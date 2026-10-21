"""Separate "can the model read?" from "does it know when to abstain?".

Greedy decoding conflates the two (see finetune.score_answer). For each dev
question this scores P("I don't know.") and the best non-abstaining answer,
then reports:

  * greedy             -- what finetune.py's evals show
  * forced answers     -- abstention banned, scored on answerable questions
                          only: the model's reading ability on its own
  * thresholded        -- abstain only when P("I don't know.") > t, with t
                          tuned on *other* dev questions, then applied to the
                          same fixed 300-question subset finetune.py reports on

Usage:
    python calibrate.py --ckpt out/finetune/ckpt.pt --data-dir data
"""

import argparse
import json
import os
import random
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch

from finetune import answer, load_eval_examples, score_answer
from model import GPT, ModelConfig
from qa_data import NO_ANSWER, exact_match, f1_score, is_abstention
from train import resolve_device, resolve_dtype


def metrics(preds: list[str], examples: list[dict]) -> dict[str, float]:
    em = f1 = no_ans = no_ans_right = has_ans = has_ans_abstained = 0
    for pred, ex in zip(preds, examples):
        golds = ex["answers"] or [NO_ANSWER]
        em += max(exact_match(pred, g) for g in golds)
        f1 += max(f1_score(pred, g) for g in golds)
        if ex["answers"]:
            has_ans += 1
            has_ans_abstained += is_abstention(pred)
        else:
            no_ans += 1
            no_ans_right += is_abstention(pred)
    n = max(1, len(examples))
    return {"em": 100 * em / n, "f1": 100 * f1 / n,
            "no_ans_acc": 100 * no_ans_right / max(1, no_ans),
            "false_abstain": 100 * has_ans_abstained / max(1, has_ans)}


def thresholded(scored: list[tuple[float, str]], t: float) -> list[str]:
    return [NO_ANSWER if p > t else forced for p, forced in scored]


def show(label: str, m: dict[str, float]) -> None:
    print(f"{label:34} EM {m['em']:5.1f}  F1 {m['f1']:5.1f}  | abstains: when unanswerable "
          f"{m['no_ans_acc']:3.0f}%, when answerable {m['false_abstain']:3.0f}%")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="out/finetune/ckpt.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--eval-examples", type=int, default=300, help="match finetune.py's --eval-qa-examples")
    p.add_argument("--tune-examples", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337, help="match finetune.py's --seed")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    ptdtype = resolve_dtype("auto", device)
    autocast_ctx = (nullcontext() if ptdtype == torch.float32
                    else torch.amp.autocast(device_type="cuda", dtype=ptdtype))
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    print(f"loaded {args.ckpt} (fine-tune step {ckpt['step']})")

    eval_set = load_eval_examples(args.data_dir, args.eval_examples, args.seed)
    with open(os.path.join(args.data_dir, "qa_val.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    eval_questions = {ex["question"] for ex in eval_set}
    rest = [r for r in records if r["question"] not in eval_questions]
    tune_set = random.Random(0).sample(rest, min(args.tune_examples, len(rest)))

    def score_all(examples):
        out = []
        for i, ex in enumerate(examples, 1):
            with autocast_ctx:
                out.append(score_answer(model, enc, ex["passages"], ex["question"], device))
            if i % 100 == 0:
                print(f"  scored {i}/{len(examples)}", flush=True)
        return out

    print(f"scoring {len(tune_set)} tuning questions")
    tune_scored = score_all(tune_set)
    print(f"scoring {len(eval_set)} eval questions")
    eval_scored = score_all(eval_set)
    with autocast_ctx:
        greedy = [answer(model, enc, ex["passages"], ex["question"], device) for ex in eval_set]

    grid = np.round(np.arange(0.02, 1.0, 0.02), 2)
    tune_f1 = [metrics(thresholded(tune_scored, t), tune_set)["f1"] for t in grid]
    best_t = float(grid[int(np.argmax(tune_f1))])

    answerable = [(s, ex) for s, ex in zip(eval_scored, eval_set) if ex["answers"]]
    forced = metrics([f for (_, f), _ in answerable], [ex for _, ex in answerable])

    print(f"\non the fixed {len(eval_set)}-question dev subset finetune.py reports on:")
    show("always \"I don't know.\"", metrics([NO_ANSWER] * len(eval_set), eval_set))
    show("greedy (finetune.py's eval)", metrics(greedy, eval_set))
    show(f"thresholded, t={best_t:.2f}", metrics(thresholded(eval_scored, best_t), eval_set))
    print(f"\nforced answers, answerable questions only ({len(answerable)}): "
          f"EM {forced['em']:.1f}  F1 {forced['f1']:.1f}   <- reading ability")
    print("\nthreshold sweep on the tuning questions (F1):")
    print("  " + "  ".join(f"{t:.1f}:{f:.1f}" for t, f in zip(grid, tune_f1) if round(t * 10) == t * 10))
    print(f"\nbest abstention threshold on the tuning questions: {best_t:.2f} "
          f"(if greedy scores as well, thresholding isn't needed -- it wasn't for run 2)")


if __name__ == "__main__":
    main()
