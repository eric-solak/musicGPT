"""Supervised fine-tuning: turn the pretrained base model into a reader that
answers a question from the passages in its context, or says it can't.

Same loop as train.py (AMP, cosine LR, grad accumulation, clipping, CSV
metrics, resumable checkpoints), with three differences:

  * Data is whole examples, not random windows: each one is a SQuAD prompt
    + answer from qa_data.py, right-padded to the longest in its batch.
  * Loss is masked to the answer. Targets on prompt and padding positions
    are -1, which GPT.forward's cross_entropy already ignores -- the model
    is trained to *produce* answers, not to model the passages.
  * Evaluation decodes answers: SQuAD EM/F1 on a fixed dev subset, plus how
    often the model abstains when it should (and when it shouldn't).

Usage:
    python qa_data.py                       # once, to build data/qa_*
    python finetune.py                      # from out/ckpt_best.pt
    python finetune.py --resume             # continue from out/finetune/ckpt.pt

Outputs (in --out-dir, default out/finetune/):
    ckpt.pt / ckpt_best.pt  latest / best-dev-F1 checkpoint (same format as train.py's)
    metrics.csv             step, train_loss, val_loss, lr, tokens/sec, em, f1, abstain metrics
    samples/                dev predictions at every eval step
"""

import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import tiktoken
import torch

from config import FinetuneConfig
from model import GPT, ModelConfig
from qa_data import NO_ANSWER, exact_match, f1_score, format_prompt, is_abstention
from train import configure_optimizer, lr_at, resolve_device, resolve_dtype


def parse_args() -> tuple[FinetuneConfig, bool]:
    defaults = FinetuneConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init-from", default=defaults.init_from)
    p.add_argument("--data-dir", default=defaults.data_dir)
    p.add_argument("--out-dir", default=defaults.out_dir)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=defaults.grad_accum_steps)
    p.add_argument("--max-steps", type=int, default=defaults.max_steps)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    p.add_argument("--eval-iters", type=int, default=defaults.eval_iters)
    p.add_argument("--eval-qa-examples", type=int, default=defaults.eval_qa_examples)
    p.add_argument("--device", default=defaults.device)
    p.add_argument("--dtype", default=defaults.dtype, choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt")
    p.add_argument("--seed", type=int, default=defaults.seed)
    args = p.parse_args()

    cfg = FinetuneConfig(**{k: v for k, v in vars(args).items() if k != "resume"})
    cfg.min_lr = cfg.lr / 10
    return cfg, args.resume


class SFTDataset:
    """Whole (prompt, answer) examples from qa_data.py's flat token files.

    Random examples per batch, right-padded to the batch's longest. Padding
    on the right is safe under a causal mask: real tokens never attend to
    the pads after them, and pad targets are -1 so they cost nothing.
    """

    def __init__(self, data_dir: str, split: str):
        prefix = os.path.join(data_dir, f"qa_{split}")
        if not os.path.exists(prefix + ".bin"):
            raise FileNotFoundError(f"{prefix}.bin not found -- run `python qa_data.py` first")
        self.tokens = np.memmap(prefix + ".bin", dtype=np.uint16, mode="r")
        self.mask = np.memmap(prefix + "_mask.bin", dtype=np.uint8, mode="r")
        self.offsets = np.load(prefix + "_idx.npy")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def get_batch(self, batch_size: int, device: str, pad_id: int):
        ix = torch.randint(len(self), (batch_size,)).tolist()
        spans = [(int(self.offsets[i]), int(self.offsets[i + 1])) for i in ix]
        T = max(end - start for start, end in spans) - 1
        x = torch.full((batch_size, T), pad_id, dtype=torch.long)
        y = torch.full((batch_size, T), -1, dtype=torch.long)
        for row, (start, end) in enumerate(spans):
            tok = torch.from_numpy(self.tokens[start:end].astype(np.int64))
            # mask[i] = token i is answer; target i predicts token i+1
            is_answer = torch.from_numpy(self.mask[start + 1:end].astype(np.bool_))
            x[row, :end - start - 1] = tok[:-1]
            y[row, :end - start - 1] = tok[1:].masked_fill(~is_answer, -1)
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def prompt_ids(model: GPT, enc, passages: list[tuple[str, str]], question: str,
               max_new_tokens: int) -> list[int]:
    """Tokenized prompt, dropping passages from the end until prompt + answer
    room fits the context."""
    passages = list(passages)
    while True:
        ids = enc.encode_ordinary(format_prompt(passages, question))
        if len(ids) + max_new_tokens <= model.config.block_size or not passages:
            break
        passages.pop()
    return ids[-(model.config.block_size - max_new_tokens):]


def _greedy(model: GPT, enc, ids: list[int], device: str, max_new_tokens: int) -> list[int]:
    """Greedy continuation of ids, stopping at (and dropping) end-of-text."""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.0,
                         eos_token=enc.eot_token, vocab_limit=enc.n_vocab)
    return [t for t in out[0, len(ids):].tolist() if t != enc.eot_token]


@torch.no_grad()
def answer(model: GPT, enc, passages: list[tuple[str, str]], question: str, device: str,
           max_new_tokens: int = 32) -> str:
    """Greedy-decode an answer to `question` from `passages`.

    No repetition penalty, on purpose: it pushes down tokens already in the
    context, and copying tokens out of the context is exactly what an
    extractive answer does.
    """
    ids = prompt_ids(model, enc, passages, question, max_new_tokens)
    return enc.decode(_greedy(model, enc, ids, device, max_new_tokens)).strip()


@torch.no_grad()
def score_answer(model: GPT, enc, passages: list[tuple[str, str]], question: str, device: str,
                 max_new_tokens: int = 32) -> tuple[float, str]:
    """(P("I don't know."), best non-abstaining answer) for thresholded abstention.

    Plain greedy decoding is biased toward abstaining: "I don't know." always
    starts with the same token, while the probability of a real answer is
    split across many possible first tokens, so " I" can win the argmax even
    when the model puts most of its mass on *some* answer. SQuAD 2.0 systems
    handle this by scoring "no answer" as a whole against the best answer and
    abstaining only above a threshold tuned on held-out data.
    """
    model.eval()
    ids = prompt_ids(model, enc, passages, question, max_new_tokens)
    abstain = enc.encode_ordinary(" " + NO_ANSWER) + [enc.eot_token]
    x = torch.tensor([ids + abstain[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x, targets=x)                       # targets only to get every position's logits
    logp = torch.log_softmax(logits[0, len(ids) - 1:].float(), dim=-1)
    p_abstain = logp.gather(-1, torch.tensor(abstain, device=device)[:, None]).sum().exp().item()

    first = logits[0, len(ids) - 1].float().clone()      # the answer's first token, abstention banned
    first[enc.n_vocab:] = float("-inf")
    first[abstain[0]] = float("-inf")
    first_tok = int(first.argmax())
    if first_tok == enc.eot_token:
        return p_abstain, ""
    rest = _greedy(model, enc, ids + [first_tok], device, max_new_tokens - 1)
    return p_abstain, enc.decode([first_tok] + rest).strip()


@torch.no_grad()
def estimate_loss(model, datasets, cfg: FinetuneConfig, autocast_ctx, pad_id: int) -> dict[str, float]:
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, cfg.device, pad_id)
            with autocast_ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def evaluate_qa(model, enc, examples: list[dict], cfg: FinetuneConfig, autocast_ctx,
                samples_path: str | None = None) -> dict[str, float]:
    """SQuAD-style EM/F1 (max over gold answers; "I don't know." is the gold
    for unanswerable questions), plus the two abstention rates that matter
    for grounding: abstaining when the answer isn't there (want high) and
    abstaining when it is (want low)."""
    model.eval()
    em = f1 = 0.0
    no_ans = no_ans_right = has_ans = has_ans_abstained = 0
    lines = []
    for ex in examples:
        with autocast_ctx:
            pred = answer(model, enc, ex["passages"], ex["question"], cfg.device, cfg.max_answer_tokens)
        golds = ex["answers"] or [NO_ANSWER]
        em += max(exact_match(pred, g) for g in golds)
        f1 += max(f1_score(pred, g) for g in golds)
        if ex["answers"]:
            has_ans += 1
            has_ans_abstained += is_abstention(pred)
        else:
            no_ans += 1
            no_ans_right += is_abstention(pred)
        lines.append(f"Q: {ex['question']}\n   gold: {golds[0]!r}\n   pred: {pred!r}\n")
    if samples_path:
        with open(samples_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines[:50]))
    model.train()
    n = max(1, len(examples))
    return {"em": 100 * em / n, "f1": 100 * f1 / n,
            "no_ans_acc": 100 * no_ans_right / max(1, no_ans),
            "false_abstain": 100 * has_ans_abstained / max(1, has_ans)}


def load_eval_examples(data_dir: str, n: int, seed: int) -> list[dict]:
    """A fixed random dev subset, so EM/F1 is comparable across evals."""
    with open(os.path.join(data_dir, "qa_val.jsonl"), encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return random.Random(seed).sample(records, min(n, len(records)))


def main() -> None:
    cfg, resume = parse_args()
    device = cfg.device = resolve_device(cfg.device)
    ptdtype = resolve_dtype(cfg.dtype, device)
    device_type = "cuda" if device.startswith("cuda") else device

    torch.manual_seed(cfg.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(os.path.join(cfg.out_dir, "samples"), exist_ok=True)
    autocast_ctx = (nullcontext() if ptdtype == torch.float32
                    else torch.amp.autocast(device_type=device_type, dtype=ptdtype))
    scaler = torch.amp.GradScaler("cuda" if device_type == "cuda" else "cpu",
                                  enabled=ptdtype == torch.float16)
    enc = tiktoken.get_encoding("gpt2")

    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    start_step, best_f1, best_val = 0, -1.0, float("inf")
    src = torch.load(ckpt_path if resume else cfg.init_from, map_location=device)
    if resume:
        model_cfg = ModelConfig(**src["model_config"])
        # "step" = optimizer updates completed, i.e. the next step to run
        start_step, best_val, best_f1 = src["step"], src["best_val_loss"], src["best_f1"]
    else:
        # the pretrained weights, with fine-tuning dropout switched on
        model_cfg = ModelConfig(**{**src["model_config"], "dropout": cfg.dropout})
    model = GPT(model_cfg).to(device)
    model.load_state_dict(src["model"])
    optimizer = configure_optimizer(model, cfg)
    if resume:
        optimizer.load_state_dict(src["optimizer"])
        print(f"resumed from {ckpt_path} at step {start_step}")
    else:
        print(f"initialized from {cfg.init_from} (pretrain step {src['step']}, "
              f"val loss {src['best_val_loss']:.4f})")
    del src

    datasets = {"train": SFTDataset(cfg.data_dir, "train"), "val": SFTDataset(cfg.data_dir, "val")}
    eval_examples = load_eval_examples(cfg.data_dir, cfg.eval_qa_examples, cfg.seed)
    pad_id = enc.eot_token
    print(f"{len(datasets['train']):,} train / {len(datasets['val']):,} val examples | "
          f"{cfg.batch_size * cfg.grad_accum_steps} examples/step | device={device} dtype={ptdtype}")

    metrics_path = os.path.join(cfg.out_dir, "metrics.csv")
    new_log = not (resume and os.path.exists(metrics_path))
    metrics_file = open(metrics_path, "w" if new_log else "a", newline="")
    metrics = csv.writer(metrics_file)
    if new_log:
        metrics.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec",
                          "em", "f1", "no_ans_acc", "false_abstain"])

    def save_ckpt(path: str, step: int) -> None:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(cfg),
            "step": step,
            "best_val_loss": best_val,
            "best_f1": best_f1,
        }, path)

    def run_eval(step: int) -> None:
        """Eval + checkpoint after `step` optimizer updates."""
        nonlocal best_f1, best_val
        losses = estimate_loss(model, datasets, cfg, autocast_ctx, pad_id)
        qa = evaluate_qa(model, enc, eval_examples, cfg, autocast_ctx,
                         os.path.join(cfg.out_dir, "samples", f"step_{step:06d}.txt"))
        print(f"step {step:6d} | eval: train {losses['train']:.4f}, val {losses['val']:.4f} | "
              f"EM {qa['em']:.1f} F1 {qa['f1']:.1f} | abstains: when unanswerable "
              f"{qa['no_ans_acc']:.0f}%, when answerable {qa['false_abstain']:.0f}%")
        metrics.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", "", "",
                          *(f"{qa[k]:.2f}" for k in ("em", "f1", "no_ans_acc", "false_abstain"))])
        metrics_file.flush()
        best_val = min(best_val, losses["val"])
        if qa["f1"] > best_f1:
            best_f1 = qa["f1"]
            save_ckpt(os.path.join(cfg.out_dir, "ckpt_best.pt"), step)
        if step > 0:
            save_ckpt(ckpt_path, step)

    if start_step == 0:
        run_eval(0)    # the base model before any fine-tuning: the "before" number

    model.train()
    x, y = datasets["train"].get_batch(cfg.batch_size, device, pad_id)
    t_last, tokens_since = time.time(), 0
    running_loss = None

    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for micro in range(cfg.grad_accum_steps):
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            tokens_since += x.numel()
            x, y = datasets["train"].get_batch(cfg.batch_size, device, pad_id)
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_item = loss.item() * cfg.grad_accum_steps
        running_loss = loss_item if running_loss is None else 0.9 * running_loss + 0.1 * loss_item
        if step % cfg.log_interval == 0:
            dt = time.time() - t_last
            tok_s = tokens_since / dt if dt > 0 else 0.0
            t_last, tokens_since = time.time(), 0
            print(f"step {step:6d} | loss {running_loss:.4f} | lr {lr:.2e} | {tok_s / 1e3:.1f}k tok/s")
            metrics.writerow([step, f"{running_loss:.4f}", "", f"{lr:.6f}", f"{tok_s:.0f}"])
            metrics_file.flush()

        done = step + 1
        if done % cfg.eval_interval == 0 or done == cfg.max_steps:
            run_eval(done)
            t_last, tokens_since = time.time(), 0   # don't count eval time in tok/s

    metrics_file.close()
    print(f"done. best dev F1 {best_f1:.1f}. checkpoints in {cfg.out_dir}/")


if __name__ == "__main__":
    main()
