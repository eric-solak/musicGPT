"""Ask the fine-tuned model a question, answered from retrieved Wikipedia passages.

    question -> BM25 top-k chunks (retrieval.py) -> prompt (qa_data.format_prompt)
             -> fine-tuned reader (finetune.py) -> answer + the passages it read

Usage:
    python ask.py "Who recorded Kind of Blue?"
    python ask.py "When was the electric guitar invented?" --show-context
    python ask.py "Who recorded Kind of Blue?" --no-retrieval     # ungrounded, for comparison
    python ask.py --eval eval/music_qa.jsonl                      # grounded vs ungrounded scorecard
"""

import argparse
import json

import tiktoken
import torch

from finetune import answer
from model import GPT, ModelConfig
from qa_data import f1_score, is_abstention, normalize_answer
from retrieval import Retriever
from train import resolve_device


def load_model(path: str, device: str) -> GPT:
    ckpt = torch.load(path, map_location=device)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def contains_answer(text: str, gold: str) -> bool:
    """Whole-word containment after SQuAD normalization: "four" must not
    match inside "fourteen", nor "4" inside "1940s"."""
    return f" {normalize_answer(gold)} " in f" {normalize_answer(text)} "


def grade(pred: str, golds: list[str]) -> str:
    """correct / abstained / wrong. Lenient on form, strict on content: a gold
    alias contained in the prediction ("Beethoven" in "Ludwig van Beethoven")
    or token F1 >= 0.5 counts as correct."""
    if is_abstention(pred):
        return "abstained"
    if any(contains_answer(pred, g) or f1_score(pred, g) >= 0.5 for g in golds):
        return "correct"
    return "wrong"


def run_eval(model, enc, retriever, path: str, k: int, device: str, max_new_tokens: int) -> None:
    """Every question twice -- with retrieved passages and with none -- plus
    retrieval recall@k, so a wrong grounded answer can be blamed on the right
    half of the system: the retriever (gold never retrieved) or the reader."""
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    tally = {mode: {"correct": 0, "abstained": 0, "wrong": 0} for mode in ("rag", "none")}
    recall_hits = 0
    wrong_with_gold_retrieved = 0
    print(f"{'':2} {'retrieved':9} {'with retrieval':28} {'no retrieval':28} question")
    for item in items:
        golds = item["answers"]
        hits = retriever.search(item["question"], k)
        passages = [(title, text) for title, text, _ in hits]
        in_context = any(contains_answer(f"{t} {x}", g) for g in golds for t, x in passages)
        recall_hits += in_context
        preds = {"rag": answer(model, enc, passages, item["question"], device, max_new_tokens),
                 "none": answer(model, enc, [], item["question"], device, max_new_tokens)}
        grades = {mode: grade(pred, golds) for mode, pred in preds.items()}
        for mode, g in grades.items():
            tally[mode][g] += 1
        wrong_with_gold_retrieved += grades["rag"] == "wrong" and in_context
        cells = [f"{grades[m][0].upper()} {preds[m][:24]:24}" for m in ("rag", "none")]
        print(f"   {'yes' if in_context else 'no':9} {cells[0]:28} {cells[1]:28} {item['question']}")

    n = len(items)
    print(f"\nretrieval recall@{k}: {recall_hits}/{n} questions had a gold answer in the passages")
    for mode, label in (("rag", "with retrieval"), ("none", "no retrieval")):
        t = tally[mode]
        print(f"{label:15} correct {t['correct']:3}/{n}  abstained {t['abstained']:3}/{n}  "
              f"wrong {t['wrong']:3}/{n}")
    print(f"grounded answers that were wrong despite the gold passage being retrieved: "
          f"{wrong_with_gold_retrieved}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?")
    p.add_argument("--ckpt", default="out/finetune/ckpt_best.pt")
    p.add_argument("--index-dir", default="index")
    p.add_argument("-k", type=int, default=2, help="passages to retrieve")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--show-context", action="store_true", help="print the retrieved passages")
    p.add_argument("--no-retrieval", action="store_true", help="answer with an empty context")
    p.add_argument("--eval", metavar="JSONL", help="score a question set with and without retrieval")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if not (args.question or args.eval):
        p.error("give a question, or --eval FILE")

    device = resolve_device(args.device)
    model = load_model(args.ckpt, device)
    enc = tiktoken.get_encoding("gpt2")
    retriever = None if args.no_retrieval and not args.eval else Retriever(args.index_dir)

    if args.eval:
        run_eval(model, enc, retriever, args.eval, args.k, device, args.max_new_tokens)
        return

    hits = retriever.search(args.question, args.k) if retriever else []
    passages = [(title, text) for title, text, _ in hits]
    print(answer(model, enc, passages, args.question, device, args.max_new_tokens))
    if hits:
        print("\nsources:")
        for i, (title, text, score) in enumerate(hits, 1):
            print(f"  [{i}] {title}  (BM25 {score:.1f})")
            if args.show_context:
                print("      " + text.replace("\n", "\n      ") + "\n")


if __name__ == "__main__":
    main()
