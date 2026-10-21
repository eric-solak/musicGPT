"""Ask the fine-tuned model a music question, answered from retrieved Wikipedia passages.

    question -> topic check -> BM25 top-k passages (retrieval.py) -> prompt (qa_data.format_prompt)
             -> fine-tuned reader (finetune.py) -> answer checks -> answer + the passages it read

Accuracy over coverage: an answer is only shown if it passes three checks,
otherwise the reply is "I don't know.":
  1. it appears word for word in a retrieved passage (no invented names);
  2. if the question names an article ("Who wrote Rubber Soul?"), it comes from
     that article's passages, not from a neighbouring one;
  3. the model's confidence (mean token log-prob) clears a threshold tuned on
     held-out dev questions for a target precision (--tune-gate).

Usage:
    python ask.py "Who recorded Kind of Blue?"
    python ask.py "When was the electric guitar invented?" --show-context
    python ask.py "Who recorded Kind of Blue?" --no-retrieval     # the raw model, ungrounded
    python ask.py --eval eval/music_qa.jsonl                      # precision / coverage scorecard
    python ask.py --eval eval/off_topic.jsonl                     # how many non-music questions are declined
    python ask.py --tune-gate data/music/music_dev.jsonl          # pick --min-confidence

Only music questions are answered: a question that matches a non-music
Wikipedia article better than any music passage is declined (see
retrieval.Retriever.topic). --no-topic-check turns that off.
"""

import argparse
import json
from dataclasses import dataclass, field

import tiktoken
import torch

from finetune import answer_with_confidence
from model import GPT, ModelConfig
from qa_data import NO_ANSWER, f1_score, is_abstention, normalize_answer
from retrieval import Retriever, search_terms
from train import resolve_device

# The reply to non-music questions. Distinct from "I don't know.", which means a
# *music* question the passages couldn't answer reliably.
OFF_TOPIC = ("I am musicGPT, a model trained on music data, and I cannot answer "
             "questions that aren't about music.")
# Tuned with --tune-gate on data/music/music_dev.jsonl (400 held-out questions):
# checks 1-2 alone gave 98% precision at 94% coverage; -0.1 gives ~99% at ~85%.
# Chosen on dev before running any test set. None = checks 1 and 2 only.
MIN_CONFIDENCE = -0.1


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


@dataclass
class Reply:
    text: str                                   # what the user sees
    status: str                                 # answered / abstained / declined
    raw: str = ""                               # the model's answer before the checks
    confidence: float = float("-inf")
    reason: str = ""                            # why an answer was withheld
    sources: list = field(default_factory=list)  # (title, text, score)


class Answerer:
    def __init__(self, model: GPT, enc, retriever: Retriever | None, device: str, k: int = 2,
                 max_new_tokens: int = 64, min_confidence: float | None = MIN_CONFIDENCE,
                 topic_check: bool = True):
        self.model, self.enc, self.retriever, self.device = model, enc, retriever, device
        self.k, self.max_new_tokens = k, max_new_tokens
        self.min_confidence, self.topic_check = min_confidence, topic_check

    def raw(self, question: str, passages: list[tuple[str, str]]) -> tuple[str, float]:
        with torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" else torch.no_grad():
            return answer_with_confidence(self.model, self.enc, passages, question, self.device,
                                          self.max_new_tokens)

    def ask(self, question: str, retrieval: bool = True, checks: bool = True) -> Reply:
        if retrieval and self.topic_check:
            on_topic, other = self.retriever.topic(question)
            if not on_topic:
                return Reply(OFF_TOPIC, "declined", reason=f"closest article: {other}")
        sources = self.retriever.search(question, self.k) if retrieval else []
        passages = [(title, text) for title, text, _ in sources]
        pred, conf = self.raw(question, passages)
        reply = Reply(pred, "answered", raw=pred, confidence=conf, sources=sources)
        if not pred or is_abstention(pred):
            reply.text, reply.status, reply.reason = NO_ANSWER, "abstained", "model"
        elif checks:
            reason = self.check(question, pred, conf, passages)
            if reason:
                reply.text, reply.status, reply.reason = NO_ANSWER, "abstained", reason
        return reply

    def check(self, question: str, pred: str, conf: float, passages: list[tuple[str, str]]) -> str:
        """The reason to withhold `pred`, or "" to show it."""
        if not any(contains_answer(f"{t} {x}", pred) for t, x in passages):
            return "not in the passages"
        query_terms = set(search_terms(question))
        named = [(t, x) for t, x in passages if self.retriever.names_article(t, query_terms)]
        if named and not any(contains_answer(f"{t} {x}", pred) for t, x in named):
            return "not from the article the question names"
        if self.min_confidence is not None and conf < self.min_confidence:
            return f"low confidence ({conf:.2f})"
        return ""


def load_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_off_topic_eval(answerer: Answerer, items: list[dict]) -> None:
    """Questions with no "answers" field are off-topic: count how many are
    declined, and what happens to the ones that slip through."""
    tally = {"declined": 0, "abstained": 0, "answered": 0}
    for item in items:
        reply = answerer.ask(item["question"])
        tally[reply.status] += 1
        shown = f"({reply.reason})" if reply.status == "declined" else reply.text
        print(f"   {reply.status:9} {shown[:34]:34} {item['question']}")
    n = len(items)
    print(f"\noff-topic: declined {tally['declined']}/{n}, slipped through but abstained "
          f"{tally['abstained']}/{n}, answered anyway {tally['answered']}/{n}")


def run_eval(answerer: Answerer, path: str) -> None:
    """Grounded (with checks) vs the raw model with no passages, plus retrieval
    recall@k, so a wrong answer can be blamed on the right half of the system."""
    items = load_items(path)
    if all("answers" not in item for item in items):
        run_off_topic_eval(answerer, items)
        return
    tally = {mode: {"correct": 0, "abstained": 0, "wrong": 0, "declined": 0}
             for mode in ("rag", "none", "rag, article in index", "rag, article not in index")}
    recall_hits = wrong_with_gold = withheld_correct = 0
    print(f"{'':2} {'retrieved':9} {'grounded + checks':30} {'raw model, no passages':28} question")
    for item in items:
        golds = item["answers"]
        rag = answerer.ask(item["question"])
        none = answerer.ask(item["question"], retrieval=False, checks=False)
        in_context = any(contains_answer(f"{t} {x}", g) for g in golds for t, x, _ in rag.sources)
        recall_hits += in_context
        grades = {"rag": "declined" if rag.status == "declined" else grade(rag.text, golds),
                  "none": grade(none.text, golds)}
        for mode, g in grades.items():
            tally[mode][g] += 1
        if "subject" in item:
            indexed = answerer.retriever.has_article(item["subject"])
            tally[f"rag, article {'in' if indexed else 'not in'} index"][grades["rag"]] += 1
        wrong_with_gold += grades["rag"] == "wrong" and in_context
        withheld_correct += rag.reason not in ("", "model") and grade(rag.raw, golds) == "correct"
        shown = rag.text if not rag.reason or rag.reason == "model" else f"[{rag.raw[:12]}: {rag.reason[:12]}]"
        print(f"   {'yes' if in_context else 'no':9} {grades['rag'][0].upper()} {shown[:28]:28} "
              f"{grades['none'][0].upper()} {none.text[:26]:26} {item['question']}")

    n = len(items)
    print(f"\nretrieval recall@{answerer.k}: {recall_hits}/{n} questions had a gold answer in the passages")
    rows = [("rag", "grounded + checks"), ("none", "raw, no passages"),
            ("rag, article in index", "  article in index"), ("rag, article not in index", "  not in index")]
    for mode, label in rows:
        t = tally[mode]
        total = sum(t.values())
        if not total:
            continue
        answered = t["correct"] + t["wrong"]
        precision = f"{100 * t['correct'] / answered:.0f}%" if answered else "n/a"
        print(f"{label:20} correct {t['correct']:3}  wrong {t['wrong']:3}  abstained {t['abstained']:3}"
              f"  declined {t['declined']:3}  |  precision {precision:>4}  coverage {100 * answered / total:3.0f}%"
              f"  (of {total})")
    print(f"wrong despite the gold passage being retrieved: {wrong_with_gold}; "
          f"correct answers withheld by the checks: {withheld_correct}")


def tune_gate(answerer: Answerer, path: str, target: float) -> None:
    """Pick --min-confidence: the lowest threshold whose precision on these
    (held-out dev) questions reaches `target`, i.e. the most coverage at that precision."""
    answerer.min_confidence = None
    scored, n = [], 0
    for item in load_items(path):
        n += 1
        reply = answerer.ask(item["question"])
        if reply.status == "answered":
            scored.append((reply.confidence, grade(reply.text, item["answers"]) == "correct"))
    scored.sort(key=lambda s: -s[0])
    print(f"{n} dev questions; {len(scored)} answers pass checks 1-2.\n")
    print(f"{'min confidence':>15} {'precision':>10} {'coverage':>9}")
    best, correct = None, 0
    for i, (conf, ok) in enumerate(scored, 1):
        correct += ok
        precision = correct / i
        if i % max(1, len(scored) // 12) == 0 or i == len(scored):
            print(f"{conf:15.3f} {100 * precision:9.0f}% {100 * i / n:8.0f}%")
        if precision >= target:
            best = conf
    if best is None:
        print(f"\nno threshold reaches {100 * target:.0f}% precision")
    else:
        print(f"\nfor >= {100 * target:.0f}% precision: --min-confidence {best:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?")
    p.add_argument("--ckpt", default="out/finetune/ckpt_best.pt")
    p.add_argument("--index-dir", default="index")
    p.add_argument("-k", type=int, default=2, help="passages to retrieve")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE,
                   help="withhold answers below this mean token log-prob (tune with --tune-gate)")
    p.add_argument("--show-context", action="store_true", help="print the retrieved passages")
    p.add_argument("--no-retrieval", action="store_true", help="the raw model: no passages, no checks")
    p.add_argument("--no-topic-check", action="store_true", help="answer non-music questions too")
    p.add_argument("--eval", metavar="JSONL", help="score a question set")
    p.add_argument("--tune-gate", metavar="JSONL", help="pick --min-confidence on dev questions")
    p.add_argument("--target-precision", type=float, default=0.9)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if not (args.question or args.eval or args.tune_gate):
        p.error("give a question, --eval FILE, or --tune-gate FILE")

    device = resolve_device(args.device)
    answerer = Answerer(load_model(args.ckpt, device), tiktoken.get_encoding("gpt2"),
                        None if args.no_retrieval and args.question else Retriever(args.index_dir),
                        device, args.k, args.max_new_tokens, args.min_confidence,
                        topic_check=not args.no_topic_check)
    if args.tune_gate:
        tune_gate(answerer, args.tune_gate, args.target_precision)
        return
    if args.eval:
        run_eval(answerer, args.eval)
        return

    reply = answerer.ask(args.question, retrieval=not args.no_retrieval, checks=not args.no_retrieval)
    print(reply.text)
    if reply.reason and reply.reason != "model":
        print(f"(withheld \"{reply.raw}\": {reply.reason})" if reply.raw else f"({reply.reason})")
    if reply.sources:
        print("\nsources:")
        for i, (title, text, score) in enumerate(reply.sources, 1):
            print(f"  [{i}] {title}  (score {score:.1f})")
            if args.show_context:
                print("      " + text.replace("\n", "\n      ") + "\n")


if __name__ == "__main__":
    main()
