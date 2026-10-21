"""Build supervised fine-tuning data from SQuAD 2.0: passages + question -> answer.

The base model learned the *shape* of Wikipedia; nothing in pretraining
rewarded a true fact over a fluent one. SQuAD 2.0 supplies that signal:
~130k questions written by people against Wikipedia paragraphs, with the
answer span marked.

Examples are shaped like what retrieval will hand the model at inference:
the gold paragraph plus a hard-negative paragraph (BM25's top hit for the
question among the *other* SQuAD paragraphs, often from the same article),
in shuffled order. "I don't know." is taught on simulated retrieval misses:
an answerable question given two negatives and no gold paragraph (25% of
them by default).

SQuAD 2.0's own ~43k unanswerable questions are left out by default
(--keep-squad-unanswerable puts them back). They are adversarial -- written
to look answerable from their paragraph -- and the first fine-tune showed a
45M model can't learn that distinction: it abstained at nearly the same
rate whether or not the answer was there. A retrieval miss is the
abstention ask.py actually needs, and a learnable one.

No HuggingFace: the official JSON is downloaded directly. Outputs, in
--data-dir (default data/):
    qa_{train,val}.bin       uint16 token stream, examples back to back
    qa_{train,val}_mask.bin  uint8, 1 where the token is part of the answer
    qa_{train,val}_idx.npy   int64 example start offsets (n_examples + 1)
    qa_val.jsonl             passages/question/gold answers, for EM/F1 eval

Usage:
    python qa_data.py
"""

import argparse
import json
import os
import random
import re
import string
from collections import Counter
from urllib.parse import unquote

import numpy as np
import requests
import tiktoken

from data import MUSIC_KEYWORDS
from retrieval import BM25Index

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/{split}-v2.0.json"
NO_ANSWER = "I don't know."
MIN_PASSAGE_TOKENS = 32    # a distractor cut shorter than this is dropped instead
PUNCTUATION = frozenset(string.punctuation)


# --- the prompt format: one definition, shared by training and ask.py ----------

def format_prompt(passages: list[tuple[str, str]], question: str) -> str:
    """Plain-text markers rather than new special tokens: no embedding resize,
    and the base model already knows what "Question:"/"Answer:" look like."""
    blocks = "\n".join(f"[{i}] {title}\n{text}" for i, (title, text) in enumerate(passages, 1))
    return f"Context:\n{blocks}\n\nQuestion: {question.strip()}\nAnswer:"


def build_example(enc, passages: list[tuple[str, str]], question: str, answer: str,
                  block_size: int, keep: tuple[int, ...] = ()):
    """Tokenize one example to fit block_size, with a loss mask over the answer.

    Passages whose index is in `keep` (the gold paragraph) are never cut --
    truncating them could cut out the answer while still labelling it. Other
    passages are trimmed from the end, then dropped, until the example fits.
    Returns (tokens, mask, final_passages), or None if it cannot fit.
    """
    items = [[title, text, i in keep] for i, (title, text) in enumerate(passages)]
    answer_ids = enc.encode_ordinary(" " + answer.strip()) + [enc.eot_token]
    while True:
        final = [(title, text) for title, text, _ in items]
        prompt_ids = enc.encode_ordinary(format_prompt(final, question))
        # +1: an example of length L trains on L-1 (input, next-token) pairs
        over = len(prompt_ids) + len(answer_ids) - (block_size + 1)
        if over <= 0:
            break
        cuttable = [j for j, (_, _, fixed) in enumerate(items) if not fixed]
        if not cuttable:
            return None
        j = cuttable[-1]
        text_ids = enc.encode_ordinary(items[j][1])
        if len(text_ids) - over < MIN_PASSAGE_TOKENS:
            items.pop(j)
        else:
            text = enc.decode(text_ids[:len(text_ids) - over])
            items[j][1] = text[:text.rfind(" ")] if " " in text else ""   # end on a word boundary
    tokens = np.asarray(prompt_ids + answer_ids, dtype=np.uint16)
    mask = np.zeros(len(tokens), dtype=np.uint8)
    mask[len(prompt_ids):] = 1
    return tokens, mask, final


# --- SQuAD metrics, written out --------------------------------------------------

def normalize_answer(s: str) -> str:
    """The official SQuAD normalization: lowercase, drop punctuation and
    articles, collapse whitespace."""
    s = "".join(ch for ch in s.lower() if ch not in PUNCTUATION)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def is_abstention(pred: str) -> bool:
    return normalize_answer(pred) == normalize_answer(NO_ANSWER)


# --- SQuAD -> examples -----------------------------------------------------------

def load_squad(split: str, data_dir: str) -> list[dict]:
    path = os.path.join(data_dir, f"squad-{split}-v2.0.json")
    if not os.path.exists(path):
        print(f"downloading {SQUAD_URL.format(split=split)}")
        r = requests.get(SQUAD_URL.format(split=split), timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["data"]


def clean_title(title: str) -> str:
    return unquote(title).replace("_", " ")


def music_density(article: dict) -> float:
    """MUSIC_KEYWORDS hits per 1000 words across an article's paragraphs.
    SQuAD has no categories or infoboxes, so data.py's classifier can't run
    here; keyword density over the prose is the closest stand-in."""
    text = " ".join(p["context"] for p in article["paragraphs"]).lower()
    hits = sum(len(re.findall(r"\b" + re.escape(kw), text)) for kw in MUSIC_KEYWORDS)
    return 1000 * hits / max(1, len(text.split()))


def make_examples(articles: list[dict], enc, block_size: int, miss_rate: float,
                  rng: random.Random, use_all_answers: bool, keep_unanswerable: bool = False):
    """Yield (article_idx, tokens, mask, record) for every question that fits."""
    paras = [(a_i, clean_title(a["title"]), p) for a_i, a in enumerate(articles)
             for p in a["paragraphs"]]
    print(f"indexing {len(paras):,} paragraphs for hard negatives")
    index = BM25Index.build(f"{title} {p['context']}" for _, title, p in paras)

    stats = Counter()
    for p_i, (a_i, title, para) in enumerate(paras):
        gold = (title, para["context"])
        for qa in para["qas"]:
            if qa["is_impossible"] and not keep_unanswerable:
                stats["skipped_squad_unanswerable"] += 1
                continue
            question = qa["question"]
            hits = [d for d, _ in index.search(question, k=4) if d != p_i]
            negatives = [(paras[d][1], paras[d][2]["context"]) for d in hits[:2]]
            golds = [] if qa["is_impossible"] else list(dict.fromkeys(
                a["text"] for a in qa["answers"]))

            if golds and len(negatives) == 2 and rng.random() < miss_rate and not any(
                    g.lower() in text.lower() for g in golds for _, text in negatives):
                # simulated retrieval miss: the answer is nowhere in the context
                passages, keep, golds = negatives, (), []
                stats["simulated_miss"] += 1
            else:
                passages = [gold] + negatives[:1]
                rng.shuffle(passages)
                keep = (passages.index(gold),)
            stats["label_no_answer" if not golds else "label_answer"] += 1

            answer = golds[0] if golds else NO_ANSWER
            built = build_example(enc, passages, question, answer, block_size, keep)
            if built is None:
                stats["too_long"] += 1
                continue
            tokens, mask, final = built
            record = {"passages": final, "question": question,
                      "answers": golds if use_all_answers else golds[:1]}
            yield a_i, tokens, mask, record
    print("  " + ", ".join(f"{k}={v:,}" for k, v in sorted(stats.items())))


def music_examples(path: str, enc, block_size: int, source_tag: bool):
    """Examples from music_qa.py (retrieved passages + infobox-derived label),
    as (article_idx=-1, tokens, mask, record) like make_examples. Passages
    containing the answer are never truncated, same rule as the SQuAD gold."""
    stats = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            passages = [tuple(p) for p in ex["passages"]]
            keep = tuple(i for i, (t, x) in enumerate(passages)
                         if ex["label"] != NO_ANSWER and ex["label"].lower() in f"{t} {x}".lower())
            built = build_example(enc, passages, ex["question"], ex["label"], block_size, keep)
            if built is None:
                stats["too_long"] += 1
                continue
            tokens, mask, final = built
            stats["label_answer" if ex["answers"] else "label_no_answer"] += 1
            record = {"passages": final, "question": ex["question"], "answers": ex["answers"],
                      "qtype": ex.get("qtype")}
            if source_tag:
                record["source"] = "music"
            yield -1, tokens, mask, record
    print("  music: " + ", ".join(f"{k}={v:,}" for k, v in sorted(stats.items())))


def write_split(prefix: str, examples, repeats_for) -> int:
    """Write examples back to back; repeats_for(article_idx) = copies to write.

    Concatenated in memory and written in one call per file: ~150k small
    writes took minutes on Colab's Google Drive mount."""
    tokens, masks, offsets = [], [], [0]
    for a_i, tok, mask, _ in examples:
        for _ in range(repeats_for(a_i)):
            tokens.append(tok)
            masks.append(mask)
            offsets.append(offsets[-1] + len(tok))
    np.concatenate(tokens).tofile(prefix + ".bin")
    np.concatenate(masks).tofile(prefix + "_mask.bin")
    np.save(prefix + "_idx.npy", np.asarray(offsets, dtype=np.int64))
    return len(offsets) - 1


def show_example(enc, tokens: np.ndarray, mask: np.ndarray, tail_chars: int = 400) -> None:
    """Decode-and-eyeball: the prompt's tail, with the trained-on span in brackets."""
    split = int(np.argmax(mask))
    prompt = enc.decode(tokens[:split].tolist())
    answer = enc.decode(tokens[split:].tolist())
    print("..." + prompt[-tail_chars:] + "[" + answer + "]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--music-boost", type=int, default=3,
                        help="write music articles' examples this many times to qa_train (1 = off)")
    parser.add_argument("--music-threshold", type=float, default=20.0,
                        help="MUSIC_KEYWORDS hits per 1000 words for an article to count as music")
    parser.add_argument("--miss-rate", type=float, default=0.25,
                        help="fraction of answerable questions given a no-gold context")
    parser.add_argument("--keep-squad-unanswerable", action="store_true",
                        help="also train on SQuAD 2.0's adversarial unanswerable questions")
    parser.add_argument("--music-dir", default=None,
                        help="music_qa.py output: adds music_train.jsonl to train and "
                             "music_dev_examples.jsonl to val (tagged, for checkpoint selection)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    prefix = {s: os.path.join(args.data_dir, f"qa_{s}") for s in ("train", "val")}
    for p in prefix.values():
        if os.path.exists(p + ".bin"):
            raise SystemExit(f"{p}.bin already exists -- delete the qa_* files first to rebuild")

    enc = tiktoken.get_encoding("gpt2")
    rng = random.Random(args.seed)

    train_articles = load_squad("train", args.data_dir)
    music = {i for i, a in enumerate(train_articles) if music_density(a) >= args.music_threshold}
    print(f"{len(music)} of {len(train_articles)} train articles count as music "
          f"(boosted x{args.music_boost}):")
    print("  " + ", ".join(sorted(clean_title(train_articles[i]["title"]) for i in music)) + "\n")

    examples = list(make_examples(train_articles, enc, args.block_size, args.miss_rate, rng,
                                  use_all_answers=False,
                                  keep_unanswerable=args.keep_squad_unanswerable))
    if args.music_dir:
        examples += music_examples(os.path.join(args.music_dir, "music_train.jsonl"), enc,
                                   args.block_size, source_tag=False)
    n_train = write_split(prefix["train"], examples,
                          lambda a_i: args.music_boost if a_i in music else 1)
    print(f"{prefix['train']}.bin: {n_train:,} examples\n")
    for i in rng.sample(range(len(examples)), 3):
        show_example(enc, examples[i][1], examples[i][2])

    # dev: held out entirely, never boosted; every gold answer kept for EM/F1
    val = list(make_examples(load_squad("dev", args.data_dir), enc, args.block_size,
                             args.miss_rate, rng, use_all_answers=True,
                             keep_unanswerable=args.keep_squad_unanswerable))
    if args.music_dir:
        val += music_examples(os.path.join(args.music_dir, "music_dev_examples.jsonl"), enc,
                              args.block_size, source_tag=True)
    n_val = write_split(prefix["val"], val, lambda a_i: 1)
    with open(prefix["val"] + ".jsonl", "w", encoding="utf-8") as f:
        for *_, record in val:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{prefix['val']}.bin: {n_val:,} examples (+ {prefix['val']}.jsonl for EM/F1)")


if __name__ == "__main__":
    main()
