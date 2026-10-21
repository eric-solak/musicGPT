"""Music questions from infobox facts: held-out test/dev sets, and training data.

SQuAD taught the reader to read, but almost none of it is about music, and
the first end-to-end scorecard showed the cost: given the right passage, the
model still answered "Who composed The Four Seasons?" with "The Four Seasons".
This generates music questions whose answers are known exactly, because they
come from the article's own infobox (facts.jsonl, written by retrieval.py):

    album Rubber Soul, Artist: the Beatles  ->  "Who recorded the album Rubber Soul?"  -> the Beatles

Articles are split by a hash of their title: 10% test, 10% dev, 80% train,
so no test article ever appears in training. Subjects of the hand-written
eval sets (eval/*.jsonl "subject") are also kept out of training.

Training contexts come from the real retriever, exactly as ask.py will see
them, in two variants: with the fact card, and prose only (no card), so the
reader also learns to find facts in text. If the retrieved passages don't
contain the answer, the label is "I don't know." -- a genuine retrieval miss.

Outputs (--out-dir, default data/music/):
    music_test.jsonl / music_dev.jsonl   question, answers, subject, qtype (ask.py --eval)
    music_train.jsonl / music_dev_examples.jsonl
                                         passages, question, answers -- for qa_data.py

Usage:
    python music_qa.py --index-dir index
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from multiprocessing import Pool

from infobox import display_title
from qa_data import NO_ANSWER, normalize_answer
from retrieval import CARD, LEAD, Retriever

# (card type, field label) -> question templates. {t} is the display title.
TEMPLATES = {
    ("album", "Artist"): ["Who recorded the album {t}?", "Which artist released the album {t}?",
                          "Who made the album {t}?", "{t} is an album by which artist?",
                          "Who wrote the album {t}?", "Whose album is {t}?",
                          "Which band or musician released {t}?"],
    ("album", "Released"): ["When was the album {t} released?", "What year did {t} come out?",
                            "In what year was {t} released?"],
    ("album", "Label"): ["Which record label released {t}?", "What label was {t} released on?"],
    ("album", "Producer"): ["Who produced the album {t}?", "Who was the producer of {t}?"],
    ("album", "Genre"): ["What genre is the album {t}?", "What style of music is {t}?"],
    ("song", "Artist"): ["Who sang {t}?", "Who performed the song {t}?", "{t} is a song by which artist?",
                         "Which artist recorded {t}?", "Who released the song {t}?"],
    ("song", "Songwriters"): ["Who wrote the song {t}?", "Who wrote {t}?", "Who are the songwriters of {t}?",
                              "Who composed the song {t}?"],
    ("song", "Album"): ["Which album is the song {t} from?", "On which album does {t} appear?"],
    ("song", "Released"): ["When was the song {t} released?", "What year did {t} come out?"],
    ("opera", "Composer"): ["Who composed the opera {t}?", "Who wrote the music for {t}?",
                            "Who is the composer of {t}?", "Which composer wrote the opera {t}?",
                            "Who wrote the opera {t}?"],
    ("opera", "Librettist"): ["Who wrote the libretto for {t}?", "Who was the librettist of {t}?"],
    ("opera", "Language"): ["In what language is the opera {t}?"],
    ("opera", "Premiere"): ["When did {t} premiere?", "When was {t} first performed?"],
    ("composition", "Composer"): ["Who composed {t}?", "Who wrote {t}?", "Which composer wrote {t}?",
                                  "Who is the composer of {t}?"],
    ("composition", "Key"): ["What key is {t} in?"],
    ("composition", "Composed"): ["When was {t} composed?", "When did the composer write {t}?"],
    ("musical artist", "Origin"): ["Where is {t} from?", "Where did {t} form?", "Where does {t} come from?"],
    ("musical artist", "Members"): ["Who are the members of {t}?", "Name a member of {t}."],
    ("musical artist", "Past members"): ["Who was a member of {t}?", "Name a former member of {t}."],
    ("musical artist", "Genres"): ["What genre of music does {t} play?", "What kind of music is {t} known for?"],
    ("musical artist", "Born"): ["When was {t} born?"],
    ("musical artist", "Birthplace"): ["Where was {t} born?"],
    ("musical artist", "Instruments"): ["What instrument does {t} play?", "What instruments did {t} play?"],
    ("person", "Born"): ["When was {t} born?", "What is the birth date of {t}?"],
    ("person", "Birthplace"): ["Where was {t} born?", "In what city was {t} born?"],
    ("person", "Died"): ["When did {t} die?"],
    ("person", "Place of death"): ["Where did {t} die?"],
    ("musical instrument", "Inventors"): ["Who invented the {t}?", "Who created the {t}?"],
    ("musical instrument", "Developed"): ["When was the {t} invented?", "When was the {t} developed?"],
    ("musical instrument", "Classification"): ["What type of instrument is the {t}?",
                                              "What kind of instrument is the {t}?"],
}
CARD_LABEL = {"album": "album", "song": "song", "opera": "opera", "musical composition": "composition",
              "musical artist": "musical artist", "instrument": "musical instrument", "person": "person"}
DATE_LABELS = {"Released", "Born", "Died", "Premiere", "Composed", "Developed"}
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_of(title: str) -> str:
    """Stable 80/10/10 split by article title."""
    h = int(hashlib.md5(title.encode("utf-8")).hexdigest(), 16) % 10
    return "test" if h == 0 else "dev" if h == 1 else "train"


def answers_for(label: str, items: list[str]) -> list[str]:
    """Acceptable answers: every listed item; for dates also the bare year."""
    out = [i for i in items if 0 < len(i) <= 60]
    if label in DATE_LABELS:
        out += [y for i in items for y in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})s?\b", i)]
    return list(dict.fromkeys(out))


def first_sentence(lead: str, title: str) -> str | None:
    """The definition an article opens with, if it looks like one."""
    text = " ".join(lead.split())
    text = re.sub(r"\(\s*[,;]?\s*[,;]?\s*\)", "", text)              # "( ; )" left by stripped templates
    sentence = SENTENCE_RE.split(text, maxsplit=1)[0].strip()
    name = display_title(title).split()[0].lower()
    if not (6 <= len(sentence.split()) <= 45) or name not in sentence.lower():
        return None
    if not re.search(r"\b(is|was|are|were)\b", sentence):
        return None
    return sentence


def make_questions(facts: list[dict], leads: dict[str, str], rng: random.Random) -> list[dict]:
    """One question per (article, field), template chosen at random, plus a
    definition question per article with a usable opening sentence."""
    out = []
    for f in facts:
        kind = CARD_LABEL[f["type"]]
        t = display_title(f["title"])
        t_inst = t.lower() if kind == "musical instrument" else t
        for label, items in f["fields"].items():
            templates = TEMPLATES.get((kind, label))
            answers = answers_for(label, items)
            if not templates or not answers or normalize_answer(answers[0]) == normalize_answer(t):
                continue
            question = rng.choice(templates).format(t=t_inst)
            if "year" in question.lower():           # "What year...?" -> the year is the answer, not the date
                answers = sorted(answers, key=lambda a: not re.fullmatch(r"\d{4}s?", a))
            out.append({"question": question, "answers": answers,
                        "subject": f["title"], "qtype": f"{kind}/{label}"})
    for title, lead in leads.items():
        sentence = first_sentence(lead, title)
        if sentence:
            t = display_title(title)
            person = bool(re.match(rf"{re.escape(t)}\s*(\(|is|was)", sentence)) and \
                bool(re.search(r"\b(born|singer|composer|musician|guitarist|pianist|rapper|songwriter)\b", sentence))
            q = rng.choice(["Who was {t}?", "Who is {t}?"] if person else ["What is {t}?", "What was {t}?",
                                                                          "Tell me about {t}."])
            out.append({"question": q.format(t=t), "answers": [sentence], "subject": title,
                        "qtype": "definition"})
    return out


def contains(text: str, answer: str) -> bool:
    return f" {normalize_answer(answer)} " in f" {normalize_answer(text)} "


# --- retrieval-shaped training examples (parallel: one retriever per worker) ---

_retriever = None


def _init_worker(index_dir: str) -> None:
    global _retriever
    _retriever = Retriever(index_dir)


def _examples(q: dict) -> list[dict]:
    """The retriever's real top-2 for the question, with and without the card.
    Labelled with the first acceptable answer found in the passages, else
    "I don't know.". Definitions only count if the opening sentence is there."""
    ranked = _retriever.ranked(q["question"])
    variants = [ranked[:2], [r for r in ranked if r[3] != CARD][:2]]
    out = []
    for i, top in enumerate(variants):
        if i == 1 and top == variants[0]:
            continue                            # no card was retrieved: one variant is enough
        passages = [[title, text] for _, title, text, _ in top]
        found = [a for a in q["answers"] if any(contains(f"{t} {x}", a) for t, x in passages)]
        out.append({"passages": passages, "question": q["question"],
                    "answers": found if found else [], "label": found[0] if found else NO_ANSWER,
                    "qtype": q["qtype"], "variant": "card" if i == 0 else "prose"})
    return out


def build_examples(questions: list[dict], index_dir: str, workers: int) -> list[dict]:
    with Pool(workers, initializer=_init_worker, initargs=(index_dir,)) as pool:
        nested = pool.map(_examples, questions, chunksize=64)
    return [ex for group in nested for ex in group]


def heldout_subjects(eval_dir: str) -> set[str]:
    subjects = set()
    for name in os.listdir(eval_dir):
        if name.endswith(".jsonl"):
            with open(os.path.join(eval_dir, name), encoding="utf-8") as f:
                subjects |= {json.loads(line).get("subject") for line in f if line.strip()}
    return subjects - {None}


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", default="index")
    p.add_argument("--out-dir", default="data/music")
    p.add_argument("--eval-dir", default="eval", help="hand-written sets whose subjects stay out of training")
    p.add_argument("--test-size", type=int, default=400)
    p.add_argument("--dev-size", type=int, default=400)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    with open(os.path.join(args.index_dir, "facts.jsonl"), encoding="utf-8") as f:
        facts = [json.loads(line) for line in f]
    retriever = Retriever(args.index_dir)
    leads = {}
    for i in (retriever.kind == LEAD).nonzero()[0]:
        title, text = retriever.chunks[int(i)]
        leads[title] = text
    del retriever

    questions = make_questions(facts, leads, rng)
    excluded = heldout_subjects(args.eval_dir)
    by_split = {"train": [], "dev": [], "test": []}
    for q in questions:
        split = split_of(q["subject"])
        if split == "train" and q["subject"] in excluded:
            continue
        by_split[split].append(q)
    print(f"{len(facts):,} fact cards, {len(leads):,} leads -> {len(questions):,} questions "
          f"({', '.join(f'{k} {len(v):,}' for k, v in by_split.items())}); "
          f"{len(excluded)} hand-written eval subjects kept out of training")
    print("question types:", dict(Counter(q["qtype"].split("/")[0] for q in questions).most_common()))

    for split, size in (("test", args.test_size), ("dev", args.dev_size)):
        sample = rng.sample(by_split[split], min(size, len(by_split[split])))
        write_jsonl(os.path.join(args.out_dir, f"music_{split}.jsonl"), sample)
        if split == "dev":
            dev_examples = build_examples(sample, args.index_dir, args.workers)
            write_jsonl(os.path.join(args.out_dir, "music_dev_examples.jsonl"), dev_examples)

    print(f"retrieving contexts for {len(by_split['train']):,} training questions...")
    train = build_examples(by_split["train"], args.index_dir, args.workers)
    write_jsonl(os.path.join(args.out_dir, "music_train.jsonl"), train)
    labels = Counter("answer" if ex["answers"] else "I don't know" for ex in train)
    print(f"{len(train):,} training examples: {dict(labels)}; "
          f"by variant {dict(Counter(ex['variant'] for ex in train))}")
    for ex in rng.sample(train, 3):
        print(f"\n  Q: {ex['question']}\n  A: {ex['label']}\n  passages: {[t for t, _ in ex['passages']]}")


if __name__ == "__main__":
    main()
