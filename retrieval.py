"""Retrieval over Wikipedia: passage chunking and BM25, written by hand.

The fine-tuned model answers from passages placed in its context; this file
finds those passages. Music articles from the same dumps used in pretraining
are split into ~120-word chunks, and a BM25 inverted index (numpy CSR
postings, no search library) ranks them against a question.

BM25 rather than dense embeddings: music questions are dense with names --
artists, albums, composers, years -- and exact term matching weighted by
rarity is exactly what finds them. It also keeps the repo free of a second
pretrained model.

Staying on topic: the index also holds the lead section of every *non*-music
article, flagged off-topic. Those chunks are never shown to the model; they
exist so a question can be checked against the whole of Wikipedia. "Who
painted the Mona Lisa?" matches the Mona Lisa article better than anything
about music, so it is declined. A music-only index can't do this: every
question finds *some* music passage, and BM25 scores on their own don't
separate on-topic from off-topic matches.

Index layout (--index-dir, default index/):
    bm25.npz           term_ptr, doc_ids, tfs, doc_len (CSR postings)
    vocab.json         term list; position = term id
    chunks.bin         every chunk as UTF-8 "title\\ntext", concatenated
    chunk_offsets.npy  byte offsets into chunks.bin (n_chunks + 1)
    chunk_kind.npy     per chunk: body / lead / card / list (music) or off-topic
    facts.jsonl        the parsed infobox facts behind each card (infobox.py)

Fact cards: infoboxes hold the cleanest facts ("artist = the Beatles") and
the text stripper deletes them with every other template, so each music
article with an infobox also gets a one-passage card rendered from it.

Usage:
    python retrieval.py build --source enwiki --parts 4      # the real index
    python retrieval.py build --max-articles 500 --index-dir index_smoke
    python retrieval.py search "who recorded kind of blue"
"""

import argparse
import json
import os
import re
from array import array
from collections import Counter
from typing import Iterable

import numpy as np

from infobox import display_title, extract_facts, fact_card

# Function words carry no topic and have the longest postings lists, so they
# are pure search cost. Question words are included: "when did X" should
# match on X, not on every passage containing "when".
STOPWORDS = frozenset("""
    a an and are as at be been but by did do does for from had has have he her
    his how i if in into is it its of on or she so than that the their them
    then there these they this to was were what when where which who whom why
    will with would
""".split())
TERM_RE = re.compile(r"\w+")

# Which articles count as music. data.py's pretraining classifier substring-
# matches keywords, so "opera" fires on "Military operations", "rock" on
# "Rocketry", "instrument" on "Measuring instruments" -- the first index held
# Apollo 11, Thermometer and Pride and Prejudice. Here: whole words only,
# and no ambiguous words ("rock", "instrument", "concert", "drum").
MUSIC_TERM_RE = re.compile(r"\b(?:" + "|".join("""
    music musical musicals musician musicians album albums song songs singles
    singer singers singing songwriter songwriters band bands composer composers
    compositions opera operas operettas jazz blues guitar guitars guitarist
    guitarists orchestra orchestras rapper rappers hip\\ hop symphony symphonies
    choir choirs piano pianos pianist pianists violin violins violinist
    violinists drummer drummers discography discographies record\\ label
    record\\ labels concerto concertos sonata sonatas ballet ballets oratorio
    oratorios
""".split()) + r")\b")
# What each chunk is (chunk_kind.npy). Only music kinds are ever shown to the model.
BODY, LEAD, CARD, LIST, OFF_TOPIC = range(5)
KIND_NAMES = ("body", "lead", "card", "list", "off-topic")

# A question using these words is about music, whatever the index holds.
MUSIC_QUESTION_RE = re.compile(MUSIC_TERM_RE.pattern[:-3] + r"|sang|sing|sings|sung|composed|lyrics|libretto|librettist)\b")

CATEGORY_RE = re.compile(r"\[\[\s*category\s*:([^\]|]*)", re.IGNORECASE)
INFOBOX_RE = re.compile(r"\{\{\s*infobox\s+([^|}]*)", re.IGNORECASE)


def is_music_topic(wikitext: str) -> bool:
    """Music by category or infobox, matched on whole words. "Infobox
    instrument" is only used for musical instruments, so it counts too.
    "Novels adapted into operas" says the *adaptation* is music, not the
    article (Pride and Prejudice, Romeo and Juliet), so those are skipped."""
    text = wikitext.lower()
    categories = " ".join(c for c in CATEGORY_RE.findall(text)
                          if "adapted into" not in c and "adaptations" not in c)
    infoboxes = " ".join(INFOBOX_RE.findall(text))
    return bool(MUSIC_TERM_RE.search(categories) or MUSIC_TERM_RE.search(infoboxes)
                or re.search(r"\binstrument\b", infoboxes))


def search_terms(text: str) -> list[str]:
    return [t for t in TERM_RE.findall(text.lower()) if t not in STOPWORDS]


def chunk_article(plain: str, max_words: int = 120) -> list[str]:
    """Pack an article's paragraphs into chunks of at most max_words words.

    Paragraphs are kept whole when they fit, so a chunk reads as prose; a
    paragraph longer than max_words is split into max_words-sized pieces.
    ~120 words is ~160 GPT-2 tokens, so two chunks plus a question fit in the
    512-token context with room to spare.
    """
    chunks, cur, n = [], [], 0
    for para in plain.split("\n"):
        words = para.split()
        while words:
            if n + len(words) <= max_words:
                cur.append(" ".join(words))
                n += len(words)
                break
            if n:                       # doesn't fit: close this chunk, retry in a fresh one
                chunks.append("\n".join(cur))
                cur, n = [], 0
                continue
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
    if cur:
        chunks.append("\n".join(cur))
    return chunks


class BM25Index:
    """Okapi BM25 over an inverted index stored as CSR arrays.

    Postings for term t are doc_ids[term_ptr[t]:term_ptr[t+1]] with matching
    term frequencies in tfs. score(d, q) = sum over query terms t of
        idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
    with the non-negative idf = ln(1 + (N - df + 0.5) / (df + 0.5)).
    """

    def __init__(self, vocab: list[str], term_ptr, doc_ids, tfs, doc_len,
                 k1: float = 1.5, b: float = 0.75):
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.terms = vocab
        self.term_ptr, self.doc_ids, self.tfs = term_ptr, doc_ids, tfs
        self.doc_len = doc_len.astype(np.float32)
        self.k1, self.b = k1, b
        self.n_docs = len(doc_len)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        df = np.diff(term_ptr).astype(np.float64)
        self.idf = np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

    @classmethod
    def build(cls, docs: Iterable[str], **kwargs) -> "BM25Index":
        """Index an iterable of documents; doc id = position in the iterable.

        Postings are accumulated as flat (term, doc, tf) arrays -- compact
        typed arrays rather than Python lists, because a few hundred thousand
        chunks produce tens of millions of postings -- then grouped by term
        with one stable sort, which keeps each term's doc ids ascending.
        """
        vocab: dict[str, int] = {}
        t_col, d_col, tf_col = array("i"), array("i"), array("H")
        doc_len = array("i")
        for d, text in enumerate(docs):
            counts = Counter(search_terms(text))
            doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                t_col.append(vocab.setdefault(term, len(vocab)))
                d_col.append(d)
                tf_col.append(min(tf, 65535))
        t = np.frombuffer(t_col, dtype=np.int32)
        order = np.argsort(t, kind="stable")
        term_ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(np.bincount(t, minlength=len(vocab)), out=term_ptr[1:])
        return cls(list(vocab), term_ptr,
                   np.frombuffer(d_col, dtype=np.int32)[order],
                   np.frombuffer(tf_col, dtype=np.uint16)[order],
                   np.frombuffer(doc_len, dtype=np.int32).copy(), **kwargs)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every doc for the query (float32, length n_docs)."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(search_terms(query)):
            tid = self.vocab.get(term)
            if tid is None:
                continue
            lo, hi = self.term_ptr[tid], self.term_ptr[tid + 1]
            docs = self.doc_ids[lo:hi]
            tf = self.tfs[lo:hi].astype(np.float32)
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[docs] / self.avgdl)
            scores[docs] += self.idf[tid] * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def search(self, query: str, k: int = 5, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-k (doc_id, score) by BM25, best first; docs scoring 0 are
        dropped. `allowed` (bool per doc) restricts the candidates."""
        return top_k(self.scores(query), k, allowed)

    def save(self, index_dir: str) -> None:
        np.savez(os.path.join(index_dir, "bm25.npz"), term_ptr=self.term_ptr,
                 doc_ids=self.doc_ids, tfs=self.tfs, doc_len=self.doc_len.astype(np.int32))
        with open(os.path.join(index_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.terms, f, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: str) -> "BM25Index":
        arrays = np.load(os.path.join(index_dir, "bm25.npz"))
        with open(os.path.join(index_dir, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab, arrays["term_ptr"], arrays["doc_ids"], arrays["tfs"], arrays["doc_len"])


def top_k(scores: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
    if allowed is not None:
        scores = np.where(allowed, scores, 0.0)
    k = min(k, len(scores))
    if k == 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


class ChunkStore:
    """Random access to chunk text without loading it all: chunks.bin is
    memory-mapped and sliced by byte offsets."""

    def __init__(self, index_dir: str):
        self.blob = np.memmap(os.path.join(index_dir, "chunks.bin"), dtype=np.uint8, mode="r")
        self.offsets = np.load(os.path.join(index_dir, "chunk_offsets.npy"))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> tuple[str, str]:
        raw = self.blob[self.offsets[i]:self.offsets[i + 1]].tobytes().decode("utf-8")
        title, _, text = raw.partition("\n")
        return title, text


def is_list_chunk(text: str) -> bool:
    """Credits, track listings, discographies: many short lines. They repeat
    terms ("Michael Landau - electric guitar" x12), which BM25 rewards, and
    they answer almost nothing."""
    lines = [line for line in text.split("\n") if line.strip()]
    return len(lines) >= 6 and sum(len(line.split()) <= 6 for line in lines) / len(lines) >= 0.6


class Retriever:
    """BM25 index + chunk text: question in, music passages and an on-topic
    verdict out.

    Ranking is BM25 over the top `pool` music chunks, then re-scored with what
    BM25 can't see: whether the question names the chunk's article ("Who wrote
    Rubber Soul?" -> the article *Rubber Soul*), and if so its fact card and
    lead first; list chunks are discounted. Weights are tuned on held-out dev
    questions (see music_qa.py), not on the test set.
    """

    # Tuned on music_dev (never the test sets). "specific": title idf-sum that
    # earns the full title bonus; "primary": extra for the undisambiguated article.
    WEIGHTS = {"title": 16.0, "card": 16.0, "lead": 16.0, "list": 0.5, "specific": 14.0, "primary": 4.0}

    def __init__(self, index_dir: str):
        if not os.path.exists(os.path.join(index_dir, "bm25.npz")):
            raise FileNotFoundError(f"no index in {index_dir} -- run `python retrieval.py build` first")
        self.index = BM25Index.load(index_dir)
        self.chunks = ChunkStore(index_dir)
        kinds = os.path.join(index_dir, "chunk_kind.npy")
        flags = os.path.join(index_dir, "chunk_is_music.npy")
        if os.path.exists(kinds):
            self.kind = np.load(kinds)
        elif os.path.exists(flags):              # older index: music vs off-topic only
            self.kind = np.where(np.load(flags), BODY, OFF_TOPIC).astype(np.uint8)
        else:                                    # oldest: all music
            self.kind = np.full(len(self.chunks), BODY, dtype=np.uint8)
        self.is_music = self.kind != OFF_TOPIC
        self.weights = dict(self.WEIGHTS)

    def has_article(self, title: str) -> bool:
        """Is this music article in the index? (Titles are read once, lazily.)"""
        if not hasattr(self, "_titles"):
            starts = np.flatnonzero((self.kind == LEAD) | (self.kind == CARD))
            self._titles = {self.chunks[int(i)][0] for i in starts}
        return title in self._titles

    def names_article(self, title: str, query_terms: set[str]) -> bool:
        """Does the question name this article? All of its title's terms
        (disambiguator dropped: "Queen (band)" -> "Queen") are in the question."""
        terms = set(search_terms(display_title(title)))
        return bool(terms) and terms <= query_terms

    def title_match(self, title: str, query_terms: set[str]) -> float:
        """0 if the question doesn't name the article, else how *specifically*
        it does, in (0, 1]: the idf-sum of the title's terms over
        weights["specific"]. A question that happens to contain "jazz" or
        "symphony" names the article *Jazz* only weakly; "Rubber Soul" fully."""
        if not self.names_article(title, query_terms):
            return 0.0
        if not self.weights["specific"]:
            return 1.0
        terms = set(search_terms(display_title(title)))
        idf = sum(float(self.index.idf[self.index.vocab[t]]) for t in terms if t in self.index.vocab)
        return min(1.0, idf / self.weights["specific"])

    @staticmethod
    def is_primary(title: str, query_terms: set[str]) -> bool:
        """"Led Zeppelin" over "Led Zeppelin (album)" -- unless the question
        says "album", in which case the disambiguated article is meant."""
        m = re.search(r"\(([^()]*)\)$", title)
        return m is None or set(search_terms(m.group(1))) <= query_terms

    def _rescored(self, scores: np.ndarray, query_terms: set[str], allowed: np.ndarray,
                  pool: int) -> list[tuple[float, str, str, int]]:
        """BM25 top-`pool` among `allowed`, re-scored, best first. Off-topic
        leads are scored like music leads, so the topic check compares like
        with like."""
        w = self.weights
        out = []
        for i, score in top_k(scores, pool, allowed):
            title, text = self.chunks[i]
            kind = int(self.kind[i])
            match = self.title_match(title, query_terms)
            if match:
                score += match * (w["title"] + (w["card"] if kind == CARD else
                                                w["lead"] if kind in (LEAD, OFF_TOPIC) else 0.0)
                                  + (w["primary"] if self.is_primary(title, query_terms) else 0.0))
            if kind == LIST:
                score *= w["list"]
            out.append((score, title, text, kind))
        out.sort(key=lambda r: -r[0])
        return out

    def ranked(self, query: str, pool: int = 200) -> list[tuple[float, str, str, int]]:
        """Music chunks as (score, title, text, kind), best first."""
        return self._rescored(self.index.scores(query), set(search_terms(query)), self.is_music, pool)

    def search(self, query: str, k: int = 2, pool: int = 200) -> list[tuple[str, str, float]]:
        """Top-k *music* passages as (title, text, score)."""
        return [(title, text, score) for score, title, text, _ in self.ranked(query, pool)[:k]]

    def topic(self, query: str, pool: int = 50) -> tuple[bool, str | None]:
        """(is the question about music?, title of the best off-topic match).

        On topic when the best music chunk outscores the best off-topic lead,
        both re-scored the same way: a question that names an article
        ("...inventing the piano?") is about that article. Music articles
        contribute every chunk and off-topic ones only their lead, so ties
        lean toward answering; a question that slips through still gets only
        music passages, and the answer checks still apply.
        """
        if MUSIC_QUESTION_RE.search(query.lower()):
            # "Who wrote the album Thriller?" is a music question even when the
            # album isn't indexed and the closest article is the film genre
            return True, None
        scores = self.index.scores(query)
        query_terms = set(search_terms(query))
        best_other = self._rescored(scores, query_terms, ~self.is_music, pool)[:1]
        if not best_other:
            return True, None
        best_music = self._rescored(scores, query_terms, self.is_music, pool)[:1]
        if not best_music:
            return False, best_other[0][1]
        return best_music[0][0] >= best_other[0][0], best_other[0][1]


def is_disambiguation(title: str, wikitext: str) -> bool:
    """Disambiguation pages list other articles; as off-topic leads they
    out-match real music articles ("King of Pop" -> "Michael Jackson
    (disambiguation)")."""
    return "(disambiguation)" in title or bool(
        re.search(r"\{\{\s*(?:disambiguation|disambig|dab|hndis|geodis)\b", wikitext.lower()))


def lead_section(wikitext: str) -> str:
    """The text before the first == heading ==: an article's summary."""
    m = re.search(r"\n==[^=]", wikitext)
    return wikitext[:m.start()] if m else wikitext


def build(args) -> None:
    from data import dump_urls, iter_articles, strip_wikitext

    os.makedirs(args.index_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.index_dir, "bm25.npz")):
        raise SystemExit(f"{args.index_dir}/bm25.npz already exists -- delete it first to rebuild")

    offsets, kinds = [0], []
    stats = Counter()

    def add(title: str, text: str, kind: int):
        raw = f"{title}\n{text}".encode("utf-8")
        blob.write(raw)
        offsets.append(offsets[-1] + len(raw))
        kinds.append(kind)
        stats[KIND_NAMES[kind]] += 1
        return f"{title} {text}"            # indexed with its title, so title terms match

    def docs():
        """Stream every article. Music ones become a fact card (if they have
        an infobox) plus their chunked text; the rest contribute their lead
        section as one off-topic chunk."""
        for url in dump_urls(args.source, args.parts):
            print(f"streaming {url}")
            for title, wikitext in iter_articles(url):
                stats["seen"] += 1
                if is_disambiguation(title, wikitext):
                    continue                    # lists of other articles: neither music nor a topic
                if is_music_topic(wikitext):
                    plain = strip_wikitext(wikitext)
                    if len(plain) < 200:
                        continue
                    stats["music articles"] += 1
                    facts = extract_facts(title, wikitext)
                    if facts:
                        facts_f.write(json.dumps(facts, ensure_ascii=False) + "\n")
                        yield add(title, fact_card(facts), CARD)
                    for j, text in enumerate(chunk_article(plain, args.max_words)):
                        yield add(title, text, LEAD if j == 0 else LIST if is_list_chunk(text) else BODY)
                elif not args.music_only:
                    lead = strip_wikitext(lead_section(wikitext))
                    if len(lead) >= 100:
                        yield add(title, chunk_article(lead, args.max_words)[0], OFF_TOPIC)
                if stats["seen"] % 5000 == 0:
                    print(f"\r  {stats['seen']:,} seen | {stats['music articles']:,} music articles | "
                          f"{stats['card']:,} cards | {stats['off-topic']:,} off-topic leads",
                          end="", flush=True)
                if args.max_articles and stats["music articles"] >= args.max_articles:
                    return
            print()

    with open(os.path.join(args.index_dir, "chunks.bin"), "wb") as blob, \
            open(os.path.join(args.index_dir, "facts.jsonl"), "w", encoding="utf-8") as facts_f:
        index = BM25Index.build(docs())
    index.save(args.index_dir)
    np.save(os.path.join(args.index_dir, "chunk_offsets.npy"), np.asarray(offsets, dtype=np.int64))
    np.save(os.path.join(args.index_dir, "chunk_kind.npy"), np.asarray(kinds, dtype=np.uint8))
    print(f"\ndone: {stats['music articles']:,} music articles -> "
          + ", ".join(f"{stats[name]:,} {name}" for name in KIND_NAMES)
          + f" chunks | {len(index.terms):,} terms, {len(index.doc_ids):,} postings")


def search(args) -> None:
    retriever = Retriever(args.index_dir)
    on_topic, other = retriever.topic(args.query)
    print(f"on topic: {on_topic}" + (f"  (best off-topic match: {other})" if other else "") + "\n")
    for rank, (title, text, score) in enumerate(retriever.search(args.query, args.k), 1):
        preview = " ".join(text.split())[:240]
        print(f"[{rank}] {score:6.2f}  {title}\n      {preview}...\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="stream Wikipedia dumps and build the index")
    b.add_argument("--source", default="enwiki", choices=["simplewiki", "enwiki"])
    b.add_argument("--parts", type=int, default=4, help="enwiki dump parts (match data.py's run)")
    b.add_argument("--max-words", type=int, default=120, help="words per chunk")
    b.add_argument("--max-articles", type=int, default=None, help="stop after N music articles")
    b.add_argument("--music-only", action="store_true",
                   help="skip the off-topic leads (no topic check possible)")
    b.add_argument("--index-dir", default="index")

    s = sub.add_parser("search", help="query the index and print the top music chunks")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--index-dir", default="index")

    args = parser.parse_args()
    build(args) if args.cmd == "build" else search(args)


if __name__ == "__main__":
    main()
