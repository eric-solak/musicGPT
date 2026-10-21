"""Retrieval over Wikipedia: passage chunking and BM25, written by hand.

The fine-tuned model answers from passages placed in its context; this file
finds those passages. Music-classified articles from the same dumps used in
pretraining are split into ~120-word chunks, and a BM25 inverted index
(numpy CSR postings, no search library) ranks them against a question.

BM25 rather than dense embeddings: music questions are dense with names --
artists, albums, composers, years -- and exact term matching weighted by
rarity is exactly what finds them. It also keeps the repo free of a second
pretrained model.

Index layout (--index-dir, default index/):
    bm25.npz           term_ptr, doc_ids, tfs, doc_len (CSR postings)
    vocab.json         term list; position = term id
    chunks.bin         every chunk as UTF-8 "title\\ntext", concatenated
    chunk_offsets.npy  byte offsets into chunks.bin (n_chunks + 1)

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

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Top-k (doc_id, score) by BM25, best first; docs scoring 0 are dropped."""
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
        k = min(k, self.n_docs)
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

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


class Retriever:
    """BM25 index + chunk text: question in, (title, text, score) passages out."""

    def __init__(self, index_dir: str):
        if not os.path.exists(os.path.join(index_dir, "bm25.npz")):
            raise FileNotFoundError(f"no index in {index_dir} -- run `python retrieval.py build` first")
        self.index = BM25Index.load(index_dir)
        self.chunks = ChunkStore(index_dir)

    def search(self, query: str, k: int = 2) -> list[tuple[str, str, float]]:
        return [(*self.chunks[i], score) for i, score in self.index.search(query, k)]


def build(args) -> None:
    from data import dump_urls, is_music_article, iter_articles, strip_wikitext

    os.makedirs(args.index_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.index_dir, "bm25.npz")):
        raise SystemExit(f"{args.index_dir}/bm25.npz already exists -- delete it first to rebuild")

    offsets = [0]
    stats = {"seen": 0, "kept": 0}

    def docs():
        """Stream music articles, write each chunk's text to chunks.bin, and
        yield the text to index (title included, so title terms match)."""
        for url in dump_urls(args.source, args.parts):
            print(f"streaming {url}")
            for title, wikitext in iter_articles(url):
                stats["seen"] += 1
                if not (args.all_articles or is_music_article(wikitext.lower())):
                    continue
                plain = strip_wikitext(wikitext)
                if len(plain) < 200:
                    continue
                for text in chunk_article(plain, args.max_words):
                    raw = f"{title}\n{text}".encode("utf-8")
                    blob.write(raw)
                    offsets.append(offsets[-1] + len(raw))
                    yield f"{title} {text}"
                stats["kept"] += 1
                if stats["kept"] % 1000 == 0:
                    print(f"\r  {stats['kept']:,} articles indexed of {stats['seen']:,} seen | "
                          f"{len(offsets) - 1:,} chunks", end="", flush=True)
                if args.max_articles and stats["kept"] >= args.max_articles:
                    return
            print()

    with open(os.path.join(args.index_dir, "chunks.bin"), "wb") as blob:
        index = BM25Index.build(docs())
    index.save(args.index_dir)
    np.save(os.path.join(args.index_dir, "chunk_offsets.npy"), np.asarray(offsets, dtype=np.int64))
    print(f"\ndone: {stats['kept']:,} articles -> {index.n_docs:,} chunks, "
          f"{len(index.terms):,} terms, {len(index.doc_ids):,} postings, "
          f"avg {index.avgdl:.0f} terms/chunk")


def search(args) -> None:
    retriever = Retriever(args.index_dir)
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
    b.add_argument("--max-articles", type=int, default=None, help="stop after N indexed articles")
    b.add_argument("--all-articles", action="store_true", help="index everything, not just music")
    b.add_argument("--index-dir", default="index")

    s = sub.add_parser("search", help="query the index and print the top chunks")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--index-dir", default="index")

    args = parser.parse_args()
    build(args) if args.cmd == "build" else search(args)


if __name__ == "__main__":
    main()
