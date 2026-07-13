"""Build training data from official Wikipedia dumps (dumps.wikimedia.org).

No HuggingFace anywhere: the dump is streamed over HTTP, decompressed with
bz2, parsed as XML, stripped of wiki markup by hand, and tokenized with the
GPT-2 BPE into flat uint16 binaries (data/train.bin, data/val.bin) that the
training loop memory-maps. uint16 works because the GPT-2 vocab (50257) fits
in 16 bits, halving disk and page-cache pressure vs int32.

Music upweighting: articles whose categories/infoboxes look music-related
(albums, songs, artists, instruments, genres, composers...) are written to
train.bin multiple times (--music-boost, default 3). Because the training
loop samples random windows from the stream, document order doesn't matter
-- duplication is upweighting. Val articles are held out entirely, never
duplicated.

Usage:
    python data.py                          # simplewiki: quick, ~50-80M tokens
    python data.py --source enwiki --parts 4    # full English wiki parts, ~1B tokens
    python data.py --max-articles 500      # tiny slice, for pipeline testing
"""

import argparse
import bz2
import html
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import requests
import tiktoken

DUMPS = "https://dumps.wikimedia.org"

# Crude but effective keyword classifier: an article counts as music-related
# if any of these appear in its [[Category:...]] tags or infobox name.
MUSIC_KEYWORDS = (
    "music", "album", "song", "singer", "band", "composer", "opera", "jazz",
    "guitar", "orchestra", "rapper", "hip hop", "symphon", "concert",
    "instrument", "musician", "choir", "piano", "violin", "drum", "rock",
    "pop group", "record label", "discograph",
)
INFOBOX_RE = re.compile(r"\{\{\s*infobox\s+([^|}]*)", re.IGNORECASE)
CATEGORY_RE = re.compile(r"\[\[\s*category\s*:([^\]|]*)", re.IGNORECASE)


def is_music_article(wikitext_lower: str) -> bool:
    tags = CATEGORY_RE.findall(wikitext_lower) + INFOBOX_RE.findall(wikitext_lower)
    blob = " ".join(tags)
    return any(kw in blob for kw in MUSIC_KEYWORDS)


# --- wikitext -> plain text, by hand -----------------------------------------
# Good enough for LM pretraining; a real pipeline would use a proper parser.

TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)        # innermost {{...}}
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)              # innermost {|...|}
REF_RE = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
FILE_LINK_RE = re.compile(  # [[File:...]] / [[Category:...]], one nesting level deep
    r"\[\[(?:File|Image|Category)\s*:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]",
    re.IGNORECASE)
WIKILINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")   # [[target|label]] -> label
EXTLINK_LABEL_RE = re.compile(r"\[\w+://[^\s\]]+\s+([^\]]*)\]")
EXTLINK_BARE_RE = re.compile(r"\[\w+://[^\]]*\]")
TAG_RE = re.compile(r"<[^>]+>")
HEADING_RE = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
LIST_RE = re.compile(r"^[*#:;]+\s*", re.MULTILINE)
TABLE_LINE_RE = re.compile(r"^\s*[|!].*$", re.MULTILINE)      # stray table rows
MAGIC_RE = re.compile(r"__[A-Z]+__")
QUOTES_RE = re.compile(r"'{2,}")   # ''italic'', '''bold''', '''''both''''' -- runs of 2+


def strip_wikitext(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = REF_RE.sub("", text)
    for pattern in (TEMPLATE_RE, TABLE_RE):    # nested constructs: peel inside-out
        for _ in range(20):
            text, n = pattern.subn("", text)
            if n == 0:
                break
    text = FILE_LINK_RE.sub("", text)
    text = WIKILINK_RE.sub(r"\1", text)
    text = EXTLINK_LABEL_RE.sub(r"\1", text)
    text = EXTLINK_BARE_RE.sub("", text)
    text = TAG_RE.sub("", text)
    text = HEADING_RE.sub(r"\1", text)
    text = LIST_RE.sub("", text)
    text = TABLE_LINE_RE.sub("", text)
    text = MAGIC_RE.sub("", text)
    # a plain "''" replace would turn '''bold''' into 'bold' -- strip whole runs
    text = QUOTES_RE.sub("", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- dump streaming -----------------------------------------------------------

def dump_urls(source: str, parts: int) -> list[str]:
    """Resolve dump file URLs. simplewiki is one file; enwiki is split into
    numbered parts whose exact names change per dump, so scrape the index."""
    if source == "simplewiki":
        return [f"{DUMPS}/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"]
    index = requests.get(f"{DUMPS}/enwiki/latest/", timeout=60,
                          headers={"User-Agent": "small-lm-data-prep (personal project)"}).text
    names = re.findall(r'href="(enwiki-latest-pages-articles-multistream(\d+)\.xml-p\d+p\d+\.bz2)"', index)
    names = sorted(set(names), key=lambda m: int(m[1]))
    if not names:
        raise RuntimeError("could not find multistream part files in the enwiki dump index")
    return [f"{DUMPS}/enwiki/latest/{name}" for name, _ in names[:parts]]


def iter_articles(url: str):
    """Stream a .xml.bz2 dump over HTTP and yield (title, wikitext) for every
    real article: namespace 0, not a redirect. Nothing touches disk."""
    with requests.get(url, stream=True, timeout=60,
                      headers={"User-Agent": "small-lm-data-prep (personal project)"}) as r:
        r.raise_for_status()
        yield from parse_dump(bz2.BZ2File(r.raw))


def parse_dump(stream):
    """Yield (title, wikitext) for each real article in a dump XML stream:
    namespace 0, not a redirect. Split from iter_articles so it can be fed
    any file-like object (tests, a local dump) without HTTP."""
    context = ET.iterparse(stream, events=("start", "end"))
    _, root = next(context)                  # first event is the <mediawiki> root
    ns, title, redirect, text = None, None, False, None
    for event, elem in context:
        if event != "end":
            continue
        tag = elem.tag.rsplit("}", 1)[-1]   # strip the xmlns prefix
        if tag == "ns":
            ns = elem.text
        elif tag == "title":
            title = elem.text
        elif tag == "redirect":
            redirect = True
        elif tag == "text":
            text = elem.text
        elif tag == "page":
            if ns == "0" and not redirect and text and not text.lstrip().lower().startswith("#redirect"):
                yield title, text
            ns, title, redirect, text = None, None, False, None
            # elem.clear() alone is not enough: finished <page> elements stay
            # attached to the root and pile up by the millions. Clearing the
            # root keeps memory flat across the whole dump.
            root.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="simplewiki", choices=["simplewiki", "enwiki"])
    parser.add_argument("--parts", type=int, default=4,
                        help="number of enwiki dump parts to use (each is roughly 100-300M tokens)")
    parser.add_argument("--music-boost", type=int, default=3,
                        help="write music-related articles this many times to train.bin (1 = off)")
    parser.add_argument("--val-every", type=int, default=200,
                        help="every Nth article is held out for validation")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="stop after N kept articles (for quick pipeline tests)")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "val.bin")
    for p in (train_path, val_path):
        if os.path.exists(p):
            raise SystemExit(f"{p} already exists -- delete it first to rebuild")

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    stats = {"kept": 0, "music": 0, "train_tokens": 0, "val_tokens": 0}
    done = False

    with open(train_path, "wb") as train_f, open(val_path, "wb") as val_f:
        for url in dump_urls(args.source, args.parts):
            if done:
                break
            print(f"streaming {url}")
            for title, wikitext in iter_articles(url):
                plain = strip_wikitext(wikitext)
                if len(plain) < 200:            # skip stubs and disambiguation husks
                    continue
                tokens = np.asarray(enc.encode_ordinary(f"{title}\n\n{plain}") + [eot],
                                    dtype=np.uint16)
                music = is_music_article(wikitext.lower())
                stats["kept"] += 1
                stats["music"] += music

                if stats["kept"] % args.val_every == 0:      # held out, never boosted
                    tokens.tofile(val_f)
                    stats["val_tokens"] += len(tokens)
                else:
                    repeats = args.music_boost if music else 1
                    for _ in range(repeats):
                        tokens.tofile(train_f)
                    stats["train_tokens"] += len(tokens) * repeats

                if stats["kept"] % 1000 == 0:
                    print(f"\r  {stats['kept']:,} articles ({stats['music']:,} music) | "
                          f"train {stats['train_tokens'] / 1e6:.1f}M tok | "
                          f"val {stats['val_tokens'] / 1e6:.1f}M tok", end="", flush=True)
                if args.max_articles and stats["kept"] >= args.max_articles:
                    done = True
                    break
            print()

    print(f"\ndone: {stats['kept']:,} articles, {stats['music']:,} music-related "
          f"(boosted x{args.music_boost})")
    print(f"{train_path}: {stats['train_tokens']:,} tokens")
    print(f"{val_path}: {stats['val_tokens']:,} tokens")
    print(f"\nrule of thumb: a ~45M-param model wants ~0.9B training tokens "
          f"(~20 tokens/param). you have {stats['train_tokens'] / 1e9:.2f}B.")


if __name__ == "__main__":
    main()
