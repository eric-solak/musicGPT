"""Correctness tests for the fine-tuning + retrieval stage. Run with:
    python -m pytest tests/ -q

The silent failures here are different from the model's: a loss mask off by
one trains on the wrong tokens, a truncated gold passage labels an answer the
model can't see, and a retriever with a sign error still returns *something*.
All run on CPU, with no network (tiktoken's GPT-2 encoding must be cached).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune import SFTDataset
from model import GPT, ModelConfig
from qa_data import NO_ANSWER, build_example, exact_match, f1_score, format_prompt
from retrieval import BM25Index, ChunkStore, chunk_article

ENC = tiktoken.get_encoding("gpt2")

DOCS = [
    "Kind of Blue is a studio album by the American jazz trumpeter Miles Davis.",
    "The saxophone is a woodwind instrument invented by Adolphe Sax.",
    "The electric guitar was invented in 1931 and is used in jazz and rock.",
    "Jazz originated in New Orleans. Jazz jazz jazz.",
]


# --- retrieval -------------------------------------------------------------------

def test_bm25_exact_match_ranks_first():
    index = BM25Index.build(DOCS)
    assert index.search("who recorded kind of blue", k=1)[0][0] == 0
    assert index.search("saxophone inventor", k=1)[0][0] == 1


def test_bm25_rare_term_outweighs_common_term():
    """'jazz' is in 3 of 4 docs, '1931' in one: the rare term must decide."""
    index = BM25Index.build(DOCS)
    assert index.idf[index.vocab["1931"]] > index.idf[index.vocab["jazz"]]
    assert index.search("jazz 1931", k=1)[0][0] == 2


def test_bm25_ignores_stopwords_and_unknown_terms():
    index = BM25Index.build(DOCS)
    assert "the" not in index.vocab
    assert index.search("the of and zzzunknown", k=3) == []


def test_bm25_save_load_roundtrip(tmp_path):
    index = BM25Index.build(DOCS)
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    for q in ("miles davis album", "woodwind", "jazz guitar 1931"):
        assert index.search(q, k=4) == loaded.search(q, k=4)


def test_chunk_store_roundtrip(tmp_path):
    chunks = [("Kind of Blue", "A modal jazz album."), ("Dvořák", "Czech composer.\nSecond line.")]
    offsets = [0]
    with open(tmp_path / "chunks.bin", "wb") as f:
        for title, text in chunks:
            raw = f"{title}\n{text}".encode("utf-8")
            f.write(raw)
            offsets.append(offsets[-1] + len(raw))
    np.save(tmp_path / "chunk_offsets.npy", np.asarray(offsets, dtype=np.int64))
    store = ChunkStore(tmp_path)
    assert [store[i] for i in range(len(store))] == chunks


def test_chunking_respects_limit_and_loses_nothing():
    plain = "\n".join(["short intro line",
                       " ".join(f"w{i}" for i in range(300)),   # one paragraph > max_words
                       "", "a closing paragraph of a few words"])
    chunks = chunk_article(plain, max_words=50)
    assert all(len(c.split()) <= 50 for c in chunks)
    assert " ".join(chunks).split() == plain.split()


# --- SFT examples ------------------------------------------------------------------

GOLD = ("Kind of Blue", DOCS[0])
DISTRACTOR = ("Filler", " ".join(["unrelated filler text"] * 200))


def test_example_mask_covers_exactly_the_answer():
    tokens, mask, _ = build_example(ENC, [GOLD], "Who recorded Kind of Blue?", "Miles Davis", 512, keep=(0,))
    assert ENC.decode(tokens[mask == 1].tolist()) == " Miles Davis<|endoftext|>"
    assert ENC.decode(tokens[mask == 0].tolist()) == format_prompt([GOLD], "Who recorded Kind of Blue?")
    assert mask[0] == 0 and mask[-1] == 1 and np.all(np.diff(mask) >= 0)   # one contiguous answer span


def test_example_truncates_distractor_never_gold():
    tokens, mask, final = build_example(ENC, [DISTRACTOR, GOLD], "Who recorded Kind of Blue?",
                                        "Miles Davis", block_size=128, keep=(1,))
    assert len(tokens) <= 128 + 1
    assert GOLD in final                                   # gold untouched
    assert all(p == GOLD or len(p[1]) < len(DISTRACTOR[1]) for p in final)


def test_example_that_cannot_fit_is_skipped():
    long_gold = ("Gold", " ".join(["miles davis"] * 400))
    assert build_example(ENC, [long_gold], "q?", "a", block_size=128, keep=(0,)) is None


# --- SFT batching ------------------------------------------------------------------

@pytest.fixture
def sft_dir(tmp_path):
    """Two hand-made examples in qa_data.py's on-disk format."""
    examples = [build_example(ENC, [GOLD], "Who recorded Kind of Blue?", "Miles Davis", 512, (0,)),
                build_example(ENC, [], "Who invented the theremin?", NO_ANSWER, 512)]
    offsets = [0]
    with open(tmp_path / "qa_train.bin", "wb") as t, open(tmp_path / "qa_train_mask.bin", "wb") as m:
        for tokens, mask, _ in examples:
            tokens.tofile(t)
            mask.tofile(m)
            offsets.append(offsets[-1] + len(tokens))
    np.save(tmp_path / "qa_train_idx.npy", np.asarray(offsets, dtype=np.int64))
    return tmp_path, examples


def test_sft_batch_shift_mask_and_padding(sft_dir):
    path, examples = sft_dir
    ds = SFTDataset(path, "train")
    torch.manual_seed(0)
    x, y = ds.get_batch(16, "cpu", pad_id=ENC.eot_token)
    assert x.shape == y.shape and x.size(1) == max(len(t) for t, _, _ in examples) - 1
    for row in range(16):
        assert (y[row] != -1).any(), "a row with no targets makes the loss NaN"
        targets = y[row][y[row] != -1].tolist()
        decoded = ENC.decode(targets)
        assert decoded in (" Miles Davis<|endoftext|>", f" {NO_ANSWER}<|endoftext|>")
        n = int((x[row] != ENC.eot_token).sum())     # real inputs; padding is eot
        assert (y[row, n:] == -1).all(), "target on a padding position"
        for i in range(n - 1):
            if y[row, i] != -1:
                assert y[row, i] == x[row, i + 1], "target is not the next input token"


def test_sft_overfit_answers_from_context(sft_dir):
    """End to end through the masked loss: a tiny model trained on the two
    examples must greedy-decode the trained answers after the prompts."""
    path, examples = sft_dir
    ds = SFTDataset(path, "train")
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=50304, block_size=128, n_layer=2, n_head=2, d_model=64))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        x, y = ds.get_batch(4, "cpu", pad_id=ENC.eot_token)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"masked SFT loss stuck at {loss.item():.3f}"
    for tokens, mask, _ in examples:
        prompt = tokens[mask == 0].tolist()
        want = tokens[mask == 1].tolist()
        out = model.generate(torch.tensor([prompt]), max_new_tokens=len(want), temperature=0.0)
        assert out[0, len(prompt):].tolist() == want


# --- metrics -----------------------------------------------------------------------

def test_squad_metrics():
    assert exact_match("The Miles Davis.", "miles davis") == 1.0
    assert exact_match("Miles", "Miles Davis") == 0.0
    assert f1_score("Miles", "Miles Davis") == pytest.approx(2 / 3)
    assert f1_score("John Coltrane", "Miles Davis") == 0.0
    assert exact_match("I don't know", NO_ANSWER) == 1.0
