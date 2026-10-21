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

from finetune import SFTDataset, score_answer
from model import GPT, ModelConfig
from qa_data import NO_ANSWER, build_example, exact_match, f1_score, format_prompt
from retrieval import BM25Index, ChunkStore, chunk_article, is_disambiguation, is_music_topic

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


def test_music_topic_matches_whole_words_only():
    """The pretraining classifier's substring matching let "Military
    operations" (opera), "Rocketry" (rock) and "Measuring instruments" into
    the music index."""
    assert is_music_topic("[[Category:1959 albums]]")
    assert is_music_topic("[[Category:Operas by Giuseppe Verdi]]")
    assert is_music_topic("{{Infobox instrument\n| name = Harp}}")
    assert not is_music_topic("[[Category:Military operations of World War II]]")
    assert not is_music_topic("[[Category:Rocketry]] [[Category:Measuring instruments]]")
    assert not is_music_topic("[[Category:Broadband]] {{Infobox spaceflight}}")
    # an adaptation being an opera doesn't make the novel music
    assert not is_music_topic("[[Category:1813 novels]] [[Category:British novels adapted into operas]]")


def test_disambiguation_pages_detected():
    """As off-topic leads they out-matched real music articles."""
    assert is_disambiguation("Michael Jackson (disambiguation)", "")
    assert is_disambiguation("Mercury", "'''Mercury''' may refer to: {{Disambiguation}}")
    assert not is_disambiguation("Queen (band)", "{{dablink|For other uses}} Queen are a rock band")


def test_search_restricted_to_allowed_docs():
    index = BM25Index.build(DOCS)
    allowed = np.array([True, True, False, True])     # doc 2 (the 1931 guitar) is off-limits
    hits = index.search("jazz 1931", k=4, allowed=allowed)
    assert 2 not in [d for d, _ in hits] and hits[0][0] == 3


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

    # thresholded scoring agrees: high P(abstain) only where abstaining was trained,
    # and the forced answer recovers the trained answer
    p_abstain, forced = score_answer(model, ENC, [GOLD], "Who recorded Kind of Blue?", "cpu", 8)
    assert p_abstain < 0.5 and forced == "Miles Davis"
    p_abstain, _ = score_answer(model, ENC, [], "Who invented the theremin?", "cpu", 8)
    assert p_abstain > 0.5


# --- metrics -----------------------------------------------------------------------

def test_squad_metrics():
    assert exact_match("The Miles Davis.", "miles davis") == 1.0
    assert exact_match("Miles", "Miles Davis") == 0.0
    assert f1_score("Miles", "Miles Davis") == pytest.approx(2 / 3)
    assert f1_score("John Coltrane", "Miles Davis") == 0.0
    assert exact_match("I don't know", NO_ANSWER) == 1.0


# --- answer checks (ask.py) ----------------------------------------------------------

def test_answer_checks():
    """An answer is shown only if it is in the passages, comes from the article
    the question names, and clears the confidence bar."""
    from ask import Answerer
    from retrieval import Retriever

    class StubRetriever:
        names_article = Retriever.names_article

    a = Answerer(model=None, enc=None, retriever=StubRetriever(), device="cpu", min_confidence=-1.0)
    passages = [("Rubber Soul", "Rubber Soul is the sixth studio album by the Beatles."),
                ("Pitchfork", "In his commentary, Ian Cohen wrote that every Beatles album shaped pop.")]
    q = "Who wrote the album Rubber Soul?"
    assert a.check(q, "the Beatles", -0.5, passages) == ""
    assert a.check(q, "Auguste Verdi", -0.5, passages) == "not in the passages"
    assert a.check(q, "Ian Cohen", -0.5, passages) == "not from the article the question names"
    assert a.check(q, "the Beatles", -2.0, passages).startswith("low confidence")


def test_music_words_mark_a_music_question():
    """A question naming a musical thing is about music even when its subject
    isn't indexed: "I don't know" fits it, the non-music reply doesn't."""
    from retrieval import MUSIC_QUESTION_RE
    for q in ("Who wrote the album Thriller?", "Who sang Hey Jude?", "Who composed Tosca?",
              "Who wrote the lyrics to Imagine?"):
        assert MUSIC_QUESTION_RE.search(q.lower()), q
    for q in ("Who painted the Mona Lisa?", "Who invented the telephone?", "When did World War II end?",
              "Which rock is the hardest?", "Who founded Microsoft?"):
        assert not MUSIC_QUESTION_RE.search(q.lower()), q
