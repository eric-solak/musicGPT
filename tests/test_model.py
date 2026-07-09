"""Correctness tests for the model. Run with:  python -m pytest tests/ -q

These check the properties that silently break transformer implementations:
attention math, causal masking, RoPE geometry, and the ability to memorize.
All run on CPU in under a minute.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import GPT, ModelConfig, apply_rope, build_rope_cache

TINY = dict(vocab_size=128, block_size=64, n_layer=2, n_head=4, d_model=64, dropout=0.0)


def make_model(**overrides) -> GPT:
    torch.manual_seed(0)
    return GPT(ModelConfig(**{**TINY, **overrides}))


def test_manual_attention_matches_flash():
    """The hand-written QK^T/sqrt(d) path and F.scaled_dot_product_attention
    must compute the same function."""
    model = make_model(flash=False)
    model_flash = make_model(flash=True)
    model_flash.load_state_dict(model.state_dict())
    model.eval(), model_flash.eval()

    idx = torch.randint(0, 128, (2, 48))
    targets = torch.randint(0, 128, (2, 48))
    logits_manual, loss_manual = model(idx, targets)
    logits_flash, loss_flash = model_flash(idx, targets)

    assert torch.allclose(logits_manual, logits_flash, atol=1e-5), \
        f"max diff {(logits_manual - logits_flash).abs().max().item()}"
    assert abs(loss_manual.item() - loss_flash.item()) < 1e-5


def test_causality():
    """Changing a token at position t must not change logits at positions < t."""
    for flash in (False, True):
        model = make_model(flash=flash)
        model.eval()
        idx = torch.randint(0, 128, (1, 32))
        logits_a, _ = model(idx, targets=idx)  # pass targets to get all positions' logits

        idx_b = idx.clone()
        idx_b[0, 20] = (idx_b[0, 20] + 1) % 128
        logits_b, _ = model(idx_b, targets=idx_b)

        assert torch.allclose(logits_a[0, :20], logits_b[0, :20], atol=1e-5), \
            f"flash={flash}: future token leaked into the past"
        assert not torch.allclose(logits_a[0, 20:], logits_b[0, 20:], atol=1e-5), \
            f"flash={flash}: changed token had no effect at all"


def test_init_loss_near_uniform():
    """At init the model should be near-uniform over the vocab: loss ~ ln(V).
    Much higher means broken init; much lower means an information leak.

    Targets must be drawn independently of the inputs here: using targets=idx
    ("predict yourself") scores ~0.9 below ln(V) even at init, because weight
    tying leaks each input token's identity into its own position's logits.
    (Found the hard way -- see README, "What broke".)"""
    model = make_model()
    idx = torch.randint(0, 128, (8, 64))
    targets = torch.randint(0, 128, (8, 64))
    _, loss = model(idx, targets=targets)
    expected = math.log(TINY["vocab_size"])
    assert abs(loss.item() - expected) < 0.5, f"init loss {loss.item():.2f}, expected ~{expected:.2f}"


def test_rope_is_a_rotation():
    """RoPE must preserve vector norms (it only rotates channel pairs)."""
    cos, sin = build_rope_cache(seq_len=32, head_dim=16, theta=10000.0)
    x = torch.randn(2, 4, 32, 16)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)
    assert not torch.allclose(x, rotated)  # ...but positions > 0 actually rotate


def test_rope_relative_positions():
    """Attention scores under RoPE depend only on relative offset: identical
    q/k content at positions (0, 4) and (8, 12) must produce the same score."""
    cos, sin = build_rope_cache(seq_len=32, head_dim=16, theta=10000.0)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)
    # place the same content at two absolute positions with the same offset
    scores = []
    for start in (0, 8):
        q_pos = apply_rope(torch.zeros(1, 1, 32, 16).index_copy(2, torch.tensor([start + 4]), q), cos, sin)
        k_pos = apply_rope(torch.zeros(1, 1, 32, 16).index_copy(2, torch.tensor([start]), k), cos, sin)
        scores.append((q_pos[0, 0, start + 4] @ k_pos[0, 0, start]).item())
    assert abs(scores[0] - scores[1]) < 1e-4, f"scores differ across absolute position: {scores}"


def test_param_counts():
    """Presets should land where the README says they do."""
    from config import MODEL_PRESETS
    base = GPT(MODEL_PRESETS["base"])
    total = base.num_params() / 1e6
    assert 40 < total < 50, f"base preset is {total:.1f}M params, expected ~45M"


def test_generate_shapes_and_range():
    model = make_model()
    idx = torch.randint(0, 128, (2, 5))
    out = model.generate(idx, max_new_tokens=10, temperature=1.0, top_k=20)
    assert out.shape == (2, 15)
    assert torch.all(out >= 0) and torch.all(out < 128)
    # greedy decoding must be deterministic
    g1 = model.generate(idx, max_new_tokens=10, temperature=0.0)
    g2 = model.generate(idx, max_new_tokens=10, temperature=0.0)
    assert torch.equal(g1, g2)


def test_generate_respects_vocab_limit():
    """The model vocab is padded past the tokenizer's (50257 -> 50304); ids in
    the padding range must never be sampled or the tokenizer can't decode.
    (The first smoke run crashed on exactly this -- see README, "What broke".)"""
    model = make_model()
    idx = torch.randint(0, 128, (2, 5))
    out = model.generate(idx, max_new_tokens=30, temperature=1.5, vocab_limit=50)
    assert torch.all(out[:, 5:] < 50), "sampled a token beyond vocab_limit"


def test_can_overfit_one_batch():
    """The canonical end-to-end check: a tiny model must be able to memorize
    a single batch. If loss won't collapse, something in the gradient path
    is broken even if every shape is right."""
    torch.manual_seed(0)
    model = make_model()
    idx = torch.randint(0, 128, (4, 32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(idx, targets=idx)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5, f"failed to overfit one batch: loss {loss.item():.3f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok: {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
