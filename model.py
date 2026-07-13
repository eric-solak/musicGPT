"""A decoder-only transformer language model, implemented from scratch.

Every core component is written out by hand rather than pulled from
torch.nn's high-level modules:

  * LayerNorm        - manual mean/variance normalization
  * RoPE             - rotary positional embeddings applied to Q and K
  * Attention        - explicit QK^T / sqrt(d) -> mask -> softmax -> V
  * Transformer block, weight init, and autoregressive generation

The only nn primitives used are Linear, Embedding, Dropout, and Parameter --
i.e. matrix multiplies and lookup tables, not pre-built transformer layers.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50304   # GPT-2 BPE is 50257; padded up to a multiple of 64 for faster matmuls
    block_size: int = 512     # max context length
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    flash: bool = True        # use fused SDPA kernel for speed; False = the manual reference path

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


class LayerNorm(nn.Module):
    """LayerNorm written out manually: y = (x - mean) / sqrt(var + eps) * g + b.

    Statistics are computed in float32 even under mixed precision, because
    variance of bf16 activations loses too much precision.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return (x * self.weight.float() + self.bias.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device=None):
    """Precompute the cos/sin tables for rotary positional embeddings.

    RoPE rotates each (even, odd) pair of channels in Q and K by an angle
    proportional to the token's absolute position, so that Q.K^T ends up
    depending only on *relative* position. Frequencies follow the original
    paper: theta^(-2i/d) for channel pair i.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)          # (seq_len, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)         # (seq_len, head_dim)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the channel pairs of x by the cached angles.

    x: (B, n_head, T, head_dim). Uses the "rotate half" formulation:
    [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin], applied per position.
    """
    T = x.size(2)
    cos, sin = cos[:T], sin[:T]                        # (T, head_dim)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


def apply_repetition_penalty(logits: torch.Tensor, idx: torch.Tensor,
                             penalty: float) -> torch.Tensor:
    """Discourage tokens that already appear in the context (CTRL-style).

    logits: (B, vocab), idx: (B, T) of already-generated/context token ids.
    Seen tokens with positive logits are divided by the penalty, negative
    ones multiplied, so the push is always away from re-selection. penalty
    > 1 suppresses repeats; 1.0 is a no-op. This is a sampling-time patch
    for the repeat loops under-trained models fall into, not a fix for them.
    """
    scores = logits.gather(-1, idx)
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits.scatter(-1, idx, scores)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with the math written out.

    softmax(QK^T / sqrt(head_dim) + causal_mask) @ V

    With config.flash=True this dispatches to F.scaled_dot_product_attention,
    which computes the identical function with a fused kernel (see
    tests/test_model.py for the equivalence check). The manual path is kept
    as the readable reference implementation.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.flash = config.flash
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        cos, sin = build_rope_cache(config.block_size, self.head_dim, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, nh, T, T)
            att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att.float(), dim=-1).to(q.dtype)              # softmax in fp32 for stability
            att = self.attn_dropout(att)
            y = att @ v                                                   # (B, nh, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: Linear -> GELU -> Linear, 4x expansion."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.proj = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x)).

    Pre-norm (norm inside the residual branch) keeps the residual stream an
    identity path, which trains stably without the LR warmup gymnastics that
    post-norm (original 2017 layout) requires.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: the input embedding and output projection share one
        # matrix. Saves vocab_size*d_model params (the single biggest tensor
        # at this scale) and acts as a regularizer.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual-branch output projections:
        # each block adds two contributions to the residual stream, so scale
        # their init by 1/sqrt(2*n_layer) to keep the stream's variance ~1
        # at depth. Without this, deeper models start with large activations
        # and the first steps of training are spent recovering.
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()   # lm_head is tied, so subtract once
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"

        x = self.drop(self.wte(idx))     # positions are injected via RoPE inside attention
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
            return logits, loss
        # Inference: only the last position's logits are needed for sampling.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 repetition_penalty: float | None = None,
                 eos_token: int | None = None, vocab_limit: int | None = None) -> torch.Tensor:
        """Autoregressive sampling with temperature, top-k, nucleus (top-p),
        and an optional repetition penalty.

        vocab_limit masks ids >= limit before sampling. The model's vocab is
        padded past the tokenizer's (50257 -> 50304 for matmul efficiency),
        and an under-trained model will happily sample those padding ids --
        which the tokenizer then cannot decode.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if vocab_limit is not None:
                logits[:, vocab_limit:] = float("-inf")
            if repetition_penalty is not None and repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, idx_cond, repetition_penalty)

            if temperature <= 0:                       # greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # keep the smallest set of tokens whose cumulative prob >= top_p
                    cutoff = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)
            if eos_token is not None and (next_token == eos_token).all():
                break
        return idx
