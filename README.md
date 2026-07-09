# musicGPT: a ~45M-parameter language model from scratch

A decoder-only transformer built and trained end-to-end — no `transformers`, no `datasets`, no `nn.TransformerDecoder`. The attention math, rotary positional embeddings, layer norm, weight init, training loop, and even the Wikipedia-markup stripper are written out by hand. The point of this repo is to demonstrate that I understand what's inside an LLM, not that I can call one.

- **Model:** 44.6M params (18.9M non-embedding), 6 layers, 8 heads, d=512, 512-token context
- **Data:** streamed directly from official [Wikipedia dumps](https://dumps.wikimedia.org), with music-related articles upweighted 3× (I wanted it to be good at music)
- **Tokenizer:** GPT-2 BPE via `tiktoken` (the one external component — see [trade-offs](#what-id-do-differently-at-scale))
- **Hardware target:** one consumer GPU, hours–days, not weeks
- **Dependencies:** `torch`, `numpy`, `tiktoken`, `requests`. That's it.

## Results

<!-- TODO after training: commit out/loss_curve.png, out/metrics.csv, and a few
     out/samples/step_*.txt files, then fill in this section:
     - final train/val loss
     - a before/after sample (step 500 gibberish vs final checkpoint)
     - tokens/sec and wall-clock time on your GPU -->

*Training in progress — loss curves and sample generations land here.*

```
python plot_loss.py        # out/metrics.csv -> out/loss_curve.png
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build the dataset (streams the dump over HTTP; nothing else touches disk)
python data.py                            # Simple English Wikipedia — quick start
python data.py --source enwiki --parts 4  # English Wikipedia — the real run (~1B tokens)

# 2. Train (checkpoints, metrics.csv, and sample generations land in out/)
python train.py                           # ~45M "base" preset
python train.py --model small             # ~30M, faster
python train.py --resume                  # continue from out/ckpt.pt

# 3. Generate
python sample.py --prompt "The electric guitar" --temperature 0.8

# 4. Tests (CPU, <1 min): attention equivalence, causality, RoPE geometry, overfit-one-batch
python -m pytest tests/ -q
```

`train.py` auto-selects CUDA and bf16 when available and falls back to fp16
(pre-Ampere) or CPU/fp32, so the repo runs anywhere — but do the real training
on a GPU. At the default settings one step processes 65,536 tokens; a mid-range
card (RTX 3060-class) does the Chinchilla-ish 0.9B-token budget in roughly a
day, a 3090/4090 in a few hours.

## Repo layout

```
model.py       the transformer: LayerNorm, RoPE, causal attention, blocks, generation
config.py      model presets (debug/small/base) + training hyperparameters
data.py        Wikipedia dump -> plain text -> GPT-2 BPE -> train.bin/val.bin
train.py       training loop: AMP, cosine LR, grad accumulation, eval, checkpoints, samples
sample.py      generate from a checkpoint (temperature / top-k / top-p)
plot_loss.py   metrics.csv -> loss_curve.png
tests/         correctness tests for the things that break silently
```

## Architecture, and why

Decoder-only, pre-norm, GPT-style — the same shape as every modern LLM, at 1/10,000th scale.

| Choice | What I did | Why |
|---|---|---|
| Norm placement | **Pre-norm** (norm inside the residual branch) | Post-norm (the original 2017 layout) makes the residual stream pass through LayerNorm, which destabilizes deep-ish stacks; pre-norm keeps an identity path and trains without fragile warmup tuning. |
| Positions | **RoPE** (rotary embeddings), implemented by hand | Rotating Q/K channel pairs by position makes attention scores depend only on *relative* offset — no learned position table, and it's what Llama-class models use. `tests/` verifies both the rotation property and the relative-position property. |
| Attention | **Manual** `softmax(QKᵀ/√d + mask)V`, with an optional fused-kernel path | The hand-written path is the reference; `flash=True` dispatches to `F.scaled_dot_product_attention` for the actual training run (same math, fused kernel — a test asserts they agree to 1e-5). Softmax runs in fp32 even under bf16 autocast. |
| LayerNorm | Written out (mean/var/rsqrt), stats in fp32 | The variance of bf16 activations is where mixed-precision LMs quietly rot. |
| Embeddings | **Tied** input/output matrix | At this scale the 50304×512 embedding is 58% of all parameters — tying saves 25M params and regularizes. (It also bit me — see below.) |
| Vocab | GPT-2's 50257, **padded to 50304** (×64) | GPU matmuls are measurably faster when the vocab dim is a multiple of 64. (This also bit me — see below.) |
| Init | N(0, 0.02), residual projections scaled by 1/√(2·n_layers) | Each block adds two branches to the residual stream; scaling their output init keeps activation variance ~1 at depth instead of growing linearly. |
| Optimizer | AdamW (β₂=0.95, wd=0.1 on matrices only), cosine LR + warmup, grad clip 1.0 | GPT-3 settings, standard at this scale. Weight decay is *not* applied to LayerNorm gains/biases — decaying a normalization scale toward zero fights the normalization. |
| Data loading | `np.memmap` over a flat uint16 token stream, random windows | No DataLoader, no padding, no epochs: every offset in a continuous token stream is a valid training example, and the OS page cache does the buffering. uint16 halves the disk footprint (50257 < 2¹⁶). |

## The data: Wikipedia, with a music bias

`data.py` streams `pages-articles` dumps straight from dumps.wikimedia.org over
HTTP → bz2 → XML, strips wiki markup with a hand-rolled set of regexes
(templates and tables are nested, so they're peeled innermost-out), and
tokenizes into a flat binary. Articles whose categories or infoboxes look
music-related (~keyword classifier: *album, song, composer, jazz, guitar,
symphon-, …*) are written **3×** to the training stream. Since training samples
random windows, duplication *is* upweighting — no sampler changes needed.
Validation articles are held out entirely and never duplicated, so val loss
stays honest.

**Honest expectations:** a 45M-param base model is a *completion* model, not a
chatbot. Trained on this mix it should continue "The saxophone is" with
coherent, music-literate encyclopedic prose — it will not reliably *answer*
"who wrote Kind of Blue?". Getting question-answering behavior would take a
small supervised fine-tune on Q&A pairs afterward (see below), which the same
training loop could do with a different data file.

## What broke

Real failures from building this, kept because they're the instructive part:

1. **Weight tying leaks the current token into its own logits.** My init-loss
   sanity test ("an untrained model should score ~ln(V)") failed at 3.95 vs the
   expected 4.85. Cause: I'd lazily used `targets=inputs`, i.e. asked the model
   to predict the token it was currently looking at — and with tied embeddings,
   position *t*'s hidden state starts as `wte[token_t]`, whose dot product with
   the tied output matrix is largest at… `token_t`. Even a random model "knows"
   what token it's standing on. The test now draws targets independently, and
   `tests/test_model.py::test_init_loss_near_uniform` documents it.

2. **Vocab padding crashes the tokenizer.** I padded the embedding to 50304
   for matmul speed; `tiktoken` only knows ids < 50257. An *untrained* model
   samples uniformly-ish, so the very first checkpoint's sample generation hit
   a padding id and died with `KeyError: 50296` — mid-training-run, in the
   checkpoint code path. Fix: `generate(..., vocab_limit=50257)` masks the
   padding ids to −inf before sampling. Lesson: any id the model *can* emit is
   an id your decoder must handle, and "can emit" includes rows you never train.

3. **Wikitext is not one regex.** Templates nest (`{{cite {{...}}}}`), tables
   nest, and a single greedy pattern either under-strips (markup soup in the
   training data) or eats whole paragraphs. The stripper peels innermost
   constructs in a loop, and I keep a decode-and-eyeball check on the first
   tokens of `train.bin` — markup that survives becomes what the model learns.

Watch out for, from the same category of silent failure: fp16 without
GradScaler underflows gradients (auto-handled here — bf16 where supported,
scaler otherwise), and eval time polluting tokens/sec metrics (timer resets
after eval). Also practical: if this repo lives in OneDrive/Dropbox, exclude
`out/` and `data/` from sync — a 1GB memmap being re-uploaded every 500 steps
while checkpoints write is misery.

## What I'd do differently at scale

- **Train my own tokenizer.** GPT-2's BPE is the one off-the-shelf component
  here, and it's frozen to 2019 web text. At scale, vocab is a tuning knob
  (size, byte-fallback, domain coverage), and I'd train BPE on my own corpus.
- **KV cache for generation.** `generate()` recomputes the full forward pass
  per token — O(T²) total. Fine for 200-token samples during training; the
  first thing to fix for real inference.
- **Real data pipeline.** Regex-stripping wikitext is "good enough for
  pretraining", not good: I'd use a proper parser, dedupe near-duplicates
  (Wikipedia is full of formulaic pages), and mix sources rather than
  oversampling one domain by crude keyword matching. Data quality is the
  highest-leverage knob at every scale.
- **Distributed training.** This repo is deliberately single-GPU. The next
  step is DDP (data parallelism), then, past ~1B params, sharded optimizers
  (ZeRO/FSDP) — the optimizer states are 2× the model in fp32.
- **Proper eval.** Val loss and eyeballing samples is fine at 45M. At scale:
  held-out perplexity per domain, and task benchmarks, tracked per checkpoint.
- **A fine-tuning stage.** To actually *answer* music questions: freeze this
  as the base model, build a few thousand Q&A pairs, and run the same loop at
  low LR on formatted examples. The infrastructure here already supports it —
  it's just a different `train.bin`.

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) — the transformer
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019) — GPT-2: decoder-only recipe, init scaling
- Su et al., [*RoFormer*](https://arxiv.org/abs/2104.09864) (2021) — rotary position embeddings
- Hoffmann et al., [*Training Compute-Optimal LLMs*](https://arxiv.org/abs/2203.15556) (2022) — the ~20 tokens/param rule used to size the token budget
- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — the spiritual ancestor of every small-GPT repo, including this one; the memmap batching trick is his
