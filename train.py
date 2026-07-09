"""Training loop, written out by hand: mixed precision, cosine LR schedule
with warmup, gradient accumulation, gradient clipping, periodic evaluation,
CSV metrics logging, checkpointing, and sample generation at each checkpoint.

Usage:
    python data.py                          # once, to build data/*.bin
    python train.py                         # train the ~45M "base" preset
    python train.py --model small --max-steps 5000
    python train.py --resume                # continue from out/ckpt.pt

Outputs (in --out-dir, default out/):
    ckpt.pt          latest checkpoint (model + optimizer + step, resumable)
    ckpt_best.pt     checkpoint with the best val loss
    metrics.csv      step, train_loss, val_loss, lr, tokens/sec
    samples/         sample generations at every eval step
"""

import argparse
import csv
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import tiktoken
import torch

from config import MODEL_PRESETS, TrainConfig
from model import GPT, ModelConfig


def parse_args() -> tuple[TrainConfig, bool]:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=defaults.model, choices=sorted(MODEL_PRESETS))
    p.add_argument("--data-dir", default=defaults.data_dir)
    p.add_argument("--out-dir", default=defaults.out_dir)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=defaults.grad_accum_steps)
    p.add_argument("--max-steps", type=int, default=defaults.max_steps)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    p.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    p.add_argument("--eval-iters", type=int, default=defaults.eval_iters)
    p.add_argument("--device", default=defaults.device)
    p.add_argument("--dtype", default=defaults.dtype, choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pt")
    p.add_argument("--seed", type=int, default=defaults.seed)
    args = p.parse_args()

    cfg = TrainConfig(**{k: v for k, v in vars(args).items() if k != "resume"})
    cfg.min_lr = cfg.lr / 10
    return cfg, args.resume


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype != "auto":
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    if device == "cuda":
        # bf16 has fp32's exponent range, so no loss-scaling is needed; fp16
        # (pre-Ampere cards) needs GradScaler to avoid gradient underflow.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class BinDataset:
    """Random fixed-length windows from a flat uint16 token file.

    The file is memory-mapped, so the OS page cache does the buffering and
    startup is instant regardless of dataset size. Sampling random offsets
    (rather than iterating an epoch) is the standard trick for LM training:
    every window boundary is a valid training example because the stream is
    continuous text.
    """

    def __init__(self, path: str, block_size: int):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run `python data.py` first")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def get_batch(self, batch_size: int, device: str):
        ix = torch.randint(len(self.data) - self.block_size - 1, (batch_size,))
        # uint16 -> int64 copy; targets are inputs shifted one to the right
        x = torch.stack([torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to cfg.lr, then cosine decay to cfg.min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def configure_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """Weight decay on matrices only. Decaying LayerNorm gains/biases hurts:
    they are per-channel scales, not capacity, and pulling them toward zero
    fights the normalization."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=cfg.device == "cuda")


@torch.no_grad()
def estimate_loss(model, datasets, cfg: TrainConfig, autocast_ctx) -> dict[str, float]:
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, cfg.device)
            with autocast_ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def write_samples(model, enc, cfg: TrainConfig, step: int, device: str) -> None:
    """Generate from the fixed prompts and save to samples/step_XXXXXX.txt,
    so the checkpoint history doubles as a qualitative eval."""
    os.makedirs(os.path.join(cfg.out_dir, "samples"), exist_ok=True)
    path = os.path.join(cfg.out_dir, "samples", f"step_{step:06d}.txt")
    raw_model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    with open(path, "w", encoding="utf-8") as f:
        for prompt in cfg.sample_prompts:
            idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
            out = raw_model.generate(idx, max_new_tokens=cfg.sample_tokens,
                                     temperature=0.8, top_k=50, eos_token=enc.eot_token,
                                     vocab_limit=enc.n_vocab)
            f.write(f"=== prompt: {prompt!r}\n{enc.decode(out[0].tolist())}\n\n")
    model.train()


def main() -> None:
    cfg, resume = parse_args()
    device = cfg.device = resolve_device(cfg.device)
    ptdtype = resolve_dtype(cfg.dtype, device)
    device_type = "cuda" if device.startswith("cuda") else device

    torch.manual_seed(cfg.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True   # tf32 matmuls: free speed on Ampere+
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(cfg.out_dir, exist_ok=True)
    autocast_ctx = (nullcontext() if ptdtype == torch.float32
                    else torch.amp.autocast(device_type=device_type, dtype=ptdtype))
    scaler_device = "cuda" if device_type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(scaler_device, enabled=ptdtype == torch.float16)

    model_cfg = MODEL_PRESETS[cfg.model]
    model = GPT(model_cfg).to(device)
    optimizer = configure_optimizer(model, cfg)
    enc = tiktoken.get_encoding("gpt2")

    start_step, best_val = 0, float("inf")
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    if resume:
        ckpt = torch.load(ckpt_path, map_location=device)
        model_cfg = ModelConfig(**ckpt["model_config"])
        model = GPT(model_cfg).to(device)
        model.load_state_dict(ckpt["model"])
        optimizer = configure_optimizer(model, cfg)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt["best_val_loss"]
        print(f"resumed from {ckpt_path} at step {start_step}")

    if cfg.compile:
        model = torch.compile(model)

    datasets = {
        "train": BinDataset(os.path.join(cfg.data_dir, "train.bin"), model_cfg.block_size),
        "val": BinDataset(os.path.join(cfg.data_dir, "val.bin"), model_cfg.block_size),
    }

    n_params = getattr(model, "_orig_mod", model).num_params()
    n_params_ne = getattr(model, "_orig_mod", model).num_params(non_embedding=True)
    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * model_cfg.block_size
    print(f"model={cfg.model}  params={n_params / 1e6:.1f}M ({n_params_ne / 1e6:.1f}M non-embedding)")
    print(f"device={device}  dtype={ptdtype}  tokens/step={tokens_per_step:,}  "
          f"total budget={tokens_per_step * cfg.max_steps / 1e9:.2f}B tokens")

    metrics_path = os.path.join(cfg.out_dir, "metrics.csv")
    new_log = not (resume and os.path.exists(metrics_path))
    metrics_file = open(metrics_path, "w" if new_log else "a", newline="")
    metrics = csv.writer(metrics_file)
    if new_log:
        metrics.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec"])

    def save_ckpt(path: str, step: int) -> None:
        raw_model = getattr(model, "_orig_mod", model)
        torch.save({
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(cfg),
            "step": step,
            "best_val_loss": best_val,
        }, path)

    model.train()
    x, y = datasets["train"].get_batch(cfg.batch_size, device)
    t_last, tokens_since = time.time(), 0
    running_loss = None

    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # gradient accumulation: simulate a larger batch than fits in memory
        optimizer.zero_grad(set_to_none=True)
        for micro in range(cfg.grad_accum_steps):
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            # prefetch the next batch while the GPU is busy with backward
            x, y = datasets["train"].get_batch(cfg.batch_size, device)
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens_since += tokens_per_step
        loss_item = loss.item() * cfg.grad_accum_steps
        running_loss = loss_item if running_loss is None else 0.9 * running_loss + 0.1 * loss_item

        if step % cfg.log_interval == 0:
            dt = time.time() - t_last
            tok_s = tokens_since / dt if dt > 0 else 0.0
            t_last, tokens_since = time.time(), 0
            print(f"step {step:6d} | loss {running_loss:.4f} | lr {lr:.2e} | {tok_s / 1e3:.1f}k tok/s")
            metrics.writerow([step, f"{running_loss:.4f}", "", f"{lr:.6f}", f"{tok_s:.0f}"])
            metrics_file.flush()

        last_step = step == cfg.max_steps - 1
        if (step > 0 and step % cfg.eval_interval == 0) or last_step:
            losses = estimate_loss(model, datasets, cfg, autocast_ctx)
            print(f"step {step:6d} | eval: train {losses['train']:.4f}, val {losses['val']:.4f}")
            metrics.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.6f}", ""])
            metrics_file.flush()
            write_samples(model, enc, cfg, step, device)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_ckpt(os.path.join(cfg.out_dir, "ckpt_best.pt"), step)
            save_ckpt(ckpt_path, step)
            t_last, tokens_since = time.time(), 0   # don't count eval time in tok/s

    metrics_file.close()
    print(f"done. best val loss {best_val:.4f}. checkpoints in {cfg.out_dir}/")


if __name__ == "__main__":
    main()
