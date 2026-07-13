"""Generate text from a trained checkpoint.

Usage:
    python sample.py --prompt "Once upon a time"
    python sample.py --ckpt out/ckpt_best.pt --temperature 0.7 --top-p 0.95 -n 300
"""

import argparse

import tiktoken
import torch

from model import GPT, ModelConfig
from train import resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt_best.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("-n", "--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.2,
                   help="penalize tokens already in the context; 1.0 disables. "
                        "1.1-1.3 tames the repeat loops of early checkpoints")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (step {ckpt['step']}, best val loss {ckpt['best_val_loss']:.4f})\n")

    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode(args.prompt)], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                             eos_token=enc.eot_token, vocab_limit=enc.n_vocab)
        print(f"--- sample {i + 1} ---")
        print(enc.decode(out[0].tolist()))
        print()


if __name__ == "__main__":
    main()
