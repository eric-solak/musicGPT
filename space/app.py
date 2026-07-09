"""Gradio demo for small-lm. Deployed as a Hugging Face Space.

This only *hosts* the model -- the model itself, model.py/config.py, is the
same from-scratch PyTorch code as the rest of the repo. Nothing here uses
transformers or the HF model-loading stack; huggingface_hub is used purely
as free file storage for the checkpoint (~180MB, too big for a plain git
push without LFS).
"""

import os

import gradio as gr
import tiktoken
import torch
from huggingface_hub import hf_hub_download

from model import GPT, ModelConfig

# Set these once you've uploaded ckpt_best.pt to a Hugging Face *model* repo
# (huggingface-cli upload <username>/small-lm-checkpoint out/ckpt_best.pt).
# If CKPT_REPO is unset, falls back to a local ckpt_best.pt next to this file
# (useful for committing the checkpoint straight into the Space repo instead).
CKPT_REPO = os.environ.get("CKPT_REPO", "")
CKPT_FILENAME = os.environ.get("CKPT_FILENAME", "ckpt_best.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")


def load_model() -> GPT:
    if CKPT_REPO:
        path = hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILENAME)
    else:
        path = os.path.join(os.path.dirname(__file__), CKPT_FILENAME)
    ckpt = torch.load(path, map_location=device)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {path} (step {ckpt['step']}, best val loss {ckpt['best_val_loss']:.4f})")
    return model


model = load_model()


@torch.no_grad()
def generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    prompt = prompt.strip() or "The"
    idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=int(max_new_tokens), temperature=temperature,
        top_k=int(top_k) if top_k > 0 else None, eos_token=enc.eot_token,
        vocab_limit=enc.n_vocab,
    )
    return enc.decode(out[0].tolist())


with gr.Blocks(title="small-lm") as demo:
    gr.Markdown(
        "# small-lm: a ~45M-parameter transformer, trained from scratch\n"
        "No HuggingFace `transformers`/`datasets` in the model or training code -- "
        "attention, RoPE, and the training loop are hand-written PyTorch. "
        "Trained on Wikipedia with music articles upweighted 3x.\n\n"
        "**This is a base (completion) model, not a chatbot.** It continues your "
        "prompt the way Wikipedia would continue it -- it won't \"answer\" a question "
        "so much as write more sentences like it. See the "
        "[repo](https://github.com/<your-username>/<your-repo>) for the full writeup."
    )
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", value="The electric guitar",
                                lines=3, placeholder="Start a sentence...")
            with gr.Row():
                max_new_tokens = gr.Slider(20, 400, value=150, step=10, label="Length (tokens)")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_k = gr.Slider(0, 200, value=50, step=5, label="Top-k (0 = off)")
            run = gr.Button("Generate", variant="primary")
        with gr.Column(scale=3):
            output = gr.Textbox(label="Output", lines=12)

    run.click(generate, inputs=[prompt, max_new_tokens, temperature, top_k], outputs=output)
    gr.Examples(
        examples=[
            ["The electric guitar", 150, 0.8, 50],
            ["Jazz is a style of music that", 150, 0.8, 50],
            ["Ludwig van Beethoven was", 150, 0.8, 50],
            ["The album was released", 150, 0.8, 50],
        ],
        inputs=[prompt, max_new_tokens, temperature, top_k],
    )

if __name__ == "__main__":
    demo.launch()
