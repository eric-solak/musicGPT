"""Model presets and the training configuration.

Sizes assume the GPT-2 BPE vocab (50257, padded to 50304) with tied
embeddings. The embedding matrix alone is vocab_size * d_model params, which
dominates at this scale -- see the README for why that shapes these choices.
"""

from dataclasses import dataclass, field

from model import ModelConfig

MODEL_PRESETS: dict[str, ModelConfig] = {
    # ~1.7M non-embedding params. For CPU debugging and the test suite.
    "debug": ModelConfig(n_layer=4, n_head=4, d_model=128, block_size=256),
    # ~30M total (10.6M non-embedding). Trains fast; good first real run.
    "small": ModelConfig(n_layer=6, n_head=6, d_model=384, block_size=512),
    # ~45M total (18.9M non-embedding). The default. Chinchilla-optimal
    # token budget (~20 tokens/param) is ~0.9B tokens -- a few parts of the
    # English Wikipedia dump, and a comfortable single-GPU job.
    "base": ModelConfig(n_layer=6, n_head=8, d_model=512, block_size=512),
}


@dataclass
class TrainConfig:
    # data / io
    data_dir: str = "data"
    out_dir: str = "out"
    model: str = "base"                 # key into MODEL_PRESETS

    # batch: effective batch = batch_size * grad_accum_steps * block_size tokens
    batch_size: int = 32
    grad_accum_steps: int = 4           # 32 * 4 * 512 = 65,536 tokens/step

    # optimizer (GPT-3 style settings, standard for this scale)
    max_steps: int = 20_000             # ~1.3B tokens at the default batch
    lr: float = 6e-4
    min_lr: float = 6e-5                # cosine decays to lr/10
    warmup_steps: int = 500
    weight_decay: float = 0.1           # applied only to matrices, not norms/biases
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # evaluation / logging / checkpoints
    eval_interval: int = 500            # steps between val-loss evals + checkpoints
    eval_iters: int = 100               # batches averaged per eval
    log_interval: int = 20              # steps between train-loss log lines
    sample_tokens: int = 200            # length of sample generations at each checkpoint
    # Completion-style prompts (the model is a base LM, not a chatbot):
    # music-flavored to track how well the domain upweighting is landing.
    sample_prompts: tuple[str, ...] = (
        "The electric guitar",
        "Jazz is a style of music that",
        "Ludwig van Beethoven was",
        "The album was released",
    )

    # system
    device: str = "auto"                # auto -> cuda if available, else mps/cpu
    dtype: str = "auto"                 # auto -> bfloat16 on Ampere+, float16 on older cuda, float32 on cpu
    compile: bool = False               # torch.compile: ~1.5-2x faster, slow first step
    seed: int = 1337
