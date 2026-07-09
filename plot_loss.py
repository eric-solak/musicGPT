"""Plot training curves from out/metrics.csv to out/loss_curve.png.

Two stacked panels sharing the step axis: loss (train + val) on top, learning
rate below. Deliberately not a dual-axis chart -- two y-scales on one plot is
unreadable.

Usage:
    python plot_loss.py                 # reads out/metrics.csv
    python plot_loss.py --out-dir out2
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (light mode): slots in fixed order.
TRAIN_COLOR = "#2a78d6"   # blue
VAL_COLOR = "#1baf7a"     # aqua (low contrast on light -> lines are direct-labeled)
LR_COLOR = "#52514e"      # secondary ink: auxiliary info, not a competing series
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
MUTED = "#898781"
INK = "#0b0b0b"


def read_metrics(path: str):
    steps_t, train, steps_v, val, steps_lr, lrs = [], [], [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if row["train_loss"]:
                steps_t.append(step)
                train.append(float(row["train_loss"]))
            if row["val_loss"]:
                steps_v.append(step)
                val.append(float(row["val_loss"]))
            if row["lr"]:
                steps_lr.append(step)
                lrs.append(float(row["lr"]))
    return (steps_t, train), (steps_v, val), (steps_lr, lrs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="out")
    args = p.parse_args()

    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    (st, train), (sv, val), (slr, lrs) = read_metrics(metrics_path)

    fig, (ax, ax_lr) = plt.subplots(
        2, 1, figsize=(8, 5.5), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )
    fig.patch.set_facecolor(SURFACE)

    for a in (ax, ax_lr):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.6)
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            a.spines[spine].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9)

    ax.plot(st, train, color=TRAIN_COLOR, linewidth=1.8, label="train")
    if sv:
        ax.plot(sv, val, color=VAL_COLOR, linewidth=1.8, marker="o", markersize=4, label="val")
    # direct labels at the line ends, in addition to the legend
    if st:
        ax.annotate("train", (st[-1], train[-1]), xytext=(6, 0), textcoords="offset points",
                    color=TRAIN_COLOR, fontsize=9, fontweight="bold", va="center")
    if sv:
        ax.annotate("val", (sv[-1], val[-1]), xytext=(6, 0), textcoords="offset points",
                    color=VAL_COLOR, fontsize=9, fontweight="bold", va="center")
    ax.set_ylabel("cross-entropy loss", color=INK, fontsize=10)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    ax.set_title("Training curves", color=INK, fontsize=12, loc="left")

    ax_lr.plot(slr, lrs, color=LR_COLOR, linewidth=1.5)
    ax_lr.set_ylabel("LR", color=INK, fontsize=10)
    ax_lr.set_xlabel("step", color=INK, fontsize=10)
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    out_path = os.path.join(args.out_dir, "loss_curve.png")
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
