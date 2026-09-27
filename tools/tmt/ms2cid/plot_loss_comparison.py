#!/usr/bin/env python
"""Plot the training histories of two MS2 model runs on one figure.

Reads ms2_history.tsv (epoch, lr, train_loss, test_loss) from each run directory and writes a light
and a dark version:

    python plot_loss_comparison.py --run "G=sweep/G=the HCD TMT model" \
        --run "H=sweep/H=random weights" --out loss_curve_G_vs_H --title "..."

Colour marks the run and line style the loss: validation solid, training dashed. Loss is on a log
scale. The runs must share one learning-rate schedule, which is drawn once below, as the rate each
epoch trained at (0 in the first epoch, then the value ai.py recorded after the previous epoch).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ms2hcd"))
from plot_loss_curve import THEMES  # noqa: E402

TICKS = (0.005, 0.007, 0.01, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15)


def plot(runs, title, theme, out_path):
    """runs: [(name, description, history)], two of them, sharing one lr schedule."""
    lw = 2 * 72 / 100
    colours = [theme["train"], theme["val"]]
    fig, (ax, ax_lr) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 4.8), dpi=100,
                                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    fig.patch.set_facecolor(theme["surface"])
    for a in (ax, ax_lr):
        a.set_facecolor(theme["surface"])
        a.grid(axis="y", color=theme["grid"], linewidth=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(theme["axis"])
        a.tick_params(which="both", colors=theme["muted"], labelsize=8, length=0)

    for (name, description, h), colour in zip(runs, colours):
        ax.plot(h["epoch"], h["test_loss"], color=colour, linewidth=lw,
                solid_capstyle="round", solid_joinstyle="round", label=f"{name} validation")
        ax.plot(h["epoch"], h["train_loss"], color=colour, linewidth=lw * 0.7, linestyle=(0, (4, 2)),
                label=f"{name} training")
        best = h.loc[h["test_loss"].idxmin()]
        ax.plot([best["epoch"]], [best["test_loss"]], marker="o", markersize=5, markerfacecolor=colour,
                markeredgecolor=theme["surface"], markeredgewidth=1.5, zorder=5)
        first = h.iloc[0]
        ax.annotate(f"{name}, epoch 1: {description}", (first["epoch"], first["test_loss"]),
                    xytext=(10, -2), textcoords="offset points", va="center", fontsize=8, color=colour)

    ax.set_yscale("log")
    losses = pd.concat([h[["train_loss", "test_loss"]] for _, _, h in runs])
    lo, hi = losses.min().min(), losses.max().max()
    ax.yaxis.set_major_locator(FixedLocator([t for t in TICKS if lo * 0.9 <= t <= hi * 1.1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("loss (log scale)", fontsize=8, color=theme["ink2"])

    # the four lines end close together, so a legend rather than labels at the line ends
    legend = ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    for text in legend.get_texts():
        text.set_color(theme["ink2"])
    x_end = max(h["epoch"].max() for _, _, h in runs)

    kept = [(name, h.loc[h["test_loss"].idxmin()]) for name, _, h in runs]
    epochs = {int(b["epoch"]) for _, b in kept}
    values = ", ".join(f"{name} {b['test_loss']:.5f}" for name, b in kept)
    text = (f"kept: epoch {epochs.pop()}, validation loss {values}" if len(epochs) == 1
            else "kept: " + ", ".join(f"{name} epoch {int(b['epoch'])} ({b['test_loss']:.5f})" for name, b in kept))
    ax.annotate(text, (kept[0][1]["epoch"], kept[0][1]["test_loss"]), xytext=(-8, 40), textcoords="offset points",
                ha="right", fontsize=8, color=theme["ink2"])
    for a in (ax, ax_lr):
        for e in {int(b["epoch"]) for _, b in kept}:
            a.axvline(e, color=theme["axis"], linewidth=0.8, zorder=0)

    h0 = runs[0][2]
    lr = pd.Series([0.0] + h0["lr"].tolist()[:-1], index=h0.index)
    ax_lr.plot(h0["epoch"], lr, color=theme["ink2"], linewidth=lw, solid_capstyle="round")
    ax_lr.set_ylabel("learning rate used", fontsize=8, color=theme["ink2"])
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_lr.yaxis.get_offset_text().set_color(theme["muted"])
    ax_lr.yaxis.get_offset_text().set_fontsize(8)
    ax_lr.set_xlabel("epoch", fontsize=8, color=theme["ink2"])
    ax_lr.set_xlim(0.5, x_end + 2)

    fig.suptitle(title, fontsize=10, color=theme["ink"], x=0.07, ha="left")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR[=DESCRIPTION]",
                    help="a run directory holding ms2_history.tsv, and what its first epoch is; give two")
    ap.add_argument("--out", required=True, help="output prefix; writes <prefix>_light.png and <prefix>_dark.png")
    ap.add_argument("--title", required=True, help="figure title")
    args = ap.parse_args(argv)
    runs = []
    for spec in args.run:
        parts = spec.split("=", 2)
        if len(parts) < 2:
            ap.error("--run expects NAME=DIR[=DESCRIPTION]")
        name, run_dir = parts[0], parts[1]
        description = parts[2] if len(parts) == 3 else "the starting model"
        runs.append((name, description, pd.read_csv(os.path.join(run_dir, "ms2_history.tsv"), sep="\t")))
    if len(runs) != 2:
        ap.error("give exactly two --run")
    if not runs[0][2]["lr"].reset_index(drop=True).equals(runs[1][2]["lr"].reset_index(drop=True)):
        ap.error("the runs do not share one learning-rate schedule")
    for mode, theme in THEMES.items():
        path = f"{args.out}_{mode}.png"
        plot(runs, args.title, theme, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
