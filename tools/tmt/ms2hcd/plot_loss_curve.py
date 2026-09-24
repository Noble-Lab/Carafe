#!/usr/bin/env python
"""Plot the training history of one MS2 model run: loss per epoch and the learning-rate schedule.

Reads ms2_history.tsv (epoch, lr, train_loss, test_loss) from a run directory written by
train_tmt_ms2.py / run_sweep.py and writes a light and a dark version for README.md:

    python plot_loss_curve.py --run sweep/C --out loss_curve        # loss_curve_light.png, _dark.png

"test_loss" in the history is ai.py's internal validation split, used to keep the best epoch; it is
labelled "validation" here to keep it apart from the held-out test set (evaluate_heldout.py). The
kept epoch is the one with the lowest validation loss. Loss is shown on a log scale because the
first epochs are several times higher than the rest. Loss and learning rate have different scales,
so they get two panels on a shared epoch axis rather than two y-axes on one.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical slots 1-2 and chart chrome, light and dark (each passes the palette checks
# against its own surface).
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "train": "#2a78d6", "val": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "train": "#3987e5", "val": "#d95926"},
}


def plot(history: pd.DataFrame, title: str, theme: dict, out_path: str):
    best = history.loc[history["test_loss"].idxmin()]
    lw = 2 * 72 / 100  # 2 px at 100 px per inch figure units -> points

    fig, (ax, ax_lr) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 4.6), dpi=100,
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

    series = (("train_loss", theme["train"], "training"), ("test_loss", theme["val"], "validation"))
    for column, colour, label in series:
        ax.plot(history["epoch"], history[column], color=colour, linewidth=lw,
                solid_capstyle="round", solid_joinstyle="round", label=label)
    ax.set_yscale("log")
    # plain decimal ticks on the log axis, a few only
    lo, hi = history[["train_loss", "test_loss"]].min().min(), history[["train_loss", "test_loss"]].max().max()
    ticks = [t for t in (0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.1) if lo * 0.9 <= t <= hi * 1.1]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # direct labels at the line ends, pushed apart if closer than one label height
    fig.canvas.draw()
    last = history.iloc[-1]
    ends = sorted(((ax.transData.transform((last["epoch"], last[c]))[1], label) for c, _, label in series))
    min_gap = 11 * fig.dpi / 72  # about one 8 pt line
    placed = []
    for y, label in ends:
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, label))
    for (y_px, label), (y0, _) in zip(placed, ends):
        ax.annotate(label, (last["epoch"], ax.transData.inverted().transform((0, y0))[1]),
                    xytext=(6, (y_px - y0) * 72 / fig.dpi), textcoords="offset points",
                    va="center", fontsize=8, color=theme["ink2"])
    ax.set_ylabel("loss (log scale)", fontsize=8, color=theme["ink2"])

    # the kept epoch: marker with a surface ring, a hairline down both panels, a text label
    ax.plot([best["epoch"]], [best["test_loss"]], marker="o", markersize=8,
            markerfacecolor=theme["val"], markeredgecolor=theme["surface"], markeredgewidth=2, zorder=5)
    for a in (ax, ax_lr):
        a.axvline(best["epoch"], color=theme["axis"], linewidth=0.8, zorder=0)
    ax.annotate(f"kept: epoch {int(best['epoch'])}, validation loss {best['test_loss']:.5f}",
                (best["epoch"], best["test_loss"]), xytext=(-8, 34), textcoords="offset points",
                ha="right", fontsize=8, color=theme["ink2"])

    legend = ax.legend(loc="upper right", fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(theme["ink2"])

    ax_lr.plot(history["epoch"], history["lr"], color=theme["ink2"], linewidth=lw, solid_capstyle="round")
    ax_lr.set_ylabel("learning rate", fontsize=8, color=theme["ink2"])
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_lr.yaxis.get_offset_text().set_color(theme["muted"])
    ax_lr.yaxis.get_offset_text().set_fontsize(8)
    ax_lr.set_xlabel("epoch", fontsize=8, color=theme["ink2"])
    ax_lr.set_xlim(0.5, history["epoch"].max() + 4)

    fig.suptitle(title, fontsize=10, color=theme["ink"], x=0.07, ha="left")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=theme["surface"])
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory holding ms2_history.tsv")
    ap.add_argument("--out", required=True, help="output prefix; writes <prefix>_light.png and <prefix>_dark.png")
    ap.add_argument("--title", help="figure title (default: from the run directory name)")
    args = ap.parse_args(argv)

    history = pd.read_csv(os.path.join(args.run, "ms2_history.tsv"), sep="\t")
    title = args.title or f"MS2 training history, run {os.path.basename(os.path.normpath(args.run))}"
    for mode, theme in THEMES.items():
        path = f"{args.out}_{mode}.png"
        plot(history, title, theme, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
