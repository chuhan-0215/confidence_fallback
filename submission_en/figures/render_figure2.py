#!/usr/bin/env python3
"""Render Figure 2 — Nature/Science-style four-panel results."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ROOT_FIG = OUT.parents[1] / "figures"

# Palette inspired by Nature/Science confidence & ablation figures
COL_GREY = "#BDBDBD"          # baseline / fixed budget
COL_GREY_DARK = "#6E6E6E"
COL_CYAN_L = "#87CEEB"        # light blue (primary series)
COL_CYAN_M = "#5BAFBF"        # medium teal-blue
COL_TEAL = "#2A9D8F"          # ours / CF
COL_TEAL_D = "#1F6F65"
COL_CORAL = "#E8988A"         # universal / warning
COL_PURPLE = "#8E7CC3"        # accent
COL_RECOVER = "#6CA07C"       # positive audit
COL_MISLEAD = "#C76B6B"       # negative audit
COL_BOTH = "#A8A8A8"
COL_AXIS = "#1A1A1A"
COL_GRID = "#ECECEC"
COL_MUTED = "#4A4A4A"

METHODS_SHORT = ["Fixed 3", "BFS", "Depth", "Univ.", "CF"]
METHODS_LEGEND = ["Fixed 3", "Auto BFS", "Depth", "Universal", "CF"]
VALUES = [83.8, 93.1, 93.6, 92.6, 95.23]
REFINE_PCT = [0.0, 0.0, 0.0, 100.0, 7.2]
DELTAS = [0.0, 9.3, 9.8, 8.8, 11.4]
COLORS = [COL_GREY, COL_CYAN_L, COL_CYAN_M, COL_CORAL, COL_TEAL]
BASELINE = 83.8

CROSS_NAMES = [
    "ProsQA",
    "ProntoQA",
    "GSM8K (self)",
    "GSM8K cp14",
    "GSM8K cp25",
    "StrategyQA",
    "ProofWriter",
    "MATH500",
]
CROSS_BASE = [83.8, 99.50, 27.07, 31.31, 32.98, 60.87, 99.75, 8.00]
CROSS_CF = [95.23, 99.62, 28.35, 31.84, 33.66, 61.49, 99.75, 14.00]
CROSS_DELTA = [11.4, 0.12, 1.28, 0.53, 0.68, 0.62, 0.00, 6.00]

AUDIT_LABELS = ["Refine recovers", "Refine misleads", "Both wrong"]
AUDIT_VALS = [14, 5, 17]
AUDIT_COLORS = [COL_RECOVER, COL_MISLEAD, COL_BOTH]
N_TOTAL = 419


def _fmt_acc(v: float) -> str:
    if v > 99 and v < 100:
        return f"{v:.2f}"
    if abs(v - round(v, 1)) < 0.05:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _style_axes(ax, grid: str = "y") -> None:
    """Clean white axes with thin black spines and light grey grid."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COL_AXIS)
        spine.set_linewidth(0.6)
    ax.tick_params(
        axis="both",
        colors=COL_AXIS,
        length=3.5,
        width=0.6,
        labelsize=7.6,
        direction="out",
        pad=3,
    )
    ax.set_facecolor("white")
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=COL_GRID, linewidth=0.55, zorder=0)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=COL_GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def _panel_title(ax, letter: str, title: str) -> None:
    ax.text(
        -0.11,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=COL_AXIS,
    )
    ax.text(
        -0.02,
        1.08,
        title,
        transform=ax.transAxes,
        fontsize=8.2,
        va="bottom",
        ha="left",
        color=COL_MUTED,
    )


def _bar_edge(col: str) -> str:
    return COL_TEAL_D if col == COL_TEAL else COL_GREY_DARK


def render() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )

    fig = plt.figure(figsize=(7.35, 5.35), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.45],
        hspace=0.46,
        wspace=0.38,
        left=0.10,
        right=0.90,
        top=0.93,
        bottom=0.10,
    )

    # --- (A) ProsQA ablation ---
    ax_a = fig.add_subplot(gs[0, 0])
    _panel_title(ax_a, "A", "ProsQA ablation (N=419)")
    x = list(range(len(METHODS_SHORT)))
    bars = ax_a.bar(
        x,
        VALUES,
        width=0.58,
        color=COLORS,
        edgecolor=[_bar_edge(c) for c in COLORS],
        linewidth=0.55,
        zorder=3,
    )
    bars[-1].set_linewidth(0.85)
    ax_a.axhline(BASELINE, color=COL_GREY_DARK, linestyle=(0, (4, 3)), linewidth=0.75, zorder=2, alpha=0.65)
    ax_a.set_ylim(82.0, 100.5)
    ax_a.set_ylabel("Accuracy (%)", labelpad=6)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(METHODS_SHORT, fontsize=7.6, rotation=10, ha="right")
    ax_a.set_yticks([85, 90, 95])
    _style_axes(ax_a, grid="y")

    for i, (bar, val, delta) in enumerate(zip(bars, VALUES, DELTAS)):
        cx = bar.get_x() + bar.get_width() / 2
        if i == 0:
            ax_a.text(cx, val + 0.45, _fmt_acc(val), ha="center", va="bottom", fontsize=7.4, color=COL_MUTED)
        else:
            label = f"{_fmt_acc(val)}\n(+{delta:.1f})"
            ax_a.text(
                cx,
                val + 0.45,
                label,
                ha="center",
                va="bottom",
                fontsize=7.4 if i < len(bars) - 1 else 7.8,
                fontweight="bold" if i == len(bars) - 1 else "normal",
                color=COL_TEAL if i == len(bars) - 1 else COL_AXIS,
                linespacing=0.9,
                clip_on=False,
            )

    # --- (B) Accuracy vs refinement ---
    ax_b = fig.add_subplot(gs[0, 1])
    _panel_title(ax_b, "B", "Accuracy vs. refinement rate")
    jitter = [-3.5, 0.0, 3.5, 0.0, 0.0]
    sizes = [34, 28, 28, 28, 52]
    for name, acc, ref, col, jx, sz in zip(METHODS_LEGEND, VALUES, REFINE_PCT, COLORS, jitter, sizes):
        is_cf = name == "CF"
        ax_b.scatter(
            ref + jx,
            acc,
            s=sz,
            c=col,
            edgecolors=COL_AXIS if is_cf else "none",
            linewidths=0.55 if is_cf else 0,
            alpha=0.92 if not is_cf else 1.0,
            zorder=4 if is_cf else 3,
        )
    ax_b.axhline(93.6, color=COL_CYAN_M, linestyle=(0, (3, 2)), linewidth=0.7, alpha=0.85, zorder=1)
    ax_b.text(
        52,
        94.35,
        "Depth-aware 93.6%",
        fontsize=6.3,
        color=COL_MUTED,
        ha="center",
        va="bottom",
    )
    ax_b.annotate(
        "CF",
        xy=(7.2, 95.23),
        xytext=(22, 97.1),
        fontsize=7.2,
        fontweight="bold",
        color=COL_TEAL,
        arrowprops=dict(arrowstyle="-", color=COL_TEAL, lw=0.55),
        clip_on=False,
    )
    ax_b.plot([100, 100], [92.6, 92.6], marker="o", color=COL_CORAL, markersize=4.5, zorder=3)
    ax_b.annotate(
        "Universal",
        xy=(100, 92.6),
        xytext=(62, 90.8),
        fontsize=6.4,
        color=COL_CORAL,
        arrowprops=dict(arrowstyle="-", color=COL_CORAL, lw=0.5),
        ha="center",
        clip_on=False,
    )
    ax_b.set_xlabel("Refinement rate (%)", labelpad=5)
    ax_b.set_ylabel("Accuracy (%)", labelpad=6)
    ax_b.set_xlim(-5, 105)
    ax_b.set_ylim(82.0, 99.0)
    ax_b.set_xticks([0, 50, 100])
    ax_b.set_yticks([85, 90, 95])
    _style_axes(ax_b, grid="both")
    b_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markeredgecolor="none", markersize=5.2, label=n)
        for n, col in zip(METHODS_LEGEND, COLORS)
    ]
    b_handles[-1].set_markeredgecolor(COL_AXIS)
    b_handles[-1].set_markeredgewidth(0.55)
    b_handles[-1].set_markersize(6.2)
    ax_b.legend(
        handles=b_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.50),
        fontsize=6.0,
        frameon=False,
        handlelength=0.9,
        handletextpad=0.35,
        labelspacing=0.35,
    )

    # --- (C) Complement audit ---
    ax_c = fig.add_subplot(gs[1, 0])
    _panel_title(ax_c, "C", "Complement audit (419 instances)")
    y_pos = [0, 1, 2]
    ax_c.barh(
        y_pos,
        AUDIT_VALS,
        height=0.52,
        color=AUDIT_COLORS,
        edgecolor=COL_AXIS,
        linewidth=0.45,
        zorder=3,
        alpha=0.88,
    )
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(AUDIT_LABELS, fontsize=7.6)
    ax_c.set_xlabel("Instance count", labelpad=5)
    ax_c.set_xlim(0, 24)
    ax_c.invert_yaxis()
    _style_axes(ax_c, grid="x")

    for y, val in zip(y_pos, AUDIT_VALS):
        pct = 100.0 * val / N_TOTAL
        label = f"{val} ({pct:.1f}%)"
        ax_c.text(
            val + 0.55,
            y,
            label,
            va="center",
            ha="left",
            fontsize=7.4,
            color=COL_AXIS,
            clip_on=True,
        )

    # --- (D) Cross-dataset transfer ---
    ax_d = fig.add_subplot(gs[1, 1])
    _panel_title(ax_d, "D", "Cross-dataset transfer")

    row_gap = 1.12
    y_d = [i * row_gap for i in range(len(CROSS_NAMES))]
    bar_h = 0.38
    pair_gap = 0.18
    offset = bar_h / 2 + pair_gap / 2
    ax_d.barh(
        [y - offset for y in y_d],
        CROSS_BASE,
        height=bar_h,
        color=COL_GREY,
        edgecolor=COL_GREY_DARK,
        linewidth=0.45,
        zorder=3,
        alpha=0.95,
    )
    ax_d.barh(
        [y + offset for y in y_d],
        CROSS_CF,
        height=bar_h,
        color=COL_TEAL,
        edgecolor=COL_TEAL_D,
        linewidth=0.45,
        zorder=3,
        alpha=0.95,
    )
    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(CROSS_NAMES, fontsize=7.4)
    ax_d.set_xlabel("Accuracy (%)", labelpad=5)
    ax_d.set_xlim(0, 118)
    ax_d.set_xticks([0, 20, 40, 60, 80, 100])
    ax_d.set_ylim(max(y_d) + 0.85, min(y_d) - 0.85)
    ax_d.invert_yaxis()
    _style_axes(ax_d, grid="x")
    ax_d.tick_params(axis="y", pad=6)
    ax_d.axvline(100, color=COL_GRID, linewidth=0.6, zorder=1)

    for y, delta, cf, base in zip(y_d, CROSS_DELTA, CROSS_CF, CROSS_BASE):
        if delta == 0:
            d_s = "0.0"
        elif delta < 1:
            d_s = f"+{delta:.2f}"
        else:
            d_s = f"+{delta:.1f}"
        bar_right = max(base, cf)
        ax_d.text(
            bar_right + 1.6,
            y,
            d_s,
            va="center",
            ha="left",
            fontsize=7.8,
            fontweight="bold" if delta == max(CROSS_DELTA) else "normal",
            color=COL_TEAL_D if delta == max(CROSS_DELTA) else COL_TEAL,
            clip_on=True,
        )

    # legend outside plot, on the right
    ax_d.legend(
        handles=[
            Patch(facecolor=COL_GREY, edgecolor=COL_GREY_DARK, label="Fixed budget"),
            Patch(facecolor=COL_TEAL, edgecolor=COL_TEAL_D, label="CF"),
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.50),
        fontsize=6.5,
        frameon=False,
        handlelength=1.1,
        labelspacing=0.45,
        borderaxespad=0.0,
    )

    for ext in ("pdf", "png"):
        path = OUT / f"figure2_ablation.{ext}"
        fig.savefig(path, dpi=320 if ext == "png" else None, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        print(f"Wrote {path}")

    ROOT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT_FIG / "figure2_ablation.pdf", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(ROOT_FIG / "figure2_ablation.png", dpi=320, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    render()
