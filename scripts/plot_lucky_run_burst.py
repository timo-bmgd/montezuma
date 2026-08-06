"""Zoomed episode-level figure of the first acquisition burst (rnd_tv_remote seed 43).

Companion to scripts/plot_lucky_run.py (full-run overview). This one zooms into
the 7.0M-9.8M window: every multi-room episode as a stem (with the near-miss
episodes as baseline ticks), plus the iteration-level intrinsic-reward, remote-
pressing, extrinsic-value, and entropy series.

Usage (venv active, from project root):
    python scripts/plot_lucky_run_burst.py \
        --run-dir analysis/HPC-Runs/runs/ALE/MontezumaRevenge-v5__rnd_tv_remote__43__1785112914 \
        --out-dir doc/figures
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BLUE = "#2a78d6"
BLUE_MID = "#3987e5"    # 2-room episodes
BLUE_DARK = "#1c5cab"   # 4+-room episodes
INK = "#3d3d3a"
MUTED = "#8a8a85"
GRID = "#e6e6e3"

LO, HI = 7_000_000, 9_800_000
FIRST_R4 = 8_028_288
VIDEO_EP = 8_630_624    # identified via frame count (4,377 = ep length 4,376 + reset frame)
R5_EP = 8_969_984


def series(ea, tag):
    evs = ea.Scalars(tag)
    return np.array([e.step for e in evs]), np.array([e.value for e in evs])


def rolling(x, n=5):
    return np.convolve(x, np.ones(n) / n, mode="same")


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def line_panel(ax, steps, vals, title, ylim=None):
    smooth = rolling(vals)  # smooth the full series first: no fake dips at window edges
    m = (steps >= LO) & (steps <= HI)
    x = steps[m] / 1e6
    ax.plot(x, vals[m], color=BLUE, linewidth=0.8, alpha=0.35)
    ax.plot(x, smooth[m], color=BLUE, linewidth=2, solid_capstyle="round")
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("doc/figures"))
    args = ap.parse_args()

    ea = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0})
    ea.Reload()

    rs, rv = series(ea, "charts/rooms_visited")
    es, ev = series(ea, "charts/episodic_return")

    fig, axes = plt.subplots(
        3, 1, figsize=(6.3, 5.6), sharex=True,
        gridspec_kw={"hspace": 0.5, "height_ratios": [1.5, 1, 1]}, dpi=200,
    )

    # --- panel A: the episodes themselves ---
    ax = axes[0]
    rooms_at = dict(zip(rs.tolist(), rv.tolist()))
    m = (es >= LO) & (es <= HI)
    near_miss = [(s, v) for s, v in zip(es[m], ev[m])
                 if v >= 100 and rooms_at.get(s, 1) < 2]
    for s, _ in near_miss:
        ax.plot([s / 1e6], [0.55], marker="|", color=MUTED, markersize=7,
                markeredgewidth=1.1)
    mm = (rs >= LO) & (rs <= HI) & (rv >= 2)
    for s, r in zip(rs[mm], rv[mm]):
        color = BLUE_DARK if r >= 4 else BLUE_MID
        ax.plot([s / 1e6, s / 1e6], [0, r], color=color, linewidth=1.6,
                solid_capstyle="round", zorder=3)
        ax.plot([s / 1e6], [r], marker="o", color=color, markersize=4, zorder=4)
    ax.set_ylim(0, 6.4)
    ax.set_yticks([2, 4, 5])
    ax.set_title("Rooms visited per episode (stems: episodes that left room 1; "
                 "gray ticks: scored but died in room 1)", loc="left", fontsize=9, color=INK)
    ax.annotate("first 4-room\nepisode", xy=(FIRST_R4 / 1e6, 4), xytext=(-6, 14),
                textcoords="offset points", ha="right", color=INK, fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.annotate("recorded video\nepisode", xy=(VIDEO_EP / 1e6, 4), xytext=(-14, 22),
                textcoords="offset points", ha="right", color=INK, fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.annotate("5 rooms, second key\n(17,005-step episode)", xy=(R5_EP / 1e6, 5),
                xytext=(8, 6), textcoords="offset points", ha="left",
                color=INK, fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.annotate("2 rooms", xy=(9.53, 2), xytext=(6, 2), textcoords="offset points",
                color=BLUE_MID, fontsize=8, fontweight="bold")
    ax.annotate("4+ rooms", xy=(9.44, 4), xytext=(6, 2), textcoords="offset points",
                color=BLUE_DARK, fontsize=8, fontweight="bold")

    # --- iteration-level panels ---
    line_panel(axes[1], *series(ea, "charts/raw_intrinsic_rew_mean"),
               "Raw intrinsic reward (batch mean, per iteration)")
    ax = axes[2]
    line_panel(ax, *series(ea, "charts/tv_action_frac"),
               "Remote-press fraction  charts/tv_action_frac", ylim=(0, 0.27))
    ax.axhline(1 / 19, color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
    ax.annotate("chance 1/19", xy=(HI / 1e6, 1 / 19), xytext=(0, 3),
                textcoords="offset points", ha="right", color=MUTED, fontsize=7.5)
    ax.set_xlabel("environment steps (millions)", fontsize=9, color=INK)

    for ax in axes:
        style_axis(ax)
        ax.set_xlim(LO / 1e6, HI / 1e6)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"lucky-run-burst_s43.{ext}", bbox_inches="tight")
    print("wrote", args.out_dir / "lucky-run-burst_s43.{png,pdf}")


if __name__ == "__main__":
    main()
