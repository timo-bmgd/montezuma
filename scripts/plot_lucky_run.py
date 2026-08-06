"""Figure for the discussion-chapter section on the 'lucky' RND run.

Plots the acquisition / near-extinction / re-acquisition timeline of
MontezumaRevenge-v5__rnd_tv_remote__43 (HPC noisy-TV batch): per-bin fraction of
episodes leaving room 1, mean episodic return, extrinsic value estimate, and
policy entropy over the 20M-step budget.

Usage (venv active, from project root):
    python scripts/plot_lucky_run.py \
        --run-dir analysis/HPC-Runs/runs/ALE/MontezumaRevenge-v5__rnd_tv_remote__43__1785112914 \
        --out-dir doc/figures
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BLUE = "#2a78d6"        # single-series panels
BLUE_MID = "#3987e5"    # rooms >= 2
BLUE_DARK = "#1c5cab"   # rooms >= 4 (nested subset: same hue, darker step)
INK = "#3d3d3a"
MUTED = "#8a8a85"
GRID = "#e6e6e3"

BIN = 500_000
TOTAL = 20_000_000

EVENTS = [
    (4_404_832, "first key + door\n(return 400)"),
    (8_028_288, "first 4-room episodes;\nrecorded video episode"),
    (16_500_000, "re-acquisition,\ncritic consolidates"),
]


def series(ea, tag):
    evs = ea.Scalars(tag)
    return np.array([e.step for e in evs]), np.array([e.value for e in evs])


def binned(steps, vals, reducer):
    edges = np.arange(0, TOTAL + BIN, BIN)
    centers = (edges[:-1] + edges[1:]) / 2
    out = np.full(len(centers), np.nan)
    idx = np.digitize(steps, edges) - 1
    for i in range(len(centers)):
        m = idx == i
        if m.any():
            out[i] = reducer(vals[m])
    return centers, out


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("doc/figures"))
    args = ap.parse_args()

    ea = EventAccumulator(str(args.run_dir), size_guidance={"scalars": 0})
    ea.Reload()

    rs, rv = series(ea, "charts/rooms_visited")
    es, ev = series(ea, "charts/episodic_return")
    ents, entv = series(ea, "losses/entropy")
    xvs, xvv = series(ea, "charts/ext_value_mean")

    c, frac2 = binned(rs, (rv >= 2).astype(float), lambda x: 100 * x.mean())
    _, frac4 = binned(rs, (rv >= 4).astype(float), lambda x: 100 * x.mean())
    _, mret = binned(es, ev, np.mean)
    ce, ment = binned(ents, entv, np.mean)
    cx, mxv = binned(xvs, xvv, np.mean)

    fig, axes = plt.subplots(
        4, 1, figsize=(6.3, 6.6), sharex=True,
        gridspec_kw={"hspace": 0.42}, dpi=200,
    )
    x = c / 1e6

    ax = axes[0]
    ax.plot(x, frac2, color=BLUE_MID, linewidth=2, solid_capstyle="round")
    ax.plot(x, frac4, color=BLUE_DARK, linewidth=2, solid_capstyle="round")
    ax.annotate("left room 1", xy=(x[-1], frac2[~np.isnan(frac2)][-1]),
                xytext=(4, 4), textcoords="offset points",
                color=BLUE_MID, fontsize=8, fontweight="bold")
    ax.annotate("reached 4+ rooms", xy=(x[-1], frac4[~np.isnan(frac4)][-1]),
                xytext=(4, -11), textcoords="offset points",
                color=BLUE_DARK, fontsize=8, fontweight="bold")
    ax.set_title("Episodes leaving room 1 (% per 500k-step bin)",
                 loc="left", fontsize=9, color=INK)
    ax.set_ylim(0, None)

    ax = axes[1]
    ax.plot(x, mret, color=BLUE, linewidth=2, solid_capstyle="round")
    ax.set_title("Mean episodic return (game score, 500k-step bins)",
                 loc="left", fontsize=9, color=INK)
    ax.set_ylim(0, None)

    ax = axes[2]
    ax.plot(cx / 1e6, mxv, color=BLUE, linewidth=2, solid_capstyle="round")
    ax.set_title("Extrinsic value estimate  charts/ext_value_mean",
                 loc="left", fontsize=9, color=INK)

    ax = axes[3]
    ax.plot(ce / 1e6, ment, color=BLUE, linewidth=2, solid_capstyle="round")
    ax.axhline(np.log(19), color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
    ax.annotate("uniform policy  ln 19 ≈ 2.94", xy=(19.8, np.log(19)),
                xytext=(0, -9), textcoords="offset points",
                ha="right", color=MUTED, fontsize=7.5)
    ax.set_title("Policy entropy (nats)", loc="left", fontsize=9, color=INK)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("environment steps (millions)", fontsize=9, color=INK)

    for ax in axes:
        style_axis(ax)
        ax.set_xlim(0, 20)
        for step, _ in EVENTS:
            ax.axvline(step / 1e6, color=MUTED, linewidth=0.8,
                       linestyle=(0, (2, 3)), zorder=0)
    for i, (step, label) in enumerate(EVENTS):
        y = axes[0].get_ylim()[1] * (0.55 if i == 2 else 1.0)
        ha, dx = ("right", -3) if i == 2 else ("left", 3)
        axes[0].annotate(label, xy=(step / 1e6, y),
                         xytext=(dx, -2), textcoords="offset points",
                         va="top", ha=ha, color=MUTED, fontsize=7.5)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"lucky-run-rnd_tv_remote_s43.{ext}",
                    bbox_inches="tight")
    print("wrote", args.out_dir / "lucky-run-rnd_tv_remote_s43.{png,pdf}")


if __name__ == "__main__":
    main()
