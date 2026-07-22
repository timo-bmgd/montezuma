#!/usr/bin/env python3
"""Overlay-compare a set of TensorBoard runs and emit derived-event summaries.

Point it at a directory whose immediate subdirectories are runs (each containing
an ``events.out.tfevents.*`` file) and it will:

  * draw one overlaid line chart per metric (all runs on shared axes), plus a
    multi-panel overview, into ``<logdir>/figures/``;
  * compute a derived-events table (when did the agent leave room 1, peak
    intrinsic reward, final return/entropy, TV capture share, ...) written to
    ``<logdir>/summary.md`` and ``<logdir>/summary.csv`` and printed to stdout.

Built for the RND noisy-TV ablation in ``analysis/`` but works on any run set.

Usage:
    python scripts/analyze_runs.py                      # defaults to --logdir analysis
    python scripts/analyze_runs.py --logdir runs --out /tmp/figs
    tensorboard --logdir analysis                       # native overlaid view
"""
from __future__ import annotations
import argparse, glob, os, csv, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# (tag, friendly title, y-label, extra-opts)
PLOTS = [
    ("charts/rooms_visited",          "Rooms visited (exploration breadth)", "rooms",          {"rooms": True}),
    ("charts/episodic_return",        "Episodic return (true game score)",   "return",         {"smooth": True}),
    ("charts/episodic_length",        "Episode length",                      "env steps",      {"smooth": True}),
    ("charts/raw_intrinsic_rew_mean", "Raw intrinsic reward (pre-norm)",     "intrinsic",      {"smooth": True}),
    ("charts/mean_intrinsic_rew",     "Normalized intrinsic reward",         "intrinsic (norm)", {"smooth": True}),
    ("charts/tv_intrinsic_share",     "TV share of intrinsic reward",        "fraction",       {"smooth": True}),
    ("charts/tv_action_frac",         "TV-directed action fraction",         "fraction",       {"smooth": True}),
    ("losses/entropy",                "Policy entropy",                      "entropy",        {"smooth": True}),
    ("losses/explained_variance",     "Explained variance (value fit)",      "EV",             {"smooth": True}),
    ("losses/approx_kl",              "Approx KL",                           "KL",             {"smooth": True}),
]
OVERVIEW = ["charts/rooms_visited", "charts/episodic_return", "charts/raw_intrinsic_rew_mean",
            "charts/tv_intrinsic_share", "losses/entropy", "losses/explained_variance"]


def label_of(dirname: str) -> str:
    """MontezumaRevenge-v5__rnd_tv_remote__1__1784631991 -> 'rnd_tv_remote s1'."""
    parts = os.path.basename(dirname).split("__")
    if len(parts) >= 3:
        return f"{parts[1]} s{parts[2]}"
    return os.path.basename(dirname)


def meta_of(dirname: str):
    parts = os.path.basename(dirname).split("__")
    exp = parts[1] if len(parts) > 1 else os.path.basename(dirname)
    seed = parts[2] if len(parts) > 2 else "?"
    algo = "ppo" if exp.startswith("ppo") else ("rnd" if exp.startswith("rnd") else exp.split("_")[0])
    m = re.search(r"tv_([a-z-]+)", exp)
    tv = m.group(1) if m else "-"
    return algo, tv, seed


def ema(xs, ys, weight=0.9):
    if not ys:
        return xs, ys
    out, acc = [], ys[0]
    for y in ys:
        acc = acc * weight + (1 - weight) * y
        out.append(acc)
    return xs, out


def last_mean(sc, n=20):
    if not sc:
        return None
    v = [s.value for s in sc[-n:]]
    return sum(v) / len(v)


def load(logdir):
    runs = []
    for d in sorted(glob.glob(os.path.join(logdir, "*"))):
        if not os.path.isdir(d):
            continue
        if not glob.glob(os.path.join(d, "events.out.tfevents.*")):
            continue
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        runs.append((d, ea, set(ea.Tags()["scalars"])))
    return runs


def series(ea, tag):
    sc = ea.Scalars(tag)
    return [s.step for s in sc], [s.value for s in sc]


def plot_metric(runs, tag, title, ylabel, opts, colors, out):
    have = [(d, ea) for d, ea, tags in runs if tag in tags]
    if not have:
        return None
    plt.figure(figsize=(9, 5.2))
    for d, ea in have:
        xs, ys = series(ea, tag)
        if not xs:
            continue
        c = colors[label_of(d)]
        if opts.get("smooth"):
            plt.plot(xs, ys, color=c, alpha=0.15, linewidth=1)
            xs, ys = ema(xs, ys)
        plt.plot(xs, ys, color=c, linewidth=1.8, label=label_of(d))
        if opts.get("rooms"):
            cross = next((s.step for s in ea.Scalars(tag) if s.value >= 2), None)
            if cross is not None:
                plt.scatter([cross], [2], color=c, s=45, zorder=5, marker="v")
    if opts.get("rooms"):
        plt.axhline(2, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        plt.text(0.01, 0.93, "▼ = first exit from room 1", transform=plt.gca().transAxes,
                 fontsize=8, color="grey")
    plt.xlabel("global step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8, loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, tag.replace("/", "__") + ".png")
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def plot_overview(runs, colors, out):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    titledict = {t: (ti, opts) for t, ti, _, opts in PLOTS}
    for ax, tag in zip(axes.flat, OVERVIEW):
        have = [(d, ea) for d, ea, tags in runs if tag in tags]
        for d, ea in have:
            xs, ys = series(ea, tag)
            if not xs:
                continue
            c = colors[label_of(d)]
            if titledict.get(tag, ("", {}))[1].get("smooth"):
                ax.plot(xs, ys, color=c, alpha=0.12, linewidth=0.8)
                xs, ys = ema(xs, ys)
            ax.plot(xs, ys, color=c, linewidth=1.5, label=label_of(d))
        ax.set_title(titledict.get(tag, (tag, {}))[0], fontsize=10)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
    fig.suptitle("Run comparison overview", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    path = os.path.join(out, "_overview.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def fmt_step(s):
    return "never" if s is None else f"{s/1e6:.2f}M"


def summarize(runs):
    rows = []
    for d, ea, tags in runs:
        algo, tv, seed = meta_of(d)
        rv = ea.Scalars("charts/rooms_visited") if "charts/rooms_visited" in tags else []
        ret = ea.Scalars("charts/episodic_return") if "charts/episodic_return" in tags else []
        intr = ea.Scalars("charts/raw_intrinsic_rew_mean") if "charts/raw_intrinsic_rew_mean" in tags else []
        ent = ea.Scalars("losses/entropy") if "losses/entropy" in tags else []
        ev = ea.Scalars("losses/explained_variance") if "losses/explained_variance" in tags else []
        tvsh = ea.Scalars("charts/tv_intrinsic_share") if "charts/tv_intrinsic_share" in tags else []
        peak = max(intr, key=lambda s: s.value) if intr else None
        rows.append({
            "run": label_of(d), "algo": algo, "tv_mode": tv, "seed": seed,
            "max_step": max((s.step for s in (ret or rv)), default=0),
            "left_room1_step": next((s.step for s in rv if s.value >= 2), None),
            "max_rooms": int(max((s.value for s in rv), default=0)),
            "final_return_l20": round(last_mean(ret) or 0, 1),
            "max_return": round(max((s.value for s in ret), default=0), 0),
            "peak_intrinsic": round(peak.value, 3) if peak else None,
            "peak_intrinsic_step": peak.step if peak else None,
            "final_entropy_l20": round(last_mean(ent) or 0, 3) if ent else None,
            "final_tv_share_l20": round(last_mean(tvsh) or 0, 3) if tvsh else None,
            "final_expl_var_l20": round(last_mean(ev) or 0, 3) if ev else None,
        })
    return rows


def write_summary(rows, logdir):
    cols = ["run", "algo", "tv_mode", "max_rooms", "left_room1_step", "final_return_l20",
            "max_return", "peak_intrinsic", "peak_intrinsic_step", "final_entropy_l20",
            "final_tv_share_l20", "final_expl_var_l20"]
    with open(os.path.join(logdir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if c in ("left_room1_step", "peak_intrinsic_step"):
                v = fmt_step(v)
            cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    md = "# Run analysis summary\n\n" + "\n".join(lines) + "\n"
    with open(os.path.join(logdir, "summary.md"), "w") as f:
        f.write(md)
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", default="analysis", help="dir whose subdirs are runs (default: analysis)")
    ap.add_argument("--out", default=None, help="figure output dir (default: <logdir>/figures)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.logdir, "figures")
    os.makedirs(out, exist_ok=True)

    runs = load(args.logdir)
    if not runs:
        raise SystemExit(f"no runs (event files) found under {args.logdir!r}")
    print(f"loaded {len(runs)} runs from {args.logdir!r}:")
    for d, _, _ in runs:
        print("  -", label_of(d))

    cmap = plt.get_cmap("tab10")
    colors = {label_of(d): cmap(i % 10) for i, (d, _, _) in enumerate(runs)}

    made = []
    for tag, title, ylabel, opts in PLOTS:
        p = plot_metric(runs, tag, title, ylabel, opts, colors, out)
        if p:
            made.append(p)
    made.append(plot_overview(runs, colors, out))
    print(f"\nwrote {len(made)} figures to {out}/")

    rows = summarize(runs)
    md = write_summary(rows, args.logdir)
    print("\n" + md)
    print(f"summary -> {args.logdir}/summary.md and summary.csv")


if __name__ == "__main__":
    main()
