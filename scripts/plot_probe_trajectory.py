"""Aggregate + plot the patch-response probe sweep (scripts/probe_patch_response.py
output) across all checkpoints, arms, and seeds.

Reads every `probe_*.csv` in --indir (each = one run's full checkpoint trajectory,
produced by slurm/run_probe_sweep.slurm), groups rows into (arm, seed) series, and
draws the metrics vs global_step:

  content_sensitivity  -- THE money plot. Does the predictor's error still depend on
      the patch CONTENT (memorisation channel open, G>0 exploitable) or has it
      collapsed to the content-invariant conditional-mean solution (H1's G=0)?
      Elevated early then decaying = the memorisation gap CLOSING over training =
      self-limiting capture DEMONSTRATED, not inferred.
  patch_contribution   -- predictor-level occlusion share (err_displayed - err_blank),
      the per-checkpoint analogue of charts/tv_intrinsic_share; expect remote/static
      to RISE (the TV takes a growing share of a shrinking error budget).
  G_proxy              -- shown for completeness; EXPECTED ~0 at --tv-refresh-every 1
      (all current runs). Its ~0 is not a null result; it is the T=1 end of P3.

Interpretation guard (also written into summary_probe.md): read the remote/static
decay AGAINST the off/sham baselines -- general predictor convergence also lowers
content_sensitivity, so the TV-specific claim is the remote-minus-off gap, not the
absolute level.

Usage:
    python scripts/plot_probe_trajectory.py --indir analysis/probe
    python scripts/plot_probe_trajectory.py --indir analysis/probe --smooth 0.6
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed visual identity so figures are comparable across re-runs.
ARM_COLOR = {
    "off":         "#7f7f7f",  # grey  -- the no-patch floor
    "sham-remote": "#ff7f0e",  # orange -- action-space control, also a floor
    "remote":      "#d62728",  # red   -- PRIMARY (behavioural channel)
    "static":      "#1f77b4",  # blue  -- signal degradation, no behavioural channel
}
SEED_STYLE = {"42": "-", "43": "--", "44": ":"}
METRIC_COLS = ("err_displayed", "err_blank", "err_fresh_mean", "err_fresh_std",
               "patch_contribution", "content_sensitivity", "G_proxy")


def fmt_step(x):
    x = float(x)
    return f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k"


def seed_of(row, fallback):
    """Seed lives in the run_name (the `checkpoint` column), which embeds env_id
    with a '/': MontezumaRevenge-v5__rnd_tv_remote__42__<ts> -> parts[-2]."""
    parts = row.get("checkpoint", "").split("__")
    if len(parts) >= 2 and parts[-2].isdigit():
        return parts[-2]
    return fallback


def load(indir):
    """-> {(arm, seed): [row, ...] sorted by global_step}"""
    series = defaultdict(list)
    files = sorted(glob.glob(os.path.join(indir, "probe_*.csv")))
    if not files:
        raise SystemExit(f"no probe_*.csv found in {indir}")
    for path in files:
        # filename convention probe_rnd_tv_<arm>__<seed>.csv is the seed fallback
        base = os.path.basename(path)[:-4]
        fb_seed = base.rsplit("__", 1)[-1] if "__" in base else "?"
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                arm = r.get("tv_mode", "off")
                seed = seed_of(r, fb_seed)
                for c in ("global_step", "iteration"):
                    r[c] = int(float(r.get(c, 0)))
                for c in METRIC_COLS:
                    r[c] = float(r.get(c, "nan"))
                series[(arm, seed)].append(r)
    for k in series:
        series[k].sort(key=lambda r: r["global_step"])
    return series


def ema(ys, w):
    if w <= 0:
        return ys
    out, m = [], ys[0]
    for y in ys:
        m = w * m + (1 - w) * y
        out.append(m)
    return out


def label(arm, seed):
    return f"{arm}·s{seed}"


def plot_metric(series, metric, ylabel, title, path, smooth, axhline=None):
    plt.figure(figsize=(9, 5.2))
    for (arm, seed) in sorted(series, key=lambda k: (list(ARM_COLOR).index(k[0])
                                                     if k[0] in ARM_COLOR else 9, k[1])):
        rows = series[(arm, seed)]
        xs = [r["global_step"] for r in rows]
        ys = [r[metric] for r in rows]
        color = ARM_COLOR.get(arm, "#333333")
        ls = SEED_STYLE.get(seed, "-.")
        # faint raw + smoothed line (or just raw with markers if smoothing off)
        if smooth > 0:
            plt.plot(xs, ys, color=color, ls=ls, lw=0.8, alpha=0.25)
            plt.plot(xs, ema(ys, smooth), color=color, ls=ls, lw=1.8, label=label(arm, seed))
        else:
            plt.plot(xs, ys, color=color, ls=ls, lw=1.6, marker="o", ms=2.5,
                     label=label(arm, seed))
    if axhline is not None:
        plt.axhline(axhline, color="k", lw=0.8, ls=":", alpha=0.5)
    plt.xlabel("global step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8, loc="best", ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def write_summary(series, path):
    rows = []
    for (arm, seed) in sorted(series):
        rs = series[(arm, seed)]
        final = rs[-1]
        cs = [r["content_sensitivity"] for r in rs]
        pc = [r["patch_contribution"] for r in rs]
        rows.append({
            "arm": arm, "seed": seed, "n_ckpts": len(rs),
            "final_step": final["global_step"],
            "patch_contrib_final": round(final["patch_contribution"], 4),
            "patch_contrib_peak": round(max(pc), 4),
            "content_sens_peak": round(max(cs), 4),
            "content_sens_final": round(final["content_sensitivity"], 4),
            "content_sens_drop": round(max(cs) - final["content_sensitivity"], 4),
            "G_proxy_final": round(final["G_proxy"], 4),
        })
    cols = list(rows[0].keys())
    with open(path + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(path + ".md", "w") as fh:
        fh.write("# Patch-response probe — cross-arm/seed summary\n\n")
        fh.write("One row per (arm, seed). `content_sens_drop = peak − final` "
                 "quantifies the memorisation gap **closing** over training "
                 "(large positive for `remote`/`static` = self-limiting demonstrated).\n\n")
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
        fh.write(
            "\n**Read against the floor.** `off`/`sham-remote` are the baselines: "
            "general predictor convergence lowers `content_sensitivity` for every arm, "
            "so the TV-specific claim is the `remote`/`static` drop *in excess of* the "
            "`off` drop — not the absolute level. `G_proxy≈0` is expected at "
            "`--tv-refresh-every 1` (the T=1 end of P3), not a null result.\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="analysis/probe", help="dir of probe_*.csv")
    ap.add_argument("--out", default=None, help="figures dir (default <indir>/figures)")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="EMA weight [0,1); 0 = raw points with markers")
    args = ap.parse_args()
    outdir = args.out or os.path.join(args.indir, "figures")
    os.makedirs(outdir, exist_ok=True)

    series = load(args.indir)
    made = []
    made.append(plot_metric(
        series, "content_sensitivity", "content_sensitivity  (std/mean of err over fresh patches)",
        "Patch content-sensitivity vs training  (gap closing = self-limiting)",
        os.path.join(outdir, "content_sensitivity_vs_step.png"), args.smooth, axhline=0.0))
    made.append(plot_metric(
        series, "patch_contribution", "patch_contribution  (err_displayed − err_blank)",
        "Predictor-level occlusion share vs training  (P1)",
        os.path.join(outdir, "patch_contribution_vs_step.png"), args.smooth, axhline=0.0))
    made.append(plot_metric(
        series, "G_proxy", "G_proxy  (err_fresh_mean − err_displayed)",
        "Memorisation gap proxy vs training  (≈0 expected at refresh_every=1)",
        os.path.join(outdir, "G_proxy_vs_step.png"), args.smooth, axhline=0.0))

    summ = write_summary(series, os.path.join(args.indir, "summary_probe"))

    print(f"series: {len(series)}  ({', '.join(sorted(f'{a}·s{s}' for a, s in series))})")
    for p in made:
        print("wrote", p)
    print("wrote", os.path.join(args.indir, "summary_probe.md"), "(+ .csv)")
    print("\narm/seed        n  final_step  patch_contrib(fin)  content_sens peak->final (drop)")
    for r in summ:
        print(f"{r['arm']:>12}·s{r['seed']:<3} {r['n_ckpts']:>2}  "
              f"{fmt_step(r['final_step']):>9}  {r['patch_contrib_final']:>10}  "
              f"{r['content_sens_peak']:>8} -> {r['content_sens_final']:<8} "
              f"({r['content_sens_drop']:+})")


if __name__ == "__main__":
    main()
