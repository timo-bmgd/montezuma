#!/bin/bash
# Fire the patch-response probe sweep as one CPU job per (seed, arm) -- the
# repeatable, cluster-native version of the hand-assembled probe command in
# doc/noisy-tv-results.md section 2. Each job traces ONE run's full checkpoint
# trajectory (content_sensitivity / patch_contribution vs step); together they
# cover the 3 seeds x 4 RND arms. Run ON THE CLUSTER LOGIN NODE from the project
# root; the jobs themselves are CPU-only and non-blocking (they neither need a
# GPU nor compete with the training matrix for one).
#
#   bash slurm/submit_probe_sweep.sh              # submit the full 3x4 matrix
#   DRY_RUN=1 bash slurm/submit_probe_sweep.sh    # print sbatch lines, submit nothing
#   SEEDS="42" bash slurm/submit_probe_sweep.sh   # just one seed (e.g. as its runs land)
#   ARMS="remote static" bash slurm/submit_probe_sweep.sh   # subset of arms
#   NUM_FRAMES=2048 bash slurm/submit_probe_sweep.sh        # ~half wall-clock per job
#
# Re-running after new checkpoints appear (e.g. once seed 44 finishes) just
# re-globs whatever exists -- that is the whole point. A job whose run has no
# checkpoints yet exits 2 with a clear message and does not block the others.
#
# PPO arms are intentionally absent: PPO checkpoints carry no RND predictor, so
# there is nothing to probe -- their TV story is behavioural (charts/tv_action_frac,
# key-grab), read from the event files via scripts/analyze_runs.py.
set -euo pipefail

SCRIPT="slurm/run_probe_sweep.slurm"
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-42 43 44}"
ARMS="${ARMS:-off remote sham-remote static}"
WALLTIME="${WALLTIME:-03:00:00}"        # ceiling; a run's ~49 ckpts take ~1-1.5 h
PARTITION="${PARTITION:-Debug_node}"    # the only partition on kiwihead01
NUM_FRAMES="${NUM_FRAMES:-4096}"
NUM_FRESH="${NUM_FRESH:-8}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="slurm-logs/probe_sweep_${STAMP}.tsv"
mkdir -p slurm-logs

# submit  seed  arm
submit() {
    local seed="$1" arm="$2"
    local exports="ALL,SEED=${seed},TV_MODE=${arm},NUM_FRAMES=${NUM_FRAMES},NUM_FRESH=${NUM_FRESH}"
    local cmd=(sbatch --parsable --partition="$PARTITION" --time="$WALLTIME"
               --export="$exports" "$SCRIPT")
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] seed=${seed} arm=${arm}: ${cmd[*]}"
        return
    fi
    local jid
    jid="$(command "${cmd[@]}")"
    printf "%s\t%s\tseed=%s\n" "$jid" "$arm" "$seed" | tee -a "$OUT"
}

echo "=== patch-response probe sweep  commit: $(git rev-parse --short HEAD)  seeds='${SEEDS}' arms='${ARMS}' frames=${NUM_FRAMES} ==="
[ "$DRY_RUN" != "1" ] && echo "Job IDs -> $OUT"

for s in $SEEDS; do
    for arm in $ARMS; do
        submit "$s" "$arm"
    done
done

echo "=== done. Monitor: squeue -u \$USER | sacct -j <id> --format=JobID,JobName,State,Elapsed,ExitCode ==="
echo "=== aggregate+plot: python scripts/plot_probe_trajectory.py --indir analysis/probe ==="
