#!/bin/bash
# Submit the RND "leave room 1" probe matrix (see doc/regression-findings.md
# "Verification plan"). Short <=5M-step runs on TV_MODE=off to find a config that
# gets the RND BASELINE reliably out of room 1 before spending any 20M budget.
# Run ON THE CLUSTER LOGIN NODE from the project root.
#
#   bash slurm/submit_rnd_probes.sh             # submit the default matrix (P0,P1,P3)
#   DRY_RUN=1 bash slurm/submit_rnd_probes.sh   # print sbatch commands, submit nothing
#   RUN_ENV64=1 bash slurm/submit_rnd_probes.sh # also fire the 64-env cell (P2)
#
# Matrix (all TV_MODE=off — get the baseline exploring first):
#   P0  baseline        current config (update_proportion=0.25, anneal ON, ext 2.0)  3M  seeds 1,2,3
#   P1  up1_noanneal    update_proportion=1.0 + constant LR              (PRIMARY)    5M  seeds 1,2
#   P2  ..._env64       P1 + num_envs=64 (oversubscribes 32 cores)       (optional)   5M  seed 1
#   P3  pure_int        P1 + ext_coef=0  (pure-intrinsic sanity)                      3M  seed 1
#
# PASS = charts/rooms_visited reaches room 2 REPEATEDLY across >=2 seeds (ideally
# 3+). If P1 clears it, re-run remote/sham/static with the winning flags via
# slurm/run_rnd_probe.slurm (TV_MODE=remote ...) — that becomes the interpretable
# TV matrix. If NOTHING clears room 1 at 32 envs, the env cap is the binding
# constraint (document as a limitation; do not engineer around it — scope is
# characterisation only).
set -euo pipefail

SCRIPT="slurm/run_rnd_probe.slurm"
DRY_RUN="${DRY_RUN:-0}"
RUN_ENV64="${RUN_ENV64:-0}"
WALLTIME="${WALLTIME:-02:30:00}"        # ceiling; job exits on finish/auto-stop
GRES="${GRES:-gpu:1}"                   # HTW KI-Werkstatt gres is untyped
PARTITION="${PARTITION:-Debug_node}"    # the only partition on kiwihead01
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="slurm-logs/rnd_probes_${STAMP}.tsv"
mkdir -p slurm-logs

# submit  label  exp_name  seed  total_timesteps  update_proportion  anneal_lr  ext_coef  [extra exports]
submit() {
    local label="$1" exp="$2" seed="$3" ts="$4" up="$5" anneal="$6" ext="$7" extra="${8:-}"
    local exports="ALL,TV_MODE=off,EXP_NAME=${exp},SEED=${seed},TOTAL_TIMESTEPS=${ts}"
    exports+=",UPDATE_PROPORTION=${up},ANNEAL_LR=${anneal},EXT_COEF=${ext}"
    [ -n "$extra" ] && exports+=",${extra}"
    local cmd=(sbatch --parsable --partition="$PARTITION" --gres="$GRES" --time="$WALLTIME"
               --export="$exports" "$SCRIPT")
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] ${label} (seed ${seed}): ${cmd[*]}"
        return
    fi
    local jid
    jid="$(command "${cmd[@]}")"
    printf "%s\t%s\tseed=%s\n" "$jid" "$label" "$seed" | tee -a "$OUT"
}

echo "=== RND leave-room-1 probes  commit: $(git rev-parse --short HEAD)  walltime=${WALLTIME} gres=${GRES} ==="
[ "$DRY_RUN" != "1" ] && echo "Job IDs -> $OUT"

# P0 — baseline (reproduce current config; control + seed variance)
for s in 1 2 3; do
    submit "P0-baseline"   rnd_probe_baseline     "$s" 3000000 0.25 1 2.0
done

# P1 — PRIMARY: update_proportion=1.0 + constant LR
for s in 1 2; do
    submit "P1-up1_noanneal" rnd_probe_up1_noanneal "$s" 5000000 1.0 0 2.0
done

# P2 — optional: P1 + 64 envs (oversubscribes 32 cores; drop if it won't schedule)
if [ "$RUN_ENV64" = "1" ]; then
    submit "P2-env64"      rnd_probe_env64        1   5000000 1.0 0 2.0 "NUM_ENVS=64"
fi

# P3 — pure-intrinsic sanity (ext_coef=0)
submit "P3-pure_int"       rnd_probe_pure_int     1   3000000 1.0 0 0.0

echo "=== done. Monitor: squeue -u \$USER | sacct -j <id> --format=JobID,JobName,State,Elapsed,ExitCode ==="
echo "=== analyse: python scripts/analyze_runs.py --logdir \$SCRATCH/montezuma/runs  (rooms_furthest_cummax is the pass/fail chart) ==="
