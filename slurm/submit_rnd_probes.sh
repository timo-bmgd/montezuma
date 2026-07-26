#!/bin/bash
# Submit the RND "leave room 1" probe matrix (see doc/regression-findings.md
# "Verification plan"). Diagnostic runs on TV_MODE=off to find whether ANY
# 32-env config gets the RND BASELINE reliably out of room 1 before committing a
# full production matrix. Run ON THE CLUSTER LOGIN NODE from the project root.
#
#   bash slurm/submit_rnd_probes.sh             # submit the default matrix (A,B,D)
#   DRY_RUN=1 bash slurm/submit_rnd_probes.sh   # print sbatch commands, submit nothing
#   RUN_ENV64=1 bash slurm/submit_rnd_probes.sh # also fire the scale cell C (num_envs=64)
#
# HONEST framing: escape = stochastic coverage of near-exit states reinforced by
# novelty before entropy decays; it scales with ENVS x FRAMES x exploratory
# window. At the 32-env cap the config levers in B are "healthier RND / longer
# window", NOT a guaranteed escape mechanism. The most likely mechanistic lever
# is C (more envs) — run it if your partition schedules >32 envs. If NOTHING
# here clears room 1, that null result IS the finding: scale is the binding
# constraint, documented as a limitation (characterisation-only scope), not
# engineered around.
#
# Budgets are 10M (not the earlier 5M): runs that escaped did so at ~4-6.4M, so a
# probe must clear ~8-10M to tell "reaches room 2 repeatedly" from "one lucky
# episode". At ~1320 SPS (32 env, A100) 10M ~ 2.1 h; env64 oversubscribes 32
# cores so ~4-5 h — the --time ceiling covers it.
#
# Matrix (all TV_MODE=off — get the baseline exploring first):
#   A  baseline   current config (up=0.25, anneal ON, ent 0.001, ext 2.0)   10M  seeds 1,2
#   B  explore    up=1.0 + constant LR + ent_coef=0.01 (healthier + longer)  10M  seeds 1,2
#   C  scale      B + num_envs=64  (THE mechanistic lever; opt-in)           10M  seed 1
#   D  pure_int   ext_coef=0 + up=1.0  (sanity: intrinsic-only)               8M  seed 1
#
# PASS = charts/rooms_visited reaches room 2 REPEATEDLY across >=2 seeds (ideally
# 3+), read from charts/rooms_furthest_cummax. If B beats A, isolate which of its
# three changes did it (they are bundled here for a first screen). If B~A and C
# helps, scale is the answer. Then re-run remote/sham/static with the winning
# flags via slurm/run_rnd_probe.slurm (TV_MODE=remote ...) for the TV matrix.
set -euo pipefail

SCRIPT="slurm/run_rnd_probe.slurm"
DRY_RUN="${DRY_RUN:-0}"
RUN_ENV64="${RUN_ENV64:-0}"
WALLTIME="${WALLTIME:-05:00:00}"        # ceiling; job exits on finish/auto-stop
GRES="${GRES:-gpu:1}"                   # HTW KI-Werkstatt gres is untyped
PARTITION="${PARTITION:-Debug_node}"    # the only partition on kiwihead01
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="slurm-logs/rnd_probes_${STAMP}.tsv"
mkdir -p slurm-logs

# submit  label  exp_name  seed  total_timesteps  update_proportion  anneal_lr  ext_coef  ent_coef  [extra exports]
submit() {
    local label="$1" exp="$2" seed="$3" ts="$4" up="$5" anneal="$6" ext="$7" ent="$8" extra="${9:-}"
    local exports="ALL,TV_MODE=off,EXP_NAME=${exp},SEED=${seed},TOTAL_TIMESTEPS=${ts}"
    exports+=",UPDATE_PROPORTION=${up},ANNEAL_LR=${anneal},EXT_COEF=${ext},ENT_COEF=${ent}"
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

# A — baseline (current config; real escape-rate reference at a fixed budget)
for s in 1 2; do
    submit "A-baseline"  rnd_probe_baseline "$s" 10000000 0.25 1 2.0 0.001
done

# B — healthier RND + longer exploratory window (up=1.0 + constant LR + 10x entropy, bundled)
for s in 1 2; do
    submit "B-explore"   rnd_probe_explore  "$s" 10000000 1.0  0 2.0 0.01
done

# C — scale: B + 64 envs (oversubscribes 32 cores; the mechanistic lever). Opt-in.
if [ "$RUN_ENV64" = "1" ]; then
    submit "C-env64"     rnd_probe_env64    1    10000000 1.0  0 2.0 0.01 "NUM_ENVS=64"
fi

# D — pure-intrinsic sanity (ext_coef=0)
submit "D-pure_int"      rnd_probe_pure_int 1     8000000 1.0  0 0.0 0.001

echo "=== done. Monitor: squeue -u \$USER | sacct -j <id> --format=JobID,JobName,State,Elapsed,ExitCode ==="
echo "=== analyse: python scripts/analyze_runs.py --logdir \$SCRATCH/montezuma/runs  (rooms_furthest_cummax is the pass/fail chart) ==="
