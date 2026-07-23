#!/bin/bash
# Submit ONE clean matched-budget comparison of the three exploration methods on the
# bug-fixed code (NEXT_STEP GAE-masking fix, PR #14). Run this ON THE CLUSTER LOGIN NODE
# after checking out the fixed commit (see doc/matched-budget-submission.md STEP 0).
#
#   bash slurm/submit_matched_matrix.sh            # submit the whole matrix
#   DRY_RUN=1 bash slurm/submit_matched_matrix.sh  # print the sbatch commands, submit nothing
#
# Matrix (all at IDENTICAL num_envs / total_timesteps / seed set -- the only intended
# difference between agents is the exploration method):
#   PPO            seeds {1,2}   ent_coef 0.01   (ppo.py standard)
#   RND            seeds {1,2}   ent_coef 0.001  (rnd.py paper value)
#   count-based    seeds {1,2}   ent_coef 0.01   (count_based.py standard)
#   RND ablation   seeds {1,2}   ent_coef 0.01   (isolates whether the ent_coef mismatch,
#                                                 not the method, drives the RND<PPO gap)
#
# Override any of these on the command line, e.g.:
#   TOTAL_TIMESTEPS=5000000 NUM_ENVS=48 SEEDS="1 2 3" bash slurm/submit_matched_matrix.sh
set -euo pipefail

# ── matched knobs (identical across every agent) ────────────────────────────
NUM_ENVS="${NUM_ENVS:-32}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
ANNEAL_LR="${ANNEAL_LR:-1}"                 # 1 = anneal ON (paper/CleanRL default)
SEEDS="${SEEDS:-1 2}"                       # >= 2 seeds per agent
ABLATION_SEEDS="${ABLATION_SEEDS:-1 2}"     # RND ent_coef=0.01 ablation seeds (1-2)

# ── matched cluster resources (confirm against your V100 partition first: ───
#    sinfo -o "%P %c %G" ; sacctmgr show qos format=Name,MaxWall) ────────────
WALLTIME="${WALLTIME:-06:00:00}"            # matched ceiling; job exits early on finish/auto-stop
GRES="${GRES:-gpu:1}"                       # HTW KI-Werkstatt gres is untyped (gpu:N); no type qualifier
PARTITION="${PARTITION:-Debug_node}"        # the only partition on kiwihead01

DRY_RUN="${DRY_RUN:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="slurm-logs/matched_matrix_${STAMP}.tsv"
mkdir -p slurm-logs

submit() {  # label  script  ent_coef  exp_name  seed
    local label="$1" script="$2" ent="$3" exp="$4" seed="$5"
    local exports="ALL,SEED=${seed},NUM_ENVS=${NUM_ENVS},TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
    exports+=",ENT_COEF=${ent},EXP_NAME=${exp},ANNEAL_LR=${ANNEAL_LR}"
    local cmd=(sbatch --parsable --partition="$PARTITION" --gres="$GRES" --time="$WALLTIME"
               --export="$exports" "$script")
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] ${label} (seed ${seed}): ${cmd[*]}"
        return
    fi
    local jid
    jid="$(command "${cmd[@]}")"
    printf "%s\t%s\t%s\n" "$jid" "$label" "seed=${seed}" | tee -a "$OUT"
}

echo "=== Matched-budget matrix: num_envs=${NUM_ENVS} total_timesteps=${TOTAL_TIMESTEPS} anneal_lr=${ANNEAL_LR} ==="
echo "=== commit: $(git rev-parse --short HEAD)  walltime=${WALLTIME} gres=${GRES} ==="
[ "$DRY_RUN" != "1" ] && echo "Job IDs -> $OUT"

for s in $SEEDS; do
    submit "PPO"          slurm/run_ppo.slurm         0.01  ppo          "$s"
    submit "RND"          slurm/run_rnd.slurm         0.001 rnd          "$s"
    submit "count-based"  slurm/run_count_based.slurm 0.01  count_based  "$s"
done
for s in $ABLATION_SEEDS; do
    submit "RND-ent0.01"  slurm/run_rnd.slurm         0.01  rnd_ent01    "$s"
done

echo "=== done. Monitor: squeue -u \$USER   |   sacct -j <id> --format=JobID,JobName,State,Elapsed,ExitCode ==="
