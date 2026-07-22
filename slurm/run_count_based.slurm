#!/bin/bash
#SBATCH --job-name=montezuma-count
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# total-timesteps=50,000,000 targets Tang et al. 2017's own reported result (15
# rooms, ~3,700 score at 50M frames -- see CLAUDE.md's literature benchmark table).
# Unlike RND, count-based exploration has no gradient-based bonus network to
# overfit, so it isn't subject to the entropy/intrinsic-collapse failure mode
# diagnosed for RND -- this is the higher-confidence near-term bet of the two.
#
# 20h walltime is a rough estimate, not a measured budget -- if the job times out
# before finishing, resubmit with --resume (see below) to continue from the last
# checkpoint; --checkpoint-interval already makes this safe to chain across
# multiple walltime-limited jobs.
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --export=ALL,SEED=1 slurm/run_count_based.slurm
#
# To resume a preempted/timed-out job, add to the python command below:
#   --resume $SCRATCH/checkpoints/<run_name>/ckpt_XXXXXX.pt
#
# --cpus-per-task=32 matches --num-envs 1:1 -- AsyncVectorEnv (the default) spawns
# one subprocess per env, so fewer CPUs than envs means workers queue for CPU time.
# This has NOT been verified against the real partition -- run:
#   sinfo -o "%P %c %G"
# before submitting. If the partition has fewer than 32 cores/GPU, lower
# --cpus-per-task to match; SLURM does not reject an over-large request, it queues
# the job indefinitely in PD state, silently burning walltime budget. Only reduce
# --num-envs as a last resort (it changes batch_size/num_iterations, so it isn't a
# free lever).

module purge
module load python/3.11
module load cuda/12.4

# Untested hypothesis, not yet confirmed by scripts/profile_throughput.py: constrain
# PyTorch's CPU thread pool so it doesn't additionally oversubscribe the same cores
# the AsyncVectorEnv worker subprocesses are already contending for.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

SCRATCH="${SCRATCH:-/scratch/$USER}/montezuma"
mkdir -p "$SCRATCH"/{runs,checkpoints,slurm-logs}

python -c "import torch; assert torch.cuda.is_available(), 'No GPU detected!'; print('GPU:', torch.cuda.get_device_name(0))"

RESUME_ARGS=()
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARGS=(--resume "$RESUME_FROM")
fi

# ── Matched-budget matrix overrides ─────────────────────────────────────────
# All optional; the defaults below preserve this script's standalone behavior.
# Set for a matched comparison via: sbatch --export=ALL,SEED=1,TOTAL_TIMESTEPS=...,NUM_ENVS=...
# See slurm/submit_matched_matrix.sh and doc/matched-budget-submission.md.
NUM_ENVS="${NUM_ENVS:-32}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000000}"
ENT_COEF="${ENT_COEF:-0.01}"               # count_based.py's standard default
EXP_NAME="${EXP_NAME:-count_based}"
ANNEAL_ARGS=()
if [ "${ANNEAL_LR:-1}" = "0" ] || [ "${ANNEAL_LR:-1}" = "false" ]; then
    ANNEAL_ARGS=(--no-anneal-lr)           # default: anneal ON (paper/CleanRL default)
fi

python src/agents/count_based.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name "$EXP_NAME" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --ent-coef "$ENT_COEF" \
    --seed "${SEED:-1}" \
    --capture-video \
    --record-room-discovery \
    --track \
    --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    --checkpoint-interval 100 \
    "${ANNEAL_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
