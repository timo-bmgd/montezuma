#!/bin/bash
#SBATCH --job-name=montezuma-ppo
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=Debug_node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Lower risk than run_rnd.slurm: PPO's prior 5M-step production run (ppo_5m_s1)
# succeeded, reaching room 2 with nonzero return -- no entropy-collapse feedback
# loop, since there's no intrinsic-reward network to overfit (see
# doc/decisions.md "RND vs PPO asymmetry" entry). Kept close to its previous
# 10M-step shape rather than redesigned from scratch.
#
# As of this round of changes, ppo.py also auto-stops (--auto-stop, on by
# default) if an entropy + frozen-PPO-update collapse is sustained -- a
# simpler variant than rnd.py's (no intrinsic-reward term, since PPO has
# none). Its default thresholds are PROVISIONAL: PPO has no prior collapse
# incident on this codebase to calibrate against, unlike RND's thresholds
# which are backed by doc/10M-RND-run-failure-documentation.md's real
# numbers. Check `sacct -j <id> --format=ExitCode` after this run: 42:0 means
# auto-stop fired (re-check whether it was a real collapse or a
# too-sensitive threshold before trusting it blindly), 0:0 means it ran to
# completion.
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --export=ALL,SEED=1 slurm/run_ppo.slurm
#
# To extend a completed or walltime-truncated run:
#   sbatch --export=ALL,SEED=1,RESUME_FROM=$SCRATCH/montezuma/checkpoints/<run_name>/ckpt_XXXXXX.pt slurm/run_ppo.slurm
# (the resume-fix in this round of changes makes this land in the same
# run_name directory instead of fragmenting into a new one -- see doc/decisions.md)
#
# Resource requests below are CONSERVATIVE GENERIC DEFAULTS pending your
# cluster's real numbers -- see doc/hpc-onboarding.md §1 and run
# `sinfo -o "%P %c %G"` before submitting. --cpus-per-task=32 matches
# --num-envs 1:1 -- AsyncVectorEnv (the default) spawns one subprocess per
# env, so fewer CPUs than envs means workers queue for CPU time. SLURM does
# not reject an over-large CPU request outright, it queues the job
# indefinitely in PD state -- see doc/hpc-onboarding.md §1. Only reduce
# --num-envs as a last resort (it changes batch_size/num_iterations).

# HTW KI-Werkstatt has NO python/cuda environment modules -- Python is a dedicated conda
# env, CUDA comes from the torch cu124 wheel's bundled runtime (see slurm/setup_hpc.sh and
# doc/matched-budget-submission.md). CONDA_ENV_PREFIX must match what setup_hpc.sh created.
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/$USER/conda-envs/montezuma}"
if [ -f /usr/etc/profile.d/conda.sh ]; then
    source /usr/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV_PREFIX"

# Untested hypothesis, not yet confirmed by scripts/profile_throughput.py: constrain
# PyTorch's CPU thread pool so it doesn't additionally oversubscribe the same cores
# the AsyncVectorEnv worker subprocesses are already contending for.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Under sbatch the script is copied to the spool dir, so $0 is NOT the repo path.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from -- submit from the repo root
# (the submit scripts / doc do). Fall back to $0-relative for direct `bash slurm/...` runs.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

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
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
ENT_COEF="${ENT_COEF:-0.01}"               # ppo.py's standard default
EXP_NAME="${EXP_NAME:-ppo}"
ANNEAL_ARGS=()
if [ "${ANNEAL_LR:-1}" = "0" ] || [ "${ANNEAL_LR:-1}" = "false" ]; then
    ANNEAL_ARGS=(--no-anneal-lr)           # default: anneal ON (paper/CleanRL default)
fi

python src/agents/ppo.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name "$EXP_NAME" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --ent-coef "$ENT_COEF" \
    --seed "${SEED:-1}" \
    --capture-video \
    --record-room-discovery \
    --checkpoint-interval 100 \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    "${ANNEAL_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
