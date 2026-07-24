#!/bin/bash
#SBATCH --job-name=montezuma-rnd
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Debug_node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Deliberately NOT the full paper-scale run (Burda et al. 2018 used
# num_envs=128, total_timesteps~2e9 -- see rnd.py's own module docstring).
# That's infeasible to commit to blind, especially with the collapse history
# below. This script is sized as a modest, checkpoint-chainable FIRST
# ATTEMPT: 3,000,000 steps at num_envs=32 (batch_size=4096, ~732 iterations).
#
# Context: both prior 10M-step seeds (rnd_10m_s1/s2, documented in
# doc/10M-RND-run-failure-documentation.md) collapsed within ~9% of their
# budget -- entropy fell to ~24-37% of ln(18), raw_intrinsic_rew_mean dropped
# ~20x, PPO updates went to no-ops (approx_kl/clipfrac -> ~0) -- and burned
# the remaining ~91% of walltime doing nothing. The root cause is still an
# OPEN investigation (doc/rnd-vs-ppo-asymmetry-investigation.md): a plain PPO
# baseline with no intrinsic reward outperformed RND on this exact codebase,
# which rules out "ent_coef was just too weak" as a full explanation. This
# script intentionally stays on rnd.py's paper-default hyperparameters
# (ent_coef=0.001, no --no-anneal-lr override) rather than shipping another
# unproven hypothesis as the default.
#
# What this run actually is: the cheap validation the open investigation
# calls for, run at matched scale to the failure case instead of guessing at
# a much larger budget. As of this round of changes, rnd.py auto-stops
# (--auto-stop, on by default -- see doc/decisions.md and
# doc/hpc-onboarding.md §7) if the same collapse signature reappears,
# sustained over --auto-stop-patience iterations, so a repeat collapse no
# longer silently burns the full 4h walltime -- check `sacct -j <id>
# --format=ExitCode`: 42:0 means auto-stop fired, 0:0 means it ran to
# completion without collapsing.
#
#   IF auto-stop does NOT fire and charts/rooms_visited > 1 and/or
#   charts/episodic_return is nonzero: real progress -- extend via a chained
#   --resume submission (see below) toward a larger budget rather than
#   stopping here.
#   IF auto-stop DOES fire: the collapse persists even under paper-default
#   hyperparameters at this scale. The next lever is NOT more compute -- see
#   the AutoresetMode.NEXT_STEP GAE-masking bug documented in
#   doc/decisions.md (explicitly out of scope for this round of changes,
#   still a live candidate root cause) before submitting this script again
#   unmodified.
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --export=ALL,SEED=1 slurm/run_rnd.slurm
#   sbatch --export=ALL,SEED=2 slurm/run_rnd.slurm
#
# To extend a completed or walltime-truncated run:
#   sbatch --export=ALL,SEED=1,RESUME_FROM=$SCRATCH/montezuma/checkpoints/<run_name>/ckpt_XXXXXX.pt slurm/run_rnd.slurm
# (the resume-fix in this round of changes makes this land in the same
# run_name directory instead of fragmenting into a new one -- see doc/decisions.md)
#
# Resource requests below are CONSERVATIVE GENERIC DEFAULTS pending your
# cluster's real numbers -- see doc/hpc-onboarding.md §1 and run
# `sinfo -o "%P %c %G"` before submitting. --cpus-per-task=32 matches
# --num-envs 1:1 (AsyncVectorEnv spawns one subprocess per env -- fewer CPUs
# than envs means workers queue for CPU time; more CPUs requested than the
# partition has per GPU means SLURM queues the job indefinitely in PD state
# instead of rejecting it outright). Only reduce --num-envs as a last resort
# -- it changes batch_size/num_iterations, reopening tuned hyperparameter
# interactions.

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

# Fail fast if GPU is not available
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
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
ENT_COEF="${ENT_COEF:-0.001}"              # rnd.py's paper-default (Burda et al. 2018)
EXP_NAME="${EXP_NAME:-rnd}"                # override (e.g. rnd_ent01) to isolate the ablation
ANNEAL_ARGS=()
if [ "${ANNEAL_LR:-1}" = "0" ] || [ "${ANNEAL_LR:-1}" = "false" ]; then
    ANNEAL_ARGS=(--no-anneal-lr)           # default: anneal ON (paper/CleanRL default)
fi

python src/agents/rnd.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name "$EXP_NAME" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --num-envs "$NUM_ENVS" \
    --ent-coef "$ENT_COEF" \
    --seed "${SEED:-1}" \
    --capture-video \
    --record-room-discovery \
    --checkpoint-interval 50 \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    "${ANNEAL_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
