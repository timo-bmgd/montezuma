#!/bin/bash
#SBATCH --job-name=montezuma-rnd
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
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

# Fail fast if GPU is not available
python -c "import torch; assert torch.cuda.is_available(), 'No GPU detected!'; print('GPU:', torch.cuda.get_device_name(0))"

RESUME_ARGS=()
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARGS=(--resume "$RESUME_FROM")
fi

python src/agents/rnd.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name rnd \
    --total-timesteps 3000000 \
    --num-envs 32 \
    --seed "${SEED:-1}" \
    --capture-video \
    --record-room-discovery \
    --checkpoint-interval 50 \
    --track \
    --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    "${RESUME_ARGS[@]}"
