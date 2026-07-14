#!/bin/bash
#SBATCH --job-name=montezuma-ppo
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
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

python src/agents/ppo.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name ppo \
    --total-timesteps 10000000 \
    --num-envs 32 \
    --seed "${SEED:-1}" \
    --capture-video \
    --record-room-discovery \
    --checkpoint-interval 100 \
    --track \
    --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    "${RESUME_ARGS[@]}"
