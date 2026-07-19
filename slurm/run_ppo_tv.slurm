#!/bin/bash
#SBATCH --job-name=montezuma-ppo-tv
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# PPO control runs for the noisy-TV experiment (doc/noisy-tv-experiment.md).
# PPO has no intrinsic reward, so PPO+TV is the no-curiosity control: the TV
# can only affect it through the observation channel (distraction) and the
# extra action -- behavioral capture should NOT occur
# (charts/tv_action_frac should hover near chance, 1/19 ~ 0.053).
#
#   TV_MODE=remote  the core control run of the matrix
#   TV_MODE=off     optional fresh PPO reference on current code
#
# Scale matches the RND runs (10M steps / 8 envs) for a like-for-like
# comparison. See slurm/run_rnd_tv.slurm's header for walltime/chaining and
# the resume --tv-* consistency guard, which apply identically here.
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --export=ALL,SEED=1,TV_MODE=remote slurm/run_ppo_tv.slurm

module purge
module load python/3.11
module load cuda/12.4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ -z "$TV_MODE" ]; then
    echo "ERROR: TV_MODE must be set (off|static|remote|sham-remote)" >&2
    exit 1
fi

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

python src/agents/ppo.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name "ppo_tv_${TV_MODE}" \
    --tv-mode "$TV_MODE" \
    --total-timesteps 10000000 \
    --num-envs 8 \
    --seed "${SEED:-1}" \
    --capture-video \
    --record-room-discovery \
    --video-episode-interval 100 \
    --checkpoint-interval 100 \
    --track \
    --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    "${RESUME_ARGS[@]}"
