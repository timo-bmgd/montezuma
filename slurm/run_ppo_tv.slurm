#!/bin/bash
#SBATCH --job-name=montezuma-ppo-tv
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=Debug_node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
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
# Scale matches the RND runs (defaults: 10M steps / num_envs=32, both
# overridable via TOTAL_TIMESTEPS / NUM_ENVS) for a like-for-like comparison.
# See slurm/run_rnd_tv.slurm's header for walltime/chaining and the resume
# --tv-* consistency guard, which apply identically here.
#
# Usage (submit from the project root -- #SBATCH --output and PROJECT_DIR
# both assume it; slurm-logs/ is committed to the repo so it always exists):
#   sbatch --export=ALL,SEED=1,TV_MODE=remote slurm/run_ppo_tv.slurm

set -eo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ -z "${TV_MODE:-}" ]; then
    echo "ERROR: TV_MODE must be set (off|static|remote|sham-remote)" >&2
    exit 1
fi

# sbatch executes a spooled COPY of this script, so $0 cannot locate the repo.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from (the project root,
# per the usage above); the $0 fallback covers direct `bash slurm/...` runs.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# HTW KI-Werkstatt has NO python/cuda modules -- dedicated conda env (see slurm/setup_hpc.sh).
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/$USER/conda-envs/montezuma}"
if [ -f /usr/etc/profile.d/conda.sh ]; then
    source /usr/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV_PREFIX"

SCRATCH="${SCRATCH:-/scratch/$USER}/montezuma"
mkdir -p "$SCRATCH"/{runs,checkpoints}

# Fail fast if GPU is not available
python -c "import torch; assert torch.cuda.is_available(), 'No GPU detected!'; print('GPU:', torch.cuda.get_device_name(0))"

RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS=(--resume "$RESUME_FROM")
fi

# Echo the resolved run config so overrides are never silently ignored (if this line
# shows total_timesteps=10000000 when you passed TOTAL_TIMESTEPS=20000000, your checkout
# predates the override -- git pull).
echo ">>> RUN CONFIG: tv_mode=$TV_MODE seed=${SEED:-1} num_envs=${NUM_ENVS:-32} total_timesteps=${TOTAL_TIMESTEPS:-10000000} resume=${RESUME_FROM:-none}"

python src/agents/ppo.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name "ppo_tv_${TV_MODE}" \
    --tv-mode "$TV_MODE" \
    --total-timesteps "${TOTAL_TIMESTEPS:-10000000}" \
    --num-envs "${NUM_ENVS:-32}" \
    --seed "${SEED:-1}" \
    --capture-video \
    --record-room-discovery \
    --video-episode-interval 100 \
    --checkpoint-interval 100 \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    "${RESUME_ARGS[@]}"
