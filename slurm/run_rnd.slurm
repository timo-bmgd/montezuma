#!/bin/bash
#SBATCH --job-name=montezuma-rnd
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Usage:
#   mkdir -p slurm-logs
#   sbatch --export=ALL,SEED=1 slurm/run_rnd.slurm
#   sbatch --export=ALL,SEED=2 slurm/run_rnd.slurm
#   sbatch --export=ALL,SEED=3 slurm/run_rnd.slurm
#
# To resume a preempted job, add to the python command below:
#   --resume $SCRATCH/checkpoints/<run_name>/ckpt_XXXXXX.pt
#
# Check your cluster's GPU resource names: sinfo -o "%P %G"

module purge
module load python/3.11
module load cuda/12.4

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

SCRATCH="${SCRATCH:-/scratch/$USER}/montezuma"
mkdir -p "$SCRATCH"/{runs,checkpoints,slurm-logs}

# Fail fast if GPU is not available
python -c "import torch; assert torch.cuda.is_available(), 'No GPU detected!'; print('GPU:', torch.cuda.get_device_name(0))"

python src/agents/rnd.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --total-timesteps 100000000 \
    --num-envs 32 \
    --seed "${SEED:-1}" \
    --track \
    --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    --checkpoint-interval 100
