#!/bin/bash
#SBATCH --job-name=montezuma-ppo
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# --cpus-per-task=32 matches --num-envs 1:1 -- AsyncVectorEnv (the default) spawns
# one subprocess per env, so fewer CPUs than envs means workers queue for CPU time.
# This has NOT been verified against the real partition -- run:
#   sinfo -o "%P %c %G"
# before submitting. If the partition has fewer than 32 cores/GPU, lower
# --cpus-per-task to match; SLURM does not reject an over-large request, it queues
# the job indefinitely in PD state, silently burning walltime budget. Only reduce
# --num-envs as a last resort (it changes batch_size/num_iterations, so it isn't a
# free lever).
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --export=ALL,SEED=1 slurm/run_ppo.slurm

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

python src/agents/ppo.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --total-timesteps 10000000 \
    --num-envs 32 \
    --seed "${SEED:-1}" \
    --track \
    --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    --checkpoint-interval 100
