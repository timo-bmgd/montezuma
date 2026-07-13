#!/bin/bash
#SBATCH --job-name=montezuma-rnd
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Do NOT submit this until slurm/run_rnd_falsify.slurm has confirmed a fix for the
# entropy/intrinsic-reward collapse documented in
# doc/10M-RND-run-failure-documentation.md and doc/rnd-vs-ppo-asymmetry-investigation.md.
# Both prior 10M-step seeds collapsed within ~9% of budget; at this job's scale
# (100M steps) an unresolved collapse would freeze the policy for the ~20-hour
# walltime instead of the ~15-18h the original failed runs took, for no additional
# benefit. This script intentionally stays on rnd.py's paper-default
# hyperparameters for now -- ent_coef=0.01/--no-anneal-lr is an unconfirmed
# hypothesis (see doc/decisions.md), not a proven fix, and RND actually
# underperformed a plain PPO baseline on this codebase, so raising ent_coef alone
# may not even be the right lever. Test hypotheses via run_rnd_falsify.slurm and
# promote whatever's confirmed there into this script's python invocation once
# it's actually validated.
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --export=ALL,SEED=1 slurm/run_rnd.slurm
#   sbatch --export=ALL,SEED=2 slurm/run_rnd.slurm
#   sbatch --export=ALL,SEED=3 slurm/run_rnd.slurm
#
# To resume a preempted job, add to the python command below:
#   --resume $SCRATCH/checkpoints/<run_name>/ckpt_XXXXXX.pt
#
# --cpus-per-task=32 matches --num-envs 1:1 -- AsyncVectorEnv (the default) spawns
# one subprocess per env, so fewer CPUs than envs means workers queue for CPU time.
# This has NOT been verified against the real partition -- run:
#   sinfo -o "%P %c %G"
# before submitting. If the partition has fewer than 32 cores/GPU, lower
# --cpus-per-task to match; SLURM does not reject an over-large request, it queues
# the job indefinitely in PD state, silently burning walltime budget. Only reduce
# --num-envs as a last resort -- it changes batch_size/num_iterations, so it isn't
# a free lever.

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

python src/agents/rnd.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --total-timesteps 100000000 \
    --num-envs 32 \
    --seed "${SEED:-1}" \
    --capture-video \
    --record-room-discovery \
    --track \
    --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    --checkpoint-interval 100
