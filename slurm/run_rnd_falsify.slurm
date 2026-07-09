#!/bin/bash
#SBATCH --job-name=montezuma-rnd-falsify
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Cheap falsification test for the entropy/intrinsic-reward collapse diagnosed in
# doc/10M-RND-run-failure-documentation.md (see worktree-rnd-10m-failure-doc branch
# if that doc isn't on your current branch), section 6.
#
# Both prior 10M-step seeds collapsed within ~9% of budget (by step ~909,312):
# entropy fell to ~24-37% of ln(18), raw_intrinsic_rew_mean dropped ~20x, and PPO
# updates went to no-ops (approx_kl/clipfrac -> ~0) for the remaining ~90% of
# training. This run tests the fix (raised ent_coef, disabled LR anneal) at a small
# step budget BEFORE committing to the full 100M-step run_rnd.slurm -- run this
# first, do not submit run_rnd.slurm until this passes.
#
# At num_envs=32, batch_size=4096, so 3,000,000 steps = ~732 iterations -- enough
# to run well past the ~222-iteration mark equivalent to the prior collapse point.
#
# Pass/fail (check TensorBoard, both seeds):
#   PASS: losses/entropy stays above ~30-40% of ln(18)=2.89, and
#         charts/raw_intrinsic_rew_mean does not collapse >10x, through at least
#         the last ~60-70% of iterations; losses/approx_kl and losses/clipfrac
#         stay meaningfully above 0 throughout.
#   FAIL: same collapse signature as before reappears early -- do not proceed to
#         the full run_rnd.slurm; revisit hyperparameters or consider the
#         alternatives in the exploration-algorithm literature survey instead.
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --export=ALL,SEED=1 slurm/run_rnd_falsify.slurm
#   sbatch --export=ALL,SEED=2 slurm/run_rnd_falsify.slurm
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

# Fail fast if GPU is not available
python -c "import torch; assert torch.cuda.is_available(), 'No GPU detected!'; print('GPU:', torch.cuda.get_device_name(0))"

python src/agents/rnd.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name rnd_falsify \
    --total-timesteps 3000000 \
    --num-envs 32 \
    --ent-coef 0.01 \
    --no-anneal-lr \
    --seed "${SEED:-1}" \
    --track \
    --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    --checkpoint-interval 100
