#!/bin/bash
#SBATCH --job-name=montezuma-ppo-smoke
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=Debug_node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Infrastructure smoke test for ppo.py -- NOT a test of training quality.
# Same purpose as run_rnd_smoke.slurm (see that script's header for the full
# rationale): confirm the pipeline works end-to-end on this cluster -- job
# runs without crashing, videos generate correctly, TensorBoard collects all
# expected metadata, checkpoint/resume works, W&B syncs -- before committing
# hours of walltime to slurm/run_ppo.slurm. See doc/hpc-onboarding.md §6 for
# the full checklist walkthrough.
#
# Resource requests below are CONSERVATIVE GENERIC DEFAULTS, not tuned to
# this cluster -- see run_rnd_smoke.slurm's header for the same caveat and
# doc/hpc-onboarding.md §1 for how to find the real numbers.
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --test-only slurm/run_ppo_smoke.slurm   # dry-run syntax check first
#   sbatch slurm/run_ppo_smoke.slurm
#
# To test the --resume fix specifically, submit a second job pointing at a
# checkpoint the first one wrote:
#   sbatch --export=ALL,RESUME_FROM=$SCRATCH/montezuma/checkpoints/<run_name>/ckpt_000010.pt slurm/run_ppo_smoke.slurm
#
# Pass/fail checklist (full detail in doc/hpc-onboarding.md §6; same as
# run_rnd_smoke.slurm minus the RND-specific TensorBoard tags -- room-discovery
# and periodic video recording both still apply, base.py's wrapper stack is shared):
#   [ ] Job completes without error (sacct -j <id> --format=State,ExitCode -> COMPLETED / 0:0)
#   [ ] slurm-logs/*.out shows steady SPS every iteration, no NaNs/tracebacks
#   [ ] At least one room_discovery/new_room_ep*.mp4 and at least one periodic
#       rl-video-episode-*.mp4, both playable
#   [ ] runs/<run_name>/ has all expected TensorBoard scalar tags, non-NaN and moving
#   [ ] checkpoints/<run_name>/ has 2+ ckpt_*.pt files; a second job with
#       --resume loads one and continues writing into the SAME run_name dir
#   [ ] W&B run visible in the montezuma-thesis-smoke project, matching TensorBoard
#
# --cpus-per-task should match --num-envs 1:1 -- see run_rnd_smoke.slurm's
# header / doc/hpc-onboarding.md §1 for why.

# HTW KI-Werkstatt has NO python/cuda environment modules -- Python is a dedicated conda
# env (see slurm/setup_hpc.sh). CONDA_ENV_PREFIX must match what setup_hpc.sh created.
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/$USER/conda-envs/montezuma}"
if [ -f /usr/etc/profile.d/conda.sh ]; then
    source /usr/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV_PREFIX"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

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
    --exp-name ppo_smoke \
    --total-timesteps 100000 \
    --num-envs 8 \
    --seed "${SEED:-1}" \
    --capture-video \
    --record-room-discovery \
    --video-episode-interval 5 \
    --checkpoint-interval 10 \
    --track \
    --wandb-project montezuma-thesis-smoke \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    "${RESUME_ARGS[@]}"
