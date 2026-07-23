#!/bin/bash
#SBATCH --job-name=montezuma-rnd-smoke
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=Debug_node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Infrastructure smoke test for rnd.py -- NOT a test of training quality.
# Purpose: confirm the pipeline works end-to-end on this cluster before
# committing hours of walltime to slurm/run_rnd.slurm: does the job run
# without crashing, do videos generate correctly, does TensorBoard collect
# all expected metadata, does checkpoint/resume work, does W&B sync. A few
# thousand steps is enough to check all of that -- see doc/hpc-onboarding.md
# §6 for the full walkthrough and what each failure mode most likely means.
#
# (This script replaces the old run_rnd_falsify.slurm, which tested a
# specific ent_coef hypothesis that was later reverted -- see doc/decisions.md.
# That hypothesis-testing purpose is moot now; this script's job is pipeline
# correctness, not algorithm tuning.)
#
# Resource requests below are CONSERVATIVE GENERIC DEFAULTS, not tuned to
# this cluster -- --gres=gpu:1 has no GPU type pinned and --cpus-per-task=8
# just matches --num-envs 1:1 with no knowledge of the real core-per-GPU
# ratio. Read doc/hpc-onboarding.md §1 and run `sinfo -o "%P %c %G"` before
# submitting, and adjust these directives to match reality.
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --test-only slurm/run_rnd_smoke.slurm   # dry-run syntax check first
#   sbatch slurm/run_rnd_smoke.slurm
#
# To test the --resume fix specifically, submit a second job pointing at a
# checkpoint the first one wrote:
#   sbatch --export=ALL,RESUME_FROM=$SCRATCH/montezuma/checkpoints/<run_name>/ckpt_000010.pt slurm/run_rnd_smoke.slurm
#
# Pass/fail checklist (full detail in doc/hpc-onboarding.md §6):
#   [ ] Job completes without error (sacct -j <id> --format=State,ExitCode -> COMPLETED / 0:0)
#   [ ] slurm-logs/*.out shows steady SPS every iteration, no NaNs/tracebacks
#   [ ] At least one room_discovery/new_room_ep*.mp4 (episode 1 always
#       qualifies) and at least one periodic rl-video-episode-*.mp4, both playable
#   [ ] runs/<run_name>/ has all expected TensorBoard scalar tags (see
#       doc/hpc-onboarding.md §6 for the full list), values non-NaN and moving
#   [ ] checkpoints/<run_name>/ has 2+ ckpt_*.pt files; a second job with
#       --resume loads one and continues writing into the SAME run_name dir
#   [ ] W&B run visible in the montezuma-thesis-smoke project, matching TensorBoard
#
# --cpus-per-task should match --num-envs 1:1 -- AsyncVectorEnv (the default)
# spawns one subprocess per env, so fewer CPUs than envs means workers queue
# for CPU time. SLURM does not reject an over-large CPU request outright, it
# queues the job indefinitely in PD state -- see doc/hpc-onboarding.md §1.

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

python src/agents/rnd.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name rnd_smoke \
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
