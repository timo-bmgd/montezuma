#!/bin/bash
#SBATCH --job-name=montezuma-count-rider-smoke
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Infrastructure smoke test for the count-driven + passive-RND-rider pipeline
# (see doc/dual-signal-rider.md) -- NOT a test of training quality. Purpose:
# confirm the full dual-signal stack works end-to-end on this cluster before
# committing hours of walltime to slurm/run_count_rnd_rider.slurm: does the
# rider run, do step-log shards and artefacts land on scratch, do videos
# generate, does checkpoint/resume restore the count table and rider state.
# Sibling of run_rnd_smoke.slurm / run_ppo_smoke.slurm -- see
# doc/hpc-onboarding.md §6 for the general smoke-test walkthrough.
#
# --rider-obs-norm-init-steps is cut from the default 50 to 5 here: the init
# phase costs (init_steps x num_steps x num_envs) env steps of walltime that
# don't count toward --total-timesteps, and at smoke scale the default would
# spend half the job on it. Production (run_count_rnd_rider.slurm) keeps the
# default 50, matching rnd.py's own obs-norm init.
#
# Resource requests below are CONSERVATIVE GENERIC DEFAULTS, not tuned to
# this cluster -- read doc/hpc-onboarding.md §1 and run `sinfo -o "%P %c %G"`
# before submitting, and adjust these directives to match reality.
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --test-only slurm/run_count_rider_smoke.slurm   # dry-run syntax check
#   sbatch slurm/run_count_rider_smoke.slurm
#
# To test resume (count table + rider state must survive):
#   sbatch --export=ALL,RESUME_FROM=$SCRATCH/montezuma/checkpoints/<run_name>/ckpt_000010.pt \
#       slurm/run_count_rider_smoke.slurm
#
# Pass/fail checklist, on top of doc/hpc-onboarding.md §6's generic one:
#   [ ] Log shows "Passive RND rider enabled" and the rider obs-norm init phase
#   [ ] runs/<run_name>/step_log/ has 1+ steps_*.npz shard; loading one shows
#       finite bonus_active AND bonus_passive, room values >= 0
#   [ ] checkpoints/<run_name>/artifacts/ has simhash_*.npz (<1 MB) and
#       rider_rnd_*.pt (~16 MB)
#   [ ] TensorBoard has rider/* tags (raw_bonus_mean, fwd_loss, ...) and
#       charts/hash_* occupancy tags; charts/hash_top_bucket_share well below
#       1.0 and charts/hash_singleton_frac well below 1.0 (pool mode healthy)
#   [ ] A --resume job continues the SAME run dir and its charts/unique_states
#       does NOT reset toward zero (count table restored)

module purge
module load python/3.11
module load cuda/12.4

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

python src/agents/count_based.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name count_rider_smoke \
    --total-timesteps 100000 \
    --num-envs 8 \
    --seed "${SEED:-1}" \
    --hash-mode pool \
    --hash-pool-size 16 \
    --hash-dim 64 \
    --exploration-coef 0.01 \
    --passive-rnd \
    --rider-obs-norm-init-steps 5 \
    --step-log \
    --step-log-flush-iters 20 \
    --artifact-interval 20 \
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
