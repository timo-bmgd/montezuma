#!/bin/bash
#SBATCH --job-name=montezuma-rnd-tv
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Noisy-TV experiment production run (the thesis's main experiment -- see
# doc/noisy-tv-experiment.md for hypothesis, conditions and analysis readouts).
#
# Scale: 10M steps at num_envs=8, deliberately matching the historical
# rnd_10m_s1/s2 baselines (doc/10M-RND-run-failure-documentation.md) so TV
# conditions are comparable against a known reference scale. One script covers
# every RND condition of the experiment matrix via TV_MODE:
#
#   TV_MODE=off          fresh baseline on current code (old baselines predate
#                        code changes: reward clipping, resume fix, ...)
#   TV_MODE=remote       PRIMARY condition -- extra NOOP-mapped action 18
#                        resamples the patch; watch charts/tv_action_frac
#                        (chance = 1/19 ~ 0.053; sustained >> chance = capture)
#   TV_MODE=sham-remote  action-space-matched control (extra action, no patch)
#   TV_MODE=static       always-on patch -- signal degradation, no behavioral
#                        channel
#
# Auto-stop expectation per mode (sacct -j <id> --format=ExitCode):
#   off / sham-remote: behaves exactly like the baseline -- 42:0 means the
#     known collapse signature reappeared (informative in itself), 0:0 means
#     the run completed without collapsing.
#   remote / static: the TV keeps raw_intrinsic_rew_mean elevated, so the
#     >=10x intrinsic-drop conjunct of the auto-stop signature should never
#     hold and the run should spend its full budget (0:0). That is the point:
#     the sustained-high-intrinsic trajectory IS the data. A 42:0 here would
#     mean intrinsic collapsed despite the TV -- surprising, look closely.
#
# Walltime: prior 10M/8-env runs sustained SPS 144-183 -> ~15-19 h. If the
# partition caps below 20 h, keep shorter blocks and chain them via
# RESUME_FROM (must repeat the SAME TV_MODE -- rnd.py aborts a --resume whose
# --tv-* flags differ from the checkpoint's, since remote and sham-remote
# load into identical network shapes and would otherwise swap silently).
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --export=ALL,SEED=1,TV_MODE=remote slurm/run_rnd_tv.slurm
#   sbatch --export=ALL,SEED=2,TV_MODE=remote slurm/run_rnd_tv.slurm
#   sbatch --export=ALL,SEED=1,TV_MODE=off slurm/run_rnd_tv.slurm
#   ...
# To extend a walltime-truncated run:
#   sbatch --export=ALL,SEED=1,TV_MODE=remote,RESUME_FROM=$SCRATCH/montezuma/checkpoints/<run_name>/ckpt_XXXXXX.pt slurm/run_rnd_tv.slurm

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

python src/agents/rnd.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name "rnd_tv_${TV_MODE}" \
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
