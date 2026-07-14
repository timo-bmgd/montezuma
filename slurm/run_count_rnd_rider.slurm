#!/bin/bash
#SBATCH --job-name=montezuma-count-rider
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Count-driven production run WITH a passive RND rider -- the primary data
# source for the thesis's dual-signal novelty comparison (see
# doc/dual-signal-rider.md). Identical to slurm/run_count_based.slurm except:
#
#   --hash-mode pool --hash-pool-size 16: the original "index" hash is
#     degenerate-collapsed on Montezuma (one bucket absorbs ~50% of visits --
#     measured by scripts/simhash_occupancy_probe.py); pool16/k=64 gives
#     healthy occupancy (top bucket ~3%, mean visit count ~16).
#   --passive-rnd: an RND observer runs on the same observation stream; its
#     bonus is logged but NEVER added to the reward (bit-identical training
#     with or without it -- verified by scripts/check_rider_noop.py).
#   --step-log: per-(step, env) records of both bonus signals + room id +
#     episode id, as .npz shards under $SCRATCH/runs/<run_name>/step_log/.
#   --artifact-interval 200: small analysis artefacts (rider predictor+target
#     ~16 MB, count table <1 MB; no optimizer states) every 200 iterations.
#
# The rider adds an RND forward pass per step and predictor training per
# iteration, so expect throughput between run_count_based.slurm and
# run_rnd.slurm.
#
# Usage:
#   mkdir -p slurm-logs
#   sbatch --export=ALL,SEED=1 slurm/run_count_rnd_rider.slurm
#
# To resume (count table AND rider state are restored from the checkpoint):
#   sbatch --export=ALL,SEED=1,RESUME_FROM=$SCRATCH/montezuma/checkpoints/<run_name>/ckpt_XXXXXX.pt \
#       slurm/run_count_rnd_rider.slurm
#
# Resource caveats: see slurm/run_count_based.slurm's header (cpus-per-task
# vs partition limits; --num-envs is not a free lever).

module purge
module load python/3.11
module load cuda/12.4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

SCRATCH="${SCRATCH:-/scratch/$USER}/montezuma"
mkdir -p "$SCRATCH"/{runs,checkpoints,slurm-logs}

python -c "import torch; assert torch.cuda.is_available(), 'No GPU detected!'; print('GPU:', torch.cuda.get_device_name(0))"

RESUME_ARGS=()
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARGS=(--resume "$RESUME_FROM")
fi

python src/agents/count_based.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --total-timesteps 50000000 \
    --num-envs 32 \
    --seed "${SEED:-1}" \
    --hash-mode pool \
    --hash-pool-size 16 \
    --hash-dim 64 \
    --exploration-coef 0.01 \
    --passive-rnd \
    --step-log \
    --artifact-interval 200 \
    --capture-video \
    --record-room-discovery \
    --track \
    --wandb-project montezuma-thesis \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    --checkpoint-interval 100 \
    "${RESUME_ARGS[@]}"
