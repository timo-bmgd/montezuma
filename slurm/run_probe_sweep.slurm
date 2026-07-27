#!/bin/bash
#SBATCH --job-name=montezuma-probe-sweep
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=Debug_node
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Post-hoc patch-response probe sweep (doc/noisy-tv-results.md section 2,
# doc/TODO-next-steps.md item 3). Runs scripts/probe_patch_response.py over EVERY
# checkpoint of ONE run (one TV arm, one seed) to trace the predictor's patch
# response across training -- content_sensitivity / patch_contribution vs step.
# The dense per-checkpoint trajectory is what turns "self-limiting capture" from
# inferred into DEMONSTRATED (the memorisation gap open early, closed late).
#
#   TV_MODE   one of the 4 RND arms: off | remote | sham-remote | static
#             (PPO arms have no rnd_model_state_dict and CANNOT be probed)
#   SEED      42 | 43 | 44
#   NUM_FRAMES   optional, default 4096 (halve to 2048 to ~halve wall-clock)
#   NUM_FRESH    optional, default 8   (content-sensitivity averaging draws)
#
# CPU job by design: the probe's cost is the sequential Atari rollout, which is
# CPU-bound regardless of --device, so a GPU would idle here and steal a slot
# from the training matrix. One run's ~49 checkpoints take ~1-1.5 h.
#
# Usage (submit from the project root; slurm-logs/ is committed so the path exists):
#   sbatch --export=ALL,SEED=42,TV_MODE=remote slurm/run_probe_sweep.slurm
# Or fire the whole 3-seed x 4-arm matrix via slurm/submit_probe_sweep.sh.

set -eo pipefail

# Single-env CPU probe: let torch use the allocated cores for the batched RND
# forward pass (the training scripts pin OMP=1 for AsyncVectorEnv; we want the
# opposite here).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [ -z "${TV_MODE:-}" ] || [ -z "${SEED:-}" ]; then
    echo "ERROR: TV_MODE (off|remote|sham-remote|static) and SEED must be set" >&2
    exit 1
fi
case "$TV_MODE" in
    off|remote|sham-remote|static) ;;
    *) echo "ERROR: TV_MODE='$TV_MODE' is not a probeable RND arm (off|remote|sham-remote|static)" >&2
       exit 1 ;;
esac

# sbatch executes a spooled COPY of this script, so $0 cannot locate the repo.
# SLURM_SUBMIT_DIR is the directory sbatch was invoked from (the project root);
# the $0 fallback covers direct `bash slurm/...` runs.
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

# Checkpoint dir: run_name embeds env_id (which contains a '/'), so the on-disk
# layout is  $SCRATCH/checkpoints/ALE/MontezumaRevenge-v5__rnd_tv_<mode>__<seed>__<ts>/
CK="$SCRATCH/checkpoints/ALE/MontezumaRevenge-v5"
GLOB="${CK}__rnd_tv_${TV_MODE}__${SEED}__*/ckpt_*.pt"

# Fail fast with a clear message if the run's checkpoints are not there yet.
if ! compgen -G "$GLOB" > /dev/null; then
    echo "ERROR: no checkpoints matched: $GLOB" >&2
    echo "  (has the SEED=${SEED} TV_MODE=${TV_MODE} training run finished a checkpoint yet?)" >&2
    exit 2
fi

OUT_DIR="$PROJECT_DIR/analysis/probe"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/probe_rnd_tv_${TV_MODE}__${SEED}.csv"

echo "=== probe sweep  arm=${TV_MODE} seed=${SEED}  frames=${NUM_FRAMES:-4096} fresh=${NUM_FRESH:-8} ==="
echo "checkpoints: $(compgen -G "$GLOB" | wc -l)   ->   $OUT"

python scripts/probe_patch_response.py --device cpu \
    --num-frames "${NUM_FRAMES:-4096}" --num-fresh "${NUM_FRESH:-8}" \
    --checkpoints $GLOB \
    --out "$OUT"

echo "=== done. Aggregate+plot once all arms/seeds are in:"
echo "    python scripts/plot_probe_trajectory.py --indir analysis/probe"
