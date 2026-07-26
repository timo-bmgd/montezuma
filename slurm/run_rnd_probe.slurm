#!/bin/bash
#SBATCH --job-name=montezuma-rnd-probe
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=02:30:00
#SBATCH --partition=Debug_node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ── RND "leave room 1" PROBE runner ──────────────────────────────────────────
# Short (<=5M step) diagnostic runs to find a config that reliably gets RND out
# of room 1, BEFORE committing any 20M-step budget. See doc/regression-findings.md
# ("Verification plan") and doc/run-analysis.md for why the 20M/32-env runs stall
# in room 1 (no bug/regression; under-scale + LR annealing to 0 + quartered
# predictor batch at 32 envs). Reference target: escape room 1 well before 10M,
# so <=5M is a valid read.
#
# This script is the run_rnd_tv.slurm boilerplate with the three probe levers
# exposed as env-var overrides, all defaulting to the CURRENT production values
# (so an unset invocation reproduces today's config):
#
#   UPDATE_PROPORTION   fraction of the minibatch used to train the RND predictor.
#                       Current 0.25 was Burda's 128-env value; at 32 envs the
#                       32-env reference is 1.0 (4x the effective predictor batch).
#   ANNEAL_LR           1/on/true keeps the linear LR->0 anneal (current default);
#                       0/off/false passes --no-anneal-lr (constant LR, Burda-aligned)
#                       so the back half of a run still learns.
#   EXT_COEF            weight of the extrinsic advantage (current 2.0). Set 0 for
#                       a pure-intrinsic sanity probe (the arm that should escape
#                       room 1 earliest if RND is working at all).
#
# Plus the usual TV_MODE / NUM_ENVS / TOTAL_TIMESTEPS / SEED / EXP_NAME /
# RESUME_FROM overrides. Probes default to TV_MODE=off (get the BASELINE exploring
# first; only once a recipe clears room 1 on off is it worth re-running remote/
# sham/static with it — otherwise the TV arms have no non-floored baseline to be
# measured against).
#
# NUM_ENVS note: --cpus-per-task above is 32 (the node's cores/GPU). NUM_ENVS>32
# oversubscribes (AsyncVectorEnv spawns one subprocess per env) — it still runs,
# just at lower SPS, buying more state diversity per predictor update. Fine for a
# probe; drop the 64-env cell if it will not schedule on your partition.
#
# Usage (submit from the project root):
#   # P0 baseline (reproduce current config)
#   sbatch --export=ALL,EXP_NAME=rnd_probe_baseline,SEED=1,TOTAL_TIMESTEPS=3000000 slurm/run_rnd_probe.slurm
#   # P1 primary bet: update_proportion=1.0 + constant LR
#   sbatch --export=ALL,EXP_NAME=rnd_probe_up1_noanneal,SEED=1,TOTAL_TIMESTEPS=5000000,UPDATE_PROPORTION=1.0,ANNEAL_LR=0 slurm/run_rnd_probe.slurm
#   # P3 pure-intrinsic sanity
#   sbatch --export=ALL,EXP_NAME=rnd_probe_pure_int,SEED=1,TOTAL_TIMESTEPS=3000000,UPDATE_PROPORTION=1.0,ANNEAL_LR=0,EXT_COEF=0 slurm/run_rnd_probe.slurm
#
# Or fire the whole matrix at once: bash slurm/submit_rnd_probes.sh
#
# Read-out (per run, from TensorBoard or scripts/analyze_runs.py):
#   PASS = charts/rooms_visited reaches 2 REPEATEDLY (not one lucky episode) and
#          ideally 3+, across >=2 seeds. Watch charts/rooms_furthest_cummax and
#          keep an eye on losses/entropy (a run that stays exploratory keeps
#          entropy up longer) and charts/raw_intrinsic_rew_mean (a working recipe
#          should NOT flatten as hard/early as the stalled 20M runs did).

set -eo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# sbatch executes a spooled COPY of this script, so $0 cannot locate the repo.
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
mkdir -p "$SCRATCH"/{runs,checkpoints}

# Fail fast if GPU is not available
python -c "import torch; assert torch.cuda.is_available(), 'No GPU detected!'; print('GPU:', torch.cuda.get_device_name(0))"

# ── build the toggle args ────────────────────────────────────────────────────
ANNEAL_ARGS=()
case "${ANNEAL_LR:-1}" in
    0|off|false|no|OFF|False) ANNEAL_ARGS=(--no-anneal-lr) ;;
    *)                        ANNEAL_ARGS=(--anneal-lr) ;;
esac

RESUME_ARGS=()
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS=(--resume "$RESUME_FROM")
fi

TV_MODE="${TV_MODE:-off}"
EXP_NAME="${EXP_NAME:-rnd_probe_${TV_MODE}}"

echo "── probe config ──────────────────────────────────────────────"
echo "  exp_name=${EXP_NAME}  tv_mode=${TV_MODE}  seed=${SEED:-1}"
echo "  total_timesteps=${TOTAL_TIMESTEPS:-3000000}  num_envs=${NUM_ENVS:-32}"
echo "  update_proportion=${UPDATE_PROPORTION:-0.25}  ext_coef=${EXT_COEF:-2.0}"
echo "  anneal_lr_args=${ANNEAL_ARGS[*]}"
echo "──────────────────────────────────────────────────────────────"

python src/agents/rnd.py \
    --env-id ALE/MontezumaRevenge-v5 \
    --exp-name "$EXP_NAME" \
    --tv-mode "$TV_MODE" \
    --total-timesteps "${TOTAL_TIMESTEPS:-3000000}" \
    --num-envs "${NUM_ENVS:-32}" \
    --seed "${SEED:-1}" \
    --update-proportion "${UPDATE_PROPORTION:-0.25}" \
    --ext-coef "${EXT_COEF:-2.0}" \
    "${ANNEAL_ARGS[@]}" \
    --capture-video \
    --record-room-discovery \
    --video-episode-interval 100 \
    --checkpoint-interval 100 \
    --runs-dir "$SCRATCH/runs" \
    --videos-dir "$SCRATCH/videos" \
    --checkpoint-dir "$SCRATCH/checkpoints" \
    "${RESUME_ARGS[@]}"
