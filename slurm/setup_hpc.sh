#!/bin/bash
# One-time environment setup for the HTW KI-Werkstatt cluster (kiwihead01).
# Run ONCE on the login node:
#   bash slurm/setup_hpc.sh
#
# This cluster has NO python/cuda environment modules (`module avail python|cuda` is
# empty) -- Python is provided by conda, and CUDA by the torch cu124 wheel's bundled
# runtime (+ the GPU node's NVIDIA driver). So this creates a dedicated conda env rather
# than the module-load + venv flow a generic HPC template would use.
#
# The env is placed on /scratch (a ~6 GB torch stack) to respect the cluster rule
# "don't waste storage in your home directory". Override the location with
#   CONDA_ENV_PREFIX=/some/other/path bash slurm/setup_hpc.sh

set -e

CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/scratch/$USER/conda-envs/montezuma}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Montezuma HPC setup (conda) ==="
echo "Project:   $PROJECT_DIR"
echo "Conda env: $CONDA_ENV_PREFIX  (python $PYTHON_VERSION)"
echo ""

# Make `conda activate` work in this non-login shell. Base conda is rooted at /usr here
# (from `conda info --envs`); fall back to the shell hook if that path ever changes.
if [ -f /usr/etc/profile.d/conda.sh ]; then
    source /usr/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found -- run this on the login node where (base) is active." >&2
    exit 1
fi

# Dedicated env on scratch. conda-forge is required: the system conda (4.10.3) predates
# python 3.11 in the defaults channel.
conda create -y -p "$CONDA_ENV_PREFIX" -c conda-forge "python=$PYTHON_VERSION"
conda activate "$CONDA_ENV_PREFIX"

pip install --upgrade pip wheel

# PyTorch, with the CUDA runtime bundled in the wheel (no system cuda toolkit needed).
#
# IMPORTANT: the wheel's CUDA tag must be <= the node's GPU driver CUDA version. Check it:
#   nvidia-smi   (on a GPU node -- top-right "CUDA Version: XX.X")
# then set TORCH_CUDA_TAG accordingly. NOTE: PyTorch only builds "+cu124" wheels up to
# torch 2.6.0; newer torch uses cu126/cu128. So the version and the CUDA tag are coupled:
#   driver CUDA >= 12.8 -> TORCH_CUDA_TAG=cu128  TORCH_VERSION=2.8.0   (default below)
#   driver CUDA  = 12.6/12.7 -> TORCH_CUDA_TAG=cu126  TORCH_VERSION=2.8.0
#   driver CUDA  = 12.4/12.5 -> TORCH_CUDA_TAG=cu124  TORCH_VERSION=2.6.0
#   driver CUDA  = 12.1..12.3 -> TORCH_CUDA_TAG=cu121  TORCH_VERSION=2.5.1
# Override without editing this file, e.g.:
#   TORCH_CUDA_TAG=cu124 TORCH_VERSION=2.6.0 bash slurm/setup_hpc.sh
# Default cu128 confirmed against kiwihead01's driver 580.95.05 (nvidia-smi CUDA 13.0,
# 2026-07-24) -- a cu128 wheel runs on this driver (drivers are backward-compatible).
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
echo ">>> installing torch==${TORCH_VERSION} (${TORCH_CUDA_TAG}) -- must be <= the node's nvidia-smi CUDA version"
pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

# Rest of dependencies (pinned to match requirements.txt / the dev venv)
pip install gymnasium==1.3.0 ale-py==0.11.2 AutoROM==0.6.1
pip install numpy==2.4.4 opencv-python-headless==4.13.0.92
pip install tensorboard==2.20.0 pillow==11.3.0   # wandb intentionally omitted (runs use TensorBoard only)
# Video-recording deps: moviepy (RecordVideo / --capture-video), imageio + imageio-ffmpeg
# (NewRoomRecorder / --record-room-discovery), matplotlib (--overlay-video). Install moviepy
# directly -- do NOT use `gymnasium[other]` (what the RecordVideo error suggests), it pulls
# numpy-2-incompatible dependencies and conflicts with the numpy pin above.
pip install moviepy imageio imageio-ffmpeg matplotlib

# Download and install Atari ROMs
python -m AutoROM --accept-license

echo ""
echo "=== Verification (login node -- no GPU here) ==="
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA build:', torch.version.cuda)"
python -c "import gymnasium; print('Gymnasium:', gymnasium.__version__)"
python -c "import ale_py; print('ale-py:', ale_py.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
echo ""
echo "NOTE: torch.cuda.is_available() is False on the login node (no GPU). Verify GPU"
echo "      visibility inside a salloc on Debug_node (see doc/matched-budget-submission.md STEP 2)."
echo ""
echo "=== Setup complete ==="
echo "The run scripts activate this env automatically (same CONDA_ENV_PREFIX default)."
echo "Manual activate:  conda activate $CONDA_ENV_PREFIX"
