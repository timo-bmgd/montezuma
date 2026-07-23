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

# PyTorch with the CUDA 12.4 runtime bundled in the wheel (no system cuda toolkit needed)
pip install torch==2.8.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Rest of dependencies (pinned to match requirements.txt / the dev venv)
pip install gymnasium==1.3.0 ale-py==0.11.2 AutoROM==0.6.1
pip install numpy==2.4.4 opencv-python-headless==4.13.0.92
pip install tensorboard==2.20.0 wandb==0.18.7 pillow==11.3.0

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
