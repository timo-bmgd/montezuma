#!/bin/bash
# One-time HPC environment setup. Run on the login node:
#   bash slurm/setup_hpc.sh
#
# Adjust module names to match your cluster:
#   sinfo -o "%P %G"              -- list partitions and GPUs
#   module avail python            -- list available Python modules
#   module avail cuda              -- list available CUDA modules

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venv"

echo "=== Setting up Montezuma HPC environment ==="
echo "Project: $PROJECT_DIR"
echo "Venv:    $VENV"
echo ""

# Load modules (adjust names for your cluster)
module load python/3.11
module load cuda/12.4

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel

# PyTorch with CUDA 12.4 (matches A100 driver requirement)
pip install torch==2.8.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Rest of dependencies
pip install gymnasium==1.3.0 ale-py==0.11.2 AutoROM==0.6.1
pip install numpy==2.4.4 opencv-python-headless==4.13.0.92
pip install tensorboard==2.20.0 wandb==0.18.7 pillow==11.3.0

# Download and install Atari ROMs
python -m AutoROM --accept-license

echo ""
echo "=== Verification ==="
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import gymnasium; print('Gymnasium:', gymnasium.__version__)"
python -c "import ale_py; print('ale-py:', ale_py.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source $VENV/bin/activate"
