#!/usr/bin/env bash
# Set up a Colab VM for training: mjlab venv, robot assets, headless rendering.
#   bash scripts/setup_colab.sh            # training only (CSV -> policy)
#   bash scripts/setup_colab.sh --with-main # also the main env's deps (latent stages, no SAM)
#   bash scripts/setup_colab.sh --with-sam # also the video stage (main deps + SAM 3.1 and SAM 3D Body venvs)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

pip install -q uv
bash "$REPO/scripts/fetch_assets.sh"

# Headless MuJoCo: Colab only registers Mesa's EGL driver, which can't reach the GPU.
EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[ -f "$EGL" ] || echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' > "$EGL"

[ -d /content/mjlab_env ] || uv venv -q -p 3.12 /content/mjlab_env
uv pip install -q -p /content/mjlab_env/bin/python mjlab==1.6.0 tensorboard
/content/mjlab_env/bin/python -c "import torch, mjlab; print('mjlab env OK, CUDA:', torch.cuda.is_available())"

if [ "${1:-}" = "--with-main" ] || [ "${1:-}" = "--with-sam" ]; then
  pip install -q -r "$REPO/requirements.txt"
fi
if [ "${1:-}" = "--with-sam" ]; then
  [ -d /content/sam3 ] || git clone -q https://github.com/facebookresearch/sam3.git /content/sam3
  [ -d /content/sam_env ] || uv venv -q -p 3.12 /content/sam_env
  uv pip install -q -p /content/sam_env/bin/python --torch-backend auto torch torchvision \
    "setuptools<81" -e "/content/sam3[notebooks]"
  /content/sam_env/bin/python -c "import torch, sam3; print('SAM env OK, CUDA:', torch.cuda.is_available())"
  bash "$REPO/scripts/setup_sam3d.sh" /content  # the 3D pose front end (A2H_SAM3D_PYTHON)
fi
echo SETUP_DONE
