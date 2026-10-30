#!/usr/bin/env bash
# SAM 3D Body's own env (pipeline.py's 3D pose front end; frontends/sam3d_run.py runs in it):
#   bash scripts/setup_sam3d.sh [prefix]     # -> <prefix>/sam3d_env, <prefix>/sam-3d-body (default /content)
# then export A2H_SAM3D_PYTHON=<prefix>/sam3d_env/bin/python A2H_SAM3D_DIR=<prefix>/sam-3d-body.
# The checkpoint (facebook/sam-3d-body-dinov3) is gated on Hugging Face and downloads on first use with HF_TOKEN.
set -euo pipefail
P="${1:-/content}"
COMMIT=b5c765a0d89d789985e186d396315e7590887b94  # frontends/sam3d_run.py SAM3D_COMMIT
command -v uv >/dev/null || pip install -q uv
if [ ! -d "$P/sam-3d-body" ]; then
  git clone -q https://github.com/facebookresearch/sam-3d-body.git "$P/sam-3d-body"
  git -C "$P/sam-3d-body" checkout -q "$COMMIT"
fi
[ -d "$P/sam3d_env" ] || uv venv -q -p 3.11 "$P/sam3d_env"
# One resolve: MoGe (the focal length) pulls triton>=3.2, which torch 2.7 satisfies. torch from PyTorch's CUDA 12.6
# index (a build step has no GPU for --torch-backend auto to detect); --no-sources: MoGe's own pyproject points uv
# at a torch index without 2.7.1.
uv pip install -q -p "$P/sam3d_env/bin/python" --no-sources --torch-backend cu126 torch==2.7.1 torchvision==0.22.1 pytorch-lightning yacs scikit-image \
  einops timm dill pandas rich hydra-core pyrootutils webdataset networkx==3.2.1 roma joblib loguru optree fvcore \
  huggingface_hub opencv-python-headless braceexpand omegaconf mediapy termcolor "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"
A2H_SAM3D_DIR="$P/sam-3d-body" MOMENTUM_ENABLED=0 PYTHONPATH="$P/sam-3d-body" "$P/sam3d_env/bin/python" -c \
  "import torch, sam_3d_body, moge; print('SAM 3D Body env OK, CUDA:', torch.cuda.is_available())"
