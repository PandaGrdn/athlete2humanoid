#!/usr/bin/env bash
# Set up (or update) the FarmShare side of remote_run.py --backend farmshare. Runs on a login node;
# idempotent, so remote_run.py calls it before every submit (seconds when nothing changed).
#   bash scripts/setup_farmshare.sh <root> [--with-main|--with-sam]
# <root>/envs/{mjlab,main,sam}: the same three environments as Colab (setup_colab.sh), built with uv's own
# Python 3.12, so no module is needed. Colab ships torch/transformers in its system Python; here the main
# env installs them explicitly. CUDA wheels: A2H_TORCH_BACKEND (default cu126; L40S nodes).
set -euo pipefail
ROOT="$1"; MODE="${2:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TORCH_BACKEND="${A2H_TORCH_BACKEND:-cu126}"
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$ROOT/uv-cache" HF_HOME="$ROOT/hf"
mkdir -p "$ROOT/envs" "$HF_HOME"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
bash "$REPO/scripts/fetch_assets.sh"

[ -d "$ROOT/envs/mjlab" ] || uv venv -q -p 3.12 "$ROOT/envs/mjlab"
uv pip install -q -p "$ROOT/envs/mjlab/bin/python" mjlab==1.6.0 tensorboard

# the batch script always activates the main env (job.py, experiments/*), so it always exists
[ -d "$ROOT/envs/main" ] || uv venv -q -p 3.12 "$ROOT/envs/main"
if [ "$MODE" = "--with-main" ] || [ "$MODE" = "--with-sam" ]; then
  uv pip install -q -p "$ROOT/envs/main/bin/python" --torch-backend "$TORCH_BACKEND" torch torchvision transformers \
    -r "$REPO/requirements.txt"
fi
if [ "$MODE" = "--with-sam" ]; then
  [ -d "$ROOT/sam3" ] || git clone -q https://github.com/facebookresearch/sam3.git "$ROOT/sam3"
  [ -d "$ROOT/envs/sam" ] || uv venv -q -p 3.12 "$ROOT/envs/sam"
  uv pip install -q -p "$ROOT/envs/sam/bin/python" --torch-backend "$TORCH_BACKEND" torch torchvision \
    "setuptools<81" -e "$ROOT/sam3[notebooks]"
  bash "$REPO/scripts/setup_sam3d.sh" "$ROOT"  # the 3D pose front end: $ROOT/sam3d_env, $ROOT/sam-3d-body
fi
# Gated weights are fetched here, on the login node, in case compute nodes can't reach the Hub.
if [ -f "$ROOT/env/HF_TOKEN" ] && [ -x "$ROOT/envs/main/bin/python" ]; then
  (cd "$REPO" && A2H_SECRETS_DIR="$ROOT/env" "$ROOT/envs/main/bin/python" - <<'PY') || echo "weight prefetch skipped"
import importlib.util, os, sys
sys.path.insert(0, os.getcwd())  # the repo, for config.get_secret
if importlib.util.find_spec("huggingface_hub"):
    import config
    from huggingface_hub import snapshot_download
    token = config.get_secret("HF_TOKEN", required=False)
    for repo in ("facebook/vjepa2-vitl-fpc64-256", "facebook/sam3.1", "facebook/sam-3d-body-dinov3",
                 "Ruicheng/moge-2-vitl-normal"):
        snapshot_download(repo, token=token)
        print("cached", repo)
PY
fi
echo SETUP_DONE
