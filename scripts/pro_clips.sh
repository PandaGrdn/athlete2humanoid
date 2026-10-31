#!/bin/bash
# New videos -> trained, robustness-tested policies in one unattended job (scripts/remote_run.py pro):
#   1. SAM 3.1 and SAM 3D Body installed at job start (the Modal image has neither), then experiments/process_clips.py
#   2. per clip, experiments/multiclip_policy.py: tracking_var / multi / multi_vjc, each from its own
#      checkpoint in inputs/ (--init-dir), trained on the clip's variants, then every robustness suite
#   [POLICIES="multi"] bash scripts/pro_clips.sh <iters> <video>...   (default: all three, compared)
#   [IN_PLACE=sprint,hurdles] ...: those videos (by file stem) run in place (pipeline.py --in-place)
#   <iters> 0: process only, nothing trained (the Modal image has no SAM, so new videos go through here)
set -u
ITERS=$1; shift
POLICIES="${POLICIES:-tracking_var multi multi_vjc}"
D=$A2H_JOB_DATA
OUT=$D/outputs/pro
mkdir -p "$OUT" "$D/inputs/clips"

if [ ! -x /content/sam_env/bin/python ]; then
  [ -d /content/sam3 ] || git clone -q https://github.com/facebookresearch/sam3.git /content/sam3
  uv venv -q -p 3.12 /content/sam_env
  uv pip install -q -p /content/sam_env/bin/python --torch-backend auto torch torchvision \
    "setuptools<81" -e "/content/sam3[notebooks]"
fi
export A2H_SAM_PYTHON=/content/sam_env/bin/python
/content/sam_env/bin/python -c "import torch, sam3; print('SAM env OK, CUDA:', torch.cuda.is_available())" || exit 1
bash scripts/setup_sam3d.sh /content || exit 1  # the 3D pose front end
export A2H_SAM3D_PYTHON=/content/sam3d_env/bin/python A2H_SAM3D_DIR=/content/sam-3d-body

python3 experiments/process_clips.py "$@" --out "$OUT/process" --in-place "${IN_PLACE:-}" || exit 1
if [ "$ITERS" = 0 ]; then
  cd "$OUT" && tar czf "$OUT/results.tar.gz" process/results.tar.gz && echo "packed $OUT/results.tar.gz (process only)"
  exit 0
fi
tar xzf "$OUT/process/results.tar.gz" -C "$D/inputs"

for c in $(ls "$OUT/process" >/dev/null && tar tzf "$OUT/process/results.tar.gz" | cut -d/ -f2 | sort -u); do
  echo "=== $c"
  python3 experiments/multiclip_policy.py "$D/inputs/clips" --out "$OUT/$c" --train "$c" --held-out \
    --policies $POLICIES --init-dir "$D/inputs" --pca "$D/inputs/vjl_pca.npz" \
    --iters "$ITERS" --full-eval || echo "$c: multiclip_policy failed"
done

cd "$OUT" && tar czf "$OUT/results.tar.gz" process/results.tar.gz $(ls -d */results.tar.gz 2>/dev/null | grep -v '^process/')
echo "packed $OUT/results.tar.gz"
