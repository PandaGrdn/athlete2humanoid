#!/usr/bin/env bash
# On a Colab GPU VM: train A2H-KickBall from the front-kick tracker, then evaluate it against the
# tracker itself (zero-shot, ball inputs at zero weight). Inputs in $IN: model_1499.pt, motion.npz.
#   IN=/content/a2h/inputs OUT=/content/a2h/outputs/kick_ball ITERS=800 bash scripts/kick_ball_job.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=/content/mjlab_env/bin/python
mkdir -p "$OUT"
bash "$REPO/scripts/setup_colab.sh"
export MUJOCO_GL=egl WANDB_MODE=disabled
"$PY" "$REPO/experiments/kick_ball_eval.py" --checkpoint "$IN/model_1499.pt" --motion "$IN/motion.npz" \
  --out "$OUT/zero_shot" -n 256
if [ -n "${PREV:-}" ]; then  # an earlier trained checkpoint, scored by this commit's evaluation
  "$PY" "$REPO/experiments/kick_ball_eval.py" --checkpoint "$PREV" --motion "$IN/motion.npz" --out "$OUT/prev" -n 256
fi
A2H_INIT_CHECKPOINT="$IN/model_1499.pt" A2H_INIT_MODE=init "$PY" "$REPO/mjlab_objects.py" A2H-KickBall-Flat-Unitree-G1 \
  --env.commands.motion.motion-file "$IN/motion.npz" --env.scene.num-envs 4096 --agent.max-iterations "${ITERS:-800}" \
  --agent.save-interval 100 --agent.logger tensorboard --log-root "$OUT/logs"
CK=$(python3 -c "import glob,sys; print(max(glob.glob('$OUT/logs/*/*/model_*.pt'), key=lambda p: int(p.rsplit('_',1)[1][:-3])))")
cp "$CK" "$OUT/"
"$PY" "$REPO/experiments/kick_ball_eval.py" --checkpoint "$CK" --motion "$IN/motion.npz" --out "$OUT/trained" -n 256
tar -czf "$OUT/../kick_ball_results.tar.gz" -C "$OUT" --exclude='logs/*/*/model_*.pt' .
echo KICK_BALL_DONE
