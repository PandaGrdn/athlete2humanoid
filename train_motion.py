"""G1 motion CSV -> mjlab motion -> PPO tracking policy -> rollout video + metrics.

Runs in the main env; mjlab steps run in mjlab's own virtualenv (A2H_MJLAB_PYTHON).

    python train_motion.py --csv outputs/squat/motion.csv --fps 30 --out outputs/squat --iters 1500
"""

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASK = "Mjlab-Tracking-Flat-Unitree-G1"


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def train_motion(csv, fps, out, iters=1500, num_envs=4096, mjlab_python=None):
    py = Path(mjlab_python or os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python"))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MUJOCO_GL="egl", WANDB_MODE="disabled")

    # 1. CSV -> motion.npz (forward kinematics at 50 fps). The converter always writes
    #    /tmp/motion.npz and then "uploads" it to W&B, which WANDB_MODE=disabled makes a no-op.
    with tempfile.TemporaryDirectory() as tmp:
        run([py, "-m", "mjlab.scripts.csv_to_npz", "--input-file", Path(csv).resolve(),
             "--output-name", out.name, "--input-fps", fps, "--output-fps", 50], env=env, cwd=tmp)
    npz = out / "motion.npz"
    shutil.copy("/tmp/motion.npz", npz)

    # 2. PPO motion tracking.
    log_root = out / "logs"
    run([py.parent / "train", TASK, "--env.commands.motion.motion-file", npz,
         "--env.scene.num-envs", num_envs, "--agent.max-iterations", iters,
         "--agent.save-interval", 250, "--agent.logger", "tensorboard", "--log-root", log_root],
        env=env, cwd=out)

    # 3. Roll out the last checkpoint headless.
    ckpt = max(glob.glob(f"{log_root}/*/*/model_*.pt"), key=lambda p: int(Path(p).stem.split("_")[1]))
    run([py, ROOT / "mjlab_rollout.py", "--checkpoint", ckpt, "--motion", npz, "--out", out], env=env)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--num-envs", type=int, default=4096)
    a = ap.parse_args()
    train_motion(a.csv, a.fps, a.out, a.iters, a.num_envs)
