"""G1 motion CSV -> mjlab motion -> PPO tracking policy -> rollout video + metrics.

Runs in the main env; mjlab steps run in mjlab's own virtualenv (A2H_MJLAB_PYTHON).

    python train_motion.py --csv outputs/squat/motion.csv --fps 30 --out outputs/squat --iters 1500
    python train_motion.py ... --task A2H-Hybrid-Flat-Unitree-G1 --bundle reward.pt  # latent reward
    python train_motion.py ... --reward-config next_reward_config.json  # muse/reward_agent.py's proposal
"""

import argparse
import fcntl
import glob
import json
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


def train_motion(csv, fps, out, iters=1500, num_envs=4096, mjlab_python=None, task=TASK, bundle=None,
                 reward_config=None, task_spec=None, importance=None):
    py = Path(mjlab_python or os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python"))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MUJOCO_GL="egl", WANDB_MODE="disabled")
    if bundle:
        env["A2H_LATENT_BUNDLE"] = str(Path(bundle).resolve())
    if task_spec:  # the task-first tasks' stage reward (skill/task.py)
        env["A2H_TASK_SPEC"] = str(Path(task_spec).resolve())
    if importance:  # per reference frame goal relevance, weighting their style terms
        env["A2H_IMPORTANCE"] = str(Path(importance).resolve())
    if reward_config:
        # Read by mjlab_tasks.latent_env_cfg (the A2H tasks); its iterations replace `iters`. The copy
        # in `out` records what this policy was trained with, for muse/reward_agent.py.
        rc = json.loads(Path(reward_config).read_text())
        iters = int(rc.get("iterations", iters))
        (out / "reward_config.json").write_text(json.dumps(rc | {"iterations": iters}, indent=1))
        env["A2H_REWARD_CONFIG"] = str(out / "reward_config.json")
    # Our tasks (mjlab_tasks.py) register themselves and then hand over to mjlab's trainer.
    trainer = [py.parent / "train"] if task == TASK else [py, ROOT / "mjlab_tasks.py"]

    # 1. CSV -> motion.npz (forward kinematics at 50 fps). The converter always writes
    #    /tmp/motion.npz and then "uploads" it to W&B, which WANDB_MODE=disabled makes a no-op.
    npz = out / "motion.npz"
    with open("/tmp/a2h_csv_to_npz.lock", "w") as lock, tempfile.TemporaryDirectory() as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX)  # trainings may run in parallel; the converter's output path is fixed
        run([py, "-m", "mjlab.scripts.csv_to_npz", "--input-file", Path(csv).resolve(),
             "--output-name", out.name, "--input-fps", fps, "--output-fps", 50], env=env, cwd=tmp)
        shutil.copy("/tmp/motion.npz", npz)

    # 2. PPO motion tracking.
    log_root = out / "logs"
    run([*trainer, task, "--env.commands.motion.motion-file", npz,
         "--env.scene.num-envs", num_envs, "--agent.max-iterations", iters,
         "--agent.save-interval", 250, "--agent.logger", "tensorboard", "--log-root", log_root],
        env=env, cwd=out)

    # 3. Roll out the last checkpoint headless.
    ckpt = max(glob.glob(f"{log_root}/*/*/model_*.pt"), key=lambda p: int(Path(p).stem.split("_")[1]))
    run([py, ROOT / "mjlab_rollout.py", "--checkpoint", ckpt, "--motion", npz, "--out", out, "--task", task], env=env)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--bundle", help="latent reward bundle (latent/distill.py) for the A2H tasks")
    ap.add_argument("--reward-config", help="reward settings for the A2H tasks (JSON); its iterations override --iters")
    ap.add_argument("--task-spec", help="skill/task.py task_spec.json for the A2H-Task tasks")
    ap.add_argument("--importance", help="importance.json weighting the A2H-Task tasks' style terms")
    a = ap.parse_args()
    train_motion(a.csv, a.fps, a.out, a.iters, a.num_envs, task=a.task, bundle=a.bundle, reward_config=a.reward_config,
                 task_spec=a.task_spec, importance=a.importance)
