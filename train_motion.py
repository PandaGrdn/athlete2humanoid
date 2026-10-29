"""G1 motion CSV -> mjlab motion -> PPO tracking policy -> rollout video + metrics.

Runs in the main env; mjlab steps run in mjlab's own virtualenv (A2H_MJLAB_PYTHON).

    python train_motion.py --csv outputs/squat/motion.csv --fps 30 --out outputs/squat --iters 1500
    python train_motion.py ... --task A2H-Hybrid-Flat-Unitree-G1 --bundle reward.pt  # latent reward
    python train_motion.py ... --reward-config next_reward_config.json  # muse/reward_agent.py's proposal
    python train_motion.py ... --init outputs/run/policy_hybrid/model_1999.pt   # build off saved weights

Weights are never thrown away: checkpoints every 250 iterations, and weights.json records what the
policy was trained from. If an earlier attempt in the same --out left checkpoints (the VM died), it
resumes from the newest instead of starting over. --init starts from another policy's weights, of
the same or another task (policy_weights.py adapts grown observations).
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


def csv_to_npz(py, csv, fps, dst, env):
    """mjlab's converter: CSV (fps) -> motion.npz at 50 fps by forward kinematics. It always writes
    /tmp/motion.npz and then "uploads" it to W&B, which WANDB_MODE=disabled makes a no-op."""
    with open("/tmp/a2h_csv_to_npz.lock", "w") as lock, tempfile.TemporaryDirectory() as tmp:
        fcntl.flock(lock, fcntl.LOCK_EX)  # trainings may run in parallel; the converter's output path is fixed
        run([py, "-m", "mjlab.scripts.csv_to_npz", "--input-file", Path(csv).resolve(),
             "--output-name", Path(dst).stem, "--input-fps", fps, "--output-fps", 50], env=env, cwd=tmp)
        shutil.copy("/tmp/motion.npz", dst)
    return Path(dst)


def train_motion(csv, fps, out, iters=1500, num_envs=4096, mjlab_python=None, task=TASK, bundle=None,
                 reward_config=None, task_spec=None, importance=None, init=None, variants=None, rollout_importance=None,
                 video_latent=None, rollout_video_latent=None, train_motion_file=None):
    """variants: experiments/variant_policy.py's variants.json, next to its motion_variants.npz (or
    train_motion_file): train on those (A2H_VARIANTS), then roll out on csv's clip (the original) with
    rollout_importance. video_latent / rollout_video_latent: the video's latent per step of the training
    / rollout motion (A2H_VIDEO_LATENT, for A2H-TrackingVJ)."""
    py = Path(mjlab_python or os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python"))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), WANDB_MODE="disabled")
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
    log_root = out / "logs"
    done = sorted(glob.glob(f"{log_root}/*/*/model_*.pt"), key=lambda p: int(Path(p).stem.split("_")[1]))
    start = {}
    if done:  # an interrupted attempt: carry on from its newest checkpoint
        last = int(Path(done[-1]).stem.split("_")[1])
        start = {"mode": "resume", "checkpoint": done[-1], "iteration": last}
        iters = max(iters - last, 1)
    elif init:
        start = {"mode": "init", "checkpoint": str(Path(init).resolve())}
    if start:
        env["A2H_INIT_CHECKPOINT"], env["A2H_INIT_MODE"] = start["checkpoint"], start["mode"]
        print(f"starting from {start}", flush=True)
    # Our tasks (mjlab_tasks.py) register themselves and then hand over to mjlab's trainer, which
    # (policy_weights.py) starts from A2H_INIT_CHECKPOINT when set.
    trainer = [py.parent / "train"] if task == TASK and not start else [py, ROOT / "mjlab_tasks.py"]

    # 1. CSV -> motion.npz (forward kinematics at 50 fps): the clip it is evaluated on.
    npz = csv_to_npz(py, csv, fps, out / "motion.npz", env)
    train_npz, roll_env = npz, env
    if variants:  # trained on the variants end to end; rolled out on the clip itself
        train_npz = Path(train_motion_file or Path(variants).resolve().parent.parent / "motion_variants.npz").resolve()
        env = dict(env, A2H_VARIANTS=str(Path(variants).resolve()))
        if rollout_importance:
            roll_env = dict(roll_env, A2H_IMPORTANCE=str(Path(rollout_importance).resolve()))
    if video_latent:
        env = dict(env, A2H_VIDEO_LATENT=str(Path(video_latent).resolve()))
    if rollout_video_latent:
        roll_env = dict(roll_env, A2H_VIDEO_LATENT=str(Path(rollout_video_latent).resolve()))

    # 2. PPO motion tracking.
    run([*trainer, task, "--env.commands.motion.motion-file", train_npz,
         "--env.scene.num-envs", num_envs, "--agent.max-iterations", iters,
         "--agent.save-interval", 250, "--agent.logger", "tensorboard", "--log-root", log_root],
        env=env, cwd=out)

    # 3. Roll out the last checkpoint headless.
    ckpt = max(glob.glob(f"{log_root}/*/*/model_*.pt"), key=lambda p: int(Path(p).stem.split("_")[1]))
    ck = Path(ckpt).resolve()
    meta = {"task": task, "checkpoint": str(ck.relative_to(out) if ck.is_relative_to(out) else ck), "iterations": int(Path(ckpt).stem.split("_")[1]) + 1,
            "started_from": start or None, "motion_csv": str(csv), "fps": fps, "num_envs": num_envs,
            **({"variants": str(Path(variants).resolve())} if variants else {}),
            **{k: str(Path(v).resolve()) for k, v in (("bundle", bundle), ("task_spec", task_spec),
                                                     ("importance", importance), ("reward_config", reward_config)) if v}}
    (out / "weights.json").write_text(json.dumps(meta, indent=1))
    run([py, ROOT / "mjlab_rollout.py", "--checkpoint", ckpt, "--motion", npz, "--out", out, "--task", task], env=roll_env)
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
    ap.add_argument("--init", help="start from this checkpoint (model_*.pt) instead of from scratch")
    ap.add_argument("--variants", help="experiments/variant_policy.py's variants.json: train on the variants")
    ap.add_argument("--rollout-importance", help="importance.json for the rollout on the clip itself (with --variants)")
    ap.add_argument("--train-motion", help="with --variants: the motion file they are laid out in")
    ap.add_argument("--video-latent", help="the video's latent per step of the training motion (A2H-TrackingVJ)")
    ap.add_argument("--rollout-video-latent", help="... and of the rollout's (the clip's own)")
    a = ap.parse_args()
    train_motion(a.csv, a.fps, a.out, a.iters, a.num_envs, task=a.task, bundle=a.bundle, reward_config=a.reward_config,
                 task_spec=a.task_spec, importance=a.importance, init=a.init, variants=a.variants,
                 rollout_importance=a.rollout_importance, video_latent=a.video_latent,
                 rollout_video_latent=a.rollout_video_latent, train_motion_file=a.train_motion)
