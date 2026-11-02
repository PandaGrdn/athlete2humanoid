"""Attempts: the policy rolled out on the clip's reference every N iterations while it trains, so a page can show
it learn (web/src/buildview/learn.tsx) without a video being rendered on the GPU.

    python -m a2h_env train <bundle.zip> --iters 1000 --attempt-every 50

Each attempt is one episode of the current policy in a one-env copy of the environment on the reference alone
(mjlab's play config: it starts at the clip's first frame, no pushes or noise), stopped where the episode ends (a
fall, or the reference lost). It is written next to the checkpoints:

    <out>/attempts/iter_0050.npz   qpos (T, 36) float32 at the env's control rate (50 Hz: root xyz, root quaternion
                                   wxyz, the 29 joints in the G1's order, as every qpos in the bundle), fps, iteration,
                                   reward (the episode's return) and, when the repository's skill/task.py is importable,
                                   the key-moment stages the attempt reached
    <out>/attempts/index.json      every attempt so far (what the server mirrors and the page lists), the reward curve
                                   (mjlab's mean episode return per iteration) and the run's settings

The npz is ~20-40 KB for a 5 s clip; a 1,000-iteration run with an attempt every 50 writes 21 of them.
"""

import json
import re
import statistics
import time
from pathlib import Path

import numpy as np

DIR = "attempts"
FILE = "iter_{:04d}.npz"
NAME = re.compile(r"^iter_(\d{4,6})\.npz$")
# The tracking bodies skill/task.py scores, in its order (server/live_worker.py BODIES); mjlab's G1 names them the same.
BODIES = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
          "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
          "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
          "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")


def _spec_eval():
    """skill.task.evaluate when the repository is importable (the GPU job runs from a checkout), else None."""
    try:
        from skill import task  # noqa: PLC0415
        return task.evaluate
    except Exception:  # noqa: BLE001  (a bundle trained away from the repository: no stage scores)
        return None


def evaluate_stages(spec, pos, fps):
    """The key-moment stages a rollout's body positions (T, 14, 3) reach, or None without skill/task.py."""
    ev = _spec_eval()
    if ev is None or not isinstance(spec, dict) or not spec.get("stages"):
        return None
    s = dict(spec)
    s.setdefault("down", spec.get("down_ref", 0.05))
    s.setdefault("air", spec.get("air_ref", 0.06))
    r = ev(s, pos, fps)
    return {"stages_completed": int(r["stages_completed"]), "stages_total": int(r["stages_total"]),
            "success": bool(r["success"]), "fell_at_s": r.get("fell_at_s")}


class Attempts:
    """Records attempts for one training run. The one-env play environment is built on the first call."""

    def __init__(self, bundle, out, iterations, every, clip_actions=None, work_dir=None, device="cpu", log=print):
        self.bundle, self.dir, self.iterations, self.every = bundle, Path(out) / DIR, int(iterations), int(every)
        self.clip_actions, self.work_dir, self.device, self.log = clip_actions, work_dir, device, log
        self.env = self.venv = None
        self.rows, self.curve = [], []
        self.task = None
        self.dir.mkdir(parents=True, exist_ok=True)

    def _build(self):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper

        from .env import env_cfg, task_of
        t0 = time.time()
        cfg = env_cfg(self.bundle, "reference", 1, play=True, work_dir=self.work_dir, device=self.device)
        self.env = ManagerBasedRlEnv(cfg=cfg, device=self.device)
        self.venv = RslRlVecEnvWrapper(self.env, clip_actions=self.clip_actions)
        robot = self.env.scene["robot"]
        names = list(robot.body_names)
        self.body_ids = [names.index(b) for b in BODIES] if all(b in names for b in BODIES) else None
        self.steps = int(len(np.load(cfg.commands["motion"].motion_file)["joint_pos"]) - 1)
        self.fps = int(round(1 / self.env.step_dt))
        self.task = task_of("reference", self.bundle.reward_config.get("feet_height"))
        self.log(f"[a2h_env] attempts: one-env play environment on the reference ({self.steps} steps at {self.fps} Hz) "
                 f"built in {time.time() - t0:.0f} s")

    def note_iteration(self, runner):
        """Called every iteration: keeps the reward curve (mjlab's mean episode return over the last 100 episodes)."""
        buf = getattr(getattr(runner, "logger", None), "rewbuffer", None)
        if buf:
            self.curve.append([int(runner.current_learning_iteration) + 1, round(float(statistics.mean(buf)), 4)])

    def record(self, runner, iteration):
        """Roll the runner's current policy out once and write iter_<iteration>.npz + index.json."""
        import torch
        if self.env is None:
            self._build()
        t0 = time.time()
        runner.alg.eval_mode()
        try:
            policy = runner.alg.get_policy()
            robot = self.env.scene["robot"]
            origin = self.env.scene.env_origins[0]
            qpos, pos, ret, fell = [], [], 0.0, False
            # no_grad, not inference_mode: a second rollout would then reset buffers that became inference tensors
            with torch.no_grad():
                obs, _ = self.venv.reset()
                for _ in range(self.steps):
                    obs, rew, dones, _ = self.venv.step(policy(obs))
                    ret += float(rew[0])
                    if bool(dones[0]):  # the episode ended: a fall, or the reference lost (play config has no time-out)
                        fell = True
                        break
                    qpos.append(torch.cat([robot.data.root_link_pos_w[0] - origin, robot.data.root_link_quat_w[0],
                                           robot.data.joint_pos[0]]).float().cpu().numpy())
                    if self.body_ids is not None:
                        pos.append((robot.data.body_link_pos_w[0, self.body_ids] - origin).float().cpu().numpy())
        finally:
            runner.alg.train_mode()
        qpos = np.stack(qpos).astype(np.float32) if qpos else np.zeros((0, 36), np.float32)
        row = {"iteration": int(iteration), "file": FILE.format(int(iteration)), "frames": int(len(qpos)),
               "seconds": round(len(qpos) / self.fps, 3), "reward": round(ret, 4),
               "reward_per_step": round(ret / max(1, len(qpos)), 5), "fell": fell,
               "fell_at_s": round(len(qpos) / self.fps, 3) if fell else None,
               "stages_completed": None, "stages_total": None, "success": None}
        if pos and len(pos) > 1:
            try:
                st = evaluate_stages(self.bundle.spec, np.stack(pos), self.fps)
            except Exception as e:  # noqa: BLE001  (a stage score must never stop training)
                self.log(f"[a2h_env] attempt {iteration}: stage score failed ({type(e).__name__}: {str(e)[:120]})")
                st = None
            if st:
                row.update({k: st[k] for k in ("stages_completed", "stages_total", "success")})
                if st.get("fell_at_s") is not None:
                    row["fell_at_s"] = st["fell_at_s"]
        tmp = self.dir / (row["file"] + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, qpos=qpos, fps=np.array(self.fps, np.int32), iteration=np.array(int(iteration), np.int32),
                                reward=np.array(ret, np.float32), fell=np.array(fell),
                                stages_completed=np.array(-1 if row["stages_completed"] is None else row["stages_completed"], np.int32),
                                stages_total=np.array(-1 if row["stages_total"] is None else row["stages_total"], np.int32))
        tmp.replace(self.dir / row["file"])
        self.rows = [r for r in self.rows if r["iteration"] != row["iteration"]] + [row]
        self.rows.sort(key=lambda r: r["iteration"])
        self.write_index()
        stages = (f", stages {row['stages_completed']}/{row['stages_total']}" if row["stages_total"] is not None else "")
        self.log(f"[a2h_env] attempt at iteration {iteration}: reward {ret:.2f}{stages}"
                 f"{', fell at %.1f s' % row['fell_at_s'] if fell else ''} ({len(qpos)} steps, {time.time() - t0:.1f} s)")
        return row

    def write_index(self):
        index = {"format": "a2h-attempts", "fps": self.fps, "steps": self.steps, "iterations": self.iterations,
                 "attempt_every": self.every, "task": self.task, "policy": "tracking_var",
                 "bundle": self.bundle.id, "attempts": self.rows, "curve": self.curve[-4000:]}
        tmp = self.dir / "index.json.tmp"
        tmp.write_text(json.dumps(index, separators=(",", ":")))
        tmp.replace(self.dir / "index.json")
        return index

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = self.venv = None
