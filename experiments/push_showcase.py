"""Recorded pushes for the website's 3D showcase: one trained policy, shoved on a grid of moments,
directions and strengths, every response saved.

    $A2H_MJLAB_PYTHON experiments/push_showcase.py --runs <folder with policy_<name>/> --policy tracking_var --out <dir>

Runs in mjlab's virtualenv (like robustness_eval.py's stage 1). One environment per push plus one
that is never pushed (the nominal run). Every episode starts at the clip's first frame exactly
(no reference-state-init noise, no observation noise, no domain randomisation), so all envs replay the
nominal run until their own push, up to float noise that the jumps amplify (`prefix_dev`: the largest
|qpos - nominal| before the push; ~0.02 midway through the clip). So each recording also keeps its own
env's last --pre frames before the push: the site crossfades from the nominal run into it there. A push is the same shove as robustness_eval.py's: the root's horizontal
velocity kicked by `mps` in one direction (training's pushes reach ~0.7 m/s).

push_showcase.npz:
    start_step        the reference frame every env starts at (after reset; 0 or 1)
    nominal  (T, 36)  root pos (m), root quat (wxyz), 29 joint positions (model order); 50 Hz
    q_<i>    (T - max(0, step_i - pre), 36) env i's run from `pre` frames before its push step to the end of the clip
    step, angle_rad, mps, fall_step (-1: never fell)  per push i; prefix_dev (max |qpos - nominal| before the push)
    fps, fall_z, fall_tilt
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_TASK = "Mjlab-Tracking-Flat-Unitree-G1"
STEP_HZ = 50
# experiments/latent_policy.py's names for the tracking policies (it can't be imported in mjlab's venv)
TASKS = {"tracking": ENV_TASK, "tracking_var": "A2H-Tracking-Flat-Unitree-G1"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(a):
    import torch

    import mjlab.tasks  # noqa: F401  (registers the tasks)
    import mjlab_tasks  # noqa: F401  (and ours)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    d = Path(a.runs) / f"policy_{a.policy}"
    ckpt = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
    task = a.task or TASKS[a.policy]

    T = len(np.load(d / "motion.npz")["joint_pos"])
    steps = [round(s * STEP_HZ) for s in np.arange(a.every, T / STEP_HZ - a.tail, a.every)]
    angles = np.arange(a.dirs) * 2 * np.pi / a.dirs
    grid = [(k, ang, v) for k in steps for ang in angles for v in a.mps]
    N = 1 + len(grid)
    log(f"{ckpt.name} ({task}): {len(steps)} moments x {a.dirs} directions x {len(a.mps)} strengths = {len(grid)} pushes")

    cfg = load_env_cfg(ENV_TASK, play=True)
    actor = load_env_cfg(task, play=True).observations["actor"]
    assert list(actor.terms) == list(cfg.observations["actor"].terms), "the policy reads other observations than the stock task"
    cfg.commands["motion"].motion_file = str(d / "motion.npz")
    cfg.commands["motion"].debug_vis = False
    cfg.scene.num_envs = N
    cfg.seed = a.seed
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}  # falls: ours, latched
    # No events: the play config still randomises friction, mass, COM and joint offsets per env at start-up,
    # which would make every env a different robot (and its run not the nominal one before its push).
    log("dropping events: " + ", ".join(f"{k} ({getattr(v, 'mode', '?')})" for k, v in cfg.events.items()))
    cfg.events = {}
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    rl = load_rl_cfg(task)
    venv = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(venv, asdict(rl), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    pi = runner.get_inference_policy(device=device)

    cmd, robot = env.command_manager.get_term("motion"), env.scene["robot"]
    zero = lambda r: {k: (0.0, 0.0) for k in r}  # noqa: E731
    cmd.cfg.pose_range, cmd.cfg.velocity_range = zero(cmd.cfg.pose_range), zero(cmd.cfg.velocity_range)
    cmd.cfg.joint_position_range = (0.0, 0.0)
    cmd.cfg.sampling_mode = "start"  # every episode from the clip's first frame (as mjlab_rollout.py's play)

    step_of = torch.tensor([-1] + [g[0] for g in grid], device=device)
    ang = torch.tensor([0.0] + [g[1] for g in grid], device=device)
    mps = torch.tensor([0.0] + [g[2] for g in grid], device=device)

    def tilt(quat):
        return torch.acos((1 - 2 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)).clamp(-1, 1))

    fall = torch.full((N,), -1, dtype=torch.long, device=device)
    t0 = time.time()
    with torch.inference_mode():
        obs, _ = venv.reset()
        start = cmd.time_steps.unique()
        assert len(start) == 1, f"envs start at different frames: {start.tolist()[:20]}"
        log(f"every env starts at reference frame {int(start)}")
        L = T - 1 - int(start)  # steps until the reference's last frame: then it would wrap
        qpos = torch.zeros(N, L, 36, device=device)
        for t in range(L):
            ids = (step_of == t).nonzero().squeeze(1)
            if len(ids):
                vel = robot.data.root_link_vel_w[ids].clone()
                vel[:, 0] += mps[ids] * torch.cos(ang[ids])
                vel[:, 1] += mps[ids] * torch.sin(ang[ids])
                robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
            obs, _, _, _ = venv.step(pi(obs))
            z = cmd.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
            down = (z < a.fall_z) | (tilt(cmd.robot_anchor_quat_w) > a.fall_tilt)
            fall = torch.where((fall < 0) & down, t, fall)
            qpos[:, t] = torch.cat([robot.data.root_link_pos_w - env.scene.env_origins,
                                    robot.data.root_link_quat_w, robot.data.joint_pos], -1)
    log(f"{L} steps in {time.time() - t0:.0f} s")

    q = qpos.cpu().numpy()
    nominal = q[0]
    out = {"nominal": nominal, "pre": np.array(a.pre), "start_step": np.array(int(start)), "fps": np.array(STEP_HZ), "fall_z": np.array(a.fall_z), "fall_tilt": np.array(a.fall_tilt)}
    dev = []
    for i, (k, _, _) in enumerate(grid, 1):
        # qpos[t] is the state after step t; the push lands before step k, so the run from k on is new
        dev.append(float(np.abs(q[i, :k] - nominal[:k]).max()) if k else 0.0)
        out[f"q_{i - 1}"] = q[i, max(0, k - a.pre):]
    out |= {"step": np.array([g[0] for g in grid]), "angle_rad": np.array([g[1] for g in grid], np.float32),
            "mps": np.array([g[2] for g in grid], np.float32), "fall_step": fall[1:].cpu().numpy(),
            "prefix_dev": np.array(dev, np.float32)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / "push_showcase.npz", **out)
    falls = fall[1:].cpu().numpy() >= 0
    summary = {"checkpoint": str(ckpt), "task": task, "pushes": len(grid), "nominal_fell": int(fall[0]) >= 0,
               "max_prefix_dev": max(dev), "fall_rate": {f"{v:g} m/s": float(falls[np.array([g[2] for g in grid]) == v].mean())
                                                         for v in a.mps}}
    (Path(a.out) / "push_showcase.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder with policy_<name>/model_*.pt + motion.npz")
    ap.add_argument("--policy", default="tracking_var")
    ap.add_argument("--task", help="its mjlab task, if not tracking / tracking_var (tracking-style observations only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=float, default=0.25, help="seconds between push moments")
    ap.add_argument("--tail", type=float, default=0.5, help="no pushes this close to the clip's end (s)")
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--mps", type=float, nargs="+", default=[0.5, 1.0], help="push strengths (m/s)")
    ap.add_argument("--pre", type=int, default=15, help="frames of each push env's own run kept before its push")
    ap.add_argument("--fall-z", type=float, default=0.35)
    ap.add_argument("--fall-tilt", type=float, default=1.4)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
