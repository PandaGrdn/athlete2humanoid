"""Run a trained mjlab tracking policy on its motion, headless: saves a video and tracking metrics.

Runs inside the mjlab virtualenv (mjlab's own `play` opens an interactive viewer instead):
    $MJLAB_PY mjlab_rollout.py --checkpoint logs/.../model_3000.pt --motion motion.npz --out outputs/clip
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mediapy
import numpy as np
import torch

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Tracking-Flat-Unitree-G1"


def rollout(checkpoint, motion_file, out_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(TASK, play=True)  # play config: starts at frame 0, no episode timeout
    agent_cfg = load_rl_cfg(TASK)
    env_cfg.commands["motion"].motion_file = str(motion_file)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    motion = env.command_manager.get_term("motion")

    n_steps = len(np.load(motion_file)["joint_pos"]) - 1
    obs, _ = venv.reset()
    robot = env.scene["robot"]
    frames, qpos, joint_err, anchor_err, resets = [], [], [], [], 0
    with torch.inference_mode():
        for _ in range(n_steps):
            obs, _, dones, _ = venv.step(policy(obs))
            joint_err.append((motion.robot_joint_pos - motion.joint_pos).abs().mean().item())
            anchor_err.append((motion.robot_anchor_pos_w - motion.anchor_pos_w).norm(dim=-1).item())
            resets += int(dones.sum().item())  # a reset mid-clip means the robot fell
            frames.append(env.render())
            # MuJoCo qpos layout (base xyz, base quat wxyz, joints) so our own renderer can
            # redraw the rollout exactly like the reference.
            qpos.append(torch.cat([robot.data.root_link_pos_w[0] - env.scene.env_origins[0],
                                   robot.data.root_link_quat_w[0], robot.data.joint_pos[0]]).cpu().numpy())

    fps = round(1 / env.step_dt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out_dir / "policy.mp4"), np.stack(frames), fps=fps)
    np.save(out_dir / "policy_qpos.npy", np.stack(qpos))
    metrics = {"checkpoint": str(checkpoint), "steps": n_steps, "fps": fps, "falls": resets,
               "mean_joint_error_rad": float(np.mean(joint_err)),
               "mean_torso_position_error_m": float(np.mean(anchor_err)),
               "max_torso_position_error_m": float(np.max(anchor_err))}
    (out_dir / "policy_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rollout(a.checkpoint, a.motion, a.out)
