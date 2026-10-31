"""Does the front kick hit the ball? A policy on A2H-KickBall-Flat-Unitree-G1 (mjlab_objects.py), run
on its clip from the start, headless, in mjlab's virtualenv:

    $MJLAB_PY experiments/kick_ball_eval.py --checkpoint model_499.pt --motion motion.npz --out outputs/kick_ball

A checkpoint of the plain tracking task (no ball inputs) is grown as policy_weights.py grows it for
training, so it acts exactly as it did without the ball: the zero-shot baseline.

Per condition (n envs each): the share of kicks the kicking leg hits, the ball's horizontal speed
along the strike direction (the best over the second after the hit, kicks that hit), how far it
travels in that second, and falls. Condition "jitter_<cm>" moves each ball up to that far from the
strike point. Env 0 of the first condition is rendered to <out>/kick_ball.mp4.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mjlab_objects  # noqa: E402  (registers A2H-KickBall)
import policy_weights  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "A2H-KickBall-Flat-Unitree-G1"


def load_policy(runner, checkpoint, device):
    try:
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        return "loaded"
    except RuntimeError:  # a checkpoint without the ball inputs
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        policy_weights.init_from(runner.alg, ckpt)
        return "grown (ball inputs at zero weight)"


def run(checkpoint, motion_file, n, jitter, render=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = load_env_cfg(TASK, play=True)
    cfg.commands["motion"].motion_file = str(motion_file)
    cfg.scene.num_envs = n
    cfg.events["ball"].params["jitter"] = jitter
    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if render else None)
    agent = load_rl_cfg(TASK)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(venv, asdict(agent), device=device)
    how = load_policy(runner, checkpoint, device)
    policy = runner.get_inference_policy(device=device)
    steps = len(np.load(motion_file)["joint_pos"]) - 1
    obs, _ = venv.reset()
    b = env.a2h_ball
    K = len(b.strike)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    leg_hit = torch.zeros(n, K, dtype=torch.bool, device=device)
    any_hit = torch.zeros(n, K, dtype=torch.bool, device=device)
    speed = torch.zeros(n, K, device=device)
    start = torch.zeros(n, K, 3, device=device)
    travel = torch.zeros(n, K, device=device)
    frames, joint_err = [], []
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, dones, _ = venv.step(policy(obs))
            fell |= dones.bool()
            joint_err.append((b.cmd.robot_joint_pos - b.cmd.joint_pos).abs().mean(-1)[~fell].mean().item()
                             if (~fell).any() else float("nan"))
            k = b.kick.clamp(min=0)
            on = (b.kick >= 0) & ~fell
            rows = torch.arange(n, device=device)
            new = on & b.hit & ~any_hit[rows, k]
            any_hit[rows[new], k[new]] = True
            start[rows[new], k[new]] = b.ball.data.root_link_pos_w[new]
            fly = on & b.flying()
            v = (b.ball.data.root_link_lin_vel_w[:, :2] * b.d[:, :2]).sum(-1)
            leg_hit[rows[fly], k[fly]] = True
            speed[rows[fly], k[fly]] = torch.maximum(speed[rows[fly], k[fly]], v[fly])
            dist = (b.ball.data.root_link_pos_w[:, :2] - start[rows, k, :2]).norm(dim=-1)
            travel[rows[fly], k[fly]] = dist[fly]
            if render:
                frames.append(env.render())
    hit = leg_hit.float()
    out = {"policy": how, "n": n, "jitter_m": jitter, "kicks_per_clip": K,
           "fall_rate": fell.float().mean().item(), "mean_joint_error_rad": float(np.nanmean(joint_err)),
           "leg_hit_rate": hit.mean().item(), "leg_hit_rate_per_kick": hit.mean(0).tolist(),
           "other_hit_rate": (any_hit & ~leg_hit).float().mean().item(),
           "ball_speed_mps": speed[leg_hit].mean().item() if leg_hit.any() else 0.0,
           "ball_travel_1s_m": travel[leg_hit].mean().item() if leg_hit.any() else 0.0}
    env.close()
    return out, frames, round(1 / env.step_dt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--jitters", default="0,0.05", help="ball offsets to test, m")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    res = {}
    for i, j in enumerate(float(x) for x in a.jitters.split(",")):
        r, frames, fps = run(a.checkpoint, a.motion, a.n, j, render=(i == 0 and not a.no_video))
        res["clean" if j == 0 else f"jitter_{round(j * 100)}cm"] = r
        print(json.dumps(r), flush=True)
        if frames:
            import mediapy
            mediapy.write_video(str(outd / "kick_ball.mp4"), np.stack(frames), fps=fps)
    (outd / "kick_ball_eval.json").write_text(json.dumps({"checkpoint": str(a.checkpoint), **res}, indent=1))
