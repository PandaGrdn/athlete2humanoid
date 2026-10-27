"""mjlab tasks that learn from the video's V-JEPA latent instead of (only) copying the retarget.
Runs inside mjlab's virtualenv; takes the same flags as mjlab's `train`:

    A2H_LATENT_BUNDLE=reward.pt $MJLAB_PY mjlab_tasks.py A2H-Hybrid-Flat-Unitree-G1 \
        --env.commands.motion.motion-file motion.npz ...

- A2H-Hybrid-Flat-Unitree-G1: latent reward + the tracking rewards at 1/5 weight, relaxed
  terminations: the robot may deviate from the retarget where that looks more like the video.
- A2H-Latent-Flat-Unitree-G1: latent reward only; the retarget is still the policy's input
  (it says what to do next) and its starting states, but is never scored.
- A2H-LatentPhase-Flat-Unitree-G1: latent reward only, and the policy never sees the retarget
  either: its inputs are its own body state and a clock (the motion's phase). The retarget only
  gives the starting states and, for the critic during training, privileged information.

A2H_REWARD_CONFIG=next_reward_config.json (muse/reward_agent.py's format; train_motion.py
--reward-config sets it) overrides the latent weight, SIGMA, tracking scale and termination
thresholds below. The latent-only tasks keep tracking_scale 0.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent.reward_net import BODIES, FPS, WINDOW, load_bundle, window_center_step  # noqa: E402

import mjlab.tasks  # noqa: E402,F401  (registers mjlab's own tasks)
from mjlab.managers.observation_manager import ObservationTermCfg  # noqa: E402
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg  # noqa: E402
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg  # noqa: E402
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: E402

SIGMA = 0.1  # reward = exp(-max(0, cos_ref - cos) / SIGMA): 1 when the robot matches the video at
             # least as well as the retarget does at that moment (cos_ref, from the bundle)


class LatentReward:
    """Per env: a ring buffer of the last ~1 s of body positions -> the distilled V-JEPA latent
    -> cosine similarity with the video's latent at the same moment of the motion."""

    def __init__(self, cfg, env):
        self.cmd = env.command_manager.get_term("motion")
        assert tuple(self.cmd.cfg.body_names) == BODIES, "reward net expects mjlab's G1 tracking bodies"
        self.net, self.target, b = load_bundle(os.environ["A2H_LATENT_BUNDLE"], env.device)
        self.video_fps, self.cos_ref, self.dt = b["video_fps"], b["cos_ref"], env.step_dt
        self.origins = env.scene.env_origins
        dev, n = env.device, env.num_envs
        self.L = int(np.ceil((WINDOW - 1) / FPS / self.dt)) + 2
        self.buf = torch.zeros(n, self.L, len(BODIES), 3, device=dev)
        self.last_ts = torch.full((n,), -10, dtype=torch.long, device=dev)
        # The 32 window frames (30 fps, ending now) as fractional positions in the 50 Hz buffer.
        pos = (self.L - 1) - (WINDOW - 1 - torch.arange(WINDOW, device=dev)) / FPS / self.dt
        self.i0 = pos.floor().long()
        self.i1 = (self.i0 + 1).clamp(max=self.L - 1)
        self.w = (pos - self.i0)[None, :, None, None]
        self.back = torch.arange(self.L, 0, -1, device=dev)  # history steps ts - L ... ts - 1

    def reset(self, env_ids=None):
        self.last_ts[slice(None) if env_ids is None else env_ids] = -10

    def __call__(self, env, sigma=SIGMA, **kwargs):
        ts = self.cmd.time_steps
        jumped = (ts != self.last_ts + 1).nonzero().squeeze(1)  # reset, or motion wrapped around
        if len(jumped):
            # Unknown past: assume the robot was on the reference until now.
            hist = (ts[jumped, None] - self.back).clamp(min=0)
            self.buf[jumped] = self.cmd.motion.body_pos_w[hist] + self.origins[jumped, None, None]
        self.buf = torch.roll(self.buf, -1, 1)
        self.buf[:, -1] = self.cmd.robot_body_pos_w
        self.last_ts = ts.clone()

        win = self.buf[:, self.i0] * (1 - self.w) + self.buf[:, self.i1] * self.w
        with torch.no_grad():
            pred = self.net(win)
        step = torch.round(window_center_step(ts * self.dt * self.video_fps)).long()
        valid = (step >= 0) & (step < len(self.target))
        step = step.clamp(0, len(self.target) - 1)
        cos = torch.nn.functional.cosine_similarity(pred, self.target[step], dim=-1)
        return torch.exp(-(self.cos_ref[step] - cos).clamp(min=0) / sigma) * valid


def motion_phase(env, command_name="motion", harmonics=4):
    """A clock: how far through the motion the robot should be, as sin/cos of its phase."""
    cmd = env.command_manager.get_term(command_name)
    t = cmd.time_steps.float() / cmd.motion.time_step_total
    k = torch.arange(1, harmonics + 1, device=t.device) * 2 * torch.pi * t[:, None]
    return torch.cat([k.sin(), k.cos()], -1)


REFERENCE_OBS = ("command", "motion_anchor_pos_b", "motion_anchor_ori_b")
TRACKING_TERMS = ("motion_global_root_pos", "motion_global_root_ori", "motion_body_pos",
                  "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel")


REWARD_CONFIG_ENV = "A2H_REWARD_CONFIG"


def reward_config(tracking_scale):
    """This task's settings: the defaults, overridden by the A2H_REWARD_CONFIG file if set."""
    rc = {"latent_weight": 3.0, "sigma": SIGMA, "tracking_scale": tracking_scale,
          # Relaxed: end an episode when the robot has fallen, not when it leaves the retarget.
          "terminations": {"anchor_pos": 0.5, "anchor_ori": 1.2, "ee_body_pos": 0.6}}
    path = os.environ.get(REWARD_CONFIG_ENV)
    if path:
        user = json.loads(Path(path).read_text())
        rc |= {k: float(user[k]) for k in ("latent_weight", "sigma") if k in user}
        if tracking_scale and "tracking_scale" in user:
            rc["tracking_scale"] = float(user["tracking_scale"])
        rc["terminations"] |= {k: float(v) for k, v in user.get("terminations", {}).items() if k in rc["terminations"]}
    return rc


def latent_env_cfg(tracking_scale, play=False, phase_only=False):
    rc = reward_config(tracking_scale)
    tracking_scale, th = rc["tracking_scale"], rc["terminations"]
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)
    if phase_only:
        actor = cfg.observations["actor"].terms
        for k in REFERENCE_OBS:
            del actor[k]
        actor["phase"] = ObservationTermCfg(func=motion_phase)
    for k in TRACKING_TERMS:
        if tracking_scale:
            cfg.rewards[k].weight *= tracking_scale
        else:
            del cfg.rewards[k]
    cfg.rewards["latent"] = RewardTermCfg(func=LatentReward, weight=rc["latent_weight"], params={"sigma": rc["sigma"]})
    cfg.terminations["anchor_pos"].params["threshold"] = th["anchor_pos"]
    cfg.terminations["anchor_ori"].params["threshold"] = th["anchor_ori"]
    if tracking_scale:
        cfg.terminations["ee_body_pos"].params["threshold"] = th["ee_body_pos"]
    else:
        del cfg.terminations["ee_body_pos"]
    return cfg


for task, scale, phase in (("A2H-Hybrid-Flat-Unitree-G1", 0.2, False), ("A2H-Latent-Flat-Unitree-G1", 0.0, False),
                          ("A2H-LatentPhase-Flat-Unitree-G1", 0.0, True)):
    register_mjlab_task(task_id=task, env_cfg=latent_env_cfg(scale, phase_only=phase),
                        play_env_cfg=latent_env_cfg(scale, play=True, phase_only=phase),
                        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(), runner_cls=MotionTrackingOnPolicyRunner)


if __name__ == "__main__":
    from mjlab.scripts.train import main
    main()
