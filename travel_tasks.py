"""Tracking tasks for motions that travel across the floor (docs/FRONTIER_TRAVEL.md). mjlab_tasks.py
registers TASKS with travel_env_cfg(); each is A2H-Tracking-Flat-Unitree-G1 with one change, so each
change can be ablated alone:

- A2H-TravelReward-Flat-Unitree-G1: two root terms next to mjlab's. mjlab's anchor position term
  (std 0.3 m, weight 0.5) is worth ~0 beyond half a metre, and nothing scores the root's own
  velocity (motion_body_lin_vel averages 14 bodies, swinging limbs included). Added: the anchor
  position at std 1 m and the anchor's horizontal velocity at std 0.7 m/s, weight 1 each.
- A2H-TravelTerm-Flat-Unitree-G1: an episode also ends when the torso is further than a threshold
  from the reference horizontally. mjlab's anchor termination is height-only, so a robot that stays
  put never fails, and adaptive sampling (which sends episodes to where they fail) never sends them
  to the travelling phase. The threshold is a curriculum: 1 m, tightened to 0.4 m over 1,000 iterations.
- A2H-TravelPhase-Flat-Unitree-G1: half of the episode starts are drawn from the travelling phase,
  weighted by how far the reference's root goes in the next second beyond 0.2 m; the other half by
  mjlab's adaptive sampling. A clip that never goes 0.2 m in a second (the in-place clips) is
  sampled as before.
- A2H-Travel-Flat-Unitree-G1: all three.

TRAVEL holds the settings; A2H_TRAVEL_CONFIG=travel.json overrides any of them. The observations
are mjlab's, so a tracking checkpoint warm-starts every one unchanged, and they are evaluated in the
stock tracking env (experiments/robustness_eval.py).
"""

import json
import os
from pathlib import Path

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

TRAVEL = {"root_pos_std": 1.0, "root_pos_weight": 1.0,  # coarse anchor position term (m)
          "root_vel_std": 0.7, "root_vel_weight": 1.0,  # anchor horizontal velocity term (m/s)
          "term_start": 1.0, "term_end": 0.4, "term_ramp_iters": 1000,  # horizontal anchor termination (m)
          "phase_mix": 0.5, "phase_window_s": 1.0, "phase_min_m": 0.2}  # travel-weighted start sampling
TRAVEL_CONFIG_ENV = "A2H_TRAVEL_CONFIG"
CHANGES = ("reward", "term", "phase")
TASKS = {"A2H-TravelReward-Flat-Unitree-G1": ("reward",), "A2H-TravelTerm-Flat-Unitree-G1": ("term",),
         "A2H-TravelPhase-Flat-Unitree-G1": ("phase",), "A2H-Travel-Flat-Unitree-G1": CHANGES}


def travel_config():
    """TRAVEL, overridden by the A2H_TRAVEL_CONFIG file if set."""
    tc = dict(TRAVEL)
    path = os.environ.get(TRAVEL_CONFIG_ENV)
    if path:
        tc |= {k: float(v) for k, v in json.loads(Path(path).read_text()).items() if k in tc}
    return tc


def _motion(env, command_name="motion"):
    return env.command_manager.get_term(command_name)


def anchor_position_error_exp(env, std, command_name="motion"):
    """mjlab's motion_global_anchor_position_error_exp, here at any std."""
    cmd = _motion(env, command_name)
    return torch.exp(-(cmd.anchor_pos_w - cmd.robot_anchor_pos_w).square().sum(-1) / std ** 2)


def anchor_lin_vel_xy_error_exp(env, std, command_name="motion"):
    """The torso's horizontal velocity against the reference's: the root's travel, without the limbs."""
    cmd = _motion(env, command_name)
    return torch.exp(-(cmd.anchor_lin_vel_w[:, :2] - cmd.robot_anchor_lin_vel_w[:, :2]).square().sum(-1) / std ** 2)


def bad_anchor_xy(env, threshold, command_name="motion"):
    """The torso further than threshold (m) from the reference's, horizontally."""
    cmd = _motion(env, command_name)
    return (cmd.anchor_pos_w[:, :2] - cmd.robot_anchor_pos_w[:, :2]).norm(dim=-1) > threshold


def ramp(start, end, steps, step):
    return start + (end - start) * min(1.0, step / max(steps, 1))


class TerminationCurriculum:
    """Moves a termination's threshold from `start` to `end` over `ramp_steps` env steps."""

    def __init__(self, cfg, env):
        self.term = env.termination_manager.get_term_cfg(cfg.params["term_name"])

    def __call__(self, env, env_ids, term_name, start, end, ramp_steps):
        th = ramp(start, end, ramp_steps, env.common_step_counter)
        self.term.params["threshold"] = th
        return torch.tensor(th)


def travel_weights(anchor_xy, window, min_m, seg_end=None):
    """(T, 2) reference torso positions -> (T,) start weights: how far the root goes in the next
    `window` steps (net, so sway cancels) beyond min_m; a window stops at its variant's last step."""
    T = len(anchor_xy)
    t = torch.arange(T, device=anchor_xy.device)
    end = (t + window).clamp(max=T - 1)
    if seg_end is not None:
        end = torch.minimum(end, seg_end)
    return ((anchor_xy[end] - anchor_xy).norm(dim=-1) - min_m).clamp(min=0)


class TravelSampling:
    """A curriculum term only in name: at start-up it wraps the motion command's adaptive sampling so
    that a `mix` share of the episodes it starts are drawn by travel_weights() instead. mjlab's own
    sampling still runs first for every env, so its failure statistics and metrics are unchanged."""

    def __init__(self, cfg, env):
        p = cfg.params
        cmd = _motion(env, p.get("command_name", "motion"))
        anchor = cmd.motion.body_pos_w[:, cmd.motion_anchor_body_index, :2]
        window = int(round(p["window_s"] / env.step_dt))
        self.w = travel_weights(anchor, window, p["min_m"], getattr(cmd, "seg_end", None))
        self.share = float(self.w.gt(0).float().mean())  # of the motion's steps: the travelling phase
        if self.w.sum() <= 0:
            return  # nothing travels: leave mjlab's sampling alone
        inner, mix = cmd._adaptive_sampling, p["mix"]

        def sampling(env_ids):
            inner(env_ids)
            pick = torch.rand(len(env_ids), device=cmd.device) < mix
            if pick.any():
                ids = env_ids[pick]
                cmd.time_steps[ids] = torch.multinomial(self.w, len(ids), replacement=True)
        cmd._adaptive_sampling = sampling

    def __call__(self, env, env_ids, **params):
        return torch.tensor(self.share)


def travel_env_cfg(base, changes, play=False):
    """base(play=play) (mjlab_tasks: A2H-Tracking's cfg) with `changes` (a subset of CHANGES). Play cfgs
    are the stock ones (their observations are the same; robustness_eval reads only those)."""
    cfg = base(play=play)
    if play:
        return cfg
    tc, steps_per_iter = travel_config(), unitree_g1_tracking_ppo_runner_cfg().num_steps_per_env
    if "reward" in changes:
        cfg.rewards["travel_root_pos"] = RewardTermCfg(func=anchor_position_error_exp, weight=tc["root_pos_weight"],
                                                       params={"std": tc["root_pos_std"]})
        cfg.rewards["travel_root_vel"] = RewardTermCfg(func=anchor_lin_vel_xy_error_exp, weight=tc["root_vel_weight"],
                                                       params={"std": tc["root_vel_std"]})
    if "term" in changes:
        cfg.terminations["anchor_xy"] = TerminationTermCfg(func=bad_anchor_xy, params={"threshold": tc["term_start"]})
        cfg.curriculum["anchor_xy"] = CurriculumTermCfg(func=TerminationCurriculum, params={
            "term_name": "anchor_xy", "start": tc["term_start"], "end": tc["term_end"],
            "ramp_steps": int(tc["term_ramp_iters"] * steps_per_iter)})
    if "phase" in changes:
        cfg.curriculum["travel_sampling"] = CurriculumTermCfg(func=TravelSampling, params={
            "mix": tc["phase_mix"], "window_s": tc["phase_window_s"], "min_m": tc["phase_min_m"]})
    return cfg

