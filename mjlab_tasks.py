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

- A2H-Task-Flat-Unitree-G1 / A2H-TaskPhase-Flat-Unitree-G1: task first. A2H_TASK_SPEC=task_spec.json
  (skill/task.py) says what the action is, as stages (stand, squat, jump, land); the reward is for
  doing them in order, however and whenever. The latent (and, for Task, 0.1x tracking) is a weak
  style term, optionally weighted in time by A2H_IMPORTANCE; both get the stability terms below
  and episodes end on their fall rule only.
- A2H-TaskCmd-Flat-Unitree-G1: TaskPhase with commanded targets (skill/commands.py): each episode
  asks for its own squat depth and jump clearance (optionally reps) inside the G1's ranges, under a
  curriculum that widens them as the task succeeds; the targets are the last actor/critic inputs,
  so a TaskPhase checkpoint warm-starts it unchanged (policy_weights.py). It also starts episodes
  with the feet on the floor (ground_starts), pays for soft landings (LandingImpact) and
  penalises the latent reward's uncertainty (LatentReward, lambda 4): docs/PIPELINE_IMPROVEMENTS.md.

- A2H-Tracking-Flat-Unitree-G1: mjlab's tracking task, here so that it too can train on variants.
- A2H-TrackingVJ-Flat-Unitree-G1: tracking whose policy also sees the video's V-JEPA latent, now and
  0.5 / 1 s ahead (A2H_VIDEO_LATENT, per motion step; latent/video_condition.py), so it is told in
  V-JEPA's latent space what the video shows. A2H-TrackingFuture-Flat-Unitree-G1 is its control: the
  retarget's joints and pelvis height 0.5 / 1 s ahead instead, the same look-ahead without V-JEPA
  (experiments/multiclip_policy.py). A2H-TrackingVJCausal sees only the video up to where the robot is:
  the latent of the ~1 s ending now and 0.5 s ago, from trailing windows (latent/video_condition.py).
A2H_VARIANTS=variants.json (experiments/variant_policy.py) makes any of them train on variants of the
clip (skill/variations.py: mirrored, re-timed, scaled) laid end to end in one motion file: each env
stays inside its variant (VariantMotionCommand) and the task reward scores it against that variant's
own keypose spec. mjlab's adaptive sampling over the whole file then favours the variants it fails.
A2H_REWARD_CONFIG=next_reward_config.json (muse/reward_agent.py's format; train_motion.py
--reward-config sets it) overrides the latent weight, SIGMA, tracking scale and termination
thresholds below. The latent-only tasks keep tracking_scale 0.

The latent-only tasks also get stability_terms(): the latent reward says what the motion should
look like, not how to stay on your feet doing it (see its docstring). The hybrid task gets them
only with "stability": {"hybrid": true} in the config.
"""

import json
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent import reward_net  # noqa: E402
from latent import video_condition  # noqa: E402
from latent.reward_net import BODIES, FPS, WINDOW, load_bundle, window_center_step  # noqa: E402
from skill import commands as skill_commands  # noqa: E402
from skill import task as skill_task  # noqa: E402

import mjlab.tasks  # noqa: E402,F401  (registers mjlab's own tasks)
from mjlab.envs import mdp as env_mdp  # noqa: E402
from mjlab.managers.curriculum_manager import CurriculumTermCfg  # noqa: E402
from mjlab.managers.observation_manager import ObservationTermCfg  # noqa: E402
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: E402
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg  # noqa: E402
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg  # noqa: E402
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg  # noqa: E402
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: E402

SIGMA = 0.1  # reward = exp(-max(0, cos_ref - cos) / SIGMA): 1 when the robot matches the video at
             # least as well as the retarget does at that moment (cos_ref, from the bundle)


class LatentReward:
    """Per env: a ring buffer of the last ~1 s of body positions -> the distilled V-JEPA latent
    -> cosine similarity with the video's latent at the same moment of the motion.

    params["uncertainty"] = {"lambda": l, "mc": m} (A2H_REWARD_CONFIG "latent_uncertainty"; l = 0,
    the default, leaves the reward as it was): the cosine minus l x how much more uncertain the
    network is here than on the reference at the same moment (latent/reward_net.py uncertainty();
    an ensemble bundle's members, or m MC-dropout samples), so motions the network was never shown
    can't be scored up by its errors."""

    def __init__(self, cfg, env):
        self.cmd = env.command_manager.get_term("motion")
        assert tuple(self.cmd.cfg.body_names) == BODIES, "reward net expects mjlab's G1 tracking bodies"
        self.net, self.target, b = load_bundle(os.environ["A2H_LATENT_BUNDLE"], env.device)
        self.video_fps, self.cos_ref, self.dt = b["video_fps"], b["cos_ref"], env.step_dt
        unc = cfg.params.get("uncertainty") or {}
        self.lam, self.mc = float(unc.get("lambda", 0.0)), int(unc.get("mc", 8))
        if self.lam:  # the reference's own uncertainty at every step: the penalty's zero
            ref, steps = self.cmd.motion.body_pos_w, torch.arange(self.cmd.motion.time_step_total, device=env.device)
            with torch.no_grad():
                self.u_ref = torch.cat([reward_net.uncertainty(reward_net.member_predictions(
                    self.net, reward_net.reference_windows(ref, ts, self.dt), self.mc)) for ts in steps.split(256)])
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

    def __call__(self, env, sigma=SIGMA, **kwargs):  # kwargs: params["uncertainty"], read at __init__
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
        if self.lam:
            with torch.no_grad():
                u = reward_net.uncertainty(reward_net.member_predictions(self.net, win, self.mc))
            cos = cos - self.lam * (u - self.u_ref[ts.clamp(max=len(self.u_ref) - 1)]).clamp(min=0)
        return torch.exp(-(self.cos_ref[step] - cos).clamp(min=0) / sigma) * valid


def motion_phase(env, command_name="motion", harmonics=4):
    """A clock: how far through the motion the robot should be, as sin/cos of its phase."""
    cmd = env.command_manager.get_term(command_name)
    t = cmd.time_steps.float() / cmd.motion.time_step_total
    k = torch.arange(1, harmonics + 1, device=t.device) * 2 * torch.pi * t[:, None]
    return torch.cat([k.sin(), k.cos()], -1)


# Stability: balance terms that never compare the robot with the retarget step by step. The only
# use of the reference is two numbers per clip, the most its torso ever leans and the lowest it
# ever goes, so a squat may go as deep as the clip does but not fold over or sit down.
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
STABILITY = {
    "hybrid": False,       # also apply to the hybrid task
    "alive": 1.0,          # per second upright: falling early costs what the rest of the clip would pay
    "termination": -100.0, # once per fall (x dt: -2)
    "upright": -2.0,       # per rad the torso leans past the clip's own max lean + tilt_margin
    "tilt_margin": 0.15,
    "height": -10.0,       # per m the torso sinks below the clip's lowest + height_margin
    "height_margin": 0.1,
    "fall_tilt": 0.35,     # fallen (episode ends): leaning this far past the clip's max lean,
    "fall_height": 0.2,    # or this far below its lowest torso height
    "feet_slip": -0.5,     # squared xy speed of feet on the floor
    "action_rate": -0.1,   # mjlab's action_rate_l2 weight
    "joint_acc": -2.5e-7,
    "push_scale": 1.5,     # training pushes ramp from mjlab's range to push_scale x it (x/y) ...
    "push_ramp_iters": 1000.0,  # ... over this many PPO iterations
}


def _tilt(quat):
    """Angle between a body's z axis and world up (quat wxyz)."""
    return torch.acos((1 - 2 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)).clamp(-1, 1))


def _torso(env, command_name="motion"):
    """(torso tilt, torso height above the floor, the clip's max tilt, the clip's min height)."""
    cmd = env.command_manager.get_term(command_name)
    a = cmd.motion_anchor_body_index
    z = cmd.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (_tilt(cmd.robot_anchor_quat_w), z, _tilt(cmd.motion.body_quat_w[:, a]).max(),
            cmd.motion.body_pos_w[:, a, 2].min())


def torso_lean_excess(env, margin):
    tilt, _, max_tilt, _ = _torso(env)
    return (tilt - max_tilt - margin).clamp(min=0)


def _extra_depth(env):
    """How much deeper than the clip's deepest squat this episode is commanded to go (TaskCmd)."""
    m = getattr(env, "a2h_task", None)
    return 0.0 if m is None else m.extra_depth


def torso_height_deficit(env, margin):
    _, z, _, min_z = _torso(env)
    return (min_z - margin - _extra_depth(env) - z).clamp(min=0)


def torso_fallen(env, tilt_margin, height_margin):
    tilt, z, max_tilt, min_z = _torso(env)
    return (tilt > max_tilt + tilt_margin) | (z < min_z - height_margin - _extra_depth(env))


def feet_slip(env, contact_height=0.06):
    """Squared horizontal speed of the feet that are on the floor (ankle below contact_height)."""
    robot = env.scene["robot"]
    ids = [robot.body_names.index(n) for n in FEET]
    on_floor = robot.data.body_link_pos_w[:, ids, 2] - env.scene.env_origins[:, None, 2] < contact_height
    return (robot.data.body_link_lin_vel_w[:, ids, :2].square().sum(-1) * on_floor).sum(-1)


FEET_CONTACT = "feet_ground_contact"


def feet_contact_sensor():
    """mjlab's G1 foot-floor contact sensor (as its velocity task's): net force per foot."""
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    return ContactSensorCfg(
        name=FEET_CONTACT, primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                                                entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce",
        num_slots=1, track_air_time=True)


def impact_excess(force, weight, threshold):
    """(B, feet, 3) foot contact forces -> (B,) how far their total is above threshold x body
    weight, in body weights (0 while standing, walking or landing softly)."""
    return (force.norm(dim=-1).sum(-1) / weight - threshold).clamp(min=0)


class LandingImpact:
    """Hard landings: the feet's total contact force above `threshold` body weights, every step it
    lasts (mjlab's soft_landing only sees the first contact step). A jump's touchdown is where the
    G1's knees and ankles take their peak loads; a stiff landing is also where the task policies
    fall (A2H-TaskPhase, 750 iterations: every play rollout falls ~0.1 s after its first landing)."""

    def __init__(self, cfg, env):
        self.sensor = env.scene[FEET_CONTACT]
        self.weight = float(env.sim.mj_model.body_mass.sum()) * 9.81

    def __call__(self, env, threshold):
        f = self.sensor.data.force
        env.extras.setdefault("log", {})["Metrics/foot_force_peak_bw"] = (f.norm(dim=-1).sum(-1) / self.weight).max()
        return impact_excess(f, self.weight, threshold)


def ground_offsets(ankle_z, floor, hover_max, keep_out=(), fps=50.0):
    """Per reference frame, how far to lower a start state so its lower foot is on the floor:
    the lower ankle's height above `floor` (the robot's standing ankle height), where that is at
    most hover_max (the retarget's feet hover, e.g. ~5 cm in squat_jump's first frame), outside the
    keep_out windows (s; take-offs and flights, which really are in the air); 0 elsewhere."""
    lift = ankle_z.min(-1).values - floor
    off = torch.where(lift <= hover_max, lift.clamp(min=0), torch.zeros_like(lift))
    t = torch.arange(len(off), device=off.device) / fps
    for a, b in keep_out:
        off = torch.where((t >= a) & (t <= b), torch.zeros_like(off), off)
    return off


def ground_starts(cmd, spec, dt, keep_margin=(0.15, 0.1)):
    """Reference-state init with the feet on the floor: mjlab puts the robot at the reference's
    root pose, so a hovering retarget frame starts the episode with a fall (squat_jump from t=0:
    4 control steps in the air, then a 3.8-4.6 body-weight impact). The motion command's start
    states are lowered by ground_offsets(), from the task spec's standing ankle height, hover level
    (down_ref) and jump stages (kept out, keep_margin s before / after). -> the offsets."""
    keep = [(st["start_s"] - keep_margin[0], st["end_s"] + keep_margin[1])
            for st in spec["stages"] if st["kind"] in ("jump", "flight")]
    ankles = cmd.motion.body_pos_w[:, [skill_task.L_ANKLE, skill_task.R_ANKLE], 2]
    off = ground_offsets(ankles, spec["foot0"], max(spec.get("down_ref", 0.1), 0.05), keep, 1 / dt)
    write = cmd._write_reference_state_to_sim

    def write_grounded(env_ids, root_pos, *rest):
        root_pos = root_pos.clone()
        root_pos[:, 2] -= off[cmd.time_steps[env_ids].clamp(max=len(off) - 1)]
        write(env_ids, root_pos, *rest)

    cmd._write_reference_state_to_sim = write_grounded
    return off


class PushCurriculum:
    """Scales the push_robot event's x/y velocity range from 1 to `scale` over `ramp_steps` env steps."""

    def __init__(self, cfg, env):
        self.event = env.event_manager.get_term_cfg(cfg.params["event_name"])
        self.base = dict(self.event.params["velocity_range"])

    def __call__(self, env, env_ids, event_name, scale, ramp_steps):
        s = 1 + (scale - 1) * min(1.0, env.common_step_counter / ramp_steps)
        self.event.params["velocity_range"] = self.base | {
            k: (lo * s, hi * s) for k, (lo, hi) in self.base.items() if k in ("x", "y")}
        return {"push_scale": torch.tensor(s)}


def stability_terms(cfg, rc):
    """What the latent reward leaves out. It saturates at cos_ref, and a folded-over squat can still
    score 1, so it neither rewards staying upright nor punishes the lean that precedes a fall; with
    only action-rate penalties besides it, an early episode's return is negative, so falling ends
    the losses. And the latent tasks drop ee_body_pos, the early fall detector, which leaves
    terminations looser than an actual fall. So: an alive bonus and a fall penalty, a clip-relative
    fall termination, hinge penalties on over-leaning and sinking, foot slip and joint-acceleration
    costs, and training pushes that grow stronger. Zero weights leave a term out."""
    s = rc["stability"]
    terms = {"alive": (env_mdp.is_alive, {}), "termination": (env_mdp.is_terminated, {}),
             "upright": (torso_lean_excess, {"margin": s["tilt_margin"]}),
             "height": (torso_height_deficit, {"margin": s["height_margin"]}),
             "feet_slip": (feet_slip, {}), "joint_acc": (env_mdp.joint_acc_l2, {})}
    for k, (func, params) in terms.items():
        if s[k]:
            cfg.rewards[k] = RewardTermCfg(func=func, weight=s[k], params=params)
    cfg.rewards["action_rate_l2"].weight = s["action_rate"]
    cfg.terminations["fallen"] = TerminationTermCfg(
        func=torso_fallen, params={"tilt_margin": s["fall_tilt"], "height_margin": s["fall_height"]})
    if "push_robot" in cfg.events and s["push_scale"] != 1:  # play cfgs have no pushes
        steps = s["push_ramp_iters"] * unitree_g1_tracking_ppo_runner_cfg().num_steps_per_env
        cfg.curriculum["push"] = CurriculumTermCfg(func=PushCurriculum, params={
            "event_name": "push_robot", "scale": s["push_scale"], "ramp_steps": max(steps, 1)})
    return cfg


REFERENCE_OBS = ("command", "motion_anchor_pos_b", "motion_anchor_ori_b")
TRACKING_TERMS = ("motion_global_root_pos", "motion_global_root_ori", "motion_body_pos",
                  "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel")


REWARD_CONFIG_ENV = "A2H_REWARD_CONFIG"


def reward_config(tracking_scale):
    """This task's settings: the defaults, overridden by the A2H_REWARD_CONFIG file if set."""
    rc = {"latent_weight": 3.0, "sigma": SIGMA, "tracking_scale": tracking_scale,
          # Relaxed: end an episode when the robot has fallen, not when it leaves the retarget.
          "terminations": {"anchor_pos": 0.5, "anchor_ori": 1.2, "ee_body_pos": 0.6},
          "stability": dict(STABILITY)}
    path = os.environ.get(REWARD_CONFIG_ENV)
    if path:
        user = json.loads(Path(path).read_text())
        rc |= {k: float(user[k]) for k in ("latent_weight", "sigma") if k in user}
        if tracking_scale and "tracking_scale" in user:
            rc["tracking_scale"] = float(user["tracking_scale"])
        rc["terminations"] |= {k: float(v) for k, v in user.get("terminations", {}).items() if k in rc["terminations"]}
        rc["stability"] |= {k: bool(v) if k == "hybrid" else float(v)
                            for k, v in user.get("stability", {}).items() if k in rc["stability"]}
    return rc


UNCERTAINTY = {"lambda": 0.0, "mc": 8}  # latent reward uncertainty penalty (LatentReward); 0: off
# TaskCmd's default lambda. On the v2 squat bundle (8 MC-dropout samples) it takes the hacked
# policies' distilled-minus-real gap from +0.115/+0.101 to +0.023/-0.027 and moves in-distribution
# rollouts by <= 0.017 (docs/PIPELINE_IMPROVEMENTS.md).
UNCERTAINTY_CMD = 4.0


def uncertainty_config(default_lambda=None):
    """LatentReward's uncertainty penalty: UNCERTAINTY, overridden by A2H_REWARD_CONFIG's
    "latent_uncertainty": {"lambda": ..., "mc": ...} (or just a number, the lambda)."""
    u = dict(UNCERTAINTY) | ({} if default_lambda is None else {"lambda": default_lambda})
    path = os.environ.get(REWARD_CONFIG_ENV)
    if path:
        user = json.loads(Path(path).read_text()).get("latent_uncertainty", {})
        u |= {"lambda": float(user)} if isinstance(user, (int, float)) else {k: float(v) for k, v in user.items() if k in u}
    return u


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
    cfg.rewards["latent"] = RewardTermCfg(func=LatentReward, weight=rc["latent_weight"],
                                          params={"sigma": rc["sigma"], "uncertainty": uncertainty_config()})
    cfg.terminations["anchor_pos"].params["threshold"] = th["anchor_pos"]
    cfg.terminations["anchor_ori"].params["threshold"] = th["anchor_ori"]
    if tracking_scale:
        cfg.terminations["ee_body_pos"].params["threshold"] = th["ee_body_pos"]
    else:
        del cfg.terminations["ee_body_pos"]
    if not tracking_scale or rc["stability"]["hybrid"]:
        stability_terms(cfg, rc)
    return cfg


TASK_SPEC_ENV, IMPORTANCE_ENV, VARIANTS_ENV = "A2H_TASK_SPEC", "A2H_IMPORTANCE", "A2H_VARIANTS"


def load_variants():
    """A2H_VARIANTS: {"segments": [{name, start, length (50 Hz steps), spec (path, relative to the file)}]},
    each segment's spec read in as "spec_data"; None when unset."""
    path = os.environ.get(VARIANTS_ENV)
    if not path:
        return None
    v = json.loads(Path(path).read_text())
    for seg in v["segments"]:
        seg["spec_data"] = json.loads((Path(path).parent / seg["spec"]).read_text())
    return v


class VariantMotionCommand(MotionCommand):
    """The motion file is several variants end to end: an env that reaches the end of its variant is
    resampled (as mjlab does at the end of the file) instead of running on into the next one."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        segs = load_variants()["segments"]
        start = torch.tensor([g["start"] for g in segs], dtype=torch.long, device=self.device)
        length = torch.tensor([g["length"] for g in segs], dtype=torch.long, device=self.device)
        assert int(start[-1] + length[-1]) == self.motion.time_step_total, "variants.json doesn't match the motion file"
        self.seg_start = start
        self.seg_of = torch.repeat_interleave(torch.arange(len(segs), device=self.device), length)
        self.seg_last = torch.zeros(self.motion.time_step_total, dtype=torch.bool, device=self.device)
        self.seg_last[start + length - 1] = True
        self.seg_end = (start + length - 1)[self.seg_of]  # per step: the last step of its variant

    def _update_command(self, env_ids=None):
        ending = self.seg_last[self.time_steps].clone()
        if env_ids is not None:
            only = torch.zeros_like(ending)
            only[env_ids] = True
            ending &= only
        super()._update_command(env_ids)
        ids = ending.nonzero().squeeze(1)
        if len(ids):
            self._resample_command(ids)
            self._env.sim.forward()
            self.update_relative_body_poses()


@dataclass(kw_only=True)
class VariantMotionCommandCfg(MotionCommandCfg):
    def build(self, env):
        return VariantMotionCommand(self, env)


def use_variants(cfg):
    """With A2H_VARIANTS set, the motion command keeps each env inside its own variant."""
    if os.environ.get(VARIANTS_ENV):
        old = cfg.commands["motion"]
        cfg.commands["motion"] = VariantMotionCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old)})
    return cfg


VIDEO_LATENT_ENV = "A2H_VIDEO_LATENT"


def _ahead(cmd, steps):
    """Each env's motion step `steps` ahead (behind, if negative), held at the end (start) of its clip
    or of its variant."""
    T = cmd.motion.time_step_total
    ts = cmd.time_steps.clamp(max=T - 1)
    seg = getattr(cmd, "seg_end", None) is not None and len(cmd.seg_end) == T
    end = cmd.seg_end[ts] if seg else torch.full_like(ts, T - 1)
    start = cmd.seg_start[cmd.seg_of[ts]] if seg else torch.zeros_like(ts)
    return torch.maximum(torch.minimum(ts + steps, end), start)


def video_latent_table(env, T, causal=False):
    """(T, K): the video's latent per motion step: env.a2h_video_latent (robustness_eval sets it per
    condition), else A2H_VIDEO_LATENT's file; re-timed to T steps when the motion was (the video plays
    along; causal: each step holds the latest latent at or before it, never blending in a later one).
    None when there is none (the observation is being sized)."""
    lat = getattr(env, "a2h_video_latent", None)
    if lat is None:
        path = os.environ.get(VIDEO_LATENT_ENV)
        if not path:
            return None
        lat = env.a2h_video_latent = torch.as_tensor(np.load(path), dtype=torch.float32, device=env.device)
    if len(lat) != T:
        idx = torch.linspace(0, len(lat) - 1, T, device=lat.device)
        i0 = idx.floor().long().clamp(max=len(lat) - 1)
        i1, w = (i0 + 1).clamp(max=len(lat) - 1), (idx - i0)[:, None] * (not causal)
        lat = env.a2h_video_latent = lat[i0] * (1 - w) + lat[i1] * w
    return lat


def video_latent(env, command_name="motion", offsets_s=video_condition.OFFSETS_S, causal=False):
    """The video's V-JEPA latent (PCA'd) now and offsets_s ahead."""
    cmd = env.command_manager.get_term(command_name)
    lat = video_latent_table(env, cmd.motion.time_step_total, causal)
    if lat is None:
        return torch.zeros(env.num_envs, video_condition.K * len(offsets_s), device=env.device)
    return torch.cat([lat[_ahead(cmd, round(o / env.step_dt))] for o in offsets_s], -1)


def video_latent_causal(env, command_name="motion"):
    """The video's latent from trailing windows (A2H_VIDEO_LATENT holds those), now and 0.5 s ago."""
    return video_latent(env, command_name, offsets_s=video_condition.CAUSAL_OFFSETS_S, causal=True)


FUTURE_S = (0.5, 1.0)


def future_reference(env, command_name="motion", offsets_s=FUTURE_S):
    """The retarget ahead: its joint positions and pelvis height offsets_s from now (the video
    latent's control: the same look-ahead, without V-JEPA)."""
    cmd = env.command_manager.get_term(command_name)
    out = []
    for o in offsets_s:
        i = _ahead(cmd, round(o / env.step_dt))
        out += [cmd.motion.joint_pos[i], cmd.motion.body_pos_w[i, 0, 2:3]]
    return torch.cat(out, -1)


def conditioned_tracking_cfg(name, term, play=False):
    """mjlab's tracking task with one more input (last, so tracking weights warm-start it)."""
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)
    for group in ("actor", "critic"):
        cfg.observations[group].terms[name] = ObservationTermCfg(func=term)
    return cfg


class TaskReward:
    """skill/task.py's stage machine as a reward term: the video's action (e.g. stand, squat, jump,
    land, in that order) scored on the robot's own body. Poses and timing are free; an episode that
    starts mid-motion starts at the stage the reference is in then."""

    def __init__(self, cfg, env):
        self.cmd = env.command_manager.get_term("motion")
        assert tuple(self.cmd.cfg.body_names) == BODIES, "skill/task.py indexes mjlab's G1 tracking bodies"
        spec = json.loads(Path(os.environ[TASK_SPEC_ENV]).read_text())
        variants = load_variants()
        if variants:  # one keypose machine over every variant's spec; each env scored against its own
            from skill.keypose import KeyposeMachine
            self.m = KeyposeMachine([g["spec_data"] for g in variants["segments"]], env.num_envs, env.device, env.step_dt)
        else:
            self.m = skill_task.machine(spec, env.num_envs, env.device, env.step_dt)  # keypose or stage spec
        self.variants = variants is not None
        if cfg.params.get("grounded_rsi"):  # before the first reset (managers are built first)
            env.a2h_ground_offset = ground_starts(self.cmd, spec, env.step_dt)
        self.dt = env.step_dt
        self.last_ts = torch.full((env.num_envs,), -10, dtype=torch.long, device=env.device)
        env.a2h_task = self.m  # for the task_stage observation

    def reset(self, env_ids=None):
        self.last_ts[slice(None) if env_ids is None else env_ids] = -10

    def __call__(self, env, **kwargs):
        ts, pos = self.cmd.time_steps, self.cmd.robot_body_pos_w
        jumped = (ts != self.last_ts + 1).nonzero().squeeze(1)  # reset, or motion wrapped around
        if len(jumped):
            t = ts[jumped]
            if self.variants:  # the variant it is in now, and the time in that variant's clip
                v = self.cmd.seg_of[t]
                self.m.set_variant(jumped, v)
                t = t - self.cmd.seg_start[v]
            self.m.reset(jumped, t * self.dt, pos[jumped])
        self.last_ts = ts.clone()
        # Terminations are computed before rewards: a fall this step refunds the attempt.
        return self.m.step(pos, fell=env.termination_manager.terminated)


class TaskCmdReward(TaskReward):
    """TaskReward with commanded targets: every episode samples its own squat depth and jump
    clearance (skill/commands.py CommandSampler, params["command"] = its curriculum settings), and
    finished episodes (end, fall, or the motion wrapping around) drive the curriculum."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        assert isinstance(self.m, skill_task.StageMachine), "commanded targets need a stand/squat/jump/land spec (--spec stages)"
        self.env = env
        self.sampler = skill_commands.CommandSampler(self.m, seed=getattr(env.cfg, "seed", None),
                                                     **cfg.params.get("command", {}))
        self.sampler.sample()
        env.a2h_cmd = self.sampler

    def reset(self, env_ids=None):
        ids = torch.arange(self.m.n, device=self.m.dev) if env_ids is None else torch.as_tensor(env_ids, device=self.m.dev)
        self.sampler.record(ids, self.env.termination_manager.terminated[ids])
        self.sampler.sample(ids)
        super().reset(env_ids)

    def __call__(self, env, **kwargs):
        ts = self.cmd.time_steps
        wrapped = ((ts != self.last_ts + 1) & (self.last_ts >= 0)).nonzero().squeeze(1)
        if len(wrapped):  # the reference ran out mid-episode: the stage sequence restarts, same targets
            self.sampler.record(wrapped, False)
        return super().__call__(env, **kwargs)


def task_command(env):
    """The commanded targets (depth, clearance, reps), normalised to [-1, 1]."""
    m = getattr(env, "a2h_task", None)
    if m is None:
        return torch.zeros(env.num_envs, len(skill_task.COMMANDS), device=env.device)
    return m.command_obs()


def command_curriculum(env, env_ids):
    """Logs the command curriculum (CommandSampler updates itself as episodes end)."""
    s = getattr(env, "a2h_cmd", None)
    return {} if s is None else {k: torch.tensor(float(v)) for k, v in s.log().items()}


def task_stage(env):
    """The policy's view of the task: which stage it is in (one-hot), its progress, finished."""
    m = getattr(env, "a2h_task", None)
    if m is None:  # observations are sized before the reward terms exist
        return torch.zeros(env.num_envs, skill_task.MAX_STAGES + 2, device=env.device)
    return m.obs()


class TimeWeighted:
    """A style term (latent or tracking) weighted by how much the reference's current moment matters
    for the goal: A2H_IMPORTANCE = importance.json {"fps": f, "weight": [per reference frame, 0..1]}
    from the goal-relevance analysis. E.g. arm-waving before a jump gets ~0: not imitated."""

    def __init__(self, cfg, env):
        inner = cfg.params["inner"]
        # a class term is built with its own params (e.g. LatentReward's uncertainty settings)
        inner_cfg = SimpleNamespace(params=cfg.params.get("inner_params") or {}, weight=cfg.weight)
        self.inner = inner(inner_cfg, env) if isinstance(inner, type) else inner
        imp = json.loads(Path(os.environ[IMPORTANCE_ENV]).read_text())
        self.w = torch.tensor(imp["weight"], dtype=torch.float32, device=env.device)
        self.fps, self.dt = float(imp["fps"]), env.step_dt
        self.cmd = env.command_manager.get_term("motion")

    def reset(self, env_ids=None):
        if hasattr(self.inner, "reset"):
            self.inner.reset(env_ids)

    def __call__(self, env, inner=None, inner_params=None, **kwargs):
        i = (self.cmd.time_steps * self.dt * self.fps).long().clamp(0, len(self.w) - 1)
        w = self.w[i]
        m = getattr(env, "a2h_task", None)
        if getattr(m, "style_until_done", False):
            # A keypose task done early must not then be asked to look like the rest of the clip:
            # kp_sj taskphase did the 9 s clip in ~3 s, then toppled imitating it from the finish pose.
            # It also means finishing early forfeits the style reward still ahead.
            w = w * ~m.done
        return self.inner(env, **(inner_params or {})) * w


def task_config():
    """Weights of the task-first tasks: the stage reward leads, V-JEPA latent style follows (both in
    seconds of unit reward; see skill/task.py WEIGHTS), and the command curriculum (skill/commands.py
    CURRICULUM). Overridable via A2H_REWARD_CONFIG ("command": {...} for the curriculum)."""
    tc = {"task_weight": 2.0, "style_weight": 1.0, "landing_weight": None, "landing_threshold": LANDING_BW,
          "grounded_rsi": None}
    cmd = {}
    path = os.environ.get(REWARD_CONFIG_ENV)
    if path:
        user = json.loads(Path(path).read_text())
        tc |= {k: (bool(user[k]) if k == "grounded_rsi" else float(user[k])) for k in tc if k in user}
        cmd = {k: v for k, v in user.get("command", {}).items() if k in skill_commands.CURRICULUM or k == "ranges"}
    return tc | {"command": cmd}


LANDING_BW = 3.0  # landing penalty above this many body weights (the clip's own landing: ~1.9 BW)
LANDING_CMD = -5.0  # TaskCmd: per s per body weight above LANDING_BW (a 3-step 5 BW landing: -0.6; a jump pays 4)


def task_env_cfg(tracking_scale, play=False, phase_only=False, commanded=False, weighted_tracking=True):
    """Task-first hybrid: the semantic stage reward leads; the V-JEPA latent and (when
    tracking_scale > 0) the tracking terms are weak style terms, weighted in time by goal importance
    when A2H_IMPORTANCE is set. "Without falling" is part of the task, so both get stability_terms();
    episodes end on its fall rule only, never for leaving the retarget."""
    cfg = latent_env_cfg(tracking_scale, play=play, phase_only=phase_only)
    if "alive" not in cfg.rewards and "fallen" not in cfg.terminations:  # the hybrid path skipped them
        stability_terms(cfg, reward_config(tracking_scale))
    tc = task_config()
    grounded = tc["grounded_rsi"] if tc["grounded_rsi"] is not None else commanded
    cfg.rewards["task"] = RewardTermCfg(func=TaskCmdReward if commanded else TaskReward, weight=tc["task_weight"],
                                        params=({"command": tc["command"]} if commanded else {})
                                        | ({"grounded_rsi": True} if grounded else {}))
    if tc["style_weight"] and os.environ.get("A2H_LATENT_BUNDLE"):
        cfg.rewards["latent"].weight = tc["style_weight"]
        if commanded:  # new targets take it off the clip the network was distilled on: penalise its guesses
            cfg.rewards["latent"].params["uncertainty"] = uncertainty_config(default_lambda=UNCERTAINTY_CMD)
    else:
        del cfg.rewards["latent"]
    if os.environ.get(IMPORTANCE_ENV):
        # TaskHybrid (weighted_tracking=False) keeps tracking on the whole clip: it is the imitation half
        for k in ("latent", *(TRACKING_TERMS if weighted_tracking else ())):
            if k in cfg.rewards:
                t = cfg.rewards[k]
                cfg.rewards[k] = RewardTermCfg(func=TimeWeighted, weight=t.weight,
                                               params={"inner": t.func, "inner_params": t.params})
    for group in ("actor", "critic"):
        if group in cfg.observations:
            cfg.observations[group].terms["task_stage"] = ObservationTermCfg(func=task_stage)
            if commanded:  # last, so TaskPhase weights warm-start it (policy_weights.py)
                cfg.observations[group].terms["task_command"] = ObservationTermCfg(func=task_command)
    if commanded:
        cfg.curriculum["command"] = CurriculumTermCfg(func=command_curriculum, params={})
    # Landing (TaskCmd by default; the other task tasks via A2H_REWARD_CONFIG "landing_weight").
    landing = tc["landing_weight"] if tc["landing_weight"] is not None else (LANDING_CMD if commanded else 0.0)
    if landing:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_contact_sensor(),)
        cfg.rewards["landing"] = RewardTermCfg(func=LandingImpact, weight=landing,
                                               params={"threshold": tc["landing_threshold"]})
    for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
        cfg.terminations.pop(k, None)
    return cfg


for task, scale, phase in (("A2H-Hybrid-Flat-Unitree-G1", 0.2, False), ("A2H-Latent-Flat-Unitree-G1", 0.0, False),
                          ("A2H-LatentPhase-Flat-Unitree-G1", 0.0, True)):
    register_mjlab_task(task_id=task, env_cfg=use_variants(latent_env_cfg(scale, phase_only=phase)),
                        play_env_cfg=use_variants(latent_env_cfg(scale, play=True, phase_only=phase)),
                        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(), runner_cls=MotionTrackingOnPolicyRunner)
# Task-first: Task keeps the retarget as the policy's input and a 0.1x tracking style term;
# TaskPhase sees only its own body, the clock and the task stage, and is never scored on the retarget.
# TaskCmd is TaskPhase asked for other squat depths and jump heights every episode.
# TaskHybrid is Task with tracking at half mjlab's weights on the whole clip (not only near key moments):
# the task's structure plus imitation, between Task (0.1x, goal-weighted) and pure tracking.
for task, scale, phase, cmd, weighted in (("A2H-Task-Flat-Unitree-G1", 0.1, False, False, True),
                                          ("A2H-TaskPhase-Flat-Unitree-G1", 0.0, True, False, True),
                                          ("A2H-TaskCmd-Flat-Unitree-G1", 0.0, True, True, True),
                                          ("A2H-TaskHybrid-Flat-Unitree-G1", 0.5, False, False, False)):
    register_mjlab_task(task_id=task, env_cfg=use_variants(task_env_cfg(scale, phase_only=phase, commanded=cmd, weighted_tracking=weighted)),
                        play_env_cfg=use_variants(task_env_cfg(scale, play=True, phase_only=phase, commanded=cmd, weighted_tracking=weighted)),
                        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(), runner_cls=MotionTrackingOnPolicyRunner)
register_mjlab_task(task_id="A2H-Tracking-Flat-Unitree-G1", env_cfg=use_variants(unitree_g1_flat_tracking_env_cfg()),
                    play_env_cfg=use_variants(unitree_g1_flat_tracking_env_cfg(play=True)),
                    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(), runner_cls=MotionTrackingOnPolicyRunner)
for task, name, term in (("A2H-TrackingVJ-Flat-Unitree-G1", "video_latent", video_latent),
                         ("A2H-TrackingVJCausal-Flat-Unitree-G1", "video_latent", video_latent_causal),
                         ("A2H-TrackingFuture-Flat-Unitree-G1", "future_reference", future_reference)):
    register_mjlab_task(task_id=task, env_cfg=use_variants(conditioned_tracking_cfg(name, term)),
                        play_env_cfg=use_variants(conditioned_tracking_cfg(name, term, play=True)),
                        rl_cfg=unitree_g1_tracking_ppo_runner_cfg(), runner_cls=MotionTrackingOnPolicyRunner)



# World model (experiments/worldmodel_policy.py, docs/FRONTIER_WORLD_MODEL.md)
PREDICTED = 2  # predicted latents per step (latent/world_model.py HORIZONS: ~0.5 s and 1 s ahead)


def video_latent_predicted(env, command_name="motion"):
    """A2H-TrackingVJPred: A2H-TrackingVJCausal's input (the causal latent now and 0.5 s ago) plus what
    V-JEPA 2's predictor imagines, from the video so far, the latent will be ~0.5 s and 1 s on.
    A2H_VIDEO_LATENT holds (T, K * (1 + PREDICTED)) per motion step: the causal latent, then the
    predictions made at that step; every column is read at or before the current step."""
    cmd = env.command_manager.get_term(command_name)
    K = video_condition.K
    lat = video_latent_table(env, cmd.motion.time_step_total, causal=True)
    if lat is None:
        return torch.zeros(env.num_envs, K * (len(video_condition.CAUSAL_OFFSETS_S) + PREDICTED), device=env.device)
    past = [lat[_ahead(cmd, round(o / env.step_dt)), :K] for o in video_condition.CAUSAL_OFFSETS_S]
    return torch.cat([*past, lat[_ahead(cmd, 0), K:K * (1 + PREDICTED)]], -1)


register_mjlab_task(task_id="A2H-TrackingVJPred-Flat-Unitree-G1",
                    env_cfg=use_variants(conditioned_tracking_cfg("video_latent", video_latent_predicted)),
                    play_env_cfg=use_variants(conditioned_tracking_cfg("video_latent", video_latent_predicted, play=True)),
                    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(), runner_cls=MotionTrackingOnPolicyRunner)

if __name__ == "__main__":
    import policy_weights
    from mjlab.scripts.train import main
    policy_weights.patch_mjlab_train()  # A2H_INIT_CHECKPOINT: resume or build off saved weights
    main()
