"""mjlab's Unitree G1 motion-tracking environment on a bundle: the reference, one variant, or every kept training
variant end to end, built the way train_motion.py / experiments/variant_policy.py build it for the stored runs.

Needs mjlab (torch, mujoco, mujoco-warp). The 30 fps qpos becomes mjlab's 50 Hz motion file by forward kinematics on
mjlab's own G1 (the same steps as mjlab.scripts.csv_to_npz, in process and without Weights & Biases), cached next to
the bundle in <bundle>.motions/. Training on every variant lays them end to end in one motion file; each env then
stays inside its own variant (VariantMotionCommand, as mjlab_tasks.py A2H_VARIANTS) and mjlab's adaptive sampling
over the whole file favours the variants the policy fails. The reward is mjlab's tracking reward, plus the
feet_height term when the bundle's reward_config.json sets one (mjlab_tasks.py FEET_HEIGHT).
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

# torch and mjlab are imported where they are used: this module imports with numpy alone (bundle inspection, the
# interpolation test), and needs mjlab (which brings torch) only to build motion files and environments.

OUTPUT_FPS = 50.0  # mjlab's tracking control rate (200 Hz physics, decimation 4)
TASK = "A2H-Tracking-Flat-Unitree-G1"
# The G1's 29 joints in the order of the bundle's qpos[:, 7:] (MuJoCo Menagerie's g1.xml and mjlab's agree).
G1_JOINTS = ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
             "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
             "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
             "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
             "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
             "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
             "right_wrist_pitch_joint", "right_wrist_yaw_joint")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
FEET_HEIGHT = {"weight": 0.0, "std": 0.1, "clearance": 0.03, "swing_only": True}  # mjlab_tasks.py's defaults


def default_device():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def task_of(variant, feet_height=None):
    """The mjlab task an environment is: mjlab's own G1 tracking on one motion with no extra term, else this repo's
    A2H-Tracking (the same task plus VariantMotionCommand and/or FeetHeight, as mjlab_tasks.py registers it)."""
    extra = variant == "train" or bool(isinstance(feet_height, dict) and feet_height.get("weight"))
    return TASK if extra else "Mjlab-Tracking-Flat-Unitree-G1"


def safe_name(name):
    """A variant's name as a file name: letters, digits, dot, dash and underscore only."""
    return re.sub(r"[^\w.-]", "_", str(name))[:80] or "variant"


# --- qpos -> mjlab motion file ----------------------------------------------------------------------------------------
def _qmul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def _slerp(a, b, t):
    """Quaternions (N, 4) wxyz, blend t (N,) -> slerp, as mjlab.utils.lab_api.math.quat_slerp: b is flipped onto a's
    hemisphere (the shortest way round), t == 0 returns a and t == 1 returns b unchanged. One difference: mjlab returns
    a untouched when the pair is within 4 float32 ulps of identical, a step of up to ~1e-3 rad that reads as up to
    ~0.06 rad/s of angular velocity at near-still frames, and float32 rounding decides which frames that hits; here such
    pairs blend linearly (equal to slerp to O(theta^2)), so near-still stretches stay smooth."""
    d = (a * b).sum(1)
    b = np.where(d[:, None] < 0, -b, b)
    theta = np.arccos(np.clip(np.abs(d), -1, 1))
    small = theta < 1e-5
    sin = np.where(small, 1.0, np.sin(theta))
    wa = np.where(small, 1 - t, np.sin((1 - t) * theta) / sin)
    wb = np.where(small, t, np.sin(t * theta) / sin)
    out = wa[:, None] * a + wb[:, None] * b
    out = np.where((t == 0)[:, None], a, out)
    out = np.where((t == 1)[:, None], b, out)
    return out


def _axis_angle(q, eps=1e-6):
    """(N, 4) wxyz -> (N, 3) rotation vectors, as mjlab.utils.lab_api.math.axis_angle_from_quat: a quaternion with
    w < 0 is negated first (the same rotation; without it a sign jump between frames reads as a ~2 pi turn)."""
    q = q * (1.0 - 2.0 * (q[:, :1] < 0))
    mag = np.linalg.norm(q[:, 1:], axis=1)
    half = np.arctan2(mag, q[:, 0])
    angle = 2 * half
    big = np.abs(angle) > eps
    k = np.where(big, np.sin(half) / np.where(big, angle, 1.0), 0.5 - angle ** 2 / 48)
    return q[:, 1:] / k[:, None]


def interpolate(qpos, fps, output_fps=OUTPUT_FPS):
    """(T, 36) at fps -> root pos, root quat (wxyz), joint pos, and their velocities at output_fps, as
    mjlab.scripts.csv_to_npz's MotionLoader (linear, slerp for the root, central differences)."""
    qpos = np.asarray(qpos, np.float64)
    n, dt_in, dt_out = len(qpos), 1.0 / fps, 1.0 / output_fps
    duration = (n - 1) * dt_in
    times = np.arange(0, duration, dt_out)
    phase = times / duration if duration > 0 else np.zeros_like(times)
    i0 = np.floor(phase * (n - 1)).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = phase * (n - 1) - i0
    pos = qpos[i0, :3] * (1 - w)[:, None] + qpos[i1, :3] * w[:, None]
    quat = _slerp(qpos[i0, 3:7], qpos[i1, 3:7], w)  # unit when the input is; not renormalised, as mjlab
    joints = qpos[i0, 7:] * (1 - w)[:, None] + qpos[i1, 7:] * w[:, None]
    lin_vel = np.gradient(pos, dt_out, axis=0) if len(pos) > 1 else np.zeros_like(pos)
    joint_vel = np.gradient(joints, dt_out, axis=0) if len(joints) > 1 else np.zeros_like(joints)
    if len(quat) > 2:
        q_rel = _qmul(quat[2:], quat[:-2] * np.array([1, -1, -1, -1]))
        omega = _axis_angle(q_rel) / (2 * dt_out)
        ang_vel = np.concatenate([omega[:1], omega, omega[-1:]])
    else:
        ang_vel = np.zeros_like(pos)
    return pos, quat, joints, lin_vel, ang_vel, joint_vel


_SCENE = {}


def _scene(device, output_fps=OUTPUT_FPS):
    """mjlab's G1 tracking scene and a one-env simulation at output_fps, compiled once per process."""
    key = (device, output_fps)
    if key not in _SCENE:
        from mjlab.scene import Scene
        from mjlab.sim.sim import Simulation, SimulationCfg
        from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
        sim_cfg = SimulationCfg()
        sim_cfg.mujoco.timestep = 1.0 / output_fps
        scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
        model = scene.compile()
        sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
        scene.initialize(sim.mj_model, sim.model, sim.data)
        _SCENE[key] = (scene, sim)
    return _SCENE[key]


def reference_motion(qpos, fps, device="cpu", output_fps=OUTPUT_FPS):
    """qpos (T, 36) at fps -> mjlab's motion arrays at output_fps {fps, joint_pos, joint_vel, body_pos_w, body_quat_w,
    body_lin_vel_w, body_ang_vel_w}, by forward kinematics on mjlab's G1 (as mjlab.scripts.csv_to_npz)."""
    import torch
    scene, sim = _scene(device, output_fps)
    robot = scene["robot"]
    idx = robot.find_joints(list(G1_JOINTS), preserve_order=True)[0]
    pos, quat, joints, lin_vel, ang_vel, joint_vel = interpolate(qpos, fps, output_fps)
    f32 = dict(dtype=torch.float32, device=device)
    log = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    scene.reset()
    for i in range(len(pos)):
        root = robot.data.default_root_state.clone()
        root[:, 0:3] = torch.tensor(pos[i], **f32)
        root[:, :2] += scene.env_origins[:, :2]
        root[:, 3:7] = torch.tensor(quat[i], **f32)
        root[:, 7:10] = torch.tensor(lin_vel[i], **f32)
        root[:, 10:] = torch.tensor(ang_vel[i], **f32)
        robot.write_root_state_to_sim(root)
        jp, jv = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        jp[:, idx] = torch.tensor(joints[i], **f32)
        jv[:, idx] = torch.tensor(joint_vel[i], **f32)
        robot.write_joint_state_to_sim(jp, jv)
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())
    return {"fps": np.array([output_fps]), **{k: np.stack(v) for k, v in log.items()}}


def _digest(qpos):
    return hashlib.sha256(np.ascontiguousarray(qpos, np.float64).tobytes()).hexdigest()[:10]


def work_dir_for(bundle, work_dir=None):
    return Path(work_dir) if work_dir else Path(bundle.path).parent / (Path(bundle.path).name.removesuffix(".zip") + ".motions")


def motion_file(bundle, variant=None, work_dir=None, device="cpu"):
    """The 50 Hz motion file of the reference (variant None / "reference") or of one variant, written on first use.
    -> Path"""
    wd = work_dir_for(bundle, work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    if variant in (None, "reference"):
        q, name = bundle.qpos, "reference"
    else:
        v = bundle.variant(variant)
        if v.qpos is None:
            raise ValueError(f"variant {variant!r} was dropped ({v.why}) or has no arrays in this bundle")
        q, name = v.qpos, safe_name(v.name)
    out = wd / f"{name}-{_digest(q)}.npz"
    if not out.is_file():
        np.savez(out, **reference_motion(q, bundle.fps, device))
    return out


def variants_file(bundle, work_dir=None, device="cpu"):
    """Every kept training variant laid end to end: (motion .npz, segments [{name, start, length, spec}]) as
    experiments/variant_policy.py builds motion_variants.npz + variants.json."""
    kept = bundle.kept("train")
    if not kept:
        raise ValueError("this bundle has no training variants with arrays; train on the reference instead (variant=None)")
    wd = work_dir_for(bundle, work_dir)
    key = hashlib.sha256("".join(f"{v.name}:{_digest(v.qpos)}" for v in kept).encode()).hexdigest()[:10]
    out, meta = wd / f"train-{key}.npz", wd / f"train-{key}.json"
    if out.is_file() and meta.is_file():
        return out, json.loads(meta.read_text())["segments"]
    parts, segs, start = [], [], 0
    for v in kept:
        m = np.load(motion_file(bundle, v.name, wd, device))
        n = len(m["joint_pos"])
        segs.append({"name": v.name, "start": start, "length": n, "params": v.params, "spec": v.spec})
        parts.append(m)
        start += n
    np.savez(out, fps=parts[0]["fps"], **{k: np.concatenate([p[k] for p in parts]) for k in parts[0].files if k != "fps"})
    meta.write_text(json.dumps({"segments": segs}))
    return out, segs


# --- the env ----------------------------------------------------------------------------------------------------------
def _mjlab():
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
    return RewardTermCfg, unitree_g1_flat_tracking_env_cfg, MotionCommand, MotionCommandCfg


def _variant_command_cfg():
    import torch
    _, _, MotionCommand, MotionCommandCfg = _mjlab()

    class VariantMotionCommand(MotionCommand):
        """The motion file is several variants end to end: an env that reaches the end of its variant is resampled
        (as mjlab does at the end of the file) instead of running on into the next one."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            segs = cfg.segments
            start = torch.tensor([g["start"] for g in segs], dtype=torch.long, device=self.device)
            length = torch.tensor([g["length"] for g in segs], dtype=torch.long, device=self.device)
            assert int(start[-1] + length[-1]) == self.motion.time_step_total, "the segments don't match the motion file"
            self.seg_start = start
            self.seg_of = torch.repeat_interleave(torch.arange(len(segs), device=self.device), length)
            self.seg_last = torch.zeros(self.motion.time_step_total, dtype=torch.bool, device=self.device)
            self.seg_last[start + length - 1] = True
            self.seg_end = (start + length - 1)[self.seg_of]

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
        segments: list = field(default_factory=list)  # [{name, start, length (50 Hz steps)}]

        def build(self, env):
            return VariantMotionCommand(self, env)

    return VariantMotionCommandCfg


def feet_height_exp(robot_z, ref_z, floor, std, clearance, swing_only=True):
    """(B, feet) ankle heights of the robot and the reference, (feet,) the reference's floor per foot -> (B,) the mean
    over feet of exp(-err^2 / std^2), each foot counted only while the reference's is more than `clearance` above its
    floor when swing_only."""
    import torch
    r = torch.exp(-(robot_z - ref_z).square() / std ** 2)
    if swing_only:
        r = r * (ref_z - floor > clearance)
    return r.mean(-1)


class FeetHeight:
    """mjlab_tasks.py's foot-height term: pays for each foot's height matching the reference's while the reference's
    foot is in swing (its floor: the 5th-percentile height over the whole motion file)."""

    def __init__(self, cfg, env):
        import torch
        self.cmd = env.command_manager.get_term("motion")
        self.ids = [list(self.cmd.cfg.body_names).index(n) for n in FEET]
        self.floor = torch.quantile(self.cmd.motion.body_pos_w[:, self.ids, 2].float(), 0.05, dim=0)

    def __call__(self, env, std, clearance, swing_only=True):
        origin = env.scene.env_origins[:, None, 2]
        robot = self.cmd.robot_body_pos_w[:, self.ids, 2] - origin
        ref = self.cmd.body_pos_w[:, self.ids, 2] - origin
        return feet_height_exp(robot, ref, self.floor, std, clearance, swing_only)


def feet_height_config(bundle):
    fh = dict(FEET_HEIGHT)
    user = bundle.reward_config.get("feet_height") if isinstance(bundle.reward_config, dict) else None
    if isinstance(user, dict):
        fh |= {k: (bool(v) if k == "swing_only" else float(v)) for k, v in user.items() if k in fh}
    return fh


def resolve_variant(bundle, variant):
    """"train" (the default everywhere) when the bundle carries training variants with arrays, else the reference;
    "reference" / None the reference; anything else a variant's name."""
    if variant == "train":
        return "train" if bundle.kept("train") else "reference"
    return "reference" if variant is None else variant


def env_cfg(bundle, variant="train", num_envs=4096, play=False, work_dir=None, device="cpu"):
    """mjlab's G1 flat tracking config on the bundle: "train" (the default) tracks every kept training variant end to
    end, each env inside its own, falling back to the reference when the bundle has none with arrays; "reference"
    (or None) tracks the retarget itself; a variant's name tracks that variant. play: mjlab's play overrides (start
    at frame 0, no pushes or noise)."""
    RewardTermCfg, g1_cfg, _, _ = _mjlab()
    cfg = g1_cfg(play=play)
    cfg.scene.num_envs = num_envs
    variant = resolve_variant(bundle, variant)
    if variant == "train":
        npz, segs = variants_file(bundle, work_dir, device)
        old = cfg.commands["motion"]
        cfg.commands["motion"] = _variant_command_cfg()(**{f.name: getattr(old, f.name) for f in fields(old)},
                                                        segments=[{k: g[k] for k in ("name", "start", "length")} for g in segs])
    else:
        npz = motion_file(bundle, variant, work_dir, device)
    cfg.commands["motion"].motion_file = str(npz)
    fh = feet_height_config(bundle)
    if fh["weight"]:
        cfg.rewards["feet_height"] = RewardTermCfg(func=FeetHeight, weight=fh["weight"],
                                                   params={k: fh[k] for k in ("std", "clearance", "swing_only")})
    return cfg


def make_env(bundle, variant="train", num_envs=4096, device=None, play=False, work_dir=None):
    """-> mjlab ManagerBasedRlEnv (see env_cfg): the training variants by default. device: cuda:0 when available."""
    from mjlab.envs import ManagerBasedRlEnv
    device = device or default_device()
    return ManagerBasedRlEnv(cfg=env_cfg(bundle, variant, num_envs, play, work_dir, device), device=device)
