"""Map a human trajectory onto G1 joint angles with mink inverse kinematics (proportion-aware by
default: the human's pose on the G1's bone lengths, feet planted; docs/RETARGET.md), and export it
as an mjlab motion CSV.

Run this file for a synthetic squat smoke test.
"""

import warnings

import mink
import mujoco
import numpy as np

import config
import render

# G1 frames we drive, and the MediaPipe Pose landmark indices that feed them.
# MediaPipe: 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles.
FRAMES = {
    # name           (frame_name,              frame_type, mediapipe indices averaged)
    "pelvis":       ("pelvis",                 "body", (23, 24)),
    "left_knee":    ("left_knee_link",         "body", (25,)),
    "right_knee":   ("right_knee_link",        "body", (26,)),
    "left_foot":    ("left_ankle_roll_link",   "body", (27,)),
    "right_foot":   ("right_ankle_roll_link",  "body", (28,)),
    "left_elbow":   ("left_elbow_link",        "body", (13,)),
    "right_elbow":  ("right_elbow_link",       "body", (14,)),
    "left_hand":    ("left_wrist_yaw_link",    "body", (15,)),
    "right_hand":   ("right_wrist_yaw_link",   "body", (16,)),
}

# How hard each frame pulls. Feet matter most (no sliding), then pelvis, then limbs.
POSITION_COST = {"pelvis": 10.0, "left_foot": 20.0, "right_foot": 20.0,
                 "left_knee": 3.0, "right_knee": 3.0, "left_elbow": 1.0, "right_elbow": 1.0,
                 "left_hand": 2.0, "right_hand": 2.0}
# Orientation targets: pelvis yaw follows the hip line (soft, so the hips can tilt), feet stay
# flat under that yaw, torso follows the human's hip-to-shoulder line. Tuned on the squat demo.
ORIENTATION_COST = {"pelvis": 0.1, "left_foot": 2.0, "right_foot": 2.0}
# The proportional retarget also uses targets named "hip_centre" (the midpoint of the G1's hip
# joint centres: it drives the pelvis frame, re-anchored to the pelvis's current tilt) and
# "<arm frame>_torso" (elbows and hands in the torso_link frame, so the arm's pose relative to the
# torso survives a torso the waist can't fully follow).
ARM_FRAMES = ("left_elbow", "right_elbow", "left_hand", "right_hand")
ARM_TORSO_GAIN = 5.0  # torso-relative arm targets are consistent with the G1's arm, so trust them more
# Driving the hip centre leaves the pelvis free to tilt, so its orientation target (roll from the
# hip line, pitch a share of the trunk's lean) needs more weight than the yaw-only one above.
HIP_CENTRE_ORIENTATION_COST = 1.5
PELVIS_LEAN = 0.5
# Feet stay flat unless the shin leans past what the ankle can pitch: then the heel (or toes)
# lift by the excess, plus this margin. (MediaPipe's heel and toe landmarks are too noisy to use:
# a still stance can read 35 degrees.)
ANKLE_MARGIN = np.radians(5.0)
POSTURE_COST = 0.1
TORSO_COST = 1.0


def bent_knee_pose(model, knee=0.5):
    """The 'stand' keyframe has straight knees: an IK singularity where the solver can't start
    a knee bend and tilts the whole pelvis instead. Start from a slight crouch, feet flat and
    lowered back onto the floor (bending the knees alone would leave them hovering)."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_kinematics(model, data)
    ankle_z = data.body("left_ankle_roll_link").xpos[2]
    for side in ("left", "right"):
        data.joint(f"{side}_hip_pitch_joint").qpos = -knee / 2
        data.joint(f"{side}_knee_joint").qpos = knee
        data.joint(f"{side}_ankle_pitch_joint").qpos = -knee / 2
    mujoco.mj_kinematics(model, data)
    data.qpos[2] -= data.body("left_ankle_roll_link").xpos[2] - ankle_z
    return data.qpos.copy()


def ball_joint_centre(model, parent, child, prefix, samples=300, seed=0):
    """Treat a pitch/roll/yaw hinge stack (e.g. "left_hip") as one ball joint: the point in the
    parent body's frame that stays equidistant from the child body over random joint angles
    (least-squares sphere fit; the G1's residual is under 3 mm)."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(samples):
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        for axis in ("pitch", "roll", "yaw"):
            j = model.joint(f"{prefix}_{axis}_joint").id
            lo, hi = model.jnt_range[j]
            data.qpos[model.jnt_qposadr[j]] = rng.uniform(max(lo, -1.0), min(hi, 1.0))
        mujoco.mj_kinematics(model, data)
        rot = data.body(parent).xmat.reshape(3, 3)
        pts.append(rot.T @ (data.body(child).xpos - data.body(parent).xpos))
    X = np.array(pts)
    # |x - c|^2 = r^2 is linear in (c, r^2 - |c|^2).
    sol = np.linalg.lstsq(np.c_[2 * X, np.ones(len(X))], (X ** 2).sum(1), rcond=None)[0]
    return sol[:3]


class G1Retargeter:
    def __init__(self, model=None):
        self.model = model if model is not None else render.load_g1()[0]
        self._measure_robot()
        self.config = mink.Configuration(self.model)
        self.config.update(bent_knee_pose(self.model))
        self.tasks = {
            name: mink.FrameTask(frame_name=frame, frame_type=ftype,
                                 position_cost=POSITION_COST[name],
                                 orientation_cost=ORIENTATION_COST.get(name, 0.0))
            for name, (frame, ftype, _) in FRAMES.items()
        }
        # Always-on: joints drift back toward the start pose; torso orientation (upright unless
        # a torso rotation is given to solve()).
        self.posture = mink.PostureTask(self.model, cost=self._posture_costs())
        self.posture.set_target_from_configuration(self.config)
        self.torso = mink.FrameTask("torso_link", "body", position_cost=0.0,
                                    orientation_cost=TORSO_COST)
        self.torso.set_target_from_configuration(self.config)
        self.tasks |= {
            f"{name}_torso": mink.RelativeFrameTask(FRAMES[name][0], FRAMES[name][1], "torso_link", "body",
                                                    position_cost=POSITION_COST[name] * ARM_TORSO_GAIN, orientation_cost=0.0)
            for name in ARM_FRAMES}
        for task in self.tasks.values():
            task.set_target_from_configuration(self.config)
        self.limits = [mink.ConfigurationLimit(self.model)]

    def _measure_robot(self):
        """Robot proportions at the straight-leg stand pose, used to scale the human."""
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key("stand").id)
        mujoco.mj_forward(self.model, data)
        pos = lambda b: data.body(b).xpos.copy()
        self.leg_length = np.mean([
            np.linalg.norm(pos(f"{s}_hip_pitch_link") - pos(f"{s}_knee_link"))
            + np.linalg.norm(pos(f"{s}_knee_link") - pos(f"{s}_ankle_roll_link"))
            for s in ("left", "right")])
        self.ankle_height = pos("left_ankle_roll_link")[2]
        hip_mid = (pos("left_hip_pitch_link") + pos("right_hip_pitch_link")) / 2
        self.pelvis_above_hips = pos("pelvis") - hip_mid

        # The G1 skeleton for the proportional retarget. Its hips and shoulders are three
        # offset hinges, not ball joints: their effective centres (fixed points in the pelvis /
        # torso frame) come from a sphere fit. Knee and elbow are hinges; their bones end at the
        # frames IK drives. Lengths are symmetrised over left and right.
        self.hip_centre = ball_joint_centre(self.model, "pelvis", "left_knee_link", "left_hip")
        self.shoulder_centre = ball_joint_centre(self.model, "torso_link", "left_elbow_link", "left_shoulder")
        mirror = np.array([1.0, -1.0, 1.0])
        hips = [pos("pelvis") + data.body("pelvis").xmat.reshape(3, 3) @ (self.hip_centre * m)
                for m in (1, mirror)]
        shoulders = [pos("torso_link") + data.body("torso_link").xmat.reshape(3, 3) @ (self.shoulder_centre * m)
                     for m in (1, mirror)]
        side = lambda a, b: np.mean([np.linalg.norm(pos(f"{s}_{a}") - pos(f"{s}_{b}")) for s in ("left", "right")])
        self.bones = {
            "hip_width": np.linalg.norm(hips[0] - hips[1]),
            "spine": np.linalg.norm(np.mean(shoulders, 0) - np.mean(hips, 0)),
            "shoulder_width": np.linalg.norm(shoulders[0] - shoulders[1]),
            "thigh": np.mean([np.linalg.norm(pos(f"{s}_knee_link") - h) for s, h in zip(("left", "right"), hips)]),
            "shin": side("knee_link", "ankle_roll_link"),
            "upper_arm": np.mean([np.linalg.norm(pos(f"{s}_elbow_link") - h) for s, h in zip(("left", "right"), shoulders)]),
            "forearm": side("elbow_link", "wrist_yaw_link"),
        }
        self.pelvis_above_hip_centres = -self.hip_centre * [0.0, 0.0, 1.0]  # in the pelvis frame
        # Ankle frame height when the sole touches the floor (the stand keyframe sinks it ~2 mm).
        sole = [g for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] == self.model.body("left_ankle_roll_link").id
                and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        self.sole_height = self.ankle_height - min(data.geom_xpos[g][2] - self.model.geom_size[g][0] for g in sole)
        self.sole_points = self.model.geom_pos[sole].copy()  # contact spheres, in the foot frame
        self.sole_radius = float(self.model.geom_size[sole[0]][0])

    def ankle_clearance(self, foot_rot):
        """(..., 3, 3) foot rotations -> (...) ankle-frame height at which the lowest sole point
        just touches the floor (sole_height for a flat foot; higher when the heel or toes lift)."""
        return self.sole_radius - (foot_rot[..., 2, :] @ self.sole_points.T).min(-1)

    def _posture_costs(self):
        # Light on the legs (they must bend to follow the pelvis), firmer on waist and arms.
        cost = np.zeros(self.model.nv)
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = self.model.joint(j).name
            leg = any(k in name for k in ("hip", "knee", "ankle"))
            cost[self.model.jnt_dofadr[j]] = POSTURE_COST * (0.01 if leg else 1.0)
        return cost

    def frame_position(self, name):
        """World position of a FRAMES frame, the hip centre, or an "<arm>_torso" frame in the
        torso_link frame."""
        if name == "hip_centre":
            pelvis = self.config.get_transform_frame_to_world("pelvis", "body")
            return pelvis.translation() + pelvis.rotation().as_matrix() @ (-self.pelvis_above_hip_centres)
        if name.endswith("_torso"):
            frame, ftype, _ = FRAMES[name.removesuffix("_torso")]
            return self.config.get_transform(frame, ftype, "torso_link", "body").translation()
        frame, ftype, _ = FRAMES[name]
        return self.config.get_transform_frame_to_world(frame, ftype).translation()

    def skeleton(self, qpos):
        """(T, nq) -> {joint: (T, 3)}: the robot's joints as in rescale_bones, for comparing
        joint angles with the human (flexion_angles)."""
        data = mujoco.MjData(self.model)
        mirror = np.array([1.0, -1.0, 1.0])
        out = {}
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_kinematics(self.model, data)
            at = lambda body, offset: data.body(body).xpos + data.body(body).xmat.reshape(3, 3) @ offset
            J = {"left_hip": at("pelvis", self.hip_centre), "right_hip": at("pelvis", self.hip_centre * mirror),
                 "left_shoulder": at("torso_link", self.shoulder_centre),
                 "right_shoulder": at("torso_link", self.shoulder_centre * mirror)}
            for s in ("left", "right"):
                J |= {f"{s}_knee": data.body(f"{s}_knee_link").xpos, f"{s}_ankle": data.body(f"{s}_ankle_roll_link").xpos,
                      f"{s}_elbow": data.body(f"{s}_elbow_link").xpos, f"{s}_wrist": data.body(f"{s}_wrist_yaw_link").xpos}
            J["hip_mid"] = (J["left_hip"] + J["right_hip"]) / 2
            J["shoulder_mid"] = (J["left_shoulder"] + J["right_shoulder"]) / 2
            for k, v in J.items():
                out.setdefault(k, []).append(v.copy())
        return {k: np.array(v) for k, v in out.items()}

    def solve(self, targets: dict, rotations: dict | None = None, iters=50, dt=0.02):
        """targets: {name: xyz}; rotations: {name or "torso": 3x3 world rotation}. Only frames
        named in targets are driven this call. Warm-starts from the previous frame's solution,
        so call in time order."""
        rotations = rotations or {}
        hips = targets.get("hip_centre")
        targets = {("pelvis" if n == "hip_centre" else n): p for n, p in targets.items()}
        for name, pos in targets.items():
            task = self.tasks[name]
            if name.endswith("_torso"):
                task.set_target(mink.SE3.from_translation(np.asarray(pos, float)))
                continue
            rot = (mink.SO3.from_matrix(rotations[name]) if name in rotations
                   else task.transform_target_to_world.rotation())
            task.set_target(mink.SE3.from_rotation_and_translation(rot, np.asarray(pos, float)))
        if "torso" in rotations:
            self.torso.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rotations["torso"]), np.zeros(3)))
        tasks = [*(self.tasks[n] for n in targets), self.posture, self.torso]
        pelvis = self.tasks["pelvis"]
        pelvis.set_orientation_cost(HIP_CENTRE_ORIENTATION_COST if hips is not None else ORIENTATION_COST["pelvis"])
        for _ in range(iters):
            if hips is not None:  # the hip centres sit below the pelvis frame, wherever it tilts
                tilt = self.config.get_transform_frame_to_world("pelvis", "body").rotation()
                pelvis.set_target(mink.SE3.from_rotation_and_translation(
                    pelvis.transform_target_to_world.rotation(),
                    np.asarray(hips, float) + tilt.as_matrix() @ self.pelvis_above_hip_centres))
            vel = mink.solve_ik(self.config, tasks, dt, "daqp", damping=1e-3, limits=self.limits)
            self.config.integrate_inplace(vel, dt)
        return self.config.q.copy()

    def retarget_sequence(self, targets_seq, rotations_seq=None):
        """-> (frames, nq) qpos array. The first frame gets extra iterations to converge from
        the start pose."""
        rotations_seq = rotations_seq or [None] * len(targets_seq)
        out = []
        for i, (t, r) in enumerate(zip(targets_seq, rotations_seq)):
            out.append(self.solve(t, r, iters=300 if i == 0 else 50))
        return np.stack(out)


def _frame_from(left, up):
    """Rotation whose columns are (forward, left, up) for the robot convention x-fwd, y-left, z-up."""
    up = up / np.linalg.norm(up)
    left = left - up * (left @ up)
    left = left / np.linalg.norm(left)
    return np.column_stack([np.cross(left, up), left, up])


def human_to_targets(P, rt: G1Retargeter, proportional=True, fps=30.0):
    """P: (T, 33, 3) human trajectory from pose.human_trajectory (metres, z-up, facing +x).
    Returns (targets_seq, rotations_seq) for G1Retargeter.retarget_sequence, with the start
    under the robot and the feet on the floor.

    proportional (default): the human's pose on the G1's skeleton, feet planted during
    contacts (proportional_motion). proportional=False: the original retarget, the whole human
    scaled uniformly by leg length, kept for comparison."""
    if not proportional:
        return _uniform_targets(P, rt)
    m = proportional_motion(P, rt, fps)
    J = m["joints"]
    targets_seq, rotations_seq = [], []
    mirror = np.array([1.0, -1.0, 1.0])
    for t in range(len(P)):
        targets = {name: J[joint][t] for name, joint in TARGET_JOINTS.items() if name not in ARM_FRAMES}
        targets["hip_centre"] = J["hip_mid"][t]
        torso = m["torso_rot"][t]
        for name in ARM_FRAMES:  # arm joints relative to their shoulder, in the torso frame
            side = name.split("_")[0]
            shoulder = rt.shoulder_centre * (1 if side == "left" else mirror)
            targets[f"{name}_torso"] = shoulder + torso.T @ (J[TARGET_JOINTS[name]][t] - J[f"{side}_shoulder"][t])
        targets_seq.append(targets)
        rotations_seq.append({"pelvis": m["pelvis_rot"][t], "left_foot": m["foot_rot"][t, 0],
                              "right_foot": m["foot_rot"][t, 1], "torso": m["torso_rot"][t]})
    return targets_seq, rotations_seq


def _uniform_targets(P, rt):
    seg = lambda a, b: np.linalg.norm(P[:, a] - P[:, b], axis=-1)
    human_leg = np.median(np.concatenate([seg(23, 25) + seg(25, 27), seg(24, 26) + seg(26, 28)]))
    P = P * (rt.leg_length / human_leg)

    # Ground: the lowest ankles (5th percentile, robust to glitches) sit at the robot's ankle height.
    P[..., 2] += rt.ankle_height - np.percentile(np.minimum(P[:, 27, 2], P[:, 28, 2]), 5)
    # Start: hips over the origin.
    P[..., :2] -= P[0, [23, 24], :2].mean(0)

    targets_seq, rotations_seq = [], []
    for p in P:
        hip_l, hip_r = p[23], p[24]
        pelvis_rot = _frame_from(hip_l - hip_r, np.array([0.0, 0, 1]))  # yaw only
        torso_rot = _frame_from(p[11] - p[12], p[[11, 12]].mean(0) - p[[23, 24]].mean(0))
        targets = {name: p[list(idx)].mean(0) for name, (_, _, idx) in FRAMES.items()}
        targets["pelvis"] = targets["pelvis"] + pelvis_rot @ rt.pelvis_above_hips
        targets_seq.append(targets)
        rotations_seq.append({"pelvis": pelvis_rot, "left_foot": pelvis_rot,
                              "right_foot": pelvis_rot, "torso": torso_rot})
    return targets_seq, rotations_seq


# The skeleton both bodies share. Limb bones: (G1 bone, parent joint, child joint, MediaPipe
# parent, MediaPipe child).
LIMBS = [
    ("thigh", "left_hip", "left_knee", 23, 25), ("shin", "left_knee", "left_ankle", 25, 27),
    ("thigh", "right_hip", "right_knee", 24, 26), ("shin", "right_knee", "right_ankle", 26, 28),
    ("upper_arm", "left_shoulder", "left_elbow", 11, 13), ("forearm", "left_elbow", "left_wrist", 13, 15),
    ("upper_arm", "right_shoulder", "right_elbow", 12, 14), ("forearm", "right_elbow", "right_wrist", 14, 16),
]
# IK frame -> skeleton joint it follows (the pelvis frame sits a fixed offset above hip_mid).
TARGET_JOINTS = {"left_knee": "left_knee", "right_knee": "right_knee",
                 "left_foot": "left_ankle", "right_foot": "right_ankle",
                 "left_elbow": "left_elbow", "right_elbow": "right_elbow",
                 "left_hand": "left_wrist", "right_hand": "right_wrist"}


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rescale_bones(P, bones):
    """P: (T, 33, 3) human. -> {joint: (T, 3)} relative to the hip midpoint: the same pose, with
    every bone keeping the human's direction (so every joint angle is the human's) but taking
    the G1's length (`bones`, as in G1Retargeter.bones). The shoulder line is squared onto the
    spine, as on the G1's rigid torso; the hip line keeps its direction."""
    hip_mid, shoulder_mid = P[:, [23, 24]].mean(1), P[:, [11, 12]].mean(1)
    spine = _unit(shoulder_mid - hip_mid)
    across = P[:, 11] - P[:, 12]
    across = _unit(across - spine * (across * spine).sum(-1, keepdims=True))
    J = {"hip_mid": np.zeros_like(hip_mid), "shoulder_mid": spine * bones["spine"]}
    for side, sign in (("left", 1), ("right", -1)):
        J[f"{side}_hip"] = sign * _unit(P[:, 23] - P[:, 24]) * bones["hip_width"] / 2
        J[f"{side}_shoulder"] = J["shoulder_mid"] + sign * across * bones["shoulder_width"] / 2
    for bone, parent, child, i, j in LIMBS:
        J[child] = J[parent] + _unit(P[:, j] - P[:, i]) * bones[bone]
    return J


def flexion_angles(J):
    """Joint angles in degrees from skeleton positions {joint: (T, 3)} (rescale_bones or
    G1Retargeter.skeleton): knee and elbow flexion (0 = straight), hip flexion and shoulder
    elevation (0 = thigh / upper arm along the trunk, pointing down)."""
    angle = lambda u, v: np.degrees(np.arccos(np.clip((_unit(u) * _unit(v)).sum(-1), -1, 1)))
    down = J["hip_mid"] - J["shoulder_mid"]
    out = {}
    for s in ("left", "right"):
        thigh, shin = J[f"{s}_knee"] - J[f"{s}_hip"], J[f"{s}_ankle"] - J[f"{s}_knee"]
        upper, fore = J[f"{s}_elbow"] - J[f"{s}_shoulder"], J[f"{s}_wrist"] - J[f"{s}_elbow"]
        out |= {f"{s}_knee": angle(thigh, shin), f"{s}_hip": angle(down, thigh),
                f"{s}_elbow": angle(upper, fore), f"{s}_shoulder": angle(down, upper)}
    return out


def _runs(mask):
    """[(start, stop)] of the True runs in a 1-D bool array."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def stance_mask(ankles, fps=30.0, height=0.1, speed=0.6, min_frames=4, max_gap=6):
    """ankles: (T, 2, 3) left/right ankle paths (metres, z-up). -> (T, 2) bool, foot planted:
    within `height` of its floor level and slower than `speed` m/s. Monocular pose is too noisy
    for tight thresholds: a planted ankle wanders by ~10 cm in height and ~0.3 m/s. Each foot's
    floor is its own lowest level (5th percentile; MediaPipe often sits one foot a few cm higher),
    capped at `height` above the other's. Lifts shorter than max_gap frames between two contacts
    are filled, then contacts shorter than min_frames dropped."""
    floor = np.percentile(ankles[..., 2], 5, axis=0)
    floor = np.minimum(floor, floor.min() + height)
    v = np.linalg.norm(np.gradient(ankles, axis=0), axis=-1) * fps
    stance = (ankles[..., 2] - floor < height) & (v < speed)
    for f in range(2):
        m = stance[:, f]
        for a, b in _runs(~m):  # fill short lifts between two contacts
            if 0 < a and b < len(m) and b - a < max_gap:
                m[a:b] = True
        for a, b in _runs(m):
            if b - a < min_frames:
                m[a:b] = False
    return stance


LOWER_BODY = ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")


def level_feet(J, stance, max_tilt=np.radians(15.0)):
    """With both feet planted, the floor says the ankles are level; monocular depth error often
    puts one a few cm higher. Pinning both would stretch one leg and fold the other (near a
    straight knee, 1 cm is 10 degrees). Instead tilt the whole lower body rigidly about the hip
    midpoint until the ankles are level, which keeps every leg angle and moves the error into the
    pelvis tilt. The tilt is bridged linearly through the frames in between."""
    d = J["left_ankle"] - J["right_ankle"]
    horiz = np.linalg.norm(d[:, :2], axis=-1)
    tilt = np.where(stance.all(1) & (horiz > 0.1), np.arctan2(d[:, 2], horiz), np.nan)
    tilt = np.clip(_bridge(tilt), -max_tilt, max_tilt)
    axis = _unit(np.cross([0.0, 0, 1], np.c_[d[:, :2], np.zeros(len(d))]) + 1e-9)
    # Rodrigues: rotating d about `axis` (horizontal, perpendicular to it) by `tilt` levels it.
    rot = lambda v: (v * np.cos(tilt)[:, None] + np.cross(axis, v) * np.sin(tilt)[:, None]
                     + axis * (axis * v).sum(-1, keepdims=True) * (1 - np.cos(tilt))[:, None])
    return J | {k: rot(J[k]) for k in LOWER_BODY}


def _bridge(x):
    """Fill NaN frames of x (T, ...) by linear interpolation in time (held at the ends); all-NaN
    columns become 0."""
    flat = x.reshape(len(x), -1).copy()
    t = np.arange(len(x))
    for c in range(flat.shape[1]):
        ok = ~np.isnan(flat[:, c])
        flat[:, c] = np.interp(t, t[ok], flat[ok, c]) if ok.any() else 0.0
    return flat.reshape(x.shape)


def plant_feet(root, ankles, stance, floor):
    """root: (T, 3) hip-midpoint path; ankles: (T, 2, 3) left/right relative to it; stance:
    (T, 2) bool; floor: ankle height on the floor, scalar or (T, 2) (a lifted heel raises it).
    -> (root, feet (T, 2, 3), foot_fix (T, 2, 3)).

    Each contact pins its foot at the contact's mean position, on the floor. While any foot is
    planted the root is placed so the planted legs reach their feet (the body is carried by the
    stance legs, so nothing slides); in flight the root follows the human, with the correction
    blended linearly between takeoff and landing. foot_fix is what had to be added to the
    skeleton's foot to keep it pinned (nonzero only in double support or when lifted off the
    floor, bridged across swing)."""
    floor = np.broadcast_to(floor, stance.shape)
    pinned = np.full(ankles.shape, np.nan)
    for f in range(2):
        for a, b in _runs(stance[:, f]):
            pinned[a:b, f] = (root[a:b] + ankles[a:b, f]).mean(0)
            pinned[a:b, f, 2] = floor[a:b, f]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames: no foot planted
        shift = np.nanmean(pinned - ankles, axis=1) - root
    root = root + _bridge(shift)
    foot_fix = _bridge(pinned - (root[:, None] + ankles))
    feet = root[:, None] + ankles + foot_fix
    feet[..., 2] = np.maximum(feet[..., 2], floor)
    return root, feet, foot_fix


def _pelvis_frame(left, trunk, lean=None):
    """Pelvis rotation: the hip line exactly (yaw and roll); pitch, which two hip landmarks can't
    show, takes `lean` of the trunk's forward lean (the G1's waist pitches only 30 degrees, so a
    deep squat's lean has to come partly from the pelvis, as in people)."""
    lean = PELVIS_LEAN if lean is None else lean
    left = left / np.linalg.norm(left)
    up = (1 - lean) * np.array([0.0, 0, 1]) + lean * trunk / np.linalg.norm(trunk)
    up = up - left * (left @ up)
    up = up / np.linalg.norm(up)
    return np.column_stack([np.cross(left, up), left, up])


def _yaw_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _pitch_rot(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def proportional_motion(P, rt: G1Retargeter, fps=30.0):
    """The human's motion on a G1-proportioned skeleton, in world coordinates on the G1's floor.
    -> {"joints": {joint: (T, 3)}, "stance": (T, 2), "pelvis_rot", "torso_rot": (T, 3, 3),
    "foot_rot": (T, 2, 3, 3), "scale": root scale}.

    Bones take the G1's lengths and keep the human's directions (rescale_bones), so joint
    angles carry over and the hands and feet land where the G1 can reach. Root translation is
    scaled by the leg-length ratio, so squat depth and jump height scale with the legs. Feet in
    contact (stance_mask) are pinned on the floor (plant_feet), their yaw held."""
    seg = lambda a, b: np.linalg.norm(P[:, a] - P[:, b], axis=-1)
    human_leg = np.median(np.concatenate([seg(23, 25) + seg(25, 27), seg(24, 26) + seg(26, 28)]))
    scale = (rt.bones["thigh"] + rt.bones["shin"]) / human_leg
    J = rescale_bones(P, rt.bones)
    root = P[:, [23, 24]].mean(1) * scale
    ankles = np.stack([J["left_ankle"], J["right_ankle"]], 1)
    # Floor first (also the fallback if nothing is ever planted): lowest ankles on the floor.
    root[:, 2] += rt.sole_height - np.percentile((root[:, None] + ankles)[..., 2].min(1), 5)
    stance = stance_mask(root[:, None] + ankles, fps)
    J = level_feet(J, stance)
    ankles = np.stack([J["left_ankle"], J["right_ankle"]], 1)

    pelvis_rot = np.stack([_pelvis_frame(l - r, s) for l, r, s in
                           zip(J["left_hip"], J["right_hip"], J["shoulder_mid"])])
    torso_rot = np.stack([_frame_from(l - r, s) for l, r, s in
                          zip(J["left_shoulder"], J["right_shoulder"], J["shoulder_mid"])])
    # Foot yaw follows the pelvis, except a planted foot keeps its contact's mean yaw. Foot pitch
    # comes from the ankle's range (a lunge's rear heel lifts); the ankle rises to match.
    yaw = np.unwrap(np.arctan2(pelvis_rot[:, 1, 0], pelvis_rot[:, 0, 0]))
    foot_yaw = np.full((len(P), 2), np.nan)
    for f in range(2):
        for a, b in _runs(stance[:, f]):
            foot_yaw[a:b, f] = yaw[a:b].mean() - yaw[a:b]
    foot_yaw = yaw[:, None] + _bridge(foot_yaw)
    shin = ankles - np.stack([J["left_knee"], J["right_knee"]], 1)
    fwd = np.stack([np.cos(foot_yaw), np.sin(foot_yaw)], -1)
    lean = np.arctan2(-(shin[..., :2] * fwd).sum(-1), -shin[..., 2])  # knee ahead of ankle > 0
    lo, hi = rt.model.jnt_range[rt.model.joint("left_ankle_pitch_joint").id]
    pitch = np.clip(0.0, lean + lo + ANKLE_MARGIN, lean + hi - ANKLE_MARGIN)  # toe down > 0
    foot_rot = np.stack([[_yaw_rot(y) @ _pitch_rot(p) for y, p in zip(ys, ps)]
                         for ys, ps in zip(foot_yaw, pitch)])

    root, feet, foot_fix = plant_feet(root, ankles, stance, rt.ankle_clearance(foot_rot))
    start = np.r_[root[0, :2], 0.0]  # hips start over the origin
    root, feet = root - start, feet - start
    joints = {k: root + v for k, v in J.items()}
    joints["left_ankle"], joints["right_ankle"] = feet[:, 0], feet[:, 1]
    # A pinned foot moved off the skeleton's (double support, or lifted off the floor): move the
    # knee half as far, the small-correction answer for a two-bone leg. (Re-solving the knee
    # exactly overreacts near a straight leg, where 1 cm of foot is 10 degrees of knee.)
    joints["left_knee"] = joints["left_knee"] + foot_fix[:, 0] / 2
    joints["right_knee"] = joints["right_knee"] + foot_fix[:, 1] / 2
    return {"joints": joints, "stance": stance, "pelvis_rot": pelvis_rot, "torso_rot": torso_rot,
            "foot_rot": foot_rot, "scale": scale}


def tracking_report(rt: G1Retargeter, qpos, targets_seq):
    """Mean and max position error (cm) per frame type: how well the G1 can follow the human.
    Hip-centre and torso-relative targets are reported under the frame they drive."""
    errs = {n: [] for n in FRAMES}
    for q, targets in zip(qpos, targets_seq):
        rt.config.update(q)
        for n, pos in targets.items():
            key = "pelvis" if n == "hip_centre" else n.removesuffix("_torso")
            errs[key].append(np.linalg.norm(rt.frame_position(n) - pos) * 100)
    return {n: (float(np.mean(e)), float(np.max(e))) for n, e in errs.items() if e}


def export_mjlab_csv(qpos, path):
    """mjlab motion CSV: base xyz, base quat (x, y, z, w), then the 29 joint angles in the G1
    joint order (identical in Menagerie and mjlab). No header."""
    wxyz = qpos[:, 3:7]
    rows = np.column_stack([qpos[:, :3], wxyz[:, 1:], wxyz[:, :1], qpos[:, 7:]])
    np.savetxt(path, rows, delimiter=",", fmt="%.6f")
    return path


def synthetic_squat(rt: G1Retargeter, depth=0.25, reps=2, fps=30, hold_s=1.0, rep_s=2.0):
    """A motion the G1 can certainly do, for testing training independently of video:
    stand still, `reps` smooth squats of `depth` metres with the feet planted, stand still."""
    feet = {k: rt.frame_position(k) for k in ("left_foot", "right_foot")}
    pelvis0 = rt.frame_position("pelvis")
    phase = np.linspace(0, 2 * np.pi * reps, int(rep_s * reps * fps), endpoint=False)
    still = np.zeros(int(hold_s * fps))
    dz = np.concatenate([still, depth * (1 - np.cos(phase)) / 2, still, [0.0]])
    return [{**feet, "pelvis": pelvis0 - [0, 0, d]} for d in dz]


if __name__ == "__main__":
    # Smoke test and training sanity check: a grounded synthetic squat, exported for mjlab.
    rt = G1Retargeter()
    seq = synthetic_squat(rt)
    qpos = rt.retarget_sequence(seq)
    report = tracking_report(rt, qpos, seq)
    print({k: f"{v[1]:.1f} cm max" for k, v in report.items()})
    print(f"{len(qpos)} frames at 30 fps, pelvis height {qpos[:, 2].min():.2f}-{qpos[:, 2].max():.2f} m")
    out = render.save_video(render.render_qpos_sequence(rt.model, qpos), config.OUTPUT_DIR / "g1_ik_squat.mp4")
    export_mjlab_csv(qpos, config.OUTPUT_DIR / "g1_ik_squat.csv")
    print(f"saved {out} and g1_ik_squat.csv")
