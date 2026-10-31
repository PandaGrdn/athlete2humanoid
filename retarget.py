"""Map a human trajectory onto G1 joint angles with mink inverse kinematics (proportion-aware by
default: the human's pose on the G1's bone lengths, feet planted; docs/RETARGET.md), and export it
as an mjlab motion CSV.

Run this file for a synthetic squat smoke test.
"""

import warnings

import mink
import mujoco
import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

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
# Ankle roll past its +-15 degree range less this margin rolls the foot target instead.
ANKLE_ROLL_MARGIN = np.radians(4.0)
# A flat foot points along its leg's bending plane (leg_heading), at most this far from the pelvis
# yaw; a planted foot whose leg turns more than PIVOT_YAW during the contact pivots with it.
LEG_YAW_MAX = np.radians(60.0)
PIVOT_YAW = np.radians(90.0)
POSTURE_COST = 0.1
TORSO_COST = 1.0
# The arms are solved continuously: each frame's shoulder, elbow and wrist joints are pulled
# toward the previous frame's (per rad). An arm straight overhead or out to the side is a gimbal
# point of the shoulder's pitch-roll-yaw chain, where the hand targets barely constrain how the
# turn is split, and the solve otherwise swung the shoulder to the speed cap below in 1-11 frames
# per clip while the hand lagged its target by up to 18 cm (docs/PIPELINE_ACCURACY.md).
ARM_CONTINUITY_COST = 0.2
ARM_BIDIRECTIONAL = True
ARM_RESEED_ERR = 0.05  # m, elbow + hand
# Fastest a joint may turn between video frames (rad per second of video). Without it, one frame's
# solve can sweep a joint across its whole range: an arm straight overhead pins the shoulder at one
# limit and then flips it to the other (shot put: 5.8 rad in one frame), and the reference hand
# teleports. Our clips peak at ~8 rad/s.
MAX_JOINT_SPEED = 20.0
# A take-off is a flight only if the pelvis rises at least this share of what gravity needs for
# its airtime (g T^2 / 8); otherwise it's a stance the pose estimate lifted off the floor (a
# pivoting or sliding foot). Real flights in our pose data measure 0.5-0.8 (the rise is under-read
# and hovering feet count as airborne early); a shot-put glide measures 0.02 (docs/PIPELINE_ACCURACY.md).
FLIGHT_RISE_FRAC = 0.4
GRAVITY = 9.81
# No contact for longer than this, and no flight: not a lifted stance but missed contacts (a run)
# or a lost foot; the floor is interpolated through the foot's touch-downs instead of grounding it.
LONG_AIRBORNE_S = 0.5
# stance_mask's speed threshold grows by this share of the root's horizontal speed.
CONTACT_ROOT_SPEED_FRAC = 0.5
# Run in place (pipeline.py --in-place): a travelling move (a sprint, a hurdle approach) as a gait on the
# spot, the vertical root motion and the contacts kept. "world": the root held where it starts, each
# contact pinned under it (plant_feet(hold_root=True)), so the stance leg stops sweeping back; "stride":
# the planted motion with its per-stride travel subtracted, which keeps the surge within a stride but
# slides each planted foot back at the running speed; "soft": as "world", but the root keeps its surge and sway within
# a stride (the planted travelling root minus its per-stride travel), squashed into +-IN_PLACE_SOFT_M of the start, so
# the stance leg may sweep a little. docs/REAL_ATHLETES.md measures them on runs.
IN_PLACE_METHODS = ("world", "stride", "soft")
IN_PLACE_DEFAULT = "world"
IN_PLACE_SOFT_M = 0.10


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
        arm = np.zeros(self.model.nv)
        for j in range(self.model.njnt):
            if any(k in self.model.joint(j).name for k in ("shoulder", "elbow", "wrist")):
                arm[self.model.jnt_dofadr[j]] = ARM_CONTINUITY_COST
        self.continuity = mink.PostureTask(self.model, cost=arm)

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

    def solve(self, targets: dict, rotations: dict | None = None, iters=50, dt=0.02, max_step=None, continuous=False):
        """targets: {name: xyz}; rotations: {name or "torso": 3x3 world rotation}. Only frames
        named in targets are driven this call. Warm-starts from the previous frame's solution,
        so call in time order. max_step: the most any joint may move this call (rad). continuous:
        True holds the arm joints near the previous solution (ARM_CONTINUITY_COST); a qpos pulls
        them toward that instead."""
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
        if continuous is not False and ARM_CONTINUITY_COST > 0:
            self.continuity.set_target(self.config.q if continuous is True else continuous)
            tasks.append(self.continuity)
        limits = self.limits if max_step is None else [*self.limits, self._step_limit(max_step / (iters * dt))]
        pelvis = self.tasks["pelvis"]
        pelvis.set_orientation_cost(HIP_CENTRE_ORIENTATION_COST if hips is not None else ORIENTATION_COST["pelvis"])
        for _ in range(iters):
            if hips is not None:  # the hip centres sit below the pelvis frame, wherever it tilts
                tilt = self.config.get_transform_frame_to_world("pelvis", "body").rotation()
                pelvis.set_target(mink.SE3.from_rotation_and_translation(
                    pelvis.transform_target_to_world.rotation(),
                    np.asarray(hips, float) + tilt.as_matrix() @ self.pelvis_above_hip_centres))
            vel = mink.solve_ik(self.config, tasks, dt, "daqp", damping=1e-3, limits=limits)
            self.config.integrate_inplace(vel, dt)
        return self.config.q.copy()

    def _step_limit(self, speed):
        """Every hinge joint at most `speed` (rad per solver second)."""
        names = [self.model.joint(j).name for j in range(self.model.njnt)
                 if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        return mink.VelocityLimit(self.model, {n: speed for n in names})

    def retarget_sequence(self, targets_seq, rotations_seq=None, fps=30.0):
        """-> (frames, nq) qpos array. The first frame gets extra iterations to converge from
        the start pose; after it, no joint moves faster than MAX_JOINT_SPEED of video time, and the
        arms are solved continuously (ARM_CONTINUITY_COST).

        An arm passing straight out to the side or overhead can leave the warm-started solve on the
        wrong pitch-roll-yaw branch of its shoulder, stuck at a joint limit with the hand far from
        its target (jumping jacks: 13 cm for 9 frames), or switching branches at the speed cap. So
        the clip is also solved backwards in time, and each run of frames where an arm still misses
        its targets by over ARM_RESEED_ERR is solved again from the arm pose of the best-tracked
        frame with the most similar target, outwards until it rejoins (_reseed). Wherever these
        solves put an arm on different branches, it takes the one that follows its elbow and hand
        targets better, spliced in where they agree (_splice_arms)."""
        rotations_seq = rotations_seq or [None] * len(targets_seq)
        n = len(targets_seq)
        cap = MAX_JOINT_SPEED / fps

        def run(order, q0):
            self.config.update(q0)
            out = {}
            for k, i in enumerate(order):
                out[i] = self.solve(targets_seq[i], rotations_seq[i], iters=300 if k == 0 else 50,
                                    max_step=None if k == 0 else cap, continuous=k > 0)
            return np.stack([out[i] for i in range(len(order))])

        qpos = run(range(n), self.config.q.copy())
        if ARM_BIDIRECTIONAL and n > 2:
            qpos = self._splice_arms(qpos, [run(range(n - 1, -1, -1), qpos[-1])], targets_seq)
            for side in ("left", "right"):
                err = self._arm_error(qpos, targets_seq, side)
                bad = [(a, b) for a, b in _runs(err > ARM_RESEED_ERR) if b - a >= 2][:5]
                cands = [self._reseed(qpos, targets_seq, rotations_seq, cap, side, err, a, b) for a, b in bad]
                qpos = self._splice_arms(qpos, [c for c in cands if c is not None], targets_seq, sides=(side,))
        self.config.update(qpos[-1])
        return qpos

    def _arm_joints(self, side):
        return [self.model.jnt_qposadr[j] for j in range(self.model.njnt)
                if self.model.joint(j).name.startswith(f"{side}_")
                and any(k in self.model.joint(j).name for k in ("shoulder", "elbow", "wrist"))]

    def _arm_error(self, qpos, targets_seq, side):
        """(T,) one arm's elbow + hand error (m) against its torso-relative targets."""
        err = []
        for q, t in zip(qpos, targets_seq):
            self.config.update(q)
            err.append(sum(np.linalg.norm(self.frame_position(k) - t[k]) for k in (f"{side}_elbow_torso", f"{side}_hand_torso") if k in t))
        return np.array(err)

    def _reseed(self, qpos, targets_seq, rotations_seq, cap, side, err, a, b, agree=0.3):
        """The clip solved again around a badly tracked run [a, b) of one arm: its worst frame from
        the arm joints of the frame whose targets are most alike among those tracked within
        ARM_RESEED_ERR / 2, then frame by frame outwards (speed-capped) until the arm is back
        within `agree` rad of qpos for 3 frames. -> (T, nq), or None without a donor frame."""
        keys = [f"{side}_elbow_torso", f"{side}_hand_torso"]
        if not all(k in targets_seq[0] for k in keys):
            return None
        tgt = np.array([np.concatenate([t[k] for k in keys]) for t in targets_seq])
        mid = a + int(np.argmax(err[a:b]))
        ok = np.flatnonzero(err < ARM_RESEED_ERR / 2)
        if not ok.size:
            return None
        donor = ok[np.argmin(np.linalg.norm(tgt[ok] - tgt[mid], axis=-1))]
        adr = self._arm_joints(side)
        out = qpos.copy()
        seed = qpos[mid].copy()
        seed[adr] = qpos[donor, adr]
        self.config.update(seed)
        out[mid] = self.solve(targets_seq[mid], rotations_seq[mid], iters=300)
        for step in (1, -1):
            self.config.update(out[mid])
            t, same = mid + step, 0
            while 0 <= t < len(qpos) and same < 3:
                out[t] = self.solve(targets_seq[t], rotations_seq[t], max_step=cap, continuous=True)
                same = same + 1 if np.abs(out[t, adr] - qpos[t, adr]).max() < agree else 0
                t += step
        return out

    def _splice_arms(self, qpos, candidates, targets_seq, sides=("left", "right"), agree=0.3):
        """Per arm, for each candidate solve in turn: frames where it differs from the current one by
        more than `agree` rad in some arm joint form runs, and each run takes whichever has the
        smaller summed elbow + hand error. At a run's edges the two agree, so the splice is a small
        step."""
        out = qpos.copy()
        for side in sides:
            adr = self._arm_joints(side)
            for cand in candidates:
                differ = (np.abs(out[:, adr] - cand[:, adr]) > agree).any(1)
                if not differ.any():
                    continue
                cur, new = self._arm_error(out, targets_seq, side), self._arm_error(cand, targets_seq, side)
                for a, b in _runs(differ):
                    if new[a:b].sum() < cur[a:b].sum():
                        out[a:b, adr] = cand[a:b, adr]
        return out


def _frame_from(left, up):
    """Rotation whose columns are (forward, left, up) for the robot convention x-fwd, y-left, z-up."""
    up = up / np.linalg.norm(up)
    left = left - up * (left @ up)
    left = left / np.linalg.norm(left)
    return np.column_stack([np.cross(left, up), left, up])


def human_to_targets(P, rt: G1Retargeter, proportional=True, fps=30.0, report=None, in_place=None):
    """P: (T, 33, 3) human trajectory from pose.human_trajectory (metres, z-up, facing +x).
    Returns (targets_seq, rotations_seq) for G1Retargeter.retarget_sequence, with the start
    under the robot and the feet on the floor.

    proportional (default): the human's pose on the G1's skeleton, feet planted during
    contacts (proportional_motion). proportional=False: the original retarget, the whole human
    scaled uniformly by leg length, kept for comparison. report: an optional dict, filled with
    the contact decisions ("airborne", "warnings", "stance_frac") for meta.json, and "in_place"
    when in_place is set (one of IN_PLACE_METHODS: the motion on the spot, proportional only)."""
    if not proportional:
        if in_place:
            raise ValueError("in_place needs the proportional retarget")
        return _uniform_targets(P, rt)
    m = proportional_motion(P, rt, fps, in_place=in_place)
    if report is not None:
        report.update(airborne=m["airborne"], warnings=m["warnings"],
                      stance_frac=[round(float(x), 3) for x in m["stance"].mean(0)])
        if in_place:
            report["in_place"] = m["in_place"]
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


def _frames(n, fps):
    """A frame count given at 30 fps, at `fps`."""
    return max(1, int(round(n * fps / 30.0)))


def stance_mask(ankles, fps=30.0, height=0.1, speed=0.6, min_frames=4, max_gap=6, root=None, brief_frames=2):
    """ankles: (T, 2, 3) left/right ankle paths (metres, z-up); root: (T, 3) hip path, optional.
    -> (T, 2) bool, foot planted. Frame counts are at 30 fps and scale with `fps`.

    Two signals per foot. A dwell: within `height` of its floor level and slower than the speed
    threshold, for at least min_frames. A touch-down: the ankle's lowest point within +-0.1 s
    (vertical velocity crossing from down to up) within `height` of its floor and under the same
    threshold; it and its lower neighbour are a contact of brief_frames, the 2-3 frames a running
    foot is down. The threshold is `speed` plus CONTACT_ROOT_SPEED_FRAC of the root's speed: a
    planted ankle reads ~0.3 m/s of jitter, more when the body moves (the root is a 2D hip track
    scaled by a drifting ruler), while a swinging foot moves at about twice the body's speed.
    Monocular pose is too noisy for tight thresholds: a planted ankle wanders by ~10 cm in height.
    Each foot's floor is its own lowest level (5th percentile; MediaPipe often sits one foot a few
    cm higher), capped at `height` above the other's. Lifts shorter than max_gap frames between two
    contacts are filled, then contacts shorter than min_frames dropped unless they hold a touch-down."""
    min_frames, max_gap, brief_frames = (_frames(n, fps) for n in (min_frames, max_gap, brief_frames))
    floor = np.percentile(ankles[..., 2], 5, axis=0)
    floor = np.minimum(floor, floor.min() + height)
    v = np.linalg.norm(np.gradient(ankles, axis=0), axis=-1) * fps
    threshold = np.full(len(ankles), float(speed))
    if root is not None:
        root_speed = np.linalg.norm(np.gradient(root[:, :2], axis=0), axis=-1) * fps
        threshold = threshold + CONTACT_ROOT_SPEED_FRAC * uniform_filter1d(root_speed, _frames(6, fps), mode="nearest")
    near, slow = ankles[..., 2] - floor < height, v < threshold[:, None]
    z = ankles[..., 2]
    bottom = z <= minimum_filter1d(z, 2 * _frames(3, fps) + 1, axis=0, mode="nearest")
    touch = near & slow & bottom
    stance = near & slow
    for f in range(2):
        m, brief = stance[:, f], np.zeros(len(stance), bool)
        for t in np.flatnonzero(touch[:, f]):
            lo, hi = max(t - brief_frames + 1, 0), min(t + brief_frames, len(m))
            nb = np.arange(lo, hi)
            brief[nb[np.argsort(z[nb, f], kind="stable")[:brief_frames]]] = True  # the lowest frames around it
        m |= brief
        for a, b in _runs(~m):  # fill short lifts between two contacts
            if 0 < a and b < len(m) and b - a < max_gap:
                m[a:b] = True
        for a, b in _runs(m):
            if b - a < min_frames and not (touch[a:b, f].any() and b - a >= brief_frames):
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


def plant_feet(root, ankles, stance, floor, hold_root=False):
    """root: (T, 3) hip-midpoint path; ankles: (T, 2, 3) left/right relative to it; stance:
    (T, 2) bool; floor: ankle height on the floor, scalar or (T, 2) (a lifted heel raises it).
    -> (root, feet (T, 2, 3), foot_fix (T, 2, 3)).

    Each contact pins its foot at the contact's mean position, on the floor. While any foot is
    planted the root is placed so the planted legs reach their feet (the body is carried by the
    stance legs, so nothing slides); in flight the root follows the human, with the correction
    blended linearly between takeoff and landing. foot_fix is what had to be added to the
    skeleton's foot to keep it pinned (nonzero only in double support or when lifted off the
    floor, bridged across swing). hold_root: the root stays as given and the feet alone are
    pinned (run in place: a root held still over a stance leg that would sweep back), so foot_fix
    takes up the whole sweep."""
    floor = np.broadcast_to(floor, stance.shape)
    pinned = np.full(ankles.shape, np.nan)
    for f in range(2):
        for a, b in _runs(stance[:, f]):
            pinned[a:b, f] = (root[a:b] + ankles[a:b, f]).mean(0)
            pinned[a:b, f, 2] = floor[a:b, f]
    if not hold_root:
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


def _take_off(root_z, s0, s1, fps):
    """A candidate flight from take-off frame s0 to landing frame s1 (-1 / len: the clip starts /
    ends in the air). -> (is a flight, pelvis rise m, rise gravity needs m)."""
    T, n = (s1 - s0) / fps, len(root_z)
    need = FLIGHT_RISE_FRAC * GRAVITY * T ** 2 / 8
    inner = root_z[max(s0 + 1, 0):min(s1, n)]
    ends = [root_z[s] for s in (s0, s1) if 0 <= s < n]
    if not ends or not inner.size:
        return False, 0.0, need
    rise = float(inner.max() - np.mean(ends))
    if s1 >= n and root_z[-1] >= root_z[s0]:
        return True, rise, need  # still above its take-off when the clip ends: never lowered
    if s0 < 0 and root_z[0] >= root_z[s1]:
        return True, rise, need
    return rise >= need, rise, need


def ground_phantom_flights(root, feet, stance, floor, fps=30.0, clear=0.01, height=0.1, report=None):
    """Stretches with neither foot planted and both above the floor (by `clear` m), tested take-off
    by take-off. Inside a stretch the lowest foot may come back near the floor (a local minimum
    within `height`, the lowest within +-0.1 s): a touch-down the contact estimator missed, which
    ends one candidate flight and starts the next. A candidate is a flight if the pelvis rises at
    least FLIGHT_RISE_FRAC of the g T^2 / 8 its airtime T needs, or if the clip ends while the pelvis
    is still above its take-off height. Flights are kept as they are. Otherwise:
      - a stretch of at most LONG_AIRBORNE_S: a stance the pose estimate lifted off the floor (a
        pivoting or sliding foot); the body is lowered until its lowest foot is down;
      - a longer one (a run whose contacts were all missed, or a lost foot): lowering the body to
        the floor everywhere would flatten every stride and any jump in it, so instead the floor is
        interpolated through the lowest foot's touch-downs (the body lowered by the foot's height
        there, linearly in between), and a warning is added to `report`.
    report: an optional list; each stretch appends a dict (start_s, dur_s, touch_downs, flights:
    [{take_off_s, dur_s, rise_cm, need_cm, ends_clip}], decision: "flight" / "lowered" /
    "interpolated", lowered_cm, and a "warning" where one applies). -> (root, feet)."""
    floor = np.broadcast_to(floor, stance.shape)
    above = (feet[..., 2] - floor).min(1)  # the lowest foot's height off the floor
    up = ~stance.any(1) & (above > clear)
    n = len(root)
    touch = (above < height) & (above <= minimum_filter1d(above, 2 * _frames(3, fps) + 1, mode="nearest"))
    lower = np.zeros(n)
    for a, b in _runs(up):
        knots = [a - 1, *np.flatnonzero(touch[a:b]) + a, b]
        value = lambda s: above[s] if a <= s < b else 0.0  # a contact either side: nothing to lower
        long = (b - a) / fps > LONG_AIRBORNE_S
        decisions = []
        for s0, s1 in zip(knots[:-1], knots[1:]):
            if s0 >= a:
                lower[s0] = value(s0)
            if s1 - s0 < 2:
                continue
            flight, rise, need = _take_off(root[:, 2], s0, s1, fps)
            t = np.arange(s0 + 1, min(s1, n))
            if flight:  # hold the touch-downs (or contacts) either side, linear in between
                xs = [s for s in (s0, s1) if 0 <= s < n]
                lower[t] = np.interp(t, xs, [value(s) for s in xs]) if xs else 0.0
            elif long:  # the floor through the foot's height at the touch-downs (or the stretch's ends)
                xs = [max(s0, a), min(s1, b - 1)]
                lower[t] = np.interp(t, xs, above[xs])
            else:
                lower[t] = above[t]
            decisions.append((s0, s1, flight, rise, need))
        flights = [d for d in decisions if d[2]]
        row = dict(start_s=round(a / fps, 2), dur_s=round((b - a + 1) / fps, 2), touch_downs=len(knots) - 2,
                   flights=[dict(take_off_s=round(max(s0, 0) / fps, 2), dur_s=round((s1 - s0) / fps, 2),
                                 rise_cm=round(100 * r, 1), need_cm=round(100 * q, 1), ends_clip=bool(s1 >= n))
                            for s0, s1, _, r, q in flights],
                   decision="flight" if len(flights) == len(decisions) else "interpolated" if long else "lowered",
                   lowered_cm=round(100 * float(lower[a:b].max()), 1))
        warn = []
        if long and row["decision"] != "flight":
            warn.append(f"no foot contact for {row['dur_s']:.2f} s from {row['start_s']:.2f} s: floor interpolated through "
                        f"{row['touch_downs']} touch-downs of the lowest foot (body lowered up to {row['lowered_cm']:.0f} cm) "
                        "instead of grounding the stretch")
        for s0, s1, _, rise, _ in flights:
            ballistic = np.sqrt(8 * max(rise, 1e-3) / GRAVITY)  # the airtime of a flight with this rise
            if s1 >= n and (s1 - s0) / fps > 1.5 * ballistic:
                warn.append(f"ends in the air {(s1 - s0) / fps:.2f} s after its last take-off with the pelvis {100 * rise:.0f} cm "
                            f"up, longer than a ballistic flight of that rise ({ballistic:.2f} s): kept, but the root may be drifting")
        if warn:
            row["warning"] = "; ".join(warn)
        if report is not None:
            report.append(row)
    root, feet = root.copy(), feet.copy()
    root[:, 2] -= lower
    feet[..., 2] -= lower[:, None]
    return root, feet


def _roll_rot(roll):
    c, s = np.cos(roll), np.sin(roll)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def leg_heading(hip, knee, ankle, yaw, thigh, shin):
    """(T,) yaw of the plane the human's leg bends in (hip, knee, ankle: (T, 3)), as a flat G1 foot
    under that leg must point: the G1's knee and ankle pitch axes are parallel, so the foot's
    forward axis lies in the leg's plane. A straight leg has no plane: there it is the pelvis
    `yaw`, blended in by the knee's bend. Held within LEG_YAW_MAX of the pelvis."""
    n = np.cross(hip - ankle, knee - ankle) / (thigh * shin)  # the knee axis, pointing left; ~sin(knee bend)
    fwd = np.cross(n, [0.0, 0, 1])
    heading = np.arctan2(fwd[:, 1], fwd[:, 0])
    w = np.clip((np.linalg.norm(fwd, axis=-1) - 0.1) / 0.3, 0, 1)  # ~6 degrees of bend: none, ~24: all
    off = np.clip(np.angle(np.exp(1j * (heading - yaw))), -LEG_YAW_MAX, LEG_YAW_MAX)
    return yaw + w * off


def ankle_roll(hip, ankle, foot_rot):
    """(...,) the ankle roll a G1 leg needs to reach `hip` from a foot at (ankle, foot_rot): the hip-
    ankle line's lean out of the foot's sagittal plane (the knee and ankle pitch axes are parallel,
    and the ankle pitches before it rolls)."""
    u = np.einsum("...ji,...j->...i", foot_rot, hip - ankle)
    return np.arctan2(u[..., 1], u[..., 2])


def knee_in_leg_plane(hip, ankle, foot_rot, knee):
    """(T, 3) the knee target's part a G1 leg can follow: `knee` projected onto the plane through
    the hip-ankle line normal to the knee axis that this hip, ankle and foot rotation leave (the
    G1's knee and ankle pitch axes are parallel and the ankle rolls by ankle_roll, so with the hip
    and foot set the knee has no freedom out of that plane). The out-of-plane part is mostly depth
    noise; the in-plane flexion is kept for the IK to weigh against the hip and foot."""
    r = ankle_roll(hip, ankle, foot_rot)
    axis = np.einsum("tij,tj->ti", foot_rot, np.stack([np.zeros_like(r), np.cos(r), -np.sin(r)], -1))
    return knee - axis * ((knee - ankle) * axis).sum(-1, keepdims=True)


def _foot_rotations(foot_yaw, shin, lo, hi):
    """(T, 2, 3, 3) foot targets: yaw `foot_yaw` (T, 2), flat unless the shin (T, 2, 3, ankle minus
    knee) leans past what the ankle pitch range [lo, hi] can absorb, less ANKLE_MARGIN: then the heel
    (or toes) lift by the excess."""
    fwd = np.stack([np.cos(foot_yaw), np.sin(foot_yaw)], -1)
    lean = np.arctan2(-(shin[..., :2] * fwd).sum(-1), -shin[..., 2])  # knee ahead of ankle > 0
    pitch = np.clip(0.0, lean + lo + ANKLE_MARGIN, lean + hi - ANKLE_MARGIN)  # toe down > 0
    return np.stack([[_yaw_rot(y) @ _pitch_rot(p) for y, p in zip(ys, ps)] for ys, ps in zip(foot_yaw, pitch)])


def proportional_motion(P, rt: G1Retargeter, fps=30.0, in_place=None):
    """The human's motion on a G1-proportioned skeleton, in world coordinates on the G1's floor.
    -> {"joints": {joint: (T, 3)}, "stance": (T, 2), "pelvis_rot", "torso_rot": (T, 3, 3),
    "foot_rot": (T, 2, 3, 3), "scale": root scale, "root_pose" / "root_planted": (T, 3) the hip
    midpoint from the pose estimate and after planting the feet (for measuring the shift), "airborne":
    ground_phantom_flights' report, "warnings": [str] for meta.json}.

    Bones take the G1's lengths and keep the human's directions (rescale_bones), so joint
    angles carry over and the hands and feet land where the G1 can reach. Root translation is
    scaled by the leg-length ratio, so squat depth and jump height scale with the legs. Feet in
    contact (stance_mask) are pinned on the floor (plant_feet), their yaw held. Knees go where the
    G1's leg can put them with that foot (knee_in_leg_plane).

    in_place (one of IN_PLACE_METHODS, or None): the same motion on the spot, its horizontal root
    travel removed after the contacts are found; the result then has "in_place" (method, travel
    removed, stride, the root travel left)."""
    seg = lambda a, b: np.linalg.norm(P[:, a] - P[:, b], axis=-1)
    human_leg = np.median(np.concatenate([seg(23, 25) + seg(25, 27), seg(24, 26) + seg(26, 28)]))
    scale = (rt.bones["thigh"] + rt.bones["shin"]) / human_leg
    J = rescale_bones(P, rt.bones)
    root = P[:, [23, 24]].mean(1) * scale
    ankles = np.stack([J["left_ankle"], J["right_ankle"]], 1)
    # Floor first (also the fallback if nothing is ever planted): lowest ankles on the floor.
    root[:, 2] += rt.sole_height - np.percentile((root[:, None] + ankles)[..., 2].min(1), 5)
    restored = floor_drift = None
    if in_place:  # a camera that tracked the runner (a dolly) loses the travel, and with it the contacts
        root, floor_drift = level_floor(root, ankles, fps)
        root, restored = restore_travel(root, ankles, fps)
    stance = stance_mask(root[:, None] + ankles, fps, root=root)
    J = level_feet(J, stance)
    ankles = np.stack([J["left_ankle"], J["right_ankle"]], 1)

    pelvis_rot = np.stack([_pelvis_frame(l - r, s) for l, r, s in
                           zip(J["left_hip"], J["right_hip"], J["shoulder_mid"])])
    torso_rot = np.stack([_frame_from(l - r, s) for l, r, s in
                          zip(J["left_shoulder"], J["right_shoulder"], J["shoulder_mid"])])
    # Foot yaw follows the leg's plane (leg_heading), except a planted foot keeps its contact's mean
    # yaw, or turns with the leg if the contact is a pivot. Foot pitch comes from the ankle's range
    # (a lunge's rear heel lifts); the ankle rises to match.
    yaw = np.unwrap(np.arctan2(pelvis_rot[:, 1, 0], pelvis_rot[:, 0, 0]))
    leg_yaw = np.stack([leg_heading(J[f"{s}_hip"], J[f"{s}_knee"], J[f"{s}_ankle"], yaw, rt.bones["thigh"], rt.bones["shin"])
                        for s in ("left", "right")], 1)
    foot_yaw = np.full((len(P), 2), np.nan)
    for f in range(2):
        smooth = uniform_filter1d(leg_yaw[:, f], _frames(9, fps), mode="nearest")
        for a, b in _runs(stance[:, f]):
            pivot = abs(smooth[b - 1] - smooth[a]) > PIVOT_YAW  # the leg turned, not just jittered
            foot_yaw[a:b, f] = (smooth[a:b] if pivot else leg_yaw[a:b, f].mean()) - leg_yaw[a:b, f]
    foot_yaw = leg_yaw + _bridge(foot_yaw)
    lo, hi = rt.model.jnt_range[rt.model.joint("left_ankle_pitch_joint").id]
    foot_rot = _foot_rotations(foot_yaw, ankles - np.stack([J["left_knee"], J["right_knee"]], 1), lo, hi)

    floor = rt.ankle_clearance(foot_rot)
    root_pose = root
    if in_place not in (None, *IN_PLACE_METHODS):
        raise ValueError(f"in_place: one of {IN_PLACE_METHODS}, not {in_place!r}")
    stride_s = stride_period(stance, fps)
    if in_place == "world":  # contacts come from the travelling motion above, where planted feet stand still
        root = np.c_[np.broadcast_to(root[0, :2], (len(root), 2)), root[:, 2]]
    elif in_place == "soft":  # the travelling root's within-stride motion, squashed into +-IN_PLACE_SOFT_M
        moving = plant_feet(root, ankles, stance, floor)[0]
        q = moving[:, :2] - stride_drift(moving, stride_s, fps)[:, :2]
        q = q - q.mean(0)
        root = np.c_[root[0, :2] + IN_PLACE_SOFT_M * np.tanh(q / IN_PLACE_SOFT_M), root[:, 2]]
    root, feet, foot_fix = plant_feet(root, ankles, stance, floor, hold_root=in_place in ("world", "soft"))
    if in_place == "stride":
        drift = stride_drift(root, stride_s, fps)
        root, feet = root - drift, feet - drift[:, None]
    root_planted, airborne = root, []
    root, feet = ground_phantom_flights(root, feet, stance, floor, fps, report=airborne)
    start = np.r_[root[0, :2], 0.0]  # hips start over the origin
    root, feet = root - start, feet - start
    joints = {k: root + v for k, v in J.items()}
    # Ankle roll: what the hip-ankle line leans out of the foot's plane beyond the joint's range
    # rolls the foot target (onto its edge, raised to clear the floor), rather than leaving the IK a
    # flat foot it can't reach, which pinned the ankle roll at its limit and gave up the knee.
    # Knees: only the part of the human's knee the G1 can follow (knee_in_leg_plane). The foot
    # pitch is then redone for the knee's new shin, and the roll and knee once more for that pitch.
    roll_lim = rt.model.jnt_range[rt.model.joint("left_ankle_roll_joint").id][1] - ANKLE_ROLL_MARGIN
    # A pinned foot moved off the skeleton's (double support, or lifted off the floor): the knee
    # moves half as far, the small-correction answer for a two-bone leg. (Re-solving the knee
    # exactly overreacts near a straight leg, where 1 cm of foot is 10 degrees of knee.)
    human_knees = [joints[f"{s}_knee"] + foot_fix[:, f] / 2 for f, s in enumerate(("left", "right"))]
    on_floor = feet.copy()
    for last in (False, True):
        for f, s in enumerate(("left", "right")):
            r = ankle_roll(joints[f"{s}_hip"], feet[:, f], foot_rot[:, f])
            foot_rot[:, f] = foot_rot[:, f] @ np.stack([_roll_rot(e) for e in np.clip(r, -roll_lim, roll_lim) - r])
        feet[..., 2] = np.maximum(on_floor[..., 2], rt.ankle_clearance(foot_rot))
        knees = np.stack([knee_in_leg_plane(joints[f"{s}_hip"], feet[:, f], foot_rot[:, f], human_knees[f])
                          for f, s in enumerate(("left", "right"))], 1)
        if not last:
            foot_rot = _foot_rotations(foot_yaw, feet - knees, lo, hi)
    joints["left_ankle"], joints["right_ankle"] = feet[:, 0], feet[:, 1]
    joints["left_knee"], joints["right_knee"] = knees[:, 0], knees[:, 1]
    out = {"joints": joints, "stance": stance, "pelvis_rot": pelvis_rot, "torso_rot": torso_rot,
           "foot_rot": foot_rot, "scale": scale, "root_pose": root_pose - start,
           "root_planted": root_planted - start, "airborne": airborne,
           "warnings": [r["warning"] for r in airborne if "warning" in r]}
    if in_place:
        travel = root_pose[-1, :2] - root_pose[0, :2]
        out["in_place"] = {"method": in_place, "travel_removed_m": round(float(np.linalg.norm(travel)), 3),
                           "mean_speed_m_s": round(float(np.linalg.norm(travel) * fps / max(len(P) - 1, 1)), 2),
                           "stride_s": round(stride_s, 3),
                           "root_travel_left_m": round(float(np.linalg.norm(root[-1, :2] - root[0, :2])), 3),
                           "travel_restored_m": restored, "floor_drift_removed_m": floor_drift}
    return out


def level_floor(root, ankles, fps=30.0, window_s=0.5, min_drift=0.03, flight=0.05, support=0.75):
    """A floor that rises or sinks through the clip (a camera pitch error turns travel into height): the lowest foot's
    lowest point per window_s, fitted by a line in time, refitted without the windows more than `flight` m above it
    (a jump or a hurdle clearance, not the floor). Applied only if at least `support` of the windows lie on the line
    and it moves more than min_drift m over the clip: the root is lowered by it. -> (root, metres removed, or None)."""
    low = (ankles[..., 2] + root[:, None, 2]).min(1)
    w = max(2, int(round(window_s * fps)))
    idx = np.array([a + int(np.argmin(low[a:a + w])) for a in range(0, len(low) - w // 2, w)], float)
    if len(idx) < 4:
        return root, None
    on = np.ones(len(idx), bool)
    for _ in range(3):
        slope, icpt = np.polyfit(idx[on], low[idx.astype(int)][on], 1)
        on = low[idx.astype(int)] - (slope * idx + icpt) < flight
        if on.sum() < 2:
            return root, None
    if on.mean() < support:
        return root, None
    drift = slope * (np.arange(len(root)) - idx[0])
    if abs(drift[-1] - drift[0]) < min_drift:
        return root, None
    return root - np.c_[np.zeros((len(root), 2)), drift], round(float(drift[-1] - drift[0]), 3)


def restore_travel(root, ankles, fps=30.0, near=0.05, window_s=1.0, min_speed=0.3):
    """root: (T, 3), ankles: (T, 2, 3) relative to it. A foot on the floor doesn't move, so while the lowest foot is
    within `near` of its floor level, the root moves at minus that foot's velocity relative to it. Where the root's
    own path falls short of that (a camera that followed the runner, whose background barely moves, so the clip reads
    as a treadmill run), add the shortfall: per frame, smoothed over window_s, bridged where no foot is down. Applied
    only if the shortfall averages over min_speed m/s. -> (root, metres of travel added, or None)."""
    z = ankles[..., 2] + root[:, None, 2]
    low = np.argmin(z, 1)
    t = np.arange(len(root))
    floor = np.percentile(z.min(1), 5)
    vel_rel = np.gradient(ankles[..., :2], axis=0)[t, low] * fps
    vel_root = np.gradient(root[:, :2], axis=0) * fps
    # the lowest foot switching sides is a jump in `rel`, not a velocity: use frames where it doesn't switch
    ok = (z.min(1) - floor < near) & (np.r_[True, low[1:] == low[:-1]] & np.r_[low[1:] == low[:-1], True])
    if ok.sum() < max(5, 0.2 * len(root)):
        return root, None
    short = np.where(ok[:, None], -vel_rel - vel_root, np.nan)
    short = uniform_filter1d(_bridge(short), max(3, int(round(window_s * fps)) | 1), axis=0, mode="nearest")
    if np.linalg.norm(short.mean(0)) < min_speed:
        return root, None
    add = np.cumsum(short, 0) / fps
    add -= add[0]
    return root + np.c_[add, np.zeros(len(root))], round(float(np.linalg.norm(add[-1])), 3)


def stride_period(stance, fps=30.0):
    """A gait's stride in seconds: the median time between one foot's successive touch-downs (the
    starts of its contacts, over both feet); 1 s when the clip has fewer than two per foot."""
    gaps = [np.diff([a for a, _ in _runs(stance[:, f])]) for f in range(stance.shape[1])]
    gaps = np.concatenate([g for g in gaps if len(g)] or [[]])
    return float(np.median(gaps) / fps) if len(gaps) else 1.0


def stride_drift(root, stride_s, fps=30.0):
    """(T, 3) the root's horizontal travel with its within-stride surge and sway averaged out: a
    moving average over one stride (odd-reflected at the ends, so steady travel stays linear there),
    from zero at the first frame. root - stride_drift is the gait on the spot."""
    w = max(3, int(round(stride_s * fps)) | 1)
    xy = np.pad(root[:, :2], ((w, w), (0, 0)), mode="reflect", reflect_type="odd")
    smooth = uniform_filter1d(xy, w, axis=0)[w:-w]
    return np.c_[smooth - smooth[0], np.zeros(len(root))]


# Bump with any change to what the retarget produces; code_version() records it with the commit.
RETARGET_VERSION = "2026-10-31 contacts (touch-downs, per-take-off flights) + leg-plane knees and feet + arm reseeding"


def code_version():
    """What produced a retarget, for meta.json: RETARGET_VERSION, the git commit of this checkout
    (and whether retarget.py differs from it), and the parameters that shape the motion."""
    import subprocess
    from pathlib import Path

    def git(*args):
        try:
            out = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = git("rev-parse", "--short=12", "HEAD")
    deg = lambda x: round(float(np.degrees(x)), 2)
    return {"version": RETARGET_VERSION, "git": commit,
            "retarget_py_modified": None if commit is None else bool(git("status", "--porcelain", "--", "retarget.py")),
            "params": {"max_joint_speed": MAX_JOINT_SPEED, "flight_rise_frac": FLIGHT_RISE_FRAC, "long_airborne_s": LONG_AIRBORNE_S,
                       "contact_root_speed_frac": CONTACT_ROOT_SPEED_FRAC, "ankle_margin_deg": deg(ANKLE_MARGIN),
                       "ankle_roll_margin_deg": deg(ANKLE_ROLL_MARGIN), "leg_yaw_max_deg": deg(LEG_YAW_MAX),
                       "pivot_yaw_deg": deg(PIVOT_YAW), "pelvis_lean": PELVIS_LEAN, "arm_continuity_cost": ARM_CONTINUITY_COST,
                       "arm_bidirectional": ARM_BIDIRECTIONAL, "arm_reseed_err": ARM_RESEED_ERR,
                       "position_cost": POSITION_COST, "orientation_cost": ORIENTATION_COST}}


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
