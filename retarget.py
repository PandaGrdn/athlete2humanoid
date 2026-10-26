"""Map a human trajectory onto G1 joint angles with mink inverse kinematics, and export it
as an mjlab motion CSV.

Run this file for a synthetic squat smoke test.
"""

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
        frame, ftype, _ = FRAMES[name]
        return self.config.get_transform_frame_to_world(frame, ftype).translation()

    def solve(self, targets: dict, rotations: dict | None = None, iters=50, dt=0.02):
        """targets: {name: xyz}; rotations: {name or "torso": 3x3 world rotation}. Only frames
        named in targets are driven this call. Warm-starts from the previous frame's solution,
        so call in time order."""
        rotations = rotations or {}
        for name, pos in targets.items():
            task = self.tasks[name]
            rot = (mink.SO3.from_matrix(rotations[name]) if name in rotations
                   else task.transform_target_to_world.rotation())
            task.set_target(mink.SE3.from_rotation_and_translation(rot, np.asarray(pos, float)))
        if "torso" in rotations:
            self.torso.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rotations["torso"]), np.zeros(3)))
        tasks = [*(self.tasks[n] for n in targets), self.posture, self.torso]
        for _ in range(iters):
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


def human_to_targets(P, rt: G1Retargeter):
    """P: (T, 33, 3) human trajectory from pose.human_trajectory (metres, z-up, facing +x).
    Scales it to G1 size, puts the feet on the floor and the start under the robot, and returns
    (targets_seq, rotations_seq) for G1Retargeter.retarget_sequence."""
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


def tracking_report(rt: G1Retargeter, qpos, targets_seq):
    """Mean and max position error (cm) per frame type: how well the G1 can follow the human."""
    errs = {n: [] for n in FRAMES}
    for q, targets in zip(qpos, targets_seq):
        rt.config.update(q)
        for n, pos in targets.items():
            errs[n].append(np.linalg.norm(rt.frame_position(n) - pos) * 100)
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
