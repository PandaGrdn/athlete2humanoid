"""Day 3: map human landmarks onto G1 joint angles with mink inverse kinematics.

The IK solver below already works (run this file for a squat demo). What's left for day 3:
`scale_landmarks` (human proportions -> robot proportions) and `landmarks_to_targets`.
"""

import mink
import mujoco
import numpy as np

import config
import render

# G1 frames we drive, and the MediaPipe Pose landmark indices that feed them.
# MediaPipe: 0 nose, 11/12 shoulders, 15/16 wrists, 23/24 hips, 27/28 ankles, 31/32 foot index.
FRAMES = {
    # name           (frame_name,              frame_type, mediapipe indices averaged)
    "pelvis":       ("pelvis",                 "body", (23, 24)),
    "left_foot":    ("left_foot",              "site", (27, 31)),
    "right_foot":   ("right_foot",             "site", (28, 32)),
    "left_hand":    ("left_wrist_yaw_link",    "body", (15,)),
    "right_hand":   ("right_wrist_yaw_link",   "body", (16,)),
    "torso":        ("torso_link",             "body", (11, 12)),
}

# How hard each frame pulls. Feet matter most (no sliding), then pelvis, then hands.
POSITION_COST = {"pelvis": 10.0, "left_foot": 20.0, "right_foot": 20.0,
                 "left_hand": 2.0, "right_hand": 2.0, "torso": 1.0}
# Orientation targets stay at the starting pose: flat feet, pelvis softly upright, torso upright.
# Day 3 could set pelvis yaw from the hip line so the robot turns with the athlete.
# These costs were tuned on the squat demo below; re-check them on your real clip.
ORIENTATION_COST = {"pelvis": 0.1, "left_foot": 5.0, "right_foot": 5.0}
POSTURE_COST = 0.1
UPRIGHT_COST = 1.0


def bent_knee_pose(model, knee=0.5):
    """The 'stand' keyframe has straight knees: an IK singularity where the solver can't start
    a knee bend and tilts the whole pelvis instead. Start from a slight crouch, feet still flat."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    for side in ("left", "right"):
        data.joint(f"{side}_hip_pitch_joint").qpos = -knee / 2
        data.joint(f"{side}_knee_joint").qpos = knee
        data.joint(f"{side}_ankle_pitch_joint").qpos = -knee / 2
    return data.qpos.copy()


class G1Retargeter:
    def __init__(self, model=None):
        self.model = model if model is not None else render.load_g1()[0]
        self.config = mink.Configuration(self.model)
        self.config.update(bent_knee_pose(self.model))
        self.tasks = {
            name: mink.FrameTask(frame_name=frame, frame_type=ftype,
                                 position_cost=POSITION_COST[name],
                                 orientation_cost=ORIENTATION_COST.get(name, 0.0))
            for name, (frame, ftype, _) in FRAMES.items()
        }
        # Always-on regularisers: joints drift back toward standing, torso stays roughly upright.
        self.posture = mink.PostureTask(self.model, cost=self._posture_costs())
        self.posture.set_target_from_configuration(self.config)
        self.upright = mink.FrameTask("torso_link", "body", position_cost=0.0,
                                      orientation_cost=UPRIGHT_COST)
        self.upright.set_target_from_configuration(self.config)
        for task in self.tasks.values():
            task.set_target_from_configuration(self.config)
        self.limits = [mink.ConfigurationLimit(self.model)]

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

    def solve(self, targets: dict, iters=50, dt=0.02):
        """targets: {name: xyz (world, metres)}. Only frames named here are driven this call.
        Warm-starts from the previous frame's solution, so call in time order."""
        for name, pos in targets.items():
            task = self.tasks[name]
            rot = task.transform_target_to_world.rotation()
            task.set_target(mink.SE3.from_rotation_and_translation(rot, np.asarray(pos, float)))
        tasks = [*(self.tasks[n] for n in targets), self.posture, self.upright]
        for _ in range(iters):
            vel = mink.solve_ik(self.config, tasks, dt, "daqp", damping=1e-3, limits=self.limits)
            self.config.integrate_inplace(vel, dt)
        return self.config.q.copy()

    def retarget_sequence(self, targets_seq):
        """targets_seq: list of target dicts, one per video frame -> (frames, nq) qpos array."""
        return np.stack([self.solve(t) for t in targets_seq])


def scale_landmarks(world_landmarks):
    """Day 3: (frames, 33, 3) MediaPipe world landmarks -> G1-sized, z-up, feet on the floor.

    Plan: convert axes (MediaPipe is y-down), scale by G1 leg length / human leg length,
    and shift so the lowest foot touches z=0 at the first frame.
    """
    raise NotImplementedError("Day 3")


def landmarks_to_targets(scaled_landmarks):
    """Day 3: average the MediaPipe indices in FRAMES -> one target dict per frame."""
    return [
        {name: frame_lms[list(idx)].mean(axis=0) for name, (_, _, idx) in FRAMES.items()}
        for frame_lms in scaled_landmarks
    ]


if __name__ == "__main__":
    # Smoke test: keep the feet planted, lower the pelvis 25 cm and back up, i.e. a squat.
    rt = G1Retargeter()
    feet = {k: rt.frame_position(k) for k in ("left_foot", "right_foot")}
    pelvis0 = rt.frame_position("pelvis")
    depth = 0.25 * np.sin(np.linspace(0, np.pi, 60))
    seq = [{**feet, "pelvis": pelvis0 - [0, 0, d]} for d in depth]
    qpos = rt.retarget_sequence(seq)

    lowest = int(np.argmax(depth))
    rt.config.update(qpos[lowest])
    err = np.linalg.norm(rt.frame_position("pelvis") - seq[lowest]["pelvis"])
    foot_err = max(np.linalg.norm(rt.frame_position(k) - feet[k]) for k in feet)
    print(f"bottom of squat: pelvis error {err*100:.1f} cm, worst foot drift {foot_err*100:.1f} cm")
    out = render.save_video(render.render_qpos_sequence(rt.model, qpos), config.OUTPUT_DIR / "g1_ik_squat.mp4")
    print(f"saved {out}")
