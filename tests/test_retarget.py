"""Proportional retarget (retarget.py): a synthetic human with long arms and short legs, squatting
with planted feet, rebuilt on the G1's skeleton. The numpy parts run anywhere mink/mujoco import;
the G1 parts need the Menagerie model (scripts/fetch_assets.sh, or A2H_ASSETS_DIR).

    python -m unittest tests.test_retarget
"""

import importlib.util
import unittest

if not all(importlib.util.find_spec(m) for m in ("numpy",)):  # e.g. a bare system Python
    raise unittest.SkipTest("needs numpy")

import numpy as np

try:
    import config
    import retarget
except ImportError as e:  # mink / mujoco / mediapy missing
    retarget = None
    SKIP = f"retarget deps missing: {e}"
else:
    SKIP = None

HAVE_G1 = retarget is not None and config.G1_SCENE_XML.exists()

# The human: arms 1.5x and legs 0.7x a typical adult's, torso and widths typical.
HUMAN = {"thigh": 0.42 * 0.7, "shin": 0.42 * 0.7, "upper_arm": 0.30 * 1.5, "forearm": 0.26 * 1.5,
         "spine": 0.50, "hip_width": 0.18, "shoulder_width": 0.34}
G1_LIKE = {"thigh": 0.33, "shin": 0.32, "upper_arm": 0.19, "forearm": 0.18, "spine": 0.41,
           "hip_width": 0.22, "shoulder_width": 0.28}


def synthetic_squat(T=60, bones=HUMAN, depth=np.radians(90), elbow=np.radians(60), lean=np.radians(30)):
    """(T, 33, 3) landmarks: feet fixed on the floor, knees bending 0 -> depth -> 0, the trunk
    leaning forward and the arms reaching forward with a bent elbow as the knees bend."""
    s = (1 - np.cos(np.linspace(0, 2 * np.pi, T))) / 2
    knee, trunk_lean, elbow_flex = s * depth, s * lean, np.radians(10) + s * elbow
    P = np.zeros((T, 33, 3))
    sag = lambda a, down=True: np.stack([np.sin(a), np.zeros_like(a), -np.cos(a) if down else np.cos(a)], -1)
    shin = sag(-knee / 2) * bones["shin"]  # hip -> knee -> ankle; the knee ahead of the ankle
    thigh = sag(knee / 2) * bones["thigh"]
    for i, y in ((0, 1), (1, -1)):
        P[:, 27 + i] = [0.0, y * bones["hip_width"] / 2, 0.08]
        P[:, 25 + i] = P[:, 27 + i] - shin
        P[:, 23 + i] = P[:, 25 + i] - thigh
    mid = P[:, [23, 24]].mean(1)
    spine = sag(trunk_lean, down=False) * bones["spine"]
    shoulder_mid = mid + spine
    for i, y in ((0, 1), (1, -1)):
        shoulder = shoulder_mid + [0, y * bones["shoulder_width"] / 2, 0]
        upper = sag(s * np.radians(60)) * bones["upper_arm"]  # arm swings forward
        fore = sag(s * np.radians(60) + elbow_flex) * bones["forearm"]
        P[:, 11 + i], P[:, 13 + i] = shoulder, shoulder + upper
        P[:, 15 + i] = P[:, 13 + i] + fore
    P[:, 0] = shoulder_mid + [0.1, 0, 0.2]  # a nose, so nothing is NaN
    return P


def human_joints(P):
    idx = {"left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26, "left_ankle": 27,
           "right_ankle": 28, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13,
           "right_elbow": 14, "left_wrist": 15, "right_wrist": 16}
    J = {k: P[:, i] for k, i in idx.items()}
    J["hip_mid"], J["shoulder_mid"] = P[:, [23, 24]].mean(1), P[:, [11, 12]].mean(1)
    return J


@unittest.skipIf(SKIP, SKIP)
class RescaleBones(unittest.TestCase):
    def setUp(self):
        self.P = synthetic_squat()
        self.J = retarget.rescale_bones(self.P, G1_LIKE)

    def test_bones_take_robot_lengths(self):
        L = lambda a, b: np.linalg.norm(self.J[a] - self.J[b], axis=-1)
        for bone, parent, child, _, _ in retarget.LIMBS:
            np.testing.assert_allclose(L(parent, child), G1_LIKE[bone], atol=1e-9, err_msg=bone)
        np.testing.assert_allclose(L("left_hip", "right_hip"), G1_LIKE["hip_width"], atol=1e-9)
        np.testing.assert_allclose(L("left_shoulder", "right_shoulder"), G1_LIKE["shoulder_width"], atol=1e-9)
        np.testing.assert_allclose(L("hip_mid", "shoulder_mid"), G1_LIKE["spine"], atol=1e-9)

    def test_joint_angles_preserved(self):
        human = retarget.flexion_angles(human_joints(self.P))
        robot = retarget.flexion_angles(self.J)
        for k in human:
            np.testing.assert_allclose(robot[k], human[k], atol=1e-6, err_msg=k)
        self.assertGreater(human["left_knee"].max(), 85)  # the motion really bends

    def test_relative_to_hip_midpoint(self):
        np.testing.assert_allclose(self.J["hip_mid"], 0)


@unittest.skipIf(SKIP, SKIP)
class Contacts(unittest.TestCase):
    def test_stance_and_planting(self):
        T = 90
        t = np.arange(T)
        # Left foot planted with 1 cm of jitter; right foot lifts and steps forward mid-clip.
        rng = np.random.default_rng(0)
        left = np.c_[np.zeros(T), np.full(T, 0.1), np.full(T, 0.05)] + rng.normal(0, 0.003, (T, 3))
        swing = (t >= 30) & (t < 50)
        right = np.c_[np.where(t < 40, 0.0, 0.3), np.full(T, -0.1), np.full(T, 0.05)]
        right[swing] += [0, 0, 0.15]
        right[swing, 0] = np.linspace(0, 0.3, swing.sum())
        world = np.stack([left, right], 1)
        stance = retarget.stance_mask(world)
        self.assertTrue(stance[:, 0].all())
        self.assertFalse(stance[35:45, 1].any())
        self.assertTrue(stance[:28, 1].all() and stance[52:, 1].all())

        root = np.c_[np.zeros(T), np.zeros(T), np.full(T, 0.7)] + rng.normal(0, 0.003, (T, 3))
        rel = world - root[:, None]
        root2, feet, fix = retarget.plant_feet(root, rel, stance, floor=0.035)
        planted = feet[stance[:, 0], 0]
        np.testing.assert_allclose(planted - planted[0], 0, atol=1e-9)  # no sliding
        np.testing.assert_allclose(planted[:, 2], 0.035)  # on the floor
        for a, b in retarget._runs(stance[:, 1]):
            np.testing.assert_allclose(feet[a:b, 1] - feet[a, 1], 0, atol=1e-9)
        self.assertTrue((feet[..., 2] >= 0.035 - 1e-12).all())  # nothing below the floor
        self.assertGreater(feet[40, 1, 2], 0.1)  # the swing foot still lifts

    def test_phantom_flight_grounded_real_flight_kept(self):
        T, fps, floor = 90, 30.0, 0.035
        stance = np.ones((T, 2), bool)
        stance[30:60] = False  # 1 s with neither foot planted
        feet = np.full((T, 2, 3), floor)
        root = np.c_[np.zeros(T), np.zeros(T), np.full(T, 0.7)]
        # Phantom: both feet 6 cm up for a second, the pelvis level (a lifted stance, not a flight).
        hover = feet.copy()
        hover[30:60, :, 2] += 0.06
        r, f = retarget.ground_phantom_flights(root, hover, stance, floor, fps)
        np.testing.assert_allclose(f[..., 2].min(1), floor, atol=1e-9)  # the lowest foot back down
        np.testing.assert_allclose(r[30:60, 2], 0.7 - 0.06)  # the body lowered with it
        # A real jump: the pelvis rises on a ballistic arc (apex g T^2 / 8 for T = 0.4 s), the feet with it.
        stance[:] = True
        stance[40:52] = False
        s = (np.arange(40, 52) - 39.5) / fps
        rise = 9.81 / 2 * s * (12 / fps - s)
        jump, jroot = feet.copy(), root.copy()
        jroot[40:52, 2] += rise
        jump[40:52, :, 2] += rise[:, None]
        r, f = retarget.ground_phantom_flights(jroot, jump, stance, floor, fps)
        np.testing.assert_allclose(r, jroot)
        np.testing.assert_allclose(f, jump)

    def test_running_contacts_need_root_speed_and_touch_downs(self):
        # A run at 4 m/s: each foot is down 2 frames per stride (a sprinter's 0.07 s) with 0.4 m/s of
        # pose jitter on the planted ankle, and swings at twice the body's speed in between.
        fps, T, v = 30.0, 60, 4.0
        rng = np.random.default_rng(1)
        t = np.arange(T)
        root = np.c_[v * t / fps, np.zeros(T), np.full(T, 0.9)]
        feet = np.zeros((T, 2, 3))
        for f, phase in ((0, 0), (1, 5)):
            down = ((t - phase) % 10) < 2
            x, last = np.zeros(T), 0.0
            for i in range(T):  # planted: stays put; in the air: catches up past the body
                last = last if down[i] else last + 2 * v / fps
                x[i] = last
            feet[:, f, 0] = x
            feet[:, f, 1] = 0.1 * (1 - 2 * f)
            lift = np.where(down, 0.0, np.sin(np.pi * (((t - phase) % 10) - 1) / 9).clip(0))
            feet[:, f, 2] = 0.08 + 0.15 * lift
        feet[..., :2] += rng.normal(0, 0.4 / fps, (T, 2, 2)).cumsum(0) * 0  # keep it deterministic
        feet[..., 0] += np.where(feet[..., 2] < 0.081, rng.normal(0, 0.005, (T, 2)), 0)
        stance = retarget.stance_mask(feet, fps, root=root)
        old = retarget.stance_mask(feet, fps, min_frames=4)  # the old window: no contacts survive
        for f, phase in ((0, 0), (1, 5)):
            down = ((t - phase) % 10) < 2
            self.assertGreater((stance[:, f] & down).sum() / down.sum(), 0.8)  # the touch-downs are found
            self.assertFalse((stance[:, f] & (feet[:, f, 2] > 0.15)).any())  # swing is never stance
        self.assertLess(old.mean(), 0.1)

    def test_missed_contacts_then_a_jump_keep_the_clearance(self):
        # Hurdles: contacts at the start, then a run-up whose touch-downs were all missed (the lowest
        # foot dips to 3 cm above the floor every 0.33 s), then a take-off and a rise of 50 cm that the
        # clip ends in. The old rule grounded the whole stretch; now the dips are interpolated
        # and the clearance kept.
        fps, T, floor = 30.0, 90, 0.035
        t = np.arange(T)
        stance = np.zeros((T, 2), bool)
        stance[:10] = True
        root = np.c_[np.zeros(T), np.zeros(T), np.full(T, 0.75)]
        low = 0.03 + 0.12 * np.abs(np.sin(np.pi * (t - 10) / 10))  # a stride every 10 frames
        low[:10] = 0
        root[60:, 2] += np.linspace(0, 0.5, T - 60)
        low[60:] += np.linspace(0, 0.5, T - 60)
        feet = np.stack([floor + low, floor + low + 0.1], 1)[..., None] * [0, 0, 1]
        report = []
        r, f = retarget.ground_phantom_flights(root, feet, stance, floor, fps, report=report)
        self.assertAlmostEqual(r[-1, 2] - r[10:60, 2].mean(), root[-1, 2] - root[10:60, 2].mean(), delta=0.035)
        for k in range(20, 60, 10):  # each missed touch-down put back on the floor
            self.assertAlmostEqual(f[k, 0, 2], floor, delta=1e-6)
        self.assertGreater(f[15, 0, 2] - floor, 0.08)  # the stride between still lifts the foot
        self.assertEqual(len(report), 1)
        self.assertTrue(report[0]["flights"][-1]["ends_clip"])
        self.assertIn("ends in the air", report[0]["warning"])

    def test_clip_ending_in_the_air_is_never_lowered(self):
        fps, T, floor = 30.0, 40, 0.035
        stance = np.ones((T, 2), bool)
        stance[30:] = False
        root = np.c_[np.zeros(T), np.zeros(T), np.full(T, 0.7)]
        feet = np.full((T, 2, 3), floor)
        rise = np.linspace(0, 0.1, 10)  # 0.33 s up, still rising when the clip stops
        root[30:, 2] += rise
        feet[30:, :, 2] += rise[:, None] + 0.02
        r, f = retarget.ground_phantom_flights(root, feet, stance, floor, fps)
        np.testing.assert_allclose(r, root)
        np.testing.assert_allclose(f, feet)

    def test_short_lifted_stance_is_grounded(self):
        fps, T, floor = 30.0, 60, 0.035
        stance = np.ones((T, 2), bool)
        stance[20:29] = False  # 0.3 s
        root = np.c_[np.zeros(T), np.zeros(T), np.full(T, 0.7)]
        feet = np.full((T, 2, 3), floor)
        feet[20:29, :, 2] += 0.05
        report = []
        r, f = retarget.ground_phantom_flights(root, feet, stance, floor, fps, report=report)
        np.testing.assert_allclose(f[..., 2].min(1), floor, atol=1e-9)
        self.assertEqual(report[0]["decision"], "lowered")
        self.assertNotIn("warning", report[0])

    def test_hops_survive_contacts_and_flights(self):
        # Jumping jacks: a 0.27 s hop of a few cm between stances, the feet spreading 25 cm each in
        # the air; the apex (near the floor, below the dwell window's 10 cm) is not a contact, and a
        # hop rising as its airtime needs is a flight.
        fps, floor = 30.0, 0.08
        T, hop = 60, np.arange(22, 30)
        z = np.full(T, floor)
        s = (hop - 21.5) / fps
        z[hop] += 9.81 / 2 * s * (len(hop) / fps - s) + 0.01
        spread = np.interp(np.arange(T), [21, 30], [0.0, 0.25])
        feet = np.stack([np.c_[np.zeros(T), y * (0.1 + spread), z] for y in (1, -1)], 1)
        stance = retarget.stance_mask(feet, fps, root=np.c_[np.zeros(T), np.zeros(T), np.full(T, 0.7)])
        self.assertFalse(stance[24:28].any())
        root = np.c_[np.zeros(T), np.zeros(T), 0.7 + z - floor]
        r, _ = retarget.ground_phantom_flights(root, feet, stance, floor, fps)
        np.testing.assert_allclose(r, root)

    def test_level_feet_keeps_leg_angles(self):
        J = retarget.rescale_bones(synthetic_squat(T=10), G1_LIKE)
        J["left_ankle"] = J["left_ankle"] + [0, 0, 0.04]  # depth error: one foot looks higher
        J["left_knee"] = J["left_knee"] + [0, 0, 0.04]
        level = retarget.level_feet(J, np.ones((10, 2), bool))
        np.testing.assert_allclose(level["left_ankle"][:, 2], level["right_ankle"][:, 2], atol=1e-9)
        before, after = retarget.flexion_angles(J), retarget.flexion_angles(level)
        for side in ("left", "right"):
            np.testing.assert_allclose(after[f"{side}_knee"], before[f"{side}_knee"], atol=1e-6)


@unittest.skipUnless(HAVE_G1, SKIP or "G1 model missing: run scripts/fetch_assets.sh")
class OnTheG1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = synthetic_squat(T=45)
        cls.rt = retarget.G1Retargeter()

    def test_measured_skeleton(self):
        b = self.rt.bones
        self.assertAlmostEqual(b["thigh"], 0.33, delta=0.02)
        self.assertAlmostEqual(b["shin"], 0.32, delta=0.02)
        self.assertAlmostEqual(b["hip_width"], 0.22, delta=0.02)
        self.assertAlmostEqual(b["upper_arm"] + b["forearm"], 0.375, delta=0.03)

    def test_motion_on_g1_proportions(self):
        m = retarget.proportional_motion(self.P, self.rt)
        J, b = m["joints"], self.rt.bones
        self.assertTrue(m["stance"].all())  # the synthetic feet never move
        for bone, parent, child, _, _ in retarget.LIMBS:
            np.testing.assert_allclose(np.linalg.norm(J[child] - J[parent], axis=-1), b[bone], atol=2e-3,
                                       err_msg=bone)
        for f in ("left_ankle", "right_ankle"):
            np.testing.assert_allclose(J[f] - J[f][0], 0, atol=1e-9)
            np.testing.assert_allclose(J[f][:, 2], self.rt.sole_height, atol=1e-9)
        human, target = retarget.flexion_angles(human_joints(self.P)), retarget.flexion_angles(J)
        for k in human:
            self.assertLess(np.abs(human[k] - target[k]).max(), 1.0, k)

    def test_ankle_roll_matches_the_g1_joint(self):
        import mujoco
        m, d = self.rt.model, mujoco.MjData(self.rt.model)
        rng = np.random.default_rng(0)
        err, off = [], []
        for _ in range(100):
            mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
            for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"):
                jid = m.joint(f"left_{j}_joint").id
                lo, hi = m.jnt_range[jid]
                lo, hi = (0.2, 1.5) if j == "knee" else (max(lo, -0.6), min(hi, 0.6)) if j.startswith("hip") else (lo, hi)
                d.qpos[m.jnt_qposadr[jid]] = rng.uniform(lo, hi)  # athletic ranges, not a full split
            mujoco.mj_kinematics(m, d)
            hip = d.body("pelvis").xpos + d.body("pelvis").xmat.reshape(3, 3) @ self.rt.hip_centre
            ankle, foot = d.body("left_ankle_roll_link").xpos, d.body("left_ankle_roll_link").xmat.reshape(3, 3)
            err.append(float(retarget.ankle_roll(hip, ankle, foot)) - float(d.joint("left_ankle_roll_joint").qpos[0]))
            # and the G1's own knee is about in the plane knee_in_leg_plane projects onto
            knee = d.body("left_knee_link").xpos
            off.append(np.linalg.norm(retarget.knee_in_leg_plane(hip[None], ankle[None], foot[None], knee[None])[0] - knee))
        # Approximate: the G1's hip is three offset hinges, whose effective centre sits ~2 cm (1 sd) off
        # the knee's plane depending on hip yaw and roll: a degree or two, which ANKLE_ROLL_MARGIN covers.
        self.assertLess(np.abs(err).mean(), 0.03)  # measured 1.0 degree mean, 4.6 max
        self.assertLess(np.abs(err).max(), 0.1)
        self.assertLess(np.mean(off), 0.02)

    def test_leg_heading(self):
        T, yaw = 4, np.zeros(4)
        c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
        ankle = np.zeros((T, 3))
        hip = np.tile([0.0, 0, 0.6], (T, 1))
        bent = np.tile([0.1 * c, 0.1 * s, 0.3], (T, 1))  # the knee forward along a 30 degree heading
        h = retarget.leg_heading(hip, bent, ankle, yaw, 0.33, 0.32)
        np.testing.assert_allclose(np.degrees(h), 30, atol=1e-6)
        straight = np.tile([0.0, 0, 0.3], (T, 1))
        np.testing.assert_allclose(retarget.leg_heading(hip, straight, ankle, yaw + 0.2, 0.33, 0.32), 0.2)

    def test_wide_stance_keeps_ankle_roll_in_range(self):
        # A wide squat with the knees out: the old foot target (flat, pelvis yaw) asked the ankle for
        # more roll than its 15 degrees and the IK gave up the knee instead.
        P = synthetic_squat(T=30)
        for i, y in ((0, 1), (1, -1)):
            P[:, 27 + i, 1] += y * 0.12
            P[:, 25 + i, 1] += y * 0.10
        rt = retarget.G1Retargeter()
        m = retarget.proportional_motion(P, rt)
        lim = rt.model.jnt_range[rt.model.joint("left_ankle_roll_joint").id][1]
        for f, s in enumerate(("left", "right")):
            roll = retarget.ankle_roll(m["joints"][f"{s}_hip"], m["joints"][f"{s}_ankle"], m["foot_rot"][:, f])
            self.assertLessEqual(np.abs(roll).max(), lim - retarget.ANKLE_ROLL_MARGIN + 1e-3)
        qpos = rt.retarget_sequence(*retarget.human_to_targets(P, rt))
        q = qpos[:, [rt.model.jnt_qposadr[rt.model.joint(f"{s}_ankle_roll_joint").id] for s in ("left", "right")]]
        self.assertLess(np.abs(q).max(), lim - np.radians(1))
        report = retarget.tracking_report(rt, qpos, retarget.human_to_targets(P, retarget.G1Retargeter())[0])
        self.assertLess(max(report["left_knee"][0], report["right_knee"][0]), 2.0)

    def test_code_version(self):
        v = retarget.code_version()
        self.assertEqual(v["version"], retarget.RETARGET_VERSION)
        self.assertEqual(v["params"]["flight_rise_frac"], retarget.FLIGHT_RISE_FRAC)
        self.assertIn("git", v)

    def test_ik_follows_angles_better_than_uniform(self):
        err = {}
        human = retarget.flexion_angles(human_joints(self.P))
        for prop in (False, True):
            rt = retarget.G1Retargeter()
            qpos = rt.retarget_sequence(*retarget.human_to_targets(self.P, rt, proportional=prop))
            # continuous: no joint faster than MAX_JOINT_SPEED (rad/s of video) after the first frame
            self.assertLessEqual(np.abs(np.diff(qpos[:, 7:], axis=0)).max() * 30, retarget.MAX_JOINT_SPEED + 1e-6)
            robot = retarget.flexion_angles(rt.skeleton(qpos))
            err[prop] = {k: np.abs(robot[k] - human[k]).mean() for k in human}
            feet = rt.skeleton(qpos)["left_ankle"]
            if prop:
                self.assertLess(np.linalg.norm(feet[:, :2] - feet[0, :2], axis=-1).max(), 0.01)
        for k in ("left_knee", "left_elbow", "left_shoulder"):
            self.assertLess(err[True][k], 5.0, k)
        self.assertLess(err[True]["left_elbow"], err[False]["left_elbow"])


@unittest.skipUnless(importlib.util.find_spec("mediapipe") and importlib.util.find_spec("imageio_ffmpeg") and not SKIP,
                     "pipeline.py needs mediapipe and imageio-ffmpeg")
class VideoFps(unittest.TestCase):
    def test_rate_from_timestamps(self):
        import tempfile

        import mediapy

        import pipeline
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/clip.mp4"
            mediapy.write_video(path, np.zeros((20, 32, 32, 3), np.uint8), fps=25)
            timing = pipeline.video_fps(path)
        self.assertAlmostEqual(timing["fps"], 25.0, places=2)
        self.assertEqual(timing["frames"], 20)


if __name__ == "__main__":
    unittest.main()
