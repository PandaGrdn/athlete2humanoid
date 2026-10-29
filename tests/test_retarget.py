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


if __name__ == "__main__":
    unittest.main()
