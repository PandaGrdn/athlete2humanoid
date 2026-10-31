"""Run in place (retarget.py in_place, pipeline.py --in-place): a synthetic jog that travels 4 m becomes a gait on the
spot whose planted feet stay put and whose vertical motion and contacts are the jog's; its keypose spec and variants
still compile. The numpy parts need mink/mujoco to import retarget; the G1 parts need the Menagerie model.

    python -m unittest tests.test_in_place
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

if not all(importlib.util.find_spec(m) for m in ("numpy",)):
    raise unittest.SkipTest("needs numpy")

import numpy as np

try:
    import config
    import retarget
except ImportError as e:
    retarget = None
    SKIP = f"retarget deps missing: {e}"
else:
    SKIP = None

HAVE_G1 = retarget is not None and config.G1_SCENE_XML.exists()
HAVE_SKILL = all(importlib.util.find_spec(m) for m in ("torch", "mujoco", "scipy"))

FPS, PERIOD, STANCE = 30.0, 18, 6  # a stride every 0.6 s, each foot down 0.2 s of it: two 0.1 s flights per stride


def synthetic_jog(T=90, speed=2.5, thigh=0.42, shin=0.42):
    """(T, 33, 3) landmarks of a jog along +x: each foot fixed on the floor through its contacts (at the hips' position
    mid-contact), swinging a stride ahead in between; the hips lowest mid-stance and highest in flight."""
    t = np.arange(T, dtype=float)
    hip_x = speed * t / FPS
    hip_z = 0.80 - 0.03 * np.cos(2 * np.pi * (t - (STANCE - 1) / 2) / (PERIOD / 2))
    L = speed * PERIOD / FPS
    P = np.zeros((T, 33, 3))
    for i, (y, off) in enumerate(((0.09, 0), (-0.09, PERIOD // 2))):
        p = (t + off) % PERIOD
        mid = t - p + (STANCE - 1) / 2  # the frame mid-way through this stride's contact
        x_c = speed * mid / FPS
        s = np.clip((p - STANCE) / (PERIOD - STANCE), 0, 1)
        ankle = np.stack([x_c + L * (1 - np.cos(np.pi * s)) / 2, np.full(T, y), 0.08 + 0.25 * np.sin(np.pi * s)], -1)
        hip = np.stack([hip_x, np.full(T, y), hip_z], -1)
        d = ankle - hip
        D = np.minimum(np.linalg.norm(d, axis=-1), thigh + shin - 1e-3)
        u = d / np.linalg.norm(d, axis=-1, keepdims=True)
        a = np.arccos(np.clip((thigh ** 2 + D ** 2 - shin ** 2) / (2 * thigh * D), -1, 1))
        knee_dir = np.stack([u[:, 0] * np.cos(a) - u[:, 2] * np.sin(a), np.zeros(T), u[:, 0] * np.sin(a) + u[:, 2] * np.cos(a)], -1)
        P[:, 23 + i], P[:, 25 + i] = hip, hip + thigh * knee_dir  # the knee ahead of the hip-ankle line
        P[:, 27 + i] = hip + D[:, None] * u
    mid = P[:, [23, 24]].mean(1)
    for i, y in ((0, 1), (1, -1)):
        swing = 0.4 * np.sin(2 * np.pi * (t + i * PERIOD / 2) / PERIOD)  # arms swing against the legs
        shoulder = mid + [0.05, y * 0.17, 0.5]
        P[:, 11 + i] = shoulder
        P[:, 13 + i] = shoulder + 0.28 * np.stack([np.sin(swing), np.zeros(T), -np.cos(swing)], -1)
        P[:, 15 + i] = P[:, 13 + i] + 0.25 * np.stack([np.sin(swing + 1.4), np.zeros(T), -np.cos(swing + 1.4)], -1)
    P[:, 0] = mid + [0.12, 0, 0.7]
    return P


@unittest.skipIf(SKIP, SKIP)
class InPlaceMotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = synthetic_jog()
        cls.rt = retarget.G1Retargeter() if HAVE_G1 else None

    def setUp(self):
        if not HAVE_G1:
            self.skipTest("needs the G1 model (scripts/fetch_assets.sh)")

    def motion(self, method):
        return retarget.proportional_motion(self.P, self.rt, FPS, in_place=method)

    def test_stride_period(self):
        st = self.motion(None)["stance"]
        self.assertAlmostEqual(retarget.stride_period(st, FPS), PERIOD / FPS, delta=1.5 / FPS)
        self.assertEqual(retarget.stride_period(np.ones((40, 2), bool), FPS), 1.0)  # no strides: the fallback

    def test_world_holds_the_root_and_plants_the_feet(self):
        travel, m = self.motion(None), self.motion("world")
        np.testing.assert_array_equal(m["stance"], travel["stance"])  # contacts found on the travelling motion
        hip = m["joints"]["hip_mid"]
        self.assertGreater(np.linalg.norm(travel["joints"]["hip_mid"][-1, :2]), 2.0)  # the jog does travel
        np.testing.assert_allclose(hip[:, :2], 0, atol=1e-9)  # and in place it doesn't
        for f, s in enumerate(("left", "right")):  # planted feet stand still and on the floor
            feet = m["joints"][f"{s}_ankle"]
            for a, b in retarget._runs(m["stance"][:, f]):
                np.testing.assert_allclose(feet[a:b, :2] - feet[a, :2], 0, atol=1e-9)
        # the vertical motion is the jog's: the same bounce, flights still above the floor
        np.testing.assert_allclose(np.ptp(hip[:, 2]), np.ptp(travel["joints"]["hip_mid"][:, 2]), atol=0.01)
        info = m["in_place"]
        self.assertEqual(info["method"], "world")
        self.assertAlmostEqual(info["travel_removed_m"], 2.5 * 89 / FPS * m["scale"], delta=0.05)
        self.assertLess(info["root_travel_left_m"], 1e-6)

    def test_stride_removes_the_travel_but_slides_the_feet(self):
        m = self.motion("stride")
        hip = m["joints"]["hip_mid"]
        self.assertLess(np.linalg.norm(hip[-1, :2] - hip[0, :2]), 0.1)
        left, st = m["joints"]["left_ankle"], m["stance"][:, 0]
        a, b = retarget._runs(st)[1]
        self.assertGreater(np.linalg.norm(left[b - 1, :2] - left[a, :2]), 0.1)  # carried back by the drift

    def test_soft_stays_within_its_bound(self):
        m = self.motion("soft")
        hip = m["joints"]["hip_mid"]
        # each horizontal axis within +-IN_PLACE_SOFT_M of a fixed centre: a band 2 * IN_PLACE_SOFT_M wide (this synthetic
        # jog's hips move at a constant speed, so its within-stride surge, and the soft root's motion, is ~0)
        self.assertTrue((np.ptp(hip[:, :2], axis=0) <= 2 * retarget.IN_PLACE_SOFT_M + 1e-6).all())
        for f, s in enumerate(("left", "right")):
            feet = m["joints"][f"{s}_ankle"]
            for a, b in retarget._runs(m["stance"][:, f]):
                np.testing.assert_allclose(feet[a:b, :2] - feet[a, :2], 0, atol=1e-9)

    def test_dolly_shot_travel_is_restored(self):
        # A camera that ran alongside: the hips stay put in the image and the planted feet slide back under them.
        P = self.P.copy()
        P[..., 0] -= (P[:, [23, 24], 0].mean(1))[:, None]
        travel = retarget.proportional_motion(self.P, self.rt, FPS)
        m = retarget.proportional_motion(P, self.rt, FPS, in_place="world")
        r = m["in_place"]["travel_restore"]
        self.assertLess(r["before_m"], 0.1 * r["after_m"])  # the treadmill read travels nowhere; restored, it runs
        self.assertGreater(r["after_m"], 0.8 * m["in_place"]["travel_removed_m"])
        np.testing.assert_allclose(m["stance"].mean(0), travel["stance"].mean(0), atol=0.06)  # its contacts are found
        _, added = retarget.restore_travel(*self._root_ankles(self.P), FPS)
        self.assertIsNone(added)  # a clip that already travels gets nothing added

    def test_level_floor(self):
        root, ankles = self._root_ankles(self.P)
        T = len(root)
        tilted = root + np.c_[np.zeros((T, 2)), np.linspace(0, 0.3, T)]  # the floor rises 30 cm through the clip
        levelled, drift = retarget.level_floor(tilted, ankles, FPS)
        self.assertAlmostEqual(drift, 0.3, delta=0.03)
        np.testing.assert_allclose(levelled[:, 2] - levelled[0, 2], root[:, 2] - root[0, 2], atol=0.03)
        self.assertIsNone(retarget.level_floor(root, ankles, FPS)[1])  # a level floor is left alone
        jump = root.copy()  # the clip ends in a long flight: the feet leave the floor, the floor doesn't rise
        jump[-25:, 2] += 0.4 * np.sin(np.linspace(0, np.pi / 2, 25))
        self.assertIsNone(retarget.level_floor(jump, ankles, FPS)[1])

    def _root_ankles(self, P):
        root = P[:, [23, 24]].mean(1)
        return root, P[:, [27, 28]] - root[:, None]

    def test_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            self.motion("treadmill")
        with self.assertRaises(ValueError):
            retarget.human_to_targets(self.P, self.rt, proportional=False, in_place="world")

    def test_report_and_default(self):
        self.assertIn(retarget.IN_PLACE_DEFAULT, retarget.IN_PLACE_METHODS)
        report = {}
        retarget.human_to_targets(self.P, self.rt, fps=FPS, report=report, in_place="world")
        self.assertEqual(report["in_place"]["method"], "world")
        report = {}
        retarget.human_to_targets(self.P, self.rt, fps=FPS, report=report)
        self.assertNotIn("in_place", report)


@unittest.skipIf(SKIP or not HAVE_G1, SKIP or "needs the G1 model")
class InPlaceOnTheG1(unittest.TestCase):
    """The solved qpos: the world method keeps planted soles still (less slip than stride), and the spec and the
    variants compile on it."""

    @classmethod
    def setUpClass(cls):
        from experiments.in_place.measure import slip
        P = synthetic_jog(T=72)
        rt = retarget.G1Retargeter()
        stance = retarget.proportional_motion(P, rt, FPS)["stance"]
        cls.qpos, cls.slip = {}, {}
        for m in retarget.IN_PLACE_METHODS:
            rt = retarget.G1Retargeter()
            cls.qpos[m] = rt.retarget_sequence(*retarget.human_to_targets(P, rt, fps=FPS, in_place=m), fps=FPS)
            cls.slip[m] = slip(rt.model, cls.qpos[m], FPS, stance)

    def test_world_slips_less_than_stride(self):
        w, s = self.slip["world"]["mean_cm_s"], self.slip["stride"]["mean_cm_s"]
        self.assertIsNotNone(w)
        self.assertLess(w, 10.0)
        self.assertLess(w, s / 5)
        root = self.qpos["world"][:, :2]
        self.assertLess(np.linalg.norm(root - root[0], axis=-1).max(), 0.15)

    @unittest.skipUnless(HAVE_SKILL, "needs torch, mujoco and scipy")
    def test_spec_and_variants_compile(self):
        from skill import keypose
        from skill import variations as V
        qpos = self.qpos["world"]
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "qpos.npy", qpos)
            (Path(d) / "meta.json").write_text(json.dumps({"fps": FPS, "in_place": {"method": "world"}}))
            spec, _ = keypose.compile_clip(d)
        sc = spec["self_check"]
        self.assertGreater(sc["stages_total"], 0)
        self.assertEqual(sc["stages_completed"], sc["stages_total"])
        vs = V.generate(qpos, FPS, [(False, 1.0, 1.0, 0.0), (True, 1.0, 1.0, 0.0), (False, 1.33, 1.0, 0.0)], log=lambda *a: None)
        self.assertTrue(next(v for v in vs if v["name"] == "o_t1_a1")["ok"], [v["why"] for v in vs])
        self.assertGreaterEqual(sum(v["ok"] for v in vs), 2, [v["why"] for v in vs])


if __name__ == "__main__":
    unittest.main()
