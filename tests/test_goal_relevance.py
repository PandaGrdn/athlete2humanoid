"""skill/importance.py: goal relevance gives the takeoff and landing high weight and injected
arm-waving ("hyping the crowd") low weight, on a synthetic squat jump (no mujoco) and, when the G1
assets and the squat_jump clip are present, on the real retarget. Plus the V-JEPA 2 predictor
masking path (latent/predictor_relevance.py) on a tiny randomly initialised model.

    python -m unittest tests.test_goal_relevance
"""

import importlib.util
import json
import os
import unittest
from pathlib import Path

import numpy as np

from skill import importance as I
from skill import task as T

FPS = 30.0
G = 9.81
REPO = Path(__file__).resolve().parents[1]
CLIP = Path(os.environ.get("A2H_TEST_CLIP", REPO / "outputs/remote_latent/clips/squat_jump"))
HYPE = (0.2, 1.4)  # s: ends before the countermovement starts


def _smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def synthetic(hype=False, swing=True, lateral=False):
    """A point-mass squat jump: stand 2 s, countermovement 0.5 s, push 0.3 s, ballistic flight,
    absorbing landing, stand. Bodies (mass): pelvis 10, torso 8, legs 4 + 4, arms 3 + 3.
    swing: arms swing down-back in the descent and up in the push (they help the jump).
    hype: arms waved up and down at 2 Hz during HYPE. lateral: arms waved sideways during the jump.
    -> (kin dict as from importance.kinematics, task body positions (T, 14, 3))."""
    t = np.arange(0, 6.0, 1 / FPS)
    z0, zb, zt = 0.80, 0.58, 0.81  # standing, squat bottom, takeoff pelvis heights
    v0 = 2.5
    t_d, t_p, t_to = 2.0, 2.5, 2.7
    t_fl = 2 * v0 / G
    t_td = t_to + t_fl
    z = np.full_like(t, z0)
    d = (t >= t_d) & (t < t_p)
    z[d] = z0 - (z0 - zb) * _smooth((t[d] - t_d) / (t_p - t_d))
    p = (t >= t_p) & (t < t_to)
    z[p] = zb + (zt - zb) * np.sin(0.5 * np.pi * (t[p] - t_p) / (t_to - t_p))
    f = (t >= t_to) & (t < t_td)
    z[f] = zt + v0 * (t[f] - t_to) - 0.5 * G * (t[f] - t_to) ** 2
    la = (t >= t_td) & (t < t_td + 0.3)
    z[la] = zt - (zt - 0.66) * np.sin(0.5 * np.pi * (t[la] - t_td) / 0.3)
    r = (t >= t_td + 0.3) & (t < t_td + 0.9)
    z[r] = 0.66 + (z0 - 0.66) * _smooth((t[r] - t_td - 0.3) / 0.6)
    feet_z = np.where(f, 0.05 + (z - zt), 0.05)  # legs extended in flight: feet rise with the body
    arm = np.zeros((len(t), 3))
    if swing:
        arm[:, 2] += np.where(d, -0.15 * _smooth((t - t_d) / 0.5), 0.0)
        arm[:, 2] += np.where(p | f, -0.15 + 0.45 * _smooth((t - t_p) / 0.3), 0.0)  # still rising at takeoff
        arm[:, 2] += np.where(t >= t_td, 0.30 * (1 - _smooth((t - t_td) / 0.6)), 0.0)
    if hype:
        w = _smooth(np.minimum(t - HYPE[0], HYPE[1] - t) / 0.2)
        arm[:, 2] += w * (0.25 + 0.25 * np.sin(2 * np.pi * 2 * t))
        arm[:, 0] += w * 0.1 * np.cos(2 * np.pi * 2 * t)
    if lateral:
        w = _smooth(np.minimum(t - t_d, t_td - t) / 0.1)
        arm[:, 1] += w * 0.2 * np.sin(2 * np.pi * 3 * t)
    n = len(t)
    pel = np.stack([np.zeros(n), np.zeros(n), z], 1)
    torso = pel + [0, 0, 0.3]
    feet = np.stack([np.stack([np.zeros(n), np.full(n, s * 0.1), feet_z], 1) for s in (1, -1)], 1)
    hands = np.stack([torso + [0, s * 0.2, -0.3] + arm * [1, s, 1] for s in (1, -1)], 1)
    com = np.stack([pel, torso, (pel + feet[:, 0]) / 2, (pel + feet[:, 1]) / 2,
                    (torso + hands[:, 0]) / 2, (torso + hands[:, 1]) / 2], 1)
    kin = {"com": com, "mass": np.array([10.0, 8, 4, 4, 3, 3]), "feet": feet, "pelvis": pel,
           "group": np.array([I.GROUPS.index(g) for g in ("torso", "torso", "left_leg", "right_leg",
                                                         "left_arm", "right_arm")])}
    pos = np.zeros((n, 14, 3), np.float32)
    pos[:] = pel[:, None]
    pos[:, T.PELVIS], pos[:, T.TORSO] = pel, torso
    pos[:, T.L_ANKLE], pos[:, T.R_ANKLE] = feet[:, 0], feet[:, 1]
    pos[:, T.L_HAND], pos[:, T.R_HAND] = hands[:, 0], hands[:, 1]
    kin["pos"] = pos
    return kin, pos


def run(kin, spec):
    ev = I.goal_events(spec, kin["pos"], FPS)
    S, summary = I.physics_scores(kin, ev, FPS)
    w, body = I.combine(S, fps=FPS)
    return w, body, summary, ev


def window(spec, kind):
    st = next(s for s in spec["stages"] if s["kind"] == kind)
    return int(st["start_s"] * FPS), int(st["end_s"] * FPS)


class TestSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kin, pos = synthetic()
        cls.spec = T.compile_spec(pos, FPS, name="synthetic")
        cls.clean = run(kin, cls.spec)
        cls.hype = run(synthetic(hype=True)[0], cls.spec)
        cls.ev = cls.clean[3]

    def test_spec(self):
        self.assertEqual([s["kind"] for s in self.spec["stages"]], ["stand", "squat", "jump", "land", "stand"])
        # the hype doesn't change what the task is
        hyped = T.compile_spec(synthetic(hype=True)[1], FPS)
        self.assertEqual([s["kind"] for s in hyped["stages"]], [s["kind"] for s in self.spec["stages"]])

    def test_hype_frames_low_takeoff_landing_high(self):
        w, body = self.hype[0], self.hype[1]
        a, b = int(HYPE[0] * FPS), int(HYPE[1] * FPS)
        self.assertLess(w[a:b].max(), I.FLOOR + 0.02)
        arms = [I.GROUPS.index("left_arm"), I.GROUPS.index("right_arm")]
        self.assertLess(body[a:b][:, arms].max(), I.FLOOR + 0.02)
        to = next(e for e in self.ev if e["kind"] == "propulsion")
        ab = next(e for e in self.ev if e["kind"] == "absorb")
        self.assertGreater(w[to["window"][0]:to["frame"] + 1].max(), 0.8)
        self.assertGreater(w[ab["window"][0]:ab["frame"] + 1].max(), 0.8)
        self.assertGreater(w[to["window"][0]:to["frame"] + 1].mean(), 0.5)  # the whole push
        # the hype changes nothing outside its window
        np.testing.assert_allclose(self.hype[0][b + 10:], self.clean[0][b + 10:], atol=1e-6)

    def test_arm_swing_credited_only_when_it_helps(self):
        share = lambda s: next(e for e in s if e["kind"] == "propulsion")["group_share"]
        with_swing = share(self.clean[2])
        still = share(run(synthetic(swing=False)[0], self.spec)[2])
        arms = lambda s: s["left_arm"] + s["right_arm"]
        self.assertGreater(arms(with_swing), 0.06)  # arm swing gives ~10% of jump height in people
        self.assertLess(arms(still), 0.01)
        # sideways waving during the jump adds (almost) no credit to the arms: it cancels out
        lateral = share(run(synthetic(swing=False, lateral=True)[0], self.spec)[2])
        self.assertLess(arms(lateral), 0.01)
        self.assertGreater(lateral["left_leg"] + lateral["right_leg"], 0.9)

    def test_completeness(self):
        """Descent credit sums to the drop of the CoM (all stance: the pelvis is the legs' doing)."""
        kin = synthetic()[0]
        q = I.group_quantities(kin, FPS)
        ev = next(e for e in self.ev if e["kind"] == "descent")
        a, b = ev["window"]
        cred = I.event_credit(q, ev, FPS)[a:b].sum()
        from scipy.ndimage import gaussian_filter1d
        com = (gaussian_filter1d(kin["com"], 1.5, axis=0, mode="nearest") * kin["mass"][None, :, None]).sum(1) / kin["mass"].sum()
        self.assertAlmostEqual(cred, com[a - 1, 2] - com[b - 1, 2], places=4)

    def test_bounds(self):
        w, body = self.clean[0], self.clean[1]
        self.assertGreaterEqual(w.min(), I.FLOOR - 1e-9)
        self.assertLessEqual(w.max(), 1 + 1e-9)
        self.assertTrue(np.all(body <= w[:, None] + 1e-9))

    def test_hype_reproduction(self):
        q = np.zeros((90, 36))
        h = I.inject_hype(q, 0.5, 2.5)
        self.assertAlmostEqual(I.hype_reproduction(h, h, q, (0.5, 2.5)), 1.0)
        self.assertAlmostEqual(I.hype_reproduction(q, h, q, (0.5, 2.5)), 0.0)
        np.testing.assert_array_equal(h[:15, 22:], q[:15, 22:])  # nothing before it starts
        np.testing.assert_array_equal(h[:, :22], q[:, :22])     # arms only


def _g1_available():
    try:
        import config
        return config.G1_SCENE_XML.exists()
    except Exception:
        return False


@unittest.skipUnless(_g1_available() and (CLIP / "qpos.npy").exists(), "needs the G1 assets and the squat_jump clip")
class TestSquatJumpRetarget(unittest.TestCase):
    def test_hype_on_retarget(self):
        import render
        model = render.load_g1()[0]
        q = np.load(CLIP / "qpos.npy")
        kin = I.kinematics(model, q)
        spec = T.compile_spec(kin["pos"], FPS, name="squat_jump")
        res = I.analyse(model, I.inject_hype(q, *HYPE), spec, FPS)
        clean = I.analyse(model, q, spec, FPS)
        w = np.array(res["weight"])
        a, b = int(HYPE[0] * FPS), int(HYPE[1] * FPS)
        self.assertLess(w[a:b].max(), I.FLOOR + 0.02)
        for g in ("left_arm", "right_arm"):
            self.assertLess(max(res["body_weight"][g][a:b]), I.FLOOR + 0.02)
        for e in res["events"]:
            if e["kind"] in ("propulsion", "absorb"):
                lo, hi = (int(x * FPS) for x in e["window_s"])
                self.assertGreater(w[lo:hi].max(), 0.8, e)
        np.testing.assert_allclose(w[b + 10:], clean["weight"][b + 10:], atol=1e-3)
        new = T.compile_spec(I.kinematics(model, I.inject_hype(q, *HYPE))["pos"], FPS)
        self.assertEqual(new["summary"], spec["summary"])


@unittest.skipUnless(importlib.util.find_spec("transformers") and _g1_available() and (CLIP / "qpos.npy").exists(),
                     "needs transformers, the G1 assets and the squat_jump clip")
class TestPredictorPlumbing(unittest.TestCase):
    def test_tiny_vjepa(self):
        import torch
        from transformers import VJEPA2Config, VJEPA2Model
        import render
        from latent import predictor_relevance as pr
        torch.manual_seed(0)
        cfg = VJEPA2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, pred_hidden_size=16,
                           pred_num_hidden_layers=1, pred_num_attention_heads=2, crop_size=64, frames_per_clip=16)
        pre = lambda x: (x[..., ::4, ::4].float() / 255 - 0.45) / 0.225
        P = pr.Predictor(model=VJEPA2Model(cfg), preprocess=pre, device="cpu", batch=13)
        model = render.load_g1()[0]
        q = np.load(CLIP / "qpos.npy")[:120]
        ev = [{"kind": "propulsion", "frame": 85, "window": (70, 86), "stage": 2, "label": "jump"}]
        S, d = pr.scores(model, q, ev, FPS, predictor=P, context=16, occlude=8, log=lambda *_: None)
        self.assertEqual(S.shape, (120, len(I.GROUPS)))
        self.assertTrue(np.isfinite(S).all() and (S >= 0).all())
        self.assertEqual(len(d["rows"]), 2 * 6)  # 2 windows x (all + 5 groups)
        self.assertTrue((S[:69] == 0).all() and (S[86:] == 0).all())  # only the context is credited
        S2, _ = pr.scores(model, q, ev, FPS, predictor=P, context=16, occlude=8, log=lambda *_: None, mode="encoder")
        self.assertTrue(np.isfinite(S2).all() and (S2 >= 0).all() and (S2[:69] == 0).all())
        # mixed into the weights only on request
        phys = np.ones((120, len(I.GROUPS)))
        np.testing.assert_array_equal(I.combine(phys, S2)[0], I.combine(phys)[0])
        # removing motion keeps the window's end points
        q2 = pr.remove_motion(q, 10, 18, pr.GROUP_COLS["left_arm"])
        np.testing.assert_array_equal(q2[18:], q[18:])
        np.testing.assert_array_equal(q2[:, :22], q[:, :22])


if __name__ == "__main__":
    unittest.main()
