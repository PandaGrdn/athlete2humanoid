"""The latent reward's uncertainty penalty (latent/reward_net.py, latent/distill.py ensembles,
mjlab_tasks.LatentReward): MC dropout and ensemble members, the bundle format (old bundles still
load), and, with mjlab, that a robot exactly on the reference pays no penalty.

    python -m unittest tests.test_latent_uncertainty
"""

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if all(importlib.util.find_spec(m) for m in ("numpy", "torch", "scipy")):
    import numpy as np
    import torch

    from latent import reward_net as rn
else:
    raise unittest.SkipTest("needs numpy, torch and scipy")

HAS_DISTILL = all(importlib.util.find_spec(m) for m in ("mujoco", "cv2"))
HAS_MJLAB = importlib.util.find_spec("mjlab") is not None
B = len(rn.BODIES)


def toy_data(n_motions=12, per=40, seed=0):
    """Windows of bodies bobbing at random heights and a latent that is a smooth function of
    each body's mean height in the window."""
    g = np.random.default_rng(seed)
    h = g.uniform(0.2, 1.2, (n_motions * per, 1, B, 1))
    X = np.broadcast_to(h, (len(h), rn.WINDOW, B, 3)).copy().astype(np.float32)
    X[..., :2] = g.normal(0, 0.05, X[..., :2].shape)
    X[..., 2] += 0.02 * np.sin(np.arange(rn.WINDOW) / 3.0)[None, :, None]
    W = g.normal(size=(B, 32)).astype(np.float32)
    Y = np.tanh(X[:, :, :, 2].mean(1) @ W)
    return X, Y, np.repeat(np.arange(n_motions), per)


class MemberPredictionTest(unittest.TestCase):
    def test_mc_dropout_and_ensembles(self):
        torch.manual_seed(0)
        net = rn.LatentRewardNet(k=8, hidden=32, depth=2).eval()
        win = torch.randn(5, rn.WINDOW, B, 3)
        p = rn.member_predictions(net, win, mc=6)
        self.assertEqual(tuple(p.shape), (6, 5, 8))
        self.assertGreater(float(rn.uncertainty(p.detach()).min()), 0)
        self.assertFalse(any(m.training for m in net.modules()))  # dropout back off
        self.assertTrue(torch.equal(net(win), net(win)))
        self.assertEqual(tuple(rn.member_predictions(net, win, mc=1).shape), (1, 5, 8))
        ens = rn.LatentRewardEnsemble([rn.LatentRewardNet(k=8, hidden=32, depth=2).eval() for _ in range(3)])
        m = rn.member_predictions(ens, win, mc=6)
        self.assertEqual(tuple(m.shape), (3, 5, 8))
        self.assertTrue(torch.allclose(ens(win), m.mean(0)))

    def test_uncertainty_is_relative_spread(self):
        p = torch.randn(1, 4, 8).expand(5, -1, -1)
        self.assertTrue(torch.allclose(rn.uncertainty(p), torch.zeros(4)))
        q = torch.randn(5, 4, 8)
        self.assertTrue(torch.allclose(rn.uncertainty(q), rn.uncertainty(10 * q), atol=1e-5))

    def test_reference_windows(self):
        T, dt = 100, 0.02
        pos = torch.arange(T, dtype=torch.float32)[:, None, None].expand(T, B, 3) * dt  # x = time
        w = rn.reference_windows(pos, torch.tensor([0, 10, 99]), dt)
        self.assertEqual(tuple(w.shape), (3, rn.WINDOW, B, 3))
        times = (torch.tensor([0, 10, 99])[:, None] * dt - (rn.WINDOW - 1 - torch.arange(rn.WINDOW)) / rn.FPS).clamp(min=0)
        self.assertTrue(torch.allclose(w[..., 0, 0], times, atol=1e-5))  # 30 fps frames ending now


@unittest.skipUnless(HAS_DISTILL, "needs the pipeline's dependencies (latent.distill)")
class EnsembleTrainingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_ensemble_is_unsure_off_distribution_and_round_trips(self):
        from latent import distill
        X, Y, seq = toy_data()
        r_mean = Y.mean(0)
        torch.manual_seed(0)
        net, U, rep = distill.train_reward_net(X, Y, seq, r_mean, k=8, epochs=40, hidden=64, depth=2,
                                               device="cpu", members=3)
        self.assertIsInstance(net, rn.LatentRewardEnsemble)
        self.assertEqual(rep["members"], 3)
        Xt = torch.as_tensor(X[:200])
        far = Xt.clone()
        far[..., 2] += 2.0  # heights the members never saw
        with torch.no_grad():
            self.assertGreater(float(rn.uncertainty(net.members(far)).mean()),
                               1.5 * float(rn.uncertainty(net.members(Xt)).mean()))
        tgt = np.zeros((10, 8), np.float32)
        distill.save_bundle(self.tmp / "e.pt", net, U, r_mean, r_mean, tgt, np.ones(10, np.float32), 30.0, rep, -150)
        net2, _, b = rn.load_bundle(self.tmp / "e.pt")
        self.assertIsInstance(net2, rn.LatentRewardEnsemble)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net2(Xt[:5]), net(Xt[:5]), atol=1e-5))
        # a bundle from before ensembles (no "members" key) loads as one network
        old = torch.load(self.tmp / "e.pt", weights_only=False)
        old.pop("members")
        torch.save(old, self.tmp / "old.pt")
        self.assertIsInstance(rn.load_bundle(self.tmp / "old.pt")[0], rn.LatentRewardNet)

    def test_one_member_is_the_old_training(self):
        from latent import distill
        X, Y, seq = toy_data(6, 20)
        torch.manual_seed(1)
        net, _, rep = distill.train_reward_net(X, Y, seq, Y.mean(0), k=4, epochs=2, hidden=32, depth=2, device="cpu")
        self.assertIsInstance(net, rn.LatentRewardNet)
        self.assertNotIn("members", rep)


@unittest.skipUnless(HAS_MJLAB and HAS_DISTILL, "needs mjlab")
class LatentRewardPenaltyTest(unittest.TestCase):
    """mjlab_tasks.LatentReward on a fake env: on the reference the penalty is exactly zero."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_on_reference_no_penalty(self):
        from latent import distill
        import mjlab_tasks as mt
        torch.manual_seed(0)
        ens = rn.LatentRewardEnsemble([rn.LatentRewardNet(k=8, hidden=32, depth=2).eval() for _ in range(3)])
        ens.kwargs = {"k": 8, "hidden": 32, "depth": 2, "features": 1}
        T = 120
        tgt = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
        distill.save_bundle(self.tmp / "b.pt", ens, np.eye(8, dtype=np.float32), np.zeros(8), np.zeros(8), tgt,
                            np.full(40, 0.5, np.float32), 30.0, {}, -150)
        ref = torch.cumsum(0.003 * torch.randn(T, B, 3), 0) + torch.tensor([0, 0, 0.8])
        ts = torch.tensor([3, 40, 90])
        cmd = SimpleNamespace(cfg=SimpleNamespace(body_names=rn.BODIES), time_steps=ts.clone(),
                              motion=SimpleNamespace(body_pos_w=ref, time_step_total=T),
                              robot_body_pos_w=ref[ts].clone())
        env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda n: cmd), device="cpu", num_envs=3,
                              step_dt=0.02, scene=SimpleNamespace(env_origins=torch.zeros(3, 3)))
        with mock.patch.dict(os.environ, {"A2H_LATENT_BUNDLE": str(self.tmp / "b.pt")}):
            plain = mt.LatentReward(SimpleNamespace(params={"sigma": 0.1}), env)
            pen = mt.LatentReward(SimpleNamespace(params={"sigma": 0.1, "uncertainty": {"lambda": 4.0}}), env)
        self.assertEqual(tuple(pen.u_ref.shape), (T,))
        self.assertTrue(torch.allclose(plain(env), pen(env), atol=1e-6))  # first call: history = reference
        cmd.time_steps += 1
        cmd.robot_body_pos_w = ref[cmd.time_steps] + torch.tensor([0.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(plain(env), pen(env), atol=1e-6))  # still on it
        cmd.time_steps += 1
        cmd.robot_body_pos_w = ref[cmd.time_steps] + 0.5 * torch.randn(3, B, 3)  # far off it
        self.assertTrue((pen(env) <= plain(env) + 1e-6).all())
        # under goal-importance weighting the inner term keeps its own settings
        imp = self.tmp / "imp.json"
        imp.write_text('{"fps": 30.0, "weight": [1.0]}')
        with mock.patch.dict(os.environ, {"A2H_LATENT_BUNDLE": str(self.tmp / "b.pt"), "A2H_IMPORTANCE": str(imp)}):
            tw = mt.TimeWeighted(SimpleNamespace(weight=1.0, params={
                "inner": mt.LatentReward, "inner_params": {"sigma": 0.1, "uncertainty": {"lambda": 4.0}}}), env)
        self.assertEqual(tw.inner.lam, 4.0)


if __name__ == "__main__":
    unittest.main()
