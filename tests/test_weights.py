"""Saved weights: policy_weights.py adapts checkpoints across tasks; train_motion.py resumes an
interrupted attempt and passes --init; latent_policy.py picks which checkpoint a policy builds off.

    python -m unittest tests.test_weights
"""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PIPELINE = all(importlib.util.find_spec(m) for m in ("torch", "mujoco", "scipy", "cv2"))


@unittest.skipUnless(HAS_TORCH, "needs torch")
class AdaptTest(unittest.TestCase):
    def test_grown_inputs_keep_old_columns_and_start_silent(self):
        import torch
        from policy_weights import adapt_state_dict
        src = {"obs_normalizer._mean": torch.ones(1, 3), "obs_normalizer._std": 2 * torch.ones(1, 3),
               "mlp.0.weight": torch.ones(4, 3), "mlp.2.weight": torch.full((2, 4), 3.0), "std": torch.ones(2)}
        dst = {"obs_normalizer._mean": torch.zeros(1, 5), "obs_normalizer._std": torch.zeros(1, 5),
               "mlp.0.weight": torch.randn(4, 5), "mlp.2.weight": torch.randn(2, 4), "std": torch.zeros(2),
               "new.bias": torch.full((2,), 7.0)}
        out, how = adapt_state_dict(src, dst)
        self.assertTrue(torch.equal(out["mlp.0.weight"][:, :3], torch.ones(4, 3)))
        self.assertTrue(torch.equal(out["mlp.0.weight"][:, 3:], torch.zeros(4, 2)))  # new inputs: no effect
        self.assertTrue(torch.equal(out["obs_normalizer._std"], torch.tensor([[2.0, 2, 2, 1, 1]])))
        self.assertTrue(torch.equal(out["obs_normalizer._mean"], torch.tensor([[1.0, 1, 1, 0, 0]])))
        self.assertTrue(torch.equal(out["mlp.2.weight"], src["mlp.2.weight"]))
        self.assertEqual(how["new.bias"], "fresh (not in source)")
        _, how = adapt_state_dict({"mlp.0.weight": torch.ones(4, 6)}, {"mlp.0.weight": torch.zeros(4, 5)})
        self.assertTrue(how["mlp.0.weight"].startswith("fresh"))  # fewer inputs: can't map, re-init


class TrainMotionStartTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def train(self, before, **kw):
        import train_motion
        calls = []
        ckpt = f"{self.tmp}/out/logs/a/b/model_1499.pt"
        with mock.patch.object(train_motion, "run", lambda cmd, **k: calls.append((cmd, k))), \
                mock.patch.object(train_motion.shutil, "copy"), \
                mock.patch.object(train_motion.glob, "glob", lambda p: before if len(calls) < 2 else [ckpt]), \
                mock.patch.object(train_motion, "open", mock.mock_open(), create=True), \
                mock.patch.object(train_motion.fcntl, "flock"):
            train_motion.train_motion(self.tmp / "m.csv", 30, self.tmp / "out", iters=2000, **kw)
        return calls[1], json.loads((self.tmp / "out" / "weights.json").read_text())

    def test_interrupted_attempt_resumes(self):
        (cmd, kw), meta = self.train([f"{self.tmp}/out/logs/x/y/model_500.pt", f"{self.tmp}/out/logs/x/y/model_1000.pt"])
        self.assertEqual(kw["env"]["A2H_INIT_MODE"], "resume")
        self.assertTrue(kw["env"]["A2H_INIT_CHECKPOINT"].endswith("model_1000.pt"))
        self.assertEqual(cmd[cmd.index("--agent.max-iterations") + 1], 1000)  # the rest of the 2000
        self.assertTrue(str(cmd[1]).endswith("mjlab_tasks.py"))  # warm start needs our entry point
        self.assertEqual(meta["started_from"]["iteration"], 1000)

    def test_init_from_other_weights(self):
        (cmd, kw), meta = self.train([], init=self.tmp / "w" / "model_1999.pt", task="A2H-Task-Flat-Unitree-G1")
        self.assertEqual(kw["env"]["A2H_INIT_MODE"], "init")
        self.assertEqual(cmd[cmd.index("--agent.max-iterations") + 1], 2000)
        self.assertEqual(meta["task"], "A2H-Task-Flat-Unitree-G1")
        self.assertEqual(meta["iterations"], 1500)

    def test_fresh_tracking_uses_mjlab_train(self):
        (cmd, kw), meta = self.train([])
        self.assertNotIn("A2H_INIT_CHECKPOINT", kw["env"])
        self.assertTrue(str(cmd[0]).endswith("train"))
        self.assertIsNone(meta["started_from"])


@unittest.skipUnless(HAS_PIPELINE, "needs the pipeline's dependencies")
class PickCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        for pol, its in (("hybrid", (999, 1999)), ("phase", (2999,))):
            (self.tmp / "run" / f"policy_{pol}").mkdir(parents=True)
            for i in its:
                (self.tmp / "run" / f"policy_{pol}" / f"model_{i}.pt").write_text("x")

    def test_init_resolution(self):
        from experiments.latent_policy import init_checkpoint
        run = str(self.tmp / "run")
        self.assertEqual(init_checkpoint("hybrid", [run]).name, "model_1999.pt")
        self.assertEqual(init_checkpoint("task", [run]).parent.name, "policy_hybrid")  # fallback
        self.assertEqual(init_checkpoint("taskphase", [run]).parent.name, "policy_phase")
        self.assertEqual(init_checkpoint("taskcmd", [run]).parent.name, "policy_phase")  # 2nd fallback
        (self.tmp / "run" / "policy_taskphase").mkdir()
        (self.tmp / "run" / "policy_taskphase" / "model_1499.pt").write_text("x")
        self.assertEqual(init_checkpoint("taskcmd", [run]).parent.name, "policy_taskphase")
        self.assertIsNone(init_checkpoint("latent", [run]))
        self.assertEqual(init_checkpoint("task", [run, "task=/w/model_5.pt"]), Path("/w/model_5.pt"))

    def test_mirror_lists_newest_settled_checkpoint(self):
        import os
        import time
        from experiments.latent_policy import mirror_checkpoints
        out = self.tmp / "out"
        logs = out / "policy_task" / "logs" / "e" / "r"
        logs.mkdir(parents=True)
        for i in (250, 500):
            (logs / f"model_{i}.pt").write_text(str(i))
            os.utime(logs / f"model_{i}.pt", (time.time() - 60,) * 2)
        (logs / "model_750.pt").write_text("still being written")  # too fresh to copy
        self.assertEqual(mirror_checkpoints(out), {"policy_task": "policy_task/latest/model_500.pt"})
        os.utime(logs / "model_750.pt", (time.time() - 60,) * 2)
        mirror_checkpoints(out)
        self.assertEqual([p.name for p in (out / "policy_task" / "latest").iterdir()], ["model_750.pt"])
        self.assertEqual(json.loads((out / "checkpoints.json").read_text())["policy_task"], "policy_task/latest/model_750.pt")

    def test_resume_seeds_the_log_folder(self):
        from experiments.latent_policy import seed_checkpoint
        d = self.tmp / "new" / "policy_hybrid"
        d.mkdir(parents=True)
        seed_checkpoint(d, "hybrid", self.tmp / "run")
        self.assertEqual([p.name for p in d.glob("logs/*/*/model_*.pt")], ["model_1999.pt"])
        seed_checkpoint(d, "phase", None)  # nothing to resume from: no-op


class SyncTest(unittest.TestCase):
    def test_sync_downloads_listed_checkpoints_once(self):
        from types import SimpleNamespace
        from scripts import remote_run as rr
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        calls = []

        def colab(*args, timeout=600, check=True):
            calls.append(args)
            remote, local = args[3], Path(args[4])
            local.write_text(json.dumps({"policy_task": "policy_task/latest/model_500.pt"})
                             if remote.endswith("checkpoints.json") else "w")
        with mock.patch.object(rr, "colab", colab), mock.patch.object(rr, "OUT", tmp), \
                mock.patch.object(rr, "SESSION", "latent_x"), mock.patch.object(rr, "log"):
            a = SimpleNamespace(job="latent", smoke=False, tag="x", target="squat")
            rr.sync_checkpoints(a)
            rr.sync_checkpoints(a)  # already there: only the index is fetched again
        self.assertEqual([p.name for p in (tmp / "checkpoints" / "policy_task").iterdir()], ["model_500.pt"])
        self.assertEqual(sum(c[3].endswith("model_500.pt") for c in calls), 1)
        self.assertTrue(all(c[0] == "download" for c in calls))  # never exec: the kernel is busy


if __name__ == "__main__":
    unittest.main()
