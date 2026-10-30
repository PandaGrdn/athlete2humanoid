"""experiments/robustness_eval.py: the joint-space DTW, the condition list (v1's 13 first), the summary
writer, and that a task-first policy refuses to load without its task spec. When mjlab is installed
(its CPU env is enough) and the local run folders exist, the mjlab stage itself, on few envs and
short horizons: a task + taskphase run (different observations, 178 / 119 dims) and a v1 run.

    python -m unittest tests.test_robustness_eval
    A2H_TEST_OUTPUTS=<folder with remote_latent/, remote_latent_sj_task/> python -m unittest ...
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if not all(importlib.util.find_spec(m) for m in ("numpy", "torch")):  # e.g. a bare system Python
    raise unittest.SkipTest("needs numpy and torch")

import numpy as np
import torch

from experiments import robustness_eval as R

REPO = Path(__file__).resolve().parents[1]
# A worktree (<repo>/.claude/worktrees/<name>) has no outputs/ of its own: use the main checkout's.
OUTPUTS = Path(os.environ.get("A2H_TEST_OUTPUTS", next(
    (p for p in (REPO / "outputs", REPO.parents[2] / "outputs") if any(p.glob("remote_latent*"))), REPO / "outputs")))
SJ_TASK, V1 = OUTPUTS / "remote_latent_sj_task", OUTPUTS / "remote_latent"
HAS_MJLAB = importlib.util.find_spec("mjlab") is not None
V1_CONDITIONS = [("clean", "nominal"), ("push", "0.5 m/s"), ("push", "0.75 m/s"), ("push", "1 m/s"),
                 ("push", "1.5 m/s"), ("init", "x1"), ("init", "x2"), ("init", "x4"), ("dynamics", "level 1"),
                 ("dynamics", "level 2"), ("dynamics", "level 3"), ("speed", "0.8x"), ("speed", "1.2x")]


def naive_dtw(a, b):
    D = np.full((len(a) + 1, len(b) + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = np.abs(a[i - 1] - b[j - 1]).mean() + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[-1, -1] / ((len(a) + len(b)) / 2)


class TestDtw(unittest.TestCase):
    def test_matches_naive_dtw_with_ragged_lengths(self):
        rng = np.random.default_rng(0)
        q, ref = rng.normal(size=(3, 12, 4)), rng.normal(size=(3, 15, 4))
        nq, nref = np.array([12, 7, 9]), np.array([15, 10, 4])
        got = R.dtw_joint(torch.tensor(q, dtype=torch.float32), torch.tensor(nq), torch.tensor(ref, dtype=torch.float32),
                          torch.tensor(nref))
        want = [naive_dtw(q[b, :nq[b]], ref[b, :nref[b]]) for b in range(3)]
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-5)

    def test_timing_is_free_poses_are_not(self):
        t = np.linspace(0, 2 * np.pi, 200)
        clip = np.stack([np.sin(t), np.cos(2 * t)], -1)
        u = np.linspace(0, 2 * np.pi, 250)  # the same motion at 0.8x
        slow = np.stack([np.sin(u), np.cos(2 * u)], -1)
        other = clip * 0.5

        def d(x):
            return R.dtw_joint(torch.tensor(x[None], dtype=torch.float32), torch.tensor([len(x)]),
                               torch.tensor(clip[None], dtype=torch.float32), torch.tensor([len(clip)])).item()

        self.assertAlmostEqual(d(clip), 0.0, places=6)
        locked = np.abs(slow[:200] - clip).mean()
        self.assertLess(d(slow), 0.05 * locked)  # same poses, other tempo: ~0, time-locked error is large
        self.assertGreater(d(other), 0.2)


class TestConditions(unittest.TestCase):
    def test_v1_conditions_first_then_the_task_ones(self):
        a = R.parse(["--runs", "x"])
        c = R.conditions(a)
        self.assertEqual([(s, lv) for s, lv, _ in c[:13]], V1_CONDITIONS)
        self.assertEqual([s for s, _, _ in c[13:]], ["push_phase", "push_phase", "start", "start", "noref"] + ["tempo"] * 3)
        noref = next(p for s, _, p in c if s == "noref")
        self.assertEqual((noref["start"], noref["reference"]), ("standing", "frozen"))
        self.assertEqual({p["push_at"] for _, _, p in c[13:15]}, {"squat", "land"})
        self.assertEqual(len(R.conditions(R.parse(["--runs", "x", "--phase-push", "--starts", "--no-noref", "--tempo"]))), 13)

    def test_frozen_and_resampled_motion(self):
        with tempfile.TemporaryDirectory() as d:
            T = 20
            src = Path(d) / "m.npz"
            rng = np.random.default_rng(1)
            np.savez(src, fps=np.array([50.0]), joint_pos=rng.normal(size=(T, 3)), joint_vel=rng.normal(size=(T, 3)),
                     body_pos_w=rng.normal(size=(T, 2, 3)), body_quat_w=np.tile([1.0, 0, 0, 0], (T, 2, 1)),
                     body_lin_vel_w=rng.normal(size=(T, 2, 3)), body_ang_vel_w=rng.normal(size=(T, 2, 3)))
            f = np.load(R.frozen_motion(src, Path(d) / "f.npz"))
            self.assertEqual(f["joint_pos"].shape, (T, 3))
            np.testing.assert_allclose(f["joint_pos"], np.repeat(f["joint_pos"][:1], T, 0))
            self.assertEqual(np.abs(f["joint_vel"]).max(), 0)
            self.assertEqual(len(np.load(R.resample_motion(src, Path(d) / "s.npz", 1.25))["joint_pos"]), 16)


class TestTaskSpecRequired(unittest.TestCase):
    pols = {"tracking": {"task": "Mjlab-Tracking-Flat-Unitree-G1"}, "task": {"task": "A2H-Task-Flat-Unitree-G1"},
            "taskphase": {"task": "A2H-TaskPhase-Flat-Unitree-G1"}}

    def setUp(self):
        self._env = os.environ.pop(R.TASK_SPEC_ENV, None)

    def tearDown(self):
        os.environ.pop(R.TASK_SPEC_ENV, None)
        if self._env is not None:
            os.environ[R.TASK_SPEC_ENV] = self._env

    def test_task_policy_without_spec_fails_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as e:
                R.task_spec_path(Path(d), self.pols)
            msg = str(e.exception)
            for s in ("task", "taskphase", "A2H_TASK_SPEC", "task_spec.json"):
                self.assertIn(s, msg)
            os.environ[R.TASK_SPEC_ENV] = str(Path(d) / "missing.json")
            with self.assertRaisesRegex(SystemExit, "not a file"):
                R.task_spec_path(Path(d), self.pols)

    def test_spec_from_run_folder_or_env(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(R.task_spec_path(Path(d), {"tracking": self.pols["tracking"]}))  # v1: compiled later
            spec = Path(d) / "elsewhere.json"
            spec.write_text("{}")
            os.environ[R.TASK_SPEC_ENV] = str(spec)
            self.assertEqual(R.task_spec_path(Path(d), self.pols), str(spec.resolve()))
            (Path(d) / "task_spec.json").write_text("{}")
            (Path(d) / "reward.pt").write_text("")
            env = R.run_env(Path(d), self.pols)  # the run folder's own spec wins
            self.assertEqual(env[R.TASK_SPEC_ENV], str((Path(d) / "task_spec.json").resolve()))
            self.assertEqual(env[R.LATENT_BUNDLE_ENV], str((Path(d) / "reward.pt").resolve()))
            self.assertNotIn(R.IMPORTANCE_ENV, env)


def fake_result(task_metrics=True):
    a = R.parse(["--runs", "x"])
    conds = []
    for suite, level, p in R.conditions(a):
        pol = {}
        for n, base in (("tracking", 0.9), ("taskphase", 0.8)):
            r = {"fall_rate": 0.1, "joint_err_rad": 0.1, "dtw_joint_rad": 0.05 if n == "tracking" else 0.15,
                 "vjepa_aligned": 0.7 if suite == "clean" else None}
            if task_metrics:
                r["task_success"] = 0.0 if (suite == "noref" and n == "tracking") else base
            pol[n] = r
        conds.append({"suite": suite, "level": level, "params": p, "policies": pol})
    meta = {"target": "squat_jump", "reps": 64, "fall_rule": {"torso_z_m": 0.35, "tilt_rad": 1.4},
            "dtw_joint_rad_still": 0.23,
            **({"task_spec": {"source": "/x/task_spec.json", "summary": "2 jumps: stand -> squat"}} if task_metrics else {})}
    return {"meta": meta, "conditions": conds, "summary": R.summarise(conds)}


class TestSummary(unittest.TestCase):
    def test_table(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(R.write_summary(fake_result(), Path(d) / "robust_summary.md")).read_text()
        rows = {line.split(" | ")[0].strip("| "): line for line in md.splitlines() if line.startswith("| ")}
        self.assertIn("No reference", rows["Policy"])
        self.assertIn("| 0% (10% fall) | 0.050 |", rows["tracking"])  # no reference -> fails; DTW to clip
        self.assertTrue(rows["taskphase"].startswith("| taskphase | 80% (10% fall)"))
        self.assertIn("standing still scores 0.230", md)
        self.assertIn("| noref | frozen | 0% / 10% fall / dtw 0.050 |", md)

    def test_v1_robust_json_without_task_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            md = Path(R.write_summary(fake_result(task_metrics=False), Path(d) / "s.md")).read_text()
        self.assertIn("predates the task metrics", md)
        self.assertIn("| tracking | 10% fall |", md)
        self.assertEqual(R.summarise(fake_result(False)["conditions"])["tracking"]["clean"]["task_success"], None)


class TestCalibrationAndDiagnostics(unittest.TestCase):
    """The clip's tracker calibration is the best tracker that stayed up; where unsuccessful rollouts stop is
    counted per stage with the feature furthest outside its tolerance; robust_summary.md prints both."""

    def setUp(self):
        from skill import keypose as K
        self.K = K
        rms = lambda x: dict(zip([f[0] for f in K.FEATURES], [x] * K.D))  # noqa: E731
        self.res = {"good": {"fall_rate": 0.0, "joint_err_rad": 0.08, "torso_err_m": 0.03, "feature_rms_m": rms(0.07),
                             "tracking_rms_m": {"pelvis": 0.03}},
                    "better_but_fell": {"fall_rate": 0.1, "joint_err_rad": 0.05, "torso_err_m": 0.03,
                                        "feature_rms_m": rms(0.01), "tracking_rms_m": {}},
                    "poor": {"fall_rate": 0.0, "joint_err_rad": 0.2, "torso_err_m": 0.1, "feature_rms_m": rms(0.3),
                             "tracking_rms_m": {}}}
        self.pols = {n: {"checkpoint": f"/x/policy_{n}/model_1.pt"} for n in self.res}

    def test_calibration_is_the_best_tracker_that_stayed_up(self):
        cal = R.tracker_calibration(self.res, "best", self.pols, "punches")
        self.assertEqual((cal["policy"], cal["clip"], cal["checkpoint"]), ("good", "punches", "model_1.pt"))
        wrist = cal["features"].index("left wrist z")
        self.assertAlmostEqual(cal["floor_m"][wrist], 0.14)  # 2 x 7 cm, above the 10 cm TOL_MIN
        self.assertAlmostEqual(cal["floor_m"][0], 2 * 0.04)  # pelvis: 2 x 7 cm capped at CALIB_MAX x 4 cm
        self.assertEqual(R.tracker_calibration(self.res, "poor", self.pols, "punches")["policy"], "poor")
        self.assertIsNone(R.tracker_calibration({"x": {**self.res["good"], "fall_rate": 0.5}}, "best", self.pols, "p"))

    def test_where_rollouts_stop(self):
        K = self.K
        spec = {"features": [f[0] for f in K.FEATURES],
                "stages": [{"kind": "settle", "name": "ready"}, {"kind": "pose", "name": "punch"}, {"kind": "settle", "name": "finish"}]}
        n = 8
        failed = torch.tensor([True] * 6 + [False] * 2)
        fell = torch.tensor([False] * 5 + [True] + [False] * 2)
        stage = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
        x = torch.zeros(n, K.D)
        wy = spec["features"].index("right wrist y")
        x[:4, wy] = torch.tensor([1.6, 1.4, 1.5, 0.5])
        rms = torch.tensor([1.2, 1.1, 1.3, 0.4, float("inf"), float("inf"), 0.1, 0.1])
        rows = R.abandoned(spec, failed, fell, stage, x, rms)
        self.assertEqual([r["name"] for r in rows], ["punch", "finish"])
        punch, finish = rows
        self.assertEqual((punch["share"], punch["fell"], punch["feature"], punch["error_tol"]), (0.5, 0.0, "right wrist y", 1.4))  # the lower median
        self.assertEqual(punch["reached"], 0.25)  # one of the four came within tolerance and didn't hold it
        self.assertEqual((finish["share"], finish["fell"]), (0.25, 0.5))
        self.assertNotIn("feature", finish)

    def test_summary_prints_calibration_stops_and_tracking_error(self):
        result = fake_result()
        clean = result["conditions"][0]["policies"]["tracking"]
        clean["abandoned"] = [{"stage": 6, "name": "turn left, step forward (finish)", "kind": "pose", "share": 1.0,
                               "fell": 0.0, "feature": "right foot height", "error_tol": 1.71, "reached": 0.0}]
        clean["tracking_rms_m"] = {"pelvis": 0.034, "left_wrist_yaw_link": 0.066}
        result["meta"]["calibration"] = {"k": 2.0, "policy": "scratch", "joint_err_rad": 0.079}
        result["meta"]["uncalibrated_clean"] = {"tracking": {"task_success": 0.0}, "taskphase": {"task_success": 0.5}}
        with tempfile.TemporaryDirectory() as d:
            md = Path(R.write_summary(result, Path(d) / "s.md")).read_text()
        self.assertIn("2 x the RMS error of scratch (joint err 0.079 rad)", md)
        self.assertIn("clean success before it tracking 0%, taskphase 50%", md)
        self.assertIn("| tracking | 6: turn left, step forward (finish) | 100% | 0% | right foot height | 1.71 | 0% |", md)
        self.assertIn("| tracking | 3.4 | - | 6.6 |", md)


@unittest.skipUnless(HAS_MJLAB, "needs mjlab (its CPU env is enough)")
class TestMjlabStage(unittest.TestCase):
    """The real thing on CPU, tiny: one env build per run, every suite once, 30 steps."""

    def run_eval(self, runs, out, *extra, env=None):
        cmd = [sys.executable, str(REPO / "experiments/robustness_eval.py"), "--runs", str(runs), "--out", str(out),
               "--reps", "2", "--score", "2", "--max-steps", "30", "--no-vjepa", "--push", "1.0", "--init", "2",
               "--dynamics", "1", "--speed", "1.2", *extra]
        env = dict(os.environ, A2H_MJLAB_PYTHON=sys.executable, MUJOCO_GL="glfw" if sys.platform == "darwin" else "egl",
                   **(env or {}))
        env.pop(R.TASK_SPEC_ENV, None)
        return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO)

    @unittest.skipUnless((SJ_TASK / "policy_task").is_dir(), f"no {SJ_TASK}")
    def test_task_and_taskphase(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_eval(SJ_TASK, d, "--target", "squat_jump", "--policies", "task", "taskphase")
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            r = json.loads((Path(d) / "robust.json").read_text())
            md = (Path(d) / "robust_summary.md").read_text()
            rollouts = np.load(Path(d) / "rollouts.npz")
            spec = json.loads((SJ_TASK / "task_spec.json").read_text())
        self.assertEqual(r["meta"]["obs_groups"]["taskphase"], {"actor": "actor_taskphase", "critic": "critic_taskphase"})
        self.assertEqual(r["meta"]["task_spec"]["source"], str((SJ_TASK / "task_spec.json").resolve()))
        self.assertTrue(r["meta"]["task_spec"]["self_check"]["success"])
        self.assertEqual(len(r["conditions"]), 13)  # with the tempo suite
        for c in r["conditions"]:
            for res in c["policies"].values():
                for k in ("task_success", "stages_completed", "stages_frac", "jump_apex_m", "squat_depth_m", "fall_rate",
                          "joint_err_rad", "pushes"):
                    self.assertIsNotNone(res[k], (c["suite"], k))
        self.assertIn("| taskphase |", md)
        # The in-sim stage machine agrees with skill/task.py evaluate() on the saved body positions.
        from skill import task as T
        clean = r["conditions"][0]["policies"]
        for n in ("task", "taskphase"):
            ev = [T.evaluate(spec, rollouts[f"0/{n}/{i}/body"], 50)["stages_completed"] for i in range(2)]
            self.assertLessEqual(abs(np.mean(ev) - clean[n]["stages_completed"]), 1.0, n)
            self.assertEqual(rollouts[f"0/{n}/0"].shape, (30, 36))
        noref = next(c for c in r["conditions"] if c["suite"] == "noref")
        self.assertEqual(noref["params"]["start"], "standing")

    @unittest.skipUnless((SJ_TASK / "policy_task").is_dir(), f"no {SJ_TASK}")
    def test_task_policy_needs_its_spec(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "run"
            shutil.copytree(SJ_TASK / "policy_task", runs / "policy_task", ignore=shutil.ignore_patterns("*.log"))
            p = self.run_eval(runs, Path(d) / "out", "--target", "squat_jump")
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("A2H_TASK_SPEC", p.stderr)
            self.assertIn("can't run without the task spec", p.stderr)
            self.assertFalse((Path(d) / "out" / "mjlab_eval.json").exists())

    @unittest.skipUnless((V1 / "policy_tracking").is_dir(), f"no {V1}")
    def test_v1_run_gets_a_spec_from_its_motion(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.run_eval(V1, d, "--policies", "tracking", "latent", "--starts", "--no-noref", "--phase-push")
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            r = json.loads((Path(d) / "robust.json").read_text())
            spec = json.loads((Path(d) / "task_spec.json").read_text())
        self.assertEqual([(c["suite"], c["level"]) for c in r["conditions"]],
                         [("clean", "nominal"), ("push", "1 m/s"), ("init", "x2"), ("dynamics", "level 1"), ("speed", "1.2x")]
                         + [("tempo", "0.5x"), ("tempo", "1.5x"), ("tempo", "2x")])
        self.assertEqual(r["meta"]["task_spec"]["source"], "compiled from motion.npz")
        self.assertIn("squat", spec["summary"])
        self.assertTrue(r["meta"]["task_spec"]["self_check"]["success"])
        self.assertEqual(r["meta"]["obs_groups"]["tracking"], {"actor": "actor", "critic": "critic"})
        self.assertGreater(r["meta"]["dtw_joint_rad_still"], 0.05)


if __name__ == "__main__":
    unittest.main()
