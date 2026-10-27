"""Commanded targets (skill/commands.py, StageMachine.set_goals, command_spec) and their evaluation
(experiments/command_eval.py scoring) on synthetic trajectories; the A2H-TaskCmd env itself and the
TaskPhase -> TaskCmd warm start when mjlab is installed (CPU, a few seconds).

    python -m unittest tests.test_commands
"""

import importlib.util
import json
import unittest
from pathlib import Path

if all(importlib.util.find_spec(m) for m in ("numpy", "torch")):
    import numpy as np
    import torch

    from experiments import command_eval as CE
    from skill import commands as C
    from skill import task as T
    from tests.test_skill_task import FPS, squat_jump
else:
    raise unittest.SkipTest("needs numpy and torch")

DATA = Path(__file__).parent / "data"
HAS_MJLAB = importlib.util.find_spec("mjlab") is not None


def run(m, pos, env=0):
    """Feed one trajectory to env `env` of machine m from t=0 -> total reward (s of unit reward)."""
    pos = torch.as_tensor(pos)
    rows = lambda t: pos[t:t + 1].expand(m.n, -1, -1)
    m.reset(torch.arange(m.n), 0.0, rows(0))
    return sum(m.step(rows(t)) * m.dt for t in range(1, len(pos)))


class CommandedGoalsTest(unittest.TestCase):
    def setUp(self):
        self.ref = squat_jump(depth=0.2, clearance=0.25)
        self.spec = T.compile_spec(self.ref, FPS)

    def test_spec_goals_by_default(self):
        m = T.StageMachine(self.spec, 2, dt=1 / FPS)
        self.assertTrue(torch.equal(m.goals[0], m.goal))
        nom = T.nominal_command(self.spec)
        self.assertAlmostEqual(nom["depth"], 0.2, delta=0.02)
        self.assertAlmostEqual(nom["clearance"], 0.25, delta=0.02)
        self.assertEqual(nom["reps"], 1)
        # commanding the clip's own targets changes nothing
        m2 = T.StageMachine(self.spec, 2, dt=1 / FPS)
        sq = [s for s in self.spec["stages"] if s["kind"] == "squat"][0]
        m2.set_goals(torch.arange(2), depth=torch.full((2,), (self.spec["h0"] - sq["z_goal"]) / 0.75),
                     clearance=torch.full((2,), self.spec["stages"][2]["clearance"]))
        self.assertTrue(torch.allclose(m2.goals, m.goals, atol=1e-5))
        self.assertTrue(torch.allclose(run(m, self.ref), run(m2, self.ref), atol=1e-4))

    def test_depth_command_decides_squat_success(self):
        m = T.StageMachine(self.spec, 2, dt=1 / FPS)
        m.set_goals(torch.arange(2), depth=torch.tensor([0.15, 0.32]))
        run(m, self.ref)
        self.assertTrue(m.done[0])                      # asked shallower than it went
        self.assertFalse(m.done[1])                     # asked deeper: stuck in the squat
        self.assertEqual(int(m.stage[1]), 1)
        self.assertGreater(float(m.extra_depth[1]), 0.1)  # stability margins follow the command
        self.assertEqual(float(m.extra_depth[0]), 0.0)

    def test_clearance_command_decides_jump_quality(self):
        m = T.StageMachine(self.spec, 3, dt=1 / FPS)
        m.set_goals(torch.arange(3), clearance=torch.tensor([0.25, 0.12, 0.30]))
        run(m, self.ref)
        q = m.q_min.tolist()
        self.assertTrue(m.done.all())
        self.assertGreater(q[0], 0.95)     # the clip's own height
        self.assertLess(q[1], 0.7)         # ~2x what was asked: overshoot is penalised
        self.assertAlmostEqual(q[2], 0.83, delta=0.06)  # too low

    def test_set_goals_matches_command_spec(self):
        for cmd in ({"depth": 0.16, "clearance": 0.14}, {"depth": 0.3, "clearance": 0.3}):
            pos = squat_jump(depth=0.18, clearance=0.2, lead=0.3)
            m = T.StageMachine(self.spec, 1, dt=1 / FPS)
            m.set_goals(torch.tensor([0]), **{k: torch.tensor([v]) for k, v in cmd.items()})
            r1 = run(m, pos)
            r2 = run(T.StageMachine(T.command_spec(self.spec, **cmd), 1, dt=1 / FPS), pos)
            self.assertTrue(torch.allclose(r1, r2, atol=1e-4), cmd)
            self.assertEqual(T.evaluate(T.command_spec(self.spec, **cmd), pos, FPS)["success"], bool(m.done[0]))

    def test_reps(self):
        two = np.concatenate([squat_jump()[:-30], squat_jump()[1:]])
        spec = T.compile_spec(two, FPS)
        self.assertEqual(T.nominal_command(spec)["reps"], 2)
        ends = T.rep_ends(spec)
        self.assertEqual(ends[-1], len(spec["stages"]))
        self.assertEqual([s["kind"] for s in spec["stages"][:ends[0]]][-2:], ["land", "stand"])
        m = T.StageMachine(spec, 2, dt=1 / FPS)
        m.set_goals(torch.arange(2), reps=torch.tensor([1, 2]))
        one_rep = squat_jump()
        run(m, one_rep)
        self.assertEqual(m.done.tolist(), [True, False])
        self.assertEqual(len(T.command_spec(spec, reps=1)["stages"]), ends[0])
        # an episode starting after the commanded reps: nothing left to do
        m.reset(torch.tensor([0]), spec["stages"][ends[0]]["start_s"] + 0.1, torch.as_tensor(one_rep[:1]))
        self.assertTrue(bool(m.done[0]))
        self.assertEqual(int(m.stage[0]), ends[0] - 1)

    def test_command_obs(self):
        m = T.StageMachine(self.spec, 2, dt=1 / FPS)
        lo, hi = T.COMMAND_RANGES["depth"]
        m.set_goals(torch.arange(2), depth=torch.tensor([lo, hi]))
        o = m.command_obs()
        self.assertEqual(tuple(o.shape), (2, len(T.COMMANDS)))
        self.assertEqual(o[:, 0].tolist(), [-1.0, 1.0])
        self.assertTrue((o.abs() <= 1).all())

    def test_achieved(self):
        r = T.evaluate(self.spec, squat_jump(depth=0.18, clearance=0.22), FPS)["metrics"]
        self.assertAlmostEqual(r["squat_depths"][0], 0.18, delta=0.02)
        self.assertAlmostEqual(r["jump_clearances"][0], 0.22, delta=0.02)
        # a fall: the collapse is not a squat
        pos = squat_jump()
        pos[40:, :, 2] = np.linspace(0.8, 0.1, len(pos) - 40)[:, None]
        r = T.evaluate(self.spec, pos, FPS)
        self.assertIsNotNone(r["fell_at_s"])
        self.assertEqual(r["metrics"]["squat_depths"], [])


class SamplerTest(unittest.TestCase):
    def setUp(self):
        self.spec = T.compile_spec(squat_jump(depth=0.2, clearance=0.25), FPS)
        self.m = T.StageMachine(self.spec, 400, dt=1 / FPS)

    def test_ranges_widen_with_level(self):
        s = C.CommandSampler(self.m, seed=0, explore=0.0)
        b0, b1 = s.bounds(0.0), s.bounds(1.0)
        self.assertEqual(b1["depth"], T.COMMAND_RANGES["depth"])
        self.assertLess(b0["depth"][1] - b0["depth"][0], 0.2 * (b1["depth"][1] - b1["depth"][0]))
        self.assertLess(b0["depth"][0], 0.2 + 1e-6)
        self.assertGreater(b0["depth"][1], 0.2 - 1e-6)
        s.sample()
        d = self.m.command[:, 0]
        self.assertTrue(((d >= b0["depth"][0] - 1e-6) & (d <= b0["depth"][1] + 1e-6)).all())
        self.assertTrue(torch.equal(self.m.command[:, 2], torch.ones(400)))  # reps not commanded

    def test_explore_samples_the_full_range(self):
        s = C.CommandSampler(self.m, seed=0, explore=0.3)
        s.sample()
        b0 = s.bounds(0.0)["depth"]
        outside = ((self.m.command[:, 0] < b0[0]) | (self.m.command[:, 0] > b0[1])).float().mean()
        self.assertGreater(float(outside), 0.15)
        self.assertLess(float(outside), 0.35)

    def test_curriculum(self):
        s = C.CommandSampler(self.m, seed=0, window=100, step=0.25)
        ids = torch.arange(100)
        s.record(ids, torch.zeros(100, dtype=torch.bool))  # first reset: no episode ran, ignored
        self.assertEqual((s.level, s.n_ep), (0.0, 0))
        self.m.steps[:] = 10
        self.m.done[:], self.m.completed[:] = True, 5
        s.record(ids, torch.zeros(100, dtype=torch.bool))
        self.assertEqual(s.level, 0.25)
        self.assertEqual(s.rate, 1.0)
        self.m.q_min[:] = 0.5  # finished, but every jump missed the commanded height
        s.record(ids, torch.zeros(100, dtype=torch.bool))
        self.assertEqual(s.level, 0.0)
        self.m.q_min[:] = 1.0
        for _ in range(8):
            s.record(ids, torch.zeros(100, dtype=torch.bool))
        self.assertEqual(s.level, 1.0)
        s.record(ids, torch.ones(100, dtype=torch.bool))  # all fell
        self.assertEqual(s.level, 0.75)
        self.assertIn("depth_lo", s.log())

    def test_fixed_commands(self):
        s = C.CommandSampler(self.m)
        s.fix(depth=torch.full((400,), 0.3), clearance=torch.full((400,), 0.1))
        s.sample(torch.tensor([3, 7]))
        self.assertAlmostEqual(float(self.m.command[3, 0]), 0.3, places=5)
        self.assertAlmostEqual(float(self.m.command[7, 1]), 0.1, places=5)
        with self.assertRaises(ValueError):
            C.CommandSampler(self.m, windw=3)

    def test_reps_curriculum(self):
        two = np.concatenate([squat_jump()[:-30], squat_jump()[1:]])
        m = T.StageMachine(T.compile_spec(two, FPS), 200)
        s = C.CommandSampler(m, seed=0, reps=True, explore=0.0)
        s.sample()
        self.assertTrue(torch.equal(m.command[:, 2], torch.full((200,), 2.0)))  # level 0: the clip's
        s.level = 1.0
        s.sample()
        self.assertEqual(set(m.command[:, 2].tolist()), {1.0, 2.0})


def traj50(depth, clearance):
    """squat_jump() resampled from 30 to 50 fps (the env's step rate)."""
    p = squat_jump(depth=depth, clearance=clearance)
    t = np.arange(0, len(p) - 1, FPS / 50.0)
    i = t.astype(int)
    w = (t - i)[:, None, None]
    return ((1 - w) * p[i] + w * p[i + 1]).astype(np.float32)


class CommandEvalTest(unittest.TestCase):
    def setUp(self):
        self.spec = T.compile_spec(traj50(0.2, 0.25), 50.0)
        self.assertIn("jump", [s["kind"] for s in self.spec["stages"]])

    def test_grid_has_the_clip_and_novel_targets(self):
        g = CE.default_grid(json.loads((DATA / "task_spec_squat_jump.json").read_text()))
        self.assertGreaterEqual(len(g), 9)
        self.assertTrue(any(abs(c["depth"] - 0.246) < 0.002 and c["clearance"] == 0.3 for c in g))
        lo, hi = T.COMMAND_RANGES["depth"]
        self.assertTrue(all(lo <= c["depth"] <= hi for c in g))

    def test_follower_vs_ignorer(self):
        cmds = [{"depth": d, "clearance": c, "reps": None} for d in (0.15, 0.2, 0.28) for c in (0.15, 0.25)]
        follow, f_sum = CE.score(self.spec, [traj50(c["depth"], c["clearance"]) for c in cmds], cmds)
        ignore, i_sum = CE.score(self.spec, [traj50(0.2, 0.25) for _ in cmds], cmds)
        self.assertAlmostEqual(f_sum["depth_slope"], 1.0, delta=0.1)
        self.assertAlmostEqual(f_sum["clearance_slope"], 1.0, delta=0.1)
        self.assertLess(abs(i_sum["depth_slope"]), 0.1)
        self.assertLess(f_sum["depth_mae"], 0.02)
        self.assertEqual(f_sum["novel"]["cmd_success"], 1.0)
        self.assertLess(i_sum["novel"]["cmd_success"], f_sum["novel"]["cmd_success"])
        self.assertEqual([r["cmd_success"] for r in ignore if r["command"]["depth"] == 0.28], [0.0, 0.0])  # too shallow
        self.assertEqual(f_sum["clip"]["commands"], 1)


@unittest.skipUnless(HAS_MJLAB, "needs mjlab (its virtualenv)")
class TaskCmdEnvTest(unittest.TestCase):
    """The A2H-TaskCmd env on CPU: commands are the last inputs, and TaskPhase weights act unchanged."""

    @classmethod
    def setUpClass(cls):
        import os
        os.environ["A2H_TASK_SPEC"] = str(DATA / "task_spec_squat_jump.json")
        os.environ.pop("A2H_LATENT_BUNDLE", None)
        os.environ.pop("A2H_IMPORTANCE", None)
        try:
            cls.env_c, cls.venv_c = CE.make_env(CE.TASK, DATA / "motion_squat_jump_head.npz", 4, "cpu")
            cls.env_p, cls.venv_p = CE.make_env(CE.PHASE_TASK, DATA / "motion_squat_jump_head.npz", 4, "cpu")
        except Exception as e:  # e.g. G1 assets missing
            raise unittest.SkipTest(f"mjlab env unavailable: {e}") from e

    def test_command_inputs_are_last(self):
        from mjlab.tasks.registry import load_env_cfg
        for g in ("actor", "critic"):
            c = list(load_env_cfg(CE.TASK).observations[g].terms)
            p = list(load_env_cfg(CE.PHASE_TASK).observations[g].terms)
            self.assertEqual(c, p + ["task_command"])
        self.assertIn("command", self.env_c.curriculum_manager.active_terms)
        self.assertNotIn("command", self.env_p.curriculum_manager.active_terms)

    def test_warm_start_acts_unchanged(self):
        import tempfile
        from dataclasses import asdict

        from tensordict import TensorDict

        from mjlab.rl import MjlabOnPolicyRunner
        from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
        torch.manual_seed(0)
        runner = (load_runner_cls(CE.PHASE_TASK) or MjlabOnPolicyRunner)(
            self.venv_p, asdict(load_rl_cfg(CE.PHASE_TASK)), device="cpu")
        with tempfile.TemporaryDirectory() as d:
            ck = Path(d) / "model_0.pt"
            runner.save(str(ck))
            pol_c, grown = CE.load_policy(self.venv_c, CE.TASK, ck, "cpu")
            pol_p, _ = CE.load_policy(self.venv_p, CE.PHASE_TASK, ck, "cpu")
        self.assertTrue(any("->" in g for g in grown))
        self.env_c.a2h_cmd.cfg["explore"] = 1.0
        k, worst = len(T.COMMANDS), 0.0
        with torch.inference_mode():
            obs, _ = self.venv_c.reset()
            self.assertGreater(obs["actor"][:, -k:].abs().max().item(), 0.0)
            for _ in range(10):
                a = pol_c(obs)
                cut = TensorDict({g: obs[g][:, :-k] for g in obs.keys()}, batch_size=obs.batch_size)
                worst = max(worst, (a - pol_p(cut)).abs().max().item())
                obs, *_ = self.venv_c.step(a)
        self.assertEqual(worst, 0.0)

    def test_grounded_starts_and_landing_term(self):
        cmd_c = self.env_c.command_manager.get_term("motion")
        cmd_p = self.env_p.command_manager.get_term("motion")
        self.venv_c.reset()
        self.venv_p.reset()
        foot0 = json.loads((DATA / "task_spec_squat_jump.json").read_text())["foot0"]
        low = lambda c: c.robot_body_pos_w[:, [T.L_ANKLE, T.R_ANKLE], 2].min(-1).values
        self.assertGreater(float(low(cmd_p).min()), foot0 + 0.04)  # the retarget's first frame hovers
        self.assertTrue(torch.allclose(low(cmd_c), torch.full((4,), foot0), atol=0.01))  # TaskCmd: on the floor
        self.assertIn("landing", self.env_c.reward_manager.active_terms)
        self.assertNotIn("landing", self.env_p.reward_manager.active_terms)
        self.assertAlmostEqual(float(self.env_c.a2h_ground_offset[0]), float(low(cmd_p)[0] - foot0), delta=0.005)

    def test_ground_offsets_and_impact(self):
        import mjlab_tasks as mt
        z = torch.tensor([[0.10, 0.20], [0.05, 0.05], [0.30, 0.40], [0.03, 0.30], [0.08, 0.08]])
        off = mt.ground_offsets(z, 0.03, 0.12, keep_out=[(0.07, 0.09)], fps=50.0)  # frame 4 is at 0.08 s
        self.assertTrue(torch.allclose(off, torch.tensor([0.07, 0.02, 0.0, 0.0, 0.0])))
        f = torch.tensor([[[0, 0, -300.0], [0, 0, -300.0]], [[0, 0, -900.0], [0, 0, -900.0]]])
        self.assertTrue(torch.allclose(mt.impact_excess(f, 300.0, 3.0), torch.tensor([0.0, 3.0])))

    def test_rollout_applies_commands(self):
        from mjlab.rl import MjlabOnPolicyRunner  # noqa: F401
        cmds = [{"depth": d, "clearance": 0.2, "reps": None} for d in (0.12, 0.18, 0.24, 0.3)]
        zero = lambda obs: torch.zeros(4, self.env_c.action_manager.total_action_dim)
        pos = CE.rollout(self.env_c, self.venv_c, zero, cmds, steps=5)
        self.assertEqual(pos.shape[:2], (4, 6))
        self.assertTrue(torch.allclose(self.env_c.a2h_task.command[:, 0], torch.tensor([0.12, 0.18, 0.24, 0.3])))


if __name__ == "__main__":
    unittest.main()
