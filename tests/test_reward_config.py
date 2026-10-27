"""The reward config's path into training, without mjlab or torch: muse/reward_agent.py's
next_reward_config.json -> train_motion.py --reward-config -> A2H_REWARD_CONFIG -> mjlab_tasks'
env cfgs. mjlab_tasks is imported against stub modules that record what it registers.

    python -m unittest discover -s tests -t .
"""

import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from muse import client as mc
from muse import reward_agent as ra

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "remote_latent"
TRACKING = ("motion_global_root_pos", "motion_global_root_ori", "motion_body_pos",
            "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel")
HYBRID, LATENT = "A2H-Hybrid-Flat-Unitree-G1", "A2H-Latent-Flat-Unitree-G1"
TASK, TASK_PHASE = "A2H-Task-Flat-Unitree-G1", "A2H-TaskPhase-Flat-Unitree-G1"


class RewardTermCfg:
    def __init__(self, func, weight, params=None):
        self.func, self.weight, self.params = func, weight, params or {}


def stub_env_cfg(play=False):
    """The parts of mjlab's G1 tracking env cfg that mjlab_tasks edits; tracking weights 1, thresholds 0.25."""
    actor = {k: None for k in ("command", "motion_anchor_pos_b", "motion_anchor_ori_b", "joint_pos")}
    critic = {k: None for k in ("command", "joint_pos")}
    return SimpleNamespace(observations={"actor": SimpleNamespace(terms=actor), "critic": SimpleNamespace(terms=critic)},
                           rewards={k: SimpleNamespace(weight=1.0, func=k, params={}) for k in TRACKING},
                           terminations={k: SimpleNamespace(params={"threshold": 0.25})
                                         for k in ("anchor_pos", "anchor_ori", "ee_body_pos")})


def import_mjlab_tasks(reward_config=None, extra_env=None):
    """-> (module, {task_id: env_cfg}) with mjlab, torch, numpy and latent.reward_net stubbed."""
    registered = {}
    mods = {n: types.ModuleType(n) for n in (
        "torch", "numpy", "latent.reward_net", "mjlab", "mjlab.tasks", "mjlab.managers",
        "mjlab.managers.reward_manager", "mjlab.managers.observation_manager", "mjlab.managers.termination_manager",
        "mjlab.tasks.registry", "mjlab.tasks.tracking",
        "mjlab.tasks.tracking.config", "mjlab.tasks.tracking.config.g1", "mjlab.tasks.tracking.config.g1.env_cfgs",
        "mjlab.tasks.tracking.config.g1.rl_cfg", "mjlab.tasks.tracking.rl")}
    rn = mods["latent.reward_net"]
    rn.BODIES, rn.FPS, rn.WINDOW, rn.load_bundle, rn.window_center_step = (), 30.0, 32, None, None
    mods["mjlab.managers.reward_manager"].RewardTermCfg = RewardTermCfg
    mods["mjlab.managers.observation_manager"].ObservationTermCfg = lambda **kw: SimpleNamespace(**kw)
    mods["mjlab.managers.termination_manager"].TerminationTermCfg = lambda **kw: SimpleNamespace(**kw)
    mods["mjlab.tasks.registry"].register_mjlab_task = lambda task_id, env_cfg, **kw: registered.update({task_id: env_cfg})
    mods["mjlab.tasks.tracking.config.g1.env_cfgs"].unitree_g1_flat_tracking_env_cfg = stub_env_cfg
    mods["mjlab.tasks.tracking.config.g1.rl_cfg"].unitree_g1_tracking_ppo_runner_cfg = lambda: None
    mods["mjlab.tasks.tracking.rl"].MotionTrackingOnPolicyRunner = object
    env = ({"A2H_REWARD_CONFIG": str(reward_config)} if reward_config else {}) | (extra_env or {})
    with mock.patch.dict(sys.modules, mods), mock.patch.dict(os.environ, env):
        for k in ("A2H_REWARD_CONFIG", "A2H_IMPORTANCE", "A2H_LATENT_BUNDLE"):
            if k not in env:
                os.environ.pop(k, None)
        sys.modules.pop("mjlab_tasks", None)
        module = importlib.import_module("mjlab_tasks")
        sys.modules.pop("mjlab_tasks", None)
    return module, registered


def summary(cfg):
    """An env cfg in reward_agent's config format (minus iterations)."""
    tracking = {cfg.rewards[k].weight for k in TRACKING if k in cfg.rewards}
    latent = cfg.rewards["latent"]
    return {"latent_weight": latent.weight, "sigma": latent.params["sigma"],
            "tracking_scale": tracking.pop() if tracking else 0.0,
            "terminations": {k: t.params["threshold"] for k, t in cfg.terminations.items()}}


class RewardConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_defaults_match_reward_agent(self):
        mt, tasks = import_mjlab_tasks()
        for task, variant in ((HYBRID, "hybrid"), (LATENT, "latent")):
            want = {k: v for k, v in ra.DEFAULTS[variant].items() if k != "iterations"}
            if variant == "latent":
                want["terminations"] = {k: v for k, v in want["terminations"].items() if k != "ee_body_pos"}
            self.assertEqual(summary(tasks[task]), want, task)
        self.assertEqual(mt.reward_config(0.2), {k: v for k, v in ra.DEFAULTS["hybrid"].items() if k != "iterations"})
        src = (ROOT / "experiments" / "latent_policy.py").read_text()
        iters = int(re.search(r'"--iters", type=int, default=(\d+)', src).group(1))
        self.assertEqual(ra.DEFAULTS["hybrid"]["iterations"], iters)

    def test_agent_proposal_round_trips_into_env_cfg(self):
        c = mc.make_client(dry_run=True, default_fixture="reward_proposal_remote_latent.json")
        ra.propose(DATA, c, "hybrid", self.tmp, frames=False)
        nxt = json.loads((self.tmp / "next_reward_config.json").read_text())
        nxt["terminations"]["anchor_pos"] = 0.8  # one nested value too
        (self.tmp / "next_reward_config.json").write_text(json.dumps(nxt))
        _, tasks = import_mjlab_tasks(self.tmp / "next_reward_config.json")
        got = summary(tasks[HYBRID])
        self.assertEqual(got, {k: v for k, v in nxt.items() if k != "iterations"})
        self.assertEqual((got["tracking_scale"], got["sigma"]), (0.5, 0.2))
        latent = summary(tasks[LATENT])  # same weight, sigma, thresholds; still latent-only
        self.assertEqual((latent["tracking_scale"], latent["sigma"], latent["terminations"]["anchor_pos"]), (0.0, 0.2, 0.8))
        self.assertNotIn("ee_body_pos", latent["terminations"])

    def test_latent_reward_uses_sigma_param(self):
        mt, tasks = import_mjlab_tasks()
        self.assertIs(tasks[HYBRID].rewards["latent"].func, mt.LatentReward)
        self.assertEqual(mt.LatentReward.__call__.__defaults__, (mt.SIGMA,))

    def test_train_motion_sets_env_and_iterations(self):
        import train_motion
        cfg = dict(ra.DEFAULTS["hybrid"], iterations=777)
        (self.tmp / "rc.json").write_text(json.dumps(cfg))
        calls = []
        with mock.patch.object(train_motion, "run", lambda cmd, **kw: calls.append((cmd, kw))), \
                mock.patch.object(train_motion.shutil, "copy"), \
                mock.patch.object(train_motion.glob, "glob", lambda p: [f"{self.tmp}/logs/a/b/model_7.pt"]), \
                mock.patch.object(train_motion, "open", mock.mock_open(), create=True), \
                mock.patch.object(train_motion.fcntl, "flock"):
            train_motion.train_motion(self.tmp / "m.csv", 30, self.tmp / "out", iters=1500,
                                      task=HYBRID, bundle=self.tmp / "reward.pt", reward_config=self.tmp / "rc.json")
        cmd, kw = calls[1]  # the PPO training call
        self.assertEqual(cmd[cmd.index("--agent.max-iterations") + 1], 777)
        written = self.tmp / "out" / "reward_config.json"
        self.assertEqual(Path(kw["env"]["A2H_REWARD_CONFIG"]), written.resolve())
        self.assertEqual(json.loads(written.read_text())["iterations"], 777)

    def test_agent_reads_the_trained_config(self):
        run = self.tmp / "run"
        shutil.copytree(DATA, run)
        (run / "policy_hybrid").mkdir()
        (run / "policy_hybrid" / "reward_config.json").write_text(json.dumps(dict(ra.DEFAULTS["hybrid"], sigma=0.15)))
        c = mc.make_client(dry_run=True, default_fixture="reward_proposal_remote_latent.json")
        p = ra.propose(run, c, "hybrid", self.tmp / "out", frames=False)
        self.assertEqual(p["_meta"]["config"]["sigma"], 0.15)

    def test_task_first_tasks(self):
        mt, tasks = import_mjlab_tasks(extra_env={"A2H_LATENT_BUNDLE": "reward.pt"})
        task, phase = tasks[TASK], tasks[TASK_PHASE]
        for cfg in (task, phase):
            self.assertIs(cfg.rewards["task"].func, mt.TaskReward)
            self.assertEqual(cfg.rewards["task"].weight, 2.0)
            self.assertEqual(cfg.rewards["latent"].weight, 1.0)
            self.assertEqual(set(cfg.terminations), {"fallen"})  # never for leaving the retarget
            self.assertIn("task_stage", cfg.observations["actor"].terms)
        self.assertEqual({task.rewards[k].weight for k in TRACKING}, {0.1})
        self.assertFalse(any(k in phase.rewards for k in TRACKING))
        self.assertFalse(any(k in phase.observations["actor"].terms for k in mt.REFERENCE_OBS))
        self.assertIn("command", task.observations["actor"].terms)

    def test_task_without_latent_bundle_or_with_importance(self):
        _, tasks = import_mjlab_tasks()
        self.assertNotIn("latent", tasks[TASK].rewards)
        mt, tasks = import_mjlab_tasks(extra_env={"A2H_LATENT_BUNDLE": "reward.pt", "A2H_IMPORTANCE": "imp.json"})
        r = tasks[TASK].rewards
        self.assertIs(r["latent"].func, mt.TimeWeighted)
        self.assertIs(r["latent"].params["inner"], mt.LatentReward)
        self.assertIs(r["motion_body_pos"].func, mt.TimeWeighted)
        self.assertAlmostEqual(r["motion_body_pos"].weight, 0.1)
        self.assertIs(r["task"].func, mt.TaskReward)  # the goal itself is never down-weighted

    def test_latent_variant_cannot_add_tracking(self):
        c = mc.make_client(dry_run=True, default_fixture="reward_proposal_remote_latent.json")
        with self.assertRaises(mc.MuseInvalid) as e:  # the fixture's patch sets /tracking_scale
            ra.propose(DATA, c, "latent", self.tmp, frames=False)
        self.assertTrue(any("stays 0" in x for x in e.exception.errors))


if __name__ == "__main__":
    unittest.main()
