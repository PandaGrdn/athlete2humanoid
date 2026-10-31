"""muse/agent_loop.py offline: the tools, their bounds, and the loop's conversation over canned rounds.

    python -m unittest tests.test_agent_loop
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from muse import agent_loop as al
from muse import client as mc
from muse.reward_agent import STABILITY
from muse.schema import strict_errors

ROOT = Path(__file__).resolve().parents[1]
KICK_SPEC = ROOT / "tests" / "data" / "task_spec_front_kick.json"


class ToolTest(unittest.TestCase):
    def test_tools_are_strict_mode_compliant(self):
        for t in al.TOOLS:
            self.assertEqual(strict_errors(t["parameters"]), [], t["name"])

    def test_function_calls_parsed_in_order(self):
        resp = {"output": [{"type": "reasoning", "summary": []},
                           {"type": "message", "content": [{"type": "output_text", "text": "Raising the bonus."}]},
                           {"type": "function_call", "call_id": "c1", "name": "set_param",
                            "arguments": '{"path": "task.done", "value": 6, "reason": "r"}'},
                           {"type": "function_call", "call_id": "c2", "name": "train_next_round", "arguments": ""}]}
        calls = al.function_calls(resp)
        self.assertEqual([c["call_id"] for c in calls], ["c1", "c2"])
        self.assertEqual(calls[1]["arguments"], "{}")
        self.assertEqual(al.output_text(resp), "Raising the bonus.")

    def test_set_param_bounds_and_limits(self):
        s = al.Session(al.defaults(json.loads(KICK_SPEC.read_text())))
        ok, _ = s.call("set_param", '{"path": "task.done", "value": 8, "reason": "finish"}')
        self.assertEqual((ok["ok"], ok["old"], ok["new"]), (True, 4.0, 8.0))
        for args, msg in (({"path": "style_weight", "value": 9}, "outside [0.0, 3.0]"),
                          ({"path": "task.secret", "value": 1}, "unknown setting"),
                          ({"path": "task.done", "value": "high"}, "must be a number"),
                          ({"path": "task.done", "value": 8}, "already 8")):
            out, dec = s.call("set_param", json.dumps(args))
            self.assertFalse(out["ok"])
            self.assertIn(msg, out["error"])
            self.assertIsNone(dec)
        self.assertEqual(s.cfg["style_weight"], 1.0)  # a rejected value changes nothing
        s.call("set_param", '{"path": "style_weight", "value": 0.5, "reason": ""}')
        s.call("set_param", '{"path": "stability.alive", "value": 2, "reason": ""}')
        out, _ = s.call("set_param", '{"path": "stability.upright", "value": -4, "reason": ""}')
        self.assertIn(f"at most {al.MAX_CHANGES}", out["error"])
        self.assertTrue(s.call("set_param", '{"path": "task.done", "value": 6, "reason": "again"}')[0]["ok"])  # same path
        self.assertEqual(s.changed(), ["stability.alive", "style_weight", "task.done"])
        s.new_round()
        self.assertTrue(s.call("set_param", '{"path": "stability.upright", "value": -4, "reason": ""}')[0]["ok"])

    def test_other_tools(self):
        s = al.Session(al.defaults(json.loads(KICK_SPEC.read_text())))
        cfg, _ = s.call("get_config", "{}")
        self.assertEqual(cfg["changes_left"], al.MAX_CHANGES)
        self.assertEqual(cfg["settings"]["terminations.fall_tilt"]["bounds"], [0.15, 1.0])
        self.assertIn("not valid JSON", s.call("set_param", "{path:")[0]["error"])
        self.assertIn("unknown tool", s.call("delete_run", "{}")[0]["error"])
        out, dec = s.call("train_next_round", '{"hypothesis": "h", "watch": "clean_stages"}')
        self.assertEqual((out, dec["tool"], dec["watch"]), ({"ok": True}, "train_next_round", "clean_stages"))
        self.assertFalse(s.call("finish", '{"best_round": "two", "summary": ""}')[0]["ok"])
        self.assertEqual(s.call("finish", '{"best_round": 2, "summary": "s"}')[1]["best_round"], 2)

    def test_config_files_reach_mjlab_tasks(self):
        """Every setting lands where mjlab_tasks.reward_config / task_config (or the spec's weights) reads it."""
        spec = json.loads(KICK_SPEC.read_text())
        cfg = al.defaults(spec)
        rc, tw = al.config_files(cfg)
        self.assertLessEqual(set(spec["weights"]), set(tw))  # an older spec may lack a weight: the default stands
        self.assertEqual({k: tw[k] for k in spec["weights"]}, {k: float(v) for k, v in spec["weights"].items()})
        self.assertLessEqual(set(rc["stability"]), set(STABILITY))
        self.assertEqual(rc["stability"], {k: float(STABILITY[k]) for k in rc["stability"]})
        self.assertEqual(set(rc) - {"stability"}, {"task_weight", "style_weight", "tracking_scale"})
        for p, (_, lo, hi, _w) in al.PARAMS.items():
            self.assertTrue(lo <= cfg[p] <= hi, p)


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        run = self.tmp / "run"
        (run / "clips" / "jumping_jacks").mkdir(parents=True)
        (run / "policy_task").mkdir()
        shutil.copy(KICK_SPEC, run / "task_spec.json")
        (run / "reward.pt").write_bytes(b"")
        for f in ("model_1499.pt", "motion.npz", "policy_qpos.npy", "policy.mp4", "policy_metrics.json"):
            (run / "policy_task" / f).write_bytes(b"")
        self.run_dir, self.seen = run, []
        canned = al.canned_round(mc.FIXTURES / "agent_round_jacks.json")

        def run_round(k, inp, tag, eval_only):
            arms = {d.name: (json.loads((d / "reward_config.json").read_text()) if (d / "reward_config.json").exists() else None,
                             json.loads((d / "task_weights.json").read_text()) if (d / "task_weights.json").exists() else None,
                             sorted(p.name for p in d.glob("model_*.pt")))
                    for d in (Path(inp) / "arms").iterdir()}
            self.seen.append((k, tag, eval_only, arms))
            return canned(k, inp, tag, eval_only)
        self.client = mc.MuseClient(fixture=mc.FIXTURES / "agent_loop_jacks.json")
        self.loop = al.Loop(run, "jumping_jacks", self.tmp / "agent_jacks", self.client, run_round, rounds=3, iters=300)

    def test_loop_over_canned_rounds(self):
        rep = self.loop.go()
        # round 0 scores the run as it is; rounds 1-3 train both arms, the agent's with its settings
        self.assertEqual([(k, tag, ev) for k, tag, ev, _ in self.seen],
                         [(0, "agent_jacks_r0", True), (1, "agent_jacks_r1", False), (2, "agent_jacks_r2", False),
                          (3, "agent_jacks_r3", False)])
        self.assertEqual(list(self.seen[0][3]), ["start"])
        r1 = self.seen[1][3]
        self.assertEqual(r1["agent"][1]["done"], 8.0)  # the accepted change
        self.assertEqual(r1["agent"][0]["style_weight"], 1.0)  # the out-of-bounds one was refused
        self.assertEqual(r1["base"][1]["done"], 4.0)  # the control keeps the defaults
        self.assertEqual(r1["agent"][2], ["model_1499.pt"])  # both from the run's checkpoint
        self.assertEqual(self.seen[2][3]["agent"][0]["style_weight"], 0.5)
        self.assertEqual(self.seen[2][3]["agent"][2], ["model_1799.pt"])  # then each from its own last one
        # the conversation: each request chains on the last response; tool results go back as outputs
        reqs = self.client.requests
        self.assertNotIn("previous_response_id", reqs[0])
        self.assertEqual([r.get("previous_response_id") for r in reqs[1:]],
                         [f"resp_fixture_{i}" for i in range(1, len(reqs))])
        self.assertEqual(reqs[0]["input"][0]["content"][0]["type"], "input_text")
        outs = {i["call_id"]: json.loads(i["output"]) for i in reqs[2]["input"] if i.get("type") == "function_call_output"}
        trained = next(o for o in outs.values() if "result" in o)  # train_next_round's output: the round's result
        self.assertEqual(sorted(o["ok"] for o in outs.values() if o is not trained), [False, True])  # set_param x2
        self.assertEqual(trained["round"], 1)
        self.assertIn("clean nominal", trained["result"])
        self.assertIn("Use the tools", reqs[3]["input"][0]["content"][0]["text"])  # a text-only reply is nudged
        self.assertEqual(rep["finished"]["best_round"], 3)
        self.assertEqual(len(rep["rows"]), 1 + 2 * 3)
        self.assertEqual(rep["tokens"]["responses"], 6)
        self.assertGreater(rep["tokens"]["usd"], 0)
        self.assertTrue((self.tmp / "agent_jacks" / "report.md").read_text().startswith("# Reward agent loop"))

    def test_transient_api_errors_are_retried(self):
        class Flaky:
            def __init__(self, inner):
                self.inner, self.fails, self.requests = inner, 1, inner.requests

            def respond(self, **req):
                if self.fails:
                    self.fails -= 1
                    err = RuntimeError("internal server error")
                    err.status_code = 500
                    raise err
                return self.inner.respond(**req)
        self.loop.client = Flaky(self.client)
        with mock.patch.object(al, "RETRY_S", (0,)):
            rep = self.loop.go()
        self.assertEqual(rep["finished"]["best_round"], 3)
        self.assertIn("retry_in_s", (self.tmp / "agent_jacks" / "agent_log.jsonl").read_text())
        self.assertFalse(al.transient(ValueError("bad request")))

    def test_resumes_without_rerunning_rounds(self):
        self.loop.go()
        n = len(self.seen)
        again = al.Loop(self.run_dir, "jumping_jacks", self.tmp / "agent_jacks", self.client, None, rounds=3)
        rep = again.go()  # finished: nothing to run, the report is rebuilt from state.json
        self.assertEqual(len(self.seen), n)
        self.assertEqual(rep["finished"]["best_round"], 3)


if __name__ == "__main__":
    unittest.main()
