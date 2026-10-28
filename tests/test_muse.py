"""Muse integrations, offline: recorded fixture responses, real run data from outputs/remote_latent
(copied to tests/data). Standard library only:

    python -m unittest discover -s tests -t .
"""

import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from muse import client as mc
from muse import moment_namer as mn
from muse import rollout_critic as rc
from muse import reward_agent as ra
from muse import skill_reader as sr
from muse.schema import validate, strict_errors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "remote_latent"


def fixture_value(name):
    return json.loads(mc.load_fixture(mc.FIXTURES / name)[0]["choices"][0]["message"]["content"])


def write_fixture(path, *values):
    path.write_text(json.dumps({"responses": [mc.completion(v) for v in values]}))
    return path


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)


class SchemaTest(unittest.TestCase):
    def test_schemas_are_strict_mode_compliant(self):
        self.assertEqual(strict_errors(sr.SKILL_SCHEMA), [])
        self.assertEqual(strict_errors(ra.PROPOSAL_SCHEMA), [])
        self.assertEqual(strict_errors(mn.NAMES_SCHEMA), [])
        self.assertEqual(strict_errors(rc.CRITIC_SCHEMA), [])

    def test_strict_errors_catches_optional_property(self):
        bad = {"type": "object", "properties": {"a": {"type": "string"}}, "required": [], "additionalProperties": False}
        self.assertTrue(strict_errors(bad))

    def test_validate(self):
        schema = sr.SKILL_SCHEMA["properties"]["tempo"]
        good = {"duration_s": 4, "reps": 2, "cycle_s": 2.0, "speed": "fast", "airborne": True}
        self.assertEqual(validate(good, schema), [])
        self.assertEqual(validate(dict(good, reps=2.0), schema), [])  # JSON may write integers as 2.0
        errs = validate(dict(good, speed="warp", extra=1, reps=True), schema)
        self.assertEqual(len(errs), 3, errs)
        self.assertTrue(validate({k: v for k, v in good.items() if k != "reps"}, schema))

    def test_fixtures_match_their_schemas(self):
        self.assertEqual(validate(fixture_value("skill_spec_squat.json"), sr.SKILL_SCHEMA), [])
        self.assertEqual(validate(fixture_value("reward_proposal_remote_latent.json"), ra.PROPOSAL_SCHEMA), [])
        self.assertEqual(validate(fixture_value("moment_names_front_kick.json"), mn.NAMES_SCHEMA), [])
        self.assertEqual(validate(fixture_value("critic_front_kick.json"), rc.CRITIC_SCHEMA), [])


class SyncTest(unittest.TestCase):
    """Constants duplicated to keep muse/ free of torch/mujoco imports must match their sources."""

    def test_bank_names_match_latent_bank(self):
        src = (ROOT / "latent" / "bank.py").read_text()
        body = src[src.index("def motion_bank"):]
        names = set(re.findall(r'"(\w+)": ', body))
        names |= {f"squat_d{d}_r{r}" for d in (10, 20, 30) for r in (1, 2, 3)}
        self.assertIn('bank[f"squat_d{int(depth * 100)}_r{reps}"]', body)
        self.assertEqual(set(sr.BANK_NAMES), names)

    def test_body_groups_cover_reward_net_bodies(self):
        src = (ROOT / "latent" / "reward_net.py").read_text()
        bodies = re.findall(r'"(\w+)"', re.search(r"BODIES = \((.*?)\)", src, re.S).group(1))
        grouped = [b for g in sr.BODY_GROUPS.values() for b in g]
        self.assertEqual(sorted(grouped), sorted(bodies))


class ClientTest(TempDir):
    def test_repair_retry_sends_errors_back(self):
        good = fixture_value("reward_proposal_remote_latent.json")
        bad = dict(good, confidence="certain")
        c = mc.MuseClient(fixture=write_fixture(self.tmp / "f.json", bad, good))
        value, info = c.ask_json("t", "sys", "hello", ra.PROPOSAL_SCHEMA)
        self.assertEqual(value, good)
        self.assertEqual(info["attempts"], 2)
        retry = c.requests[1]["messages"]
        self.assertEqual([m["role"] for m in retry], ["developer", "user", "assistant", "user"])
        self.assertIn("confidence", retry[-1]["content"])

    def test_invalid_after_retries_raises(self):
        c = mc.MuseClient(fixture=write_fixture(self.tmp / "f.json", {"nope": 1}))
        with self.assertRaises(mc.MuseInvalid) as e:
            c.ask_json("t", "sys", "hello", ra.PROPOSAL_SCHEMA)
        self.assertTrue(any("missing" in x for x in e.exception.errors))

    def test_not_json(self):
        c = mc.MuseClient(fixture=write_fixture(self.tmp / "f.json", "not json {"))
        with self.assertRaises(mc.MuseInvalid):
            c.ask_json("t", "sys", "hello", ra.PROPOSAL_SCHEMA, retries=0)

    def test_request_shape(self):
        c = mc.MuseClient(fixture=mc.FIXTURES / "reward_proposal_remote_latent.json")
        c.ask_json("reward_proposal", "sys", [mc.text_part("hi"), mc.image_part(b"x" * 1000)], ra.PROPOSAL_SCHEMA)
        req = c.requests[0]
        self.assertEqual(req["model"], "muse-spark-1.3")
        self.assertEqual(req["response_format"]["type"], "json_schema")
        self.assertTrue(req["response_format"]["json_schema"]["strict"])
        self.assertEqual(req["messages"][0]["role"], "developer")
        c.save_requests(self.tmp / "req.json")
        saved = (self.tmp / "req.json").read_text()
        self.assertIn("<image/png, ", saved)
        self.assertNotIn("base64,eHh4", saved)

    def test_api_key_prefers_muse_key_and_explains_missing(self):
        secrets = {"MUSE_API_KEY": "m", "META_API_KEY": "old"}
        with mock.patch.object(config, "get_secret", lambda n, required=True: secrets.get(n)):
            self.assertEqual(mc.api_key(), "m")
        with mock.patch.object(config, "get_secret", lambda n, required=True: {"META_API_KEY": "old"}.get(n)):
            self.assertEqual(mc.api_key(), "old")
        with mock.patch.object(config, "get_secret", lambda n, required=True: None):
            with self.assertRaisesRegex(RuntimeError, "MUSE_API_KEY"):
                mc.api_key()

    def test_default_model(self):
        if "MUSE_MODEL" not in os.environ:
            self.assertEqual(config.MUSE_MODEL, "muse-spark-1.3")


class SkillReaderTest(TempDir):
    def clip_dir(self):
        d = self.tmp / "squat"
        shutil.copytree(DATA / "clips" / "squat", d)
        video = self.tmp / "squat.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)  # never decoded in fixture mode
        return d, video

    def test_read_skill_fixture(self):
        d, video = self.clip_dir()
        c = mc.make_client(dry_run=True, default_fixture="skill_spec_squat.json")
        spec = sr.read_skill(video, d, c, fps=4)
        self.assertEqual(spec["action"]["label"], "bodyweight_squat")
        saved = sr.load(d / "skill_spec.json")
        self.assertEqual(saved["_meta"]["mode"], "fixture:skill_spec_squat.json")
        req = json.loads((d / "skill_spec.request.json").read_text())[0]
        parts = req["messages"][1]["content"]
        self.assertEqual(parts[1]["type"], "video_url")
        self.assertEqual(parts[1]["video_url"]["fps"], 4)
        self.assertIn("10.20 s", parts[0]["text"])  # clip length from meta.json (306 frames at 30 fps)
        self.assertIn("left_foot 2.7/15.5", parts[0]["text"])  # retarget_report.json

    def test_check_catches_bad_phases(self):
        spec = fixture_value("skill_spec_squat.json")
        self.assertEqual(sr.check(spec, duration=10.2), [])
        spec["phases"][2]["end_s"] = 12.0  # past the clip
        spec["phases"][3]["keyframe_s"] = 0.1  # outside its phase
        spec["key_bodies"][0]["weight"] = 2
        errs = sr.check(spec, duration=10.2)
        self.assertEqual(len(errs), 4, errs)  # the long phase also overlaps the next one

    def test_out_of_range_phase_is_repaired(self):
        d, video = self.clip_dir()
        good = fixture_value("skill_spec_squat.json")
        bad = json.loads(json.dumps(good))
        bad["phases"][-1]["end_s"] = 30.0
        c = mc.MuseClient(fixture=write_fixture(self.tmp / "f.json", bad, good))
        spec = sr.read_skill(video, d, c)
        self.assertEqual(spec["_meta"]["attempts"], 2)
        self.assertIn("clip length", c.requests[1]["messages"][-1]["content"])

    def test_cli_skips_existing(self):
        d, video = self.clip_dir()
        sr.main([str(video), "--out", str(d), "--dry-run"])
        mtime = (d / "skill_spec.json").stat().st_mtime_ns
        sr.main([str(video), "--out", str(d), "--dry-run"])
        self.assertEqual((d / "skill_spec.json").stat().st_mtime_ns, mtime)


class RewardAgentTest(TempDir):
    def test_default_config_reads_iters_from_status(self):
        self.assertEqual(ra.default_config("hybrid", DATA)["iterations"], 2000)  # --iters 2000 in status.json
        self.assertEqual(ra.target_name(DATA), "squat")
        self.assertEqual(ra.default_config("latent")["tracking_scale"], 0.0)

    def test_propose_on_real_run(self):
        out, log = self.tmp / "out", self.tmp / "log.jsonl"
        c = mc.make_client(dry_run=True, default_fixture="reward_proposal_remote_latent.json")
        p = ra.propose(DATA, c, "hybrid", out, log, frames=False)
        nxt = json.loads((out / "next_reward_config.json").read_text())
        self.assertEqual(nxt["tracking_scale"], 0.5)
        self.assertEqual(nxt["sigma"], 0.2)
        self.assertEqual(nxt["iterations"], 2000)
        self.assertEqual(p["_meta"]["config"]["tracking_scale"], 0.2)
        text = c.requests[0]["messages"][1]["content"][0]["text"]
        for s in ("0.5792810320854187", "val_cosine_p10", "bend", "'hybrid'"):
            self.assertIn(s, text)
        row = json.loads(log.read_text())
        self.assertEqual(row["metrics"]["aligned"], 0.5792810320854187)
        self.assertEqual(ra.read_history(log, "hybrid")[0]["patch"], p["patch"])
        self.assertFalse((DATA / "reward_proposal.json").exists())  # input data stays read-only

    def test_history_reaches_next_prompt(self):
        log = self.tmp / "log.jsonl"
        for i in range(2):
            c = mc.make_client(dry_run=True, default_fixture="reward_proposal_remote_latent.json")
            ra.propose(DATA, c, "hybrid", self.tmp / f"out{i}", log, frames=False)
        self.assertIn("Earlier proposals", c.requests[0]["messages"][1]["content"][0]["text"])

    def test_check_rejects_bad_patches(self):
        cfg = ra.default_config("hybrid")
        op = lambda path, value: {"op": "replace", "path": path, "value": value, "reason": "r"}  # noqa: E731
        base = {"diagnosis": "d", "expected_effect": "e", "confidence": "low", "stop": False}
        cases = {
            "outside": [op("/sigma", 0.9)],
            "whole number": [op("/iterations", 1234.5)],
            "already": [op("/sigma", 0.1)],
            "at most once": [op("/sigma", 0.2), op("/sigma", 0.3)],
            "1-3 ops": [op("/sigma", 0.2), op("/latent_weight", 2), op("/iterations", 900), op("/tracking_scale", 0.4)],
        }
        for msg, patch in cases.items():
            errs = ra.check(dict(base, patch=patch), cfg)
            self.assertTrue(any(msg in e for e in errs), (msg, errs))
        self.assertTrue(ra.check(dict(base, patch=[op("/sigma", 0.2)], stop=True), cfg))
        self.assertEqual(ra.check(dict(base, patch=[], stop=True), cfg), [])
        self.assertTrue(validate(dict(base, patch=[op("/ppo/lr", 0.1)]), ra.PROPOSAL_SCHEMA))  # path enum

    def test_apply_patch_nested(self):
        cfg = ra.default_config("hybrid")
        nxt = ra.apply_patch(cfg, [{"op": "replace", "path": "/terminations/anchor_pos", "value": 0.8},
                                   {"op": "replace", "path": "/iterations", "value": 3000.0}])
        self.assertEqual(nxt["terminations"]["anchor_pos"], 0.8)
        self.assertIsInstance(nxt["iterations"], int)
        self.assertEqual(cfg["terminations"]["anchor_pos"], 0.5)

    def test_cli_loop_over_runs(self):
        runs = self.tmp / "runs"
        for name in ("run_a", "run_b"):
            shutil.copytree(DATA, runs / name)
        (runs / "not_a_run").mkdir()
        argv = [str(runs), "--dry-run", "--no-frames", "--log", str(self.tmp / "log.jsonl")]
        ra.main(argv)
        self.assertTrue((runs / "run_a" / "reward_proposal.json").exists())
        self.assertTrue((runs / "run_b" / "next_reward_config.json").exists())
        make = lambda: mc.make_client(dry_run=True, default_fixture="reward_proposal_remote_latent.json")  # noqa: E731
        self.assertEqual(ra.process([runs], make, "hybrid", frames=False), 0)  # nothing new
        self.assertEqual(len((self.tmp / "log.jsonl").read_text().splitlines()), 2)


KICK_SPEC = ROOT / "tests" / "data" / "task_spec_front_kick.json"  # keypose.py's names, before Muse


def kick_folder(path):
    """A folder like a live job's: the unnamed front-kick spec and a stand-in processed.mp4."""
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(KICK_SPEC, path / "task_spec.json")
    (path / "processed.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 not a real video")
    return path


class MomentNamerTest(TempDir):
    def name(self, folder, **kw):
        client = mc.make_client(dry_run=True, default_fixture="moment_names_front_kick.json")
        return mn.name_moments(folder / "task_spec.json", client, **kw), client

    def test_names_the_moments_and_keeps_the_measured_ones(self):
        before = json.loads(KICK_SPEC.read_text())
        spec, client = self.name(kick_folder(self.tmp / "job"))
        poses = mn.moments(spec)
        answer = fixture_value("moment_names_front_kick.json")
        self.assertEqual([st["description"] for st in poses], [m["description"] for m in answer["moments"]])
        self.assertEqual([st["measured"] for st in poses], [st["name"] for st in mn.moments(before)])
        self.assertEqual(spec["stages"][-1]["name"], "finish")  # settle stages keep their names
        self.assertEqual(spec["action"], answer["action"]["label"])
        self.assertIn(poses[0]["name"], spec["summary"])
        self.assertEqual(spec["named_by"]["mode"], "fixture:moment_names_front_kick.json")
        # the reward never reads names: everything else in the spec is unchanged
        strip = lambda st: {k: v for k, v in st.items() if k not in ("name", "label", "description", "measured")}  # noqa: E731
        self.assertEqual([strip(st) for st in spec["stages"]], [strip(st) for st in before["stages"]])
        prompt = client.requests[0]["messages"][1]["content"][0]["text"]
        self.assertIn("1. 1.40 s, required. Measured: right foot up, right foot forward.", prompt)
        self.assertEqual(client.requests[0]["messages"][1]["content"][1]["type"], "video_url")
        self.assertTrue((self.tmp / "job" / "moment_names.request.json").exists())

    def test_renaming_starts_from_the_measured_names(self):
        folder = kick_folder(self.tmp / "job")
        first, _ = self.name(folder)
        again, client = self.name(folder)
        self.assertEqual(again["stages"], first["stages"])
        self.assertNotIn("(finish) (finish)", again["summary"])
        self.assertIn("Measured: right foot down, right foot left.", client.requests[0]["messages"][1]["content"][0]["text"])

    def test_check(self):
        good = fixture_value("moment_names_front_kick.json")
        self.assertEqual(mn.check(good, 4), [])
        self.assertTrue(any("exactly one entry" in e for e in mn.check(good, 5)))
        bad = json.loads(json.dumps(good))
        bad["moments"][0]["name"] = "the right leg swings all the way up high"
        bad["moments"][1]["description"] = "Then you lower your leg."
        bad["action"]["label"] = "Side Kicks!"
        errs = mn.check(bad, 4)
        self.assertEqual(len(errs), 3, errs)

    def test_needs_a_keypose_spec_and_a_video(self):
        folder = self.tmp / "job"
        folder.mkdir()
        (folder / "task_spec.json").write_text(json.dumps({"type": "phases", "stages": []}))
        with self.assertRaises(ValueError):
            self.name(folder)
        shutil.copy(KICK_SPEC, folder / "task_spec.json")
        spec = json.loads(KICK_SPEC.read_text())
        spec["name"] = "no_such_clip"
        (folder / "task_spec.json").write_text(json.dumps(spec))
        with self.assertRaises(FileNotFoundError):
            self.name(folder)

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "skill.keypose needs torch")
    def test_summarize_matches_keypose(self):
        from skill.keypose import summarize
        spec = json.loads(KICK_SPEC.read_text())
        self.assertEqual(mn.summarize(spec["stages"]), summarize(spec["stages"]))
        self.assertEqual(mn.summarize(spec["stages"]), spec["summary"])


class RolloutCriticTest(TempDir):
    def runs(self):
        """Two runs of the front kick, as in outputs/: three policies in one, the hybrid in the other."""
        a = kick_folder(self.tmp / "remote_latent_kp_kick")  # the folder name seeds the blind order
        mn.name_moments(a / "task_spec.json", mc.make_client(dry_run=True, default_fixture="moment_names_front_kick.json"))
        b = self.tmp / "remote_latent_hyb_kick"
        evals = {"tracking": (0.726, 0), "task": (0.46, 0), "taskphase": (0.363, 0), "taskhybrid": (0.636, 1)}
        for run, names in ((a, ("tracking", "task", "taskphase")), (b, ("taskhybrid",))):
            for n in names:
                (run / f"policy_{n}").mkdir(parents=True)
                (run / f"policy_{n}" / "policy.mp4").write_bytes(b"robot " + n.encode())
            (run / "eval.json").write_text(json.dumps({n: {"aligned": evals[n][0], "falls": evals[n][1],
                                                           "task_stages": "6/6"} for n in names}))
        return a, b

    def test_blind_critique_by_policy_name(self):
        a, b = self.runs()
        client = mc.make_client(dry_run=True, default_fixture="critic_front_kick.json")
        res = rc.critique([a, b], client)
        self.assertEqual(sorted(res["policies"]), ["task", "taskhybrid", "taskphase", "tracking"])
        self.assertEqual(res["ranking"][0], "tracking")
        self.assertEqual(res["policies"]["tracking"]["does_action"], "yes")
        content = client.requests[0]["messages"][1]["content"]
        self.assertEqual([c["type"] for c in content], ["text"] + ["video_url"] * 5)
        prompt = content[0]["text"]
        self.assertNotIn("tracking", prompt)  # blind: the robots are numbers
        self.assertIn("1. 1.40 s: " + mn.moments(json.loads((a / "task_spec.json").read_text()))[0]["name"], prompt)
        ag = res["agreement"]
        self.assertEqual(ag["vjepa_rank"], ["tracking", "taskhybrid", "task", "taskphase"])
        self.assertEqual(ag["falls_agree"], "3/4")  # the stand-in eval says the hybrid fell; Muse saw no fall
        self.assertEqual(ag["by_policy"]["task"]["moments_reached"]["stage_checker"], "6/6")
        self.assertEqual(json.loads((a / "critic.json").read_text())["blind_order"], res["blind_order"])

    def test_blind_order_is_shuffled_and_repeatable(self):
        names = ["task", "taskhybrid", "taskphase", "tracking"]
        orders = set()
        for seed in range(6):
            o = sorted(names)
            rc.random.Random(seed).shuffle(o)
            orders.add(tuple(o))
        self.assertGreater(len(orders), 1)
        a, b = self.runs()
        r1 = rc.critique([a, b], mc.make_client(dry_run=True, default_fixture="critic_front_kick.json"))
        r2 = rc.critique([a, b], mc.make_client(dry_run=True, default_fixture="critic_front_kick.json"))
        self.assertEqual(r1["blind_order"], r2["blind_order"])

    def test_check(self):
        good = fixture_value("critic_front_kick.json")
        self.assertEqual(rc.check(good, 4, 4), [])
        bad = json.loads(json.dumps(good))
        bad["ranking"] = [1, 1, 2, 3]
        bad["robots"][0]["likeness"] = 11
        bad["robots"][1]["moments"] = bad["robots"][1]["moments"][:2]
        self.assertEqual(len(rc.check(bad, 4, 4)), 3, rc.check(bad, 4, 4))
        self.assertTrue(any("one entry per robot" in e for e in rc.check(good, 5, 4)))

    def test_no_policies(self):
        with self.assertRaises(FileNotFoundError):
            rc.critique([self.tmp], mc.make_client(dry_run=True, default_fixture="critic_front_kick.json"))


if __name__ == "__main__":
    unittest.main()
