"""experiments/adaptation_curves.py: log parsing, checkpoint picking, summaries and the report (no GPU)."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import adaptation_curves as AC  # noqa: E402

LOG = """\x1b[1m                 Learning iteration 0/1500                  \x1b[0m
                        Collection time: 1.675s
                           Time elapsed: 0:00:02
################################################################################
\x1b[1m                 Learning iteration 99/1500                  \x1b[0m
                           Time elapsed: 0:02:58
\x1b[1m                 Learning iteration 1499/1500                  \x1b[0m
                           Time elapsed: 1 day, 1:02:03
"""


def test_elapsed_by_iter():
    assert AC.elapsed_by_iter(LOG) == {0: 2, 99: 178, 1499: 86400 + 3723}


def test_elapsed_stitches_a_resumed_log():
    resumed = LOG.split("##")[0] + """Learning iteration 50/1500
                           Time elapsed: 0:01:00
Learning iteration 40/1500
                           Time elapsed: 0:00:05
Learning iteration 41/1500
                           Time elapsed: 0:00:10
"""
    e = AC.elapsed_by_iter(resumed)
    assert e[50] == 60 and e[40] == 2 + 5 and e[41] == 2 + 10  # resumed after iteration 39: none logged, so from 0's 2 s


def test_pick_is_closest_to_n_updates():
    ck = {k: None for k in (0, 50, 100, 150, 1499)}
    assert AC.pick(ck, 100, 25) == 100  # model_99 isn't saved; model_100 is one update more
    assert AC.pick(ck, 1500, 25) == 1499
    assert AC.pick(ck, 1000, 25) is None
    assert AC.pick({}, 100, 25) is None
    assert AC.pick({99: None, 100: None}, 100, 25) == 99


def test_parse_start():
    assert AC.parse_start("multi=/a/model_1999.pt") == ("multi", Path("/a/model_1999.pt"), AC.TRACK)
    assert AC.parse_start("big=/a/m.pt@A2H-Other") == ("big", Path("/a/m.pt"), "A2H-Other")
    with pytest.raises(SystemExit):
        AC.parse_start("multi_vj=/a.pt")


def test_find_policies(tmp_path):
    for name, task in (("scratch_100", AC.TRACK), ("multi_100", AC.TRACK), ("half", None)):
        d = tmp_path / f"policy_{name}"
        d.mkdir()
        (d / "model_50.pt").write_bytes(b"")
        (d / "model_100.pt").write_bytes(b"")
        (d / "motion.npz").write_bytes(b"")
        if task:
            (d / "task.txt").write_text(task + "\n")
    p = AC.find_policies(tmp_path, None)
    assert sorted(p) == ["multi_100", "scratch_100"]
    assert p["multi_100"]["checkpoint"].endswith("model_100.pt") and p["multi_100"]["task"] == AC.TRACK
    assert list(AC.find_policies(tmp_path, ["multi_100"])) == ["multi_100"]
    with pytest.raises(SystemExit):
        AC.find_policies(tmp_path / "policy_half", None)


def cond(suite, pols):
    return {"suite": suite, "level": "x", "policies": pols}


def r(ts, fall, je=0.1):
    return {"task_success": ts, "fall_rate": fall, "stages_frac": ts, "dtw_joint_rad": je, "joint_err_rad": je}


def fake_run(tmp_path, starts=("scratch", "multi")):
    """A run folder as `run` leaves it: timing.json, train logs, eval/{short,full}/robust.json."""
    out = tmp_path / "run"
    at = [100, 1500]
    short = [cond("clean", {f"{s}_{n}": r(0.9 if (s, n) != ("scratch", 100) else 0.0, 0.0) for s in starts for n in at}),
             cond("transfer", {f"{s}_{n}": r(0.5, 0.1) for s in starts for n in at}),
             cond("transfer", {f"{s}_{n}": r(0.7, 0.3) for s in starts for n in at}),
             cond("push", {f"{s}_{n}": r(0.8, 0.0) for s in starts for n in at}),
             cond("tempo", {f"{s}_{n}": r(None, 0.5) for s in starts for n in at})]
    full = [cond("clean", {f"{s}_1500": r(1.0, 0.0) for s in starts}), cond("dynamics", {f"{s}_1500": r(0.5, 0.2) for s in starts})]
    for kind, c in (("short", short), ("full", full)):
        (out / "eval" / kind).mkdir(parents=True)
        (out / "eval" / kind / "robust.json").write_text(json.dumps({"conditions": c}))
    for s in starts:
        (out / f"policy_{s}").mkdir()
        (out / f"policy_{s}" / "train.log").write_text(LOG)
    (out / "timing.json").write_text(json.dumps({s: {"startup_s": 60, "wall_s": 999} for s in starts}))
    return out, {s: {"checkpoint": None, "task": AC.TRACK} for s in starts}, at


def test_summarise_and_report(tmp_path):
    out, starts, at = fake_run(tmp_path)
    res = AC.summarise(out, "discus", starts, at, build_s=120)
    c = res["starts"]["multi"]["curve"]
    assert c["100"]["ppo_s"] == 178 and c["100"]["minutes"] == pytest.approx((120 + 60 + 178) / 60)
    assert c["100"]["suites"]["transfer"]["task_success"] == pytest.approx(0.6)
    assert c["100"]["suites"]["tempo"]["task_success"] is None  # every condition missing it
    assert res["starts"]["multi"]["full"]["dynamics"]["fall_rate"] == 0.2
    assert AC.minutes_to(res["starts"]["scratch"]["curve"], 0.8) == (1500, c["1500"]["minutes"])
    assert AC.minutes_to(c, 0.8)[0] == 100
    assert AC.minutes_to(c, 0.95) is None
    md = (out / "curves.md").read_text()
    assert "| multi | 100 | 6 | 90% (0%) | 60% (20%) | 80% (0%) | - (50%) | 0.100 |" in md

    pytest.importorskip("matplotlib")
    rep = tmp_path / "report"
    AC.report(Namespace(runs=[str(out)], out=str(rep), thresholds=[0.5, 0.8]))
    text = (rep / "adaptation_curves.md").read_text()
    assert "| discus | scratch | 1505 min (1500 it) | 1505 min (1500 it) | not reached |" in text
    assert (rep / "adaptation_curves.png").stat().st_size > 1000


def test_robust_mean():
    assert AC.robust_mean({s: {"task_success": 0.5} for s in AC.SHORT}) == 0.5
    assert AC.robust_mean({"clean": {"task_success": 1.0}}) is None
