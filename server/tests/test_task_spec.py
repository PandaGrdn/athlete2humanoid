"""The task environment: GET /api/runs/{id}/task_spec, GET /api/jobs/{id}/task_spec and the 'task spec:' log line."""

import json

from server.runs import body_part
from server.stages import classify

from .test_api import wait_done

FEATURES = ["pelvis height", "torso x", "left ankle x", "right ankle x", "right foot height", "left wrist z",
            "right wrist y", "root forward", "root turn"]


def keypose_spec():
    """A small skill/keypose.py spec (shaped like outputs/keypose/front_kick.json)."""
    rel = [0.0] * len(FEATURES)
    kick = list(rel)
    kick[3], kick[4] = 0.51, 1.0  # right ankle x, right foot height
    punch = list(rel)
    punch[5], punch[0] = 0.9, 0.1  # left wrist z (pelvis far below the floor)
    return {
        "type": "keypose", "name": "front_kick", "fps": 30.0, "duration_s": 5.567, "features": FEATURES,
        "summary": "3 key moments, 1 jump: ready -> right foot up -> jump -> land -> left hand up -> finish",
        "warnings": ["short\x1b[31m clip"],
        "stages": [
            {"kind": "settle", "start_s": 0.0, "end_s": 0.0, "name": "ready", "feet_down": [True, True]},
            {"kind": "pose", "level": 2.59, "optional": False, "start_s": 0.033, "end_s": 1.4, "frame": 42,
             "name": "right foot up‮", "goal": [0.79] * len(FEATURES), "tol": [1.0] * len(FEATURES),
             "relevance": kick, "feet_down": [True, False]},
            {"kind": "flight", "start_s": 1.4, "end_s": 1.8, "name": "jump", "clearance": 0.3, "min_air_s": 0.12},
            {"kind": "settle", "start_s": 1.8, "end_s": 1.8, "name": "land", "hold_s": 0.1, "feet_down": [True, True]},
            {"kind": "pose", "level": 0.7, "optional": True, "start_s": 1.8, "end_s": 4.2, "frame": 126,
             "name": "left hand up " + "x" * 500, "relevance": punch, "feet_down": [True, True]},
            {"kind": "settle", "start_s": 4.2, "end_s": 5.567, "name": "finish", "feet_down": [True, True]},
            "junk", {"kind": "pose", "name": "no times"},
        ],
        "self_check": {"stages_total": 6, "stages_completed": 6, "success": True, "stages_skipped": 0,
                       "fell_at_s": None, "metrics": {"falls": 0}},
    }


OLD_SPEC = {  # skill/task.py before keypose: no type, kinds stand/squat/jump/land, no self_check
    "name": "squat_jump", "duration_s": 9.033, "summary": "1 jump: stand -> squat -> jump -> land -> stand",
    "warnings": [],
    "stages": [
        {"kind": "stand", "start_s": 0.0, "end_s": 1.867},
        {"kind": "squat", "start_s": 1.967, "end_s": 2.7, "depth": 0.21, "z_goal": 0.62},
        {"kind": "jump", "start_s": 2.8, "end_s": 3.3, "clearance": 0.3, "ref_clearance": 0.36},
        {"kind": "land", "start_s": 3.3, "end_s": 4.233, "absorb_z": 0.519},
        {"kind": "stand", "start_s": 4.333, "end_s": 9.0},
    ],
}


def test_run_task_spec_keypose(client, runs_dir):
    (runs_dir / "remote_latent" / "task_spec.json").write_text(json.dumps(keypose_spec()))
    listing = {r["id"]: r for r in client.get("/api/runs").json()}
    assert listing["latent"]["has_task_spec"] and not listing["latent_sparse"]["has_task_spec"]
    r = client.get("/api/runs/latent/task_spec")
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    v = r.json()
    assert v["format"] == "keypose" and v["name"] == "front_kick" and v["duration_s"] == 5.567
    assert v["summary"].startswith("3 key moments")
    assert [s["kind"] for s in v["stages"]] == ["settle", "pose", "flight", "settle", "pose", "settle"]  # junk dropped
    kick, jump, opt = v["stages"][1], v["stages"][2], v["stages"][4]
    assert kick["at_s"] == 1.4 and jump["at_s"] == 1.4 and v["stages"][0]["at_s"] == 0.0
    assert kick["name"] == "right foot up" and kick["frame"] == 42 and kick["level"] == 2.59 and not kick["optional"]
    assert kick["body_parts"] == [{"part": "right leg", "weight": 1.0}] and kick["feet_down"] == [True, False]
    assert jump["clearance_m"] == 0.3 and jump["body_parts"] == []
    assert opt["optional"] and len(opt["name"]) <= 120
    assert [p["part"] for p in opt["body_parts"]] == ["left arm"]  # pelvis at 0.1 / 0.9 is below the cut
    assert v["counts"] == {"moments": 2, "required": 1, "optional": 1, "flights": 1}
    assert v["self_check"] == {"stages_completed": 6, "stages_total": 6, "success": True, "stages_skipped": 0,
                               "fell_at_s": None, "source": "spec"}
    assert v["warnings"] == ["short [31m clip"]
    raw = r.text
    assert "goal" not in raw and "tol" not in raw and "relevance" not in raw and "‮" not in raw


def test_run_task_spec_old_format(client, runs_dir):
    run = runs_dir / "remote_latent"
    (run / "task_spec.json").write_text(json.dumps(OLD_SPEC))
    ev = json.loads((run / "eval.json").read_text())
    ev["retarget"].update(task_stages="5/5", task_success=True)
    (run / "eval.json").write_text(json.dumps(ev))
    v = client.get("/api/runs/latent/task_spec").json()
    assert v["format"] == "phases"
    assert [(s["name"], s["kind"]) for s in v["stages"]] == [
        ("stand", "settle"), ("squat", "pose"), ("jump", "flight"), ("land", "settle"), ("stand", "settle")]
    assert v["stages"][1]["body_parts"] == [{"part": "torso/pelvis", "weight": 1.0}]
    assert v["stages"][2]["feet_down"] is None and v["stages"][0]["feet_down"] == [True, True]
    assert v["counts"] == {"moments": 1, "required": 1, "optional": 0, "flights": 1}
    sc = v["self_check"]
    assert (sc["stages_completed"], sc["stages_total"], sc["success"], sc["source"]) == (5, 5, True, "eval")


def test_run_task_spec_missing_or_bad(client, runs_dir):
    assert client.get("/api/runs/latent/task_spec").status_code == 404  # no spec
    assert client.get("/api/runs/nope/task_spec").status_code == 404
    spec = runs_dir / "remote_latent" / "task_spec.json"
    for bad in ["{not json", json.dumps({"stages": "x"}), json.dumps([1, 2]), json.dumps({"stages": [{"kind": "pose"}]})]:
        spec.write_text(bad)
        assert client.get("/api/runs/latent/task_spec").status_code == 404, bad
    spec.write_text(json.dumps({**OLD_SPEC, "pad": "x" * (2 * 1048576)}))  # over the size cap
    assert client.get("/api/runs/latent/task_spec").status_code == 404
    # a symlink out of the run folder is not followed
    outside = runs_dir.parent / "elsewhere.json"
    outside.write_text(json.dumps(OLD_SPEC))
    spec.unlink()
    spec.symlink_to(outside)
    assert client.get("/api/runs/latent/task_spec").status_code == 404


def test_job_task_spec_follows_the_replay(make_client, runs_dir):
    run = runs_dir / "remote_latent"
    (run / "task_spec.json").write_text(json.dumps(keypose_spec()))
    with open(run / "job.log", "a") as f:
        f.write("\n[01:28:43] task spec: 3 key moments, 1 jump: ready -> right foot up -> jump\n")
    slow = make_client(demo_seconds=60)
    jid = slow.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    assert slow.get(f"/api/jobs/{jid}/task_spec").status_code == 404  # replay hasn't reached the environment
    slow.post(f"/api/jobs/{jid}/cancel")
    assert slow.get("/api/jobs/nope/task_spec").status_code == 404

    fast = make_client()
    jid = fast.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    done = wait_done(fast, jid)
    env = [line[2] for line in done["logs"] if line[1] == "environment"]
    assert env == ["task spec: 3 key moments, 1 jump: ready -> right foot up -> jump"]
    v = fast.get(f"/api/jobs/{jid}/task_spec").json()
    assert v["format"] == "keypose" and v["counts"]["moments"] == 2


def test_classify_task_spec_line():
    assert classify("[01:28:43] task spec: 7 key moments, 2 jumps: ready -> crouch -> jump") == (
        "environment", "task spec: 7 key moments, 2 jumps: ready -> crouch -> jump")
    assert classify("[19:47:27] task spec: 2 jumps: stand -> squat -> jump -> land -> stand")[0] == "environment"
    # the line comes after the first latent-reward step; the job doesn't step back to the environment
    assert classify("[01:28:43] target squat_jump: viewpoint -154 deg (renders at -150)")[0] == "latent"


def test_late_environment_line_keeps_the_current_stage(client):
    store = client.app.state.jobs
    job = store.create("colab")
    store.set_stage(job, "latent")
    store.set_stage(job, "environment")
    st = {s.key: s.status for s in job.stages}
    assert job.stage == "latent" and st["environment"] == "done" and st["latent"] == "running"


def test_body_parts():
    assert body_part("right ankle x") == "right leg" and body_part("right foot height") == "right leg"
    assert body_part("left wrist z") == "left arm" and body_part("torso y") == "torso/pelvis"
    assert body_part("pelvis height") == "torso/pelvis" and body_part("root turn") == "root"
    assert body_part("something else") is None and body_part(3) is None
