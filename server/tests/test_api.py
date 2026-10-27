"""API tests: run listing, artifact serving, upload validation and the demo runner's job lifecycle.

    python -m pytest server/tests -q
"""

import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app
from server.stages import classify

REPO = Path(__file__).resolve().parents[2]
MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64
FAKE_TOKEN = "hf_" + "A" * 30


@pytest.fixture
def runs_dir(tmp_path):
    root = tmp_path / "runs"
    run = root / "remote_latent"
    shutil.copytree(REPO / "tests" / "data" / "remote_latent", run)
    ev = json.loads((run / "eval.json").read_text())
    (run / "comparison.mp4").write_bytes(MP4)
    (run / "clips" / "squat" / "side_by_side.mp4").write_bytes(MP4)
    (run / "clips" / "squat" / "motion.csv").write_text("0,0,0.8\n")
    (run / "notes.py").write_text("secret = 1\n")
    pol = run / "policy_latent"
    pol.mkdir()
    (pol / "model_9.pt").write_bytes(b"pt")
    (pol / "train.log").write_text("".join(
        f"\x1b[1m  Learning iteration {i}/10  \x1b[0m\n  Mean reward: {i * 0.5:.2f}\n" for i in range(10)))
    (run / "job.log").write_text("\n".join([
        "Loading weights:  50%|#####     | noise",
        "SAM 3.1 ('person'): object 1, visible in 100% of frames",
        "pose found in 100% of frames",
        "retarget error, cm (mean, max): {'pelvis': (0.5, 0.9)}",
        f"HF_TOKEN={FAKE_TOKEN}",
        "squat: 35 windows, 153 dense steps (15/s) x 1024 dims",
        f"[04:09:55] training the reward network token: {FAKE_TOKEN}",
        "[04:10:04] training latent (A2H-Latent-Flat-Unitree-G1), log: train.log",
        "[06:29:32] latent finished with code 0",
        "[06:32:12] latent   : aligned 0.532, dtw 0.549",
        "[06:32:19] done: eval.json",
    ]))
    robust = root / "remote_robust"
    robust.mkdir()
    (robust / "robust.json").write_text(json.dumps({
        "meta": {"reps": 64},
        "reference_check": {k: {"eval_json_aligned": ev[k]["aligned"]} for k in ("tracking", "hybrid", "latent") if k in ev},
        "conditions": [{"suite": "push", "level": "1 m/s", "policies": {"latent": {"fall_rate": 0.5, "vjepa_aligned": 0.4}}}],
        "summary": {},
    }))
    sparse = root / "remote_latent_sparse"  # only an eval.json, with junk in it
    sparse.mkdir()
    (sparse / "eval.json").write_text(json.dumps({"retarget": {"aligned": "n/a"}, "latent": {"aligned": 0.5}, "_x": 1}))
    broken = root / "remote_latent_broken"
    broken.mkdir()
    (broken / "eval.json").write_text("{not json")
    smoke = root / "remote_latent_smoke"
    smoke.mkdir()
    (smoke / "eval.json").write_text("{}")
    return root


@pytest.fixture
def make_client(runs_dir, tmp_path):
    def make(**kw):
        s = Settings()
        s.runs_dir, s.data_dir = runs_dir, tmp_path / "data"
        s.demo_seconds, s.demo_run, s.runners, s.enable_colab = 0.3, "latent", ["demo"], False
        for k, v in kw.items():
            setattr(s, k, v)
        return TestClient(create_app(s))
    return make


@pytest.fixture
def client(make_client):
    return make_client()


def wait_done(client, jid, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {jid} still {j['status']}")


# --- runs ----------------------------------------------------------------------------------
def test_health_lists_only_demo_by_default(client, make_client):
    h = client.get("/api/health").json()
    assert h["runners"] == ["demo"] and h["demo_run"] == "latent"
    assert [s["key"] for s in h["stages"]] == ["silhouette", "pose", "retarget", "latent", "train", "eval"]
    # colab needs both the runner list and the explicit enable flag
    assert make_client(runners=["demo", "colab"]).get("/api/health").json()["runners"] == ["demo"]
    assert make_client(runners=["demo", "colab"], enable_colab=True).get("/api/health").json()["runners"] == ["demo", "colab"]


def test_run_listing_is_tolerant(client):
    runs = {r["id"]: r for r in client.get("/api/runs").json()}
    assert set(runs) == {"latent", "latent_sparse", "latent_broken"}  # smoke runs hidden
    assert runs["latent"]["has_robust"] and runs["latent"]["target"] == "squat"
    assert runs["latent"]["hero"]["url"].endswith("clips/squat/side_by_side.mp4")
    sparse = client.get("/api/runs/latent_sparse").json()
    assert sparse["reference"]["aligned"] is None and sparse["policies"][0]["metrics"]["aligned"] == 0.5
    assert sparse["robust"] is None and sparse["clips"] == [] and sparse["video"] is None
    broken = client.get("/api/runs/latent_broken").json()
    assert broken["policies"] == [] and broken["headline"]["best_aligned"] is None
    assert client.get("/api/runs/nope").status_code == 404


def test_run_detail(client):
    d = client.get("/api/runs/latent").json()
    latent = next(p for p in d["policies"] if p["key"] == "latent")
    assert latent["curve"][0] == [0, 0.0] and latent["curve"][-1] == [9, 4.5]
    assert latent["checkpoint"]["path"] == "policy_latent/model_9.pt" and latent["iterations"] == 10
    assert d["robust"]["reps"] == 64 and d["robust"]["conditions"][0]["policies"]["latent"]["fall_rate"] == 0.5
    assert d["clips"][0]["name"] == "squat" and d["clips"][0]["is_target"]
    kinds = {x["kind"] for x in d["downloads"]}
    assert {"checkpoint", "motion", "metrics"} <= kinds


def test_artifact_serving(client):
    r = client.get("/api/runs/latent/files/comparison.mp4")
    assert r.status_code == 200 and r.content == MP4 and r.headers["content-type"] == "video/mp4"
    r = client.get("/api/runs/latent/files/comparison.mp4", headers={"Range": "bytes=0-9"})
    assert r.status_code == 206 and len(r.content) == 10
    r = client.get("/api/runs/latent/files/clips/squat/motion.csv?download=1")
    assert r.status_code == 200 and r.headers["content-disposition"] == 'attachment; filename="a2h-latent-clips-squat-motion.csv"'
    for bad in ["notes.py", "job.log", "../remote_robust/robust.json", "..%2Fremote_robust%2Frobust.json",
                "clips/../../remote_robust/robust.json", "missing.mp4"]:
        assert client.get(f"/api/runs/latent/files/{bad}").status_code == 404, bad


# --- jobs ----------------------------------------------------------------------------------
def test_demo_job_lifecycle(client):
    r = client.post("/api/jobs", data={"run_id": "latent"})
    assert r.status_code == 201
    job = r.json()
    assert job["status"] in ("queued", "running") and job["demo"] and job["source_run"] == "latent"
    done = wait_done(client, job["id"])
    assert done["status"] == "succeeded" and done["result_run"] == "latent" and done["progress"] == 1.0
    assert all(s["status"] == "done" for s in done["stages"])
    assert {line[1] for line in done["logs"]} == {"silhouette", "pose", "retarget", "latent", "train", "eval"}
    text = "\n".join(line[2] for line in done["logs"])
    assert FAKE_TOKEN not in text and "Loading weights" not in text
    assert done["curves"]["latent"][-1] == [9, 4.5]
    assert any(j["id"] == job["id"] for j in client.get("/api/jobs").json())


def test_job_events_stream(client):
    jid = client.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    with client.stream("GET", f"/api/jobs/{jid}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1]["status"] == "succeeded"
    assert [e["progress"] for e in events] == sorted(e["progress"] for e in events)


def test_cancel(make_client):
    c = make_client(demo_seconds=30)
    jid = c.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    c.post(f"/api/jobs/{jid}/cancel")
    assert wait_done(c, jid)["status"] == "cancelled"


def test_active_job_limit(make_client):
    c = make_client(demo_seconds=30, max_active_jobs=1)
    jid = c.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    assert c.post("/api/jobs", data={"run_id": "latent"}).status_code == 429
    c.post(f"/api/jobs/{jid}/cancel")


def test_job_needs_input(client):
    assert client.post("/api/jobs", data={}).status_code == 400
    assert client.post("/api/jobs", data={"run_id": "nope"}).status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


# --- uploads -------------------------------------------------------------------------------
def test_upload_runs_demo_and_serves_input(client, runs_dir):
    r = client.post("/api/jobs", files={"file": ("my clip.mp4", MP4, "video/mp4")})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["input"]["filename"] == "my clip.mp4" and job["input"]["size"] == len(MP4)
    assert job["source_run"] == "latent"  # uploads replay the configured demo run
    inp = client.get(job["input"]["url"])
    assert inp.status_code == 200 and inp.content == MP4
    assert wait_done(client, job["id"])["status"] == "succeeded"


@pytest.mark.parametrize("name,body,ctype,code", [
    ("clip.exe", MP4, "video/mp4", 415),             # extension
    ("clip.mp4", MP4, "text/html", 415),             # declared type
    ("clip.mp4", b"<html>" + b"x" * 64, "video/mp4", 415),  # content isn't a video
    ("clip.mp4", b"", "video/mp4", 400),             # empty
])
def test_upload_validation(client, name, body, ctype, code):
    assert client.post("/api/jobs", files={"file": (name, body, ctype)}).status_code == code


def test_upload_size_limit(make_client, tmp_path):
    c = make_client(max_upload_mb=0.001)  # ~1 kB
    r = c.post("/api/jobs", files={"file": ("big.mov", MP4 + b"\x00" * 4096, "video/quicktime")})
    assert r.status_code == 413
    assert not any((tmp_path / "data" / "uploads" / "_incoming").iterdir())  # partial file removed


def test_runner_must_be_enabled(client):
    r = client.post("/api/jobs", data={"runner": "colab"}, files={"file": ("a.mp4", MP4, "video/mp4")})
    assert r.status_code == 400


def test_webm_accepted(client):
    webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 64
    assert client.post("/api/jobs", files={"file": ("a.webm", webm, "video/webm")}).status_code == 201


# --- log classifier ------------------------------------------------------------------------
def test_classify():
    assert classify("[03:58:26] target squat: viewpoint -145 deg")[0] == "latent"
    assert classify("pose found in 100% of frames") == ("pose", "pose found in 100% of frames")
    assert classify("  25/253 motions rendered + embedded (81 s)")[0] == "latent"
    assert classify("tracking : aligned 0.714, dtw 0.739")[0] == "eval"
    assert classify("random warning")[0] is None
    assert FAKE_TOKEN not in classify(f"SAM 3.1 api_key={FAKE_TOKEN}")[1]
