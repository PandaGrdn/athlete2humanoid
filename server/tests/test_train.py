"""The modal runner (server/train.py): a finished live build trained on a GPU, with a stand-in Modal.

The fake backend keeps the Volume as a dict and lets a test play the GPU job: it writes the job's log, the
attempts' index and files, weights.json and the checkpoint, and sets the call's state. No modal client, no GPU.
"""

import json
import re
import time
import zipfile
from pathlib import Path

import pytest

from server import train as T
from server.config import Settings
from server.jobs import JobStore
from server.journal import Journal

from .test_api import REPO, wait_done
from .test_live import upload

CODE = "kick-2026"


class FakeBackend:
    def __init__(self):
        self.files, self.spawned, self.cancelled, self.removed = {}, [], [], []
        self.state_, self.ok = "running", (True, "ok")

    def status(self, max_age=600.0):
        return self.ok

    def upload(self, session, files):
        for name, p in files.items():
            self.files[f"runs/{session}/inputs/{name}"] = Path(p).read_bytes()

    def spawn(self, session, cmd, branch):
        self.spawned.append((session, cmd, branch))
        return f"fc-{len(self.spawned)}"

    def state(self, call_id):
        return self.state_

    def cancel(self, call_id):
        self.cancelled.append(call_id)

    def read(self, session, rel, limit=T.MAX_LOG_BYTES):
        return self.files.get(f"runs/{session}/{rel}")

    def download(self, session, rel, local, limit=T.MAX_ATTEMPT_BYTES):
        data = self.read(session, rel)
        if data is None:
            return False
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(data)
        return True

    def remove(self, session):
        self.removed.append(session)
        return True

    # --- the GPU job, played by the test ---------------------------------------------------------
    def put(self, session, rel, data):
        self.files[f"runs/{session}/{rel}"] = data if isinstance(data, bytes) else data.encode()

    def job_log(self, session, upto, total=1000):
        lines = []
        for it in range(upto):
            lines.append(f"\x1b[1m  Learning iteration {it}/{total}  \x1b[0m\n  Mean reward: {it * 0.01:.2f}\n")
        self.put(session, f"jobs/{session}/log", "rendering with egl\n" + "".join(lines))

    def attempts(self, session, iterations, every=50, total=1000):
        rows = [{"iteration": it, "file": f"iter_{it:04d}.npz", "frames": 30, "seconds": 0.6, "reward": it / 100,
                 "reward_per_step": 0.01, "fell": it < 100, "fell_at_s": 0.6 if it < 100 else None,
                 "stages_completed": 1 + it // 100, "stages_total": 6, "success": it >= 500} for it in iterations]
        self.put(session, "outputs/policy/attempts/index.json", json.dumps({
            "format": "a2h-attempts", "fps": 50, "steps": 276, "iterations": total, "attempt_every": every,
            "task": "A2H-Tracking-Flat-Unitree-G1", "policy": "tracking_var", "attempts": rows,
            "curve": [[i, i * 0.01] for i in range(1, max(iterations) + 1)]}))
        for it in iterations:
            self.put(session, f"outputs/policy/attempts/iter_{it:04d}.npz", b"PK\x03\x04npz-" + str(it).encode())

    def finish(self, session, rc=0):
        self.put(session, "outputs/policy/weights.json", json.dumps({"checkpoint": "logs/2026-11-01/model_999.pt", "iterations": 1000}))
        self.put(session, "outputs/policy/logs/2026-11-01/model_999.pt", b"pt-weights")
        self.put(session, f"jobs/{session}/status.json", json.dumps({"state": "done" if rc == 0 else "failed", "returncode": rc}))
        self.state_ = "done" if rc == 0 else "failed"


@pytest.fixture
def train_client(live_client):
    """live_client with the modal runner on: (client, fake backend)."""
    def make(**kw):
        fake = FakeBackend()
        opts = dict(runners=["demo", "modal"], modal_token=True, train_code=CODE, train_backend=fake, train_poll_s=0.05,
                    train_attempt_every=50, train_iters=1000, rate_train_per_hour=100000) | kw
        c = live_client(**opts)
        return c, fake
    return make


def built(c):
    """A finished live build (the fake worker wrote env.zip)."""
    jid = upload(c).json()["id"]
    j = wait_done(c, jid)
    assert j["status"] == "succeeded" and j["stages"][-1]["status"] == "skipped"
    return jid


def wait_for(fn, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError("timed out")


# --- offering it -------------------------------------------------------------------------------------
def test_health_offers_training_only_with_a_token_and_a_gate(live_client, train_client):
    c = live_client(runners=["demo", "modal"], modal_token=False, train_code=CODE)
    h = c.get("/api/health").json()
    assert "modal" not in h["runners"] and h["train"] is None
    c = live_client(runners=["demo", "modal"], modal_token=True, train_backend=FakeBackend())  # no code, no allowance
    assert c.get("/api/health").json()["train"] is None
    c, fake = train_client()
    h = c.get("/api/health").json()
    assert h["runners"] == ["demo", "live", "modal"]
    assert h["train"] == {"backend": "modal", "code_required": True, "code_accepted": True, "per_day": 0, "left_today": 0,
                          "iterations": 1000, "attempt_every": 50, "minutes": 30, "busy": False}
    assert c.get("/readyz").json()["train_backend"] == "ok"
    fake.ok = (False, "Modal app athlete2humanoid not deployed or unreachable (NotFoundError)")
    h = c.get("/api/health").json()
    assert "modal" not in h["runners"] and h["train"] is None and h["degraded"] == ["modal runner unavailable: " + fake.ok[1]]


def test_settings_warn_without_a_token_or_a_gate(monkeypatch):
    monkeypatch.setenv("A2H_RUNNERS", "demo,modal")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    s = Settings()
    assert any("MODAL_TOKEN_ID" in w for w in s.validate()) and "modal" not in s.enabled_runners
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-y")
    s = Settings()
    assert any("A2H_TRAIN_CODE" in w for w in s.validate())
    monkeypatch.setenv("A2H_TRAIN_PER_DAY", "2")
    s = Settings()
    s.live_ok = True
    assert not s.validate() and "modal" in s.enabled_runners and s.train_open


def test_the_gate(train_client):
    c, fake = train_client()
    jid = built(c)
    assert c.post(f"/api/jobs/{jid}/train").status_code == 403  # no code
    r = c.post(f"/api/jobs/{jid}/train", data={"code": "nope"})
    assert r.status_code == 403 and "invite code" in r.json()["detail"]
    assert c.post("/api/jobs/nope/train", data={"code": CODE}).status_code == 404
    r = c.post("/api/jobs", data={"runner": "demo", "run_id": "latent"})
    assert r.status_code == 201, r.text
    demo = r.json()["id"]
    assert c.post(f"/api/jobs/{demo}/train", data={"code": CODE}).status_code == 404  # not a live job
    assert c.post("/api/jobs", data={"runner": "modal"}).status_code == 400  # it trains a live build, not an upload
    assert fake.spawned == []


def test_a_daily_allowance_without_a_code(train_client):
    c, fake = train_client(train_code="", train_per_day=1)
    t = c.get("/api/health").json()["train"]
    assert (t["code_required"], t["per_day"], t["left_today"]) == (False, 1, 1)
    a, b = built(c), built(c)
    assert c.post(f"/api/jobs/{a}/train").status_code == 200
    assert c.get("/api/health").json()["train"]["left_today"] == 0
    r = c.post(f"/api/jobs/{b}/train")
    assert r.status_code == 403 and "used up" in r.json()["detail"]
    fake.finish(f"live_{a[:12]}")
    wait_done(c, a)
    assert c.post(f"/api/jobs/{b}/train").status_code == 403  # the day's allowance, not the queue


# --- running one ---------------------------------------------------------------------------------------
def test_training_follows_the_gpu_job(train_client):
    c, fake = train_client()
    jid = built(c)
    j = c.post(f"/api/jobs/{jid}/train", data={"code": CODE}).json()
    assert j["status"] == "running" and j["training"]["state"] in ("starting", "running") and j["training"]["iterations"] == 1000
    r = c.post(f"/api/jobs/{jid}/train", data={"code": CODE})
    assert r.status_code == 409 and "training already" in r.json()["detail"]
    assert c.get("/api/health").json()["train"]["busy"] is True
    session = f"live_{jid[:12]}"
    wait_for(lambda: fake.spawned)
    (s, cmd, branch), = fake.spawned
    assert s == session and branch == "main"
    assert cmd == (f"$A2H_MJLAB_PYTHON -m a2h_env train /vol/runs/{session}/inputs/env.zip --iters 1000 --num-envs 4096 "
                   f"--attempt-every 50 --out /vol/runs/{session}/outputs/policy --work-dir /vol/runs/{session}/outputs/motions")
    inputs = {k.split("/inputs/")[1] for k in fake.files if "/inputs/" in k}
    assert {"env.zip", "task_spec.json", "processed.mp4"} <= inputs and "HF_TOKEN" not in str(fake.files)
    j = wait_for(lambda: (x := c.get(f"/api/jobs/{jid}").json()) and x["training"]["state"] == "running" and x)
    assert j["stages"][-1]["status"] == "running" and j["stage"] == "policy"
    assert c.get(f"/api/jobs/{jid}/attempts").json()["attempts"] == []

    # the GPU job runs: 120 iterations logged, two attempts written
    fake.job_log(session, 120)
    fake.attempts(session, [50, 100])
    j = wait_for(lambda: (x := c.get(f"/api/jobs/{jid}").json()) and x["training"]["attempts"] == 2 and x)
    assert j["training"]["iteration"] == 120 and j["training"]["reward"] == pytest.approx(1.19)
    assert j["training"]["last"]["iteration"] == 100 and j["training"]["last"]["stages_completed"] == 2
    assert j["curves"]["policy"][0] == [1, 0.0] and j["curves"]["policy"][-1] == [120, 1.19]
    assert 0.1 < j["stages"][-1]["progress"] < 0.2 and 0.4 < j["progress"] < 0.6
    lines = [line for _, st, line in j["logs"] if st == "policy"]
    assert any(line.startswith("iteration 120/1000: mean reward 1.19") for line in lines)
    assert any("attempt at iteration 100: reward 1.00, 2/6 key moments" in line for line in lines)
    assert not any("Learning iteration" in line for line in lines)
    a = c.get(f"/api/jobs/{jid}/attempts").json()
    assert [x["iteration"] for x in a["attempts"]] == [50, 100] and a["attempt_every"] == 50 and a["iterations"] == 1000
    want = {"reward": 0.5, "fell": True, "success": False, "stages_total": 6, "url": f"/api/jobs/{jid}/attempts/iter_0050.npz"}
    assert {k: a["attempts"][0][k] for k in want} == want
    assert a["curve"][-1] == [100, 1.0] and a["training"]["state"] == "running"
    r = c.get(f"/api/jobs/{jid}/attempts/iter_0050.npz")
    assert r.status_code == 200 and r.content == b"PK\x03\x04npz-50" and r.headers["cache-control"] == "private, max-age=86400, immutable"
    assert c.get(f"/api/jobs/{jid}/attempts/iter_0075.npz").status_code == 404
    assert c.get(f"/api/jobs/{jid}/attempts/../index.json").status_code == 404
    assert c.get(f"/api/jobs/{jid}/policy.pt").status_code == 404

    # ... and ends: the last attempt, weights and the checkpoint come back, the Volume folder goes
    fake.job_log(session, 1000)
    fake.attempts(session, [50, 100, 1000])
    fake.finish(session)
    j = wait_done(c, jid)
    assert j["status"] == "succeeded" and j["training"]["state"] == "done" and j["training"]["checkpoint"] == "policy.pt"
    assert j["stages"][-1]["status"] == "done" and j["progress"] == 1.0 and j["error"] is None
    assert any("training finished: 3 attempts recorded, final reward 10.00" in line for _, _, line in j["logs"])
    assert fake.removed == [session]
    a = c.get(f"/api/jobs/{jid}/attempts").json()
    assert [x["iteration"] for x in a["attempts"]] == [50, 100, 1000] and a["attempts"][-1]["success"] is True
    r = c.get(f"/api/jobs/{jid}/policy.pt")
    assert r.status_code == 200 and r.content == b"pt-weights" and "attachment" in r.headers["content-disposition"]
    assert c.get("/api/health").json()["train"]["busy"] is False
    # the job list (local) carries the training state without the log
    js = {x["id"]: x for x in c.get("/api/jobs").json()}
    assert js[jid]["training"]["attempts"] == 3


def test_cancel_stops_the_gpu_job(train_client):
    c, fake = train_client()
    jid = built(c)
    c.post(f"/api/jobs/{jid}/train", data={"code": CODE})
    wait_for(lambda: fake.spawned)
    c.post(f"/api/jobs/{jid}/cancel")
    j = wait_done(c, jid)
    # the build stays succeeded; the training is the thing cancelled, and it can be started again
    assert j["status"] == "succeeded" and j["training"]["state"] == "cancelled" and j["stages"][-1]["status"] == "skipped"
    assert fake.cancelled == ["fc-1"] and fake.removed == [f"live_{jid[:12]}"]
    assert c.post(f"/api/jobs/{jid}/train", data={"code": CODE}).status_code == 200
    wait_for(lambda: len(fake.spawned) == 2)
    assert fake.spawned[1][0] == f"live_{jid[:12]}_1"  # its own folder on the Volume


def test_a_failed_gpu_job_fails_the_training(train_client):
    c, fake = train_client()
    jid = built(c)
    c.post(f"/api/jobs/{jid}/train", data={"code": CODE})
    wait_for(lambda: fake.spawned)
    session = f"live_{jid[:12]}"
    fake.put(session, f"jobs/{session}/status.json", json.dumps({"state": "failed", "returncode": 1}))
    fake.state_ = "failed"
    j = wait_done(c, jid)
    assert j["status"] == "succeeded" and j["training"]["state"] == "failed"
    assert j["error"] == j["training"]["error"] == "the GPU job failed (job state failed); no policy came back"
    assert j["stages"][-1]["status"] == "failed" and fake.removed == [session]
    assert c.get(f"/api/jobs/{jid}/env.zip").status_code == 200  # the build is still there
    # a failed training can be tried again (9); a trained one can't
    assert c.post(f"/api/jobs/{jid}/train", data={"code": CODE}).status_code == 200
    wait_for(lambda: len(fake.spawned) == 2)
    j = c.get(f"/api/jobs/{jid}").json()
    assert j["training"]["state"] in ("starting", "running") and j["training"]["history"][0]["state"] == "failed"
    fake.finish(f"live_{jid[:12]}_1")
    j = wait_done(c, jid)
    assert j["training"]["state"] == "done" and j["status"] == "succeeded" and j["error"] is None
    r = c.post(f"/api/jobs/{jid}/train", data={"code": CODE})
    assert r.status_code == 409 and "trained" in r.json()["detail"]


def test_a_refused_spawn_cleans_up(train_client):
    c, fake = train_client()
    fake.spawn = lambda *a: (_ for _ in ()).throw(RuntimeError("spend limit"))
    jid = built(c)
    c.post(f"/api/jobs/{jid}/train", data={"code": CODE})
    j = wait_done(c, jid)
    assert j["training"]["state"] == "failed" and "didn't take the job" in j["error"] and fake.removed == [f"live_{jid[:12]}"]


def test_one_training_at_a_time(train_client):
    c, fake = train_client()
    a, b = built(c), built(c)
    assert c.post(f"/api/jobs/{a}/train", data={"code": CODE}).status_code == 200
    r = c.post(f"/api/jobs/{b}/train", data={"code": CODE})
    assert r.status_code == 429 and "another visitor" in r.json()["detail"] and r.headers["retry-after"]
    fake.finish(f"live_{a[:12]}")
    wait_done(c, a)
    assert c.post(f"/api/jobs/{b}/train", data={"code": CODE}).status_code == 200


def test_timeout_stops_a_runaway_job(train_client):
    c, fake = train_client()
    jid = built(c)
    c.app.state.settings.train_timeout_h = 0.00001  # 36 ms (too small for the startup check, set after it)
    c.post(f"/api/jobs/{jid}/train", data={"code": CODE})
    j = wait_done(c, jid)
    assert j["training"]["state"] == "failed" and "ran past" in j["error"] and fake.cancelled == ["fc-1"] and fake.removed == [f"live_{jid[:12]}"]


def test_a_crash_after_the_spawn_cancels_the_gpu_job(train_client):
    """(3) Whatever ends the follow loop, the call is cancelled unless it is known to have finished."""
    c, fake = train_client()
    jid = built(c)
    fake.read = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    c.post(f"/api/jobs/{jid}/train", data={"code": CODE})
    j = wait_done(c, jid)
    assert j["training"]["state"] == "failed" and "internal error" in j["error"]
    assert fake.cancelled == ["fc-1"] and fake.removed == [f"live_{jid[:12]}"]


def test_lost_contact_is_not_a_failed_job(train_client):
    """(3) A gRPC-style error from the call is this host's problem: the follow goes on, and a long loss stops the job."""
    c, fake = train_client()
    jid = built(c)
    c.post(f"/api/jobs/{jid}/train", data={"code": CODE})
    wait_for(lambda: fake.spawned)

    class Unavailable(Exception):
        pass
    fake.state_ = "unknown"  # what the real backend answers to a gRPC-style error (below)
    real = T.ModalBackend()
    real._call = lambda cid: type("C", (), {"get": lambda self, timeout=0: (_ for _ in ()).throw(Unavailable("x"))})()
    assert real.state("fc-1") == "unknown"
    real._call = lambda cid: type("C", (), {"get": lambda self, timeout=0: (_ for _ in ()).throw(ValueError("cancelled"))})()
    assert real.state("fc-1") == "failed"
    time.sleep(0.3)  # several polls of "unknown"
    j = c.get(f"/api/jobs/{jid}").json()
    assert j["training"]["state"] == "running" and fake.cancelled == []
    fake.finish(f"live_{jid[:12]}")
    assert wait_done(c, jid)["training"]["state"] == "done"


def test_shutdown_cancels_running_trainings(train_client):
    """(1) A Cloud Run instance going away must not leave its GPU job running unwatched."""
    c, fake = train_client()
    jid = built(c)
    c.post(f"/api/jobs/{jid}/train", data={"code": CODE})
    wait_for(lambda: fake.spawned)
    c.app.state.jobs.close()
    j = c.app.state.jobs.get(jid)
    assert j.training["state"] == "failed" and "server stopped" in j.error and j.status == "succeeded"
    assert set(fake.cancelled) == {"fc-1"} and f"live_{jid[:12]}" in fake.removed  # (the follow thread's exit cancels too)


def test_train_requests_are_rate_limited_per_client(train_client):
    """(11) Guessing the invite code is bounded by its own bucket, not the general one."""
    c, fake = train_client(rate_train_per_hour=3)
    jid = built(c)
    codes = [c.post(f"/api/jobs/{jid}/train", data={"code": "nope"}).status_code for _ in range(5)]
    assert codes[:3] == [403, 403, 403] and codes[3] == 429 and codes[4] == 429


def test_restart_stops_a_training_it_cannot_follow(tmp_path, runs_dir):
    s = Settings()
    s.runs_dir, s.data_dir, s.runners, s.hosted, s.live_mode = runs_dir, tmp_path / "data", ["demo", "modal"], False, "0"
    journal = Journal(s.data_dir / "jobs.sqlite3")
    body = {"id": "a" * 32, "runner": "live", "status": "running", "stage": "policy", "stages": [], "logs": [], "curves": {},
            "created_at": 1.0, "updated_at": 2.0, "version": 3,
            "training": {"backend": "modal", "state": "running", "session": "live_aaaaaaaaaaaa", "call_id": "fc-9",
                         "started_at": time.time() - 100, "attempts": 1}}
    journal.save([("a" * 32, "running", 1.0, 2.0, None, body)])
    fake = FakeBackend()
    store = JobStore(s, journal, train_backend=fake)
    j = store.get("a" * 32)
    # (2) the training is marked failed, the build stays succeeded, nothing counts as running any more
    assert j.status == "succeeded" and "GPU job was stopped" in j.error and j.training["state"] == "failed"
    assert fake.cancelled == ["fc-9"] and fake.removed == ["live_aaaaaaaaaaaa"] and store.train.active_count() == 0
    # (A) ... and the store says so too, so the job doesn't come back as "training" once it leaves memory
    rec = store.persist.get_record("a" * 32)
    assert rec["training"]["state"] == "failed" and rec["status"] == "succeeded"
    assert [r["state"] for r in store.persist.trainings(0)] == ["failed"]
    store.close()
    journal.close()


# --- stored runs -----------------------------------------------------------------------------------------
def test_run_attempts(client, runs_dir):
    run = runs_dir / "remote_latent"
    assert client.get("/api/runs/latent/attempts").status_code == 404 and client.get("/api/runs/latent").json()["attempts"] is None
    att = run / "attempts"
    att.mkdir()
    rows = [{"iteration": 50, "file": "iter_0050.npz", "reward": 1.5, "fell": True, "stages_completed": 1, "stages_total": 6, "success": False},
            {"iteration": 100, "file": "iter_0100.npz", "reward": 3.0, "fell": False, "stages_completed": 6, "stages_total": 6, "success": True},
            {"iteration": 150, "file": "iter_0150.npz", "reward": 4.0},  # listed, not here: left out
            {"iteration": 7, "file": "../eval.json"}]
    (att / "index.json").write_text(json.dumps({"iterations": 1000, "attempt_every": 50, "fps": 50, "policy": "tracking_var",
                                                "attempts": rows, "curve": [[1, 0.1], [2, 0.2], ["x", 1]]}))
    (att / "iter_0050.npz").write_bytes(b"a")
    (att / "iter_0100.npz").write_bytes(b"b")
    v = client.get("/api/runs/latent/attempts").json()
    assert [x["iteration"] for x in v["attempts"]] == [50, 100] and v["curve"] == [[1, 0.1], [2, 0.2]]
    assert v["attempts"][1]["url"] == "/api/runs/latent/files/attempts/iter_0100.npz" and v["policy"] == "tracking_var"
    assert client.get(v["attempts"][1]["url"]).content == b"b"
    assert client.get("/api/runs/latent").json()["attempts"]["iterations"] == 1000


def test_names_match_the_modal_app():
    src = (REPO / "scripts" / "modal_app.py").read_text()
    m = re.search(r'APP, VOLUME, MOUNT = "([^"]+)", "([^"]+)", "([^"]+)"', src)
    assert (T.APP, T.VOLUME, T.MOUNT) == m.groups() and f"def {T.FUNCTION}(" in src
    assert T.INPUTS[0] == "env.zip"
    with zipfile.ZipFile(REPO / "tests" / "data" / "remote_latent" / "eval.json", "r") if False else open(REPO / "a2h_env" / "attempts.py") as f:
        assert "iter_{:04d}.npz" in f.read()  # the file names the server mirrors (T.NPZ)
