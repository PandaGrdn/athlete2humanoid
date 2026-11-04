"""The durable store (server/store.py): a visitor's builds and trainings survive a restart, listed under their key.

The local backend is exercised through the API with the fake live worker; the GCS backend against an in-memory
stand-in for google-cloud-storage's bucket (no emulator, no network).
"""

import json
import time
from pathlib import Path

import pytest

from server import store as S
from server.config import Settings
from server.jobs import JobStore
from server.journal import Journal

from .test_api import MP4, wait_done

KEY = "a" * 32
OTHER = "b" * 32


def upload(c, key=KEY):
    return c.post("/api/jobs", data={"runner": "live"}, files={"file": ("kick.mp4", MP4, "video/mp4")},
                  headers={"X-A2H-Owner": key} if key else {})


def test_owner_key_is_read_from_the_header():
    assert S.owner_key([(b"x-a2h-owner", KEY.encode())]) == KEY
    assert S.owner_key([(b"X-A2H-Owner", b" " + KEY.upper().encode())]) == KEY
    assert S.owner_key([(b"x-a2h-owner", b"nope")]) is None and S.owner_key([]) is None


def test_kept_paths_cannot_escape_the_job():
    assert S.safe_rel("train/attempts/iter_0050.npz") and S.safe_rel("live.json")
    for bad in ("../../etc/passwd", "a/../b", "/etc/passwd", "a/", "", "live.json\n", "..", "a/./b"):
        assert not S.safe_rel(bad), bad
    assert S.REL.match("../x") is False and S.REL.match("x/y.npz") is True


def test_store_url_setting(monkeypatch):
    monkeypatch.setenv("A2H_STORE", "s3://nope")
    with pytest.raises(Exception, match="A2H_STORE"):
        Settings().validate()
    monkeypatch.setenv("A2H_STORE", "gcs://athlete2humanoid-jobs/v1")
    s = Settings()
    s.validate()
    st = S.store_for(s)
    assert isinstance(st, S.GcsStore) and st.bucket_name == "athlete2humanoid-jobs" and st.prefix == "v1/"
    monkeypatch.delenv("A2H_STORE")
    assert isinstance(S.store_for(Settings()), S.LocalStore)


def test_a_finished_build_is_kept_and_listed(live_client, tmp_path):
    c = live_client()
    jid = upload(c).json()["id"]
    j = wait_done(c, jid)
    assert j["status"] == "succeeded" and j["kept"] is True
    persist = tmp_path / "data" / "persist" / "jobs" / jid
    assert {p.name for p in persist.iterdir()} >= {"processed.mp4", "live.json", "task_spec.json", "env.zip"}
    assert c.get("/api/health").json()["store"] == {"kind": "local", "durable": True} and c.get("/readyz").json()["checks"]["store"] is True
    assert c.get("/api/my/jobs").status_code == 400  # no key
    mine = c.get("/api/my/jobs", headers={"X-A2H-Owner": KEY}).json()
    assert mine["owner"] == KEY and [x["id"] for x in mine["jobs"]] == [jid]
    me = mine["jobs"][0]
    assert me["status"] == "succeeded" and me["filename"] == "kick.mp4" and me["kept"] and me["env"]["url"] == f"/api/jobs/{jid}/env.zip"
    assert me["summary"] and me["training"] is None
    assert c.get("/api/my/jobs", headers={"X-A2H-Owner": OTHER}).json()["jobs"] == []  # someone else's key
    # a running job of the same visitor is listed too, before it has a record
    c.mode("slow")
    j2 = upload(c).json()["id"]
    ids = [x["id"] for x in c.get("/api/my/jobs", headers={"X-A2H-Owner": KEY}).json()["jobs"]]
    assert ids == [j2, jid]
    assert c.post(f"/api/jobs/{j2}/cancel", headers={"X-A2H-Owner": OTHER}).status_code == 403  # not theirs
    assert c.post(f"/api/jobs/{j2}/cancel").status_code == 403  # a key-made job needs its key
    assert c.post(f"/api/jobs/{j2}/cancel", headers={"X-A2H-Owner": KEY}).status_code == 200
    wait_done(c, j2)


def test_a_kept_build_survives_a_restart_and_expiry(live_client, tmp_path):
    c = live_client()
    jid = upload(c).json()["id"]
    wait_done(c, jid)
    app = c.app
    store = app.state.jobs
    # expiry sweeps the upload and the job out of memory and the journal; the store keeps it
    with store.lock:
        store.jobs[jid].updated_at -= 400 * 3600
    store.sweep()
    assert jid not in store.jobs and not store.upload_dir(jid).exists()
    j = c.get(f"/api/jobs/{jid}").json()
    assert j["status"] == "succeeded" and j["kept"] is True and j["input"]["filename"] == "kick.mp4"
    assert c.get(f"/api/jobs/{jid}/live").json()["frames"] == 60  # live.json came back from the store
    assert c.get(f"/api/jobs/{jid}/env.zip").status_code == 200
    assert c.get(f"/api/jobs/{jid}/task_spec").status_code == 200
    assert [x["id"] for x in c.get("/api/my/jobs", headers={"X-A2H-Owner": KEY}).json()["jobs"]] == [jid]
    # a fresh process on the same host, with the journal gone (a new Cloud Run revision)
    store.close()
    (tmp_path / "data" / "jobs.sqlite3").unlink()
    s = Settings()
    s.runs_dir, s.data_dir, s.runners, s.hosted, s.live_mode = app.state.settings.runs_dir, tmp_path / "data", ["demo"], False, "0"
    fresh = JobStore(s, Journal(s.data_dir / "jobs.sqlite3"))
    j = fresh.get(jid)
    assert j.status == "succeeded" and fresh.owner_key[jid] == KEY and time.time() - j.updated_at < 5  # not the record's age
    live = fresh.live_dir(j)
    assert (live / "env.zip").is_file() and (live / "live.json").is_file()
    (live / "env.zip").unlink()  # a file gone (or one that failed to come) is fetched again on the next access
    assert (fresh.live_dir(j) / "env.zip").is_file()
    assert [r["id"] for r in fresh.mine(KEY)] == [jid] and fresh.mine(OTHER) == []
    fresh.close()


def test_a_job_recorded_mid_way_is_failed_when_revived(tmp_path):
    st = S.LocalStore(tmp_path / "persist")
    st.save_record("c" * 32, KEY, {"id": "c" * 32, "runner": "live", "status": "running", "stages": [], "logs": [], "curves": {},
                                    "created_at": 1.0, "updated_at": 2.0, "version": 1, "kept": ["live.json"],
                                    "training": {"state": "running"}})
    s = Settings()
    s.data_dir, s.live_mode, s.store_backend = tmp_path / "data", "0", st
    store = JobStore(s)
    j = store.get("c" * 32)
    assert j.status == "succeeded" and "restarted" in j.error and j.training["state"] == "failed"  # the build stands
    assert store.get("d" * 32) is None and store.get("../x") is None
    store.close()


def test_no_key_means_no_listing(live_client):
    c = live_client()
    jid = upload(c, key=None).json()["id"]
    wait_done(c, jid)
    assert c.get("/api/my/jobs", headers={"X-A2H-Owner": KEY}).json()["jobs"] == []
    assert c.get(f"/api/jobs/{jid}").json()["kept"] is True  # kept all the same, under its id


# --- the GCS backend against a stand-in bucket ---------------------------------------------------------------
class FakeBlob:
    def __init__(self, bucket, name):
        self.b, self.name = bucket, name

    def exists(self):
        return self.name in self.b.objects

    def upload_from_filename(self, path):
        self.b.objects[self.name] = Path(path).read_bytes()

    def upload_from_string(self, data, content_type=None):
        self.b.objects[self.name] = data.encode() if isinstance(data, str) else data

    def download_to_filename(self, path):
        Path(path).write_bytes(self.b.objects[self.name])

    def download_as_bytes(self):
        return self.b.objects[self.name]

    def delete(self):
        self.b.objects.pop(self.name, None)


class FakeBucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self, name)

    def list_blobs(self, prefix="", max_results=None):
        out = [FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)]
        return out[:max_results] if max_results else out

    def exists(self):
        return True


class FakeClient:
    def __init__(self):
        self.b = FakeBucket()

    def bucket(self, name):
        return self.b


def test_restart_mid_training_counts_and_cancels(tmp_path, runs_dir):
    """(1, 2) A training started by an instance that died is in the store's index: the next instance counts it for
    the day, cancels its call when it meets the job, and never sees it as running."""
    from .test_train import FakeBackend
    from .test_live import FAKE
    from .test_task_spec import keypose_spec
    from .test_api import MP4
    import sys
    # a live client with the modal runner and a daily allowance of 1, no code
    (tmp_path / "mode").write_text("ok")
    (tmp_path / "input.mp4").write_bytes(MP4)
    (tmp_path / "spec.json").write_text(json.dumps(keypose_spec()))
    (tmp_path / "fake_worker.py").write_text(FAKE.format(mode=str(tmp_path / "mode"), mp4=str(tmp_path / "input.mp4"), spec=str(tmp_path / "spec.json")))
    py = tmp_path / "fakepy"
    py.write_text(f"#!/bin/sh\nexec {sys.executable} {tmp_path / 'fake_worker.py'} \"$@\"\n")
    py.chmod(0o755)
    from fastapi.testclient import TestClient
    from server.main import create_app
    fake = FakeBackend()
    s = Settings()
    s.runs_dir, s.data_dir, s.runners, s.hosted, s.live_mode, s.live_python = runs_dir, tmp_path / "data", ["demo", "modal"], False, "1", str(py)
    s.modal_token, s.train_code, s.train_per_day, s.train_backend, s.train_poll_s, s.rate_train_per_hour = True, "", 1, fake, 0.05, 1000
    c = TestClient(create_app(s))
    jid = upload(c).json()["id"]
    wait_done(c, jid)
    assert c.post(f"/api/jobs/{jid}/train").status_code == 200
    store = c.app.state.jobs
    end = time.time() + 5
    while time.time() < end and not ((store.persist.get_record(jid) or {}).get("training") or {}).get("call_id"):
        time.sleep(0.02)
    idx = store.persist.trainings(0)
    assert len(idx) == 1 and idx[0]["state"] == "running" and idx[0]["call_id"] == "fc-1"
    rec = store.persist.get_record(jid)
    assert rec["training"]["state"] == "running" and rec["training"]["call_id"] == "fc-1"
    # the instance dies: its journal is gone, the store stays; a new instance comes up on it
    die(store)
    (s.data_dir / "jobs.sqlite3").unlink()
    fake2 = FakeBackend()
    s2 = Settings()
    s2.runs_dir, s2.data_dir, s2.runners, s2.hosted, s2.live_mode = runs_dir, tmp_path / "data", ["demo", "modal"], False, "0"
    s2.modal_token, s2.train_code, s2.train_per_day, s2.train_backend, s2.store_backend = True, "", 1, fake2, store.persist
    fresh = JobStore(s2, Journal(s2.data_dir / "jobs.sqlite3"), train_backend=fake2)
    assert fresh.train.today() == 1  # the day's allowance is spent: no second GPU job per idle cycle
    fresh.train.stale_s = 0.2  # a heartbeat older than this means the follower is gone (90 s for real)
    assert fresh.train.active_count() == 1 and fake2.cancelled == []  # the heartbeat is fresh still: left alone
    time.sleep(0.3)
    assert fresh.train.active_count() == 0  # ... stale now: a stray, stopped
    assert fake2.cancelled == ["fc-1"] and fake2.removed == [f"live_{jid[:12]}"]
    j = fresh.get(jid)
    assert j.status == "succeeded" and j.training["state"] == "failed" and "restarted" in j.error
    assert fresh.persist.trainings(0)[0]["state"] == "failed"
    fresh.close()


def die(store):
    """What a dead process leaves: no more heartbeats, no record of an end (the thread is told to stop quietly)."""
    store.train._index = lambda job: None
    store.train._end = lambda *a, **k: None
    store.save_record = lambda job: None
    store.stop.set()
    for ev in store.cancel.values():
        ev.set()
    time.sleep(0.15)


def test_a_second_live_instance_leaves_a_followed_training_alone(live_client, tmp_path):
    """(C) Two instances during a rollout: the second counts the first's training as running and cancels nothing
    while the first heartbeats; once the heartbeat goes stale it is a stray."""
    from .test_train import FakeBackend, built, wait_for
    fake = FakeBackend()
    c = live_client(runners=["demo", "modal"], modal_token=True, train_code="", train_per_day=5, train_backend=fake,
                    train_poll_s=0.05, rate_train_per_hour=1000)
    jid = built(c)
    assert c.post(f"/api/jobs/{jid}/train").status_code == 200
    wait_for(lambda: c.get(f"/api/jobs/{jid}").json()["training"]["state"] == "running")
    st = c.app.state.settings
    s = Settings()
    s.runs_dir, s.data_dir, s.runners, s.hosted, s.live_mode = st.runs_dir, tmp_path / "inst2", ["demo", "modal"], False, "0"
    s.data_dir.mkdir()
    s.modal_token, s.train_per_day, s.store_backend = True, 5, c.app.state.jobs.persist
    fake2 = FakeBackend()
    other = JobStore(s, Journal(s.data_dir / "jobs.sqlite3"), train_backend=fake2)
    other.train.stale_s = 0.5
    for _ in range(3):
        assert other.train.active_count() == 1 and other.train.today() == 1
        time.sleep(0.1)
    assert fake2.cancelled == [] and c.get(f"/api/jobs/{jid}").json()["training"]["state"] == "running"
    j2 = other.get(jid)  # a GET on instance 2 revives it as running, followed elsewhere
    assert j2.training["state"] == "running" and fake2.cancelled == []
    die(c.app.state.jobs)
    time.sleep(0.6)
    assert other.train.active_count() == 0 and fake2.cancelled == ["fc-1"]
    assert other.get(jid).training["state"] == "failed"
    other.close()


def test_the_allowance_counts_trainings_not_jobs(live_client):
    """(B) A retry on the same job is a training too: with two a day, the third start on one job is refused."""
    from .test_train import FakeBackend, built, wait_for
    fake = FakeBackend()
    c = live_client(runners=["demo", "modal"], modal_token=True, train_code="", train_per_day=2, train_backend=fake,
                    train_poll_s=0.05, rate_train_per_hour=1000)
    jid = built(c)
    for n in (1, 2):
        assert c.post(f"/api/jobs/{jid}/train").status_code == 200
        wait_for(lambda: len(fake.spawned) == n)
        c.post(f"/api/jobs/{jid}/cancel")
        wait_done(c, jid)
    assert c.app.state.jobs.train.today() == 2 and c.get("/api/health").json()["train"]["left_today"] == 0
    r = c.post(f"/api/jobs/{jid}/train")
    assert r.status_code == 403 and "used up" in r.json()["detail"] and len(fake.spawned) == 2
    assert len(c.app.state.jobs.persist.trainings(0)) == 2  # one row per training


def test_local_store_prunes_by_age_and_size(tmp_path):
    st = S.LocalStore(tmp_path / "p", ttl_s=100, max_bytes=5)
    f = tmp_path / "f"
    for i, jid in enumerate(("1" * 32, "2" * 32, "3" * 32)):
        f.write_bytes(b"xx")
        st.put(jid, "live.json", f)
        st.save_record(jid, KEY, {"id": jid})
        with st.lock:
            st.db.execute("UPDATE records SET updated_at = ? WHERE id = ?", (1000 + i, jid))
    assert st.prune(now=1050) == 1  # 6 bytes > 5: the oldest goes
    assert [r["id"] for r in st.records_for(KEY)] == ["3" * 32, "2" * 32]
    assert st.prune(now=1300) == 2  # all older than 100 s
    assert st.records_for(KEY) == [] and st.files("3" * 32) == []
    st.close()


def test_gcs_store_layout(tmp_path):
    client = FakeClient()
    st = S.GcsStore("bkt", "v1", client=client)
    src = tmp_path / "live"
    src.mkdir()
    (src / "live.json").write_text("{}")
    (src / "env.zip").write_bytes(b"zip")
    (src / "train" / "attempts").mkdir(parents=True)
    (src / "train" / "attempts" / "iter_0050.npz").write_bytes(b"npz")
    (src / "train" / "attempts" / "junk.txt").write_bytes(b"no")
    jid = "e" * 32
    assert st.keep(jid, src) == ["live.json", "env.zip", "train/attempts/iter_0050.npz"]
    st.save_record(jid, KEY, {"id": jid, "updated_at": 5})
    assert set(client.b.objects) == {f"v1/jobs/{jid}/live.json", f"v1/jobs/{jid}/env.zip", f"v1/jobs/{jid}/train/attempts/iter_0050.npz",
                                     f"v1/jobs/{jid}/record.json", f"v1/owners/{KEY}/{jid}"}
    assert st.files(jid) == ["env.zip", "live.json", "train/attempts/iter_0050.npz"]
    rec = {"id": jid, "updated_at": 5, "owner": KEY}
    assert st.get_record(jid) == rec and st.get_record("f" * 32) is None
    assert st.records_for(KEY) == [rec] and st.records_for(OTHER) == []
    out = tmp_path / "again"
    assert st.materialize(jid, out, ["env.zip", "train/attempts/iter_0050.npz", "../etc/passwd"]) == 2
    assert (out / "env.zip").read_bytes() == b"zip" and not (out / "etc").exists() and not list(out.glob("*.part"))
    assert st.ping()
    st.put_training("live_x_1", {"job": jid, "session": "live_x_1", "started_at": time.time(), "state": "running", "call_id": "fc-1"})
    assert [t["call_id"] for t in st.trainings(time.time() - 60)] == ["fc-1"] and st.trainings(time.time() + 60) == []
    st.delete(jid)
    assert set(client.b.objects) == {"v1/trainings/live_x_1"}


def test_local_store_round_trip(tmp_path):
    st = S.LocalStore(tmp_path / "p")
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    st.put("g" * 32, "train/policy.pt", f)
    assert st.files("g" * 32) == ["train/policy.pt"]
    dest = tmp_path / "out" / "policy.pt"
    dest.parent.mkdir()
    assert st.fetch("g" * 32, "train/policy.pt", dest) and dest.read_bytes() == b"abc"
    assert not st.fetch("g" * 32, "nope", tmp_path / "x")
    st.save_record("g" * 32, KEY, {"id": "g" * 32, "t": time.time()})
    assert st.records_for(KEY)[0]["id"] == "g" * 32
    st.delete("g" * 32)
    assert st.files("g" * 32) == [] and st.get_record("g" * 32) is None
    st.close()
    assert json.loads(json.dumps({"ok": True}))  # the module needs nothing beyond the standard library here
