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
    assert c.get("/api/health").json()["store"] == {"kind": "local"} and c.get("/readyz").json()["checks"]["store"] is True
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
    c.post(f"/api/jobs/{j2}/cancel")
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
    assert fresh.get(jid).status == "succeeded" and fresh.owner_key[jid] == KEY
    assert (fresh.live_dir(fresh.get(jid)) / "env.zip").is_file()
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
    assert j.status == "failed" and "restarted" in j.error and j.training["state"] == "failed"
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

    def list_blobs(self, prefix=""):
        return [FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)]

    def exists(self):
        return True


class FakeClient:
    def __init__(self):
        self.b = FakeBucket()

    def bucket(self, name):
        return self.b


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
    assert (out / "env.zip").read_bytes() == b"zip" and not (out / "etc").exists()
    assert st.ping()
    st.delete(jid)
    assert client.b.objects == {}


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
