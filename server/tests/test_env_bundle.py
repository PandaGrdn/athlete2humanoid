"""The training environment the API hands out: GET /api/runs/{id}/env.zip (built on demand from the run folder and
cached), its entry in the run detail, and a live job's /api/jobs/{id}/env.zip (server/live_worker.py writes it)."""

import io
import json
import os
import shutil
import zipfile

from a2h_env.export import npy_header

from .test_api import REPO, wait_done
from .test_live import live_client, upload  # noqa: F401
from .test_task_spec import keypose_spec

QPOS = REPO / "tests" / "data" / "qpos_front_kick.npy"  # copied as bytes: the server (and CI's server job) has no numpy


def give_spec_and_retarget(runs_dir, run="remote_latent"):
    """The fixture run gets what a bundle needs: a task spec naming its clip, and the clip's retarget. -> frames"""
    d = runs_dir / run
    (d / "task_spec.json").write_text(json.dumps({**keypose_spec(), "name": "squat"}))
    shutil.copy(QPOS, d / "clips" / "squat" / "qpos.npy")
    return npy_header(QPOS.read_bytes())[0][0]


def test_run_env_zip(client, runs_dir):
    assert client.get("/api/runs/latent").json()["env"] is None  # no task spec yet
    assert client.get("/api/runs/latent/env.zip").status_code == 404
    frames = give_spec_and_retarget(runs_dir)
    d = client.get("/api/runs/latent").json()
    env = d["env"]
    assert env["url"] == "/api/runs/latent/env.zip" and env["filename"] == "latent.a2h-env.zip" and env["size"] > 1000
    assert "G1 reference" in env["inside"] and "loader for mjlab" in env["inside"]
    assert env["variants"]["train"]["total"] is None  # the fixture has no variants
    r = client.get("/api/runs/latent/env.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == 'attachment; filename="latent.a2h-env.zip"'
    assert r.headers["x-content-type-options"] == "nosniff" and len(r.content) == env["size"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        m = json.loads(z.read("manifest.json"))
        assert m["id"] == "latent" and m["clip"] == "squat" and m["reference"]["frames"] == frames
        assert z.read("reference/qpos.npy") == QPOS.read_bytes()
        assert "a2h_env/env.py" in z.namelist() and "README.md" in z.namelist()
    # cached: the same bytes until the run changes; a changed spec rebuilds it
    cache = client.app.state.settings.data_dir / "env"
    assert (cache / "latent.a2h-env.zip").is_file()
    first = (cache / "latent.a2h-env.zip").stat().st_mtime_ns
    client.get("/api/runs/latent/env.zip")
    assert (cache / "latent.a2h-env.zip").stat().st_mtime_ns == first
    spec = {**keypose_spec(), "name": "squat", "summary": "changed"}
    (runs_dir / "remote_latent" / "task_spec.json").write_text(json.dumps(spec))
    os.utime(runs_dir / "remote_latent" / "task_spec.json", ns=(first + 10 ** 9, first + 10 ** 9))
    r = client.get("/api/runs/latent/env.zip")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert json.loads(z.read("task_spec.json"))["summary"] == "changed"


def test_run_env_zip_shipped_and_bad_ids(client, runs_dir):
    """A run folder that ships its own env.zip (scripts/export_env.py, build_space.py) serves that file; unknown
    and traversing ids are 404."""
    give_spec_and_retarget(runs_dir)
    shipped = runs_dir / "remote_latent" / "env.zip"
    with zipfile.ZipFile(shipped, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": "a2h-env", "format_version": 1, "id": "latent", "inside": "shipped",
                                                "reference": {"frames": 1, "fps": 30}, "task_spec": {"moments": 0}, "variants": {}}))
    r = client.get("/api/runs/latent/env.zip")
    assert r.status_code == 200 and r.content == shipped.read_bytes()
    assert client.get("/api/runs/latent").json()["env"]["inside"] == "shipped"
    for bad in ["nope", "..", "%2e%2e", "latent%2F..%2Fremote_robust", "remote_latent"]:
        assert client.get(f"/api/runs/{bad}/env.zip").status_code == 404, bad
    assert client.get("/api/runs/latent/files/env.zip").status_code == 404  # not a servable artifact type


def test_live_job_env_zip(live_client):  # noqa: F811
    c = live_client()
    r = upload(c)
    jid = r.json()["id"]
    j = wait_done(c, jid)
    assert j["status"] == "succeeded"
    assert any("training environment: " in line[2] for line in j["logs"])
    live = c.get(f"/api/jobs/{jid}/live").json()
    assert live["env"]["file"] == "env.zip" and live["env"]["size"] > 0 and live["env"]["variants"] == 1
    z = c.get(f"/api/jobs/{jid}/env.zip")
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
    assert z.headers["content-disposition"] == f'attachment; filename="live-{jid[:8]}.a2h-env.zip"'
    assert len(z.content) == live["env"]["size"]
    other = c.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    assert c.get(f"/api/jobs/{other}/env.zip").status_code == 404  # not a live job
    assert c.get("/api/jobs/nope/env.zip").status_code == 404
