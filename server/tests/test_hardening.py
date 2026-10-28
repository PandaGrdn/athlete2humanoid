"""Security and robustness: body/rate limits, hosted lockdown, headers, config validation, error hygiene,
job persistence across restarts, expiry, health endpoints and GPU-backend fallback.

    python -m pytest server/tests -q
"""

import time

import pytest
from fastapi.testclient import TestClient

from server import main as M
from server.config import ConfigError, Settings
from server.main import create_app
from server.security import RateLimiter, client_ip

from .test_api import MP4, wait_done


def settings_for(runs_dir, data_dir, **kw):
    s = Settings()
    s.runs_dir, s.data_dir = runs_dir, data_dir
    s.demo_seconds, s.demo_run, s.runners, s.enable_colab, s.hosted = 0.3, "latent", ["demo"], False, False
    s.live_mode = "0"
    for k, v in kw.items():
        setattr(s, k, v)
    return s


# --- request size ----------------------------------------------------------------------------
def test_declared_body_over_limit_is_refused_before_parsing(make_client, tmp_path):
    c = make_client(max_upload_mb=0.001)
    r = c.post("/api/jobs", content=b"x" * 200_000, headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413
    assert not (tmp_path / "data" / "uploads" / "_incoming").exists() or \
        not any((tmp_path / "data" / "uploads" / "_incoming").iterdir())


def test_streamed_body_over_limit_is_cut_off(make_client, tmp_path):
    """No Content-Length (chunked): the limit is applied while the body streams in."""
    c = make_client(max_upload_mb=0.001)
    sent = []

    def body():
        for _ in range(100):
            sent.append(1)
            yield b"x" * 8192
    r = c.post("/api/jobs", content=body(), headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413 and "larger than" in r.json()["detail"]


def test_small_endpoints_take_small_bodies(client):
    jid = client.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    assert client.post(f"/api/jobs/{jid}/cancel", content=b"x" * 100_000).status_code == 413


# --- rate limits -------------------------------------------------------------------------------
def test_rate_limiter_refills():
    t = [0.0]
    rl = RateLimiter(clock=lambda: t[0])
    assert [rl.hit("k", 2, 60)[0] for _ in range(3)] == [True, True, False]
    ok, wait = rl.hit("k", 2, 60)
    assert not ok and 0 < wait <= 30
    t[0] += 30
    assert rl.hit("k", 2, 60)[0]
    assert rl.hit("other", 2, 60)[0]  # buckets are per key


def test_job_creation_rate_limit(make_client):
    c = make_client(rate_jobs_per_hour=2, max_jobs_per_ip=10)
    codes = [c.post("/api/jobs", data={"run_id": "latent"}).status_code for _ in range(3)]
    assert codes == [201, 201, 429]
    r = c.post("/api/jobs", data={"run_id": "latent"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert c.get("/api/runs").status_code == 200  # reads keep working


def test_request_rate_limit(make_client):
    c = make_client(rate_requests_per_min=3)
    assert [c.get("/api/runs").status_code for _ in range(4)] == [200, 200, 200, 429]
    assert c.get("/healthz").status_code == 200  # probes aren't rate limited


def test_per_client_active_jobs(make_client):
    c = make_client(demo_seconds=30, max_jobs_per_ip=1)
    jid = c.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    r = c.post("/api/jobs", data={"run_id": "latent"})
    assert r.status_code == 429 and "already have" in r.json()["detail"]
    c.post(f"/api/jobs/{jid}/cancel")


def test_client_ip_trusts_only_configured_hops():
    scope = {"client": ("10.0.0.9", 1), "headers": [(b"x-forwarded-for", b"6.6.6.6, 1.2.3.4")]}
    assert client_ip(scope, 0) == "10.0.0.9"   # no proxy trusted: the socket peer
    assert client_ip(scope, 1) == "1.2.3.4"    # what our proxy appended, not the spoofable left entry
    assert client_ip(scope, 2) == "6.6.6.6"
    assert client_ip({"client": ("10.0.0.9", 1), "headers": []}, 1) == "10.0.0.9"


def test_stream_cap(make_client):
    c = make_client(max_streams=1)
    jid = c.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    c.app.state.streams = 1
    r = c.get(f"/api/jobs/{jid}/events")
    assert r.status_code == 503 and r.headers["retry-after"] == "5"


def test_upload_storage_cap(make_client, tmp_path):
    c = make_client(max_upload_storage_mb=0.00001)  # ~10 bytes: the first upload fills it
    assert c.post("/api/jobs", files={"file": ("a.mp4", MP4, "video/mp4")}).status_code == 201
    assert c.post("/api/jobs", files={"file": ("b.mp4", MP4, "video/mp4")}).status_code == 503


# --- hosted lockdown ---------------------------------------------------------------------------
def test_hosted_can_never_enable_colab(make_client, monkeypatch):
    c = make_client(hosted=True, runners=["demo", "colab"], enable_colab=True)
    assert c.get("/api/health").json()["runners"] == ["demo"]
    r = c.post("/api/jobs", data={"runner": "colab"}, files={"file": ("a.mp4", MP4, "video/mp4")})
    assert r.status_code == 400
    assert c.get("/api/jobs").status_code == 404  # no job list: ids are the key to uploads
    monkeypatch.setenv("SPACE_ID", "someone/athlete2humanoid-api")
    monkeypatch.setenv("A2H_RUNNERS", "demo,colab")
    monkeypatch.setenv("A2H_ENABLE_COLAB", "1")
    assert Settings().hosted and Settings().enabled_runners == ["demo"]


def test_job_list_hides_inputs(client):
    client.post("/api/jobs", files={"file": ("a.mp4", MP4, "video/mp4")})
    assert all("input" not in j for j in client.get("/api/jobs").json())


def test_hosted_rejects_wildcard_cors(runs_dir, tmp_path):
    with pytest.raises(ConfigError, match="CORS"):
        create_app(settings_for(runs_dir, tmp_path / "d", hosted=True, cors_origins=["*"]))


def test_config_validation_lists_every_problem(monkeypatch):
    monkeypatch.setenv("A2H_DEMO_SECONDS", "forty")
    monkeypatch.setenv("A2H_MAX_ACTIVE_JOBS", "0")
    monkeypatch.setenv("A2H_RUNNERS", "demo,gpu")
    monkeypatch.setenv("A2H_REMOTE_BACKEND", "aws")
    with pytest.raises(ConfigError) as e:
        Settings().validate()
    msg = str(e.value)
    for name in ("A2H_DEMO_SECONDS", "A2H_MAX_ACTIVE_JOBS", "A2H_RUNNERS", "A2H_REMOTE_BACKEND"):
        assert name in msg
    monkeypatch.delenv("A2H_DEMO_SECONDS")
    monkeypatch.delenv("A2H_MAX_ACTIVE_JOBS")
    monkeypatch.delenv("A2H_RUNNERS")
    monkeypatch.delenv("A2H_REMOTE_BACKEND")
    assert Settings().validate() == [] or all("CORS" not in w for w in Settings().validate())


# --- responses ---------------------------------------------------------------------------------
def test_security_headers(client):
    r = client.get("/api/runs")
    h = r.headers
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
    assert "default-src 'none'" in h["content-security-policy"] and h["referrer-policy"] == "no-referrer"
    assert len(h["x-request-id"]) == 16 and "server" not in h
    assert "content-security-policy" not in client.get("/docs").headers  # Swagger UI needs its scripts


def test_cors_allows_only_configured_origin(client):
    ok = client.options("/api/jobs", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.get("/api/runs", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_runner_errors_dont_leak_internals(make_client):
    c = make_client()
    store = c.app.state.jobs

    def boom(job):
        raise FileNotFoundError("/home/user/app/secret/thing.json")
    store._run_demo = boom
    jid = c.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    j = wait_done(c, jid)
    assert j["status"] == "failed" and "secret" not in j["error"] and "ref " in j["error"]


def test_unhandled_errors_are_generic(make_client, monkeypatch):
    c = TestClient(make_client().app, raise_server_exceptions=False)
    monkeypatch.setattr(M.runs, "find_runs", lambda *a, **k: (_ for _ in ()).throw(OSError("/Users/x/private")))
    r = c.get("/api/runs")
    assert r.status_code == 500 and "private" not in r.text and "ref" in r.json()["detail"]


@pytest.mark.parametrize("name,body,ctype", [
    ("a.webm", MP4, "video/webm"),                      # mp4 bytes named .webm
    ("a.mp4", b"\x1a\x45\xdf\xa3" + b"\0" * 64, "video/mp4"),  # webm bytes named .mp4
])
def test_magic_must_match_extension(client, name, body, ctype):
    assert client.post("/api/jobs", files={"file": (name, body, ctype)}).status_code == 415


def test_display_name_is_sanitised(client):
    r = client.post("/api/jobs", files={"file": ("../../<script>‮.mp4", MP4, "video/mp4")})
    name = r.json()["input"]["filename"]
    assert "/" not in name and "<" not in name and "‮" not in name and name.endswith(".mp4")
    assert r.json()["input"]["stored"] == "input.mp4"


# --- persistence and expiry ----------------------------------------------------------------------
def test_jobs_survive_a_restart(runs_dir, tmp_path):
    data = tmp_path / "data"
    c1 = TestClient(create_app(settings_for(runs_dir, data)))
    done = c1.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    wait_done(c1, done)
    c1.app.state.jobs.flush()
    slow = TestClient(create_app(settings_for(runs_dir, data, demo_seconds=30)))
    running = slow.post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    time.sleep(0.2)
    c2 = TestClient(create_app(settings_for(runs_dir, data)))  # "the restart": reads the journal
    j = c2.get(f"/api/jobs/{done}").json()
    assert j["status"] == "succeeded" and j["result_run"] == "latent" and j["logs"]
    r = c2.get(f"/api/jobs/{running}").json()
    assert r["status"] == "failed" and "restarted" in r["error"]
    slow.app.state.jobs.close()


def test_expiry(make_client, tmp_path):
    c = make_client(upload_ttl_h=1, job_ttl_h=24)
    store = c.app.state.jobs
    jid = c.post("/api/jobs", files={"file": ("a.mp4", MP4, "video/mp4")}).json()["id"]
    wait_done(c, jid)
    (tmp_path / "data" / "uploads" / "orphan").mkdir()
    assert store.sweep() == (1, 0)  # only the orphan folder: the upload is fresh
    assert c.get(f"/api/jobs/{jid}/input").status_code == 200
    assert store.sweep(now=time.time() + 2 * 3600) == (1, 0)
    assert c.get(f"/api/jobs/{jid}/input").status_code == 404 and c.get(f"/api/jobs/{jid}").status_code == 200
    assert store.sweep(now=time.time() + 25 * 3600) == (0, 1)
    assert c.get(f"/api/jobs/{jid}").status_code == 404
    assert store.journal.load() == []


def test_incoming_leftovers_removed_at_startup(runs_dir, tmp_path):
    inc = tmp_path / "data" / "uploads" / "_incoming"
    inc.mkdir(parents=True)
    (inc / "half.mp4").write_bytes(b"x")
    create_app(settings_for(runs_dir, tmp_path / "data"))
    assert not any(inc.iterdir())


# --- health ------------------------------------------------------------------------------------
def test_health_endpoints(make_client, tmp_path):
    c = make_client()
    assert c.get("/healthz").json() == {"ok": True}
    r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"] == {"runs": True, "data_dir_writable": True, "journal": True}
    empty = tmp_path / "empty"
    empty.mkdir()
    r = make_client(runs_dir=empty).get("/readyz")
    assert r.status_code == 503 and r.json()["checks"]["runs"] is False


def test_gpu_backend_missing_falls_back_to_demo(make_client, monkeypatch):
    monkeypatch.setattr(M.shutil, "which", lambda name: None)
    c = make_client(runners=["demo", "colab"], enable_colab=True)
    h = c.get("/api/health").json()
    assert h["runners"] == ["demo"] and "colab CLI not found" in h["degraded"][0]
    r = c.post("/api/jobs", data={"runner": "colab"}, files={"file": ("a.mp4", MP4, "video/mp4")})
    assert r.status_code == 201 and r.json()["runner"] == "demo"
    j = wait_done(c, r.json()["id"])
    assert any("falling back to the demo runner" in line[2] for line in j["logs"])
