"""The live runner (server/live.py): an upload -> its task environment on the host's CPU.

A fake worker (a script standing in for `python -m server.live_worker`) drives the runner's paths:
success, a clip it can't use, a crash, a timeout, a cancel, the one-at-a-time queue and the worker's
bare environment. test_real_worker runs the real pipeline when MediaPipe, mink and a clip are here.
"""

import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from .test_api import MP4, REPO, wait_done
from .test_task_spec import keypose_spec

FAKE = """
import json, os, sys, time
from pathlib import Path
args = sys.argv[3:]  # -m server.live_worker <video> <out> --max-seconds N
out, mode = Path(args[1]), Path({mode!r}).read_text().strip()
(out / "env.json").write_text(json.dumps(sorted(os.environ)))
(out / "argv.json").write_text(json.dumps(args))
def emit(**m): print(json.dumps(m), flush=True)
emit(stage="video", progress=0.5)
(out / "processed.mp4").write_bytes(Path({mp4!r}).read_bytes())
emit(stage="video", done=True, log="video: 2.0 s")
if mode == "bad":
    emit(error="a person was found in only 10% of frames"); sys.exit(2)
if mode == "crash":
    print("Traceback: /secret/path/live_worker.py", file=sys.stderr); sys.exit(1)
if mode == "slow":
    time.sleep(30)
live = {{"fps": 30, "frames": 60, "duration_s": 2.0, "video": {{"width": 360, "height": 640}},
        "landmarks": {{"names": ["nose"], "edges": [], "frames": [[[0.5, 0.2]]] * 60}}, "pose_found": 1.0,
        "robot": None, "retarget_error_cm": None, "timings_s": {{"video": 0.1}}}}
(out / "live.json").write_text(json.dumps(live))
emit(stage="pose", progress=0.5); emit(stage="pose", done=True, log="pose found in 100% of frames")
emit(stage="retarget", done=True, log="retarget error 1.0 cm")
(out / "task_spec.json").write_text(Path({spec!r}).read_text())
emit(stage="environment", done=True, log="task spec: 3 key moments")
emit(stage="environment", note="variants: 18 of 18 kept")
import zipfile
with zipfile.ZipFile(out / "env.zip", "w") as z:  # the variants stage's output: the training environment
    z.writestr("manifest.json", json.dumps({{"format": "a2h-env", "id": "live", "inside": "G1 reference, task spec, 1 training variants, loader"}}))
    z.writestr("reference/qpos.npy", b"x")
live["env"] = {{"file": "env.zip", "size": (out / "env.zip").stat().st_size, "inside": "G1 reference, task spec, 1 training variants, loader", "variants": 1}}
(out / "live.json").write_text(json.dumps(live))
emit(stage="environment", note="training environment: 0.0 MB zip (G1 reference, task spec, 1 training variants, loader)")
"""


@pytest.fixture
def live_client(make_client, tmp_path):
    """make_client with the live runner on and a fake worker; .mode(name) picks its behaviour."""
    (tmp_path / "mode").write_text("ok")
    (tmp_path / "input.mp4").write_bytes(MP4)
    (tmp_path / "spec.json").write_text(json.dumps(keypose_spec()))
    (tmp_path / "fake_worker.py").write_text(FAKE.format(mode=str(tmp_path / "mode"), mp4=str(tmp_path / "input.mp4"),
                                                          spec=str(tmp_path / "spec.json")))
    py = tmp_path / "fakepy"
    py.write_text(f"#!/bin/sh\nexec {sys.executable} {tmp_path / 'fake_worker.py'} \"$@\"\n")
    py.chmod(0o755)

    def make(**kw):
        c = make_client(live_mode="1", live_python=str(py), **kw)
        c.mode = lambda m: (tmp_path / "mode").write_text(m)
        return c
    return make


def upload(c, runner="live"):
    return c.post("/api/jobs", data={"runner": runner}, files={"file": ("kick.mp4", MP4, "video/mp4")})


def test_live_job_builds_the_environment(live_client):
    c = live_client()
    h = c.get("/api/health").json()
    assert "live" in h["runners"] and [s["key"] for s in h["live_stages"]] == ["video", "pose", "retarget", "environment", "policy"]
    r = upload(c)
    assert r.status_code == 201
    j = wait_done(c, r.json()["id"])
    assert j["status"] == "succeeded" and j["progress"] == 1.0, j
    assert {s["key"]: s["status"] for s in j["stages"]} == {"video": "done", "pose": "done", "retarget": "done",
                                                            "environment": "done", "policy": "skipped"}
    assert any("needs a GPU" in line[2] for line in j["logs"])
    assert ["environment", "variants: 18 of 18 kept"] in [line[1:] for line in j["logs"]]  # a note: a log line only
    live = c.get(f"/api/jobs/{j['id']}/live").json()
    assert live["frames"] == 60 and live["landmarks"]["frames"][0] == [[0.5, 0.2]]
    v = c.get(f"/api/jobs/{j['id']}/live/video")
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4" and v.content == MP4
    spec = c.get(f"/api/jobs/{j['id']}/task_spec").json()
    assert spec["format"] == "keypose" and spec["reward"]["weights"]["stage"] == 0.0  # the fixture has no weights
    assert any("order" in n for n in spec["reward"]["notes"])


def test_live_needs_an_upload_and_the_runner(live_client, make_client):
    c = live_client()
    assert c.post("/api/jobs", data={"runner": "live", "run_id": "latent"}).status_code == 400
    assert upload(make_client()).status_code == 400  # live off on this host
    other = make_client().post("/api/jobs", data={"run_id": "latent"}).json()["id"]
    assert c.get(f"/api/jobs/{other}/live").status_code == 404  # not a live job


def test_a_clip_it_cannot_use_says_why(live_client):
    c = live_client()
    c.mode("bad")
    j = wait_done(c, upload(c).json()["id"])
    assert j["status"] == "failed" and j["error"] == "a person was found in only 10% of frames"
    assert {s["key"]: s["status"] for s in j["stages"]}["video"] == "done"


def test_a_crash_does_not_leak_internals(live_client):
    c = live_client()
    c.mode("crash")
    j = wait_done(c, upload(c).json()["id"])
    assert j["status"] == "failed" and j["error"].startswith("internal error building the environment (ref ")
    assert "secret" not in json.dumps(j)


def test_timeout_and_cancel_kill_the_worker(live_client):
    c = live_client(live_timeout_s=1.0)
    c.mode("slow")
    t = time.time()
    j = wait_done(c, upload(c).json()["id"], timeout=10)
    assert j["status"] == "failed" and "longer than 1 s" in j["error"] and time.time() - t < 8
    c = live_client()
    jid = upload(c).json()["id"]
    time.sleep(0.8)
    assert c.post(f"/api/jobs/{jid}/cancel").status_code == 200
    assert wait_done(c, jid, timeout=5)["status"] == "cancelled"


def test_one_build_at_a_time(live_client):
    c = live_client(live_concurrency=1)
    c.mode("slow")
    a = upload(c).json()["id"]
    time.sleep(0.5)
    b = upload(c).json()["id"]
    time.sleep(0.6)
    jb = c.get(f"/api/jobs/{b}").json()
    assert jb["status"] == "queued" and any("waiting for another" in line[2] for line in jb["logs"])
    c.post(f"/api/jobs/{a}/cancel")
    c.mode("ok")
    assert wait_done(c, b, timeout=10)["status"] == "succeeded"


def test_the_worker_gets_no_secrets(live_client, monkeypatch, make_client):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "x" * 30)
    monkeypatch.setenv("MUSE_API_KEY", "k")
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-x")
    c = live_client()
    jid = wait_done(c, upload(c).json()["id"])["id"]
    data_dir = c.app.state.settings.data_dir
    seen = json.loads((data_dir / "uploads" / jid / "live" / "env.json").read_text())
    assert "HF_TOKEN" not in seen and "MUSE_API_KEY" not in seen and "PATH" in seen
    assert "MODAL_TOKEN_ID" not in seen and "MODAL_TOKEN_SECRET" not in seen  # the CPU pose needs no token
    assert "A2H_OUTPUT_DIR" in seen  # config.py's folders go to the job's scratch, not the read-only image
    assert c.get("/api/health").json()["live"]["pose"] == "cpu"
    assert c.get(f"/api/jobs/{jid}/live/mask").status_code == 404  # no SAM 3.1 mask without the GPU


def test_gpu_pose_mode_hands_the_worker_the_modal_token(live_client, monkeypatch):
    """A2H_LIVE_POSE=modal: the worker runs with --pose modal and the Modal token, and nothing else of ours."""
    monkeypatch.setenv("HF_TOKEN", "hf_" + "x" * 30)
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-x")
    c = live_client(live_pose="modal")
    assert c.get("/api/health").json()["live"]["pose"] == "modal"
    jid = wait_done(c, upload(c).json()["id"])["id"]
    live_dir = c.app.state.settings.data_dir / "uploads" / jid / "live"
    seen = json.loads((live_dir / "env.json").read_text())
    assert "MODAL_TOKEN_ID" in seen and "MODAL_TOKEN_SECRET" in seen and "HF_TOKEN" not in seen
    assert json.loads((live_dir / "argv.json").read_text())[-2:] == ["--pose", "modal"]


def test_gpu_pose_reader_ships_in_the_bundle():
    """The worker reads the GPU's output through frontends/sam3d_body.py: the deploy bundle must carry it (a hosted
    build once crashed importing it after the GPU had answered)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_space", REPO / "scripts" / "build_space.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    assert "frontends/sam3d_body.py" in bs.LIVE_MODULES and "COPY frontends frontends" in bs.LIVE_CODE
    assert all((REPO / m).is_file() for m in bs.LIVE_MODULES)


def test_gpu_pose_settings(monkeypatch):
    from server.config import Settings
    monkeypatch.setenv("A2H_LIVE_POSE", "tpu")
    with pytest.raises(Exception, match="A2H_LIVE_POSE"):
        Settings().validate()
    monkeypatch.setenv("A2H_LIVE_POSE", "modal")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    s = Settings()
    assert any("MODAL_TOKEN_ID" in w for w in s.validate()) and s.live_pose == "cpu"  # no token: CPU, with a warning
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-x")
    s = Settings()
    assert not [w for w in s.validate() if "MODAL" in w] and s.live_pose == "modal"


def test_gpu_pose_client_checks_and_saves(tmp_path, monkeypatch):
    """server/live_pose.py: a result is checked against the worker's frames and written as the npz files the worker
    reads; the worker's gpu_pose falls back (returns None, says why) on a bad result, an error or a missing token."""
    import io

    np = pytest.importorskip("numpy")  # the worker's Python has it; the API-only CI job does not
    from server import live_pose, live_worker

    T, H, W = 4, 6, 8
    masks = np.zeros((T, H, W), bool)
    masks[:, 1:5, 2:6] = True
    sam3d = io.BytesIO()
    np.savez_compressed(sam3d, kp3d=np.zeros((T, 70, 3), np.float32), kp2d=np.zeros((T, 70, 2), np.float32),
                        cam_t=np.zeros((T, 3), np.float32), image_wh=np.array([W, H]))
    m = io.BytesIO()
    np.savez_compressed(m, masks=masks)
    result = {"sam3d": sam3d.getvalue(), "masks": m.getvalue(), "frames": T, "width": W, "height": H,
              "seen": 1.0, "timings_s": {"sam": 1.5, "sam3d": 2.0}, "gpu": "L40S"}
    assert live_pose.check(result, T, W, H) is None
    assert "frames" in live_pose.check(result, T + 1, W, H)
    assert live_pose.check({"sam3d": b""}, T, W, H) == "not a pose result"
    out = tmp_path / "live"
    out.mkdir()
    path, got = live_pose.save(result, out)
    assert path == out / "sam3d.npz" and got.dtype == bool and (got == masks).all()
    assert np.load(out / "masks.npz")["masks"].shape == (T, H, W)
    video = tmp_path / "processed.mp4"
    video.write_bytes(b"not really a video")

    whys = []
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    assert live_worker.gpu_pose(video, out, T, W, H, 5, whys.append) is None and "token" in whys[-1]
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-x")
    monkeypatch.setattr(live_pose, "fetch", lambda b, t, prompt="athlete": result)
    p, mk, r = live_worker.gpu_pose(video, out, T, W, H, 5, whys.append)
    assert p == out / "sam3d.npz" and mk.shape == (T, H, W) and r["gpu"] == "L40S"
    monkeypatch.setattr(live_pose, "fetch", lambda b, t, prompt="athlete": {**result, "frames": T + 2})
    assert live_worker.gpu_pose(video, out, T, W, H, 5, whys.append) is None and "frames" in whys[-1]

    def boom(b, t, prompt="athlete"):
        raise TimeoutError("no container in 5 s")
    monkeypatch.setattr(live_pose, "fetch", boom)
    assert live_worker.gpu_pose(video, out, T, W, H, 5, whys.append) is None and whys[-1].startswith("TimeoutError")


def _have(*mods):
    return all(importlib.util.find_spec(m) for m in mods)


@pytest.mark.skipif(not _have("mediapipe", "mink", "mujoco", "torch", "cv2", "imageio_ffmpeg")
                    or not (REPO / "assets" / "pose_landmarker_heavy.task").is_file()
                    or not (REPO / "clips" / "front_kick.mov").is_file(), reason="needs the CPU pipeline and a clip")
def test_real_worker(tmp_path):
    from server import live_worker
    out = tmp_path / "out"
    assert live_worker.main([str(REPO / "clips" / "front_kick.mov"), str(out), "--max-seconds", "8"]) == 0
    live = json.loads((out / "live.json").read_text())
    spec = json.loads((out / "task_spec.json").read_text())
    assert live["robot"]["frames"] and len(live["robot"]["frames"]) == live["frames"] and live["pose_found"] > 0.9
    assert spec["type"] == "keypose" and spec["self_check"]["success"]
    # the build view's data: the 3D skeleton and qpos per frame, and the variants' preview
    assert len(live["skeleton"]["frames"]) == len(live["qpos"]) == live["frames"] and len(live["qpos"][0]) == 36
    v = live["variants"]
    assert v["total"] == len(v["rows"]) >= v["kept"] > 0 and v["ghosts"] and len(v["ghosts"][0]["qpos"][0]) == 36
    # the worker's body positions are the ones the task spec is scored on everywhere else
    from latent.reward_net import BODIES
    assert tuple(live_worker.BODIES) == BODIES


@pytest.mark.skipif(not _have("mink", "mujoco", "scipy", "torch") or not (REPO / "assets" / "mujoco_menagerie").is_dir(),
                    reason="needs the retarget and the G1 model")
def test_variants_stop_at_their_deadline(monkeypatch):
    """The variants run on the job's clock after its environment is done: past their deadline they stop, keeping
    those made (the ghost variants come first), and with none made the worker skips them."""
    import retarget
    from server import live_worker
    rt = retarget.G1Retargeter()
    q = rt.retarget_sequence(retarget.synthetic_squat(rt))
    with pytest.raises(TimeoutError):
        live_worker.variant_preview(q, rt.model, deadline=time.monotonic() - 1)
    clock = iter(range(1000))  # each check of the clock is one second later
    monkeypatch.setattr(time, "monotonic", lambda: float(next(clock)))
    v = live_worker.variant_preview(q, rt.model, deadline=2.5)
    assert v["total"] == 3 and [r["params"]["mirror"] for r in v["rows"]][0] is True  # the mirrored ghost first


def test_live_data_is_bounded(tmp_path):
    from server import live
    (tmp_path / "live.json").write_text("x" * (live.MAX_LIVE_BYTES + 1))
    assert live.live_data(tmp_path) is None
    (tmp_path / "live.json").write_text("{not json")
    assert live.live_data(tmp_path) is None
    shutil.rmtree(tmp_path)
    assert live.live_data(Path(os.devnull)) is None


def test_muse_names_the_moments_when_on(live_client, monkeypatch, tmp_path):
    """A2H_MUSE_NAMES=1: the API process (not the worker) has Muse name the moments; a failure keeps the
    measured names and doesn't fail the job."""
    from muse import client as mc
    real = mc.MuseClient
    monkeypatch.setattr(mc, "MuseClient", lambda: real(fixture=mc.FIXTURES / "moment_names_front_kick.json"))
    (tmp_path / "spec.json").write_text((REPO / "tests" / "data" / "task_spec_front_kick.json").read_text())
    c = live_client(live_muse_names=True)
    j = wait_done(c, upload(c).json()["id"])
    assert j["status"] == "succeeded" and any("Muse named the key moments" in line[2] for line in j["logs"]), j["logs"]
    spec = c.get(f"/api/jobs/{j['id']}/task_spec").json()
    poses = [s for s in spec["stages"] if s["kind"] == "pose"]
    assert poses[0]["measured"] == "right foot up, right foot forward" and poses[0]["description"]
    assert spec["named_by"] == "muse-spark-1.3" and spec["action"]
    (tmp_path / "spec.json").write_text(json.dumps(keypose_spec()))  # a different number of moments: Muse's answer won't fit
    j = wait_done(c, upload(c).json()["id"])
    assert j["status"] == "succeeded" and any("couldn't name the moments" in line[2] for line in j["logs"])
    assert c.get(f"/api/jobs/{j['id']}/task_spec").json().get("named_by") is None
