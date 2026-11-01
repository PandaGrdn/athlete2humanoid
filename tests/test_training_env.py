"""The training environment bundle (a2h_env/, docs/TRAINING_ENV.md): export a run folder, load the zip back, and get
the run's own arrays; the manifest's fields; the live build's export; the loader's interpolation against mjlab's.

    python -m pytest -q tests/test_training_env.py
"""

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a2h_env import bundle as B  # noqa: E402
from a2h_env import export as X  # noqa: E402

DATA = ROOT / "tests" / "data"
QPOS = DATA / "qpos_front_kick.npy"
SPEC = DATA / "task_spec_front_kick.json"


def fake_variant(name, params, qpos, ok=True, rules=(), why=""):
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    q = qpos[::2] + rng.normal(0, 0.01, (len(qpos[::2]), qpos.shape[1]))
    return {"name": name, "params": params, "qpos": q.astype(np.float32), "src_frame": np.arange(len(q), dtype=np.float32) * 2,
            "frames": len(q), "limit_clipped": 0.0, "ok": ok, "rules": list(rules), "why": why,
            "spec": json.loads(SPEC.read_text()) | {"name": name}}


@pytest.fixture
def run(tmp_path):
    """A run folder shaped like outputs/remote_latent_<tag>, with a sibling variants run (skill/variations.py write())."""
    root = tmp_path / "outputs"
    run = root / "remote_latent_t1"
    clip = run / "clips" / "front_kick"
    clip.mkdir(parents=True)
    qpos = np.load(QPOS).astype(np.float64)
    np.save(clip / "qpos.npy", qpos)
    (clip / "meta.json").write_text(json.dumps({"fps": 30.0, "frames": len(qpos), "frontend": "sam3d", "sam": True,
                                                  "retarget_version": {"version": "v-test", "git": "abc123"},
                                                  "in_place": {"method": "world", "travel_removed_m": 1.2}}))
    (clip / "silhouette.mp4").write_bytes(b"\x00" * 100)
    (clip / "video_latent.npz").write_bytes(b"\x00" * (X.MAX_EXTRA_BYTES + 1))  # too big: left out
    spec = json.loads(SPEC.read_text())
    spec["stages"][1]["muse_label"] = "kick right leg forward"
    spec["named_by"] = {"model": "muse-spark-1.3"}
    (run / "task_spec.json").write_text(json.dumps(spec))
    (run / "importance.json").write_text(json.dumps({"fps": 30.0, "weight": [0.5] * len(qpos)}))
    (run / "web.json").write_text(json.dumps({"title": "Kick", "credit": {"text": "A. Person, \"Kick\", Pexels",
                                                                          "url": "https://www.pexels.com/x", "licence": "Pexels License"}}))
    pol = run / "policy_tracking_var"
    pol.mkdir()
    (pol / "weights.json").write_text(json.dumps({"task": "A2H-Tracking-Flat-Unitree-G1", "iterations": 1000,
                                                  "started_from": {"mode": "init", "checkpoint": "/content/a2h/inputs/policy_tracking_var/model_1499.pt"},
                                                  "variants": "/content/a2h/outputs/athletes_x/variants/variants.json"}))
    (pol / "reward_config.json").write_text(json.dumps({"feet_height": {"weight": 2.0, "std": 0.1, "clearance": 0.03, "swing_only": True}}))
    var = root / "remote_variants_t1"
    train = [fake_variant("o_t1_a1", {"mirror": False, "tempo": 1.0, "amplitude": 1.0, "warp": 0.0, "seed": 0}, qpos),
             fake_variant("m_t1.33_a1.1", {"mirror": True, "tempo": 1.33, "amplitude": 1.1, "warp": 0.0, "seed": 1}, qpos),
             fake_variant("o_t0.75_a0.85", {"mirror": False, "tempo": 0.75, "amplitude": 0.85, "warp": 0.0, "seed": 2}, qpos,
                          ok=False, rules=["speed"], why="a joint at 21.0 rad/s (cap 20)")]
    held = [fake_variant("o_t2_a1", {"mirror": False, "tempo": 2.0, "amplitude": 1.0, "warp": 0.0, "seed": 1000}, qpos)]
    _write(train, var / "variants")
    _write(held, var / "held_out")
    return run


def _write(variants, out_dir):
    """skill.variations.write without retarget's CSV export (no mink here): the same files but motion.csv."""
    out_dir = Path(out_dir)
    index = []
    for v in variants:
        row = {k: v[k] for k in ("name", "params", "frames", "limit_clipped", "ok", "rules", "why")}
        if v["ok"]:
            d = out_dir / v["name"]
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "qpos.npy", v["qpos"])
            np.save(d / "src_frame.npy", v["src_frame"])
            (d / "task_spec.json").write_text(json.dumps(v["spec"], indent=1))
            row["summary"] = v["spec"]["summary"]
        index.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))


def test_export_then_load_round_trips_the_run(run, tmp_path):
    out = tmp_path / "t1.a2h-env.zip"
    m = X.export_run(run, out, generate=False)
    assert m and out.is_file() and m["bytes"] == out.stat().st_size
    assert out.stat().st_mode & 0o044 == 0o044  # readable by other users: a deploy image serves it as a different uid
    b = B.load_bundle(out)
    qpos = np.load(run / "clips" / "front_kick" / "qpos.npy")
    assert np.array_equal(b.qpos, qpos) and b.fps == 30.0 and b.in_place and b.clip == "front_kick" and b.id == "latent_t1"
    assert b.spec["stages"][1]["muse_label"] == "kick right leg forward"
    assert [label for kind, label in b.stages() if kind == "pose"][0] == "kick right leg forward"
    assert b.importance["weight"][0] == 0.5 and b.credit["licence"] == "Pexels License"
    assert b.reward_config["feet_height"]["weight"] == 2.0 and b.reward_config["iterations"] == 1000
    assert [p["policy"] for p in b.reward_config["policies"]] == ["tracking_var"]
    # variants: every verdict, arrays for the kept ones only, equal to the variants run's
    assert [v.name for v in b.train] == ["o_t1_a1", "m_t1.33_a1.1", "o_t0.75_a0.85"]
    kept = b.kept("train")
    assert [v.name for v in kept] == ["o_t1_a1", "m_t1.33_a1.1"]
    for v in kept:
        assert np.array_equal(v.qpos, np.load(run.parent / "remote_variants_t1" / "variants" / v.name / "qpos.npy").astype(np.float64))
        assert v.spec["name"] == v.name and len(v.src_frame) == len(v.qpos)
    dropped = b.variant("o_t0.75_a0.85")
    assert not dropped.ok and dropped.rules == ["speed"] and dropped.qpos is None
    assert [v.name for v in b.kept("held_out")] == ["o_t2_a1"]
    # the same from the unzipped folder
    folder = tmp_path / "unzipped"
    with zipfile.ZipFile(out) as z:
        z.extractall(folder)
    b2 = B.load_bundle(folder)
    assert np.array_equal(b2.qpos, b.qpos) and len(b2.train) == 3
    # contacts come with MuJoCo and the G1 model, as (T, 2) bool, from the spec's own rule
    if b.contacts is not None:
        assert b.contacts.shape == (len(qpos), 2) and b.contacts.dtype == bool
        assert b.contacts.any()


def test_manifest_fields(run, tmp_path):
    out = tmp_path / "t1.a2h-env.zip"
    m = X.export_run(run, out, generate=False)
    assert m["format"] == "a2h-env" and m["format_version"] == 1 and m["id"] == "latent_t1" and m["run"] == "remote_latent_t1"
    assert m["title"] == "Kick" and m["clip"] == "front_kick"
    assert m["pipeline"]["retarget_version"] == "v-test" and m["pipeline"]["retarget_git"] == "abc123" and m["pipeline"]["pose_frontend"] == "sam3d"
    assert m["pipeline"]["commit"] == "abc123"  # the retarget's commit, from the clip's meta.json
    assert m["pipeline"]["exported_at_commit"] is None or (len(m["pipeline"]["exported_at_commit"]) == 12 and m["pipeline"]["dirty"] in (True, False))
    spec = json.loads((run / "task_spec.json").read_text())
    assert m["spec_sha256"] == X.sha256(X.canonical(spec)) and len(m["qpos_sha256"]) == 64
    assert m["reference"]["frames"] == len(np.load(QPOS)) and m["reference"]["fps"] == 30.0 and m["reference"]["in_place"] is True
    poses, flights = (sum(st["kind"] == k for st in spec["stages"]) for k in ("pose", "flight"))
    assert m["task_spec"]["moments"] == poses and m["task_spec"]["flights"] == flights and m["task_spec"]["named_by"] == "muse-spark-1.3"
    assert m["variants"]["train"] == {"total": 3, "kept": 2, "with_arrays": 2, "index": "variants/train/index.json"}
    assert m["variants"]["held_out"]["with_arrays"] == 1 and m["variants"]["source"] == "remote_variants_t1"
    assert m["reward"]["feet_height"]["weight"] == 2.0 and m["reward"]["policy"] == "tracking_var"
    assert m["reward"]["started_from"] == {"mode": "init", "checkpoint": "model_1499.pt"} and m["reward"]["trained_on"] == "variants: athletes_x"
    assert m["credit"]["url"] == "https://www.pexels.com/x"
    assert m["extras"] == {"silhouette": "extras/silhouette.mp4"}  # the latent was too big
    assert m["robot"]["nq"] == 36 and "mjlab" in m["loader"]["requires"]
    paths = {f["path"] for f in m["files"]}
    assert {"reference/qpos.npy", "reference/meta.json", "task_spec.json", "reward_config.json", "importance.json", "credit.json",
            "variants/train/index.json", "variants/train/o_t1_a1/qpos.npy", "variants/held_out/o_t2_a1/task_spec.json",
            "extras/silhouette.mp4", "robot/README.md", "a2h_env/__init__.py", "a2h_env/env.py", "a2h_env/train.py"} <= paths
    assert "variants/train/o_t0.75_a0.85/qpos.npy" not in paths
    assert "2 training variants and 1 held out" in m["inside"] and "named by muse-spark-1.3" in m["inside"]
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        readme = z.read("README.md").decode()
        assert names == paths | {"manifest.json", "README.md"}
        assert "python -m a2h_env train" in readme and "Pexels License" in readme
        assert json.loads(z.read("manifest.json"))["files"] == m["files"]
    assert X.manifest_of(out)["bytes"] == out.stat().st_size and X.manifest_of(QPOS) is None


def test_verdicts_only_and_no_spec(run, tmp_path):
    """Without a variants run, the run's variants.json view gives the verdicts (no arrays); without a spec, no bundle."""
    shutil.rmtree(run.parent / "remote_variants_t1")
    (run / "variants.json").write_text(json.dumps({"total": 2, "kept": 1, "rows": [
        {"name": "o_t1_a1", "params": {"mirror": False, "tempo": 1.0, "amplitude": 1.0, "warp": 0.0}, "frames": 10, "ok": True, "rules": [], "why": ""},
        {"name": "m_t1_a1", "params": {"mirror": True, "tempo": 1.0, "amplitude": 1.0, "warp": 0.0}, "frames": 10, "ok": False, "rules": ["self-check"], "why": "x"}],
        "ghosts": []}))
    m = X.export_run(run, tmp_path / "v.zip", generate=False)
    assert m["variants"]["train"] == {"total": 2, "kept": 1, "with_arrays": 0, "index": "variants/train/index.json"}
    assert m["variants"]["held_out"] is None and m["variants"]["source"].startswith("verdicts only")
    b = B.load_bundle(tmp_path / "v.zip")
    assert len(b.train) == 2 and b.train[0].ok and b.train[0].qpos is None and b.kept() == []
    assert "verdicts of 2 variants" in m["inside"]
    (run / "task_spec.json").write_text(json.dumps({"name": "../front_kick", "stages": []}))
    assert X.export_run(run, tmp_path / "x.zip") is None
    (run / "task_spec.json").unlink()
    assert X.export_run(run, tmp_path / "x.zip") is None and not (tmp_path / "x.zip").exists()


def test_live_export(tmp_path):
    qpos = np.load(QPOS).astype(np.float64)
    spec = json.loads(SPEC.read_text())
    pos = np.zeros((len(qpos), 14, 3), np.float32)
    pos[:, [3, 6], 2] = spec["foot0"] + 0.01  # both feet down
    pos[10:20, 6, 2] = spec["foot0"] + 0.3  # right foot up for 10 frames
    vs = [fake_variant("m_t1_a1", {"mirror": True, "tempo": 1.0, "amplitude": 1.0, "warp": 0.0, "seed": 0}, qpos),
          fake_variant("o_t0.75_a1", {"mirror": False, "tempo": 0.75, "amplitude": 1.0, "warp": 0.0, "seed": 1}, qpos, ok=False, rules=["ballistic"])]
    m = X.export_live(tmp_path, qpos=qpos, fps=30.0, spec=spec, variants=vs, pos=pos, name="clip", job_id="abcdef0123",
                      meta={"frontend": "MediaPipe"})
    assert (tmp_path / "env.zip").is_file() and m["id"] == "live-abcdef01" and m["variants"]["train"]["with_arrays"] == 1  # server/live.py env_filename
    b = B.load_bundle(tmp_path / "env.zip")
    assert np.array_equal(b.qpos, qpos) and b.reference_meta["frontend"] == "MediaPipe" and not b.in_place
    assert b.contacts.shape == (len(qpos), 2) and b.contacts[0].tolist() == [True, True] and b.contacts[15].tolist() == [True, False]
    assert np.array_equal(b.kept()[0].qpos, vs[0]["qpos"].astype(np.float64)) and b.held_out == []
    assert b.reward_config["feet_height"] is None and "1 training variants" in m["inside"]


def test_npy_header_without_numpy():
    data = (QPOS).read_bytes()
    shape, dtype, start = X.npy_header(data)
    a = np.load(QPOS)
    assert shape == a.shape and np.dtype(dtype) == a.dtype and len(data) - start == a.nbytes


def test_info_cli(run, tmp_path):
    out = tmp_path / "t1.a2h-env.zip"
    X.export_run(run, out, generate=False)
    r = subprocess.run([sys.executable, "-m", "a2h_env", "info", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "latent_t1: Kick" in r.stdout and "kick right leg forward" in r.stdout and "train       2 of 3 kept" in r.stdout
    assert "feet_height (weight 2.0)" in r.stdout and "run in place" in r.stdout
    r = subprocess.run([sys.executable, "-m", "a2h_env", "info", str(out), "--json"], capture_output=True, text=True, cwd=ROOT)
    assert json.loads(r.stdout)["id"] == "latent_t1"


def test_loader_is_self_contained(run, tmp_path):
    """The vendored a2h_env/ imports and reads a bundle with nothing else of this repository on the path, and with numpy
    alone: a2h_env.env and a2h_env.train import torch and mjlab only when an environment is built."""
    out = tmp_path / "t1.a2h-env.zip"
    X.export_run(run, out, generate=False)
    with zipfile.ZipFile(out) as z:
        z.extractall(tmp_path / "clean")
    code = (f"import sys; sys.path = [p for p in sys.path if not p.startswith({str(ROOT)!r})]; sys.path.insert(0, '.')\n"
            "import a2h_env, a2h_env.env, a2h_env.train\n"
            f"b = a2h_env.load_bundle({str(out)!r}); print(len(b.qpos), len(b.kept()))\n"
            "from a2h_env.env import interpolate\n"
            "pos, quat, joints, lin_vel, ang_vel, joint_vel = interpolate(b.qpos, b.fps)\n"
            "print(len(pos), 'torch' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path / "clean")
    assert r.returncode == 0, r.stderr
    n = len(np.load(QPOS))
    assert r.stdout.split() == [str(n), "2", str(len(np.arange(0, (n - 1) / 30.0, 1 / 50.0))), "False"]


def test_axis_angle_flips_the_sign_of_w():
    """A quaternion and its negation are the same rotation: both give the same small rotation vector (mjlab's
    axis_angle_from_quat negates w < 0; without it a sign jump between frames reads as a ~2 pi turn)."""
    sys.path.insert(0, str(ROOT))
    from a2h_env.env import _axis_angle
    q = np.array([[np.cos(0.01), np.sin(0.01), 0.0, 0.0]])  # 0.02 rad about x
    assert np.allclose(_axis_angle(q), [[0.02, 0, 0]], atol=1e-9) and np.allclose(_axis_angle(-q), [[0.02, 0, 0]], atol=1e-9)
    assert np.allclose(_axis_angle(np.array([[1.0, 0, 0, 0]])), 0) and np.allclose(_axis_angle(np.array([[-1.0, 0, 0, 0]])), 0)


def test_interpolate_matches_mjlab_motion_loader():
    """The loader's qpos -> 50 Hz interpolation equals mjlab.scripts.csv_to_npz's MotionLoader on the same CSV."""
    pytest.importorskip("mjlab")
    from mjlab.scripts.csv_to_npz import MotionLoader
    sys.path.insert(0, str(ROOT))
    from a2h_env.env import interpolate
    import tempfile
    qpos = np.load(QPOS).astype(np.float64)
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "motion.csv"
        wxyz = qpos[:, 3:7]
        np.savetxt(csv, np.column_stack([qpos[:, :3], wxyz[:, 1:], wxyz[:, :1], qpos[:, 7:]]), delimiter=",", fmt="%.6f")
        m = MotionLoader(str(csv), 30, 50, "cpu")
        rows = np.loadtxt(csv, delimiter=",")  # the same rounded values mjlab read
    qp = rows[:, [0, 1, 2, 6, 3, 4, 5] + list(range(7, 36))]
    pos, quat, joints, lin_vel, ang_vel, joint_vel = interpolate(qp, 30.0)
    assert pos.shape == m.motion_base_poss.shape
    assert np.allclose(pos, m.motion_base_poss.numpy(), atol=1e-5) and np.allclose(joints, m.motion_dof_poss.numpy(), atol=1e-5)
    assert np.allclose(np.abs((quat * m.motion_base_rots.numpy()).sum(1)), 1, atol=1e-5)
    assert np.allclose(lin_vel, m.motion_base_lin_vels.numpy(), atol=1e-3) and np.allclose(joint_vel, m.motion_dof_vels.numpy(), atol=1e-3)
    # Angular velocity is the axis-angle of q[t+1] q[t-1]^-1 over 2 dt = 0.04 s. mjlab's slerp returns the first quaternion
    # untouched when a pair is within 4 float32 ulps of identical (a step of up to ~1e-3 rad, ~0.06 rad/s at near-still
    # frames), and float32 rounding of the dot product decides which frames that hits; the loader blends those smoothly
    # (a2h_env.env._slerp), so they are compared loosely. Everywhere else the two agree to ~3e-5 rad/s (the float32
    # products, measured on this clip, the lunge, the jog and the squat jump); 1e-3 leaves a factor ~30.
    n = len(rows)
    phase = np.arange(0, (n - 1) / 30.0, 1 / 50.0) / ((n - 1) / 30.0)
    i0 = np.floor(phase * (n - 1)).astype(int)
    # mjlab's shortcut fires on |dot| within 4.77e-7 of 1 in float32 arithmetic, whose own rounding reaches ~5e-7; an output
    # frame on an input frame's boundary belongs to either neighbouring pair (float32 floors the phase differently)
    dots = np.abs((qp[:-1, 3:7] * qp[1:, 3:7]).sum(1)) > 1 - 2e-6  # input pair k = (k, k + 1) near-identical
    pair = dots[np.minimum(i0, n - 2)] | dots[np.maximum(i0 - 1, 0)]
    near_still = pair | np.roll(pair, 1) | np.roll(pair, -1)  # the central difference reaches one frame either side
    diff = np.abs(ang_vel - m.motion_base_ang_vels.numpy()).max(1)
    assert diff[~near_still].max() < 1e-3, diff[~near_still].max()
    assert diff.max() < 0.1, diff.max()
