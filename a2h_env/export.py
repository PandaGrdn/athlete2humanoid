"""A run folder (or a live build) -> its training environment zip. Standard library only; numpy, mujoco and the
repository's skill/ are used when importable (foot contacts, generating variants) and skipped otherwise, so the
API server builds bundles with nothing but Python.

    python scripts/export_env.py outputs/remote_latent_s3_kick [...]   # -> outputs/remote_latent_s3_kick/env.zip

What goes in (bundle.py has the layout): the clip the run's task spec is named after (clips/<name>/qpos.npy and
meta.json), task_spec.json, importance.json, the variants (a sibling outputs/remote_variants_<tag>/ run's training and
held-out sets with their verdicts; else the run's variants.json view, verdicts only; else, with generate=True,
skill/variations.py's TRAIN and HELD_OUT grids on the retarget), the reward config the tracking policy used, the
credit from web.json, the silhouette and V-JEPA latent when small, the G1's licence, and the a2h_env loader.
"""

import hashlib
import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
FORMAT, FORMAT_VERSION = "a2h-env", 1
ZIP_NAME = "env.zip"
TASK = "A2H-Tracking-Flat-Unitree-G1"
MAX_EXTRA_BYTES = 2 * 1048576  # the silhouette and the latent ride along only when small
DOWN = 0.05  # m above the standing ankle height that counts as a foot on the floor (skill/task.py DOWN)
LOADER_FILES = ("__init__.py", "__main__.py", "bundle.py", "env.py", "train.py")
# mjlab 1.6.0's G1 flat tracking task (mjlab/tasks/tracking/tracking_env_cfg.py), the reward the stored tracking
# policies were trained with; recorded here so the bundle says what it pays without importing mjlab.
MJLAB_TRACKING = {
    "version": "1.6.0", "task": "Mjlab-Tracking-Flat-Unitree-G1 (BeyondMimic re-implementation)",
    "control_hz": 50, "physics_hz": 200, "episode_s": 10.0,
    "rewards": {"motion_global_root_pos": {"weight": 0.5, "std": 0.3}, "motion_global_root_ori": {"weight": 0.5, "std": 0.4},
                "motion_body_pos": {"weight": 1.0, "std": 0.3}, "motion_body_ori": {"weight": 1.0, "std": 0.4},
                "motion_body_lin_vel": {"weight": 1.0, "std": 1.0}, "motion_body_ang_vel": {"weight": 1.0, "std": 3.14},
                "action_rate_l2": {"weight": -0.1}, "joint_limit": {"weight": -10.0},
                "self_collisions": {"weight": -10.0, "force_threshold": 10.0}},
    "terminations": {"anchor_pos": 0.25, "anchor_ori": 0.8, "ee_body_pos": 0.25, "time_out": True},
    "bodies": ["pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
               "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
               "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"],
}
ROBOT = {"name": "Unitree G1 (29 DoF)", "nq": 36,
         "qpos_layout": "root xyz, root quaternion (w, x, y, z), 29 joint angles in the G1's joint order",
         "env_asset": "mjlab.asset_zoo.robots.unitree_g1 (g1.xml, shipped with mjlab)",
         "kinematics_asset": "MuJoCo Menagerie unitree_g1/scene.xml (the retarget's model; joint order identical)",
         "licence": "BSD-3-Clause (MuJoCo Menagerie); see robot/LICENSE when included"}


# --- helpers (standard library) ----------------------------------------------------------------------------------------
def npy_header(data):
    """(shape, dtype, header length) of a .npy file's bytes, without numpy."""
    if data[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = data[6]
    if major == 1:
        hlen, start = struct.unpack("<H", data[8:10])[0], 10
    else:
        hlen, start = struct.unpack("<I", data[8:12])[0], 12
    header = data[start:start + hlen].decode("latin1")
    shape = tuple(int(x) for x in re.search(r"'shape':\s*\(([^)]*)\)", header).group(1).split(",") if x.strip())
    dtype = re.search(r"'descr':\s*'([^']+)'", header).group(1)
    return shape, dtype, start + hlen


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def git_state(repo=REPO):
    """(commit, dirty) of the checkout the export runs from; (None, None) outside a git checkout (the Space image)."""
    def git(*args):
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    commit = git("rev-parse", "--short=12", "HEAD")
    if commit is None:
        return None, None
    status = git("status", "--porcelain", "--untracked-files=no")
    return commit, (bool(status) if status is not None else None)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def run_id(folder):
    name = Path(folder).name
    return name[len("remote_"):] if name.startswith("remote_") else name


def clean_credit(v):
    """web.json "credit" -> {text, url (https only), licence}, or None."""
    if not isinstance(v, dict) or not isinstance(v.get("text"), str) or not v["text"].strip():
        return None
    url = v.get("url") if isinstance(v.get("url"), str) and re.fullmatch(r"https://[^\s\"'<>]{1,400}", v["url"]) else None
    lic = v.get("licence") if isinstance(v.get("licence"), str) else None
    return {"text": v["text"].strip()[:300], "url": url, "licence": lic and lic.strip()[:60]}


def npy_bytes(array):
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(array))
    return buf.getvalue()


# --- the parts of a bundle ----------------------------------------------------------------------------------------------
class Parts:
    """path in the zip -> bytes or a Path to copy, plus the manifest fields gathered along the way."""

    def __init__(self):
        self.files = {}

    def add(self, name, data):
        if data is not None:
            self.files[name] = data

    def add_file(self, name, path, max_bytes=None):
        p = Path(path)
        if p.is_file() and (max_bytes is None or p.stat().st_size <= max_bytes):
            self.files[name] = p
            return True
        return False

    def add_json(self, name, obj):
        if obj is not None:
            self.files[name] = json.dumps(obj, indent=1, ensure_ascii=False).encode("utf-8")

    def size(self, name):
        d = self.files[name]
        return d.stat().st_size if isinstance(d, Path) else len(d)

    def read(self, name):
        d = self.files[name]
        return d.read_bytes() if isinstance(d, Path) else d


def contacts_from_positions(pos, foot0, down=DOWN):
    """(T, bodies, 3) body positions (mjlab's G1 tracking bodies) -> (T, 2) bool: left and right foot on the floor."""
    import numpy as np
    pos = np.asarray(pos)
    return (pos[:, [3, 6], 2] - float(foot0)) < float(down)


def body_positions(qpos):
    """(T, 36) -> (T, 14, 3) via MuJoCo on the Menagerie G1, or None when mujoco or the model aren't here."""
    try:
        import mujoco
        import numpy as np
        sys.path.insert(0, str(REPO))
        import config
        model = mujoco.MjModel.from_xml_path(str(config.G1_SCENE_XML))
    except Exception:  # noqa: BLE001  (no mujoco, no assets: contacts are optional)
        return None
    data = mujoco.MjData(model)
    ids = [model.body(b).id for b in MJLAB_TRACKING["bodies"]]
    out = np.empty((len(qpos), len(ids), 3), np.float32)
    for t, q in enumerate(np.asarray(qpos, np.float64)):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        out[t] = data.xpos[ids]
    return out


def reference_meta(meta, fps, frames):
    """reference/meta.json: the clip's meta.json with in_place made a flag (its details kept)."""
    meta = dict(meta or {})
    ip = meta.get("in_place")
    meta["in_place"] = bool(ip)
    if isinstance(ip, dict):
        meta["in_place_details"] = ip
    meta.setdefault("fps", fps)
    meta["frames"] = frames
    return meta


def variant_rows(index):
    """index.json rows -> the verdict rows a bundle keeps."""
    out = []
    for r in index or []:
        if not isinstance(r, dict) or not isinstance(r.get("name"), str):
            continue
        out.append({"name": r["name"], "params": r.get("params") or {}, "frames": int(r.get("frames") or 0),
                    "limit_clipped": r.get("limit_clipped"), "ok": bool(r.get("ok")), "rules": list(r.get("rules") or []),
                    "why": r.get("why") or "", "summary": r.get("summary")})
    return out


def add_variant_set(parts, which, rows, arrays):
    """variants/<which>/: index.json with every verdict, and each kept variant's arrays when `arrays(name)` has them
    -> {total, kept, with_arrays}."""
    rows = variant_rows(rows)
    parts.add_json(f"variants/{which}/index.json", rows)
    n = 0
    for r in rows:
        if not r["ok"]:
            continue
        got = arrays(r["name"])
        if not got:
            continue
        base = f"variants/{which}/{r['name']}/"
        for k, v in got.items():
            parts.add(base + k, v)
        n += 1
    return {"total": len(rows), "kept": sum(r["ok"] for r in rows), "with_arrays": n, "index": f"variants/{which}/index.json"}


def from_folder(d):
    """A variant folder (skill/variations.py write()) -> its arrays as bytes."""
    d = Path(d)
    if not (d / "qpos.npy").is_file():
        return None
    out = {"qpos.npy": d / "qpos.npy"}
    for f in ("task_spec.json", "src_frame.npy"):
        if (d / f).is_file():
            out[f] = d / f
    return out


def from_generated(v):
    """A skill.variations.generate() variant -> its arrays as bytes."""
    if v.get("qpos") is None:
        return None
    out = {"qpos.npy": npy_bytes(v["qpos"])}
    if v.get("spec") is not None:
        out["task_spec.json"] = json.dumps(v["spec"], indent=1).encode()
    if v.get("src_frame") is not None:
        out["src_frame.npy"] = npy_bytes(v["src_frame"])
    return out


def generate_variants(qpos, fps, log=lambda *a: None):
    """skill/variations.py's TRAIN and HELD_OUT grids on the retarget -> (train rows, held-out rows), each a
    generate() list; (None, None) when the generator or the G1 model isn't here."""
    try:
        sys.path.insert(0, str(REPO))
        from skill import variations as V
        g1 = V.G1()
    except Exception as e:  # noqa: BLE001
        log(f"variants not generated ({type(e).__name__}: {e})")
        return None, None
    import numpy as np
    q = np.asarray(qpos, np.float64)
    return V.generate(q, fps, V.TRAIN, 0, g1, log), V.generate(q, fps, V.HELD_OUT, 1000, g1, log)


def reward_config_of(run_dir):
    """What the run's tracking policy trained with: {task, policy, iterations, started_from, trained_on, feet_height,
    policies: [every policy_* folder's task, iterations, start and data], mjlab}. The bundle's arrays are the
    reference, its spec and the variants this pipeline makes for it; `policies` says what each stored policy actually
    trained on (the reference alone, or a variants run) and which checkpoint it was built off."""
    run_dir = Path(run_dir)
    rc = {"task": TASK, "policy": None, "iterations": None, "started_from": None, "trained_on": None, "feet_height": None,
          "policies": [], "mjlab": MJLAB_TRACKING,
          "note": "mjlab's tracking reward on the reference (or on every training variant, each env inside its own); "
                  "feet_height adds a foot-height term while the reference's foot is in swing (null: not used). "
                  "`policies` records what each of the run's stored policies trained on, which may differ from this bundle's variants."}
    for pdir in sorted(run_dir.glob("policy_*")):
        w = read_json(pdir / "weights.json") or {}
        user = read_json(pdir / "reward_config.json")
        if user is None and isinstance(w.get("reward_config"), str):  # trained elsewhere: the config by its name in the repo
            for cand in (REPO / "docs" / "real_athletes" / Path(w["reward_config"]).name, run_dir / Path(w["reward_config"]).name):
                user = read_json(cand)
                if user is not None:
                    break
        user = user if isinstance(user, dict) else {}
        started = w.get("started_from") if isinstance(w.get("started_from"), dict) else None
        entry = {"policy": pdir.name.removeprefix("policy_"), "task": w.get("task"),
                 "iterations": user.get("iterations") if isinstance(user.get("iterations"), int) else w.get("iterations"),
                 "started_from": {"mode": started.get("mode"), "checkpoint": Path(str(started.get("checkpoint"))).name} if started else None,
                 "trained_on": ("variants: " + Path(str(w["variants"])).parent.parent.name) if w.get("variants") else
                               ("the clip's reference" if w.get("motion_csv") else None),
                 "feet_height": user.get("feet_height") if isinstance(user.get("feet_height"), dict) else None}
        rc["policies"].append(entry)
        if entry["policy"].startswith("tracking") and rc["policy"] is None:
            rc |= {k: entry[k] for k in ("policy", "iterations", "started_from", "trained_on", "feet_height")}
            rc["task"] = entry["task"] or rc["task"]
    return rc


def inside(m):
    """One line on what the bundle holds, for the download button."""
    ref, v = m["reference"], m["variants"]
    bits = [f"G1 reference ({ref['frames']} frames at {ref['fps']:g} fps" + (", in place" if ref.get("in_place") else "") + ")",
            f"task spec ({m['task_spec']['moments']} key moments" + (f", named by {m['task_spec']['named_by']}" if m['task_spec'].get("named_by") else "") + ")"]
    tr, ho = v.get("train") or {}, v.get("held_out") or {}
    if tr.get("with_arrays"):
        bits.append(f"{tr['with_arrays']} training variants" + (f" and {ho['with_arrays']} held out" if ho.get("with_arrays") else ""))
    elif tr.get("total"):
        bits.append(f"verdicts of {tr['total']} variants")
    bits += ["reward config", "loader for mjlab"]
    return ", ".join(bits)


def readme(m):
    v = m["variants"]
    tr, ho = v.get("train") or {}, v.get("held_out") or {}
    fh = m["reward"].get("feet_height")
    lines = [
        f"# {m.get('title') or m['id']}: a training environment for the Unitree G1",
        "",
        "A phone video of a person became this environment (athlete2humanoid, https://github.com/PandaGrdn/athlete2humanoid):",
        "the person's motion retargeted onto the G1, the action read off it as ordered key moments, and variants of it",
        "to train on. Everything a policy needs is in this zip; nothing else of that repository is required.",
        "",
        "## Train",
        "",
        "```bash",
        "uv venv -p 3.12 mjlab_env && uv pip install -p mjlab_env/bin/python mjlab==1.6.0 tensorboard   # or pip, in a venv",
        "source mjlab_env/bin/activate",
        f"unzip {m['id']}.a2h-env.zip -d {m['id']} && cd {m['id']}",
        f"python -m a2h_env info  ../{m['id']}.a2h-env.zip",
        f"python -m a2h_env train ../{m['id']}.a2h-env.zip --iters 1500 --num-envs 4096    # a GPU; --device cpu --num-envs 8 for a smoke test",
        "```",
        "",
        "`train` uses every kept training variant (`--variant train`, the default); `--variant reference` trains on the clip's",
        "retarget alone, `--variant <name>` on one variant. Checkpoints (rsl_rl `model_<iteration>.pt`) land in",
        "`<bundle>.train/logs/<timestamp>/`; mjlab's `play` script or the repository's `mjlab_rollout.py` run them. The zip itself on",
        "`PYTHONPATH` works too (`PYTHONPATH=<zip> python -m a2h_env ...`, not with `python -I`). On macOS under a cleared",
        "environment keep `/usr/sbin` on `PATH` (MuJoCo's warp backend calls `sysctl`).",
        "",
        "```python",
        "import a2h_env",
        f"b = a2h_env.load_bundle('{m['id']}.a2h-env.zip')     # numpy only: b.qpos (T, 36), b.fps, b.spec, b.train, b.held_out",
        "env = a2h_env.make_env(b, num_envs=4096)                    # mjlab ManagerBasedRlEnv: every kept training variant (the default)",
        "env = a2h_env.make_env(b, variant='reference')              # ... or the reference alone; or a variant by name",
        "```",
        "",
        "Dependencies: numpy to read; mjlab 1.6.0 (brings torch, mujoco, mujoco-warp, rsl-rl) and tensorboard to train.",
        "",
        "## What each file is",
        "",
        "| File | What |",
        "|---|---|",
        "| `manifest.json` | ids, versions (retarget, pipeline commit, spec hash), counts, every file's size |",
        f"| `reference/qpos.npy` | the retargeted G1, ({m['reference']['frames']}, 36): root xyz, root quaternion (w, x, y, z), 29 joints, at {m['reference']['fps']:g} fps"
        + ("; run in place (the clip's travel removed)" if m["reference"].get("in_place") else "") + " |",
        "| `reference/meta.json` | the clip's processing record: fps, pose front end, retarget version and parameters, in-place details |",
        "| `reference/contacts.npy` | (T, 2) bool, left and right foot on the floor, by the task spec's rule (when the exporter had MuJoCo) |",
        f"| `task_spec.json` | the action as ordered key moments ({m['task_spec']['moments']} poses, {m['task_spec']['flights']} flights"
        + (f", named by {m['task_spec']['named_by']} as `muse_label`" if m["task_spec"].get("named_by") else "")
        + "): goals, tolerances, relevance per body feature, and the reward machine's weights (skill/keypose.py) |",
        "| `importance.json` | goal relevance per reference frame, for time-weighted style terms (when the run had it) |",
        "| `reward_config.json` | the tracking reward: mjlab's terms and weights, and `feet_height`"
        + (f" (weight {fh.get('weight')})" if isinstance(fh, dict) else " (not used)") + " |",
        f"| `variants/train/` | `index.json`: {tr.get('total', 0)} variants (mirror, tempo, amplitude, warp), {tr.get('kept', 0)} kept, the rest with the physics-filter rule they broke; "
        f"{tr.get('with_arrays', 0)} kept ones with `qpos.npy`, their own `task_spec.json` and `src_frame.npy` (the clip frame each frame shows) |",
        f"| `variants/held_out/` | the same for the held-out grid (tempos and amplitudes outside the training range): {ho.get('kept', 0)} of {ho.get('total', 0)} kept, {ho.get('with_arrays', 0)} with arrays |",
        "| `credit.json` | the source video's attribution and licence, when it has one |",
        "| `extras/silhouette.mp4` | SAM 3.1's silhouette of the person (what V-JEPA 2 compares), when small |",
        "| `extras/video_latent.npz` | the video's V-JEPA 2 latent (`emb` per window, `dense` per step), when small |",
        "| `robot/` | which G1 model the environment uses (mjlab's) and the Menagerie licence |",
        "| `a2h_env/` | the loader: `load_bundle`, `make_env`, `train`, and the `info` / `train` commands |",
        "",
        "## Notes",
        "",
        "- The environment is mjlab's G1 motion tracking (BeyondMimic re-implementation), with each env kept inside its own",
        "  variant when training on all of them. The key-moment task reward of the repository's task policies (`skill/`) is",
        "  not vendored here; the spec is, so another reward can be built on it.",
        "- The arrays are the reference, its spec and the variants this pipeline generates for it. What each of the run's stored",
        "  policies actually trained on (the reference alone or a variants run, and the checkpoint it was built off) is",
        "  `reward_config.json` `policies`; `manifest.json` `variants.source` says where these variants came from.",
        "- Every variant was kept only if its own reference completes its spec and passes the physics filter (joint speed cap,",
        "  ballistic flights, no lost flight, no floor penetration). `index.json` says why the others were dropped.",
    ]
    if m.get("credit"):
        c = m["credit"]
        lines.append(f"- Source video: {c['text']}" + (f" ({c['licence']})" if c.get("licence") else "") + (f", {c['url']}" if c.get("url") else "") + ".")
    lines.append("- The G1 model is from MuJoCo Menagerie (BSD-3-Clause); mjlab ships its own copy.")
    return "\n".join(lines) + "\n"


def write_zip(out, parts, manifest):
    """Write the zip: manifest.json first (with every file's size), README.md, then the parts. -> manifest."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest["files"] = [{"path": n, "bytes": parts.size(n)} for n in sorted(parts.files)]
    manifest["inside"] = inside(manifest)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=out.name + ".", suffix=".tmp")  # unique: concurrent builders never share one
    try:
        with os.fdopen(fd, "wb") as f, zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
            z.writestr("README.md", readme(manifest))
            for name in sorted(parts.files):
                d = parts.files[name]
                if isinstance(d, Path):
                    z.write(d, name)
                else:
                    z.writestr(name, d)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    manifest["bytes"] = out.stat().st_size
    return manifest


def add_loader(parts):
    for f in LOADER_FILES:
        parts.add_file(f"a2h_env/{f}", PACKAGE / f)
    lic = REPO / "assets" / "mujoco_menagerie" / "unitree_g1" / "LICENSE"
    parts.add_file("robot/LICENSE", lic, 64 * 1024)
    parts.add("robot/README.md", ("The environment runs on mjlab's Unitree G1 (mjlab.asset_zoo.robots.unitree_g1, g1.xml, 29 DoF), which mjlab\n"
                                  "installs. The reference was retargeted on MuJoCo Menagerie's unitree_g1/scene.xml; the two share the joint\n"
                                  "order, so qpos[:, 7:] goes straight into mjlab's joints. Licence: BSD-3-Clause (robot/LICENSE).\n").encode())


def spec_summary(spec):
    stages = [st for st in spec.get("stages", []) if isinstance(st, dict)]
    named = spec.get("named_by") if isinstance(spec.get("named_by"), dict) else None
    return {"file": "task_spec.json", "name": spec.get("name"), "type": spec.get("type"), "stages": len(stages),
            "moments": sum(st.get("kind") == "pose" for st in stages), "flights": sum(st.get("kind") == "flight" for st in stages),
            "named_by": named and named.get("model"), "summary": spec.get("summary"),
            "self_check": {k: spec["self_check"].get(k) for k in ("stages_completed", "stages_total", "success")}
            if isinstance(spec.get("self_check"), dict) else None}


def base_manifest(rid, title, clip, qpos_bytes, fps, spec, meta, credit):
    shape, dtype, _ = npy_header(qpos_bytes)
    rv = meta.get("retarget_version") if isinstance(meta, dict) else None
    retarget_git = rv.get("git") if isinstance(rv, dict) else None
    commit, dirty = git_state()
    return {
        "format": FORMAT, "format_version": FORMAT_VERSION, "id": rid, "title": title, "clip": clip,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # commit: the pipeline that made the retarget (the clip's meta.json), when recorded; exported_at_commit: the checkout
        # that wrote this bundle, which may have had uncommitted changes (dirty)
        "pipeline": {"commit": retarget_git, "retarget_version": rv.get("version") if isinstance(rv, dict) else rv,
                     "retarget_git": retarget_git, "exported_at_commit": commit, "dirty": dirty,
                     "pose_frontend": (meta or {}).get("frontend"), "loader_version": FORMAT_VERSION},
        "spec_sha256": sha256(canonical(spec)), "qpos_sha256": sha256(qpos_bytes),
        "reference": {"file": "reference/qpos.npy", "frames": shape[0], "fps": float(fps), "duration_s": round(shape[0] / float(fps), 3),
                      "dtype": dtype, "in_place": bool((meta or {}).get("in_place")), "contacts": None, "layout": ROBOT["qpos_layout"]},
        "task_spec": spec_summary(spec),
        "variants": {"train": None, "held_out": None, "source": None},
        "reward": {"file": "reward_config.json", "task": TASK, "feet_height": None},
        "robot": ROBOT, "credit": credit, "extras": {},
        "loader": {"package": "a2h_env/", "requires": "mjlab==1.6.0 (torch, mujoco, mujoco-warp, rsl-rl-lib), numpy"},
    }


# --- stored runs ---------------------------------------------------------------------------------------------------------
def target_clip(run_dir):
    spec = read_json(Path(run_dir) / "task_spec.json")
    name = spec.get("name") if isinstance(spec, dict) else None
    if not isinstance(name, str) or not re.fullmatch(r"[\w.-]+", name):
        return None, spec
    return (name if (Path(run_dir) / "clips" / name / "qpos.npy").is_file() else None), spec


def variants_run(run_dir):
    """The sibling outputs/remote_variants_<tag>/ run whose variants this run's policies trained on, if any."""
    run_dir = Path(run_dir)
    tag = run_dir.name.removeprefix("remote_latent_").removeprefix("remote_latent")
    d = run_dir.parent / f"remote_variants_{tag}" if tag else None
    return d if d and (d / "variants" / "index.json").is_file() else None


def export_run(run_dir, out=None, generate=False, variants_dir=None, log=lambda *a: None):
    """A run folder -> its bundle at `out` (default <run_dir>/env.zip). -> the manifest (with "bytes"), or None when the
    run has no task spec naming a clip with a retarget. generate: make the TRAIN and HELD_OUT grids when the run has no
    variants of its own (needs numpy, mujoco, scipy, torch and the G1 model)."""
    run_dir = Path(run_dir)
    clip, spec = target_clip(run_dir)
    if clip is None:
        return None
    rid = run_id(run_dir)
    cdir = run_dir / "clips" / clip
    web = read_json(run_dir / "web.json") or {}
    meta = read_json(cdir / "meta.json") or {}
    qpos_bytes = (cdir / "qpos.npy").read_bytes()
    fps = float(meta.get("fps") or spec.get("fps") or 30.0)
    credit = clean_credit(web.get("credit"))
    title = web.get("title") if isinstance(web.get("title"), str) else None
    m = base_manifest(rid, title, clip, qpos_bytes, fps, spec, meta, credit)
    m["run"] = run_dir.name
    parts = Parts()
    parts.add("reference/qpos.npy", qpos_bytes)
    parts.add_json("reference/meta.json", reference_meta(meta, fps, m["reference"]["frames"]))
    parts.add_json("task_spec.json", spec)
    parts.add_json("importance.json", read_json(run_dir / "importance.json"))
    parts.add_json("credit.json", credit)
    rc = reward_config_of(run_dir)
    parts.add_json("reward_config.json", rc)
    m["reward"] |= {k: rc[k] for k in ("task", "feet_height", "policy", "iterations", "started_from", "trained_on")}
    # contacts (MuJoCo + the G1 model: the repo's environment, or the live image)
    try:
        import numpy as np
        pos = body_positions(np.load(io.BytesIO(qpos_bytes)))
    except ImportError:
        pos = None
    if pos is not None and isinstance(spec.get("foot0"), (int, float)):
        parts.add("reference/contacts.npy", npy_bytes(contacts_from_positions(pos, spec["foot0"], spec.get("down", DOWN))))
        m["reference"]["contacts"] = "reference/contacts.npy"
    # variants: the sibling variants run, else the run's view (verdicts), else generated
    vdir = Path(variants_dir) if variants_dir else variants_run(run_dir)
    if vdir is not None:
        m["variants"]["train"] = add_variant_set(parts, "train", read_json(vdir / "variants" / "index.json"), lambda n: from_folder(vdir / "variants" / n))
        m["variants"]["held_out"] = add_variant_set(parts, "held_out", read_json(vdir / "held_out" / "index.json"), lambda n: from_folder(vdir / "held_out" / n))
        m["variants"]["source"] = vdir.name
    else:
        train = held = None
        if generate:
            log(f"{rid}: generating variants")
            train, held = generate_variants(np.load(io.BytesIO(qpos_bytes)), fps, log) if pos is not None else (None, None)
        if train is not None:
            by = {v["name"]: v for v in train + held}
            m["variants"]["train"] = add_variant_set(parts, "train", train, lambda n: from_generated(by[n]))
            m["variants"]["held_out"] = add_variant_set(parts, "held_out", held, lambda n: from_generated(by[n]))
            m["variants"]["source"] = "generated"
        else:
            view = read_json(run_dir / "variants.json")
            rows = view.get("rows") if isinstance(view, dict) else None
            if rows:
                m["variants"]["train"] = add_variant_set(parts, "train", rows, lambda n: None)
                m["variants"]["source"] = "verdicts only (variants.json)"
    # extras
    if parts.add_file("extras/silhouette.mp4", cdir / "silhouette.mp4", MAX_EXTRA_BYTES):
        m["extras"]["silhouette"] = "extras/silhouette.mp4"
    if parts.add_file("extras/video_latent.npz", cdir / "video_latent.npz", MAX_EXTRA_BYTES):
        m["extras"]["video_latent"] = "extras/video_latent.npz"
    add_loader(parts)
    out = Path(out) if out else run_dir / ZIP_NAME
    return write_zip(out, parts, m)


# --- a live build (server/live_worker.py) --------------------------------------------------------------------------------
def export_live(out_dir, qpos, fps, spec, variants=None, pos=None, name="clip", job_id=None, meta=None):
    """An uploaded video's build -> <out_dir>/env.zip. qpos (T, 36) at fps; spec its keypose spec; variants:
    skill.variations.generate()'s list (kept ones go in with their arrays); pos (T, 14, 3) body positions for the
    contacts. -> the manifest."""
    import numpy as np
    out_dir = Path(out_dir)
    rid = f"live-{str(job_id)[:8]}" if job_id else "live"  # server/live.py env_filename: the same name on download
    qpos_bytes = npy_bytes(np.asarray(qpos, np.float64))
    meta = dict(meta or {}) | {"fps": float(fps), "frontend": (meta or {}).get("frontend", "mediapipe"), "live": True}
    m = base_manifest(rid, f"Live build: {name}", name, qpos_bytes, fps, spec, meta, None)
    parts = Parts()
    parts.add("reference/qpos.npy", qpos_bytes)
    parts.add_json("reference/meta.json", reference_meta(meta, fps, len(qpos)))
    parts.add_json("task_spec.json", spec)
    rc = {"task": TASK, "policy": None, "iterations": None, "feet_height": None, "mjlab": MJLAB_TRACKING,
          "note": "mjlab's tracking reward; feet_height not set (a live build trains nothing)."}
    parts.add_json("reward_config.json", rc)
    if pos is not None and isinstance(spec.get("foot0"), (int, float)):
        parts.add("reference/contacts.npy", npy_bytes(contacts_from_positions(pos, spec["foot0"], spec.get("down", DOWN))))
        m["reference"]["contacts"] = "reference/contacts.npy"
    if variants:
        by = {v["name"]: v for v in variants}
        m["variants"]["train"] = add_variant_set(parts, "train", variants, lambda n: from_generated(by[n]))
        m["variants"]["source"] = "generated on the host (preview grid)"
    add_loader(parts)
    return write_zip(out_dir / ZIP_NAME, parts, m)


def manifest_of(zip_path):
    """A bundle's manifest (with "bytes"), read from the zip; None if it isn't one."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            m = json.loads(z.read("manifest.json"))
        if m.get("format") != FORMAT:
            return None
        m["bytes"] = Path(zip_path).stat().st_size
        return m
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None


def is_current(zip_path, run_dir):
    """Whether a bundle still matches its run folder: the manifest's spec and retarget hashes equal the folder's
    task_spec.json and clips/<clip>/qpos.npy. False for a missing or foreign zip, or a run with no spec."""
    m = manifest_of(zip_path)
    clip, spec = target_clip(run_dir)
    if m is None or clip is None:
        return False
    try:
        qpos = (Path(run_dir) / "clips" / clip / "qpos.npy").read_bytes()
    except OSError:
        return False
    return m.get("spec_sha256") == sha256(canonical(spec)) and m.get("qpos_sha256") == sha256(qpos) and m.get("clip") == clip


def main(runs, out_dir=None, generate=True, force=False, log=print):
    """scripts/export_env.py: each run folder -> its env.zip (in the run folder, or out_dir/<run>.a2h-env.zip). An
    existing zip is kept while it matches the run's spec and retarget (is_current), rebuilt otherwise or with force."""
    done = []
    for r in runs:
        r = Path(r)
        out = Path(out_dir) / f"{run_id(r)}.a2h-env.zip" if out_dir else r / ZIP_NAME
        if out.is_file() and not force:
            m = manifest_of(out)
            if m and is_current(out, r):
                log(f"{out}: already there and current ({m['bytes'] / 1e6:.1f} MB; --force to rebuild)")
                done.append((r, m))
                continue
            if m:
                log(f"{out}: stale (the spec or the retarget changed); rebuilding")
        m = export_run(r, out, generate=generate, log=log)
        if m is None:
            log(f"{r}: no task spec naming a clip with a retarget; skipped")
            continue
        v = m["variants"]
        tr = v.get("train") or {}
        log(f"{out}: {m['bytes'] / 1e6:.1f} MB; {m['reference']['frames']} frames, {m['task_spec']['moments']} key moments, "
            f"variants {tr.get('with_arrays', 0)}/{tr.get('total', 0)} with arrays ({v.get('source') or 'none'})")
        done.append((r, m))
    return done
