"""Finished runs on disk (outputs/remote_* folders) -> JSON the frontend renders.

A run is any folder with an eval.json. Every other file is optional: missing or malformed files
just leave their fields empty. Robustness results (robust.json) live in their own folder
(remote_robust[_tag]) and are attached to the run whose eval.json they reproduce.
"""

import fnmatch
import json
import re
from pathlib import Path

from .stages import classify, redact

POLICY_INFO = {
    "tracking": ("Tracking", "Joint-target imitation (mjlab baseline)"),
    "hybrid": ("Hybrid", "Joint targets + V-JEPA 2 latent reward"),
    "latent": ("Latent", "V-JEPA 2 latent reward; retarget used only as observation and resets"),
    "task": ("Task", "The video's key moments as the reward (keypose spec); retarget as input, weak style terms"),
    "taskphase": ("Task (blind)", "The keypose task with only a clock and the stage as input, no retarget"),
}
METRIC_KEYS = ["aligned", "dtw", "chamfer", "dtw_rank_vs_bank", "reward_net_aligned", "real_aligned_in_reward_space",
               "steps", "fps", "falls", "mean_joint_error_rad", "mean_torso_position_error_m", "max_torso_position_error_m"]
SERVABLE = {".mp4", ".webm", ".json", ".csv", ".onnx", ".pt", ".npz", ".npy", ".png", ".jpg"}
_cache = {}


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def run_id(folder):
    return folder.name[len("remote_"):] if folder.name.startswith("remote_") else folder.name


def find_runs(runs_dir, exclude=()):
    """{run id: folder}, newest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return {}
    found = [d for d in runs_dir.iterdir()
             if d.is_dir() and (d / "eval.json").is_file() and not any(fnmatch.fnmatch(d.name, p) for p in exclude)]
    found.sort(key=lambda d: (_finished(d) or "", d.stat().st_mtime), reverse=True)
    return {run_id(d): d for d in found}


def _finished(d):
    s = read_json(d / "status.json") or {}
    return s.get("finished") or s.get("started")


def _cmd_flag(cmd, flag):
    m = re.search(rf"{flag}\s+(\S+)", cmd)
    return m.group(1).strip("'\"") if m else None


def file_url(rid, rel):
    return f"/api/runs/{rid}/files/{rel}"


def _file(folder, rid, rel, label=None, kind=None):
    p = folder / rel
    if not p.is_file():
        return None
    return {"label": label or p.name, "path": rel, "url": file_url(rid, rel), "size": p.stat().st_size, "kind": kind}


def train_curve(policy_dir, max_points=120):
    """[[iteration, mean reward], ...] from curve.json, else parsed from rsl_rl's train.log."""
    pre = read_json(policy_dir / "curve.json")
    if isinstance(pre, list):
        return pre
    log = policy_dir / "train.log"
    if not log.is_file():
        return []
    key = (str(log), log.stat().st_mtime)
    if key not in _cache:
        pts, it = [], None
        with open(log, errors="ignore") as f:
            for line in f:
                if "Learning iteration" in line:
                    m = re.search(r"Learning iteration (\d+)/(\d+)", line)
                    it = int(m.group(1)) if m else None
                elif it is not None and "Mean reward:" in line:
                    m = re.search(r"Mean reward:\s*(-?[\d.]+)", line)
                    if m:
                        pts.append([it, float(m.group(1))])
                    it = None
        step = max(1, len(pts) // max_points)
        _cache[key] = pts[::step] + (pts[-1:] if pts and (len(pts) - 1) % step else [])
    return _cache[key]


def _total_iters(folder, policy_dir):
    for ck in sorted(policy_dir.glob("model_*.pt")):
        m = re.match(r"model_(\d+)\.pt", ck.name)
        if m:
            return int(m.group(1)) + 1
    return None


def _policies(folder, rid, ev):
    out = []
    for key, m in ev.items():
        if key == "retarget" or key.startswith("_") or not isinstance(m, dict):
            continue
        base = key.removeprefix("prev_")
        label, blurb = POLICY_INFO.get(base, (base.title(), ""))
        pdir = folder / f"policy_{key}"
        ck = sorted(pdir.glob("model_*.pt")) if pdir.is_dir() else []
        onnx = sorted(pdir.glob("*.onnx")) if pdir.is_dir() else []
        out.append({
            "key": key, "base": base, "label": label + (" (previous run)" if key.startswith("prev_") else ""),
            "description": blurb, "previous": key.startswith("prev_"),
            "metrics": {k: _num(m.get(k)) for k in METRIC_KEYS},
            "video": _file(folder, rid, f"policy_{key}/policy.mp4"),
            "checkpoint": _file(folder, rid, f"policy_{key}/{ck[-1].name}", kind="checkpoint") if ck else None,
            "onnx": _file(folder, rid, f"policy_{key}/{onnx[0].name}", kind="onnx") if onnx else None,
            "curve": train_curve(pdir) if pdir.is_dir() else [],
            "iterations": _total_iters(folder, pdir) if pdir.is_dir() else None,
        })
    order = {"task": 0, "tracking": 1, "hybrid": 2, "latent": 3, "taskphase": 4}
    out.sort(key=lambda p: (p["previous"], order.get(p["base"], 9)))
    return out


def _clips(folder, rid, target):
    cdir = folder / "clips"
    if not cdir.is_dir():
        return []
    out = []
    for c in sorted(d for d in cdir.iterdir() if d.is_dir()):
        meta = read_json(c / "meta.json") or {}
        rep = read_json(c / "retarget_report.json") or {}
        errs = {k: v[0] for k, v in rep.items() if isinstance(v, list) and v and _num(v[0]) is not None}
        rel = f"clips/{c.name}"
        out.append({
            "name": c.name, "is_target": c.name == target,
            "fps": _num(meta.get("fps")), "frames": _num(meta.get("frames")),
            "view_azimuth": _num(meta.get("view_azimuth")), "sam": meta.get("sam"),
            "retarget_error_cm": errs,
            "retarget_error_mean_cm": round(sum(errs.values()) / len(errs), 2) if errs else None,
            "side_by_side": _file(folder, rid, f"{rel}/side_by_side.mp4"),
            "silhouette": _file(folder, rid, f"{rel}/silhouette.mp4"),
            "motion_csv": _file(folder, rid, f"{rel}/motion.csv", f"{c.name}_motion.csv", "motion"),
        })
    out.sort(key=lambda c: (not c["is_target"], c["name"]))
    return out


def robust_for(runs_dir, folder, ev):
    """The robust.json whose reference_check reproduces this run's eval.json, else one named after its tag."""
    own = read_json(folder / "robust.json")
    if isinstance(own, dict):
        return own
    for p in sorted(Path(runs_dir).glob("remote_robust*/robust.json")):
        r = read_json(p)
        ref = r.get("reference_check") if isinstance(r, dict) else None
        shared = [k for k in (ref or {}) if isinstance(ev.get(k), dict) and isinstance(ref[k], dict)]
        if shared and all(abs((_num(ref[k].get("eval_json_aligned")) or 9) - (_num(ev[k].get("aligned")) or 0)) < 1e-4
                          for k in shared):
            return r
    tag = folder.name.removeprefix("remote_latent")
    p = Path(runs_dir) / f"remote_robust{tag}" / "robust.json"
    r = read_json(p) if p.is_file() else None
    return r if isinstance(r, dict) else None


def _robust_view(r):
    if not isinstance(r, dict):
        return None
    conds = []
    for c in r.get("conditions") or []:
        if not isinstance(c, dict):
            continue
        pols = {}
        for k, v in (c.get("policies") or {}).items():
            if isinstance(v, dict):
                pols[k] = {m: _num(v.get(m)) for m in ("fall_rate", "survival", "torso_err_m", "joint_err_rad", "vjepa_aligned", "vjepa_dtw")}
        conds.append({"suite": c.get("suite"), "level": c.get("level"), "params": c.get("params") or {}, "policies": pols})
    meta = r.get("meta") or {}
    return {"conditions": conds, "summary": r.get("summary") or {},
            "reps": _num(meta.get("reps")), "step_hz": _num(meta.get("step_hz"))}


def replay_lines(folder):
    """[(stage, line)] worth showing from a run's job.log, in order."""
    log = folder / "job.log"
    if not log.is_file():
        return []
    out = []
    with open(log, errors="ignore") as f:
        for raw in f:
            for part in raw.split("\r"):
                stage, text = classify(part)
                if stage and text and "Learning iteration" not in text:
                    out.append((stage, text[:300]))
    return out


def summarize(runs_dir, rid, folder, detail=False):
    ev = read_json(folder / "eval.json")
    ev = ev if isinstance(ev, dict) else {}
    st = read_json(folder / "status.json") or {}
    cmd = " ".join(st.get("cmd") or []) if isinstance(st.get("cmd"), list) else str(st.get("cmd") or "")
    web = read_json(folder / "web.json") or {}  # optional hand-written title/description
    target = web.get("target") or _cmd_flag(cmd, "--target")
    policies = _policies(folder, rid, ev)
    current = [p for p in policies if not p["previous"]]
    best = max(current, key=lambda p: p["metrics"]["aligned"] or -1, default=None)
    tag = rid.removeprefix("latent").strip("_")
    rob = robust_for(runs_dir, folder, ev)
    iters = _cmd_flag(cmd, "--iters")
    s = {
        "id": rid,
        "title": web.get("title") or f"{(target or rid).replace('_', ' ').title()}" + (f" · {tag}" if tag else ""),
        "description": web.get("description") or "",
        "target": target, "tag": tag or None,
        "iterations": int(iters) if iters and iters.isdigit() else None,
        "started": st.get("started"), "finished": st.get("finished"), "state": st.get("state") or "done",
        # hero: the person next to the G1 retarget; video: what V-JEPA 2 compares (silhouettes)
        "hero": (_file(folder, rid, f"clips/{target}/side_by_side.mp4") if target else None)
                or _file(folder, rid, "comparison.mp4"),
        "video": _file(folder, rid, "comparison.mp4"),
        "poster": _file(folder, rid, "poster.jpg"),
        "policies_trained": [p["key"] for p in current],
        "headline": {
            "retarget_aligned": _num((ev.get("retarget") or {}).get("aligned")),
            "best_policy": best["key"] if best else None,
            "best_aligned": best["metrics"]["aligned"] if best else None,
            "falls": sum(p["metrics"]["falls"] or 0 for p in current) if current else None,
        },
        "has_robust": rob is not None,
        "has_task_spec": safe_path(folder, TASK_SPEC) is not None,
        "clip_count": len([d for d in (folder / "clips").iterdir() if d.is_dir()]) if (folder / "clips").is_dir() else 0,
    }
    if not detail:
        return s
    tgt_clip = folder / "clips" / (target or "")
    downloads = [d for d in [
        *[p["onnx"] for p in current if p["onnx"]],
        *[p["checkpoint"] for p in current if p["checkpoint"]],
        _file(folder, rid, f"clips/{target}/motion.csv", f"{target}_motion.csv", "motion") if target and tgt_clip.is_dir() else None,
        _file(folder, rid, "reward.pt", "reward.pt", "reward"),
        _file(folder, rid, "eval.json", "eval.json", "metrics"),
    ] if d]
    s.update({
        "reference": {k: _num(v) for k, v in (ev.get("retarget") or {}).items()},
        "policies": policies,
        "bank_top5": [x for x in (ev.get("_bank_top5_dtw") or []) if isinstance(x, list) and len(x) == 2],
        "distill": read_json(folder / "distill_report.json"),
        "robust": _robust_view(rob),
        "clips": _clips(folder, rid, target),
        "downloads": downloads,
    })
    return s


def safe_path(folder, rel):
    """The file under folder named by rel, or None when rel escapes it or isn't a servable type."""
    try:
        p = (folder / rel).resolve()
        p.relative_to(folder.resolve())
    except (ValueError, OSError):
        return None
    if not p.is_file() or p.suffix.lower() not in SERVABLE or any(part.startswith(".") for part in Path(rel).parts):
        return None
    return p


# --- task environment (task_spec.json) -----------------------------------------------------------
# skill/keypose.py's spec (type "keypose": key moments for any action) or skill/task.py's older one
# (kinds stand/squat/jump/land). Served as a small view: names, kinds, times, flags and body parts, never
# the raw goals and tolerances, with every string cleaned and every list bounded.
TASK_SPEC = "task_spec.json"
MAX_SPEC_BYTES = 2 * 1048576
MAX_SPEC_STAGES = 200
_OLD_KINDS = {"stand": "settle", "land": "settle", "squat": "pose", "jump": "flight"}
_LEG = {"ankle", "ankles", "foot", "feet", "knee", "knees", "hip", "hips", "toe", "toes", "heel", "leg", "legs", "shin", "thigh"}
_ARM = {"wrist", "wrists", "hand", "hands", "elbow", "elbows", "shoulder", "shoulders", "arm", "arms", "finger"}
_TORSO = {"pelvis", "torso", "waist", "chest", "spine", "head", "neck", "trunk", "base", "com"}


def _text(v, n=160):
    if not isinstance(v, str):
        return None
    return redact(re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069]", " ", v)).strip()[:n] or None


def _sec(v):
    v = _num(v)
    return round(float(v), 3) if v is not None and v == v and abs(v) < 1e6 else None


def body_part(feature):
    """'right ankle x' -> 'right leg', 'left wrist z' -> 'left arm', 'pelvis height' -> 'torso/pelvis',
    'root turn' -> 'root'; None for anything unrecognised."""
    if not isinstance(feature, str):
        return None
    words = re.findall(r"[a-z]+", feature.lower())
    if "root" in words:
        return "root"
    side = next((w for w in words if w in ("left", "right")), None)
    for group, name in ((_LEG, "leg"), (_ARM, "arm")):
        if any(w in group for w in words):
            return f"{side} {name}" if side else f"{name}s"
    if any(w in _TORSO for w in words):
        return "torso/pelvis"
    return None


def _body_parts(relevance, features, top=4, floor=0.3):
    """The parts a key moment's goal weighs most: max relevance per part, those within `floor` of the top."""
    if not isinstance(relevance, list) or len(relevance) != len(features):
        return []
    best = {}
    for f, r in zip(features, relevance):
        part, r = body_part(f), _num(r)
        if part and r is not None and r > 0:
            best[part] = max(best.get(part, 0.0), float(r))
    if not best:
        return []
    hi = max(best.values())
    keep = sorted(((p, w) for p, w in best.items() if w >= floor * hi), key=lambda x: -x[1])[:top]
    return [{"part": p, "weight": round(w / hi, 2)} for p, w in keep]


def _feet(v):
    return [bool(x) for x in v] if isinstance(v, list) and len(v) == 2 and all(isinstance(x, bool) for x in v) else None


def _spec_stage(st, features, keypose, fps):
    kind = st.get("kind") if isinstance(st.get("kind"), str) else ""
    if keypose:
        k = kind if kind in ("pose", "flight", "settle") else "pose"
        name = _text(st.get("name"), 120) or k
        parts = _body_parts(st.get("relevance"), features) if k == "pose" else []
        feet = _feet(st.get("feet_down"))
    else:  # stand / squat / jump / land
        k = _OLD_KINDS.get(kind, "pose")
        name = _text(kind, 40) or k
        parts = [{"part": "torso/pelvis", "weight": 1.0}] if kind == "squat" else []
        feet = None if k == "flight" else [True, True]
    level = _num(st.get("level"))
    frame = _num(st.get("frame"))
    frame = int(frame) if frame is not None and 0 <= frame < 1e7 else None
    start, end = _sec(st.get("start_s")), _sec(st.get("end_s"))
    # when the retarget reaches it: a pose's key frame, a flight's take-off, a settle's start
    at = (round(frame / fps, 3) if frame is not None and fps else end) if k == "pose" else start
    return {
        "name": name, "kind": k, "start_s": start, "end_s": end, "at_s": at,
        "optional": st.get("optional") is True,
        "level": round(float(level), 2) if level is not None and abs(level) < 1e6 else None,
        "frame": frame,
        "feet_down": feet, "body_parts": parts,
        "clearance_m": _sec(st.get("clearance")) if k == "flight" else None,
    }


def _self_check(spec, ev):
    sc = spec.get("self_check")
    if isinstance(sc, dict):
        done, total = _num(sc.get("stages_completed")), _num(sc.get("stages_total"))
        if done is not None and total is not None:
            return {"stages_completed": int(done), "stages_total": int(total), "success": sc.get("success") is True,
                    "stages_skipped": int(_num(sc.get("stages_skipped")) or 0), "fell_at_s": _sec(sc.get("fell_at_s")),
                    "source": "spec"}
    # older specs: the evaluation scored the retarget against the spec ("task_stages": "9/9")
    ref = (ev or {}).get("retarget") if isinstance(ev, dict) else None
    m = re.fullmatch(r"(\d{1,4})/(\d{1,4})", str((ref or {}).get("task_stages") or ""))
    if m:
        return {"stages_completed": int(m.group(1)), "stages_total": int(m.group(2)),
                "success": (ref or {}).get("task_success") is True, "stages_skipped": 0,
                "fell_at_s": _sec((ref or {}).get("task_fell_at_s")), "source": "eval"}
    return None


def _reward(spec, keypose):
    """How the stage machine pays (skill/keypose.py / skill/task.py WEIGHTS, seconds of unit reward)."""
    w = spec.get("weights") if isinstance(spec.get("weights"), dict) else {}
    num = {k: _num(w.get(k)) for k in ("progress", "stage", "jump", "done", "hold")}
    num = {k: round(float(v), 3) if v is not None and abs(v) < 1e4 else 0.0 for k, v in num.items()}
    notes = ["Progress toward the next key moment pays as it happens, gated on staying upright.",
             "Each key moment reached pays a bonus; a jump pays extra for its height.",
             "Finishing pays a bonus scaled by the moments reached."]
    if num["hold"]:
        notes.append("Once done, holding the end pose pays per second, and the style terms stop.")
    elif keypose:
        notes.append("This spec predates the end-pose hold (trained without it).")
    notes += ["A fall gives back what the attempt earned since the last steady stance.",
              "Timing is free: only the order of the moments counts."]
    if keypose:
        notes.append("Optional moments pay too, but may be skipped.")
    return {"weights": num, "notes": notes}


def task_spec_view(spec, ev=None):
    """The view of a parsed task_spec.json the frontend renders, or None when it isn't one."""
    if not isinstance(spec, dict) or not isinstance(spec.get("stages"), list):
        return None
    keypose = spec.get("type") == "keypose"
    feats = spec.get("features") if isinstance(spec.get("features"), list) else []
    feats = [f if isinstance(f, str) else "" for f in feats[:256]]
    fps = _num(spec.get("fps"))
    fps = float(fps) if fps is not None and 0 < fps < 1e4 else None
    stages = [_spec_stage(st, feats, keypose, fps) for st in spec["stages"][:MAX_SPEC_STAGES] if isinstance(st, dict)]
    stages = [st for st in stages if st["start_s"] is not None and st["end_s"] is not None and st["at_s"] is not None]
    if not stages:
        return None
    duration = _sec(spec.get("duration_s"))
    if duration is None or duration <= 0:
        duration = max(st["end_s"] for st in stages)
    poses = [st for st in stages if st["kind"] == "pose"]
    return {
        "format": "keypose" if keypose else "phases",
        "name": _text(spec.get("name"), 80),
        "summary": _text(spec.get("summary"), 600),
        "duration_s": duration,
        "stages": stages,
        "counts": {"moments": len(poses), "required": sum(not st["optional"] for st in poses),
                   "optional": sum(st["optional"] for st in poses),
                   "flights": sum(st["kind"] == "flight" for st in stages)},
        "body_parts": sorted({p["part"] for st in stages for p in st["body_parts"]}),
        "self_check": _self_check(spec, ev),
        "reward": _reward(spec, keypose),
        "warnings": [w for w in (_text(x, 200) for x in (spec.get("warnings") or [])[:10]
                                 if isinstance(spec.get("warnings"), list)) if w],
    }


def task_spec(folder):
    """The task environment of a run (or a live job's output) folder, or None when it has no readable spec."""
    if folder is None:
        return None
    p = safe_path(Path(folder), TASK_SPEC)  # contained in the folder, no dotfiles, a regular file
    if p is None:
        return None
    ev_p = safe_path(Path(folder), "eval.json")  # the self-check of older specs lives in the evaluation
    try:
        st, ev_st = p.stat(), ev_p.stat() if ev_p else None
    except OSError:
        return None
    if st.st_size > MAX_SPEC_BYTES:
        return None
    key = ("task_spec", str(p), st.st_mtime_ns, st.st_size, ev_st and ev_st.st_mtime_ns)
    if key not in _cache:
        ev = read_json(ev_p) if ev_st and ev_st.st_size <= MAX_SPEC_BYTES else None
        _cache[key] = task_spec_view(read_json(p), ev)
    return _cache[key]
