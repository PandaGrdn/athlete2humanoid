"""Finished runs on disk (outputs/remote_* folders) -> JSON the frontend renders.

A run is any folder with an eval.json. Every other file is optional: missing or malformed files
just leave their fields empty. Robustness results (robust.json) live in their own folder
(remote_robust[_tag]) and are attached to the run whose eval.json they reproduce.
"""

import fnmatch
import json
import re
from pathlib import Path

from .stages import classify

POLICY_INFO = {
    "tracking": ("Tracking", "Joint-target imitation (mjlab baseline)"),
    "hybrid": ("Hybrid", "Joint targets + V-JEPA 2 latent reward"),
    "latent": ("Latent", "V-JEPA 2 latent reward; retarget used only as observation and resets"),
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
    order = {"tracking": 0, "hybrid": 1, "latent": 2}
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
