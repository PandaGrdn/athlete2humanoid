"""Real-athlete clips (docs/REAL_ATHLETES.md): their licences, the inputs of the athletes job, and its results as the
demo's run folders.

    python experiments/real_athletes.py inputs outputs/remote_pro_ra --out outputs/athletes_inputs
        # the processed clips (remote_run.py pro --iters 0) whose spec its own retarget completes, + the warm start
    python experiments/real_athletes.py audit outputs/remote_pro_ra    # -> docs/results/real_athletes/audit.json
    python experiments/real_athletes.py assemble outputs/remote_athletes_ra
        # each clip's results -> outputs/remote_latent_ra_<short> (the run the site lists: eval.json, the policy, the
        # clip, web.json with the source's credit, robust.json), remote_variants_ra_<short> (its variants, for
        # scripts/build_variants_view.py) and remote_pushes_ra_<short> (for scripts/build_showcase.py)

CLIPS (docs/real_athletes/clips.json) is the record of every clip: where it came from, its licence and author, the
credit line the site shows, and whether it runs in place.
"""

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLIPS = ROOT / "docs" / "real_athletes" / "clips.json"
WARM_START = "remote_variants_s3_sj"  # the squat-jump variants tracker (policy_tracking_var)
TAG = "ra"


def clips():
    return {c["name"]: c for c in json.loads(CLIPS.read_text())["clips"]}


def unpack(f, dst):
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(f) as tar:
        tar.extractall(dst, filter="data")
    return dst


def inputs(a):
    """remote_run.py pro --iters 0's results -> a --reuse folder for remote_run.py athletes: clips/<name>/ for each clip
    whose own retarget completes its spec (compiled here, as variant_policy.py would), and policy_tracking_var/."""
    from skill import keypose
    src, out = Path(a.processed), Path(a.out)
    proc = src / "process"
    if not (proc / "clips").is_dir():
        unpack(src / "process" / "results.tar.gz", proc)
    known = clips()
    rows = {}
    for d in sorted((proc / "clips").iterdir()):
        spec, _ = keypose.compile_clip(d)
        sc = spec["self_check"]
        ok = sc["stages_completed"] == sc["stages_total"] and d.name in known
        rows[d.name] = {"spec": spec["summary"], "self_check": f"{sc['stages_completed']}/{sc['stages_total']}", "kept": ok}
        print(f"{d.name}: {spec['summary']}; retarget completes {sc['stages_completed']}/{sc['stages_total']}"
              + ("" if ok else " -> not trained"))
        if ok:
            shutil.copytree(d, out / "clips" / d.name, dirs_exist_ok=True)
    warm = sorted((ROOT / "outputs" / WARM_START / "policy_tracking_var").glob("model_*.pt"))[-1]
    (out / "policy_tracking_var").mkdir(parents=True, exist_ok=True)
    shutil.copy2(warm, out / "policy_tracking_var" / warm.name)
    (out / "inputs.json").write_text(json.dumps({"warm_start": f"{WARM_START}/policy_tracking_var/{warm.name}", "clips": rows}, indent=1))
    print(f"-> {out}: {sum(r['kept'] for r in rows.values())}/{len(rows)} clips, warm start {warm}")


def reretarget(a):
    """Redo a processed clip's retarget from its human.npy with this checkout's retarget.py (in place as given):
    qpos.npy, motion.csv, retarget_report.json, meta.json's retarget fields, and side_by_side.mp4 from the source video
    (resampled to 30 fps / 720p as process_clips.py does). No GPU: the pose is the stored one."""
    import subprocess
    import imageio_ffmpeg
    import mediapy
    import numpy as np
    import render
    import retarget
    d = Path(a.clip)
    meta = json.loads((d / "meta.json").read_text())
    P, fps = np.load(d / "human.npy"), meta["fps"]
    rt, contacts = retarget.G1Retargeter(), {}
    targets, rots = retarget.human_to_targets(P, rt, fps=fps, report=contacts, in_place=a.in_place)
    qpos = rt.retarget_sequence(targets, rots, fps=fps)
    report = retarget.tracking_report(rt, qpos, targets)
    np.save(d / "qpos.npy", qpos)
    retarget.export_mjlab_csv(qpos, d / "motion.csv")
    (d / "retarget_report.json").write_text(json.dumps(report, indent=1))
    meta.update(retarget_version=retarget.code_version(), retarget_warnings=contacts["warnings"],
                stance_frac=contacts["stance_frac"], in_place=contacts.get("in_place"))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    if a.video:
        tmp = d / "_clip30.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", a.video, "-vf", "fps=30,scale=-2:720",
                        "-an", str(tmp)], check=True)
        frames = np.asarray(mediapy.read_video(str(tmp)))[:len(qpos)]
        tmp.unlink()
        robot = render.render_qpos_sequence(rt.model, qpos, size=(480, 480))
        render.save_video(render.side_by_side(frames, robot), d / "side_by_side.mp4", fps=fps)
    print(f"{d.name}: in place {contacts.get('in_place')}, stance {contacts['stance_frac']}, warnings {contacts['warnings']}")


def audit(a):
    """The pipeline audit's numbers per processed clip (experiments/pipeline_audit): pose noise and bones (pose_stats),
    the retarget's IK error, joint speed and limits, the feet (retarget_audit, run in place as meta.json says), the
    camera's motion (camera.json), the spec and its self-check, and how many variants the filter keeps."""
    import numpy as np
    from experiments.pipeline_audit.pose_stats import clip_stats
    from experiments.pipeline_audit.retarget_audit import audit as retarget_audit
    from skill import keypose
    from skill import variations as V
    proc = Path(a.processed) / "process" / "clips"
    rows = {}
    for d in sorted(proc.iterdir()):
        meta = json.loads((d / "meta.json").read_text())
        P, fps = np.load(d / "human.npy"), meta["fps"]
        ip = (meta.get("in_place") or {}).get("method")
        r = retarget_audit(P, fps, in_place=ip)
        spec, _ = keypose.compile_clip(d)
        sc = spec["self_check"]
        vs = V.generate(np.load(d / "qpos.npy"), fps, log=lambda *x: None)
        cam = json.loads((d / "camera.json").read_text()) if (d / "camera.json").exists() else {}
        rows[d.name] = {
            "frames": meta["frames"], "seconds": round(meta["frames"] / fps, 2), "frontend": meta.get("frontend"),
            "in_place": meta.get("in_place"), "retarget_warnings": meta.get("retarget_warnings"),
            "camera": {"height_px": (cam.get("height") or {}).get("height_px"),
                       **{k: (cam.get("camera") or {}).get(k) for k in ("static", "compensated", "max_shift_px", "max_zoom")},
                       "cuts": len((cam.get("camera") or {}).get("cuts") or []), "warnings": cam.get("warnings")},
            "pose": clip_stats(P, fps, meta.get("view_azimuth") or 0.0),
            "retarget": {k: r[k] for k in ("tracking_cm", "joint_speed", "limits", "feet", "angle_err_deg")},
            "spec": {"summary": spec["summary"], "self_check": f"{sc['stages_completed']}/{sc['stages_total']}",
                     "warnings": spec.get("warnings")},
            "variants": {"kept": sum(v["ok"] for v in vs), "total": len(vs),
                         "dropped_for": sorted({w for v in vs if not v["ok"] for w in v.get("rules", [])})},
        }
        t = r["tracking_cm"]
        print(f"{d.name}: {rows[d.name]['seconds']} s, IK feet {t['left_foot']['mean']}/{t['right_foot']['mean']} cm, "
              f"hands {t['left_hand']['mean']}/{t['right_hand']['mean']} cm, slide {r['feet']['slide_cm_per_s']} cm/s, "
              f"spec {rows[d.name]['spec']['self_check']}, variants {rows[d.name]['variants']['kept']}/{len(vs)}")
    out = ROOT / "docs" / "results" / "real_athletes" / "audit.json"
    out.write_text(json.dumps(rows, indent=1, default=float))
    print("->", out)


def run_eval(spec, metrics, robust):
    """eval.json as server/runs.py reads it: the retarget's own spec check, and the policy's clean-clip numbers."""
    sc = spec["self_check"]
    clean = next(c["policies"]["tracking_var"] for c in robust["conditions"] if c["suite"] == "clean")
    held = robust["summary"]["tracking_var"].get("transfer", {})
    return {"retarget": {"task_success": bool(sc.get("success", sc["stages_completed"] == sc["stages_total"])),
                         "task_stages": f"{sc['stages_completed']}/{sc['stages_total']}"},
            "tracking_var": {**{k: v for k, v in metrics.items() if k != "checkpoint"},
                             "task_success": clean["task_success"] >= 0.5, "task_success_rate": clean["task_success"],
                             "task_stages": f"{clean['stages_completed']:.1f}/{sc['stages_total']}",
                             "fall_rate": clean["fall_rate"], "held_out_success": held.get("task_success"),
                             "held_out_fall_rate": held.get("fall_rate")}}


def assemble(a):
    res, known, outputs = Path(a.results), clips(), ROOT / "outputs"
    rows = {}
    for log in sorted(res.glob("*.log")):
        name = log.stem
        c = known.get(name)
        if c is None or not (res / name / "results.tar.gz").exists():
            print(f"{name}: no results or not in {CLIPS.name}")
            continue
        run = res / name / "run"
        if not (run / "task_spec.json").exists():
            unpack(res / name / "results.tar.gz", run)
        short = c["short"]
        robust = json.loads((res / name / "robust" / "robust.json").read_text())
        spec = json.loads((run / "task_spec.json").read_text())
        metrics = json.loads((run / "policy_tracking_var" / "policy_metrics.json").read_text())
        ev = run_eval(spec, metrics, robust)

        var = outputs / f"remote_variants_{TAG}_{short}"  # build_variants_view.py's trained variant set
        shutil.copytree(run, var, dirs_exist_ok=True)
        if (res / name / "pushes").is_dir():
            shutil.copytree(res / name / "pushes", outputs / f"remote_pushes_{TAG}_{short}", dirs_exist_ok=True)
        dst = outputs / f"remote_latent_{TAG}_{short}"
        (dst / "policy_tracking_var").mkdir(parents=True, exist_ok=True)
        shutil.copytree(run / "clips", dst / "clips", dirs_exist_ok=True)
        for f in ("task_spec.json", "importance.json"):
            shutil.copy2(run / f, dst / f)
        for f in run.joinpath("policy_tracking_var").iterdir():
            if f.suffix in (".mp4", ".json", ".npy", ".npz", ".pt") or f.name == "train.log":
                shutil.copy2(f, dst / "policy_tracking_var" / f.name)
        shutil.copy2(res / name / "robust" / "robust.json", dst / "robust.json")
        (dst / "eval.json").write_text(json.dumps(ev, indent=1))
        iters = robust["meta"].get("iters") or c.get("iters")
        (dst / "status.json").write_text(json.dumps({"state": "done", "finished": c.get("trained"),
                                                     "cmd": ["experiments/variant_policy.py", f"--target {name}",
                                                             f"--iters {iters or a.iters}"]}))
        (dst / "web.json").write_text(json.dumps({
            "target": name, "title": c["title"],
            "description": c["description"] + (" Run in place: the clip's forward travel is removed, so the G1 performs "
                                                "the gait on the spot." if c.get("in_place") else ""),
            "credit": {"text": c["credit"], "url": c["page"], "licence": c["licence"]}}, indent=1))
        rows[name] = {"run": dst.name, **{k: ev["tracking_var"][k] for k in ("task_success_rate", "held_out_success", "fall_rate")}}
        print(f"{name}: -> {dst.name}; clean {ev['tracking_var']['task_success_rate']:.0%}, "
              f"held-out {ev['tracking_var']['held_out_success']}, spec {ev['retarget']['task_stages']}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inputs")
    p.add_argument("processed", help="remote_run.py pro --iters 0's folder (process/results.tar.gz)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("reretarget")
    p.add_argument("clip")
    p.add_argument("--in-place", choices=("world", "stride", "soft"))
    p.add_argument("--video", help="the source video, to re-render side_by_side.mp4")
    p = sub.add_parser("audit")
    p.add_argument("processed", help="remote_run.py pro --iters 0's folder")
    p = sub.add_parser("assemble")
    p.add_argument("results", help="remote_run.py athletes' folder")
    p.add_argument("--iters", type=int)
    a = ap.parse_args()
    {"inputs": inputs, "reretarget": reretarget, "audit": audit, "assemble": assemble}[a.cmd](a)
