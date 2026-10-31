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
import re
import shutil
import sys
import tarfile
import time
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


def showcase(a):
    """Each named clip's assembled run -> a 3D showcase bundle (scripts/build_showcase.py, its source video credited)
    and its tab in web/public/showcase/index.json, so /showcase and the build view's stage 6 play it. A run whose policy
    doesn't complete its clip (eval.json tracking_var.task_success) is refused unless --force (to inspect)."""
    import subprocess
    known, outputs = clips(), ROOT / "outputs"
    index_f = ROOT / "web" / "public" / "showcase" / "index.json"
    index = json.loads(index_f.read_text())
    for name in a.clips:
        c = known[name]
        run = outputs / f"remote_latent_{TAG}_{c['short']}"
        ev = json.loads((run / "eval.json").read_text())["tracking_var"]
        if ev.get("task_success") is not True and not a.force:
            raise SystemExit(f"{name}: its policy doesn't complete the clip (clean success {ev.get('task_success_rate')}); "
                             "not a showcase (--force to build it anyway)")
        iters = re.search(r"--iters (\d+)", " ".join(json.loads((run / "status.json").read_text()).get("cmd") or []))
        sid = name.replace("_", "-")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_showcase.py"), "--id", sid, "--run", run.name,
                        "--policy", "tracking_var", "--pushes", f"remote_pushes_{TAG}_{c['short']}",
                        "--video", str(run / "clips" / name / "side_by_side.mp4"), "--video-label", f"{c['licence']} video",
                        "--task-spec", str(run / "task_spec.json")], check=True)
        facts = [{"v": f"{ev['task_success_rate']:.0%}", "l": "of clean runs complete the clip's key moments"}]
        if ev.get("held_out_success") is not None:
            facts.append({"v": f"{ev['held_out_success']:.0%}", "l": "of held-out variants completed"})
        entry = {"id": sid, "tab": c["tab"], "title": c["showcase_title"], "kicker": "Free-licence video → one policy",
                 "lede": c["description"] + (" The clip travels; the G1 runs it in place." if c.get("in_place") else ""),
                 "video": f"{c['author']}'s clip ({c['licence']}). SAM 3.1 segments the person, SAM 3D Body lifts the pose "
                          "to 3D" + (", and the run's forward travel is taken out." if c.get("in_place") else "."),
                 "policy": "PPO in mjlab on 36 automatic variants of the clip, warm-started from the squat-jump tracker, "
                           (f"{iters.group(1)} iterations. " if iters else "") + "The network's own run in MuJoCo, recorded.",
                 "facts": facts, "credit": {"text": c["credit"], "url": c["page"], "licence": c["licence"]},
                 "fallback": None}
        index["clips"] = [e for e in index["clips"] if e["id"] != sid] + [{k: v for k, v in entry.items() if v is not None}]
        print(f"{sid}: bundle + tab")
    index_f.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")


def facts(a):
    """The numbers docs/REAL_ATHLETES.md quotes that no other results file holds, each from its source in outputs/
    (gitignored), -> docs/results/real_athletes/facts.json: SAM 3.1 / SAM 3D Body coverage and the intake's warnings
    (the processing job's log), the kick's reference tilt and training terminations, how far each clip trained on
    Modal before the spend limit, the jogs' ankle heights (reference vs the final policy's clean rollouts) and the
    spec stage they stop at, the boxer's tracking error, the warm start's own scores, the synthetic jog's slip, and
    the jobs' wall-clock times."""
    import numpy as np
    from skill import task as T
    out_d = ROOT / "outputs"
    res = {"sources": {}}

    log = (out_d / "remote_pro_ra" / "job.log").read_text(errors="ignore")
    res["sources"]["processing"] = "outputs/remote_pro_ra/job.log"
    proc, start = {}, 0
    for m in re.finditer(r"\] (\w+): processed", log):
        chunk = log[start:m.start()]
        num = lambda pat: (lambda x: float(x[-1]) if x else None)(re.findall(pat, chunk))
        proc[m.group(1)] = {"sam31_visible_pct": num(r"SAM 3.1: athlete visible in (\d+)% of frames"),
                            "sam3d_pose_found_pct": num(r"pose \(sam3d\) found in (\d+)% of frames"),
                            "intake_warnings": re.findall(r"intake warning: (.*)", chunk)}
        start = m.end()
    res["processing"] = proc

    kick = out_d / "remote_athletes_ra3_partial" / "taekwondo_kick"
    klog = (kick / "robust_cpu.log").read_text(errors="ignore")
    tl = (kick / "policy_tracking_var" / "train.log").read_text(errors="ignore")
    last = tl[tl.rfind("Learning iteration"):]
    res["kick"] = {"ref_tilt_max_rad": float(re.findall(r"'ref_tilt_max_rad': ([\d.]+)", klog)[-1]),
                   "fall_rule_tilt_rad": float(re.findall(r"'tilt_rad': ([\d.]+)", klog)[-1]),
                   "clip_steps_50hz": int(re.findall(r"'clip_steps': (\d+)", klog)[-1]),
                   "eval_refused": "AssertionError: fall rule fires on the reference" in klog,
                   "last_terminations_per_window": {k: float(v) for k, v in re.findall(r"Episode_Termination/(\w+): ([\d.]+)", last)}}
    res["sources"]["kick"] = f"{kick.relative_to(ROOT)}/robust_cpu.log, policy_tracking_var/train.log"

    modal = {}
    for c in ("jog_lakeside", "jog_bridge", "shadow_boxing", "taekwondo_kick"):
        t = (out_d / "remote_athletes_ra3_partial" / c / "policy_tracking_var" / "train.log").read_text(errors="ignore")
        jl = (out_d / "remote_athletes_ra3_partial" / f"{c}.log").read_text(errors="ignore")
        modal[c] = {"training_started_utc": (re.findall(r"\[(\d\d:\d\d:\d\d)\] training tracking_var", jl) or [None])[0],
                    "training_variants": (re.findall(r"(\d+/\d+) training variants", jl) or [None])[0],
                    "last_iteration_logged": int(re.findall(r"Learning iteration (\d+)/", t)[-1]),
                    "time_elapsed_at_it": re.findall(r"Time elapsed: ([\d:]+)", t)[-1],
                    "iteration_time_s": float(re.findall(r"Iteration time: ([\d.]+)s", t)[-1]),
                    "mean_episode_length": float(re.findall(r"Mean episode length: ([\d.]+)", t)[-1])}
    res["modal_training_before_the_limit"] = modal

    feet = {}
    for c in ("jog_lakeside", "jog_bridge"):
        ref = T.body_positions_from_qpos(np.load(out_d / "remote_pro_ra" / "process" / "clips" / c / "qpos.npy"))
        z = np.load(out_d / "remote_athletes_ra5" / c / "robust" / "rollouts.npz")
        pol = np.concatenate([z[k][..., [3, 6], 2] for k in z.files if k.startswith("0/tracking_var/") and k.endswith("/body")])
        spec = json.loads((out_d / "remote_athletes_ra5" / c / "run" / "task_spec.json").read_text())
        names = spec["features"]
        st = [x for x in spec["stages"] if x["kind"] == "pose"][1]
        feet[c] = {"reference_ankle_z_m": {"max": np.round(ref[:, [3, 6], 2].max(0), 3).tolist(),
                                            "p90": np.round(np.percentile(ref[:, [3, 6], 2], 90, axis=0), 3).tolist()},
                   "policy_clean_ankle_z_m": {"rollouts": int(len(pol) // max(1, len(z[next(k for k in z.files if k.endswith('/body'))]))),
                                               "max": np.round(pol.max(0), 3).tolist(),
                                               "p90": np.round(np.percentile(pol, 90, axis=0), 3).tolist(),
                                               "median": np.round(np.median(pol, 0), 3).tolist()},
                   "left_right": "left, right ankle roll link",
                   "second_key_moment": {"name": st["name"], "feet_down": st["feet_down"], "hold_s": st.get("hold_s"),
                                         "foot_height_tol_m": [round(st["tol"][names.index(f"{s} foot height")], 4) for s in ("left", "right")]}}
    res["jog_feet"] = feet
    res["sources"]["jog_feet"] = "reference: outputs/remote_pro_ra/process/clips/<clip>/qpos.npy; policy: outputs/remote_athletes_ra5/<clip>/robust/rollouts.npz (clean, the rollouts it saves)"

    r = json.loads((out_d / "remote_athletes_ra5" / "shadow_boxing" / "robust" / "robust.json").read_text())
    res["boxing_clean_tracking_rms_m"] = r["conditions"][0]["policies"]["tracking_var"]["tracking_rms_m"]

    wr = json.loads((out_d / "remote_robust_s3_sj" / "robust.json").read_text())
    w, wc = wr["summary"]["tracking_var"], wr["conditions"][0]["policies"]["tracking_var"]
    res["warm_start_squat_jump"] = {"clean_task_success": w["clean"]["task_success"], "held_out_task_success": w["transfer"]["task_success"],
                                    "clean_joint_err_rad": round(wc["joint_err_rad"], 3), "clean_torso_err_m": round(wc["torso_err_m"], 3)}
    res["sources"]["warm_start_squat_jump"] = "outputs/remote_robust_s3_sj/robust.json summary.tracking_var"

    import retarget
    from experiments.in_place.measure import slip
    sys.path.insert(0, str(ROOT / "tests"))
    from test_in_place import FPS, synthetic_jog
    P = synthetic_jog(T=72)
    stance = retarget.proportional_motion(P, retarget.G1Retargeter(), FPS)["stance"]
    syn = {}
    for m in retarget.IN_PLACE_METHODS:
        rt = retarget.G1Retargeter()
        q = rt.retarget_sequence(*retarget.human_to_targets(P, rt, fps=FPS, in_place=m), fps=FPS)
        syn[m] = slip(rt.model, q, FPS, stance)
    res["synthetic_jog_slip"] = syn

    times = {}
    for run in ("remote_pro_ra", "remote_athletes_ra3", "remote_athletes_ra5"):
        f = out_d / run / "runner.log"
        times[run] = [ln for ln in f.read_text().splitlines() if re.search(r"running on Modal|job launched|job state|L4 ready|A100 ready|VM stopped", ln)] if f.exists() else None
    times["note"] = ("remote_athletes_ra3: the Modal container was stopped by the spend limit; its runner was stopped by hand "
                     "after the call had failed, so its log has no end time (see modal_training_before_the_limit)")
    res["job_times"] = times
    rnd = lambda x: (round(x, 4) if isinstance(x, float) else [rnd(v) for v in x] if isinstance(x, list)
                     else {k: rnd(v) for k, v in x.items()} if isinstance(x, dict) else x)
    out = ROOT / "docs" / "results" / "real_athletes" / "facts.json"
    out.write_text(json.dumps(rnd(res), indent=1))
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
        if not (ev["tracking_var"]["task_success"] or a.all):
            print(f"{name}: clean success {ev['tracking_var']['task_success_rate']:.0%}: not a demo run (--all to assemble anyway)")
            continue

        var = outputs / f"remote_variants_{TAG}_{short}"  # build_variants_view.py's trained variant set
        shutil.copytree(run, var, dirs_exist_ok=True)
        if (res / name / "pushes").is_dir():
            shutil.copytree(res / name / "pushes", outputs / f"remote_pushes_{TAG}_{short}", dirs_exist_ok=True)
        dst = outputs / f"remote_latent_{TAG}_{short}"
        (dst / "policy_tracking_var").mkdir(parents=True, exist_ok=True)
        shutil.copytree(run / "clips", dst / "clips", dirs_exist_ok=True)
        shutil.copy2(run / "importance.json", dst / "importance.json")
        # the run's spec is its original variant's (named o_t1_a1); the site and build_variants_view look for the clip
        (dst / "task_spec.json").write_text(json.dumps({**spec, "name": name}, indent=1))
        for f in run.joinpath("policy_tracking_var").iterdir():
            if f.suffix in (".mp4", ".json", ".npy", ".npz", ".pt") or f.name == "train.log":
                shutil.copy2(f, dst / "policy_tracking_var" / f.name)
        shutil.copy2(res / name / "robust" / "robust.json", dst / "robust.json")
        (dst / "eval.json").write_text(json.dumps(ev, indent=1))
        done = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime((res / name / "results.tar.gz").stat().st_mtime))
        (dst / "status.json").write_text(json.dumps({"state": "done", "finished": done,
                                                     "cmd": ["experiments/variant_policy.py", f"--target {name}",
                                                             f"--iters {a.iters}"]}))
        (dst / "web.json").write_text(json.dumps({
            "target": name, "title": c["title"],
            "description": c["description"] + (" Run in place: the clip's forward travel is removed, so the G1 does "
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
    p = sub.add_parser("showcase")
    p.add_argument("clips", nargs="+")
    p.add_argument("--force", action="store_true", help="also a clip whose policy doesn't pass (to inspect; don't commit it)")
    sub.add_parser("facts")
    p = sub.add_parser("assemble")
    p.add_argument("results", help="remote_run.py athletes' folder")
    p.add_argument("--iters", type=int, required=True, help="the policies' training iterations (for the run page)")
    p.add_argument("--all", action="store_true", help="also the clips whose policy doesn't complete its clip (to inspect)")
    a = ap.parse_args()
    {"inputs": inputs, "reretarget": reretarget, "audit": audit, "assemble": assemble, "showcase": showcase, "facts": facts}[a.cmd](a)
