"""Counterfactual check of goal relevance in simulation: does weighting the style terms by
importance.json stop the robot from copying movements that don't serve the goal, without costing
the task? GPU job (scripts/remote_run.py relevance).

    python experiments/goal_relevance.py --inputs outputs/remote_latent_v2 --target squat_jump \
        --out outputs/goal_relevance_cf --iters 1000            # what the GPU job runs
    python experiments/goal_relevance.py ... --importance-only   # importance.json only, no training

1. the target's retarget, and a "hype" variant with arm-waving injected before the countermovement
   (skill/importance.inject_hype, --hype START END)
2. importance.json for both: physics attribution (--vjepa predictor|encoder adds the V-JEPA 2
   masking scores as a diagnostic, vjepa_scores.npz, not mixed into the weights)
3. A2H-Task policies on the hype variant, in parallel: `plain` (no importance) and `weighted`
   (--importance), optionally `clean` (the clean clip, no importance) as a reference. The hype
   variant's latent target is V-JEPA's embedding of its robot-silhouette render (centred on the
   robot-domain mean), since no human video of it exists; the reward network is the reused bundle's.
4. eval: task_success / stages (skill/task.evaluate), hype_reproduction (0 = arms as in the clean
   clip, 1 = copies the waving) over the hype window, and arm/leg joint error to the reference
   inside the goal windows (does weighting lose the arm swing that matters?) -> counterfactual.json
"""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skill import importance as I  # noqa: E402
from skill import task as skill_task  # noqa: E402

TASK = "A2H-Task-Flat-Unitree-G1"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def importance_for(name, qpos, spec, model, out, vjepa, azimuth, fps=30.0, hype=None):
    """importance.json (+ png) for one motion; with vjepa = (Predictor, mode), also its V-JEPA
    masking scores (vjepa_scores.npz, a diagnostic: not mixed into the weights)."""
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "importance.json"
    if f.exists():
        return f
    V = None
    if vjepa:
        from latent import predictor_relevance as pr
        events = I.goal_events(spec, I.kinematics(model, qpos)["pos"], fps)
        t0 = time.time()
        V, detail = pr.scores(model, qpos, events, fps, azimuth=azimuth, predictor=vjepa[0], log=log, mode=vjepa[1])
        np.savez(d / "vjepa_scores.npz", scores=V, **detail)
        log(f"{name}: V-JEPA predictor scoring took {time.time() - t0:.0f} s")
    res = I.analyse(model, qpos, spec, fps, V)
    res["clip"] = name
    I.save(res, f)
    I.plot(res, spec, d / "importance.png", hype=hype)
    return f


def hype_bundle(src, dst, qpos, model, azimuth, vj):
    """The reused reward bundle pointed at a robot-rendered target: its centred V-JEPA latents,
    centred on the robot-domain mean (r_mean; there is no human video of it)."""
    from latent import distill, silhouette
    from latent.distill import unit
    from latent.reward_net import load_bundle
    from scipy.ndimage import uniform_filter1d
    net, _, b = load_bundle(src)
    sil = silhouette.robot_silhouettes(model, qpos, azimuth=azimuth)
    target = distill.video_target(vj.embed_centred(sil)[0], b["r_mean"], b["basis"])
    pred = distill.predict(net, distill.body_positions(model, qpos))
    n = min(len(pred), len(target))
    cos = (unit(pred[:n]) * unit(target[:n])).sum(-1)
    cos_ref = uniform_filter1d(np.pad(cos, (0, len(target) - n), mode="edge"), 5)
    distill.save_bundle(dst, net, b["basis"], b["r_mean"], b["h_mean"], target, cos_ref, 30.0,
                        dict(b["report"], target="robot-rendered hype variant"), b["azimuth"])


def to_30fps(q50):
    from latent import distill
    return distill.resample(q50, int(round((len(q50) - 1) * 30 / 50)) + 1)


def joint_err(p, ref, frames, cols):
    n = min(len(p), len(ref))
    f = frames[frames < n]
    return float(np.sqrt(((p[f][:, cols] - ref[f][:, cols]) ** 2).mean())) if len(f) else None


def evaluate(name, d, spec, model, hype_q, clean_q, imp, window, fps=30.0):
    from latent import distill
    f = d / "policy_qpos.npy"
    if not f.exists():
        return {"error": "no rollout"}
    q = to_30fps(np.load(f))
    sem = skill_task.evaluate(spec, distill.body_positions(model, q), fps)
    w = np.array(json.loads(Path(imp).read_text())["weight"])
    goal = np.flatnonzero(w > 0.5)  # frames that matter (propulsion, absorb, descent)
    ref = hype_q if name != "clean" else clean_q
    return {"task_success": sem["success"], "task_stages": f"{sem['stages_completed']}/{sem['stages_total']}",
            "task_jump_quality": sem["jump_quality"], "task_fell_at_s": sem["fell_at_s"],
            "hype_reproduction": round(I.hype_reproduction(q, hype_q, clean_q, window, fps), 3),
            "arm_rmse_goal_frames": joint_err(q, ref, goal, slice(22, 36)),
            "leg_rmse_goal_frames": joint_err(q, ref, goal, slice(7, 19)),
            "arm_rmse_hype_window_vs_clean": joint_err(q, clean_q, np.arange(int(window[0] * fps), int(window[1] * fps)),
                                                       slice(22, 36))}


def main(a):
    import render
    import retarget
    from latent import predictor_relevance as pr
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inp = Path(a.inputs)
    clip = inp / "clips" / a.target
    fps = float(json.loads((clip / "meta.json").read_text())["fps"])
    model = render.load_g1()[0]
    clean_q = np.load(clip / "qpos.npy")
    hype_q = I.inject_hype(clean_q, *a.hype, fps=fps)
    np.save(out / "hype_qpos.npy", hype_q)
    azimuth = pr.azimuth_of(clip)
    spec = skill_task.compile_spec(I.kinematics(model, clean_q)["pos"], fps, name=a.target)
    spec_f = out / "task_spec.json"
    spec_f.write_text(json.dumps(spec, indent=1))
    log(f"{a.target}: {spec['summary']}; hype {a.hype[0]}-{a.hype[1]} s; viewpoint {azimuth} deg")

    predictor = None if a.vjepa == "off" else (pr.Predictor(), a.vjepa)
    imp = {"clean": importance_for(a.target, clean_q, spec, model, out, predictor, azimuth, fps),
           "hype": importance_for(f"{a.target}_hype", hype_q, spec, model, out, predictor, azimuth, fps, a.hype)}
    for k, f in imp.items():
        log(f"importance ({k}): " + json.dumps(I.summarize(json.loads(f.read_text()), spec,
                                                           [("hype", *a.hype)])))
    if a.importance_only:
        return pack(out)
    del predictor
    import torch
    torch.cuda.empty_cache()

    # the hype variant's motion and reward bundle
    hype_csv = retarget.export_mjlab_csv(hype_q, out / "hype_motion.csv")
    bundle_src = Path(a.reward or inp / "reward.pt")
    bundle_h = out / "reward_hype.pt"
    if not bundle_h.exists():
        from latent.vjepa import VJepa
        hype_bundle(bundle_src, bundle_h, hype_q, model, azimuth, VJepa())
    runs = {"plain": (hype_csv, bundle_h, None), "weighted": (hype_csv, bundle_h, imp["hype"])}
    if a.with_clean:
        runs["clean"] = (clip / "motion.csv", bundle_src, None)
    procs = {}
    for name, (csv, bundle, importance) in runs.items():
        d = out / f"policy_{name}"
        if (d / "policy_qpos.npy").exists():
            continue
        d.mkdir(exist_ok=True)
        cmd = [sys.executable, ROOT / "train_motion.py", "--csv", csv, "--fps", 30, "--out", d, "--iters", a.iters,
               "--num-envs", a.num_envs, "--task", TASK, "--bundle", bundle, "--task-spec", spec_f]
        if importance:
            cmd += ["--importance", importance]
        procs[name] = subprocess.Popen(list(map(str, cmd)), stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT)
        log(f"training {name}, log {d / 'train.log'}")
    for name, p in procs.items():
        log(f"{name} finished with code {p.wait()}")

    res = {"hype_s": a.hype, "iters": a.iters, "importance": {k: str(v.relative_to(out)) for k, v in imp.items()}}
    for name in runs:
        res[name] = evaluate(name, out / f"policy_{name}", spec, model, hype_q, clean_q, imp["hype"], a.hype, fps)
        log(f"{name}: {res[name]}")
    (out / "counterfactual.json").write_text(json.dumps(res, indent=1))
    pack(out)


def pack(out):
    with tarfile.open(out / "results.tar.gz", "w:gz") as tar:
        for f in out.rglob("*"):
            if f.is_file() and f.suffix in {".json", ".png", ".npz", ".npy", ".mp4", ".log"} and "logs" not in f.parts \
                    and f.name != "results.tar.gz":
                tar.add(f, str(f.relative_to(out)))
    log(f"done: {out / 'results.tar.gz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="an earlier latent run's folder: clips/<target>/, reward.pt")
    ap.add_argument("--target", default="squat_jump")
    ap.add_argument("--out", default="outputs/goal_relevance_cf")
    ap.add_argument("--hype", nargs=2, type=float, default=[0.2, 1.4], metavar=("START_S", "END_S"))
    ap.add_argument("--reward", help="reward bundle (default <inputs>/reward.pt)")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--with-clean", action="store_true", help="also train on the clean clip (a third policy)")
    ap.add_argument("--importance-only", action="store_true", help="importance.json only, no training")
    ap.add_argument("--vjepa", default="off", choices=["off", "predictor", "encoder"],
                    help="also save V-JEPA 2 masking scores (diagnostic, a few GPU minutes)")
    main(ap.parse_args())
