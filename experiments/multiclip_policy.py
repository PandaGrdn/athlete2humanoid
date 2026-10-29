"""One policy, several clips: does seeing the video's V-JEPA latent help it on clips it never saw?

    python experiments/multiclip_policy.py inputs/clips --out outputs/multiclip \
        --init /content/a2h/inputs/policy_tracking_var/model_1499.pt --iters 2000

1. Every training clip's TRAIN variants (skill/variations.py) are laid end to end in one motion file,
   as experiments/variant_policy.py does for one clip; HELD_OUT clips are never trained on.
2. Each clip's silhouette video, and its mirror image, are embedded with V-JEPA 2 (dense: one vector
   per 2 frames) and reduced by a PCA fitted on the training clips only (latent/video_condition.py).
   Each variant gets the latent of what its video shows at each step (re-timed variants read the video
   at their source frames, mirrored ones the mirrored video's latent).
3. Three policies train on the same variants from the same warm start, differing only in one input:
     multi         mjlab's tracking inputs
     multi_vj      + the video's V-JEPA latent now and 0.5 / 1 s ahead
     multi_future  + the retarget's joints and pelvis height 0.5 / 1 s ahead (the control: the same
                   look-ahead, no V-JEPA; if multi_vj only beats multi, it may be the look-ahead)
4. experiments/robustness_eval.py runs each clip, trained on or held out, with all three: the clip
   itself, its HELD_OUT variants (each with its own latent), tempo changes and pushes. Summarised in
   multiclip_summary.md.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.latent_policy import TASKS, log, mirror_checkpoints  # noqa: E402

TRAIN_CLIPS = ("squat_jump", "front_kick", "squat", "arms", "lunge")
HELD_OUT_CLIPS = ("jumping_jacks", "punches")
POLICIES = ("multi", "multi_vj", "multi_future")
ROLLOUT_CLIP = "squat_jump"  # train_motion.py's rollout video and metrics


def clip_qpos(d):
    return np.load(d / "qpos.npy"), float(json.loads((d / "meta.json").read_text())["fps"])


def latents(clips_dir, names, smoke):
    """name -> (dense latent, mirrored video's dense latent), from V-JEPA 2 on the silhouette videos.
    smoke: the stored latent for both (no GPU, no model download)."""
    from latent import video_condition as VC
    if smoke:
        return {c: (np.load(clips_dir / c / "video_latent.npz")["dense"],) * 2 for c in names}
    from latent.vjepa import VJepa
    vj = VJepa()
    out = {}
    for c in names:
        out[c] = VC.dense_latents(clips_dir / c / "silhouette.mp4", vj)
        log(f"{c}: V-JEPA latents {out[c][0].shape} (+ mirrored)")
    import torch
    del vj
    torch.cuda.empty_cache()  # the GPU is the trainers' from here on
    return out


def build(clips_dir, out, py, env, train, held, seed=0, smoke=False):
    """-> out/motion_variants.npz (+ .vjl.npy), variants/variants.json, vjl_pca.npz, and per clip
    evals/<clip>/{motion.npz, motion.vjl.npy, task_spec.json, transfer/<variant>.npz (+ .vjl.npy)}."""
    from latent import video_condition as VC
    from skill import variations as V
    from skill import task as T
    from skill.keypose import compile_keypose_spec
    from train_motion import csv_to_npz
    g1 = V.G1()
    raw = latents(clips_dir, [*train, *held], smoke)
    pca = VC.fit_pca([x for c in train for x in raw[c]])
    np.savez(out / "vjl_pca.npz", **pca)
    z = {c: tuple(VC.project(x, pca) for x in raw[c]) for c in raw}
    log(f"PCA on the {len(train)} training clips' latents: {VC.K} components keep {pca['explained']:.0%} of the variance")

    grid = V.TRAIN[:2] if smoke else V.TRAIN
    held_grid = V.HELD_OUT[:2] if smoke else V.HELD_OUT
    segs, parts, lats, start = [], [], [], 0
    for c in train:
        qpos, fps = clip_qpos(clips_dir / c)
        log(f"{c}: training variants")
        vs = V.generate(qpos, fps, grid, seed, g1, log)
        for v in vs:
            v["name"] = f"{c}.{v['name']}"
        V.write(vs, out / "variants")
        for v in (v for v in vs if v["ok"]):
            d = out / "variants" / v["name"]
            m = dict(np.load(csv_to_npz(py, d / "motion.csv", fps, d / "motion.npz", env)))
            n50 = len(m["joint_pos"])
            lats.append(VC.per_step(z[c][int(v["params"]["mirror"])], v["src_frame"], fps, n50))
            segs.append({"name": v["name"], "clip": c, "start": start, "length": n50,
                         "spec": f"{v['name']}/task_spec.json", "params": v["params"]})
            parts.append(m)
            start += n50
    np.savez(out / "motion_variants.npz", fps=parts[0]["fps"],
             **{k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "fps"})
    np.save(out / "motion_variants.vjl.npy", np.concatenate(lats))
    (out / "variants" / "variants.json").write_text(json.dumps({"clips": list(train), "segments": segs}, indent=1))
    log(f"{len(segs)} training variants of {len(train)} clips ({start / 50:.0f} s of motion)")

    for c in [*train, *held]:  # what every policy is tested on, clip by clip
        qpos, fps = clip_qpos(clips_dir / c)
        e = out / "evals" / c
        (e / "transfer").mkdir(parents=True, exist_ok=True)
        n50 = len(np.load(csv_to_npz(py, clips_dir / c / "motion.csv", fps, e / "motion.npz", env))["joint_pos"])
        np.save(e / "motion.vjl.npy", VC.per_step(z[c][0], np.arange(len(qpos), dtype=float), fps, n50))
        spec = compile_keypose_spec(T.body_positions_from_qpos(qpos), fps, name=c)
        (e / "task_spec.json").write_text(json.dumps(spec, indent=1))
        vs = V.generate(qpos, fps, held_grid, seed + 1000, g1, log)
        for v in (v for v in vs if v["ok"]):
            d = e / "held_out" / v["name"]
            V.write([v], e / "held_out")
            f = csv_to_npz(py, d / "motion.csv", fps, e / "transfer" / f"{v['name']}.npz", env)
            np.save(f.with_suffix(".vjl.npy"),
                    VC.per_step(z[c][int(v["params"]["mirror"])], v["src_frame"], fps, len(np.load(f)["joint_pos"])))
        log(f"{c}: eval set, {len(list((e / 'transfer').glob('*.npz')))} held-out variants")


def train_all(a, out, clips_dir):
    procs = {}
    roll = out / "evals" / ROLLOUT_CLIP
    for name in a.policies:
        d = out / f"policy_{name}"
        d.mkdir(exist_ok=True)
        cmd = [sys.executable, ROOT / "train_motion.py", "--csv", clips_dir / ROLLOUT_CLIP / "motion.csv", "--fps", 30,
               "--out", d, "--iters", a.iters, "--num-envs", a.num_envs, "--task", TASKS[name],
               "--variants", out / "variants" / "variants.json", "--train-motion", out / "motion_variants.npz"]
        if name == "multi_vj":
            cmd += ["--video-latent", out / "motion_variants.vjl.npy", "--rollout-video-latent", roll / "motion.vjl.npy"]
        if a.init:
            cmd += ["--init", a.init]
        procs[name] = subprocess.Popen(list(map(str, cmd)), stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT)
        log(f"training {name} ({TASKS[name]}); log {d / 'train.log'}")
    while procs:
        for name, p in list(procs.items()):
            if p.poll() is not None:
                log(f"{name} finished with code {p.returncode}")
                del procs[name]
        mirror_checkpoints(out)
        if procs:
            time.sleep(20 if a.smoke else 60)


def newest(d):
    c = sorted(Path(d).glob("logs/*/*/model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return c[-1] if c else None


def evaluate(a, out, clips):
    """robustness_eval per clip, all policies at once."""
    for c in clips:
        e = out / "evals" / c
        run = e / "run"
        if (e / "robust" / "robust.json").exists():
            continue
        for name in a.policies:
            ck = newest(out / f"policy_{name}")
            if ck is None:
                log(f"{name}: no checkpoint, left out of the eval")
                continue
            d = run / f"policy_{name}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(ck, d / ck.name)
            shutil.copy(e / "motion.npz", d / "motion.npz")
        shutil.copy(e / "task_spec.json", run / "task_spec.json")
        transfer = [f"{f.stem}={f}" for f in sorted((e / "transfer").glob("*.npz"))]
        cmd = [sys.executable, ROOT / "experiments" / "robustness_eval.py", "--runs", run, "--out", e / "robust",
               "--target", c, "--reps", 4 if a.smoke else a.reps, "--no-vjepa", "--push", *([] if a.smoke else [1.0]),
               "--init", "--dynamics", 0, "--speed", "--phase-push", "--starts", "--no-noref",
               "--tempo", *([] if a.smoke else [0.5, 2.0]), "--transfer", *transfer]
        if a.smoke:
            cmd += ["--max-steps", 20]
        log(f"evaluating on {c} ({len(transfer)} held-out variants)")
        r = subprocess.run(list(map(str, cmd)), env=dict(os.environ, A2H_VIDEO_LATENT=str(e / "motion.vjl.npy")))
        if r.returncode:
            log(f"{c}: robustness_eval failed ({r.returncode})")


def summarise(out, train, held, policies):
    """Success / fall rate per policy: on the clips, on their held-out variants, trained vs held-out clips."""
    rows = {}
    for c in [*train, *held]:
        f = out / "evals" / c / "robust" / "robust.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())["summary"]
        rows[c] = {p: {suite: {k: s[p][suite].get(k) for k in ("task_success", "fall_rate", "dtw_joint_rad")}
                       for suite in s[p]} for p in policies if p in s}
    groups = {"trained clips": [c for c in train if c in rows], "held-out clips": [c for c in held if c in rows]}

    def mean(cs, p, suite, k):
        v = [rows[c][p][suite][k] for c in cs if p in rows[c] and suite in rows[c][p] and rows[c][p][suite][k] is not None]
        return float(np.mean(v)) if v else None
    table = {g: {p: {suite: {k: mean(cs, p, suite, k) for k in ("task_success", "fall_rate")}
                     for suite in ("clean", "transfer", "tempo", "push")} for p in policies} for g, cs in groups.items()}
    res = {"clips": rows, "groups": groups, "table": table}
    (out / "multiclip.json").write_text(json.dumps(res, indent=1))
    pct = lambda x: "-" if x is None else f"{x:.0%}"  # noqa: E731
    lines = ["# One policy, several clips: with and without the video's V-JEPA latent", "",
             f"Trained on {', '.join(train)}; never trained on {', '.join(held)}. Task success (fall rate).", ""]
    for g, cs in groups.items():
        lines += [f"## {g.capitalize()} ({', '.join(cs)})", "", "| Policy | The clip | Its held-out variants | Tempo 0.5x / 2x | Pushed 1 m/s |",
                  "|---|---|---|---|---|"]
        for p in policies:
            t = table[g][p]
            lines.append(f"| {p} | " + " | ".join(f"{pct(t[s]['task_success'])} ({pct(t[s]['fall_rate'])})"
                                                for s in ("clean", "transfer", "tempo", "push")) + " |")
        lines.append("")
    lines += ["## Per clip", "", "| Clip | " + " | ".join(policies) + " |", "|---|" + "---|" * len(policies)]
    for c in rows:
        lines.append(f"| {c}{' (held out)' if c in held else ''} | " + " | ".join(
            f"{pct(rows[c][p]['clean']['task_success'])} / {pct(rows[c][p].get('transfer', {}).get('task_success'))}"
            if p in rows[c] else "-" for p in policies) + " |")
    lines += ["", "Each cell: the clip / its held-out variants."]
    (out / "multiclip_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def pack(out):
    with tarfile.open(out / "results.tar.gz", "w:gz") as tar:
        for f in ["multiclip.json", "multiclip_summary.md", "vjl_pca.npz"]:
            if (out / f).exists():
                tar.add(out / f, f)
        tar.add(out / "variants" / "variants.json", "variants/variants.json")
        for e in (out / "evals").glob("*/robust"):
            for f in ["robust.json", "robust_summary.md"]:
                if (e / f).exists():
                    tar.add(e / f, f"evals/{e.parent.name}/robust/{f}")
        for d in out.glob("policy_*"):
            for f in ["policy.mp4", "policy_metrics.json", "policy_qpos.npy", "train.log", "motion.npz", "weights.json"]:
                if (d / f).exists():
                    tar.add(d / f, f"{d.name}/{f}")
            ck = newest(d)
            if ck:
                tar.add(ck, f"{d.name}/{ck.name}")


def main(a):
    clips_dir, out = Path(a.clips).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train, held = (a.train[:2], a.held_out[:1]) if a.smoke else (a.train, a.held_out)
    py = Path(os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python"))
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), WANDB_MODE="disabled")
    if not (out / "motion_variants.npz").exists():
        build(clips_dir, out, py, env, train, held, a.seed, a.smoke)
    if a.build_only:
        return
    train_all(a, out, clips_dir)
    evaluate(a, out, [*train, *held])
    summarise(out, train, held, a.policies)
    pack(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", help="folder of processed clips (qpos.npy, meta.json, motion.csv, silhouette.mp4, video_latent.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", nargs="+", default=list(TRAIN_CLIPS))
    ap.add_argument("--held-out", nargs="+", default=list(HELD_OUT_CLIPS))
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    ap.add_argument("--init", help="checkpoint all three start from (a tracking policy)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=64, help="rollouts per policy per eval condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 clips, 2 variants, a few iterations; no V-JEPA model")
    main(ap.parse_args())
