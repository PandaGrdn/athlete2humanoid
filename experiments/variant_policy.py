"""Train on variants of a clip, hold others out: does the policy learn the action or the recording?

    python experiments/variant_policy.py inputs/clips/squat_jump --out outputs/variants_sj \
        --policies taskhybrid_var tracking_var --init /content/a2h/inputs --iters 1500

1. skill/variations.py makes TRAIN variants of the clip (mirrored, re-timed, scaled, warped; nothing
   about what the action is) and HELD_OUT ones outside that range, each with its own keypose spec;
   variants whose own reference can't complete their spec, or that push joints past their limits,
   are dropped.
2. mjlab's converter turns each into a 50 Hz motion file. The TRAIN ones are laid end to end in
   motion_variants.npz, with variants/variants.json saying where each starts (mjlab_tasks.py
   A2H_VARIANTS), and their goal-importance weights likewise (importance_variants.json).
3. Each policy trains on all of them at once (train_motion.py --variants), then is rolled out on the
   clip itself (policy_<name>/motion.npz, as any run's), so experiments/robustness_eval.py compares
   it with the single-clip policies; held_out/<name>.npz go to its --transfer.

No V-JEPA style term: the video's latent is tied to the original's timing, which the variants change.
The style half is tracking (TaskHybrid), which follows whichever variant's reference it is shown.
"""

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.latent_policy import TASK_FIRST, TASKS, init_checkpoint, log, mirror_checkpoints  # noqa: E402

ORIGINAL = "o_t1_a1"  # skill.variations.name(False, 1.0, 1.0, 0.0): the clip itself


def npz_frames(f):
    return len(np.load(f)["joint_pos"])


def importance_50hz(spec, n30, n50):
    """keypose.importance's per-frame weights (30 fps) resampled to a 50 Hz motion file's frames."""
    from skill import keypose
    w = np.asarray(keypose.importance(spec, n30, 30.0)["weight"])
    return w[np.minimum(np.round(np.arange(n50) * 30.0 / 50.0).astype(int), n30 - 1)]


def build(clip, out, py, env, seed=0):
    """Variants -> out/variants/<name>/, out/held_out/<name>.npz, out/motion_variants.npz, variants.json,
    importance_variants.json, and the clip itself (task_spec.json, importance.json, motion.csv)."""
    from skill import variations as V
    from train_motion import csv_to_npz
    meta = json.loads((clip / "meta.json").read_text())
    qpos, fps = np.load(clip / "qpos.npy"), float(meta["fps"])
    g1 = V.G1()
    log(f"variants of {clip.name}: training grid")
    train = V.write(V.generate(qpos, fps, V.TRAIN, seed, g1, log), out / "variants")
    log("held-out grid")
    held = V.write(V.generate(qpos, fps, V.HELD_OUT, seed + 1000, g1, log), out / "held_out")
    kept = [r for r in train if r["ok"]]
    if not any(r["name"] == ORIGINAL for r in kept):
        raise SystemExit(f"the clip itself fails its own spec: {next(r['why'] for r in train if r['name'] == ORIGINAL)}")
    segs, parts, weights, start = [], [], [], 0
    for r in kept:
        d = out / "variants" / r["name"]
        f = csv_to_npz(py, d / "motion.csv", fps, d / "motion.npz", env)
        n50 = npz_frames(f)
        spec = json.loads((d / "task_spec.json").read_text())
        segs.append({"name": r["name"], "start": start, "length": n50, "spec": f"{r['name']}/task_spec.json",
                     "params": r["params"]})
        parts.append(np.load(f))
        weights.append(importance_50hz(spec, r["frames"], n50))
        start += n50
    np.savez(out / "motion_variants.npz", fps=parts[0]["fps"],
             **{k: np.concatenate([p[k] for p in parts]) for k in parts[0].files if k != "fps"})
    (out / "variants" / "variants.json").write_text(json.dumps({"clip": clip.name, "segments": segs}, indent=1))
    (out / "importance_variants.json").write_text(json.dumps({"fps": 50.0, "weight": np.round(np.concatenate(weights), 3).tolist(),
                                                              "source": "keypose, per variant"}))
    for r in held:
        if r["ok"]:
            d = out / "held_out" / r["name"]
            csv_to_npz(py, d / "motion.csv", fps, out / "held_out" / f"{r['name']}.npz", env)
    orig = out / "variants" / ORIGINAL
    spec = json.loads((orig / "task_spec.json").read_text())
    (out / "task_spec.json").write_text(json.dumps(spec, indent=1))
    from skill import keypose
    (out / "importance.json").write_text(json.dumps(keypose.importance(spec, len(qpos), fps)))
    log(f"{len(kept)}/{len(train)} training variants ({start / 50:.0f} s of motion), "
        f"{sum(r['ok'] for r in held)}/{len(held)} held out")
    return orig / "motion.csv", fps


def main(a):
    import os
    clip, out = Path(a.clip), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    py = Path(os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python"))
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), WANDB_MODE="disabled")
    csv, fps = build(clip, out, py, env, a.seed)
    if a.build_only:
        return
    procs = {}
    for name in a.policies:
        d = out / f"policy_{name}"
        d.mkdir(exist_ok=True)
        cmd = [sys.executable, ROOT / "train_motion.py", "--csv", csv, "--fps", fps, "--out", d, "--iters", a.iters,
               "--num-envs", a.num_envs, "--task", TASKS[name], "--variants", out / "variants" / "variants.json"]
        init = init_checkpoint(name, a.init)
        if init:
            cmd += ["--init", init]
        if name in TASK_FIRST:
            cmd += ["--task-spec", out / "task_spec.json", "--importance", out / "importance_variants.json",
                    "--rollout-importance", out / "importance.json"]
        procs[name] = subprocess.Popen(list(map(str, cmd)), stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT)
        log(f"training {name} ({TASKS[name]}) on the variants" + (f", from {init}" if init else "") + f"; log {d / 'train.log'}")
    while procs:
        for name, p in list(procs.items()):
            if p.poll() is not None:
                log(f"{name} finished with code {p.returncode}")
                del procs[name]
        mirror_checkpoints(out)
        if procs:
            time.sleep(60)
    pack(out, clip)


def pack(out, clip):
    """What robustness_eval needs to compare these with other runs, and the variants themselves."""
    with tarfile.open(out / "results.tar.gz", "w:gz") as tar:
        for f in ["task_spec.json", "importance.json", "importance_variants.json"]:
            tar.add(out / f, f)
        tar.add(out / "variants", "variants", filter=lambda t: None if t.name.endswith("motion.npz") else t)
        tar.add(out / "held_out", "held_out")
        for f in clip.iterdir():  # the clip, so this folder can be --reuse'd like a latent run's
            tar.add(f, f"clips/{clip.name}/{f.name}")
        for d in out.glob("policy_*"):
            for f in ["policy.mp4", "policy_metrics.json", "policy_qpos.npy", "train.log", "motion.npz", "weights.json"]:
                if (d / f).exists():
                    tar.add(d / f, f"{d.name}/{f}")
            ckpts = sorted(d.glob("logs/*/*/model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            if ckpts:
                tar.add(ckpts[-1], f"{d.name}/{ckpts[-1].name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip", help="a processed clip folder (qpos.npy, meta.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policies", nargs="+", default=["taskhybrid_var", "tracking_var"],
                    choices=[k for k in TASKS if k.endswith("_var")])
    ap.add_argument("--init", nargs="*", default=[], help="as experiments/latent_policy.py --init")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-only", action="store_true", help="make the variants and motion files, train nothing")
    main(ap.parse_args())
