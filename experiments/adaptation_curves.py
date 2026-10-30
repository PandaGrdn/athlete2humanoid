"""How fast does a new skill become a robust policy? PPO adaptation curves on clips no start trained on.

    # 1. here: the clips re-retargeted with the current retarget.py, and each start's checkpoint
    python experiments/adaptation_curves.py prepare outputs/adapt_inputs \
        --clip <dir>/jumping_jacks <dir>/punches <dir>/discus \
        --start specialist=<...>/policy_tracking_var/model_1499.pt --start multi=<...>/policy_multi/model_1999.pt
    # 2. Modal, one L40S per clip, running the pushed --branch (docs/MODAL.md)
    python experiments/adaptation_curves.py launch outputs/adapt_inputs --tag adapt_v1 --branch <branch>
    #    each GPU job: python experiments/adaptation_curves.py run <inputs> --clip discus --out <dir>
    # 3. here: every run's curves -> tables and a chart
    python experiments/adaptation_curves.py report outputs/remote_adapt_v1_* --out docs/results/adaptation

1. For one clip, one policy per start (from scratch, or from a trained tracker's weights) is trained
   with PPO on the clip's automatic variants (skill/variations.py, laid end to end as
   experiments/multiclip_policy.py does), saving a checkpoint every --save-interval iterations.
   --parallel 1 (default) trains the starts one after another, so each has the GPU to itself and its
   minutes are what one new clip costs.
2. At each of --at iterations, every start's checkpoint goes through experiments/robustness_eval.py's
   short set: the clip, its held-out variants, 1 m/s pushes and 0.5x / 2x tempo. The last one also
   gets every mjlab suite (pushes, stage-timed pushes, start noise and poses, friction and mass,
   speed, tempo, no reference).
3. Minutes at N iterations = the variant build + the trainer's start-up + PPO time to N (the
   trainer's own "Time elapsed"), all on the job's GPU.

A new start (e.g. a better generalist) is one more curve: prepare a folder with only it (--no-scratch
--start frontier=<model.pt>[@<task>]), launch it with its own tag; report merges every run it is given.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRACK = "A2H-Tracking-Flat-Unitree-G1"  # the task the specialist and the 5-clip generalist were trained in
AT = (100, 300, 1000, 1500)
SHORT = ("clean", "transfer", "push", "tempo")  # the suites tested at every checkpoint
METRICS = ("task_success", "fall_rate", "stages_frac", "dtw_joint_rad", "joint_err_rad")
SUITE_NAMES = {"clean": "the clip", "transfer": "held-out variants", "push": "pushed 1 m/s", "tempo": "tempo 0.5x / 2x"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- prepare (here)

def max_joint_speed(qpos, fps):
    """Fastest hinge joint, rad/s (qpos: root pos + quat, then the joints)."""
    return float(np.abs(np.diff(qpos[:, 7:], axis=0)).max() * fps)


def reretarget(src, dst):
    """src's human.npy through the current retarget.py -> dst (qpos.npy, motion.csv, retarget_report.json,
    and src's meta, video latent and silhouette). -> what changed."""
    import retarget
    meta = json.loads((src / "meta.json").read_text())
    fps = float(meta["fps"])
    rt = retarget.G1Retargeter()
    targets, rotations = retarget.human_to_targets(np.load(src / "human.npy"), rt, fps=fps)
    qpos = rt.retarget_sequence(targets, rotations, fps=fps)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("human.npy", "video_latent.npz", "silhouette.mp4"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    (dst / "retarget_report.json").write_text(json.dumps(retarget.tracking_report(rt, qpos, targets), indent=1))
    np.save(dst / "qpos.npy", qpos)
    retarget.export_mjlab_csv(qpos, dst / "motion.csv")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (dst / "meta.json").write_text(json.dumps({**meta, "retarget": "proportional", "retarget_commit": commit}, indent=1))
    old = np.load(src / "qpos.npy")
    return {"frames": len(qpos), "fps": fps, "max_joint_speed_before": max_joint_speed(old, fps),
            "max_joint_speed_after": max_joint_speed(qpos, fps)}


def parse_start(s):
    """'name=path[@task]' (or 'scratch') -> (name, path or None, task)."""
    name, _, rest = s.partition("=")
    path, _, task = rest.partition("@")
    if not re.fullmatch(r"[a-z][a-z0-9]*", name):
        raise SystemExit(f"start names are lowercase letters and digits: {name!r}")
    return name, (Path(path) if path else None), task or TRACK


def prepare(a):
    out = Path(a.out).resolve()
    (out / "clips").mkdir(parents=True, exist_ok=True)
    starts = {} if a.no_scratch else {"scratch": {"checkpoint": None, "task": TRACK}}
    for s in a.start:
        name, path, task = parse_start(s)
        rel = None
        if path:
            rel = f"policy_{name}/{path.name}"
            (out / rel).parent.mkdir(exist_ok=True)
            shutil.copy(path, out / rel)
        starts[name] = {"checkpoint": rel, "task": task, "source": str(path) if path else None}
    (out / "starts.json").write_text(json.dumps(starts, indent=1))
    report = {}
    for c in a.clip:
        c = Path(c)
        report[c.name] = reretarget(c, out / "clips" / c.name)
        log(f"{c.name}: re-retargeted ({report[c.name]})")
    (out / "reretarget.json").write_text(json.dumps(report, indent=1))
    log(f"inputs in {out}: starts {', '.join(starts)}; clips {', '.join(report)}")


# ---------------------------------------------------------------- run (the GPU job)

def clip_qpos(d):
    return np.load(d / "qpos.npy"), float(json.loads((d / "meta.json").read_text())["fps"])


def build(clip_dir, out, py, env, seed=0, smoke=False):
    """-> out/motion_variants.npz, out/variants/variants.json (the training variants, as multiclip_policy.py
    lays them out), and out/eval/{motion.npz, task_spec.json, transfer/<held-out variant>.npz}."""
    from skill import task as T
    from skill import variations as V
    from skill.keypose import compile_keypose_spec
    from train_motion import csv_to_npz
    g1, c = V.G1(), clip_dir.name
    qpos, fps = clip_qpos(clip_dir)
    vs = V.generate(qpos, fps, V.TRAIN[:2] if smoke else V.TRAIN, seed, g1, log)
    for v in vs:
        v["name"] = f"{c}.{v['name']}"
    V.write(vs, out / "variants")
    segs, parts, start = [], [], 0
    for v in (v for v in vs if v["ok"]):
        d = out / "variants" / v["name"]
        m = dict(np.load(csv_to_npz(py, d / "motion.csv", fps, d / "motion.npz", env)))
        segs.append({"name": v["name"], "clip": c, "start": start, "length": len(m["joint_pos"]),
                     "spec": f"{v['name']}/task_spec.json", "params": v["params"]})
        parts.append(m)
        start += len(m["joint_pos"])
    (out / "variants" / "variants.json").write_text(json.dumps({"clips": [c], "segments": segs}, indent=1))
    np.savez(out / "motion_variants.npz", fps=parts[0]["fps"],
             **{k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "fps"})
    log(f"{len(segs)} training variants ({start / 50:.0f} s of motion)")
    e = out / "eval"
    (e / "transfer").mkdir(parents=True, exist_ok=True)
    csv_to_npz(py, clip_dir / "motion.csv", fps, e / "motion.npz", env)
    (e / "task_spec.json").write_text(json.dumps(compile_keypose_spec(T.body_positions_from_qpos(qpos), fps, name=c), indent=1))
    for v in (v for v in V.generate(qpos, fps, V.HELD_OUT[:2] if smoke else V.HELD_OUT, seed + 1000, g1, log) if v["ok"]):
        V.write([v], e / "held_out")
        csv_to_npz(py, e / "held_out" / v["name"] / "motion.csv", fps, e / "transfer" / f"{v['name']}.npz", env)
    log(f"eval set: the clip + {len(list((e / 'transfer').glob('*.npz')))} held-out variants")


ITER_RE = re.compile(r"Learning iteration (\d+)/\d+")
ELAPSED_RE = re.compile(r"Time elapsed: (?:(\d+) days?, )?(\d+):(\d\d):(\d\d)")


def elapsed_by_iter(text):
    """rsl_rl's train log -> {iteration: seconds of PPO (collection + learning) since iteration 0}."""
    out, it = {}, None
    for line in re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines():
        if m := ITER_RE.search(line):
            it = int(m[1])
        elif (m := ELAPSED_RE.search(line)) and it is not None:
            out[it] = int(m[1] or 0) * 86400 + int(m[2]) * 3600 + int(m[3]) * 60 + int(m[4])
    return out


def checkpoints(d):
    """policy folder -> {k: path of model_k.pt} (model_k: the weights after k + 1 PPO updates)."""
    return {int(p.stem.split("_")[1]): p for p in Path(d).glob("logs/*/*/model_*.pt")}


def pick(ckpts, n, tol):
    """The checkpoint closest to n PPO updates (model_{n-1}), within tol updates; else None."""
    if not ckpts:
        return None
    k = min(ckpts, key=lambda k: (abs(k + 1 - n), k))
    return k if abs(k + 1 - n) <= tol else None


def train(a, out, clip_dir, inp, starts, fps):
    """Every start's policy, a.parallel at a time -> out/timing.json {start: {startup_s, wall_s}}."""
    tf = out / "timing.json"
    timing = json.loads(tf.read_text()) if tf.exists() else {}
    todo = [n for n in starts if not (out / f"policy_{n}" / "weights.json").exists()]
    running = {}
    while todo or running:
        while todo and len(running) < a.parallel:
            name = todo.pop(0)
            d = out / f"policy_{name}"
            d.mkdir(exist_ok=True)
            s = starts[name]
            cmd = [sys.executable, ROOT / "train_motion.py", "--csv", clip_dir / "motion.csv", "--fps", fps, "--out", d,
                   "--iters", a.iters, "--num-envs", a.num_envs, "--task", s["task"], "--save-interval", a.save_interval,
                   "--variants", out / "variants" / "variants.json", "--train-motion", out / "motion_variants.npz"]
            if s["checkpoint"]:
                cmd += ["--init", inp / s["checkpoint"]]
            p = subprocess.Popen(list(map(str, cmd)), stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT)
            running[name] = (p, time.time(), None)
            log(f"training {name} from {s['checkpoint'] or 'scratch'} ({s['task']}); log {d / 'train.log'}")
        time.sleep(1 if a.smoke else 5)
        for name, (p, t0, t_up) in list(running.items()):
            if t_up is None and "Learning iteration" in (out / f"policy_{name}" / "train.log").read_text(errors="ignore"):
                running[name] = (p, t0, t_up := time.time())
            if p.poll() is not None:
                timing[name] = {"startup_s": (t_up or time.time()) - t0, "wall_s": time.time() - t0,
                                "returncode": p.returncode, "parallel": a.parallel}
                tf.write_text(json.dumps(timing, indent=1))
                log(f"{name} finished with code {p.returncode} after {timing[name]['wall_s'] / 60:.1f} min")
                del running[name]
    return timing


def find_policies(runs, wanted):
    """robustness_eval.find_policies for any policy names: each policy_<name>/ has its task in task.txt."""
    pols = {}
    for d in sorted(Path(runs).glob("policy_*")):
        name = d.name.removeprefix("policy_")
        ck = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if (not wanted or name in wanted) and ck and (d / "motion.npz").exists() and (d / "task.txt").exists():
            pols[name] = {"checkpoint": str(ck[-1]), "task": (d / "task.txt").read_text().strip(), "motion": str(d / "motion.npz")}
    if not pols:
        raise SystemExit(f"no policy_*/{{model_*.pt, motion.npz, task.txt}} in {runs}")
    return pols


def robust(run, out, clip, transfer, flags, reps, max_steps=None):
    cmd = [sys.executable, __file__, "robust", "--runs", run, "--out", out, "--target", clip, "--reps", reps,
           "--transfer", *transfer, *flags, *(["--max-steps", max_steps] if max_steps else [])]
    t0 = time.time()
    r = subprocess.run(list(map(str, cmd)))
    log(f"robustness_eval {Path(out).name}: code {r.returncode}, {(time.time() - t0) / 60:.1f} min")
    return r.returncode


def evaluate(a, out, starts):
    """robust.json for every start at every --at (eval/short) and at the last (eval/full)."""
    e = out / "eval"
    transfer = [f"{f.stem}={f}" for f in sorted((e / "transfer").glob("*.npz"))]
    final = max(a.at)
    for kind, ns, flags in (
            ("short", a.at, ["--no-vjepa", "--push", *([] if a.smoke else [1.0]), "--init", "--dynamics", 0, "--speed",
                             "--phase-push", "--starts", "--no-noref", "--tempo", *([] if a.smoke else [0.5, 2.0])]),
            ("full", [final], ["--no-vjepa"])):
        if (e / kind / "robust.json").exists():
            continue
        run = e / f"run_{kind}"
        for name, s in starts.items():
            ck = checkpoints(out / f"policy_{name}")
            for n in ns:
                k = pick(ck, n, a.save_interval // 2)
                if k is None:
                    log(f"{name}: no checkpoint near {n} iterations")
                    continue
                d = run / f"policy_{name}_{n}"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(ck[k], d / ck[k].name)
                shutil.copy(e / "motion.npz", d / "motion.npz")
                (d / "task.txt").write_text(s["task"])
        shutil.copy(e / "task_spec.json", run / "task_spec.json")
        robust(run, e / kind, a.clip, transfer, flags, 4 if a.smoke else a.reps, 20 if a.smoke else None)


def suite_means(conditions, policy):
    """robust.json conditions -> {suite: {metric: mean over that suite's conditions}} for one policy."""
    out = {}
    for suite in dict.fromkeys(c["suite"] for c in conditions):
        rs = [c["policies"][policy] for c in conditions if c["suite"] == suite and policy in c["policies"]]
        out[suite] = {k: (float(np.mean(v)) if (v := [r[k] for r in rs if r.get(k) is not None]) else None) for k in METRICS}
    return out


def summarise(out, clip, starts, at, build_s):
    """-> out/curves.json: per start, per iteration count, the short set's suite means, and minutes."""
    e = out / "eval"
    timing = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    short = json.loads((e / "short" / "robust.json").read_text())["conditions"] if (e / "short" / "robust.json").exists() else []
    full = json.loads((e / "full" / "robust.json").read_text())["conditions"] if (e / "full" / "robust.json").exists() else []
    res = {"clip": clip, "at": list(at), "build_s": build_s, "starts": {}}
    for name, s in starts.items():
        d = out / f"policy_{name}"
        el = elapsed_by_iter((d / "train.log").read_text(errors="ignore")) if (d / "train.log").exists() else {}
        t = timing.get(name, {})
        curve = {}
        for n in at:
            k = pick({i: None for i in el}, n, 10**9)
            pol = f"{name}_{n}"
            curve[str(n)] = {"ppo_s": el.get(k), "suites": suite_means(short, pol) if short and pol in short[0]["policies"] else {}}
            if el.get(k) is not None:
                curve[str(n)]["minutes"] = (build_s + t.get("startup_s", 0) + el[k]) / 60
        fpol = f"{name}_{max(at)}"
        res["starts"][name] = {"checkpoint": s["checkpoint"], "source": s.get("source"), "task": s["task"], "timing": t,
                               "curve": curve, "full": suite_means(full, fpol) if full and fpol in full[0]["policies"] else {}}
    (out / "curves.json").write_text(json.dumps(res, indent=1))
    (out / "curves.md").write_text(render([res]))
    print(render([res]))
    return res


def pack(out):
    with tarfile.open(out / "results.tar.gz", "w:gz") as tar:
        for f in ("curves.json", "curves.md", "timing.json", "variants/variants.json", "eval/task_spec.json"):
            if (out / f).exists():
                tar.add(out / f, f)
        for kind in ("short", "full"):
            for f in ("robust.json", "robust_summary.md"):
                if (out / "eval" / kind / f).exists():
                    tar.add(out / "eval" / kind / f, f"eval/{kind}/{f}")
        for d in out.glob("policy_*"):
            for f in ("train.log", "weights.json", "policy_metrics.json", "policy.mp4", "policy_qpos.npy"):
                if (d / f).exists():
                    tar.add(d / f, f"{d.name}/{f}")
            if ck := checkpoints(d):
                tar.add(ck[max(ck)], f"{d.name}/{ck[max(ck)].name}")


def run(a):
    inp, out = Path(a.inputs).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    starts = json.loads((inp / "starts.json").read_text())
    starts = {n: s for n, s in starts.items() if not a.starts or n in a.starts}
    if a.smoke:
        a.iters, a.save_interval, a.at = min(a.iters, 30), 10, [10, 30]
    clip_dir = inp / "clips" / a.clip
    _, fps = clip_qpos(clip_dir)
    py = Path(os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python"))
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), WANDB_MODE="disabled")
    t0 = time.time()
    if not (out / "motion_variants.npz").exists():
        build(clip_dir, out, py, env, a.seed, a.smoke)
        (out / "build_s.txt").write_text(f"{time.time() - t0:.1f}")
    build_s = float((out / "build_s.txt").read_text()) if (out / "build_s.txt").exists() else 0.0
    log(f"variants built in {build_s:.0f} s")
    train(a, out, clip_dir, inp, starts, fps)
    evaluate(a, out, starts)
    summarise(out, a.clip, starts, a.at, build_s)
    pack(out)


# ---------------------------------------------------------------- report (here)

def pct(x):
    return "-" if x is None else f"{x:.0%}"


def minutes_to(curve, threshold, suite="clean"):
    """First tested iteration count whose task success on suite reaches threshold -> (n, minutes), else None."""
    for n in sorted(curve, key=int):
        p = curve[n]
        s = p["suites"].get(suite, {}).get("task_success")
        if s is not None and s >= threshold:
            return int(n), p.get("minutes")
    return None


def robust_mean(suites):
    """Task success averaged over the short set's four suites (None if any is missing)."""
    v = [suites.get(s, {}).get("task_success") for s in SHORT]
    return None if any(x is None for x in v) else float(np.mean(v))


def render(results):
    """curves.json's (one per run; several runs may share a clip) -> markdown tables."""
    by_clip = {}
    for r in results:
        by_clip.setdefault(r["clip"], {}).update(r["starts"])
    lines = []
    for clip, starts in by_clip.items():
        lines += [f"### {clip}", "", "Task success (fall rate); joint error is the clean clip's, time-locked (rad).", "",
                  "| Start | Iterations | Minutes | The clip | Held-out variants | Pushed 1 m/s | Tempo 0.5x / 2x | Joint error |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, s in starts.items():
            for n in sorted(s["curve"], key=int):
                p = s["curve"][n]
                su = p["suites"]
                cells = [f"{pct(su.get(k, {}).get('task_success'))} ({pct(su.get(k, {}).get('fall_rate'))})" for k in SHORT]
                je = su.get("clean", {}).get("joint_err_rad")
                m = p.get("minutes")
                lines.append(f"| {name} | {n} | {'-' if m is None else f'{m:.0f}'} | " + " | ".join(cells)
                             + f" | {'-' if je is None else f'{je:.3f}'} |")
        lines.append("")
        fulls = {n: s["full"] for n, s in starts.items() if s.get("full")}
        if fulls:
            suites = [k for k in ("clean", "push", "push_phase", "init", "start", "speed", "tempo", "dynamics", "noref", "transfer")
                      if any(k in f for f in fulls.values())]
            lines += ["Full suite at the last checkpoint, task success (fall rate):", "",
                      "| Start | " + " | ".join(suites) + " |", "|---|" + "---|" * len(suites)]
            for name, f in fulls.items():
                lines.append(f"| {name} | " + " | ".join(
                    f"{pct(f.get(k, {}).get('task_success'))} ({pct(f.get(k, {}).get('fall_rate'))})" for k in suites) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def report(a):
    results = []
    for d in a.runs:
        f = next(iter(Path(d).glob("**/curves.json")), None)
        if f:
            results.append(json.loads(f.read_text()))
        else:
            log(f"{d}: no curves.json")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adaptation_curves.json").write_text(json.dumps(results, indent=1))
    lines = ["# Adaptation curves", "", render(results), "## Minutes to a success rate on the clip", "",
             "| Clip | Start | " + " | ".join(f"{t:.0%}" for t in a.thresholds) + " | Robust 80% |",
             "|---|---|" + "---|" * (len(a.thresholds) + 1)]
    for r in results:
        for name, s in r["starts"].items():
            cells = []
            for t in a.thresholds:
                hit = minutes_to(s["curve"], t)
                cells.append("not reached" if hit is None else f"{hit[1]:.0f} min ({hit[0]} it)")
            rob = next(((int(n), p.get("minutes")) for n, p in sorted(s["curve"].items(), key=lambda x: int(x[0]))
                        if (m := robust_mean(p["suites"])) is not None and m >= 0.8), None)
            cells.append("not reached" if rob is None else f"{rob[1]:.0f} min ({rob[0]} it)")
            lines.append(f"| {r['clip']} | {name} | " + " | ".join(cells) + " |")
    lines += ["", "Robust 80%: task success averaged over the clip, its held-out variants, 1 m/s pushes and 0.5x / 2x tempo."]
    (out / "adaptation_curves.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    chart(results, out / "adaptation_curves.png")


def chart(results, path):
    """Task success on the clip (solid) and averaged over the short set (dashed) vs minutes, one panel per clip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_clip = {}
    for r in results:
        by_clip.setdefault(r["clip"], {}).update(r["starts"])
    colors = ["#2a6fdb", "#e0832b", "#2b9e5f", "#9b4fd1", "#c23b3b"]
    names = list(dict.fromkeys(n for s in by_clip.values() for n in s))
    fig, axes = plt.subplots(1, len(by_clip), figsize=(4.2 * len(by_clip), 3.4), sharey=True, squeeze=False)
    for ax, (clip, starts) in zip(axes[0], by_clip.items()):
        for name, s in starts.items():
            pts = [(p["minutes"], p["suites"]) for n, p in sorted(s["curve"].items(), key=lambda x: int(x[0]))
                   if p.get("minutes") is not None and p["suites"]]
            if not pts:
                continue
            c = colors[names.index(name) % len(colors)]
            x = [m for m, _ in pts]
            ax.plot(x, [su.get("clean", {}).get("task_success") or 0 for _, su in pts], "-o", color=c, label=name, ms=4)
            ax.plot(x, [robust_mean(su) or 0 for _, su in pts], "--", color=c, alpha=0.6, lw=1)
        ax.set_title(clip)
        ax.set_xlabel("minutes on one L40S")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("task success")
    axes[0][0].legend(frameon=False, fontsize=8, title="solid: the clip; dashed: robust mean", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    log(f"wrote {path}")


# ---------------------------------------------------------------- launch (here -> Modal)

def launch(a):
    """One Modal job (one L40S) per clip, all at once; waits for them and unpacks each into
    outputs/remote_<tag>_<clip>/. --attach: follow jobs already submitted with this tag."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import modal
    from modal_backend import ModalRun
    if not a.tag.startswith("adapt_"):
        raise SystemExit("tags start with adapt_")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", a.branch):
        raise SystemExit(f"odd branch name: {a.branch}")
    if not a.attach and a.branch not in subprocess.run(["git", "ls-remote", "--heads", "origin", a.branch], cwd=ROOT,
                                                       capture_output=True, text=True).stdout:
        raise SystemExit(f"branch {a.branch} isn't on GitHub: push it first (the GPU job clones it from there)")
    inp = Path(a.inputs).resolve()
    clips = a.clips or sorted(p.name for p in (inp / "clips").iterdir() if p.is_dir())
    ModalRun.check_login()
    jobs = {}
    for clip in clips:
        session = f"{a.tag}_{clip}"
        out = ROOT / "outputs" / f"remote_{session}"
        out.mkdir(parents=True, exist_ok=True)

        def say(msg, out=out):
            line = f"{time.strftime('%H:%M:%S')} {msg}"
            print(f"[{out.name}] {line}", flush=True)
            with open(out / "runner.log", "a") as f:
                f.write(line + "\n")
        mr = ModalRun(session, out, say, a.modal_gpu)
        jobs[clip] = mr
        if a.attach:
            mr.call = modal.FunctionCall.from_id(json.loads((out / "modal_call.json").read_text())["call_id"])
            continue
        tar_f = out / "inputs.tar.gz"
        with tarfile.open(tar_f, "w:gz") as tar:
            tar.add(inp / "starts.json", "starts.json")
            tar.add(inp / "clips" / clip, f"clips/{clip}")
            for f in inp.glob("policy_*/*.pt"):
                tar.add(f, f"{f.parent.name}/{f.name}")
        if clip == clips[0]:
            mr.deploy()
        mr.upload([], tar_f, ROOT / "no-secrets-needed")  # no HF token: nothing here downloads gated weights
        mr.submit(f"python3 experiments/adaptation_curves.py run {mr.run_dir}/inputs --clip {clip} "
                  f"--out {mr.run_dir}/outputs/adapt {a.args}", a.branch)
    try:
        while jobs:
            time.sleep(a.poll_s)
            for clip, mr in list(jobs.items()):
                st = mr.state()
                if st in ("running", "unknown"):
                    continue
                mr.log(f"job state: {st}")
                for local, remote in (("job.log", f"{mr.run_dir}/jobs/{mr.session}/log"),
                                      ("results.tar.gz", f"{mr.run_dir}/outputs/adapt/results.tar.gz")):
                    try:
                        mr.download(remote, mr.out / local)
                    except Exception as e:  # noqa: BLE001
                        mr.log(f"could not download {local}: {str(e)[-200:]}")
                if (mr.out / "results.tar.gz").exists():
                    with tarfile.open(mr.out / "results.tar.gz") as tar:
                        tar.extractall(mr.out, filter="data")
                    mr.log("unpacked results.tar.gz")
                mr.cleanup()
                del jobs[clip]
    except KeyboardInterrupt:
        for mr in jobs.values():
            mr.cancel()
        raise


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="re-retarget clips and gather the starts' checkpoints into one inputs folder")
    p.add_argument("out")
    p.add_argument("--clip", nargs="+", default=[], help="processed clip folders (human.npy, meta.json, ...)")
    p.add_argument("--start", action="append", default=[], help="name=model_*.pt[@task] (repeatable)")
    p.add_argument("--no-scratch", action="store_true", help="no from-scratch start (an extra curve for an earlier run)")
    p = sub.add_parser("run", help="the GPU job: train every start on one clip, test along the way")
    p.add_argument("inputs", help="a prepare'd folder (starts.json, clips/, policy_*/)")
    p.add_argument("--clip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--starts", nargs="*", help="only these starts (default: all in starts.json)")
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--at", type=int, nargs="+", default=list(AT), help="iteration counts tested")
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--parallel", type=int, default=1, help="starts trained at once on the GPU (1: clean minutes)")
    p.add_argument("--reps", type=int, default=64, help="rollouts per policy per eval condition")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="2 variants, 30 iterations, tiny eval: plumbing test")
    p = sub.add_parser("launch", help="one Modal job per clip (docs/MODAL.md)")
    p.add_argument("inputs")
    p.add_argument("--tag", required=True, help="adapt_<...>: names the runs (outputs/remote_<tag>_<clip>)")
    p.add_argument("--branch", required=True, help="the pushed branch the jobs run")
    p.add_argument("--clips", nargs="*", help="default: every clip in the inputs")
    p.add_argument("--args", default="", help="extra flags for run (e.g. '--at 50 100 300')")
    p.add_argument("--modal-gpu", default=os.environ.get("A2H_MODAL_GPU"))
    p.add_argument("--poll-s", type=int, default=120)
    p.add_argument("--attach", action="store_true", help="follow already-submitted jobs (modal_call.json)")
    p = sub.add_parser("report", help="runs' curves.json -> tables, minutes-to-X, chart")
    p.add_argument("runs", nargs="+", help="result folders (outputs/remote_<tag>_<clip>)")
    p.add_argument("--out", required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.8, 0.9])
    return ap


if __name__ == "__main__":
    if sys.argv[1:2] == ["robust"]:  # robustness_eval.py, with policies named by this file (find_policies)
        from experiments import robustness_eval as RE
        sys.argv = [RE.__file__, *sys.argv[2:]]  # its mjlab stage re-runs RE with sys.argv[1:]
        RE.find_policies = find_policies
        RE.main(RE.parse())
    else:
        a = parser().parse_args()
        {"prepare": prepare, "run": run, "launch": launch, "report": report}[a.cmd](a)
