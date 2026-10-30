"""Motions that travel: each of travel_tasks.py's changes alone, against plain tracking, on one clip.

    # 1. here: clips re-retargeted with the current retarget.py, the warm start, earlier policies to compare
    python experiments/travel_policy.py prepare outputs/travel_inputs --clip <dir>/shot_put <dir>/lunge \
        --start multi=<...>/policy_multi/model_1999.pt --compare base_pro=<...>/policy_multi/model_1499.pt
    python experiments/travel_policy.py build outputs/travel_inputs --clips shot_put lunge --sets stock synced
    # 2. Modal, one L40S per clip, running the pushed --branch (docs/MODAL.md)
    python experiments/travel_policy.py launch outputs/travel_inputs --tag travel_v1 --branch <branch> --clips shot_put \
        --args "--policies base=A2H-Tracking-Flat-Unitree-G1 reward=A2H-TravelReward-Flat-Unitree-G1"
    #    each GPU job: python experiments/travel_policy.py run <inputs> --clip shot_put --out <dir> --policies ...
    # 3. here: every run -> one table
    python experiments/travel_policy.py report outputs/remote_travel_v1_* --out docs/results/travel

1. run builds the clip's training variants (skill/variations.py, laid end to end as
   experiments/multiclip_policy.py does) and, for policies marked @synced, the same grid through
   skill/travel_variants.py (holds synced on root-moving stages: more of a travelling clip's variants
   survive). The eval set is always the stock one: the clip and its held-out variants as
   skill/variations.py keeps them, so every policy is tested on the same thing.
2. Every --policies name=TASK[@synced] trains from --start (train_motion.py, all at once on the GPU,
   --iters PPO iterations); --compare checkpoints from the inputs (earlier runs) join the eval untrained.
3. experiments/robustness_eval.py tests them all in one env: the clip, its held-out variants, 1 m/s
   pushes, 0.5x / 2x tempo, start noise (no V-JEPA). travel.json adds, from the clean rollouts, how
   far the torso went: its net horizontal travel as a share of the reference's, and its distance from
   the reference's at the end.
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

TRACK = "A2H-Tracking-Flat-Unitree-G1"
SUITES = ("clean", "transfer", "push", "tempo", "init")
METRICS = ("task_success", "fall_rate", "stages_frac", "torso_err_m", "joint_err_rad")
EVAL_FLAGS = ["--no-vjepa", "--score", 16, "--push", 1.0, "--init", 1, "--dynamics", 0, "--speed", "--phase-push", "--starts",
              "--no-noref", "--tempo", 0.5, 2.0]
CLIP_FILES = ("qpos.npy", "meta.json", "motion.csv")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clip_qpos(d):
    return np.load(d / "qpos.npy"), float(json.loads((d / "meta.json").read_text())["fps"])


# ---------------------------------------------------------------- prepare (here)

def retarget_clip(src, dst):
    """src/human.npy -> dst/{qpos.npy, motion.csv, meta.json, retarget_report.json} with the current retarget."""
    import retarget
    human, meta = np.load(src / "human.npy"), json.loads((src / "meta.json").read_text())
    rt = retarget.G1Retargeter()
    targets, rotations = retarget.human_to_targets(human, rt, fps=meta["fps"])
    qpos = rt.retarget_sequence(targets, rotations, fps=meta["fps"])
    (dst / "retarget_report.json").write_text(json.dumps(retarget.tracking_report(rt, qpos, targets), indent=1))
    np.save(dst / "qpos.npy", qpos)
    retarget.export_mjlab_csv(qpos, dst / "motion.csv")
    (dst / "meta.json").write_text(json.dumps(meta | {"frames": len(qpos)}, indent=1))
    return qpos


def prepare(a):
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    for c in map(Path, a.clip):
        d = out / "clips" / c.name
        d.mkdir(exist_ok=True)
        if (c / "human.npy").exists() and not a.no_retarget:
            old = np.load(c / "qpos.npy") if (c / "qpos.npy").exists() else None
            q = retarget_clip(c, d)
            same = old is not None and old.shape == q.shape and np.abs(old - q).max() < 1e-3
            log(f"{c.name}: re-retargeted, {len(q)} frames ({'unchanged' if same else 'changed'} vs {c})")
        else:
            for f in CLIP_FILES:
                shutil.copy(c / f, d / f)
            log(f"{c.name}: copied")
    pols = {}
    for kind, specs in (("start", a.start), ("compare", a.compare)):
        for s in specs:
            name, rest = s.split("=", 1)
            path, task = rest.split("@") if "@" in rest else (rest, TRACK)
            d = out / f"{kind}_{name}"
            d.mkdir(exist_ok=True)
            shutil.copy(path, d / Path(path).name)
            pols[f"{kind}:{name}"] = {"checkpoint": f"{d.name}/{Path(path).name}", "task": task, "source": str(path)}
    (out / "inputs.json").write_text(json.dumps(pols, indent=1))
    log(f"{out}: {len(a.clip)} clips, {len(pols)} checkpoints")


# ---------------------------------------------------------------- run (the GPU job)

def build_variants(clip_dir, out, py, env, kind, seed=0, smoke=False):
    """-> out/variants_<kind>/variants.json + out/motion_variants_<kind>.npz. kind: "stock"
    (skill/variations.py) or "synced" (skill/travel_variants.py)."""
    from skill import travel_variants as TV
    from skill import variations as V
    from train_motion import csv_to_npz
    vdir, npz = out / f"variants_{kind}", out / f"motion_variants_{kind}.npz"
    if (vdir / "variants.json").exists() and npz.exists():
        return vdir / "variants.json", npz
    c = clip_dir.name
    qpos, fps = clip_qpos(clip_dir)
    gen = TV.generate if kind == "synced" else V.generate
    vs = gen(qpos, fps, V.TRAIN[:2] if smoke else V.TRAIN, seed, V.G1(), log)
    for v in vs:
        v["name"] = f"{c}.{v['name']}"
    V.write(vs, vdir)
    segs, parts, start = [], [], 0
    for v in (v for v in vs if v["ok"]):
        d = vdir / v["name"]
        m = dict(np.load(csv_to_npz(py, d / "motion.csv", fps, d / "motion.npz", env)))
        segs.append({"name": v["name"], "clip": c, "start": start, "length": len(m["joint_pos"]),
                     "spec": f"{v['name']}/task_spec.json", "params": v["params"], "synced": bool(v.get("synced"))})
        parts.append(m)
        start += len(m["joint_pos"])
    (vdir / "variants.json").write_text(json.dumps({"clips": [c], "segments": segs}, indent=1))
    np.savez(npz, fps=parts[0]["fps"], **{k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "fps"})
    log(f"{kind} variants: {len(segs)} of {len(vs)} kept ({start / 50:.0f} s of motion)")
    return vdir / "variants.json", npz


def build_eval(clip_dir, out, py, env, seed=0, smoke=False):
    """-> out/eval/{motion.npz, task_spec.json, transfer/<held-out variant>.npz}: multiclip_policy.py's eval set."""
    from skill import task as T
    from skill import variations as V
    from skill.keypose import compile_keypose_spec
    from train_motion import csv_to_npz
    e = out / "eval"
    if (e / "task_spec.json").exists():
        return e
    qpos, fps = clip_qpos(clip_dir)
    (e / "transfer").mkdir(parents=True, exist_ok=True)
    csv_to_npz(py, clip_dir / "motion.csv", fps, e / "motion.npz", env)
    for v in (v for v in V.generate(qpos, fps, V.HELD_OUT[:2] if smoke else V.HELD_OUT, seed + 1000, V.G1(), log) if v["ok"]):
        V.write([v], e / "held_out")
        csv_to_npz(py, e / "held_out" / v["name"] / "motion.csv", fps, e / "transfer" / f"{v['name']}.npz", env)
    (e / "task_spec.json").write_text(json.dumps(compile_keypose_spec(T.body_positions_from_qpos(qpos), fps,
                                                                      name=clip_dir.name), indent=1))
    log(f"eval set: the clip + {len(list((e / 'transfer').glob('*.npz')))} held-out variants")
    return e


def parse_policies(specs):
    """["name=TASK[@synced]", ...] -> {name: {"task", "variants"}}."""
    out = {}
    for s in specs:
        name, rest = s.split("=", 1)
        task, _, kind = rest.partition("@")
        if kind not in ("", "synced"):
            raise SystemExit(f"{s}: the variant set is @synced or nothing")
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise SystemExit(f"{name}: policy names are [a-z0-9_]")
        out[name] = {"task": task, "variants": kind or "stock"}
    return out


def checkpoints(d):
    return {int(p.stem.split("_")[1]): p for p in Path(d).glob("logs/*/*/model_*.pt")}


def train(a, out, clip_dir, inp, pols, sets, fps):
    """Every policy from the start, a.parallel at a time -> out/timing.json."""
    tf = out / "timing.json"
    timing = json.loads(tf.read_text()) if tf.exists() else {}
    start = json.loads((inp / "inputs.json").read_text()).get(f"start:{a.start}") if a.start else None
    todo = [n for n in pols if not (out / f"policy_{n}" / "weights.json").exists()]
    running = {}
    while todo or running:
        while todo and len(running) < a.parallel:
            name = todo.pop(0)
            d = out / f"policy_{name}"
            d.mkdir(exist_ok=True)
            vj, npz = sets[pols[name]["variants"]]
            cmd = [sys.executable, ROOT / "train_motion.py", "--csv", clip_dir / "motion.csv", "--fps", fps, "--out", d,
                   "--iters", a.iters, "--num-envs", a.num_envs, "--task", pols[name]["task"],
                   "--variants", vj, "--train-motion", npz]
            if start:
                cmd += ["--init", inp / start["checkpoint"]]
            p = subprocess.Popen(list(map(str, cmd)), stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT)
            running[name] = (p, time.time())
            log(f"training {name} ({pols[name]['task']}, {pols[name]['variants']} variants) from "
                f"{start['checkpoint'] if start else 'scratch'}; log {d / 'train.log'}")
        time.sleep(1 if a.smoke else 30)
        for name, (p, t0) in list(running.items()):
            if p.poll() is not None:
                timing[name] = {"wall_s": time.time() - t0, "returncode": p.returncode, "parallel": a.parallel}
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


def evaluate(a, out, pols, inp):
    """Every trained policy (last checkpoint) and every --compare checkpoint -> out/eval/robust/robust.json."""
    e = out / "eval"
    if (e / "robust" / "robust.json").exists():
        return
    run = e / "run"
    entries = {n: (ck[max(ck)], TRACK) for n in pols if (ck := checkpoints(out / f"policy_{n}"))}
    known = json.loads((inp / "inputs.json").read_text())
    for name in a.compare:
        c = known[f"compare:{name}"]
        entries[name] = (inp / c["checkpoint"], c["task"])
    for name, (ck, task) in entries.items():
        d = run / f"policy_{name}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(ck, d / Path(ck).name)
        shutil.copy(e / "motion.npz", d / "motion.npz")
        (d / "task.txt").write_text(task)  # the travel tasks' observations are tracking's: run in its env
    shutil.copy(e / "task_spec.json", run / "task_spec.json")
    transfer = [f"{f.stem}={f}" for f in sorted((e / "transfer").glob("*.npz"))]
    flags = ["--no-vjepa", "--push", "--init", "--dynamics", 0, "--speed", "--phase-push", "--starts", "--no-noref",
             "--tempo"] if a.smoke else EVAL_FLAGS
    cmd = [sys.executable, __file__, "robust", "--runs", run, "--out", e / "robust", "--target", a.clip,
           "--reps", 4 if a.smoke else a.reps, "--transfer", *transfer, *flags, *(["--max-steps", 20] if a.smoke else [])]
    t0 = time.time()
    r = subprocess.run(list(map(str, cmd)))
    log(f"robustness_eval: code {r.returncode}, {(time.time() - t0) / 60:.1f} min")


def travel_metrics(rollouts, ref_xy, condition=0):
    """Clean-condition rollouts (robustness_eval's rollouts.npz) -> {policy: {travel_frac, end_err_m}}:
    the torso's net horizontal travel along the reference's, as a share of it, and how far it ends
    from where the reference ends (m), over rollouts that start at the clip's start."""
    per = {}
    for k in rollouts.files:
        parts = k.split("/")
        if len(parts) != 3 or parts[0] != str(condition):
            continue
        q, s = rollouts[k], int(rollouts[k + "/start"])
        if s > 2 or len(q) < 2:  # the clean condition starts at the clip's first steps
            continue
        n = min(len(q), len(ref_xy) - s) - 1
        ref = ref_xy[s + n] - ref_xy[s]
        got = q[n, :2] - q[0, :2]
        dist = float(np.linalg.norm(ref))
        frac = float(got @ ref / dist ** 2) if dist > 1e-6 else None
        per.setdefault(parts[1], []).append((frac, float(np.linalg.norm(got - ref))))
    return {p: {"travel_frac": float(np.mean([f for f, _ in v])) if v[0][0] is not None else None,
                "end_err_m": float(np.mean([d for _, d in v])), "rollouts": len(v)} for p, v in per.items()}


def suite_means(conditions, policy):
    out = {}
    for suite in dict.fromkeys(c["suite"] for c in conditions):
        rs = [c["policies"][policy] for c in conditions if c["suite"] == suite and policy in c["policies"]]
        out[suite] = {k: (float(np.mean(v)) if (v := [r[k] for r in rs if r.get(k) is not None]) else None) for k in METRICS}
    return out


def summarise(a, out, pols):
    e = out / "eval"
    f = e / "robust" / "robust.json"
    if not f.exists():
        log("no robust.json: nothing to summarise")
        return None
    conds = json.loads(f.read_text())["conditions"]
    names = list(conds[0]["policies"])
    ref = np.load(e / "motion.npz")["body_pos_w"][:, 0, :2]
    trav = travel_metrics(np.load(e / "robust" / "rollouts.npz"), ref) if (e / "robust" / "rollouts.npz").exists() else {}
    res = {"clip": a.clip, "iters": a.iters, "start": a.start, "ref_travel_m": float(np.linalg.norm(ref[-1] - ref[0])),
           "variants": {k: len(json.loads((out / f"variants_{k}" / "variants.json").read_text())["segments"])
                        for k in ("stock", "synced") if (out / f"variants_{k}" / "variants.json").exists()},
           "timing": json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {},
           "policies": {n: {**(pols.get(n) or {"task": "compare", "variants": None}), "suites": suite_means(conds, n),
                            "travel": trav.get(n, {})} for n in names}}
    (out / "travel.json").write_text(json.dumps(res, indent=1))
    (out / "travel.md").write_text(render([res]))
    print(render([res]))
    return res


def pct(x):
    return "-" if x is None else f"{x:.0%}"


def num(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def render(results):
    lines = []
    for r in results:
        lines += [f"## {r['clip']} ({r['iters']} iterations from {r['start'] or 'scratch'}; reference travels "
                  f"{r['ref_travel_m']:.2f} m; variants {r['variants']})", "",
                  "| Policy | Task | The clip | Held-out variants | Pushed 1 m/s | Tempo 0.5x / 2x | Start noise | "
                  "Stages done | Torso error (m) | Travel done | End error (m) | Joint error (rad) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, p in r["policies"].items():
            s = p["suites"]

            def cell(suite):
                x = s.get(suite, {})
                return f"{pct(x.get('task_success'))} ({pct(x.get('fall_rate'))})" if x else "-"
            c, t = s.get("clean", {}), p.get("travel", {})
            task = p["task"].removeprefix("A2H-").removesuffix("-Flat-Unitree-G1") + ("@synced" if p.get("variants") == "synced" else "")
            lines.append(f"| {n} | {task} | " + " | ".join(cell(x) for x in SUITES)
                         + f" | {pct(c.get('stages_frac'))} | {num(c.get('torso_err_m'))} | {pct(t.get('travel_frac'))}"
                         f" | {num(t.get('end_err_m'))} | {num(c.get('joint_err_rad'), 3)} |")
        lines += ["", "Cells: task success (fall rate). Stages, torso and joint error: the clip, clean.", ""]
    return "\n".join(lines)


def pack(out):
    with tarfile.open(out / "results.tar.gz", "w:gz") as tar:
        for f in ("travel.json", "travel.md", "timing.json", "eval/task_spec.json", "eval/robust/robust.json",
                  "eval/robust/robust_summary.md", "variants_stock/variants.json", "variants_synced/variants.json"):
            if (out / f).exists():
                tar.add(out / f, f)
        for d in out.glob("policy_*"):
            for f in ("train.log", "weights.json", "policy_metrics.json", "policy.mp4", "policy_qpos.npy"):
                if (d / f).exists():
                    tar.add(d / f, f"{d.name}/{f}")
            if ck := checkpoints(d):
                tar.add(ck[max(ck)], f"{d.name}/{ck[max(ck)].name}")


def run(a):
    inp, out = Path(a.inputs).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (inp / "inputs.json").exists():  # the job didn't unpack them (a warm container): do it here
        tar_f = inp.parent / "inputs.tar.gz"
        for _ in range(12):
            if tar_f.exists():
                break
            log(f"waiting for {tar_f}")
            time.sleep(10)
        with tarfile.open(tar_f) as tar:
            tar.extractall(inp, filter="data")
        log(f"unpacked {tar_f}")
    if a.smoke:
        a.iters = min(a.iters, 20)
    pols = parse_policies(a.policies)
    clip_dir = inp / "clips" / a.clip
    _, fps = clip_qpos(clip_dir)
    py = Path(os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python"))
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), WANDB_MODE="disabled")
    if a.travel_config:
        os.environ["A2H_TRAVEL_CONFIG"] = str(Path(a.travel_config).resolve())  # read by travel_tasks in training
    built = inp / "built" / a.clip
    if built.is_dir() and not a.smoke:  # motion files made by `build` before the job: no GPU time on them
        shutil.copytree(built, out, dirs_exist_ok=True)
        log(f"using the motion files built in {built}")
    t0 = time.time()
    sets = {k: build_variants(clip_dir, out, py, env, k, a.seed, a.smoke) for k in {p["variants"] for p in pols.values()}}
    build_eval(clip_dir, out, py, env, a.seed, a.smoke)
    log(f"built in {(time.time() - t0) / 60:.1f} min")
    if pols:
        train(a, out, clip_dir, inp, pols, sets, fps)
    evaluate(a, out, pols, inp)
    summarise(a, out, pols)
    pack(out)


def build(a):
    """Here, before launching: the variants and eval set of each clip, as run makes them, into
    <inputs>/built/<clip>/ (run copies them). mjlab's CSV -> npz converter takes ~13 s a motion on a
    laptop CPU and ~2 min on the Modal job, where the GPU would sit idle meanwhile."""
    inp = Path(a.inputs).resolve()
    py = Path(os.environ.get("A2H_MJLAB_PYTHON", sys.executable))
    env = dict(os.environ, WANDB_MODE="disabled")
    for clip in a.clips:
        out = inp / "built" / clip
        out.mkdir(parents=True, exist_ok=True)
        for k in a.sets:
            build_variants(inp / "clips" / clip, out, py, env, k, a.seed)
        build_eval(inp / "clips" / clip, out, py, env, a.seed)
        for d in out.glob("variants_*/*/"):  # the per-variant npz files are in motion_variants_*.npz already
            (d / "motion.npz").unlink(missing_ok=True)


# ---------------------------------------------------------------- launch (here -> Modal)

def launch(a):
    """One Modal job (one L40S) per clip, all at once; waits for them and unpacks each into
    outputs/remote_<tag>_<clip>/. --attach: follow jobs already submitted with this tag."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import modal
    from modal_backend import ModalRun
    if not a.tag.startswith("travel_"):
        raise SystemExit("tags start with travel_")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", a.branch):
        raise SystemExit(f"odd branch name: {a.branch}")
    if not a.attach and a.branch not in subprocess.run(["git", "ls-remote", "--heads", "origin", a.branch], cwd=ROOT,
                                                       capture_output=True, text=True).stdout:
        raise SystemExit(f"branch {a.branch} isn't on GitHub: push it first (the GPU job clones it from there)")
    inp = Path(a.inputs).resolve()
    ModalRun.check_login()
    if a.app:  # scripts/modal_app.py deployed under another name (its own GPU type), the shared app left alone
        import modal_backend
        modal_backend.APP = a.app

        def deploy(self):
            self.log(f"deploying scripts/modal_app.py as {a.app} on {a.modal_gpu or 'L40S'}")
            r = subprocess.run([sys.executable, "-m", "modal", "deploy", str(ROOT / "scripts" / "modal_app.py"), "--name", a.app],
                               capture_output=True, text=True, env=dict(os.environ, **({"A2H_MODAL_GPU": a.modal_gpu} if a.modal_gpu else {})))
            if r.returncode:
                raise RuntimeError(f"modal deploy failed: {(r.stderr or r.stdout)[-300:]}")
        ModalRun.deploy = deploy
    jobs = {}
    for clip in a.clips:
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
        tar_f = pack_inputs(inp, clip, out)
        if clip == a.clips[0]:
            mr.deploy()
        mr.upload([], tar_f, ROOT / "no-secrets-needed")  # no HF token: nothing here downloads gated weights
        mr.submit(f"python3 experiments/travel_policy.py run {mr.run_dir}/inputs --clip {clip} "
                  f"--out {mr.run_dir}/outputs/travel {a.args}", a.branch)
    try:
        while jobs:
            time.sleep(a.poll_s)
            for clip, mr in list(jobs.items()):
                st = mr.state()
                if st in ("running", "unknown"):
                    continue
                mr.log(f"job state: {st}")
                for local, remote in (("job.log", f"{mr.run_dir}/jobs/{mr.session}/log"),
                                      ("results.tar.gz", f"{mr.run_dir}/outputs/travel/results.tar.gz")):
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


def launch_colab(a):
    """The same jobs on Colab A100 VMs (scripts/remote_run.py's Colab helpers), one per clip, one after
    another: for when Modal has no GPU to give. No secrets are uploaded: nothing here needs one."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import remote_run as RR
    inp = Path(a.inputs).resolve()
    for clip in a.clips:
        session = f"{a.tag}_{clip}"
        RR.SESSION, RR.BRANCH, RR.REMOTE = session, a.branch, "/content/a2h"
        RR.OUT = out = ROOT / "outputs" / f"remote_{session}"
        out.mkdir(parents=True, exist_ok=True)
        tar_f = pack_inputs(inp, clip, out)
        RR.colab("new", "-s", session, "--gpu", "A100", timeout=900)
        RR.log("A100 ready; uploading inputs")
        try:
            r = RR.REMOTE
            RR._exec(f"import os; os.makedirs('{r}/inputs', exist_ok=True)")
            RR.colab("upload", "-s", session, str(tar_f), f"{r}/inputs.tar.gz", timeout=900)
            RR._exec(f"import tarfile, os; os.makedirs('{r}/inputs', exist_ok=True); "
                     f"tarfile.open('{r}/inputs.tar.gz').extractall('{r}/inputs')")
            inner = ("cd /content/athlete2humanoid && bash scripts/setup_colab.sh --with-main && "
                     f"python3 experiments/travel_policy.py run {r}/inputs --clip {clip} --out {r}/outputs/travel {a.args}")
            RR._exec(f"""
import os, subprocess
R = "/content/athlete2humanoid"
if not os.path.exists(R):
    subprocess.run(["git", "clone", "-q", "--branch", "{a.branch}", "https://github.com/PandaGrdn/athlete2humanoid.git", R], check=True)
env = dict(os.environ, MUJOCO_GL="egl", A2H_OUTPUT_DIR="{r}/outputs", A2H_ASSETS_DIR=R + "/assets",
           A2H_MJLAB_PYTHON="/content/mjlab_env/bin/python")
subprocess.Popen(["python3", R + "/scripts/job.py", "{session}", "--", "bash", "-c", {inner!r}], env=env,
                 start_new_session=True, cwd="/content", stdout=open("{r}/job_{session}.out", "w"), stderr=subprocess.STDOUT)
print("launched")
""")
            RR.log(f"{session} launched on Colab")
            keeper, state = RR.keep_busy(), "running"
            while state in ("running", "unknown"):
                time.sleep(a.poll_s)
                state = RR.status()
                if state == "running" and keeper.poll() is not None:
                    keeper = RR.keep_busy()
            keeper.kill()
            RR.log(f"job state: {state}")
            RR.fetch_results({"results.tar.gz": f"{r}/outputs/travel/results.tar.gz"})
        finally:
            RR.colab("stop", "-s", session, check=False)
            RR.log("VM stopped")


def pack_inputs(inp, clip, out):
    tar_f = out / "inputs.tar.gz"
    with tarfile.open(tar_f, "w:gz") as tar:
        tar.add(inp / "inputs.json", "inputs.json")
        tar.add(inp / "clips" / clip, f"clips/{clip}")
        if (inp / "built" / clip).is_dir():
            tar.add(inp / "built" / clip, f"built/{clip}")
        for f in [*inp.glob("start_*/*.pt"), *inp.glob("compare_*/*.pt"), *inp.glob("*.json")]:
            if f.name != "inputs.json":
                tar.add(f, str(f.relative_to(inp)))
    return tar_f


def report(a):
    res = [json.loads((Path(r) / "travel.json").read_text()) for r in a.runs if (Path(r) / "travel.json").exists()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "travel.json").write_text(json.dumps(res, indent=1))
    (out / "travel.md").write_text(render(res))
    print(render(res))


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="re-retarget clips, gather checkpoints into one inputs folder")
    p.add_argument("out")
    p.add_argument("--clip", nargs="+", default=[], help="processed clip folders (human.npy or qpos.npy, meta.json)")
    p.add_argument("--no-retarget", action="store_true", help="copy qpos.npy / motion.csv as they are")
    p.add_argument("--start", action="append", default=[], help="name=model_*.pt[@task]: a warm start (repeatable)")
    p.add_argument("--compare", action="append", default=[], help="name=model_*.pt[@task]: evaluated, not trained")
    p = sub.add_parser("run", help="the GPU job: train the policies on one clip, test them")
    p.add_argument("inputs", help="a prepare'd folder")
    p.add_argument("--clip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policies", nargs="*", default=[], help="name=TASK[@synced] (synced: skill/travel_variants.py)")
    p.add_argument("--start", help="the warm start's name (prepare --start), else from scratch")
    p.add_argument("--compare", nargs="*", default=[], help="prepare --compare names to evaluate alongside")
    p.add_argument("--travel-config", help="travel_tasks.TRAVEL overrides (JSON)")
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--parallel", type=int, default=8, help="policies trained at once on the GPU")
    p.add_argument("--reps", type=int, default=64, help="rollouts per policy per eval condition")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="2 variants, 20 iterations, tiny eval: plumbing test")
    p = sub.add_parser("build", help="here: the clips' variants and eval sets, so the GPU job needn't make them")
    p.add_argument("inputs")
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--sets", nargs="+", default=["stock"], choices=["stock", "synced"])
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("launch", help="one Modal job per clip (docs/MODAL.md)")
    p.add_argument("inputs")
    p.add_argument("--tag", required=True, help="travel_<...>: names the runs (outputs/remote_<tag>_<clip>)")
    p.add_argument("--branch", required=True, help="the pushed branch the jobs run")
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--args", default="", help="flags for run (--policies ..., --start ..., --compare ...)")
    p.add_argument("--modal-gpu", default=os.environ.get("A2H_MODAL_GPU"))
    p.add_argument("--poll-s", type=int, default=120)
    p.add_argument("--attach", action="store_true", help="follow already-submitted jobs (modal_call.json)")
    p.add_argument("--app", help="deploy the Modal app under this name (e.g. with --modal-gpu A100-40GB when the "
                   "shared app's GPU type has no capacity)")
    p.add_argument("--backend", choices=["modal", "colab"], default="modal", help="colab: A100 VMs, one clip at a time")
    p = sub.add_parser("report", help="runs' travel.json -> one table")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    return ap


if __name__ == "__main__":
    if sys.argv[1:2] == ["robust"]:  # robustness_eval.py, with policies named by this file (find_policies)
        from experiments import robustness_eval as RE
        sys.argv = [RE.__file__, *sys.argv[2:]]  # its mjlab stage re-runs RE with sys.argv[1:]
        RE.find_policies = find_policies
        RE.main(RE.parse())
    else:
        a = parser().parse_args()
        if a.cmd == "launch" and a.backend == "colab":
            launch_colab(a)
        else:
            {"prepare": prepare, "build": build, "run": run, "launch": launch, "report": report}[a.cmd](a)
