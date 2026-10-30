"""A zero-shot generalist: one tracker over many motions, tested on clips it never trained on.

The 5-clip generalist (`multi`, experiments/multiclip_policy.py) completes 0% of jumping jacks, punches or
the discus, none of which it saw. This trains the same tracker on more and more motions, level by level,
and measures whether that changes (docs/FRONTIER_GENERALIST.md):

    real        the real clips that aren't test clips and pass trackable() (squat jump, front kick, squat,
                arms, lunge, shot put; the hurdles film ends in mid-air), each re-retargeted from its
                human.npy with today's retarget.py
    real_bank   + the procedural bank (latent/bank.py), minus its jumping jacks and punches: the test
                clips' procedural twins
    all         + recombinations of all of those (latent/distill.py variation()), as the latent policy's
                distillation set builds them

Every motion is checked trackable before it is trained on (trackable(): joint limits and speed, upright,
on the floor, starting and ending on it, flights gravity explains, no sliding stance foot), then expanded
into its automatic variants (skill/variations.py, which drops any variant whose own reference fails its
keypose spec).

1. Locally, on the CPU: the pool, as clip folders and motion CSVs (no mjlab needed)
       python experiments/generalist_policy.py pool --out outputs/gen_inputs --clips <clip folders>
2. On the GPU, through the multiclip job (multiclip_policy.py hands --level runs to this file):
       python3 scripts/remote_run.py multiclip --tag gen_real --reuse outputs/gen_inputs --with-weights \\
           --backend modal --branch <branch> --iters 2000 --args "--level real --baselines multi multi_future"
   - every CSV of the level and of the eval sets -> mjlab motion.npz, all in one process (convert());
   - the policies train side by side: gen_multi from the 5-clip multi, gen_future from the 5-clip
     multi_future (mjlab's inputs plus the retarget 0.5 / 1 s ahead), the _l ones (a bigger network)
     and _s ones from scratch;
   - experiments/robustness_eval.py on each clip: the zero-shot clips with every suite, the trained real
     clips on the clean clip and its held-out variants. --baselines evaluates those policies from the
     inputs too, untrained here.
   -> generalist.json, generalist_summary.md, results.tar.gz
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


def log(msg):  # as experiments/latent_policy.py's, which this can't import in mjlab's env (no mink there)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


REAL_TRAIN = ("squat_jump", "front_kick", "squat", "arms", "lunge", "shot_put", "hurdles")  # candidates
ZERO_SHOT = ("jumping_jacks", "punches", "discus")  # shared by the four frontier directions
BANK_EXCLUDE = ("jumping_jacks", "punch")  # the bank's procedural twins of two zero-shot clips
LEVELS = {"real": ("real",), "real_bank": ("real", "bank"), "all": ("real", "bank", "recombo")}
# Variant grids (mirror, tempo, amplitude, warp): real clips get skill/variations.py's full TRAIN grid; the
# bank is already parametric (no warps); recombinations are random already (mirror only).
BANK_GRID = [(m, s, a, 0.0) for m in (False, True) for s in (0.75, 1.0, 1.33) for a in (0.85, 1.0, 1.1)]
RECOMBO_GRID = [(False, 1.0, 1.0, 0.0), (True, 1.0, 1.0, 0.0)]
# name -> (task, the 5-clip policy in --init-dir it starts from; None: from scratch)
POLICIES = {"gen_multi": ("A2H-Tracking-Flat-Unitree-G1", "multi"),
            "gen_future": ("A2H-TrackingFuture-Flat-Unitree-G1", "multi_future"),
            "gen_future_s": ("A2H-TrackingFuture-Flat-Unitree-G1", None),
            "gen_multi_l": ("A2H-TrackingL-Flat-Unitree-G1", None),
            "gen_future_l": ("A2H-TrackingFutureL-Flat-Unitree-G1", None)}
ROLLOUT_CLIP = "squat_jump"  # train_motion.py's rollout video and metrics

# trackable(): what a G1 reference must satisfy to be worth tracking
MAX_TILT = 1.0  # rad, pelvis from upright (the front kick leans 0.95; the eval's fall test is 1.4)
MIN_TORSO_Z = 0.45  # m (the bank's deepest squat: 0.51; the eval's fall test is 0.35)
MAX_SINK = 0.03  # m, lowest foot below standing ankle height
MAX_SLIDE = 0.5  # m/s, a planted foot's 95th-percentile horizontal speed
MIN_RANGE = 0.05  # rad: a motion whose every joint moves less than this is a frozen pose


# ---------------------------------------------------------------- 1. the pool (local)

def re_retarget(src, dst):
    """A processed clip folder -> dst, its qpos re-made from human.npy by today's retarget.py. -> report."""
    import retarget
    meta = json.loads((src / "meta.json").read_text())
    fps = float(meta["fps"])
    H = np.load(src / "human.npy")
    rt = retarget.G1Retargeter()
    q = rt.retarget_sequence(*retarget.human_to_targets(H, rt, fps=fps), fps=fps)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("human.npy", "silhouette.mp4", "video_latent.npz", "skill_spec.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    np.save(dst / "qpos.npy", q)
    retarget.export_mjlab_csv(q, dst / "motion.csv")
    (dst / "meta.json").write_text(json.dumps(meta | {"frames": len(q), "re_retargeted_from": str(src)}, indent=1))
    old = np.load(src / "qpos.npy")
    speed = lambda x: float(np.abs(np.diff(x[:, 7:], axis=0)).max() * fps)  # noqa: E731
    return {"frames": len(q), "max_joint_speed": round(speed(q), 1), "old_max_joint_speed": round(speed(old), 1),
            "max_change_rad": round(float(np.abs(q[:, 7:] - old[:, 7:]).max()), 3) if len(old) == len(q) else None}


def _runs(mask):
    """(start, end) of each run of True, end exclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def stand_foot_height(g1):
    q = g1.model.key("stand").qpos[None].astype(np.float64)
    from skill.variations import L_ANKLE
    return float(g1.bodies(q)[0, L_ANKLE, 2])


def trackable(q, g1, foot0, fps=30.0, slide=True):
    """Can the G1 track this reference at all? -> (qpos with joints clipped to their limits, reasons it can't:
    [] if it can). The rules a real retarget obeys (retarget.py's joint-speed cap and gravity check), plus
    the things latent/distill.py's random recombinations break: upright, on the floor, feet not sliding.
    slide=False skips the last for real clips: retarget.py pins their stance feet, and the moves that travel
    (lunge, glide, spin) read as sliding at this resolution though trackers learn them (docs/PRO_CLIPS.md)."""
    import retarget
    from skill import task as T
    from skill.variations import L_ANKLE, MAX_LIMIT_FRAC, R_ANKLE
    q, why = np.asarray(q, np.float64).copy(), []
    over = (q[:, 7:] < g1.lo - 0.02) | (q[:, 7:] > g1.hi + 0.02)
    if over.mean() > MAX_LIMIT_FRAC:
        why.append(f"{over.mean():.0%} of joint samples past their limits")
    q[:, 7:] = np.clip(q[:, 7:], g1.lo, g1.hi)
    if np.ptp(q[:, 7:], axis=0).max() < MIN_RANGE:
        why.append("a frozen pose")
    speed = np.abs(np.diff(q[:, 7:], axis=0)).max() * fps
    if speed > retarget.MAX_JOINT_SPEED * 1.05:
        why.append(f"a joint at {speed:.0f} rad/s")
    w, x, y, z = q[:, 3:7].T / np.linalg.norm(q[:, 3:7], axis=1)
    tilt = np.arccos(np.clip(1 - 2 * (x ** 2 + y ** 2), -1, 1))
    if tilt.max() > MAX_TILT:
        why.append(f"tilts {tilt.max():.2f} rad")
    pos = g1.bodies(q)
    if pos[:, 7, 2].min() < MIN_TORSO_Z:  # BODIES[7] = torso_link
        why.append(f"torso down to {pos[:, 7, 2].min():.2f} m")
    feet = pos[:, [L_ANKLE, R_ANKLE], :]
    low = feet[:, :, 2].min(1) - foot0
    if low.min() < -MAX_SINK:
        why.append(f"a foot {-low.min() * 100:.0f} cm into the floor")
    if max(low[0], low[-1]) > T.DOWN:
        # an episode started there begins in mid-air: the hurdles film ends in flight, and its last
        # frames took a third of all of mjlab's adaptive resets while failing every one
        why.append("starts or ends in the air")
    for a, b in _runs(low > T.AIR):  # both feet up: only if the pelvis rises as a real flight must
        ends = q[[max(a - 1, 0), min(b, len(q) - 1)], 2].mean()
        if q[a:b, 2].max() - ends < retarget.FLIGHT_RISE_FRAC * retarget.GRAVITY * ((b - a) / fps) ** 2 / 8:
            why.append(f"floats for {(b - a) / fps:.2f} s")
            break
    planted = feet[:, :, 2] - foot0 < T.DOWN
    both = planted[1:] & planted[:-1]
    v = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=-1) * fps
    if slide and both.any() and np.percentile(v[both], 95) > MAX_SLIDE:
        why.append(f"a planted foot slides at {np.percentile(v[both], 95):.1f} m/s")
    return q, why


def grounded_span(q, g1, foot0):
    """(first, last + 1): q's frames from the first to the last with a foot on the floor."""
    from skill import task as T
    from skill.variations import L_ANKLE, R_ANKLE
    pos = g1.bodies(q)
    on = np.flatnonzero(np.minimum(pos[:, L_ANKLE, 2], pos[:, R_ANKLE, 2]) - foot0 <= T.DOWN)
    return (int(on[0]), int(on[-1]) + 1) if len(on) else (0, 0)


def bank_motions():
    from latent import bank
    return {k: q for k, q in bank.motion_bank().items() if k not in BANK_EXCLUDE}


def recombinations(base, n, g1, foot0, seed=0, max_tries=None):
    """n trackable latent/distill.py variations of base (name -> qpos at 30 fps) -> (name -> qpos, rejections)."""
    from latent import distill
    rng, out, why = np.random.default_rng(seed), {}, {}
    tries = 0
    while len(out) < n and tries < (max_tries or 20 * n):
        tries += 1
        q, w = trackable(distill.variation(base, rng), g1, foot0)
        if w:
            for r in w:
                k = re.sub(r"[\d.]+", "#", r)
                why[k] = why.get(k, 0) + 1
        else:
            out[f"rc{len(out):03d}"] = q
    return out, {"tries": tries, "kept": len(out), "rejected": why}


def pool(a):
    """-> out/clips/<clip>/ (re-retargeted), out/clips/_pool/{variants/<segment>/, levels.json, evals.json,
    pool.json}, out/policy_<name>/ (the 5-clip policies to start from and compare with)."""
    from skill import variations as V
    from skill import task as T
    from skill.keypose import compile_keypose_spec
    out = Path(a.out).resolve()
    clips_out, P = out / "clips", out / "clips" / "_pool"
    (P / "variants").mkdir(parents=True, exist_ok=True)
    g1 = V.G1()
    foot0 = stand_foot_height(g1)
    report = {"retarget": {}, "sources": {}, "trackable": {}}

    srcs = {Path(c).name: Path(c) for c in a.clips}
    missing = [c for c in (*REAL_TRAIN, *ZERO_SHOT) if c not in srcs]
    if missing:
        raise SystemExit(f"--clips lacks {missing}")
    real = {}
    for c, src in srcs.items():
        report["retarget"][c] = re_retarget(src, clips_out / c)
        log(f"{c}: re-retargeted {report['retarget'][c]}")
        real[c] = np.load(clips_out / c / "qpos.npy")
    train = []
    for c in REAL_TRAIN:
        why = trackable(real[c], g1, foot0, slide=False)[1]
        report["trackable"][f"clip/{c}"] = why
        if why:
            log(f"{c}: not trained on ({'; '.join(why)})")
        else:
            train.append(c)

    groups = {"real": {c: (real[c], V.TRAIN) for c in train}}
    bank = bank_motions()
    if a.smoke:
        bank = dict(list(bank.items())[:3])
    kept_bank = {}
    for k, q in bank.items():
        q, why = trackable(q, g1, foot0)
        report["trackable"][f"bank/{k}"] = why
        if why:
            log(f"bank/{k}: not trackable ({'; '.join(why)})")
        else:
            kept_bank[f"bank_{k}"] = q
    groups["bank"] = {k: (q, BANK_GRID) for k, q in kept_bank.items()}
    base = {f"bank/{k[5:]}": q for k, q in kept_bank.items()} | {f"clip/{c}": real[c] for c in train}
    rc, rep = recombinations(base, 4 if a.smoke else a.recombinations, g1, foot0, a.seed)
    report["recombinations"] = rep
    log(f"recombinations: {rep}")
    groups["recombo"] = {k: (q, RECOMBO_GRID) for k, q in rc.items()}

    segs = {}
    for g, motions in groups.items():
        segs[g] = []
        for name, (q, grid) in motions.items():
            vs = V.generate(q, 30.0, grid[:2] if a.smoke else grid, a.seed, g1, log=lambda *_: None)
            for v in vs:
                v["name"] = f"{name}.{v['name']}"
            V.write(vs, P / "variants")
            ok = [v for v in vs if v["ok"]]
            report["sources"][name] = {"group": g, "variants": len(vs), "kept": len(ok)}
            segs[g] += [{"name": v["name"], "clip": name, "frames": v["frames"], "params": v["params"],
                         "csv": f"variants/{v['name']}/motion.csv", "spec": f"variants/{v['name']}/task_spec.json"}
                        for v in ok]
        log(f"{g}: {len(motions)} motions -> {len(segs[g])} variants ({sum(s['frames'] for s in segs[g]) / 30:.0f} s)")
    levels = {lv: [s for g in gs for s in segs[g]] for lv, gs in LEVELS.items()}
    (P / "levels.json").write_text(json.dumps(levels, indent=1))

    evals = {}  # what every policy is tested on, clip by clip, as multiclip_policy.py builds it
    for c in (*train, *ZERO_SHOT):
        e = P / "evals" / c
        e.mkdir(parents=True, exist_ok=True)
        spec = compile_keypose_spec(T.body_positions_from_qpos(real[c]), 30.0, name=c)
        (e / "task_spec.json").write_text(json.dumps(spec, indent=1))
        held = V.HELD_OUT[:2] if a.smoke else V.HELD_OUT
        vs = [v for v in V.generate(real[c], 30.0, held, a.seed + 1000, g1, log=lambda *_: None) if v["ok"]]
        V.write(vs, e / "held_out")
        evals[c] = {"csv": f"../{c}/motion.csv", "spec": f"evals/{c}/task_spec.json",
                    "held_out": {v["name"]: f"evals/{c}/held_out/{v['name']}/motion.csv" for v in vs}}
        log(f"{c}: eval set, {len(vs)} held-out variants; spec: {spec['summary'][:80]}")
    (P / "evals.json").write_text(json.dumps(evals, indent=1))

    for name, ck in (("multi", a.multi), ("multi_future", a.multi_future)):
        if ck:
            (out / f"policy_{name}").mkdir(exist_ok=True)
            shutil.copy(ck, out / f"policy_{name}" / Path(ck).name)
    report["real_train"] = train
    report["levels"] = {lv: {"segments": len(s), "seconds": round(sum(x["frames"] for x in s) / 30)} for lv, s in levels.items()}
    (P / "pool.json").write_text(json.dumps(report, indent=1))
    log(f"pool in {out}: " + ", ".join(f"{lv} {v['segments']} variants / {v['seconds']} s" for lv, v in report["levels"].items()))


# ---------------------------------------------------------------- 2. CSV -> mjlab motion.npz (mjlab's env)

def convert(jobs_file, batch=256):
    """In mjlab's virtualenv: every [csv, fps, dst] in jobs_file -> dst, the motion.npz mjlab.scripts.csv_to_npz
    would write (50 Hz, by forward kinematics), but for all of them in one process, `batch` at a time in
    parallel environments (csv_to_npz builds a simulation per file, one env)."""
    import inspect

    import torch
    from mjlab.scripts import csv_to_npz as C
    jobs = json.loads(Path(jobs_file).read_text())
    jobs = [j for j in jobs if not Path(j[2]).exists()]
    if not jobs:
        return
    names = re.findall(r'"(\w+_joint)"', inspect.getsource(C.main))  # the G1 joint order csv_to_npz uses
    assert len(names) == 29, names
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n = min(batch, len(jobs))
    sim_cfg = C.SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / 50
    scene_cfg = C.unitree_g1_flat_tracking_env_cfg().scene
    scene_cfg.num_envs = n
    scene = C.Scene(scene_cfg, device=device)
    model = scene.compile()
    sim = C.Simulation(num_envs=n, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot = scene["robot"]
    idx = robot.find_joints(names, preserve_order=True)[0]
    t0 = time.time()
    for b in range(0, len(jobs), n):
        part = jobs[b:b + n]
        ms = [C.MotionLoader(motion_file=c, input_fps=f, output_fps=50, device=device, line_range=None) for c, f, _ in part]
        T = max(m.output_frames for m in ms)
        pad = lambda x: torch.cat([x, x[-1:].expand(T - len(x), *x.shape[1:])]) if len(x) < T else x  # noqa: E731
        fields = ("motion_base_poss", "motion_base_rots", "motion_base_lin_vels", "motion_base_ang_vels",
                  "motion_dof_poss", "motion_dof_vels")
        st = {f: torch.stack([pad(getattr(m, f)) for m in ms], 1) for f in fields}  # (T, envs, ...)
        rows = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
        e = len(part)
        scene.reset()
        for t in range(T):
            root = robot.data.default_root_state.clone()
            root[:e, 0:3] = st["motion_base_poss"][t]
            root[:, :2] += scene.env_origins[:, :2]
            root[:e, 3:7] = st["motion_base_rots"][t]
            root[:e, 7:10] = st["motion_base_lin_vels"][t]
            root[:e, 10:] = st["motion_base_ang_vels"][t]
            robot.write_root_state_to_sim(root)
            jp, jv = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            jp[:e, idx], jv[:e, idx] = st["motion_dof_poss"][t], st["motion_dof_vels"][t]
            robot.write_joint_state_to_sim(jp, jv)
            sim.forward()
            scene.update(sim.mj_model.opt.timestep)
            d = robot.data
            for k, x in (("joint_pos", d.joint_pos), ("joint_vel", d.joint_vel), ("body_pos_w", d.body_link_pos_w),
                         ("body_quat_w", d.body_link_quat_w), ("body_lin_vel_w", d.body_link_lin_vel_w),
                         ("body_ang_vel_w", d.body_link_ang_vel_w)):
                rows[k].append(x[:e].cpu().numpy().copy())
        stacked = {k: np.stack(v, 1) for k, v in rows.items()}  # (envs, T, ...)
        origin = scene.env_origins[:e, :2].cpu().numpy()
        for i, (m, (_, _, dst)) in enumerate(zip(ms, part)):
            arr = {k: v[i, :m.output_frames] for k, v in stacked.items()}
            arr["body_pos_w"] = arr["body_pos_w"].copy()
            arr["body_pos_w"][..., :2] -= origin[i]  # as csv_to_npz's single env at the origin
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, fps=[50], **arr)
        log(f"converted {b + e}/{len(jobs)} motions ({time.time() - t0:.0f} s)")


def to_npz(jobs, py, env, work):
    f = work / "convert_jobs.json"
    f.write_text(json.dumps([[str(c), float(fps), str(d)] for c, fps, d in jobs]))
    subprocess.run([str(py), __file__, "convert", str(f)], check=True, env=env)


# ---------------------------------------------------------------- 3. train + evaluate (GPU)

def build_level(a, out, P, py, env):
    """The level's variants -> out/motion_variants.npz + out/variants/variants.json; the eval sets ->
    out/evals/<clip>/{motion.npz, task_spec.json, transfer/<variant>.npz}."""
    segs = json.loads((P / "levels.json").read_text())[a.level]
    evals = json.loads((P / "evals.json").read_text())
    if a.smoke:
        segs = segs[:6]
    jobs = [(P / s["csv"], 30.0, out / "npz" / f"{s['name']}.npz") for s in segs]
    to_npz(jobs + eval_jobs(out, P, evals, eval_clips(a)), py, env, out)
    parts, rows, start = [], [], 0
    for s in segs:
        m = dict(np.load(out / "npz" / f"{s['name']}.npz"))
        n = len(m["joint_pos"])
        rows.append({"name": s["name"], "clip": s["clip"], "start": start, "length": n,
                     "spec": str((P / s["spec"]).resolve()), "params": s["params"]})
        parts.append(m)
        start += n
    np.savez(out / "motion_variants.npz", fps=parts[0]["fps"],
             **{k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "fps"})
    (out / "variants").mkdir(exist_ok=True)
    (out / "variants" / "variants.json").write_text(json.dumps(
        {"clips": sorted({s["clip"] for s in segs}), "segments": rows}, indent=1))
    shutil.rmtree(out / "npz")
    copy_specs(out, P, evals, eval_clips(a))
    log(f"level {a.level}: {len(rows)} variants of {len({s['clip'] for s in segs})} motions ({start / 50:.0f} s)")


def eval_jobs(out, P, evals, clips):
    jobs = []
    for c in clips:
        jobs.append((P / evals[c]["csv"], 30.0, out / "evals" / c / "motion.npz"))
        jobs += [(P / f, 30.0, out / "evals" / c / "transfer" / f"{v}.npz") for v, f in evals[c]["held_out"].items()]
    return jobs


def copy_specs(out, P, evals, clips):
    for c in clips:
        shutil.copy(P / evals[c]["spec"], out / "evals" / c / "task_spec.json")


def eval_clips(a):
    return [*a.eval_train, *a.zero_shot]


def train_all(a, out, clips_dir):
    from experiments.latent_policy import mirror_checkpoints
    procs, roll = {}, clips_dir / (ROLLOUT_CLIP if (clips_dir / ROLLOUT_CLIP).exists() else a.eval_train[0])
    for name in a.policies:
        task, start = POLICIES[name]
        d = out / f"policy_{name}"
        d.mkdir(exist_ok=True)
        cmd = [sys.executable, ROOT / "train_motion.py", "--csv", roll / "motion.csv", "--fps", 30, "--out", d,
               "--iters", a.iters_scratch if start is None and a.iters_scratch else a.iters, "--num-envs", a.num_envs,
               "--task", task, "--variants", out / "variants" / "variants.json", "--train-motion", out / "motion_variants.npz"]
        init = newest(Path(a.init_dir) / f"policy_{start}", flat=True) if start else None
        if start and init is None:
            raise SystemExit(f"{name} starts from {start}, but {a.init_dir}/policy_{start} has no model_*.pt")
        if init:
            cmd += ["--init", init]
        procs[name] = subprocess.Popen(list(map(str, cmd)), stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT)
        log(f"training {name} ({task}, from {init or 'scratch'}); log {d / 'train.log'}")
    while procs:
        for name, p in list(procs.items()):
            if p.poll() is not None:
                log(f"{name} finished with code {p.returncode}")
                del procs[name]
        mirror_checkpoints(out)
        if procs:
            time.sleep(20 if a.smoke else 60)


def newest(d, flat=False):
    c = sorted(Path(d).glob("model_*.pt" if flat else "logs/*/*/model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return c[-1] if c else None


def robust(args, env):
    """experiments/robustness_eval.py, knowing this file's policy names (latent_policy.TASKS doesn't)."""
    code = ("import sys; sys.path.insert(0, %r); from experiments import latent_policy as L, generalist_policy as G; "
            "L.TASKS.update({k: v[0] for k, v in G.POLICIES.items()}); import os, json; "
            "L.TASKS.update(json.loads(os.environ.get('A2H_GEN_TASKS', '{}'))); from experiments import robustness_eval as R; "
            "sys.argv = [R.__file__, *sys.argv[1:]]; R.main(R.parse())") % str(ROOT)
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], env=env).returncode


def evaluate(a, out, clips_dir):
    """robustness_eval per clip, every policy at once: the zero-shot clips with every suite (and the V-JEPA
    score against the clip's video), the trained ones on the clean clip and its held-out variants only."""
    for c in eval_clips(a):
        e = out / "evals" / c
        run = e / "run"
        if (e / "robust" / "robust.json").exists():
            continue
        for name in [*a.policies, *a.baselines]:
            ck = newest(out / f"policy_{name}") if name in a.policies else newest(Path(a.init_dir) / f"policy_{name}", flat=True)
            if ck is None:
                log(f"{name}: no checkpoint, left out of the eval")
                continue
            d = run / f"policy_{name}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(ck, d / ck.name)
            shutil.copy(e / "motion.npz", d / "motion.npz")
        shutil.copy(e / "task_spec.json", run / "task_spec.json")
        transfer = [f"{f.stem}={f}" for f in sorted((e / "transfer").glob("*.npz"))]
        cmd = ["--runs", run, "--out", e / "robust", "--target", c, "--reps", 4 if a.smoke else a.reps, "--transfer", *transfer]
        full = c in a.zero_shot and not a.short_eval
        if full:
            (run / "clips").mkdir(exist_ok=True)
            if not (run / "clips" / c).exists():
                (run / "clips" / c).symlink_to(clips_dir / c)
        elif c in a.zero_shot and a.cpu:  # no V-JEPA, no dynamics: the suites a CPU gets through
            cmd += ["--no-vjepa", "--push", 0.5, 1.0, "--init", 1, 2, "--dynamics", 0, "--speed", "--phase-push",
                    "--starts", "--no-noref", "--tempo", 0.5, 2.0]
        else:
            cmd += ["--no-vjepa", "--push", "--init", "--dynamics", 0, "--speed", "--phase-push", "--starts",
                    "--no-noref", "--tempo"]
        if a.smoke:
            cmd += ["--max-steps", 20, "--no-vjepa"] if full else ["--max-steps", 20]
        log(f"evaluating on {c} ({len(transfer)} held-out variants, {'every suite' if full else 'clean + variants'})")
        if robust(cmd, dict(os.environ)):
            log(f"{c}: robustness_eval failed")


METRICS = ("task_success", "fall_rate", "stages_frac", "survival", "joint_err_rad", "dtw_joint_rad")


def summarise(a, out, policies):
    """Per policy: on the zero-shot clips (the clip, its held-out variants, every suite) and the trained ones."""
    rows = {}
    for c in eval_clips(a):
        f = out / "evals" / c / "robust" / "robust.json"
        if f.exists():
            s = json.loads(f.read_text())["summary"]
            rows[c] = {p: {suite: {k: s[p][suite].get(k) for k in METRICS} for suite in s[p]} for p in policies if p in s}

    def mean(cs, p, suite, k):
        v = [rows[c][p][suite][k] for c in cs if p in rows.get(c, {}) and suite in rows[c][p]
             and rows[c][p][suite][k] is not None]
        return float(np.mean(v)) if v else None
    groups = {"zero-shot": [c for c in a.zero_shot if c in rows], "trained": [c for c in a.eval_train if c in rows]}
    table = {g: {p: {suite: {k: mean(cs, p, suite, k) for k in METRICS} for suite in ("clean", "transfer")}
                 for p in policies} for g, cs in groups.items()}
    res = {"level": a.level, "clips": rows, "groups": groups, "table": table}
    (out / "generalist.json").write_text(json.dumps(res, indent=1))
    pct = lambda x: "-" if x is None else f"{x:.0%}"  # noqa: E731
    f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    lines = [f"# Generalist, level {a.level}", "", f"Zero-shot: {', '.join(a.zero_shot)}. Trained real clips: "
             f"{', '.join(a.eval_train)}. Task success (fall rate), stages done, joint error (rad).", ""]
    for g, cs in groups.items():
        lines += [f"## {g.capitalize()} ({', '.join(cs)})", "",
                  "| Policy | The clip | Stages | Joint err | Held-out variants | Stages | Joint err |", "|---|---|---|---|---|---|---|"]
        for p in policies:
            t = table[g][p]
            lines.append(f"| {p} | " + " | ".join(f"{pct(t[s]['task_success'])} ({pct(t[s]['fall_rate'])}) | "
                                                f"{pct(t[s]['stages_frac'])} | {f3(t[s]['joint_err_rad'])}"
                                                for s in ("clean", "transfer")) + " |")
        lines.append("")
    lines += ["## Per clip (the clip / its held-out variants: task success, fall rate)", "",
              "| Clip | " + " | ".join(policies) + " |", "|---|" + "---|" * len(policies)]
    for c, r in rows.items():
        cell = lambda p: "-" if p not in r else " / ".join(  # noqa: E731
            f"{pct(r[p].get(s, {}).get('task_success'))} ({pct(r[p].get(s, {}).get('fall_rate'))})" for s in ("clean", "transfer"))
        lines.append(f"| {c}{' (zero-shot)' if c in a.zero_shot else ''} | " + " | ".join(cell(p) for p in policies) + " |")
    (out / "generalist_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def pack(out):
    with tarfile.open(out / "results.tar.gz", "w:gz") as tar:
        for f in ("generalist.json", "generalist_summary.md"):
            if (out / f).exists():
                tar.add(out / f, f)
        tar.add(out / "variants" / "variants.json", "variants/variants.json")
        for e in (out / "evals").glob("*/robust"):
            for f in ("robust.json", "robust_summary.md"):
                if (e / f).exists():
                    tar.add(e / f, f"evals/{e.parent.name}/robust/{f}")
        for d in out.glob("policy_*"):
            for f in ("policy.mp4", "policy_metrics.json", "train.log", "motion.npz", "weights.json"):
                if (d / f).exists():
                    tar.add(d / f, f"{d.name}/{f}")
            ck = newest(d)
            if ck:
                tar.add(ck, f"{d.name}/{ck.name}")


def run(a):
    clips_dir, out = Path(a.clips).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    P = Path(a.pool).resolve() if a.pool else clips_dir / "_pool"
    a.init_dir = a.init_dir or str(clips_dir.parent)
    if a.eval_train is None:  # the real clips that passed trackable()
        a.eval_train = json.loads((P / "pool.json").read_text())["real_train"]
    if a.smoke:
        a.eval_train, a.zero_shot = a.eval_train[:1], a.zero_shot[:1]
        a.iters, a.iters_scratch, a.num_envs = min(a.iters, 3), None, 64
    py = Path(os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python"))
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), WANDB_MODE="disabled")
    if not (out / "motion_variants.npz").exists():
        build_level(a, out, P, py, env)
    if a.build_only:
        return
    if a.policies:
        train_all(a, out, clips_dir)
    evaluate(a, out, clips_dir)
    summarise(a, out, [*a.baselines, *a.policies])
    pack(out)


def compare(a):
    """Checkpoints from several runs (NAME=BASE:model.pt; BASE, a POLICIES name or multi / multi_future, gives
    the task), evaluated side by side on the eval clips, e.g. on a CPU when the GPU job stopped before its eval."""
    from experiments.latent_policy import TASKS
    P, out = Path(a.clips).resolve() / "_pool", Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    a.eval_train = json.loads((P / "pool.json").read_text())["real_train"] if a.eval_train is None else a.eval_train
    a.level, a.smoke, a.short_eval, a.policies, a.baselines = a.label, False, False, [], []
    a.init_dir, tasks = str(out / "checkpoints"), {}
    for spec in a.policy:
        name, rest = spec.split("=", 1)
        base, ck = rest.split(":", 1)
        tasks[name] = POLICIES[base][0] if base in POLICIES else TASKS[base]
        d = out / "checkpoints" / f"policy_{name}"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / Path(ck).name).exists():
            shutil.copy(ck, d / Path(ck).name)
        a.baselines.append(name)
    evals = json.loads((P / "evals.json").read_text())
    py = Path(os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python"))
    to_npz(eval_jobs(out, P, evals, eval_clips(a)), py, dict(os.environ, WANDB_MODE="disabled"), out)
    copy_specs(out, P, evals, eval_clips(a))
    os.environ["A2H_GEN_TASKS"] = json.dumps(tasks)
    evaluate(a, out, Path(a.clips).resolve())
    summarise(a, out, a.baselines)


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("pool", help="1: the motion pool (local)")
    p.add_argument("--out", required=True)
    p.add_argument("--clips", nargs="+", required=True, help="processed clip folders (human.npy, meta.json, ...): "
                   "the real training clips and the zero-shot ones")
    p.add_argument("--recombinations", type=int, default=300, help="trackable distill.variation() motions to keep")
    p.add_argument("--multi", help="the 5-clip multi checkpoint (start and baseline)")
    p.add_argument("--multi-future", help="... and multi_future")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="3 bank motions, 4 recombinations, 2 variants each")
    e = sub.add_parser("compare", help="checkpoints of several runs, evaluated side by side (compare())")
    e.add_argument("clips", help="the pool's clips/ folder")
    e.add_argument("--out", required=True)
    e.add_argument("--policy", nargs="+", required=True, help="NAME=BASE:path/to/model_*.pt")
    e.add_argument("--label", "--level", default="compare")
    e.add_argument("--iters", type=int, help=argparse.SUPPRESS)  # the multiclip job's, unused
    e.add_argument("--eval-train", nargs="*", help="default: the pool's real training clips")
    e.add_argument("--zero-shot", nargs="*", default=list(ZERO_SHOT))
    e.add_argument("--reps", type=int, default=64)
    e.add_argument("--cpu", action="store_true", help="the zero-shot clips without V-JEPA or dynamics suites")
    c = sub.add_parser("convert", help=argparse.SUPPRESS)
    c.add_argument("jobs")
    return ap


def run_parser():
    """Step 2's flags (multiclip_policy.py's command line: <clips> --out --iters, plus these)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", help="the pool's clips/ folder (the multiclip job's inputs/clips)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", required=True, choices=list(LEVELS))
    ap.add_argument("--pool", help="default: <clips>/_pool")
    ap.add_argument("--init-dir", help="policy_<name>/model_*.pt to start from and compare with; default: <clips>/..")
    ap.add_argument("--policies", nargs="*", default=["gen_multi", "gen_future"], choices=list(POLICIES))
    ap.add_argument("--baselines", nargs="*", default=[], choices=["multi", "multi_future"],
                    help="also evaluate these from --init-dir, untrained here")
    ap.add_argument("--eval-train", nargs="*", help="trained clips to evaluate (clean + variants); default: the pool's")
    ap.add_argument("--zero-shot", nargs="*", default=list(ZERO_SHOT))
    ap.add_argument("--short-eval", action="store_true", help="zero-shot clips on the clean clip + variants only")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--iters-scratch", type=int, help="iterations for the policies trained from scratch (default --iters)")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--cpu", action="store_true", help="the zero-shot clips without V-JEPA or dynamics suites")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="6 variants, one eval clip of each kind, 3 iterations")
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--compare" in argv:  # through the multiclip job: <clips> --out --iters --level ... --compare --policy ...
        argv = ["compare", *[x for x in argv if x != "--compare"]]
    if argv and argv[0] in ("pool", "convert", "compare"):
        a = parser().parse_args(argv)
        return {"pool": pool, "compare": compare}[a.cmd](a) if a.cmd != "convert" else convert(a.jobs)
    return run(run_parser().parse_args(argv))


if __name__ == "__main__":
    main()
