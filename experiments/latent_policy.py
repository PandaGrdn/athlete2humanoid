"""Video -> latent -> policy, end to end, and the comparison that says whether it worked.

    python experiments/latent_policy.py clips/*.mov --target squat --out outputs/latent_squat

1. every clip: SAM 3.1 -> pose -> retarget, and its V-JEPA silhouette latent (cached per clip;
   all clips together give the human-domain mean)
2. distillation dataset at the target video's viewpoint: bank + retargets + earlier policies'
   rollouts (--extra-motions, DAgger-style) + random variations, rendered and embedded with V-JEPA
3. reward network (robot body trajectory -> V-JEPA latent), bundled with the target's latent
4. policies on the target clip, in parallel: tracking (copy the retarget), hybrid (latent reward
   + weak tracking), latent (latent reward only), phase (latent reward only, and the policy sees
   only its own state and a clock, not the retarget); task / taskphase: task first, the semantic
   stage reward from skill/task.py's spec of the target (task_spec.json) leads, the latent is style
5. evaluation with the real V-JEPA (not the distilled network): each rollout re-rendered as a
   silhouette and compared with the video, time-aligned and with DTW, against the bank

Each stage is skipped when its output exists. --smoke runs every stage tiny, on a synthetic
target (a bank squat standing in for the video), to test the plumbing in minutes.
"""

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import render  # noqa: E402
import retarget  # noqa: E402
from experiments.cross_embodiment_probe import AZ_BUCKET, _processed, chamfer, dtw  # noqa: E402
from latent import distill, silhouette  # noqa: E402
from latent import bank as motion_bank  # noqa: E402
from latent.distill import unit  # noqa: E402

TASKS = {"tracking": "Mjlab-Tracking-Flat-Unitree-G1", "hybrid": "A2H-Hybrid-Flat-Unitree-G1",
         "latent": "A2H-Latent-Flat-Unitree-G1", "phase": "A2H-LatentPhase-Flat-Unitree-G1",
         "task": "A2H-Task-Flat-Unitree-G1", "taskphase": "A2H-TaskPhase-Flat-Unitree-G1"}
TASK_FIRST = ("task", "taskphase")  # trained on the semantic task spec (skill/task.py)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def humans(clips, vj, smoke, out):
    """name -> {dense, centred, az, qpos, csv, dir} for every clip, plus the human-domain means
    of both embeddings (dense: evaluation, as in earlier runs; centred: the reward's target)."""
    from latent.video_latent import video_latent
    if smoke:  # robot squat silhouettes stand in for a human video
        g1 = render.load_g1()[0]
        res = {}
        for name, q in (("squat", motion_bank.squat(0.25, 2)), ("march", motion_bank.march(0.15))):
            d = out / "smoke_clips" / name
            d.mkdir(parents=True, exist_ok=True)
            csv = retarget.export_mjlab_csv(q, d / "motion.csv")
            sil = silhouette.robot_silhouettes(g1, q, azimuth=-150)
            res[name] = {"dense": vj.embed_dense(sil), "centred": vj.embed_centred(sil)[0],
                         "az": -150.0, "qpos": q, "csv": csv, "dir": d}
    else:
        import mediapy
        res = {}
        for d in map(_processed, clips):
            video_latent(d, vj)
            lat = np.load(d / "video_latent.npz")
            cf = out / "human_centred" / f"{d.name}.npy"
            if not cf.exists():
                cf.parent.mkdir(exist_ok=True)
                np.save(cf, vj.embed_centred(np.asarray(mediapy.read_video(str(d / "silhouette.mp4"))))[0])
            res[d.name] = {"dense": lat["dense"], "centred": np.load(cf), "az": float(lat["view_azimuth"]),
                           "qpos": np.load(d / "qpos.npy"), "csv": d / "motion.csv", "dir": d}
    h_mean = np.concatenate([h["dense"] for h in res.values()]).mean(0)
    h_mean_c = np.concatenate([h["centred"] for h in res.values()]).mean(0)
    return res, h_mean, h_mean_c


def bank_motions(cache):
    f = cache / "bank_qpos.npz"
    if f.exists():
        return dict(np.load(f))
    bank = motion_bank.motion_bank()
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, **bank)
    return bank


def to_30fps(q50):
    return distill.resample(q50, int(round((len(q50) - 1) * 30 / 50)) + 1)


def aligned(H, R):
    n = min(len(H), len(R))
    return float((unit(H[:n]) * unit(R[:n])).sum(-1).mean())


def main(a):
    import config
    from latent.vjepa import VJepa
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vj = VJepa()
    g1 = render.load_g1()[0]

    # Earlier policies' rollouts (50 Hz qpos), named after their folder: policy_latent -> latent.
    extra = {Path(f).parent.name.removeprefix("policy_"): to_30fps(np.load(f)) for f in a.extra_motions}

    # 1. clips -> latents
    hum, h_mean, h_mean_c = humans(a.clips, vj, a.smoke, out)
    tgt = hum[a.target]
    bucket = int(round(tgt["az"] / AZ_BUCKET) * AZ_BUCKET)
    log(f"target {a.target}: viewpoint {tgt['az']:.0f} deg (renders at {bucket}), {len(tgt['dense'])} latent steps")

    # What the action is, read off the target's retarget (G1-sized) and Muse's skill spec if any.
    from skill import task as skill_task
    spec_f = out / "task_spec.json"
    if not spec_f.exists():
        sk = tgt["dir"] / "skill_spec.json"
        spec = skill_task.compile_spec(distill.body_positions(g1, tgt["qpos"]), 30.0,
                                       json.loads(sk.read_text()) if sk.exists() else None,
                                       a.max_clearance, name=a.target)
        spec_f.write_text(json.dumps(spec, indent=1))
    spec = json.loads(spec_f.read_text())
    log(f"task spec: {spec['summary']}" + "".join(f"; warning: {w}" for w in spec["warnings"]))

    bundle = out / "reward.pt"
    if a.reward and not bundle.exists():  # an earlier run's reward network: skips stages 2 and 3
        import shutil
        shutil.copy(a.reward, bundle)
        if (Path(a.reward).parent / "distill_report.json").exists():
            shutil.copy(Path(a.reward).parent / "distill_report.json", out / "distill_report.json")
        retarget_bundle(bundle, tgt, g1, bucket, a.target)

    # 2. distillation dataset
    ds_f = out / "distill_dataset.npz"
    if not ds_f.exists() and not bundle.exists():
        bank = bank_motions(config.OUTPUT_DIR / "cache")
        if a.smoke:
            bank = dict(list(bank.items())[:4])
        base = {f"bank/{k}": q for k, q in bank.items()} | {f"clip/{k}": h["qpos"] for k, h in hum.items()}
        base |= {f"dagger/{k}": q for k, q in extra.items()}
        rng = np.random.default_rng(0)
        motions = base | {f"var/{i}": distill.variation(base, rng) for i in range(a.variations)}
        log(f"distillation dataset: {len(motions)} motions ({sum(len(q) for q in motions.values())} frames)")
        np.savez(ds_f, **distill.build_dataset(motions, bucket, vj, workers=a.workers, every=a.every))
    # 3. reward network
    if not bundle.exists():
        ds = np.load(ds_f)
        names, seq, Y = ds["names"], ds["seq"], ds["Y"]
        is_bank = np.char.startswith(names.astype(str), "bank/")
        r_mean = Y[is_bank[seq]].astype(np.float32).mean(0)
        bank_c = {str(n)[5:]: Y[seq == i].astype(np.float32) for i, n in enumerate(names) if is_bank[i]}
        log("training the reward network")
        net, U, rep = distill.train_reward_net(ds["X"], Y, seq, r_mean, k=a.k, epochs=a.epochs,
                                               hidden=a.hidden, depth=a.depth)
        target = distill.video_target(tgt["centred"], h_mean_c, U)
        pred = distill.predict(net, distill.body_positions(g1, tgt["qpos"]))
        n = min(len(pred), len(target))
        cos = (unit(pred[:n]) * unit(target[:n])).sum(-1)
        cos_ref = uniform_filter1d(np.pad(cos, (0, len(target) - n), mode="edge"), 5)
        # Does the 64-d space keep what the probe found? Rank of the target's own retarget by DTW.
        own_i = list(names.astype(str)).index(f"clip/{a.target}")
        cand = {k: (v - r_mean) @ U for k, v in bank_c.items()} | {"(own retarget)": (Y[seq == own_i] - r_mean) @ U}
        ranked = sorted(cand, key=lambda k: -dtw(unit(target), unit(cand[k])))
        rep |= {"cos_ref_mean": float(cos_ref.mean()), "pca_own_retarget_dtw_rank": ranked.index("(own retarget)") + 1,
                "pca_dtw_top3": ranked[:3]}
        distill.save_bundle(bundle, net, U, r_mean, h_mean_c, target, cos_ref, 30.0, rep, bucket)
        (out / "distill_report.json").write_text(json.dumps(rep, indent=1))
    log("reward bundle ready: " + (out / "distill_report.json").read_text().replace("\n", ""))

    # 4. policies, in parallel (the GPU has room for three mjlab trainings)
    import torch
    torch.cuda.empty_cache()
    procs = {}
    for name in a.policies:
        task = TASKS[name]
        d = out / f"policy_{name}"
        if (d / "policy_metrics.json").exists():
            continue
        d.mkdir(exist_ok=True)
        seed_checkpoint(d, name, a.resume_from)
        cmd = [sys.executable, ROOT / "train_motion.py", "--csv", tgt["csv"], "--fps", 30, "--out", d,
               "--iters", a.iters, "--num-envs", a.num_envs, "--task", task, "--bundle", bundle]
        init = init_checkpoint(name, a.init)
        if init:
            cmd += ["--init", init]
        if name in TASK_FIRST:
            cmd += ["--task-spec", spec_f] + (["--importance", Path(a.importance).resolve()] if a.importance else [])
        if a.reward_config and name != "tracking":
            cmd += ["--reward-config", Path(a.reward_config).resolve()]
        procs[name] = subprocess.Popen(list(map(str, cmd)), stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT)
        log(f"training {name} ({task}), log: {d / 'train.log'}")
    for name, p in procs.items():
        rc = p.wait()
        log(f"{name} finished with code {rc}")

    # 5. evaluation with the real V-JEPA, in its dense embedding (comparable across runs), plus how
    # the reward network and the real centred embedding score each rollout in the reward's space
    from latent.reward_net import load_bundle
    net, target_k, b = load_bundle(bundle)
    U, r_mean = b["basis"], b["r_mean"]
    H = unit(tgt["dense"] - h_mean)
    bank_d = eval_bank(bank_motions(config.OUTPUT_DIR / "cache"), bucket, vj, g1, out, a.smoke)
    r_mean_d = np.concatenate(list(bank_d.values())).mean(0)
    cands = {"retarget": tgt["qpos"]} | {f"prev_{k}": q for k, q in extra.items()}
    for name in TASKS:
        f = out / f"policy_{name}" / "policy_qpos.npy"
        if f.exists():
            cands[name] = to_30fps(np.load(f))
    results, sils = {}, {}
    bank_n = {k: unit(v - r_mean_d) for k, v in bank_d.items()}
    for name, q in cands.items():
        sil = silhouette.robot_silhouettes(g1, q, azimuth=bucket)
        dense = vj.embed_dense(sil)
        R = unit(dense - r_mean_d)
        pred = distill.predict(net, distill.body_positions(g1, q))
        real_k = (vj.embed_centred(sil)[0] - r_mean) @ U
        others = {k: dtw(H, v) for k, v in bank_n.items()}
        score = dtw(H, R)
        res = {"aligned": aligned(H, R), "dtw": float(score), "chamfer": float(chamfer(H, R)),
               "dtw_rank_vs_bank": 1 + int(sum(v > score for v in others.values())),
               "reward_net_aligned": aligned(target_k.numpy(), pred), "real_aligned_in_reward_space": aligned(target_k.numpy(), real_k)}
        sem = skill_task.evaluate(spec, distill.body_positions(g1, q), 30.0)
        res |= {"task_success": sem["success"], "task_stages": f"{sem['stages_completed']}/{sem['stages_total']}",
                "task_jump_quality": sem["jump_quality"], "task_fell_at_s": sem["fell_at_s"]}
        m = out / f"policy_{name}" / "policy_metrics.json"
        if m.exists():
            res |= {k: v for k, v in json.loads(m.read_text()).items() if k != "checkpoint"}
        results[name] = res
        if not name.startswith("prev_"):
            sils[name] = sil
        log(f"{name:9s}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()))
    best_bank = sorted(((dtw(H, v), k) for k, v in bank_n.items()), reverse=True)[:5]
    results["_bank_top5_dtw"] = [(k, round(v, 3)) for v, k in best_bank]
    (out / "eval.json").write_text(json.dumps(results, indent=1))
    comparison_video(tgt, sils, out / "comparison.mp4")
    pack(out, hum)
    log(f"done: {out / 'eval.json'}")


def eval_bank(bank, bucket, vj, g1, out, smoke):
    """The bank's dense V-JEPA embeddings at the target's viewpoint (the evaluation's reference)."""
    f = out / "eval_bank_dense.npz"
    if not f.exists():
        if smoke:
            bank = dict(list(bank.items())[:4])
        np.savez(f, **{k: vj.embed_dense(silhouette.robot_silhouettes(g1, q, azimuth=bucket)) for k, q in bank.items()})
    return dict(np.load(f))


# A task-first policy can build off these when no weights of its own name are given.
INIT_FALLBACK = {"task": ("hybrid", "tracking"), "taskphase": ("phase",)}


def _newest(d):
    c = sorted(Path(d).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return c[-1] if c else None


def init_checkpoint(name, specs):
    """--init entries (name=checkpoint, or a run folder with policy_<name>/model_*.pt) -> the
    checkpoint this policy starts from, or None. policy_weights.py adapts other tasks' weights."""
    for s in specs:
        if "=" in s:
            k, v = s.split("=", 1)
            if k == name:
                return Path(v)
    for s in (s for s in specs if "=" not in s):
        for src in (name, *INIT_FALLBACK.get(name, ())):
            c = _newest(Path(s) / f"policy_{src}")
            if c:
                return c
    return None


def seed_checkpoint(d, name, resume_from):
    """--resume-from DIR: put DIR/policy_<name>'s newest checkpoint where train_motion.py looks for
    an interrupted attempt, so it continues from there (same task, same motion)."""
    c = resume_from and _newest(Path(resume_from) / f"policy_{name}")
    if c and not any(d.glob("logs/*/*/model_*.pt")):
        dst = d / "logs" / "resumed" / "from" / c.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(c, dst)
        log(f"{name}: resuming from {c}")


def retarget_bundle(bundle, tgt, g1, bucket, name):
    """Point a reused reward bundle at this run's target: its network maps robot motion into the
    latent space of every clip rendered at its viewpoint, so only the target latent and cos_ref
    (how well this target's retarget scores) change."""
    from latent.reward_net import load_bundle
    net, _, b = load_bundle(bundle)
    net.kwargs = b["net_kwargs"]  # kept by save_bundle, so the network reloads with its own shape
    if int(round(b["azimuth"])) != bucket:
        raise SystemExit(f"--reward was distilled at {b['azimuth']:.0f} deg; {name} renders at {bucket}")
    target = distill.video_target(tgt["centred"], b["h_mean"], b["basis"])
    pred = distill.predict(net, distill.body_positions(g1, tgt["qpos"]))
    n = min(len(pred), len(target))
    cos = (unit(pred[:n]) * unit(target[:n])).sum(-1)
    cos_ref = uniform_filter1d(np.pad(cos, (0, len(target) - n), mode="edge"), 5)
    rep = dict(b["report"], target=name, cos_ref_mean=float(cos_ref.mean()))
    distill.save_bundle(bundle, net, b["basis"], b["r_mean"], b["h_mean"], target, cos_ref,
                        b["video_fps"], rep, b["azimuth"])
    log(f"reward bundle retargeted to {name}: cos_ref_mean {cos_ref.mean():.3f}")


def comparison_video(tgt, sils, path):
    """Video silhouette next to each robot's silhouette, same framing V-JEPA saw."""
    import cv2
    import mediapy
    human = tgt["dir"] / "silhouette.mp4"
    panels = {"video": np.asarray(mediapy.read_video(str(human)))} if human.exists() else {}
    panels |= sils
    n = min(len(v) for v in panels.values())
    frames = []
    for t in range(n):
        row = []
        for name, v in panels.items():
            f = np.ascontiguousarray(v[t])
            cv2.putText(f, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            row.append(f)
        frames.append(np.concatenate(row, 1))
    mediapy.write_video(str(path), np.stack(frames), fps=30)


def pack(out, hum):
    """One small archive with everything worth keeping (pulled back to the laptop)."""
    with tarfile.open(out / "results.tar.gz", "w:gz") as tar:
        for f in ["eval.json", "distill_report.json", "reward.pt", "comparison.mp4", "task_spec.json"]:
            if (out / f).exists():
                tar.add(out / f, f)
        for d in out.glob("policy_*"):
            for f in ["policy.mp4", "policy_metrics.json", "policy_qpos.npy", "train.log", "motion.npz"]:
                if (d / f).exists():
                    tar.add(d / f, f"{d.name}/{f}")
            ckpts = sorted(d.glob("logs/*/*/model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            if ckpts:
                tar.add(ckpts[-1], f"{d.name}/{ckpts[-1].name}")
        for name, h in hum.items():  # per-clip results, so later runs can skip SAM
            for f in ["video_latent.npz", "qpos.npy", "meta.json", "motion.csv", "retarget_report.json",
                      "silhouette.mp4", "side_by_side.mp4", "human.npy"]:
                if (h["dir"] / f).exists():
                    tar.add(h["dir"] / f, f"clips/{name}/{f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--target", default="squat")
    ap.add_argument("--out", default="outputs/latent_policy")
    ap.add_argument("--variations", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--every", type=int, default=2, help="distil at every n-th dense step")
    ap.add_argument("--extra-motions", nargs="*", default=[],
                    help="earlier policies' policy_qpos.npy (50 Hz), added to the distillation data")
    ap.add_argument("--reward", help="reuse this reward bundle (reward.pt) instead of distilling one")
    ap.add_argument("--policies", nargs="+", default=["tracking", "hybrid", "latent"], choices=list(TASKS))
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--reward-config", help="hybrid/latent reward settings, e.g. muse/reward_agent.py's "
                                            "next_reward_config.json (its iterations override --iters)")
    ap.add_argument("--importance", help="importance.json (per reference frame goal relevance) weighting the "
                                         "task-first policies' style terms")
    ap.add_argument("--init", nargs="*", default=[],
                    help="build policies off saved weights: name=model_*.pt, or a run folder (policy_<name>/"
                         "model_*.pt; task <- hybrid/tracking, taskphase <- phase if it has none of its own)")
    ap.add_argument("--resume-from", help="a run folder whose policy_<name>/model_*.pt checkpoints these policies "
                                          "continue from (same task and motion); --iters is the new total")
    ap.add_argument("--max-clearance", type=float, default=0.30, help="most jump clearance (m) asked of the G1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.variations, a.epochs, a.iters, a.num_envs, a.target = 6, 3, 3, 64, "squat"
    main(a)
