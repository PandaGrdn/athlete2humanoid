"""Video -> latent -> policy, end to end, and the comparison that says whether it worked.

    python experiments/latent_policy.py clips/*.mov --target squat --out outputs/latent_squat

1. every clip: SAM 3.1 -> pose -> retarget, and its V-JEPA silhouette latent (cached per clip;
   all clips together give the human-domain mean)
2. distillation dataset at the target video's viewpoint: bank + retargets + random variations,
   rendered and embedded with V-JEPA
3. reward network (robot body trajectory -> V-JEPA latent), bundled with the target's latent
4. three policies on the target clip, in parallel: tracking (copy the retarget), hybrid (latent
   reward + weak tracking), latent (latent reward only)
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
         "latent": "A2H-Latent-Flat-Unitree-G1"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def humans(clips, vj, smoke, out):
    """name -> {dense, az, qpos, csv, sil} for every clip, plus the human-domain mean."""
    from latent.video_latent import video_latent
    if smoke:  # robot squat silhouettes stand in for a human video
        g1 = render.load_g1()[0]
        res = {}
        for name, q in (("squat", motion_bank.squat(0.25, 2)), ("march", motion_bank.march(0.15))):
            d = out / "smoke_clips" / name
            d.mkdir(parents=True, exist_ok=True)
            csv = retarget.export_mjlab_csv(q, d / "motion.csv")
            res[name] = {"dense": vj.embed_dense(silhouette.robot_silhouettes(g1, q, azimuth=-150)),
                         "az": -150.0, "qpos": q, "csv": csv, "dir": d}
    else:
        res = {}
        for d in map(_processed, clips):
            video_latent(d, vj)
            lat = np.load(d / "video_latent.npz")
            res[d.name] = {"dense": lat["dense"], "az": float(lat["view_azimuth"]), "qpos": np.load(d / "qpos.npy"),
                           "csv": d / "motion.csv", "dir": d}
    h_mean = np.concatenate([h["dense"] for h in res.values()]).mean(0)
    return res, h_mean


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

    # 1. clips -> latents
    hum, h_mean = humans(a.clips, vj, a.smoke, out)
    tgt = hum[a.target]
    bucket = int(round(tgt["az"] / AZ_BUCKET) * AZ_BUCKET)
    log(f"target {a.target}: viewpoint {tgt['az']:.0f} deg (renders at {bucket}), {len(tgt['dense'])} latent steps")

    # 2. distillation dataset
    ds_f = out / "distill_dataset.npz"
    if not ds_f.exists():
        bank = bank_motions(config.OUTPUT_DIR / "cache")
        if a.smoke:
            bank = dict(list(bank.items())[:4])
        base = {f"bank/{k}": q for k, q in bank.items()} | {f"clip/{k}": h["qpos"] for k, h in hum.items()}
        rng = np.random.default_rng(0)
        motions = base | {f"var/{i}": distill.variation(base, rng) for i in range(a.variations)}
        log(f"distillation dataset: {len(motions)} motions ({sum(len(q) for q in motions.values())} frames)")
        np.savez(ds_f, **distill.build_dataset(motions, bucket, vj, workers=a.workers))
    ds = np.load(ds_f)
    names, seq, Y = ds["names"], ds["seq"], ds["Y"]
    is_bank = np.char.startswith(names.astype(str), "bank/")
    r_mean = Y[is_bank[seq]].astype(np.float32).mean(0)
    bank_dense = {str(n)[5:]: Y[seq == i].astype(np.float32) for i, n in enumerate(names) if is_bank[i]}

    # 3. reward network
    bundle = out / "reward.pt"
    if not bundle.exists():
        log("training the reward network")
        net, U, rep = distill.train_reward_net(ds["X"], Y, seq, r_mean, k=a.k, epochs=a.epochs)
        target = distill.video_target(tgt["dense"], h_mean, U)
        pred = distill.predict(net, distill.body_positions(g1, tgt["qpos"]))
        n = min(len(pred), len(target))
        cos = (unit(pred[:n]) * unit(target[:n])).sum(-1)
        cos_ref = uniform_filter1d(np.pad(cos, (0, len(target) - n), mode="edge"), 5)
        # Does the 64-d space keep what the probe found? Rank of the target's own retarget by DTW.
        own_i = list(names.astype(str)).index(f"clip/{a.target}")
        cand = {k: (v - r_mean) @ U for k, v in bank_dense.items()} | {"(own retarget)": (Y[seq == own_i] - r_mean) @ U}
        ranked = sorted(cand, key=lambda k: -dtw(unit(target), unit(cand[k])))
        rep |= {"cos_ref_mean": float(cos_ref.mean()), "pca_own_retarget_dtw_rank": ranked.index("(own retarget)") + 1,
                "pca_dtw_top3": ranked[:3]}
        distill.save_bundle(bundle, net, U, r_mean, h_mean, target, cos_ref, 30.0, rep, bucket)
        (out / "distill_report.json").write_text(json.dumps(rep, indent=1))
    log("reward bundle ready: " + (out / "distill_report.json").read_text().replace("\n", ""))

    # 4. policies, in parallel (the GPU has room for three mjlab trainings)
    import torch
    torch.cuda.empty_cache()
    procs = {}
    for name, task in TASKS.items():
        d = out / f"policy_{name}"
        if (d / "policy_metrics.json").exists():
            continue
        d.mkdir(exist_ok=True)
        cmd = [sys.executable, ROOT / "train_motion.py", "--csv", tgt["csv"], "--fps", 30, "--out", d,
               "--iters", a.iters, "--num-envs", a.num_envs, "--task", task, "--bundle", bundle]
        procs[name] = subprocess.Popen(list(map(str, cmd)), stdout=open(d / "train.log", "w"), stderr=subprocess.STDOUT)
        log(f"training {name} ({task}), log: {d / 'train.log'}")
    for name, p in procs.items():
        rc = p.wait()
        log(f"{name} finished with code {rc}")

    # 5. evaluation with the real V-JEPA
    from latent.reward_net import load_bundle
    net, target_k, b = load_bundle(bundle)
    U = b["basis"]
    H = unit(tgt["dense"] - h_mean)
    cands = {"retarget": tgt["qpos"]}
    for name in TASKS:
        f = out / f"policy_{name}" / "policy_qpos.npy"
        if f.exists():
            cands[name] = to_30fps(np.load(f))
    results, sils = {}, {}
    bank_c = {k: unit(v - r_mean) for k, v in bank_dense.items()}
    for name, q in cands.items():
        sil = silhouette.robot_silhouettes(g1, q, azimuth=bucket)
        dense = vj.embed_dense(sil)
        R = unit(dense - r_mean)
        pred = distill.predict(net, distill.body_positions(g1, q))
        real_k = (dense - r_mean) @ U
        others = {k: dtw(H, v) for k, v in bank_c.items()}
        score = dtw(H, R)
        res = {"aligned": aligned(H, R), "dtw": float(score), "chamfer": float(chamfer(H, R)),
               "dtw_rank_vs_bank": 1 + int(sum(v > score for v in others.values())),
               "reward_net_aligned": aligned(target_k.numpy(), pred), "real_aligned_in_reward_space": aligned(target_k.numpy(), real_k)}
        m = out / f"policy_{name}" / "policy_metrics.json"
        if m.exists():
            res |= {k: v for k, v in json.loads(m.read_text()).items() if k != "checkpoint"}
        results[name] = res
        sils[name] = sil
        log(f"{name:9s}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()))
    best_bank = sorted(((dtw(H, v), k) for k, v in bank_c.items()), reverse=True)[:5]
    results["_bank_top5_dtw"] = [(k, round(v, 3)) for v, k in best_bank]
    (out / "eval.json").write_text(json.dumps(results, indent=1))
    comparison_video(tgt, sils, out / "comparison.mp4")
    pack(out, hum)
    log(f"done: {out / 'eval.json'}")


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
        for f in ["eval.json", "distill_report.json", "reward.pt", "comparison.mp4"]:
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
    ap.add_argument("--variations", type=int, default=220)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.variations, a.epochs, a.iters, a.num_envs, a.target = 6, 3, 3, 64, "squat"
    main(a)
