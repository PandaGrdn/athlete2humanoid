"""V-JEPA 2 as a world model for the multi-clip tracker: does seeing the *imagined* future help, with
no future leaking in? docs/FRONTIER_WORLD_MODEL.md.

    python experiments/worldmodel_policy.py inputs/clips --out outputs/wm --stage latents
    python experiments/worldmodel_policy.py inputs/clips --out outputs/wm --latents outputs/wm/wm \\
        --pca-vjc inputs/pca_vjc.npz --pca-vj inputs/pca_vj.npz --init inputs/policy_tracking_var/model_1499.pt \\
        --init-dir inputs --iters 2000

Stage "latents" (GPU, V-JEPA 2): for every clip and its mirror image, per V-JEPA step (latent/world_model.py):
the trailing latent (what multi_vjc sees), the encoder's per-step tokens of that trailing window and the
predictor's imagined tokens of the next 15 steps (~1 s), plus the centred dense latent multi_vj sees.
Then how good the imagined future is (wm_quality.json / .md): V-JEPA's predictor, a ridge predictor
learned from the training clips' latents (cross-fitted, so a training clip is predicted by a model that
never saw it), and copying the present, against the true future latent.

Stage "train": the latents from stage 1 (--latents). One new policy, multi_vjp (A2H-TrackingVJPred),
on the same five clips' variants, from the same warm start and for as long as multi / multi_vj / multi_vjc
(experiments/multiclip_policy.py); its input is multi_vjc's (the causal latent now and 0.5 s ago) plus the
predicted latent ~0.5 s and 1 s ahead, made at each step from frames up to it only. Then every clip,
trained on or not, is evaluated as multiclip_policy.py does with all four policies. Each policy reads a
different latent, and robustness_eval holds one latent table per run, so a clip is evaluated in three
groups: {multi_vjp, multi} on the predicted table, {multi_vjc} on the causal one, {multi_vj} on the
non-causal one, each with its own run's PCA. The groups share the motions, the held-out variants, the
conditions and their seeds.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import multiclip_policy as MP  # noqa: E402
from experiments.latent_policy import TASKS, log, mirror_checkpoints  # noqa: E402
from latent import video_condition as VC  # noqa: E402
from latent import world_model as WM  # noqa: E402

TRAIN_CLIPS = MP.TRAIN_CLIPS
HELD_OUT_CLIPS = ("jumping_jacks", "punches", "discus")
PRED_POLICY = {"vjepa": "multi_vjp", "learned": "multi_vjl"}  # the policy that sees each predictor's future


# ---------------------------------------------------------------- stage 1: the latents

def latents_stage(a, clips_dir, out, names):
    import mediapy
    wm = WM.WorldModel()
    (out / "wm").mkdir(exist_ok=True)
    for c in names:
        f = out / "wm" / f"{c}.npz"
        if f.exists():
            continue
        sil = np.asarray(mediapy.read_video(str(clips_dir / c / "silhouette.mp4")))
        t0, res = time.time(), {}
        for tag, v in (("", sil), ("_m", np.ascontiguousarray(sil[:, :, ::-1]))):
            for k, x in wm.imagine(v).items():
                res[k + tag] = x
            res["dense" + tag] = wm.embed_dense(v)
        np.savez_compressed(f, **res)
        log(f"{c}: {len(res['raw'])} steps, imagined {res['pred'].shape[1]} steps ahead ({time.time() - t0:.0f} s)")
    report = quality_report(out / "wm", a.train, [n for n in names if n not in a.train])
    (out / "wm_quality.json").write_text(json.dumps(report, indent=1))
    (out / "wm_quality.md").write_text(quality_md(report))
    print(quality_md(report))


def load_latents(d, names):
    return {c: dict(np.load(Path(d) / f"{c}.npz")) for c in names}


def ln_pca(lat, train):
    """The PCA of the layer-normed trailing latents (the space the predictor predicts in), training clips only."""
    return VC.fit_pca([WM.trailing(lat[c][k]) for c in train for k in ("tub", "tub_m")])


def learned_predictions(z, train):
    """{(clip, mirror): (S, K)} -> {(clip, mirror): {h: (S, K)}}: LatentPredictor fitted on the training
    clips' latents (both mirror images), its ridge strength by leave-one-clip-out; cross-fitted: a training
    clip is predicted by a fit on the others."""
    fits, out = {}, {}
    for (c, m), zz in z.items():
        key = c if c in train else None
        if key not in fits:
            fits[key] = WM.LatentPredictor.select({t: [z[(t, "")], z[(t, "_m")]] for t in train if t != key})
        out[(c, m)] = {h: fits[key].predict(zz, h) for h in WM.HORIZONS}
    return out


def quality_report(wm_dir, train, held):
    """Imagined vs true future, per clip and horizon, in the PCA space of the layer-normed latents:
    V-JEPA 2's predictor, the learned ridge predictor (cross-fitted), and copying the present."""
    names = [*train, *held]
    lat = load_latents(wm_dir, names)
    pca = ln_pca(lat, train)
    z = {(c, m): VC.project(WM.trailing(lat[c]["tub" + m]), pca) for c in names for m in ("", "_m")}
    learned = learned_predictions(z, train)
    rows = {}
    for c in names:
        rows[c] = {}
        for h in WM.HORIZONS:
            parts = {"vjepa": [], "learned": [], "present": [], "actual": []}
            for m in ("", "_m"):
                L = lat[c]
                keep = slice(0, max(len(z[(c, m)]) - h, 1))  # the last h steps have no true future
                parts["present"].append(z[(c, m)][keep])
                parts["actual"].append(WM.shifted(z[(c, m)], h)[keep])
                parts["vjepa"].append(VC.project(WM.predicted_trailing(L["tub" + m], L["pred" + m], h), pca)[keep])
                parts["learned"].append(learned[(c, m)][h][keep])
            P = {k: np.concatenate(v) for k, v in parts.items()}
            rows[c][h] = {k: WM.quality(P["present"], P[k], P["actual"]) for k in ("vjepa", "learned")}
    groups = {"trained clips": list(train), "held-out clips": list(held)}
    table = {g: {h: {k: {m: float(np.mean([rows[c][h][k][m] for c in cs])) for m in ("cos_pred", "cos_copy", "mse_ratio", "cos_change")}
                     for k in ("vjepa", "learned")} for h in WM.HORIZONS} for g, cs in groups.items() if cs}
    return {"pca_explained": pca["explained"], "clips": rows, "table": table, "groups": groups,
            "horizons_s": {h: h * 2 / 30 for h in WM.HORIZONS}}


def quality_md(r):
    lines = ["# How well the future latent is imagined", "",
             "Cosine with the true future latent (PCA of V-JEPA's layer-normed latents, centred), against copying the present; "
             "error: squared error relative to the copy's (< 1: better than copying). Change: cosine between the predicted "
             "and the true change from now.", "",
             "| Clips | Ahead | Copy the present: cos | V-JEPA predictor: cos / error / change | Learned (ridge): cos / error / change |",
             "|---|---|---|---|---|"]
    for g, t in r["table"].items():
        for h, row in t.items():
            v, l_ = row["vjepa"], row["learned"]
            lines.append(f"| {g} | {r['horizons_s'][h]:.2f} s | {v['cos_copy']:.2f} | {v['cos_pred']:.2f} / {v['mse_ratio']:.2f} / "
                         f"{v['cos_change']:.2f} | {l_['cos_pred']:.2f} / {l_['mse_ratio']:.2f} / {l_['cos_change']:.2f} |")
    lines += ["", "Per clip (V-JEPA / learned error relative to the copy, 0.53 s and 1 s ahead):", ""]
    for c, row in r["clips"].items():
        lines.append(f"- {c}: " + "; ".join(f"{r['horizons_s'][h]:.2f} s {row[h]['vjepa']['mse_ratio']:.2f} / {row[h]['learned']['mse_ratio']:.2f}"
                                           for h in row))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- stage 2: the tables, training, eval

def tables(lat, train, pca_vjc, pca_vj, predictors, pca_ln=None):
    """name -> {"vjc": (table, mirrored), "vj": ..., <predictor>: ...}, per V-JEPA step, already projected.
    vjc: the causal latent (its run's PCA); vj: the centred one (its run's PCA); per predictor: [the causal
    latent | predicted ~0.5 s | ~1 s], the predictions made at each step from frames up to it ("vjepa":
    V-JEPA 2's predictor, in the PCA of its layer-normed latents; "learned": the ridge predictor in vjc's PCA,
    cross-fitted on the training clips). pca_ln: the run's own, when carrying on from it (a refit can flip
    a component's sign on another machine, and the policy was trained on the first fit's)."""
    pca_ln = pca_ln or ln_pca(lat, train)
    zc = {(c, m): VC.project(lat[c]["raw" + m], pca_vjc) for c in lat for m in ("", "_m")}
    learned = learned_predictions(zc, train) if "learned" in predictors else None
    out = {}
    for c, L in lat.items():
        out[c] = {"vjc": (zc[(c, "")], zc[(c, "_m")]),
                  "vj": tuple(VC.project(L["dense" + m], pca_vj) for m in ("", "_m"))}
        for p in predictors:
            t = []
            for m in ("", "_m"):
                if p == "vjepa":
                    pr = [VC.project(WM.predicted_trailing(L["tub" + m], L["pred" + m], h), pca_ln) for h in WM.HORIZONS]
                else:
                    pr = [learned[(c, m)][h] for h in WM.HORIZONS]
                t.append(np.concatenate([zc[(c, m)], *pr], 1).astype(np.float32))
            out[c][p] = tuple(t)
    return out, pca_ln


def group_evals(out, tabs, names, clips_dir, group, causal):
    """out/eval_<group>/evals/<clip>/: the build's motions and held-out variants, with this group's latent."""
    root = out / f"eval_{group}"
    for c in names:
        src, e = out / "evals" / c, root / "evals" / c
        (e / "transfer").mkdir(parents=True, exist_ok=True)
        for f in ("motion.npz", "task_spec.json"):
            if not (e / f).exists():
                (e / f).symlink_to(src / f)
        _, fps = MP.clip_qpos(clips_dir / c)
        n_frames = len(np.load(clips_dir / c / "qpos.npy"))
        n50 = len(np.load(src / "motion.npz")["joint_pos"])
        z = tabs[c][group]
        np.save(e / "motion.vjl.npy", VC.per_step(z[0], np.arange(n_frames, dtype=float), fps, n50, causal))
        for f in sorted((src / "transfer").glob("*.npz")):
            if not (e / "transfer" / f.name).exists():
                (e / "transfer" / f.name).symlink_to(f)
            sf = np.load(src / "held_out" / f.stem / "src_frame.npy")
            np.save((e / "transfer" / f.name).with_suffix(".vjl.npy"),
                    VC.per_step(z[int(f.stem.startswith("m_"))], sf, fps, len(np.load(f)["joint_pos"]), causal))
    return root


def variant_latents(out, tabs, key, clips_dir):
    """The training variants' latent file for table `key`, as multiclip_policy.build lays it out."""
    segs = json.loads((out / "variants" / "variants.json").read_text())["segments"]
    lats = []
    for s in segs:
        _, fps = MP.clip_qpos(clips_dir / s["clip"])
        sf = np.load(out / "variants" / s["name"] / "src_frame.npy")
        lats.append(VC.per_step(tabs[s["clip"]][key][int(s["params"]["mirror"])], sf, fps, s["length"], True))
    return np.concatenate(lats)


def train_policies(a, out, clips_dir, jobs):
    """jobs: {policy: (training latent file, rollout latent file)}, trained at once from the same warm start."""
    procs = {}
    for name, (vjl, roll) in jobs.items():
        d = out / f"policy_{name}"
        d.mkdir(exist_ok=True)
        cmd = [sys.executable, ROOT / "train_motion.py", "--csv", clips_dir / MP.ROLLOUT_CLIP / "motion.csv", "--fps", 30,
               "--out", d, "--iters", a.iters, "--num-envs", a.num_envs, "--task", TASKS[name],
               "--variants", out / "variants" / "variants.json", "--train-motion", out / "motion_variants.npz",
               "--video-latent", vjl, "--rollout-video-latent", roll]
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


def merge(out, names, roots):
    """Every group's robust.json per clip -> out/merged/evals/<clip>/robust/robust.json (one summary)."""
    for c in names:
        summ = {}
        for root in roots:
            f = root / "evals" / c / "robust" / "robust.json"
            if f.exists():
                summ.update(json.loads(f.read_text())["summary"])
        d = out / "merged" / "evals" / c / "robust"
        d.mkdir(parents=True, exist_ok=True)
        (d / "robust.json").write_text(json.dumps({"summary": summ}, indent=1))


def pack(out):
    with tarfile.open(out / "results.tar.gz", "w:gz") as tar:
        for f in ["wm_quality.json", "wm_quality.md", "wm_pca.npz", "merged/multiclip.json", "merged/multiclip_summary.md",
                  "variants/variants.json"]:
            if (out / f).exists():
                tar.add(out / f, f)
        for f in out.glob("wm/*.npz"):
            tar.add(f, str(f.relative_to(out)))
        for f in [*out.glob("evals/*/robust/robust*"), *out.glob("eval_*/evals/*/robust/robust*")]:
            if f.suffix in (".json", ".md"):
                tar.add(f, str(f.relative_to(out)))
        for d in out.glob("policy_*"):
            for f in ["policy.mp4", "policy_metrics.json", "policy_qpos.npy", "train.log", "motion.npz", "weights.json"]:
                if (d / f).exists():
                    tar.add(d / f, f"{d.name}/{f}")
            ck = MP.newest(d)
            if ck:
                tar.add(ck, f"{d.name}/{ck.name}")


def train_stage(a, clips_dir, out, train, held):
    lat = load_latents(a.latents, [*train, *held])
    preds = list(a.predictors)
    kept = out / "wm_pca.npz"
    kept = {k[3:]: v for k, v in np.load(kept).items() if k.startswith("ln_")} if kept.exists() else None
    tabs, pca_ln = tables(lat, train, dict(np.load(a.pca_vjc)), dict(np.load(a.pca_vj)), preds, kept)
    np.savez(out / "wm_pca.npz", **{f"ln_{k}": v for k, v in pca_ln.items()}, horizons=np.array(WM.HORIZONS))
    py = Path(os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python"))
    env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), WANDB_MODE="disabled")
    first = preds[0]
    if not (out / "motion_variants.npz").exists():
        # multiclip_policy.build, fed the first predictor's table (already projected: an identity "PCA")
        MP.latents = lambda clips_dir, names, smoke, causal=False: {c: tabs[c][first] for c in names}
        width = tabs[train[0]][first][0].shape[1]
        MP.build(clips_dir, out, py, env, train, held, a.seed, a.smoke,
                 {"mean": np.zeros(width, np.float32), "components": np.eye(width, dtype=np.float32), "explained": 1.0},
                 causal=True)
    assert np.allclose(variant_latents(out, tabs, first, clips_dir), np.load(out / "motion_variants.vjl.npy"), atol=1e-5)
    names = [*train, *held]
    roots, jobs = {first: out}, {}
    for p in preds:
        if p != first:
            roots[p] = group_evals(out, tabs, names, clips_dir, p, causal=True)
            np.save(out / f"motion_variants.{p}.vjl.npy", variant_latents(out, tabs, p, clips_dir))
        jobs[PRED_POLICY[p]] = (out / ("motion_variants.vjl.npy" if p == first else f"motion_variants.{p}.vjl.npy"),
                                roots[p] / "evals" / MP.ROLLOUT_CLIP / "motion.vjl.npy")
    if a.build_only:
        return
    todo = {k: v for k, v in jobs.items() if a.retrain or not MP.newest(out / f"policy_{k}")}
    if todo:
        train_policies(a, out, clips_dir, todo)
    groups = [(roots[p], [PRED_POLICY[p]], ["multi"] if p == first else []) for p in preds]
    groups += [(group_evals(out, tabs, names, clips_dir, "vjc", causal=True), [], ["multi_vjc"]),
               (group_evals(out, tabs, names, clips_dir, "vj", causal=False), [], ["multi_vj"])]
    for root, pols, compare in groups:
        for name in pols:  # trained in out/, evaluated from this group's root
            if root != out and not (root / f"policy_{name}").exists():
                (root / f"policy_{name}").symlink_to(out / f"policy_{name}")
        ns = SimpleNamespace(policies=pols, compare=compare, init_dir=a.init_dir, reps=a.reps, smoke=a.smoke,
                             full_eval=False)
        MP.evaluate(ns, root, names, clips_dir)
    merge(out, names, [g[0] for g in groups])
    MP.summarise(out / "merged", train, held, ["multi", "multi_vjc", *[PRED_POLICY[p] for p in preds], "multi_vj"])


def main(a):
    clips_dir, out = Path(a.clips).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train, held = (a.train[:2], a.held_out[:1]) if a.smoke else (a.train, a.held_out)
    a.train = train
    if a.stage in ("latents", "all"):
        latents_stage(a, clips_dir, out, [*train, *held])
        a.latents = a.latents or out / "wm"
    if a.stage in ("train", "all"):
        train_stage(a, clips_dir, out, train, held)
    pack(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", help="folder of processed clips (qpos.npy, meta.json, motion.csv, silhouette.mp4)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["latents", "train", "all"], default="train")
    ap.add_argument("--latents", help="stage 1's wm/ folder (<clip>.npz)")
    ap.add_argument("--predictors", nargs="+", choices=list(PRED_POLICY), default=["vjepa", "learned"],
                    help="one policy per predictor (multi_vjp, multi_vjl), trained at once")
    ap.add_argument("--pca-vjc", help="multi_vjc's run's vjl_pca.npz (causal latents)")
    ap.add_argument("--pca-vj", help="multi_vj's run's vjl_pca.npz (centred latents)")
    ap.add_argument("--train", nargs="+", default=list(TRAIN_CLIPS))
    ap.add_argument("--held-out", nargs="*", default=list(HELD_OUT_CLIPS))
    ap.add_argument("--init", help="the checkpoint multi_vjp starts from (the others' warm start: tracking_var)")
    ap.add_argument("--init-dir", help="policy_{multi,multi_vj,multi_vjc}/model_*.pt, evaluated next to it")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 clips, 2 variants, a few iterations")
    main(ap.parse_args())
