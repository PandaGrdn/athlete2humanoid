"""Stage 3 of the audit: each clip's keypose spec from its stored qpos.npy (what the policies trained on): stages,
required moments, tolerances against the retarget's own noise, hold and dwell times, the self-check under the
reference's and the robot's thresholds, pass rate and key-moment stability when the whole body is perturbed by
temporally correlated noise, and the reference played at 0.7x and 1.4x."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import render  # noqa: E402
from experiments.pipeline_audit import RESULTS, clip_dirs, dump  # noqa: E402
from latent.distill import body_positions  # noqa: E402
from skill import keypose as K  # noqa: E402
from skill import task as T  # noqa: E402


def ev(spec, pos, fps, robot_thresholds=False):
    s = spec if robot_thresholds else {**spec, "down": spec["down_ref"], "air": spec["air_ref"]}
    r = T.evaluate(s, pos, fps)
    return r["stages_completed"], r["stages_total"], r["success"]


def smooth_noise(pos, sigma_cm, fps, rng, tau_s=0.1, bodies=None):
    """Temporally correlated noise (tau_s) of sigma_cm per coordinate, on all bodies or the given ones."""
    n = gaussian_filter1d(rng.normal(size=pos.shape), tau_s * fps, axis=0)
    n *= sigma_cm / 100 / n.std()
    if bodies is not None:
        keep = np.zeros(pos.shape[1], bool)
        keep[list(bodies)] = True
        n[:, ~keep] = 0
    return pos + n


def resample(pos, speed):
    idx = np.clip(np.arange(0, len(pos) - 1, speed), 0, len(pos) - 1)
    i0 = np.floor(idx).astype(int)
    w = (idx - i0)[:, None, None]
    return pos[i0] * (1 - w) + pos[np.minimum(i0 + 1, len(pos) - 1)] * w


def audit(pos, fps, name, sigmas=(1.0, 2.0, 3.0), trials=6, seed=0):
    spec = K.compile_keypose_spec(pos, fps, name=name)
    sc, sc_robot = ev(spec, pos, fps), ev(spec, pos, fps, robot_thresholds=True)
    poses = [s for s in spec["stages"] if s["kind"] == "pose"]
    F = K.trajectory_features(pos, spec["foot0"]).numpy()
    sig = K.noise_sigma(F, fps)
    feat = spec["features"]
    tol_sig, wrist_tol = [], []
    for s in poses:
        for i, (t, r) in enumerate(zip(s["tol"], s["relevance"])):
            if r > 0:
                tol_sig.append(t / max(sig[i], 1e-4))
                if "wrist" in feat[i]:
                    wrist_tol.append(t)
    rng = np.random.default_rng(seed)
    stab = {}
    for sigma in sigmas:
        passes, done, matched, count = [], [], [], []
        for _ in range(trials):
            p2 = smooth_noise(pos, sigma, fps, rng)
            c, n, ok = ev(spec, p2, fps)
            passes.append(ok)
            done.append(c)
            try:
                k2 = [s["frame"] for s in K.compile_keypose_spec(p2, fps, name=name)["stages"] if s["kind"] == "pose"]
                k1 = [s["frame"] for s in poses]
                matched.append(sum(any(abs(a - b) <= 0.15 * fps for b in k2) for a in k1) / max(len(k1), 1))
                count.append(len(k2) / max(len(k1), 1))
            except ValueError:
                matched.append(0.0)
                count.append(0.0)
        stab[f"{sigma:g}cm"] = dict(ref_passes=round(float(np.mean(passes)), 2), stages_done_mean=round(float(np.mean(done)), 1),
                                  keymoments_matched=round(float(np.mean(matched)), 2), keymoment_count_ratio=round(float(np.mean(count)), 2))
    tempo = {f"x{sp}": ev(spec, resample(pos, sp), fps)[0] for sp in (0.7, 1.4)}
    return dict(
        summary=spec["summary"], stages=len(spec["stages"]), poses=len(poses), required=sum(not s.get("optional") for s in spec["stages"]),
        flights=sum(s["kind"] == "flight" for s in spec["stages"]),
        rdp_eps_cm=round(100 * spec["rdp_eps"], 1), down_ref_cm=round(100 * spec["down_ref"], 1),
        self_check=f"{sc[0]}/{sc[1]}", self_check_robot_thresholds=f"{sc_robot[0]}/{sc_robot[1]}",
        noise_sigma_cm=dict(pelvis=round(100 * float(sig[0]), 2), wrist=round(100 * float(np.mean(sig[7:13])), 2),
                            foot_h=round(100 * float(np.mean(sig[13:15])), 2), ankle_xy=round(100 * float(np.mean(sig[4:8])), 2)),
        tol_over_sigma=dict(min=round(float(np.min(tol_sig)), 1), median=round(float(np.median(tol_sig)), 1)) if tol_sig else None,
        wrist_tol_cm=dict(min=round(100 * min(wrist_tol), 1), median=round(100 * float(np.median(wrist_tol)), 1), n=len(wrist_tol)) if wrist_tol else None,
        holds_s=[s["hold_s"] for s in poses], ref_dwell_s=[s["ref_dwell_s"] for s in poses],
        levels=[s["level"] for s in poses], optional=[bool(s.get("optional")) for s in poses],
        noise_stability=stab, tempo_stages_done=tempo, names=[s["name"] for s in spec["stages"]],
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--out", default=RESULTS / "spec_audit.json")
    a = ap.parse_args(argv)
    model = render.load_g1()[0]
    rows = {}
    for name, d in clip_dirs(a.clips).items():
        fps = json.loads((d / "meta.json").read_text())["fps"]
        pos = body_positions(model, np.load(d / "qpos.npy"))
        try:
            rows[name] = r = audit(pos, fps, name)
        except ValueError as e:
            rows[name] = {"error": str(e)}
            print(name, "error:", e)
            continue
        print(f"{name:14s} self-check {r['self_check']:>6s} poses {r['poses']:2d} required {r['required']:2d} eps {r['rdp_eps_cm']} cm "
              f"min tol/sigma {r['tol_over_sigma']} wrist tol {r['wrist_tol_cm']} tempo {r['tempo_stages_done']}")
        print("    noise:", json.dumps(r["noise_stability"]))
    print("->", dump(rows, a.out))


if __name__ == "__main__":
    main()
