"""Which predicate breaks a keypose spec under small perturbations: the pose tolerances or the holds. Noise goes on
the limbs only; the pelvis, torso and foot heights stay exact, as a simulated robot's are (the upright gate and the
fall rule read the 4.4 cm pelvis->torso_link vector, so noise there reads as a fall, which is not what is being
tested). Each spec is scored as is, with holds off, with tolerances x2 and x3, and both."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import render  # noqa: E402
from experiments.pipeline_audit import RESULTS, clip_dirs, dump  # noqa: E402
from experiments.pipeline_audit.spec_audit import smooth_noise  # noqa: E402
from latent.distill import body_positions  # noqa: E402
from skill import keypose as K  # noqa: E402
from skill import task as T  # noqa: E402

LIMBS = [b for b in range(14) if b not in (K.PELVIS, K.TORSO)]


def variant(spec, hold_scale=1.0, tol_scale=1.0):
    s = json.loads(json.dumps(spec))
    for st in s["stages"]:
        if st["kind"] == "pose":
            if st.get("hold_s"):
                st["hold_s"] = round(st["hold_s"] * hold_scale, 3)
            st["tol"] = [t * tol_scale for t in st["tol"]]
    return s


def done(spec, pos, fps):
    return T.evaluate({**spec, "down": spec["down_ref"], "air": spec["air_ref"]}, pos, fps)["stages_completed"]


def audit(pos, fps, name, sigmas=(2.0, 4.0, 6.0), trials=8, seed=1):
    spec = K.compile_keypose_spec(pos, fps, name=name)
    n_st = len(spec["stages"])
    variants = {"as_is": spec, "no_hold": variant(spec, hold_scale=0.0), "tol_x2": variant(spec, tol_scale=2.0),
                "tol_x3": variant(spec, tol_scale=3.0), "no_hold_tol_x2": variant(spec, 0.0, 2.0)}
    rng = np.random.default_rng(seed)
    out = {"stages": n_st}
    for sigma in sigmas:
        res = {k: [] for k in variants}
        for _ in range(trials):
            p2 = smooth_noise(pos, sigma, fps, rng, tau_s=0.15, bodies=LIMBS)
            p2[:, [K.L_ANKLE, K.R_ANKLE], 2] = pos[:, [K.L_ANKLE, K.R_ANKLE], 2]  # feet keep the reference's heights
            for k, s in variants.items():
                res[k].append(done(s, p2, fps))
        out[f"{sigma:g}cm"] = {k: dict(stages_done=round(float(np.mean(v)), 1), pass_rate=round(float(np.mean(np.array(v) >= n_st)), 2))
                               for k, v in res.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--out", default=RESULTS / "spec_noise.json")
    a = ap.parse_args(argv)
    model = render.load_g1()[0]
    rows = {}
    for name, d in clip_dirs(a.clips).items():
        fps = json.loads((d / "meta.json").read_text())["fps"]
        pos = body_positions(model, np.load(d / "qpos.npy"))
        rows[name] = r = audit(pos, fps, name)
        print(f"{name:14s} " + " | ".join(f"{k}: " + ", ".join(f"{v}={x['pass_rate']:.0%}" for v, x in r[k].items()) for k in r if k != "stages"))
    print("->", dump(rows, a.out))


if __name__ == "__main__":
    main()
