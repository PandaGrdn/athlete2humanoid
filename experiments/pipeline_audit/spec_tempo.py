"""Fix 2's acceptance check (docs/PIPELINE_ACCURACY.md, "Fixes applied"): every clip's keypose spec must pass its own
reference played at 0.7x and 1.4x as well as at 1x, and the upright gate must not bind on the reference. Also the
older stand/squat/jump spec (skill/task.py compile_spec, what v1 runs are scored with) on the same references.
For each abandoned stage, the feature that kept it from passing (skill/task.py abandoned_stage)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import render  # noqa: E402
from experiments.pipeline_audit import RESULTS, clip_dirs, dump  # noqa: E402
from experiments.pipeline_audit.spec_audit import resample  # noqa: E402
from latent.distill import body_positions  # noqa: E402
from skill import keypose as K  # noqa: E402
from skill import task as T  # noqa: E402

SPEEDS = (0.7, 1.0, 1.4)


def run(spec, pos, fps):
    r = T.evaluate({**spec, "down": spec["down_ref"], "air": spec["air_ref"]}, pos, fps)
    return {"done": f"{r['stages_completed']}/{r['stages_total']}", "pass": r["success"], "abandoned": r["abandoned"]}


def audit(pos, fps, name):
    out = {}
    up = T.features(pos, 0.0)[3].numpy()
    out["lean_max_deg"] = round(float(np.degrees(np.arccos(np.clip(up, -1, 1))).max()), 1)
    out["gate_min"] = round(float(T.upright_gate(T.features(pos, 0.0)[3]).min()), 3)
    spec = K.compile_keypose_spec(pos, fps, name=name)
    out["keypose"] = {f"x{s:g}": run(spec, resample(pos, s), fps) for s in SPEEDS}
    out["holds_s"] = [s["hold_s"] for s in spec["stages"] if s["kind"] == "pose"]
    try:
        st = T.compile_spec(pos, fps, name=name)
        out["stage_spec"] = {f"x{s:g}": run(st, resample(pos, s), fps) for s in SPEEDS}
    except ValueError as e:
        out["stage_spec"] = {"error": str(e)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--out", default=RESULTS / "spec_tempo.json")
    a = ap.parse_args(argv)
    model = render.load_g1()[0]
    rows = {}
    for name, d in clip_dirs(a.clips).items():
        fps = json.loads((d / "meta.json").read_text())["fps"]
        rows[name] = r = audit(body_positions(model, np.load(d / "qpos.npy")), fps, name)
        kp = r["keypose"]
        st = r["stage_spec"]
        print(f"{name:14s} lean max {r['lean_max_deg']:4.0f} deg gate min {r['gate_min']:.2f} | keypose "
              + " ".join(f"{k} {v['done']}{'' if v['pass'] else ' FAIL'}" for k, v in kp.items())
              + " | stage spec " + (" ".join(f"{k} {v['done']}{'' if v['pass'] else ' FAIL'}" for k, v in st.items())
                                    if "error" not in st else st["error"]))
        for k, v in kp.items():
            for ab in v["abandoned"]:
                print(f"    keypose {k}: {ab}")
    print("->", dump(rows, a.out))


if __name__ == "__main__":
    main()
