"""Is a wrong move still rejected once a good tracker passes? A robustness_eval.py run's saved clean rollouts
(<out>/rollouts.npz with --score N: every env's body positions) re-scored offline with skill/task.py evaluate():
against their own clip's spec, against other clips' keypose specs (uncalibrated, and with every tolerance floor at its
cap, CALIB_MAX x TOL_MIN: the loosest a calibration can make them), and with one arm held still.

    python experiments/pipeline_audit/wrong_move.py --run outputs/robust_punches --others arms front_kick jumping_jacks
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import render  # noqa: E402
from experiments.pipeline_audit import RESULTS, clip_dirs, dump  # noqa: E402
from latent.distill import body_positions  # noqa: E402
from skill import keypose as K  # noqa: E402
from skill import task as T  # noqa: E402

STEP_HZ = 50.0
L_ARM, R_ARM = [9, 10], [12, 13]  # elbow, wrist (latent/reward_net.py BODIES)


def clean_rollouts(run, condition=0):
    """{policy: [(T, B, 3) body positions]} of the run's condition (0: the clean clip)."""
    r = np.load(Path(run) / "rollouts.npz")
    out = {}
    for k in r.files:
        parts = k.split("/")
        if len(parts) == 4 and parts[0] == str(condition) and parts[3] == "body":
            out.setdefault(parts[1], []).append(r[k])
    return out


def frozen_arm(pos, arm):
    """The arm's elbow and wrist kept where they were at the start, relative to the pelvis."""
    out = pos.copy()
    out[:, arm] = pos[:, [0]] + (pos[0, arm] - pos[0, [0]])
    return out


def rate(spec, rolls, fps=STEP_HZ):
    return round(float(np.mean([T.evaluate(spec, p, fps)["success"] for p in rolls])), 3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="a robustness_eval.py output folder (rollouts.npz, task_spec*.json)")
    ap.add_argument("--others", nargs="*", default=["arms", "front_kick", "jumping_jacks", "shot_put", "discus"],
                    help="clips (experiments/pipeline_audit CLIPS) whose specs the rollouts must fail")
    ap.add_argument("--out", default=RESULTS / "wrong_move.json")
    a = ap.parse_args(argv)
    run = Path(a.run)
    spec = json.loads(((run / "task_spec_calibrated.json") if (run / "task_spec_calibrated.json").exists()
                       else Path(json.loads((run / "mjlab_eval.json").read_text())["meta"]["task_spec"]["source"])).read_text())
    rolls = clean_rollouts(run)
    loosest = K.calibration(np.full(K.D, 1.0))  # every floor at its cap
    model = render.load_g1()[0]
    others = {}
    for name, d in clip_dirs(a.others).items():
        fps = json.loads((d / "meta.json").read_text())["fps"]
        others[name] = K.compile_keypose_spec(body_positions(model, np.load(d / "qpos.npy")), fps, name=name)
    rows = {}
    for pol, rs in rolls.items():
        rows[pol] = row = {"rollouts": len(rs), "own_spec": rate(spec, rs),
                           "left_arm_still": rate(spec, [frozen_arm(p, L_ARM) for p in rs]),
                           "right_arm_still": rate(spec, [frozen_arm(p, R_ARM) for p in rs]),
                           "other_specs": {n: rate(s, rs) for n, s in others.items()},
                           "other_specs_floors_at_cap": {n: rate(K.calibrate(s, loosest), rs) for n, s in others.items()}}
        print(f"{pol:18s} own {row['own_spec']:.0%}  left arm still {row['left_arm_still']:.0%}  right arm still "
              f"{row['right_arm_still']:.0%}  other specs " + ", ".join(f"{n} {x:.0%}" for n, x in row["other_specs"].items())
              + " | floors at cap " + ", ".join(f"{n} {x:.0%}" for n, x in row["other_specs_floors_at_cap"].items()))
    print("->", dump(rows, a.out))


if __name__ == "__main__":
    main()
