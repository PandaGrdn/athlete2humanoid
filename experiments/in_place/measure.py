"""Run in place (retarget.py in_place): which way of removing a run's travel leaves the feet planted? For each clip
folder (human.npy, meta.json), retarget it travelling and with each of retarget.IN_PLACE_METHODS, then measure on the
G1 (MuJoCo forward kinematics of the solved qpos, not the targets):
  - foot slip: each foot's horizontal speed on the frames the retarget plants it (its contacts, found on the
    travelling motion, so the same frames for every method) and its sole is down (lowest contact sphere within 2 cm
    of the floor), mean and 95th percentile, cm/s;
  - root travel left: the pelvis's net horizontal travel, and its largest distance from where it starts;
  - the pipeline audit's IK numbers (experiments/pipeline_audit/retarget_audit.audit): tracking error, penetration,
    joint speed.

    python experiments/in_place/measure.py outputs/remote_process_pro/clips/hurdles --out docs/results/in_place.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import retarget  # noqa: E402
from experiments.pipeline_audit.retarget_audit import audit, sole_heights  # noqa: E402

DOWN = 0.02  # m: a sole this close to the floor is down


def slip(model, qpos, fps, stance):
    """{mean, p95} cm/s of a planted, down foot's horizontal speed, and the share of frames with one."""
    import mujoco
    data = mujoco.MjData(model)
    xy = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        xy.append([data.xpos[model.body(f"{s}_ankle_roll_link").id][:2].copy() for s in ("left", "right")])
    xy = np.array(xy)
    sole = sole_heights(model, qpos)
    sole_down = (sole[1:] < DOWN) & (sole[:-1] < DOWN)
    down = sole_down & stance[1:] & stance[:-1]
    v = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * fps * 100
    s, lowest = v[down], v[np.arange(len(v)), sole[1:].argmin(1)][sole_down.any(1)]
    r = lambda x, f: round(float(f(x)), 1) if x.size else None
    return {"mean_cm_s": r(s, np.mean), "p95_cm_s": r(s, lambda x: np.percentile(x, 95)),
            "foot_down_frac": round(float(down.any(1).mean()), 3),
            # whatever the contacts say: the lowest sole's speed whenever a sole is down
            "lowest_sole_median_cm_s": r(lowest, np.median)}


def measure(clip, methods=(None, *retarget.IN_PLACE_METHODS)):
    P = np.load(clip / "human.npy")
    fps = json.loads((clip / "meta.json").read_text())["fps"]
    rows = {}
    for m in methods:
        stance = retarget.proportional_motion(P, retarget.G1Retargeter(), fps, in_place=m)["stance"]
        rt = retarget.G1Retargeter()
        report = {}
        qpos = rt.retarget_sequence(*retarget.human_to_targets(P, rt, fps=fps, report=report, in_place=m), fps=fps)
        root = qpos[:, :2]
        a = audit(P, fps, in_place=m)
        rows[m or "travelling"] = {
            "slip": slip(rt.model, qpos, fps, stance),
            "root_travel_m": {"net": round(float(np.linalg.norm(root[-1] - root[0])), 3),
                              "max_from_start": round(float(np.linalg.norm(root - root[0], axis=-1).max()), 3)},
            "in_place": report.get("in_place"), "stance_frac": report.get("stance_frac"),
            "audit": {k: a[k] for k in ("tracking_cm", "joint_speed", "feet", "angle_err_deg")},
        }
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = {Path(c).name: measure(Path(c)) for c in a.clips}
    for name, rows in res.items():
        for m, r in rows.items():
            t, ang = r["audit"]["tracking_cm"], r["audit"]["angle_err_deg"]["mean"]
            print(f"{name:12s} {m:10s} slip {r['slip']['mean_cm_s']} cm/s (p95 {r['slip']['p95_cm_s']}), "
                  f"root net {r['root_travel_m']['net']} m (max {r['root_travel_m']['max_from_start']}), "
                  f"feet IK {t['left_foot']['mean']}/{t['right_foot']['mean']} cm, pelvis {t['pelvis']['mean']} cm, "
                  f"knee {ang['knee']} deg, lowest sole {r['slip']['lowest_sole_median_cm_s']} cm/s, stance {r['stance_frac']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
