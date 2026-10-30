"""Stage 4 of the audit: generate each clip's TRAIN variants (skill/variations.py) and check what the filter keeps
against physics: flights whose pelvis rise does not match their airtime (g T^2 / 8), hops lost to amplitude scaling,
joints faster than the retarget's own cap, feet in the floor, and why the dropped ones were dropped.

    python experiments/pipeline_audit/variants_audit.py                          # the stored qpos.npy
    python experiments/pipeline_audit/variants_audit.py --qpos-dir /tmp/fresh    # <clip>_qpos.npy from retarget_audit.py --save-qpos
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.pipeline_audit import RESULTS, clip_dirs, dump  # noqa: E402
from retarget import MAX_JOINT_SPEED  # noqa: E402
from skill import variations as V  # noqa: E402

LO, HI = V.BALLISTIC


def audit(qpos, fps, g1, grid=V.TRAIN, seed=0):
    vs = V.generate(qpos, fps, grid, seed, g1, log=lambda *a: None)
    base_pos = g1.bodies(qpos)
    foot0 = float(np.percentile(base_pos[:, [V.L_ANKLE, V.R_ANKLE], 2].min(1), 5))
    base_fl = V.flights(base_pos, foot0, fps)
    why, dropped, phys = {}, [], []
    for v in vs:
        if not v["ok"]:
            for rule in v.get("rules") or ["?"]:
                why[rule] = why.get(rule, 0) + 1
            dropped.append(dict(name=v["name"], why=v["why"]))
            continue
        q = v["qpos"].astype(np.float64)
        fl = V.flights(g1.bodies(q), foot0, fps)
        speed = np.abs(np.diff(q[:, 7:], axis=0)) * fps
        phys.append(dict(name=v["name"], frames=v["frames"], flights=len(fl), flight_ratios=[f["ratio"] for f in fl],
                         flights_unballistic=sum(1 for f in fl if not LO <= f["ratio"] <= HI),
                         max_joint_speed=round(float(speed.max()), 1), limit_clipped=v["limit_clipped"],
                         penetration_cm=round(-100 * float(min(g1.soles(q).min(), 0)), 1),
                         poses=sum(s["kind"] == "pose" for s in v["spec"]["stages"])))
    ratios = sorted(r for p in phys for r in p["flight_ratios"])
    return dict(kept=len(phys), dropped=len(vs) - len(phys), drop_reasons=why,
                base_flights=[{k: f[k] for k in ("air_s", "rise", "ratio")} for f in base_fl],
                base_max_joint_speed=round(float(np.abs(np.diff(qpos[:, 7:], axis=0)).max() * fps), 1),
                kept_flights=len(ratios), kept_flight_ratio_range=[ratios[0], ratios[-1]] if ratios else None,
                kept_flights_unballistic=sum(p["flights_unballistic"] for p in phys),
                kept_variants_lost_a_flight=sum(1 for p in phys if p["flights"] < len(base_fl)),
                kept_variants_over_cap=sum(p["max_joint_speed"] > MAX_JOINT_SPEED * V.SPEED_SLACK for p in phys),
                max_joint_speed_kept=max([p["max_joint_speed"] for p in phys] or [0]),
                max_penetration_kept_cm=max([p["penetration_cm"] for p in phys] or [0]),
                pose_count_spread=sorted({p["poses"] for p in phys}), per_variant=phys, dropped_variants=dropped)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--qpos-dir", help="take each clip's qpos from <dir>/<clip>_qpos.npy instead of its stored qpos.npy")
    ap.add_argument("--out", default=RESULTS / "variants_audit.json")
    a = ap.parse_args(argv)
    g1 = V.G1()
    rows = {}
    for name, d in clip_dirs(a.clips).items():
        fps = json.loads((d / "meta.json").read_text())["fps"]
        src = Path(a.qpos_dir) / f"{name}_qpos.npy" if a.qpos_dir else d / "qpos.npy"
        try:
            rows[name] = r = audit(np.load(src).astype(np.float64), fps, g1)
        except Exception as e:  # noqa: BLE001 - one clip's failure is itself a finding
            rows[name] = {"error": f"{type(e).__name__}: {e}"}
            print(name, "error:", e)
            continue
        print(f"{name:14s} kept {r['kept']:2d}/{r['kept'] + r['dropped']} drops {r['drop_reasons']} | base flights {[f['ratio'] for f in r['base_flights']]} | "
              f"kept flights {r['kept_flights']} ratios {r['kept_flight_ratio_range']} unballistic {r['kept_flights_unballistic']} "
              f"lost {r['kept_variants_lost_a_flight']} | max {r['max_joint_speed_kept']} rad/s, over {MAX_JOINT_SPEED:g} {r['kept_variants_over_cap']} | "
              f"pen {r['max_penetration_kept_cm']} cm | poses {r['pose_count_spread']}")
    print("->", dump(rows, a.out))


if __name__ == "__main__":
    main()
