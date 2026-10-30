"""Stage 4 of the audit: generate each clip's TRAIN variants (skill/variations.py) and check what the filter keeps
against physics: flights whose pelvis rise does not match their airtime (g T^2 / 8), hops lost to amplitude scaling,
joints faster than the retarget's own cap, and why the dropped ones were dropped."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.pipeline_audit import RESULTS, clip_dirs, dump  # noqa: E402
from retarget import GRAVITY, MAX_JOINT_SPEED  # noqa: E402
from skill import task as T  # noqa: E402
from skill import variations as V  # noqa: E402


def flights(pos, foot0, fps):
    """Flights (both feet > 10 cm off the floor for at least MIN_FLIGHT): airtime, pelvis rise and the ballistic ratio."""
    lo = np.minimum(pos[:, V.L_ANKLE, 2], pos[:, V.R_ANKLE, 2]) - foot0
    air = lo > 0.10
    edges = np.flatnonzero(np.diff(np.r_[0, air.astype(int), 0]))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        Tair = (b - a) / fps
        if Tair < T.MIN_FLIGHT:
            continue
        z = pos[:, 0, 2]
        rise = z[a:b].max() - z[[max(a - 1, 0), min(b, len(z) - 1)]].mean()
        ballistic = GRAVITY * Tair ** 2 / 8
        out.append(dict(air_s=round(Tair, 2), rise_cm=round(100 * rise, 1), ballistic_cm=round(100 * ballistic, 1), ratio=round(float(rise / ballistic), 2)))
    return out


def audit(qpos, fps, g1, grid=V.TRAIN, seed=0):
    vs = V.generate(qpos, fps, grid, seed, g1, log=lambda *a: None)
    base_pos = g1.bodies(qpos)
    foot0 = float(np.percentile(base_pos[:, [V.L_ANKLE, V.R_ANKLE], 2].min(1), 5))
    base_fl = flights(base_pos, foot0, fps)
    why, phys = {}, []
    for v in vs:
        if not v["ok"]:
            key = "self-check" if "completes" in v["why"] else ("limits" if "limit" in v["why"] else v["why"][:30])
            why[key] = why.get(key, 0) + 1
            continue
        q = v["qpos"].astype(np.float64)
        fl = flights(g1.bodies(q), foot0, fps)
        speed = np.abs(np.diff(q[:, 7:], axis=0)) * fps
        phys.append(dict(name=v["name"], flights=len(fl), flight_ratios=[f["ratio"] for f in fl],
                         flights_unballistic=sum(1 for f in fl if not 0.6 <= f["ratio"] <= 1.5),
                         max_joint_speed=round(float(speed.max()), 1), limit_clipped=v["limit_clipped"],
                         poses=sum(s["kind"] == "pose" for s in v["spec"]["stages"])))
    return dict(kept=len(phys), dropped=len(vs) - len(phys), drop_reasons=why, base_flights=base_fl,
                kept_flights=sum(p["flights"] for p in phys), kept_flights_unballistic=sum(p["flights_unballistic"] for p in phys),
                kept_variants_lost_a_flight=sum(1 for p in phys if p["flights"] < len(base_fl)),
                kept_variants_over_cap=sum(p["max_joint_speed"] > MAX_JOINT_SPEED for p in phys),
                max_joint_speed_kept=max([p["max_joint_speed"] for p in phys] or [0]),
                pose_count_spread=sorted({p["poses"] for p in phys}), per_variant=phys)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--out", default=RESULTS / "variants_audit.json")
    a = ap.parse_args(argv)
    g1 = V.G1()
    rows = {}
    for name, d in clip_dirs(a.clips).items():
        fps = json.loads((d / "meta.json").read_text())["fps"]
        try:
            rows[name] = r = audit(np.load(d / "qpos.npy").astype(np.float64), fps, g1)
        except Exception as e:  # noqa: BLE001 - one clip's failure is itself a finding
            rows[name] = {"error": f"{type(e).__name__}: {e}"}
            print(name, "error:", e)
            continue
        print(f"{name:14s} kept {r['kept']:2d}/{r['kept'] + r['dropped']} drops {r['drop_reasons']} | base flights {[f['ratio'] for f in r['base_flights']]} | "
              f"kept flights {r['kept_flights']} unballistic {r['kept_flights_unballistic']} lost {r['kept_variants_lost_a_flight']} | "
              f"over {MAX_JOINT_SPEED:g} rad/s {r['kept_variants_over_cap']} | poses {r['pose_count_spread']}")
    print("->", dump(rows, a.out))


if __name__ == "__main__":
    main()
