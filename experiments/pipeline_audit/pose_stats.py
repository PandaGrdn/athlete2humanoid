"""Stage 1 of the audit: where MediaPipe's 3D is noisy, per clip, from human.npy (T, 33, 3) metres, z-up,
athlete facing +x: noise along the camera's depth axis vs the image plane, bone-length jitter and left/right
asymmetry, and the feet against the floor (hover, jitter, double-support disagreement)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.pipeline_audit import MAJOR, RESULTS, clip_dirs, dump, noise  # noqa: E402

BONES = {"thigh": [(23, 25), (24, 26)], "shin": [(25, 27), (26, 28)], "upper_arm": [(11, 13), (12, 14)],
         "forearm": [(13, 15), (14, 16)], "hip_width": [(23, 24)], "shoulder_width": [(11, 12)]}


def clip_stats(P, fps, view_azimuth_deg):
    az = np.radians(view_azimuth_deg)
    depth = np.array([np.cos(az), np.sin(az), 0.0])  # the camera's look direction in the athlete frame
    side = np.array([-np.sin(az), np.cos(az), 0.0])  # image horizontal
    up = np.array([0, 0, 1.0])
    n_depth, n_side, n_up = (noise(P[:, MAJOR] @ a, fps).mean() for a in (depth, side, up))
    bones = {}
    for b, pairs in BONES.items():
        L = [np.linalg.norm(P[:, i] - P[:, j], axis=-1) for i, j in pairs]
        cv = np.mean([np.std(x) / np.mean(x) for x in L])
        asym = abs(np.mean(L[0]) - np.mean(L[1])) / np.mean(L) if len(L) == 2 else 0.0
        rng = np.mean([(np.percentile(x, 95) - np.percentile(x, 5)) / np.mean(x) for x in L])
        bones[b] = dict(mean_cm=100 * float(np.mean(L)), cv_pct=100 * float(cv), p5p95_pct=100 * float(rng), asym_pct=100 * float(asym))
    ank = P[:, [27, 28], 2]
    floor = np.percentile(ank, 5, axis=0)
    low = ank.min(1) - floor.min()
    slow = np.linalg.norm(np.gradient(P[:, [27, 28]], axis=0), axis=-1) * fps < 0.6
    stance = slow & ((ank - floor) < 0.10)  # retarget.stance_mask's window, before its gap filling
    hover = (ank - floor)[stance]
    both = stance.all(1)
    lr = np.abs(ank[both, 0] - ank[both, 1])
    seg = lambda a, b: np.linalg.norm(P[:, a] - P[:, b], axis=-1)
    leg = np.median(np.concatenate([seg(23, 25) + seg(25, 27), seg(24, 26) + seg(26, 28)]))
    return dict(
        frames=len(P), fps=fps, view_azimuth_deg=round(float(view_azimuth_deg), 1), dur_s=round(len(P) / fps, 2),
        noise_cm=dict(depth=round(100 * n_depth, 2), image_horizontal=round(100 * n_side, 2), vertical=round(100 * n_up, 2),
                      depth_ratio=round(float(n_depth / max(n_side, 1e-6)), 2)),
        bones=bones, human_leg_cm=round(100 * leg, 1),
        feet=dict(frames_lower_ankle_gt2cm_pct=round(100 * float((low > 0.02).mean()), 1),
                  lower_ankle_max_cm=round(100 * float(low.max()), 1),
                  stance_hover_mean_cm=round(100 * float(hover.mean()), 2) if hover.size else None,
                  stance_hover_sd_cm=round(100 * float(hover.std()), 2) if hover.size else None,
                  double_support_lr_diff_mean_cm=round(100 * float(lr.mean()), 2) if lr.size else None,
                  double_support_lr_diff_max_cm=round(100 * float(lr.max()), 2) if lr.size else None,
                  stance_frac=round(float(stance.any(1).mean()), 2)),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*", help="clip names (default: every clip in CLIPS that exists)")
    ap.add_argument("--out", default=RESULTS / "pose_stats.json")
    a = ap.parse_args(argv)
    rows = {}
    for name, d in clip_dirs(a.clips).items():
        meta = json.loads((d / "meta.json").read_text())
        rows[name] = clip_stats(np.load(d / "human.npy"), meta["fps"], meta["view_azimuth"])
    print(f"{'clip':14s} {'depth':>6s} {'horiz':>6s} {'vert':>6s} {'ratio':>5s} | thigh shin uarm farm cv% | hover mean/sd | lr diff mean/max | ank>2cm%")
    for n, r in rows.items():
        b, f, nz = r["bones"], r["feet"], r["noise_cm"]
        print(f"{n:14s} {nz['depth']:6.2f} {nz['image_horizontal']:6.2f} {nz['vertical']:6.2f} {nz['depth_ratio']:5.2f} | "
              f"{b['thigh']['cv_pct']:5.1f} {b['shin']['cv_pct']:4.1f} {b['upper_arm']['cv_pct']:4.1f} {b['forearm']['cv_pct']:4.1f} | "
              f"{f['stance_hover_mean_cm']} / {f['stance_hover_sd_cm']} | {f['double_support_lr_diff_mean_cm']} / {f['double_support_lr_diff_max_cm']} | "
              f"{f['frames_lower_ankle_gt2cm_pct']}")
    print("->", dump(rows, a.out))


if __name__ == "__main__":
    main()
