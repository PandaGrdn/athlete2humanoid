"""Fix 3 of the audit: the front-end A/B. The stored MediaPipe human.npy against SAM 3D Body (frontends/sam3d_body.py)
on the same 30 fps frames, scored with this audit's own metrics:

- before the retarget (pose_stats.clip_stats): bone-length CV, double-support ankle disagreement, stance hover,
  depth-to-image noise ratio (after the shared smoothing; the raw, pre-smoothing ratio too);
- after it, with the current retarget.py (retarget_audit.audit): knee IK error, limit hits, cap hits, root shift,
  phantom-flight decisions.

    python experiments/pipeline_audit/frontend_ab.py      # every clip with outputs/frontend_ab/sam3d/<clip>.npz (frontends/sam3d_modal.py)

Front ends: "mediapipe" (the stored human.npy), "sam3d_camera" (SAM 3D Body's keypoints + its predicted camera
translation: skeleton and root from the model), "sam3d_image" (SAM 3D Body's skeleton with MediaPipe's root rule:
the 2D hip track, depth motion zero). The decision rule (docs/PIPELINE_ACCURACY.md, fix 3): adopt a front end if
it halves the double-support ankle disagreement and the knee IK error on shot_put and hurdles without worsening
squat_jump.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.pipeline_audit import MAJOR, RESULTS, ROOT, clip_dirs, dump, noise  # noqa: E402
from experiments.pipeline_audit.pose_stats import clip_stats  # noqa: E402
from experiments.pipeline_audit.retarget_audit import audit  # noqa: E402
from frontends import sam3d_body  # noqa: E402

CLIPS = ("squat_jump", "shot_put", "hurdles")
ATHLETIC = ("shot_put", "hurdles")
SAM3D = ROOT / "outputs" / "frontend_ab" / "sam3d"


def raw_noise(cam, fps):
    """Noise (cm) along the camera axes (x right, y down, z depth) before any smoothing, of points (T, J, 3), and
    the depth / image-plane ratio."""
    n = noise(cam.reshape(len(cam), -1), fps).reshape(cam.shape[1], 3).mean(0)
    return dict(ratio=round(float(n[2] / n[:2].mean()), 2), **{a: round(100 * float(n[i]), 2) for i, a in enumerate("xyz")})


def soles(P, fps):
    """Double support on the lowest point of each foot (ankle, heel or toe) instead of the ankle: a lifted heel
    (a throw's rear foot, a sprinter's toe-off) is not a disagreement there. Same window as pose_stats."""
    ank = P[:, [27, 28]]
    sole = np.stack([P[:, [27, 29, 31], 2].min(1), P[:, [28, 30, 32], 2].min(1)], 1)
    floor = np.percentile(sole, 5, axis=0)
    slow = np.linalg.norm(np.gradient(ank, axis=0), axis=-1) * fps < 0.6
    both = (slow & ((sole - floor) < 0.10)).all(1)
    lr = np.abs(sole[both, 0] - sole[both, 1])
    floor_a = np.percentile(ank[..., 2], 5, axis=0)
    both_a = (slow & ((ank[..., 2] - floor_a) < 0.10)).all(1)
    return dict(double_support_frames_ankle=int(both_a.sum()), double_support_frames_sole=int(both.sum()),
                double_support_sole_lr_cm=round(100 * float(lr.mean()), 2) if lr.size else None)


def trajectories(name, d):
    """{front end: (P (T, 33, 3), view azimuth, extra)}."""
    meta = json.loads((d / "meta.json").read_text())
    out = {"mediapipe": (np.load(d / "human.npy"), meta["view_azimuth"], {})}
    f = SAM3D / f"{name}.npz"
    if f.exists():
        s = sam3d_body.load(f)
        kp = s["kp3d"][:, sam3d_body.MHR70_TO_MP33]
        # hip-centred, as raw_pose.py measured MediaPipe's world landmarks; the root (camera translation) apart
        extra = dict(raw_noise_hip_centred_cm=raw_noise(kp[:, MAJOR] - kp[:, [23, 24]].mean(1, keepdims=True), meta["fps"]),
                     raw_noise_root_cm=raw_noise(s["cam_t"][:, None], meta["fps"]), focal_px=round(float(s["focal"])),
                     box_found=round(float((s["box_scores"] > 0).mean()), 3))
        for root in ("camera", "image"):
            P, az = sam3d_body.human_trajectory(s, root=root, return_view=True)
            if len(P) != len(out["mediapipe"][0]):
                raise SystemExit(f"{name}: SAM 3D Body has {len(P)} frames, human.npy {len(out['mediapipe'][0])}")
            out[f"sam3d_{root}"] = (P, az, extra)
    return out, meta["fps"]


def summary(stats, ret):
    f, tr, c = stats["feet"], ret["tracking_cm"], ret["contacts"]
    return dict(
        bone_cv_pct={b: round(stats["bones"][b]["cv_pct"], 1) for b in ("thigh", "shin", "upper_arm", "forearm")},
        double_support_lr_cm=f["double_support_lr_diff_mean_cm"], double_support_lr_max_cm=f["double_support_lr_diff_max_cm"],
        stance_hover_cm=f["stance_hover_mean_cm"], stance_hover_sd_cm=f["stance_hover_sd_cm"],
        depth_ratio=stats["noise_cm"]["depth_ratio"], noise_cm=stats["noise_cm"],
        knee_ik_cm=[tr["left_knee"]["mean"], tr["right_knee"]["mean"]], knee_ik_mean_cm=round((tr["left_knee"]["mean"] + tr["right_knee"]["mean"]) / 2, 2),
        limits_frac=ret["limits"]["frames_any"], limits_worst=ret["limits"]["worst"], cap_frames=ret["joint_speed"]["frames_any_joint_at_cap"],
        root_shift_cm=c["root_shift_cm"], no_contact_frac=c["no_contact_frac"], stance_frac=c["stance_frac"],
        phantom_lowering_cm_max=c["phantom_lowering_cm_max"], pelvis_rise_cm=c["pelvis_rise_cm"], travel_cm=c["travel_cm"],
        flights=[{k: s.get(k) for k in ("start_s", "end_s", "dur_s", "decision", "rise_cm", "needed_cm") if k in s} or s for s in c["stretches"] or []],
        angle_err_deg=ret["angle_err_deg"]["mean"])


def decide(rows):
    """The fix 3 rule, per SAM 3D Body variant: halve both numbers on both athletic clips, squat_jump no worse
    (on the rule's three clips; the others get the "no worse" check, reported apart)."""
    out = {}
    for fe in ("sam3d_camera", "sam3d_image"):
        checks = {}
        for clip, r in rows.items():
            if fe not in r:
                continue
            b, a = r["mediapipe"]["summary"], r[fe]["summary"]
            # no double-support frames in the baseline (hurdles: MediaPipe never has both feet down): knee alone
            ank = a["double_support_lr_cm"] / b["double_support_lr_cm"] if a["double_support_lr_cm"] and b["double_support_lr_cm"] else None
            knee = a["knee_ik_mean_cm"] / b["knee_ik_mean_cm"]
            limit = 0.5 if clip in ATHLETIC else 1.0
            ok = knee <= limit and (ank is None or ank <= limit)
            checks[clip] = dict(ankle_ratio=None if ank is None else round(ank, 2), knee_ratio=round(knee, 2), passes=bool(ok),
                                rule="<= 0.5 on both" if clip in ATHLETIC else "<= 1.0 on both")
        rule = {c: v for c, v in checks.items() if c in CLIPS}
        out[fe] = dict(checks=checks, adopt=len(rule) == len(CLIPS) and all(c["passes"] for c in rule.values()),
                       other_clips_no_worse=[c for c, v in checks.items() if c not in CLIPS and v["passes"]])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*", help="default: every clip with a SAM 3D Body npz")
    ap.add_argument("--out", default=RESULTS / "frontend_ab" / "frontend_ab.json")
    a = ap.parse_args(argv)
    rows = {}
    for name, d in clip_dirs(a.clips or [f.stem for f in SAM3D.glob("*.npz")]).items():
        fronts, fps = trajectories(name, d)
        rows[name] = {}
        for fe, (P, az, extra) in fronts.items():
            stats, ret = clip_stats(P, fps, az), audit(P, fps)
            rows[name][fe] = dict(view_azimuth_deg=round(float(az), 1), **extra, summary=summary(stats, ret) | soles(P, fps),
                                  pose_stats=stats, retarget=ret)
            s = rows[name][fe]["summary"]
            print(f"{name:11s} {fe:13s} bone cv thigh/shin/uarm/farm {list(s['bone_cv_pct'].values())} | ds ankle {s['double_support_lr_cm']} "
                  f"| hover {s['stance_hover_cm']} | depth ratio {s['depth_ratio']} | knee IK {s['knee_ik_cm']} | limits {s['limits_frac']:.0%} "
                  f"| cap {s['cap_frames']} | root shift {s['root_shift_cm']} | no contact {s['no_contact_frac']:.0%}", flush=True)
    rows["decision"] = decide({k: v for k, v in rows.items()})
    print(json.dumps(rows["decision"], indent=1))
    print("->", dump(rows, a.out))


if __name__ == "__main__":
    main()
