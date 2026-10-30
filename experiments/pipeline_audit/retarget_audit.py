"""Stage 2 of the audit: re-run the current retarget.py on every clip's human.npy and measure what it loses: IK error
per frame type, joint-limit hits, frames at the joint-speed cap, foot penetration and sliding, joint-angle error
against the human, the root shift plant_feet applies, and every no-contact stretch with ground_phantom_flights'
decision on it. Also compares with the stored qpos.npy (which retarget version it came from)."""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import retarget  # noqa: E402
from experiments.pipeline_audit import RESULTS, clip_dirs, dump  # noqa: E402

ANGLES = ("knee", "hip", "elbow", "shoulder")


def human_skeleton(P):
    idx = {"left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
           "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16}
    J = {k: P[:, i] for k, i in idx.items()}
    J["hip_mid"], J["shoulder_mid"] = P[:, [23, 24]].mean(1), P[:, [11, 12]].mean(1)
    return J


def sole_heights(model, qpos):
    """(T, 2): lowest point of each foot's contact spheres above the floor (negative = penetration)."""
    data = mujoco.MjData(model)
    soles = [[g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(f"{s}_ankle_roll_link").id
              and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE] for s in ("left", "right")]
    out = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        out.append([min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in gs) for gs in soles])
    return np.array(out)


def contact_report(P, rt, fps):
    """proportional_motion's contact steps, one by one, to see what ground_phantom_flights decided and why."""
    seg = lambda a, b: np.linalg.norm(P[:, a] - P[:, b], axis=-1)
    human_leg = np.median(np.concatenate([seg(23, 25) + seg(25, 27), seg(24, 26) + seg(26, 28)]))
    scale = (rt.bones["thigh"] + rt.bones["shin"]) / human_leg
    J = retarget.rescale_bones(P, rt.bones)
    root = P[:, [23, 24]].mean(1) * scale
    ankles = np.stack([J["left_ankle"], J["right_ankle"]], 1)
    root[:, 2] += rt.sole_height - np.percentile((root[:, None] + ankles)[..., 2].min(1), 5)
    stance = retarget.stance_mask(root[:, None] + ankles, fps)
    J = retarget.level_feet(J, stance)
    ankles = np.stack([J["left_ankle"], J["right_ankle"]], 1)
    floor = np.full(stance.shape, rt.sole_height)  # flat feet; proportional_motion raises a lifted heel's floor
    root1, feet, foot_fix = retarget.plant_feet(root, ankles, stance, floor)
    root2, _ = retarget.ground_phantom_flights(root1, feet, stance, floor, fps)
    above = (feet[..., 2] - floor).min(1)
    up = ~stance.any(1) & (above > 0.01)
    stretches = []
    for a, b in retarget._runs(up):
        T = (b - a + 1) / fps
        ends = root1[[max(a - 1, 0), min(b, len(root1) - 1)], 2].mean()
        rise = root1[a:b, 2].max() - ends
        need = retarget.FLIGHT_RISE_FRAC * retarget.GRAVITY * T ** 2 / 8
        stretches.append(dict(start_s=round(a / fps, 2), dur_s=round(T, 2), rise_cm=round(100 * rise, 1), need_cm=round(100 * need, 1),
                              ballistic_rise_cm=round(100 * retarget.GRAVITY * T ** 2 / 8, 1), grounded=bool(rise < need),
                              lowered_cm=round(100 * float(above[a:b].mean()), 1)))
    shift = np.linalg.norm(root1 - root, axis=-1)
    return dict(scale=round(float(scale), 3), stance_frac=[round(float(stance[:, f].mean()), 2) for f in range(2)],
                no_contact_frac=round(float((~stance.any(1)).mean()), 2),
                foot_fix_cm=dict(mean=round(100 * float(np.linalg.norm(foot_fix, axis=-1).mean()), 2), max=round(100 * float(np.linalg.norm(foot_fix, axis=-1).max()), 2)),
                root_shift_cm=dict(mean=round(100 * float(shift.mean()), 2), max=round(100 * float(shift.max()), 2)),
                phantom_lowering_cm_max=round(100 * float(np.abs(root2[:, 2] - root1[:, 2]).max()), 2), stretches=stretches)


def audit(P, fps, stored=None, save_qpos=None):
    rt = retarget.G1Retargeter()
    model = rt.model
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    hnames = [model.joint(j).name for j in hinges]
    adr = [model.jnt_qposadr[j] for j in hinges]
    lo, hi = model.jnt_range[hinges].T
    t0 = time.time()
    targets, rots = retarget.human_to_targets(P, rt, fps=fps)
    qpos = rt.retarget_sequence(targets, rots, fps=fps)
    if save_qpos:
        np.save(save_qpos, qpos)
    errs = {n: [] for n in retarget.FRAMES}
    for q, tg in zip(qpos, targets):
        rt.config.update(q)
        for n, p in tg.items():
            key = "pelvis" if n == "hip_centre" else n.removesuffix("_torso")
            errs[key].append(np.linalg.norm(rt.frame_position(n) - p) * 100)
    track = {n: dict(mean=round(float(np.mean(e)), 2), p95=round(float(np.percentile(e, 95)), 2), max=round(float(np.max(e)), 2),
                     frames_gt3cm=round(float((np.array(e) > 3).mean()), 3)) for n, e in errs.items() if e}
    q = qpos[:, adr]
    speed = np.abs(np.diff(q, axis=0)) * fps
    at_cap = speed >= retarget.MAX_JOINT_SPEED * 0.98
    hit = (q < lo + np.radians(1)) | (q > hi - np.radians(1))
    sole = sole_heights(model, qpos)
    stance = retarget.proportional_motion(P, rt, fps)["stance"]
    feet = np.stack([rt.skeleton(qpos)[f"{s}_ankle"] for s in ("left", "right")], 1)
    step = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1)
    both = stance[1:] & stance[:-1]
    human, robot = retarget.flexion_angles(human_skeleton(P)), retarget.flexion_angles(rt.skeleton(qpos))
    ang = lambda f: {a: round(float(f([np.abs(human[f"{s}_{a}"] - robot[f"{s}_{a}"]) for s in ("left", "right")])), 1) for a in ANGLES}
    return dict(
        frames=len(qpos), seconds=round(time.time() - t0, 1),
        stored_vs_recomputed_max_diff_rad=None if stored is None or len(stored) != len(qpos) else round(float(np.abs(stored - qpos).max()), 4),
        tracking_cm=track,
        joint_speed=dict(max_rad_s=round(float(speed.max()), 1), p99_rad_s=round(float(np.percentile(speed, 99)), 1),
                         frames_any_joint_at_cap=int(at_cap.any(1).sum()),
                         cap_joints={hnames[i]: int(at_cap[:, i].sum()) for i in np.argsort(-at_cap.sum(0))[:4] if at_cap[:, i].any()}),
        limits=dict(frames_any=round(float(hit.any(1).mean()), 3),
                    worst={hnames[i]: round(float(hit[:, i].mean()), 3) for i in np.argsort(-hit.mean(0))[:4] if hit[:, i].any()}),
        feet=dict(penetration_max_cm=round(float(max(0, -sole.min()) * 100), 2), penetration_frames_gt1cm=round(float((sole < -0.01).any(1).mean()), 3),
                  stance_float_mean_cm=round(float(np.abs(sole[stance]).mean() * 100), 2) if stance.any() else None,
                  stance_float_p95_cm=round(float(np.percentile(np.abs(sole[stance]), 95) * 100), 2) if stance.any() else None,
                  lowest_sole_gt2cm_frac=round(float((sole.min(1) > 0.02).mean()), 3),
                  slide_cm_per_s=round(float(step[both].sum() / max(both.sum(), 1) * fps * 100), 2)),
        angle_err_deg=dict(mean=ang(lambda d: np.mean([x.mean() for x in d])), max=ang(lambda d: max(x.max() for x in d))),
        contacts=contact_report(P, rt, fps),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--out", default=RESULTS / "retarget_audit.json")
    ap.add_argument("--save-qpos", help="folder for the recomputed <clip>_qpos.npy")
    a = ap.parse_args(argv)
    if a.save_qpos:
        Path(a.save_qpos).mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, d in clip_dirs(a.clips).items():
        fps = json.loads((d / "meta.json").read_text())["fps"]
        rows[name] = audit(np.load(d / "human.npy"), fps, np.load(d / "qpos.npy") if (d / "qpos.npy").exists() else None,
                           Path(a.save_qpos) / f"{name}_qpos.npy" if a.save_qpos else None)
        r = rows[name]
        print(f"{name:14s} knees {r['tracking_cm']['left_knee']['mean']}/{r['tracking_cm']['right_knee']['mean']} cm | limits {r['limits']['frames_any']:.0%} {r['limits']['worst']} | "
              f"cap frames {r['joint_speed']['frames_any_joint_at_cap']} | root shift {r['contacts']['root_shift_cm']} | no contact {r['contacts']['no_contact_frac']:.0%} | "
              f"phantom lowering {r['contacts']['phantom_lowering_cm_max']} cm | stored diff {r['stored_vs_recomputed_max_diff_rad']} rad")
    print("->", dump(rows, a.out))


if __name__ == "__main__":
    main()
