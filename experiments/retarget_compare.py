"""Uniform vs proportional retarget on the clips' human trajectories: IK tracking error, joint
limit hits, foot sliding and penetration, and joint-angle agreement with the human. Writes
compare.json, a markdown table, each clip's new qpos/motion.csv, and optionally a render.

    python experiments/retarget_compare.py                        # every clip under --clips
    python experiments/retarget_compare.py --render squat_jump    # + athlete | uniform | proportional video
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import render  # noqa: E402
import retarget  # noqa: E402

METHODS = {"uniform": False, "proportional": True}
ANGLES = ("knee", "hip", "elbow", "shoulder")
LIMIT_MARGIN = np.radians(1.0)


def human_skeleton(P):
    """The raw human's joints, named as in retarget.rescale_bones (no rescaling)."""
    idx = {"left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26, "left_ankle": 27,
           "right_ankle": 28, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13,
           "right_elbow": 14, "left_wrist": 15, "right_wrist": 16}
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


def limit_hits(model, qpos):
    """Fraction of frames with any hinge within 1 degree of its range, and the worst joints."""
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    q = qpos[:, [model.jnt_qposadr[j] for j in hinges]]
    lo, hi = model.jnt_range[hinges].T
    hit = (q < lo + LIMIT_MARGIN) | (q > hi - LIMIT_MARGIN)
    worst = {model.joint(hinges[i]).name: round(float(hit[:, i].mean()), 3)
             for i in np.argsort(-hit.mean(0))[:3] if hit[:, i].any()}
    return float(hit.any(1).mean()), worst


def evaluate(rt, P, qpos, targets, stance, fps):
    rep = {"tracking_cm": {k: round(v[0], 2) for k, v in retarget.tracking_report(rt, qpos, targets).items()}}
    rep["limit_frames"], rep["limit_worst"] = limit_hits(rt.model, qpos)
    feet = np.stack([rt.skeleton(qpos)[f"{s}_ankle"] for s in ("left", "right")], 1)
    step = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1)  # (T-1, 2)
    both = stance[1:] & stance[:-1]
    rep["slide_cm_per_s"] = float(step[both].sum() / max(both.sum(), 1) * fps * 100)
    drift = [np.linalg.norm(feet[a:b, f, :2] - feet[a, f, :2], axis=-1).max()
             for f in range(2) for a, b in retarget._runs(stance[:, f])]
    rep["drift_cm"] = float(np.mean(drift) * 100) if drift else 0.0
    sole = sole_heights(rt.model, qpos)
    rep["penetration_cm"] = float(max(0.0, -sole.min()) * 100)
    rep["penetration_frames"] = float((sole < -0.01).any(1).mean())
    rep["stance_float_cm"] = float(np.abs(sole[stance]).mean() * 100) if stance.any() else 0.0
    human = retarget.flexion_angles(human_skeleton(P))
    robot = retarget.flexion_angles(rt.skeleton(qpos))
    rep["angle_err_deg"] = {a: float(np.mean([np.abs(human[f"{s}_{a}"] - robot[f"{s}_{a}"]).mean()
                                              for s in ("left", "right")])) for a in ANGLES}
    return rep


def run_clip(clip_dir, out_dir, fps):
    P = np.load(clip_dir / "human.npy")
    rt = retarget.G1Retargeter()
    stance = retarget.proportional_motion(P, rt, fps)["stance"]  # contacts from the human, for both
    res, qposes = {"frames": len(P), "stance_frac": stance.mean(0).round(2).tolist()}, {}
    for name, prop in METHODS.items():
        rt = retarget.G1Retargeter()
        targets, rotations = retarget.human_to_targets(P, rt, proportional=prop, fps=fps)
        qpos = rt.retarget_sequence(targets, rotations)
        res[name] = evaluate(rt, P, qpos, targets, stance, fps)
        qposes[name] = qpos
        np.save(out_dir / f"qpos_{name}.npy", qpos)
    retarget.export_mjlab_csv(qposes["proportional"], out_dir / "motion.csv")
    return res, qposes, rt


def table(results):
    """Markdown: one row per clip and method."""
    cols = ["clip", "method", "track pelvis/feet/knees/elbows/hands (cm)", "limit-hit frames",
            "slide (cm/s)", "drift / contact (cm)", "penetration max (cm)", "stance float (cm)",
            "knee err (deg)", "hip err", "elbow err", "shoulder err"]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for clip, res in results.items():
        for m in METHODS:
            r = res[m]
            t = r["tracking_cm"]
            pair = lambda k: (t[f"left_{k}"] + t[f"right_{k}"]) / 2
            track = f"{t['pelvis']:.1f} / {pair('foot'):.1f} / {pair('knee'):.1f} / {pair('elbow'):.1f} / {pair('hand'):.1f}"
            a = r["angle_err_deg"]
            rows.append(f"| {clip} | {m} | {track} | {r['limit_frames']:.0%} | {r['slide_cm_per_s']:.1f} | "
                        f"{r['drift_cm']:.1f} | {r['penetration_cm']:.1f} | {r['stance_float_cm']:.1f} | "
                        f"{a['knee']:.1f} | {a['hip']:.1f} | {a['elbow']:.1f} | {a['shoulder']:.1f} |")
    return "\n".join(rows)


def render_compare(rt, clip_dir, qposes, video, out_path, fps):
    """athlete | uniform | proportional, robots seen from the video camera's side."""
    import mediapy
    meta = json.loads((clip_dir / "meta.json").read_text()) if (clip_dir / "meta.json").exists() else {}
    cam = render.tracking_camera(rt.model, azimuth=meta.get("view_azimuth", 135.0))
    robots = [render.render_qpos_sequence(rt.model, qposes[m], size=(480, 480), cam=cam) for m in METHODS]
    frames = np.concatenate(robots, axis=2)
    if video and Path(video).exists():
        frames = render.side_by_side(np.asarray(mediapy.read_video(str(video))), frames)
    return render.save_video(frames, out_path, fps=fps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, default=config.OUTPUT_DIR / "remote_latent_v2" / "clips")
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "retarget_compare")
    ap.add_argument("--only", nargs="*", help="clip names (default: all with a human.npy)")
    ap.add_argument("--render", nargs="*", default=[], help="clips to render")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    results = {}
    for clip_dir in sorted(p.parent for p in a.clips.glob("*/human.npy")):
        if a.only and clip_dir.name not in a.only:
            continue
        out_dir = a.out / clip_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        results[clip_dir.name], qposes, rt = run_clip(clip_dir, out_dir, a.fps)
        print(clip_dir.name, json.dumps(results[clip_dir.name]), flush=True)
        if clip_dir.name in a.render:
            video = next(iter(config.CLIPS_DIR.glob(f"{clip_dir.name}.*")), None)
            print("rendered", render_compare(rt, clip_dir, qposes, video, out_dir / "compare.mp4", a.fps))
    (a.out / "compare.json").write_text(json.dumps(results, indent=1))
    (a.out / "compare.md").write_text(table(results) + "\n")
    print(table(results))
