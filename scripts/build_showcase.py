"""The website's 3D showcase -> web/public/showcase/<id>/: the G1 as meshes plus its kinematic tree (the
browser does the forward kinematics), the trained policy's own run, the retarget it tracks, and the
policy's recorded responses to pushes (experiments/push_showcase.py, run as `remote_run.py pushes`).

    python scripts/build_showcase.py                      # squat jump, tracking_var (needs numpy, mujoco, fast-simplification)
    python scripts/build_showcase.py --robot-only         # just the meshes

Outputs (all static; the site reads nothing else):
    g1.json / g1.bin   bodies (parent, pose in the parent, hinge joint), and per body its visual meshes,
                       decimated: float32 positions, uint16 triangles, a tone (the MJCF's silver / black)
    showcase.json      timing, clip stats, and the push index: moments x directions x strengths
    nominal.bin        the policy's unpushed run (int16 frames, below); reference.bin the retarget
    push_<step>.bin    every push at that moment, end to end, every --stride-th frame
A frame is 36 int16: root position (mm), root quaternion wxyz (x 32767), 29 joint angles (x 1e4 rad).

Each recording starts --pre frames before its push, on its own env's run: the envs follow the
nominal run only up to float noise that the jumps amplify, so the site crossfades into a recording
over those frames instead of cutting. A pushed recording ends once the robot is back on the unpushed run (within --rejoin-m of its root
position and --rejoin-rad of every joint, for 0.2 s); the site then carries on with the unpushed run.
A recording that falls is kept for --after-fall-s after the fall, then the site restarts the clip.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAIN = REPO.parents[2] if REPO.parent.name == "worktrees" else REPO  # outputs/ lives in the main checkout
G1_XML = MAIN / "assets/mujoco_menagerie/unitree_g1/g1_mjlab.xml"


def outputs(name):
    """outputs/<name> here, else in the main checkout (a worktree has its own jobs' results only)."""
    return next((b / "outputs" / name for b in (REPO, MAIN) if (b / "outputs" / name).exists()), MAIN / "outputs" / name)


SCALE = np.array([1000] * 3 + [32767] * 4 + [1e4] * 29, np.float32)


def frames_i16(q):
    q = np.asarray(q, np.float32).copy()
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    return np.clip(np.round(q * SCALE), -32767, 32767).astype("<i2")


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rotate(q, v):
    return quat_mul(quat_mul(q, np.r_[0, v]), q * [1, -1, -1, -1])[1:]


def fk(bodies, qpos):
    """Body world positions, as the browser computes them (checked against MuJoCo below)."""
    pos, quat = [np.zeros(3)], [np.array([1.0, 0, 0, 0])]
    for b in bodies:
        if b["free"]:
            p, q = qpos[:3].astype(float), qpos[3:7] / np.linalg.norm(qpos[3:7])
        else:
            pp, pq = pos[b["parent"]], quat[b["parent"]]
            p, q = pp + rotate(pq, np.array(b["pos"])), quat_mul(pq, np.array(b["quat"]))
            if b["joint"] >= 0:
                th = qpos[7 + b["joint"]]
                r = np.r_[np.cos(th / 2), np.sin(th / 2) * np.array(b["axis"])]
                jp = np.array(b["jpos"])
                p = p + rotate(q, jp - rotate(r, jp))
                q = quat_mul(q, r)
        pos.append(p)
        quat.append(q)
    return np.array(pos[1:])


def build_robot(out, tris):
    import fast_simplification
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    bodies = []
    for i in range(1, m.nbody):
        j = [k for k in range(m.njnt) if m.jnt_bodyid[k] == i]
        assert len(j) <= 1, m.body(i).name
        free = bool(j) and m.jnt_type[j[0]] == mujoco.mjtJoint.mjJNT_FREE
        hinge = j[0] if j and not free else None
        bodies.append({"name": m.body(i).name, "parent": int(m.body_parentid[i]), "free": free,
                       "pos": m.body_pos[i].round(6).tolist(), "quat": m.body_quat[i].round(6).tolist(),
                       "joint": int(m.jnt_qposadr[hinge] - 7) if hinge is not None else -1,
                       "jpos": m.jnt_pos[hinge].round(6).tolist() if hinge is not None else [0, 0, 0],
                       "axis": m.jnt_axis[hinge].round(6).tolist() if hinge is not None else [0, 0, 1]})
    # forward kinematics as the browser does it == MuJoCo's, on a random pose
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    d.qpos[:3], d.qpos[3:7] = [0.1, -0.2, 0.8], [0.9, 0.1, -0.2, 0.3]
    d.qpos[7:] = rng.uniform(-1, 1, m.nq - 7)
    mujoco.mj_kinematics(m, d)
    err = np.abs(fk(bodies, d.qpos.copy()) - d.xpos[1:]).max()
    assert err < 1e-5, f"forward kinematics differs from MuJoCo by {err} m"

    geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2]
    faces_all = sum(int(m.mesh_facenum[m.geom_dataid[g]]) for g in geoms)
    parts, pos_chunks, idx_chunks, voff, ioff = [], [], [], 0, 0
    for g in geoms:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        keep = max(80, int(len(f) * tris / faces_all))
        if keep < len(f):
            v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1 - keep / len(f))
        q = m.geom_quat[g]
        v = np.array([rotate(q, x) for x in v]) + m.geom_pos[g]  # mesh frame -> body frame
        assert len(v) < 65536
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        parts.append({"body": int(m.geom_bodyid[g]) - 1, "tone": "dark" if rgba[0] < 0.4 else "light",
                      "v": [voff, len(v)], "i": [ioff, f.size]})
        pos_chunks.append(v.astype("<f4").ravel())
        idx_chunks.append(f.astype("<u2").ravel())
        voff, ioff = voff + len(v), ioff + f.size
    pos, idx = np.concatenate(pos_chunks), np.concatenate(idx_chunks)
    (out / "g1.bin").write_bytes(pos.tobytes() + idx.tobytes())
    (out / "g1.json").write_text(json.dumps({
        "source": "MuJoCo Menagerie unitree_g1 (g1_mjlab.xml), visual meshes decimated",
        "bodies": bodies, "parts": parts, "bin": "g1.bin", "positions": [0, int(pos.size)],
        "indices": [int(pos.nbytes), int(idx.size)], "joints": [m.joint(k).name for k in range(1, m.njnt)]}))
    print(f"robot: {len(bodies)} bodies, {len(parts)} meshes, {ioff // 3} triangles ({faces_all} before), "
          f"{(pos.nbytes + idx.nbytes) / 1e3:.0f} kB")


def rejoin_frame(rec, nominal, s, k, a, fps):
    """rec: a pushed run from clip frame s. The first frame >= k + 0.5 s from which it stays on the
    nominal run for 0.2 s (absolute clip frame), or None."""
    nom = nominal[s:s + len(rec)]
    n = len(nom)
    near = (np.linalg.norm(rec[:n, :3] - nom[:, :3], axis=1) < a.rejoin_m) & (np.abs(rec[:n, 7:] - nom[:, 7:]).max(1) < a.rejoin_rad)
    hold = int(0.2 * fps)
    for f in range(k - s + int(0.5 * fps), n - hold):
        if near[f: f + hold].all():
            return s + f
    return None


def build_motion(out, a):
    run = outputs(a.run)
    pol = run / f"policy_{a.policy}"
    z = np.load(outputs(a.pushes) / "push_showcase.npz")
    fps = int(z["fps"])
    nominal = z["nominal"]
    ref = np.load(pol / "motion.npz")
    # nominal[t] is the state after env step t, when the reference is at frame start_step + t + 1
    s0 = int(z["start_step"]) + 1
    reference = np.concatenate([ref["body_pos_w"][:, 0], ref["body_quat_w"][:, 0], ref["joint_pos"]], 1)[s0:s0 + len(nominal)]
    print(f"nominal vs the retarget it tracks: mean joint diff {np.abs(reference[:, 7:] - nominal[:len(reference), 7:]).mean():.3f} rad")
    clean = np.load(pol / "policy_qpos.npy")
    print(f"nominal vs the run's own policy_qpos.npy: max joint diff {np.abs(clean[:len(nominal), 7:] - nominal[:len(clean), 7:]).max():.3f} rad")
    (out / "nominal.bin").write_bytes(frames_i16(nominal).tobytes())
    (out / "reference.bin").write_bytes(frames_i16(reference).tobytes())

    steps, angles, mps, pre = z["step"], z["angle_rad"], z["mps"], int(z["pre"])
    moments, total = [], 0
    for k in sorted(set(steps.tolist())):
        rows, blobs = [], []
        for i in np.nonzero(steps == k)[0]:
            rec, s = z[f"q_{i}"], max(0, k - pre)  # env i's own run from clip frame s (pre-roll, then the push at k)
            fall = int(z["fall_step"][i])
            if fall >= 0:
                end, rejoin = min(s + len(rec), fall + int(a.after_fall_s * fps)), None
            else:
                rejoin = rejoin_frame(rec, nominal, s, k, a, fps)
                end = rejoin if rejoin is not None else s + len(rec)
            seg = frames_i16(rec[:end - s:a.stride])
            rows.append({"angle": round(float(angles[i]), 4), "mps": float(mps[i]), "start": s, "frames": len(seg),
                         "fall": fall if fall >= 0 else None, "rejoin": rejoin, "offset": sum(len(b) for b in blobs) // (36 * 2)})
            blobs.append(seg.tobytes())
        data = b"".join(blobs)
        (out / f"push_{k}.bin").write_bytes(data)
        total += len(data)
        moments.append({"step": int(k), "file": f"push_{k}.bin", "pushes": rows})
    falls = {f"{v:g}": float(np.mean([r["fall"] is not None for mo in moments for r in mo["pushes"] if r["mps"] == v]))
             for v in sorted(set(mps.tolist()))}
    rejoined = float(np.mean([r["rejoin"] is not None for mo in moments for r in mo["pushes"] if r["fall"] is None]))
    meta = json.loads((outputs(a.pushes) / "push_showcase.json").read_text())
    # the source video: the example's side-by-side, cropped to its person panel
    prim = json.loads((REPO / "web/public/examples" / a.example / "example.json").read_text())["primary"]
    person = next(p for p in prim["panels"] if p["label"] == "Athlete")
    pm = json.loads((pol / "policy_metrics.json").read_text())
    show = {
        "id": a.id, "example": a.example, "policy": a.policy, "fps": fps, "frames": len(nominal),
        "format": "int16 x 36 per frame: root pos (mm), root quat wxyz (x32767), 29 joints (x1e4 rad), model order",
        "nominal": "nominal.bin", "reference": "reference.bin", "push_stride": a.stride,
        "video": {"src": f"examples/{prim['src']}", "poster": f"examples/{prim['poster']}",
                  "size": [prim["width"], prim["height"]], "crop": [person["from"], person["to"]], "offset_s": s0 / fps},
        "clean": {"falls": pm["falls"], "joint_err_rad": round(pm["mean_joint_error_rad"], 3),
                  "torso_err_m": round(pm["mean_torso_position_error_m"], 3)},
        "pushes": {"kind": "root velocity kick (as robustness_eval.py and training)", "strengths_mps": sorted(set(mps.tolist())),
                   "directions": int(len(set(angles.tolist()))), "fall_rate": falls, "rejoined": rejoined,
                   "rejoin": {"m": a.rejoin_m, "rad": a.rejoin_rad}, "after_fall_s": a.after_fall_s, "pre_frames": pre,
                   "drift": {"median": round(float(np.median(z["prefix_dev"])), 4), "max": round(float(z["prefix_dev"].max()), 4)},
                   "checkpoint": Path(meta["checkpoint"]).name, "moments": moments},
    }
    (out / "showcase.json").write_text(json.dumps(show, separators=(",", ":")))
    print(f"motion: {len(moments)} moments, {len(steps)} pushes, {total / 1e6:.2f} MB of pushes; fall rate {falls}; "
          f"{rejoined:.0%} of the upright ones rejoin the unpushed run")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="squat-jump")
    ap.add_argument("--example", default="squat-jump-variants", help="its examples/<id>/ (the source video)")
    ap.add_argument("--run", default="remote_variants_var_sj")
    ap.add_argument("--policy", default="tracking_var")
    ap.add_argument("--pushes", default="remote_pushes_sj", help="outputs/<folder> with push_showcase.npz")
    ap.add_argument("--tris", type=int, default=36000, help="triangle budget for the whole robot")
    ap.add_argument("--stride", type=int, default=2, help="keep every n-th frame of the pushes (the site interpolates)")
    # about the run-to-run spread of unpushed envs (prefix_dev), so the ease back onto the nominal run is small
    ap.add_argument("--rejoin-m", type=float, default=0.05)
    ap.add_argument("--rejoin-rad", type=float, default=0.15)
    ap.add_argument("--after-fall-s", type=float, default=1.5)
    ap.add_argument("--robot-only", action="store_true")
    a = ap.parse_args()
    out = REPO / "web/public/showcase" / a.id
    out.mkdir(parents=True, exist_ok=True)
    build_robot(out, a.tris)
    if not a.robot_only:
        build_motion(out, a)
    sys.exit(0)
