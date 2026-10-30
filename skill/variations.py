"""Variations of any clip, made without knowing what the action is: the training set for a policy that
should do the action, not replay one recording of it.

    python -m skill.variations outputs/remote_latent_kp_kick/clips/front_kick --out /tmp/kick_variants

Every operation is defined on the G1's joint positions and the clip's own contacts, never on the
action, so a squat jump, a kick and a lunge get the same treatment:

- mirror: left and right swapped through the sagittal plane (exact: the G1's left and right joint axes
  are the same vectors, so roll and yaw joints change sign).
- tempo: the clip played faster or slower, uniformly or with a different speed between each pair of
  key moments (a random monotone time warp). Ballistic phases (both feet off the floor) keep their
  real duration: gravity sets how long a flight takes, so a slowed-down jump would be impossible.
  Nor may a speed-up push a joint past the retarget's own cap (retarget.MAX_JOINT_SPEED): where it
  would, that stretch plays only as fast as the cap allows.
- amplitude: every joint, the root's height and travel, and its turn scaled about the clip's first
  frame (alpha 0.85 = a smaller version of the same motion: a shallower squat, a lower kick). The
  root is then shifted vertically so the feet touch the floor where the original's did, and each
  flight is re-timed to the airtime its new pelvis rise needs (sqrt(8 rise / g)): a higher jump
  takes longer, a lower one less.

Each variant gets its own keypose spec, compiled by the same rule as any clip (skill/keypose.py), and
is kept only if its own reference completes that spec (self-check), stays within the joint limits,
and is physical: no joint faster than the cap, every flight ballistic (pelvis rise within
[0.6, 1.5] x g T^2 / 8 of its airtime T; a flight the clip ends or starts in has an unknown airtime and
only the lower bound applies), every flight of the clip still there, and the feet no more than
PENETRATION into the floor. A dropped variant's "why" says which rule it broke.
TRAIN and HELD_OUT are disjoint: a policy is trained on TRAIN and evaluated on HELD_OUT, which has
tempos and amplitudes outside the training range, so success there is generalisation, not recall.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from skill import task as T

FPS = 30.0
L_ANKLE, R_ANKLE = 3, 6  # in latent.reward_net.BODIES, as skill/keypose.py
# (mirror, tempo, amplitude, warp): warp = how far each key-moment segment's speed may vary (0 = uniform)
TRAIN = [(m, s, a, w) for m in (False, True) for s in (0.75, 1.0, 1.33) for a in (0.85, 1.0, 1.1) for w in (0.0, 0.25)]
HELD_OUT = [(False, 0.5, 1.0, 0.0), (False, 2.0, 1.0, 0.0), (False, 1.0, 0.7, 0.0), (False, 1.0, 1.25, 0.0),
            (True, 0.6, 0.8, 0.0), (True, 1.6, 1.2, 0.0), (False, 1.0, 1.0, 0.5)]
MAX_LIMIT_FRAC = 0.02  # share of joint samples allowed at a limit (clipped) before a variant is dropped
FLIGHT_AIR = 0.10  # m: both feet this far above the floor is a flight (the spec's own air_ref floor)
BALLISTIC = (0.6, 1.5)  # a flight's pelvis rise / (g T^2 / 8) outside this is not a jump gravity allows
PENETRATION = 0.02  # m a foot's contact spheres may go below the floor (the retargets reach 1.8 cm)
SPEED_SLACK = 1.001  # the retarget's cap is met exactly on capped frames; this is float noise, not motion


class G1:
    """The G1 model's joint layout: mirror permutation and signs, ranges, forward kinematics."""

    def __init__(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import mujoco
        import render
        self.model = m = render.load_g1()[0]
        joints = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        adr = [int(m.jnt_qposadr[m.joint(n).id]) for n in joints]
        assert adr == list(range(7, m.nq)), "expected the free base then one qpos per joint"
        swap = lambda n: n.replace("left", "@").replace("right", "left").replace("@", "right")  # noqa: E731
        self.perm = np.array([joints.index(swap(n)) for n in joints])
        self.sign = np.array([-1.0 if ("roll" in n or "yaw" in n) else 1.0 for n in joints])
        self.lo = np.array([m.jnt_range[m.joint(n).id][0] for n in joints])
        self.hi = np.array([m.jnt_range[m.joint(n).id][1] for n in joints])

    def bodies(self, qpos):
        from latent.distill import body_positions
        return body_positions(self.model, qpos)

    def soles(self, qpos):
        """(T,) height of the lowest point of either foot's contact spheres above the floor (< 0: in it)."""
        import mujoco
        m, data = self.model, mujoco.MjData(self.model)
        feet = {m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")}
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in feet and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        out = np.empty(len(qpos))
        for t, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_kinematics(m, data)
            out[t] = min(data.geom_xpos[g][2] - m.geom_size[g][0] for g in gs)
        return out


def _rot(q):
    return Rotation.from_quat(q[:, [4, 5, 6, 3]])  # MuJoCo wxyz -> scipy xyzw


def _quat(r):
    x = r.as_quat()
    return x[:, [3, 0, 1, 2]]


def mirror(qpos, g1):
    """Reflect through the x-z plane: y -> -y, the root's roll and yaw change sign, left and right swap."""
    out = qpos.copy()
    out[:, 1] *= -1
    out[:, 4] *= -1  # quaternion (w, x, y, z) -> (w, -x, y, -z)
    out[:, 6] *= -1
    out[:, 7:] = qpos[:, 7:][:, g1.perm] * g1.sign
    return out


def resample(qpos, src_t):
    """qpos sampled at fractional source frames src_t: positions and joints linearly, the root's
    orientation by slerp."""
    n = len(qpos)
    src_t = np.clip(src_t, 0, n - 1)
    i0 = np.minimum(np.floor(src_t).astype(int), n - 2)
    w = (src_t - i0)[:, None]
    out = qpos[i0] * (1 - w) + qpos[i0 + 1] * w
    out[:, 3:7] = _quat(Slerp(np.arange(n), _rot(qpos))(src_t))
    return out


def airborne(pos, foot0, air=T.AIR):
    """Frames with both feet off the floor."""
    return np.minimum(pos[:, L_ANKLE, 2], pos[:, R_ANKLE, 2]) - foot0 > air


def flights(pos, foot0, fps, air=FLIGHT_AIR):
    """Flights (both feet more than `air` off the floor for at least T.MIN_FLIGHT) -> [{start, end (frames,
    end exclusive), air_s, rise (pelvis peak above its height where the feet cross `air`), ratio (rise /
    g T^2 / 8)}]. Rise and airtime are both measured from where the feet cross `air`: for a body that holds
    its shape in the air, the time spent above any level and the rise above it obey the same g T^2 / 8.
    The crossings are interpolated between frames: a 5-frame hop counted in whole frames is +-20% in T.
    open: the clip starts or ends in it (retarget.ground_phantom_flights keeps such a stretch as a flight
    when the clip ends with the pelvis up). Its airtime is censored, so only rise / ballistic >= 1 follows
    from gravity: a ballistic body still in the air after T has risen at least g T^2 / 8 by then."""
    from retarget import GRAVITY
    lo = np.minimum(pos[:, L_ANKLE, 2], pos[:, R_ANKLE, 2]) - foot0
    edges = np.flatnonzero(np.diff(np.r_[0, (lo > air).astype(int), 0]))
    z, out = pos[:, 0, 2], []

    def cross(i, j):  # fractional frame between i and j where lo crosses air, and the pelvis height there
        w = float(np.clip((air - lo[i]) / (lo[j] - lo[i]), 0, 1)) if lo[j] != lo[i] else 0.5
        return i + w * (j - i), z[i] + w * (z[j] - z[i])
    for a, b in zip(edges[::2], edges[1::2]):
        if (b - a) / fps < T.MIN_FLIGHT:
            continue
        (t0, z0), (t1, z1) = cross(max(a - 1, 0), a), cross(min(b, len(z) - 1), b - 1)
        t_air = max(t1 - t0, 1.0) / fps
        # an open end has no crossing: its last (first) frame is still in the air, so rise is from the other one
        base = z1 if a == 0 and b < len(z) else z0 if b == len(z) and a > 0 else (z0 + z1) / 2
        rise = float(z[a:b].max() - base)
        out.append({"start": int(a), "end": int(b), "air_s": round(t_air, 4), "rise": round(rise, 4),
                    "ratio": round(rise / (GRAVITY * t_air ** 2 / 8), 2), "open": bool(a == 0 or b == len(z))})
    return out


def ballistic(f):
    """Is flight f (from flights()) one gravity allows? An open one only has to rise enough for its airtime."""
    lo, hi = BALLISTIC
    return lo <= f["ratio"] and (f["open"] or f["ratio"] <= hi)


def _src_times(seg_speed):
    """Per-interval playback speed (source frames per output frame) -> the source frame of each output frame."""
    out_len = np.concatenate([[0.0], np.cumsum(1.0 / seg_speed)])  # output frames per source frame
    m = int(np.floor(out_len[-1] + 1e-9)) + 1
    return np.interp(np.arange(m), out_len, np.arange(len(seg_speed) + 1))


def time_map(n, keys, speed, warp, air, rng, joint_speed=None, cap=None, floor=1.0):
    """Source frame for each output frame. Between consecutive key frames the clip plays at `speed`
    times a random factor in [1 - warp, 1 + warp]; airborne frames always play at 1x.
    joint_speed: (n - 1,) each source interval's fastest joint (rad/s); where the chosen speed would take
    it past `cap`, the interval plays only as fast as the cap allows, but never slower than `floor` x
    min(speed, 1) (a scaled-up variant may slow to 1 / alpha to stay under the cap; a joint already past the cap
    in the clip itself is left to the filter)."""
    knots = sorted({0, *[int(k) for k in keys if 0 < k < n - 1], n - 1})
    seg_speed = np.ones(n - 1)
    for a, b in zip(knots, knots[1:]):
        seg_speed[a:b] = speed * (1 + rng.uniform(-warp, warp) if warp else 1.0)
    if joint_speed is not None:
        allowed = cap / np.maximum(joint_speed, 1e-9)
        seg_speed = np.minimum(seg_speed, np.maximum(allowed, floor * np.minimum(seg_speed, 1.0)))
    seg_speed = np.where(air[:-1] & air[1:], 1.0, seg_speed)  # gravity sets a flight's duration
    return _src_times(seg_speed)


def ballistic_retime(q, g1, foot0, fps, ref):
    """Each flight of q re-timed to the airtime its pelvis rise needs, sqrt(8 rise / g) -> source frame of
    each output frame. The stretch is uniform over the flight's frames and one either side, which hold
    both crossings of the air threshold, so the crossing-to-crossing time scales exactly.
    ref: the source clip's flights (as `flights()`); a flight is re-timed only if its source flight was
    ballistic and lands in the clip. Squeezing a non-ballistic one into the airtime its rise allows (hurdles'
    1.65 s, 15 cm float into 0.35 s) only multiplies its joint speeds; the filter reports it as it is."""
    from retarget import GRAVITY
    seg_speed = np.ones(len(q) - 1)
    for f in flights(g1.bodies(q), foot0, fps):
        src = [r for r in ref if r["start"] < f["end"] and f["start"] < r["end"]]
        if f["rise"] <= 0 or f["open"] or not src or not all(ballistic(r) and not r["open"] for r in src):
            continue
        need = np.sqrt(8 * f["rise"] / GRAVITY)
        seg_speed[max(f["start"] - 1, 0):f["end"]] = f["air_s"] / need
    return _src_times(seg_speed)


def scale(qpos, alpha, g1, foot0):
    """Amplitude alpha about the first frame; then the feet put back on the floor where they were: the
    lowest point of the soles, not the ankle (a scaled ankle pitch tips the toe or heel into the floor)."""
    q0 = qpos[:1]
    out = qpos.copy()
    out[:, :3] = q0[:, :3] + alpha * (qpos[:, :3] - q0[:, :3])
    rel = _rot(q0).inv() * _rot(qpos)
    out[:, 3:7] = _quat(_rot(q0) * Rotation.from_rotvec(alpha * rel.as_rotvec()))
    out[:, 7:] = q0[:, 7:] + alpha * (qpos[:, 7:] - q0[:, 7:])
    out_of = lambda q: (q[:, 7:] < g1.lo) | (q[:, 7:] > g1.hi)  # noqa: E731
    clipped = np.mean(out_of(out) & ~out_of(qpos))  # newly past a limit (the retarget itself grazes some)
    out[:, 7:] = np.clip(out[:, 7:], g1.lo, g1.hi)
    before = g1.bodies(qpos)
    low_b = np.minimum(before[:, L_ANKLE, 2], before[:, R_ANKLE, 2])
    ground = low_b - foot0 < T.DOWN + 0.02
    dz = g1.soles(qpos) - g1.soles(out)
    if ground.any():  # grounded frames keep their contact; in between (flights), interpolate the shift
        dz = np.interp(np.arange(len(qpos)), np.flatnonzero(ground), dz[ground])
    out[:, 2] += dz
    return out, float(clipped)


def regrounded(q, t, g1, sole=None):
    """resample(q, t) with the lowest foot kept at the height the source's has there. Blending two frames'
    joints linearly moves a foot along the chord of its arc: between frames 0.6 rad apart (a capped
    retarget) a planted foot dips 3-4 cm into the floor that neither frame has it in."""
    sole = g1.soles(q) if sole is None else sole
    out = resample(q, t)
    out[:, 2] += np.interp(t, np.arange(len(q)), sole) - g1.soles(out)
    return out


def make(qpos, g1, mirrored, speed, alpha, warp, keys, foot0, seed=0, fps=FPS):
    """One variant of qpos (at fps) -> (qpos, joint-limit share clipped, the source frame of each of its
    frames: what the video was showing then, for the video's latent)."""
    from retarget import MAX_JOINT_SPEED
    rng = np.random.default_rng(seed)
    q = mirror(qpos, g1) if mirrored else qpos
    clipped = 0.0
    src = np.arange(len(qpos), dtype=np.float64)
    keys = np.asarray(keys, np.float64)
    if alpha != 1.0:
        q, clipped = scale(q, alpha, g1, foot0)
        t = ballistic_retime(q, g1, foot0, fps, flights(g1.bodies(qpos), foot0, fps))
        if len(t) != len(q) or np.abs(t - src).max() > 1e-9:
            q, src, keys = regrounded(q, t, g1), t, np.interp(keys, t, np.arange(len(t)))
    joint_speed = np.abs(np.diff(q[:, 7:], axis=0)).max(1) * fps
    if speed != 1.0 or warp or joint_speed.max() > MAX_JOINT_SPEED * SPEED_SLACK:
        air = airborne(g1.bodies(q), foot0)
        t = time_map(len(q), keys, speed, warp, air, rng, joint_speed, MAX_JOINT_SPEED, 1 / max(alpha, 1.0))
        q, src = regrounded(q, t, g1), np.interp(t, np.arange(len(src)), src)
    return q, clipped, src


def physics(q, g1, foot0, fps, ref_flights):
    """Why a variant's reference is not something a body can do: [(rule, reason)] ([] if it is)."""
    from retarget import MAX_JOINT_SPEED
    lo, hi = BALLISTIC
    why = []
    speed = float(np.abs(np.diff(q[:, 7:], axis=0)).max() * fps)
    if speed > MAX_JOINT_SPEED * SPEED_SLACK:
        why.append(("speed", f"a joint at {speed:.1f} rad/s (cap {MAX_JOINT_SPEED:g})"))
    fl = flights(g1.bodies(q), foot0, fps)
    bad = [f for f in fl if not ballistic(f)]
    if bad:
        why.append(("ballistic", "; ".join(
            f"a flight the clip {'starts' if f['start'] == 0 else 'ends'} in: {f['air_s']:.2f} s in the air with a "
            f"{100 * f['rise']:.0f} cm rise, which needs at least {lo:g} x ballistic (it is {f['ratio']:g})" if f["open"] else
            f"flight rise / ballistic {f['ratio']:g} outside [{lo:g}, {hi:g}]" for f in bad)))
    if len(fl) < len(ref_flights):
        why.append(("lost flight", f"{len(fl)} of the clip's {len(ref_flights)} flights left"))
    depth = -float(g1.soles(q).min())
    if depth > PENETRATION:
        why.append(("penetration", f"a foot {100 * depth:.1f} cm into the floor"))
    return why


def name(mirrored, speed, alpha, warp):
    return f"{'m' if mirrored else 'o'}_t{speed:g}_a{alpha:g}" + (f"_w{warp:g}" if warp else "")


def generate(qpos, fps=FPS, grid=None, seed=0, g1=None, log=print):
    """Variants of a clip: [{name, params, qpos, spec, ok, why, rules}] for each (mirror, tempo, amplitude,
    warp) in grid (default TRAIN). spec is the variant's own keypose spec; ok = it may be trained on;
    rules = the filter rules it broke ("self-check", "spec", "limits", "speed", "ballistic", "lost flight",
    "penetration"), why = the same in words."""
    from skill.keypose import compile_keypose_spec
    g1 = g1 or G1()
    qpos = np.asarray(qpos, np.float64)
    pos = g1.bodies(qpos)
    base = compile_keypose_spec(pos, fps)
    keys = [s["frame"] for s in base["stages"] if s["kind"] == "pose"]
    foot0 = float(np.percentile(pos[:, [L_ANKLE, R_ANKLE], 2].min(1), 5))  # the spec's, before it is rounded
    ref_flights = flights(pos, foot0, fps)
    out = []
    for i, (mirrored, speed, alpha, warp) in enumerate(grid or TRAIN):
        q, clipped, src = make(qpos, g1, mirrored, speed, alpha, warp, keys, foot0, seed + i, fps)
        v = {"name": name(mirrored, speed, alpha, warp),
             "params": {"mirror": mirrored, "tempo": speed, "amplitude": alpha, "warp": warp, "seed": seed + i},
             "qpos": q.astype(np.float32), "src_frame": src.astype(np.float32), "frames": len(q),
             "limit_clipped": round(clipped, 4)}
        try:
            p = g1.bodies(q)
            spec = compile_keypose_spec(p, fps, name=v["name"])
            sc = T.evaluate({**spec, "down": spec["down_ref"], "air": spec["air_ref"]}, p, fps)
            spec["self_check"] = sc
            v["spec"] = spec
            why = [] if sc["success"] else [("self-check", f"its own reference completes {sc['stages_completed']}/{sc['stages_total']}"
                                             + (f", stuck at {sc['next_stage']}" if sc["next_stage"] else ""))]
        except ValueError as e:  # too many stages, etc.
            v["spec"], why = None, [("spec", str(e))]
        if clipped > MAX_LIMIT_FRAC:
            why.append(("limits", f"{clipped:.0%} of joint samples beyond their limits"))
        why += physics(q, g1, foot0, fps, ref_flights)
        v["ok"], v["rules"], v["why"] = not why, [r for r, _ in why], "; ".join(w for _, w in why)
        log(f"  {v['name']:22s} {len(q) / fps:5.2f} s  " + (f"{v['spec']['summary'][:70]}" if v["ok"] else f"dropped: {v['why']}"))
        out.append(v)
    return out


def write(variants, out_dir):
    """Each kept variant -> out_dir/<name>/{qpos.npy, src_frame.npy, motion.csv, task_spec.json}; index.json
    lists them all."""
    from retarget import export_mjlab_csv
    out_dir = Path(out_dir)
    index = []
    for v in variants:
        row = {k: v[k] for k in ("name", "params", "frames", "limit_clipped", "ok", "rules", "why")}
        if v["ok"]:
            d = out_dir / v["name"]
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "qpos.npy", v["qpos"])
            np.save(d / "src_frame.npy", v["src_frame"])
            export_mjlab_csv(v["qpos"], d / "motion.csv")
            (d / "task_spec.json").write_text(json.dumps(v["spec"], indent=1))
            row["summary"] = v["spec"]["summary"]
        index.append(row)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip", help="a clip folder from pipeline.py (qpos.npy, meta.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--held-out", action="store_true", help="the held-out grid instead of the training one")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    clip = Path(a.clip)
    fps = json.loads((clip / "meta.json").read_text())["fps"]
    variants = generate(np.load(clip / "qpos.npy"), fps, HELD_OUT if a.held_out else TRAIN, a.seed)
    index = write(variants, a.out)
    drops = {}
    for r in index:
        for rule in r["rules"]:
            drops[rule] = drops.get(rule, 0) + 1
    print(f"{a.out}: {sum(r['ok'] for r in index)}/{len(index)} variants kept" + (f"; dropped for {drops}" if drops else ""))


if __name__ == "__main__":
    main()
