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
- amplitude: every joint, the root's height and travel, and its turn scaled about the clip's first
  frame (alpha 0.85 = a smaller version of the same motion: a shallower squat, a lower kick). The
  root is then shifted vertically so the feet touch the floor where the original's did.

Each variant gets its own keypose spec, compiled by the same rule as any clip (skill/keypose.py), and
is kept only if its own reference completes that spec (self-check) and stays within the joint limits.
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


def time_map(n, keys, speed, warp, air, rng):
    """Source frame for each output frame. Between consecutive key frames the clip plays at `speed`
    times a random factor in [1 - warp, 1 + warp]; airborne frames always play at 1x."""
    knots = sorted({0, *[int(k) for k in keys if 0 < k < n - 1], n - 1})
    seg_speed = np.ones(n - 1)
    for a, b in zip(knots, knots[1:]):
        seg_speed[a:b] = speed * (1 + rng.uniform(-warp, warp) if warp else 1.0)
    seg_speed = np.where(air[:-1] & air[1:], 1.0, seg_speed)  # gravity sets a flight's duration
    out_len = np.concatenate([[0.0], np.cumsum(1.0 / seg_speed)])  # output frames per source frame
    m = int(np.floor(out_len[-1])) + 1
    return np.interp(np.arange(m), out_len, np.arange(n))


def scale(qpos, alpha, g1, foot0):
    """Amplitude alpha about the first frame; then the feet put back on the floor where they were."""
    q0 = qpos[:1]
    out = qpos.copy()
    out[:, :3] = q0[:, :3] + alpha * (qpos[:, :3] - q0[:, :3])
    rel = _rot(q0).inv() * _rot(qpos)
    out[:, 3:7] = _quat(_rot(q0) * Rotation.from_rotvec(alpha * rel.as_rotvec()))
    out[:, 7:] = q0[:, 7:] + alpha * (qpos[:, 7:] - q0[:, 7:])
    out_of = lambda q: (q[:, 7:] < g1.lo) | (q[:, 7:] > g1.hi)  # noqa: E731
    clipped = np.mean(out_of(out) & ~out_of(qpos))  # newly past a limit (the retarget itself grazes some)
    out[:, 7:] = np.clip(out[:, 7:], g1.lo, g1.hi)
    before, after = g1.bodies(qpos), g1.bodies(out)
    low_b = np.minimum(before[:, L_ANKLE, 2], before[:, R_ANKLE, 2])
    low_a = np.minimum(after[:, L_ANKLE, 2], after[:, R_ANKLE, 2])
    ground = low_b - foot0 < T.DOWN + 0.02
    dz = low_b - low_a
    if ground.any():  # grounded frames keep their contact; in between (flights), interpolate the shift
        dz = np.interp(np.arange(len(qpos)), np.flatnonzero(ground), dz[ground])
    out[:, 2] += dz
    return out, float(clipped)


def make(qpos, g1, mirrored, speed, alpha, warp, keys, foot0, seed=0):
    """One variant of qpos (FPS) -> (qpos, joint-limit share clipped, the source frame of each of its
    frames: what the video was showing then, for the video's latent)."""
    rng = np.random.default_rng(seed)
    q = mirror(qpos, g1) if mirrored else qpos
    clipped = 0.0
    src = np.arange(len(qpos), dtype=np.float64)
    if alpha != 1.0:
        q, clipped = scale(q, alpha, g1, foot0)
    if speed != 1.0 or warp:
        air = airborne(g1.bodies(q), foot0)
        src = time_map(len(q), keys, speed, warp, air, rng)
        q = resample(q, src)
    return q, clipped, src


def name(mirrored, speed, alpha, warp):
    return f"{'m' if mirrored else 'o'}_t{speed:g}_a{alpha:g}" + (f"_w{warp:g}" if warp else "")


def generate(qpos, fps=FPS, grid=None, seed=0, g1=None, log=print):
    """Variants of a clip: [{name, params, qpos, spec, ok, why}] for each (mirror, tempo, amplitude,
    warp) in grid (default TRAIN). spec is the variant's own keypose spec; ok = it may be trained on."""
    from skill.keypose import compile_keypose_spec
    g1 = g1 or G1()
    qpos = np.asarray(qpos, np.float64)
    pos = g1.bodies(qpos)
    base = compile_keypose_spec(pos, fps)
    keys = [s["frame"] for s in base["stages"] if s["kind"] == "pose"]
    out = []
    for i, (mirrored, speed, alpha, warp) in enumerate(grid or TRAIN):
        q, clipped, src = make(qpos, g1, mirrored, speed, alpha, warp, keys, base["foot0"], seed + i)
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
            why = [] if sc["success"] else [f"its own reference completes {sc['stages_completed']}/{sc['stages_total']}"]
        except ValueError as e:  # too many stages, etc.
            v["spec"], why = None, [str(e)]
        if clipped > MAX_LIMIT_FRAC:
            why.append(f"{clipped:.0%} of joint samples beyond their limits")
        v["ok"], v["why"] = not why, "; ".join(why)
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
        row = {k: v[k] for k in ("name", "params", "frames", "limit_clipped", "ok", "why")}
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
    print(f"{a.out}: {sum(r['ok'] for r in index)}/{len(index)} variants kept")


if __name__ == "__main__":
    main()
