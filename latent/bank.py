"""A bank of varied G1 motions, made with IK. It gives V-JEPA a fixed reference set: the mean
embedding over the bank is what we subtract (shared robot/scene/camera appearance), and the bank
seeds the distillation dataset.

Each generator returns a (frames, 36) qpos array at 30 fps.
"""

import numpy as np

import retarget

FPS = 30


def _rot(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(axis)


def ik_clip(fn, seconds=4.0):
    """fn(rest, t) -> (targets, rotations) for t in [0, 1]; rest = start positions of all frames."""
    rt = retarget.G1Retargeter()
    rest = {k: rt.frame_position(k) for k in retarget.FRAMES}
    t = np.linspace(0, 1, int(seconds * FPS) + 1)
    targets, rotations = zip(*(fn(rest, x) for x in t))
    return rt.retarget_sequence(list(targets), list(rotations))


def cycle(t, reps):
    """0 -> 1 -> 0, `reps` times over t in [0, 1]."""
    return (1 - np.cos(2 * np.pi * reps * t)) / 2


def legs(rest, **moves):
    """Feet and pelvis targets, each optionally offset: legs(rest, pelvis=[0, 0, -0.2])."""
    return {k: rest[k] + np.asarray(moves.get(k, 0.0)) for k in ("pelvis", "left_foot", "right_foot")}


def squat(depth, reps, seconds=4.0):
    return ik_clip(lambda r, t: (legs(r, pelvis=[0, 0, -depth * cycle(t, reps)]), {}), seconds)


def lunge(side):
    foot = f"{side}_foot"
    def fn(r, t):
        s = cycle(t, 1)
        return legs(r, pelvis=[0.2 * s, 0, -0.18 * s], **{foot: [0.45 * s, 0, 0]}), {}
    return ik_clip(fn)


def march(lift, reps=4):
    def fn(r, t):
        s, left = cycle(t, reps), int(t * reps * 2) % 2 == 0
        lifted = [0.05 + 0.1 * lift, 0, lift * s]
        return legs(r, **{"left_foot" if left else "right_foot": lifted}), {}
    return ik_clip(fn)


def kick(side, height):
    foot = f"{side}_foot"
    def fn(r, t):
        s = cycle(t, 2)
        return legs(r, pelvis=[-0.05 * s, 0, -0.05 * s], **{foot: [0.45 * s, 0, height * s]}), {}
    return ik_clip(fn)


def side_shuffle():
    def fn(r, t):
        s = cycle(t, 1) * 0.3
        lead = min(1.0, 2 * cycle(t, 1)) * 0.3
        return legs(r, pelvis=[0, s, 0], left_foot=[0, lead, 0], right_foot=[0, s, 0]), {}
    return ik_clip(fn)


def vertical_jump(height=0.2):
    """Kinematic jump: crouch, extend and leave the ground, land, recover."""
    def fn(r, t):
        if t < 0.3:
            pz, fz = -0.15 * cycle(t / 0.6, 1), 0.0
        elif t < 0.55:
            u = (t - 0.3) / 0.25
            pz, fz = -0.15 + (0.15 + height) * np.sin(np.pi / 2 * u) if u < 0.5 else height * np.sin(np.pi * u), height * np.sin(np.pi * u)
        elif t < 0.8:
            pz, fz = -0.12 * cycle((t - 0.55) / 0.5, 1), 0.0
        else:
            pz, fz = 0.0, 0.0
        return legs(r, pelvis=[0, 0, pz], left_foot=[0, 0, fz], right_foot=[0, 0, fz]), {}
    return ik_clip(fn)


def jumping_jacks(reps=3):
    def fn(r, t):
        s = cycle(t, reps)
        tg = legs(r, left_foot=[0, 0.15 * s, 0], right_foot=[0, -0.15 * s, 0], pelvis=[0, 0, -0.03 * s])
        tg["left_hand"] = r["left_hand"] + s * np.array([0.0, 0.25, 0.75])
        tg["right_hand"] = r["right_hand"] + s * np.array([0.0, -0.25, 0.75])
        return tg, {}
    return ik_clip(fn)


def arms(kind, reps=2):
    def fn(r, t):
        s, tg = cycle(t, reps), legs(r)
        if kind == "raise_front":
            for h in ("left_hand", "right_hand"):
                tg[h] = r[h] + s * np.array([0.45, 0, 0.55])
        elif kind == "raise_side":
            tg["left_hand"] = r["left_hand"] + s * np.array([0, 0.45, 0.45])
            tg["right_hand"] = r["right_hand"] + s * np.array([0, -0.45, 0.45])
        elif kind == "wave":
            tg["right_hand"] = r["right_hand"] + np.array([0.05, -0.15 + 0.15 * (2 * s - 1), 0.7])
        elif kind == "punch":
            left = int(t * reps * 2) % 2 == 0
            h = "left_hand" if left else "right_hand"
            tg[h] = r[h] + s * np.array([0.4, 0, 0.3])
        return tg, {}
    return ik_clip(fn)


def torso(kind, reps=2):
    def fn(r, t):
        s = cycle(t, reps)
        if kind == "bend":
            return legs(r, pelvis=[-0.08 * s, 0, -0.03 * s]), {"torso": _rot("y", 0.8 * s)}
        if kind == "twist":
            return legs(r), {"torso": _rot("z", 0.6 * (2 * s - 1))}
        raise ValueError(kind)
    return ik_clip(fn)


def sway():
    return ik_clip(lambda r, t: (legs(r, pelvis=[0, 0.06 * np.sin(2 * np.pi * 2 * t), 0]), {}))


def motion_bank():
    """name -> qpos (30 fps). About 25 clips, 4 s each."""
    bank = {}
    for depth in (0.1, 0.2, 0.3):
        for reps in (1, 2, 3):
            bank[f"squat_d{int(depth * 100)}_r{reps}"] = squat(depth, reps)
    bank |= {
        "lunge_left": lunge("left"), "lunge_right": lunge("right"),
        "march": march(0.12), "high_knees": march(0.25, reps=6),
        "kick_left": kick("left", 0.3), "kick_right_high": kick("right", 0.45),
        "side_shuffle": side_shuffle(), "vertical_jump": vertical_jump(),
        "jumping_jacks": jumping_jacks(),
        "arms_raise_front": arms("raise_front"), "arms_raise_side": arms("raise_side"),
        "wave": arms("wave", reps=3), "punch": arms("punch", reps=4),
        "bend": torso("bend"), "twist": torso("twist"), "sway": sway(),
        "stand": squat(0.0, 1),
    }
    return bank
