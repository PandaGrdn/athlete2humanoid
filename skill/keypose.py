"""Any action as a task: key moments picked from the retarget by a statistical rule, each turned into
a goal on the body parts that matter for it, plus the contacts (flight, landing) between them. No
list of action types: a squat jump, a front kick, a punch and a lunge all compile the same way.

    python -m skill.keypose outputs/remote_latent_final/clips/front_kick   # -> task_spec.json + self-check

1. Features: a heading-invariant description of the pose in metres (pelvis height, torso and wrists
   relative to the pelvis in its own heading frame, the ankles' horizontal offsets from it, ankle
   heights above the floor) plus how
   far the root has moved and turned since the last key moment.
2. Key moments (key_moments): the vertices of a piecewise-linear simplification of the feature
   trajectory (Ramer-Douglas-Peucker) under a mass-weighted metric, with the tolerance set from the
   clip itself: several sigmas of its pose noise (median absolute deviation), and a fraction of its
   largest excursion, so a punch and a squat jump are each judged at their own scale. A key moment is where the motion stops being
   explained by a straight line between its neighbours: the bottom of a squat, a kick's apex, a
   punch's extension, the return to standing. Flights (both feet off the floor) are found from
   contacts, as in skill/task.py.
   RDP is hierarchical: each key moment's level is the deviation at which it split the path. A loose
   task: the end goal, the contacts and the REQUIRED_TOP moments at the top of the hierarchy are
   required; the finer ones (how this person happened to do it: the in-between crouch, an arm swing)
   are optional, so the policy is paid for them but may skip them. Once done, holding the end pose pays.
3. Goal relevance (relevance): each feature's share of the mass-weighted, noise-corrected motion into
   that key moment. A kick's goal is mostly the kicking ankle; arms waving through a squat move
   little mass and get a loose tolerance or none.
4. Names (name_moment): a deterministic description of what changed most ("right foot up",
   "crouch", "hands forward"), until Muse labels them (muse/skill_reader.py, skill.task.attach_skill).

KeyposeMachine is skill/task.py's StageMachine for this spec: the same API (reset/step/obs), the
same potential-based progress, stage bonuses, upright gating and fall refund, with three stage kinds:
"pose" (the core features each within tolerance, the rest on average, the reference's support feet
on the floor; going further than the key moment along its own motion is free up to OVERSHOOT),
"flight" (both feet off the floor; jump quality as in skill/task.py) and "settle" (the start: feet
down, upright and still; a landing: feet down and upright; the end: the feet that are down at the
clip's end, upright and still).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from skill import task as T

PELVIS, L_HIP, L_ANKLE, R_HIP, R_ANKLE, TORSO, L_WRIST, R_WRIST = 0, 1, 3, 4, 6, 7, 10, 13
YAW_R = 0.3  # m: a turn counts as this radius times its angle (so features share one unit)
# name, mass weight (share of the body the feature moves), minimum tolerance (m)
FEATURES = ([("pelvis height", 1.0, 0.04)]
            + [(f"torso {a}", 0.5, 0.05) for a in "xyz"]
            + [(f"{s} ankle {a}", 0.25, 0.12) for s in ("left", "right") for a in "xy"]  # re-placing a foot isn't a goal
            + [(f"{s} wrist {a}", 0.08, 0.10) for s in ("left", "right") for a in "xyz"]
            + [("left foot height", 0.25, 0.04), ("right foot height", 0.25, 0.04)]
            + [("root forward", 1.0, 0.15), ("root lateral", 1.0, 0.15), ("root turn", 1.0, 0.12)])
D_POSE, D = 16, len(FEATURES)  # the first D_POSE depend on the pose only; the last 3 on an anchor
MASS = torch.tensor([f[1] for f in FEATURES])
TOL_MIN = torch.tensor([f[2] for f in FEATURES])
TOL_FRAC = 0.3  # tolerance as a fraction of the feature's excursion into the key moment
REL_MIN = 0.15  # features below this share of the most relevant one are free at that key moment
RDP_K = 6.0  # RDP tolerance: at least this many noise sigmas of the metric,
RDP_FRAC = 0.3  # and at least this fraction of the clip's largest excursion from its first pose,
RDP_MIN = 0.01  # and never below this (m, mass-weighted RMS)
MIN_GAP_S = 0.2  # key moments closer than this merge
ABSORB_S = 0.5  # s after a touchdown that are the landing (its "settle" stage), not key moments
CORE = 0.5  # features at least this relevant must each be within tolerance; the rest only on average
OVERSHOOT = 0.5  # going further than the key moment along its motion is free up to this share of it
POSE, FLIGHT, SETTLE = range(3)
KINDS = ("pose", "flight", "settle")
HOLD = {POSE: 0.0, SETTLE: 0.3}  # a key pose may be passed through (one control step)
REQUIRED_TOP = 2  # a loose task: the end goal, the contacts (jumps, landings) and the 2 key moments at the
# top of the hierarchy are required; the rest (how this person happened to do it) pay but may be skipped
WEIGHTS = {"progress": 1.0, "stage": 1.0, "jump": 2.0, "done": 4.0,
           "hold": 1.0}  # per s in the finish pose once done: the end goal is a state to keep


def heading(pos):
    """(..., B, 3) -> yaw (rad) of the pelvis's forward direction (perpendicular to the hip axis)."""
    lat = pos[..., L_HIP, :2] - pos[..., R_HIP, :2]
    return torch.atan2(lat[..., 1], lat[..., 0]) - torch.pi / 2


def _to_heading(v, yaw):
    c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    return torch.cat([c * v[..., :1] + s * v[..., 1:2], -s * v[..., :1] + c * v[..., 1:2], v[..., 2:]], -1)


def pose_features(pos, foot0):
    """(..., B, 3) world body positions -> (..., D_POSE) heading-invariant pose features (m)."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    yaw, pel = heading(pos), pos[..., PELVIS, :]
    rel = [_to_heading(pos[..., b, :] - pel, yaw) for b in (TORSO, L_ANKLE, R_ANKLE, L_WRIST, R_WRIST)]
    rel[1], rel[2] = rel[1][..., :2], rel[2][..., :2]  # an ankle's height is pelvis height + foot height
    feet = pos[..., [L_ANKLE, R_ANKLE], 2] - foot0
    return torch.cat([pel[..., 2:3], *rel, feet], -1)


def root_features(pos, anchor_xy, anchor_yaw):
    """Root displacement (forward, lateral, in the anchor's heading frame) and turn since the anchor."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    d = pos[..., PELVIS, :2] - anchor_xy
    c, s = torch.cos(anchor_yaw), torch.sin(anchor_yaw)
    turn = torch.remainder(heading(pos) - anchor_yaw + torch.pi, 2 * torch.pi) - torch.pi
    return torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], YAW_R * turn], -1)


def trajectory_features(pos, foot0):
    """(T, B, 3) -> (T, D): pose features, and root motion relative to the first frame (unwrapped turn)."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    f = pose_features(pos, foot0)
    yaw = torch.as_tensor(np.unwrap(heading(pos).numpy()), dtype=torch.float32)
    r = root_features(pos, pos[0, PELVIS, :2], yaw[0])
    r[:, 2] = YAW_R * (yaw - yaw[0])
    return torch.cat([f, r], -1)


def noise_sigma(F, fps):
    """Per feature, the pose noise: MAD of the residual from a ~0.2 s moving average (robust to the motion)."""
    k = max(3, int(round(0.2 * fps)) | 1)
    res = F - uniform_filter1d(F, k, axis=0, mode="nearest")
    return 1.4826 * np.median(np.abs(res - np.median(res, 0)), 0)


def _metric(x, mass=None):
    mass = MASS.numpy() if mass is None else mass
    return np.sqrt((mass * x ** 2).sum(-1) / MASS.numpy().sum())


# For a key moment's level: where the feet are placed on the floor is left out (a step shows in the
# root's motion, a kick in the foot's height), so re-placing a foot while standing is a detail.
LEVEL_MASS = np.array([0.0 if " ankle " in f[0] else f[1] for f in FEATURES])


def rdp(F, eps, lo=0, hi=None, out=None):
    """Ramer-Douglas-Peucker on the rows of F (the metric above): {kept vertex: its deviation from the
    chord it split}. The deviation is the vertex's level in the hierarchy: the first splits are the
    motion's largest departures from a straight path, the last ones its details."""
    hi = len(F) - 1 if hi is None else hi
    out = {lo: np.inf, hi: np.inf} if out is None else out
    if hi - lo < 2:
        return out
    t = np.linspace(0, 1, hi - lo + 1)[:, None]
    dev = _metric(F[lo:hi + 1] - ((1 - t) * F[lo] + t * F[hi]))
    i = int(np.argmax(dev))
    if dev[i] > eps:
        out[lo + i] = float(dev[i])
        rdp(F, eps, lo, lo + i, out)
        rdp(F, eps, lo + i, hi, out)
    return out


def key_moments(F, fps, air_mask=None, scale=1.0):
    """(T, D) features -> sorted key-moment frames (not counting frame 0), the RDP tolerance and the
    per-feature noise. scale > 1 coarsens the tolerance (compile_keypose_spec, to fit MAX_STAGES)."""
    sig = noise_sigma(F, fps)
    Fs = uniform_filter1d(F, max(1, int(round(0.1 * fps))), axis=0, mode="nearest")
    eps = scale * max(RDP_MIN, RDP_K * float(_metric(sig)), RDP_FRAC * float(_metric(Fs - Fs[0]).max()))
    dev = rdp(Fs, eps)
    ks = sorted(dev)
    if air_mask is not None:  # a flight and its landing are their own stages: no pose goals inside
        ks = [k for k in ks if not air_mask[k]]
    gap, out = MIN_GAP_S * fps, []
    for k in ks[1:]:
        if out and k - out[-1] < gap:  # keep the one further from its predecessor's pose
            prev = out[-2] if len(out) > 1 else 0
            if _metric(Fs[k] - Fs[prev]) > _metric(Fs[out[-1]] - Fs[prev]):
                out[-1] = k
            continue
        out.append(k)
    # A key moment must itself be a significant change from the last one kept (not just a bend in
    # the path): a foot shuffle while standing is a vertex but not a goal.
    kept = []
    for k in out:
        if _metric(Fs[k] - Fs[kept[-1] if kept else 0]) >= eps:
            kept.append(k)
    # Level: deviation from the straight path between its kept neighbours (the hierarchy, measured on
    # the final simplification), in tolerances; the clip's end is always required.
    level, nb = {}, [0, *kept]
    for i, k in enumerate(kept):
        a, b = nb[i], kept[i + 1] if i + 1 < len(kept) else None
        if b is None:
            level[k] = np.inf
            continue
        t = (k - a) / max(b - a, 1)
        level[k] = float(_metric(Fs[k] - ((1 - t) * Fs[a] + t * Fs[b]), LEVEL_MASS)) / eps
    return kept, eps, sig, level


def relevance(delta, sig):
    """Excursion of each feature, weighted by the square root of the mass it moves (so a punch's hand
    still counts next to a weight shift), into a key moment beyond what its noise or its minimum
    tolerance would accept anyway, as a share of the largest one; below REL_MIN -> 0 (free)."""
    e = np.sqrt(MASS.numpy()) * np.maximum(np.abs(delta) - np.maximum(2 * sig, TOL_MIN.numpy()), 0)
    w = e / max(e.max(), 1e-9)
    return np.where(w >= REL_MIN, w, 0.0)


def name_moment(delta, w):
    """What changed most, in words (Muse's stand-in)."""
    words = []
    for i in np.argsort(-w)[:2]:
        if w[i] <= 0:
            continue
        n, d = FEATURES[i][0], delta[i]
        if n == "pelvis height":
            words.append("crouch" if d < 0 else "rise")
        elif n.endswith("foot height"):
            words.append(f"{n.split()[0]} foot {'up' if d > 0 else 'down'}")
        elif n.startswith("root"):
            words.append({"root forward": "step " + ("forward" if d > 0 else "back"),
                          "root lateral": "step " + ("left" if d > 0 else "right"),
                          "root turn": "turn " + ("left" if d > 0 else "right")}[n])
        else:
            part, axis = n.rsplit(" ", 1)
            part = {"torso": "lean" if axis != "z" else "torso", "left ankle": "left foot",
                    "right ankle": "right foot", "left wrist": "left hand", "right wrist": "right hand"}[part]
            way = {"x": ("forward", "back"), "y": ("left", "right"), "z": ("up", "down")}[axis][int(d < 0)]
            words.append(f"{part} {way}" if part != "lean" else f"lean {way}")
    words = list(dict.fromkeys(words))
    return ", ".join(words) if words else "hold"


def compile_keypose_spec(pos, fps, skill=None, max_clearance=T.MAX_CLEARANCE, name=None):
    """pos: (T, bodies, 3) retarget body positions (G1-sized) at fps -> keypose task spec (JSON-able).
    Too many stages for the policy's stage input (MAX_STAGES): the RDP tolerance coarsens until it fits."""
    for i in range(12):
        try:
            return _compile(pos, fps, skill, max_clearance, name, scale=1.25 ** i)
        except _TooMany:
            continue
    raise ValueError(f"more than {T.MAX_STAGES} stages even at a coarse tolerance: trim the clip")


class _TooMany(ValueError):
    pass


def _compile(pos, fps, skill, max_clearance, name, scale):
    pos = np.asarray(pos, np.float32)
    foot0 = float(np.percentile(pos[:, [L_ANKLE, R_ANKLE], 2].min(1), 5))
    z, lo, hi, up = (x.numpy() for x in T.features(pos, foot0))
    ground = lo < 0.10
    h0 = float(np.percentile(z[ground], 75)) if ground.any() else float(np.median(z))
    calm = ground & (np.abs(z - h0) < 0.03)
    down_ref = float(np.clip(np.percentile(hi[calm], 95) + 0.02, T.DOWN, 0.15)) if calm.any() else T.DOWN
    air_ref = max(T.AIR, down_ref + 0.01, 0.10)
    air = T._runs(np.where(lo > air_ref, "air", "ground"), fps, {"air": T.MIN_FLIGHT, "ground": 0.0})
    flights = [(s, e) for lab, s, e in air if lab == "air"]
    air_mask = np.zeros(len(pos), bool)
    for s, e in flights:
        air_mask[max(0, s - int(0.1 * fps)):e + int(ABSORB_S * fps)] = True  # takeoff jitter, landing
    F = trajectory_features(pos, foot0).numpy()
    keys, eps, sig, dev = key_moments(F, fps, air_mask, scale)
    yaw = np.unwrap(heading(torch.as_tensor(pos)).numpy())
    finite = [dev[k] for k in keys if np.isfinite(dev[k])]
    need = sorted(finite, reverse=True)[REQUIRED_TOP - 1] if len(finite) >= REQUIRED_TOP else 0.0

    events = [("pose", k) for k in keys] + [("flight", s, e) for s, e in flights]
    events.sort(key=lambda ev: ev[1])
    stages, prev_k, prev_t = [{"kind": "settle", "start_s": 0.0, "end_s": 0.0, "name": "ready",
                               "feet_down": [True, True]}], 0, 0
    for ev in events:
        if ev[0] == "flight":
            s, e = ev[1], ev[2]
            stages.append({"kind": "flight", "start_s": round(prev_t / fps, 3), "end_s": round(e / fps, 3),
                           "name": "jump", "clearance": round(float(min(lo[s:e].max(), max_clearance)), 3),
                           "ref_clearance": round(float(lo[s:e].max()), 3),
                           "min_air_s": round(float(np.clip(0.5 * (e - s) / fps, 0.05, 0.12)), 3)})
            stages.append({"kind": "settle", "start_s": round(e / fps, 3), "end_s": round(e / fps, 3),
                           "name": "land", "hold_s": 0.1, "still": False, "feet_down": [True, True]})
            prev_k, prev_t = e, e
            continue
        k = ev[1]
        delta = F[k] - F[prev_k]
        w = relevance(delta, sig)
        if not w.any():
            continue
        tol = np.maximum(TOL_MIN.numpy(), TOL_FRAC * np.abs(delta)) / np.sqrt(np.maximum(w, 1e-3))
        # root targets: in the previous key moment's heading frame, as root_features() measures them
        dxy, c, s_ = pos[k, PELVIS, :2] - pos[prev_k, PELVIS, :2], np.cos(yaw[prev_k]), np.sin(yaw[prev_k])
        root = [c * dxy[0] + s_ * dxy[1], -s_ * dxy[0] + c * dxy[1], YAW_R * float(yaw[k] - yaw[prev_k])]
        goal = np.concatenate([F[k, :D_POSE], root])
        # Required down: a foot the reference has planted by the robot's own rule (T.DOWN), which is what the
        # robot is held to. One the reference merely has low (on its toes, a lunge's back foot, hovering up
        # to down_ref) is left free: required by the looser rule, the robot could never meet it (kp lunge:
        # the reference itself scored 13/16 under the robot's thresholds, the task policy stuck there).
        feet_down = [bool(pos[k, a, 2] - foot0 < T.DOWN) for a in (L_ANKLE, R_ANKLE)]
        level = dev[k]
        stages.append({"kind": "pose", "level": round(min(level, 99.0), 2), "optional": bool(level < need), "start_s": round(prev_t / fps, 3), "end_s": round(k / fps, 3),
                       "name": name_moment(delta, w), "goal": np.round(goal, 4).tolist(),
                       "tol": np.round(np.minimum(tol, 9.99), 4).tolist(), "relevance": np.round(w, 3).tolist(),
                       "delta": np.round(np.concatenate([delta[:D_POSE], root]), 4).tolist(),
                       "feet_down": feet_down, "frame": int(k)})
        prev_k, prev_t = k, k
    end_down = [bool(pos[-1, a, 2] - foot0 < T.DOWN) for a in (L_ANKLE, R_ANKLE)]
    if stages[-1]["kind"] == "pose" and len(pos) - 1 - stages[-1]["frame"] < MIN_GAP_S * fps:
        stages[-1] |= {"still": True, "optional": False, "name": stages[-1]["name"] + " (finish)"}  # the end pose is the finish
    else:
        stages.append({"kind": "settle", "start_s": round(prev_t / fps, 3), "end_s": round(len(pos) / fps, 3),
                       "name": "finish", "hold_s": 0.1, "feet_down": end_down if any(end_down) else [True, True]})
    for a, b in zip(stages, stages[1:]):  # strictly increasing starts: stage_at() picks one per time
        b["start_s"] = round(max(b["start_s"], a["start_s"] + 1 / fps), 3)
    if len(stages) > T.MAX_STAGES:
        raise _TooMany(len(stages))
    spec = {"type": "keypose", "name": name, "fps": fps, "duration_s": round(len(pos) / fps, 3),
            "h0": round(h0, 3), "foot0": round(foot0, 3), "down": T.DOWN, "air": T.AIR,
            "down_ref": round(down_ref, 3), "air_ref": round(air_ref, 3), "weights": dict(WEIGHTS),
            "features": [f[0] for f in FEATURES], "rdp_eps": round(eps, 4), "stages": stages,
            "summary": summarize(stages), "warnings": []}
    if skill:
        T.attach_skill(spec, skill)
        for st in spec["stages"]:
            st["name"] = st.get("label", st["name"])
    return spec


def importance(spec, n_frames, fps, floor=0.1, before_s=0.4, after_s=0.15):
    """Goal relevance per reference frame for the style terms (mjlab_tasks.TimeWeighted's
    importance.json): the approach to each key moment counts by its level in the hierarchy (required
    ones fully), take-offs and landings fully, the rest of the clip at `floor`."""
    w = np.zeros(n_frames)
    t = np.arange(n_frames) / fps
    top = max([s["level"] for s in spec["stages"] if s["kind"] == "pose" and s["level"] < 99] or [1.0])
    for s in spec["stages"]:
        if s["kind"] == "pose":
            c, v = s["frame"] / fps, (1.0 if not s.get("optional") else min(1.0, s["level"] / top))
        elif s["kind"] == "flight":
            c, v = s["end_s"], 1.0  # the landing; the take-off is the key moment before it
        else:
            continue
        w = np.maximum(w, v * ((t >= c - before_s) & (t <= c + after_s)))
    w = uniform_filter1d(w, max(1, int(0.1 * fps)), mode="nearest")
    return {"fps": fps, "weight": np.round(np.maximum(w, floor), 3).tolist(), "source": "keypose"}


def summarize(stages):
    poses = [s for s in stages if s["kind"] == "pose"]
    jumps = sum(s["kind"] == "flight" for s in stages)
    head = f"{len(poses)} key moment{'s' * (len(poses) != 1)}" + (f", {jumps} jump{'s' * (jumps != 1)}" if jumps else "")
    return head + ": " + " -> ".join(s["name"] for s in stages)


class KeyposeMachine:
    """Batched reward machine over a keypose spec, with StageMachine's interface (skill/task.py)."""

    def __init__(self, spec, num_envs, device="cpu", dt=0.02, weights=None):
        self.spec, self.n, self.dev, self.dt = spec, num_envs, device, dt
        self.w = {**WEIGHTS, **spec.get("weights", {}), **(weights or {})}
        st = spec["stages"]
        self.S = len(st)
        self.kind = torch.tensor([KINDS.index(s["kind"]) for s in st], device=device)
        zeros = [0.0] * D
        self.goal = torch.tensor([s.get("goal", zeros) for s in st], dtype=torch.float32, device=device)
        self.tol = torch.tensor([s.get("tol", [1.0] * D) for s in st], dtype=torch.float32, device=device)
        self.rel = torch.tensor([s.get("relevance", zeros) for s in st], dtype=torch.float32, device=device)
        delta = torch.tensor([s.get("delta", zeros) for s in st], dtype=torch.float32, device=device)
        self.sign = torch.where(delta < 0, -1.0, 1.0)
        self.span = delta.abs() * (self.rel > 0)
        self.need_down = torch.tensor([s.get("feet_down", [s["kind"] == "settle"] * 2) for s in st], device=device)
        self.still = torch.tensor([s.get("still", s["kind"] == "settle") for s in st], device=device)
        self.optional = torch.tensor([bool(s.get("optional")) and i + 1 < self.S for i, s in enumerate(st)], device=device)
        self.hold_need = torch.tensor([s.get("min_air_s", s.get("hold_s", HOLD.get(KINDS.index(s["kind"]), 0.1)))
                                       for s in st], dtype=torch.float32, device=device)
        self.start = torch.tensor([s["start_s"] for s in st], dtype=torch.float32, device=device)
        self.end = torch.tensor([s["end_s"] for s in st], dtype=torch.float32, device=device)
        self.clear = torch.tensor([s.get("clearance", 0.0) for s in st], dtype=torch.float32, device=device)
        # A settle stage right after a flight pays that jump's quality (skill/task.py's land stage).
        self.jump_goal = torch.tensor([st[i - 1].get("clearance", 0.0) if i and st[i - 1]["kind"] == "flight" else 0.0
                                       for i in range(self.S)], dtype=torch.float32, device=device)
        z = lambda dtype=torch.float32: torch.zeros(num_envs, dtype=dtype, device=device)
        self.stage, self.completed = z(torch.long), z(torch.long)
        self.hold, self.phi, self.apex, self.z_prev, self.p, self.banked = z(), z(), z(), z(), z(), z()
        self.d0 = torch.ones(num_envs, device=device)  # weighted distance to the goal when the stage began
        self.root_scale = torch.ones(num_envs, device=device)  # share of the root goal left (mid-stage starts)
        self.anchor_xy, self.anchor_yaw = torch.zeros(num_envs, 2, device=device), z()
        self.done = z(torch.bool)
        self.stats = {"advances": z(torch.long), "jump_q": z()}
        self.n_stages = torch.full((num_envs,), self.S, dtype=torch.long, device=device)
        self.q_min, self.steps = torch.ones(num_envs, device=device), z(torch.long)
        self.skipped = z(torch.long)
        self.extra_depth = z()  # StageMachine's commanded extra squat depth (mjlab_tasks fall rule): none here
        self.style_until_done = True  # mjlab_tasks.TimeWeighted: no style terms once the task is done

    def _eval(self, stage, pos, vz, ids=slice(None)):
        """Progress in [0, 1] and whether the predicate holds, for envs ids (state buffers indexed by
        ids) in stage, plus the relevance-weighted RMS distance to a pose goal in tolerances."""
        s = self.spec
        _, lo, _, up = T.features(pos, s["foot0"])
        down = (pos[..., [L_ANKLE, R_ANKLE], 2] - s["foot0"]) < s["down"]
        k = self.kind[stage]
        f = torch.cat([pose_features(pos, s["foot0"]), root_features(pos, self.anchor_xy[ids], self.anchor_yaw[ids])], -1)
        g = self.goal[stage].clone()
        g[:, D_POSE:] = g[:, D_POSE:] * self.root_scale[ids][:, None]
        # error along the key moment's motion: short of the goal counts, beyond it only past OVERSHOOT
        e = (g - f) * self.sign[stage]
        e = torch.where(e >= 0, e, (-e - OVERSHOOT * self.span[stage]).clamp(min=0))
        x = e / self.tol[stage]
        r = self.rel[stage]
        worst = torch.where(r >= CORE, x, torch.zeros_like(x)).max(-1).values
        rms = ((r * x ** 2).sum(-1) / r.sum(-1).clamp(min=1e-6)).sqrt()
        d0 = self.d0[ids]
        p_pose = ((d0 - rms) / (d0 - 1).clamp(min=1e-3)).clamp(0, 1)
        support = (~self.need_down[stage] | down).all(-1)
        p = torch.where(k == POSE, p_pose, torch.where(k == FLIGHT, (lo / self.clear[stage].clamp(min=1e-3)).clamp(0, 1),
                        support.float() * ((up - 0.6) / 0.25).clamp(0, 1)))
        calm = ~self.still[stage] | (vz.abs() < 0.5)
        settled = support & (up > T.UPRIGHT) & calm
        sat = torch.where(k == POSE, (worst <= 1) & (rms <= 1) & support & calm, torch.where(k == FLIGHT, lo > s["air"], settled))
        return p, sat, rms

    def _enter(self, ids, pos, frac=None):
        """Stage entry for envs ids (pos: theirs): anchor the root goal here, measure the starting distance."""
        self.anchor_xy[ids], self.anchor_yaw[ids] = pos[:, PELVIS, :2], heading(pos)
        self.root_scale[ids] = 1.0 if frac is None else frac
        self.d0[ids] = 2.0  # placeholder so _eval's progress is defined; replaced just below
        _, _, rms = self._eval(self.stage[ids], pos, torch.zeros(len(pos), device=self.dev), ids)
        self.d0[ids] = rms.clamp(min=1.0 + 1e-3)

    def stage_at(self, t):
        return (torch.searchsorted(self.start, t.float().contiguous(), right=True) - 1).clamp(0, self.S - 1)

    def reset(self, env_ids, t, pos):
        """Episodes of env_ids (re)start at reference time t (s) with bodies at pos (len(env_ids), B, 3).
        Mid-stage, the root goal is scaled to the share of the stage still ahead."""
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self.dev)
        ids = torch.as_tensor(env_ids, device=self.dev)
        t = torch.as_tensor(t, dtype=torch.float32, device=self.dev).expand(len(ids))
        stage = self.stage_at(t)
        # "ready" lasts no time: an episode that starts at the clip's first frame is past it, like one
        # that starts a frame later. Otherwise only exact frame-0 starts see it (~1 in 264 RSI starts in
        # training), and a policy without the retarget as input never learns it (kp_jacks taskphase:
        # every play rollout from frame 0 fell within 5 steps; from frame 1 on, none did).
        ready = (stage == 0) & (self.end[0] <= self.start[0]) if self.S > 1 else torch.zeros_like(stage, dtype=torch.bool)
        stage = torch.where(ready, 1, stage)
        self.stage[ids] = stage
        span = (self.end[stage] - self.start[stage]).clamp(min=1e-3)
        self._enter(ids, pos, ((self.end[stage] - t) / span).clamp(0, 1))
        p, _, _ = self._eval(stage, pos, torch.zeros(len(ids), device=self.dev), ids)
        self.p[ids], self.phi[ids] = p, stage + p
        for buf in (self.hold, self.apex, self.completed, self.steps, self.banked, self.skipped):
            buf[ids] = 0
        self.completed[ids] = ready.to(self.completed.dtype)
        self.q_min[ids] = 1.0
        self.done[ids] = False
        self.z_prev[ids] = pos[:, PELVIS, 2]

    def step(self, pos, fell=None):
        """As StageMachine.step: progress, stage bonuses, jump quality, finish bonus, fall refund."""
        pos = torch.as_tensor(pos, dtype=torch.float32, device=self.dev)
        z = pos[:, PELVIS, 2]
        vz, self.z_prev = (z - self.z_prev) / self.dt, z
        p, sat, _ = self._eval(self.stage, pos, vz)
        hold_end = self.done & sat  # done: the stage stays the last one, whose predicate is the end pose
        # An optional key moment (a detail of how the clip did it) may be skipped: if the next stage
        # already holds, move on to it, without the optional one's bonus.
        opt = (self.optional[self.stage] & ~sat & ~self.done).nonzero().squeeze(1)
        if len(opt):
            saved = (self.stage[opt].clone(), self.anchor_xy[opt].clone(), self.anchor_yaw[opt].clone(),
                     self.root_scale[opt].clone(), self.d0[opt].clone())
            self.stage[opt] += 1
            self._enter(opt, pos[opt])
            p2, sat2, _ = self._eval(self.stage[opt], pos[opt], vz[opt], opt)
            back = ~sat2
            for buf, v in zip((self.stage, self.anchor_xy, self.anchor_yaw, self.root_scale, self.d0), saved):
                buf[opt[back]] = v[back]
            skip = opt[sat2]
            if len(skip):
                p, sat = p.clone(), sat.clone()
                p[skip], sat[skip] = p2[sat2], True
                self.hold[skip] = 0.0
                self.skipped[skip] += 1
                self.phi[skip] = self.phi[skip] + 1.0  # the skipped stage is behind it: no progress reward for it
        _, lo, _, _ = T.features(pos, self.spec["foot0"])
        k = self.kind[self.stage]
        after_flight = self.jump_goal[self.stage] > 0
        self.apex = torch.where((k == FLIGHT) | after_flight, torch.maximum(self.apex, lo), torch.zeros_like(lo))
        self.hold = torch.where(sat & ~self.done, self.hold + self.dt, torch.zeros_like(self.hold))
        adv = sat & (self.hold >= self.hold_need[self.stage] - 1e-6) & ~self.done

        jg = self.jump_goal[self.stage]
        ratio = self.apex / jg.clamp(min=1e-3)
        q = torch.where(jg > 0, ratio.clamp(0, 1) - ((ratio - 1.25) / 0.75).clamp(0, 1), torch.zeros_like(ratio))
        bonus = adv * (self.w["stage"] + self.w["jump"] * q)

        self.completed += adv.long()
        new_stage = self.stage + adv.long()
        just_done = new_stage >= self.n_stages
        self.done |= just_done
        self.stage = torch.minimum(new_stage, self.n_stages - 1)
        moved = (adv & ~just_done).nonzero().squeeze(1)
        if len(moved):
            self._enter(moved, pos[moved])
            p = p.clone()
            p[moved] = 0.0
        self.p = torch.where(adv, torch.zeros_like(p), p)
        self.hold = torch.where(adv, torch.zeros_like(self.hold), self.hold)
        bonus = bonus + just_done * self.w["done"] * self.completed / self.n_stages

        phi = torch.where(self.done, self.n_stages.float(), self.stage + self.p)
        d, self.phi = phi - self.phi, phi
        _, _, _, up = T.features(pos, self.spec["foot0"])
        gate = ((up - 0.6) / 0.3).clamp(0, 1)
        prog = torch.where(d > 0, d * gate, d) * self.w["progress"]
        self.stats["advances"] += adv.long()
        self.stats["jump_q"] = torch.where(adv & (jg > 0), q, self.stats["jump_q"])
        self.q_min = torch.where(adv & (jg > 0), torch.minimum(self.q_min, q), self.q_min)
        self.steps += 1
        r = prog + bonus + hold_end * self.w["hold"] * self.dt
        self.banked = self.banked + r
        if fell is not None:
            r = torch.where(fell, r - self.banked, r)
            self.banked = torch.where(fell, torch.zeros_like(r), self.banked)
        rep_done = (adv & (k == SETTLE)) | just_done
        self.banked = torch.where(rep_done, torch.zeros_like(r), self.banked)
        return r / self.dt

    def obs(self):
        one = torch.nn.functional.one_hot(self.stage, T.MAX_STAGES).float()
        return torch.cat([one, self.p[:, None], self.done.float()[:, None]], -1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip", help="a clip folder from pipeline.py (qpos.npy, meta.json; skill_spec.json optional)")
    ap.add_argument("--out", help="default <clip>/task_spec.json")
    ap.add_argument("--skill", help="Muse skill_spec.json (default <clip>/skill_spec.json if present)")
    a = ap.parse_args(argv)
    clip = Path(a.clip)
    fps = json.loads((clip / "meta.json").read_text())["fps"]
    pos = T.body_positions_from_qpos(np.load(clip / "qpos.npy"))
    skill_path = Path(a.skill) if a.skill else clip / "skill_spec.json"
    skill = json.loads(skill_path.read_text()) if skill_path.exists() else None
    spec = compile_keypose_spec(pos, fps, skill, name=clip.name)
    spec["self_check"] = T.evaluate({**spec, "down": spec["down_ref"], "air": spec["air_ref"]}, pos, fps)
    out = Path(a.out or clip / "task_spec.json")
    out.write_text(json.dumps(spec, indent=1))
    sc = spec["self_check"]
    print(f"{out}: {spec['summary']}; retarget completes {sc['stages_completed']}/{sc['stages_total']}"
          + "".join(f"\n  warning: {w}" for w in spec["warnings"]))


if __name__ == "__main__":
    main()
