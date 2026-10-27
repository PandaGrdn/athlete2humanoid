"""Goal relevance: per reference frame (and per body group), how much a moment of the clip matters
for achieving its task stages (skill/task.py), so the style terms imitate what serves the goal
(the arm swing before a takeoff) and not what doesn't (waving to the crowd seconds earlier).

    python -m skill.importance outputs/remote_latent/clips/squat_jump    # -> importance.json + .png
    python -m skill.importance <clip> --hype 0.3 1.6     # the same with arm-waving injected (validation)
    python -m skill.importance <clip> --vjepa auto       # + V-JEPA 2 predictor masking (needs the weights)

Two attributions, combined into importance.json {"fps", "weight": [per frame, floor..1],
"body_weight": {group: [per frame]}, ...}, which mjlab_tasks.TimeWeighted reads (A2H_IMPORTANCE):

- Physics (always; cheap and causal). Every goal event of the task spec (squat bottom, takeoff,
  apex, touchdown, settling) has a goal quantity that is a sum over body groups: the centre of
  mass's height (descent), its velocity along the takeoff direction (propulsion), the feet's
  clearance below a ballistic CoM (flight), its vertical velocity (absorbing the landing), its
  horizontal offset from the feet (balance). Its change over the event's causal window then splits
  exactly into per-frame, per-group increments (completeness, as in integrated gradients). Each
  group-frame is credited with the part that moved the quantity toward the goal, summed over a
  short window first so back-and-forth motion (waving) cancels. Frames outside every causal window
  (the stand before the countermovement) get no credit, whatever the arms do there.
- World model (optional, latent/predictor_relevance.py): V-JEPA 2's predictor predicts the latent of
  each goal event from the ~1 s of video before it; a window (or one body group's motion in it) is
  removed by interpolation and re-rendered, and the increase in prediction error is its relevance.

Combined: physics and V-JEPA scores (each scaled to [0, 1]) are averaged with weight --vjepa-mix,
smoothed (Gaussian, --smooth s), and lifted to a floor (--floor, default 0.1) so style still
regularises a little everywhere. See docs/GOAL_RELEVANCE.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from skill import task as skill_task

GROUPS = ("left_leg", "right_leg", "torso", "left_arm", "right_arm")
# qpos columns of each group's joints (G1: base 0:7, legs 7:19, waist 19:22, arms 22:36)
QPOS_GROUPS = {"left_leg": slice(7, 13), "right_leg": slice(13, 19), "torso": slice(19, 22),
               "left_arm": slice(22, 29), "right_arm": slice(29, 36)}
# Events: how much each kind counts toward the frame weight (the stage machine pays the jump most).
EVENT_WEIGHT = {"propulsion": 1.0, "absorb": 1.0, "flight": 0.6, "descent": 0.6, "balance": 0.4}
CANCEL_S = 0.27  # s: signed credit is summed over this window before taking its positive part
FLOOR, SMOOTH_S, VJEPA_MIX = 0.1, 0.07, 0.35


def group_of(body_name):
    for side in ("left", "right"):
        if body_name.startswith(side):
            return f"{side}_leg" if any(k in body_name for k in ("hip", "knee", "ankle")) else f"{side}_arm"
    return "torso"  # pelvis, waist, torso


# ---------------------------------------------------------------- kinematics

def kinematics(model, qpos):
    """(T, 36) G1 qpos -> {com (T, nb, 3) body CoMs, mass (nb,), group (nb,) index into GROUPS,
    feet (T, 2, 3) ankles, pelvis (T, 3), pos (T, 14, 3) skill/task.py's tracking bodies}."""
    import mujoco
    from latent.reward_net import BODIES
    data = mujoco.MjData(model)
    bodies = list(range(1, model.nbody))
    feet = [model.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    track = [model.body(b).id for b in BODIES]
    T = len(qpos)
    com, ft, pos = np.empty((T, len(bodies), 3)), np.empty((T, 2, 3)), np.empty((T, len(track), 3), np.float32)
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        com[t], ft[t], pos[t] = data.xipos[bodies], data.xpos[feet], data.xpos[track]
    group = np.array([GROUPS.index(group_of(model.body(b).name)) for b in bodies])
    return {"com": com, "mass": model.body_mass[bodies].copy(), "group": group, "feet": ft,
            "pelvis": com[:, 0], "pos": pos}


# ---------------------------------------------------------------- goal events

def goal_events(spec, pos, fps):
    """The task spec's stages -> goal events with their causal windows (frames, inclusive start,
    exclusive end): [{kind, frame, window, stage, label}]."""
    T = len(pos)
    z, lo, _, _ = (x.numpy() for x in skill_task.features(pos, spec["foot0"]))
    h0 = spec["h0"]
    fr = lambda s: int(np.clip(round(s * fps), 0, T - 1))
    ev, stages = [], spec["stages"]
    bottom_before = {}
    for i, st in enumerate(stages):
        a, b = fr(st["start_s"]), max(fr(st["end_s"]), fr(st["start_s"]) + 1)
        label = st.get("label", st["kind"])
        if st["kind"] == "squat":
            bot = a + int(np.argmin(z[a:b + 1]))
            on = bot
            while on > 0 and z[on - 1] < h0 - 0.03 * h0:  # back to where the descent left standing height
                on -= 1
            on = max(0, on - int(0.1 * fps))
            ev.append({"kind": "descent", "frame": bot, "window": (on, bot + 1), "stage": i, "label": label})
            ev.append({"kind": "balance", "frame": bot, "window": (on, min(b + 1, T)), "stage": i, "label": label})
            bottom_before[i + 1] = bot
        elif st["kind"] == "jump":
            to = a
            start = bottom_before.get(i, max(0, to - int(0.4 * fps)))
            apex = a + int(np.argmax(lo[a:b + 1]))
            ev.append({"kind": "propulsion", "frame": to, "window": (start, to + 1), "stage": i, "label": label})
            ev.append({"kind": "flight", "frame": apex, "window": (to, apex + 1), "stage": i, "label": label})
        elif st["kind"] == "land":
            td = a
            settle = min(b + 1, T)
            bot = td + int(np.argmin(z[td:settle])) if settle > td else td
            ev.append({"kind": "absorb", "frame": bot, "window": (max(0, td - 2), bot + 1), "stage": i, "label": label})
            ev.append({"kind": "balance", "frame": settle - 1, "window": (bot, settle), "stage": i, "label": label})
    return ev


# ---------------------------------------------------------------- physics attribution

def _group_sum(x, group, n=len(GROUPS)):
    """(T, nb, ...) per body -> (T, n, ...) per group."""
    return np.stack([x[:, group == g].sum(1) for g in range(n)], 1)


def group_quantities(kin, fps, air=0.10, sigma=1.5):
    """Per-frame, per-group increments of the goal quantities (T, G, ...), credited to the group
    that causes them rather than the mass that moves. The CoM is the pelvis plus each group's
    mass-weighted offset from it: c = p + sum_g (m_g / M)(c_g - p). Each group owns its offset
    (an arm swung up raises the CoM relative to the body); the pelvis's own motion is the stance
    legs' doing (split between the feet within `air` m of the ground) and nobody's in flight.
    -> {dpos, dvel (T, G, 3), dclear (T, G), doffset (T, G, 2), com_vel (T, 3)}."""
    com = gaussian_filter1d(kin["com"], sigma, axis=0, mode="nearest")
    feet = gaussian_filter1d(kin["feet"], sigma, axis=0, mode="nearest")
    pel = gaussian_filter1d(kin["pelvis"], sigma, axis=0, mode="nearest")
    m, g = kin["mass"], kin["group"]
    w = (m / m.sum())[None, :, None]
    rel = _group_sum((com - pel[:, None]) * w, g)          # (T, G, 3) offsets of the CoM from the pelvis
    rel_v = np.gradient(rel, 1.0 / fps, axis=0)
    pel_v = np.gradient(pel, 1.0 / fps, axis=0)
    legs = [GROUPS.index("left_leg"), GROUPS.index("right_leg")]
    h = feet[..., 2] - np.percentile(feet[..., 2].min(1), 5)
    stance = (h < air).astype(float)
    stance /= np.maximum(stance.sum(1, keepdims=True), 1)  # (T, 2), rows sum to 1 or 0 (flight)
    d = lambda x: np.diff(x, axis=0, prepend=x[:1])
    dpos, dvel = d(rel), d(rel_v)
    for side, k in enumerate(legs):
        dpos[:, k] += stance[:, side, None] * d(pel)
        dvel[:, k] += stance[:, side, None] * d(pel_v)
    # foot clearance below the (ballistic) CoM: mean feet z - c_z = sum_legs 0.5 (foot - p)_z - sum_g rel_z
    clear = -rel[..., 2].copy()
    # horizontal offset of the CoM from the support centre: sum_legs 0.5 (p - foot)_xy + sum_g rel_xy
    off = rel[..., :2].copy()
    for side, k in enumerate(legs):
        clear[:, k] += 0.5 * (feet[:, side, 2] - pel[:, 2])
        off[:, k] += 0.5 * (pel[:, :2] - feet[:, side, :2])
    return {"dpos": dpos, "dvel": dvel, "dclear": d(clear), "offset": off, "doffset": d(off),
            "com_vel": pel_v + rel_v.sum(1)}


def event_credit(q, ev, fps):
    """One event -> per-frame, per-group signed credit (T, G): increments of its goal quantity
    (toward the goal is positive). physics_scores keeps only the event's causal window."""
    k = ev["kind"]
    if k == "descent":  # lower the CoM
        cred = -q["dpos"][..., 2]
    elif k == "propulsion":  # CoM velocity along its takeoff direction
        u = q["com_vel"][ev["frame"]]
        u = u / max(np.linalg.norm(u), 1e-6) if u[2] > 0.1 else np.array([0.0, 0.0, 1.0])
        cred = q["dvel"] @ u
    elif k == "flight":  # feet up under the ballistic CoM
        cred = q["dclear"]
    elif k == "absorb":  # brake the downward CoM velocity
        cred = q["dvel"][..., 2]
    elif k == "balance":  # bring the CoM over the feet: -d|offset| ~ -<d offset_g, offset / |offset|>
        tot = q["offset"].sum(1)
        u = tot / np.maximum(np.linalg.norm(tot, axis=-1, keepdims=True), 1e-3)
        cred = -(q["doffset"] * u[:, None]).sum(-1)
    else:
        raise ValueError(k)
    return cred


def physics_scores(kin, events, fps, cancel_s=CANCEL_S):
    """-> (scores (T, G) >= 0, per-event summary). Each event's positive credit (after summing the
    signed credit over cancel_s, so oscillations cancel) is normalised to sum to its event weight,
    spread over its window: a short decisive push gets more per frame than a long settling."""
    q = group_quantities(kin, fps)
    T = len(q["dclear"])
    S = np.zeros((T, len(GROUPS)))
    summary = []
    n = max(1, int(round(cancel_s * fps)))
    for ev in events:
        full = event_credit(q, ev, fps)
        a, b = ev["window"]
        c = np.zeros_like(full)
        c[a:b] = full[a:b]
        net = uniform_filter1d(full, n, axis=0, mode="nearest") * n  # signed sum over the cancel window
        pos = np.clip(net, 0, None)
        pos[:a], pos[b:] = 0, 0
        tot = pos.sum()
        if tot <= 1e-9:
            continue
        # Coherence: how much of the credited motion actually moved the goal quantity on net. ~1 for
        # a push or a braking; ~0 when the reference just holds a balanced pose (its sway is noise).
        coh = float(np.clip(c.sum() * n / tot, 0, 1))  # pos sums each frame n times
        S += coh * EVENT_WEIGHT[ev["kind"]] * pos / tot * (b - a) / fps  # per-frame density ~ weight
        share = pos.sum(0) / tot
        summary.append({"kind": ev["kind"], "label": ev["label"], "stage": ev["stage"],
                        "t_s": round(ev["frame"] / fps, 3), "window_s": [round(a / fps, 3), round(b / fps, 3)],
                        "goal_change": round(float(c.sum()), 4), "coherence": round(coh, 3),
                        "group_share": {g: round(float(s), 3) for g, s in zip(GROUPS, share)}})
    return S, summary


# ---------------------------------------------------------------- combining

def _unit_scale(x, q=98):
    ref = np.percentile(x[x > 0], q) if (x > 0).any() else 1.0
    return np.clip(x / max(ref, 1e-9), 0, 1)


def combine(phys, vjepa=None, vjepa_mix=VJEPA_MIX, floor=FLOOR, smooth_s=SMOOTH_S, fps=30.0):
    """Per-group scores (T, G) from physics (and optionally V-JEPA) -> (weight (T,) in [floor, 1],
    body_weight (T, G) in [floor, 1]). Frame importance = the group total scaled to [0, 1]; a group's
    weight = frame importance x its share relative to the most relevant group at that frame."""
    def norm(S):
        S = gaussian_filter1d(S, max(smooth_s * fps, 1e-3), axis=0, mode="nearest")
        frame = _unit_scale(S.sum(1))
        rel = S / np.maximum(S.max(1, keepdims=True), 1e-12)
        return frame, rel
    f, r = norm(phys)
    if vjepa is not None:
        fv, rv = norm(vjepa)
        f, r = (1 - vjepa_mix) * f + vjepa_mix * fv, (1 - vjepa_mix) * r + vjepa_mix * rv
    weight = floor + (1 - floor) * f
    body = floor + (1 - floor) * f[:, None] * r
    return weight, body


# ---------------------------------------------------------------- synthetic hype (validation)

def inject_hype(qpos, start_s, end_s, fps=30.0, freq=2.0, ramp_s=0.25):
    """Both arms raised overhead and waved side to side ("hype the crowd") between start_s and
    end_s, blended in and out over ramp_s; everything else unchanged."""
    q = np.array(qpos, dtype=float, copy=True)
    t = np.arange(len(q)) / fps
    a = np.clip(np.minimum(t - start_s, end_s - t) / ramp_s, 0, 1)
    a = a * a * (3 - 2 * a)  # smoothstep
    wave = np.sin(2 * np.pi * freq * (t - start_s))
    for side, col in ((1, 22), (-1, 29)):  # left arm columns 22:29, right 29:36
        target = q[:, col:col + 7].copy()
        target[:, 0] = -2.6                          # shoulder pitch: arm up overhead
        target[:, 1] = side * (0.35 + 0.35 * wave)   # shoulder roll: sway out and in
        target[:, 3] = 0.6 + 0.5 * wave              # elbow pumping
        q[:, col:col + 7] = (1 - a[:, None]) * q[:, col:col + 7] + a[:, None] * target
    return q


def hype_reproduction(policy_qpos, hype_qpos, clean_qpos, window, fps=30.0):
    """How much of the injected arm-waving a policy reproduces, in [0, 1] (can exceed 1): over the
    hype window, arm-joint distance to the clean reference relative to the hype's own distance from
    it (1 = copies the hype, 0 = arms as without it). All at the same fps, starting at frame 0."""
    a, b = int(window[0] * fps), int(window[1] * fps)
    n = min(len(policy_qpos), len(hype_qpos), len(clean_qpos), b)
    arms = slice(22, 36)
    p, h, c = (np.asarray(x)[a:n, arms] for x in (policy_qpos, hype_qpos, clean_qpos))
    proj = ((p - c) * (h - c)).sum() / max(((h - c) ** 2).sum(), 1e-9)  # projection onto the hype direction
    return float(proj)


# ---------------------------------------------------------------- output

def analyse(model, qpos, spec, fps, vjepa=None, floor=FLOOR, smooth_s=SMOOTH_S, vjepa_mix=VJEPA_MIX):
    kin = kinematics(model, qpos)
    events = goal_events(spec, kin["pos"], fps)
    phys, summary = physics_scores(kin, events, fps)
    weight, body = combine(phys, vjepa, vjepa_mix, floor, smooth_s, fps)
    return {"fps": fps, "weight": [round(float(x), 4) for x in weight],
            "body_weight": {g: [round(float(x), 4) for x in body[:, i]] for i, g in enumerate(GROUPS)},
            "groups": {g: f"qpos[{QPOS_GROUPS[g].start}:{QPOS_GROUPS[g].stop}]" for g in GROUPS},
            "floor": floor, "smooth_s": smooth_s, "vjepa_mix": vjepa_mix if vjepa is not None else 0.0,
            "sources": ["physics"] + (["vjepa_predictor"] if vjepa is not None else []),
            "events": summary, "physics_raw": phys, "vjepa_raw": vjepa}


def summarize(res, spec, extra_windows=()):
    """Mean weight per stage (and per extra (name, start_s, end_s) window) for reports."""
    w, fps = np.array(res["weight"]), res["fps"]
    out = {}
    for i, st in enumerate(spec["stages"]):
        a, b = int(st["start_s"] * fps), max(int(st["end_s"] * fps), int(st["start_s"] * fps) + 1)
        out[f"{i}:{st['kind']}"] = {"t": [st["start_s"], st["end_s"]], "mean": round(float(w[a:b].mean()), 3),
                                    "arms": round(float(np.mean([np.array(res["body_weight"][g])[a:b].mean()
                                                                 for g in ("left_arm", "right_arm")])), 3),
                                    "legs": round(float(np.mean([np.array(res["body_weight"][g])[a:b].mean()
                                                                 for g in ("left_leg", "right_leg")])), 3)}
    for name, s, e in extra_windows:
        a, b = int(s * fps), int(e * fps)
        out[name] = {"t": [s, e], "mean": round(float(w[a:b].mean()), 3),
                     "arms": round(float(np.mean([np.array(res["body_weight"][g])[a:b].mean()
                                                  for g in ("left_arm", "right_arm")])), 3)}
    return out


def save(res, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in res.items() if not k.endswith("_raw")}, indent=1))


def plot(res, spec, path, title="", hype=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fps, w = res["fps"], np.array(res["weight"])
    t = np.arange(len(w)) / fps
    body = np.array([res["body_weight"][g] for g in GROUPS])
    fig, (ax, hm) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    colors = {"stand": "#e8e8e8", "squat": "#cfe3f7", "jump": "#f9d4c6", "land": "#d8efd2"}
    for st in spec["stages"]:
        ax.axvspan(st["start_s"], st["end_s"], color=colors[st["kind"]], lw=0)
        ax.text((st["start_s"] + st["end_s"]) / 2, 1.04, st.get("label", st["kind"]), ha="center", va="bottom",
                fontsize=7, color="#555")
    if hype:
        ax.axvspan(*hype, ymin=0, ymax=0.06, color="#c0392b", lw=0)
        ax.text(hype[0], 0.07, "injected arm-waving", fontsize=7, color="#c0392b")
    ax.plot(t, w, color="#1f4e79", lw=1.8, label="frame weight (style terms)")
    for ev in res["events"]:
        if ev["kind"] in ("propulsion", "absorb"):
            ax.axvline(ev["t_s"], color="#333", lw=0.6, ls=":")
    ax.axhline(res["floor"], color="#999", lw=0.6, ls="--")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("importance")
    ax.set_title(title or f"Goal relevance: {res.get('clip', spec.get('name', ''))} ({spec['summary']}; "
                 f"{' + '.join(res['sources'])})", fontsize=9)
    ax.legend(loc="center left", fontsize=7, frameon=False)
    im = hm.imshow(body, aspect="auto", origin="upper", cmap="viridis", vmin=0, vmax=1,
                   extent=(0, len(w) / fps, len(GROUPS) - 0.5, -0.5), interpolation="nearest")
    hm.set_yticks(range(len(GROUPS)), [g.replace("_", " ") for g in GROUPS], fontsize=8)
    hm.set_xlabel("time (s)")
    fig.colorbar(im, ax=[ax, hm], shrink=0.5, pad=0.01, label="body-group weight")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip", help="a clip folder (qpos.npy, meta.json; task_spec.json made if missing)")
    ap.add_argument("--spec", help="task_spec.json (default <clip>/task_spec.json, compiled if absent)")
    ap.add_argument("--out", help="output folder (default outputs/goal_relevance/<clip>[_hype])")
    ap.add_argument("--hype", nargs=2, type=float, metavar=("START_S", "END_S"),
                    help="inject arm-waving between these times first (the validation case)")
    ap.add_argument("--vjepa", default="off", help="'off', 'auto' (run latent/predictor_relevance.py), "
                                                   "or a vjepa_scores.npz from an earlier run")
    ap.add_argument("--device", default=None)
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--smooth", type=float, default=SMOOTH_S)
    ap.add_argument("--vjepa-mix", type=float, default=VJEPA_MIX)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config
    import render
    clip = Path(a.clip)
    fps = json.loads((clip / "meta.json").read_text())["fps"]
    qpos = np.load(clip / "qpos.npy")
    model = render.load_g1()[0]
    spec_f = Path(a.spec) if a.spec else clip / "task_spec.json"
    if spec_f.exists():
        spec = json.loads(spec_f.read_text())
    else:  # the clean clip's spec: the hype doesn't change the stages (tests/test_skill_task checks)
        spec = skill_task.compile_spec(kinematics(model, qpos)["pos"], fps, name=clip.name)
    name = clip.name + ("_hype" if a.hype else "")
    out = Path(a.out) if a.out else config.OUTPUT_DIR / "goal_relevance" / name
    out.mkdir(parents=True, exist_ok=True)
    if a.hype:
        qpos = inject_hype(qpos, *a.hype, fps=fps)
        np.save(out / "qpos.npy", qpos)
    vjepa = None
    if a.vjepa == "auto":
        from latent import predictor_relevance as pr
        kin = kinematics(model, qpos)
        events = goal_events(spec, kin["pos"], fps)
        vjepa, detail = pr.scores(model, qpos, events, fps, azimuth=pr.azimuth_of(clip), device=a.device)
        np.savez(out / "vjepa_scores.npz", scores=vjepa, **detail)
    elif a.vjepa != "off":
        vjepa = np.load(a.vjepa)["scores"]
    res = analyse(model, qpos, spec, fps, vjepa, a.floor, a.smooth, a.vjepa_mix)
    res["clip"], res["hype_s"] = name, a.hype
    res["stage_summary"] = summarize(res, spec, [("hype", *a.hype)] if a.hype else ())
    save(res, out / "importance.json")
    (out / "task_spec.json").write_text(json.dumps(spec, indent=1))
    plot(res, spec, out / "importance.png", hype=a.hype)
    print(f"{out / 'importance.json'}: sources {res['sources']}")
    for k, v in res["stage_summary"].items():
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
