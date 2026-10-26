"""Feasibility probe: does V-JEPA 2 (Meta's video world model) put "the same action, done
differently" closer to a reference than "a different action"?

All clips are the G1 rendered identically, so only the motion differs:
  same action:  shallower squat, faster squat, the trained policy's physics rollout
  different:    standing still, arm raises, marching in place, the bedroom jumps

    python experiments/vjepa_probe.py --policy-qpos outputs/squat/policy_qpos.npy --policy-fps 50 \
        --bedroom-qpos outputs/colab_bedroom_test/qpos.npy --out outputs/vjepa_probe
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render  # noqa: E402
import retarget  # noqa: E402

FPS = 30
SIZE = (256, 256)


def squat(depth=0.25, reps=2, rep_s=2.0):
    rt = retarget.G1Retargeter()
    return rt.retarget_sequence(retarget.synthetic_squat(rt, depth=depth, reps=reps, rep_s=rep_s))


def periodic(make_targets, seconds=6.0, reps=2):
    """IK clip where make_targets(rt, s) gives targets for s in [0, 1] (0 = rest, 1 = peak)."""
    rt = retarget.G1Retargeter()
    t = np.linspace(0, 1, int(seconds * FPS) + 1)
    s = (1 - np.cos(2 * np.pi * reps * t)) / 2
    return rt.retarget_sequence([make_targets(rt, x, i / len(t)) for i, x in enumerate(s)])


def arm_raise(rt, s, _):
    if not hasattr(rt, "_rest"):
        rt._rest = {k: rt.frame_position(k) for k in ("pelvis", "left_foot", "right_foot", "left_hand", "right_hand")}
    r = rt._rest
    up = lambda p, side: p + s * (np.array([0.05, side * 0.15, 0.75]))
    return {"pelvis": r["pelvis"], "left_foot": r["left_foot"], "right_foot": r["right_foot"],
            "left_hand": up(r["left_hand"], 1), "right_hand": up(r["right_hand"], -1)}


def march(rt, s, phase):
    if not hasattr(rt, "_rest"):
        rt._rest = {k: rt.frame_position(k) for k in ("pelvis", "left_foot", "right_foot")}
    r = rt._rest
    left = int(phase * 4) % 2 == 0  # alternate legs each half-cycle
    lift = np.array([0.05, 0, 0.15 * s])
    return {"pelvis": r["pelvis"],
            "left_foot": r["left_foot"] + (lift if left else 0),
            "right_foot": r["right_foot"] + (0 if left else lift)}


def resample(qpos, src_fps):
    t_src = np.arange(len(qpos)) / src_fps
    t = np.arange(0, t_src[-1], 1 / FPS)
    return np.stack([np.interp(t, t_src, qpos[:, j]) for j in range(qpos.shape[1])], 1)


def embed(model, processor, frames, n=64):
    import torch
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    video = torch.from_numpy(frames[idx]).permute(0, 3, 1, 2)  # T, C, H, W
    inputs = processor(video, return_tensors="pt").to(model.device)
    with torch.no_grad():
        feats = model.get_vision_features(**inputs)  # (1, tokens, dim)
    return feats.mean(1)[0].float().cpu().numpy()


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = {"squat_ref": squat(), "squat_shallow": squat(depth=0.15), "squat_fast": squat(reps=3, rep_s=4 / 3)}
    if a.policy_qpos:
        clips["squat_policy_rollout"] = resample(np.load(a.policy_qpos), a.policy_fps)
    clips |= {"stand_still": squat(depth=0.0), "arm_raise": periodic(arm_raise), "march": periodic(march, reps=4)}
    if a.bedroom_qpos:
        clips["bedroom_jumps"] = np.load(a.bedroom_qpos)
    same = {"squat_shallow", "squat_fast", "squat_policy_rollout"}

    model_g1 = render.load_g1()[0]
    videos = {}
    for name, q in clips.items():
        videos[name] = render.render_qpos_sequence(model_g1, q, size=SIZE)
        render.save_video(videos[name], out / f"{name}.mp4", fps=FPS)
    print("rendered:", {k: len(v) for k, v in videos.items()}, flush=True)

    from transformers import AutoModel, AutoVideoProcessor
    model = AutoModel.from_pretrained(a.model).to("cuda").eval()
    processor = AutoVideoProcessor.from_pretrained(a.model)
    E = np.stack([embed(model, processor, v) for v in videos.values()])
    names = list(videos)

    # Cosine similarity after centring: every clip shares the robot, scene and camera, so the
    # mean embedding is mostly appearance. Subtracting it leaves the motion.
    Ec = E - E.mean(0)
    Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
    sim = Ec @ Ec.T
    raw = (E / np.linalg.norm(E, axis=1, keepdims=True)) @ (E / np.linalg.norm(E, axis=1, keepdims=True)).T

    # Baseline: plain joint-angle distance after resampling to 64 frames.
    def joints(q):
        idx = np.linspace(0, len(q) - 1, 64).round().astype(int)
        return q[idx, 7:]
    ref_j = joints(clips["squat_ref"])

    rows = []
    for i, n in enumerate(names):
        if n == "squat_ref":
            continue
        rows.append({"clip": n, "group": "same" if n in same else "different",
                     "vjepa_centred_cos": float(sim[0, i]), "vjepa_raw_cos": float(raw[0, i]),
                     "joint_dist_rad": float(np.abs(joints(clips[n]) - ref_j).mean())})
    rows.sort(key=lambda r: -r["vjepa_centred_cos"])
    print(f"{'clip':22s} {'group':10s} {'V-JEPA (centred)':>17s} {'V-JEPA (raw)':>13s} {'joint dist':>11s}")
    for r in rows:
        print(f"{r['clip']:22s} {r['group']:10s} {r['vjepa_centred_cos']:17.3f} {r['vjepa_raw_cos']:13.4f} {r['joint_dist_rad']:11.3f}")
    s_same = [r["vjepa_centred_cos"] for r in rows if r["group"] == "same"]
    s_diff = [r["vjepa_centred_cos"] for r in rows if r["group"] == "different"]
    margin = min(s_same) - max(s_diff)
    verdict = "SEPARATES" if margin > 0 else "DOES NOT SEPARATE"
    print(f"\nmargin (worst same - best different): {margin:+.3f} -> V-JEPA 2 {verdict} same-action from different-action")
    (out / "probe_results.json").write_text(json.dumps({"rows": rows, "margin": margin, "names": names,
                                                        "sim_centred": sim.tolist()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-qpos")
    ap.add_argument("--policy-fps", type=float, default=50)
    ap.add_argument("--bedroom-qpos")
    ap.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--out", default="outputs/vjepa_probe")
    main(ap.parse_args())
