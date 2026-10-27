"""Skill reader: Muse Spark watches the clip and writes a training spec for the robot.

    python -m muse.skill_reader clips/squat.mov                # -> outputs/squat/skill_spec.json
    python -m muse.skill_reader clips/*.mov --out-root outputs/remote_latent/clips --dry-run
    python pipeline.py clips/squat.mov --skill live            # as the last pipeline stage

skill_spec.json (schema: SKILL_SCHEMA) holds the action label and nearest bank motions, tempo,
phases with keyframes and how each should be matched, the bodies that matter, and success
criteria in metrics the simulator can measure. If the clip's folder has pipeline.py's meta.json
and retarget_report.json, Muse gets the clip length and the per-body retarget error as well.
Everything is phrased for the robot; nothing in the spec addresses the person in the video.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from muse import media  # noqa: E402
from muse.client import make_client, now, read_json, text_part, video_part, write_json  # noqa: E402
from muse.schema import BOOL, INT, NUM, STR, arr, enum, obj, validate  # noqa: E402

# latent/bank.py motion_bank() names (tests check they stay in sync).
BANK_NAMES = [f"squat_d{d}_r{r}" for d in (10, 20, 30) for r in (1, 2, 3)] + [
    "lunge_left", "lunge_right", "march", "high_knees", "kick_left", "kick_right_high", "side_shuffle",
    "vertical_jump", "jumping_jacks", "arms_raise_front", "arms_raise_side", "wave", "punch", "bend",
    "twist", "sway", "stand"]

# Body groups Muse can name -> the mjlab G1 tracking bodies (latent/reward_net.py BODIES).
BODY_GROUPS = {
    "pelvis": ("pelvis",),
    "torso": ("torso_link",),
    "left_leg": ("left_hip_roll_link", "left_knee_link"),
    "right_leg": ("right_hip_roll_link", "right_knee_link"),
    "left_foot": ("left_ankle_roll_link",),
    "right_foot": ("right_ankle_roll_link",),
    "left_arm": ("left_shoulder_roll_link", "left_elbow_link"),
    "right_arm": ("right_shoulder_roll_link", "right_elbow_link"),
    "left_hand": ("left_wrist_yaw_link",),
    "right_hand": ("right_wrist_yaw_link",),
}

# What a success criterion can measure; all computable from a rollout's body positions.
METRICS = {
    "pelvis_drop_m": "lowest pelvis height below its standing height, metres",
    "pelvis_rise_m": "highest pelvis height above its standing height, metres (jumps)",
    "airborne_s": "longest time with both feet off the ground, seconds",
    "foot_height_max_m": "highest either foot gets above the ground, metres",
    "hand_height_max_m": "highest either hand gets above the pelvis, metres",
    "torso_pitch_max_deg": "largest forward lean of the torso from vertical, degrees",
    "reps": "repetitions completed",
    "cycle_s": "mean duration of one repetition, seconds",
    "falls": "times the robot falls",
}

MATCH = {
    "pose": "copy the retargeted pose closely (tracking reward), e.g. the bottom of a squat",
    "style": "only look like the video (V-JEPA latent reward); the exact joint angles may differ",
    "free": "not scored (set-up, transitions, frames where the person is hidden)",
}

SKILL_SCHEMA = obj(
    action=obj(label=STR, description=STR, nearest_bank=arr(enum(BANK_NAMES))),
    tempo=obj(duration_s=NUM, reps=INT, cycle_s=NUM, speed=enum(["slow", "moderate", "fast", "explosive"]),
              airborne=BOOL),
    phases=arr(obj(name=STR, start_s=NUM, end_s=NUM, keyframe_s=NUM, match=enum(MATCH),
                   key_bodies=arr(enum(BODY_GROUPS)), description=STR)),
    key_bodies=arr(obj(body=enum(BODY_GROUPS), weight=NUM, reason=STR)),
    success_criteria=arr(obj(name=STR, metric=enum(METRICS), op=enum([">=", "<="]), value=NUM, description=STR)),
)

SYSTEM = (
    "You write training specs for a Unitree G1 humanoid robot (1.3 m tall, 35 kg) that learns a motion "
    "in a physics simulator from a video of a person doing it. Describe the motion so the robot's reward "
    "and evaluation can be set up; never give the person advice or judge their technique. "
    "Times are seconds from the start of the clip. Answer with JSON only."
)


def _bullets(d):
    return "\n".join(f"- {k}: {v}" for k, v in d.items())


def prompt(meta=None, retarget_report=None, fps=4):
    lines = [f"The video is sampled at {fps} frames per second."]
    if meta:
        lines.append(f"Clip length: {meta['frames'] / meta['fps']:.2f} s ({meta['frames']} frames at {meta['fps']:g} fps).")
    if retarget_report:
        err = ", ".join(f"{k} {v[0]:.1f}/{v[1]:.1f}" for k, v in retarget_report.items())
        lines.append("How closely the robot's retargeted motion follows the person, per body, cm mean/max "
                     f"(high = the robot can't reach that pose, so don't make it a 'pose' target): {err}.")
    lines += [
        "",
        "1. action: a short snake_case label, one sentence describing the motion, and the 1-3 bank motions "
        f"closest to it, most similar first. Bank: {', '.join(BANK_NAMES)}.",
        "2. tempo: clip duration, repetitions, seconds per repetition, overall speed, and whether both feet "
        "ever leave the ground.",
        "3. phases: cover the clip in time order, at most 12, each with one keyframe inside it, the bodies "
        "that matter in that phase, and how it should be matched:\n" + _bullets(MATCH),
        "4. key_bodies: the bodies that carry the motion, weight 0-1 (1 = most important). Bodies: "
        + ", ".join(BODY_GROUPS) + ".",
        "5. success_criteria: 2-6 checks a rollout must pass, each 'metric op value' in these metrics, "
        "scaled to the robot's size, not the person's:\n" + _bullets(METRICS),
    ]
    return "\n".join(lines)


def check(spec, duration=None):
    """Checks beyond the schema: counts, ranges, time order. -> list of problems."""
    errors = []
    bank = spec["action"]["nearest_bank"]
    if not 1 <= len(bank) <= 3 or len(set(bank)) != len(bank):
        errors.append("action.nearest_bank: 1-3 different bank motions")
    t = spec["tempo"]
    if t["duration_s"] <= 0 or t["cycle_s"] <= 0 or t["reps"] < 1:
        errors.append("tempo: duration_s and cycle_s must be > 0, reps >= 1")
    phases = spec["phases"]
    if not 1 <= len(phases) <= 12:
        errors.append("phases: 1-12 phases")
    end = duration + 0.5 if duration else float("inf")
    prev_end = 0.0
    for i, p in enumerate(phases):
        where = f"phases[{i}] ({p['name']})"
        if not 0 <= p["start_s"] < p["end_s"] <= end:
            errors.append(f"{where}: need 0 <= start_s < end_s <= clip length"
                          + (f" ({duration:.2f} s)" if duration else ""))
        if not p["start_s"] <= p["keyframe_s"] <= p["end_s"]:
            errors.append(f"{where}: keyframe_s must lie inside the phase")
        if p["start_s"] < prev_end - 0.1:
            errors.append(f"{where}: starts before the previous phase ends; phases go in time order")
        if len(set(p["key_bodies"])) != len(p["key_bodies"]):
            errors.append(f"{where}: key_bodies repeated")
        prev_end = p["end_s"]
    bodies = [b["body"] for b in spec["key_bodies"]]
    if not bodies or len(set(bodies)) != len(bodies):
        errors.append("key_bodies: at least one, each body once")
    errors += [f"key_bodies ({b['body']}): weight must be in [0, 1]" for b in spec["key_bodies"]
               if not 0 <= b["weight"] <= 1]
    crit = spec["success_criteria"]
    if not 1 <= len(crit) <= 6 or len({c["name"] for c in crit}) != len(crit):
        errors.append("success_criteria: 1-6 criteria with different names")
    errors += [f"success_criteria ({c['name']}): value must be >= 0" for c in crit if c["value"] < 0]
    return errors


def load(path):
    """A skill_spec.json, validated -> spec (with its _meta)."""
    spec = read_json(path)
    body = {k: v for k, v in spec.items() if k != "_meta"}
    errors = validate(body, SKILL_SCHEMA) or check(body)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return spec


def read_skill(video, out_dir, client, fps=4, max_side=480):
    """Muse Spark watches `video` -> out_dir/skill_spec.json. Returns the spec."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, report = read_json(out_dir / "meta.json"), read_json(out_dir / "retarget_report.json")
    duration = meta["frames"] / meta["fps"] if meta else None
    clip = media.prepare_video(video, out_dir / "muse_input.mp4", max_side, required=client.live)
    content = [text_part(prompt(meta, report, fps)), video_part(clip, fps)]
    spec, info = client.ask_json("skill_spec", SYSTEM, content, SKILL_SCHEMA, check=lambda s: check(s, duration))
    spec["_meta"] = {"source": str(video), "created": now(), "sample_fps": fps, **info}
    write_json(out_dir / "skill_spec.json", spec)
    if not client.live:
        client.save_requests(out_dir / "skill_spec.request.json")
    print(f"{out_dir / 'skill_spec.json'}: {spec['action']['label']}, {len(spec['phases'])} phases, "
          f"nearest bank {spec['action']['nearest_bank']} ({info['mode']})")
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", help="output folder (one video only); default <out-root>/<video name>")
    ap.add_argument("--out-root", default=config.OUTPUT_DIR)
    ap.add_argument("--fps", type=float, default=4, help="frames per second Muse samples")
    ap.add_argument("--max-side", type=int, default=480, help="downscale the video to this many px first")
    ap.add_argument("--dry-run", action="store_true", help="no API call: bundled fixture response")
    ap.add_argument("--fixture", help="no API call: responses from this recorded file")
    ap.add_argument("--record", help="save live exchanges here (they work as --fixture files)")
    ap.add_argument("--model")
    ap.add_argument("--force", action="store_true", help="redo clips that already have skill_spec.json")
    a = ap.parse_args(argv)
    if a.out and len(a.videos) > 1:
        ap.error("--out takes one video; use --out-root for several")
    for v in a.videos:
        out = Path(a.out or Path(a.out_root) / Path(v).stem)
        if (out / "skill_spec.json").exists() and not a.force:
            print(f"{out / 'skill_spec.json'} exists; skipping (--force to redo)")
            continue
        client = make_client(a.dry_run, a.fixture, "skill_spec_squat.json", a.model, a.record)
        read_skill(v, out, client, a.fps, a.max_side)


if __name__ == "__main__":
    main()
