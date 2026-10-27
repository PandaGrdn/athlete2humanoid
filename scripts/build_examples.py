"""Curated demo examples for the web app: small JSON stories + web-friendly videos, served statically.

    python scripts/build_examples.py            # (re)build web/public/examples/ from outputs/
    python scripts/build_examples.py --check    # validate the built examples (no outputs/ needed)

Every number comes from a run's own files (eval.json, robust.json, compare.json, importance.json, ...);
each example lists the files it read under "sources". The frontend reads examples/index.json and
examples/<id>/example.json with no backend, so a static deploy is a full demo.

Examples:
  squat-v1           three policies on the squat clip (outputs/remote_latent)
  retarget           athlete | uniform | proportional retarget (outputs/retarget_compare)
  goal-relevance     which moments of squat_jump matter (outputs/goal_relevance)
  robustness         pushes, noisy starts, friction/mass, speed (outputs/remote_robust/robust.json)
  squat-jump-final   the task-reward squat_jump policy; a placeholder until outputs/remote_latent_final
                     exists, then filled by re-running this script

Videos are re-encoded to H.264 yuv420p (plays everywhere, unlike the 4:4:4 clip renders), at most
720 px tall, 30 fps, a keyframe every 0.5 s so synchronized scrubbing is snappy. Encodes are cached by
mtime; --force redoes them. ffmpeg comes from imageio-ffmpeg when there is no system ffmpeg.

The goal-relevance video renders the arm-waving ("hype") variant of the G1 reference with MuJoCo when
it's available (A2H_ASSETS_DIR must hold mujoco_menagerie); otherwise it falls back to the clean clip.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "web" / "public" / "examples"
MAX_FILE_MB = 25
MAX_TOTAL_MB = 60
POLICY_COLOR = {"tracking": "#3987e5", "hybrid": "#d95926", "latent": "#199e70", "task": "#c98bf0", "taskphase": "#e0a43a"}
POLICY_INFO = {
    "tracking": ("Tracking", "Copies the retarget's joint angles (the standard motion-imitation reward)."),
    "hybrid": ("Hybrid", "Joint tracking plus a V-JEPA 2 latent reward: rewarded for looking like the clip."),
    "latent": ("Latent", "Mostly the V-JEPA 2 latent reward; the retarget is only an observation and a reset pose."),
    "task": ("Task", "Rewarded for completing the action's stages in order (stand, squat, jump, land), plus "
                     "V-JEPA 2 style. Ends only on a fall."),
    "taskphase": ("Task (no reference)", "Like Task, but the policy never sees the retarget: only its own body, "
                                         "a clock and the current stage."),
}
STAGE_LABEL = {"stand": "Stand", "squat": "Squat", "jump": "Jump", "land": "Land"}


# --- helpers -------------------------------------------------------------------------------------

def default_runs_dir():
    """outputs/ of this checkout, or of the main checkout when this is a git worktree (outputs/ is gitignored)."""
    here = REPO / "outputs"
    if (here / "remote_latent").is_dir():
        return here
    try:
        common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--git-common-dir"], capture_output=True,
                                text=True, check=True).stdout.strip()
        main = (REPO / common).resolve().parent / "outputs"
        if main.is_dir():
            return main
    except (OSError, subprocess.CalledProcessError):
        pass
    return here


def read_json(p):
    return json.loads(Path(p).read_text())


def ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # bundled binary; pip install imageio-ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path):
    """(width, height, duration_s) from ffmpeg's banner (there's no ffprobe in imageio-ffmpeg)."""
    err = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    wh = re.search(r"Video: .*?(\d{2,5})x(\d{2,5})", err)
    if not d or not wh:
        raise RuntimeError(f"can't read video {path}")
    return int(wh[1]), int(wh[2]), int(d[1]) * 3600 + int(d[2]) * 60 + float(d[3])


def r(v, d=3):
    return None if v is None else round(float(v), d)


class Media:
    """Collects the encodes an example needs and runs them (cached by mtime)."""

    def __init__(self, out_dir, force=False):
        self.out_dir, self.force = out_dir, force

    def _fresh(self, dst, srcs):
        return (not self.force and dst.exists() and dst.stat().st_size > 0
                and all(dst.stat().st_mtime >= Path(s).stat().st_mtime for s in srcs))

    def video(self, ex_id, name, src, *, max_h=720, crf=27, filter_complex=None, extra_inputs=(), label="",
              panels=None, poster_at=None):
        """src -> <out>/<id>/<name>.mp4 + .jpg poster; returns the JSON entry for it."""
        d = self.out_dir / ex_id
        d.mkdir(parents=True, exist_ok=True)
        dst, poster = d / f"{name}.mp4", d / f"{name}.jpg"
        srcs = [src, *extra_inputs]
        if not self._fresh(dst, srcs):
            cmd = [ffmpeg(), "-y", "-loglevel", "error"]
            for s in srcs:
                cmd += ["-i", str(s)]
            scale = f"scale=-2:'min({max_h},ih)':flags=lanczos,fps=30,format=yuv420p"
            if filter_complex:
                cmd += ["-filter_complex", f"{filter_complex},{scale}[v]", "-map", "[v]"]
            else:
                cmd += ["-vf", scale]
            cmd += ["-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-profile:v", "high",
                    "-g", "15", "-movflags", "+faststart", str(dst)]
            subprocess.run(cmd, check=True)
            print(f"  encoded {dst.relative_to(self.out_dir)} ({dst.stat().st_size / 1e6:.2f} MB)")
        w, h, dur = probe(dst)
        t = min(poster_at if poster_at is not None else dur * 0.4, max(dur - 0.1, 0))
        if not self._fresh(poster, [dst]):
            subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(dst), "-frames:v", "1",
                            "-q:v", "4", str(poster)], check=True)
        entry = {"src": f"{ex_id}/{dst.name}", "poster": f"{ex_id}/{poster.name}", "label": label,
                 "width": w, "height": h, "duration_s": round(dur, 2)}
        if panels:
            entry["panels"] = panels
        return entry


def task_stages(spec):
    return [{"kind": s["kind"], "label": STAGE_LABEL.get(s["kind"], s["kind"].title()),
             "start_s": r(s["start_s"]), "end_s": r(s["end_s"])} for s in spec["stages"]]


def compile_task_spec(clip_dir):
    """The clip's task spec (stand/squat/jump/land stages) via skill.task, if its deps import; else None."""
    if (clip_dir / "task_spec.json").exists():
        return read_json(clip_dir / "task_spec.json")
    try:
        sys.path.insert(0, str(REPO))
        import numpy as np
        from skill import task
        qpos = np.load(clip_dir / "qpos.npy")
        fps = read_json(clip_dir / "meta.json").get("fps", 30.0)
        return task.compile_spec(task.body_positions_from_qpos(qpos), fps, name=clip_dir.name)
    except Exception as e:  # torch / mujoco / G1 assets missing: the example just has no stage timeline
        print(f"  (no task stages for {clip_dir.name}: {type(e).__name__}: {e})")
        return None


def semantic_result(doc, run_label):
    """'5/5 stages, no falls' for a run from docs/SEMANTIC_REWARD.md's results table (v1 eval.json predates
    task_success; the doc's table is where those numbers were recorded)."""
    m = re.search(rf"^\s*\|\s*{re.escape(run_label)}\s*\|\s*([^|]+)\|\s*(\d+)/(\d+) stages, ([^|]+)\|", doc.read_text(), re.M)
    if not m:
        raise RuntimeError(f"no '{run_label}' row in {doc}")
    return {"policies": [p.strip() for p in m[1].split(",")], "done": int(m[2]), "total": int(m[3]), "falls": m[4].strip()}


def policy_rows(ev, keys):
    """Plain-language metric rows per policy from an eval.json."""
    rows = []
    for k in keys:
        e = ev[k]
        label, what = POLICY_INFO.get(k, (k.title(), ""))
        row = {"key": k, "label": label, "color": POLICY_COLOR.get(k, "#8a96a3"), "reward": what, "values": {}}
        v = row["values"]
        if "task_stages" in e:
            v["stages"] = {"value": e["task_stages"], "display": e["task_stages"], "good": bool(e.get("task_success"))}
        if e.get("falls") is not None:
            v["falls"] = {"value": e["falls"], "display": str(e["falls"]), "good": e["falls"] == 0}
        if e.get("aligned") is not None:
            v["vjepa"] = {"value": r(e["aligned"]), "display": f"{e['aligned']:.2f}"}
        if e.get("dtw_rank_vs_bank") is not None:
            v["rank"] = {"value": e["dtw_rank_vs_bank"], "display": f"#{e['dtw_rank_vs_bank']}",
                         "good": e["dtw_rank_vs_bank"] == 1}
        if e.get("mean_joint_error_rad") is not None:
            deg = e["mean_joint_error_rad"] * 57.29578
            v["joint_deg"] = {"value": r(deg, 1), "display": f"{deg:.1f}°"}
        if e.get("mean_torso_position_error_m") is not None:
            cm = e["mean_torso_position_error_m"] * 100
            v["torso_cm"] = {"value": r(cm, 1), "display": f"{cm:.1f} cm"}
        if e.get("task_jump_quality"):
            q = e["task_jump_quality"]
            v["jump_q"] = {"value": q, "display": " / ".join(f"{x:.2f}" for x in q)}
        rows.append(row)
    return rows


POLICY_COLUMNS = {
    "stages": ("Stages done", "Stages of the action completed in order, judged on the robot's own body"),
    "falls": ("Falls", "Falls in the evaluation rollout"),
    "jump_q": ("Jump quality", "Per jump: 1 = apex within the clip's height range"),
    "vjepa": ("V-JEPA 2 similarity", "Cosine similarity to the clip in V-JEPA 2's video embedding (1 = identical)"),
    "rank": ("Rank vs. motion bank", "Rank among a bank of other motions by V-JEPA similarity to the clip; #1 = closest"),
    "joint_deg": ("Joint-angle gap", "Mean joint-angle difference from the retarget"),
    "torso_cm": ("Torso offset", "Mean torso position difference from the retarget"),
}


def columns_for(rows):
    keys = [k for k in POLICY_COLUMNS if any(k in row["values"] for row in rows)]
    return [{"key": k, "label": POLICY_COLUMNS[k][0], "note": POLICY_COLUMNS[k][1]} for k in keys]


# Rollout renders are 320x240 with the robot in the middle ~60%: crop to it and upscale 2x so it reads on screen.
POLICY_CROP = "[0:v]crop=208:156:64:62,scale=416:312:flags=lanczos"


def policy_videos(media, ex_id, run, keys, poster_at=None):
    out = []
    for k in keys:
        f = run / f"policy_{k}" / "policy.mp4"
        if f.exists():
            w, h, _ = probe(f)
            v = media.video(ex_id, f"policy_{k}", f, label=POLICY_INFO.get(k, (k.title(),))[0], poster_at=poster_at,
                            filter_complex=POLICY_CROP if (w, h) == (320, 240) else None)
            v["color"], v["policy"] = POLICY_COLOR.get(k, "#8a96a3"), k
            out.append(v)
    return out


def clip_duration(clip_dir):
    meta = read_json(clip_dir / "meta.json")
    return round(meta["frames"] / meta["fps"], 2)


# --- examples ------------------------------------------------------------------------------------

def ex_squat_v1(runs, media):
    run = runs / "remote_latent"
    ev, status = read_json(run / "eval.json"), read_json(run / "status.json")
    clip = run / "clips" / "squat"
    sem = semantic_result(REPO / "docs" / "SEMANTIC_REWARD.md", "v1")
    keys = [k for k in ["tracking", "hybrid", "latent"] if k in ev]
    rows = policy_rows(ev, keys)
    for row in rows:  # v1's eval.json predates task_success: the doc's table recorded it for all three
        if row["key"] in sem["policies"]:
            s = f"{sem['done']}/{sem['total']}"
            row["values"] = {"stages": {"value": s, "display": s, "good": sem["done"] == sem["total"]}, **row["values"]}
    iters = re.search(r"--iters (\d+)", " ".join(status.get("cmd", [])))
    spec = compile_task_spec(clip)
    stages = task_stages(spec) if spec else None
    squats = sum(1 for s in stages or [] if s["kind"] == "squat")
    vj = {k: ev[k]["aligned"] for k in keys}
    bank_best = ev["_bank_top5_dtw"][0]
    dur = clip_duration(clip)
    bottom = next((s["start_s"] + 0.6 for s in stages or [] if s["kind"] == "squat"), None)
    source = media.video("squat-v1", "source", clip / "side_by_side.mp4", label="Phone video → G1 reference",
                         panels=_sbs_panels(clip / "side_by_side.mp4"), poster_at=bottom)
    return {
        "id": "squat-v1",
        "kicker": "Learning the action",
        "title": "Squat: three rewards, one skill",
        "tagline": (f"Shown one phone video of {_n(squats, 'squat')}, three policies trained with different rewards "
                    f"all complete it ({sem['done']}/{sem['total']} stages, {sem['falls']}), though their V-JEPA 2 "
                    f"similarity to the clip ranges from {min(vj.values()):.2f} to {max(vj.values()):.2f}."),
        "story": {
            "shown": (f"A {dur:.0f}-second phone video of one person doing {_n(squats, 'squat')}. SAM 3.1 isolates "
                      "the person, MediaPipe lifts their pose to 3D, and inverse kinematics rebuilds it on the "
                      "Unitree G1's proportions: the reference on the right of the first video."),
            "learned": ("Three control policies, trained with PPO in mjlab"
                        + (f" for {int(iters[1]):,} iterations each" if iters else "")
                        + ". Each drives a physically simulated G1 with its own reward: copying joint angles "
                          "(tracking), joint angles plus Meta's V-JEPA 2 video model (hybrid), or mostly V-JEPA 2 "
                          "alone (latent)."),
            "how": (f"A stage checker reads only the robot's own body: stand, squat, stand, squat, stand. All three "
                    f"finish every stage without falling. V-JEPA 2 scores how much each rollout looks like the video: "
                    f"each is closer to the clip than any other motion in the bank (best other: "
                    f"'{bank_best[0]}' at {bank_best[1]:.2f} DTW), yet the latent policy scores lower than tracking. "
                    f"A lower similarity score was not a failure to squat."),
        },
        "duration_s": dur,
        "stages": stages,
        "primary": source,
        "grid": policy_videos(media, "squat-v1", run, keys, poster_at=bottom),
        "grid_caption": "Solid robot: the trained policy, physically simulated. Green ghost: the reference it was shown.",
        "extra": [media.video("squat-v1", "vjepa_view", run / "comparison.mp4",
                              filter_complex="[0:v]crop=iw:ih-34:0:34",  # drop the burned-in labels; the page labels panels
                              label="What V-JEPA 2 compares: silhouettes of the video and each robot",
                              panels=_even_panels(["Video", "Retarget", "Tracking", "Hybrid", "Latent"]),
                              poster_at=bottom)],
        "metrics": [
            {"label": "Squat stages completed", "display": f"{sem['done']}/{sem['total']}",
             "note": f"by all {len(sem['policies'])} policies, {sem['falls']}", "tone": "good"},
            {"label": "Falls in evaluation", "display": str(sum(ev[k]["falls"] for k in keys)),
             "note": f"across {len(keys)} policies × {ev[keys[0]]['steps'] / ev[keys[0]]['fps']:.1f} s rollouts", "tone": "good"},
            {"label": "V-JEPA 2 similarity", "display": " / ".join(f"{vj[k]:.2f}" for k in keys),
             "note": f"{' / '.join(keys)}; the retarget itself scores {ev['retarget']['aligned']:.2f}"},
            {"label": "Closest motion in the bank", "display": "#1",
             "note": "for every policy: each looks more like the clip than any other bank motion"},
        ],
        "policies": rows,
        "columns": columns_for(rows),
        "sources": [_rel(runs, run / "eval.json"), _rel(runs, run / "status.json"), "docs/SEMANTIC_REWARD.md",
                    _rel(runs, clip / "qpos.npy")],
        "notes": ["Stage results are from docs/SEMANTIC_REWARD.md (the v1 run's eval.json predates the stage checker).",
                  "One deterministic evaluation rollout per policy; the robustness example has 64 per condition."],
    }


def ex_retarget(runs, media):
    cmp = read_json(runs / "retarget_compare" / "compare.json")
    clip = "squat_jump"
    clips = sorted(cmp)

    def mean(method, f):
        return sum(f(cmp[c][method]) for c in clips) / len(clips)

    rows = [
        ("Foot sliding while planted", "cm/s", lambda m: m["slide_cm_per_s"]),
        ("Planted-foot drift", "cm", lambda m: m["drift_cm"]),
        ("Foot floating above the floor", "cm", lambda m: m["stance_float_cm"]),
        ("Frames at a joint limit", "%", lambda m: 100 * m["limit_frames"]),
        ("Knee angle error", "°", lambda m: m["angle_err_deg"]["knee"]),
        ("Hip angle error", "°", lambda m: m["angle_err_deg"]["hip"]),
        ("Elbow angle error", "°", lambda m: m["angle_err_deg"]["elbow"]),
        ("Shoulder angle error", "°", lambda m: m["angle_err_deg"]["shoulder"]),
    ]
    compare = []
    for label, unit, f in rows:
        b, a = mean("uniform", f), mean("proportional", f)
        cb, ca = f(cmp[clip]["uniform"]), f(cmp[clip]["proportional"])
        compare.append({"label": label, "unit": unit, "before": r(b, 2), "after": r(a, 2),
                        "ratio": r(b / a, 1) if a > 1e-9 else None,
                        "clip_before": r(cb, 2), "clip_after": r(ca, 2)})
    angle = [c for c in compare if c["label"].endswith("angle error")]
    ratios = sorted(c["ratio"] for c in angle)
    slide = compare[0]
    video = runs / "retarget_compare" / clip / "compare.mp4"
    w, h, _ = probe(video)
    athlete_w = w - 2 * h  # athlete | robot (h x h) | robot (h x h)
    panels = [{"label": "Athlete", "from": 0, "to": r(athlete_w / w)},
              {"label": "Uniform scaling", "from": r(athlete_w / w), "to": r((athlete_w + h) / w), "tone": "bad"},
              {"label": "Proportional (ours)", "from": r((athlete_w + h) / w), "to": 1, "tone": "good"}]
    spec = read_json(runs / "goal_relevance" / clip / "task_spec.json")
    return {
        "id": "retarget",
        "kicker": "Retargeting",
        "title": "A human body on G1 proportions",
        "tagline": (f"Rebuilding the person on the G1's own bone lengths keeps their joint angles "
                    f"({ratios[0]:.1f}–{ratios[-1]:.0f}× smaller angle error) and plants the feet: foot sliding drops "
                    f"from {slide['before']:.1f} to {slide['after']:.1f} cm/s, averaged over {len(clips)} clips."),
        "story": {
            "shown": ("The same squat-jump video, retargeted two ways. Uniform scaling shrinks the whole person by "
                      "leg length; the G1's hips are still about 60% wider than the scaled human's, so the inverse "
                      "kinematics chases targets no G1 pose can reach: feet slide and elbows straighten."),
            "learned": ("Proportional retargeting walks the skeleton from the hips and gives every bone the human's "
                        "direction but the G1's length, then pins each planted foot to the floor. Every joint angle "
                        "is the human's, and the hands and feet land where the G1 can actually reach."),
            "how": (f"Measured on all {len(clips)} recorded clips against the human's own joint angles and foot "
                    f"contacts. The angle columns compare robot and human flexion at the knee, hip, elbow and "
                    f"shoulder. The bars show the {len(clips)}-clip mean; the video is {clip}."),
        },
        "duration_s": clip_duration(runs / "remote_latent" / "clips" / clip),
        "stages": task_stages(spec),
        "primary": media.video("retarget", "compare", video, label="Athlete | uniform | proportional",
                               panels=panels, poster_at=2.6),
        "grid": [],
        "extra": [],
        "metrics": [
            {"label": "Foot sliding", "display": f"{slide['before']:.1f} → {slide['after']:.1f}", "unit": "cm/s",
             "note": f"mean of {len(clips)} clips, uniform → proportional", "tone": "good"},
            {"label": "Joint-angle error", "display": f"{ratios[0]:.1f}–{ratios[-1]:.0f}× lower",
             "note": "knee, hip, elbow and shoulder vs. the human", "tone": "good"},
            {"label": "Frames at a joint limit", "display": f"{compare[3]['before']:.0f}% → {compare[3]['after']:.0f}%",
             "note": "the IK pinned against the G1's range"},
        ],
        "compare": compare,
        "compare_labels": {"before": "Uniform scaling", "after": "Proportional"},
        "sources": [_rel(runs, runs / "retarget_compare" / "compare.json"), "docs/RETARGET.md"],
        "notes": ["Retargeting parameters were tuned on these same seven clips; there is no held-out clip."],
    }


def ex_goal_relevance(runs, media):
    clean = read_json(runs / "goal_relevance" / "squat_jump" / "importance.json")
    hype_dir = runs / "goal_relevance" / "squat_jump_hype"
    imp = read_json(hype_dir / "importance.json")
    spec = read_json(hype_dir / "task_spec.json")
    fps, w = imp["fps"], imp["weight"]
    a, b = imp["hype_s"]
    ia, ib = int(round(a * fps)), int(round(b * fps))
    bw = imp["body_weight"]
    arms = [(x + y) / 2 for x, y in zip(bw["left_arm"], bw["right_arm"])]
    legs = [(x + y) / 2 for x, y in zip(bw["left_leg"], bw["right_leg"])]

    import numpy as np
    qpos = np.load(hype_dir / "qpos.npy")
    groups = imp["groups"]
    cols = lambda g: list(range(*map(int, re.findall(r"\d+", groups[g]))))  # "qpos[22:29]" -> 22..28  # noqa: E731
    arm_cols = cols("left_arm") + cols("right_arm")
    speed = np.r_[0, np.linalg.norm(np.diff(qpos[:, arm_cols], axis=0), axis=1) * fps]
    speed = np.convolve(speed, np.ones(5) / 5, mode="same")
    motion = (speed / speed.max()).round(3).tolist()

    ev = {e["kind"]: e for e in imp["events"] if e["kind"] in ("propulsion", "absorb")}
    prop, absorb = ev["propulsion"], ev["absorb"]
    win = lambda e: max(w[int(e["window_s"][0] * fps): int(e["window_s"][1] * fps) + 1])  # noqa: E731
    events = [{"t": r(e["t_s"]), "label": {"propulsion": "Takeoff push", "absorb": "Landing"}[e["kind"]],
               "peak": r(win(e), 2)}
              for e in imp["events"] if e["kind"] in ("propulsion", "absorb")]
    hype_mean = sum(w[ia:ib]) / (ib - ia)
    hype_arms = sum(arms[ia:ib]) / (ib - ia)
    hype_motion = sum(motion[ia:ib]) / (ib - ia)
    rest_motion = sum(motion[ib:]) / (len(motion) - ib)
    legs_prop = prop["group_share"]["left_leg"] + prop["group_share"]["right_leg"]
    arms_prop = prop["group_share"]["left_arm"] + prop["group_share"]["right_arm"]
    same_after = max(abs(x - y) for x, y in zip(w[ib:], clean["weight"][ib:]))

    clip = runs / "remote_latent" / "clips" / "squat_jump"
    sbs = clip / "side_by_side.mp4"
    rendered = render_hype(qpos, read_json(clip / "meta.json"))
    if rendered:
        sw, sh, _ = probe(sbs)
        aw = sw - sh  # athlete | robot (h x h)
        primary = media.video("goal-relevance", "hype", sbs, extra_inputs=[rendered],
                              filter_complex=f"[0:v]crop={aw}:{sh}:0:0,setsar=1[a];[1:v]scale={sh}:{sh},setsar=1[b];[a][b]hstack=inputs=2",
                              label="Athlete | G1 reference with arm-waving added",
                              panels=[{"label": "Athlete", "from": 0, "to": r(aw / (aw + sh))},
                                      {"label": f"G1 + arm-waving {a:g}–{b:g} s", "from": r(aw / (aw + sh)), "to": 1,
                                       "tone": "warn"}], poster_at=(a + b) / 2)
    else:
        primary = media.video("goal-relevance", "clip", sbs, label="Athlete | G1 reference",
                              panels=_sbs_panels(sbs), poster_at=prop["t_s"])
    return {
        "id": "goal-relevance",
        "kicker": "What matters",
        "title": "Imitate what serves the jump",
        "tagline": (f"Arm-waving added before the jump gets the minimum weight ({hype_mean:.2f}, the same as standing "
                    f"still) while the takeoff push and the landing peak at {min(x['peak'] for x in events):.2f}–"
                    f"{max(x['peak'] for x in events):.2f}."),
        "story": {
            "shown": (f"The squat-jump reference with a distraction injected: from {a:g} to {b:g} s both arms go "
                      "overhead and pump, like someone hyping a crowd before a jump. The jump itself is unchanged."),
            "learned": ("For every frame and body part, how much it matters for the goal. The goal comes from the "
                        "action's stages: lower the centre of mass, push off, fly, absorb the landing. Credit is split "
                        "by physics: the change in each goal quantity is divided exactly among the body parts that "
                        "caused it, and back-and-forth motion cancels."),
            "how": (f"In the waving window the arms move {hype_motion / max(rest_motion, 1e-9):.1f}× faster than they do "
                    f"for the rest of the clip, yet the frame weight stays at the floor ({imp['floor']:.2f}). The legs "
                    f"get {legs_prop:.0%} of the takeoff credit, the arms {arms_prop:.0%}. After the waving the "
                    + ("weights are identical to the clean clip's. " if same_after < 5e-4 else
                       f"weights match the clean clip's to within {same_after:.3f}. ")
                    + f"Training scales the imitation (style) "
                    f"terms by this weight, so the policy isn't paid to copy the waving."),
        },
        "duration_s": r(len(w) / fps, 2),
        "stages": task_stages(spec),
        "primary": primary,
        "grid": [],
        "extra": [],
        "chart": {
            "fps": fps,
            "weight": [r(x) for x in w],
            "arms": [r(x) for x in arms],
            "legs": [r(x) for x in legs],
            "arm_motion": motion,
            "floor": imp["floor"],
            "hype": [a, b],
            "events": events,
        },
        "metrics": [
            {"label": "Weight while waving", "display": f"{hype_mean:.2f}",
             "note": f"the floor is {imp['floor']:.2f}; arms {hype_arms:.2f}", "tone": "good"},
            {"label": "Arm motion while waving", "display": f"{hype_motion / max(rest_motion, 1e-9):.1f}×",
             "note": "arm joint speed vs. the rest of the clip"},
            {"label": "Peak weight", "display": f"{max(x['peak'] for x in events):.2f}",
             "note": "the takeoff push and the landing", "tone": "good"},
            {"label": "Takeoff credit to the legs", "display": f"{legs_prop:.0%}",
             "note": f"arms {arms_prop:.0%}, torso {prop['group_share']['torso']:.0%}"},
        ],
        "sources": [_rel(runs, hype_dir / "importance.json"), _rel(runs, hype_dir / "qpos.npy"),
                    _rel(runs, runs / "goal_relevance" / "squat_jump" / "importance.json"), "docs/GOAL_RELEVANCE.md"],
        "notes": ["Weights are physics-only; V-JEPA 2's own relevance scores turned out to track salience, not the "
                  "goal (docs/GOAL_RELEVANCE.md).",
                  "The waving is synthetic (injected into the retarget), not a filmed clip."],
    }


def render_hype(qpos, meta):
    """The hype variant rendered from the clip camera's side, 480x480, cached in build/. None without MuJoCo."""
    cache = REPO / "build" / "examples_cache"
    out = cache / "hype_g1.mp4"
    if out.exists():
        return out
    try:
        sys.path.insert(0, str(REPO))
        import mediapy
        import render
        model, _ = render.load_g1()
        cam = render.tracking_camera(model, azimuth=meta.get("view_azimuth", 135.0))
        frames = render.render_qpos_sequence(model, qpos, size=(480, 480), cam=cam)
        cache.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(str(out), frames, fps=meta.get("fps", 30.0))
        print(f"  rendered {out}")
        return out
    except Exception as e:
        print(f"  (no hype render: {type(e).__name__}: {e}; using the clean clip)")
        return None


def ex_robustness(runs, videos):
    rb = read_json(runs / "remote_robust" / "robust.json")
    meta = rb["meta"]
    pols = list(meta["policies"])
    names = {"clean": "As trained", "push": "Pushed", "init": "Noisy start", "dynamics": "Friction & mass",
             "speed": "Tempo"}

    def level(c):
        p = c["params"]
        if c["suite"] == "push":
            return f"{p['push_mps']:g} m/s shove"
        if c["suite"] == "init":
            return f"{p['init_scale']:g}× start noise"
        if c["suite"] == "dynamics":
            f, m = p["friction"], p["mass_scale"]
            return f"friction {f[0]:g}–{f[1]:g}, mass ×{m[0]:g}–{m[1]:g}"
        if c["suite"] == "speed":
            return f"{p['speed']:g}× speed"
        return "nominal"

    conditions = [{"suite": c["suite"], "suite_label": names.get(c["suite"], c["suite"]), "level": level(c),
                   "cells": {k: {"fall_rate": r(v["fall_rate"]), "vjepa": r(v.get("vjepa_aligned"))}
                             for k, v in c["policies"].items()}} for c in rb["conditions"]]
    summ = rb["summary"]
    dyn = {k: summ[k]["dynamics"]["fall_rate"] for k in pols}
    push = next(c for c in rb["conditions"] if c["suite"] == "push" and c["params"]["push_mps"] == 0.5)["policies"]
    clean = next(c for c in rb["conditions"] if c["suite"] == "clean")["policies"]
    most = min(pols, key=lambda k: summ[k]["push"]["fall_rate"] + summ[k]["init"]["fall_rate"])
    least = max(pols, key=lambda k: summ[k]["push"]["fall_rate"] + summ[k]["init"]["fall_rate"])
    label = lambda k: POLICY_INFO.get(k, (k.title(),))[0]  # noqa: E731
    return {
        "id": "robustness",
        "kicker": "Stress test",
        "title": "Pushed, tripped, re-weighted",
        "tagline": (f"{meta['reps']} simulated rollouts per condition: the squat policies barely notice friction and mass "
                    f"changes, and {label(most).lower()} is the sturdiest under shoves and noisy starts. "
                    f"{label(least)} falls most, which is why the task reward only pays while the robot stays upright."),
        "story": {
            "shown": ("The three squat policies, each run from scratch under conditions they never saw in training: "
                      "random shoves of increasing strength, starts from a perturbed pose at a random point in the "
                      "clip, wider friction and mass ranges, and the clip played faster or slower."),
            "learned": ("Whether a policy learned to squat or only memorised one trajectory. A policy that learned "
                        "the action keeps its balance and its motion as conditions move away from training."),
            "how": (f"A fall is the torso dropping below {meta['fall_rule']['torso_z_m']:g} m or tilting past "
                    f"{meta['fall_rule']['tilt_rad']:g} rad. Each cell is the fall rate over {meta['reps']} rollouts; "
                    f"underneath, V-JEPA 2 similarity to the clip on {meta['vjepa_rollouts']} of them. Under a "
                    f"0.5 m/s shove, falls are " + ", ".join(f"{label(k).lower()} {push[k]['fall_rate']:.0%}" for k in pols)
                    + "." + (f" Even as trained, the latent policy falls in {clean['latent']['fall_rate']:.0%} of "
                             "randomised rollouts." if "latent" in clean else "")),
        },
        "duration_s": None,
        "stages": None,
        "primary": None,
        "grid": [v for v in videos if v.get("policy") in pols],  # squat-v1's videos: the same checkpoints
        "grid_caption": "The same three policies, as trained (no perturbation).",
        "extra": [],
        "metrics": [
            {"label": "Rollouts per condition", "display": str(meta["reps"]),
             "note": f"{len(rb['conditions'])} conditions × {len(pols)} policies"},
            {"label": "Falls under friction & mass changes", "display": " / ".join(f"{dyn[k]:.0%}" for k in pols),
             "note": " / ".join(pols)},
            {"label": "Falls under a 0.5 m/s shove", "display": " / ".join(f"{push[k]['fall_rate']:.0%}" for k in pols),
             "note": " / ".join(pols)},
        ],
        "robust": {"policies": [{"key": k, "label": label(k), "color": POLICY_COLOR.get(k, "#8a96a3")} for k in pols],
                   "conditions": conditions, "reps": meta["reps"]},
        "sources": [_rel(runs, runs / "remote_robust" / "robust.json")],
        "notes": ["Policies are the squat v1 checkpoints; the reference check reproduces their eval.json scores.",
                  "The robustness suite randomises starts, so 'as trained' falls can differ from the single "
                  "deterministic evaluation rollout."],
    }


def ex_final(runs, media):
    run = runs / "remote_latent_final"
    base = {"id": "squat-jump-final", "kicker": "The task policy", "title": "Squat jump: stand, squat, jump, land"}
    if not (run / "eval.json").exists():
        status = read_json(run / "status.json") if (run / "status.json").exists() else {}
        return {**base, "status": "pending",
                "tagline": ("The final run: a policy rewarded for completing the squat jump's stages in order, "
                            "without falling. Results appear here when training finishes."),
                "story": {"shown": "The squat-jump phone video: two countermovement jumps.",
                          "learned": ("Training on an A100" if status.get("state") == "running" else "Not trained yet")
                                     + ": a policy rewarded for the stages, not for matching the clip frame by frame.",
                          "how": "Stage completion, falls and jump height, judged on the robot's own body."},
                "duration_s": None, "stages": None, "primary": None, "grid": [], "extra": [], "metrics": [],
                "sources": [], "notes": []}
    ev = read_json(run / "eval.json")
    status = read_json(run / "status.json") if (run / "status.json").exists() else {}
    cmd = " ".join(status.get("cmd", []))
    target = (re.search(r"--target (\S+)", cmd) or [None, "squat_jump"])[1]
    keys = [k for k in ev if not k.startswith(("_", "prev_")) and k != "retarget" and isinstance(ev[k], dict)]
    keys.sort(key=lambda k: (not ev[k].get("task_success", False), list(POLICY_INFO).index(k) if k in POLICY_INFO else 99))
    clip = next((d for d in [run / "clips" / target, runs / "remote_latent" / "clips" / target] if (d / "side_by_side.mp4").exists()), None)
    spec_f = next((f for f in [run / "task_spec.json", runs / "goal_relevance" / target / "task_spec.json"] if f.exists()), None)
    spec = read_json(spec_f) if spec_f else None
    rows = policy_rows(ev, keys)
    best = keys[0]
    e = ev[best]
    takeoff = next((s["start_s"] for s in (spec or {}).get("stages", []) if s["kind"] == "jump"), None)
    primary = (media.video(base["id"], "source", clip / "side_by_side.mp4", label="Phone video → G1 reference",
                           panels=_sbs_panels(clip / "side_by_side.mp4"), poster_at=takeoff) if clip else None)
    succ = [k for k in keys if ev[k].get("task_success")]
    metrics = []
    if "task_stages" in e:
        metrics.append({"label": "Stages completed", "display": e["task_stages"],
                        "note": f"{POLICY_INFO.get(best, (best,))[0]} policy, in order", "tone": "good" if e.get("task_success") else None})
    if e.get("falls") is not None:
        metrics.append({"label": "Falls", "display": str(e["falls"]), "note": "in the evaluation rollout",
                        "tone": "good" if e["falls"] == 0 else "bad"})
    if e.get("task_jump_quality"):
        metrics.append({"label": "Jump quality", "display": " / ".join(f"{x:.2f}" for x in e["task_jump_quality"]),
                        "note": "per jump; 1 = apex within the clip's height range"})
    if e.get("aligned") is not None and "retarget" in ev:
        metrics.append({"label": "V-JEPA 2 similarity", "display": f"{e['aligned']:.2f}",
                        "note": f"the retarget itself scores {ev['retarget']['aligned']:.2f}"})
    stage_txt = " → ".join(STAGE_LABEL.get(s["kind"], s["kind"]) for s in spec["stages"]) if spec else "the action's stages"
    return {**base, "status": "ready",
            "tagline": (f"Rewarded for the action, not for copying the clip: "
                        + (f"the {' and '.join(POLICY_INFO.get(k, (k,))[0] for k in succ)} "
                           f"polic{'y' if len(succ) == 1 else 'ies'} completed every stage ({e.get('task_stages')})"
                           if succ else f"the best policy completed {e.get('task_stages', '?')} stages")
                        + (f" with {e['falls']} falls." if e.get("falls") is not None else ".")),
            "story": {
                "shown": f"The {target.replace('_', ' ')} phone video, retargeted to G1 proportions. Its stages: {stage_txt}.",
                "learned": ("A policy rewarded mainly for completing those stages in order, with V-JEPA 2 similarity "
                            "as a style term weighted toward the moments that matter. The episode ends only on a fall."),
                "how": "A stage checker reads only the robot's own body: feet on or off the floor, centre-of-mass height.",
            },
            "duration_s": clip_duration(clip) if clip else None,
            "stages": task_stages(spec) if spec else None,
            "primary": primary,
            "grid": policy_videos(media, base["id"], run, keys, poster_at=takeoff),
            "grid_caption": "Solid robot: the trained policy, physically simulated. Green ghost: the reference it was shown.",
            "extra": ([media.video(base["id"], "vjepa_view", run / "comparison.mp4",
                                   label="What V-JEPA 2 compares: silhouettes of the video and each robot",
                                   poster_at=takeoff)] if (run / "comparison.mp4").exists() else []),
            "metrics": metrics,
            "policies": rows,
            "columns": columns_for(rows),
            "sources": [_rel(runs, run / "eval.json")] + ([_rel(runs, spec_f)] if spec_f else []),
            "notes": []}


# --- small formatting helpers ---------------------------------------------------------------------

def _n(k, word):
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return f"{words[k] if k < len(words) else k} {word}{'' if k == 1 else 's'}"


def _rel(runs, p):
    return f"outputs/{Path(p).relative_to(runs)}"


def _even_panels(labels):
    n = len(labels)
    return [{"label": l, "from": r(i / n), "to": r((i + 1) / n)} for i, l in enumerate(labels)]


def _sbs_panels(path):
    w, h, _ = probe(path)
    return [{"label": "Athlete", "from": 0, "to": r((w - h) / w)},
            {"label": "G1 reference", "from": r((w - h) / w), "to": 1, "tone": "good"}]


# --- build / check -------------------------------------------------------------------------------

BUILDERS = ["squat-v1", "retarget", "goal-relevance", "robustness", "squat-jump-final"]


def build(runs, out, force=False):
    media = Media(out, force)
    exs, built = [], {}
    for name in BUILDERS:
        print(f"{name}:")
        t = time.time()
        ex = {"squat-v1": lambda: ex_squat_v1(runs, media),
              "retarget": lambda: ex_retarget(runs, media),
              "goal-relevance": lambda: ex_goal_relevance(runs, media),
              "robustness": lambda: ex_robustness(runs, built["squat-v1"]["grid"]),
              "squat-jump-final": lambda: ex_final(runs, media)}[name]()
        built[name] = ex
        ex.setdefault("status", "ready")
        (out / ex["id"]).mkdir(parents=True, exist_ok=True)
        (out / ex["id"] / "example.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False) + "\n")
        exs.append(ex)
        print(f"  ok ({time.time() - t:.1f} s)")
    # The finished task policy leads once it exists; a pending slot goes last.
    exs.sort(key=lambda e: (e["status"] != "ready", 0 if e["id"] == "squat-jump-final" else 1))
    index = {"version": 1, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "examples": [{k: e.get(k) for k in ("id", "kicker", "title", "tagline", "status")}
                          | {"poster": _poster(e), "video": (e.get("primary") or {}).get("src")} for e in exs]}
    (out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    # Drop media no example references any more (e.g. a renamed video).
    keep = {out / p for p in referenced(out)} | {out / e["id"] / "example.json" for e in exs} | {out / "index.json"}
    for f in out.rglob("*"):
        if f.is_file() and f not in keep:
            f.unlink()
            print(f"removed stale {f.relative_to(out)}")
    return check(out)


def _poster(e):
    for v in [e.get("primary"), *e.get("grid", []), *e.get("extra", [])]:
        if v:
            return v["poster"]
    return None


def referenced(out):
    """Every media path the examples' JSON points at, relative to out/."""
    refs = []
    index = read_json(out / "index.json")
    for item in index["examples"]:
        ex = read_json(out / item["id"] / "example.json")
        for v in [ex.get("primary"), *ex.get("grid", []), *ex.get("extra", [])]:
            if v:
                refs += [v["src"], v["poster"]]
        refs += [p for p in (item.get("poster"), item.get("video")) if p]
    return sorted(set(refs))


REQUIRED = ["id", "kicker", "title", "tagline", "status", "story", "metrics", "sources"]


def check(out):
    """Validate the built examples: JSON shape, every referenced file present, size budgets."""
    problems = []
    if not (out / "index.json").exists():
        print(f"FAIL: {out / 'index.json'} missing; run scripts/build_examples.py")
        return 1
    index = read_json(out / "index.json")
    ids = [e["id"] for e in index["examples"]]
    for i in ids:
        f = out / i / "example.json"
        if not f.exists():
            problems.append(f"{f} missing")
            continue
        ex = read_json(f)
        problems += [f"{i}: no '{k}'" for k in REQUIRED if k not in ex]
        if ex.get("status") == "ready" and not (ex.get("primary") or ex.get("grid") or ex.get("robust")):
            problems.append(f"{i}: ready but has nothing to show")
        if ex.get("status") == "ready" and not ex.get("sources"):
            problems.append(f"{i}: ready but cites no source files")
    for rel in referenced(out):
        p = out / rel
        if not p.is_file():
            problems.append(f"missing {rel}")
        elif p.stat().st_size > MAX_FILE_MB * 1e6:
            problems.append(f"{rel} is {p.stat().st_size / 1e6:.1f} MB (> {MAX_FILE_MB} MB)")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    if total > MAX_TOTAL_MB * 1e6:
        problems.append(f"examples total {total / 1e6:.1f} MB (> {MAX_TOTAL_MB} MB)")
    for p in problems:
        print(f"FAIL: {p}")
    ready = sum(1 for i in ids if read_json(out / i / "example.json").get("status") == "ready") if not problems else 0
    print(f"{'ok' if not problems else 'FAILED'}: {len(ids)} examples ({ready} ready), "
          f"{len(referenced(out))} media files, {total / 1e6:.1f} MB total")
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=None, help="folder of runs (default: outputs/, or the main checkout's)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--assets", type=Path, default=None,
                    help="assets folder with mujoco_menagerie, for the hype render (default: A2H_ASSETS_DIR or "
                         "<runs>/../assets)")
    ap.add_argument("--force", action="store_true", help="re-encode every video")
    ap.add_argument("--check", action="store_true", help="only validate the built examples")
    a = ap.parse_args(argv)
    if a.check:
        return check(a.out)
    runs = (a.runs or default_runs_dir()).resolve()
    assets = a.assets or (runs.parent / "assets" if (runs.parent / "assets" / "mujoco_menagerie").is_dir() else None)
    print(f"runs: {runs}\nout:  {a.out}")
    if assets:  # config.py reads it at import: the G1 model for task stages and the hype render
        os.environ.setdefault("A2H_ASSETS_DIR", str(assets.resolve()))
    a.out.mkdir(parents=True, exist_ok=True)
    return build(runs, a.out, a.force)


if __name__ == "__main__":
    sys.exit(main())
