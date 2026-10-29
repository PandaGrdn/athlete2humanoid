"""Recorded live builds for the static site: /live replays them when no pipeline API is reachable.

    python scripts/build_live_demo.py            # -> web/public/live-demo/<id>/ and index.json
    python scripts/build_live_demo.py --muse     # also have Muse name the key moments (needs MUSE_API_KEY)

Each sample clip goes through the real live worker (server/live_worker.py, as the API runs it), with
its stage messages recorded at the times they arrived: the replay is that build, not an animation.
Then the policies trained on that clip (outputs/), with their training curves from train.log and
each rollout scored under today's keypose checker (skill/keypose.py), so a policy that passed an
older, looser checker says so.

Writes per clip: processed.mp4, live.json (LiveData), task_spec.json (the API's task-spec view),
replay.json (events, policies) and policy_<name>.mp4/.jpg.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "web" / "public" / "live-demo"
RUNS = REPO / "outputs"

# id, clip (clips/<name>.*), title, [(policy label key, run/policy folder)]
CLIPS = [
    ("front-kick", "front_kick", "Front kick",
     [("tracking", "remote_latent_kp_kick/policy_tracking"), ("taskhybrid", "remote_latent_fix_kick/policy_taskhybrid"),
      ("task", "remote_latent_fix_kick/policy_task"),
      ("tracking_var", "remote_variants_var_kick/policy_tracking_var"),
      ("taskhybrid_var", "remote_variants_var_kick/policy_taskhybrid_var")]),
    ("squat-jump", "squat_jump", "Squat jump",
     [("tracking", "remote_latent_final/policy_tracking"), ("taskhybrid", "remote_latent_fix_sj/policy_taskhybrid"),
      ("task", "remote_latent_fix_sj/policy_task"),
      ("tracking_var", "remote_variants_var_sj/policy_tracking_var"),
      ("taskhybrid_var", "remote_variants_var_sj/policy_taskhybrid_var")]),
    ("jumping-jacks", "jumping_jacks", "Jumping jacks", [("task", "remote_latent_kp_jacks/policy_task")]),
    ("lunge", "lunge", "Lunge", [("task", "remote_latent_loose_lunge/policy_task")]),
    ("punches", "punches", "Punches", [("task", "remote_latent_loose_punch/policy_task")]),
]
POLICY_LABEL = {"tracking": "Tracking", "task": "Task", "taskhybrid": "Task + tracking", "taskphase": "Task (no reference)",
                "tracking_var": "Tracking, trained on variants", "taskhybrid_var": "Task + tracking, trained on variants"}
STAGES = [("video", "Video check"), ("pose", "Pose"), ("retarget", "Retarget"), ("environment", "Task environment"),
          ("policy", "Policy")]


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def web_video(src, dst, height=360):
    """H.264, no audio, faststart; and a poster frame next to it."""
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-vf", f"scale=-2:{height}", "-an",
                    "-c:v", "libx264", "-crf", "28", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(dst)], check=True)
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-ss", "1", "-i", str(dst), "-frames:v", "1", "-q:v", "4",
                    str(dst.with_suffix(".jpg"))], check=True)


def record_build(video, out):
    """Run the live worker on video as the API does; its stage messages with their arrival times."""
    if out.exists():
        shutil.rmtree(out)
    (out / "scratch").mkdir(parents=True)
    from server.live import _env
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable, "-m", "server.live_worker", str(video), str(out), "--max-seconds", "20"],
                            cwd=REPO, env=_env(out), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    events = []
    for line in proc.stdout:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if isinstance(msg, dict) and msg.get("stage"):
            events.append({"t": round(time.monotonic() - t0, 2), **{k: msg[k] for k in ("stage", "progress", "done", "log") if k in msg}})
    if proc.wait() != 0:
        raise RuntimeError(f"the live worker failed on {video}")
    shutil.rmtree(out / "scratch", ignore_errors=True)
    return events


def curve(train_log, points=60):
    """(iteration, mean episode reward, mean episode length) from mjlab's console log, thinned."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", Path(train_log).read_text(errors="replace"))
    rows = []
    for block in text.split("Learning iteration ")[1:]:
        it = re.match(r"(\d+)/(\d+)", block)
        rew = re.search(r"Mean reward:\s*(-?[\d.]+)", block)
        length = re.search(r"Mean episode length:\s*([\d.]+)", block)
        if it and rew and length:
            rows.append((int(it.group(1)), float(rew.group(1)), float(length.group(1))))
    if not rows:
        return None
    total = int(re.search(r"Learning iteration \d+/(\d+)", text).group(1))
    keep = rows[:: max(1, len(rows) // points)] + [rows[-1]]
    return {"iterations": total, "points": [list(r) for r in keep]}


def score(policy_dir, clip):
    """The rollout under today's checker, against the spec its own training clip compiles to now (the
    live worker's retarget of the phone video differs a little from the training run's)."""
    from skill import task as T
    from skill.keypose import compile_keypose_spec
    run = policy_dir.parent
    ref_dir = next(d for d in (run / "clips" / clip, RUNS / "remote_latent_final" / "clips" / clip) if (d / "qpos.npy").exists())
    fps = json.loads((ref_dir / "meta.json").read_text())["fps"]
    spec = compile_keypose_spec(T.body_positions_from_qpos(np.load(ref_dir / "qpos.npy")), fps)
    q = np.load(policy_dir / "policy_qpos.npy")
    r = T.evaluate(spec, T.body_positions_from_qpos(q), 50)
    return {"stages_completed": r["stages_completed"], "stages_total": r["stages_total"], "success": r["success"],
            "fell_at_s": r["fell_at_s"]}


def build(cid, clip, title, policies, muse=False):
    from server.runs import task_spec_view
    video = next(p for p in (REPO / "clips").glob(f"{clip}.*") if p.suffix.lower() in (".mov", ".mp4"))
    d = OUT / cid
    work = REPO / "build" / "live_demo" / cid
    print(f"{cid}: building the environment from {video.name}")
    events = record_build(video, work)
    if muse:
        from muse.client import MuseClient
        from muse.moment_namer import name_moments
        name_moments(work / "task_spec.json", MuseClient(), video=work / "processed.mp4")
    d.mkdir(parents=True, exist_ok=True)
    web_video(work / "processed.mp4", d / "processed.mp4", 480)
    shutil.copy(work / "live.json", d / "live.json")
    spec = json.loads((work / "task_spec.json").read_text())
    (d / "task_spec.json").write_text(json.dumps(task_spec_view(spec)))
    pols = []
    for key, rel in policies:
        src = RUNS / rel
        if not (src / "policy.mp4").exists():
            print(f"  {rel}: no rollout, skipped")
            continue
        web_video(src / "policy.mp4", d / f"policy_{key}.mp4")
        pols.append({"key": key, "label": POLICY_LABEL.get(key, key), "video": f"policy_{key}.mp4",
                     "poster": f"policy_{key}.jpg", "run": src.parent.name,
                     "curve": curve(src / "train.log") if (src / "train.log").exists() else None,
                     "checker": score(src, clip)})
        print(f"  {key}: {pols[-1]['checker']}")
    replay = {"id": cid, "title": title, "clip": video.name, "recorded": time.strftime("%Y-%m-%d"),
              "stages": [{"key": k, "label": label} for k, label in STAGES], "events": events, "policies": pols,
              "named_by": (spec.get("named_by") or {}).get("model")}
    (d / "replay.json").write_text(json.dumps(replay, indent=1))
    return {"id": cid, "title": title, "summary": spec.get("summary"), "duration_s": round(json.loads((d / "live.json").read_text())["duration_s"], 1),
            "poster": "processed.jpg", "policies": [p["label"] for p in pols]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", help="these clip ids")
    ap.add_argument("--muse", action="store_true", help="Muse names the key moments (MUSE_API_KEY)")
    a = ap.parse_args(argv)
    index_f = OUT / "index.json"
    index = {e["id"]: e for e in json.loads(index_f.read_text())["clips"]} if index_f.exists() else {}
    for cid, clip, title, policies in CLIPS:
        if a.only and cid not in a.only:
            continue
        index[cid] = build(cid, clip, title, policies, a.muse)
    order = [c[0] for c in CLIPS]
    index_f.write_text(json.dumps({"clips": sorted(index.values(), key=lambda e: order.index(e["id"]))}, indent=1))
    print(f"{index_f}: {len(index)} recorded builds")


if __name__ == "__main__":
    main()
