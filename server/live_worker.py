"""The live runner's worker: an uploaded video -> its RL task environment, on CPU, in seconds.

    python -m server.live_worker <video> <out_dir> [--max-seconds 20]

Run by server/live.py as a sandboxed subprocess (its own session, no secrets in its environment,
CPU / memory / wall-clock limits). It writes into out_dir and reports on stdout, one JSON object per
line: {"stage", "progress"} while a stage runs, {"stage", "done": true, "log"} when it ends, and
{"error"} (exit 2) for a clip it can't use. Anything else that goes wrong exits 1 with the traceback on
stderr, which the server logs and never shows.

  video        ffmpeg re-encodes the upload: first --max-seconds s, longer side <= 640 px, 30 fps,
               H.264, no audio or metadata -> processed.mp4 (also what the browser plays)
  pose         MediaPipe on the whole frame (no SAM 3.1 on CPU, so: one person in shot); or, with --sam3d, SAM 3D
               Body's output for these same frames (frontends/sam3d_run.py on processed.mp4, run on a GPU
               beforehand: scripts/build_live_demo.py's recorded builds)
  retarget     mink IK onto the Unitree G1
  environment  skill/keypose.py: key moments, contacts, the reward machine, and the retarget's own
               score against it (the self-check) -> task_spec.json

live.json grows as stages finish: what the page draws (2D landmarks and the 3D skeleton, the robot's bodies
and qpos, timings), then a preview of the clip's variants (skill/variations.py, best effort, after the
environment: the build view's ghosts) and the training environment they go into (env.zip, a2h_env/export.py:
the reference, the spec and the variants made, downloadable from the page).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FPS = 30
MAX_SIDE = 640
MIN_SECONDS = 1.0
MIN_POSE_FOUND = 0.5
VARIANT_MAX_S = 12  # longer clips get the four ghost variants only (CPU time)
# The variants run after the environment is complete, under the job's own wall clock: skipped once the build has
# used this share of its timeout, and stopped (keeping what is done) at the other share or VARIANT_BUDGET_S.
VARIANT_START_BY, VARIANT_STOP_BY, VARIANT_BUDGET_S = 0.4, 0.7, 45.0
# --pose modal: the GPU pose may use this share of the job's timeout (a cold container boots in it), the rest is
# the CPU stages'; the mask video it leaves beside processed.mp4 is what the build view plays over the clip
GPU_POSE_SHARE, MASK_VIDEO = 0.6, "mask.mp4"
# MediaPipe landmarks drawn over the video: the body, not the face or hands
LANDMARKS = {0: "nose", 11: "left shoulder", 12: "right shoulder", 13: "left elbow", 14: "right elbow",
             15: "left wrist", 16: "right wrist", 23: "left hip", 24: "right hip", 25: "left knee",
             26: "right knee", 27: "left ankle", 28: "right ankle", 29: "left heel", 30: "right heel",
             31: "left toe", 32: "right toe"}
LANDMARK_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25),
                  (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)]
# latent/reward_net.py BODIES: the G1 bodies the task spec scores (skill/task.py indexes them)
BODIES = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
          "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
          "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
          "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")
BODY_NAMES = ("pelvis", "left hip", "left knee", "left ankle", "right hip", "right knee", "right ankle", "torso",
              "left shoulder", "left elbow", "left wrist", "right shoulder", "right elbow", "right wrist")
BODY_EDGES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (7, 11), (11, 12), (12, 13)]


class ClipError(Exception):
    """A clip this pipeline can't use; the message is shown to the user."""


def emit(**msg):
    print(json.dumps(msg), flush=True)


def transcode(src, dst, max_seconds):
    import imageio_ffmpeg
    scale = (f"scale='if(gt(iw,ih),min({MAX_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({MAX_SIDE},ih))'"
             f":flags=bilinear,fps={FPS},format=yuv420p")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
           "-t", str(max_seconds), "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
           "-vf", scale, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart",
           "-threads", "2", "-y", str(dst)]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        raise ClipError("couldn't decode that video (no readable video stream)")


def body_positions(model, qpos):
    """(T, 36) G1 qpos -> (T, len(BODIES), 3) world body positions (as latent.distill.body_positions)."""
    import mujoco
    data = mujoco.MjData(model)
    ids = [model.body(b).id for b in BODIES]
    out = np.empty((len(qpos), len(ids), 3), np.float32)
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        out[t] = data.xpos[ids]
    return out


def _round(a, nd):
    """Nested lists, NaN -> None (JSON null)."""
    a = np.round(np.asarray(a, np.float64), nd)
    return [[None if np.isnan(p).any() else p.tolist() for p in f] for f in a]


def gpu_pose(video, out, n_frames, width, height, timeout_s, log):
    """The processed clip through SAM 3.1 and SAM 3D Body on the Modal GPU (server/live_pose.py), waiting at most
    timeout_s. -> (sam3d.npz path, (T, H, W) masks, the result dict), or None with the reason given to log."""
    from server import live_pose

    if not live_pose.configured():
        log("no Modal token on this host")
        return None
    try:
        r = live_pose.fetch(Path(video).read_bytes(), timeout_s)
        why = live_pose.check(r, n_frames, width, height)
        if why:
            raise RuntimeError(why)
        path, masks = live_pose.save(r, out)
        return path, masks, r
    except Exception as e:  # noqa: BLE001  (the GPU is optional: the build goes on without it)
        log(f"{type(e).__name__}: {str(e)[:160]}" if str(e) else type(e).__name__)
        return None


def run(video, out, max_seconds, name="clip", sam3d=None, timeout_s=240.0, job_id=None, pose_backend="cpu"):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    live = {"fps": FPS, "frames": 0, "duration_s": 0.0, "video": None, "landmarks": None, "pose_found": None,
            "skeleton": None, "robot": None, "qpos": None, "retarget_error_cm": None, "variants": None, "timings_s": {}}

    def save():
        tmp = out / "live.json.tmp"
        tmp.write_text(json.dumps(live, separators=(",", ":")))
        tmp.replace(out / "live.json")

    def done(stage, t0, log):
        live["timings_s"][stage] = round(time.monotonic() - t0, 2)
        save()
        emit(stage=stage, done=True, log=log)

    # video
    t0 = time.monotonic()
    emit(stage="video", progress=0.1)
    processed = out / "processed.mp4"
    transcode(video, processed, max_seconds)
    import imageio_ffmpeg
    import mediapy
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())  # no system ffmpeg on the Space
    frames = np.asarray(mediapy.read_video(str(processed)))
    if frames.ndim != 4 or len(frames) < MIN_SECONDS * FPS:
        raise ClipError(f"the clip is too short (needs at least {MIN_SECONDS:g} s)")
    T, H, W = frames.shape[:3]
    live.update(frames=T, duration_s=round(T / FPS, 3), video={"width": W, "height": H})
    done("video", t0, f"video: {T / FPS:.1f} s at {FPS} fps, {W}x{H} (first {max_seconds:g} s, rescaled)")

    # pose
    t0 = time.monotonic()
    import pose
    import track
    emit(stage="pose", progress=0.0)
    d = None
    if pose_backend == "modal" and not sam3d:
        # the full SAM pipeline on a Modal GPU (server/live_pose.py); any failure -> MediaPipe here, with a line why
        emit(stage="pose", note="SAM 3.1 and SAM 3D Body on a Modal GPU (a cold start adds a minute or two)")
        got = gpu_pose(processed, out, T, W, H, max(30.0, timeout_s * GPU_POSE_SHARE - (time.monotonic() - started)),
                       lambda why: emit(stage="pose", note=f"GPU pose unavailable ({why}); MediaPipe on the CPU instead"))
        if got is not None:
            sam3d, masks, r = got
            seen = float(masks.any((1, 2)).mean())
            live["segmenter"], live["mask_seen"] = "SAM 3.1", round(seen, 3)
            live["gpu"] = {"name": str(r.get("gpu", ""))[:40],
                           "timings_s": {str(k)[:20]: round(float(v), 2) for k, v in (r.get("timings_s") or {}).items()
                                         if isinstance(v, (int, float))}}
            mediapy.write_video(str(out / MASK_VIDEO), track.overlay(frames, masks), fps=FPS)  # the build view's mask
            emit(stage="pose", note=f"SAM 3.1: person visible in {seen:.0%} of frames", progress=0.8)
    if sam3d:
        from frontends import sam3d_body  # not in the Space image: imported only when asked for
        d = sam3d_body.cached(sam3d, frames)
        if d is None:
            raise RuntimeError(f"{sam3d} is not SAM 3D Body's output for these {T} frames at {W}x{H}")
        world, image_xy, boxes = sam3d_body.landmarks(d)
    else:
        crops, boxes = track.crop_to_mask(frames, np.ones((T, H, W), bool))
        world, image_xy, _ = pose.landmarks_from_crops(crops, FPS, progress=lambda i, n: emit(stage="pose", progress=i / n))
        del crops
    model = "SAM 3D Body" if d is not None else "MediaPipe"
    live["pose_model"] = model
    found = float((~np.isnan(world[:, 0, 0])).mean())
    x0, y0, side = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3]
    px = np.stack([(x0 + image_xy[..., 0] * side) / W, (y0 + image_xy[..., 1] * side) / H], -1)
    keep = list(LANDMARKS)
    live["landmarks"] = {"names": [LANDMARKS[i] for i in keep],
                         "edges": [[keep.index(a), keep.index(b)] for a, b in LANDMARK_EDGES],
                         "frames": [None if np.isnan(f).all() else f for f in _round(px[:, keep], 4)]}
    live["pose_found"] = round(found, 3)
    if found < MIN_POSE_FOUND:
        save()
        raise ClipError(f"a person was found in only {found:.0%} of frames; film one person, whole body in shot")
    # the 3D skeleton the retarget reads: metres, z up, the athlete facing +x (ground not yet at 0)
    human = pose.human_trajectory(world, image_xy, boxes)
    live["skeleton"] = {"names": [LANDMARKS[i] for i in keep], "edges": live["landmarks"]["edges"],
                        "frames": _round(human[:, keep], 3)}
    done("pose", t0, f"pose ({model}) found in {found:.0%} of frames")

    # retarget
    t0 = time.monotonic()
    import retarget
    emit(stage="retarget", progress=0.1)
    rt = retarget.G1Retargeter()
    targets, rotations = retarget.human_to_targets(human, rt, fps=FPS)
    emit(stage="retarget", progress=0.4)
    qpos = rt.retarget_sequence(targets, rotations)
    report = retarget.tracking_report(rt, qpos, targets)
    pos = body_positions(rt.model, qpos)
    np.save(out / "qpos.npy", qpos)
    live["robot"] = {"names": list(BODY_NAMES), "edges": [list(e) for e in BODY_EDGES], "frames": _round(pos, 3)}
    live["qpos"] = [[round(float(x), 4) for x in q] for q in qpos]
    live["retarget_error_cm"] = {k: [round(float(x), 1) for x in v] for k, v in report.items()}
    mean = np.mean([v[0] for v in report.values()]) if report else float("nan")
    done("retarget", t0, f"retarget error {mean:.1f} cm (mean over end effectors)")

    # environment
    t0 = time.monotonic()
    from skill import keypose
    from skill import task as skill_task
    emit(stage="environment", progress=0.2)
    try:
        spec = keypose.compile_keypose_spec(pos, FPS, name=name)
    except ValueError as e:  # the compiler's own verdict on the clip (too many moments even at a coarse tolerance)
        raise ClipError(f"couldn't compile the key moments: {e}") from e
    spec["self_check"] = skill_task.evaluate({**spec, "down": spec["down_ref"], "air": spec["air_ref"]}, pos, FPS)
    (out / "task_spec.json").write_text(json.dumps(spec))
    sc = spec["self_check"]
    done("environment", t0, f"task spec: {spec['summary']} (the retarget scores "
                            f"{sc['stages_completed']}/{sc['stages_total']})")

    # variants (best effort: the environment is done, and a failure or running out of time never fails the build)
    vs = []
    used = time.monotonic() - started
    if used > VARIANT_START_BY * timeout_s:
        emit(stage="environment", note=f"variants skipped (the build already took {used:.0f} s)")
    else:
        try:
            deadline = min(started + VARIANT_STOP_BY * timeout_s, time.monotonic() + VARIANT_BUDGET_S)
            vs = variant_set(qpos, rt.model, deadline)
            if not vs:
                raise TimeoutError("no time left for variants")
            live["variants"] = variant_view(vs)
            save()
            v = live["variants"]
            emit(stage="environment", note=f"variants: {v['kept']} of {v['total']} kept (mirrored, re-timed, rescaled; "
                                           f"each passes its own key-moment check and the physics filter)")
        except Exception as e:  # noqa: BLE001
            vs = []
            emit(stage="environment", note=f"variants skipped ({type(e).__name__})")
    # the training environment: the variants stage's output, a zip anyone with mjlab can train on (a2h_env/export.py);
    # with no variants it still holds the reference and the spec
    try:
        from a2h_env import export as env_export
        m = env_export.export_live(out, qpos=qpos, fps=FPS, spec=spec, variants=vs, pos=pos, name=name, job_id=job_id,
                                   meta={"frontend": model, "retarget_version": retarget.code_version()})
        tr = m["variants"].get("train") or {}
        live["env"] = {"file": env_export.ZIP_NAME, "filename": f"{m['id']}.a2h-env.zip", "size": m["bytes"], "inside": m["inside"],
                       "variants": tr.get("with_arrays", 0)}
        save()
        emit(stage="environment", note=f"training environment: {m['bytes'] / 1e6:.1f} MB zip ({m['inside']})")
    except Exception as e:  # noqa: BLE001
        emit(stage="environment", note=f"training environment skipped ({type(e).__name__})")


def variant_set(qpos, model, deadline=None):
    """skill/variations.py on this retarget: the four ghost variants first, then the rest of PREVIEW's 18 (none above
    VARIANT_MAX_S), until the deadline -> generate()'s list (those made in time; [] if none)."""
    from skill import variations
    g1 = variations.G1()
    g1.bodies = lambda q: body_positions(model, q)  # this module's copy: latent/ isn't in the Space image
    ghosts = [g[:4] for g in variations.GHOSTS]
    grid = ghosts + ([g for g in variations.PREVIEW if g not in ghosts] if len(qpos) <= VARIANT_MAX_S * FPS else [])
    return variations.generate(qpos, fps=FPS, grid=grid, g1=g1, log=lambda *a: None, deadline=deadline)


def variant_view(vs):
    """generate()'s list -> the build view's preview (every verdict, a few kept ones' qpos as ghosts)."""
    from skill import variations
    by_name = {v["name"]: v["qpos"] for v in vs}
    return variations.preview(vs, by_name.__getitem__, fps=FPS)


def variant_preview(qpos, model, deadline=None):
    """variant_set + variant_view; TimeoutError when no variant was made before the deadline."""
    vs = variant_set(qpos, model, deadline)
    if not vs:
        raise TimeoutError("no time left for variants")
    return variant_view(vs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--name", default="clip")
    ap.add_argument("--sam3d", help="SAM 3D Body's npz for this video's processed frames: the pose instead of MediaPipe")
    ap.add_argument("--timeout-s", type=float, default=240.0, help="the job's wall-clock limit (the variants' budget)")
    ap.add_argument("--job-id", help="the job's id, named in the training environment's manifest")
    ap.add_argument("--pose", choices=("cpu", "modal"), default="cpu",
                    help="modal: SAM 3.1 + SAM 3D Body on the deployed Modal GPU app (server/live_pose.py), MediaPipe "
                         "on the CPU if that fails; cpu: MediaPipe")
    a = ap.parse_args(argv)
    try:
        run(Path(a.video), Path(a.out), a.max_seconds, a.name, a.sam3d, a.timeout_s, a.job_id, a.pose)
    except ClipError as e:
        emit(error=str(e))
        return 2
    except subprocess.TimeoutExpired:
        emit(error="decoding the video took too long")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
