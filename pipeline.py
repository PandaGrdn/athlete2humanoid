"""Video -> G1 reference motion (mjlab CSV) + a side-by-side preview.

    python pipeline.py clips/squat.mp4                # SAM 3.1 tracking (needs a GPU)
    python pipeline.py clips/squat.mp4 --no-sam       # whole frame, single person (any machine)
    python pipeline.py clips/squat.mp4 --skill live   # then Muse Spark writes skill_spec.json
    python pipeline.py clips/squat.mp4 --task-spec    # then the task (skill/keypose.py): task_spec.json, its
                                                      # key moments named by Muse Spark when MUSE_API_KEY is set
    python pipeline.py clips/meet.mp4 --athlete 640,360   # several people: the one under this pixel
                                                          # on the first frame (or x0,y0,x1,y1, or an index)
    python pipeline.py clips/squat.mp4 --frontend mediapipe   # 3D pose from MediaPipe even where SAM 3D Body runs

3D pose comes from SAM 3D Body where it is set up (scripts/setup_sam3d.sh, a GPU; frontends/), else MediaPipe
(any machine): docs/PIPELINE_ACCURACY.md, fix 3.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import mediapy
import numpy as np

import athlete
import camera
import config
import pose
import render
import retarget
import track
from frontends import sam3d_body


def video_fps(video_path):
    """The video's real frame rate, from its decoded frames' timestamps: (frames - 1) / (last - first
    timestamp). The rate in the container header (what mediapy reports) can be off: our phone clips
    say 27.7-29.6 fps but their frames are 1/30 s apart. -> {"fps", "frames", "container_fps",
    "irregular_gaps": intervals over 1.5x the median (dropped frames)}."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(video_path), "-map", "0:v:0", "-vf", "showinfo", "-f", "null", "-"]
    log = subprocess.run(cmd, capture_output=True, text=True).stderr
    t = np.array([float(x) for x in re.findall(r"pts_time:\s*(-?[\d.]+)", log)])
    nominal = re.search(r"Video:.*?([\d.]+) fps", log)
    if len(t) < 2 or t[-1] <= t[0]:
        raise ValueError(f"{video_path}: no frame timestamps from ffmpeg")
    dt = np.diff(t)
    return {"fps": round(float((len(t) - 1) / (t[-1] - t[0])), 4), "frames": len(t),
            "container_fps": float(nominal.group(1)) if nominal else None,
            "irregular_gaps": int((dt > 1.5 * np.median(dt)).sum())}


def video_to_motion(video_path, out_dir=None, use_sam=True, prompt=config.SAM_PROMPT, athlete_select=None,
                    compensate_camera=True, frontend="auto"):
    video_path = Path(video_path)
    out_dir = Path(out_dir or config.OUTPUT_DIR / video_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = mediapy.read_video(str(video_path))
    frames = np.asarray(video)
    timing = video_fps(video_path)
    fps = timing["fps"]
    if timing["frames"] != len(frames):
        print(f"warning: ffmpeg timestamped {timing['frames']} frames, mediapy read {len(frames)}")
    if abs(fps - video.metadata.fps) > 0.01 * fps:
        print(f"frame rate: {fps:.3f} fps from the timestamps; the file's header says {video.metadata.fps:g}")

    if use_sam:
        masks = track.athlete_masks(video_path, len(frames), prompt=prompt, select=athlete_select)
        print(f"SAM 3.1: athlete visible in {masks.any((1, 2)).mean():.0%} of frames")
        np.savez_compressed(out_dir / "masks.npz", masks=masks)  # for the latent (silhouette) stage
        render.save_video(track.overlay(frames, masks), out_dir / "tracking.mp4", fps=fps)
    elif athlete_select is not None:  # no SAM, several people: follow the chosen one (athlete.py, CPU)
        masks, _ = athlete.track_people(frames, athlete_select)
    else:
        masks = np.ones(frames.shape[:3], bool)

    # SAM 3D Body on the athlete's mask boxes (no mask: its own person detector). A sam3d.npz already in out_dir
    # for these frames is reused, so the later stages re-run without a GPU.
    d = sam3d_body.cached(out_dir / "sam3d.npz", frames) if frontend != "mediapipe" else None
    frontend = "sam3d" if d is not None else sam3d_body.choose(frontend)
    if frontend == "sam3d":
        if d is None:
            d = sam3d_body.run(video_path, len(frames), sam3d_body.mask_boxes(masks) if masks.mean() < 0.9 else None)
            np.savez_compressed(out_dir / "sam3d.npz", **d)
        else:
            print(f"pose: reusing {out_dir / 'sam3d.npz'}")
        world, image_xy, boxes = sam3d_body.landmarks(d)
    else:
        crops, boxes = track.crop_to_mask(frames, masks)
        world, image_xy, vis = pose.landmarks_from_crops(crops, fps)
    print(f"pose ({frontend}) found in {(~np.isnan(world[:, 0, 0])).mean():.0%} of frames")
    cam, intake = camera.intake(frames, masks, image_xy, boxes, compensate=compensate_camera)
    (out_dir / "camera.json").write_text(json.dumps(intake, indent=1))
    human, view_azimuth = pose.human_trajectory(world, image_xy, boxes, return_view=True, camera=cam)
    np.save(out_dir / "human.npy", human)

    rt = retarget.G1Retargeter()
    contacts = {}
    targets, rotations = retarget.human_to_targets(human, rt, fps=fps, report=contacts)
    for w in contacts["warnings"]:
        print("retarget warning:", w)
    qpos = rt.retarget_sequence(targets, rotations, fps=fps)
    report = retarget.tracking_report(rt, qpos, targets)
    (out_dir / "retarget_report.json").write_text(json.dumps(report, indent=1))
    print("retarget error, cm (mean, max):", {k: tuple(round(x, 1) for x in v) for k, v in report.items()})

    np.save(out_dir / "qpos.npy", qpos)
    csv = retarget.export_mjlab_csv(qpos, out_dir / "motion.csv")
    robot = render.render_qpos_sequence(rt.model, qpos, size=(480, 480))
    render.save_video(render.side_by_side(frames, robot), out_dir / "side_by_side.mp4", fps=fps)
    meta = {"fps": fps, "frames": len(qpos), "view_azimuth": view_azimuth, "sam": use_sam, "frontend": frontend,
            "video_timing": timing, "retarget": "proportional", "retarget_version": retarget.code_version(),
            "retarget_warnings": contacts["warnings"], "stance_frac": contacts["stance_frac"]}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"csv": csv, "out_dir": out_dir, **meta}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--prompt", default=config.SAM_PROMPT)
    ap.add_argument("--out")
    ap.add_argument("--athlete", help="pick the athlete on the first frame: x,y (a click), x0,y0,x1,y1 (a box) "
                    "or an index counted from the left; default SAM's most-seen person")
    ap.add_argument("--no-camera", action="store_true", help="skip the pan/zoom compensation (camera.py)")
    ap.add_argument("--frontend", choices=["auto", "sam3d", "mediapipe"], default="auto",
                    help="3D pose: SAM 3D Body (needs scripts/setup_sam3d.sh and a GPU) or MediaPipe; auto = SAM 3D Body "
                         "where it is set up")
    ap.add_argument("--skill", choices=["live", "dry-run"],
                    help="Muse Spark reads the clip -> skill_spec.json (muse/skill_reader.py)")
    ap.add_argument("--task-spec", action="store_true",
                    help="last stage: the key-moment task (skill/keypose.py) -> task_spec.json; with MUSE_API_KEY set, "
                         "Muse Spark names its moments (muse/moment_namer.py), else they keep the measured names")
    a = ap.parse_args()
    res = video_to_motion(a.video, a.out, use_sam=not a.no_sam, prompt=a.prompt, athlete_select=a.athlete,
                          compensate_camera=not a.no_camera, frontend=a.frontend)
    if a.skill:
        from muse import client, skill_reader
        skill_reader.read_skill(a.video, res["out_dir"],
                                client.make_client(a.skill == "dry-run", default_fixture="skill_spec_squat.json"))
    if a.task_spec:
        from muse.moment_namer import name_if_available
        from skill import keypose
        spec, spec_f = keypose.compile_clip(res["out_dir"])
        print(f"task spec: {spec['summary']}")
        name_if_available(spec_f, video=a.video)
        res["task_spec"] = str(spec_f)
    print(res)
