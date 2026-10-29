"""Video -> G1 reference motion (mjlab CSV) + a side-by-side preview.

    python pipeline.py clips/squat.mp4                # SAM 3.1 tracking (needs a GPU)
    python pipeline.py clips/squat.mp4 --no-sam       # whole frame, single person (any machine)
    python pipeline.py clips/squat.mp4 --skill live   # then Muse Spark writes skill_spec.json
"""

import argparse
import json
from pathlib import Path

import mediapy
import numpy as np

import config
import pose
import render
import retarget
import track


def video_to_motion(video_path, out_dir=None, use_sam=True, prompt=config.SAM_PROMPT):
    video_path = Path(video_path)
    out_dir = Path(out_dir or config.OUTPUT_DIR / video_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = mediapy.read_video(str(video_path))
    fps = video.metadata.fps
    frames = np.asarray(video)

    if use_sam:
        masks = track.athlete_masks(video_path, len(frames), prompt=prompt)
        print(f"SAM 3.1: athlete visible in {masks.any((1, 2)).mean():.0%} of frames")
        np.savez_compressed(out_dir / "masks.npz", masks=masks)  # for the latent (silhouette) stage
        render.save_video(track.overlay(frames, masks), out_dir / "tracking.mp4", fps=fps)
    else:
        masks = np.ones(frames.shape[:3], bool)
    crops, boxes = track.crop_to_mask(frames, masks)

    world, image_xy, vis = pose.landmarks_from_crops(crops, fps)
    print(f"pose found in {(~np.isnan(world[:, 0, 0])).mean():.0%} of frames")
    human, view_azimuth = pose.human_trajectory(world, image_xy, boxes, return_view=True)
    np.save(out_dir / "human.npy", human)

    rt = retarget.G1Retargeter()
    targets, rotations = retarget.human_to_targets(human, rt, fps=fps)
    qpos = rt.retarget_sequence(targets, rotations, fps=fps)
    report = retarget.tracking_report(rt, qpos, targets)
    (out_dir / "retarget_report.json").write_text(json.dumps(report, indent=1))
    print("retarget error, cm (mean, max):", {k: tuple(round(x, 1) for x in v) for k, v in report.items()})

    np.save(out_dir / "qpos.npy", qpos)
    csv = retarget.export_mjlab_csv(qpos, out_dir / "motion.csv")
    robot = render.render_qpos_sequence(rt.model, qpos, size=(480, 480))
    render.save_video(render.side_by_side(frames, robot), out_dir / "side_by_side.mp4", fps=fps)
    meta = {"fps": fps, "frames": len(qpos), "view_azimuth": view_azimuth, "sam": use_sam}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return {"csv": csv, "out_dir": out_dir, **meta}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--prompt", default=config.SAM_PROMPT)
    ap.add_argument("--out")
    ap.add_argument("--skill", choices=["live", "dry-run"],
                    help="last stage: Muse Spark reads the clip -> skill_spec.json (muse/skill_reader.py)")
    a = ap.parse_args()
    res = video_to_motion(a.video, a.out, use_sam=not a.no_sam, prompt=a.prompt)
    if a.skill:
        from muse import client, skill_reader
        skill_reader.read_skill(a.video, res["out_dir"],
                                client.make_client(a.skill == "dry-run", default_fixture="skill_spec_squat.json"))
    print(res)
