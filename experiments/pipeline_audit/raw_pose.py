"""Raw MediaPipe on the whole frame (no SAM crop), CPU: the file's real fps, detection rate, the athlete's size in
pixels, landmark visibility, pre- vs post-smoothing noise, bone-length jitter, and how consistent MediaPipe's 3D is
with its own 2D (weak-perspective residual). Writes <out>/<clip>.npz (landmarks) and raw_pose.json."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pose  # noqa: E402
from experiments.pipeline_audit import MAJOR, RESULTS, ROOT, VIDEOS, dump, noise  # noqa: E402

BONES = [(23, 25), (24, 26), (25, 27), (26, 28), (11, 13), (12, 14), (13, 15), (14, 16)]


def frames_of(path, max_side=1280):
    """Square-padded frames (as the pipeline reads the file: mediapy, rotation applied), the real fps, frame size."""
    import mediapy
    try:
        import imageio_ffmpeg
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    v = mediapy.read_video(str(path))
    fps, out = v.metadata.fps, []
    for f in np.asarray(v):
        h, w = f.shape[:2]
        side = max(h, w)
        sq = np.full((side, side, 3), 114, np.uint8)
        y0, x0 = (side - h) // 2, (side - w) // 2
        sq[y0:y0 + h, x0:x0 + w] = f
        if side > max_side:
            sq = cv2.resize(sq, (max_side, max_side), interpolation=cv2.INTER_AREA)
        out.append(sq)
    return np.stack(out), fps, (h, w), side


def summarize(world, xy, vis, fps, hw, side):
    T, det = len(world), ~np.isnan(world[:, 0, 0])
    row = dict(frames=T, fps=round(float(fps), 2), frame_hw=[int(hw[0]), int(hw[1])], detected_pct=round(100 * float(det.mean()), 1))
    if det.sum() < 5:
        return row
    px = xy * side
    row["person_height_px"] = round(float(np.nanmedian(np.abs(px[:, 0, 1] - np.maximum(px[:, 27, 1], px[:, 28, 1])))), 0)
    n_raw = noise(world[det][:, MAJOR], fps).mean(0) * 100
    try:
        n_sm = noise(pose.fill_and_smooth(world)[:, MAJOR], fps).mean(0) * 100
    except RuntimeError:
        n_sm = [np.nan] * 3
    L = [np.linalg.norm(world[:, i] - world[:, j], axis=-1) for i, j in BONES]
    cv = np.mean([np.nanstd(x) / np.nanmean(x) for x in L]) * 100
    res = []  # per frame: image_xy ~ s * world_xy + t on the major joints; residual back in metres
    for t in np.flatnonzero(det):
        A = np.c_[world[t, MAJOR, :2].reshape(-1), np.tile(np.eye(2), (len(MAJOR), 1))]
        b = px[t, MAJOR].reshape(-1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        res.append(np.sqrt(np.mean(((A @ sol - b) / sol[0]) ** 2)) * 100)
    hip_px = px[:, [23, 24]].mean(1)
    torso_px = np.linalg.norm(px[:, [11, 12]].mean(1) - hip_px, axis=-1)
    torso_m = np.linalg.norm(world[:, [11, 12]].mean(1) - world[:, [23, 24]].mean(1), axis=-1)
    mpp = torso_m / torso_px  # human_trajectory's ruler, per frame
    row |= dict(visibility=dict(ankles=round(float(np.nanmean(vis[det][:, [27, 28]])), 2), wrists=round(float(np.nanmean(vis[det][:, [15, 16]])), 2),
                                knees=round(float(np.nanmean(vis[det][:, [25, 26]])), 2),
                                ankle_frames_below_0_5=round(100 * float((vis[det][:, [27, 28]] < 0.5).any(1).mean()), 1)),
                noise_raw_cm=dict(x=round(float(n_raw[0]), 2), y=round(float(n_raw[1]), 2), depth=round(float(n_raw[2]), 2)),
                noise_smoothed_cm=dict(x=round(float(n_sm[0]), 2), y=round(float(n_sm[1]), 2), depth=round(float(n_sm[2]), 2)),
                bone_cv_raw_pct=round(float(cv), 1),
                weak_persp_residual_cm=dict(mean=round(float(np.mean(res)), 2), p95=round(float(np.percentile(res, 95)), 2)),
                metres_per_px_cv_pct=round(100 * float(np.nanstd(mpp) / np.nanmean(mpp)), 1))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", nargs="*", help="clip names (default: every video in VIDEOS that exists)")
    ap.add_argument("--out", default=RESULTS / "raw_pose", help="folder for <clip>.npz and raw_pose.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, rel in VIDEOS.items():
        if (a.clips and name not in a.clips) or not (ROOT / rel).exists():
            continue
        t0 = time.time()
        crops, fps, hw, side = frames_of(ROOT / rel)
        world, xy, vis = pose.landmarks_from_crops(crops, fps)
        np.savez_compressed(out / f"{name}.npz", world=world, xy=xy, vis=vis, fps=fps, hw=hw, side=side)
        rows[name] = summarize(world, xy, vis, fps, hw, side) | {"seconds": round(time.time() - t0, 1)}
        print(name, json.dumps(rows[name]))
    print("->", dump(rows, out / "raw_pose.json"))


if __name__ == "__main__":
    main()
