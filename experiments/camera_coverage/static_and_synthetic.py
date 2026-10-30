"""Fix 6 check, part 1 (CPU): on a static clip, the camera stage must leave the root unchanged; on a
synthetic panned and zoomed copy of the same clip (MediaPipe re-run on the warped video), the
compensated root must match the static one about as well as the true camera motion would.

    python experiments/camera_coverage/static_and_synthetic.py <repo> <name> <video> <landmarks.npz>

landmarks.npz comes from landmarks.py. Prints one JSON line.
"""

import json
import sys
import time

import imageio_ffmpeg
import mediapy
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, sys.argv[1])
import camera  # noqa: E402
import pose  # noqa: E402
import track  # noqa: E402

mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


def run_pose(frames, fps, masks):
    crops, boxes = track.crop_to_mask(frames, masks)
    world, image_xy, _ = pose.landmarks_from_crops(crops, fps)
    return world, image_xy, boxes


def root(world, image_xy, boxes, Hs=None):
    P = pose.human_trajectory(world, image_xy, boxes, camera=Hs)
    pel = P[:, [23, 24]].mean(1)
    return pel - pel[0]


def identity_if_none(Hs, T):
    return np.tile(np.eye(3), (T, 1, 1)) if Hs is None else Hs


name, video, cache = sys.argv[2], sys.argv[3], sys.argv[4]
v = mediapy.read_video(video)
fps = v.metadata.fps
frames = np.asarray(v)
T, H, W = frames.shape[:3]
z = np.load(cache)
world, image_xy, boxes = z["world"], z["image_xy"], z["boxes"]
masks = np.unpackbits(z["masks"], axis=-1)[..., :W].astype(bool) if "masks" in z else np.ones((T, H, W), bool)
real_mask = masks.mean() < 0.9
px = pose.landmark_pixels(pose.fill_and_smooth(image_xy), boxes)
res = {"clip": name, "frames": T, "size": [W, H], "height": pose.height_check(px)}

# Static: the clip as filmed.
t0 = time.time()
Hs, report = camera.intake(frames, masks, image_xy, boxes)
Hs = identity_if_none(Hs, T)
info = report["camera"]
res["static"] = {k: info[k] for k in ("static", "max_shift_px", "max_zoom", "failed", "cuts")}
r0 = root(world, image_xy, boxes)
r1 = root(world, image_xy, boxes, Hs)
res["static"]["root_change_cm_max"] = float(np.abs(r1 - r0).max() * 100)
res["static"]["estimate_s"] = round(time.time() - t0, 1)

# Synthetic: a crop of the frame that pans to follow the athlete (plus a sway), tilts and zooms in 10%.
tt = np.arange(T) / (T - 1)
c = min(0.9, max(0.6, res["height"]["height_px"] / H * 1.15 * 1.1))  # the athlete still fits after the zoom
oh, ow = int(H * c), int(W * c)
zooms = 1 + 0.1 * np.sin(np.pi * tt / 2)
m = 0.9 * (W - ow) / 2  # stay inside the frame: no reflected border
xa = uniform_filter1d(px[:, [23, 24], 0].mean(1), 15, mode="nearest")
sx = np.clip(-zooms * (xa - W / 2) + 0.5 * m * np.sin(2 * np.pi * tt), -m, m)
sy = 0.3 * (H - oh) / 2 * np.sin(np.pi * tt)
shifts = np.stack([sx, sy], -1)
syn, Ht = camera.warp_synthetic(frames, shifts, zooms, out_shape=(oh, ow))
if real_mask:
    warped = camera.warp_synthetic(np.repeat(masks[..., None], 3, -1).astype(np.uint8) * 255, shifts, zooms, out_shape=(oh, ow))[0]
    smasks = warped[..., 0] > 127
else:
    smasks = np.ones(syn.shape[:3], bool)
sw, si, sb = run_pose(syn, fps, smasks)
spx = pose.landmark_pixels(pose.fill_and_smooth(si), sb)
He, sreport = camera.intake(syn, smasks, si, sb)
He = identity_if_none(He, T)
sinfo = sreport["camera"]
pts = spx[:, [23, 24]].mean(1)
herr = np.linalg.norm(camera.apply(He, pts) - camera.apply(Ht, pts), axis=-1)
truemove = np.linalg.norm(camera.apply(Ht, pts) - pts, axis=-1)


def cm(r):
    d = np.linalg.norm(r - r0, axis=-1) * 100
    return [round(float(d.mean()), 1), round(float(d.max()), 1)]


res["synthetic"] = {
    "crop": round(c, 2), "pose_found": float((~np.isnan(sw[:, 0, 0])).mean()),
    "camera_motion_px_max": round(float(truemove.max())), "zoom_max": round(float(zooms.max()), 2),
    "camera_err_px_mean_max": [round(float(herr.mean()), 2), round(float(herr.max()), 2)],
    "detected_static": sinfo["static"], "cuts": sinfo["cuts"], "failed": len(sinfo["failed"]),
    "root_vs_static_cm_mean_max": {"uncompensated": cm(root(sw, si, sb)), "compensated": cm(root(sw, si, sb, He)),
                                   "true_camera": cm(root(sw, si, sb, Ht))}}
print(json.dumps(res))
