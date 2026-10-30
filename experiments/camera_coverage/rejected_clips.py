"""Fix 6 check, part 2 (CPU): a rejected pro clip through the CPU front end (athlete tracker ->
MediaPipe -> camera intake -> human_trajectory with and without compensation). Prints JSON and writes
<out>.png (pelvis x, z and top view), <out>_track.jpg (the tracked box on 24 frames) and <out>.npz.

    python experiments/camera_coverage/rejected_clips.py <repo> <name> <video> <first> <end> <athlete> <fps|0> <out>
"""

import json
import sys
import time

import cv2
import imageio_ffmpeg
import matplotlib
import mediapy
import numpy as np

sys.path.insert(0, sys.argv[1])
import athlete  # noqa: E402
import camera  # noqa: E402
import pose  # noqa: E402
import track  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
name, src, out, sel = sys.argv[2], sys.argv[3], sys.argv[8], sys.argv[6]
t0, t1, fps_override = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[7])
v = mediapy.read_video(src)
fps = fps_override or v.metadata.fps  # some phone files report 600 fps
frames = np.asarray(v)[t0:t1]
T = len(frames)
tic = time.time()
masks, tinfo = athlete.track_people(frames, None if sel == "largest" else sel)
crops, boxes = track.crop_to_mask(frames, masks)
world, image_xy, _ = pose.landmarks_from_crops(crops, fps)
Hs, rep = camera.intake(frames, masks, image_xy, boxes)
res = {"clip": name, "frames": [t0, t1], "fps": round(fps, 2), "tracker": tinfo, "continuity": athlete.continuity(masks),
       "pose_found": round(float((~np.isnan(world[:, 0, 0])).mean()), 3), "height": rep["height"],
       "camera": {k: rep["camera"][k] for k in ("static", "max_shift_px", "max_zoom", "cuts", "compensated")},
       "camera_failed_frames": len(rep["camera"]["failed"]), "keyframes": len(rep["camera"]["keyframes"]),
       "teleports": rep["teleports"], "warnings": rep["warnings"]}


def stats(P):
    pel = P[:, [23, 24]].mean(1)
    pel = pel - pel[0]
    sp = np.linalg.norm(np.diff(pel[:, :2], axis=0), axis=-1) * fps
    return pel, {"travel_xy_m": round(float(np.linalg.norm(pel[-1, :2])), 2), "path_len_m": round(float(np.sum(sp) / fps), 2),
                 "speed_p95_ms": round(float(np.percentile(sp, 95)), 1), "speed_max_ms": round(float(sp.max()), 1),
                 "z_range_m": round(float(np.ptp(pel[:, 2])), 2)}


P0 = pose.human_trajectory(world, image_xy, boxes)
P1 = pose.human_trajectory(world, image_xy, boxes, camera=Hs) if Hs is not None else P0
a, res["root_uncompensated"] = stats(P0)
b, res["root_compensated"] = stats(P1)
res["seconds"] = round(time.time() - tic)
np.savez_compressed(out + ".npz", uncomp=a, comp=b, Hs=Hs if Hs is not None else np.tile(np.eye(3), (T, 1, 1)))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
tt = np.arange(T) / fps
for lab, p in (("uncompensated", a), ("compensated", b)):
    ax[0].plot(tt, p[:, 0], label=lab)
    ax[1].plot(tt, p[:, 2], label=lab)
    ax[2].plot(p[:, 0], p[:, 1], label=lab)
ax[0].set_title(f"{name}: pelvis x (m)")
ax[1].set_title("pelvis z (m)")
ax[2].set_title("pelvis path, top view (m)")
ax[2].axis("equal")
for x in ax[:2]:
    x.set_xlabel("s")
ax[0].legend()
fig.tight_layout()
fig.savefig(out + ".png", dpi=80)

px = pose.landmark_pixels(pose.fill_and_smooth(image_xy), boxes)
tiles = []
for i in np.linspace(0, T - 1, 24).astype(int):
    f = frames[i].copy()
    ys, xs = np.nonzero(masks[i])
    if len(xs):
        cv2.rectangle(f, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (255, 0, 0), 3)
    for q in px[i, [11, 12, 23, 24, 25, 26, 27, 28]]:
        cv2.circle(f, tuple(int(c) for c in q), 6, (0, 255, 0), -1)
    f = cv2.resize(f, (400, int(400 * f.shape[0] / f.shape[1])))
    cv2.putText(f, str(i), (5, 25), 0, 0.8, (255, 255, 0), 2)
    tiles.append(f)
sheet = np.concatenate([np.concatenate(tiles[r * 6:(r + 1) * 6], 1) for r in range(4)], 0)
cv2.imwrite(out + "_track.jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
print(json.dumps(res))
