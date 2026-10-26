"""Silhouette videos: the same white-on-black body shape for the athlete (from SAM 3.1 masks)
and the robot (from MuJoCo segmentation), framed identically. This is how V-JEPA compares the
motion in a video with the robot's motion without the human-vs-robot appearance gap.
"""

import mujoco
import numpy as np
from scipy.ndimage import uniform_filter1d

SIZE = 256


def normalize(masks, size=SIZE, pad=0.25):
    """(T, H, W) bool masks -> (T, size, size, 3) uint8 white-on-black, one crop for the whole
    clip: square side from the largest body extent, horizontal centre follows the body slowly
    (about a second), vertical centre fixed, so jumps and squats stay visible as motion."""
    import cv2

    T, H, W = masks.shape
    seen = masks.any((1, 2))
    if not seen.any():
        raise RuntimeError("Empty masks.")
    box = np.full((T, 4), np.nan)
    for t in np.flatnonzero(seen):
        ys, xs = np.nonzero(masks[t])
        box[t] = xs.min(), ys.min(), xs.max(), ys.max()
    t_all = np.arange(T)
    for j in range(4):
        box[:, j] = np.interp(t_all, t_all[seen], box[seen, j])
    extent = max(np.max(box[:, 2] - box[:, 0]), np.max(box[:, 3] - box[:, 1]))
    side = int(extent * (1 + 2 * pad))
    cx = uniform_filter1d((box[:, 0] + box[:, 2]) / 2, size=31, mode="nearest")
    cy = np.full(T, np.median((box[:, 1] + box[:, 3]) / 2))

    padded = np.zeros((H + 2 * side, W + 2 * side), np.uint8)
    out = np.empty((T, size, size, 3), np.uint8)
    for t in range(T):
        padded[side:side + H, side:side + W] = masks[t] * 255
        x0, y0 = int(round(cx[t] - side / 2)) + side, int(round(cy[t] - side / 2)) + side
        crop = cv2.resize(padded[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)
        out[t] = crop[..., None]
    return out


def robot_masks(model, qpos, azimuth=135.0, elevation=-5.0, distance=4.0, size=512):
    """(T, size, size) robot masks rendered from a viewpoint matching the video. The camera
    follows the pelvis horizontally only, at a fixed height, so vertical motion is kept."""
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size)
    model.vis.global_.offheight = max(model.vis.global_.offheight, size)
    xy = uniform_filter1d(qpos[:, :2], size=31, axis=0, mode="nearest")
    out = np.empty((len(qpos), size, size), bool)
    with mujoco.Renderer(model, size, size) as r:
        r.enable_segmentation_rendering()
        for t, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            cam.lookat[:] = (xy[t, 0], xy[t, 1], 0.6)
            r.update_scene(data, camera=cam)
            seg = r.render()
            is_floor = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == floor)
            out[t] = (seg[..., 0] >= 0) & ~is_floor
    return out


def robot_silhouettes(model, qpos, azimuth=135.0, **kw):
    return normalize(robot_masks(model, qpos, azimuth=azimuth, **kw))


def athlete_silhouettes(masks):
    return normalize(masks)
