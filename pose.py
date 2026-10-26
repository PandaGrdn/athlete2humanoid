"""MediaPipe Pose on SAM-cropped frames -> a smoothed, metric, z-up human trajectory.

Uses the MediaPipe Tasks API (PoseLandmarker); recent mediapipe releases dropped the legacy
mp.solutions.pose. SAM 3D Body can replace this stage later with the same output shape.
"""

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from scipy.signal import savgol_filter

import config

# MediaPipe landmark indices (the person's own left/right).
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24


def landmarks_from_crops(crops, fps, model_path=config.POSE_MODEL):
    """crops: (T, S, S, 3) uint8. Returns world (T, 33, 3) metres, hip-centred, camera axes
    (x right, y down, z depth); image_xy (T, 33, 2) normalised to the crop; visibility (T, 33).
    Frames with no detection are NaN."""
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    T = len(crops)
    world = np.full((T, 33, 3), np.nan)
    image_xy = np.full((T, 33, 2), np.nan)
    vis = np.zeros((T, 33))
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        for t, crop in enumerate(crops):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
            res = landmarker.detect_for_video(image, int(t * 1000 / fps))
            if not res.pose_world_landmarks:
                continue
            world[t] = [(p.x, p.y, p.z) for p in res.pose_world_landmarks[0]]
            image_xy[t] = [(p.x, p.y) for p in res.pose_landmarks[0]]
            vis[t] = [p.visibility for p in res.pose_landmarks[0]]
    return world, image_xy, vis


def fill_and_smooth(x, window=9):
    """Interpolate NaN frames, then Savitzky-Golay along time. x: (T, ...)."""
    T = len(x)
    flat = x.reshape(T, -1).copy()
    ok = ~np.isnan(flat[:, 0])
    if ok.sum() < max(2, T // 2):
        raise RuntimeError(f"Pose found in only {ok.sum()}/{T} frames.")
    t = np.arange(T)
    for j in range(flat.shape[1]):
        flat[:, j] = np.interp(t, t[ok], flat[ok, j])
    window = min(window, T - (1 - T % 2))  # odd and <= T
    if window >= 5:
        flat = savgol_filter(flat, window, 3, axis=0)
    return flat.reshape(x.shape)


def human_trajectory(world, image_xy, boxes, return_view=False):
    """Combine hip-centred 3D pose with the hips' 2D image motion -> (T, 33, 3) metres,
    z-up, ground not yet set, athlete facing +x. With return_view, also the video camera's
    viewing direction as a MuJoCo camera azimuth (degrees) in that athlete-facing frame, so the
    robot can be rendered from the same viewpoint.

    MediaPipe world landmarks have no global translation (the hips are always at the origin),
    so a jump would vanish. We recover root motion in the image plane from the 2D hip track,
    converted to metres with the torso length as a ruler. Depth motion is assumed zero, so
    side-view clips work best.
    """
    world = fill_and_smooth(world)
    image_xy = fill_and_smooth(image_xy)
    x0, y0, side = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3]
    px = np.stack([x0 + image_xy[..., 0] * side, y0 + image_xy[..., 1] * side], -1)  # full-frame pixels

    hip_px = px[:, [L_HIP, R_HIP]].mean(1)
    torso_px = np.linalg.norm(px[:, [L_SHOULDER, R_SHOULDER]].mean(1) - hip_px, axis=-1)
    hip_w = world[:, [L_HIP, R_HIP]].mean(1)
    torso_m = np.linalg.norm(world[:, [L_SHOULDER, R_SHOULDER]].mean(1) - hip_w, axis=-1)
    metres_per_px = np.median(torso_m / torso_px)
    root = np.zeros((len(world), 3))
    root[:, :2] = (hip_px - hip_px[0]) * metres_per_px  # camera x right, y down
    cam = world + root[:, None, :]

    # Camera axes (x right, y down, z depth) -> z-up world.
    P = np.stack([cam[..., 0], cam[..., 2], -cam[..., 1]], -1)

    # Chirality check: the nose should be in front of the ears in the facing direction
    # derived from the hips. If not, the depth axis is mirrored for this clip; flip it.
    view = np.array([0.0, 1.0, 0.0])  # the camera looks along +depth
    if np.median(_facing_dot(P)) < 0:
        P[..., 1] *= -1
        view[1] = -1.0

    # Rotate about z so the athlete faces +x (the G1's forward).
    fwd = np.median(_forward(P), axis=0)
    yaw = np.arctan2(fwd[1], fwd[0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if return_view:
        v = rot @ view
        return P @ rot.T, float(np.degrees(np.arctan2(v[1], v[0])))
    return P @ rot.T


def _forward(P):
    left = P[:, L_HIP] - P[:, R_HIP]
    left[:, 2] = 0
    fwd = np.cross(left, [0, 0, 1])
    return fwd / np.linalg.norm(fwd, axis=-1, keepdims=True)


def _facing_dot(P):
    head = P[:, NOSE] - P[:, [L_EAR, R_EAR]].mean(1)
    return np.einsum("ij,ij->i", head, _forward(P))
