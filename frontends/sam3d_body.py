"""SAM 3D Body output (frontends/sam3d_modal.py's npz) -> the (T, 33, 3) human trajectory retarget.py takes.

SAM 3D Body predicts, per frame, the MHR rig's 70 keypoints in metres (camera axes, root-relative) and the
camera translation that places them. Two ways to get a trajectory from that:

    root="camera": keypoints + predicted camera translation, the model's own metric root, depth included
                   (valid for a static camera; the three A/B clips are);
    root="image":  the keypoints as pose.human_trajectory treats MediaPipe's world landmarks, with the root from
                   the 2D hip track and the torso ruler (depth motion zero): the skeleton alone, MediaPipe's root.

Both apply the same fill + Savitzky-Golay smoothing as the MediaPipe path, so the comparison is like for like.

pipeline.py uses the second (`landmarks`: the same world / image_xy / boxes MediaPipe gives, so camera compensation
and the intake checks apply unchanged), with SAM 3D Body run by `run` in its own env (frontends/sam3d_run.py).
"""

import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import config
import pose

# MediaPipe's 33 landmarks from MHR70 (sam_3d_body/metadata/mhr70.py). Eyes inner/outer take the eye, the mouth
# corners the nose, hand landmarks the finger bases (MediaPipe's are at the knuckles); only 0, 7, 8, 11-16
# and 23-28 are read downstream (human_trajectory, rescale_bones).
MHR70_TO_MP33 = np.array([
    0,                  # nose
    1, 1, 1, 2, 2, 2,   # left eye inner / eye / outer, right
    3, 4,               # ears
    0, 0,               # mouth corners
    5, 6, 7, 8,         # shoulders, elbows
    62, 41,             # wrists
    61, 40, 49, 28, 45, 24,  # pinky, index, thumb bases (left, right)
    9, 10, 11, 12, 13, 14,   # hips, knees, ankles
    17, 20, 15, 18,     # heels, big toes
])


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def camera_points(d):
    """(T, 33, 3) camera-frame metres: keypoints + the predicted camera translation."""
    return (d["kp3d"] + d["cam_t"][:, None, :])[:, MHR70_TO_MP33]


def available():
    """Whether SAM 3D Body can run here: its env (config.SAM3D_PYTHON), or sam_3d_body importable with a GPU."""
    if config.SAM3D_PYTHON:
        return Path(config.SAM3D_PYTHON).exists()
    if importlib.util.find_spec("sam_3d_body") is None:
        return False
    import torch

    return torch.cuda.is_available()


def choose(frontend="auto"):
    """"mediapipe" | "sam3d" for pipeline.py's --frontend: auto takes SAM 3D Body where it can run."""
    if frontend == "auto":
        return "sam3d" if available() else "mediapipe"
    if frontend == "sam3d" and not available():
        raise RuntimeError("SAM 3D Body is not set up here: run scripts/setup_sam3d.sh and set A2H_SAM3D_PYTHON "
                           "(and A2H_SAM3D_DIR), or use --frontend mediapipe")
    return frontend


def mask_boxes(masks):
    """(T, H, W) athlete masks -> (T, 4) x0 y0 x1 y1 pixels, frames without the athlete interpolated."""
    seen = masks.any((1, 2))
    if not seen.any():
        raise RuntimeError("Mask is empty in every frame.")
    boxes = np.full((len(masks), 4), np.nan)
    for t in np.flatnonzero(seen):
        ys, xs = np.nonzero(masks[t])
        boxes[t] = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    t_all = np.arange(len(masks))
    return np.stack([np.interp(t_all, t_all[seen], boxes[seen, j]) for j in range(4)], 1)


def run(video_path, n_frames, boxes=None):
    """SAM 3D Body on the video (frontends/sam3d_run.py), in its own env when config.SAM3D_PYTHON is set.
    boxes: (T, 4) athlete boxes (mask_boxes), else its person detector picks them. -> the npz as a dict."""
    config.get_secret("HF_TOKEN")  # the checkpoint is gated; exported for the subprocess
    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/sam3d.npz"
        if config.SAM3D_PYTHON:
            extra = []
            if boxes is not None:
                np.save(f"{tmp}/boxes.npy", boxes)
                extra = ["--boxes", f"{tmp}/boxes.npy"]
            subprocess.run([config.SAM3D_PYTHON, str(Path(__file__).with_name("sam3d_run.py")), str(Path(video_path).resolve()),
                            "--out", out, *extra], check=True, cwd=config.ROOT,
                           env=dict(os.environ, A2H_SAM3D_DIR=os.environ.get("A2H_SAM3D_DIR", "/content/sam-3d-body")))
            d = load(out)
        else:
            import mediapy

            from frontends import sam3d_run

            d = sam3d_run.estimate(np.asarray(mediapy.read_video(str(video_path))), boxes)
    if len(d["kp3d"]) != n_frames:
        raise RuntimeError(f"SAM 3D Body read {len(d['kp3d'])} frames, the pipeline {n_frames}")
    return d


def landmarks(d):
    """MediaPipe's outputs from SAM 3D Body's: world (T, 33, 3) hip-centred metres in camera axes, image_xy (T, 33, 2)
    and boxes (T, 3) such that pose.landmark_pixels(image_xy, boxes) is the keypoints' full-frame pixels. These go
    where landmarks_from_crops' and crop_to_mask's do (camera.intake, pose.human_trajectory)."""
    kp = d["kp3d"][:, MHR70_TO_MP33].astype(float)
    world = kp - kp[:, [23, 24]].mean(1, keepdims=True)  # hip-centred, like MediaPipe's world landmarks
    return world, d["kp2d"][:, MHR70_TO_MP33].astype(float), np.tile([0.0, 0.0, 1.0], (len(kp), 1))


def human_trajectory(d, root="camera", return_view=False):
    """d: load(npz). -> (T, 33, 3) metres, z-up, facing +x (pose.human_trajectory's frame); with
    return_view also the camera azimuth."""
    if root == "camera":
        return pose.camera_to_world(pose.fill_and_smooth(camera_points(d)), return_view)
    if root != "image":
        raise ValueError(root)
    return pose.human_trajectory(*landmarks(d), return_view=return_view)
