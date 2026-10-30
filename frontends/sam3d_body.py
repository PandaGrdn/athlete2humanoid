"""SAM 3D Body output (frontends/sam3d_modal.py's npz) -> the (T, 33, 3) human trajectory retarget.py takes.

SAM 3D Body predicts, per frame, the MHR rig's 70 keypoints in metres (camera axes, root-relative) and the
camera translation that places them. Two ways to get a trajectory from that:

    root="camera": keypoints + predicted camera translation, the model's own metric root, depth included
                   (valid for a static camera; the three A/B clips are);
    root="image":  the keypoints as pose.human_trajectory treats MediaPipe's world landmarks, with the root from
                   the 2D hip track and the torso ruler (depth motion zero): the skeleton alone, MediaPipe's root.

Both apply the same fill + Savitzky-Golay smoothing as the MediaPipe path, so the comparison is like for like.
"""

import numpy as np

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


def human_trajectory(d, root="camera", return_view=False):
    """d: load(npz). -> (T, 33, 3) metres, z-up, facing +x (pose.human_trajectory's frame); with
    return_view also the camera azimuth."""
    if root == "camera":
        return pose.camera_to_world(pose.fill_and_smooth(camera_points(d)), return_view)
    if root != "image":
        raise ValueError(root)
    kp = d["kp3d"][:, MHR70_TO_MP33]
    world = kp - kp[:, [23, 24]].mean(1, keepdims=True)  # hip-centred, like MediaPipe's world landmarks
    px = d["kp2d"][:, MHR70_TO_MP33]
    boxes = np.tile([0.0, 0.0, 1.0], (len(px), 1))  # landmark_pixels(px, (0, 0, 1)) = px
    return pose.human_trajectory(world, px, boxes, return_view=return_view)
