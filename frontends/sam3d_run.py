"""SAM 3D Body on a video's frames -> per-frame MHR70 keypoints + camera translation, as an npz that
frontends/sam3d_body.py reads. Runs in SAM 3D Body's own environment (Python 3.11, torch 2.7; setup:
scripts/setup_colab.sh --with-sam3d), which pipeline.py calls as a subprocess, like sam_track.py for SAM 3.1:

    $A2H_SAM3D_PYTHON frontends/sam3d_run.py clip.mp4 --out sam3d.npz [--boxes boxes.npy]

The code is facebookresearch/sam-3d-body at SAM3D_COMMIT, on PYTHONPATH (A2H_SAM3D_DIR, default
/content/sam-3d-body). The checkpoint (facebook/sam-3d-body-dinov3) is gated on Hugging Face: accept its licence,
and have HF_TOKEN in the environment. Per clip:

1. the athlete's box per frame: given (--boxes, (T, 4) x0 y0 x1 y1 pixels: pipeline.py passes SAM 3.1's mask
   boxes), else torchvision's COCO Faster R-CNN, following the person that overlaps the previous box most;
2. one focal length for the clip: MoGe-2's, median over 8 frames (a moving camera's zoom is not followed);
3. SAM 3D Body (body decoder, no hand refinement) on each frame with that box and focal.

Fix 3 of docs/PIPELINE_ACCURACY.md measured it against MediaPipe.
"""

import argparse
import os
import sys
import time

import numpy as np

SAM3D_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"  # facebookresearch/sam-3d-body
REPO = "facebook/sam-3d-body-dinov3"
KEYS = {"kp3d": "pred_keypoints_3d", "kp2d": "pred_keypoints_2d", "cam_t": "pred_cam_t", "joints": "pred_joint_coords",
        "global_rot": "global_rot", "body_pose": "body_pose_params", "scale": "scale_params", "shape": "shape_params"}


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def person_boxes(frames, log=print):
    """(T, 4) boxes and (T,) scores (0 = not found, previous box kept) from torchvision's COCO Faster R-CNN."""
    import torch
    import torchvision

    T, H, W = frames.shape[:3]
    det = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").cuda().eval()
    boxes, scores, prev = np.zeros((T, 4)), np.zeros(T), None
    with torch.no_grad():
        for t in range(T):
            o = det([torch.from_numpy(frames[t]).permute(2, 0, 1).float().cuda() / 255])[0]
            keep = (o["labels"] == 1) & (o["scores"] > 0.3)
            b, s = o["boxes"][keep].cpu().numpy(), o["scores"][keep].cpu().numpy()
            if len(b) == 0:
                boxes[t] = prev if prev is not None else [0, 0, W, H]
                continue
            i = int(np.argmax(s)) if prev is None else int(np.argmax([_iou(prev, bb) + 1e-3 * ss for bb, ss in zip(b, s)]))
            boxes[t], scores[t], prev = b[i], s[i], b[i]
    log(f"boxes: person found in {(scores > 0).mean():.0%} of frames")
    return boxes, scores


def estimate(frames, boxes=None, log=print):
    """frames (T, H, W, 3) uint8 RGB -> {"kp3d" (T, 70, 3) metres camera axes root-relative, "kp2d" (T, 70, 2)
    pixels, "cam_t" (T, 3), ..., "boxes", "box_scores", "focal", "image_wh"}."""
    import torch

    sam3d_dir = os.environ.get("A2H_SAM3D_DIR", "/content/sam-3d-body")
    if sam3d_dir not in sys.path:
        sys.path.insert(0, sam3d_dir)
    os.environ.setdefault("MOMENTUM_ENABLED", "0")  # set = the TorchScript MHR in the checkpoint, no pymomentum
    t0 = time.time()
    T, H, W = frames.shape[:3]
    if boxes is None:
        boxes, scores = person_boxes(frames, log)
    else:
        boxes, scores = np.asarray(boxes, float), np.ones(T)

    from tools.build_fov_estimator import FOVEstimator

    fov = FOVEstimator(name="moge2", device="cuda")
    focals = [float(fov.get_cam_intrinsics(frames[t])[0, 1, 1]) for t in np.linspace(0, T - 1, 8).astype(int)]
    focal = float(np.median(focals))
    del fov
    cam_int = torch.tensor([[[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]]], dtype=torch.float32)
    log(f"focal {focal:.0f} px (8 frames: {min(focals):.0f}-{max(focals):.0f})")

    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf

    model, cfg = load_sam_3d_body_hf(REPO)
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    out = {k: [] for k in KEYS}
    for t in range(T):
        r = est.process_one_image(frames[t], bboxes=boxes[t][None], cam_int=cam_int, inference_type="body")[0]
        for k, src in KEYS.items():
            out[k].append(np.asarray(r[src], np.float32))
        if t % 50 == 0:
            log(f"  SAM 3D Body frame {t}/{T} ({time.time() - t0:.0f} s)")
    return {**{k: np.stack(v) for k, v in out.items()}, "boxes": boxes, "box_scores": scores, "focal": focal,
            "focals": np.array(focals), "image_wh": np.array([W, H]), "repo": REPO, "commit": SAM3D_COMMIT}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes", help="(T, 4) x0 y0 x1 y1 per frame (.npy); default: a person detector")
    a = ap.parse_args()
    import mediapy

    frames = np.asarray(mediapy.read_video(a.video))
    np.savez_compressed(a.out, **estimate(frames, np.load(a.boxes) if a.boxes else None))
    print(f"SAM 3D Body: {len(frames)} frames -> {a.out}")
