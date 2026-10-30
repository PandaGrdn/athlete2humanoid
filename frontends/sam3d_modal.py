"""SAM 3D Body on a Modal GPU: video -> per-frame MHR70 keypoints + camera translation (frontends/sam3d_body.py
turns that into the pipeline's (T, 33, 3) human trajectory). Fix 3 of docs/PIPELINE_ACCURACY.md.

    MODAL_PROFILE=a2h python -m modal run --detach frontends/sam3d_modal.py --videos clips/squat_jump.mov,outputs/frontend_ab/hurdles30.mp4
    MODAL_PROFILE=a2h python -m modal volume get a2h-frontend-ab sam3d outputs/frontend_ab/
        # -> outputs/frontend_ab/sam3d/<stem>.npz

`modal run` is ephemeral: the app stops when the call returns. The checkpoint (facebook/sam-3d-body-dinov3, gated:
accept its licence on Hugging Face first) downloads inside the container with the HF_TOKEN from .env, passed as an
in-memory secret. Per clip:

1. the frames, decoded with mediapy as pipeline.py does (feed the 30 fps / 720p resample for films, as
   experiments/process_clips.py does, so frame t matches the stored human.npy);
2. the athlete's box per frame: torchvision's COCO Faster R-CNN, the person box that overlaps the previous one most
   (the first frame takes the highest-scoring person). All three A/B clips have one person;
3. one focal length for the clip (static camera): MoGe-2's, median over 8 frames;
4. SAM 3D Body (body decoder, no hand refinement) on each frame with that box and focal.
"""

import io
import os
from pathlib import Path

import modal

SAM3D_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"  # facebookresearch/sam-3d-body
REPO = "facebook/sam-3d-body-dinov3"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    # one resolve: MoGe pulls triton>=3.2 (FlexGEMM), which torch 2.7 satisfies
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "pytorch-lightning", "yacs", "scikit-image", "einops", "timm", "dill", "pandas", "rich", "hydra-core",
                 "pyrootutils", "webdataset", "networkx==3.2.1", "roma", "joblib", "loguru", "optree", "fvcore",
                 "huggingface_hub", "opencv-python-headless", "braceexpand", "omegaconf", "mediapy", "termcolor",
                 "git+https://github.com/microsoft/MoGe.git")
    .run_commands(f"git clone -q https://github.com/facebookresearch/sam-3d-body.git /opt/sam3d && "
                  f"git -C /opt/sam3d checkout -q {SAM3D_COMMIT}")
    .env({"PYTHONPATH": "/opt/sam3d", "MOMENTUM_ENABLED": "0"})  # set = use the TorchScript MHR in the checkpoint
)
app = modal.App("a2h-frontend-ab", image=image)
vol = modal.Volume.from_name("a2h-frontend-ab", create_if_missing=True)  # results survive a dropped client


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


@app.function(gpu="L40S", timeout=3600, cpu=8, memory=32768, volumes={"/vol": vol})
def run_all(videos: dict, hf_token: str):
    """{file name: bytes} -> /vol/sam3d/<stem>.npz, one clip after another on one GPU."""
    os.makedirs("/vol/sam3d", exist_ok=True)
    for name, video in videos.items():
        Path(f"/vol/sam3d/{_stem(name)}.npz").write_bytes(run_clip(name, video, hf_token))
        vol.commit()


def _stem(name):
    return Path(name).stem.removesuffix("30")


def run_clip(name: str, video: bytes, hf_token: str) -> bytes:
    import tempfile
    import time

    import mediapy
    import numpy as np
    import torch
    import torchvision

    os.environ["HF_TOKEN"] = hf_token
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix) as f:
        f.write(video)
        f.flush()
        frames = np.asarray(mediapy.read_video(f.name))
    T, H, W = frames.shape[:3]
    print(f"{name}: {T} frames {W}x{H}", flush=True)

    # 2. the athlete's box
    det = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").cuda().eval()
    boxes, scores, prev = np.zeros((T, 4)), np.zeros(T), None
    with torch.no_grad():
        for t in range(T):
            x = torch.from_numpy(frames[t]).permute(2, 0, 1).float().cuda() / 255
            o = det([x])[0]
            keep = (o["labels"] == 1) & (o["scores"] > 0.3)
            b, s = o["boxes"][keep].cpu().numpy(), o["scores"][keep].cpu().numpy()
            if len(b) == 0:
                boxes[t], scores[t] = (prev if prev is not None else [0, 0, W, H]), 0.0
                continue
            i = int(np.argmax(s)) if prev is None else int(np.argmax([_iou(prev, bb) + 1e-3 * ss for bb, ss in zip(b, s)]))
            boxes[t], scores[t], prev = b[i], s[i], b[i]
    del det
    print(f"boxes: person found in {(scores > 0).mean():.0%} of frames ({time.time() - t0:.0f} s)", flush=True)

    # 3. one focal length for the clip
    from tools.build_fov_estimator import FOVEstimator

    fov = FOVEstimator(name="moge2", device="cuda")
    focals = [float(fov.get_cam_intrinsics(frames[t])[0, 1, 1]) for t in np.linspace(0, T - 1, 8).astype(int)]
    focal = float(np.median(focals))
    del fov
    cam_int = torch.tensor([[[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]]], dtype=torch.float32)
    print(f"focal {focal:.0f} px (8 frames: {min(focals):.0f}-{max(focals):.0f}; default would be {np.hypot(W, H):.0f})", flush=True)

    # 4. SAM 3D Body per frame
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf

    model, cfg = load_sam_3d_body_hf(REPO)
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    keys = {"kp3d": "pred_keypoints_3d", "kp2d": "pred_keypoints_2d", "cam_t": "pred_cam_t", "joints": "pred_joint_coords",
            "global_rot": "global_rot", "body_pose": "body_pose_params", "scale": "scale_params", "shape": "shape_params"}
    out = {k: [] for k in keys}
    for t in range(T):
        r = est.process_one_image(frames[t], bboxes=boxes[t][None], cam_int=cam_int, inference_type="body")[0]
        for k, src in keys.items():
            out[k].append(np.asarray(r[src], np.float32))
        if t % 20 == 0:
            print(f"  frame {t}/{T} ({time.time() - t0:.0f} s)", flush=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.stack(v) for k, v in out.items()}, boxes=boxes, box_scores=scores,
                        focal=focal, focals=np.array(focals), image_wh=np.array([W, H]), repo=REPO, commit=SAM3D_COMMIT)
    print(f"{name}: done in {time.time() - t0:.0f} s", flush=True)
    return buf.getvalue()


@app.local_entrypoint()
def main(videos: str):
    import re

    env = (Path(__file__).resolve().parents[1] / ".env")
    if not env.exists():
        env = Path(__file__).resolve().parents[4] / ".env"  # a worktree under .claude/worktrees/
    token = re.search(r"HF_TOKEN\s*=\s*['\"]?([^'\"\s]+)", env.read_text()).group(1)
    paths = [Path(v) for v in videos.split(",")]
    run_all.remote({p.name: p.read_bytes() for p in paths}, token)
    # then: modal volume get a2h-frontend-ab sam3d outputs/frontend_ab/ (run with --detach, it finishes
    # even if this client drops)
