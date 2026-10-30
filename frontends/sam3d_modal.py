"""SAM 3D Body on a Modal GPU: video -> per-frame MHR70 keypoints + camera translation (frontends/sam3d_body.py
turns that into the pipeline's (T, 33, 3) human trajectory). Fix 3 of docs/PIPELINE_ACCURACY.md.

    MODAL_PROFILE=a2h python -m modal run --detach frontends/sam3d_modal.py --videos clips/squat_jump.mov,outputs/frontend_ab/hurdles30.mp4
    MODAL_PROFILE=a2h python -m modal volume get a2h-frontend-ab sam3d outputs/frontend_ab/
        # -> outputs/frontend_ab/sam3d/<stem>.npz

`modal run` is ephemeral: the app stops when the call returns. The checkpoint (facebook/sam-3d-body-dinov3, gated:
accept its licence on Hugging Face first) downloads inside the container with the HF_TOKEN from .env, passed as an
in-memory secret. The frames are decoded with mediapy as pipeline.py does (feed the 30 fps / 720p resample for films,
as experiments/process_clips.py does, so frame t matches the stored human.npy); frontends/sam3d_run.py does the rest
(boxes from a person detector, one MoGe-2 focal length, SAM 3D Body per frame).
"""

import io
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SAM3D_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"  # facebookresearch/sam-3d-body, as in sam3d_run.py

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    # one resolve: MoGe pulls triton>=3.2 (FlexGEMM), which torch 2.7 satisfies
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "pytorch-lightning", "yacs", "scikit-image", "einops", "timm", "dill", "pandas", "rich", "hydra-core",
                 "pyrootutils", "webdataset", "networkx==3.2.1", "roma", "joblib", "loguru", "optree", "fvcore",
                 "huggingface_hub", "opencv-python-headless", "braceexpand", "omegaconf", "mediapy", "termcolor",
                 "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a")
    .run_commands(f"git clone -q https://github.com/facebookresearch/sam-3d-body.git /opt/sam3d && "
                  f"git -C /opt/sam3d checkout -q {SAM3D_COMMIT}")
    .env({"PYTHONPATH": "/opt/a2h", "A2H_SAM3D_DIR": "/opt/sam3d"})
    .add_local_file(HERE / "sam3d_run.py", "/opt/a2h/sam3d_run.py")  # the inference itself, shared with pipeline.py
)
app = modal.App("a2h-frontend-ab", image=image)
vol = modal.Volume.from_name("a2h-frontend-ab", create_if_missing=True)  # results survive a dropped client


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

    import mediapy
    import numpy as np
    import sam3d_run

    os.environ["HF_TOKEN"] = hf_token
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix) as f:
        f.write(video)
        f.flush()
        frames = np.asarray(mediapy.read_video(f.name))
    print(f"{name}: {len(frames)} frames {frames.shape[2]}x{frames.shape[1]}", flush=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **sam3d_run.estimate(frames, log=lambda m: print(m, flush=True)))
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
