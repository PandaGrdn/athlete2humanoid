"""The live demo's GPU pose stage on Modal: SAM 3.1 mask tracking + SAM 3D Body for one uploaded clip.

    MODAL_PROFILE=a2h python -m modal deploy frontends/live_pose_modal.py        # once; rebuilds only when this file changes
    MODAL_PROFILE=a2h python -m modal run frontends/live_pose_modal.py --video clip.mp4 --out outputs/live_pose   # a test call

The hosted server (server/live_pose.py) looks up the deployed function, `a2h-live-pose` / `pose`, and sends it the
already-transcoded clip (H.264 mp4, 30 fps, longest side 640, at most 20 s). `pose` runs the repo's own code paths,
so the result is byte-for-byte what pipeline.py produces on a Colab GPU:

  1. track.athlete_masks   -> sam_track.py in SAM 3.1's env (/content/sam_env, numpy<2), prompt "athlete" with the
                             "person" fallback, the most-seen object's (T, H, W) masks;
  2. frontends.sam3d_body.run -> frontends/sam3d_run.py in SAM 3D Body's env (/content/sam3d_env), the athlete's mask
                             boxes as the person boxes, one MoGe-2 focal length, MHR70 keypoints per frame.

The image is scripts/setup_colab.sh's --with-sam recipe (the sam3 checkout and the two venvs at the /content paths the
scripts expect) plus the main environment (requirements.txt) and the repo modules the stage imports, under /opt/a2h.
Differences from Colab: the venvs' torch comes from PyTorch's cu126 index (an image build has no GPU for
`--torch-backend auto` to detect), and a system ffmpeg decodes the clip (mediapy) instead of imageio-ffmpeg's.

Secret `a2h-hf` (HF_TOKEN; both checkpoints are gated, facebook/sam3.1 and facebook/sam-3d-body-dinov3):
    MODAL_PROFILE=a2h python -m modal secret create a2h-hf HF_TOKEN=...
Volume `a2h-live-pose`, mounted at /vol, holds the Hugging Face cache (HF_HOME=/vol/hf: SAM 3.1, SAM 3D Body, MoGe-2)
so a cold start downloads nothing after the first call; the function commits it after each call.

Cost: one L40S at $1.95/h, billed per second while a call runs plus the 5 min scaledown window that keeps the
container warm for the next upload (scaledown_window=300). Measured on a 6 s / 180-frame clip: 113 s warm
(SAM 3.1 61 s, SAM 3D Body 51 s; each call starts the two envs' subprocesses, which load their models again, as
pipeline.py's do), 147 s in a fresh container with the weights cached, 170 s the very first time (downloads):
about $0.06-0.09 per 6 s clip, roughly $0.15-0.20 for a 20 s one, plus $0.16 for the idle window after the last
call. No min_containers: nothing runs between uploads.
"""

import io
import os
import tempfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent  # frontends/
ROOT = HERE.parent
GPU = "L40S"
SAM3D_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"  # facebookresearch/sam-3d-body, as in sam3d_run.py
# the repo modules the stage imports (track -> config; frontends.sam3d_body -> pose, config; sam_track -> track;
# athlete for --athlete selection), kept at the repo's layout under /opt/a2h
MODULES = ["config.py", "track.py", "athlete.py", "sam_track.py", "pose.py", "frontends/sam3d_body.py",
           "frontends/sam3d_run.py"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "ffmpeg", "libgl1", "libglib2.0-0", "libegl1", "libgles2")
    .pip_install("uv")
    # SAM 3.1's env: scripts/setup_colab.sh --with-sam (sam3 pins numpy<2, so it keeps its own Python 3.12)
    .run_commands(
        "git clone -q https://github.com/facebookresearch/sam3.git /content/sam3",
        "uv venv -q -p 3.12 /content/sam_env",
        'uv pip install -q -p /content/sam_env/bin/python --torch-backend cu126 torch torchvision "setuptools<81" '
        '-e "/content/sam3[notebooks]"',
        "/content/sam_env/bin/python -c 'import sam3, decord, numpy; print(\"sam env OK, numpy\", numpy.__version__)'",
    )
    # SAM 3D Body's env: scripts/setup_sam3d.sh /content
    .run_commands(
        "git clone -q https://github.com/facebookresearch/sam-3d-body.git /content/sam-3d-body && "
        f"git -C /content/sam-3d-body checkout -q {SAM3D_COMMIT}",
        "uv venv -q -p 3.11 /content/sam3d_env",
        "uv pip install -q -p /content/sam3d_env/bin/python --no-sources --torch-backend cu126 torch==2.7.1 torchvision==0.22.1 "
        "pytorch-lightning yacs scikit-image einops timm dill pandas rich hydra-core pyrootutils webdataset networkx==3.2.1 "
        "roma joblib loguru optree fvcore huggingface_hub opencv-python-headless braceexpand omegaconf mediapy termcolor "
        '"git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"',
        "cd /tmp && MOMENTUM_ENABLED=0 PYTHONPATH=/content/sam-3d-body /content/sam3d_env/bin/python -c "
        "'import torch, sam_3d_body, moge; print(\"SAM 3D Body env OK\")'",
    )
    # the main environment (Colab's own Python): requirements.txt, and the repo modules the stage imports
    .add_local_file(ROOT / "requirements.txt", "/opt/a2h/requirements.txt", copy=True)
    .run_commands("uv pip install -q --system -r /opt/a2h/requirements.txt")
    .env({
        "PYTHONPATH": "/opt/a2h",
        "A2H_SAM_PYTHON": "/content/sam_env/bin/python",
        "A2H_SAM3D_PYTHON": "/content/sam3d_env/bin/python",
        "A2H_SAM3D_DIR": "/content/sam-3d-body",
        "A2H_OUTPUT_DIR": "/tmp/a2h_out",
        "A2H_CLIPS_DIR": "/tmp/a2h_clips",
        "A2H_ASSETS_DIR": "/opt/a2h/assets",
        "HF_HOME": "/vol/hf",          # both checkpoints and MoGe-2, persisted on the Volume
        "TORCH_HOME": "/vol/torch",    # sam3d_run's Faster R-CNN fallback (boxes=None), if it is ever taken
        "PYTHONUNBUFFERED": "1",
    })
)
for _m in MODULES:
    image = image.add_local_file(ROOT / _m, f"/opt/a2h/{_m}", copy=True)

app = modal.App("a2h-live-pose", image=image)
vol = modal.Volume.from_name("a2h-live-pose", create_if_missing=True)


@app.function(gpu=GPU, timeout=900, scaledown_window=300, cpu=4, memory=16384,
              secrets=[modal.Secret.from_name("a2h-hf")], volumes={"/vol": vol})
def pose(video: bytes, prompt: str = "athlete") -> dict:
    """video: the transcoded clip (mp4 bytes). -> {"sam3d": npz bytes (frontends/sam3d_body.load's keys), "masks": npz
    bytes (key "masks", (T, H, W) bool), "seen": fraction of frames with the athlete, "frames", "width", "height",
    "timings_s": {"sam", "sam3d", "total"}, "gpu"}."""
    import mediapy
    import numpy as np

    import track
    from frontends import sam3d_body

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/clip.mp4"
        Path(path).write_bytes(video)
        frames = np.asarray(mediapy.read_video(path))  # as server/live_worker.py reads it: T, H, W must match
        T, H, W = frames.shape[:3]
        print(f"clip: {T} frames {W}x{H}, prompt {prompt!r}", flush=True)
        t1 = time.time()
        masks = track.athlete_masks(path, T, prompt=prompt)  # sam_track.py in SAM 3.1's env
        t2 = time.time()
        d = sam3d_body.run(path, T, sam3d_body.mask_boxes(masks) if masks.mean() < 0.9 else None)  # as pipeline.py
        t3 = time.time()
    vol.commit()  # the checkpoints the first call downloaded (a no-op afterwards)
    seen = float(masks.any((1, 2)).mean())
    timings = {"sam": round(t2 - t1, 2), "sam3d": round(t3 - t2, 2), "total": round(time.time() - t0, 2)}
    print(f"done: athlete in {seen:.0%} of frames, {timings}", flush=True)
    return {"sam3d": _npz(**d), "masks": _npz(masks=masks.astype(bool)), "seen": seen, "frames": int(T),
            "width": int(W), "height": int(H), "timings_s": timings, "gpu": GPU}


def _npz(**arrays) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


@app.local_entrypoint()
def main(video: str, out: str = "outputs/live_pose"):
    """One clip through the deployed stage: writes <out>/sam3d.npz and <out>/masks.npz."""
    t0 = time.time()
    r = pose.remote(Path(video).read_bytes())
    wall = time.time() - t0
    os.makedirs(out, exist_ok=True)
    Path(out, "sam3d.npz").write_bytes(r["sam3d"])
    Path(out, "masks.npz").write_bytes(r["masks"])
    print(f"{r['frames']} frames {r['width']}x{r['height']}, athlete seen in {r['seen']:.0%}; "
          f"timings {r['timings_s']} on {r['gpu']}; wall {wall:.1f} s -> {out}/sam3d.npz, {out}/masks.npz")
