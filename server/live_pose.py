"""The live runner's GPU pose: SAM 3.1 masks and SAM 3D Body on a Modal GPU, for an uploaded video.

frontends/live_pose_modal.py is the deployed app ("a2h-live-pose", function "pose"): it takes the processed clip's
bytes and returns SAM 3D Body's npz and the person's masks. This module is the client side, used by
server/live_worker.py when the server runs with A2H_LIVE_POSE=modal (server/config.py): the worker sends the clip,
waits (a cold start downloads nothing, the weights live on a Volume, but still takes a minute or two to boot), and
writes sam3d.npz and masks.npz into the job's folder, which it then reads like a stored run's. Any failure makes the
worker fall back to MediaPipe on the CPU, with a log line, so a visitor's build never fails because the GPU didn't come.

The Modal token reaches the worker as MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in its environment (server/live.py passes
them through only in this mode); the modal client reads those itself. Nothing here prints them.
"""
import io
import os
from pathlib import Path

import numpy as np

APP, FUNCTION = "a2h-live-pose", "pose"
RESULT_KEYS = ("sam3d", "masks", "frames", "width", "height")
SAM3D_NPZ, MASKS_NPZ = "sam3d.npz", "masks.npz"


def configured():
    """Whether this process has a Modal token to call the app with."""
    return bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))


def fetch(video_bytes, timeout_s, prompt="athlete"):
    """The deployed function's result for the clip, waiting at most timeout_s. Raises on any failure."""
    import modal  # the client only; imported when asked for, so hosts without it still run the CPU path

    f = modal.Function.from_name(APP, FUNCTION)
    call = f.spawn(video_bytes, prompt)
    try:
        return call.get(timeout=timeout_s)
    except BaseException:
        try:
            call.cancel()
        except Exception:  # noqa: BLE001  (best effort: the container stops on its own at the function's timeout)
            pass
        raise


def check(result, n_frames, width, height):
    """The result's problem as a string, or None when it is SAM output for a clip of this shape."""
    if not isinstance(result, dict) or any(k not in result for k in RESULT_KEYS):
        return "not a pose result"
    shape = (result["frames"], result["width"], result["height"])
    if shape != (n_frames, width, height):
        return f"the GPU read {shape[0]} frames at {shape[1]}x{shape[2]}, this worker {n_frames} at {width}x{height}"
    if not isinstance(result["sam3d"], (bytes, bytearray)) or not isinstance(result["masks"], (bytes, bytearray)):
        return "the result's arrays are not npz bytes"
    return None


def save(result, out):
    """Write the result's sam3d.npz and masks.npz into out. -> (sam3d path, masks (T, H, W) bool)."""
    out = Path(out)
    sam3d = out / SAM3D_NPZ
    sam3d.write_bytes(bytes(result["sam3d"]))
    with np.load(io.BytesIO(bytes(result["masks"]))) as z:
        masks = z["masks"].astype(bool)
    np.savez_compressed(out / MASKS_NPZ, masks=masks)
    return sam3d, masks
