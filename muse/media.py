"""Media for Muse: small MP4s (Muse reads MP4; fewer pixels, fewer tokens) and frame contact sheets.

Both depend on optional tools (ffmpeg, OpenCV) and are imported lazily, so the rest of muse/ runs on
a bare Python for tests and dry runs.
"""

import shutil
import subprocess
from pathlib import Path


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # in requirements.txt; ships its own ffmpeg binary
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def prepare_video(src, dst, max_side=480, required=True):
    """src -> dst as a silent H.264 MP4 whose longer side is max_side px. Without ffmpeg
    an MP4 is used as is. required=False (dry runs) is best effort: when the conversion can't be
    done, src is passed through unchanged instead of raising."""
    src, dst = Path(src), Path(dst)
    exe = ffmpeg_exe()
    if exe is None:
        if src.suffix.lower() == ".mp4" or not required:
            return src
        raise RuntimeError(f"Converting {src.name} to MP4 needs ffmpeg (pip install imageio-ffmpeg).")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    scale = f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease:force_divisible_by=2"
    try:
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-vf", scale, "-an", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
    except subprocess.CalledProcessError:
        dst.unlink(missing_ok=True)  # don't let a partial file pass the cache check next time
        if required:
            raise
        print(f"ffmpeg couldn't convert {src.name}; using it as is (dry run)")
        return src
    return dst


def contact_sheet(video, n=6, max_width=1600, max_height=3000):
    """n frames evenly spaced through `video`, stacked top to bottom -> PNG bytes. None when OpenCV
    is missing or the video can't be read."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            rows.append(frame)
    cap.release()
    if not rows:
        return None
    sheet = np.concatenate(rows, 0)
    s = min(1.0, max_width / sheet.shape[1], max_height / sheet.shape[0])
    if s < 1:
        sheet = cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", sheet)[1].tobytes()
