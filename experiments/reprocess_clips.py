"""Re-run the stored clips (experiments/pipeline_audit CLIPS) through the current pipeline.py with SAM 3D Body's
pose, in place, without a GPU: each clip's sam3d.npz (frontends/sam3d_modal.py, on the same frames) is reused, and
everything after pose (camera intake, human trajectory, retarget, motion.csv, the side-by-side) is recomputed.

    python experiments/reprocess_clips.py --videos-root ~/athlete2humanoid      # clips/*.mov, clips_pro/*.mp4 there
        [--sam3d outputs/frontend_ab/sam3d] [--clips squat lunge]

Replaced files go to outputs/_backup_pre_sam3d/<clip>/ first. silhouette.mp4 and video_latent.npz stay: they come
from the SAM 3.1 masks and the video, not the pose. Films are resampled to 30 fps / 720p first, as
experiments/process_clips.py does, so the frames match the stored clip's.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline  # noqa: E402
from experiments.pipeline_audit import ROOT, VIDEOS, clip_dirs  # noqa: E402

REPLACED = ["human.npy", "qpos.npy", "motion.csv", "meta.json", "retarget_report.json", "side_by_side.mp4",
            "camera.json", "sam3d.npz"]


def video_30(src, tmp):
    """The phone clips go in as they are (1/30 s frames); films are resampled like process_clips.py's _processed."""
    if src.suffix.lower() == ".mov":
        return src
    import imageio_ffmpeg

    dst = Path(tmp) / f"{src.stem}30.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", "fps=30,scale=-2:720", "-an", str(dst)], check=True)
    return dst


def reprocess(name, d, video, sam3d, backup):
    old = json.loads((d / "meta.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        out.mkdir()
        shutil.copyfile(sam3d, out / "sam3d.npz")
        res = pipeline.video_to_motion(video_30(video, tmp), out, use_sam=False, frontend="sam3d")
        if res["frontend"] != "sam3d" or res["frames"] != old["frames"]:
            raise RuntimeError(f"{name}: {res['frontend']}, {res['frames']} frames (stored {old['frames']})")
        backup.mkdir(parents=True, exist_ok=True)
        for f in REPLACED:
            if (d / f).exists() and not (backup / f).exists():
                shutil.copy2(d / f, backup / f)
        meta = json.loads((out / "meta.json").read_text())
        # sam: whether SAM 3.1 masked the athlete for the stored silhouette / latent (kept); this pose run used none
        meta |= {k: v for k, v in old.items() if k not in meta or k == "sam"}
        meta["reprocessed"] = {"date": time.strftime("%Y-%m-%d"), "frontend": "sam3d",
                               "pose_boxes": "person detector (frontends/sam3d_run.py); no SAM 3.1 masks",
                               "previous": str(backup.relative_to(ROOT))}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        for f in REPLACED:
            if (out / f).exists():
                shutil.copyfile(out / f, d / f)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--videos-root", default=str(ROOT), help="folder with clips/ and clips_pro/ (VIDEOS)")
    ap.add_argument("--sam3d", default=str(ROOT / "outputs" / "frontend_ab" / "sam3d"), help="<clip>.npz per clip")
    ap.add_argument("--clips", nargs="*")
    a = ap.parse_args(argv)
    for name, d in clip_dirs(a.clips).items():
        sam3d = Path(a.sam3d) / f"{name}.npz"
        if not sam3d.exists():
            print(f"{name}: no {sam3d}, skipped")
            continue
        meta = reprocess(name, d, Path(a.videos_root) / VIDEOS[name], sam3d, ROOT / "outputs" / "_backup_pre_sam3d" / name)
        print(f"{name}: {meta['frames']} frames, stance {meta['stance_frac']}, warnings {meta['retarget_warnings']}")


if __name__ == "__main__":
    main()
