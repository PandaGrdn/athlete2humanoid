"""New videos -> processed clip folders, nothing trained: SAM 3.1 -> pose -> retarget, and the V-JEPA
silhouette latent. The archive is what variant_policy.py / multiclip_policy.py read as clips/<name>/.

    python experiments/process_clips.py clips_pro/*.mp4 --out outputs/process_pro
    python experiments/process_clips.py clips_pro/*.mp4 --out outputs/process_pro --in-place sprint,hurdles
        # those clips (by file stem) run in place (pipeline.py --in-place)

Needs the SAM env (scripts/setup_colab.sh --with-sam), so it runs on Colab (scripts/remote_run.py process).
"""

import argparse
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.cross_embodiment_probe import _processed  # noqa: E402
from experiments.latent_policy import log  # noqa: E402

KEEP = ["video_latent.npz", "qpos.npy", "meta.json", "motion.csv", "retarget_report.json", "silhouette.mp4",
        "side_by_side.mp4", "human.npy"]  # as latent_policy.pack's clips/<name>/


def record_source(video, d):
    """Add the source video's own frame timing to d/meta.json (_processed resamples it to 30 fps
    before the pipeline, whose "fps" is then the resampled file's)."""
    import pipeline
    path = Path(d) / "meta.json"
    meta = json.loads(path.read_text())
    meta["source_video"] = {"name": Path(video).name, **pipeline.video_fps(video)}
    path.write_text(json.dumps(meta, indent=1))


def main(a):
    from latent.video_latent import video_latent
    from latent.vjepa import VJepa
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import retarget
    vj, done = VJepa(), {}
    in_place = {s.strip().lower() for s in a.in_place.split(",") if s.strip()}
    for v in a.videos:
        try:
            d = _processed(v, retarget.IN_PLACE_DEFAULT if Path(v).stem.lower() in in_place else None)
            if Path(v).suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
                record_source(v, d)
            video_latent(d, vj)
            done[d.name] = d
            log(f"{d.name}: processed")
        except Exception as e:  # one bad video shouldn't lose the others
            log(f"{Path(v).name}: failed: {type(e).__name__}: {e}")
    with tarfile.open(out / "results.tar.gz", "w:gz") as tar:
        for name, d in done.items():
            for f in KEEP:
                if (d / f).exists():
                    tar.add(d / f, f"clips/{name}/{f}")
    log(f"{len(done)}/{len(a.videos)} clips processed -> {out / 'results.tar.gz'}")
    if not done:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--in-place", default="", help="comma-separated video stems to run in place (pipeline.py --in-place)")
    main(ap.parse_args())
