"""New videos -> processed clip folders, nothing trained: SAM 3.1 -> pose -> retarget, and the V-JEPA
silhouette latent. The archive is what variant_policy.py / multiclip_policy.py read as clips/<name>/.

    python experiments/process_clips.py clips_pro/*.mp4 --out outputs/process_pro

Needs the SAM env (scripts/setup_colab.sh --with-sam), so it runs on Colab (scripts/remote_run.py process).
"""

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.cross_embodiment_probe import _processed  # noqa: E402
from experiments.latent_policy import log  # noqa: E402

KEEP = ["video_latent.npz", "qpos.npy", "meta.json", "motion.csv", "retarget_report.json", "silhouette.mp4",
        "side_by_side.mp4", "human.npy"]  # as latent_policy.pack's clips/<name>/


def main(a):
    from latent.video_latent import video_latent
    from latent.vjepa import VJepa
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vj, done = VJepa(), {}
    for v in a.videos:
        try:
            d = _processed(v)
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
    main(ap.parse_args())
