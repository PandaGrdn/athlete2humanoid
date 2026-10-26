"""Video -> latent: the athlete's SAM 3.1 masks become a silhouette video, and V-JEPA 2 turns it
into one embedding per ~1 s window. This latent sequence is the training target.

    python -m latent.video_latent outputs/squat      # a pipeline.py output folder (has masks.npz)
"""

import argparse
import json
from pathlib import Path

import mediapy
import numpy as np

from latent import silhouette
from latent.vjepa import VJepa


def video_latent(out_dir, vj=None):
    out_dir = Path(out_dir)
    masks = np.load(out_dir / "masks.npz")["masks"]
    meta = json.loads((out_dir / "meta.json").read_text())
    sil = silhouette.athlete_silhouettes(masks)
    mediapy.write_video(str(out_dir / "silhouette.mp4"), sil, fps=meta["fps"])
    vj = vj or VJepa()
    emb, starts = vj.embed_windows(sil)
    dense = vj.embed_dense(sil)
    np.savez(out_dir / "video_latent.npz", emb=emb, starts=starts, dense=dense, fps=meta["fps"],
             view_azimuth=meta["view_azimuth"])
    print(f"{out_dir.name}: {len(emb)} windows, {len(dense)} dense steps ({meta['fps'] / 2:.0f}/s) x {emb.shape[1]} dims")
    return emb, starts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    vj = VJepa()
    for d in ap.parse_args().out_dirs:
        video_latent(d, vj)
