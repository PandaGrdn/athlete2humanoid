"""Make-or-break test for video -> latent -> policy: is a *human* video's V-JEPA latent closest to
the *robot* doing the same action?

For each processed human clip (a pipeline.py output folder named after its action, e.g.
outputs/squat), compare its silhouette latent with robot silhouettes of every motion in the bank,
rendered from the same viewpoint as the video, plus the robot replaying that clip's own retarget
(a positive control). Each domain is centred on its own mean so shared appearance cancels out.

    python experiments/cross_embodiment_probe.py outputs/squat outputs/lunge outputs/jacks ...
    python experiments/cross_embodiment_probe.py clips/squat.mp4 clips/lunge.mp4 ...  # runs the
        # full SAM 3.1 -> pose -> retarget pipeline on each video first
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render  # noqa: E402
from latent import bank as motion_bank  # noqa: E402
from latent import silhouette  # noqa: E402
from latent.video_latent import video_latent  # noqa: E402
from latent.vjepa import VJepa  # noqa: E402

# Which bank motions count as "the same action" for each clip name.
CATEGORY = {"squat_jump": ("vertical_jump", "squat_"), "squat": "squat_", "lunge": "lunge_",
            "front_kick": "kick_", "jumping_jacks": "jumping_jacks", "punches": "punch", "jacks": "jumping_jacks", "kick": "kick_",
            "arms": "arms_raise_", "jump": "vertical_jump", "march": "march", "punch": "punch"}


def _processed(path, in_place=None):
    """A pipeline output folder; videos are run through pipeline.py into outputs/<name> first (in_place: its
    --in-place method, or None)."""
    path = Path(path)
    if path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
        import subprocess
        import config
        import imageio_ffmpeg
        import pipeline
        out = config.OUTPUT_DIR / path.stem.lower()
        if not (out / "masks.npz").exists():
            config.get_secret("HF_TOKEN")  # SAM 3.1's subprocess downloads gated weights
            # 30 fps like the robot renders (phones often record 60), 720p for SAM and MediaPipe.
            out.mkdir(parents=True, exist_ok=True)
            clip = out / "clip30.mp4"
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
                            "-vf", "fps=30,scale=-2:720", "-an", str(clip)], check=True)
            pipeline.video_to_motion(clip, out, use_sam=True, in_place=in_place)
        return out
    return path


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


AZ_BUCKET = 15  # degrees: robot renders are cached per viewpoint bucket


def cached_bank(vj, g1, az):
    """Bank motions -> dense robot latents for one viewpoint bucket, cached on disk. Rendering
    26 motions dominates the probe's run time, and clips filmed from similar angles share it."""
    import config
    bucket = int(round(az / AZ_BUCKET) * AZ_BUCKET)
    cache = config.OUTPUT_DIR / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"bank_dense_az{bucket:+04d}_sil{silhouette.VERSION}.npz"
    if f.exists():
        d = np.load(f)
        return {k: d[k] for k in d.files}, bucket
    qf = cache / "bank_qpos.npz"
    if qf.exists():
        bank = dict(np.load(qf))
    else:
        bank = motion_bank.motion_bank()
        np.savez(qf, **bank)
    t0 = time.time()
    dense = {k: vj.embed_dense(silhouette.robot_silhouettes(g1, q, azimuth=bucket)) for k, q in bank.items()}
    np.savez(f, **dense)
    print(f"rendered + embedded {len(dense)} bank motions at {bucket} deg in {time.time() - t0:.0f} s", flush=True)
    return dense, bucket


def chamfer(H, R):
    """Order-free match: every human moment finds its best robot moment and vice versa.
    Robust to rep counts and tempo."""
    S = H @ R.T
    return 0.5 * (S.max(1).mean() + S.max(0).mean())


def dtw(H, R):
    """Order-aware match: dynamic time warping of the human sequence against the robot sequence
    (repeated to cover the human clip, so 1 robot rep can align with several human reps)."""
    R = np.concatenate([R] * int(np.ceil(len(H) / len(R))))
    cost = 1 - H @ R.T
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    L = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = min(((D[i - 1, j - 1], L[i - 1, j - 1]), (D[i - 1, j], L[i - 1, j]), (D[i, j - 1], L[i, j - 1])))
            D[i, j], L[i, j] = cost[i - 1, j - 1] + prev[0], prev[1] + 1
    end = int(np.argmin(D[n, 1:] / L[n, 1:])) + 1  # open end: the robot sequence may stop early
    return 1 - D[n, end] / L[n, end]


SCORES = {"clip_mean": lambda H, R: float(unit(H.mean(0)) @ unit(R.mean(0))), "chamfer": chamfer, "dtw": dtw}


def main(dirs, out):
    dirs = [_processed(d) for d in dirs]
    vj = VJepa()
    g1 = render.load_g1()[0]

    human = {}
    for d in map(Path, dirs):
        video_latent(d, vj)
        lat = np.load(d / "video_latent.npz")
        human[d.name] = {"dense": lat["dense"], "az": float(lat["view_azimuth"]), "own": np.load(d / "qpos.npy")}
    # Centre each domain on its own mean: shared "human" / "robot silhouette" appearance cancels.
    h_mean = np.concatenate([h["dense"] for h in human.values()]).mean(0)

    results, hits = {}, {k: [] for k in SCORES}
    for name, h in human.items():
        bank, bucket = cached_bank(vj, g1, h["az"])
        r_mean = np.concatenate(list(bank.values())).mean(0)
        own = vj.embed_dense(silhouette.robot_silhouettes(g1, h["own"], azimuth=bucket))
        H = unit(h["dense"] - h_mean)
        robots = {k: unit(v - r_mean) for k, v in bank.items()} | {"(own retarget)": unit(own - r_mean)}
        prefix = next((p for c, p in CATEGORY.items() if name.lower().startswith(c)), None)
        prefixes = prefix if isinstance(prefix, tuple) else (prefix,) if prefix else ()
        res = {"view_azimuth": h["az"], "bucket": bucket}
        print(f"\n{name} (viewpoint {h['az']:.0f} deg, {len(H)} latent steps)", flush=True)
        for score_name, fn in SCORES.items():
            sims = {k: float(fn(H, R)) for k, R in robots.items()}
            ranked = sorted(sims.items(), key=lambda kv: -kv[1])
            bank_ranked = [k for k, _ in ranked if k != "(own retarget)"]
            hit = next((i + 1 for i, k in enumerate(bank_ranked) if any(k.startswith(p) for p in prefixes)), None)
            own_rank = [k for k, _ in ranked].index("(own retarget)") + 1
            if hit:
                hits[score_name].append(hit)
            res[score_name] = {"rank_of_same_action": hit, "own_retarget_rank": own_rank, "top5": ranked[:5]}
            print(f"  {score_name:9s}: same action #{hit}, own retarget #{own_rank} | top: "
                  + ", ".join(f"{k} {v:+.2f}" for k, v in ranked[:4]), flush=True)
        results[name] = res

    print()
    for score_name, hs in hits.items():
        if hs:
            print(f"{score_name:9s}: top-1 {np.mean([x == 1 for x in hs]):.0%}, top-3 {np.mean([x <= 3 for x in hs]):.0%}"
                  f", mean rank {np.mean(hs):.1f} over {len(hs)} clips (26 bank motions; chance top-1 ~4%)")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "cross_embodiment.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="outputs/cross_embodiment")
    a = ap.parse_args()
    main(a.dirs, a.out)
