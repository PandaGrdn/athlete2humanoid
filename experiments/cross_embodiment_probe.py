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
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render  # noqa: E402
from latent import bank as motion_bank  # noqa: E402
from latent import silhouette  # noqa: E402
from latent.video_latent import video_latent  # noqa: E402
from latent.vjepa import VJepa  # noqa: E402

# Which bank motions count as "the same action" for each clip name.
CATEGORY = {"squat_jump": ("vertical_jump", "squat_"), "squat": "squat_", "lunge": "lunge_", "jacks": "jumping_jacks", "kick": "kick_",
            "arms": "arms_raise_", "jump": "vertical_jump", "march": "march", "punch": "punch"}


def _processed(path):
    """A pipeline output folder; videos are run through pipeline.py into outputs/<name> first."""
    path = Path(path)
    if path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
        import subprocess
        import config
        import imageio_ffmpeg
        import pipeline
        out = config.OUTPUT_DIR / path.stem.lower()
        if not (out / "masks.npz").exists():
            # 30 fps like the robot renders (phones often record 60), 720p for SAM and MediaPipe.
            out.mkdir(parents=True, exist_ok=True)
            clip = out / "clip30.mp4"
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
                            "-vf", "fps=30,scale=-2:720", "-an", str(clip)], check=True)
            pipeline.video_to_motion(clip, out, use_sam=True)
        return out
    return path


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main(dirs, out):
    dirs = [_processed(d) for d in dirs]
    vj = VJepa()
    g1 = render.load_g1()[0]
    bank = motion_bank.motion_bank()

    human = {}
    for d in map(Path, dirs):
        emb, _ = video_latent(d, vj)
        human[d.name] = {"emb": emb, "az": json.loads((d / "meta.json").read_text())["view_azimuth"],
                         "own": np.load(d / "qpos.npy")}
    h_mean = np.concatenate([h["emb"] for h in human.values()]).mean(0)

    results = {}
    for name, h in human.items():
        robot = {k: vj.embed_windows(silhouette.robot_silhouettes(g1, q, azimuth=h["az"]))[0] for k, q in bank.items()}
        own = vj.embed_windows(silhouette.robot_silhouettes(g1, h["own"], azimuth=h["az"]))[0]
        r_mean = np.concatenate(list(robot.values())).mean(0)
        hv = unit(h["emb"].mean(0) - h_mean)
        sims = {k: float(unit(v.mean(0) - r_mean) @ hv) for k, v in robot.items()}
        sims["(own retarget)"] = float(unit(own.mean(0) - r_mean) @ hv)
        ranked = sorted(sims.items(), key=lambda kv: -kv[1])
        prefix = next((p for c, p in CATEGORY.items() if name.lower().startswith(c)), None)
        bank_ranked = [k for k, _ in ranked if k != "(own retarget)"]
        hit = next((i + 1 for i, k in enumerate(bank_ranked) if prefix and k.startswith(prefix)), None)
        results[name] = {"view_azimuth": h["az"], "rank_of_same_action": hit, "top5": ranked[:5],
                         "own_retarget": sims["(own retarget)"]}
        print(f"\n{name} (viewpoint {h['az']:.0f} deg): same action ranks #{hit} of {len(bank_ranked)} bank motions")
        for k, s in ranked[:5]:
            print(f"   {s:+.3f}  {k}")

    hits = [r["rank_of_same_action"] for r in results.values() if r["rank_of_same_action"]]
    if hits:
        print(f"\ntop-1 accuracy {np.mean([h == 1 for h in hits]):.0%}, top-3 {np.mean([h <= 3 for h in hits]):.0%}"
              f" over {len(hits)} clips (chance top-1 ~{1 / len(bank):.0%})")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "cross_embodiment.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="outputs/cross_embodiment")
    a = ap.parse_args()
    main(a.dirs, a.out)
