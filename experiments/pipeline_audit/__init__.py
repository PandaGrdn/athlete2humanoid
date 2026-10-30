"""Measurement tools behind docs/PIPELINE_ACCURACY.md: where the athlete's motion is lost between the video and
the simulator, stage by stage, on the clips already processed under outputs/. Nothing here changes the pipeline.

    python experiments/pipeline_audit/pose_stats.py        # human.npy: depth noise, bone-length jitter, feet vs floor
    python experiments/pipeline_audit/raw_pose.py          # MediaPipe on the whole frame, CPU: pre-smoothing noise, fps, 3D/2D consistency
    python experiments/pipeline_audit/retarget_audit.py    # re-run retarget.py: IK error, limits, speed cap, root shift, phantom flights
    python experiments/pipeline_audit/spec_audit.py        # keypose specs: tolerances vs noise, key-moment stability, tempo
    python experiments/pipeline_audit/spec_noise.py        # which predicate (tolerance, hold) breaks a spec under limb noise
    python experiments/pipeline_audit/variants_audit.py    # the variant filter against physics (ballistics, joint speed)

Each writes a JSON next to docs/results/pipeline_audit/ (override with --out). All run on CPU in the main
environment (mujoco, mink, torch, scipy; raw_pose.py also needs mediapipe and the pose model under assets/).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "docs" / "results" / "pipeline_audit"

# Processed clip folders (human.npy, qpos.npy, meta.json), relative to the repo root.
CLIPS = {
    "squat_jump": "outputs/mc_inputs/clips/squat_jump", "squat": "outputs/mc_inputs/clips/squat",
    "lunge": "outputs/mc_inputs/clips/lunge", "arms": "outputs/mc_inputs/clips/arms",
    "front_kick": "outputs/mc_inputs/clips/front_kick", "jumping_jacks": "outputs/mc_inputs/clips/jumping_jacks",
    "punches": "outputs/mc_inputs/clips/punches",
    "shot_put": "outputs/remote_process_pro/clips/shot_put", "discus": "outputs/remote_process_pro/clips/discus",
    "hurdles": "outputs/remote_process_pro/clips/hurdles",
}
# The source videos, for raw_pose.py.
VIDEOS = {"squat_jump": "clips/squat_jump.mov", "squat": "clips/squat.mov", "lunge": "clips/lunge.mov", "arms": "clips/arms.mov",
          "front_kick": "clips/front_kick.mov", "jumping_jacks": "clips/jumping_jacks.mov", "punches": "clips/punches.mov",
          "shot_put": "clips_pro/shot_put.mp4", "discus": "clips_pro/discus.mp4", "hurdles": "clips_pro/hurdles.mp4"}
MAJOR = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # MediaPipe shoulders, elbows, wrists, hips, knees, ankles


def clip_dirs(only=None):
    """{name: Path} of the processed clips that exist locally (optionally only the named ones)."""
    out = {n: ROOT / d for n, d in CLIPS.items() if (ROOT / d / "human.npy").exists()}
    return {n: d for n, d in out.items() if not only or n in only}


def noise(x, fps):
    """Per column, the pose noise: 1.4826 x MAD of the residual from a ~0.2 s moving average (robust to the motion)."""
    k = max(3, int(round(0.2 * fps)) | 1)
    res = x - uniform_filter1d(x, k, axis=0, mode="nearest")
    return 1.4826 * np.median(np.abs(res - np.median(res, 0)), 0)


def dump(rows, path):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1, default=float))
    return path
