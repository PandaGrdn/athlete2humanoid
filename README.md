# Athlete2Humanoid

Turn any athlete video into a Unitree G1 humanoid performing the same move.

**Pipeline:** video → **SAM 3.1** tracks the athlete → MediaPipe 3D pose → **mink** IK retargets it onto the G1 → MuJoCo replay, then physics and RL → **Muse Spark** labels the phases, coaches, and designs the rewards.

## Layout

| File | Day | Status |
|---|---|---|
| `config.py` | 1 | ✅ paths, model names, env overrides |
| `render.py` | 1 | ✅ load G1, physics and kinematic rendering, save video · ⬜ `side_by_side` (day 3) |
| `coach.py` | 1/4 | ✅ Muse Spark client (`ask`, with optional images or JSON) · ⬜ phases (4), infeasibility (6), rewards (7) |
| `track.py` | 1 | ✅ SAM 3.1 tracking, main-athlete mask, overlay, athlete-following crops |
| `pose.py` | 1 | ✅ MediaPipe 3D pose + 2D root motion → metric, z-up human trajectory |
| `retarget.py` | 1 | ✅ G1 IK retargeting (limbs, pelvis, torso orientation), error report, mjlab CSV export |
| `pipeline.py` | 1 | ✅ video → G1 motion CSV + side-by-side preview (`--no-sam` runs on any machine) |
| `mjlab_rollout.py` | 1 | ✅ headless rollout of a trained mjlab policy → video + metrics (untested on GPU) |
| `notebooks/video_to_policy.ipynb` | 1 | ✅ Colab: clip → SAM → pose → retarget → mjlab PPO → policy video |
| `app.py` | 5 | ⬜ Gradio demo |
| `train_rl.py` | 6–8 | ⬜ physics playback, PPO imitation (MuJoCo Playground) |


## Run it

**Colab (main path):** open `notebooks/video_to_policy.ipynb` on an A100 and run it top to bottom.

**Locally (robot parts only, no GPU needed):**
```bash
pip install -r requirements.txt
bash scripts/fetch_assets.sh
python render.py      # outputs/g1_stand.mp4
python retarget.py    # outputs/g1_ik_squat.mp4
python pipeline.py clips/your_clip.mp4 --no-sam   # outputs/your_clip/side_by_side.mp4 + motion.csv
```

## Gotchas found during setup

- **SAM 3.1** defaults to FlashAttention 3, which only runs on H100s. `track.py` passes `use_fa3=False`, and you need an **L4 or A100**; a T4 won't work.
- **sam3 pins `numpy<2`**, which has no Python 3.13 build (Colab's current Python). SAM 3.1 runs in its own Python 3.12 env via `sam_track.py`, and mjlab in another; the main env keeps Colab's numpy 2.
- **G1 "stand" keyframe has straight knees,** a singularity where IK can't start a knee bend and tilts the pelvis instead. `retarget.bent_knee_pose` starts from a slight crouch.
- **mink `lm_damping`** is left at 0 (the default). Tested from the straight-knee pose, any value above 0 stalled the solver about 13 cm short of the target.
- **G1 IK costs** (`retarget.py`) were tuned on a synthetic squat: feet flat (cost 5), pelvis orientation soft (0.1), torso upright (1.0). Re-check them on real clips.
- **Muse Spark API** is OpenAI-compatible (`openai` SDK, `base_url=https://api.meta.ai/v1`, model `muse-spark-1.1`). Confirm both at kickoff; override them with `META_API_BASE` and `MUSE_MODEL`.
- **MediaPipe:** use the Tasks API (`PoseLandmarker`), not the legacy `mp.solutions.pose`.
