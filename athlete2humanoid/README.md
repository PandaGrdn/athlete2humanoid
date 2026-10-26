# Athlete2Humanoid

Turn any athlete video into a Unitree G1 humanoid performing the same move.

**Pipeline:** video → **SAM 3.1** tracks the athlete → MediaPipe 3D pose → **mink** IK retargets it onto the G1 → MuJoCo replay, then physics and RL → **Muse Spark** labels the phases, coaches, and designs the rewards.

## Layout

| File | Day | Status |
|---|---|---|
| `config.py` | 1 | ✅ paths, model names, env overrides |
| `render.py` | 1 | ✅ load G1, physics and kinematic rendering, save video · ⬜ `side_by_side` (day 3) |
| `coach.py` | 1/4 | ✅ Muse Spark client (`ask`, with optional images or JSON) · ⬜ phases (4), infeasibility (6), rewards (7) |
| `track.py` | 1/2 | ✅ SAM 3.1 tracking, main-athlete mask, overlay · ⬜ `crop_to_mask` (2) |
| `pose.py` | 2 | ⬜ MediaPipe PoseLandmarker on crops, smoothing |
| `retarget.py` | 3 | ✅ G1 IK solver (squat demo) · ⬜ `scale_landmarks` (3) |
| `app.py` | 5 | ⬜ Gradio demo |
| `train_rl.py` | 6–8 | ⬜ physics playback, PPO imitation (MuJoCo Playground) |
| `notebooks/day1_setup.ipynb` | 1 | ✅ runs every day-1 check in Colab |

## Run it

**Colab (main path):** open `notebooks/day1_setup.ipynb` and follow the instructions at the top.

**Locally (robot parts only, no GPU needed):**
```bash
pip install -r requirements.txt
bash scripts/fetch_assets.sh
python render.py      # outputs/g1_stand.mp4
python retarget.py    # outputs/g1_ik_squat.mp4
```

## Gotchas found during setup

- **SAM 3.1** defaults to FlashAttention 3, which only runs on H100s. `track.py` passes `use_fa3=False`, and you need an **L4 or A100**; a T4 won't work.
- **sam3 pins `numpy<2`**, so the whole project stays on numpy 1.26. That needs `scipy<1.17`, since newer SciPy crashes mink on import. The notebook's install cell restarts the runtime when numpy gets downgraded.
- **G1 "stand" keyframe has straight knees,** a singularity where IK can't start a knee bend and tilts the pelvis instead. `retarget.bent_knee_pose` starts from a slight crouch.
- **mink `lm_damping`** is left at 0 (the default). Tested from the straight-knee pose, any value above 0 stalled the solver about 13 cm short of the target.
- **G1 IK costs** (`retarget.py`) were tuned on a synthetic squat: feet flat (cost 5), pelvis orientation soft (0.1), torso upright (1.0). Re-check them on real clips.
- **Muse Spark API** is OpenAI-compatible (`openai` SDK, `base_url=https://api.meta.ai/v1`, model `muse-spark-1.1`). Confirm both at kickoff; override them with `META_API_BASE` and `MUSE_MODEL`.
- **MediaPipe:** use the Tasks API (`PoseLandmarker`), not the legacy `mp.solutions.pose`.
