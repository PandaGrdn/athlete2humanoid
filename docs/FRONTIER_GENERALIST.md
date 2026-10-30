# Frontier 1: a scaled generalist, zero-shot (2026-10-29)

**Question:** the 5-clip tracker `multi` completes 0% of a clip it never trained on. If one tracker trains on many
more motions, does it start to track new clips **without retraining**?

**Zero-shot test clips** (shared by the four frontier directions, never trained on): jumping jacks, punches, discus.

## Method

`experiments/generalist_policy.py`. Two steps: the pool is built locally; training and evaluation run on Modal
through the multiclip job.

### 1. The motion pool (local, CPU, ~8 min)

**Real clips, re-retargeted.** All ten clips were re-made from `human.npy` with today's `retarget.py`
(the joint-speed cap and the gravity check on flights, commit 42f3d64):

| Clip | Fastest joint before → after (rad/s) | Largest joint change (rad) |
|---|---|---|
| jumping jacks (zero-shot) | 75.9 → 20.0 | 3.0 (the overhead-arm flips) |
| discus (zero-shot) | 65.2 → 20.0 | 3.1 |
| punches (zero-shot) | 10.1 → 10.1 | 0.25 |
| hurdles | 78.0 → 20.0 | 2.8 |
| front kick | 21.7 → 19.9 | 0.33 |
| squat jump, squat, arms, lunge, shot put | unchanged | 0 |

The re-retargeted discus and jumping jacks are not the references earlier docs were tested on.

**Three sources of motion, nested into three levels:**

| Level | Adds | Motions | Variants trained on | Motion |
|---|---|---|---|---|
| (baseline) `multi` | squat jump, front kick, squat, arms, lunge | 5 | 159 | 20 min |
| `real7` | + shot put, hurdles (the two non-test Olympic films) | 7 | 207 | 23 min |
| `real7_bank` | + the procedural bank (`latent/bank.py`) | 29 | 563 | 48 min |
| `all` | + 300 recombinations (`latent/distill.py variation()`) | 329 | 1,151 | 85 min |

- **The bank** leaves out its `jumping_jacks` and `punch`: they are procedural copies of two test clips, and
  training on them would not be zero-shot.
- **Recombinations** are built as the latent policy's 536-motion distillation set is: another motion's upper
  body, half-done motions, joint noise, leaning, re-timing.
- **Variants** (`skill/variations.py`): the real clips get the full training grid (mirror × 3 tempos ×
  3 amplitudes × 2 warps); bank motions mirror × tempo × amplitude; recombinations mirror only.

**Every motion is checked trackable first** (`trackable()`, then `skill/variations.py`'s own filters):

| Check | Rule |
|---|---|
| Joint limits | at most 2% of samples past a limit (then clipped), as `skill/variations.py` |
| Joint speed | ≤ 20 rad/s, `retarget.py`'s cap |
| Upright | pelvis tilt ≤ 1.0 rad; torso ≥ 0.45 m (the eval's fall test: 1.4 rad, 0.35 m) |
| On the floor | no foot more than 3 cm into it |
| Flights | both feet up only where the pelvis rises as gravity requires (`retarget.py`'s rule) |
| No sliding | a planted foot's 95th-percentile speed ≤ 0.5 m/s |
| Moving | not a frozen pose |
| Its own task | each variant's reference completes its own keypose spec (self-check) |

Of 2,178 random recombinations, 300 passed (14%); the main rejections were sliding feet (1,522), floating
(1,264) and feet through the floor (1,193). Of the bank's 24 motions, 22 passed (`high_knees` slides; `stand`
is frozen).

### 2. Training and evaluation (Modal L40S)

- **Motions to mjlab.** Every CSV becomes mjlab's 50 Hz `motion.npz` in one process, many in parallel
  environments (`convert()`). The output matches `mjlab.scripts.csv_to_npz` to 1e-6.
- **Policies,** trained side by side on the level's variants (`VariantMotionCommand`, adaptive sampling):

| Policy | Inputs | Starts from |
|---|---|---|
| `gen_multi` | mjlab's tracking inputs | the 5-clip `multi` |
| `gen_future` | + the retarget's joints and pelvis height 0.5 / 1 s ahead | the 5-clip `multi_future` |
| `gen_future_s` | the same as `gen_future` | scratch |
| `gen_future_l` | the same, with a (1024, 512, 256) network instead of (512, 256, 128) | scratch |

- **Evaluation** (`experiments/robustness_eval.py`, 64 rollouts per condition):
  - zero-shot clips: every suite (the `multiclip_policy.py --full-eval` reference), including the held-out
    variants and the V-JEPA score;
  - trained real clips: the clean clip and its held-out variants.
- **Baselines:** the 5-clip `multi` and `multi_future`, evaluated on the same re-retargeted references.

## Results

(pending)
