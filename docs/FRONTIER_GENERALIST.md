# Frontier 1: a scaled generalist, zero-shot (2026-10-29)

**Question:** the 5-clip tracker `multi` completes 0% of a clip it never trained on. If one tracker trains on many
more motions, does it start to track new clips **without retraining**?

**Zero-shot test clips** (shared by the four frontier directions, never trained on): jumping jacks, punches, discus.

## Method

`experiments/generalist_policy.py`. The pool is built locally. Training ran on Modal through the multiclip job;
the evaluation ran on a Colab A100 (`compare`, see Compute).

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
| `real` | + shot put (the non-test Olympic film that passes the filter) | 6 | 183 | 22 min |
| `real_bank` | + the procedural bank (`latent/bank.py`) | 28 | 539 | 47 min |
| `all` | + 300 recombinations (`latent/distill.py variation()`) | 328 | 1,133 | 84 min |

The hurdles film was meant to be the seventh real clip. It is left out because it ends in mid-flight
(below).

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
| On the floor | no foot more than 3 cm into it; a foot on the floor in the first and last frame |
| Flights | both feet up only where the pelvis rises as gravity requires (`retarget.py`'s rule) |
| No sliding | a planted foot's 95th-percentile speed ≤ 0.5 m/s (not for real clips: see below) |
| Moving | not a frozen pose |
| Its own task | each variant's reference completes its own keypose spec (self-check) |

- **Recombinations:** 300 of 2,023 passed (15%). The main rejections were sliding feet (1,369), floating
  (1,138), feet through the floor (1,051) and starting or ending in the air (923).
- **The bank:** 22 of its 24 motions passed. `high_knees` slides; `stand` is frozen.
- **Real clips get every rule except sliding.** The lunge, the shot-put glide, the discus spin and the
  jacks read as 1.1–1.8 m/s "slides" at 30 fps, though trackers learn them (`docs/PRO_CLIPS.md`).

**Why the start-and-end rule exists.** The first launch trained on hurdles, whose film ends in mid-air.
- mjlab's adaptive sampling sends episodes to where the policy fails. 11–38% of all episode starts went to
  one bin in the hurdles' last frames, where a policy can only fall.
- Every level was affected: episodes were still 20–37 steps long at iteration 1,000, against 200 for the
  5-clip run.
- Those runs were stopped after about 30 minutes. With the rule, the busiest bin takes 2–10% of starts,
  and episodes reach 260–400 steps.

### 2. Training (Modal L40S) and evaluation (Colab A100)

- **Motions to mjlab.** Every CSV becomes mjlab's 50 Hz `motion.npz` in one process, many in parallel
  environments (`convert()`). The output matches `mjlab.scripts.csv_to_npz` to 1e-6.
- **Policies,** trained side by side on the level's variants (`VariantMotionCommand`, adaptive sampling):

| Policy | Inputs | Starts from |
|---|---|---|
| `gen_multi` | mjlab's tracking inputs | the 5-clip `multi` |
| `gen_future` | + the retarget's joints and pelvis height 0.5 / 1 s ahead | the 5-clip `multi_future` |
| `gen_future_s` | the same as `gen_future` | scratch |
| `gen_future_l` | the same, with a (1024, 512, 256) network instead of (512, 256, 128) | scratch |

- **Evaluation** (`experiments/robustness_eval.py`, 64 rollouts per condition, all 10 policies batched in
  one environment; `generalist_policy.py compare`):
  - zero-shot clips: every suite (the `multiclip_policy.py --full-eval` reference), including the held-out
    variants and the V-JEPA score;
  - trained real clips: the clean clip and its held-out variants.
- **Baselines:** the 5-clip `multi` and `multi_future`, evaluated on the same re-retargeted references.

**What actually ran.** Both Modal workspaces ran out of credits mid-run (shared with the other sessions).
- Every training run stopped at iteration **1,750**, the last checkpoint to reach the volume. The scale runs
  were meant to reach 2,000; the network-size runs (from scratch) were meant to reach 4,000.
- The evaluation ran on one Colab A100 from the synced checkpoints.
- The scratch policies have 1,750 iterations against the warm-started ones' 2,000 + 1,750, so the
  network-size comparison is between two undertrained policies.

## Results

`outputs/remote_multiclip_gen2_compare/job.log`; recovered in `docs/results/frontier_generalist/`:
the summary, every condition's numbers per policy (`conditions.json`), and the pool report (`pool.json`).
The run's summary was printed to that log; the robust.json
files were lost with the VM, because `compare` didn't pack them (fixed since). Task success, with the fall
rate in brackets. "Stages" is the share of the clip's key moments reached. V-JEPA is the aligned cosine to
the human video.

### The scale curve, zero-shot (jumping jacks, punches, discus; averaged)

| Policy | Motions | The clip | Stages | Held-out variants | Stages | V-JEPA |
|---|---|---|---|---|---|---|
| `multi` (baseline) | 5 clips | 0% (67%) | 36% | 0% (70%) | 14% | 0.34 |
| `multi_future` (baseline) | 5 clips | 0% (69%) | 30% | 0% (84%) | 13% | 0.34 |
| `gen_multi`, real | 6 clips | 0% (49%) | 41% | 1% (54%) | 34% | 0.44 |
| `gen_future`, real | 6 clips | 0% (67%) | 31% | 1% (57%) | 31% | 0.36 |
| `gen_multi`, real_bank | 28 | 0% (67%) | 39% | 0% (66%) | 39% | 0.39 |
| `gen_future`, real_bank | 28 | 0% **(32%)** | 42% | 1% (56%) | 37% | **0.45** |
| `gen_multi`, all | 328 | **1%** (44%) | **46%** | 4% (44%) | **47%** | 0.41 |
| **`gen_future`, all** | 328 | 0% (34%) | 33% | 5% **(40%)** | 31% | 0.43 |
| `gen_future_s`, all, scratch | 328 | 0% (40%) | 36% | **7%** (50%) | 37% | 0.40 |
| `gen_future_l`, all, scratch, bigger network | 328 | 0% (46%) | 40% | 2% (61%) | 35% | 0.43 |

**Per zero-shot clip** (the clip / its held-out variants):

| Policy | Jumping jacks | Punches | Discus |
|---|---|---|---|
| `multi` (baseline) | 0% (100%) / 0% (100%) | 0% (0%) / 0% (11%) | 0% (100%) / 0% (100%) |
| `multi_future` (baseline) | 0% (100%) / 0% (100%) | 0% (6%) / 1% (52%) | 0% (100%) / 0% (100%) |
| `gen_multi`, real | 0% (50%) / 0% (66%) | 0% (0%) / 0% (0%) | 0% (98%) / 4% (96%) |
| `gen_future`, real_bank | 0% (0%) / 1% (66%) | 0% (0%) / 0% (3%) | 0% (95%) / 2% (98%) |
| `gen_multi`, all | 2% (33%) / 12% (33%) | 0% (0%) / 0% (0%) | 0% (100%) / 1% (99%) |
| `gen_future`, all | 0% **(2%)** / 16% (21%) | 0% (0%) / 0% (0%) | 0% (100%) / 0% (98%) |
| `gen_future_s`, all, scratch | 0% (17%) / **21%** (44%) | 0% (2%) / 0% (7%) | 0% (100%) / 0% (100%) |

**Every zero-shot suite** (averaged over the three clips; task success / fall rate):

| Suite | `multi` | `multi_future` | `gen_multi` all | `gen_future` all | `gen_future` real_bank |
|---|---|---|---|---|---|
| Clean | 0% / 67% | 0% / 69% | 1% / 44% | 0% / 34% | 0% / 32% |
| Pushes 0.5–1.5 m/s | 0% / 69% | 0% / 73% | 0% / 51% | 0% / 36% | 0% / 37% |
| Stage-timed pushes | 0% / 67% | 0% / 72% | 0% / 50% | 0% / 37% | 0% / 35% |
| Start noise | 1% / 68% | 1% / 72% | 4% / 54% | 1% / 50% | 2% / 50% |
| Start pose | 0% / 67% | 0% / 68% | 0% / 36% | 0% / 35% | 0% / 33% |
| Speed 0.8 / 1.2x | 0% / 67% | 0% / 70% | 2% / 58% | 0% / 36% | 0% / 34% |
| Tempo 0.5–2x | 0% / 70% | 0% / 77% | **8%** / 54% | 0% / 56% | 0% / 64% |
| Friction / mass | 0% / 67% | 0% / 72% | 1% / 44% | 0% / 37% | 0% / 35% |
| Held-out variants | 0% / 65% | 0% / 81% | 4% / 38% | 5% / 34% | 1% / 50% |

### The trained real clips (six; averaged)

| Policy | The clip | Held-out variants | Joint DTW (rad) |
|---|---|---|---|
| `multi` (baseline) | 37% (30%) | 32% (31%) | 0.160 |
| `multi_future` (baseline) | 67% (17%) | 51% (22%) | 0.152 |
| `gen_multi`, real | **77% (2%)** | 49% (2%) | 0.162 |
| `gen_future`, real_bank | 67% (0%) | **52% (4%)** | 0.160 |
| `gen_multi`, all | 38% (0%) | 38% (11%) | 0.169 |
| `gen_future`, all | 67% (5%) | 48% (10%) | 0.163 |
| `gen_future_s`, all, scratch | 53% (14%) | 32% (27%) | 0.175 |
| `gen_future_l`, all, scratch | 47% (47%) | 40% (37%) | 0.162 |

- **The shot put becomes trainable.** It was 0% with 100% falls for both baselines (they never saw it).
  `gen_multi` real reaches 67% on the clip.
- **Lunge stays at 0% for everyone,** as in every earlier run.
- **More motions don't cost the old clips.** Squat jump, arms and squat stay at 90–100% for the look-ahead
  policies. Falls on the trained clips drop from 17–30% to 0–5% for five of the six warm-started policies; `gen_multi` real_bank still falls 29% of the time.

## What it says

1. **No zero-shot task success.** Across 10 policies and 3 clips, the best is 1% on a clean clip and
   5–7% on held-out variants. Today's numbers say the same as the 5-clip result: tracking an action the
   policy never trained on is not solved by 328 motions.
2. **But there is a trend, and it is in staying up.** Zero-shot falls on the clean clips go from 67–69%
   (5 clips) to 32–34% (28–328 motions, with the look-ahead). They are lower in every robustness suite. Key moments
   reached on the held-out variants go from 13–14% to 31–47%. The V-JEPA match to the video rises from 0.34
   to 0.43–0.45, mostly because the robot stays up and keeps moving like a person.
3. **Jumping jacks is where the curve shows most.**
   - The baselines fall on every rollout.
   - `gen_future` all falls 2% of the time and completes 16% of the held-out variants; scratch reaches 21%.
   - The pool has the pieces (side arm raises, jumps, squats) but not the combination, and the policy
     tracks the combination well enough to stay up.
4. **The discus doesn't move:** 95–100% falls for all. Nothing in the pool spins 429° on one foot; the
   bank's `twist` turns the torso, not the whole body. Without a motion of its kind, more data of other
   kinds doesn't help.
5. **Punches never fall and never finish,** before or after (0–6% falls, 0% success). An arm-only motion is
   easy to stay up through. Its spec asks for hand positions that the policy doesn't reach precisely without
   training on them.
6. **Look-ahead (`gen_future`) vs plain (`gen_multi`):** the look-ahead keeps the robot up more (34% vs 44%
   falls zero-shot). Plain tracking reaches more key moments (46% vs 33%) and gets the most tempo-changed
   success (8%). Neither dominates.
7. **Bank vs recombinations:**
   - The bank alone gives most of the fall reduction (`gen_future`: 67% → 32%).
   - The recombinations add the first held-out-variant successes (1% → 5%) and help on jumping jacks.
   - On the trained clips, `all` is slightly worse than `real_bank` (48% vs 52% on variants, more falls):
     the random recombinations dilute the real clips.
8. **Network size: inconclusive.** Both were trained from scratch for 1,750 of the planned 4,000 iterations.
   - The default network does better than the bigger one on held-out variants (7% vs 2%) and on the
     trained clips (53% vs 47%, 14% vs 47% falls).
   - The bigger network is further from converged, so this says little about size.

**Honest conclusion.** Scaling from 5 clips to 328 motions halves zero-shot falls and triples the key
moments reached on unseen variants, but it does not produce zero-shot task success. The one test motion
whose parts are in the pool (jumping jacks) moves from "always falls" to "rarely falls, sometimes
completes". The one with nothing like it in the pool (the discus spin) doesn't move at all. That is the
signature of a tracker that interpolates between what it has seen. Zero-shot tracking of an arbitrary
athletic move needs a motion library that covers it, at AMASS scale (thousands of clips), not 328
procedural and recombined motions.

## The best checkpoint (start for fast adaptation)

**`gen_future`, level `all`, iteration 1,750:** the lowest zero-shot falls on held-out variants (40%), the
most jumping-jacks variants completed among the warm-started policies, and 67% / 48% on the trained clips.
- Checkpoint (full, with optimizer state):
  `.claude/worktrees/frontier-generalist-771588/outputs/remote_multiclip_gen2_all/checkpoints/policy_gen_future/model_1750.pt`
- Task: `A2H-TrackingFuture-Flat-Unitree-G1` (mjlab tracking inputs plus the retarget 0.5 / 1 s ahead).
- Plain-tracking alternative (`A2H-Tracking-Flat-Unitree-G1`, for a `multi`-style start):
  `.../outputs/remote_multiclip_gen2_all/checkpoints/policy_gen_multi/model_1750.pt`
- The others: `.../outputs/remote_multiclip_gen2_{real,bank,size}/checkpoints/policy_*/model_1750.pt`.

## Cost

| Item | Compute | Cost |
|---|---|---|
| Smoke runs (3) and the first, preempted launch | Modal L40S, minutes | ~$1 |
| First full launch, with the hurdles (stopped after 30 min) | 4 × L40S, 0.5 h | ~$4 |
| The training runs (4, to iteration 1,750) | 4 × L40S, ~0.8 h | ~$6.5 |
| The evaluation (10 policies × 9 clips) | Colab A100, 2.7 h | ~18 compute units (estimate) |
| **Total** | | **~$11.5 on Modal + ~18 Colab units** |

Both Modal workspaces' credits ran out with the other sessions' jobs; this direction's share was about
$11.5 of the $30 budget.

## Next

- **Data that covers the motion.** The one result that moved, jumping jacks, is the one whose parts were in
  the pool. Zero-shot on arbitrary moves needs a large human motion library, retargeted with `retarget.py`.
  AMASS is the standard one. **It needs your approval to download** (non-commercial licence, registration
  at amass.is.tue.mpg.de). Not done here.
- **Finish the training:** 250 more iterations for the scale runs, 2,250 for the size runs, then rerun
  `compare`. The size question can only be answered by policies that converged.
- **Spinning and travelling motions in the pool.** The discus and lunge failures both point at whole-body
  turning and travel, which the bank lacks. Bank generators for pivots and walking turns are cheap.
