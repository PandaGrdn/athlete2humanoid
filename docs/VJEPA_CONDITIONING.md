# The video's V-JEPA latent as a policy input (2026-10-28)

**Question:** if the policy also sees the video in V-JEPA 2's latent space, does it generalise better, including to an action it was never trained on?

## Setup

`experiments/multiclip_policy.py`, run `outputs/remote_multiclip_vj1`: one Modal L40S, about 2 h, about $4.

**One tracking policy over five clips:** squat jump, front kick, squat, arms, lunge.
- It trains on all their automatic variants at once (`skill/variations.py`): 165 variants.
- It starts from the squat jump's variant-trained tracker and trains 2,000 iterations.

**Jumping jacks and punches are never trained on.** They stand in for "something that isn't in the variants".

**Three policies, identical except for one input:**

| Policy | Extra input |
|---|---|
| `multi` | none: mjlab's tracking inputs |
| `multi_vj` | the video's V-JEPA latent now, 0.5 s and 1 s ahead (3 × 32) |
| `multi_future` | the retarget's joints and pelvis height 0.5 s and 1 s ahead: the same look-ahead, without V-JEPA (the control) |

**How the latent is made (`latent/video_condition.py`):**
- The athlete's silhouette video is embedded densely: one 1024-d vector per 2 frames, each with about 1 s of context.
- A PCA fitted on the training clips only reduces it to 32 components, which keep 93% of the variance.
- It is laid out per motion step:
  - re-timed variants read the video at their source frames;
  - mirrored variants read the mirrored video's latent.
- A new clip's latent comes from its own video, through the same PCA.

**The test** (`experiments/robustness_eval.py`) runs on every clip, trained on or held out, 64 rollouts per condition:
- the clip itself;
- its held-out variants (outside the training grid);
- 0.5x and 2x tempo;
- 1 m/s pushes.

**Success:** every key moment of the clip's keypose spec, in order, with no fall.

## Results

Task success, with the fall rate in brackets.

**Trained clips** (averaged over the five):

| Policy | The clip | Its held-out variants | Tempo 0.5x / 2x | Pushed 1 m/s |
|---|---|---|---|---|
| multi | 44% (16%) | 38% (17%) | 22% (19%) | 38% (21%) |
| **multi_vj** | **80% (0%)** | 61% (3%) | **40% (3%)** | 57% (2%) |
| multi_future | **80% (0%)** | **63% (6%)** | 30% (19%) | **67% (2%)** |

**Held-out clips** (jumping jacks, punches):

| Policy | The clip | Its held-out variants |
|---|---|---|
| multi | 0% (50%) | 0% (54%) |
| multi_vj | 0% (50%) | 0% (51%) |
| multi_future | 0% (56%) | 0% (79%) |

**Per clip** (the clip / its held-out variants):

| Clip | multi | multi_vj | multi_future |
|---|---|---|---|
| squat jump | 100% / 77% | 100% / 73% | 100% / 75% |
| front kick | 100% / 56% | 100% / 67% | 100% / 58% |
| squat | 0% / 29% | 98% / 63% | 100% / 85% |
| arms | 20% / 27% | 100% / 100% | 100% / 93% |
| lunge | 0% / 0% | 0% / 0% | 0% / 0% |
| jumping jacks (held out) | 0% / 0%, every rollout falls | the same | the same |
| punches (held out) | 0% / 0%, no falls, 48% of stages | 0% / 0%, 31% of stages | 0% / 0%, 32% of stages |

## What it says

1. **One policy for many clips needs to know what is coming.**
   - Plain `multi` fails the squat and the arms clip. Tracking only the current frame, it can't tell which of five motions it is in until it is already late.
   - Both added inputs fix that: 44% → 80% on the clips, and falls 16% → 0%.
2. **V-JEPA does as well as seeing the retarget ahead, but not better overall.**
   - It is better under tempo changes: 40% vs 30% success, and 3% vs 19% falls. The latent describes the motion, not exact joint targets, so it degrades more gracefully when the timing is off.
   - The look-ahead is better on pushes and on the squat's variants.
   - So far, the latent is a compact, timing-tolerant description of what is coming, not a source of new knowledge.
3. **Neither generalises to an action it never trained on.**
   - Jumping jacks (a jump with the arms overhead) make every policy fall.
   - On punches, all three stay up, but none completes the punches. Plain tracking gets furthest: 48% of the stages.
   - Five clips is far too few to learn tracking in general. Tracking an arbitrary new motion zero-shot takes motion libraries of thousands of clips (the AMASS-scale trackers).
4. **Lunge fails for all three.** It walks, and only 15 of its 36 variants survive the variant filters (`docs/GENERALIZATION.md`).

## Next

- **Adapting to a new clip (running next):** each of the three policies is fine-tuned briefly on jumping jacks, against the plain single-clip start. Does a generalist that sees the video latent adapt faster? That is the practical form of "any video becomes a policy".
- **Scale:** the latent's point is to describe any motion, but it can only help once there are many motions to learn from. The distillation dataset already has 536 robot motions with V-JEPA embeddings (`latent/distill.py`); training the conditioned tracker on those is the real test of "it understands in latent space what is going on".
