# V-JEPA 2 as a world model: the robot imagines what the video does next (2026-10-29)

**Question:** the video's V-JEPA latent helps the multi-clip tracker most when it sees ahead (`multi_vj`, 80%), but
seeing ahead leaks the future: no robot watching a video live has it. The leak-free latent (`multi_vjc`, 75%) sees only
the past. Can V-JEPA 2, used as a *world model*, close that gap: predict the latent 0.5 s and 1 s ahead from the video
seen so far, and give the policy that?

## Method

Code: `latent/world_model.py`, `experiments/worldmodel_policy.py`, `experiments/worldmodel_remote.py` (the Modal
launcher), the task `A2H-TrackingVJPred-Flat-Unitree-G1` (`mjlab_tasks.py`, observation `video_latent_predicted`).

**Imagining the future with V-JEPA 2's predictor.** `facebook/vjepa2-vitl-fpc64-256` ships the predictor network it
was pretrained with: given the encoder's tokens of the visible part of a clip and the positions of the hidden part, it
predicts the hidden part's tokens (in the encoder's layer-normed token space). Hugging Face exposes it as
`VJEPA2Model.predictor(encoder_hidden_states, context_mask, target_mask)`. For each V-JEPA step *s* (2 frames; step *s*
ends at frame 2*s*):

1. The **encoder sees only the 32 frames up to frame 2*s*** (~1 s, the first frame repeated before the start): 16 time
   steps × 256 patches.
2. The **predictor** gets those tokens as context and the positions of the next 15 time steps (frames 2*s* + 1 … 2*s*
   + 30, 1 s) as targets, and predicts their tokens. The predictor is called directly with the context tokens, so the
   encoder never sees the future frames, not even blanked (the blanking the goal-relevance diagnostic used would put
   empty frames in the encoder's attention).
3. **The latent *h* steps ahead** is the trailing window that would end at frame 2(*s* + *h*): its first 16 − *h* time
   steps are the last ones the encoder saw, the other *h* imagined, averaged over space and time
   (`predicted_trailing`). *h* = 8 (0.53 s) and 15 (1 s).

**A learned latent predictor, for comparison and as the fallback.** `LatentPredictor`: ridge regression from the
causal latent now, ~0.27 s and ~0.53 s ago to its change 0.53 s and 1 s ahead, fitted on the five training clips'
latents (both mirror images), the ridge strength picked by leaving one clip out. It is cross-fitted: a training clip's
predictions come from a fit on the other four, so the policy trains on predictions as imperfect as the ones it gets on a
new clip.

**No future leakage, checked by tests** (`tests/test_world_model.py`):
- `imagine`'s outputs at step *s* are unchanged when every frame after 2*s* is replaced (a fake encoder and a
  predictor that mixes all its context).
- On a small random V-JEPA 2, the predictor's output doesn't change when the target slots of its input are filled with
  noise instead of zeros: it reads only the context.
- The learned predictor's output at step *s* is unchanged when the latents after *s* are.
- The policy's observation at motion step *t* reads no row of the latent table after *t* (the predictions it reads are
  the ones made at *t*).
- Laid out per motion step, a re-timed variant reads the latest step whose window ends at or before the video frame it
  shows (`video_condition.at_frames_causal`, already tested).

**The policies.** All five are trained on the same 165 variants of squat jump, front kick, squat, arms and lunge, from
the same warm start (the squat jump's variant-trained tracker), for 2,000 iterations with 4,096 environments; only their
extra input differs:

| Policy | Extra input | Leaks the future? |
|---|---|---|
| `multi` | none | no |
| `multi_vjc` | the causal latent now and 0.5 s ago (2 × 32) | no |
| **`multi_vjp`** | `multi_vjc`'s + V-JEPA 2's imagined latent 0.53 s and 1 s ahead (+ 2 × 32) | **no** |
| **`multi_vjl`** | `multi_vjc`'s + the learned predictor's latent 0.53 s and 1 s ahead (+ 2 × 32) | **no** |
| `multi_vj` | the centred latent now, 0.5 s and 1 s ahead (3 × 32) | yes, ~1.5 s (upper bound) |

`multi`, `multi_vjc` and `multi_vj` are the earlier runs' checkpoints (`outputs/mc_adapt_inputs`,
`outputs/remote_multiclip_vjc1`), re-evaluated here next to the new ones. The imagined latents are in the PCA (32
components) of the layer-normed trailing latents, fitted on the training clips; the causal latent keeps `multi_vjc`'s
PCA, so `multi_vjp`'s input is exactly `multi_vjc`'s plus the prediction.

**The test** is `experiments/robustness_eval.py` as in `docs/VJEPA_CONDITIONING.md`, 64 rollouts per condition: each
clip itself, its held-out variants, 0.5x / 2x tempo, 1 m/s pushes. Trained clips: the five. Zero-shot: jumping jacks,
punches, discus. Each policy reads its own latent table, and an eval run holds one, so each clip is evaluated in four
runs that share the motions, the held-out variants, the conditions and their seeds: {`multi_vjp`, `multi`},
{`multi_vjl`}, {`multi_vjc`}, {`multi_vj`}.

## Is the imagined future any good? (`outputs/remote_wm_lat1`, ~5 min on an L40S)

`wm_quality.md`. The imagined latent *h* steps ahead against the true trailing latent at *s* + *h* (from the real
video), in the PCA of the layer-normed trailing latents (32 components, 98% of the variance), centred on the training
clips' mean. The baseline is **copying the present**: the trailing latent at *s*. Error is squared error relative to the
copy's (below 1 beats it). Change is the cosine between the predicted change from now and the true one (0 means it
doesn't know which way the motion goes).

| Clips | Ahead | Copy the present: cos | V-JEPA 2 predictor: cos / error / change | Learned ridge: cos / error / change |
|---|---|---|---|---|
| trained (cross-fitted) | 0.53 s | 0.74 | 0.70 / 2.04 / 0.06 | 0.74 / 1.15 / 0.12 |
| trained (cross-fitted) | 1.00 s | 0.50 | 0.47 / 2.80 / −0.04 | 0.50 / 1.19 / 0.15 |
| zero-shot | 0.53 s | 0.84 | 0.68 / 3.12 / −0.06 | 0.84 / 0.94 / 0.18 |
| zero-shot | 1.00 s | 0.75 | 0.48 / 5.13 / −0.10 | 0.76 / 0.92 / 0.22 |

Per clip, the learned predictor's error is 0.88–0.98 of the copy's on seven of the eight clips. The exception is arms
(1.8–2.1): a 1.7 s clip, extrapolated by a fit on the other four. That exception is what lifts its trained-clip average
above 1.

**V-JEPA 2's predictor can't imagine the future of these videos: it is worse than copying the present.** Its error is
2–6 times the copy's, and its predicted change is uncorrelated with the real one (−0.15 to 0.10).
- **It is offset and blurred.** Per time step, its predicted tokens are a near-constant offset away from the real ones:
  the bias has norm ≈ 14, while the real tokens vary by ≈ 5–6 around their mean. Even with the offset removed, they agree
  with the real future less than the present does (cosine 0.08–0.31 vs 0.37–0.75 on the full-body clips).
- **What signal there is, is small.** A ridge readout fitted on the training clips, from [the present, V-JEPA's
  prediction] to the future, beats one from the present alone by only 1–4 points of relative error on the zero-shot
  clips.
- **Why:** it is the self-supervised predictor V-JEPA 2 was pretrained with. It learned to fill in masked tubes that
  span the clip's whole duration, so every target always had visible context at the same time step. Predicting whole
  future time steps with no spatial context at all is outside what it learned. `docs/GOAL_RELEVANCE.md` found the same
  thing from the other side: it falls back on a near-mean prediction that barely uses the context. Meta's
  action-conditioned V-JEPA 2-AC, trained to predict future frames, is the model built for this. Its checkpoint is a
  separate download and is not tried here.

**The learned latent world model is only slightly better than copying.** It knows which way the latent will move
(change cosine 0.10–0.33), but the move it predicts is small. From five clips, it has mostly learned that motions drift
back towards the average pose.

So the brief's fallback applies: train on the learned predictor's future. The V-JEPA-predicted policy is trained too,
since it costs nothing extra in the same job (both train at once): a check that an input no better than noise neither
helps nor hurts.

## Policies with the imagined future (`outputs/remote_wm_vjp1`, `outputs/remote_multiclip_wm_eval1`)

`multi_vjp` and `multi_vjl` trained together on a Modal L40S (`remote_wm_vjp1`). The Modal workspace hit its spend
limit two clips from the end of the first eval group, so the remaining 26 of the 32 clip evals ran on a Colab A100
from the same build, the same checkpoints and the same PCA (`worldmodel_remote.py --backend colab --resume`). The PCA
had to be carried over: a refit flips one component's sign on another machine. `merged/multiclip_summary.md`. Task
success, fall rate in brackets, 64 rollouts per condition:

| Policy | Leaks? | Trained clips: the clip | Held-out variants | Tempo 0.5x / 2x | Pushed 1 m/s | Zero-shot clips: the clip | Held-out variants |
|---|---|---|---|---|---|---|---|
| `multi` | no | 44% (16%) | 38% (17%) | 22% (18%) | 40% (19%) | 0% (67%) | 2% (68%) |
| `multi_vjc` (past only) | no | **75% (0%)** | **61% (4%)** | 43% (7%) | **61% (5%)** | 0% (69%) | 0% (84%) |
| **`multi_vjp`** (+ V-JEPA 2's imagined future) | no | 66% (0%) | **61% (4%)** | 43% (16%) | 57% (4%) | 0% (67%) | 4% (66%) |
| **`multi_vjl`** (+ the learned predictor's) | no | 69% (20%) | 58% (18%) | 43% (28%) | 59% (24%) | 0% (97%) | 0% (91%) |
| `multi_vj` (true future) | yes | **78% (0%)** | 60% (3%) | 38% (4%) | 53% (3%) | 0% (66%) | 0% (67%) |

Per clip (the clip / its held-out variants):

| Clip | `multi` | `multi_vjc` | `multi_vjp` | `multi_vjl` | `multi_vj` |
|---|---|---|---|---|---|
| squat jump | 100% / 77% | 100% / 81% | 100% / 75% | 97% / 74% | 100% / 73% |
| front kick | 100% / 56% | 75% / 43% | 100% / 72% | 100% / 68% | 100% / 64% |
| squat | 0% / 29% | 100% / 84% | 28% / 57% | 47% / 57% | 92% / 62% |
| arms | 20% / 27% | 100% / 97% | 100% / 99% | 100% / 94% | 100% / 100% |
| lunge | 0% / 0% | 0% / 0% | 0% / 0% | 0% (100% falls) / 0% | 0% / 0% |
| jumping jacks (zero-shot) | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% |
| punches (zero-shot) | 0% / 0% | 0% / 0% | 0% / 11% | 0% (92% falls) / 0% | 0% / 0% |
| discus (zero-shot) | 0% / 5% | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 1% |

**Neither imagined future closes the gap; neither beats seeing only the past.**
- **`multi_vjp` (V-JEPA 2's predictor)** ties `multi_vjc` on the held-out variants (61%) but is 9 points lower on
  the clips themselves (66% vs 75%): it loses the squat (28% vs 100%), partly made up on the front kick (100% / 72%
  vs 75% / 43%). With predictions no better than copying the present, the extra 64 inputs mostly add noise
  the policy has to learn to ignore. It also falls more under tempo changes (16% vs 7%).
- **`multi_vjl` (the learned predictor)** tracks about as well as `multi_vjp` where it stays up, but it falls far more:
  20% on the trained clips, 91–97% on the zero-shot ones. The extra falls are on the lunge (100%) and punches (92%),
  where every other policy stays up while missing the task. A ridge model fitted on a handful of clips makes
  off-distribution inputs where it extrapolates, and the policy trusts them. A
  predictor fitted on five clips makes off-distribution inputs on a new clip, and the policy trusts them.
- **The leaky upper bound is barely above the causal latent here** (78% vs 75% on the clips; 60% vs 61% on the
  variants). Both are the earlier runs' checkpoints, so this is a re-evaluation, not new training. Most of the "seeing
  ahead" advantage the brief started from is within noise between runs. So the headroom a world model could recover on
  this benchmark is small to begin with.
- **Zero-shot clips stay at ~0% for every policy.** None of the latents, past, imagined or true future, makes five
  training clips enough for a new motion. That is the same finding as the other multi-clip runs.

## Cost

| Run | Where | Time | Cost |
|---|---|---|---|
| Smoke runs (two killed: missing inputs, then a spend limit) | Modal L40S, first account | ~15 min | ~$0.6 |
| `wm_lat1`: V-JEPA 2 latents and predictions, quality report | Modal L40S | ~5 min | ~$0.2 |
| `wm_vjp1`: build, train both policies, 6 of 32 evals | Modal L40S, second account | ~1 h 40 min | ~$3.3 |
| `wm_eval1`: the other 26 evals | Colab A100 (one stalled launch, one refused upload, then the run) | ~1 h 25 min | ~9.5 compute units |

## Conclusion

V-JEPA 2's pretrained predictor can't serve as a world model for this task. Given the past ~1 s of video, its
imagined latent 0.5 s and 1 s ahead is further from the truth than copying the present (2–5 times the error), and its
predicted change is uncorrelated with the real one. A small learned latent predictor is only slightly better than
copying. Fed to the policy, neither prediction helps: `multi_vjp` ties the past-only `multi_vjc` on held-out variants
and is lower on the clips; `multi_vjl` falls far more on clips where its predictor extrapolates. `multi_vjc` remains
the best leak-free policy. The leaky `multi_vj`'s margin over it is ~3 points, so there is little to recover this way
on five clips. A world model worth feeding the policy needs to be trained to predict forward in time: V-JEPA 2-AC, or
a predictor trained on many more motions (below).

## Not attempted: rewarding the robot's own imagined future

The stretch idea is a policy rewarded when the latent *it* would produce next, as V-JEPA imagines it from a render of
the robot, matches the video's. It needs a predictor whose imagined future beats copying the present, and neither here
does. The reward would score the robot against a guess that is no better than "keep doing what you're doing".

What would make it, and the input above, worth trying:
- **V-JEPA 2-AC**, Meta's action-conditioned predictor, trained to roll the latent forward in time. It is a different
  checkpoint (a new download), and robot-arm actions are what it is conditioned on, so it would need fine-tuning on the G1.
- **A learned predictor on many motions.** The distillation set (`latent/distill.py`, 536 robot motions with V-JEPA
  embeddings) is about 100 times more data than the five clips here.
- **Predicting in the robot's own space.** `multi_future`'s 80% shows the retarget ahead is what helps. A small model that
  predicts the retarget's next joints from the past video latent is the same world model, with a target that's easier
  to learn.
