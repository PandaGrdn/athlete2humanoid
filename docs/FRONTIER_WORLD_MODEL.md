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
