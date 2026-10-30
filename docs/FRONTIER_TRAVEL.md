# Motions that travel (2026-10-29)

**Question:** why does the tracking policy stay roughly in place on moves that cross the floor, and which change
makes it travel? Two clips fail this way:
- **Shot put** (`docs/PRO_CLIPS.md`): after the retarget fixes the robot no longer falls and tracks the joints at
  0.18 rad, but it doesn't do the 1.1 m sideways glide. Torso error is 0.4–0.5 m, it never reaches the first key
  moment ("step left, turn left") and task success is 0%.
- **Lunge:** 0% for every policy in `docs/VJEPA_CONDITIONING.md` and `docs/GENERALIZATION.md`, and only 15 of its
  36 training variants survive the filters.

## Diagnosis

Read before changing anything: mjlab's tracking task (`mjlab/tasks/tracking/tracking_env_cfg.py`, `mdp/`), the
baseline shot put run (`outputs/remote_multiclip_pro_shot_put2`, policy `multi`) and the variant filter.

**1. The root's reward is too weak where it matters.** mjlab's anchor (torso) position term is
exp(−e²/0.3²) at weight 0.5:

| Torso error | 0.1 m | 0.3 m | 0.5 m | 1.0 m |
|---|---|---|---|---|
| mjlab's term (std 0.3) | 0.89 | 0.37 | 0.06 | 0.00002 |
| its pull, weight × d/de | 1.0 | 1.2 | 0.34 | ~0 |
| std 1.0, weight 1 (added) | 0.99 | 0.91 | 0.78 | 0.37 |
| its pull | 0.2 | 0.5 | 0.78 | 0.74 |

- Once the robot is half a metre behind, catching up pays almost nothing. Nothing else measures where the robot is: body positions are scored relative to the robot's own torso.
- No term scores the root's velocity. `motion_body_lin_vel` averages 14 bodies' velocity errors, swinging limbs included.

**2. Staying put is never a failure.** mjlab's anchor termination is height-only (`bad_anchor_pos_z_only`), as is the end-effector one.
- So a robot that doesn't travel is never reset.
- mjlab's adaptive sampling sends episodes to where they fail, so it never sends them to the travelling phase.
- In the baseline, 56% of episode starts ended in a single 1 s bin (`sampling_top1_prob`), where `ee_body_pos` terminations were.
- The torso error grew from 0.10 to 0.35 m over training as episodes lengthened from 9 to 357 steps: drift accumulated, and nothing corrected it.

**3. The spec makes travel a hard requirement.** The shot put's first key moment asks for 0.92 m of lateral root travel within 0.28 m, held for 0.4 s, by 2.0 s. The baseline arrives 0.4 m short.

**4. The lunge filter wrongly drops travelling variants.** Of the 21 variants it drops, 12 really do exceed the
joint limits (amplitude 1.1). The other 9 fail only their own reference's self-check:
- The compiler sets a key pose's hold from how long the reference dwells in it, measuring the root goal from the previous key frame.
- The machine measures the root goal from where the previous stage was first satisfied, often frames earlier.
- On stages that move the root ("step forward", "crouch, step right"), the reference then dwells less than the hold asks. The stage completes only when a later repetition passes the same pose, the machine falls behind the clip, and the variant fails.
- `keypose.fit_holds` halves holds only at the stage the self-check stops at, which by then is a later one.
- So the cause isn't scaled travel making the planted foot slide, as `docs/GENERALIZATION.md` guessed. The lunge at 1.33x tempo and amplitude 1.0 already fails.

## Changes

Each is a new registered task or an opt-in variant set. No existing task's defaults change.

| Change | Where | What |
|---|---|---|
| **reward** | `A2H-TravelReward-Flat-Unitree-G1` | + torso position exp(std 1 m), weight 1; + torso horizontal velocity exp(std 0.7 m/s), weight 1. mjlab's terms unchanged |
| **term** | `A2H-TravelTerm-Flat-Unitree-G1` | + a horizontal torso termination; threshold curriculum 1.0 → 0.4 m over 1,000 iterations. It also steers adaptive sampling to where the robot falls behind |
| **phase** | `A2H-TravelPhase-Flat-Unitree-G1` | half the episode starts drawn by how far the reference's root goes in the next second, beyond 0.2 m (net, so sway cancels); the rest by mjlab's adaptive sampling. A no-op for in-place clips, none of which goes 0.2 m in a second |
| all | `A2H-Travel-Flat-Unitree-G1` | the three together |
| **synced** | `skill/travel_variants.py` (`name=TASK@synced` in the runner) | variants dropped only by a failed self-check get their holds synced: the first stage completed late has its hold halved until the reference passes. Goals, tolerances and the joint-limit filter are unchanged. Lunge keeps 24 of 36 instead of 15; shot put, squat jump and jumping jacks are unchanged |

- **Code:** `travel_tasks.py`, registered at the end of `mjlab_tasks.py`.
- **Settings:** the settings live in `travel_tasks.TRAVEL`. `A2H_TRAVEL_CONFIG=travel.json` overrides any of them.
- **Observations and evaluation:** all the tasks have mjlab's observations. A tracking checkpoint warm-starts them unchanged, and they are evaluated in the stock tracking env.

## Setup

`experiments/travel_policy.py`:
- **Steps:**
  - `prepare`: the clips re-retargeted from `human.npy` with the current `retarget.py`, plus the checkpoints.
  - `build`: the motion files, made on a laptop. mjlab's CSV → npz converter takes about 10 s a motion on a Mac CPU and about 2 min on a Modal job, where the GPU sits idle meanwhile.
  - `launch`: one GPU job per clip.
  - `report`: the tables below.
- **Training:** every policy starts from the 5-clip generalist `multi` (`outputs/mc_adapt_inputs/policy_multi/model_1999.pt`) and trains 1,500 iterations on the clip's training variants, 4,096 envs. All of a clip's policies train at once on one GPU.
- **Baseline:** `base` is plain tracking (`A2H-Tracking`), retrained in the same job. `base_pro` is the earlier run's `multi` (`remote_multiclip_pro_shot_put2`), the same recipe, so the two `base` rows show seed noise.
- **Eval:** `experiments/robustness_eval.py` in one env, 64 rollouts per condition:
  - the clip;
  - its held-out variants (always the stock filter's, so every policy is tested on the same set);
  - 1 m/s pushes;
  - 0.5x and 2x tempo;
  - start noise.

  It also reports the share of the reference's net horizontal travel the torso achieved, and its distance from the reference's end point, from 16 clean rollouts per policy.

## Shot put: each change alone

Run `outputs/remote_travel_c3_shot_put` (`docs/results/travel/travel.md`). Each cell is task success with the fall rate in brackets, 64 rollouts per condition. The held-out cell is the mean over its 5 variants. The last three columns are on the clean clip.

| Policy | The clip | Held-out variants | Pushed 1 m/s | Tempo 0.5x / 2x | Start noise | Torso error | Glide done | End error | Joint error |
|---|---|---|---|---|---|---|---|---|---|
| base_pro (earlier run) | 0% (0%) | 16% (5%) | 8% (3%) | 32% (33%) | 66% (8%) | 0.38 m | 63% | 0.45 m | 0.189 |
| base (tracking) | 8% (0%) | 7% (0%) | 23% (0%) | 14% (4%) | 77% (0%) | 0.31 m | 73% | 0.42 m | 0.184 |
| reward | 77% (0%) | 40% (0%) | 64% (0%) | 50% (34%) | 88% (0%) | 0.14 m | 95% | 0.09 m | 0.210 |
| term | **100%** (0%) | 40% (0%) | 73% (5%) | 48% (52%) | 83% (11%) | 0.12 m | 93% | 0.08 m | 0.201 |
| phase | 81% (0%) | 40% (0%) | 53% (0%) | 50% (4%) | 80% (5%) | 0.23 m | 97% | 0.23 m | 0.194 |
| **all** | **100%** (0%) | 40% (0%) | **94%** (2%) | 50% (2%) | **89%** (3%) | **0.09 m** | 96% | **0.05 m** | 0.216 |

- **Each change alone turns the shot put from a fail into a pass.** Success goes from 0–8% to 77–100%. The torso now follows the glide: 93–97% of the reference's 1.12 m, against 63–73% before.
- **The termination curriculum matters most alone (100%).** It does two things at once: it ends episodes where the robot falls behind, and it steers adaptive sampling there.
- **The two reward terms close the gap best at the end of the clip.** The robot ends 0.09 m from where the reference ends. With phase-balanced starts it gets the travel but ends 0.23 m off: it is taught the glide more often, not where to finish.
- **Together they are best on every robustness suite:**
  - pushes: 94% vs 23–73%;
  - start noise: 89%;
  - falls at 2x tempo: 2–3%, where the term-only and reward-only policies fall 34–52% of the time.

  The combination removes the 2x-tempo falls that each of those two causes alone. Why isn't measured here.
- **Held-out variants stay at 40% for every travel policy.** 2 of the 5 pass at 100% (0.5x, warp 0.5). The other 3 fail for every policy, baselines included:
  - amplitude 0.7;
  - mirrored at 0.6x tempo and amplitude 0.8;
  - 2x tempo.

  All three are outside the training range.
- **The cost:** joint error 0.216 vs 0.184 rad. The robot trades a little pose accuracy for being in the right place.
- **Seed noise:** the two tracking baselines (same recipe, separate runs) differ by 8 points on the clip and 0.07 m in torso error. Every travel row is far outside that.

## Lunge: the filter fix alone, and with all three changes

Run `outputs/remote_travel_c3_lunge`. Rows marked @synced train on the 24 variants `skill/travel_variants.py` keeps; the others train on the 15 stock variants.

| Policy | The clip | Held-out variants | Pushed 1 m/s | Tempo 0.5x / 2x | Start noise | Stages done | Torso error | Joint error |
|---|---|---|---|---|---|---|---|---|
| base | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 13% | 0.11 m | 0.155 |
| synced | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 14% | 0.12 m | 0.170 |
| all | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (5%) | 12% | 0.10 m | 0.178 |
| all_synced | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (3%) | 12% | 0.09 m | 0.199 |

**Nothing here moves the lunge.** None of the changes is wrong; the lunge's blocker isn't travel:
- The torso already follows the reference to 0.10 m, and the lunge's net travel is only 0.09 m.
- Every policy completes 2 key moments and stalls at the first required one, "step forward, step right" (1.8 s).
- That stage asks the pelvis to move 0.32 m forward, within 0.15 m, from where the previous stage ended. That is a 0.26 m shift onto the front leg in about 0.3 s.
- The reference reaches 0.26 m and passes on margin. Every policy gets 0.08 m at best and stays in that stage for the rest of the clip, so the 12 key moments after it are never scored.

**The open question** is whether the G1 can hold that weight shift and depth, or whether the spec's root goal is too strict for a lunge. Neither is a tracking-reward problem.

The filter fix is still right: it keeps 9 more lunge variants, all correct. It just doesn't matter while every policy stalls before the first required stage.

## Regression check: squat jump (in place)

Run `outputs/remote_travel_c3_squat_jump`: the same warm start and recipe, plain tracking against all three changes.
- **Known numbers:** `docs/GENERALIZATION.md`'s `tracking_var` scores 100% on the clip, 78% (0%) on its held-out variants and 0.11 rad joint error.
- **Phase sampling:** it is a no-op on this clip, since the squat jump never travels 0.2 m in a second.

| Policy | The clip | Held-out variants | Pushed 1 m/s | Tempo 0.5x / 2x | Start noise | Torso error | Joint error |
|---|---|---|---|---|---|---|---|
| tracking_var (`GENERALIZATION.md`) | 100% | 78% (0%) | | | | | 0.11 |
| base (tracking) | 100% (0%) | 77% (0%) | 67% (9%) | 50% (0%) | 92% (0%) | 0.11 m | 0.112 |
| all | 100% (0%) | **93%** (0%) | 47% (3%) | 50% (0%) | 94% (0%) | 0.06 m | 0.144 |

- **The clip is unhurt, and the held-out variants improve.** 93% vs 77–78%. The gains are the two hardest variants:
  - 2x tempo: 77% vs 19%;
  - mirrored 1.6x at amplitude 1.2: 77% vs 23%.
- **The cost is under pushes:** 47% vs 67% success at 1 m/s, though with fewer falls (3% vs 9%).
  - Likely (not measured): a shoved robot pulled back toward the reference's position keeps its feet but misses key moments.
  - Joint error rises from 0.11 to 0.14 rad.
- **So:** use `A2H-Travel` for clips that travel. For in-place clips it is a trade, not a free win: better held-out variants, worse pushes.

## Not run: the zero-shot set (jumping jacks, punches, discus)

Their motion files are built (`experiments/travel_policy.py build`), but Colab returned "Service Unavailable" for an A100 on three attempts over 15 minutes, and both Modal workspaces were at their spend limits.
- **Discus** is the one that matters here: it travels 0.99 m. It is the next run, `base` against `all`, as for the squat jump.
- **Jumping jacks and punches** stay in place, so phase sampling is a no-op on them, and the squat jump result above is the expected pattern.

## Cost

Both Modal workspaces hit their spend limits during this work: the first mid-run, the second within minutes. Everything that trained ran on Colab A100s, one at a time, because Colab refuses a second A100.

| Run | Where | Time | Cost |
|---|---|---|---|
| Two smoke runs, and jobs killed by the spend limits before training | Modal L40S | about 50 min | about $1.6 |
| Shot put: 5 policies + eval | Colab A100 | 2.8 h | about 33 CU |
| Lunge: 4 policies + eval | Colab A100 | 2.6 h | about 30 CU |
| Squat jump: 2 policies + eval | Colab A100 | 1.4 h | about 17 CU |
| Motion files for all 6 clips (`build`) | laptop CPU | about 25 min | 0 |

- **Total:** about $1.6 on Modal and about 80 Colab compute units (about $8), against the $15 budget.
- **Speed:** with 5 policies on one A100, training ran at about 5 s per iteration; with 2, about 2.3 s.

## Conclusion

- **The policy stayed in place because nothing in mjlab's tracking task made it move.** The root position term saturates by half a metre, nothing scores the root's velocity, and the anchor termination is height-only. Adaptive sampling, driven by failures, therefore never visits the travel phase.
- **Each fix alone makes the shot put glide.** Together (`A2H-Travel-Flat-Unitree-G1`), the shot put goes from 0–8% to 100% task success: 0.09 m torso error, 96% of the 1.1 m glide, 94% under 1 m/s pushes.
- **The lunge isn't a travel problem.** It stalls at a required 0.32 m forward weight shift that no policy makes, travel task or not. That is the next thing to look at, in the spec or in what the G1 can do.
- **The lunge filter was dropping 9 valid variants.** Those were false rejections from a hold/anchor mismatch on root-moving stages. `skill/travel_variants.py` keeps them without changing any spec's goals. Since fix 2 of `docs/PIPELINE_ACCURACY.md` (tempo-relative holds) is on main, the stock filter keeps the same 24 lunge variants, so `travel_variants.py` is a no-op fallback there.
- **On an in-place clip (squat jump), `A2H-Travel`:**
  - doesn't hurt the clip;
  - raises held-out variants from 77% to 93%;
  - costs 20 points of push success and 0.03 rad of joint error.
- **Default for new clips:** `A2H-Travel` when the clip's root moves more than about 0.2 m in a second (`travel_tasks.travel_weights` > 0), plain tracking otherwise.
