# Any action as an RL task: the keypose spec

The first task spec (`skill/task.py`, [SEMANTIC_REWARD.md](SEMANTIC_REWARD.md)) knows four stage kinds: stand, squat, jump, land. A kick, a punch or a lunge compiles to little more than "stand". `skill/keypose.py` replaces those kinds with **key moments**, picked by a statistical rule from the retarget, for any action. It is the default in `experiments/latent_policy.py`; `--spec stages` keeps the old spec, which the commanded-target task (`A2H-TaskCmd`) still needs.

```bash
python -m skill.keypose outputs/remote_latent_final/clips/front_kick    # -> task_spec.json + self-check
```

## How a clip becomes a task

1. **Features.** The pose as 19 numbers in metres, invariant to where the person faces. The first 16 depend on the pose only:
   - the pelvis height;
   - the torso and both wrists relative to the pelvis, in its own heading frame;
   - both ankles' horizontal offsets from the pelvis;
   - both ankle heights above the floor.

   The last 3 are the root's forward and sideways motion and its turn since the last key moment. A turn counts as 0.3 m per radian.
2. **Key moments: a hierarchical simplification.** Ramer–Douglas–Peucker on the feature trajectory, under a mass-weighted metric.
   - A key moment is a frame where the motion stops being explained by a straight line between its neighbours: the bottom of a squat, a kick's apex, a punch's extension, the return to standing.
   - The tolerance comes from the clip itself. It is the largest of three:
     - six sigmas of its pose noise (median absolute deviation from a 0.2 s moving average);
     - 30% of its largest excursion, so a punch and a squat jump are each judged at their own scale;
     - a 1 cm floor.
   - It coarsens automatically if the clip would need more than 16 stages.
   - A moment must also be a real change from the last one kept, so a foot shuffle while standing is dropped.
3. **Required or optional: its level in the hierarchy.**
   - Each moment's level is its deviation from the path between its neighbours, in tolerances. Foot placement is left out, since a step shows in the root's motion and a kick in the foot's height.
   - **A loose task.** Required:
     - the end goal (the clip's last pose);
     - the contacts (each jump and its landing);
     - the 2 key moments at the top of the hierarchy (`REQUIRED_TOP`).

     Every other moment is how this person happened to do it (an in-between crouch, an arm swing). It pays a bonus but may be skipped. Squat jump: 3 required poses and 2 jumps out of 12 stages. Lunge: 3 of 16.
4. **Goal relevance per feature.**
   - What counts is each feature's motion into that moment, beyond what its noise or its minimum tolerance would accept anyway, weighted by the square root of the mass it moves.
   - A kick's goal is the kicking foot; arms swinging through a squat get a loose tolerance or none.
   - Features below 15% of the most relevant one are free. The core ones (at least 50%) must each be within tolerance; the rest only on average.
   - Going further than the key moment along its own motion is free, up to half its excursion.
5. **Contacts.**
   - Flights (both feet off the floor for at least 0.15 s) are their own stages, with skill/task.py's jump-quality band. Each is followed by a landing: feet down, upright. The 0.5 s after touchdown belong to the landing, not to key moments.
   - Every pose also requires the feet that were on the floor in the reference to be on the floor.
6. **Names.**
   - A deterministic description of what changed most: "right foot up, right foot forward", "crouch", "left hand right".
   - It stands in for Meta's Muse, whose skill reader labels the stages by time overlap (`skill.task.attach_skill`) once the API key exists.

The reward machine (`KeyposeMachine`) has `StageMachine`'s interface and rules, so the mjlab task envs, warm starts and `evaluate()` take either spec:
- potential-based progress toward the current goal, gated on staying upright;
- a bonus per stage, plus jump quality and a finish bonus;
- a fall refunds the attempt;
- the stage is part of the policy's observation, the same 18 numbers as before.

**Once done, the end goal is a state to keep.** Holding the end pose pays 1 per second, and a fall gives that back. The style terms stop: the clip is no longer imitated once the task is complete, so finishing early forfeits the style reward still ahead.

**The robot chooses its own timing:** only the order of the moments counts. The style terms (V-JEPA 2) are time-weighted toward the approach to each key moment and each landing (`keypose.importance`).

## Checks on the seven clips

Each clip's spec scored on every clip's retarget, and on standing still at the clip's first pose (stages passed / total):

| Spec \ motion | squat jump | squat | front kick | jumping jacks | lunge | punches | arms | stand |
|---|---|---|---|---|---|---|---|---|
| squat jump | **12/12** | 2/12 | 1/12 | 1/12 | 2/12 | 1/12 | 1/12 | 1/12 |
| squat | 9/9 | **9/9** | 1/9 | 1/9 | 1/9 | 1/9 | 1/9 | 1/9 |
| front kick | 1/6 | 1/6 | **6/6** | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |
| jumping jacks | 1/13 | 1/13 | 1/13 | **13/13** | 1/13 | 1/13 | 3/13 | 1/13 |
| lunge | 3/16 | 3/16 | 1/16 | 2/16 | **16/16** | 2/16 | 1/16 | 1/16 |
| punches | 4/7 | 6/7 | 1/7 | 3/7 | 6/7 | **7/7** | 2/7 | 1/7 |
| arms | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 1/3 |

- **Every retarget passes its own spec,** and standing still fails every spec.
- **The specs tell the actions apart.** The exceptions: a squat jump contains the squat spec (it does squat). The arms clip is weak, since its only key moment is "hands down".
- **The punches spec is the loosest** (half its moments are optional), so the squat and lunge clips get far into it.
- **Against the squat jump's keypose spec**, the final run's policies, trained on the old spec, score:
  - tracking: 12/12;
  - task and taskphase: 6/12. They jump and land, but skip an arm swing the person did between reps, which is required at this level.

  A policy trained on the keypose spec learns it; the training runs below test that.

`tests/test_keypose.py` covers the same properties on synthetic squats, kicks, punches and jumps: every action compiles and passes its own spec, standing fails, actions tell apart, the kick's goal is the kicking foot, timing and heading are free, flights are stages, optional moments can be skipped, a fall refunds, and noise alone gives no required moment.

## Training runs (2026-10-27)

All warm-started from the final squat-jump weights (same observation size):

| Run | Clip | Policies | Iterations |
|---|---|---|---|
| `outputs/remote_latent_kp_kick` | front kick | tracking, task, taskphase | 1,500 |
| `outputs/remote_latent_kp_sj` | squat jump | task, taskphase | 1,500 |
| `outputs/remote_latent_kp_jacks` | jumping jacks | task, taskphase | 1,500 |

The squat-jump run is the regression check against the old spec's results.

**Squat jump** (each policy scored on its keypose spec; one 9 s rollout from the first frame):

| Policy | Stages | Jump quality | Falls | Torso error (mean / max) |
|---|---|---|---|---|
| retarget | 12/12 | 1.0, 1.0 | – | – |
| final run's task (old spec) | 6/12 | 0.68 | 0 | – |
| **keypose task** | **12/12** | **1.0, 1.0** | **0** | 0.18 / 0.42 m |
| keypose taskphase | 0/12 → 12/12* | 0.83, 0.83* | 9 → 2* | 0.62 / 2.38 m* |

- **The task policy learns the keypose spec:** 6/12 → 12/12 in 1,500 iterations, including the arm swing the old spec never asked for. Kept as `weights/kp_squat_jump_task`.
- **Taskphase** (* re-scored after the fix below, same checkpoint) now completes the spec, but falls twice on the way. Its later stages are partly credited to the reset after a fall, so it is not a clean pass. Its training episodes were still getting longer at 1,500 iterations, so it needs more.

**Front kick**, an action no earlier run trained on. Its spec has 4 key moments, two kicks with the right foot:

| Policy | Stages | Falls | Torso error (mean / max) |
|---|---|---|---|
| retarget | 6/6 | – | – |
| final run's squat-jump policies | 1/6 | 0 | – |
| **keypose tracking** | **6/6** | **0** | 0.12 / 0.21 m |
| **keypose task** | **6/6** | **0** | 0.85 / 1.87 m |
| **keypose taskphase** | **6/6** | **0** | 0.49 / 1.33 m |

- **All three learn the kick from the clip alone,** with no kick-specific reward. The warm start knew only squat jumps (1/6).
- **The task policies kick, but not where the person did.** Position and timing are free, so they drift up to 1.9 m from the reference path. Tracking keeps to it.
- Kept as `weights/kp_front_kick_{tracking,task,taskphase}`.

**Jumping jacks.** Its spec has 10 key moments: arms up and down, a crouch, then a jump and its landing.

| Policy | Stages | Jump quality | Falls | Torso error (mean / max) |
|---|---|---|---|---|
| retarget | 13/13 | 1.0 | – | – |
| final run's squat-jump policies | 1/13 | – | 0 | – |
| **keypose task** | **13/13** | **1.0** | **0** | 0.85 / 1.02 m |
| **keypose taskphase** | 1/13 → **13/13*** | 0.90* | 53 → **0*** | 0.93 / 1.08 m* |

- **The task policy learns jumping jacks from the clip alone.** Kept as `weights/kp_jumping_jacks_task`.
- **Taskphase** (* re-scored after the fix below, same checkpoint) learns jumping jacks too. Kept as `weights/kp_jumping_jacks_taskphase`; its `policy.mp4` is still the pre-fix rollout.

**The frame-0 fault (fixed).** Every keypose spec opens with a "ready" stage that lasts no time (0.0–0.0 s). The stage machine resets at the reference time after one step:
- An episode that starts at frame 1 or later is past the first key moment's start (0.033 s), so it begins at stage 1.
- One that starts at exactly frame 0 begins in "ready", which asks the robot to be still (the jumping-jacks clip's first frame moves at 1.1 m/s).

Training starts at frame 0 about once in 264 episodes, so a policy that doesn't see the retarget (taskphase) never learned "ready". In play, which always starts at frame 0, it fell within 5 steps of every reset; from any other start frame, it never fell. `KeyposeMachine.reset` now starts past a zero-length first stage and counts it as completed, so frame 0 behaves like frame 1. Re-scored on CPU (mjlab's play rollout, scored with `skill.task.evaluate` at 50 Hz); the Colab numbers in the other rows come from the GPU simulator and differ slightly.

## Why the policies fall (2026-10-28)

**Not physical impossibility.** On all three clips, the tracking and task policies complete every key moment without falling. The end pose of the squat jump asks only for the pelvis at 0.78 ± 0.04 m with the feet down, which is standing.

**Partly too little training.** Falls per training iteration flatten out at:

| Clip | Task | Taskphase (blind) |
|---|---|---|
| Squat jump | 0.2 | 1.5 (still falling: 5.0 → 2.5 → 1.6 → 1.5 at iterations 500/1,000/1,250/1,500) |
| Jumping jacks | 0.75 | 1.7 (flat since iteration 750) |
| Front kick | 0.4 | 1.3 (flat since iteration 1,000) |

**Mostly two reward faults, both fixed:**
1. **Frame 0** (above): taskphase fell at every reset of the evaluation.
2. **After the finish.** Every squat-jump taskphase fall comes after it has completed 12/12:
   - Timing is free, and PPO discounts, so it did the 9 s clip in about 3 s.
   - Then the task paid nothing, and the clock-based style terms asked it to look like the clip's middle (crouching, jumping) from the finish pose.
   - It tipped over within 0.3 s, each time.

   Now holding the end pose pays, and the style terms stop once done.

   **Checked:** squat-jump taskphase continued for 1,000 iterations with both fixes (`outputs/remote_latent_loose_sj_tp`, warm-started from `kp_sj`):
   - 12/12 stages with **0 falls** (before: 0/12 with 9 falls; then 12/12 with 2 falls after the frame-0 fix alone).
   - It stopped rushing: torso error 0.29 / 0.48 m mean / max, down from 0.62 / 2.38 m.
   - Training falls dropped to 0.33 per iteration (from 1.5), the task policy's level.

   Kept as `weights/kp_squat_jump_taskphase`.

**What works best so far: the task policy.** It gets the keypose reward, sees the retarget as input, and has weak style terms (V-JEPA 2, plus 0.1× tracking). It completes all three actions without falling, including two it had never trained on, from the same squat-jump warm start. Seeing the retarget keeps it in time with the clip, so it neither rushes nor drifts. The blind taskphase policy is the stronger generalisation claim, since it needs only a clock at run time, but it learns more slowly and falls 2–7× more.

**Lunge, a fourth action** (`outputs/remote_latent_loose_lunge`, the loose spec, task policy, 1,500 iterations from the squat-jump weights):
- It scored 13/16 with **0 falls**, but the reference itself also scored 13/16 under the robot's thresholds. That pointed at the spec, not the policy.
- **Cause:** a foot was required down if it was within `down_ref` (15 cm here, the reference's hover allowance), but the robot is held to `T.DOWN` (5 cm). A lunge's back foot is up on its toes, so stage 13 could never be met.
- **Fix:** only feet planted by the robot's rule are required down; a merely low foot is free.
- Every clip's reference now passes its spec under the robot's thresholds. The cross-action table is unchanged.
- The same lunge rollout scores **16/16** against the corrected spec. Kept as `weights/kp_lunge_task`.

`experiments/latent_policy.py` now trains `task` by default. The web app's Colab job runs `--policies task` with `--init-weights outputs/remote_latent_final` (`A2H_COLAB_POLICIES`, `A2H_COLAB_INIT`).

## The checker was too lenient: tuck-hops and flicked kicks (2026-10-28)

Muse's blind rollout critic (`muse/rollout_critic.py`) said that on squat jump and front kick only the tracking policy really does the action, although every task-family policy had passed the stage checker (12/12, 6/6). Measured on the rollouts, it was right:

| | Person | Tracking | Task, taskphase, taskhybrid |
|---|---|---|---|
| Squat jump: time in the air | 0.53 s per jump | 0.42 s | 0.04–0.14 s |
| Squat jump: highest pelvis | 1.14 m (standing 0.78) | 1.05 m | 0.78 m: never above standing |
| Front kick: kicks above 0.35 m | 2, each ~0.5 s wide | 2, ~0.5 s | task 2 at 0.13–0.16 s; taskphase and taskhybrid 1 |

The policies found two shortcuts the spec allowed:
- **Tuck-hops.** A flight asked for foot clearance and 0.05–0.12 s in the air. Tucking the feet up with the pelvis at standing height met it.
- **Flicks.** A key pose could be passed through in one control step, so a 0.15 s flick from a crouch counted as the kick.

**The fix (`skill/keypose.py`):**
- A flight needs the pelvis to rise `FLIGHT_RISE_FRAC` = 0.5 of the reference's rise (at most 0.25 m) and the feet off the floor for `FLIGHT_AIR_FRAC` = 0.6 of its airtime (at most 0.35 s). Its progress pays half for foot clearance, half for pelvis rise.
- A key pose is held for `HOLD_FRAC` = 0.5 of the time the reference itself stays within tolerance of it (`ref_dwell_s`), at most 0.4 s.

**Rescored under the new rules:**
- squat jump: the reference 12/12, tracking 12/12, every task-family policy 1/12;
- front kick: the reference 6/6, tracking 6/6, every task-family policy 1/6.

Those policies had learned the shortcut, so the robustness numbers in `docs/GENERALIZATION.md` flatter them.

**Retrained on the fixed spec** (`outputs/remote_latent_fix_sj`, `outputs/remote_latent_fix_kick`): 1,500 iterations each, warm-started from the clip's tracking policy.

| Clip | Policy | Checker | Falls | V-JEPA (before) | Measured |
|---|---|---|---|---|---|
| squat jump | task | 12/12 | 0 | 0.47 (0.33) | pelvis to 1.03 m, flights 0.40 / 0.38 s |
| squat jump | taskhybrid | 12/12 | 0 | 0.54 (0.45) | pelvis to 1.02 m, flights 0.40 / 0.40 s |
| squat jump | tracking (reference point) | 12/12 | 0 | 0.70 | pelvis to 1.05 m, flights 0.42 / 0.42 s |
| front kick | task | 6/6 | 0 | 0.50 (0.36) | one high kick (1.08 m, 0.38 s above 0.35 m) |
| front kick | taskhybrid | 6/6 | 0 | 0.61 (0.57) | one high kick (1.06 m, 0.39 s) |
| front kick | tracking (reference point) | 6/6 | 0 | 0.73 | two kicks (0.74 m, ~0.5 s each) |

- **Squat jump: fixed.** The task policies now really jump, nearly as high and as long as tracking.
- **Front kick: better, not done.** The first kick is real and held, higher than the person's. The second (the sideways one) stays under 0.35 m but still meets its goal's tolerance.

Muse's critic agrees on the kick: tracking > taskhybrid > task, the same order as V-JEPA. On the squat jump it ranked task first and said tracking "barely leaves the ground", which the measurements contradict; an earlier call had ranked tracking first. One critic call is noisy. Average several blind orders before trusting a ranking.

## For the live demo

A new video becomes an environment within minutes: SAM 3.1, pose and retarget, then this spec and its self-check. The web app shows the key moments as soon as the spec exists; the trained policy follows after 1–2 hours of GPU training.
