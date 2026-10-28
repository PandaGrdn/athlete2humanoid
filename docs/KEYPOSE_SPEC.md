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
   - Moments at the top levels are required: at least 2.5, or half the clip's top level if that is lower. The finer ones are how this person happened to do it (an in-between crouch, an arm swing); they pay a bonus but may be skipped.
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
| keypose taskphase | 0/12 | – | 9 | 0.17 / 0.72 m |

- **The task policy learns the keypose spec:** 6/12 → 12/12 in 1,500 iterations, including the arm swing the old spec never asked for. Kept as `weights/kp_squat_jump_task`.
- **Taskphase is not converged.** Its training episodes grew from 280 to 468 of 500 steps and were still rising at the end, but it falls in the evaluation rollout. It needs more iterations.

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

## For the live demo

A new video becomes an environment within minutes: SAM 3.1, pose and retarget, then this spec and its self-check. The web app shows the key moments as soon as the spec exists; the trained policy follows after 1–2 hours of GPU training.
