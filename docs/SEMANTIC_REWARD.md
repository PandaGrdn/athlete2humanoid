# Task-first rewards: learn what the action is, not the clip

## Why v1/v2 were only semi-okay

1. **We trained on the wrong clip.** Every run so far targeted `squat`. The user's clip is `squat_jump`: two jumps, reaching 0.38–0.40 m of foot clearance in the G1-sized retarget.
2. **V-JEPA cosine has little dynamic range as a score.**
   - On the v1 squat, the best *wrong* bank motion (`bend`) scores 0.518 DTW, the latent policy 0.549, the retarget 0.726.
   - Almost any humanoid motion lands near 0.5, so the reward barely separates "the action" from "a person moving".
3. **The learned reward gets hacked.** In v2 the distilled reward network scored the policies much higher (hybrid 0.35 → 0.49) while real V-JEPA stayed flat (0.35 → 0.37). That is classic overoptimisation of a learned reward (Gao et al. 2210.10760).
4. **Nothing in the latent reward pays for balance**, and the terminations measure distance to the retarget, not falling. The `latent-stability` session is fixing this.
5. **The retarget has artifacts, and its proportions are wrong.**
   - Uniform scaling by leg length gets the arms and torso wrong. The `retarget-prop` session is fixing this.
   - In `squat_jump`, the feet hover about 10 cm above the floor between the two jumps.
6. **Physics gap.** In simulation every policy stands about 7 cm lower than the retarget (bent knees). Anything keyed to the retarget's absolute heights is off by that much.
7. **The metric undervalued the policies.** Under the semantic spec below, all five existing squat policies complete the squat:

   | Run | Policies | Result |
   |---|---|---|
   | v1 | tracking, hybrid, latent | 5/5 stages, no falls |
   | v2 | hybrid, latent | 5/5 stages, no falls |

   Low V-JEPA alignment was not the same as failing the action.

## What changes

`skill/task.py` turns the video into a **task spec**: an ordered list of stages (`stand`, `squat`, `jump`, `land`), each with a predicate on the robot's own body.

**The spec is built from the retarget's kinematics.**
- The retarget is G1-sized, so squat depth and jump clearance come out in the robot's proportions.
- Jump clearance is clipped to 0.30 m (`--max-clearance`).

**Muse's `skill_spec.json` is used when present.**
- It names the stages ("crouch", "takeoff_and_flight").
- It is cross-checked against the kinematics. For example, if Muse says the motion is airborne but the retarget never leaves the ground, the spec records a warning.

**Thresholds are physical.**
- A foot within 5 cm of the floor counts as down; both feet above 6 cm count as in the air.
- A flight must last at least 0.15 s, so glitches aren't counted as jumps.
- The retarget's own noise level is used only to detect stages and for its self-check. The retarget must pass its own spec (`squat_jump`: 9/9).

**`StageMachine` is the reward.** It is a batched reward machine (Icarte et al., 2010.03950):
- *Potential-based progress* toward the current stage's goal (Ng et al. 1999). Bobbing up and down earns at most one stage's worth; a test checks this.
- A *one-off bonus* when the stage's predicate holds long enough.
- A *jump-quality bonus* when the landing completes. Full marks for apex in [goal, 1.5 × goal]; less for lower jumps, and less again for overshooting.
- A *finish bonus*, scaled by the fraction of stages completed this episode.
- Progress is *gated on staying upright*.
- Timing and poses are free; only the order of stages matters.
- Episodes that start mid-motion start at the stage the reference is in at that moment.
- The current stage is also part of the **policy's observation** (`task_stage`), so the reward stays Markov: the policy knows "squat now", "jump now".

**New mjlab tasks** in `mjlab_tasks.py`:

| Task | Policy sees | Style terms | Episode ends |
|---|---|---|---|
| `A2H-Task` | the retarget as before, plus the stage | V-JEPA latent at weight 1; tracking at 0.1× | on falls only (torso < 0.35 m or tilt > 1.4 rad) |
| `A2H-TaskPhase` | its own body, a clock, and the stage; never the retarget | V-JEPA latent at weight 1 | on falls only |

- The task reward has weight 2.
- By episode budget, task ≈ 75% of the positive reward, in line with the 0.7–0.9 task share that AMP-style task-plus-style work uses.
- Wire-up: `experiments/latent_policy.py --policies task taskphase`. It compiles `task_spec.json` for the target and adds `task_success` and `task_stages` to `eval.json` for every policy.
- `--reward` now re-points a reused reward bundle at the new target. The bundle's network covers every clip rendered at its viewpoint; `squat` and `squat_jump` share the −150° bucket.

## Which movements matter (goal relevance)

A person's clip has movements that don't serve the goal. For example, waving the arms to hype the crowd before a long jump.

- **The stage reward already ignores them.** A test checks that arm waving before the jump doesn't change success.
- **The style terms still imitate them.** The `TimeWeighted` wrapper multiplies the latent and tracking terms by `A2H_IMPORTANCE`: per reference frame, how much that moment matters for the goal. Without it, every weight is 1.
- **Producing `importance.json`**: `python -m skill.importance <clip>`; method and validation in `docs/GOAL_RELEVANCE.md`:
  - V-JEPA 2's predictor as a world model: mask time windows or body regions and measure how much harder the goal-event latents (takeoff, apex, landing) become to predict.
  - Checked by counterfactual ablations in simulation.

## Evaluate "learned the action, not copied the clip"

- **Semantic success rate** (`task_success`), including under the robustness suite's pushes, start noise and speed changes. Not just V-JEPA alignment.
- **High success at a non-trivial distance from the clip.** Report joint-space DTW to the retarget alongside success.
- `experiments/robustness_eval.py` does both, for every policy of a run (v1 runs get a spec compiled from their `motion.npz`):
  - Success per condition: v1's 13 conditions, plus pushes timed to the squat and to touchdown, starts crouched or standing still, and no reference at all (standing still, reference frozen).
  - Stages done, jump apex and quality, squat depth, and DTW to the clip, next to falls, tracking errors and V-JEPA.
  - `robust_summary.md` is the table for the write-up.
  - Task-first policies need their `task_spec.json` in the run folder (or `A2H_TASK_SPEC`); without it they refuse to load.
- **Commanded targets** (`A2H-TaskCmd`): every episode samples its own squat depth and jump clearance, under a curriculum. `experiments/command_eval.py` checks whether the policy follows them; see `docs/PIPELINE_IMPROVEMENTS.md`.
- **Next steps:**
  - Have Muse judge rendered robot rollouts phase by phase.
  - Check specs from other squat-jump videos agree.

## References

From the literature review, 2026-10-27.

**Reward machines and shaping**
- Icarte et al., reward machines, 2010.03950.
- Ng, Harada, Russell, potential-based shaping, ICML 1999.

**LLM- and video-written rewards**
- L2R, 2306.08647: an LLM fills parameters of fixed reward templates. That is the reliability argument for a fixed predicate library over free-form code.
- Eureka, 2310.12931; SDS, 2410.11571; E-SDS on the G1, 2512.16446.
- CurricuLLM, 2409.18382.

**Similarity rewards fail on dynamic motions**
- VLM-RM, 2310.12921; RoboCLIP, 2310.07899.

**Combining task and style**
- AMP, 2104.02180; SkillMimic, 2408.15270 (multiplicative gating).
- PBHC, 2506.12851.

**Retargeting across bodies**
- GMR, 2510.02252; OmniRetarget, 2509.26633.
