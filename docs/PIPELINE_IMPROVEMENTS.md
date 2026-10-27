# Generalisation: commanded targets, and what to improve next

The task-first policies (`docs/SEMANTIC_REWARD.md`) are trained on one clip's targets: one squat depth, one jump height, two reps. They can learn that one trajectory and nothing about "squat this deep, jump this high". This doc covers:

- what changed to make them generalise,
- which further improvements are worth it, by value and effort,
- what was checked on CPU, and the GPU runs that remain.

## 1. Commanded targets: `A2H-TaskCmd-Flat-Unitree-G1`

TaskCmd is TaskPhase (its own body, a clock, the stage; it never sees the retarget), except that **every episode asks for its own targets**.

| Command | Range (G1) | Clip value (`squat_jump`) | How it is scored |
|---|---|---|---|
| squat depth | 0.10–0.32 m below standing | 0.246 m | squat predicate at ¾ of it (as `compile_spec`); squat progress |
| jump clearance | 0.08–0.30 m | 0.30 m (already clipped to the G1 max) | jump-quality band: full marks for an apex in [1, 1.5] × command, less for lower or higher |
| reps (off by default) | 1 … the clip's | 2 | the stage sequence ends after that many reps |

**How it works.**

- **Per-env goals.** `StageMachine.set_goals` (`skill/task.py`) holds the goals per env. `command_spec()` gives the same targets as a spec that `evaluate()` reads; a test checks that both give identical rewards.
  - Without commands, the machine is bit-identical to before. It was checked against the committed version on 1,200 batched steps with 108 stage advances: max |reward/obs diff| = 0.
- **The policy sees the targets.** `task_command` gives them normalised to [-1, 1] over the *fixed G1 ranges* (not the clip's), so the numbers mean the same thing for every clip.
  - They are the **last actor and critic inputs**, so a TaskPhase checkpoint warm-starts TaskCmd: `policy_weights.py` gives the 3 new inputs zero weights.
  - Verified on CPU: `command_eval.py --check-warm-start` loads TaskPhase `model_750` into TaskCmd. The action difference against TaskPhase on the same observations is **max 0.0**, with command inputs up to ±0.96 (119 → 122 inputs).
- **Curriculum** (`skill/commands.py CommandSampler`):
  - At level 0 the ranges are the clip's value ±15 % of the way to the range ends; level 1 is the full range.
  - Every 2,048 finished episodes, the level moves by ±0.1. It goes up when the commanded success rate is above 0.6 and down below 0.3.
  - Commanded success means: every stage done, no fall, and every jump at quality ≥ 0.7 (apex 0.7–1.8 × the command).
  - 20 % of episodes always sample the full ranges.
  - The level and ranges are logged as `Curriculum/command/*`.
  - This is the success-gated scheme of terrain curricula (Rudin et al. 2109.11978) applied to goal parameters, as in parameterised-environment teachers (Portelas et al. 1910.07224). Conditioning the policy on the goal is the UVFA/HER idea (Schaul et al. 2015; Andrychowicz et al. 1707.01495).
- **Stability margins follow the command.** The clip-relative "sank too low" penalty and termination allow the extra depth of a squat commanded deeper than the clip's.
- **Settings.** Everything is in `A2H_REWARD_CONFIG` → `"command": {...}`, e.g. `{"command": {"reps": true, "up": 0.5}}`.

**Evaluation: `experiments/command_eval.py`** (CPU is fine; 48 envs × 450 steps ≈ 30 s per policy).

- **The grid.** Every env gets one command, the clip's own plus targets it never showed: depth {0.14, 0.19, 0.246, 0.30} × clearance {0.12, 0.20, 0.30}, 2–4 envs each.
- **Reported** per command and policy:
  - success under the commanded spec;
  - `cmd_success` (jumps in the band);
  - falls;
  - achieved squat depth and jump clearance per rep vs commanded. Only reps actually done count, and nothing from the 0.3 s before a fall.
- **Summary:** clip vs novel commands, MAE, and the **slope of achieved on commanded** (1 = follows the command, 0 = ignores it). A unit test checks that a synthetic follower scores ≈1 and an ignorer ≈0.
- **TaskPhase checkpoints** load through the same zero-weight path, so they are the "ignores the command" baseline.
- **A2H-Task checkpoints** are refused: they see the retarget, so their inputs don't fit.

**Baseline today.** The only local checkpoint is the plateaued 750-iteration TaskPhase run. In the play env it does stand → squat → jump within its first 0.7 s, then falls on the landing in every env: 48/48, any command. That run's ~58 % training falls look like rushed, stiff landings. Its numbers are a harness check, not a result.

## 2. Next improvements, ranked by value / effort

| # | Candidate | Value | Effort | Verdict |
|---|---|---|---|---|
| A | **Latent-reward uncertainty penalty** (mean − λ·excess spread) | high: the distilled V-JEPA reward was hacked in v2 | low (MC dropout, no re-distillation) to medium (ensemble) | **done**, validated on the real v2 bundle |
| B | **Grounded reference-state init** (start states with the feet on the floor) | high: 64 % of `squat_jump` start states began with a drop | low | **done**, validated in the CPU env |
| C | **Landing-impact penalty** (foot contact force above 3 BW) | medium-high: stiff landings are where policies fall, and where a real G1 takes peak load | low (mjlab's sensor) | **done** (TaskCmd default) |
| D | RSI at stage boundaries | low now: mjlab's adaptive sampling already oversamples the failing bins (the landings); B fixes the actual RSI bug | low | not done |
| E | Contact-based flight/contact predicates | low in sim (ankle heights are exact there); matters for **deployment**, since the stage observation must be computed on the robot | medium | design note below |
| F | AMP-style discriminator on V-JEPA features or body trajectories | high in principle (AMP 2104.02180, ASE 2205.01906): a learned, harder-to-hack style score | high, needs GPU iterations | later |
| G | Multi-clip training (several videos of one action → one policy) | high for "learn the action, not the clip" | medium; needs ≥ 3 clips of the same action, which we don't have | later; TaskCmd already removes the single-target crutch |
| H | Sim-to-real: DR, latency, observation noise | needed for hardware | low per item | mostly in place; gaps listed below |
| I | ONNX export and G1 deployment notes | needed for a demo on hardware | low | `scripts/export_onnx.py` exists; notes below |

### A. Uncertainty-penalised latent reward

**Problem** (`docs/SEMANTIC_REWARD.md` §3). In v2 the distilled network scored the hybrid and latent policies 0.115 and 0.101 *above* real V-JEPA. That is overoptimisation of a learned reward (Gao et al. 2210.10760). The v1 rollouts and the retarget were in its training data, and there the gap is ≤ 0.015.

**Fix.** Subtract the network's uncertainty where it is higher than on the reference at the same moment. This is uncertainty-weighted optimisation with reward-model ensembles (Coste et al. 2310.02743). Ensembles mitigate but do not eliminate hacking (Eisenstein et al. 2312.09244), hence a penalty rather than a hard filter.

```
score = cos(net(window), target) − λ · max(0, u(window) − u_ref[t])
u = |std of member predictions| / |their mean|
```

- **Members.** An ensemble's members (`latent_policy.py --ensemble 5`: bootstrap over training motions, each with its own seed; Lakshminarayanan et al. 1612.01474), or 8 MC-dropout samples of an existing single network (Gal & Ghahramani 1506.02142). The second needs no new bundle.
- **`u_ref`.** `u_ref[t]` is computed once from the reference motion's own windows, so the retarget keeps full marks, as with `cos_ref`.
- **Code.** `latent/reward_net.py` (`member_predictions`, `uncertainty`, `LatentRewardEnsemble`, `reference_windows`), `latent/distill.py` (`train_reward_net(members=)`), `mjlab_tasks.LatentReward`.
- **Bundles.** Old bundles load as before. Ensemble bundles also carry `state_dict` = member 0, so older code still reads them.

**Validation, CPU, the real v2 squat bundle and rollouts.**

- **The network reproduces eval.json.** Its scores match the `reward_net_aligned` column exactly.
- **The uncertainty flags the hacked windows.** Per-window AUROC, hacked v2 policy vs in-distribution (retarget + v1 policies):
  - relative-spread `u`: **0.88**;
  - spread of the cosine itself: 0.70.
- **The penalty closes the gap.** Distilled − real gap per rollout:

| rollout | gap now | λ = 1 | λ = 2 | **λ = 4** |
|---|---|---|---|---|
| retarget | +0.010 | +0.007 | +0.005 | −0.001 |
| v1 hybrid / latent / tracking | +0.001 / +0.008 / +0.014 | +0.003 … +0.008 | −0.007 … +0.002 | −0.016 … −0.011 |
| **v2 hybrid (hacked)** | **+0.115** | +0.092 | +0.069 | **+0.023** |
| **v2 latent (hacked)** | **+0.101** | +0.069 | +0.037 | **−0.027** |

  So λ = 4 removes ~80–100 % of the hacked gap and moves in-distribution rollouts by ≤ 0.017.
- **Caveat.** This is an in-sample calibration on one bundle. Whether the *policy* then stops drifting into such motions can only be shown by training.

**Defaults.**

- TaskCmd uses λ = 4: its new targets take it off the clip the network was distilled on.
- Every other task stays at λ = 0, i.e. unchanged. Opt in with `"latent_uncertainty": 4` in `A2H_REWARD_CONFIG`.
- Cost: 8 extra forward passes of the reward MLP per step. On CPU, the TaskCmd env with all its extras took 2.36 s per 60 steps × 16 envs, against 1.79 s for TaskPhase. On an A100 the extra is estimated at < 10 % of iteration time (not measured).

### B. Grounded reference-state init

mjlab's RSI (DeepMimic-style, Peng et al. 1804.02717) puts the robot at the reference's root pose. The proportional retarget's feet hover:

- **In this clip.** In `squat_jump`, **64 % of frames** have the lower ankle more than 2 cm above its standing height (0.034 m in sim), up to 11 cm.
- **At t = 0.** The first frame hovers 5.2 cm. Every episode starting there spends 4 control steps in the air and lands with **3.8–4.6 body weights** of foot force, before the policy has done anything.

**Fix** (`mjlab_tasks.ground_starts`).

- Start states are lowered so the lower foot is on the floor, wherever the reference hovers by at most its own hover level (the spec's `down_ref`).
- Jump stages are left out (0.15 s before to 0.1 s after), because those frames really are in the air.

**Result.** From the first frame, the start ankle is at 0.028 m and the first-0.2 s peak force is **1.6–2.4 BW**. Default on for TaskCmd; Task/TaskPhase can opt in with `"grounded_rsi": true`.

### C. Landing-impact penalty

**What it penalises.** `LandingImpact` uses mjlab's G1 foot-floor contact sensor (as in its velocity task). It charges the feet's total contact force above **3 body weights**, for every step the excess lasts. mjlab's own `soft_landing` only sees the first contact step.

**Calibration** (CPU env, G1 = 33.3 kg):

| Case | Peak foot force |
|---|---|
| The clip's own landing (pelvis acceleration) | ~1.9 BW |
| The 750-iteration policy's own landings | 2.2–2.6 BW |
| Ungrounded start drop | 3.8–4.6 BW |

**Weight.** TaskCmd uses −5 per second per BW of excess, so a 3-step 5 BW landing costs ~0.6 against the 4 a jump pays. Elsewhere it is opt-in: `"landing_weight"`, `"landing_threshold"`. Logged as `Metrics/foot_force_peak_bw`.

### E. Stage observation on a real robot (design note)

`task_stage` and the reward machine read world-frame pelvis and ankle heights. Ankle heights are exact in sim; on a G1 they come from leg forward kinematics plus IMU attitude, assuming a stance foot is on the ground.

- **Flight** is then better detected from contact: the G1 has no foot force sensors, so contact must be estimated from joint torques or with a contact estimator.
- **The change needed** is `StageMachine.step(pos, contact=None)` taking a contact mask for its `feet_down`/airborne predicates, fed by the same contact sensor in sim.
- **It is not done yet,** because a training/eval mismatch (contacts in training, heights in `evaluate`) would need the eval to record contacts too.

### H. Sim-to-real: what's in place, what's missing

**In place.** mjlab's G1 tracking config already has the BeyondMimic sim-to-real recipe (Liao et al. 2508.08241), which transferred zero-shot to a real G1:

- pushes, ramped up to 1.5× by our curriculum;
- foot friction 0.3–1.2;
- torso CoM offset;
- joint encoder bias;
- uniform observation noise on velocities and joints.

**Missing, cheapest first.** Each is a few lines as an mjlab event or action wrapper:
1. **Action latency** of 0–20 ms (one control step), randomised per episode. Hwangbo et al. 1901.08652 and ASAP (He et al. 2502.01143) both name actuation delay a main gap.
2. **PD gain and motor-strength scaling** of ±10–20 % (Peng et al. 1710.06537).
3. **Link mass scaling** of ±10 %. `robustness_eval.py` already tests it at eval time.

**Noise on the new inputs.** The command and stage inputs carry no noise, which is right for the command. The stage input should get the state-estimation noise of E once it is computed on hardware.

### I. Export and deployment

`scripts/export_onnx.py run_folder --verify` exports `policy_*/model_*.pt` to ONNX (obs → normaliser → MLP → 29 joint targets), checked against rsl_rl's actor.

The obs vector for TaskCmd, in order, is mjlab's tracking actor terms minus the reference terms:
- `base_lin_vel`, `base_ang_vel`, `joint_pos`, `joint_vel`, `actions`;
- `phase` (8);
- `task_stage` (18);
- `task_command` (3).

On the robot, `phase` comes from a clock, `task_stage` from E, and `task_command` from the operator: "squat 0.2, jump 0.15". Control runs at 50 Hz with mjlab's G1 PD gains and action scale; `base_lin_vel` needs a state estimator.

## First TaskCmd result (2026-10-27, short run)

The run used the last Colab units:
- `A2H-TaskCmd`, 500 iterations, warm-started from the final squat-jump TaskPhase policy (1,200 iterations), with the goal-importance weights.
- The VM was stopped right after training, so the V-JEPA evaluation was skipped.
- The policy is in `weights/sj_cmd_taskcmd_500`; the raw results are in `outputs/remote_latent_sj_cmd/command_eval.json`.

`experiments/command_eval.py`, run on CPU: 12 commands (squat depth × jump clearance, 11 of them away from the clip's own targets), 3 envs each.

| Policy | Success, novel commands | Hits the commanded height (q ≥ 0.7) | Depth slope | Clearance slope | Depth error | Clearance error |
|---|---|---|---|---|---|---|
| TaskCmd, 500 iterations | 97% | 33% | 0.11 | 0.06 | 8.6 cm | 6.1 cm |
| TaskPhase (ignores commands) | 94% | 58% | 0.06 | 0.13 | 8.7 cm | 6.5 cm |

- **Generalisation holds.** Both policies still complete the squat jump for almost every command.
- **Command following isn't learned yet.** A slope of 1 means doing what is asked and 0 means ignoring it; TaskCmd sits near 0. After 500 iterations it still squats about 29 cm deep and jumps about 24 cm high whatever it's asked.
- **The curriculum hadn't widened** past the first levels (success 0.33, under the 0.6 needed to advance).
- **Next:**
  - Train longer: 1,500 iterations, as planned in section 3.
  - Or make the command matter more early: a stronger quality-band weight, or lower the curriculum threshold with `{"command": {"up": 0.4}}` in the reward config.

## 3. GPU runs (not launched)

Before any run, **merge this branch and push main**: the Colab VM pulls GitHub main.

**TaskCmd, warm-started from the final TaskPhase run, plus TaskPhase continued as the fair baseline.** Same iterations, same start weights:

```bash
python3 scripts/remote_run.py latent --reuse outputs/remote_latent_<final> --with-weights \
    --target squat_jump --tag sj_cmd --iters 1500 \
    --args "--reward /content/a2h/inputs/reward.pt --policies taskcmd taskphase --init /content/a2h/inputs"
```

Notes on this command:
- `<final>` is the main session's run folder: the one with `policy_taskphase/model_*.pt` and `reward.pt`. `--with-weights` uploads those checkpoints.
- `--init /content/a2h/inputs` builds taskcmd off `policy_taskphase` (`INIT_FALLBACK`) and continues taskphase from its own weights.
- Add `--importance /content/a2h/docs/goal_relevance/squat_jump_importance.json` inside `--args` if the final run used importance weighting.
- Cost: two policies at 4,096 envs ≈ 2.4 s/iteration, so 1,500 iterations ≈ 60 min. With setup, evaluation and VM overhead that is ≈ 1.4 h, about **7.5 units**. TaskCmd alone (`--policies taskcmd`) is about 1 h, about **5.5 units**.
- Watch `Curriculum/command/level` and `success` in `policy_taskcmd/train.log`. If the level stays at 0 after ~500 iterations, the warm start's success is below 0.6: rerun with a reward config `{"command": {"up": 0.4}}`.

**Afterwards, locally on CPU (~1 min):**

```bash
A2H_TASK_SPEC=outputs/remote_latent_sj_cmd/task_spec.json $MJLAB_PY experiments/command_eval.py \
    --motion outputs/remote_latent_sj_cmd/policy_taskcmd/motion.npz --per-command 4 \
    taskcmd=outputs/remote_latent_sj_cmd/policy_taskcmd/model_1499.pt \
    taskphase=outputs/remote_latent_sj_cmd/policy_taskphase/model_1499.pt --out outputs/remote_latent_sj_cmd/command_eval.json
```

**Success looks like this:**
- taskcmd `depth_slope` and `clearance_slope` near 1, against ~0 for taskphase;
- novel-command `cmd_success` well above taskphase's;
- clip-command success no worse.

**Caveat for the robustness suite.** `experiments/robustness_eval.py` (another session's) runs policies in the plain tracking env, where there is no stage machine. A TaskCmd policy would see `task_command` = 0 there, i.e. the middle of every range (depth 0.21 m, clearance 0.19 m), not the clip's targets. Before adding taskcmd to that suite, it should set the commands to the clip's (`env.a2h_cmd.fix(...)`) or run in the TaskCmd env.

**Optional: an ensemble reward bundle for A** (`--ensemble 5`). It needs a fresh distillation, since the dataset isn't kept: drop `--reward` and add `--ensemble 5`. Distillation is ~5× the network training time plus the usual render and V-JEPA dataset. Budget **+0.5 h, ~3 units** over the run above. MC dropout on the existing bundle already gives the validated signal.

## References

**Reward hacking and uncertainty**
- Gao, Schulman, Hilton, reward-model overoptimisation, 2210.10760.
- Coste et al., reward-model ensembles against overoptimisation, 2310.02743.
- Eisenstein et al., ensembles mitigate but don't eliminate reward hacking, 2312.09244.
- Lakshminarayanan et al., deep ensembles, 1612.01474.
- Gal & Ghahramani, MC dropout, 1506.02142.

**Goals, curricula, initial states**
- Schaul et al., UVFA, ICML 2015.
- Andrychowicz et al., HER, 1707.01495.
- Rudin et al., massively parallel RL with a success-gated terrain curriculum, 2109.11978.
- Portelas et al., teacher algorithms for parameterised environments, 1910.07224.
- Peng et al., DeepMimic (reference-state init), 1804.02717.

**Style terms**
- Peng et al., AMP, 2104.02180; ASE, 2205.01906.

**Sim-to-real**
- Liao et al., BeyondMimic, 2508.08241.
- He et al., ASAP, 2502.01143.
- Hwangbo et al., actuator networks and latency, 1901.08652.
- Peng et al., dynamics randomisation, 1710.06537.
- Tobin et al., domain randomisation, 1703.06907.
