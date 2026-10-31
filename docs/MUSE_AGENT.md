# Muse Spark as a live agent in the pipeline

Two parts of the pipeline now call Meta's Muse Spark (`muse-spark-1.3`, Meta Model API) live:

1. **Moment naming:** Muse watches the clip and names every key moment the keypose compiler found ("arms overhead legs apart", "sink into right lunge", "jump over hurdle"). The names go into `task_spec.json` under a stable key, `muse_label`.
2. **Reward agent loop:** Muse tunes a policy's reward over short training rounds. Each round it reads the evaluation and watches the rollout video, then calls bounded tools to change reward weights and termination settings. The next round trains on Modal from the last checkpoint, next to an untouched control with the same budget.

The verification calls, and what the integration plan got wrong, are in [`META_INTEGRATION.md` §7](META_INTEGRATION.md#7-verified-live-oct-30).

**Result in one line.** On jumping jacks, three agent rounds (900 PPO iterations) beat the same 900 iterations without the agent on likeness to the video: clean V-JEPA 2 similarity was **0.410 ± 0.008 vs 0.388 ± 0.017** (start 0.357). Success and falls stayed level. The agent did **not** reach its other target: the optional key moments it set out to recover were still skipped. One training run per arm; details and caveats [below](#the-run-jumping-jacks).

## What is live

| | Where | When it runs | Without the key |
|---|---|---|---|
| Moment naming | `muse/moment_namer.py` | `pipeline.py --task-spec`, `python -m skill.keypose <clip>`, `experiments/latent_policy.py` (when it compiles a spec), live uploads with `A2H_MUSE_NAMES=1` (`server/live.py`) | does nothing, says nothing; the compiler's names stay |
| Reward agent loop | `muse/agent_loop.py` (local, Responses API) + `experiments/agent_round.py` (GPU, one round) | `python -m muse.agent_loop <run> ...` | `--dry-run` replays fixtures |
| Older single-call tools | `muse/skill_reader.py`, `muse/reward_agent.py` (one proposal from a finished run), `muse/rollout_critic.py` | by hand | `--dry-run` |

The key is `MUSE_API_KEY`, read through `config.get_secret` (environment or `.env`). It stays on the machine that calls Muse: the GPU jobs never hold it. The Modal jobs get only the HF token line, as before.

## Moment naming

The keypose compiler (`skill/keypose.py`) picks the key moments by measurement and names each one after what changed most: "right hand up, left hand up". Muse gets four things:
- the clip at 4 fps, 480 px;
- a sheet of the frame at each moment, captioned with its number and time;
- the measured change at each moment;
- the times of each flight (both feet off the ground).

It returns a plain-English name and one sentence per moment, plus the action. Strict `json_schema` output is checked again in code: one entry per moment, 1–5 words, nothing addressed to the person. On any failure Muse sees the errors and gets one retry.

**The schema (`skill/spec_schema.py`).** Each named pose or flight stage gets four fields:
- `muse_label`: the stable key front ends read (PR #32's build view reads it);
- `description`: Muse's one sentence;
- `measured`: the compiler's name;
- `name`: Muse's name, as the existing server display expects.

Settle stages (ready, land, finish) are never named. `stage_label(st)` falls back from `muse_label` to `label` (the skill reader's), then `name`, then `kind`. `text_errors(spec)` checks the fields, and the tests run it on every named spec. The reward never reads any of these: naming a spec changes nothing a policy learns.

**The ten stored specs, named live.**

| Spec | Muse's action | Moments | First moment: Muse / compiler |
|---|---|---|---|
| `remote_latent_s3_jacks` | jumping_jacks | 11 | arms overhead legs apart / right hand up, left hand up |
| `remote_latent_s3_kick` | double_right_kick | 4 | kick right leg forward / right foot up, right foot forward |
| `remote_latent_s3_lunge` | alternating_forward_lunges | 13 | step right foot out / crouch, right foot left |
| `remote_latent_s3_punch` | alternating_forward_punches | 11 | raise right hand / right hand up |
| `remote_latent_s3_sj` | double_squat_jump | 9 (incl. 2 jumps) | sink into squat / crouch |
| `remote_latent_s3_squat` | double_bodyweight_squat | 4 | squat with hands forward / crouch, right hand forward |
| `mc_inputs/clips/arms` | lower_arms | 1 | lower arms to sides / left hand down, right hand down |
| `remote_process_pro/clips/discus` | spinning_overhead_throw | 1 | step left, turn left / turn left, step left |
| `remote_process_pro/clips/hurdles` | hurdle_jump | 1 (a jump) | jump over hurdle / jump |
| `remote_process_pro/clips/shot_put` | throw_ball_upward | 1 | step left, turn left / step left, turn left |

Notes on the ten:
- The first six specs live in their training runs' folders. The last four had no spec, so they were compiled with `python -m skill.keypose <clip> --no-names`, then named.
- Hurdles compiles to no key pose, only a flight. The namer therefore names flights as well, so every clip gets at least one label.
- All ten answers were valid on the first attempt. In total: 10 calls, 33k input and 10k output tokens, **$0.086**, ~10 s per clip.

## The reward agent loop

```
round 0   score the run's policy as it is ─────────────────────────────┐
          │                                                             │
          ▼                                                             │
  Muse turn (Responses API, previous_response_id carries its reasoning) │
     input: round result (numbers) + rollout video (+ the person's video, once)
     calls: get_config · set_param(path, value, reason) ×≤3 · train_next_round(hypothesis, watch) | finish
          │                                                             │
          ▼                                                             │
  round k on Modal (scripts/remote_run.py agentround → experiments/agent_round.py)
     agent arm: Muse's settings ─┐  both resume their own checkpoint,   │
     base arm:  defaults ────────┘  300 PPO iterations, in parallel      │
     score both alike: robustness_eval.py on the fixed task spec (64 robots × clean / pushes / noisy starts,
     V-JEPA 2 on rendered rollouts) + the rollout's stage check + the last iteration's reward terms
          │
          └── train_next_round's tool output = the agent arm's result; its rollout video follows ──▶ next turn
```

**The tools.** `set_param` changes one setting within bounds. A call out of bounds, an unknown setting, a value that is already set, or a fourth change in the same round gets an error the model can read and correct. All four were exercised in the tests; live, Muse never sent an invalid call.

| Setting | Default | Bounds | What it is |
|---|---|---|---|
| `task.progress`, `task.stage`, `task.done`, `task.hold`, `task.jump` | 1, 1, 4, 1, 2 | 0.25–4, 0.25–4, 1–12, 0–4, 0.5–6 | the stage machine's weights, written into the spec the agent arm trains on |
| `task_weight`, `style_weight`, `tracking_scale` | 2, 1, 0.1 | 0.5–6, 0–3, 0.02–0.5 | the stage reward term, the V-JEPA style term, the retarget-tracking terms |
| `stability.alive`, `.upright`, `.feet_slip`, `.action_rate`, `.push_scale` | 1, −2, −0.5, −0.1, 1.5 | 0–5, −10–0, −3–0, −1–−0.01, 1–2.5 | balance terms and training pushes |
| `terminations.fall_tilt`, `.fall_height` | 0.35 rad, 0.2 m | 0.15–1, 0.05–0.5 | when an episode ends as fallen (the task-first tasks' only terminations) |

**Rules that keep the comparison honest.**
- Every arm and round is scored on the original task spec; the agent can change what the arm trains on, never what it is scored on.
- The base arm gets the same iterations, the same resume mechanics and the same GPU job.
- Muse never sees the base arm.
- Not exposed: the clip's tempo. Tempo variants need a variant set built per round (`skill/variations.py`), more than a 300-iteration round can afford.

**Resumable.** `state.json` is saved after round 0, after each decision and after each round. A rerun reuses every round already scored, and a decision made before a failed round is replayed, not asked again. Transient API errors (5xx, timeouts) are retried after 10/30/60/120 s.

## The run: jumping jacks

**Starting point:** `outputs/remote_latent_s3_jacks`'s task policy (A2H-Task, 1500 iterations). In the simulator it already finishes the task and never falls: 100% success, clean and under 0.75 m/s pushes. That success counts optional stages as skippable, though. On average it completes 9 of the 11 stages, skipping three optional "arms overhead legs apart" moments: it does jumping jacks without its arms fully overhead. Its V-JEPA similarity to the video is 0.35, against 0.72 for the retargeted motion. So the headroom was:
- the skipped moments;
- likeness to the video;
- staying up under harder conditions.

**Objective given to Muse:** keep success and zero falls; then complete more key moments; then look more like the video.

**What Muse did** (its reasons, abridged from `agent_log.jsonl`):

| Before round | Changes | Its reason |
|---|---|---|
| 1 | `task.stage` 1→2, `task.progress` 1→2, `tracking_scale` 0.1→0.2 | stages 1, 3, 5 skipped; the rollout stops at stage 9 with the right wrist 2.7× outside tolerance; "robot arms not fully overhead vs person" in the video |
| 2 | `task.jump` 2→4, `stability.action_rate` −0.1→−0.04, `tracking_scale` 0.2→0.3 | legs-apart moments still skipped; the action-rate penalty (−3.1 per episode) dominates; V-JEPA 0.387 vs 0.72 |
| 3 | `task.done` 4→8, `style_weight` 1→2, `stability.feet_slip` −0.5→−0.2 | stages stuck at 9.0 for three rounds; "narrow arms vs person wide overhead clap" |
| finish | best round: 3 | "raising tracking_scale 0.1→0.3 and style_weight 1→2 helped the robot's overhead arm spread and overall likeness. Raising task.stage/progress/jump/done did not help: the robot still skips the early legs-apart overhead stages" |

One misstep: `task.jump` has no effect on jumping jacks, because the spec has no flight stages. The setting's description says so, and Muse raised it anyway. Its final summary is accurate, and matches the numbers below.

**Per round** (64 robots per condition; V-JEPA is the mean of 2 rendered rollouts, clean condition):

| Round | Arm | Clean success | Stages | Falls, noisy start (2×) | V-JEPA |
|---|---|---|---|---|---|
| 0 | start | 100% | 9.00 | 0% | 0.351 |
| 1 | agent / base | 100% / 100% | 9.00 / 9.00 | 0% / 3% | 0.387 / 0.363 |
| 2 | agent / base | 100% / 100% | 9.00 / 9.00 | 0% / 5% | 0.393 / 0.361 |
| 3 | agent / base | 100% / 100% | 9.00 / 9.00 | 0% / 2% | 0.416 / 0.373 |

**Confirmation** (`outputs/remote_agentround_agent_jacks_confirm`): the three final policies scored again in one environment, with 8 rendered rollouts per arm per condition and harder perturbations.

| Condition | Start (1500 it.) | Base (+900, defaults) | Agent (+900, Muse) |
|---|---|---|---|
| Clean: V-JEPA (mean ± sd, n = 8) | 0.357 ± 0.023 | 0.388 ± 0.017 | **0.410 ± 0.008** |
| Push 0.75 m/s: V-JEPA | 0.384 ± 0.019 | 0.403 ± 0.024 | **0.426 ± 0.011** |
| Push 1.0 m/s: V-JEPA | 0.378 ± 0.023 | 0.387 ± 0.038 | 0.407 ± 0.031 |
| Push 1.5 m/s: success / falls | 100% / 0% | 98% / 2% | 98% / 2% |
| Noisy start 4×: success / falls | 61% / 39% | 66% / 34% | 69% / 30% |
| Stages completed, clean | 9.00 | 9.00 | 9.00 |
| Joint DTW to the clip (rad, lower = closer) | 0.396 | 0.411 | **0.382** |
| V-JEPA of the deterministic rollout | 0.339 | 0.391 | **0.410** |

**What this says.**
- **Likeness: a small, consistent gain over the control.** The agent arm is ahead in every round, and at 0.75 m/s pushes. In the confirmation it leads by +0.022 on the clean clip (Welch t ≈ 3.3 over 8 + 8 rollouts) and +0.023 at 0.75 m/s (t ≈ 2.5). The lead shrinks to +0.020 at 1.0 m/s (t ≈ 1.6) and is a tie at 1.5 m/s. It is also closer to the retarget in joint space, the opposite of the control, which drifts away with more training.
  - The changes that plausibly did it are the ones Muse credited: more tracking weight (0.1→0.3) and more V-JEPA style weight (1→2).
  - Training reward isn't comparable across arms (the agent changed the weights), so it isn't used as evidence.
- **Key moments: a null result.** No arm completed more stages: 9.00 in every round, for all three policies. Raising the stage, progress and finish bonuses did not make the policy go back for the optional overhead poses.
- **Robustness: unchanged within noise.** At 4× start noise the success rates (69% vs 66%, 64 robots) are within each other's noise.
- **Caveats.**
  - One training run per arm: seed variance between two 900-iteration fine-tunes was not measured, and could be the size of the likeness gap.
  - V-JEPA's absolute numbers stay far below the retarget's 0.72.
  - The offline cross-check of saved rollouts (`task_success_evaluate`, skill/task.py at 50 Hz) disagrees with the in-sim stage machine and swings between rounds (clean, over rounds 1-3 and the confirmation: the agent arm read 50%, 100%, 100%, 100%; the base arm 100%, 0%, 50%, 12%), so it isn't used here.

## Cost

| | Amount | Cost |
|---|---|---|
| Muse Spark, agent loop | 12 responses (10 in the conversation that counts, 2 from an attempt cut off by an HTTP 500); 87.5k input tokens (55.6k cached), 11.7k output (9.4k reasoning) | **$0.098** |
| Muse Spark, moment naming | 10 calls, 33.2k in / 10.4k out | **$0.086** |
| Muse Spark, verification calls | 6 calls | $0.024 |
| Modal GPU (L40S, `MODAL_PROFILE=a2h`) | 72 GPU-minutes of job wall clock: round 0, 10 min; rounds 1–3, 20/14/16 min; confirmation, 8 min; two 2-min starts that found no inputs (fixed: `vol.reload()` in `scripts/modal_app.py`) | **~$2.35** |

A round costs ~14–20 GPU-minutes: ~11 min of two parallel 300-iteration trainings, then ~2.5 min of scoring once the bank's V-JEPA latents are cached (6 min before). The agent's API cost is about 4% of the GPU cost. No Modal container is left running.

## Running it

```bash
MODAL_PROFILE=a2h python -m muse.agent_loop outputs/remote_latent_s3_jacks --target jumping_jacks \
    --video clips/jumping_jacks.mov --out outputs/agent_jacks --rounds 3 --iters 300 --branch muse-agent-loop
```

```bash
python -m muse.agent_loop outputs/remote_latent_s3_jacks --target jumping_jacks --out /tmp/agent --dry-run
```

```bash
python -m muse.moment_namer outputs/remote_latent_s3_kick --video clips/front_kick.mov
```

`--branch` must be pushed: the GPU job clones it. The jacks run's files are in `outputs/agent_jacks/`:
- `report.md`;
- `agent_log.jsonl`, with every request (media redacted), response, tool call and tool result;
- `state.json`;
- the rounds' results, in `outputs/remote_agentround_agent_jacks_r{0,1,2,3,confirm}/`.

Tests: `tests/test_agent_loop.py` (tools, bounds, tool-call parsing, the conversation over canned rounds, resume, retries) and `tests/test_muse.py` (schema fields, naming flights, the optional step), all offline.

## For the submission: use of Meta tooling

> Meta's models run the whole loop from phone video to trained robot. SAM 3.1 segments the athlete, SAM 3D Body lifts them to 3D, and V-JEPA 2 is both the robot's "look like the video" reward and the judge of whether it does. Muse Spark is the agent on top, called live through the Meta Model API. It watches each clip and names the key moments of the robot's task in plain English ("arms overhead legs apart", "sink into right lunge"), written into the task spec the web app displays. Then it tunes the reward between training rounds. Through the Responses API it reads each round's evaluation, watches the robot's rollout video, and calls bounded tools to change reward weights and termination thresholds, each change with a stated reason. The next round trains on Modal from the last checkpoint. Measured against an identical run without the agent, three rounds (900 PPO iterations) raised the robot's V-JEPA likeness to the person's video from 0.388 to 0.410 (start: 0.357), at equal success and falls. Muse's own end-of-run summary named what worked (more tracking and style weight) and what didn't (bigger stage bonuses never recovered the skipped overhead poses), and the numbers agree. The whole agent run cost $0.10 in API credits, against $2.35 of GPU time.
