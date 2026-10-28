# Tracking, task or both: which policy generalises?

## Why tracking looks better

Tracking is rewarded for matching the retarget's every joint at every step, so it looks the most like the video. The task policies are rewarded for reaching the video's key moments in order, at their own timing. Between those moments the pose is free; the style terms that pull it toward the video are weak (Task: tracking at 0.1x, weighted toward the key moments, plus the V-JEPA latent).

Measured on the clean clip (`eval.json`):

| Clip | Policy | V-JEPA aligned | Joint error (rad) | Torso error (m) | Key moments |
|---|---|---|---|---|---|
| front kick | tracking | **0.73** | **0.11** | **0.12** | 6/6 |
| front kick | task | 0.46 | 0.43 | 0.85 | 6/6 |
| front kick | taskphase | 0.36 | 0.51 | 0.49 | 6/6 |
| squat jump (final) | tracking | **0.70** | **0.11** | **0.11** | 9/9 |
| squat jump (final) | task | 0.43 | 0.42 | 0.41 | 9/9 |

All of them do the action. Tracking does it the way the person did; the task policies do it their own way (they drift up to ~1 m and keep their own tempo).

## What looking right costs

The robustness eval of the final squat-jump policies (`outputs/remote_robust_final/robust_summary.md`, 64 rollouts per condition): task success per suite, with the fall rate in brackets.

| Policy | Clean | Pushes | Start noise | Speed 0.8 / 1.2x | Friction / mass | No reference | Distance to clip (DTW) |
|---|---|---|---|---|---|---|---|
| tracking | 100% | 65% (21%) | 54% (27%) | 97% | 100% | **0%** | **0.10** |
| task | 100% | **94% (6%)** | **83% (17%)** | 100% | 99% | **100%** | 0.41 |
| taskphase | 100% | 83% (17%) | 73% (27%) | 100% | 98% | **100%** | 0.41 |

- At the hardest settings the gap is large:
  - 1.5 m/s shoves: tracking falls 66% of the time, task 19%.
  - 4x start noise: tracking falls 64%, task 41%.
- Tracking cannot run without the reference: frozen, it stands still.
- Tracking is a copy of the clip: closest to it, and the first to fail away from it.

## The generalisation test

`experiments/robustness_eval.py` now also asks each policy for things it never trained on:
- **Tempo:** the reference at 0.5x, 1.5x and 2x speed (the speed suite's 0.8x and 1.2x are mild). The task must still be done before the clip ends.
- **Transfer to a similar new task:** a clip the policy never saw, with a keypose spec compiled from it:
  - `--mirror`, for any clip, the mirror image (the kick with the other leg, mirrored exactly: forward kinematics of the mirrored joints reproduces the mirrored bodies to 1e-5 m);
  - `--transfer squat=<motion.npz>` for the squat-jump policies: the plain squat clip.

The policies compared, for squat jump and front kick:
- tracking;
- task (keypose reward, 0.1x goal-weighted tracking);
- taskphase (blind: clock and stage only);
- **taskhybrid**, new (`A2H-TaskHybrid`): the keypose task plus tracking at half strength over the whole clip. It is meant to keep tracking's look and the task's robustness.

Runs: `outputs/remote_latent_hyb_sj`, `outputs/remote_latent_hyb_kick` (training), then `outputs/remote_robust_gen_sj`, `outputs/remote_robust_gen_kick` (the evaluation).

**First signal (CPU smoke, 4 rollouts):** every front-kick policy, tracking included, falls on the mirrored kick. A policy trained on one clip has learned that clip. If the full run agrees, generalisation has to be trained, not hoped for. That means training on variations of the clip and holding others out:
- mirrored copies;
- tempo randomisation;
- the commanded targets of `A2H-TaskCmd`, which already make squat depth and jump height a command.

## Results

**Squat jump** (`outputs/remote_robust_gen_sj/robust_summary.md`, 64 rollouts per condition): task success, with the fall rate in brackets.

| Policy | Clean | Pushes | Start noise | Tempo 0.5–2x | No reference | Mirrored | Plain squat | V-JEPA |
|---|---|---|---|---|---|---|---|---|
| tracking | 100% | 80% (18%) | 70% (27%) | 70% (0%) | 0% | 0% (100%) | 0% (100%) | **0.65** |
| task | 100% | **97% (3%)** | **87% (13%)** | **100%** | 100% | 0% (100%) | 0% (100%) | 0.33 |
| taskphase | 100% | 91% (9%) | 78% (22%) | 76% (24%) | 100% | **100% (0%)** | 0% (98%) | 0.26 |
| taskhybrid | 100% | 96% (4%) | 80% (20%) | **100%** | 100% | 0% (100%) | 0% (100%) | 0.45 |

- **Taskhybrid:** as robust as task, and closer to the video by V-JEPA (0.45 vs 0.33).
- **Tracking fails off its clip:**
  - 2x tempo: 14% success;
  - frozen reference: 0%;
  - 1.5 m/s shoves: falls 52% of the time.
- **Nothing transfers to a new clip.** The blind taskphase passes the mirrored squat jump only because a squat jump is nearly symmetric and it never looks at the reference.

**Caveat, found after these runs (`docs/KEYPOSE_SPEC.md`, "The checker was too lenient"):** the task-family policies were tuck-hopping. Feet off the floor for 0.04–0.14 s, the pelvis never above standing height; Muse's critic saw no jumps. Their robustness here is partly the robustness of an easier motion. They are being retrained on the fixed spec from the tracking weights. This table will be redone with those.

**Front kick:** the generalisation eval is running (`outputs/remote_robust_gen_kick`).
