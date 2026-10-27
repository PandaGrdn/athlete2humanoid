# Robustness: squat_jump

64 envs per policy per condition, 18 conditions. Task: 2 jumps: stand -> squat -> jump -> land -> stand -> squat -> jump -> land -> stand (spec: task_spec.json).

Task success (fall rate) per suite, averaged over its conditions

| Policy | Clean | Pushes | Stage-timed pushes | Start noise | Start pose | Speed | Friction/mass | No reference | DTW to clip | Joint err | V-JEPA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tracking | 100% (0% fall) | 65% (21% fall) | 98% (0% fall) | 54% (27% fall) | 100% (0% fall) | 97% (1% fall) | 100% (0% fall) | 0% (0% fall) | 0.101 | 0.103 | 0.651 |
| task | 100% (0% fall) | 94% (6% fall) | 99% (1% fall) | 83% (17% fall) | 100% (0% fall) | 100% (0% fall) | 99% (1% fall) | 100% (0% fall) | 0.411 | 0.424 | 0.396 |
| taskphase | 100% (0% fall) | 83% (17% fall) | 100% (0% fall) | 73% (27% fall) | 80% (20% fall) | 100% (0% fall) | 98% (2% fall) | 100% (0% fall) | 0.411 | 0.426 | 0.257 |

- Success: every stage of the task spec done in order, from the stage the clip is in at the start, with no fall (torso below 0.35 m or tilted past 1.4 rad) before the clip ends.
- Pushes: random shoves every 1-3 s; stage-timed pushes: one per squat (halfway down) or per touchdown; start noise: random start in the first half of the clip, 1-4x training's init noise; start pose: crouched or standing still at t = 0; no reference: standing still and the reference frozen at t = 0.
- DTW to clip: joint-space DTW to the retarget on the clean clip, mean rad per matched frame, over rollouts that stayed up; standing still scores 0.171. Joint err: the same, time-locked. V-JEPA: aligned cosine with the video, clean clip.

## Per condition

| Suite | Level | tracking | task | taskphase |
|---|---|---|---|---|
| clean | nominal | 100% / 0% fall / dtw 0.101 | 100% / 0% fall / dtw 0.411 | 100% / 0% fall / dtw 0.411 |
| push | 0.5 m/s | 92% / 0% fall / dtw 0.104 | 100% / 0% fall / dtw 0.408 | 98% / 2% fall / dtw 0.404 |
| push | 0.75 m/s | 92% / 0% fall / dtw 0.107 | 98% / 2% fall / dtw 0.406 | 97% / 3% fall / dtw 0.403 |
| push | 1 m/s | 61% / 20% fall / dtw 0.113 | 95% / 5% fall / dtw 0.403 | 81% / 19% fall / dtw 0.399 |
| push | 1.5 m/s | 14% / 66% fall / dtw 0.121 | 81% / 19% fall / dtw 0.399 | 55% / 45% fall / dtw 0.396 |
| init | x1 | 80% / 0% fall / dtw 0.102 | 100% / 0% fall / dtw 0.420 | 100% / 0% fall / dtw 0.411 |
| init | x2 | 61% / 17% fall / dtw 0.107 | 89% / 11% fall / dtw 0.421 | 89% / 11% fall / dtw 0.412 |
| init | x4 | 22% / 64% fall / dtw 0.113 | 59% / 41% fall / dtw 0.417 | 31% / 69% fall / dtw 0.410 |
| dynamics | level 1 | 100% / 0% fall / dtw 0.102 | 100% / 0% fall / dtw 0.410 | 100% / 0% fall / dtw 0.409 |
| dynamics | level 2 | 100% / 0% fall / dtw 0.102 | 98% / 2% fall / dtw 0.410 | 100% / 0% fall / dtw 0.409 |
| dynamics | level 3 | 100% / 0% fall / dtw 0.102 | 100% / 0% fall / dtw 0.409 | 94% / 6% fall / dtw 0.407 |
| speed | 0.8x | 100% / 0% fall / dtw 0.107 | 100% / 0% fall / dtw 0.428 | 100% / 0% fall / dtw 0.434 |
| speed | 1.2x | 94% / 2% fall / dtw 0.114 | 100% / 0% fall / dtw 0.397 | 100% / 0% fall / dtw 0.388 |
| push_phase | squat 0.75 m/s | 95% / 0% fall / dtw 0.103 | 100% / 0% fall / dtw 0.408 | 100% / 0% fall / dtw 0.405 |
| push_phase | land 0.75 m/s | 100% / 0% fall / dtw 0.107 | 98% / 2% fall / dtw 0.408 | 100% / 0% fall / dtw 0.403 |
| start | crouched | 100% / 0% fall / dtw 0.110 | 100% / 0% fall / dtw 0.435 | 61% / 39% fall / dtw 0.421 |
| start | standing | 100% / 0% fall / dtw 0.105 | 100% / 0% fall / dtw 0.427 | 100% / 0% fall / dtw 0.420 |
| noref | frozen | 0% / 0% fall / dtw 0.214 | 100% / 0% fall / dtw 0.465 | 100% / 0% fall / dtw 0.419 |

