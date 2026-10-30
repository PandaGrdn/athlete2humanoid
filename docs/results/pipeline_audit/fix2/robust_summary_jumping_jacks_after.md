# Robustness: jumping_jacks

16 envs per policy per condition, 7 conditions. Task: 10 key moments, 1 jump: ready -> right hand up, left hand up -> right hand down, left hand down -> right hand up, left hand up -> rise, right hand down -> left hand down, crouch -> jump -> land -> right hand down, left hand down -> crouch, right hand up -> left hand down, right hand down -> left hand up, right hand up -> right hand down, left hand down (finish) (spec: task_spec.json).

Task success (fall rate) per suite, averaged over its conditions

| Policy | Clean | Pushes | Tempo 0.5-2x | No reference | New similar clip | DTW to clip | Joint err | V-JEPA |
|---|---|---|---|---|---|---|---|---|
| multi_1500 | 0% (0% fall) | 6% (0% fall) | 0% (6% fall) | 0% (0% fall) | 0% (0% fall) | 0.117 | 0.119 | - |
| scratch_1500 | 0% (0% fall) | 0% (25% fall) | 0% (50% fall) | 0% (0% fall) | 0% (0% fall) | 0.129 | 0.129 | - |
| specialist_1500 | 0% (0% fall) | 0% (12% fall) | 0% (50% fall) | 0% (0% fall) | 0% (0% fall) | 0.152 | 0.153 | - |

- Success: every stage of the task spec done in order, from the stage the clip is in at the start, with no fall (torso below 0.35 m or tilted past 1.4 rad) before the clip ends.
- Pushes: random shoves every 1-3 s; stage-timed pushes: one per squat (halfway down) or per touchdown; start noise: random start in the first half of the clip, 1-4x training's init noise; start pose: crouched or standing still at t = 0; no reference: standing still and the reference frozen at t = 0.
- DTW to clip: joint-space DTW to the retarget on the clean clip, mean rad per matched frame, over rollouts that stayed up; standing still scores 0.234. Joint err: the same, time-locked. V-JEPA: aligned cosine with the video, clean clip.

## Where unsuccessful rollouts stop (clean clip)

The stage they were in when the clip ended or they fell, as a share of all rollouts; for a key pose, the feature most often furthest outside its tolerance at their closest approach (error / tolerance, median).

| Policy | Stage | Rollouts | Fell | Furthest feature | Error / tol | Reached, not held |
|---|---|---|---|---|---|---|
| multi_1500 | 6: jump | 100% | 0% | - | - | - |
| scratch_1500 | 6: jump | 100% | 0% | - | - | - |
| specialist_1500 | 6: jump | 100% | 0% | - | - | - |

## Tracking error on the clean clip (RMS while upright, cm)

| Policy | pelvis | torso | L hand | R hand | L foot | R foot |
|---|---|---|---|---|---|---|
| multi_1500 | 36.5 | 36.3 | 40.9 | 39.5 | 41.9 | 49.6 |
| scratch_1500 | 43.5 | 43.8 | 48.2 | 47.2 | 49.8 | 54.1 |
| specialist_1500 | 42.1 | 42.0 | 47.3 | 44.6 | 52.0 | 52.2 |

## Per condition

| Suite | Level | multi_1500 | scratch_1500 | specialist_1500 |
|---|---|---|---|---|
| clean | nominal | 0% / 0% fall / dtw 0.117 | 0% / 0% fall / dtw 0.129 | 0% / 0% fall / dtw 0.152 |
| push | 1 m/s | 0% / 0% fall / dtw 0.123 | 0% / 0% fall / dtw 0.135 | 0% / 6% fall / dtw 0.153 |
| push | 2 m/s | 12% / 0% fall / dtw 0.141 | 0% / 50% fall / dtw 0.159 | 0% / 19% fall / dtw 0.169 |
| noref | frozen | 0% / 0% fall / dtw 0.227 | 0% / 0% fall / dtw 0.239 | 0% / 0% fall / dtw 0.251 |
| tempo | 0.5x | 0% / 0% fall / dtw 0.157 | 0% / 0% fall / dtw 0.169 | 0% / 0% fall / dtw 0.220 |
| tempo | 2x | 0% / 12% fall / dtw 0.251 | 0% / 100% fall | 0% / 100% fall |
| transfer | mirrored | 0% / 0% fall / dtw 0.119 | 0% / 0% fall / dtw 0.135 | 0% / 0% fall / dtw 0.144 |

