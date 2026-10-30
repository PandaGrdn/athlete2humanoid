# Robustness: punches

16 envs per policy per condition, 6 conditions. Task: 5 key moments: ready -> right hand up, left hand up -> left hand right, left foot back -> left hand left, right hand right -> left hand right, right hand left -> left hand left, right hand down (finish) (spec: task_spec.json).

Task success (fall rate) per suite, averaged over its conditions

| Policy | Clean | Pushes | Tempo 0.5-2x | No reference | DTW to clip | Joint err | V-JEPA |
|---|---|---|---|---|---|---|---|
| multi_300 | 0% (0% fall) | 12% (9% fall) | 0% (0% fall) | 0% (0% fall) | 0.180 | 0.181 | - |
| tracking_var_300 | 100% (0% fall) | 59% (31% fall) | 75% (0% fall) | 0% (0% fall) | 0.139 | 0.139 | - |

- Success: every stage of the task spec done in order, from the stage the clip is in at the start, with no fall (torso below 0.35 m or tilted past 1.4 rad) before the clip ends.
- Pushes: random shoves every 1-3 s; stage-timed pushes: one per squat (halfway down) or per touchdown; start noise: random start in the first half of the clip, 1-4x training's init noise; start pose: crouched or standing still at t = 0; no reference: standing still and the reference frozen at t = 0.
- DTW to clip: joint-space DTW to the retarget on the clean clip, mean rad per matched frame, over rollouts that stayed up; standing still scores 0.170. Joint err: the same, time-locked. V-JEPA: aligned cosine with the video, clean clip.
- Tolerances floored by the clip's tracker calibration: max(TOL_MIN, 2 x the RMS error of scratch_1500 (joint err 0.079 rad) per feature).

## Where unsuccessful rollouts stop (clean clip)

The stage they were in when the clip ended or they fell, as a share of all rollouts; for a key pose, the feature most often furthest outside its tolerance at their closest approach (error / tolerance, median).

| Policy | Stage | Rollouts | Fell | Furthest feature | Error / tol | Reached, not held |
|---|---|---|---|---|---|---|
| multi_300 | 2: left hand right, left foot back | 100% | 0% | left wrist y | 1.14 | 0% |

## Tracking error on the clean clip (RMS while upright, cm)

| Policy | pelvis | torso | L hand | R hand | L foot | R foot |
|---|---|---|---|---|---|---|
| multi_300 | 4.1 | 4.1 | 14.5 | 9.2 | 6.4 | 4.9 |
| tracking_var_300 | 4.8 | 4.8 | 10.8 | 9.4 | 6.3 | 4.8 |

## Per condition

| Suite | Level | multi_300 | tracking_var_300 |
|---|---|---|---|
| clean | nominal | 0% / 0% fall / dtw 0.180 | 100% / 0% fall / dtw 0.139 |
| push | 1 m/s | 6% / 19% fall / dtw 0.199 | 88% / 12% fall / dtw 0.157 |
| push | 2 m/s | 19% / 0% fall / dtw 0.215 | 31% / 50% fall / dtw 0.177 |
| noref | frozen | 0% / 0% fall / dtw 0.227 | 0% / 0% fall / dtw 0.214 |
| tempo | 0.5x | 0% / 0% fall / dtw 0.218 | 100% / 0% fall / dtw 0.166 |
| tempo | 2x | 0% / 0% fall / dtw 0.230 | 50% / 0% fall / dtw 0.187 |

