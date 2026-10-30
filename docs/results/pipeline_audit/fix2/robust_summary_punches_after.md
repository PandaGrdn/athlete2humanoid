# Robustness: punches

16 envs per policy per condition, 7 conditions. Task: 5 key moments: ready -> right hand up, left hand up -> left hand right, left foot back -> left hand left, right hand right -> left hand right, right hand left -> left hand left, right hand down (finish) (spec: task_spec.json).

Task success (fall rate) per suite, averaged over its conditions

| Policy | Clean | Pushes | Tempo 0.5-2x | No reference | New similar clip | DTW to clip | Joint err | V-JEPA |
|---|---|---|---|---|---|---|---|---|
| multi_1500 | 100% (0% fall) | 81% (9% fall) | 100% (0% fall) | 0% (0% fall) | 100% (0% fall) | 0.104 | 0.104 | - |
| scratch_1500 | 100% (0% fall) | 47% (50% fall) | 100% (0% fall) | 0% (6% fall) | 100% (0% fall) | 0.078 | 0.079 | - |
| specialist_1500 | 100% (0% fall) | 72% (19% fall) | 100% (0% fall) | 0% (0% fall) | 100% (0% fall) | 0.099 | 0.099 | - |

- Success: every stage of the task spec done in order, from the stage the clip is in at the start, with no fall (torso below 0.35 m or tilted past 1.4 rad) before the clip ends.
- Pushes: random shoves every 1-3 s; stage-timed pushes: one per squat (halfway down) or per touchdown; start noise: random start in the first half of the clip, 1-4x training's init noise; start pose: crouched or standing still at t = 0; no reference: standing still and the reference frozen at t = 0.
- DTW to clip: joint-space DTW to the retarget on the clean clip, mean rad per matched frame, over rollouts that stayed up; standing still scores 0.170. Joint err: the same, time-locked. V-JEPA: aligned cosine with the video, clean clip.
- Tolerances floored by the clip's tracker calibration: max(TOL_MIN, 2 x the RMS error of scratch_1500 (joint err 0.079 rad) per feature): clean success before it multi_1500 100%, scratch_1500 100%, specialist_1500 100%.

## Tracking error on the clean clip (RMS while upright, cm)

| Policy | pelvis | torso | L hand | R hand | L foot | R foot |
|---|---|---|---|---|---|---|
| multi_1500 | 3.4 | 3.4 | 6.6 | 9.5 | 1.7 | 5.0 |
| scratch_1500 | 3.5 | 3.6 | 7.1 | 6.0 | 1.1 | 4.9 |
| specialist_1500 | 3.6 | 3.7 | 7.0 | 6.8 | 2.8 | 4.9 |

## Per condition

| Suite | Level | multi_1500 | scratch_1500 | specialist_1500 |
|---|---|---|---|---|
| clean | nominal | 100% / 0% fall / dtw 0.104 | 100% / 0% fall / dtw 0.078 | 100% / 0% fall / dtw 0.099 |
| push | 1 m/s | 100% / 0% fall / dtw 0.121 | 94% / 0% fall / dtw 0.102 | 94% / 0% fall / dtw 0.115 |
| push | 2 m/s | 62% / 19% fall / dtw 0.139 | 0% / 100% fall | 50% / 38% fall / dtw 0.132 |
| noref | frozen | 0% / 0% fall / dtw 0.194 | 0% / 6% fall / dtw 0.183 | 0% / 0% fall / dtw 0.217 |
| tempo | 0.5x | 100% / 0% fall / dtw 0.130 | 100% / 0% fall / dtw 0.105 | 100% / 0% fall / dtw 0.129 |
| tempo | 2x | 100% / 0% fall / dtw 0.142 | 100% / 0% fall / dtw 0.102 | 100% / 0% fall / dtw 0.135 |
| transfer | mirrored | 100% / 0% fall / dtw 0.088 | 100% / 0% fall / dtw 0.066 | 100% / 0% fall / dtw 0.083 |

