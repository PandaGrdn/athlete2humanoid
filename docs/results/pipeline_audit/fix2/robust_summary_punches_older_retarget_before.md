# Robustness: punches

16 envs per policy per condition, 6 conditions. Task: 6 key moments: ready -> right hand up, left hand up -> left hand right, left foot back -> left hand left, right hand right -> left hand right, right hand back -> left hand left, right hand down -> turn left, step forward (finish) (spec: task_spec.json).

Task success (fall rate) per suite, averaged over its conditions

| Policy | Clean | Pushes | Tempo 0.5-2x | No reference | DTW to clip | Joint err | V-JEPA |
|---|---|---|---|---|---|---|---|
| multi_300 | 0% (0% fall) | 0% (9% fall) | 0% (0% fall) | 0% (0% fall) | 0.180 | 0.181 | - |
| tracking_var_300 | 0% (0% fall) | 0% (31% fall) | 0% (0% fall) | 0% (0% fall) | 0.139 | 0.139 | - |

- Success: every stage of the task spec done in order, from the stage the clip is in at the start, with no fall (torso below 0.35 m or tilted past 1.4 rad) before the clip ends.
- Pushes: random shoves every 1-3 s; stage-timed pushes: one per squat (halfway down) or per touchdown; start noise: random start in the first half of the clip, 1-4x training's init noise; start pose: crouched or standing still at t = 0; no reference: standing still and the reference frozen at t = 0.
- DTW to clip: joint-space DTW to the retarget on the clean clip, mean rad per matched frame, over rollouts that stayed up; standing still scores 0.170. Joint err: the same, time-locked. V-JEPA: aligned cosine with the video, clean clip.

## Per condition

| Suite | Level | multi_300 | tracking_var_300 |
|---|---|---|---|
| clean | nominal | 0% / 0% fall / dtw 0.180 | 0% / 0% fall / dtw 0.139 |
| push | 1 m/s | 0% / 19% fall / dtw 0.199 | 0% / 12% fall / dtw 0.157 |
| push | 2 m/s | 0% / 0% fall / dtw 0.215 | 0% / 50% fall / dtw 0.177 |
| noref | frozen | 0% / 0% fall / dtw 0.227 | 0% / 0% fall / dtw 0.214 |
| tempo | 0.5x | 0% / 0% fall / dtw 0.218 | 0% / 0% fall / dtw 0.166 |
| tempo | 2x | 0% / 0% fall / dtw 0.230 | 0% / 0% fall / dtw 0.187 |

