# Robustness: punches

16 envs per policy per condition, 6 conditions. Task: 6 key moments: ready -> right hand up, left hand up -> left hand right, left foot back -> left hand left, right hand right -> left hand right, right hand back -> left hand left, right hand down -> turn left, step forward (finish) (spec: task_spec.json).

Task success (fall rate) per suite, averaged over its conditions

| Policy | Clean | Pushes | Tempo 0.5-2x | No reference | DTW to clip | Joint err | V-JEPA |
|---|---|---|---|---|---|---|---|
| multi_1500 | 0% (0% fall) | 3% (9% fall) | 0% (0% fall) | 0% (0% fall) | 0.104 | 0.104 | - |
| scratch_1500 | 0% (0% fall) | 0% (50% fall) | 0% (0% fall) | 0% (6% fall) | 0.078 | 0.079 | - |
| specialist_1500 | 0% (0% fall) | 0% (19% fall) | 0% (0% fall) | 0% (0% fall) | 0.099 | 0.099 | - |

- Success: every stage of the task spec done in order, from the stage the clip is in at the start, with no fall (torso below 0.35 m or tilted past 1.4 rad) before the clip ends.
- Pushes: random shoves every 1-3 s; stage-timed pushes: one per squat (halfway down) or per touchdown; start noise: random start in the first half of the clip, 1-4x training's init noise; start pose: crouched or standing still at t = 0; no reference: standing still and the reference frozen at t = 0.
- DTW to clip: joint-space DTW to the retarget on the clean clip, mean rad per matched frame, over rollouts that stayed up; standing still scores 0.170. Joint err: the same, time-locked. V-JEPA: aligned cosine with the video, clean clip.

## Per condition

| Suite | Level | multi_1500 | scratch_1500 | specialist_1500 |
|---|---|---|---|---|
| clean | nominal | 0% / 0% fall / dtw 0.104 | 0% / 0% fall / dtw 0.078 | 0% / 0% fall / dtw 0.099 |
| push | 1 m/s | 0% / 0% fall / dtw 0.121 | 0% / 0% fall / dtw 0.102 | 0% / 0% fall / dtw 0.115 |
| push | 2 m/s | 6% / 19% fall / dtw 0.139 | 0% / 100% fall | 0% / 38% fall / dtw 0.132 |
| noref | frozen | 0% / 0% fall / dtw 0.194 | 0% / 6% fall / dtw 0.183 | 0% / 0% fall / dtw 0.217 |
| tempo | 0.5x | 0% / 0% fall / dtw 0.130 | 0% / 0% fall / dtw 0.105 | 0% / 0% fall / dtw 0.129 |
| tempo | 2x | 0% / 0% fall / dtw 0.142 | 0% / 0% fall / dtw 0.102 | 0% / 0% fall / dtw 0.135 |

