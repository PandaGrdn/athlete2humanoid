# Squat jump: final run evaluation

Copied from `outputs/remote_latent_final/eval.json` (evaluation rollout from the start of the clip, scored with the real V-JEPA 2 and the task spec).

| motion | aligned | dtw | dtw_rank_vs_bank | task_success | task_stages | task_jump_quality | falls | mean_joint_error_rad | mean_torso_position_error_m |
|---|---|---|---|---|---|---|---|---|---|
| retarget | 0.732 | 0.732 | 1 | True | 9/9 | [1.0, 1.0] | - | - | - |
| tracking | 0.696 | 0.696 | 1 | True | 9/9 | [0.872, 0.86] | 0 | 0.107 | 0.106 |
| task | 0.431 | 0.431 | 11 | True | 9/9 | [0.51, 0.682] | 0 | 0.420 | 0.407 |
| taskphase | 0.312 | 0.325 | 20 | True | 9/9 | [0.791, 1.0] | 0 | 0.419 | 1.059 |
