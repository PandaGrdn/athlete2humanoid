## shot_put (1500 iterations from multi; reference travels 1.12 m; variants {'stock': 24, 'synced': 24})

| Policy | Task | The clip | Held-out variants | Pushed 1 m/s | Tempo 0.5x / 2x | Start noise | Stages done | Torso error (m) | Travel done | End error (m) | Joint error (rad) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | Travel | 100% (0%) | 40% (0%) | 94% (2%) | 50% (2%) | 89% (3%) | 100% | 0.09 | 96% | 0.05 | 0.216 |
| base | Tracking | 8% (0%) | 7% (0%) | 23% (0%) | 14% (4%) | 77% (0%) | 65% | 0.31 | 73% | 0.42 | 0.184 |
| base_pro | compare | 0% (0%) | 16% (5%) | 8% (3%) | 32% (33%) | 66% (8%) | 49% | 0.38 | 63% | 0.45 | 0.189 |
| phase | TravelPhase | 81% (0%) | 40% (0%) | 53% (0%) | 50% (4%) | 80% (5%) | 94% | 0.23 | 97% | 0.23 | 0.194 |
| reward | TravelReward | 77% (0%) | 40% (0%) | 64% (0%) | 50% (34%) | 88% (0%) | 92% | 0.14 | 95% | 0.09 | 0.210 |
| term | TravelTerm | 100% (0%) | 40% (0%) | 73% (5%) | 48% (52%) | 83% (11%) | 100% | 0.12 | 93% | 0.08 | 0.201 |

Cells: task success (fall rate). Stages, torso and joint error: the clip, clean.

## lunge (1500 iterations from multi; reference travels 0.09 m; variants {'stock': 15, 'synced': 24})

| Policy | Task | The clip | Held-out variants | Pushed 1 m/s | Tempo 0.5x / 2x | Start noise | Stages done | Torso error (m) | Travel done | End error (m) | Joint error (rad) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | Travel | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (5%) | 12% | 0.10 | 220% | 0.10 | 0.178 |
| all_synced | Travel@synced | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (3%) | 12% | 0.09 | 185% | 0.08 | 0.199 |
| base | Tracking | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 13% | 0.11 | 217% | 0.10 | 0.155 |
| synced | Tracking@synced | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 14% | 0.12 | 239% | 0.13 | 0.170 |

Cells: task success (fall rate). Stages, torso and joint error: the clip, clean.

## squat_jump (1500 iterations from multi; reference travels 0.06 m; variants {'stock': 36})

| Policy | Task | The clip | Held-out variants | Pushed 1 m/s | Tempo 0.5x / 2x | Start noise | Stages done | Torso error (m) | Travel done | End error (m) | Joint error (rad) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | Travel | 100% (0%) | 93% (0%) | 47% (3%) | 50% (0%) | 94% (0%) | 88% | 0.06 | 34% | 0.05 | 0.144 |
| base | Tracking | 100% (0%) | 77% (0%) | 67% (9%) | 50% (0%) | 92% (0%) | 80% | 0.11 | -20% | 0.10 | 0.112 |

Cells: task success (fall rate). Stages, torso and joint error: the clip, clean.
