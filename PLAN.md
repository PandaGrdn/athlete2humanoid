# Plan v2: humanoids that learn athletic skills from video

## Positioning (decides stage 1 pass/fail)
Kickoff listed **"sports analyzers or coaches"** as an overdone project. We are not one. This is **robot learning from internet video**: sports clips are just the hardest, most dynamic source of human motion. Muse Spark acts as the **reward engineer that trains the robot**, not as a coach for people.
- Never pitch, label, or demo it as "coaching". Rename `coach.py` to `reward_agent.py` when we touch it next.

## Pipeline (Meta models in bold)
1. **SAM 3.1** tracks the athlete through occlusion and crowds.
2. **SAM 3D Body** recovers a full-body 3D mesh and skeleton from each SAM-masked frame, which is far better than MediaPipe on fast sports poses. MediaPipe stays as the fallback.
3. **Retarget** onto the G1 with mink IK (works now), then clean up foot contacts and ground penetration.
4. **Physics tracking policy:** PPO in MuJoCo MJX (Playground G1). Train one clip first, then one policy across a library of 5+ moves, with domain randomization.
5. **Muse Spark reward agent** (Eureka-style): it sees the clip frames plus rollout telemetry (where the robot fell, which joints lagged), rewrites the reward weights or code, and retrains. We measure tracking error per iteration against a hand-tuned baseline. This is the headline technical result.
6. *Stretch:* **fine-tune a Llama model** (LoRA on Modal credits) into a fast "robot failure critic", trained on thousands of simulator rollouts. The simulator labels the data for free. Evaluate it against Muse Spark on held-out rollouts.
- *Fallback for step 4:* **Meta Motivo** tracks the motion zero-shot in physics on its own humanoid. This only works if MHR → SMPL conversion is easy, so check it on day 4.

## Milestones
| Day | Date | Goal | Done when |
|---|---|---|---|
| 1 | Oct 26 | Credits claimed, themes confirmed, notebook green, clips collected | all day-1 checks pass |
| 2 | Oct 27 | SAM 3.1 + SAM 3D Body on the squat clip | `(T, joints, 3)` saved; compared with MediaPipe |
| 3 | Oct 28 | Squat retargeted, side-by-side video | **A: squat replay** |
| 4 | Oct 29 | Physics PD playback; MJX env; first PPO run | training curve going down |
| 5 | Oct 30 | Squat tracked in physics; sports clips through steps 1–3 | **B: backup demo video recorded** |
| 6 | Oct 31 | Muse Spark reward-agent loop on 2 sports moves | error drops across iterations |
| 7 | Nov 1 | Multi-move policy (kick, jump, lunge, ...) + domain randomization | **C: 3+ sports moves in physics** |
| 8 | Nov 2 | Llama critic fine-tune *or* more moves and metrics | **feature freeze at end of day** |
| 9 | Nov 3 | App, README, results charts, 2–3 min video | |
| 10 | Nov 4 | Submit by midday | |
| | Nov 5–8 | Rehearse the live demo (top 5 present live on Nov 9) | demo runs in under 2 min |

**Cut order if behind:** Llama fine-tune → multi-move policy → domain randomization. Milestones A and B are never cut.

## Compute
- **Colab** (L4/A100): day-to-day development.
- **Modal** ($100, for Meta open-weights models): SAM 3.1 on H100 (FlashAttention 3 works there), SAM 3D Body batch runs, Llama fine-tune.
- **Meta Model API** ($150): Muse Spark reward-agent calls.
- RL training: Colab A100 or Modal, whichever credits allow.
