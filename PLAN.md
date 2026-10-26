# Plan v3: video in, trained humanoid policy out

## Fit (stage 1 is pass/fail)
- **Problem statement: Push the Frontier.** The guide names robotics and asks us to "give a machine a new capability". We turn any video of a human move into a physics policy for a real humanoid, the Unitree G1.
- **Never a "sports analyzer/coach".** That's on the anti-project list. Sports clips are just the hardest human motion to learn from, and nothing in the product coaches people.
- **Counted Meta models:** Muse Spark/Wonder/Image/Voice Transcribe and SAM 3.1 via the Meta Model API, and Muse Glimmer / Muse Spark Open as open weights. Other Meta models (SAM 3D Body, Motivo) are extras, not the qualifying use.

## Rubric → what we optimise
| Criterion | Weight | Our answer |
|---|---|---|
| Impact | 30% | Teaching robots new skills from ordinary video instead of motion capture or hand-tuned controllers |
| Use of Meta AI tooling | 25% | SAM 3.1 tracking; a **Muse Wonder/Spark agent** that watches the robot's rollouts and rewrites its reward; a **fine-tuned Muse Glimmer/Spark Open** critic |
| Technical depth | 20% | IK retargeting, GPU RL (mjlab, MuJoCo Warp), agent-in-the-loop reward design, measured tracking error |
| Demo | 25% | 3-min video: clip → tracking → robot copies it in physics; agent improves a failed skill live |

Bonus prizes in reach: Best Agentic Use-Case ($5K), Best Build on Meta Open Weight Models ($10K), Most Viral ($20K, post on X).

## Pipeline
1. **SAM 3.1** tracks the athlete → masked crops (`track.py`) ✅
2. MediaPipe 3D pose + 2D root motion → metric human trajectory (`pose.py`) ✅. Upgrade to SAM 3D Body if time allows.
3. mink IK onto the G1 (feet, knees, pelvis, elbows, hands, torso orientation) → mjlab CSV (`retarget.py`) ✅
4. **mjlab** PPO motion tracking on GPU → policy, video, metrics (`notebooks/video_to_policy.ipynb`, `mjlab_rollout.py`) ✅ written, **not yet run on a GPU**
5. **Reward agent:** Muse Wonder watches the reference and rollout videos and reads the metrics, then edits reward weights and termination thresholds and retrains. Each round is logged.
6. **Open-weights critic:** fine-tune Muse Glimmer or Spark Open on simulator-labelled rollouts (Modal / Fireworks).

## Milestones (submit by Wed Nov 4, 9 PM ET)
| Day | Date | Goal |
|---|---|---|
| 1 | Mon Oct 26 | **One clip → trained policy, end to end** (squat) |
| 2 | Oct 27 | Policy quality on the squat; 3 sports clips through steps 1–3; Agents 101 session |
| 3 | Oct 28 | Reward agent v1 (Muse Wonder/Spark) on a move the default reward fails |
| 4 | Oct 29 | Agent loop measurably improves tracking; **backup demo video recorded** |
| 5 | Oct 30 | Multi-move: kick, jump, lunge; Fireworks fine-tuning session |
| 6 | Oct 31 | Rollout dataset for the critic; start the fine-tune |
| 7 | Nov 1 | Critic fine-tuned and compared with Muse Spark |
| 8 | Nov 2 | Upload-a-clip demo app; **feature freeze** |
| 9 | Nov 3 | README, charts, 3-min video, X post |
| 10 | Nov 4 | Buffer, submit before 9 PM ET |

**Cut order if behind:** critic fine-tune → multi-move → app polish. Never cut: Day 1 end-to-end, and the Day 4 backup video.

## Compute
- Colab A100: mjlab training. Modal ($100): batch SAM and the fine-tune. Meta API ($150): agent calls. Fireworks: Muse Open fine-tuning, if selected.
