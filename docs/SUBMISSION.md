# Submission: athlete2humanoid

Track: **Push the Frontier** (robotics). Bonus prizes we're entering: Best Agentic Use-Case, Best Open Weights, Most Viral.

**Tagline:** Any video of a person moving becomes a physics RL policy for a Unitree G1 humanoid that learns the action, not the clip.

## Inspiration

Robots still learn new skills from motion capture studios or hand-written reward functions. Meanwhile, people film millions of hours of movement every day. We asked whether a humanoid could learn a move from one phone video.

The obvious approach, copying the clip joint by joint, gives a robot that needs the clip. Shove it, start it from an odd pose, or take the reference away, and it has nothing to fall back on. We wanted a robot that learns *what the action is*: squat, jump, land, stand back up, without falling.

## What it does

You give it a video of a person, for example a squat jump. It produces trained reinforcement-learning policies that run the Unitree G1 humanoid in full physics simulation.

- **SAM 3.1** finds and tracks the person.
- **Retargeting in the robot's proportions.** The pose is rebuilt on the G1's own bone lengths, so joint angles carry over and planted feet stay planted.
- **A task spec.** The motion becomes an ordered list of stages with goals in G1 units: stand → squat → jump → land → stand, twice.
  - **Muse Spark**, Meta's multimodal model, names the stages and its spec is cross-checked against the kinematics.
- **A reward machine** trains the robot to complete those stages in order, staying upright.
  - **V-JEPA 2**, Meta's video world model, adds a light "look like the video" style term.
  - Goal relevance weights that term toward the moments that matter: the push before takeoff and the landing, not arm-waving for the crowd.
- **An evaluation of the claim.** 64 rollouts × 18 conditions: shoves, odd starts, tempo changes, friction and mass, and the reference clip frozen.
- **A web app**: upload a video, follow the job, see the robot. The examples page works offline.

This is infrastructure for robot learning: video in, policy out. Nothing in it gives feedback to the person in the video.

## How we built it

- **Tracking:** SAM 3.1 with a text prompt. It keeps the track visible in the most frames and runs in its own Python 3.12 environment.
- **Proportion-aware retargeting:** the person's bone directions on the G1's bone lengths, stance feet pinned, the lower body tilted rather than one leg stretched.
  - Joint-angle error is 3–10× lower than uniform scaling.
  - Foot sliding falls from 29 to 0.4 cm/s.
- **Task spec and reward machine** (`skill/task.py`):
  - Stages are detected from the retarget's pelvis and feet. Glitches shorter than a real flight are ignored, and jump clearance is clipped to what the G1 can do.
  - **Progress is potential-based**, so bobbing up and down earns nothing.
  - **Bonuses:** one per completed stage; one per jump, scaled by a target height band; one for finishing the sequence.
  - **Gating:** everything is gated on staying upright, and a fall refunds the attempt.
  - **The stage is part of the policy's observation**, so the reward stays Markov.
- **Goal relevance** (`skill/importance.py`): a physics attribution of which moments and body parts change each goal event's outcome. We tested masking V-JEPA 2's predictor as a second signal, but it didn't separate relevant motion, so we report it as a diagnostic.
- **RL:** custom mjlab tasks (MuJoCo Warp, PPO, 4,096 envs).
  - **Stability terms:** alive bonus, fall penalty, a clip-relative fall rule, foot slip, and a push curriculum.
  - **Episodes end only on a fall,** never for leaving the retarget.
  - **Warm starts:** every policy builds off saved weights. A policy whose inputs grew starts out acting exactly like its source.
- **Ops:** `scripts/remote_run.py` drives a Colab A100 (or Stanford FarmShare via Slurm) from a laptop, unattended.
  - It keeps the VM alive, syncs checkpoints mid-run, and retries from them if the VM is lost.
  - CI runs the tests, lint, frontend build and image build. The backend deploys to a Hugging Face Space, the frontend to Cloudflare Pages.

## Challenges we ran into

- **A video-similarity reward alone isn't enough.** Our first two rounds rewarded the robot for its V-JEPA 2 embedding matching the video's.
  - Almost any humanoid motion scores about 0.5, so the reward barely separates "the action" from "a person moving".
  - When we retrained the distilled reward on the policies' own rollouts, the policy gamed it: the learned reward rose from 0.35 to 0.49 while real V-JEPA stayed flat (0.35 → 0.37).
  - That pushed us to the task spec.
- **The embodiment gap.** A person's arms and legs aren't a G1's. Uniform scaling put targets out of reach and let feet hover about 10 cm above the floor between jumps; proportion-aware retargeting fixed both.
- **Jump, then crash.** The first squat-jump task policies over-jumped (0.34 m against a 0.30 m goal) and collapsed on landing, because the jump paid before the landing did. Stability terms and warm starts fixed training. Now a fall also refunds the attempt.
- **Long GPU jobs from a laptop.** Colab reclaims idle VMs, and its kernel is busy while a job runs. A keeper cell, download-only checkpoint sync and automatic retry keep multi-hour jobs alive with nobody watching.

## Accomplishments that we're proud of

**A G1 that learned a squat jump from one phone video, and still does it without the clip.**

| Policy | Stages (clean) | Random pushes | Push 1.5 m/s | Start noise | **Reference frozen** | Joint distance to clip |
|---|---|---|---|---|---|---|
| Tracking (copies the clip) | 9/9 | 65% | 14% | 54% | **0%** | 0.10 rad |
| Task (rewarded for the action) | 9/9 | **94%** | **81%** | **83%** | **100%** | 0.41 rad |
| Task, never sees the clip | 9/9 | 83% | 55% | 73% | **100%** | 0.41 rad |

Success means every stage in order with no fall, over 64 rollouts per condition.

- **Without the clip,** the copier stands still while both task policies squat, jump, land and stand.
- **Their own way.** They do the action far from the clip's joint angles: 0.41 rad, more than standing still (0.17).

## What we learned

- **Reward the action, use the video for style.** A semantic task spec in the robot's own units, plus a light style term, gave both success and robustness. Similarity alone gave neither.
- **Measure the claim directly.** "Learns the action, not the clip" became testable once we froze the reference and shoved the robot.
- **Learned rewards get hacked.** Any distilled reward needs a held-out real metric and penalties for uncertainty. We added a dropout-uncertainty penalty, which shrinks the gamed gap from +0.11 to about 0.
- **Proportions matter.** Retargeting on the robot's skeleton fixed foot contacts that every downstream stage depended on.

## What's next for athlete2humanoid

- **Commanded targets:** one policy for a range of squat depths and jump heights, under a curriculum. Already implemented as `A2H-TaskCmd`.
- **Live Muse:** the skill reader on new videos, and the reward-design agent closing the loop on real runs. It's integrated and tested against recorded fixtures; the live run is waiting on an API key.
- **More actions** (lunge, kick, jumping jacks), and several videos of one action into one policy.
- **Sim-to-real** on a physical G1.

## Built with

Python · PyTorch · **SAM 3.1** · **V-JEPA 2** · **Muse Spark** (Meta Model API) · Hugging Face Transformers · MediaPipe · mink · MuJoCo · MuJoCo Warp · mjlab · MuJoCo Menagerie (Unitree G1) · FastAPI · React · Google Colab (A100) · Stanford FarmShare (Slurm) · GitHub Actions · Hugging Face Spaces · Cloudflare Pages

Code and weights: https://github.com/PandaGrdn/athlete2humanoid (policy weights to be published on the Hugging Face Hub).

---

## Tweet thread draft (Most Viral)

Each tweet is at most 280 characters. Attach media where noted.

**1/** *(video: phone clip | robot squat-jumping)*
One phone video of a squat jump. Hours later, a Unitree G1 humanoid learned to squat, jump, land and stand in physics sim. No mocap, no hand-written controller. 🧵

**2/** *(video: tracking robot standing still | task robot jumping, reference frozen)*
The test: freeze the reference clip. The robot that copies the video just stands there (0%). Ours still does the whole squat jump (100%). It learned the action, not the clip.

**3/**
How: Meta's SAM 3.1 tracks the person; we rebuild their pose on the robot's own skeleton; the motion becomes stages (squat → jump → land) with goals in robot units. The robot is rewarded for doing them, without falling.

**4/** *(image: push robustness chart)*
Shove it at 1.5 m/s: the clip-copier finishes 14% of the time, ours 81%. Meta's V-JEPA 2 keeps the style close to the video where it matters: takeoff and landing, not arm-waving.

**5/**
Code + weights are open: github.com/PandaGrdn/athlete2humanoid. Built for the Meta Global AI Developer Hackathon. #MetaAI
