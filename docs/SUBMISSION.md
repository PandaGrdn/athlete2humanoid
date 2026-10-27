# Submission: athlete2humanoid

Track: **Push the Frontier** (robotics). Bonus prizes we're entering: Best Agentic Use-Case, Most Viral.

**Tagline:** Any video of a person moving becomes a physics RL policy for a Unitree G1 humanoid.

## Inspiration

Robots still learn new skills from motion capture studios or hand-written reward functions. Meanwhile, people film millions of hours of movement for free every day. We asked whether a humanoid could learn a move from one phone video, and whether we could reward it for *doing the action* instead of replaying the clip joint by joint. Meta's V-JEPA 2 understands motion in video without labels, so we used it as the judge of "does the robot look like it's doing the same thing?"

## What it does

You give it a video of a person: a squat, a lunge, a kick. It produces a trained reinforcement-learning policy that runs the Unitree G1 humanoid in full physics simulation.

- **SAM 3.1** finds and tracks the person.
- The motion is retargeted onto the G1 with inverse kinematics.
- The video is turned into a **V-JEPA 2** latent trajectory, which becomes a reward: "move so that your silhouette video embeds like the human's."
- Three policies are trained on GPU, each with a different mix of rewards: copy the joints, latent reward only, or a hybrid of the two. Each is scored by the real V-JEPA 2 against a bank of 26 other motions.
- *(In progress)* **Muse Spark** reads the video into a training spec, then acts as a reward-design agent. It watches the robot's attempts, reads the scores and rewrites the reward between rounds.

This is infrastructure for robot learning: video in, policy out. Nothing in it gives feedback to the person in the video.

## How we built it

- **Tracking:** SAM 3.1 with a text prompt ("athlete", falling back to "person"). It keeps the track visible in the most frames and runs in its own Python 3.12 environment.
- **Pose and retargeting:** MediaPipe 3D pose plus 2D root motion gives a metric trajectory. mink IK drives nine G1 frames and the torso orientation, and the result is exported as an mjlab motion.
- **Latent target:** human (SAM masks) and robot (MuJoCo segmentation) are rendered as identical silhouette videos at matched scale and viewpoint. V-JEPA 2 ViT-L embeds them at 15 steps per second, and each domain is centred on its own mean.
- **Distilled reward:** V-JEPA can't run inside RL with 4,096 parallel robots, so we render 253 robot motions (an IK bank, the clip retargets, and 220 randomised variations including falls and freezes). From them we train an MLP that maps ~1 s of body positions to V-JEPA's latent, projected to 64 principal components.
- **RL:** custom mjlab tasks (MuJoCo Warp, PPO) add the latent reward and relax terminations. Tracking, hybrid and latent-only policies train in parallel on one A100: 2,000 iterations × 4,096 envs, about 2.3 h.
- **Evaluation:** rollouts are re-rendered and scored with the real V-JEPA 2, never the distilled net, using time-aligned cosine and DTW rank against the bank.
- **Ops:** `scripts/remote_run.py` drives a Colab A100 from a laptop, unattended. It waits for clips, checks tokens, launches the job, keeps the VM alive, retries if the VM is lost, and pulls the results back.

## Challenges we ran into

- **Embodiment gap.** A human and a robot look nothing alike to a video model. Silhouettes at a matched body scale, plus per-domain mean-centring, were what made a human squat embed closest to a robot squat. Before that, raw cosine was 0.87–1.0 for everything.
- **V-JEPA in the RL loop.** It is much too slow per step, so we distilled it. The distilled net reaches only 0.62 cosine on held-out motions, and it is the main reason the latent-only policy trails tracking.
- **IK singularities.** The G1's straight-knee "stand" pose can't start a knee bend, so we seed IK from a slight crouch.
- **Dependency conflicts.** SAM 3.1 pins `numpy<2`, which has no build for Colab's Python 3.13, and FlashAttention 3 needs an H100. We split the work into three environments and disabled FA3.
- **Long GPU jobs from a laptop.** Colab reclaims idle VMs, so a keeper cell, status polling and an automatic retry keep a 3-hour job alive with nobody watching.

## Accomplishments that we're proud of

- **A robot that learned a squat from a video embedding alone.** The latent-only policy has no tracking reward, never falls, and real V-JEPA ranks it closer to the human video than all 26 bank motions.
- **A complete pipeline from video to policy:** phone video to three trained G1 policies and an honest evaluation, in one unattended command.
- **Measured results, not vibes.** Every policy is scored by an independent model (real V-JEPA 2) against a 26-motion bank, with falls and torso error reported alongside.

| Robot motion | V-JEPA cosine to video | DTW rank vs 26 motions | Falls | Torso error |
|---|---|---|---|---|
| Retarget (kinematic) | 0.726 | #1 | n/a | n/a |
| Tracking policy | 0.714 | #1 | 0 | 7.1 cm |
| Hybrid policy | 0.579 | #1 | 0 | 13.3 cm |
| Latent-only policy | 0.532 | #1 | 0 | 16.6 cm |
| Latent-only, DAgger reward net | TODO(v2) | TODO(v2) | TODO(v2) | TODO(v2) |

## What we learned

- **V-JEPA 2 does transfer across embodiments, but only with care.** Matched silhouettes, the same viewpoint and per-domain centring all mattered. In a 7-clip probe, the human clip's own robot retarget ranked #1 by DTW for 6 of 7 clips. The miss was a subtle arm-raise.
- **A distilled reward is only as good as its training states.** The policy explores states the IK bank never showed, which is why we're now training a DAgger-refined net on the policies' own rollouts.
- **Tracking is a strong baseline.** The interesting question is not "beat tracking on its own metric". It is whether a latent reward lets the robot depart from a noisy retarget where that still looks like the move.

## What's next for athlete2humanoid

- DAgger-refined reward net (training now). TODO(v2): results.
- Muse Spark skill reader and reward-design agent loop, scored by real V-JEPA so it can't game its own reward (in progress).
- Train the six other clips we've already processed: lunge, front kick, jumping jacks, punches, squat jump, arms.
- Generalisation tests on variations the clip never showed (depth, height, pushes), then sim-to-real on a physical G1.
- An open-weights rollout critic with Muse Glimmer.

## Built with

Python · PyTorch · **SAM 3.1** · **V-JEPA 2** · **Muse Spark** (Meta Model API) · Hugging Face Transformers · MediaPipe · mink · MuJoCo · MuJoCo Warp · mjlab · MuJoCo Menagerie (Unitree G1) · Google Colab (A100) · OpenAI Python SDK (for the Meta Model API)

---

## Tweet thread draft (Most Viral)

Each tweet is at most 280 characters. Attach media where noted.

**1/** *(video: phone clip | robot squatting)*
One phone video of a squat. A few hours later, a Unitree G1 humanoid learned the squat in physics sim. No mocap, no hand-tuned controller. Just the video. 🧵

**2/**
Step 1: Meta's SAM 3.1 tracks the person. We retarget their pose onto the robot. But copying joints from one noisy phone video is brittle, so we wanted the robot to learn what the move *looks* like.

**3/** *(image: human silhouette | robot silhouette)*
So we turn both the human and the robot into silhouettes and ask Meta's V-JEPA 2, a video world model, "are these doing the same thing?" Its answer becomes the robot's reward.

**4/**
One catch: V-JEPA is far too slow to run 4,096 robots at 50 Hz. So we distilled it into a tiny net that predicts V-JEPA's view from the robot's body positions. Three policies, one A100, ~2.3 h.

**5/** *(image: results table)*
Result: a policy that never saw a joint-tracking reward learned a stable squat, with zero falls. Real V-JEPA ranks it closer to the video than all 26 other robot motions. Not as good as plain tracking yet. That's next.

**6/**
Next: a Muse Spark agent watches the robot's attempts and rewrites its reward, then we add more moves. Code is open: TODO(v2): repo link. Built for the Meta Global AI Developer Hackathon. #MetaAI
