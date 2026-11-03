# Voiceover script for the demo video

Cut: `My Movie.mp4` as of 2026-11-02 13:49, 3:00.8 long. Timecodes are what is on screen in that cut; the
animated version (`outputs/demo_kit/v2/athlete2humanoid_demo_v2.mp4`) keeps every section and cue at the same
second, so the same timings hold for both. Read at a normal pace (about 150 words a minute); the narration lands a
little ahead of each cut, which is where you want it.

| Time | On screen | Say |
|---|---|---|
| 0:00 – 0:07 | THE PROBLEM: "Teaching a humanoid one new skill" | Humanoid robots can do incredible things. But teaching one a single new skill, a jump, a kick, a dance move, still takes a whole lab. |
| 0:08 – 0:31 | The four cards land: Data collection 0:08, Retargeting 0:14, Reward design 0:18, Training 0:25 | First, data collection: a motion-capture studio, suits and markers. Then retargeting: cleaning that data up and fitting it to the robot's body. Then reward design: a reward function written and tuned by hand, for every single skill. And finally training: reinforcement learning in simulation, and then checking whether it holds up. |
| 0:32 – 0:38 | Weeks of work. Specialized equipment. Expert time. | Weeks of work, specialised equipment and expert time. For one skill. |
| 0:38 – 0:49 | A Unitree G1 fetching a pillow (footage: Unitree Robotics, credited on screen) | And the next skill starts the loop all over again. |
| 0:49 – 0:54 | "What everyone has" … "a phone." | But there is one thing everyone already has. A phone. |
| 0:55 – 1:05 | OUR SOLUTION: athlete2humanoid | So we built athlete2humanoid. Any phone video of a person doing an action goes in. Out comes everything a Unitree G1 needs to learn it, built on Meta's SAM 3.1, SAM 3D Body, V-JEPA 2 and Muse Spark. |
| 1:06 – 1:25 | Three cards: A training environment 1:06, A dataset 1:14, A robust policy 1:21 | A training environment: the action as an RL task, with its key moments, a reward and the robot's reference motion. A dataset: dozens of variants of the move, mirrored, re-timed and rescaled, each one checked against physics. And a robust policy, trained with RL, that recovers from shoves and handles variants it never saw. |
| 1:26 – 1:30 | "Demo", then the home page | Here it is, live. |
| 1:30 – 1:36 | The Live page; Finder; dancer.mov dropped in | Take a clip off a phone and drop it in. |
| 1:36 – 1:47 | Video check, Skeleton 1:40, Moments 1:44 | SAM 3.1 segments and tracks the person. SAM 3D Body lifts them into 3D, seventeen joints in metres, and proportion-aware inverse kinematics rebuilds the motion on the G1's twenty-nine joints, under a centimetre off. |
| 1:48 – 2:04 | Policy: attempts at iteration 500 to 1300 of 1500; ghosts; key-moment captions | Now the robot learns. The retargeted motion is read off as an ordered task of key moments, and Meta's Muse Spark watches the clip and names each one: turn right, left foot back. Then four thousand simulated G1s train with PPO in mjlab, rewarded for hitting those moments in order, without falling. These are its own attempts, one every fifty iterations. |
| 2:04 – 2:20 | The trained policy; Nudge, Shove, Push | V-JEPA 2, Meta's video world model, is in the loop three ways. Its embedding of the clip is an input the policy sees, past frames only. A style term rewards looking like the video. And it is the judge of the result. Shove the trained policy, and it recovers. |
| 2:20 – 2:44 | Scroll: 7 key moments, 36 of 36 variants, the reward curve, Download training environment | We also tried V-JEPA as a true world model, imagining the robot's future. It did worse than copying the present, so it stays as input and judge. And the pipeline is agentic: between training rounds, Muse Spark reads the scores, watches the rollout video, and retunes the reward with tool calls. Over three rounds it beat an untouched control on likeness to the video. |
| 2:44 – 2:49 | Download; Finder; the unzipped folder | And it is yours. One download: the reference motion, the task spec, the variants, the reward config and a loader. Train it yourself with mjlab. |
| 2:49 – 3:00 | Closing card | athlete2humanoid. Any video in. A humanoid skill out. |

About 440 words in all. The first half reads with room to spare; the build stage from 1:36 to 2:49 is dense,
about 165 words in 73 seconds, a brisk but clear read. If it feels rushed, drop "Shove the trained policy, and it
recovers." and the "one every fifty iterations" sentence.

## Notes on the technical lines

- **Muse naming.** On screen, the dance build's moment captions come from the task compiler, not Muse; Muse's
  names are on the showcase clips (10 of 10 clips named, `named_by: muse-spark-1.3` in their task specs). The
  line is true of the pipeline, so say it over the captions as written, or cut to the front-kick build's Moments
  stage, whose caption reads "named by muse-spark-1.3".
- **V-JEPA 2 three ways.** Policy input (the clip's latent, past frames only, no future leakage), style reward
  term, and judge of likeness (`docs/VJEPA_CONDITIONING.md`, `docs/SEMANTIC_REWARD.md`).
- **World model.** V-JEPA 2's pretrained predictor, used to imagine the robot's future, had 2 to 6 times the
  error of copying the present frame, and feeding its imagined future to the policy did not help
  (`docs/FRONTIER_WORLD_MODEL.md`). Saying this out loud is a strength for judges: it is an honest negative.
- **Reward agent.** Muse Spark, through the Responses API with bounded tool calls, read each round's evaluation
  and rollout video and changed reward weights and termination settings with a stated reason. Jumping jacks,
  three rounds, 900 PPO iterations: likeness 0.410 against 0.388 for the control, one run per arm, about $0.10
  of API (`docs/MUSE_AGENT.md`). If asked, the gain came from upweighting the likeness terms.

## The closing card

`outputs/demo_kit/closing/closing_card.mp4`, 18 s, in the site's pixel style: the headline types out next to the
squat-jump figures from the home page, then the tagline, the two links and a credit line. Everything is on screen
by 8 s and it fades out over the last second, so trimming it from the end works.

- Headline: **Any video in. A humanoid skill out.**
- Tagline: **No mocap studio. No hand-written reward. A phone.**
- Links: athlete2humanoid.pages.dev and github.com/PandaGrdn/athlete2humanoid
- Credit line: Built with Meta SAM 3.1, SAM 3D Body and V-JEPA 2. Trained in mjlab on the Unitree G1.
