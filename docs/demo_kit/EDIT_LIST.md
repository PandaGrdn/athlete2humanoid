# Demo video kit: edit list

One clip per shot of the storyboard in [`docs/DEMO_SCRIPT.md`](../DEMO_SCRIPT.md), plus a rough cut. All clips are
1920×1080, 30 fps, H.264, no audio, and trimmed to the storyboard's length. They live in `outputs/demo_kit/`, which is
gitignored. The voice-over and the final cut are yours: the shot clips are clean, and only the rough cut has captions.

- **Rough cut:** `outputs/demo_kit/rough_cut.mp4`, **3:00.00** against the 3:00 target. It has no voice-over. Title
  cards come from the storyboard, and the numbers are burned in as lower-third captions.
- **Rebuild:** `python scripts/demo_kit/build_kit.py` re-cuts every shot and the rough cut. Pass shot numbers to
  rebuild only those. Trims, captions and card text are in its `SHOTS` table.
- **Re-record a take:** `scripts/demo_kit/record_takes.mjs` (`build_sj`, `build_kick`, `showcase`, `live <clip>`). The
  header explains the setup.

## Shots

Times are positions in the rough cut. "Source" gives each segment in order, as `file [in–out s]`. For the raw takes, `build_kit.py` resolves these from the take's `.marks.json` (an action plus an offset), so a re-recorded take is cut at the same moments.

| # | Time | File | Length | Source |
|---|---|---|---|---|
| 1 | 0:00–0:12 | `shot01_hook.mp4` | 12.0 s | `demo/final/grid.mp4` [0–9], at 0.75× speed |
| 2 | 0:12–0:30 | `shot02_problem.mp4` | 18.0 s | title card (5 s); `final/push.mp4` [0–9], left panel (tracking); `final/frozen_reference.mp4` [0–4], left panel. Both are cropped so the % labels stay hidden until shot 8 |
| 3 | 0:30–0:48 | `shot03_pipeline_build_sj.mp4` | 18.0 s | `raw/build_sj.mp4` [0.8–18.8]: the `/build/latent_s3_sj` replay, stages 01→06 |
| 4 | 0:48–1:03 | `shot04_retarget.mp4` | 15.0 s | `raw/build_sj.mp4` [21.3–27.3]: orbit on 03 Retarget; `retarget_compare/squat_jump/compare.mp4` [0–9], labelled person / uniform scale / G1 proportions |
| 5 | 1:03–1:18 | `shot05_goal_relevance.mp4` | 15.0 s | `docs/goal_relevance/squat_jump_hype_importance.png`, slow push-in |
| 6 | 1:18–1:33 | `shot06_reward_machine.mp4` | 15.0 s | `raw/build_sj.mp4` [27.9–33.2]: 04 Moments, with the ring and name at each moment; `raw/build_sj.mp4` [47.1–56.8]: scroll to the RL-environment panel (key-moment timeline and its rules) |
| 7 | 1:33–1:53 | `shot07_results_grid.mp4` | 20.0 s | `final/grid.mp4` looped twice, then the last frame held 1.9 s |
| 8 | 1:53–2:25 | `shot08_robustness.mp4` | 32.0 s | `final/frozen_reference.mp4` (+1 s hold); `final/push.mp4` (+3 s hold); `final/robustness.mp4` (+4 s hold) |
| 9 | 2:25–2:40 | `shot09_muse.mp4` | 15.0 s | `raw/build_kick.mp4` [28.0–33.0]: 04 Moments, with "named by muse-spark-1.3" in the caption box; `raw/build_kick.mp4` [49.6–55.6]: the kick's key-moment timeline with Muse's names; reward-agent card (4 s) |
| 10 | 2:40–3:00 | `shot10_live_and_end.mp4` | 20.0 s | `raw/live_upload.mp4` [1.6–23.4] at 3.63×: `clips/punches.mov` built live; `raw/build_kick.mp4` [39.0–43.5]: 06 Policy, Shove + Push; `raw/showcase.mp4` [23.3–27.8]: showcase front kick pushed; end card (5 s) |

Paths under `demo/`, `final/` and `retarget_compare/` are relative to `outputs/`. `final/` means `outputs/demo/final/`.

## Narration and on-screen numbers

The narration is the storyboard's. Lines marked *changed* were updated in `DEMO_SCRIPT.md` in this change.

| # | Narration | Captions (rough cut, seconds within the shot) |
|---|---|---|
| 1 | "One phone video of a squat jump. A few hours later, a Unitree G1 humanoid squats, jumps, lands and stands in full physics. No mocap, no hand-written controller." | 0.3–5.8 One phone video → a Unitree G1 policy in full physics · 6.2–11.8 **9/9 stages · 0 falls** · no mocap, no hand-written controller |
| 2 | "Teaching a robot from video usually means copying the clip joint by joint. But a copier needs the clip. Shove it, start it somewhere new, or take the clip away, and it has nothing to fall back on. We wanted a robot that learns the action, not the clip." | 5.2–13.8 A clip copier (tracking policy) shoved at **1.5 m/s** · 14.1–17.9 Same copier, clip frozen: it just stands |
| 3 | *changed:* "Meta's SAM 3.1 finds the person. We rebuild their pose on the robot. Then we read off what the action is, and Meta's Muse names its key moments: sink into squat, jump straight up." | 0.5–4.5 SAM 3.1 finds the person · 4.6–8.4 SAM 3D Body: **17 body joints** shown (the model's MHR70 keypoints, mapped to 33 landmarks) · 8.5–11.8 Retargeted onto the G1's **29 joints** · 11.9–17.9 Key moments named by Muse Spark: sink into squat · jump straight up |
| 4 | *changed:* "Bodies differ. We rebuild the pose on the G1's own bone lengths, so joint angles carry over and the feet stay planted. Joint-angle error drops two to ten times." | 0.3–5.8 Rebuilt on the G1's own bone lengths · IK error **0.1 cm** (against the retarget's own targets, not fidelity to the person) · 6.2–14.8 Joint-angle error **2–10× lower** (elbow **13.4° → 1.3°**) · feet stay planted |
| 5 | "Meta's V-JEPA 2 keeps the robot's style close to the video, but only where it matters. Physics tells us the push before takeoff and the landing count. Waving at the crowd doesn't." | 0.5–7.0 V-JEPA 2 style reward, weighted by what the physics needs · 7.4–14.8 Arm-waving: floor weight **0.1** · takeoff push and landing: **1.0** (`docs/GOAL_RELEVANCE.md`) |
| 6 | "The robot is rewarded for finishing each stage in order, and only while it stays up. A fall voids the attempt. Timing and exact poses are up to it." | 0.3–5.1 Each key moment is a stage of the reward · 5.5–14.8 Reach the stages in order, own timing · a fall voids the attempt |
| 7 | "All three policies finish every stage without falling. The copier stays within eleven centimetres of the clip. Our task policies do the jump their own way, forty centimetres to a metre away." | 0.3–9.5 All three policies: **9/9 stages · 0 falls** · 9.8–19.8 Distance from the clip: copier **0.11 m** · task **0.41 m** · task, no clip **1.06 m** |
| 8 | "Now the real test. Freeze the clip: the copier just stands there. Both task policies still squat, jump and land, a hundred percent of the time. Shove them at 1.5 metres per second: the copier finishes 14 percent of the time, ours 81. Across 18 conditions and 64 rollouts each, the robot rewarded for the action holds up." | 0.3–9.8 Clip frozen: copier **0%** · task **100%** · task, no clip **100%** · 10.2–21.8 Shoved at 1.5 m/s: copier **14%** · ours **81%** · 22.2–31.8 **18 conditions × 64 rollouts** per policy |
| 9 | *changed:* "Muse closes the loop. It watches the clip and names each key moment, and between training rounds it reads the results, watches the robot, and proposes the next reward." | 0.3–10.8 Muse Spark watches the clip and names its key moments · 11.1–14.9 …then tunes the reward between training rounds. The card shows V-JEPA 2 likeness **0.410 ± 0.008 vs 0.388 ± 0.017** without the agent (start 0.357), jumping jacks, 3 rounds / 900 PPO iterations, one training run per arm |
| 10 | *changed:* "Try it live: drop in a phone clip and the same stages build in under a minute. Then push the robot yourself. Any video of a person moving can become robot training data. Code and weights are open. athlete2humanoid." | 0.2–5.9 Live: upload a phone clip · this CPU build took **21 s** (shown 3.6×; hosted `/live` takes 15–40 s, plus a cold start) · 6.1–14.9 Push the trained robot yourself · recorded **1 m/s** recoveries. End card: github.com/PandaGrdn/athlete2humanoid · athlete2humanoid.pages.dev |

The numbers come from the files listed under "Numbers quoted, and where they come from" in `DEMO_SCRIPT.md`.

## Raw screen takes (`outputs/demo_kit/raw/`)

Each take was recorded headless from a local server (`server.main` + Vite) at 1280×720 CSS px × 1.5, so the output is
1920×1080. That ran on this Mac's GPU at about 29 fps of real frames for the 3D views and 50–60 fps for the 2D pages,
re-timed to 30 fps.

Why local rather than athlete2humanoid.pages.dev: the local data is identical, including Muse's names, but local
removes network stalls from the takes. The runs folder held only the six `remote_latent_s3_*` runs and their
`remote_robust_s3_*`, as the Space does. Every take's `<name>.marks.json` gives the second at which each action
starts.

| Take | Length | What happens (seconds) |
|---|---|---|
| `build_sj.mp4` | 56.8 s | `/build/latent_s3_sj`: 1.0 "Replay the build" · 16.8 all six stages ready · 19.3 swing to 03 · 21.5 orbit · 28.2 04 Moments · 33.2 05 Variants · 36.7 06 Policy · 39.2 Shove + Push · 43.7 Push again · 48.2 scroll to the RL-environment panel · 49.8–56.8 hold |
| `build_kick.mp4` | 56.8 s | `/build/latent_s3_kick`, the same actions at the same times |
| `live_upload.mp4` | 37.4 s | `/live`: 2.0 `clips/punches.mov` chosen · 2.3 build starts · 11.8 video ready · 15.8 skeleton · 18.4 retarget · 19.9 moments · 23.1 variants ready, policy offered · 25.6 scroll to the stage · 28.4 "Play a pre-trained policy (squat)" · 32.4 Shove + Push |
| `showcase.mp4` | 34.6 s | `/showcase`: 3.0 Shove + Push (squat jump) · 8.0 orbit · 12.5 Push · 17.6 switch to Front kick · 23.6 Push · 29.1 scroll to the stats (100% held-out variants, 0 falls, 100% after 1 m/s) |

## Notes for the final cut

- **Live upload:** the 21 s CPU build plays at 3.6× in shot 10 (the job clock on screen runs fast). The caption says
  so. Stage 06 of a live upload is a pre-trained policy, labelled **pre-trained** on screen: the CPU host can't train.
  The shot cuts away before that label, to the kick's own trained policy.
- **Two kinds of policy:**
  - The robots that recover from pushes in shots 3, 6, 9 and 10 (the build view and the showcase) are the
    `remote_latent_s3_*` tracking policies. Those were trained on 36 variants with pushes.
  - The "copier" that falls in shots 2 and 8 is the final run's tracking policy (`remote_latent_final`), trained
    without them.
  - The narration never calls the s3 robots "copiers". Keep it that way if you re-cut.
- **Who appears:** only the user's own phone clips appear (squat jump, front kick, punches). The 1900 discus and
  hurdles athletes are not used.
- **Hosted site:** athlete2humanoid.pages.dev shows the same build view and labels if you'd rather record from it.
  Its task specs carry `named_by: muse-spark-1.3`.
- **Upgrades if there's time:**
  - Shot 5 is a static plot with a push-in. An animated plot would move more.
  - Shot 7 loops `grid.mp4` twice. A second camera angle would avoid the repeat.
