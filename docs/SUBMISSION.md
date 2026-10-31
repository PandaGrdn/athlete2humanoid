# Submission: athlete2humanoid

The text for the Cerebral Valley submission form, one section per rubric criterion. Every number cites the doc it
comes from; all docs are in https://github.com/PandaGrdn/athlete2humanoid/tree/main/docs.

- **Track:** Push the Frontier (robotics: "give a machine a new capability").
- **Bonus prizes:** Best Agentic (the Muse Spark reward agent), Best Open Weights (SAM 3.1, SAM 3D Body and V-JEPA 2 run from their open weights), Most Viral.
- **Live site:** https://athlete2humanoid.pages.dev
- **Code:** https://github.com/PandaGrdn/athlete2humanoid
- **Video:** VIDEO_URL (3 min; storyboard in `docs/DEMO_SCRIPT.md`)

**Tagline:** a phone video of a person becomes a reinforcement-learning policy for a Unitree G1 humanoid. The robot is always the subject; nothing in the system gives feedback to the person in the video.

---

## 1. Impact (30%)

**The claim:** any phone video of a person becomes a robust humanoid policy in about 20 minutes, with no mocap and no motion library.

**Where it holds today, measured:**
- **18 minutes, one GPU.** The 1900 Olympic discus throw, a clip no policy had trained on, became a policy in 18 minutes on one L40S, warm-started from a generalist trained on five phone clips (`docs/FRONTIER_ADAPTATION.md`).
  - 12 of those minutes are PPO (1,000 iterations); 6 build the clip's training variants.
  - At that point it completes the throw on 100% of clean rollouts, 78% of held-out variants and 88% under 1 m/s pushes. From scratch, the same 1,000 iterations complete 0%, and the first pass comes at 24 minutes.
  - The 18 minutes exclude processing the clip (SAM 3.1, pose, retarget), which is the same for every start.
- **Robust to variations it never saw.** A squat jump and a front kick, each from one phone video, complete 82% and 100% of held-out variants (tempos, amplitudes and mirror images outside the training range) with 0% falls (`docs/PIPELINE_ACCURACY.md`, Fix 3, "Policies retrained on the SAM 3D Body references").
- **Moves that cross the floor.** The shot put glide went from 0–8% to 100% task success once the training task rewarded travel: 96% of the 1.1 m glide, 94% success under 1 m/s pushes (`docs/FRONTIER_TRAVEL.md`).

**Where it doesn't hold yet:** on two of the three unseen clips (punches, jumping jacks), 1,500 iterations, about half an hour, were not enough for any starting point (`docs/FRONTIER_ADAPTATION.md`). "About 20 minutes" is a measured result for one move, not a guarantee for every move.

**Real-athlete clips:** REAL_ATHLETE_RESULTS (added by the session processing real-athlete footage).

**Who benefits:**
- **Robot learning teams and humanoid makers.** A new skill today means a mocap session or a hand-written reward. Here the input is a video anyone can film, and the output is a policy plus the evaluation that says how robust it is.
- **Researchers without a mocap lab.** The pipeline, the task specs and the evaluation harness are open; a new move costs about $0.8 of GPU per starting policy (`docs/FRONTIER_ADAPTATION.md`, Cost).
- **Anyone with a phone.** The Live page accepts an upload and builds the robot's version of it in the browser.

**Why now:**
- Meta's open models made monocular video good enough to train on. SAM 3D Body cut bone-length variation 3–6x on our phone clips compared with MediaPipe (`docs/PIPELINE_ACCURACY.md`, Fix 3), and that consistency is what lets the retarget keep planted feet planted.
- GPU physics (MuJoCo Warp through mjlab) trains 4,096 simulated G1s at once, so a new skill is minutes of PPO, not days.
- A general-purpose humanoid, the G1, is a shared target: a policy for it is useful to everyone who has one.

**Problem statement:** Push the Frontier. The new capability is a humanoid that picks up a skill from an ordinary video.

## 2. Use of Meta AI tooling (25%)

Five Meta models, one at each stage from video to trained robot. Each was measured against the alternative it replaced.

| Model | What it does here | Measured effect | What was unexpected |
|---|---|---|---|
| **SAM 3.1** (open weights, `facebook/sam3.1`) | Segments and tracks the athlete from a text prompt. Its masks crop the frame for the pose model, select one athlete in a crowd, and become the silhouette video that V-JEPA 2 embeds. | On the 1900 hurdles film the athlete is never detected on the whole frame (0% of frames); with SAM's crop the pipeline finds them (`docs/PIPELINE_ACCURACY.md`, Stage 1). | The masks do double duty: on silhouettes, V-JEPA 2 compares motion rather than clothing or background. |
| **SAM 3D Body** (open weights, `facebook/sam-3d-body-dinov3`) | Lifts the masked athlete to a 3D skeleton, which is retargeted onto the G1. | On the seven phone clips: bone-length variation 3–6x lower; on the squat jump, double-support ankle disagreement 6.6 → 1.4 cm and knee IK error 1.1 → 0.5 cm; jumping jacks gets its foot contacts back (no-contact frames 41% → 6%). Policies retrained on its references: squat jump 78% → 82% of held-out variants (`docs/PIPELINE_ACCURACY.md`, Fix 3). | It failed the audit's own adoption rule on the two athletic films. A more faithful pose put the shot put's shoulder at the G1's joint limit in 100% of frames (44% before). We adopted it for phone clips, the footage the pipeline is for, and kept MediaPipe as the fallback. |
| **V-JEPA 2** (open weights, `facebook/vjepa2-vitl-fpc64-256`) | The video's latent as a policy input; a weak "look like the video" style reward; the judge of likeness. | One policy over five clips, seeing only past video (no future leakage): success 44% → 75%, falls 16% → 0%; best of all inputs under tempo changes, 44% vs 30% (`docs/VJEPA_CONDITIONING.md`). | As the only reward it was gamed: a distilled V-JEPA reward rose 0.35 → 0.49 while true V-JEPA similarity stayed at 0.35 → 0.37 (`docs/SEMANTIC_REWARD.md`). Used as a world model, its pretrained predictor was worse than copying the present, 2–6x the error (`docs/FRONTIER_WORLD_MODEL.md`). |
| **Muse Spark**, moment naming (Meta Model API, `muse-spark-1.3`) | Watches the clip and names each key moment the task compiler found ("arms overhead legs apart", "sink into right lunge"), written into the task spec as `muse_label` and shown in the 3D build view. | 10 clips named, 10/10 valid on the first attempt, $0.086 in total (`docs/MUSE_AGENT.md`). The reward never reads the names. | Video costs about 95 tokens per sampled frame, against the ~1.4k the image rate predicted; reasoning, not video, drives the cost (`docs/META_INTEGRATION.md` §7). |
| **Muse Spark**, reward-design agent (Responses API, tool calls) | Between training rounds it reads the evaluation, watches the robot's rollout video, and calls bounded tools to change reward weights and termination settings, each with a stated reason. The next round trains on Modal from the last checkpoint, beside an untouched control with the same budget. | Jumping jacks, three rounds (900 PPO iterations): V-JEPA likeness to the video 0.410 ± 0.008 vs 0.388 ± 0.017 for the control (start 0.357), with success and falls level. $0.098 of API against ~$2.35 of GPU (`docs/MUSE_AGENT.md`). | Its own end-of-run summary matched the numbers: more tracking and style weight helped, bigger stage bonuses did not. |

**Caveats on the agent loop** (`docs/MUSE_AGENT.md`):
- The likeness gain came from Muse raising the tracking and V-JEPA style weights, which aim straight at the metric reported. It shows that upweighting the likeness terms raised likeness; it does not show the agent beats a one-line heuristic, and that comparison was not run.
- One training run per arm; seed variance is unmeasured and could be the size of the gap.
- Its other goal failed: the optional key moments it set out to recover were still skipped (9.00 of 11 stages in every round, both arms).
- One misstep: it raised the jump bonus on a clip with no flight stages, where that setting has no effect.

**Also:** Muse Spark as a blind rollout critic said the task policies "never really jump", while our stage checker passed them all. Measured, it was right: the policies were tuck-hopping with the pelvis never above standing height. That finding led to the checker fix in Technical depth (`docs/META_INTEGRATION.md` §6, `docs/KEYPOSE_SPEC.md`).

## 3. Technical depth (20%)

**Method: audit every stage, then fix the biggest losses first.** We measured where the athlete's motion is lost between video and simulation, stage by stage, on all ten clips, and ranked fixes by fidelity gained per day (`docs/PIPELINE_ACCURACY.md`). Examples of what the audit found and what fixed it:
- **Feet are the weakest measurement, and everything downstream trusts them.** In double support MediaPipe put the two ankles 3–8 cm apart in height. Fix 3 (SAM 3D Body) and the contact fixes address this; see the table above.
- **The task spec's timing wasn't free.** Holds were absolute seconds, so a reference played 1.4x faster failed its own spec. With tempo-relative holds, the lunge at 1.4x goes from 3/16 to 16/16 and the shot put from 2/4 to 4/4, and every testable spec passes its own reference at 0.7x, 1x and 1.4x; the hurdles film ends mid-air and can't be tested (`docs/PIPELINE_ACCURACY.md`, Fix 2).
- **The checker was too lenient.** After the Muse critic's report, a flight now needs the pelvis to rise half the reference's rise. Re-scored, the tuck-hopping policies pass 1 of 12 stages. Retrained on the fixed spec, they jump: pelvis to 1.03 m (standing is 0.78 m), flights of 0.40 s (`docs/KEYPOSE_SPEC.md`).
- **Retargeting in the robot's proportions.** Rebuilding the person on the G1's own bone lengths, rather than scaling them uniformly, cut mean knee angle error 11.9° → 3.0° and elbow error 13.4° → 1.3°, and stance-foot sliding 29.1 → 0.4 cm/s (`docs/RETARGET.md`).
- **Why the robot wouldn't travel.** mjlab's torso-position reward saturates by half a metre of error, nothing scores the root's velocity, and the only termination checks height. Adding a wider position term, a velocity term and a horizontal termination curriculum took the shot put glide from 0–8% to 100% (`docs/FRONTIER_TRAVEL.md`).
- **Generalisation is trained, not hoped for.** Every clip becomes mirrored, re-timed and rescaled variants, each checked against its own spec and a physics filter. A tracker trained on them never falls on a held-out variant; every single-clip policy falls on every mirrored one (`docs/GENERALIZATION.md`).

**Honest negatives**, each its own doc:
- **No zero-shot generalist.** Scaling from 5 clips to 328 motions halves zero-shot falls (67–69% → 32–34%) but gives at most 1% success on an unseen clip and 5–7% on its variants. A tracker that works on arbitrary moves needs a motion library that covers them (`docs/FRONTIER_GENERALIST.md`). That is why the product adapts per video instead.
- **V-JEPA 2 as a world model didn't help.** Fed to the policy, its imagined future tied or trailed the past-only latent; a small learned predictor made the robot fall far more on clips it extrapolated to (`docs/FRONTIER_WORLD_MODEL.md`).
- **V-JEPA 2 can't say which moments matter.** Masking its predictor gave changes of ±0.1–0.8% with inconsistent sign, so goal relevance uses physics attribution, and V-JEPA is kept as a diagnostic (`docs/GOAL_RELEVANCE.md`).
- **The lunge still fails.** Every policy stalls at a required 0.32 m forward weight shift; the travel fixes don't move it (`docs/FRONTIER_TRAVEL.md`).
- **Not every move adapts fast:** punches and jumping jacks, above (`docs/FRONTIER_ADAPTATION.md`).
- **Simulation only;** no sim-to-real transfer yet.

**Stack:** SAM 3.1 → SAM 3D Body → proportion-aware mink IK onto the G1 → keypose task spec (named by Muse Spark) → automatic variants with a physics filter → PPO in mjlab (MuJoCo Warp, 4,096 envs), warm-started, with V-JEPA 2 as input, style term and judge → a 64-rollout robustness suite (pushes, start noise, tempo, friction and mass, frozen reference).

## 4. Demo (25%)

Everything below runs in a browser at https://athlete2humanoid.pages.dev.

| Click | What to expect |
|---|---|
| `/build/latent_s3_kick` | The 3D build view: a front-kick clip goes through the pipeline one stage at a time on one floor. 01 the video and its SAM 3.1 mask, 02 the SAM 3D Body skeleton, 03 the G1 snapping onto it, 04 the key moments with Muse's names, 05 the variants as ghosts, 06 the trained policy. Each stage chip names the model that did it. Click 06, then Push: the policy's recorded recovery plays. Drag to orbit. |
| `/showcase` | The trained policies (squat jump, front kick, squat) in 3D; click the robot to push it. |
| `/live` | Upload a phone clip of one person. The server builds it stage by stage on CPU, about 15–40 s (`docs/DEMO_SCRIPT.md`). Training needs a GPU, so stage 06 offers the nearest stored policy, labelled pre-trained. |
| `/live/demo/front-kick` | A recorded live build, if you'd rather not upload. |
| `/research` | Each experiment's results with its videos. |

**Cold start:** the API runs on Cloud Run with no minimum instance, so it scales to zero about 15 minutes after its last request (`docs/DEPLOY.md`). The first request after that starts a new instance: about 23 s, measured on 2026-10-31; once it is up, a request takes 0.15 s (`docs/DEPLOY.md`). The header shows "API online" once it answers. The examples, showcase and research pages are static and never wait.

**The video (3 min):** VIDEO_URL. The person's clip, the robot learning it, the build view, the push test, and the Muse agent's round-by-round decisions.

## Built with

Python · PyTorch · **SAM 3.1** · **SAM 3D Body** · **V-JEPA 2** · **Muse Spark** (Meta Model API) · Hugging Face Transformers · MediaPipe (fallback) · mink · MuJoCo · MuJoCo Warp · mjlab · MuJoCo Menagerie (Unitree G1) · FastAPI · React · three.js · Modal · Google Colab · Stanford FarmShare · Google Cloud Run · Cloudflare Pages · GitHub Actions

---

## Tweet thread draft (Most Viral)

Each tweet is at most 280 characters.

**1/** *(video: phone clip | robot doing it)*
One phone video of a person. A Unitree G1 humanoid learns the move in physics simulation. No mocap, no motion library. A new skill took 18 minutes on one GPU. 🧵

**2/** *(video: build view)*
Meta's SAM 3.1 finds the person, SAM 3D Body lifts them to 3D, we rebuild the pose on the robot's own skeleton, and Muse Spark names the key moments. Then thousands of simulated robots practise.

**3/** *(video: shot put glide)*
A 1900 Olympic shot put: the robot first stayed in place (0–8% success). One fix to what it's rewarded for, and it does the 1.1 m glide, 100% of the time.

**4/** *(video: Muse agent rounds)*
Muse Spark watches each training round and retunes the reward with tool calls. It made the robot look more like the video, and it said honestly what didn't work.

**5/**
Try it: athlete2humanoid.pages.dev. Code is open: github.com/PandaGrdn/athlete2humanoid. Built for the Meta AI Developer Hackathon.
