# Demo video: storyboard and narration (3:00)

Every visual comes from `scripts/make_demo_assets.py`, which writes 1920x1080 / 30 fps silent clips
to `outputs/demo/`. It also writes `rough_cut.mp4`: all shots back to back at the timings below,
with the captions burned in. Record the voice over the rough cut, then swap in clean clips and real
captions in the editor.

```bash
python3 -m venv ~/.venvs/a2h-demo && ~/.venvs/a2h-demo/bin/pip install imageio-ffmpeg pillow numpy
~/.venvs/a2h-demo/bin/python scripts/make_demo_assets.py --src outputs/remote_latent
```

The metric cards read `<src>/eval.json` and `<src>/distill_report.json`. When the refined run lands,
rerun the script against its output folder and the numbers update. Then check the narration in
shots 10 and 11, which quotes the numbers.

## Arc and rubric weighting

| Criterion (weight) | Shots | Time | Share |
|---|---|---|---|
| Impact (30%) | 1 hook, 3 problem, 12 impact, 13 end | 49 s | 27% |
| Meta tooling (25%) | 5 SAM 3.1, 7 V-JEPA 2 reward, 11 Muse Spark loop (the title and pipeline also name all three) | 55 s | 31% |
| Technical depth (20%) | 4 pipeline, 6 retarget, 8 PPO in mjlab | 35 s | 19% |
| Demo (25%) | 2 title, 9 results grid, 10 metrics (the hook is demo footage too) | 41 s | 23% |

The narration runs at about 2.5 words per second, roughly 460 words in all. Frame the project as
**robot learning from video**. Nothing in it should be addressed to the person in the clip: no form
feedback and no "coach" wording.

## Shots

### 1 · Hook · 0:00–0:10 (10 s)
- **Visual:** `hook.mp4`. The phone video (left) plays next to the latent-only G1 policy (right), in sync.
- **Caption:** One phone video. One humanoid policy.
- **Narration:** "That's me, squatting in my dorm, filmed on a phone. That's a Unitree G1 that learned the squat, and its reward never gave it a single joint angle."

### 2 · Title · 0:10–0:16 (6 s)
- **Visual:** `title.mp4`. The name, the tagline and the model line, over the dimmed silhouette strip.
- **Caption:** none (the card is the caption).
- **Narration:** "This is athlete2humanoid: any video in, a humanoid policy out."

### 3 · Problem · 0:16–0:30 (14 s)
- **Visual:** `problem.mp4`. Three bullets build in, then the punch line in cyan.
- **Caption:** A new robot skill today: a mocap studio or a hand-written reward
- **Narration:** "A new humanoid skill today takes a mocap studio, or a hand-written reward per skill. Copy joint angles off a video, and the robot just replays that clip. Meanwhile, there are billions of hours of people moving on video."

### 4 · Pipeline · 0:30–0:45 (15 s)
- **Visual:** `pipeline.mp4`. Six stages build left to right, each with a live thumbnail: phone video, SAM 3.1 mask, retarget, the V-JEPA 2 latent (real values from the clip), the reward MLP and the PPO policy. The Muse Spark bracket appears under them.
- **Caption:** Video in, policy out, with three Meta models
- **Narration:** "So we built a pipeline from any clip to a trained policy. Three Meta models do the heavy lifting: SAM 3.1 sees the person, V-JEPA 2 understands the motion, and Muse Spark runs the whole loop as an agent."

### 5 · SAM 3.1 · 0:45–0:55 (10 s)
- **Visual:** `sam_split.mp4`. The input video next to the SAM 3.1 mask.
- **Caption:** SAM 3.1 isolates the person: no suit, no markers
- **Narration:** "SAM 3.1 finds the person and tracks them through every frame. No suit, no markers, no green screen. Just a phone in a messy room."

### 6 · Retarget · 0:55–1:05 (10 s)
- **Visual:** `retarget_split.mp4`. The human next to the kinematic G1 retarget.
- **Caption:** Retarget: a kinematic first guess on the Unitree G1
- **Narration:** "We retarget that body onto the G1. It's a good first guess, but it's only kinematic: no physics, no balance, nothing a real robot could run."

### 7 · V-JEPA 2, distilled into a reward · 1:05–1:25 (20 s)
- **Visual:** `pipeline_reward.mp4`. The same diagram with V-JEPA 2 and the reward MLP lit and the other stages dimmed; the explanation builds in underneath.
- **Caption:** V-JEPA 2 knows what the action looks like; a small MLP turns that into a reward
- **Narration:** "Here's the key idea. V-JEPA 2, Meta's video world model, captures what an action *looks like*. It's too slow for RL, so we distil it: a small network predicts its latent from one second of robot body positions, trained on 253 motions, falls included. The reward: does this look like the video?"

### 8 · PPO in mjlab · 1:25–1:35 (10 s)
- **Visual:** `strip.mp4`. Silhouettes as V-JEPA sees them (video, retarget, tracking, hybrid, latent) with each one's score underneath.
- **Caption:** PPO in mjlab: one clip, three reward recipes
- **Narration:** "Then PPO in mjlab trains three policies from the same clip: one tracks joints, one mixes joints with V-JEPA, and one gets V-JEPA only."

### 9 · Results · 1:35–1:55 (20 s)
- **Visual:** `grid.mp4`, played twice. Human, retarget, tracking policy and latent policy, with each one's score and fall count.
- **Caption:** Latent policy: no joint targets, still a squat
- **Narration:** "Top left, the video. Top right, the retarget. Bottom left, the tracking policy, which follows the joints closely. Bottom right is the one I care about. It never saw a joint target. It was only rewarded for *looking like the video*, and it squats and never falls. It learned the action, not the clip."

### 10 · Metrics · 1:55–2:10 (15 s)
- **Visual:** `metrics.mp4`. Animated bars of the real V-JEPA 2 similarity, each with its fall count and bank rank.
- **Caption:** Scored by the real V-JEPA 2, not the reward net
- **Narration:** "We score everything with the real V-JEPA 2, not our reward net. Tracking reaches 0.71, right next to the retarget's 0.73. The latent policy reaches 0.53. All of them rank first out of 26 motions, with zero falls."

### 11 · The agentic loop: Muse Spark · 2:10–2:35 (25 s)
- **Visual:** `muse_loop.mp4`: Muse Spark at the centre and four stages (skill reader, train, judge, revise) with a dot running the loop. **Swap the middle ~12 s for real footage once the Muse integration lands:** a screen capture of the agent's tool calls, or a chart drawn from `agent_log.jsonl` showing the score across rounds.
- **Caption:** Muse Spark reads the clip, designs the reward, judges the rollout
- **Narration:** "Muse Spark makes this agentic. As a skill reader, it watches the raw clip and writes the training spec: the phases, the nearest known motions, the reward weights. Then, as a reward designer, it launches training runs as tool calls, watches the rollouts, reads the scores, and rewrites the reward. It's judged by the real V-JEPA, so it can't game its own reward."
- **Before recording:** match this to what the Muse sessions actually ship. The card and the narration follow the plan in `docs/META_INTEGRATION.md`, integrations #1 and #2.

### 12 · Impact · 2:35–2:50 (15 s)
- **Visual:** `impact.mp4`. Three bullets, then the punch line.
- **Caption:** Every video of a person moving is now robot training data
- **Narration:** "Any clip of a person (a dance, a chore, a rehab exercise) becomes a robot training signal. The reward is about how the motion looks, not one body's joint angles, so it carries across bodies. And perception and reward run on open weights."

### 13 · End card · 2:50–3:00 (10 s)
- **Visual:** `end.mp4`. The name, the tagline, the repo URL and the model line, over the dimmed latent policy.
- **Caption:** none.
- **Narration:** "Every video of a person moving is now robot training data. This is athlete2humanoid."

## Assets

| File | Length | Used in |
|---|---|---|
| `hook.mp4` | 10.2 s | 1 |
| `title.mp4` | 6.0 s | 2 |
| `problem.mp4` | 14.0 s | 3 |
| `pipeline.mp4` | 15.0 s | 4 |
| `sam_split.mp4` | 10.2 s | 5 |
| `retarget_split.mp4` | 10.2 s | 6 |
| `pipeline_reward.mp4` | 20.0 s | 7 |
| `strip.mp4` | 10.17 s | 8 |
| `grid.mp4` | 10.2 s | 9 (looped) |
| `metrics.mp4` | 15.0 s | 10 |
| `muse_loop.mp4` | 25.0 s | 11 |
| `impact.mp4` | 15.0 s | 12 |
| `end.mp4` | 10.0 s | 13 |
| `rough_cut.mp4` | 180.0 s | the whole video, captions burned in |

To rebuild a single clip, pass `--only grid metrics`. If `clips/squat.mov` is missing, the human
footage falls back to the left half of `side_by_side.mp4`.

## Open items before the final cut
- **Muse footage (shot 11):** this needs the agent's real run, either a screen capture or a rounds chart from `agent_log.jsonl`.
- **Refined run:** rerun the script with `--src` set to the new output folder, then update the numbers in the shot 10 narration.
- **Policy video resolution:** `policy_*/policy.mp4` is 320x240, so the hook and the grid upscale it and it looks soft. Re-rendering `policy_qpos.npy` at 1280x720 would sharpen shots 1, 4, 9 and 13.
