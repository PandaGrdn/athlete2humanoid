# Demo video: storyboard and narration (3:00)

Frame it as **robot learning from video**. Nothing is addressed to the person in the clip: no form feedback, no "coach" wording. The narration is about 390 words, about 2.2 words per second; shots 3 and 6 are the densest (73 words in 33 s, 62 in 30 s).

**Format:** a screen recording of the site with a voice-over, plus a few full-screen cards. Demo is 25% of the score, and the
3D build view already plays like a moving slide deck, so slides alone would show less. Record the voice-over one shot at
a time (each shot's clip below is exactly its narration slot), then lay it over the shot clips in any editor.

## The kit

```bash
python scripts/demo_kit/render_policies.py     # outputs/demo_kit/src/: jog, discus, shot_put, agent (1080p policy renders)
node scripts/demo_kit/record_takes.mjs build_jog|env_jog|live <clip> live_kick   # screen takes (header has the setup)
python scripts/demo_kit/build_kit.py           # outputs/demo_kit/shotNN_*.mp4 and rough_cut.mp4 (captioned, no audio)
```

[`docs/demo_kit/EDIT_LIST.md`](demo_kit/EDIT_LIST.md) lists every shot's sources, trims and captions, and every take.
The first version of this storyboard (the squat-jump final run: a "copier" against task policies) is superseded. Its
frozen-reference and shove numbers came from policies later found to be tuck-hopping (`docs/KEYPOSE_SPEC.md`), and its
shot clips are in `outputs/demo_kit/v1/`.

## Rubric weighting

| Criterion (weight) | Shots | Time |
|---|---|---|
| Impact (30%) | 1 hook, 2 problem, 4 speed, 8 close | 67 s |
| Meta tooling (25%) | 3 the build (SAM 3.1, SAM 3D Body, Muse names, V-JEPA 2), 6 the Muse reward agent | 63 s |
| Technical depth (20%) | 5 robustness and the travel fix, plus 3's retarget and variants | 25 s plus parts of 3 |
| Demo (25%) | 7 live upload and the training environment, plus the build view in 3 and the pushes in 5 | 25 s plus parts of 3 and 5 |

## Shots

| # | Time | Visual | Narration |
|---|---|---|---|
| 1 | 0:00–0:12 | The Pexels jogger (credit burned in) beside the G1 running the policy trained on that clip, twice | "A stock video of a stranger jogging. And a Unitree G1 humanoid in physics, running the policy it learned from that one video. No mocap, no motion library." |
| 2 | 0:12–0:27 | Card: how a humanoid learns a move today → what everyone has | "Today a robot learns a new move in a motion-capture studio or from a hand-written reward. Both are slow and expensive. We wanted the input to be a video anyone can film, and the output a policy that survives a shove." |
| 3 | 0:27–1:00 | `/build/latent_ra_jog` replaying 01→06 (each chip's model badge lights up); orbit on 03; the front kick's 04 with Muse's names; the jog's 05 variants | "Here's the build. Meta's SAM 3.1 finds and tracks the person. SAM 3D Body lifts them to 3D. We rebuild the motion on the G1's own skeleton, so planted feet stay planted. The key moments become the task, and Muse Spark watches the clip and names them. Then come variants, mirrored, faster, slower, bigger, and PPO across thousands of simulated robots, with V-JEPA 2 judging how much it still looks like the video." |
| 4 | 1:00–1:20 | The adaptation chart (push-in); the discus policy (robot only); the "18 minutes" card | "How long does a new move take? A 1900 Olympic discus throw it had never seen: starting from a policy trained on five phone clips, it was doing the throw in 18 minutes on one GPU. From scratch, 24. Punches and jumping jacks didn't make it in half an hour." |
| 5 | 1:20–1:45 | Shove + Push on the jog and the front kick in the build view; the shot put before / after the travel task | "It isn't replaying the clip. Shove it and it recovers. On variants it never trained on, it completes the squat jump 82% of the time and the front kick 100%. On a shot-put glide the robot fell 40 centimetres short; a travel reward and termination took it from 8% to 100%." |
| 6 | 1:45–2:15 | Muse's tool calls with their reasons, from the run's log; start / without the agent / with it; the numbers card | "Muse Spark also tunes the reward. Between training rounds it reads the evaluation, watches the robot's video, and calls tools to change reward weights, with a reason for each. Against the same run without it, likeness to the video rose from 0.388 to 0.410, because it upweighted the likeness terms. It said itself what didn't work. The run cost ten cents." |
| 7 | 2:15–2:40 | `/live`: the Mixkit kick uploaded and built (12.7×); the jog's stage 05 "Download training environment"; `python -m a2h_env info` on the zip | "Try it: upload a clip on the live page and the same stages build on the server. Training needs a GPU, so every build hands back a training environment: one small zip with the motion, the task, the variants and the reward. With mjlab installed, one command starts training." |
| 8 | 2:40–3:00 | "Not yet" card; end card | "What doesn't work yet: one policy for every move, without training, tops out at 7%, so we train per video. Any video of a person moving can become robot training data. athlete2humanoid." |

## Numbers quoted, and where they come from

- **The jog** (shot 1): 100% of clean rollouts, 98% under 1 m/s pushes, 7/7 held-out variants, no falls: `docs/REAL_ATHLETES.md`. Video: Pavel Danilyuk, Pexels 4065467 (Pexels License; `docs/real_athletes/clips.json`).
- **Feet stay planted** (shot 3): stance-foot sliding 29.1 → 0.4 cm/s with proportional retargeting: `docs/RETARGET.md`. **36 of 36 variants** kept for the jog: its build view and `python -m a2h_env info`.
- **Muse's names** (shot 3): the front kick's spec is `named_by: muse-spark-1.3`. The jog's isn't (its stage 04 badge says "keypose compiler"), which is why shot 3 cuts to the kick for this.
- **18 / 24 minutes; 100% clean and 88% under 1 m/s pushes; punches and jumping jacks not in 1,500 iterations** (shot 4): `docs/FRONTIER_ADAPTATION.md`. Use the discus robot only: the 1900 film shows an unclothed athlete (`docs/PRO_CLIPS.md`).
- **82% / 100% of held-out variants** (shot 5): `docs/PIPELINE_ACCURACY.md`, Fix 3, "Policies retrained on the SAM 3D Body references".
- **Shot put** (shot 5), run `remote_travel_c3_shot_put`: plain tracking does 73% of the glide and ends 0.42 m short, at 8% task success. The travel task does 96% and ends 0.05 m off, at 100%. Both are on the clean clip; on held-out variants the travel policy reaches 40%. Source: `docs/FRONTIER_TRAVEL.md`.
- **Reward agent** (shot 6): V-JEPA 2 likeness 0.410 ± 0.008 vs 0.388 ± 0.017 for the same 900 PPO iterations without the agent (start 0.357), success and falls level, $0.098 of API, one training run per arm: `docs/MUSE_AGENT.md`. The tool calls on the card are verbatim from `outputs/agent_jacks/agent_log.jsonl`, abridged. Don't claim more: the gain came from upweighting the likeness terms, and the optional moments it targeted were still skipped.
- **Live build** (shot 7): the 6.4 s 1080p kick took 101 s on the hosted API's CPU (the take's marks). Short phone clips take 15–40 s, plus up to about 23 s of cold start (`docs/DEPLOY.md`). Don't say "under a minute" over this take.
- **The training environment** (shot 7): 2.0 MB for the jog; `python -m a2h_env info` output on the card is the real output. Only a 20-iteration CPU training has been tested (`docs/TRAINING_ENV.md`), so the card shows the `train` command without output.
- **7%** (shot 8): the best zero-shot success on held-out variants across ten generalist policies: `docs/FRONTIER_GENERALIST.md`. **0.32 m** lunge weight shift: `docs/FRONTIER_TRAVEL.md`.

## Showing the 3D build view live

The build view (`web/src/buildview/`) turns the pipeline into the demo: one black stage where each step appears as its
data arrives, in order. **01** the clip (with SAM 3.1's mask beside it on GPU builds; 2D landmarks on a CPU build),
**02** the 3D skeleton from SAM 3D Body (or MediaPipe), **03** the G1 snapping onto it and stepping aside (blue lines join
each wrist, elbow, knee and ankle to the body it drives), **04** the key moments on the timeline (Muse's names when the
spec has them; a ring pulses under the robot and the name flashes as each passes), **05** the variants fanning out
behind as ghosts (mirrored, 0.75× and 1.33× tempo, 1.1× amplitude) with the kept/dropped grid below, **06** a trained
policy you can push. Click a stage chip to swing the camera to it; drag to orbit.

Where it appears:

| Page | What it shows |
|---|---|
| `/live` → drop a clip | A real build of the upload on the API's CPU, stage by stage from the job's events (about 15–40 s on short phone clips; `punches.mov` took 21 s on a local CPU, the 6.4 s 1080p Mixkit kick 101 s on the hosted API). Training needs a GPU, so stage 06 offers the nearest stored policy, labelled **pre-trained**. |
| `/build/latent_s3_kick` (any `/api/runs` id; "Build in 3D" on a run page) | A stored run replayed from its own files, one stage every 2.6 s. "Replay the build" restarts it. |
| `/live/demo/front-kick` | A recorded live build; its 3D pieces come from the stored run of the same clip. |
| `/jobs/<id>` for a demo-runner job | The same view, each stage revealed as the replayed job reaches it. |

To show it live (about 60 s):

1. Open `/build/latent_s3_kick` and let it run: say what each stage is as its chip lights up. Stop on 03 and orbit
   once around the skeleton and the G1 to show the retarget lines.
2. Click **06 Policy**, press **Push** (or click the robot) with **Shove**: the policy's recorded recovery plays.
3. Go to `/live`, drop `clips/punches.mov` (or any phone clip of one person): the same stages build from scratch on the
   server. Point at 05 (variants generated on this host from the upload) and 06 (pre-trained, not this clip's policy).

The hosted API has the build view's data for the six runs (`variants.json`, the skeletons, the named stages). If a run is
added, run `python scripts/build_variants_view.py outputs/remote_latent_s3_*` (the repo's venv) and redeploy
(`docs/DEPLOY.md`; `scripts/build_space.py` ships `variants.json` and `human.npy`). On an older API the view still
works and marks the missing stages "n/a".
