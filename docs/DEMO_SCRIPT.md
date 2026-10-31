# Demo video: storyboard and narration (3:00)

Frame it as **robot learning from video**. Nothing is addressed to the person in the clip: no form feedback, no "coach" wording. Narration runs about 2.5 words per second, roughly 440 words in all.

## Assets

```bash
python scripts/make_final_demo.py      # outputs/demo/final/: grid, frozen_reference, push, robustness (+ .png)
python scripts/make_demo_assets.py --src outputs/remote_latent --render-robots   # v1 shots: title, pipeline, SAM, retarget
```

**Final run** (`scripts/make_final_demo.py`), 1920×1080 at 30 fps, from `outputs/remote_latent_final` and `outputs/remote_robust_final`:

| File | What it shows |
|---|---|
| `grid.mp4` | The person, tracking, task and task-never-sees-the-clip, with the stage strip underneath |
| `frozen_reference.mp4` | The three policies with the reference frozen |
| `push.mp4` | 1.5 m/s shoves, tracking vs task |
| `robustness.mp4` / `.png` | The success chart |

**Older shots** (`scripts/make_demo_assets.py`): title, pipeline cards, the SAM split and the retarget split are reused from the v1 assets.

Also useful:
- `outputs/retarget_compare/squat_jump/compare.mp4`: person, uniform and proportional retarget.
- `docs/goal_relevance/squat_jump_hype_importance.png`.
- The web app's `/examples` page, which you can screen-record.

## Rubric weighting

| Criterion (weight) | Shots | Time |
|---|---|---|
| Impact (30%) | 1 hook, 2 problem, 10 impact | 45 s |
| Meta tooling (25%) | 3 pipeline (SAM 3.1, V-JEPA 2, Muse), 5 V-JEPA style + goal relevance, 9 Muse | 45 s |
| Technical depth (20%) | 4 retarget, 6 reward machine | 30 s |
| Demo (25%) | 7 grid, 8 frozen + push + chart | 60 s |

## Shots

| # | Time | Visual | Narration |
|---|---|---|---|
| 1 | 0:00–0:12 | `grid.mp4`: the person, then the three robots squat-jumping | "One phone video of a squat jump. A few hours later, a Unitree G1 humanoid squats, jumps, lands and stands in full physics. No mocap, no hand-written controller." |
| 2 | 0:12–0:30 | Title, then a robot copying a clip and falling when shoved | "Teaching a robot from video usually means copying the clip joint by joint. But a copier needs the clip. Shove it, start it somewhere new, or take the clip away, and it has nothing to fall back on. We wanted a robot that learns the action, not the clip." |
| 3 | 0:30–0:48 | Pipeline card: video → SAM 3.1 → pose → retarget → task spec → RL, with V-JEPA 2 and Muse | "Meta's SAM 3.1 finds the person. We rebuild their pose on the robot. Then we read off what the action is, and Meta's Muse names its stages: stand, squat, jump, land." |
| 4 | 0:48–1:03 | `compare.mp4`: person, uniform and proportional retarget | "Bodies differ. We rebuild the pose on the G1's own bone lengths, so joint angles carry over and the feet stay planted. Joint-angle error drops three to ten times." |
| 5 | 1:03–1:18 | Goal-relevance plot with hype arm-waving at about zero, takeoff and landing peaking | "Meta's V-JEPA 2 keeps the robot's style close to the video, but only where it matters. Physics tells us the push before takeoff and the landing count. Waving at the crowd doesn't." |
| 6 | 1:18–1:33 | The stage strip filling in, then the reward-machine card | "The robot is rewarded for finishing each stage in order, and only while it stays up. A fall voids the attempt. Timing and exact poses are up to it." |
| 7 | 1:33–1:53 | `grid.mp4` with the labels | "All three policies finish every stage without falling. The copier stays within eleven centimetres of the clip. Our task policies do the jump their own way, forty centimetres to a metre away." |
| 8 | 1:53–2:30 | `frozen_reference.mp4`, then `push.mp4`, then `robustness.mp4` | "Now the real test. Freeze the clip: the copier just stands there. Both task policies still squat, jump and land, a hundred percent of the time. Shove them at 1.5 metres per second: the copier finishes 14 percent of the time, ours 81. Across 18 conditions and 64 rollouts each, the robot rewarded for the action holds up." |
| 9 | 2:30–2:45 | Muse card: skill spec JSON and the reward agent's proposal | "Muse closes the loop: it reads the video into a task spec, and after each run it reads the results and proposes the next reward." |
| 10 | 2:45–3:00 | Web app examples page, then the end card with the repo | "Any video of a person moving can become robot training data. Code and weights are open. athlete2humanoid." |

## Numbers quoted, and where they come from

- **9/9 stages, 0 falls; 0.11 / 0.41 / 1.06 m from the clip:** `outputs/remote_latent_final/eval.json` (`docs/results/squat_jump_eval.md`).
- **Frozen reference** 0% / 100% / 100%; **1.5 m/s shove** 14% vs 81%; **18 conditions × 64 rollouts:** `outputs/remote_robust_final/robust.json` (`docs/results/squat_jump_robustness.md`).
- **Retarget "three to ten times":** `docs/RETARGET.md`.
- **Muse runs on recorded fixtures** until the API key exists. Say "Muse names its stages" only once the live run is in, or show the fixture as "example output".

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
| `/live` → drop a clip | A real build of the upload on the API's CPU, stage by stage from the job's events (about 15–40 s). Training needs a GPU, so stage 06 offers the nearest stored policy, labelled **pre-trained**. |
| `/build/latent_s3_kick` (any `/api/runs` id; "Build in 3D" on a run page) | A stored run replayed from its own files, one stage every 2.6 s. "Replay the build" restarts it. |
| `/live/demo/front-kick` | A recorded live build; its 3D pieces come from the stored run of the same clip. |
| `/jobs/<id>` for a demo-runner job | The same view, each stage revealed as the replayed job reaches it. |

To show it live (about 60 s):

1. Open `/build/latent_s3_kick` and let it run: say what each stage is as its chip lights up. Stop on 03 and orbit
   once around the skeleton and the G1 to show the retarget lines.
2. Click **06 Policy**, press **Push** (or click the robot) with **Shove**: the policy's recorded recovery plays.
3. Go to `/live`, drop `clips/punches.mov` (or any phone clip of one person): the same stages build from scratch on the
   server. Point at 05 (variants generated on this host from the upload) and 06 (pre-trained, not this clip's policy).

Before a live demo, the API needs this branch's server (skeleton, qpos and variants in `live.json`) and the runs need
`variants.json` and `human.npy`: run `python scripts/build_variants_view.py outputs/remote_latent_s3_*` (the repo's
venv), then redeploy (`docs/DEPLOY.md`; `scripts/build_space.py` ships both files). On an older API the view still
works and marks the missing stages "n/a".
