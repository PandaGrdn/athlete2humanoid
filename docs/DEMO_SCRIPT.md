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
