# Real athletes: the robot learns the move from free-licence video (2026-10-30)

**Question:** the stack was built on our own phone clips. Does it also work on video of other people, filmed by
someone else, that the submission may legally show? And can a move that travels (a run) be learned when the G1 has to
do it on the spot?

## The clips

Licences were read on each clip's page on 2026-10-30. The record, with authors, trims and the credit lines the site
shows, is `docs/real_athletes/clips.json`; the downloads stay out of git (`clips_pro/athletes/`, with
`clips_pro/LICENSES.md`).
- Pexels and Mixkit licences need no attribution. The site credits every clip anyway (`web.json` "credit", shown on the
  run and build pages and the showcase tab).
- These are stock-footage athletes (runners, a taekwondo black belt, a boxer), not famous professionals. Clean,
  openly licensed footage of famous athletes doing one move is close to nonexistent; broadcast footage is copyrighted.

Each clip was checked by eye on contact sheets for one athlete, full body, and no cuts:

| Clip | Source | Licence | Trim | Verdict |
|---|---|---|---|---|
| `jog_lakeside` | [Pavel Danilyuk, Pexels 4065467](https://www.pexels.com/video/a-man-wearing-sunglasses-running-in-the-street-4065467/) | Pexels License | 0–6.0 s | kept: side-on dolly shot of a jogger; a second runner enters at 7 s |
| `jog_bridge` | [RDNE Stock project, Pexels 8402092](https://www.pexels.com/video/a-beautiful-woman-jogging-on-a-sunny-day-8402092/) | Pexels License | 0–7.5 s | kept: side-on jogger, slowly panning camera; cut at about 8 s |
| `taekwondo_kick` | [rubenvelasco, Mixkit 49634](https://mixkit.co/free-stock-video/a-young-man-performs-a-karate-kick-to-the-air-49634/) | Mixkit Stock Video Free License | whole 6.4 s | kept: static camera, black background, a side kick |
| `shadow_boxing` | [Ron Lach, Pexels 8745108](https://www.pexels.com/video/woman-doing-shadow-boxing-8745108/) | Pexels License | 0–5.5 s | kept: backlit, the head briefly hidden by a foreground object |
| Pexels 7187054 / 7187058 (sprinters) | RDNE Stock project | Pexels License | | rejected: cuts every second or so; one runs toward the camera |
| Neymar's free kick (Commons) | Rodrigogomesonetwo | **CC BY-SA 4.0** (the file page's `{{self|cc-by-sa-4.0}}`), not CC BY | | excluded by licence, not downloaded |

**Gaps:**
- **Jogs, not a sprint.** The two side-on sprinter clips on the free sites have cuts every second or so.
- **No hurdle or high-jump approach.** The only ones under an allowed licence were vertical phone footage looking down
  the lane.

## Pipeline quality (the pipeline audit's numbers)

Processing ran on Modal (job `pro_ra`, an L40S, 12 minutes for all four): SAM 3.1, then SAM 3D Body, the camera
intake, and the retarget. The two jogs then had their retargets redone locally with the final in-place steps
(`experiments/real_athletes.py reretarget`).

`experiments/real_athletes.py audit` writes the numbers to `docs/results/real_athletes/audit.json`. Columns:
- **IK error:** the mean distance between the G1's frames and their targets.
- **Angle error:** the mean flexion error to the human.
- **Limits:** the share of frames with any joint within 1° of a limit.
- **Variants kept:** of the 36 training variants, after the physical filter (`skill/variations.py`).

| Clip | SAM 3.1 visible | SAM 3D Body pose | IK error, knees / feet / hands (cm) | Angle error, knee / hip (°) | Fastest joint (rad/s) | Limits | Spec: key moments, self-check | Variants kept |
|---|---|---|---|---|---|---|---|---|
| `jog_lakeside` (in place) | 100% | 100% | 4.6–6.1 / 0.2–0.4 / 0.1 | 19.9 / 12.9 | 12.7 | 0% | 11, 12/12 | 36/36 |
| `jog_bridge` (in place) | 100% | 100% | 4.3–4.4 / 0.1 / 0.1 | 11.0 / 13.3 | 10.3 | 16% (ankle pitch) | 10, 11/11 | 35/36 |
| `shadow_boxing` | 99% | 100% | 1.0–1.5 / 0.0–0.1 / 0.1 | 3.8 / 3.6 | 13.4 | 5% | 3, 5/5 | 36/36 |
| `taekwondo_kick` | 100% | 100% | 2.0–5.4 / 0.3–1.2 / 0.1 | 12.0 / 9.0 | 20.0 (at the cap) | **67%** | 4, 6/6 | 24/36 |

Notes from the intake:
- The bridge jogger is 383 px tall, under the 400 px the intake wants.
- The kick's black background gives the camera compensation nothing to register, so it was left uncompensated, which
  is right for a static camera.
- The in-place jogs' knee and hip errors are the cost of running on the spot (see "Run in place" below), not pose
  error. Travelling, the same clips retarget at 1–2° of knee error.

**The kick does not fit the G1 as filmed.**
- The reference pins the right hip yaw at its limit in 45% of frames, the hip pitch in 33% and the knee in 25%. A
  side kick at hip height asks for more hip range than the G1 has.
- Its torso tilts to 1.39 rad (79°). The robustness eval's fall rule (1.4 rad) fires on the reference itself, so the
  eval refuses to score it.
- Training agreed: after 1,000 iterations its episodes lasted 87 of about 320 steps, ending on mjlab's hand and foot
  height check. A lower kick, or a front kick (as on our own clip), is what this robot can learn.

## Policies: not finished. The GPU budget ran out mid-training.

**Result: no real-athlete policy passes its spec yet.**
- The trackers stay up: no falls on the clean clip, almost none under 1 m/s pushes.
- But they don't do every key moment yet. The jogs shuffle their feet instead of lifting them.
- The failure is under-training, not the pipeline: the spend limit stopped training at 450–490 of 1,000 iterations.
- The demo wiring is ready, but no run was added to the hosted demo, since none passes.

**Recipe.** As for the squat jump:
- `tracking_var` trained on the clip's variants (`experiments/variant_policy.py`), one policy per clip.
- Warm-started from the squat-jump variants tracker (`remote_variants_s3_sj/policy_tracking_var/model_1499.pt`).
- 1,000 iterations, the four clips in parallel on one Modal L40S (`remote_run.py athletes`, `scripts/real_athletes.sh`).
- Then the held-out variants and recorded pushes.

**What happened:**
1. **Building the variants took about 30 minutes inside the job:** about 40 motion files per clip, four clips sharing
   8 CPUs while the GPU idled. Build them beforehand next time (`variant_policy.py --build-only`, or reuse a build with
   the new `--reuse-build` / `real_athletes.sh --from`).
2. **The Modal workspace (`a2h`) hit its billing-cycle spend limit** at about 00:15. That stopped the container with
   the jogs and the boxing at iterations 448–489. Checkpoints are saved every 250 iterations, so iteration 250 was
   kept. A new job was refused (`ResourceExhaustedError`). The kick had finished its 1,000 iterations, but not its
   rollout.
3. **Training continued on this Mac's CPU** (mjlab on Warp's CPU backend; mjlab runs on CPU with
   `CUDA_VISIBLE_DEVICES=` empty), from the GPU iteration-250 checkpoints.
   - 512 envs at about 10 s per iteration: an eighth of the samples per iteration that a GPU's 4,096 envs give.
   - The processes ended at CPU iteration 551–591. Iteration-500 checkpoints were kept.

**Evaluation.** `experiments/robustness_eval.py` on CPU, 32 rollouts per condition. "Stages" is the share of the
spec's key moments done in order; success needs all of them. Numbers are in `docs/results/real_athletes/policies.json`.

| Clip | Checkpoint | Clean: success / stages / falls | Joint error (rad) | 1 m/s pushes: success / falls | Held-out variants: success |
|---|---|---|---|---|---|
| `jog_lakeside` (in place) | GPU 250 | 0% / 17% / 0% | 0.258 | n/a | 2/7 variants (half speed, and mirrored at 1.6x) |
| | GPU 250 + CPU 500 | 0% / 17% / 0% | 0.167 | 0% / 0% | 2/7 (the same two, 100% each) |
| `jog_bridge` (in place) | GPU 250 | 0% / 12% / 0% | 0.208 | n/a | 0/5 |
| | GPU 250 + CPU 500 | 0% / 18% / 0% | 0.193 | 0% / 6% | 0/5 |
| `shadow_boxing` | GPU 250 | 0% / 21% / 0% | 0.210 | n/a | 0/7 |
| | GPU 250 + CPU 500 | 0% / 21% / 0% | 0.172 | 0% / 0% | 0/7 |
| `taekwondo_kick` | GPU 1,000 | not scorable: the reference itself trips the fall rule (above) | | | |

The joint error falls with training: 0.26 → 0.17 rad on the lakeside jog; the finished squat-jump tracker reaches
0.09.

**Where the jogs fail.** The reference lifts each ankle about 20 cm every stride. The policy lifts the left foot once,
at the start, and then keeps both feet at 3–4 cm, so it never completes the second key moment ("left foot up, right
foot forward"). That stage needs the left foot off the floor for 0.1 s, with ankle-height tolerances of 5–6 cm.

Slowed down, it does jog. The lakeside policy completes every key moment of the half-speed variant and of the mirrored
1.6x-slower one, in all 32 rollouts, with no falls. That's the expected path of a tracker learning a gait: the slower
version first.

**Squat-jump comparison.** The squat-jump tracker that is the warm start was trained the same way, for 1,500 GPU
iterations. It reaches 100% clean and 82% on held-out variants (`outputs/remote_robust_s3_sj`, the showcase's figure).

### To finish (needs GPU budget)

The variants are built and the inputs are packed. One job retrains the three clips from the squat-jump tracker:
about 30 minutes of building plus about 45 minutes of training on one L40S, around $3. That's less if the builds are
reused: they're kept locally in `outputs/remote_athletes_ra3_partial/` and need uploading.

```bash
MODAL_PROFILE=a2h python3 scripts/remote_run.py athletes --backend modal --reuse outputs/athletes_inputs --with-weights \
    --iters 1000 --tag ra --branch real-athletes --args "jog_lakeside jog_bridge shadow_boxing"
python3 experiments/real_athletes.py assemble outputs/remote_athletes_ra    # -> outputs/remote_latent_ra_{jog,bridge,box}
python3 scripts/build_variants_view.py outputs/remote_latent_ra_*          # the build view's variants.json
python3 experiments/real_athletes.py showcase jog_lakeside ...             # the passing clips: 3D bundles + credited tabs
```

Then map each passing clip to its bundle in `web/src/buildview/data.ts` (`OWN`), so the build view's stage 6 plays
its own policy.

**What the deploy needs:**
- `scripts/build_space.py` already ships `remote_latent_ra_*` and their `robust.json`; then the usual Space redeploy
  (`docs/DEPLOY.md`).
- The credit lines are served by the API (`web.json` "credit") and shown on the run, build and showcase pages. The
  web build has to be redeployed for those components.

## Run in place (`pipeline.py --in-place`)

A run's reference travels several metres. The tracking tasks learn travel poorly (docs/FRONTIER_TRAVEL.md), and a
demo floor is finite. So `--in-place` turns the travelling motion into the same gait on the spot, keeping the vertical
root motion (the bounce, the flights) and the foot contacts.

It is an option of the retarget (`retarget.proportional_motion(in_place=...)`). It is passed through `pipeline.py`,
`experiments/process_clips.py`, `scripts/pro_clips.sh` (`IN_PLACE=`) and `scripts/remote_run.py --in-place`.
`experiments/real_athletes.py reretarget` redoes a stored clip's retarget without a GPU. The clip's `meta.json`
records it, for example the bridge jog:

```json
"in_place": {"method": "world", "travel_removed_m": 5.776, "mean_speed_m_s": 0.77, "stride_s": 1.333,
             "root_travel_left_m": 0.0, "travel_restored_m": 3.733, "floor_drift_removed_m": 0.303}
```

### Steps, in order

1. **Level the floor** (`level_floor`). A camera pitch error turns travel into height. On the bridge jog the floor
   rose 0.30 m over 7.5 s, so its second half had no foot near the floor: no contacts, and every variant was dropped,
   the clip itself included.
   - The fix: a line through the lowest foot's lowest point per 0.5 s, refitted without the windows more than 5 cm
     above it (flights), and applied only when at least 75% of the windows lie on it.
   - The hurdles film, which ends in a 1.8 s flight, is left alone.
2. **Restore the ground speed a following camera hides** (`restore_travel`). The lakeside jog is a dolly shot. Its
   background is far away and barely moves, so the camera compensation recovers 0.2 m of a run of about 6 m.
   - The pose reads as a treadmill run: each stance foot slides back under a still hip at 1.8 m/s, and the contact
     detector, which needs a planted foot to stand still, finds no contacts at all.
   - The fix: while the lowest foot is on the floor, the root must move at minus that foot's velocity relative to
     it. The shortfall, smoothed over 1 s, is added back where it averages over 0.3 m/s.
   - That restores 5.7 m on the lakeside jog and 3.7 m on the bridge jog, whose camera pans.
   - Only with `--in-place`: the default retarget is unchanged.
3. **Find the contacts** on that travelling motion, where a planted foot stands still.
4. **Remove the travel**, by one of three methods:

| Method | What it does | What it costs |
|---|---|---|
| `world` (default) | The root is held where it starts. Each contact is pinned under it at the contact's mean position (`plant_feet(hold_root=True)`), and the pin is bridged across the swing. | The stance leg can't sweep back under a still hip, so it bends instead: knee angles differ from the human's. |
| `soft` | As `world`, but the root keeps its surge and sway within a stride, squashed into ±10 cm of the start (`IN_PLACE_SOFT_M`). | Measured below: it doesn't help. |
| `stride` | The planted motion minus its per-stride travel (a moving average of the root over one stride, `stride_drift`). | Every planted foot slides back at the running speed. |

### Measured

`experiments/in_place/measure.py` measures on the G1, from MuJoCo forward kinematics of the solved qpos. Results are
in `docs/results/real_athletes/in_place_*.json`.
- **Slip:** a foot's horizontal speed on the frames the retarget plants it and its sole is within 2 cm of the floor.
- **Knee error:** the mean knee-angle error to the human.

| Clip | Method | Slip (cm/s) | Knee error (°) | Root's farthest from start (m) |
|---|---|---|---|---|
| Lakeside jog | travelling | n/a: no contacts found; the lowest sole moves at 179 | 1.1 | 0.40 |
| | **world** | **0.5** | 19.9 | 0.02 |
| | soft (±10 cm) | 0.6 | 20.6 | 0.08 |
| | stride | 103 | 1.7 | 0.10 |
| Bridge jog | travelling | 0.1 | 1.6 | 6.62 |
| | **world** | **0.3** | 11.0 | 0.02 |
| | soft (±10 cm) | 0.3 | 10.4 | 0.10 |
| | stride | 78 | 1.7 | 0.13 |
| Hurdles run-up (1900 film) | travelling | 0.0 | 1.3 | 3.17 |
| | **world** | **0.5** | 21.1 | 0.08 |
| | soft (±10 cm) | 0.6 | 18.6 | 0.16 |
| | stride | 85 | 2.9 | 0.28 |

- **`world` keeps the feet planted.** `stride` slides them at about the speed it removed. So `world` is the default.
- **The cost is the knee:** 11–21° of mean knee-angle error, against 1–2° travelling. A stance leg that stays under the
  hip instead of sweeping behind it is a run in place, not a treadmill run. The arms, trunk and rhythm are the
  athlete's; the legs are the G1 jogging on the spot.
- **The soft hold doesn't buy the knee back.** It changes the knee error by -2.5 to +0.7°, at the same slip. A jog's
  stance sweep (how far the planted foot moves back relative to the hip) is several times the 10 cm the root may drift.
  `soft` stays available, but isn't the default.

### Tests

`tests/test_in_place.py` uses a synthetic 2.5 m/s jog (0.6 s strides, 0.1 s flights). In place:
- the root holds, planted feet stay put, and the bounce is kept;
- `world` slips 0.6 cm/s against `stride`'s 192;
- the soft hold stays within its bound;
- a dolly-shot version (hips held still in the image) gets its travel and contacts back;
- a rising floor is levelled, and a flight at the clip's end isn't;
- the keypose spec compiles, and its own retarget completes it;
- the variant filter keeps the original and a mirrored and a re-timed variant.
