# Pipeline accuracy audit: where the athlete's motion is lost between video and sim (2026-10-30)

A diagnosis, not a fix: nothing in the pipeline was changed. The numbers come from the measurement scripts in
`experiments/pipeline_audit/` run on CPU against the clips already in `outputs/` (seven phone clips in
`outputs/mc_inputs/clips`, three 1900 films in `outputs/remote_process_pro/clips`), plus a fresh run of the
current `retarget.py` on every clip's `human.npy`, and a raw MediaPipe pass on the ten videos. The JSON behind
every table is in `docs/results/pipeline_audit/` (`pose_stats.json`, `raw_pose/raw_pose.json`,
`retarget_audit.json`, `spec_audit.json`, `spec_noise.json`, `variants_audit.json`).

No model weights or datasets were downloaded. The WHAM / GVHMR / TRAM / SAM 3D Body comparison is therefore
from the literature plus our own error profile, with a concrete A/B proposed below.

## Summary

1. **The feet are the weakest measurement, and everything downstream trusts them.** In double support
   MediaPipe puts the two ankles 3-8 cm apart in height (max 12 cm); a planted ankle hovers 1.3-4 cm with
   1-4 cm of jitter. Stance detection has to use a 10 cm / 0.6 m/s window to cope, and that window fails on
   anything faster than a walk: the hurdles clip gets contacts in 7% of frames, so the current
   `ground_phantom_flights` treats the whole run as a phantom flight and pulls the hurdler down by up to
   70 cm during the real clearance. To keep planted feet still, `plant_feet` moves the athlete's root by
   4-14 cm on average and up to 25 cm.
2. **The IK is accurate where it is asked to be and silently wrong where it is not.** Pelvis, feet and hands
   track to under 1 cm. The knees absorb the inconsistency (right knee 5-10 cm mean error on four clips, over
   3 cm in 79-91% of frames). Ankle roll sits at its joint limit in 4-22% of frames on phone clips and 34-86%
   on the 1900 films. The 20 rad/s cap fires on 5 of 10 clips and leaves the hand up to 13-18 cm behind its
   target while it does.
3. **The task spec's tolerances are fine; its holds and its time base are not.** With realistic limb noise
   (pelvis, torso and feet exact) the pose tolerances pass to 6 cm on most clips, but the hold rule fails
   first: lunge 25% pass at 4 cm (100% with holds off), shot put 50% (100%), punches 25% at 6 cm (100% at
   3x tolerance). A hold is a fixed number of seconds, so a reference played 1.4x faster passes 3/16 (lunge)
   and 2/4 (shot put): "timing is free" holds only for slowing down. The key moments themselves are not
   stable: re-compiling after 1 cm of noise re-finds only 52-86% of them within 0.15 s.
4. **The variant filter checks joint limits and the self-check, nothing physical.** It keeps jumping-jack
   variants whose joints move at 45-82 rad/s, 14 of 36 that lost their hop, and 3 whose flight rises 1.6x what
   its airtime allows; it keeps shot-put variants at 76-172 rad/s. It drops 21 of 36 lunge variants, 13 of
   them because the variant's own spec fails its own reference.
5. **Stored retargets are inconsistent.** The `qpos.npy` in `outputs/mc_inputs` and
   `outputs/remote_process_pro` predates the 20 rad/s cap and gravity check for 6 of 10 clips (shot put differs
   by 5.8 rad, jumping jacks by 3.0 rad). Tables in the docs mix retarget versions. The hurdles reference in
   `outputs/` flies; the one the current code produces is dragged along the floor.
6. **Coverage is limited by the absence of a camera model, not by SAM.** Five of eight downloaded pro clips were
   rejected for panning, crowds or cuts. `pose.human_trajectory` turns any camera motion into root motion 1:1.
7. **Two small front-end facts worth knowing:** before smoothing, MediaPipe's depth is 2-4.5x noisier than its
   image plane and its 3D disagrees with its own 2D by 2.3-3.1 cm per joint; and every `meta.json` records
   30.0 fps for files that are 24.7-29.6 fps, a 3-8% time-scale error that reaches the speed cap, the holds and
   the flight airtimes.

Ranking by fidelity gained per day of work (details in the last section):

| # | Fix | Days | What it buys |
|---|---|---|---|
| 1 | Contacts: a real foot-contact estimator plus a correct phantom-flight test | 1.5 | un-breaks running and glides, removes most of the 4-25 cm root shift, stops bodies being dragged to the floor |
| 2 | Spec: flicker-tolerant holds, tempo-relative holds, noise-calibrated tolerance floor, log the failing feature | 1 | success rates become trustworthy; punches-type false zeros disappear |
| 3 | Front end A/B: SAM 3D Body (per-frame skeleton) and GVHMR/WHAM (world trajectory + contacts) on 3 clips with this audit's metrics | 2-3 + GPU | the only route to form fidelity on far-side limbs and real feet; needs weight downloads |
| 4 | Variants: physics filter and ballistic re-timing | 0.5 | no impossible training targets |
| 5 | Retarget: ankle-roll and shoulder-singularity handling, knee plane | 1 | removes the 34-86% limit-hit frames on athletic clips |
| 6 | Coverage: background-homography camera compensation, click-to-select athlete | 1.5 | accepts panning and multi-person footage |
| 7 | Hygiene: retarget version in `meta.json`, re-process stored clips | 0.5 | tables that compare like with like |

## Stage 1: 3D pose (SAM 3.1 crop -> MediaPipe Pose heavy -> fill, Savitzky-Golay, 2D-hip root)

What `human.npy` carries (after smoothing; the raw-pass table below has the pre-smoothing values):

| clip | frames | noise depth / horiz / vert (cm) | depth ratio | thigh / shin / upper arm / forearm length CV (%) | stance hover mean / sd (cm) | double-support L-R ankle height diff mean / max (cm) | frames with lower ankle > 2 cm off floor |
|---|---|---|---|---|---|---|---|
| squat_jump | 271 | 0.47 / 0.30 / 0.45 | 1.6 | 3.7 / 3.0 / 5.6 / 7.9 | 3.9 / 2.0 | 6.6 / 10.3 | 82% |
| squat | 306 | 0.26 / 0.16 / 0.18 | 1.7 | 3.3 / 3.7 / 5.1 / 9.1 | 2.9 / 1.9 | 6.5 / 10.3 | 82% |
| lunge | 297 | 0.49 / 0.34 / 0.32 | 1.4 | 5.7 / 5.7 / 6.1 / 10.0 | 3.4 / 2.6 | 3.0 / 7.6 | 91% |
| arms | 102 | 0.18 / 0.15 / 0.10 | 1.2 | 3.2 / 2.3 / 11.2 / 7.4 | 2.5 / 1.6 | 3.2 / 4.5 | 80% |
| front_kick | 167 | 0.56 / 0.41 / 0.24 | 1.3 | 4.3 / 4.2 / 3.4 / 5.8 | 1.5 / 1.4 | 3.4 / 10.3 | 29% |
| jumping_jacks | 159 | 1.29 / 1.40 / 2.87 | 0.9 | 2.9 / 2.7 / 18.5 / 13.2 | 2.5 / 2.4 | 4.8 / 12.0 | 87% |
| punches | 168 | 0.65 / 0.58 / 0.42 | 1.1 | 3.1 / 2.6 / 8.3 / 12.8 | 1.3 / 0.9 | 5.9 / 8.1 | 16% |
| shot_put (1900) | 128 | 0.87 / 0.72 / 0.62 | 1.2 | 7.9 / 9.9 / 21.4 / 18.2 | 3.4 / 3.8 | 2.8 / 6.2 | 84% |
| discus (1900) | 102 | 1.35 / 1.14 / 0.64 | 1.2 | 5.8 / 8.2 / 14.6 / 18.1 | 3.6 / 3.5 | 7.7 / 9.7 | 87% |
| hurdles (1900) | 89 | 1.20 / 6.09 / 1.27 | 0.2 | 14.1 / 12.2 / 17.0 / 19.9 | -0.4 / 1.3 | n/a | 89% |

Noise = 1.4826 x MAD of the residual from a 0.2 s moving average, over the 12 major joints, along the camera's
depth axis and the two image axes (from `meta.json`'s `view_azimuth`). CV = std / mean of each bone's length
over the clip. Floor = each ankle's 5th-percentile height.

What this says:

- **Depth is only 1.1-1.7x noisier than the image plane after smoothing.** The Savitzky-Golay filter
  (window 9, order 3) hides most of the frame-to-frame depth jitter. The damage shows up instead as slow
  bone-length drift and left/right disagreement, which smoothing does not remove.
- **Bone lengths wobble 3-6% on phone clips (1-2 cm on a thigh) and 8-21% on the 1900 films.** The
  proportional retarget fixes lengths to the G1's, which is right, but the direction error that comes with a
  wrong length is kept: a 5% thigh error at a near-straight knee is about 10 degrees of knee.
- **The feet are the worst joints.** In double support the two ankles should be level; MediaPipe disagrees by
  3-8 cm on average and up to 12 cm. The lower ankle is more than 2 cm above its own floor level in 80-90% of
  frames on most clips. Every contact decision (stance, flight, landing, "feet down" predicates) sits on top of
  this, which is why `stance_mask` needs a 10 cm height and 0.6 m/s speed window.
- **Fast motion is where MediaPipe degrades most:** jumping jacks has 3-10x the in-plane noise of the squat and
  18% upper-arm CV (overhead arms). The hurdles clip has 6 cm of horizontal noise: the runner crosses the
  frame, and the root comes from the 2D hip track scaled by the torso length, so every torso-length wobble
  becomes horizontal root error.
- **Root depth is assumed zero** (`human_trajectory`: "depth motion is assumed zero"). Any approach to or
  retreat from the camera is lost, and any camera pan or zoom becomes root motion.

### Raw MediaPipe pass (no SAM crop, whole frame, CPU)

MediaPipe heavy on the whole (square-padded, 1280 px) frame, read with mediapy as `pipeline.py` does, without
the SAM crop. "3D-2D residual" is the per-frame RMS disagreement between MediaPipe's own 3D landmarks and its 2D
landmarks under the best weak-perspective fit (scale + offset), in centimetres of 3D: how self-consistent the
lifting is. "Ruler CV" is the spread of the metres-per-pixel scale `human_trajectory` takes from the torso.

| clip | true fps (meta.json says 30.0) | person height (px) | detected | raw noise x / y / depth (cm) | depth ratio raw (smoothed) | raw bone CV | 3D-2D residual mean (p95) | ruler CV | ankle visibility (frames < 0.5) |
|---|---|---|---|---|---|---|---|---|---|
| squat_jump | 29.19 | 1064 | 100% | 0.31 / 0.26 / 0.70 | 2.4 (1.6) | 5.3% | 2.6 (3.1) | 8.2% | 0.93 (0%) |
| squat | 29.54 | 1079 | 100% | 0.17 / 0.12 / 0.42 | 2.9 (1.7) | 5.4% | 2.5 (3.0) | 5.6% | 0.94 (0%) |
| lunge | 28.97 | 857 | 100% | 0.27 / 0.24 / 0.62 | 2.4 (1.4) | 7.1% | 2.4 (2.9) | 5.4% | 0.87 (4%) |
| arms | 27.73 | 909 | 100% | 0.08 / 0.14 / 0.50 | 4.5 (1.2) | 6.8% | 2.3 (3.0) | 2.2% | 0.86 (0%) |
| front_kick | 29.63 | 843 | 100% | 0.32 / 0.26 / 0.66 | 2.3 (1.3) | 4.8% | 2.3 (2.8) | 2.3% | 0.92 (4%) |
| jumping_jacks | 29.17 | 1016 | 100% | 1.38 / 1.51 / 1.63 | 1.1 (0.9) | 8.6% | 2.9 (3.7) | 2.4% | 0.93 (0%) |
| punches | 28.69 | 1081 | 100% | 0.50 / 0.44 / 0.85 | 1.8 (1.1) | 7.7% | 3.1 (3.5) | 3.7% | **0.71 (82%)** |
| shot_put (1840x1010) | 25.03 | 564 | 100% | 0.80 / 0.47 / 1.05 | 1.7 (1.2) | 12.8% | 3.1 (3.6) | 8.4% | 0.99 (0%) |
| discus (1912x1012) | 24.76 | 539 | 95% | 1.67 / 0.55 / 2.12 | 1.9 (1.2) | 10.7% | 2.7 (3.6) | 4.6% | 0.97 (0%) |
| hurdles (1920x1000) | 24.74 | (small, far left) | **0%** | - | - | - | - | - | - |

- **Depth is 2-4.5x noisier than the image plane before smoothing** (1.1-1.7x after). The Savitzky-Golay
  filter hides the depth jitter but not its slow component, which is what the bone-length CV (5-13%) and the
  double-support ankle disagreement measure.
- **MediaPipe's 3D disagrees with its own 2D by 2.3-3.1 cm RMS per joint**, 3-4 cm at the 95th percentile. That
  is the GHUM lifting's own inconsistency, independent of the camera: a floor on what any IK on these landmarks
  can reproduce.
- **The torso ruler drifts 2-8%.** The root path is 2D hip pixels times this ruler, so root travel carries a
  2-8% scale error on top of the zero-depth assumption. On the shot put's 1.1 m glide that is up to 9 cm.
- **Frame rates are wrong in `meta.json`.** The phone clips are 27.7-29.6 fps variable-rate files and the films
  are 24.7-25.0 fps; every `meta.json` says 30.0. The stored film clips also have more frames than the files
  (shot put 128 vs 107, hurdles 89 vs 73), so the Modal job resampled them. A 3-8% time-scale error reaches the
  retarget's speed cap, the spec's hold seconds, the flight airtimes and the "tempo" claims in the docs.
- **SAM's crop is what makes distant subjects work.** Without it the hurdler (a few hundred pixels tall at the
  left edge of a 1920 px frame) is never detected; with it the pipeline reported 100%. Subject height in the
  films is about 550 px against 850-1080 px on the phone clips, and their noise is 2-3x higher.
- **Punches: the ankles are below 0.5 visibility in 82% of frames** (the camera is too close and the feet are
  cut off or occluded). The feet in that clip are guessed, which is why its root shift reaches 25 cm.

### Is a world-grounded front end better than MediaPipe + IK?

Not run locally (weights need the user's approval; both Colab GPUs were busy). From the literature, with the
caveat that protocols differ between papers:

| model | what it outputs | reported accuracy | what it would fix here |
|---|---|---|---|
| MediaPipe Pose (GHUM lifting) | per-frame 33 landmarks, hip-centred, no root | 3D MAE 36-45 mm on its own test set (aggregator figure, unverified; the official docs report 2D PCK/mAP only) | baseline |
| WHAM (CVPR 2024) | SMPL pose + world trajectory + foot contacts, from video + camera motion (DPVO) | EMDB-2: WA-MPJPE 135.6 mm, foot sliding 4.4, jitter 22.5 | root shift, phantom flights, camera pan, contacts |
| GVHMR (SIGGRAPH Asia 2024) | same, gravity-view coordinates, stationary labels for feet/hands | EMDB-2: WA-MPJPE 111.0 mm (7-10 mm better than WHAM on world metrics) | same, better vertical consistency |
| TRAM (ECCV 2024) | SMPL + camera-scaled trajectory via SLAM | EMDB-2: WA-MPJPE 76.4 mm (as listed by a 2026 follow-up; protocol differs from the GVHMR paper) | same, best on moving cameras |
| SAM 3D Body (Meta, Nov 2025, CVPR 2026) | per-image full-body mesh on the MHR rig (skeleton separate from shape), hands and feet | 3DPW MPJPE about 55 mm, EMDB about 62 mm (third-party reproduction of the checkpoint table) | bone-length consistency, far-side limbs, hands and feet; no world trajectory on its own |

Judgement: the three world-grounded models address exactly the failures measured above (feet, root, flights,
camera), and SAM 3D Body addresses the limb-direction and bone-length failures. None is a drop-in: each needs
the same retarget and the same contact and spec logic afterwards, so the audit metrics here are the right
yardstick. The proposed A/B is in the plan (fix 3).

On the user's question about simulating other camera angles with a world model and running SAM on each: it
cannot add information about what the camera did not see, because the synthesised view is the generator's
prior, not evidence. The same prior is already inside a single-view 3D model, one step shorter. It could help
as test-time augmentation for the 2D detector on blurred or partly occluded frames, which is a small gain for a
lot of GPU. If a Meta-model story is wanted, SAM 3D Body replacing MediaPipe is the direct version.

Sources: [WHAM](https://arxiv.org/pdf/2312.07531), [GVHMR](https://arxiv.org/abs/2409.06662),
[TRAM numbers as listed by OnlineHMR](https://arxiv.org/pdf/2603.17355),
[SAM 3D Body](https://arxiv.org/abs/2602.15989v1), [SAM 3D Body research page](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/),
[MediaPipe pose docs](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md),
[BlazePose aggregator figures](https://www.emergentmind.com/topics/blazepose).

## Stage 2: retarget (`retarget.py`, proportional, current code re-run on every `human.npy`)

| clip | IK error mean (cm): pelvis / L knee / R knee / feet / hands | R knee > 3 cm | any joint at a limit | worst limit joint | frames at the 20 rad/s cap | angle error mean (deg) knee / hip / elbow / shoulder | angle error max | root shift by `plant_feet` mean / max (cm) | no-contact frames |
|---|---|---|---|---|---|---|---|---|---|
| squat_jump | 0.4 / 1.1 / **5.6** / 0.0 / 0.1 | 79% | 5% | L ankle roll 4% | 0 | 2.3 / 4.9 / 0.9 / 3.8 | 12 / 15 / 3 / 12 | 9.4 / 13.9 | 14% |
| squat | 0.4 / 1.1 / **5.0** / 0.1 / 0.1 | 91% | 5% | L ankle roll 5% | 0 | 4.2 / 4.2 / 1.0 / 2.7 | 13 / 15 / 7 / 12 | 9.7 / 13.4 | 0% |
| lunge | 0.4 / 2.4 / 3.5 / 0.1 / 0.1 | 58% | 36% | L ankle roll 16%, L ankle pitch 14% | 0 | 3.0 / 4.9 / 1.1 / 1.8 | 19 / 21 / 3 / 7 | 8.7 / 24.8 | 4% |
| arms | 0.1 / 1.7 / 1.6 / 0.0 / 0.1 | 0% | 0% | - | 0 | 0.8 / 2.1 / 2.1 / 2.5 | 2 / 5 / 5 / 5 | 1.7 / 2.7 | 0% |
| front_kick | 1.3 / 1.8 / **6.1** / 0.2 / 0.1 | 74% | 26% | R ankle roll 22% | 1 | 4.0 / 6.1 / 1.3 / 2.2 | 26 / 22 / 11 / 7 | 6.0 / 15.3 | 0% |
| jumping_jacks | 0.1 / 1.5 / 1.8 / 0.0 / 0.4 (L hand max 13) | 15% | 13% | L ankle roll 13% | 9 | 1.7 / 3.0 / 3.0 / 2.9 | 21 / 13 / **43** / **41** | 6.3 / 16.8 | 49% |
| punches | 0.8 / 3.2 / 3.8 / 0.1 / 0.1 | 64% | 0% | - | 0 | 5.3 / 5.3 / 0.8 / 1.6 | 20 / 16 / 2 / 3 | 4.3 / 25.6 | 0% |
| shot_put | 0.9 / 2.3 / 3.8 / 0.1 / 0.3 (L hand max 18) | 53% | **94%** | R ankle roll 86%, R shoulder pitch 40% | 7 | 5.4 / 7.1 / 3.5 / 8.7 | 36 / 26 / **41** / **49** | 13.7 / 22.4 | 58% |
| discus | **2.7** / **6.2** / **9.8** / 0.5 / 0.4 (R hand max 11) | 91% | **77%** | L ankle roll 37%, R knee 28% | 11 | **12.8** / 6.7 / 4.2 / 4.2 | **91** / 33 / **62** / 26 | 10.2 / 18.8 | 2% |
| hurdles | 1.3 / 2.5 / 3.8 / 0.3 / 0.2 | 66% | **66%** | R ankle roll 64%, L ankle roll 40% | 4 | 1.7 / 2.2 / 2.3 / 2.3 | 7 / 6 / **46** / 10 | 4.5 / 5.4 | **88%** |

Angle errors are robot vs the raw human skeleton (`flexion_angles`), averaged over both sides. Root shift is
how far `plant_feet` moved the hip midpoint from the pose estimate's path to keep planted feet from sliding.

Findings:

- **Knees are where the IK gives up.** Pelvis, feet and hands have 20x, 10x and 10x the knee's weight, so when
  the human knee is not on a circle the G1 leg can reach (depth error puts it off the hip-ankle plane), the
  knee target is sacrificed: 5-10 cm mean error, over 3 cm in most frames. The right knee is 3-5x worse than the
  left on all four home clips where it matters; the camera sees the athlete's left side nearer on those clips,
  but the same asymmetry appears on the 1900 films where the right side is nearer, so the cause is not settled.
  In angle terms it is 2-5 degrees mean and 12-26 degrees at the worst frame on phone clips, and 13 degrees mean
  / 91 degrees max on the discus.
- **Ankle roll lives at its limit.** `level_feet` tilts the lower body to make the ankles level, the pelvis
  roll follows the hip line, and the foot orientation is held flat with the contact's yaw. Whatever roll that
  leaves is asked of the G1's ankle roll joint, which has a small range, so it saturates in 4-22% of frames on
  phone clips and 34-86% on athletic clips. When a joint saturates the error goes somewhere unmeasured (the
  foot's actual orientation, or the knee).
- **The 20 rad/s cap is a patch over a singularity.** It fires in 1-11 frames per clip on 5 of 10 clips. It
  removed the 5.8 rad shoulder flip, but a capped arm needs 9 frames to cross 5.8 rad, and during those frames
  the hand lags by up to 13 cm (jumping jacks) and 18 cm (shot put). Arms straight overhead (jacks, shot put,
  discus release) hit the G1 shoulder's pitch/roll/yaw ordering ambiguity; the fix is in the arm IK (an elbow
  orientation target or a quaternion-continuous shoulder solve), not in the speed limit.
- **Root shift of 4-14 cm mean, up to 25 cm**, is the retarget's own correction for the pose's feet. On the
  punches the root is moved 25 cm at the worst frame to keep a foot that MediaPipe wandered. On the shot put
  (14 cm mean) and discus (10 cm) this is a meaningful share of the athlete's actual travel (about 1.1 m for the
  glide). The body's path is therefore partly an artefact of which foot was declared planted.
- **Contact detection is the single point of failure for anything faster than a walk.** `stance_mask` needs the
  ankle within 10 cm of its floor and under 0.6 m/s for at least 4 frames. On the hurdles clip the near-floor
  ankle speed is 1.35 m/s at the median (0.34 at the 10th percentile), so contacts are found in 7% and 6% of
  frames for the two feet; a sprinter's foot is on the ground for 2-3 frames at 30 fps, under the 4-frame
  minimum. The shot put's pivoting stance foot is found 17-31% of the time.

### What `ground_phantom_flights` gets wrong

The rule: a stretch with no planted foot is a flight only if the pelvis rises at least 0.25 of g T^2 / 8 for
its length T; otherwise the body is lowered until a foot is on the floor. Measured on every no-contact stretch:

| clip | stretch | length | pelvis rise | needed (0.25 x ballistic) | ballistic rise for that airtime | decision | effect |
|---|---|---|---|---|---|---|---|
| squat_jump | 2.77 s, 6.90 s | 0.63 s each | 39.6, 37.9 cm | 12.3 cm | 49.2 cm | flight (correct) | - |
| jumping_jacks | 7 hops | 0.13-0.43 s | 2.7-14.4 cm | 0.5-5.8 cm | 2.2-23 cm | flights (correct) | - |
| lunge | 5.70 s | 0.17 s | 3.5 cm | 0.9 cm | 3.4 cm | flight | a foot shuffle counted as a hop |
| shot_put | 1.10 s | 0.33 s | 5.8 cm | 3.4 cm | 13.6 cm | flight | doubtful: the film shows a hop in the glide, plausible |
| shot_put | 2.17 s | 1.63 s | 5.9 cm | 81.8 cm | 327 cm | grounded, lowered 5.9 cm | correct: the pivoting foot was never off the floor |
| **hurdles** | 1.23 s to the end | 1.77 s | 18.5 cm | 95.7 cm | 383 cm | **grounded, lowered up to 70 cm** | **wrong: the stretch is the run-up plus the real clearance** |

Three defects:

1. **It reasons about stretches between detected contacts, so a missed contact merges a run-up with a jump.**
   The hurdles run has no detected contacts, so the whole clip after 1.23 s is one 1.77 s stretch that no
   jump could satisfy, and the real clearance (pelvis +60 cm in the pose estimate) is flattened to the floor.
   Any running approach (long jump, hurdles, a layup) will do the same.
2. **A stretch that runs to the clip's end has no landing frame, and the rise is measured against the
   airborne end frame itself**, so even a true flight at the end of a clip reads as little rise.
3. **The threshold is loose at 0.25.** Real flights in the pose data have a rise of 0.5-0.8 of the ballistic
   value (the pelvis rise is under-read, or the feet count as airborne before take-off because they hover), so a
   threshold near 0.4 would still keep them while rejecting the 0.33 s shot-put hop at 0.43 as doubtful. The
   lunge's 0.17 s "flight" (3.5 cm rise, ratio 1.0) is a shuffle that only passed because its length is short.

Also: the stored `qpos.npy` in `outputs/remote_process_pro` for the hurdles is the pre-fix retarget (pelvis
rises to 1.16 m), so the "ends mid-air" exclusion in the generalist was made on a reference that flew. The
current code would hand the generalist a hurdler who never leaves the ground.

## Stage 3: key-moment spec (`skill/keypose.py`)

Compiled from the stored `qpos.npy` (what the policies trained on):

| clip | stages (poses, flights) | required | RDP eps (cm) | noise sigma pelvis / wrist / foot (cm) | min tolerance / sigma | wrist tolerances (cm) min / median | self-check | 1.4x tempo | 0.7x tempo |
|---|---|---|---|---|---|---|---|---|---|
| squat_jump | 12 (7, 2) | 8 | 4.8 | 0.29 / 0.36 / 0.07 | 14 | 10.0 / 12.5 | 12/12 | 12 | 12 |
| squat | 9 (7, 0) | 5 | 3.8 | 0.02 / 0.17 / 0.05 | 64 | 10.0 / 18.4 | 9/9 | 9 | 9 |
| lunge | 16 (15, 0) | 4 | 7.0 | 0.26 / 0.38 / 0.16 | 16 | 14.1 / 20.0 | 16/16 | **3** | 16 |
| arms | 3 (1, 0) | 3 | 3.1 | 0.04 / 0.17 / 0.06 | 160 | 20.0 / 20.5 | 3/3 | 3 | 3 |
| front_kick | 6 (4, 0) | 5 | 6.6 | 0.09 / 0.45 / 0.12 | 34 | - | 6/6 | 6 | 6 |
| jumping_jacks | 13 (10, 1) | 6 | 6.8 | 1.14 / 2.91 / 2.10 | **2.9** | 10.0 / 18.4 | 13/13 | 13 | 13 |
| punches | 7 (6, 0) | 4 | 5.3 | 0.12 / 0.66 / 0.16 | 8.7 | 10.0 / 12.3 | 7/7 | 7 | 7 |
| shot_put | 4 (3, 0) | 4 | 15.5 | 0.41 / 0.83 / 0.36 | 19 | 26.2 / 26.9 | 4/4 | **2** | 4 |
| discus | 2 (1, 0) | 2 | 27.4 | 0.45 / 1.67 / 0.57 | 50 | - | 2/2 | 2 | 2 |
| hurdles | 4 (0, 1) | 4 | 35.5 | 0.84 / 1.67 / 0.97 | - | - | **2/4** | 2 | 2 |

Robustness of each spec to perturbations of its own reference (8 trials each; noise is temporally correlated,
0.15 s; pelvis, torso and foot heights kept exact, as a simulated robot's are):

| clip | 2 cm limb noise | 4 cm | 6 cm | 4 cm, holds off | 6 cm, tolerances x3 |
|---|---|---|---|---|---|
| squat_jump | 100% | 100% | 100% | 100% | 100% |
| squat | 100% | 100% | 100% | 100% | 100% |
| lunge | 75% | **25%** | 25% | **100%** | 100% |
| front_kick | 100% | 88% | 100% | 100% | 100% |
| jumping_jacks | 100% | 100% | 100% | 100% | 100% |
| punches | 100% | 88% | **25%** | 100% | **100%** |
| shot_put | 62% | **50%** | 50% | 75% | 100% |
| discus | 100% | 100% | 100% | 100% | 100% |

Key-moment stability: re-compiling the spec after 1 cm of noise re-finds 52% (jumping jacks), 60% (squat),
72% (punches), 79% (squat jump), 83-100% (the rest) of the original key moments within 0.15 s, and the count
changes by up to 40%.

Findings:

- **How often is the derived key moment wrong or untestable?** On these ten clips: hurdles untestable (ends
  mid-air: 1 of 10). Discus and arms compile to a single key moment, so "success" there is a loose bar (the
  discus policies "succeed" at 100% with 0.23 rad of joint error, twice the phone-clip level). Punches (6
  moments, all wrist-driven) is testable but its tolerance floor is the problem below. Jumping jacks' 10 key
  moments sit at 2.9 noise sigmas, the only clip where the RDP tolerance is near the noise. So roughly 3 of 10
  specs are not a fair test of the action, and the key-moment set itself moves by 20-50% under the clip's own
  noise.
- **Tolerances are floors, not calibration.** `TOL_MIN` (4 cm pelvis, 5 cm torso, 10 cm wrists, 4 cm foot
  height, 12-15 cm root) dominates: wrist tolerances are at the 10 cm floor on every punching key moment, and
  they are 15-160 noise sigmas on calm clips. The noise term (6 sigma of MAD) never binds except on the
  jumping jacks. So the pose-noise self-calibration exists but is inactive; what binds is the fixed floor
  compared with the policy's tracking error, which is unknown per feature.
- **Holds, not tolerances, are what break under noise.** Turning holds off takes the lunge from 25% to 100%
  at 4 cm noise and the shot put from 50% to 75-100%. The hold counter resets whenever the predicate flickers
  out of tolerance for one control step, so a rollout that sits at the tolerance edge, as a 0.14 rad tracker
  does, never accumulates its 0.2-0.4 s.
- **Holds are absolute seconds, so timing is not free.** A reference played 1.4x faster fails the lunge (3/16)
  and the shot put (2/4): each hold is 0.5 x the reference's dwell, but the faster performance dwells 1/1.4 as
  long. Slowing down passes everywhere. The variants audit (next section) shows the same thing: 13 of the 21
  dropped lunge variants fail their own self-check at tempo 1.33 or amplitude 0.85.
- **Punches:** the tracking policy completes 4.95 of 7 stages with 0.14 rad RMS joint error and 3.8 cm torso
  error (`remote_multiclip_adapt_punches` robust.json), stuck at "left hand left, right hand down" (left
  wrist y 10 cm, right wrist z 10 cm, both at the floor, with a hold). A 0.14 rad error over the G1's 0.37 m
  arm is 5 cm of hand error before shoulder error is added, so a good tracker lives at the edge of a 10 cm box
  with a hold that resets on each excursion. The eval does not record which feature failed; that is the first
  thing to add (plan, fix 2).
- **The "upright" gate and fall rule measure the pelvis-to-torso_link vector, which is 4.4 cm long on the G1.**
  Exact in sim, so harmless there, but it is not torso lean, and it makes any evaluation on estimated positions
  (a real robot, or a reference check with pose noise) read falls that did not happen. My first brittleness
  test fell into this: with 2 cm of noise on every body the fall rule fired at 2.5 s in every clip.

### Per-clip tolerance calibration from the pose noise itself

The spec already computes a per-feature noise sigma from the retarget. Proposal:

- tolerance_i = max(TOL_MIN_i, k_noise x sigma_i, TOL_FRAC x |delta_i|) with k_noise about 3, as now, but
  **also** a per-feature floor from the tracking policy's own error: measure the tracking policy's per-body
  RMS error against the reference on the clean clip (already available in `robust.json` as a scalar, needs to
  be per body) and set TOL_MIN_i = max(TOL_MIN_i, 2 x that RMS). Then "a good tracker passes its own clip" is
  guaranteed by construction instead of hoped for.
- holds: require the predicate for hold_s within a window, not consecutively (for example 80% of the steps in
  the last hold_s), and express hold_s as a fraction of the time the rollout itself spends near the goal, or
  of the stage's reference duration scaled by the rollout's tempo, so a faster performance is not penalised.
- record, per rollout and stage, the feature with the largest normalised error at the time the stage was
  abandoned. One extra field in `evaluate()`.

## Stage 4: variants (`skill/variations.py`, TRAIN grid, 36 per clip)

| clip | kept | dropped (reason) | hops in base | kept variants that lost the hop | flights with rise / ballistic outside [0.6, 1.5] | kept variants with a joint above 20 rad/s (max) |
|---|---|---|---|---|---|---|
| squat_jump | 36 | 0 | 2 (ratios 1.01, 0.98) | 0 | 0 | 0 |
| squat | 36 | 0 | 0 | - | - | 0 |
| lunge | **15** | 21 (8 limits, 13 self-check 3-7/16) | 0 | - | - | 0 |
| arms | 36 | 0 | 0 | - | - | 0 |
| front_kick | 36 | 0 | 0 | - | - | **15** (24 rad/s) |
| jumping_jacks | 36 | 0 | 1 (ratio 1.25) | **14** | **3** (ratio 1.6) | **36** (45-82 rad/s) |
| punches | 36 | 0 | 0 | - | - | 0 |
| shot_put | 24 | 12 (6 limits at 9%, 6 self-check 2/4) | 0 | - | - | **24** (76-172 rad/s) |
| discus | 24 | 12 (limits) | 0 | - | - | **24** |
| hurdles | 0 | 36 (self-check) | 1 | - | - | - |

Findings:

- **The filter checks two things: joint limits (at most 2% of samples clipped) and the variant's own
  self-check.** It does not check joint speed, flight ballistics or foot penetration. So it keeps every
  jumping-jack variant at 45-82 rad/s and every shot-put variant at 76-172 rad/s (those come from the pre-fix
  stored retargets, which the variant builder in the adaptation runs did not use, but the filter would have let
  them through), and 15 front-kick variants at 20-24 rad/s because tempo 1.33 multiplies joint speed and the
  cap applies only to the retarget.
- **Amplitude scaling breaks flights.** Scaling joints and root about frame 0 by 0.85 lowers a 4 cm hop's
  pelvis rise and foot clearance below the air threshold, so 14 of 36 jumping-jack variants have no flight
  stage at all while the body still hops; at 1.1 the rise grows but the airtime does not, so the flight rises
  1.6x what gravity allows. The tempo warp correctly leaves airborne frames at 1x, but amplitude needs the same
  care: airtime = sqrt(8 x rise / g) after scaling.
- **Self-check drops are the spec's tempo problem again.** 13 lunge variants and 6 shot-put variants fail
  their own self-check, all at tempo 1.33 or amplitude 0.85: the compiled holds and tolerances are not
  invariant to the transformations the variant builder applies. These are the training examples closest to
  "the same move, a bit faster or smaller", which is what the variants are for.
- **Limit drops are honest** (deep lunge at 1.1, shot put at 1.1 with 9% clipped) but they say something about
  the retarget: the base clip is already grazing the ankle limits, so a 10% larger version crosses them.

## Stage 5: coverage (`clips/CANDIDATES.md`, `docs/PRO_CLIPS.md`)

Of 13 shortlisted and 8 downloaded pro clips, 3 were kept and 1 of those (hurdles) is unusable. Rejections:
panning camera (2), distant + crowded + cuts (1), the jump never shown (1), spliced out of order (1), cut before
landing (1). Mapping each to the pipeline:

- **Panning or zooming camera.** `human_trajectory` builds the root from the 2D hip track in pixels times a
  torso-length ruler. A pan is indistinguishable from the athlete moving; a zoom changes the ruler. What it
  takes: estimate camera motion from the background (feature tracks outside the SAM mask, a per-frame
  homography; OpenCV, CPU, about a day) and subtract it from the hip track; or use a front end that solves the
  camera (WHAM/TRAM with DPVO, GVHMR). The homography route handles pan and zoom on a static scene; it does not
  handle a camera that translates (a dolly) without a depth model.
- **Multi-person frames.** SAM 3.1 picks the object with the highest summed confidence over the clip
  (`main_track_masks`), and the crop greys out everything outside the dilated mask, so MediaPipe cannot latch
  onto a bystander once the mask is right. The failure is choosing the wrong person or an identity switch. What
  it takes: a point or box prompt from the user's click in the web app (SAM 3.1 supports it), and a continuity
  check (mask IoU between frames; re-prompt on a drop). Half a day.
- **Distance.** Pose noise scales with 1 / person height in pixels. The raw pass gives each clip's pixel
  height: 850-1080 px on the phone clips, about 550 px on the films (2-3x the noise), and the hurdler is not
  detected at all without SAM's crop. A threshold (about 400 px of subject height after the crop) belongs in
  the intake checks, and the crop should be upsampled to MediaPipe's input size rather than left small.
- **Frame rate.** Record the file's real fps (the phone clips are 27.7-29.6 fps, the films 24.7-25.0 fps;
  `meta.json` says 30.0 for all ten) and resample to a fixed rate explicitly.
- **Cuts and incomplete motions.** No pipeline fix; the intake check "starts and ends on the floor" already
  exists in the generalist. A shot-boundary detector (histogram difference) would reject spliced clips before
  processing.

## Plan: the top three, concretely

### Fix 1 (1.5 days): contacts that survive running, and a phantom-flight test that reasons about take-offs

Where: `retarget.stance_mask`, `retarget.ground_phantom_flights`, `retarget.proportional_motion`.

1. Replace the single speed/height window with a two-signal contact estimator per foot: (a) the ankle's
   vertical velocity crosses zero at a local height minimum within `height` of the floor, (b) horizontal speed
   below a threshold that scales with the root's speed (a running foot is at rest relative to the ground but
   the pose's ankle still shows about 0.3 m/s of jitter). Allow contacts of 2 frames at 30 fps. Keep the
   existing fill-short-gaps logic.
2. Phantom-flight test per **take-off**, not per stretch: a stretch becomes a flight only from the last frame
   where the lowest foot was within `clear` of the floor and moving up; its length is measured to the first
   frame where a foot comes back down (or the clip ends: then treat the stretch as a flight if the pelvis is
   still above its take-off height, and never lower it). Compare the pelvis rise with g T^2 / 8 at a threshold
   of about 0.4 (real flights measure 0.5-0.8; the shot-put glide measures 0.02).
3. Where no contact is detected for more than 0.5 s and the pelvis did not rise, do not lower the body to the
   floor; instead interpolate the floor through the lowest foot's local minima (a running athlete's feet touch
   down at each stride) and log a warning in `meta.json`.
4. Measure before and after on the hurdles, shot put and jumping jacks with `retarget_audit.py`: no-contact
   fraction, phantom lowering, root shift, the clearance of the hurdle (pelvis +60 cm must survive).

Expected gain: the hurdles reference becomes usable (today it is destroyed by the current code), the shot put
glide keeps its foot pivot without a 1.1 m travel being half absorbed into root shift, and the 4-14 cm mean
root shift on every clip drops because fewer wrong frames are declared planted.

### Fix 2 (1 day): a success metric that a good tracker passes

Where: `skill/keypose.py` (`_compile`, `KeyposeMachine.step`), `skill/task.py evaluate`, `experiments/robustness_eval.py`.

1. Holds tolerant to flicker: count the predicate over a sliding window of hold_s and advance when at least
   80% of its steps satisfied it. Today one step out resets the counter.
2. Holds relative to time, not seconds: hold_s = HOLD_FRAC x the reference dwell x (rollout duration /
   reference duration), or, simpler, cap hold_s at 0.5 x the stage's reference duration and let a faster
   rollout satisfy it in proportion. Check: the reference at 0.7x and 1.4x must pass every spec (today the
   lunge and shot put fail at 1.4x).
3. Tolerance floor from the tracker: extend the eval to record per-body RMS error of the tracking policy on the
   clean clip, and set each feature's floor to max(TOL_MIN, 2 x that RMS). Re-score punches.
4. Diagnostics: `evaluate()` returns, per abandoned stage, the feature with the largest normalised error, and
   `robust_summary.md` prints it. This is the measurement tool that would have explained punches in one line.
5. Replace the pelvis-to-torso_link "upright" vector with pelvis-to-shoulder-midpoint (0.4 m lever arm) for
   the gate and the fall rule; same meaning in sim, usable with estimated positions later.

Expected gain: the success rates in every doc become comparable across clips; punches-type false zeros stop;
the variant builder stops dropping the "a bit faster" lunges.

### Fix 3 (2-3 days plus GPU; needs approval to download weights): the front-end A/B

Candidates: SAM 3D Body (Meta; per-frame skeleton on the MHR rig, consistent bone lengths, hands and feet) and
GVHMR or WHAM (world trajectory with foot contacts). Clips: squat_jump (clean baseline), shot_put (pivot,
limits), hurdles (running, camera-crossing, ends mid-air). Protocol:

1. Run each model on the three clips on one Colab GPU; convert output joints to the 33-landmark layout
   `human_trajectory` expects (or add a loader that takes joints + root + contacts directly).
2. Score the same metrics as this audit, before retargeting: bone-length CV, double-support ankle disagreement,
   stance hover, depth-to-image noise ratio; after retargeting with the current code: knee IK error, limit-hit
   fraction, cap hits, root shift, phantom-flight decisions. For the world-grounded models, feed their contacts
   into `plant_feet` instead of `stance_mask` and measure root shift again.
3. Decision rule: adopt a front end if it halves the double-support ankle disagreement and the knee IK error on
   the two athletic clips without worsening the squat jump. Expect SAM 3D Body to win on limbs and the
   world-grounded models on feet and root; the likely answer is SAM 3D Body for the skeleton plus either a
   world-grounded trajectory or the background-homography root (coverage fix) for the camera.

Cost: about 1 GPU hour for inference on three short clips; the time is in the loaders and the comparison.

### Fixes 4-7, briefly

- **Variants (0.5 day):** add to the filter: max joint speed above the cap, any flight with rise / ballistic
  outside [0.6, 1.5], and penetration; after amplitude scaling re-time each flight to sqrt(8 x rise / g).
- **Retarget (1 day):** ankle roll: project the leftover roll into the ankle range by letting the pelvis roll
  absorb the rest, instead of saturating; shoulder: an elbow orientation target or a continuous shoulder solve
  so the 20 rad/s cap is never the active constraint; knees: give the knee target its component in the hip-ankle
  plane only (the out-of-plane part is depth noise the G1 cannot and should not follow).
- **Coverage (1.5 days):** background homography camera compensation; click-to-select athlete prompt; a
  pixel-height intake check.
- **Hygiene (0.5 day):** write the retarget code hash, parameters and the file's real fps into `meta.json`,
  re-process the ten clips with one version, and mark which tables in the docs were computed on which.

## What was not measured

- No world-grounded model or SAM 3D Body was run (weights need approval); the front-end comparison is from the
  literature and the error profile.
- Policy rollouts for punches are not stored locally, so the failing feature at its stuck stage is inferred
  from the spec's tolerances and the policy's scalar joint error, not read off the rollout.
- The right-knee asymmetry is reported without a settled cause.
- Pixel height of the SAM crop itself (masks are not kept in `outputs/`), so the "distance" threshold is an
  estimate from the whole-frame heights.

## Fixes applied

### Fix 6: coverage (camera compensation, athlete selection, intake checks), 2026-10-30

What changed:

- **`camera.py` (new): background camera compensation.** For every frame, the motion that maps it onto the
  first frame is estimated from the background only. The athlete is masked out with SAM's mask joined to a
  silhouette drawn from the pose landmarks, so `--no-sam` works too. `pose.human_trajectory(camera=...)`
  maps the 2D landmarks through it before lifting to metres, so a pan or zoom no longer becomes root motion
  and the torso ruler is measured in one image plane.
  - Shi-Tomasi corners, tracked with pyramidal Lucas-Kanade (forward-backward checked) against a keyframe
    rather than chained pair by pair, so errors do not add up. RANSAC fit. ORB, then phase correlation, as
    fallbacks.
  - Motion model: a similarity (pan, tilt, roll, zoom). A full homography is available
    (`estimate(model="homography")`) but was less stable on frames with few features.
  - Static cameras are a strict no-op: under 2 px of accumulated motion at the athlete, every matrix is the
    identity and the root is bit-identical.
  - An estimate is "unreliable" when fewer than 80% of the distinct frames register from features, as on the
    1900 films (grain, flicker, mostly repeated frames). It is then applied only to a clear pan, over 5% of
    the frame width.
  - Like the audit said, a homography covers rotation and zoom, not a camera that translates.
- **`athlete.py` (new): athlete selection.** `pipeline.py --athlete` takes `x,y` (a click), `x0,y0,x1,y1` (a
  box) or an index counted from the left, on the first frame.
  - With SAM it picks that object from SAM 3.1's outputs instead of the most-seen one
    (`track.main_track_masks(select=)`; `sam_track.py --athlete`).
  - Without SAM a CPU tracker follows the chosen person: MediaPipe in a crop around the last position,
    upscaled so distant athletes are detected; a torso-colour check so it does not jump to someone else;
    full-frame re-detection after a loss.
  - `athlete.continuity` flags mask jumps (identity switches or same-shot splices).
- **Intake checks, in `camera.json` beside each clip's outputs (the `meta.json` writer is untouched):**
  - the athlete's pixel height (`pose.height_check`, a warning under 400 px);
  - shot cuts (colour-histogram jumps);
  - hip teleports and mask jumps (splices of the same shot, which a histogram cannot see);
  - frames where camera motion could not be measured.
- `pipeline.py`: a minimal hook (`camera.intake` before `human_trajectory`; `--athlete`, `--no-camera`).

Measured on CPU, no downloads. The scripts are in `experiments/camera_coverage/` and the JSON in
`docs/results/camera_coverage/`. Each clip's landmarks were cached on the whole frame (phone clips) or with
the CPU tracker (films), because SAM masks are not stored.

**Static clips and synthetic pans.** Static: the clip as filmed; "root change" is the maximum change in any
root coordinate with the camera stage on vs off. Synthetic: a cropped copy of the same clip whose camera pans
to follow the athlete with a sway, tilts and zooms in 10%, with MediaPipe re-run on the warped video. Cells
are mean / max cm of the pelvis from the static clip's pelvis, so the "true camera" column is the floor that
pose noise on the re-cropped video allows.

| clip | height px | static: detected / root change | synthetic camera motion (px) | camera error, mean / max (px) | uncompensated | compensated | true camera |
|---|---|---|---|---|---|---|---|
| squat_jump | 1377 | yes / 0.00 cm | 59 | 0.13 / 0.6 | 4.9 / 7.1 | 0.7 / 2.0 | 0.7 / 2.0 |
| squat | 1361 | yes / 0.00 cm | 53 | 0.08 / 0.3 | 5.3 / 7.3 | 0.8 / 1.7 | 0.8 / 1.7 |
| lunge | 1078 | yes / 0.00 cm | 192 | 0.45 / 0.9 | 17.6 / 31.2 | 3.1 / 7.1 | 3.1 / 7.1 |
| arms | 1127 | yes / 0.00 cm | 100 | 0.11 / 0.5 | 10.8 / 15.5 | 0.6 / 1.3 | 0.6 / 1.3 |
| front_kick | 1027 | yes / 0.00 cm | 132 | 0.21 / 0.5 | 12.9 / 21.7 | 2.1 / 2.8 | 2.1 / 2.7 |
| jumping_jacks | 1313 | yes / 0.00 cm | 70 | 0.16 / 0.3 | 7.7 / 10.3 | 0.5 / 1.1 | 0.5 / 1.1 |
| punches | 1316 | yes / 0.00 cm | 95 | 0.07 / 0.3 | 7.4 / 13.5 | 1.0 / 1.6 | 1.0 / 1.6 |
| shot_put (1900) | 699 | yes (unreliable) / 0.00 cm | 203 | 11.1 / 18.7 | 29.3 / 56.9 | 7.3 / 9.9 | 2.9 / 4.6 |
| discus (1900) | 685 | yes (unreliable) / 0.00 cm | 139 | 59.3 / 92.9 | 19.3 / 40.4 | 16.8 / 26.3 | 9.2 / 16.1 |
| hurdles (1900) | 416 | yes (unreliable) / 0.00 cm | 463 | 10.5 / 17.6 | 94.9 / 207.6 | 6.9 / 11.3 | 6.1 / 8.8 |

- **No regression on static clips.** All ten keep their root bit for bit.
  - On the phone clips the measured motion is 0.1-0.6 px.
  - The films read 12-32 px with an unreliable estimate and are left alone. Hurdles probably does drift about
    27 px (phase correlation on the roof agrees), so that is a correction this conservative rule forgoes.
- **On a phone-quality pan the compensation is as good as knowing the camera.** Camera error is under 1 px
  and the root equals the true-camera root to 0.1 cm, against 5-31 cm of phantom travel without it.
- **On the films it removes most but not all of the pan.** Grain and repeated frames leave 10-60 px of
  error. Hurdles drops from 95 / 208 cm to 7 / 11 cm, discus from 19 / 40 to 17 / 26.
- The pixel heights confirm the audit's estimate. The films are at 416-699 px with the tracker's crop.
  Without it, MediaPipe on the whole frame finds the hurdler in 0% of frames (shot put 73%, discus 89%);
  with the CPU tracker, 100% for all three (the hurdler needs a box on frame 0).

**The rejected pro clips** (`docs/PRO_CLIPS.md`, already on disk from the first download; CPU tracker with a
box on the first frame, since SAM needs a GPU). Root numbers are over the frames where the tracker was
verifiably on the athlete (checked on a contact sheet). Speeds are 0.3 s means of the pelvis's in-plane
speed.

| clip (why rejected) | athlete px | camera | root without compensation | root with compensation | plausible now? |
|---|---|---|---|---|---|
| High jump, Sudamericanos (distant, crowded, cuts) | 274 (warned) | pans 1740 px following the curved run-up | 0.2 m net over the 3.7 s run-up, peak 2.1 m/s: running on the spot | 12.3 m, accelerating from 0.5 to 5.1 m/s | **run-up yes.** The flop is lost: MediaPipe does not detect the inverted body over the bar, and the tracker ends on an official (flagged). No histogram cut was found in the source. |
| Javelin, EuroGames (distant, panning) | 200 (warned) | pans 1470 px with the thrower | 0.4 m over the 1.9 s approach, peak 1.5 m/s | 5.5 m, 2.3 to 5.0 m/s | **approach yes**, a typical javelin run-up speed. The thrower is lost in the crowd at the throw (frame ~60, flagged); 200 px is too small. |
| Biron running long jump (distant, crowded, panning) | 1115 | whip pan, 6200 px | net -1.1 m, peak 9.8 m/s | 11.8 m net, but peaks at 20 m/s and the pelvis dips 1.9 m | **no.** The pan is removed, but the jumper runs straight at the camera (300 to 1100 px tall). The fixed torso ruler and zero-depth assumption are wrong for that, which no homography fixes. |
| Sweeney long jump (spliced out of order) | 385 (warned) | static, unreliable estimate, left alone | | | **flagged.** Mask jumps at the splices (frames 20-29, 56, 85, 116 of the excerpt). The histogram cut test finds nothing on a same-shot splice. |
| Sweeney high jump (cut before landing) | | | | | not run: an incomplete motion is not a camera problem. The "starts and ends on the floor" intake check covers it. |

What is left:

- **Depth motion.** Clips filmed head-on or with a depth component need a per-frame ruler or a
  world-grounded front end (Fix 3).
- **Inverted or distant bodies.** MediaPipe loses them (the flop over the bar; the 200 px thrower in a crowd).
  SAM's masks with a click prompt keep identity, but the pose is still MediaPipe's.
- **Not run on a GPU.** SAM with `--athlete` selection is tested on fake SAM outputs only.
- **Other entry points.** `experiments/process_clips.py` calls `video_to_motion` with defaults, so camera
  compensation is now on there. It is a no-op on all ten stored clips, per the table above.
  `server/live_worker.py` does not call the camera stage yet.
