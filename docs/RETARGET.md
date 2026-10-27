# Proportion-aware retargeting (human → G1)

`retarget.human_to_targets(P, rt)` now retargets **proportionally** by default.
`proportional=False` keeps the original method for comparison: the whole human is scaled uniformly by
leg length, then the IK chases absolute elbow, hand and knee positions.

## Why uniform scaling was wrong

After uniform scaling by leg length (×0.83 on our clips), the human still doesn't have the G1's shape:

| bone (m) | G1 | human × 0.83 |
|---|---|---|
| hip width (hip-joint centres) | 0.22 | 0.14 |
| hip → shoulder midpoint | 0.41 | 0.40 |
| shoulder width | 0.28 | 0.27 |
| upper arm / forearm | 0.19 / 0.18 | 0.17–0.25 / 0.15–0.21 |
| thigh / shin | 0.33 / 0.32 | 0.28–0.34 / 0.31–0.37 |

The G1's hips are about 60% wider. On top of that, MediaPipe's own left and right bones differ by up to
20% (depth error), so uniform scaling hands the IK targets that no G1 configuration can reach. The IK
then compromises. Elbows straighten to reach a hand that sits too far out, feet slide, and joint angles
stop matching the human's: the mismatch you see when your arm length isn't the robot's.

## Method (`retarget.proportional_motion`)

1. **Measure the G1 skeleton** (`G1Retargeter._measure_robot`). The G1's hips and shoulders are three
   offset hinges, not ball joints. A least-squares sphere fit (`ball_joint_centre`) gives each one's
   effective centre in the pelvis or torso frame, with under 3 mm of residual. Bone lengths run from
   those centres to the frames the IK drives: knee, ankle-roll, elbow and wrist-yaw links.
2. **Rebuild the human on G1 bones** (`rescale_bones`). Walk the tree from the hip midpoint. Every bone
   keeps the human's direction and takes the G1's length, so every joint angle is the human's and the
   hands and feet land where the G1 can reach. The shoulder line is squared onto the spine, like the
   G1's rigid torso.
3. **Root translation** is scaled by the leg-length ratio, so jump height and squat depth scale with the
   legs.
4. **Contacts.**
   - `stance_mask` marks a foot as planted when it is near its floor level and slow. Thresholds are loose
     (10 cm, 0.6 m/s) because a planted MediaPipe ankle wanders about that much.
   - `level_feet`: in double support, the lower body is tilted rigidly about the hip midpoint until both
     ankles are level. This moves MediaPipe's left/right height error into pelvis tilt instead of
     stretching one leg and folding the other; near a straight knee, 1 cm of ankle is about 10° of knee.
   - `plant_feet` pins each contact at its mean position on the floor. While a foot is planted the root
     is carried by the stance legs, and between contacts it blends linearly back to the human's path, so
     planted feet don't slide by construction.
   - A planted foot keeps its contact's yaw. Its pitch stays flat unless the shin leans past the G1's
     ankle-pitch range (a lunge's rear heel), and then the ankle rises so the sole spheres stay on the
     floor. MediaPipe's heel and toe landmarks are too noisy to use: a still stance can read 35°.
5. **IK targets.**
   - The pelvis is placed by its **hip-joint centre**, re-anchored to the pelvis's current tilt every
     iteration. Its orientation follows the hip line for yaw and roll; pitch is half the trunk's lean,
     because the G1's waist pitches only ±30°.
   - Elbows and hands are driven **relative to the torso** (`mink.RelativeFrameTask`). If the torso can't
     follow the human, the arm's pose relative to the shoulder still survives.
   - The torso and foot orientation targets work as before.

## How others do it

- **H2O / PHC** ([He et al. 2024, arXiv:2403.04436](https://arxiv.org/abs/2403.04436)) fit an SMPL body
  shape to the robot, then pose that robot-shaped SMPL with the human's joint rotations and run IK on its
  keypoints. That is the same idea as step 2. We have MediaPipe landmarks rather than SMPL, so we rescale
  bones directly.
- **GMR** ([Araujo, Ze et al. 2025, arXiv:2510.02252](https://arxiv.org/abs/2510.02252)) uses per-body
  non-uniform scaling followed by two-stage IK (mink). Its main finding: retargeting artifacts (foot
  sliding, penetration, unfaithful poses) measurably hurt the downstream tracking policy. On the G1 it
  is more faithful than PHC and ProtoMotions retargets.
- **OmniRetarget** ([Yang et al. 2025, arXiv:2509.26633](https://arxiv.org/abs/2509.26633)) keeps an
  interaction mesh between body, terrain and objects, and minimises its Laplacian deformation under hard
  constraints: joint limits, no penetration, foot contacts. It targets object and scene interaction,
  which we don't need yet.

## Before / after

Source: `python experiments/retarget_compare.py --render squat_jump`, run on the seven clips'
`human.npy` (`outputs/remote_latent_v2/clips`). Contacts come from the human (`stance_mask`) and are the
same for both methods.

- **Track:** mean IK error against each method's *own* targets (pelvis / feet / knees / elbows / hands, cm).
- **Limit:** frames with any hinge within 1° of its range.
- **Slide:** stance-foot path length per second in stance.
- **Drift:** mean max displacement within one contact.
- **Pen.:** deepest sole sphere below the floor.
- **Float:** mean sole height while in stance.
- **Angle errors:** robot vs human flexion (knee, hip = trunk–thigh, elbow, shoulder = trunk–upper arm),
  degrees, averaged over both sides.

**Mean over the 7 clips:**

| method | track (cm) | limit | slide (cm/s) | drift (cm) | pen. (cm) | float (cm) | knee | hip | elbow | shoulder |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 0.7 / 1.4 / 4.9 / 4.9 / 2.7 | 40% | 29.1 | 15.7 | 0.9 | 5.2 | 11.9° | 10.7° | 13.4° | 8.9° |
| **proportional** | 0.5 / 0.1 / 2.9 / 0.5 / 0.1 | **12%** | **0.4** | **0.2** | **0.3** | **0.1** | **3.0°** | **4.4°** | **1.3°** | **2.3°** |

**Per clip:**

| clip | method | track (cm) | limit | slide | drift | pen. | float | knee | hip | elbow | shoulder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| squat_jump | uniform | 0.5 / 2.4 / 4.4 / 5.7 / 3.5 | 17% | 32.5 | 19.3 | 2.0 | 6.8 | 7.4 | 9.9 | 14.8 | 8.8 |
| squat_jump | proportional | 0.4 / 0.0 / 3.3 / 0.5 / 0.1 | 5% | 0.1 | 0.1 | 0.3 | 0.0 | 2.3 | 4.9 | 0.9 | 3.8 |
| squat | uniform | 0.6 / 1.9 / 4.2 / 5.5 / 2.6 | 9% | 19.6 | 16.3 | 0.7 | 6.1 | 7.4 | 9.3 | 15.9 | 9.9 |
| squat | proportional | 0.4 / 0.1 / 3.0 / 0.5 / 0.1 | 5% | 0.1 | 0.1 | 0.1 | 0.0 | 4.2 | 4.2 | 1.0 | 2.7 |
| lunge | uniform | 0.9 / 0.7 / 7.4 / 4.4 / 2.1 | 49% | 31.4 | 16.7 | 1.3 | 6.0 | 20.7 | 20.9 | 20.9 | 7.7 |
| lunge | proportional | 0.4 / 0.1 / 2.9 / 0.3 / 0.1 | 36% | 0.2 | 0.1 | 0.9 | 0.1 | 3.0 | 4.9 | 1.1 | 1.8 |
| front_kick | uniform | 1.8 / 2.2 / 7.5 / 5.0 / 1.9 | 85% | 42.4 | 11.5 | 0.2 | 3.9 | 13.2 | 10.5 | 12.1 | 8.9 |
| front_kick | proportional | 1.4 / 0.3 / 4.0 / 0.5 / 0.1 | 26% | 1.9 | 0.8 | 0.1 | 0.1 | 4.0 | 6.2 | 1.3 | 2.2 |
| jumping_jacks | uniform | 0.3 / 0.5 / 3.1 / 5.4 / 2.8 | 36% | 36.0 | 9.2 | 1.3 | 5.9 | 13.1 | 9.1 | 13.1 | 9.4 |
| jumping_jacks | proportional | 0.1 / 0.0 / 1.7 / 0.8 / 0.1 | 13% | 0.2 | 0.0 | 0.1 | 0.0 | 1.7 | 3.0 | 1.8 | 1.8 |
| punches | uniform | 0.4 / 0.1 / 4.2 / 4.3 / 1.4 | 80% | 29.2 | 25.4 | 0.9 | 3.2 | 12.7 | 5.4 | 9.7 | 12.5 |
| punches | proportional | 0.8 / 0.1 / 3.5 / 0.3 / 0.1 | 0% | 0.4 | 0.5 | 0.3 | 0.1 | 5.2 | 5.4 | 0.8 | 1.6 |
| arms | uniform | 0.6 / 1.8 / 3.5 / 3.7 / 4.4 | 0% | 12.8 | 11.6 | 0.0 | 4.3 | 9.0 | 9.5 | 7.0 | 5.3 |
| arms | proportional | 0.1 / 0.0 / 1.6 / 0.7 / 0.1 | 0% | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 | 2.1 | 2.1 | 2.5 |

On every clip, the proportional method is better or tied on every contact, limit and angle metric
(the hip angle on `punches` ties at 5.4°). The tracking columns aren't comparable across methods,
because each is measured against its own targets. The uniform path reproduces the existing `remote_latent_v2` `qpos.npy` to 1e-11.

**Notes:**

- **Remaining limit hits:**
  - `lunge` (36%): ankle pitch and roll in the deep lunge.
  - `front_kick` (26%): the kicking hip's yaw and roll. The human's kick leaves the G1's hip range.
- **Not measured:** V-JEPA alignment of the new retarget. It needs the GPU; the old retarget scored 0.726.
- **Tuning:** parameters were tuned on these same seven clips (pelvis orientation weight 1.5, pelvis
  lean 0.5, arm weight ×5, ankle margin 5°). There is no held-out clip.
- **Arm smoothness:** at arm weight ×5, 95th-percentile arm joint speed stays below the uniform
  retarget's on every clip checked. Jumping jacks still show shoulder-yaw flips with overhead arms,
  under both methods.

**Render:** `outputs/retarget_compare/squat_jump/compare.mp4` shows athlete | uniform | proportional,
with the robots seen from the video camera's side. New per-clip `qpos_proportional.npy` and
`motion.csv` files are next to it.
