# Objects: from a ball in the video to a ball in the G1's simulation (plan, 2026-10-29)

**Question:** can athlete2humanoid see the object an athlete uses (a basketball, say), put a matching sim-ready object
in the robot's scene, and train the G1 to do the skill with it, e.g. dribbling?

**Short answer:** yes, in stages. Detection is the easy part: the SAM 3.1 call we already make takes any noun phrase. The
hard part is the RL: contact-rich object skills take an object reward, a contact reward and a curriculum on top of
tracking. For balls we don't need SimReady at all: a MuJoCo sphere with the regulation radius and mass is more exact
and much faster than any mesh. SimReady is useful later, for objects that aren't primitives (a kettlebell, a bat, a box).

Nothing was downloaded and no GPU job was run for this plan. Claims marked *(unverified)* come from search snippets or
third-party summaries, not the primary source.

## 1. SimReady assets

| | |
|---|---|
| What | NVIDIA's spec for simulation-ready OpenUSD assets: geometry, physics, semantic labels and packaging, each checked by validators. Specs: [NVIDIA/simready-foundation](https://github.com/NVIDIA/simready-foundation) (Apache-2.0). |
| "Sim Ready Central" | An **index**, not one dataset, launched 2026-10-21 *(date from press coverage)*. Each company hosts its own HF dataset (pattern `<org>/simready-assets`); NVIDIA's index is [`nvidia/simready-central-library`](https://huggingface.co/datasets/nvidia/simready-central-library) (Apache-2.0 card, gated: accept conditions), browsed in the Space [`nvidia/simready-central`](https://huggingface.co/spaces/nvidia/simready-central). Docs: [SimReady Central Library](https://docs.omniverse.nvidia.com/simready/latest/simready-central.html). [`nvidia/simready-catalog`](https://huggingface.co/datasets/nvidia/simready-catalog) also exists (401 to an anonymous fetch). |
| Search | Natural language, via [`simready-search`](https://pypi.org/project/simready-search/) (`AssetLibrary.add_service_source(...)`, `SearchFilterClass`, `SearchFilterLLMQuery`) or the endpoint `https://nvidia-simready-central.hf.space`. Returns `asset_path` plus a relevance score. |
| Licence | **Per package**, declared in the dataset card and the package metadata. Any licence is allowed, non-commercial included; NVIDIA recommends CC-BY-4.0. Check each asset. |
| Format | OpenUSD packages with their materials and textures. Physics through `UsdPhysics` (rigid body, mass, colliders, physics materials). The `Robotics-Prop` profile and the rigid-body feature (FET003) set the requirements; the exact fields weren't readable from the overview pages. |
| Semantic labels | `Open-Taxonomy-*` profiles tie labels to closed vocabularies (COCO, ADE20K, ImageNet-1K, …). COCO has `sports ball`, so a detection's class can be matched to an asset's label. |
| Older sibling | [`nvidia/PhysicalAI-SimReady-Warehouse-01`](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01): ~753 warehouse props with USDPhysics colliders and a CSV catalog giving name, category, tags and **mass** *(count and CC-BY-4.0 licence unverified)*. No sports balls seen. |

**Is there a basketball?** Not confirmed: the Space didn't render for a text fetch. Search it (`simready-search`, query
"basketball") before relying on it. For balls it doesn't matter (see the asset table in §2).

### Getting a USD asset into MuJoCo / mjlab

1. **MuJoCo's own USD import** ([docs](https://mujoco.readthedocs.io/en/stable/OpenUSD/importing.html)) is
   experimental and needs a MuJoCo built with USD. We run mujoco 3.11.0 from pip, and the docs imply the pip wheel
   doesn't include it. Not worth a custom build.
2. **[usd2mjcf](https://pypi.org/project/usd2mjcf/)** (LightwheelAI, v0.0.1): too early to rely on *(unverified
   coverage of UsdPhysics)*.
3. **Recommended: a small converter of our own (~1 day).**
   - Read the asset with `usd-core` (`pxr`): mesh points and faces, `UsdPhysics.MassAPI` (mass, centre of mass,
     inertia), `UsdPhysics.MaterialAPI` (static and dynamic friction, restitution), and the collider approximation
     (`MeshCollisionAPI`: convexHull, convexDecomposition, …).
   - Visual mesh: decimate to ~5k faces (trimesh / Open3D quadric decimation), export OBJ.
   - Collision: if the USD already has convex parts, keep them. If not, run [CoACD](https://github.com/SarahWeiii/CoACD)
     (what `obj2mjcf` uses), max ~8 hulls.
   - Build an `MjSpec` body with a free joint. MuJoCo Warp collides convex meshes, but primitives are much cheaper per
     step, so fit a box, capsule or sphere wherever one fits within ~1 cm.
   - Map the physics material: friction straight across; restitution through `solref`. MuJoCo's contacts are soft,
     so a bouncing ball needs a stiffer, less damped `solref` (e.g. negative `solref` = direct stiffness/damping),
     tuned until a dropped ball returns 0.75–0.85 of its height (a regulation basketball's bounce).

## 2. Detection

### The same SAM 3.1 call, another prompt
`track.track(video, prompt=...)` already sends a free-text concept prompt to SAM 3.1 and returns every matching object
per frame (`out_obj_ids`, `out_probs`, `out_boxes_xywh`, `out_binary_masks`). SAM 3 is built for short noun phrases
("basketball", "soccer ball", "kettlebell"), so a second call with the object's name works as is.

Changes needed:
- `sam_track.py --prompt basketball`, keeping **all** tracks, not just `main_track_masks`' single best. A dribble can
  be lost under the hand or blurred; fill short gaps from the physics fit below.
- Small, fast balls are the known weak spot. The [SAM 2 ball-tracking write-up](https://alpha.sievedata.com/blog/ball-tracking-with-sam2)
  found detectors miss blurred balls. Expect gaps at the fastest frames.
- **Which object?** Run the prompt fallback the athlete prompt already uses: the user's class name if given, else a
  short fixed list (`sports ball`, `basketball`, `soccer ball`, …), and keep the class SAM tracks with the most
  confidence near the hands or feet. A VLM labelling the first frame (Llama, Meta's own) is an optional upgrade.

### 3D trajectory from one camera
A ball's size is known, and the person gives a metric scale.
1. **Scale at the athlete's depth.** MediaPipe world landmarks are metric. Their pixel lengths in the crop (torso,
   thigh) give metres-per-pixel `s` at the athlete's depth `Z_p`.
2. **Ball depth from its apparent size.** Expected pixel diameter at depth `Z_p` is `D / s`, where `D` is the class
   diameter (basketball 0.24 m). Measured diameter `d` from the mask's fitted circle gives `Z_b = Z_p · (D/s) / d`.
   Lateral position follows from the pixel offset, scaled by `Z_b / Z_p`.
3. **Physics fit.**
   - Split the track at contacts (ball mask touching a hand or foot landmark, or the floor).
   - Fit each free flight as a parabola with **g fixed at 9.81 m/s²**. This is the gravity constraint retarget.py already
     applies to the athlete's flights (commit 42f3d64).
   - The fitted vertical acceleration in pixels is also an independent check of `s`.
   - The size-based depth is noisy, so let the parabola smooth it.
4. **Put it in the robot's frame.** Use the same root transform and leg-length scale the retarget applies to the human.
   Don't scale the ball. Do scale its path relative to the hands, so it stays in the robot's reach.

Output: `object.npz` with per-frame `pos (T,3)`, `vel (T,3)`, `visible (T,)`, `contacts (T, k)` (which body part
touches it), and `class`. The keypose spec gains object events: hand–ball contacts, releases, bounces, apex.

### Class → asset
A table in code, not a search, for the classes we support. Search only for the long tail:

| Class | Asset | Physics |
|---|---|---|
| basketball | MJCF sphere r = 0.119 m | 0.62 kg, hollow-shell inertia, bounce 0.75–0.85 |
| soccer ball | sphere r = 0.11 m | 0.43 kg |
| volleyball | sphere r = 0.105 m | 0.27 kg |
| tennis ball | sphere r = 0.033 m | 0.058 kg |
| anything else | `simready-search` (class + `SearchFilterLLMQuery`), checked for licence and the `Robotics-Prop` profile, then the converter above | from `UsdPhysics` |

## 3. Training

### How it's done today

| Work | What it shows | What we'd take |
|---|---|---|
| [SkillMimic](https://arxiv.org/abs/2408.15270) (CVPR 2025) | Simulated humanoid basketball (pickup, dribble, layup, shot) learned from mocap with **one** reward setup for every skill. Reported success: dribble-forward ~80%, layup ~99% *(figures via a third-party review)*. | The **contact-graph reward**: match which body parts touch the ball, frame by frame. Without it, policies use the wrong body parts. Body and object rewards are **multiplied**, so neither can be ignored. |
| [HDMI](https://arxiv.org/abs/2509.16757) (CMU, 2025) | Unitree **G1** object interaction (doors, boxes, suitcase) from **monocular human video**, run on the real robot. | Object pose in the robot's root frame; reference contact points in the object's frame. Interaction reward: hand-to-contact-point distance × capped contact force, gated by the reference contact flag. Residual actions around the reference pose. |
| [Learning to Ball](https://arxiv.org/abs/2509.22442) (2025) | Long basketball sequences by composing policies; adversarial (discriminator) rewards plus rule penalties: ball contact only with the hands, no travelling. | Dribbling as its own policy, composed with other skills later. |
| [Dribble Master](https://arxiv.org/abs/2505.12679) (2025) | Humanoid **soccer** dribbling with a two-stage curriculum, run on hardware. | The curriculum: walking first, then the ball. |
| [PAiD](https://arxiv.org/abs/2602.05310) (2026), [code](https://soccer-humanoid.github.io) | **G1 kicks**: motion tracking, then ball perception, then sim-to-real; 91.3% kick success. | Closest to our first stage: a tracked kick with a ball added. |

### Adding an object to our mjlab tasks (`mjlab_tasks.py`)
mjlab 1.6 already does objects. Its lift-cube task adds a free-joint body as a scene entity: `EntityCfg(spec_fn=...)`
in `mjlab/tasks/manipulation/config/yam/env_cfgs.py`. It also has object observations and contact sensors
(`ContactSensorCfg`). A new `A2H-Object-Flat-Unitree-G1` family would add:

- **Scene:** `cfg.scene.entities["ball"] = EntityCfg(spec_fn=ball_spec)`, plus a `ContactSensorCfg` between the ball and
  `left/right_rubber_hand`, the feet and the floor.
  - The mjlab G1 has **fingerless rubber hands**. That suits dribbling and kicking, since the palm pushes. Holding takes
    a two-hand squeeze.
- **Reference:** the motion npz gains the object columns. `VariantMotionCommand` resets the ball to the reference
  object state at the sampled start time (reference-state init, as tracking does for the body).
  - `skill/variations.py` must warp the object path with the body's. Time warps are free; spatial variants keep the ball
    at the same place relative to the hand.
- **Observations:**
  - ball position and velocity in the anchor frame;
  - the reference ball position at the `FUTURE_S` offsets (as `future_reference` does);
  - the reference contact flags for the next key moment.
- **Rewards:**
  - `object_pos`: `exp(-‖p_ball − p_ref‖² / σ²)`, with σ around 0.15 m at first, shrinking later.
  - `contact_graph`: SkillMimic's reward, from the contact sensor against the reference contacts.
  - `interaction`: HDMI's, for holds.
  - Kicks get an outcome term instead: ball velocity after contact against the reference.
  - Multiply into the tracking reward (body × object), and keep `TaskReward`'s stage machine, with object events as
    stages ("contact at t_k", "release", "bounce").
- **Terminations:** ball more than ~0.5 m from its reference while a contact is expected (like `anchor_pos`), and ball
  touching the floor during a hold.
- **Robustness eval:** ball mass ±20%, radius ±5%, restitution and friction ranges, and pushes on the ball, added to
  `robustness_eval.py`.

### Difficulty and compute
- **Our baseline:** a tracking run is ~1,500 iterations, about 1 h on a Modal L40S (~$2), per `docs/MODAL.md`.
- **Object simulation cost:** one sphere adds little per step.
- **Learning cost:** object skills usually need several times more samples. The policy must discover contact timing, and
  a ball fumbled once ends the episode.
- **Main risk:** a ball path from monocular video won't be physically consistent with the G1's (smaller) body. The
  object reward must be loose, so the policy can satisfy the physics, not the video.

## 4. Staged plan

| Stage | Demo | What's new | Effort | Compute |
|---|---|---|---|---|
| **0. See the ball** | A clip with a ball, the ball's mask, and its 3D path drawn next to the athlete's skeleton | `sam_track.py` multi-prompt; `object.npz` lifter and parabola fit; overlay render. **Needs a clip with a ball: none of ours has one; downloading one needs your OK.** | 2 days | one SAM pass |
| **1. Kick a ball** | `front_kick` sends a ball forward | A ball placed at the kicking foot's apex position, at rest. Reward: foot–ball contact at the kick's key moment, plus ball speed and direction after it. No ball path from video needed. | 2 days | 1–2 runs, ~1–2 h |
| **2. Hold / carry** | The robot holds a ball through a squat or walk | Reference ball pinned between the hands; `interaction` + `object_pos` rewards; curriculum: a weld between hands and ball that fades to zero over training. | 3 days | ~3 runs |
| **3. One bounce** | Drop, bounce, catch (or one push-down and recover) | Contact restitution tuned; video ball path used for the first time; contact-graph reward. | 4–5 days | ~5 h |
| **4. Dribble** | Stand-in-place dribble, then walking dribble | A periodic ball reference (phase from the keypose clock); ball observation essential. Curriculum: dribble standing, then add root motion. Expect the most tuning here. | 1–2 weeks | 10–20 h |
| **5. Any object** | User names an object; the pipeline fetches a SimReady asset | Class → asset search, licence check, USD → MJCF converter | 3 days | — |

Stages 0 and 1 are independent and can run in parallel. Stage 1 alone is a good demo ("the kick now hits a ball")
and needs no new perception.
