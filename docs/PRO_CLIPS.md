# The best stack on professional athletes' footage (2026-10-29)

**Question:** does the stack that works best on our own clips also work on footage of elite athletes it has never seen? And does starting from the multi-clip generalist, with or without the video's V-JEPA latent, help on a new clip?

## The clips

There is little clean, openly licensed footage of famous athletes doing a single move. The candidates are listed in `clips/CANDIDATES.md`. Eight were downloaded from Wikimedia Commons and checked by eye:

| Clip | Verdict |
|---|---|
| Biron, standing long jump (CC0) | rejected: the jump is never shown |
| Biron, running long jump (CC0) | rejected: distant, crowded, the camera pans |
| Javelin, EuroGames 2023 | rejected: distant and panning |
| High jump, Sudamericanos 2019 | rejected: distant and crowded, with cuts |
| Sweeney, long jump (1900) | rejected: spliced out of order |
| Sweeney, high jump (1900) | rejected: every attempt is cut before the landing |
| **McCracken, shot put (1900)** | kept: one throw, 4.2 s |
| **McCracken, discus (1900)** | kept: wind-up, a 429° spin, and the release, 3.3 s |
| **Kraenzlein, 110 m hurdles (1900)** | kept: run-up and clearance, 2.9 s; ends mid-air |

The three kept clips are Étienne-Jules Marey's films from the 1900 Paris Olympics: public domain, a static side-on camera, and one athlete each.
- McCracken won Olympic silver in the shot put.
- Kraenzlein won four Olympic golds.
- The discus and hurdles athletes are unclothed in the originals. Show only the robot renders publicly.

Trimmed copies are in `clips_pro/`, which is gitignored.

## Setup

One unattended Modal job per run: `scripts/pro_clips.sh` (`remote_run.py pro`), then `remote_run.py multiclip` for the discus rerun.

**Processing:**
- SAM 3.1 is installed at job start, since the Modal image has none.
- `experiments/process_clips.py` runs SAM 3.1 → MediaPipe → mink IK, plus the V-JEPA silhouette latent.
- On all three films, SAM found the athlete in 100% of frames and MediaPipe found the pose in 100%.

**Three policies per clip,** each trained 1,500 iterations on the clip's automatic variants (`skill/variations.py`), from its own starting point:

| Policy | Starts from |
|---|---|
| `tracking_var` | the squat-jump tracker trained on variants: the single-clip stack that won on our own clips (`docs/GENERALIZATION.md`) |
| `multi` | the 5-clip generalist (`docs/VJEPA_CONDITIONING.md`) |
| `multi_vjc` | the 5-clip generalist with the causal V-JEPA input: the video so far only, no future leakage |

**Test:** the full robustness suite (`multiclip_policy.py --full-eval`), 64 rollouts per condition:
- pushes of 0.5–1.5 m/s;
- pushes timed to the stages;
- noisy and unusual start poses;
- friction and mass changes;
- 0.8x and 1.2x speed;
- 0.5x–2x tempo;
- the clip's held-out variants;
- the V-JEPA score against the film.

**Results:** `outputs/remote_multiclip_pro_discus` and `outputs/remote_pro_v1`.

## Discus: all three learn it, and the generalist start is the most robust

Task success, with the fall rate in brackets:

| Policy | The clip | Pushes | Stage-timed pushes | Start noise | Speed | Tempo 0.5–2x | Friction / mass | Held-out variants | Joint error (rad) |
|---|---|---|---|---|---|---|---|---|---|
| `tracking_var` | 86% (0%) | 55% (2%) | 77% (0%) | 57% (20%) | 84% (0%) | **43%** (19%) | 85% (2%) | 80% (0%) | **0.225** |
| `multi` | **100%** (0%) | **92%** (1%) | **100%** (0%) | 49% (18%) | **100%** (0%) | 32% (68%) | **99%** (1%) | **85%** (0%) | 0.237 |
| `multi_vjc` | **100%** (0%) | 85% (2%) | 98% (0%) | 51% (23%) | **100%** (0%) | 33% (67%) | 96% (4%) | 79% (4%) | 0.248 |

1. **The pipeline turns a 1900 Olympic film into a working policy.** The source is grainy, flickering and 25 fps. The policy completes the throw from the clean start 86–100% of the time, with no falls.
2. **Starting from the 5-clip generalist is better than starting from the squat-jump specialist** on a new athlete's motion:
   - 100% vs 86% on the clip;
   - 92% vs 55% under pushes;
   - 85% vs 80% on unseen variants.
   - Pre-training on several clips leaves a more robust starting point, even for a motion it never saw.
3. **The V-JEPA input adds nothing on top of that here.**
   - `multi_vjc` roughly matches `multi`, a few points lower on pushes and variants.
   - On the five training clips, the causal latent helped most under tempo changes. On the discus, all three are weak there: 32–43% success, and the generalists fall 67% of the time.
4. **Caveat:** the discus spec has only one key moment ("turn left, step left"), so task success is a loose bar. The joint error to the reference (0.22–0.25 rad) is about twice that on our own clips (~0.11). The policies do the throw, but follow the film less closely than they follow a phone video.

## Shot put: the reference itself can't be tracked

All three policies fail: 0% success and 95–100% falls in every condition. Training never got going: mean episodes last about 25 steps (0.5 s) after 1,500 iterations, against about 360 for the discus. Episodes end on mjlab's hand/foot height check: 165 terminations per log window, against 4 for the discus.

The retarget has two defects that no policy can follow:

1. **The arm flips at a joint limit.**
   - As the throwing arm goes straight overhead at the release, the IK pins the right shoulder pitch at its lower limit (−3.09 rad), then jumps to its upper limit (+2.67 rad) in one frame. That is a 5.8 rad flip (172 rad/s); the reference hand teleports.
   - The left shoulder jumps 1–2 rad per frame twice during the glide.
   - The squat jump's fastest joint moves at 8 rad/s.
2. **The glide floats.**
   - For about 1.2 s around the throw, both feet are 4–9 cm off the floor while the body travels sideways, with no rise of the pelvis.
   - A real flight that long would need a jump of over a metre. The 1900 film was probably shot faster than it plays, so a hop of a fraction of a second becomes a slow float.

**Fixes for the pipeline, not yet built:**
- Keep the IK continuous: penalise joint velocity, or reject a solution that jumps across a joint's range.
- Check flights against gravity: airtime vs pelvis rise. That detects slowed footage and re-times it, or lets the spec drop the phantom flight.

**The discus works because it never leaves the ground,** so its playback speed doesn't matter.

## Hurdles: not trained

Every attempt in the film is cut while Kraenzlein is in the air. The reference can't land, so it can't complete its own task, and every variant is dropped. A clip has to contain a whole motion.

## A pipeline bug the discus found

The discus first failed the variant filter: its own reference "completed 1/2 key moments", so every variant was dropped.

- **Cause:** a key moment's turn target was set from the unwrapped heading (429° here), but the check measures the turn wrapped into ±180°. A spin past 180° asked for a turn that nothing could show.
- **Fix:** `skill/keypose.py` now wraps the target the same way (commit 3ab2cd2).
- **Effect:** the discus keeps 24 of its 36 training variants. Every earlier clip turns less than 180° between key moments, so their specs are unchanged.

## Cost

| Run | Time | Cost |
|---|---|---|
| Processing, shot put and hurdles (`pro_v1`) | about 1 h 20 min on an L40S | about $2.60 |
| Discus | about 1 h 20 min | about $2.60 |

## What it says about the stack

- **Video → SAM 3.1 → retarget → variant-trained tracker works on elite athletes' footage** when the retarget is physically sound (discus).
- **For a new clip, start from the multi-clip generalist,** not the single-clip specialist. Adding the causal V-JEPA input to that generalist didn't help further on this clip.
- **The weak link on archive footage is the retarget,** not the policy: joint-limit flips and slowed-down flights. Both can be detected automatically before any GPU time is spent.
