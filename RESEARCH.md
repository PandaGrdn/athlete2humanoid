# Prior art and positioning: "skill, not clip"

Scan done 2026-10-26. Items marked *(unverified)* were cited from memory and still need checking.

## The method
Muse watches one athlete video and writes a **skill program**: the phases of the move plus physics invariants the simulator can check. For a squat jump these are countermovement depth, near-simultaneous hip/knee/ankle extension, the arm swing leading takeoff, takeoff velocity, and landing knee flexion. The policy trains on that program's reward plus a *loose* style term, and is evaluated on variations the clip never showed. Muse then watches rollout videos and revises the program.

## Safe novelty claim
> LLM-authored, verifiable biomechanical invariants extracted from a single video, with training aimed at the skill rather than the clip, and evaluated on variations the clip never showed.

Do **not** claim "first LLM reward from video" (Video2Reward) or "first VLM rollout critique" (RDA, GROVE).

## Closest work
| Work | What it does | How we differ |
|---|---|---|
| [TRACC](https://arxiv.org/abs/2609.36924) (Sept 2026) | G1 humanoid. GPT-5 infers intent from a failed video and proposes task rewards; tracking weight decays to zero at takeoff | No phase-structured invariant program; no revision loop |
| [Video2Reward](https://arxiv.org/abs/2412.05515) (ECAI'24) | Video → keypoints → LLM reward code, refined by a VLM | Locomotion gaits; no style term; no evaluation on variations |
| [GROVE](https://openaccess.thecvf.com/content/CVPR2025/html/Cui_GROVE_A_Generalized_Reward_for_Learning_Open-Vocabulary_Physical_Skill_CVPR_2025_paper.html) (CVPR'25) | LLM-written physical constraints refined with VLM feedback | Starts from text, not a video demo |
| [TaskNPoint](https://arxiv.org/abs/2606.26215) (June 2026) | G1, one video per skill, generalizes to unseen goals | No LLM; key interaction window labelled by hand |
| [RDA](https://arxiv.org/abs/2606.01672) (RLC'26) | VLM watches rollouts and rewrites reward code | No demo video |
| [MVR](https://arxiv.org/abs/2603.01694) | VLM reward alone fails on humanoids; works added to a task reward | Supports our fundamentals + style split |

Background: [Eureka](https://arxiv.org/abs/2310.12931), [DrEureka](https://arxiv.org/abs/2406.01967), [Text2Reward](https://arxiv.org/abs/2309.11489), [VLM-RM](https://arxiv.org/abs/2310.12921), [RoboCLIP](https://arxiv.org/abs/2310.07899), [DeepMimic](https://arxiv.org/abs/1804.02717), [AMP](https://arxiv.org/abs/2104.02180), [ASE](https://arxiv.org/abs/2205.01906), [MaskedMimic](https://arxiv.org/abs/2409.14393), [SFV](https://arxiv.org/abs/1810.03599), [VideoMimic](https://arxiv.org/abs/2505.03729), [mjlab](https://arxiv.org/abs/2601.22074); BeyondMimic, ASAP, GMT, KungfuBot, HumanPlus *(unverified IDs)*.

## Build decisions
- **Style term:** apply tracking only at Muse-chosen keyframes (bottom of the countermovement, takeoff, landing) and fade it out after takeoff, as TRACC does. An AMP discriminator would mean changing the PPO loss, which is too risky for this timeline.
- **mjlab hooks:** reward terms are `f(env, ...) -> (num_envs,)` tensors in the task's `rewards` dict. The tracking task ends an attempt when the torso or end effectors drift 25 cm vertically from the reference, which is what killed the bedroom clip; a skill task relaxes these checks. To register a custom task without forking mjlab, a launcher imports our task module and then calls `mjlab.scripts.train.main()` (single-GPU training runs in-process).
- **V-JEPA 2** ([paper](https://arxiv.org/abs/2506.09985), [code](https://github.com/facebookresearch/vjepa2)): use it only as an evaluation metric (style similarity between rollout and clip). It's too slow for a per-step reward, and the action-conditioned 2-AC variant is trained on robot arms. *(Weights license and inference cost unverified.)*

## Evaluation plan
1. **Unseen targets:** jump-height and box-height sweeps beyond the clip; report success rate.
2. **Push recovery:** the largest impulse survived in each phase.
3. **Two-axis plot:** skill success and invariant pass rate against frame-matching error (MPJPE). Ablations: tracking only, skill only, both, and no Muse revision.
