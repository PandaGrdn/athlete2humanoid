# Training environments: one zip per video that anyone with mjlab can train on (2026-11-01)

**Question:** a run page showed a video of our robot. What does someone who wants to *use* this take away?

**Short answer:** every stored run and every live upload now hands back a training environment: one file,
`<id>.a2h-env.zip`, 1.3–3.1 MB, holding the person's motion as a G1 reference, the action as a key-moment spec, every
training and held-out variant with the physics filter's verdict, the reward config, the source credit and a loader.
With mjlab installed and nothing else of this repository, `python -m a2h_env train <zip>` trains a tracking policy
on it. Smoke-tested on CPU: 20 PPO iterations on the front kick's reference and on its 36 variants end to end, from
a clean folder with only the zip and mjlab 1.6.0. Not tested: a full GPU training from a bundle (the stored policies
were trained by `train_motion.py` on the same arrays, through the same mjlab task).

## Where to get one

| Where | What | How |
|---|---|---|
| Run page (`/runs/<id>`) | "Download training environment", with the size and one line on what is inside | `GET /api/runs/<id>/env.zip` (`server/runs.py env_bundle`): the run folder's `env.zip` when it ships one, else built on demand from the run folder and cached under `A2H_DATA_DIR/env/`, rebuilt when the run's inputs change |
| Build view (`/build/<id>`), stage 05 "Variants" | the same button, in the stage's caption and under the variant grid | the run's `env` entry |
| Live page (`/live/<job>`) | the button once the variants stage has written the zip | `server/live_worker.py` exports the bundle right after the variants (stage 05's output); with no variants in time, it still holds the reference and the spec. `GET /api/jobs/<job>/env.zip`; it expires with the upload |
| Batch | `outputs/remote_<run>/env.zip` for each run, shipped by `scripts/build_space.py` | `python scripts/export_env.py outputs/remote_latent_s3_* outputs/remote_latent_ra_*` (the repo's venv: numpy, mujoco, scipy, torch) |

`a2h_env/export.py` builds every one of them. It is standard library only (numpy, MuJoCo and `skill/` are used when
importable: foot contacts, generating variants), so the API server needs nothing beyond Python to serve a bundle.

## What is inside

| File | What |
|---|---|
| `manifest.json` | ids (run, clip), versions (`RETARGET_VERSION`, the pipeline commit, the pose front end, format version), the spec's and the retarget's sha256, counts, every file's size, and `inside`: the one line the button shows |
| `README.md` | what each file is and how to train, written per bundle |
| `reference/qpos.npy` | the retargeted G1, (T, 36) at the clip's fps: root xyz, root quaternion (w, x, y, z), the 29 joints in the G1's order |
| `reference/meta.json` | the clip's `meta.json`: fps, front end, retarget version and parameters, `in_place` as a flag with its details (the lakeside jog) |
| `reference/contacts.npy` | (T, 2) bool, left and right foot on the floor by the spec's own rule (ankle within 5 cm of its standing height), when the exporter had MuJoCo |
| `task_spec.json` | `skill/keypose.py`'s spec: the key moments with Muse's `muse_label`s, flights, settles, goals, tolerances, relevance, the reward machine's weights, and the retarget's self-check |
| `importance.json` | goal relevance per reference frame, when the run had it |
| `reward_config.json` | the tracking task and its mjlab reward terms (recorded, so the bundle says what it pays without mjlab), the policy it came from and its iterations, and `feet_height` when the run used it (the jog: weight 2.0) |
| `variants/train/` | `index.json`: the 36 TRAIN variants (mirror × tempo × amplitude × warp) kept or dropped and why; each kept one's `qpos.npy`, its own `task_spec.json` and `src_frame.npy` |
| `variants/held_out/` | the same for the 7 HELD_OUT variants (tempos and amplitudes outside the training range) |
| `credit.json` | the source video's attribution and licence (`web.json`), when it has one |
| `extras/silhouette.mp4`, `extras/video_latent.npz` | SAM 3.1's silhouette and the V-JEPA 2 latent, each only under 2 MB |
| `robot/` | which G1 the environment uses (mjlab's own asset) and the Menagerie licence; the meshes are not copied, mjlab installs them |
| `a2h_env/` | the loader (below), vendored so the zip works on its own |

Where the variants come from, per run (`manifest.variants.source`): the sibling `outputs/remote_variants_<tag>/` run
when the run's policies trained on variants (kick, squat jump, jog); otherwise `scripts/export_env.py` generates both
grids from the retarget with `skill/variations.py` (squat, jumping jacks, lunge, punches), which is the same generator
but not what those runs' stored policies trained on (they tracked the clip alone). The API's on-demand build never
generates: it takes the sibling run, or the verdicts from `variants.json` with no arrays.

## The loader

```bash
uv venv -p 3.12 mjlab_env && uv pip install -p mjlab_env/bin/python mjlab==1.6.0 tensorboard
unzip latent_s3_kick.a2h-env.zip -d latent_s3_kick && cd latent_s3_kick
python -m a2h_env info  ../latent_s3_kick.a2h-env.zip
python -m a2h_env train ../latent_s3_kick.a2h-env.zip --iters 1500 --num-envs 4096   # --variant train (default) | reference | <name>
```

```python
import a2h_env
b = a2h_env.load_bundle("latent_s3_kick.a2h-env.zip")   # numpy only: b.qpos, b.fps, b.contacts, b.spec, b.train, b.held_out, b.reward_config
env = a2h_env.make_env(b, variant="train", num_envs=4096)   # mjlab ManagerBasedRlEnv
```

- `load_bundle` reads the zip (or its unzipped folder) with numpy alone.
- `make_env` builds mjlab's G1 flat tracking environment the way `train_motion.py` and `experiments/variant_policy.py`
  do: the qpos becomes mjlab's 50 Hz motion file by forward kinematics on mjlab's G1 (`mjlab.scripts.csv_to_npz`'s
  steps, in process, no Weights & Biases; `tests/test_training_env.py` checks the interpolation against mjlab's
  `MotionLoader`), cached in `<bundle>.motions/`. `variant="train"` lays every kept training variant end to end and keeps
  each env inside its own (`VariantMotionCommand`, as `mjlab_tasks.py` `A2H_VARIANTS`); `feet_height` is applied when
  the bundle's reward config sets it (`mjlab_tasks.py FEET_HEIGHT`).
- `train` runs mjlab's `MotionTrackingOnPolicyRunner` (rsl_rl PPO, mjlab's G1 tracking PPO config) and writes
  `logs/<timestamp>/model_<it>.pt` and `weights.json`.
- Dependencies: numpy to read; mjlab 1.6.0 (torch, mujoco, mujoco-warp, rsl-rl-lib) and tensorboard to train. A GPU for
  a real run; CPU works for a smoke test.
- Not in the bundle: the key-moment task reward of the task policies (`skill/task.py`, `skill/keypose.py`), which needs
  the repository. The spec is there, so another reward can be built on it.

## Bundle per stored run

Exported on 2026-11-01 at commit b8186830df9b, each loaded back with `python -m a2h_env info`:

| Run | Clip | Reference | Key moments | Training variants | Held out | Variants from | feet_height | Zip |
|---|---|---|---|---|---|---|---|---|
| `latent_s3_kick` | front kick | 167 frames, 5.6 s | 4, named by Muse | 36 of 36 (209 s) | 7 of 7 | `remote_variants_s3_kick` | no | 1.75 MB |
| `latent_s3_sj` | squat jump | 271 frames, 9.0 s | 7 | 36 of 36 (336 s) | 7 of 7 | `remote_variants_s3_sj` | no | 2.76 MB |
| `latent_s3_squat` | squat | 306 frames, 10.2 s | 4 | 36 of 36 (383 s) | 7 of 7 | generated | no | 3.03 MB |
| `latent_s3_jacks` | jumping jacks | 159 frames, 5.3 s | 11 | 20 of 36 (124 s); 16 dropped: speed | 3 of 7 | generated | no | 1.28 MB |
| `latent_s3_lunge` | lunge | 297 frames, 9.9 s | 13 | 36 of 36 (371 s) | 7 of 7 | generated | no | 3.08 MB |
| `latent_s3_punch` | punches | 168 frames, 5.6 s | 11 | 36 of 36 (210 s) | 7 of 7 | generated | no | 1.78 MB |
| `latent_ra_jog` | lakeside jog, in place | 180 frames, 6.0 s | 13 | 36 of 36 (225 s) | 7 of 7 | `remote_variants_ra_jog` | weight 2.0 | 1.96 MB |

The jumping-jacks drops are the joint-speed cap on its stored retarget (docs/PIPELINE_ACCURACY.md, fix 4).

## Smoke test: the bundle trains

On a MacBook (Apple silicon, CPU, `--device cpu`), in a clean folder holding only `latent_s3_kick.a2h-env.zip`
unzipped and a fresh virtualenv with `mjlab==1.6.0` (nothing of this repository on the path; environment cleared with
`env -i`):

| Command | Motion file | PPO | Mean reward (iteration 0 → 19) | Mean episode length (steps) | Wall |
|---|---|---|---|---|---|
| `python -m a2h_env train ../latent_s3_kick.a2h-env.zip --iters 20 --num-envs 8 --device cpu --variant reference` | 1 clip, 50 Hz | 20 iterations × 8 envs × 24 steps | −2.10 → −2.78 | 11.1 → 19.0 | 16 s (9.6 s of PPO) |
| the same with `--variant train` | 36 variants end to end, 10,384 steps (208 s) | 20 × 8 × 24 | −1.82 → −2.69 | 11.5 → 18.6 | 35 s (9.6 s of PPO) |

Both wrote `model_0.pt` and `model_19.pt` and a `weights.json`; `info` printed the bundle's manifest. Twenty iterations
on eight envs say nothing about learning (the rewards above are noise around an untrained policy); they say the zip,
the loader and mjlab's trainer fit together with no server and no repository. The stored policies were trained by
`train_motion.py` on these same arrays through the same mjlab task, 1,000–1,500 iterations on one GPU; a full training
from a bundle was not run.

## Tests

- `tests/test_training_env.py`: export a run folder (with a sibling variants run) → load the zip and its unzipped folder →
  arrays equal to the run folder's; the manifest's fields; verdicts-only and no-spec cases; the live export (contacts from
  the spec's rule); the `.npy` header parser; the `info` command; the vendored loader importing with nothing of the
  repository on the path; and, with mjlab installed, the loader's interpolation against mjlab's `MotionLoader`.
- `server/tests/test_env_bundle.py`: `GET /api/runs/{id}/env.zip` (404 without a spec, content type, attachment name,
  the cache and its rebuild, a shipped `env.zip`, unknown and traversing ids), the run detail's `env` entry, and a live
  job's `env.zip` through the fake worker.
