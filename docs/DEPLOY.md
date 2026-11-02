# Web app: local dev and deploy

The product UI is two pieces:

- `server/`: a FastAPI backend. It lists finished runs (`outputs/remote_*`), serves their
  artifacts, accepts uploads and runs jobs with progress over server-sent events.
- `web/`: a React + Vite + TypeScript frontend with a landing page and gallery, live job
  progress, and a results page.

Jobs go through one of four runners:

| Runner | What it does | Where |
|---|---|---|
| `demo` (default) | Replays a finished run: its real log lines and training curves, on a ~40 s clock. Results are that run's results. | Everywhere. |
| `live` | Builds a real RL task environment from the uploaded video: the pose (SAM 3.1 masks and SAM 3D Body on a Modal GPU with `A2H_LIVE_POSE=modal`, below; MediaPipe on the CPU otherwise, or when the GPU doesn't answer), mink retarget onto the G1, then the keypose task spec (`skill/keypose.py`) and the variants. It runs `python -m server.live_worker` as a sandboxed subprocess and stops before training (that needs a GPU for 1-2 h). | On whenever the pipeline imports (`A2H_LIVE=auto`). The Space image includes it. |
| `colab` | Copies the upload into `clips/` and runs `scripts/remote_run.py latent` on a Colab A100 (about 3 hours). It follows `runner.log`, `job.log` and `status.json`, and the finished run joins the gallery. | Local only. Needs `A2H_RUNNERS=demo,colab` **and** `A2H_ENABLE_COLAB=1`, plus a logged-in `colab` CLI. It spends compute units, so it is off by default. |
| `modal` | Trains a finished live build's environment on a Modal GPU, started by the visitor with "Train on a GPU" on the build's page (`POST /api/jobs/{id}/train`): the build's `env.zip` goes to the Volume `a2h-runs`, `scripts/modal_app.py`'s `run_job` runs `python -m a2h_env train ... --attempt-every 50` (mjlab's G1 tracking task on the reference and the variants, 1,000 PPO iterations, ~20-30 min on an L40S, about $1), and every 50 iterations the robot's attempt on the clip comes back to the page, which plays it on the G1 against the retarget while the GPU keeps training (`server/train.py`). The trained checkpoint is downloadable at the end. | Hosted too. Needs `A2H_RUNNERS=demo,modal`, the Modal token on the service (as for the GPU pose) and a gate: an invite code (`A2H_TRAIN_CODE`) and/or free trainings a day (`A2H_TRAIN_PER_DAY`). One training at a time by default. Setup below, "Live training on a GPU". |

The UI says so when a job is a demo replay, including for uploads.

**The live runner on the free Space.** A free CPU Space has 2 vCPU and 16 GB of RAM. A 6 s 1080p clip
takes ~9 s on an M-series Mac. In the Space image, emulated on that Mac (linux/amd64 under Docker), a 5.6 s
front-kick clip took 28 s end to end, 20 s of it in MediaPipe. The container peaked at ~1 GB of memory, server
included. On the Space's own x86 CPUs, expect roughly 15–40 s per clip. Limits: one build at a time
(`A2H_LIVE_CONCURRENCY`, default 1; later jobs wait in "queued"), and only the first ~20 s of a clip is used
(`A2H_LIVE_MAX_SECONDS`). Each build also has a wall-clock timeout and a memory cap (table below). While a build
runs, the API stays responsive: the worker runs at nice 10.

## Local quickstart

You need Python 3.10+ and Node 18+.

```bash
make install
```

```bash
make dev
```

Then open http://localhost:5173. `make dev` starts the API on :8000 (`uvicorn server.main:app --reload`) and the
Vite dev server on :5173, which proxies `/api` to the API. To run them in two terminals instead:

```bash
python3 -m uvicorn server.main:app --reload --port 8000
```

```bash
cd web && npm run dev
```

By default the API reads runs from `outputs/`. To serve another folder, set `A2H_RUNS_DIR` (for example a built
Space bundle: `A2H_RUNS_DIR=build/space/data/runs`).

Tests (API pytest, the existing unittest suite, and a frontend typecheck and build):

```bash
make test
```

### ONNX policies

Run checkpoints are in rsl_rl format. To add a `policy.onnx` next to each checkpoint (so the results page
offers ONNX downloads), run this with any Python that has torch and onnx. The mjlab venv has both:

```bash
python scripts/export_onnx.py outputs/remote_latent outputs/remote_latent_v2 --verify
```

`--verify` rebuilds the actor with rsl_rl's own `MLPModel` and checks that the exported graph gives identical
actions. Input: the 160-dim policy observation. Output: 29 joint actions.

## Settings (environment variables)

All settings are server-side. The client only ever gets `API_BASE`. The server checks every value at startup and
refuses to start with a list of the bad ones. Limits and their rationale: `docs/SECURITY.md`. Job lifecycle,
failure modes and health endpoints: `docs/SYSTEM_DESIGN.md`.

| Variable | Default | Meaning |
|---|---|---|
| `A2H_RUNS_DIR` | `outputs/` | Folder of run folders (any sub-folder with an `eval.json`) |
| `A2H_DATA_DIR` | `outputs/_web` | Writable scratch for uploads |
| `A2H_CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated origins allowed to call the API |
| `A2H_RUNNERS` / `A2H_ENABLE_COLAB` | `demo` / unset | See runners above |
| `A2H_DEMO_RUN` | newest run with a video | The run that uploads replay |
| `A2H_DEMO_SECONDS` | `40` | Length of a demo replay |
| `A2H_MAX_UPLOAD_MB` | `200` (`100` on the Space) | Upload limit, enforced on the request body as it streams. The extension, content type and file magic bytes are all checked. |
| `A2H_MAX_UPLOAD_STORAGE_MB` | `2000` | Total stored uploads before new ones get a 503 |
| `A2H_MAX_ACTIVE_JOBS` | `4` | Concurrent jobs. Further requests get a 429. |
| `A2H_MAX_JOBS_PER_IP` | `2` | Concurrent jobs per client |
| `A2H_RATE_JOBS` / `A2H_RATE_REQUESTS` | `30` per hour / `600` per minute | Per-client rate limits (429 + `Retry-After`) |
| `A2H_MAX_STREAMS` | `100` | Open live-progress streams; beyond it clients poll |
| `A2H_PROXY_HOPS` | `0` (`1` on the Space) | Trusted proxies in front of the API; the client IP is read that far from the right of `X-Forwarded-For` |
| `A2H_UPLOAD_TTL_H` / `A2H_JOB_TTL_H` | `1` / `24` | Hours an upload / a finished job is kept |
| `A2H_JOB_TIMEOUT_H` | `8` | A GPU-runner job still running after this is stopped |
| `A2H_LIVE` | `auto` | Live runner: `auto` turns it on when the pipeline imports, `1` forces it on, `0` off |
| `A2H_LIVE_MAX_SECONDS` | `20` | Only the first this-many seconds of an upload are used |
| `A2H_LIVE_CONCURRENCY` | `1` | Live builds at once; later jobs queue |
| `A2H_LIVE_TIMEOUT_S` / `A2H_LIVE_MEMORY_MB` | `240` / `8192` | Per-build wall-clock limit (CPU-time limit is twice that) and address-space cap (Linux) |
| `A2H_LIVE_PYTHON` | the server's Python | The interpreter the live worker runs in |
| `A2H_LIVE_POSE` | `cpu` | `modal`: the pose stage runs SAM 3.1 and SAM 3D Body on the deployed Modal GPU app (below); needs `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the server's environment, else it logs a warning and stays `cpu` (MediaPipe) |
| `A2H_HOSTED` | unset (`1` in the Space image; a Space's `SPACE_ID` implies it) | Public deploy: the GPU runner can't be enabled, the job list is off, CORS `*` is refused |
| `A2H_REMOTE_BACKEND` | `colab` | GPU backend for the local GPU runner: `colab` or `farmshare` (docs/FARMSHARE.md) |
| `A2H_TRAIN_CODE` | unset | Invite code a visitor types to train a live build on a Modal GPU (the `modal` runner). Put it in Secret Manager (`scripts/cloudrun_train.sh`), not in a plain variable. |
| `A2H_TRAIN_PER_DAY` | `0` | Trainings a day anyone may start without the code (`0`: code only). With both set, the code works once the day's allowance is used. |
| `A2H_TRAIN_ITERS` / `A2H_TRAIN_ATTEMPT_EVERY` | `1000` / `50` | PPO iterations per training, and iterations between the attempts the page plays |
| `A2H_TRAIN_NUM_ENVS` | `4096` | Parallel environments on the GPU |
| `A2H_TRAIN_CONCURRENCY` | `1` | Trainings at once (each one is a GPU: a second visitor gets a 429 with the minutes left) |
| `A2H_TRAIN_TIMEOUT_H` | `3` | A training still running after this is cancelled on Modal and the job fails |
| `A2H_TRAIN_BRANCH` | `main` | The repository branch the GPU job clones (`scripts/modal_app.py` `run_job`) |
| `A2H_STORE` | unset | Where finished builds and trainings are kept past a restart (`server/store.py`): unset = files under `A2H_DATA_DIR/persist`; `gcs://<bucket>[/<prefix>]` = a Cloud Storage bucket (`scripts/cloudrun_store.sh`) |
| `A2H_STORE_TTL_D` / `A2H_STORE_MAX_MB` | `30` / `2000` | The local store's caps: days a kept job stays, total size (the oldest go first). The bucket gets a 30-day lifecycle rule from the script instead. |
| `A2H_LIST_JOBS` | on locally, off hosted | `GET /api/jobs` (debugging) |
| `A2H_LOG_FORMAT` / `A2H_LOG_LEVEL` | `json` / `INFO` | Structured logs on stderr; `text` for humans |
| `A2H_RUN_EXCLUDE` | `*smoke*` | Glob patterns of run folders to hide |
| `A2H_STATIC_DIR` | unset | Serve a built `web/dist` from the API too (single-process hosting) |
| `API_BASE` (frontend, build time) | empty | The backend origin, e.g. `https://you-athlete2humanoid-api.hf.space`. Empty means same origin or the dev proxy. |

## Hosting choices

**Backend: your own machine through a Cloudflare Tunnel (planned: a Mac mini), or a Hugging Face Space.**
- **Hugging Face Space.** Since 2026, Docker Spaces need HF PRO ($9/month). The free tier now serves only static
  Spaces: `hf repos create` answers 402. The image and script are ready for when that's paid.
- **Your own machine.** `scripts/deploy.sh tunnel` runs the same hardened image in Docker (read-only, no
  capabilities, localhost only, restarted after reboots) and publishes it with `cloudflared`. A Mac mini runs
  the live pipeline fast and costs nothing monthly. The API is up only while the machine is on, awake and
  online, and training still needs a GPU (Colab).
- **Google Cloud Run (free tier).** `scripts/deploy.sh cloudrun [project]` pushes the same image to Artifact
  Registry and runs it as one Cloud Run service. Settings:
  - 2 vCPU, 4 GiB, min 0 and max 1 instance (jobs live in its memory);
  - CPU kept on between requests, so a job's thread runs;
  - uploads capped to 400 MB, since `/tmp` is memory there.

  The free tier (instance-based: 240k vCPU-s and 450k GiB-s a month) covers demo traffic: an instance stays
  up about 15 min after its last request, then scales to zero. Images over Artifact Registry's free 0.5 GB
  cost cents a month.
  - **Cold start, measured 2026-10-30:** after 25 minutes with no request from us, the first `GET /readyz` took
    23 s; the next took 0.15 s. Load `/readyz` once before a demo.
  - **Cloud Run needs a billing account linked to the project, even at $0:** a card on file, added by you in
    the console. A budget alert there is cheap insurance.
  - **Once:** `gcloud auth login`, a project (`gcloud projects create <id>`), billing linked
    (console.cloud.google.com/billing).
- **Free 512 MB tiers** (Render, Koyeb) are too small: the live pipeline peaks around 1-1.5 GB.

**Until the API is hosted:** `scripts/deploy.sh static` deploys the website alone, live at
https://athlete2humanoid.pages.dev. The header says "Offline demo". Without an API:
- **Examples:** work as usual.
- **`/live`:** replays recorded builds (`/live/demo/<clip>`) of five clips: front kick, squat jump, jumping jacks, lunge, punches.
- **`/build/<run>`:** a stored run's build replayed fast from its own files (about 28 s: the stages, then the policy's
  recorded attempts every ~100th iteration, then the trained policy). The pacing is the run's `web.json`
  `"replay": {"seconds": 28, "attempt_stride": 100}` (defaults when absent; `web/src/buildview/replay.ts`). A showcase
  tab with `"replay_build": "<run>"` in `web/public/showcase/index.json` opens on that replay (the dance does).
  A real-athlete run whose policy doesn't pass its spec ships only with `"ship": true` in its `web.json` (the dance:
  shown with its caveat).
  - Each is a real run of the live worker, recorded by `scripts/build_live_demo.py`: its stage messages at the times they arrived, pose landmarks, the G1 retarget, and the task environment with Muse's moment names.
  - Then a fast replay of PPO training (the real run's reward curve), and the policies trained on that clip, each scored under today's checker.
- **Your own video:** a dropped file stays in the browser. The page says live builds need the API and points at the recordings.

Re-record with `REBUILD=1 MUSE=1 PYTHON=<the repo's venv python> scripts/deploy.sh static` once newer policies exist (`CLIPS` in the script picks the runs).

**Frontend: Cloudflare Pages.** The free tier has unlimited bandwidth and requests and allows commercial use
(Vercel's Hobby plan doesn't). It serves the SPA fallback from `web/public/_redirects`, deploys with one
`wrangler` command, and makes adding a domain easy later. A domain bought through Cloudflare Registrar is sold at
cost and connects to the project in two clicks.

## Deploy

**On a Mac mini (or any always-on machine with Docker):**

```bash
brew install cloudflared
```

```bash
npx wrangler login
```

```bash
scripts/deploy.sh tunnel
```

- **Address.** Without a domain this gives a free quick tunnel, `https://<random>.trycloudflare.com`. It changes
  whenever cloudflared restarts, so after a reboot re-run the script; it redeploys the website with the new
  address.
- **A stable address.** Add a domain to your Cloudflare account (Cloudflare Registrar sells them at cost, about
  $10/year), then:
  - run `cloudflared tunnel login` once, picking that domain;
  - deploy with `TUNNEL_HOSTNAME=api.<your-domain> scripts/deploy.sh tunnel`;
  - keep the tunnel up across reboots with `sudo cloudflared service install`, pointed at `build/tunnel.yml`.
- **Image architecture.** The image's packages are locked for linux/amd64. On Apple silicon, Docker Desktop runs
  it under Rosetta (turn on "Use Rosetta for x86_64/amd64 emulation" in its settings). A live build then takes
  about 30 s instead of 10.

**On a Hugging Face Space (needs HF PRO):** log in once to each service, then run one script. You need Node 18+ (`brew install node`) and
[uv](https://docs.astral.sh/uv/).

```bash
hf auth login
```

```bash
npx wrangler login
```

```bash
scripts/deploy.sh
```

`hf auth login` needs a token with **write** access from https://huggingface.co/settings/tokens. If `hf` isn't
installed, run `uvx --from huggingface_hub hf auth login`; it stores the same login. `make deploy` runs the same
script. Re-run it to redeploy.

The script works out your Hugging Face username with `hf auth whoami`, or takes it as `scripts/deploy.sh <hf-user>`.
Then it:

1. Creates the Pages project `athlete2humanoid` if it doesn't exist yet, and reads back its `*.pages.dev` origin
   for the API's CORS list.
2. Builds the Space bundle (`build/space`, below). It refuses to upload if the bundle contains a `.env`, a key, a
   log, or anything shaped like a token.
3. Creates the Space `<hf-user>/athlete2humanoid-api` if needed and uploads the bundle.
4. Waits for `https://<hf-user>-athlete2humanoid-api.hf.space/readyz`. The first image build takes 5–10 minutes.
5. Builds the frontend with `API_BASE` set to the Space URL and deploys it to Pages.
6. Prints both URLs.

Options are set through the environment:

- `SPACE_NAME` and `PAGES_PROJECT` override the names.
- `CORS=https://your-domain.com` adds an origin.
- `ONNX_PY=<mjlab venv>/bin/python` ships ONNX policies. It needs a Python with torch and onnx.
- `SKIP_WEB=1` deploys the backend only.

The script never prints a token: `hf` and `wrangler` use their own stored logins.

### The live runner's pose on a GPU

By default a live build's pose stage is MediaPipe on the host's CPU, because SAM 3.1 and SAM 3D Body need CUDA and
gated weights. With `A2H_LIVE_POSE=modal` the worker instead sends the processed clip (H.264, 30 fps, longest side
640, at most `A2H_LIVE_MAX_SECONDS`) to the deployed Modal app `a2h-live-pose` (`frontends/live_pose_modal.py`:
SAM 3.1 masks, then SAM 3D Body on the mask boxes, on an L40S; the weights stay on a Modal Volume so a cold start
only boots the container), writes back `sam3d.npz` and `masks.npz` and goes on exactly as a stored run does: the
build view then plays SAM 3.1's mask over the clip and the stage badge says SAM 3D Body. If the GPU does not answer
inside its share of the build's wall clock, or anything else fails, the build logs why and finishes with MediaPipe.
It costs about $0.05-0.15 a clip on Modal's per-second billing (the Starter plan's $30 a month covers a few hundred);
`scaledown_window` keeps the container warm for five minutes between uploads, so back-to-back demos are fast.

One-time setup, in this order:

1. Deploy the app (needs the `a2h-hf` Modal secret with `HF_TOKEN`, for the gated checkpoints):
   ```bash
   MODAL_PROFILE=a2h python -m modal deploy frontends/live_pose_modal.py
   ```
2. Give Cloud Run the Modal token and switch the pose on. The script reads the token from `~/.modal.toml`, puts it in
   Secret Manager (never on a command line or in the terminal) and sets the two variables on the service:
   ```bash
   MODAL_PROFILE=a2h PATH="$HOME/google-cloud-sdk/bin:$PATH" scripts/cloudrun_modal_pose.sh
   ```
   `deploy.sh` keeps those settings on later deploys (`--update-env-vars`). `A2H_LIVE_TIMEOUT_S` goes to 900 s so a
   cold container fits in the build's wall clock. `curl $API/api/health` then shows `"live": {..., "pose": "modal"}`.

Locally, the same thing is `A2H_LIVE_POSE=modal` with `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` exported (from
`python -m modal token new`) before `uvicorn server.main:app`.

### Live training on a GPU

Once a live build has its environment, the page can offer "Train on a GPU": the `modal` runner (`server/train.py`)
sends the build's `env.zip` to the Volume `a2h-runs` and spawns the same `run_job` as `scripts/remote_run.py --backend
modal`, which trains `python -m a2h_env train <env.zip> --iters 1000 --attempt-every 50` on an L40S: mjlab's G1
tracking task on the reference and every variant the build kept (`docs/TRAINING_ENV.md`), the policy the download
trains. Every 50 iterations the trainer rolls the policy out once on the clip (`a2h_env/attempts.py`, no video:
~30 KB of qpos) and the server copies it into the job's folder; the build view plays each attempt on the G1 with the
retarget as a ghost, with the reward curve under it, as they arrive. At the end the checkpoint comes back
(`GET /api/jobs/{id}/policy.pt`), the Volume folder is deleted, and the job's policy stage is done. About 20-30 min
and $1 of GPU per training; the server cancels the Modal call if the visitor cancels the job, if the training runs
past `A2H_TRAIN_TIMEOUT_H`, if the follow loop ends any other way than the call finishing, or when the process stops
or restarts under it. A failed or cancelled training leaves the build as it was (the job stays succeeded, the policy
stage says what happened) and can be started again; a trained one can't. `A2H_RATE_TRAIN` (12 an hour per client)
bounds invite-code guessing.

No secret goes up with a training (it needs no Hugging Face token), and nobody can start one without the gate: an
invite code (`A2H_TRAIN_CODE`) the visitor types on the page, or a daily allowance (`A2H_TRAIN_PER_DAY`), or both.
`/api/health` shows `"train": {...}` with the gate and the queue when the runner is on; `/readyz` reports
`train_backend`. The trained policy and the attempts live with the upload and expire with it (`A2H_UPLOAD_TTL_H`).

Setup, after the GPU pose (the Modal token is already on the service):

1. Deploy the GPU app once (cached after): `MODAL_PROFILE=a2h python -m modal deploy scripts/modal_app.py`.
2. Turn the runner on and store the invite code in Secret Manager (typed at a prompt, never on a command line):
   ```bash
   MODAL_PROFILE=a2h PATH="$HOME/google-cloud-sdk/bin:$PATH" scripts/cloudrun_train.sh
   ```
   `A2H_TRAIN_PER_DAY=2 scripts/cloudrun_train.sh` adds two free trainings a day. `deploy.sh` keeps the settings.
3. Optionally set a spending limit in the Modal dashboard: the gate bounds how many trainings start, the limit
   bounds what they can cost.

Locally: `A2H_RUNNERS=demo,modal A2H_TRAIN_CODE=<code> A2H_TRAIN_BRANCH=<branch>` with `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET` exported, then `uvicorn server.main:app`. The tests drive the runner with a stand-in Modal
(`server/tests/test_train.py`); a stored run's attempts for the gallery come from `scripts/record_attempts.py`.

### Keeping environments between visits

Job state lives in memory and in a SQLite journal under `A2H_DATA_DIR`, and a live build's files sit with its upload
there. On Cloud Run that folder is `/tmp`: a new revision or a restart forgets every build, and an upload's files
expire after `A2H_UPLOAD_TTL_H`. The store (`server/store.py`) keeps what a visitor made past both: when a live
build finishes it copies the build's files (`processed.mp4`, `mask.mp4`, `live.json`, `task_spec.json`, `qpos.npy`,
`env.zip`) and the job's record (its public JSON with a short log, who owns it, what was kept) to the store; when a
training finishes, `train/weights.json`, `train/policy.pt` and `train/attempts/*` follow. A job that isn't in memory
any more (pruned, expired, or from before a restart) comes back from its record on the next `GET /api/jobs/{id}`, and
its files are fetched into the job's folder again before they are served. Raw uploads are never kept.

Two backends behind one interface, chosen by `A2H_STORE`: local files under `A2H_DATA_DIR/persist` with a SQLite
record table (the default, and what the tests use; on a single always-on machine that is already durable, capped by
`A2H_STORE_TTL_D` and `A2H_STORE_MAX_MB`; on Cloud Run that folder is `/tmp`, so `/api/health` says `"durable": false`
and the page says so), or a Cloud Storage bucket (`gcs://<bucket>[/<prefix>]`: objects `jobs/<id>/<file>`, the record at
`jobs/<id>/record.json`, an owner index `owners/<key>/<id>` so a visitor's list is one prefix listing, and
`trainings/<id>` for GPU trainings). No Firestore: a JSON record per job in the bucket is enough for this, and it
needs one dependency (`google-cloud-storage`) and one IAM binding (`objectAdmin`; `/readyz` lists one object, which
needs no bucket metadata permission).

The training index matters on a host that scales to zero: a GPU training is written to the store the moment it starts
(one row per training, a retry included, and again with its Modal call id), and the instance following it refreshes
the row's heartbeat every poll. Another instance counts a row with a fresh heartbeat as running (two can run briefly
during a rollout) and leaves it alone; a row whose heartbeat is older than three polls (90 s at least) belongs to a
process that died, so the next instance to meet it cancels the call and marks the training failed. The day's
allowance counts rows. `/api/health` and `/readyz` read counts cached for 10 s. Restores of a kept job's
files take one lock per job, write whole files under unique temporary names, fetch `live.json` last, and fetch again
whatever is still missing on the next request.

**No accounts.** The browser makes a 32-hex key on its first visit (`localStorage` `a2h_owner`), sends it as
`X-A2H-Owner` on every API call, and the server records it on each job it creates. `GET /api/my/jobs` lists the
visitor's builds and trainings (status, date, the task spec's one line, links to the build view, `env.zip` and the
trained policy), which is the `/my` page ("Mine" in the navigation). The page shows a link with `?key=` that restores
the key on another device (in the URL fragment, so it reaches no server log, and the page takes it out of the
address at once); anyone with the link sees that list, so it is theirs to share or not. Without the key
nobody can list anyone's jobs; a job's id stays the only key to its files, and the hosted job list stays off.
`/api/health` reports `"store": {"kind": "local" | "gcs"}`; `/readyz` checks the store answers.

One-time setup on Google Cloud (the bucket, the service account's access, `A2H_STORE` on the service):

```bash
PATH="$HOME/google-cloud-sdk/bin:$PATH" scripts/cloudrun_store.sh
```

`BUCKET=<name>` picks another bucket name (default `<project>-jobs`, in the service's region, uniform access, public
access prevented). `deploy.sh` keeps `A2H_STORE` on later deploys. Locally, `A2H_STORE=gcs://<bucket>` works with
`gcloud auth application-default login`.

### What the bundle contains

`scripts/build_space.py` (also `make space`) copies only what the app shows. Training environments
(`docs/TRAINING_ENV.md`): a run's `env.zip` is shipped when the run folder has one, else built while bundling; since
`deploy.sh` runs `build_space.py` without numpy or MuJoCo, a bundle built there has no foot contacts and only the
variants of a sibling `remote_variants_*` run, so run `python scripts/export_env.py outputs/remote_latent_*` from the
pipeline's environment first. The bundle also ships each clip's `qpos.npy`, which the API checks a shipped `env.zip`
against; a run folder with the zip and the spec but no retarget still serves the zip while its manifest matches the
spec. A `--code-only` redeploy (CI) ships no run data and so no bundles; the API then builds
them on demand into `A2H_DATA_DIR/env/` from whatever run data the host already has.

It copies:

- videos, metrics and task specs (`task_spec.json`);
- the latest checkpoint per policy, motion CSVs and the reward net;
- training curves as JSON, and the demo-replay log lines.

It never copies `.env`, raw clips or full logs.

The default runs are the ones trained on the SAM 3D Body references (`docs/PIPELINE_ACCURACY.md`, fix 3):
`outputs/remote_latent_s3_*` (any folder with an `eval.json`; `DEFAULT_RUNS` in the script). The MediaPipe-era runs
(`remote_latent_final`, `remote_latent_kp_*`, `remote_latent_loose_*`) are no longer shipped. To ship other runs,
pass their folders explicitly.

The home page features `latent_s3_kick`, the front kick, whenever that run is in the bundle. `--demo-run <id>` picks
another.

The live runner adds about 50 MB:

- its pipeline modules (`config.py`, `pose.py`, `track.py`, `retarget.py`, `render.py`, `skill/`);
- the MediaPipe pose model;
- the G1 model files `scene.xml` references (meshes plus Menagerie's LICENSE);
- a second hash-locked dependency set, `server/requirements-live.lock`. It holds CPU-only torch from the PyTorch
  index, MediaPipe, MuJoCo, mink, daqp, SciPy and headless OpenCV.

`--live off` leaves all of that out. The assets come from `scripts/fetch_assets.sh`.

Sizes on 2026-10-28, for six runs (final, three `kp_*`, `loose_sj_tp` and `loose_lunge`):

| What | Size |
|---|---|
| Bundle, total | 330 MB |
| Bundle, runs | 279 MB |
| Bundle, live pipeline and assets | 50 MB |
| Bundle, distinct files | 278 MB |
| Image, compressed | 0.9 GB |
| Image, unpacked | 2.3 GB, of which Python packages are 1.8 GB |

Git LFS stores each distinct file once. Two runs share a reward net and every run carries the same clip renders,
so the upload is the distinct-files size.

The image runs as uid 1000 from a digest-pinned base, with hash-locked dependencies. After changing
`server/requirements.txt` or `server/requirements-live.txt`, run `make lock`. It resolves for the Space (linux
x86_64, Python 3.12), not for your machine.

The image is built to run with a read-only root filesystem. The live worker writes only to its job's scratch
folder under `/tmp`. Its caches (matplotlib, XDG, torch) are pointed there too, and nothing is rendered, so it
needs no `MUJOCO_GL`. To try it locally (on Apple silicon, add `--platform linux/amd64` to both commands, since the
lock is for x86_64):

```bash
docker build -t a2h-space build/space && docker run --rm --read-only --tmpfs /tmp --cap-drop ALL -p 7860:7860 a2h-space
```

### By hand

These are the steps the script runs, if you'd rather run them yourself.

```bash
uv run --no-project --with imageio-ffmpeg python scripts/build_space.py --cors https://athlete2humanoid.pages.dev
```

```bash
hf repos create <hf-user>/athlete2humanoid-api --type space --space-sdk docker --exist-ok
```

```bash
hf upload <hf-user>/athlete2humanoid-api build/space . --repo-type space
```

To follow the image build:

```bash
hf spaces logs <hf-user>/athlete2humanoid-api
```

`/readyz` answers 200 once runs are loaded and storage works; `/healthz` only says the process is up.
`/api/health` lists `"runners": ["demo", "live"]` when the live pipeline loaded:

```bash
curl https://<hf-user>-athlete2humanoid-api.hf.space/readyz
```

Then build and deploy the frontend:

```bash
cd web && npm ci && API_BASE=https://<hf-user>-athlete2humanoid-api.hf.space npm run build
```

```bash
npx wrangler pages project create athlete2humanoid --production-branch main
```

```bash
npx wrangler pages deploy dist --project-name athlete2humanoid
```

The frontend build writes `dist/_headers`, which Cloudflare Pages applies. It holds a Content-Security-Policy that
allows scripts and styles from the site itself, and network or media requests only to `API_BASE`, plus the usual
hardening headers. If you later add a third-party origin (fonts, analytics), add it in `web/vite.config.ts`. If
the printed `*.pages.dev` URL isn't `https://athlete2humanoid.pages.dev`, allow it on the API (next section).

### Changing the allowed origin, or adding a domain later

Space variables override the Dockerfile defaults and don't need a rebuild of the bundle:

```bash
hf spaces variables add <hf-user>/athlete2humanoid-api -e A2H_CORS_ORIGINS=https://athlete2humanoid.pages.dev,https://your-domain.com
```

```bash
hf spaces restart <hf-user>/athlete2humanoid-api
```

To add a custom domain to the frontend: Cloudflare dashboard → Workers & Pages → athlete2humanoid → Custom
domains → add `your-domain.com`. The API can stay on `hf.space`. To move it under your domain too (for example
`api.your-domain.com`), proxy it with a Cloudflare Worker or a CNAME, then rebuild the frontend with the new
`API_BASE`.

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every pull request and push to `main`: ruff, the server tests, the unit tests
(CPU torch + MuJoCo + the G1 model), the frontend typecheck and build (plus a scan of the bundle for secrets and a
check of the CSP headers), `pip-audit` on both locks and `npm audit` (reported, never blocking), and a build of the
Space image (with the live pipeline and its assets) that is then started read-only and probed: hosted lockdown,
`live` among the runners, and the pipeline's modules, MediaPipe model and G1 model loading on the read-only
filesystem. No secrets are needed for any of it.

`.github/workflows/keepalive.yml` GETs `$API_BASE/readyz` every 12 hours, so the free Space never reaches its
48-hour sleep. It runs only when the `API_BASE` variable is set, and needs no secrets. GitHub pauses scheduled
workflows after 60 days without commits; re-enable it from the Actions tab if so.

`.github/workflows/deploy.yml` runs on push to `main` (and by hand from the Actions tab). Each half runs only when
its secrets and variables exist; otherwise it is skipped, not failed. To turn them on, in the GitHub repo go to
Settings → Secrets and variables → Actions:

| Kind | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | A Hugging Face token with **write** access to the Space (a fine-grained token scoped to that one Space is best) |
| Variable | `HF_SPACE` | `<hf-user>/athlete2humanoid-api` |
| Variable (optional) | `A2H_CORS` | The frontend origin(s), comma-separated. Default `https://athlete2humanoid.pages.dev` |
| Variable (optional) | `A2H_DEMO_RUN` | Run id the demo runner replays for uploads |
| Secret | `CLOUDFLARE_API_TOKEN` | A Cloudflare API token with the **Cloudflare Pages: Edit** permission (dashboard → My Profile → API Tokens → Create Token → Custom) |
| Secret | `CLOUDFLARE_ACCOUNT_ID` | Your account id (dashboard → Workers & Pages, right-hand column) |
| Variable | `API_BASE` | `https://<hf-user>-athlete2humanoid-api.hf.space` |
| Variable (optional) | `CF_PAGES_PROJECT` | The Pages project name, default `athlete2humanoid` |

The Space deploy uploads **code only** (`server/`, the live pipeline's modules, `Dockerfile`, `README.md`). The
runs come from `outputs/` and the live runner's assets from `assets/`, and neither is in git. So do the first
deploy with `scripts/deploy.sh`. It puts `data/runs` and `assets/` on the Space and creates the Pages project.
After that, every push to `main` updates the code, leaves the data and assets alone, and waits until `/readyz`
answers. To ship new runs, run `scripts/deploy.sh` again.

Both deploy jobs use a GitHub environment named `production`; to require a manual approval before each deploy,
add yourself as a required reviewer under Settings → Environments → production.

## Demo examples (work offline)

`/examples` presents curated results as short stories: what the robot was shown, what it learned, and how we
know, with synchronized videos and the numbers from each run's own files. They are static files in
`web/public/examples/` (about 7 MB), so the Pages deploy is a full demo even when the Space is asleep or down. The
header then shows "Offline demo", uploads are disabled, and the page keeps checking the API every 30 s.

Rebuild them after a new run lands in `outputs/` (needs `imageio-ffmpeg`; MuJoCo and the G1 assets are optional,
for the arm-waving render and the squat's stage timeline):

```bash
python scripts/build_examples.py
```

The squat-jump task policy slot fills itself from `outputs/remote_latent_final/` (`eval.json`, `policy_*/policy.mp4`)
and moves to the top. The keypose front kick (`front-kick`) comes second. It reads `outputs/remote_latent_kp_kick/`
(`eval.json`, `task_spec.json`, `status.json`). The deps for a full rebuild, without installing anything into
your own Python:

```bash
uv run --no-project --python 3.12 --with numpy --with scipy --with mujoco --with mediapy --with imageio-ffmpeg --with torch python scripts/build_examples.py
```
 To validate the built files (no `outputs/` needed; also run by `make test`):

```bash
python scripts/build_examples.py --check
```

## Before recording the demo

- `/examples/<id>` pages are made for recording: one shared play/pause, scrubber and 0.5× speed for every video
  on the page, with the action's stages on the timeline.

- Load the Space URL once so it is awake. Free Spaces sleep after 48 h idle; the keepalive workflow prevents
  that once `API_BASE` is set.
- Upload a short clip with one person in shot (whole body, ≤ 20 s) to show the live runner build its task
  environment.
- Gallery → "Replay pipeline" on the main run gives a ~40 s job with live stages, logs and training curves,
  then "View results".
- To shorten or lengthen the replay: `hf spaces variables add <hf-user>/athlete2humanoid-api -e A2H_DEMO_SECONDS=30`.
