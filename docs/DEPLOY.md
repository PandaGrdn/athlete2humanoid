# Web app: local dev and deploy

The product UI is two pieces:

- `server/`: a FastAPI backend. It lists finished runs (`outputs/remote_*`), serves their
  artifacts, accepts uploads and runs jobs with progress over server-sent events.
- `web/`: a React + Vite + TypeScript frontend with a landing page and gallery, live job
  progress, and a results page.

Jobs go through one of three runners:

| Runner | What it does | Where |
|---|---|---|
| `demo` (default) | Replays a finished run: its real log lines and training curves, on a ~40 s clock. Results are that run's results. | Everywhere. |
| `live` | Builds a real RL task environment from the uploaded video on CPU: MediaPipe pose, mink retarget onto the G1, then the keypose task spec (`skill/keypose.py`). It runs `python -m server.live_worker` as a sandboxed subprocess and stops before training (that needs a GPU). | On whenever the pipeline imports (`A2H_LIVE=auto`). The Space image includes it. |
| `colab` | Copies the upload into `clips/` and runs `scripts/remote_run.py latent` on a Colab A100 (about 3 hours). It follows `runner.log`, `job.log` and `status.json`, and the finished run joins the gallery. | Local only. Needs `A2H_RUNNERS=demo,colab` **and** `A2H_ENABLE_COLAB=1`, plus a logged-in `colab` CLI. It spends compute units, so it is off by default. |

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
| `A2H_HOSTED` | unset (`1` in the Space image; a Space's `SPACE_ID` implies it) | Public deploy: the GPU runner can't be enabled, the job list is off, CORS `*` is refused |
| `A2H_REMOTE_BACKEND` | `colab` | GPU backend for the local GPU runner: `colab` or `farmshare` (docs/FARMSHARE.md) |
| `A2H_LIST_JOBS` | on locally, off hosted | `GET /api/jobs` (debugging) |
| `A2H_LOG_FORMAT` / `A2H_LOG_LEVEL` | `json` / `INFO` | Structured logs on stderr; `text` for humans |
| `A2H_RUN_EXCLUDE` | `*smoke*` | Glob patterns of run folders to hide |
| `A2H_STATIC_DIR` | unset | Serve a built `web/dist` from the API too (single-process hosting) |
| `API_BASE` (frontend, build time) | empty | The backend origin, e.g. `https://you-athlete2humanoid-api.hf.space`. Empty means same origin or the dev proxy. |

## Hosting choices

**Backend: Hugging Face Spaces (Docker, free CPU basic).** It fits the open-weights story: the API sits next to
the Meta models it talks about. It gives 2 vCPU and 16 GB of RAM for free, builds from a plain Dockerfile, and
stores videos and checkpoints with Git LFS (`hf upload` handles that). Render's free tier would also work, but it
spins down after 15 minutes idle (a slow first load on camera) and its free instances have no persistent disk.
A free Space sleeps only after 48 hours without traffic.

**Frontend: Cloudflare Pages.** The free tier has unlimited bandwidth and requests and allows commercial use
(Vercel's Hobby plan doesn't). It serves the SPA fallback from `web/public/_redirects`, deploys with one
`wrangler` command, and makes adding a domain easy later. A domain bought through Cloudflare Registrar is sold at
cost and connects to the project in two clicks.

## Deploy

Log in once to each service, then run one script. You need Node 18+ (`brew install node`) and
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

### What the bundle contains

`scripts/build_space.py` (also `make space`) copies only what the app shows:

- videos, metrics and task specs (`task_spec.json`);
- the latest checkpoint per policy, motion CSVs and the reward net;
- training curves as JSON, and the demo-replay log lines.

It never copies `.env`, raw clips or full logs.

The default runs are the final squat-jump run and every keypose run that has finished:
`outputs/remote_latent_final`, `outputs/remote_latent_kp_*` and `outputs/remote_latent_loose_*` (any folder with an
`eval.json`; `DEFAULT_RUNS` in the script). A new keypose run therefore joins the next deploy by itself. To ship
other runs, pass their folders explicitly.

The home page features `latent_kp_kick`, the front kick learned from its video alone, whenever that run is in the
bundle. `--demo-run <id>` picks another.

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
