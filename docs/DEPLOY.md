# Web app: local dev and deploy

The product UI is two pieces:

- `server/`: a FastAPI backend. It lists finished runs (`outputs/remote_*`), serves their
  artifacts, accepts uploads and runs jobs with progress over server-sent events.
- `web/`: a React + Vite + TypeScript frontend with a landing page and gallery, live job
  progress, and a results page.

Jobs go through one of two runners:

| Runner | What it does | Where |
|---|---|---|
| `demo` (default) | Replays a finished run: its real log lines and training curves, on a ~40 s clock. Results are that run's results. | Everywhere. This is what the hosted deploy uses, because free tiers have no GPU. |
| `colab` | Copies the upload into `clips/` and runs `scripts/remote_run.py latent` on a Colab A100 (about 3 hours). It follows `runner.log`, `job.log` and `status.json`, and the finished run joins the gallery. | Local only. Needs `A2H_RUNNERS=demo,colab` **and** `A2H_ENABLE_COLAB=1`, plus a logged-in `colab` CLI. It spends compute units, so it is off by default. |

The UI says so when a job is a demo replay, including for uploads.

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

All settings are server-side. The client only ever gets `API_BASE`.

| Variable | Default | Meaning |
|---|---|---|
| `A2H_RUNS_DIR` | `outputs/` | Folder of run folders (any sub-folder with an `eval.json`) |
| `A2H_DATA_DIR` | `outputs/_web` | Writable scratch for uploads |
| `A2H_CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated origins allowed to call the API |
| `A2H_RUNNERS` / `A2H_ENABLE_COLAB` | `demo` / unset | See runners above |
| `A2H_DEMO_RUN` | newest run with a video | The run that uploads replay |
| `A2H_DEMO_SECONDS` | `40` | Length of a demo replay |
| `A2H_MAX_UPLOAD_MB` | `200` (`100` on the Space) | Upload limit. The extension, content type and file magic bytes are all checked. |
| `A2H_MAX_ACTIVE_JOBS` | `4` | Concurrent jobs. Further requests get a 429. |
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

Replace `<hf-user>` with your Hugging Face username. The Space URL is then
`https://<hf-user>-athlete2humanoid-api.hf.space`; the Pages URL is `https://athlete2humanoid.pages.dev` (or
`https://<project>.pages.dev` if that name is taken: wrangler prints it).

### 1. Backend: Hugging Face Space

Log in once. This needs a token with **write** access from https://huggingface.co/settings/tokens:

```bash
pip install -U huggingface_hub
```

```bash
hf auth login
```

Build the Space folder from your finished runs. Point `--onnx-python` at a Python with torch and onnx (the
mjlab venv) to ship ONNX policies; leave it out to ship checkpoints only. Use a Python with `imageio-ffmpeg`
(the mjlab venv again), or have `ffmpeg` on PATH, to get gallery poster frames.

```bash
python3 scripts/build_space.py outputs/remote_latent outputs/remote_latent_v2 --cors https://athlete2humanoid.pages.dev --demo-run latent --onnx-python <mjlab venv>/bin/python
```

The script prints the bundle size (about 85 MB for these two runs) and copies only what the app shows: videos,
metrics, the latest checkpoint per policy, motion CSVs, the reward net, training curves as JSON, and the
demo-replay log lines. It never copies `.env`, raw clips or full logs.

Create the Space and upload. Re-run the upload to redeploy:

```bash
hf repos create <hf-user>/athlete2humanoid-api --type space --space-sdk docker --exist-ok
```

```bash
hf upload spaces/<hf-user>/athlete2humanoid-api build/space . --repo-type space
```

The first build takes about 2 minutes. To follow it:

```bash
hf spaces logs <hf-user>/athlete2humanoid-api
```

Check it:

```bash
curl https://<hf-user>-athlete2humanoid-api.hf.space/api/health
```

### 2. Frontend: Cloudflare Pages

```bash
cd web && npm ci && API_BASE=https://<hf-user>-athlete2humanoid-api.hf.space npm run build
```

```bash
npx wrangler login
```

```bash
npx wrangler pages project create athlete2humanoid --production-branch main
```

```bash
npx wrangler pages deploy dist --project-name athlete2humanoid
```

Open the printed `*.pages.dev` URL. If it isn't `https://athlete2humanoid.pages.dev`, allow it on the API (see
"Changing the allowed origin" below).

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

## Before recording the demo

- Load the Space URL once so it is awake (free Spaces sleep after 48 h idle).
- Gallery → "Replay pipeline" on the main run gives a ~40 s job with live stages, logs and training curves,
  then "View results".
- To shorten or lengthen the replay: `hf spaces variables add <hf-user>/athlete2humanoid-api -e A2H_DEMO_SECONDS=30`.
