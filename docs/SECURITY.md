# Security review: web backend, Space image, frontend, CI

Scope: `server/` (FastAPI, including the `live` runner's deploy-side controls added 2026-10-28), the Hugging Face
Space image built by `scripts/build_space.py`, the Cloudflare Pages
frontend (`web/`), the GPU runners (`colab`, and the FarmShare backend in `scripts/remote_run.py`) and the
GitHub Actions workflows. Reviewed on 2026-10-27 at commit `f8e21fe`. Each finding lists what was wrong, what
changed, and the test that covers it (`server/tests/test_hardening.py` unless noted).

Threat model: the hosted demo is public and anonymous. Anyone can list runs, download artifacts, upload a video
and start a demo job. What we protect: other visitors' uploads, the host's disk/CPU/bandwidth (free tier), the
server's internals (paths, tracebacks), the user's secrets (HF token, deploy tokens), and the GPU budget (no
anonymous user may start a real GPU job).

## Findings

### High

**H1. Upload size was only enforced after the whole body had been spooled to disk.**
Starlette's multipart parser writes the full upload to a temp file before the handler runs, so the
`A2H_MAX_UPLOAD_MB` check in `_save_upload` came too late: a 20 GB POST would fill `/tmp` on the Space.
*Fix:* `security.BodyLimit` ASGI middleware. A `Content-Length` over the limit gets a 413 before any byte is read;
a chunked body (or a lying `Content-Length`) is counted while it streams and cut off the moment it passes the limit,
and the reply is a 413 whatever the inner parser makes of it. Uploads may be `A2H_MAX_UPLOAD_MB` + 64 kB of
multipart framing; every other endpoint 64 kB. The per-file check in `_save_upload` stays as a second line.
Tests: `test_declared_body_over_limit_is_refused_before_parsing`, `test_streamed_body_over_limit_is_cut_off`,
`test_small_endpoints_take_small_bodies`, `test_upload_size_limit` (test_api).

**H2. Anyone could see anyone's uploaded video.** `GET /api/jobs` listed the 20 latest jobs with their ids and
input URLs, and a job id is the only key to `/api/jobs/{id}/input` (and to cancelling it). Ids were also 48-bit.
*Fix:* the job list is off on hosted deploys (`A2H_LIST_JOBS`, default off when `A2H_HOSTED`) and never includes
`input`; ids are 128-bit (`uuid4().hex`); uploads are deleted `A2H_UPLOAD_TTL_H` (1 h) after the job ends. The
frontend never used the list. Tests: `test_hosted_can_never_enable_colab`, `test_job_list_hides_inputs`, `test_expiry`.

**H3. No abuse limits besides a global active-job cap.** One client could hold all 4 job slots, create jobs in a
loop, open unlimited SSE streams (each polls every 250 ms) or fill the disk with uploads.
*Fix:* per-client token buckets (`security.RateLimit`): `A2H_RATE_REQUESTS` (600/min) on `/api/*` and
`A2H_RATE_JOBS` (30/h, burst 5) on job creation, 429 + `Retry-After`; `A2H_MAX_JOBS_PER_IP` (2) active jobs per
client; `A2H_MAX_STREAMS` (100) open SSE streams (503 → the client already falls back to polling); total upload
storage `A2H_MAX_UPLOAD_STORAGE_MB` (2 GB, 503 when full); uvicorn `--limit-concurrency 200`; colab jobs time out
after `A2H_JOB_TIMEOUT_H`. The client is identified by the socket peer, or `A2H_PROXY_HOPS` entries from the right of
`X-Forwarded-For` (never the client-controlled left end); the Space sets 1. Tests: `test_rate_limiter_refills`,
`test_job_creation_rate_limit`, `test_request_rate_limit`, `test_per_client_active_jobs`,
`test_client_ip_trusts_only_configured_hops`, `test_stream_cap`, `test_upload_storage_cap`.

### Medium

**M1. The `colab` runner (real GPU spend) was only off by configuration.** A Space variable pair
(`A2H_RUNNERS=demo,colab`, `A2H_ENABLE_COLAB=1`) would have offered it to the public. (It would then have failed,
since the image has no `colab` CLI or scripts, but "fails by accident" is not a control.)
*Fix:* hosted mode. `A2H_HOSTED=1` is baked into the Space image, and a Space's own `SPACE_ID` variable implies it.
In hosted mode `enabled_runners` is `["demo"]` regardless of every other variable, and a startup warning says the
request was ignored. CI runs the built image with both colab flags set and asserts `runners == ["demo"]`. Tests:
`test_hosted_can_never_enable_colab`; CI job `space-image`.

**M2. Runner exceptions were returned to the client verbatim.** `error = f"{type(e).__name__}: {e}"` exposes file
paths and internals (`FileNotFoundError: /home/user/app/...`); the colab runner also logged the absolute path of
`remote_run.py` into the job log. *Fix:* the traceback goes to the server log with a short reference; the client
sees `internal error in the demo runner (ref 1a2b3c4d)`. Unhandled request errors return
`internal error (ref <request id>)`. Tests: `test_runner_errors_dont_leak_internals`, `test_unhandled_errors_are_generic`.

**M3. Frontend dependencies with known vulnerabilities.** `npm audit`: react-router-dom 6.28.0 (high: open
redirect / XSS, GHSA-2w69-qvjg-hvjx, GHSA-2j2x-hqr9-3h42) and vite 5.4.10 → esbuild (moderate: dev server readable
by any site, GHSA-67mh-4wv8-2f99). *Fix:* vite 6.4.5, @vitejs/plugin-react 4.7.0, and (2026-10-28) react-router 7.18.4
in library mode, replacing react-router-dom, which also clears the two moderate 6.x advisories (GHSA-wrjc-x8rr-h8h6,
GHSA-337j-9hxr-rhxg). All pinned exactly. `npm audit`: 0 vulnerabilities.

**M4. No security headers or CSP.** *Fix (API):* every `/api/*` response carries `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, HSTS, a `Permissions-Policy`, and
`Content-Security-Policy: default-src 'none'; frame-ancestors 'none'; sandbox` (JSON and media need nothing, so an
HTML file served by mistake can't run). `Server` headers are dropped. *Fix (frontend):* the Vite build writes
`dist/_headers` for Cloudflare Pages with a CSP of `script-src 'self'; style-src 'self'` (no `unsafe-inline`; React
style props go through CSSOM, which CSP doesn't block) and only the configured `API_BASE` allowed for
`connect-src`/`media-src`/`img-src`, plus `frame-ancestors 'none'`, `object-src 'none'`, `base-uri 'self'`, nosniff,
HSTS, COOP. Verified in a browser against a local API on another origin: landing page, gallery videos, a live job
over SSE and the charts all work with zero CSP violations. Tests: `test_security_headers`; CI job `web` greps `_headers`.

**M5. Container ran its own code as a writable user, unpinned.** The image `pip install --user`'d unpinned ranges
and `COPY --chown=user` made code and packages writable by the running process.
*Fix:* base image pinned by digest (`python:3.12.16-slim-bookworm@sha256:6dd8…`); dependencies installed from
`server/requirements.lock` with `--require-hashes --no-deps` as root; code root-owned; the app runs as uid 1000 and
writes only `/tmp/a2h`; `HEALTHCHECK` on `/healthz`; `--no-server-header`, keep-alive and graceful-shutdown
timeouts. CI runs it with `--read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges` and probes it.
(Hugging Face doesn't let a Space set those runtime flags; the image is built so they'd work, and the uid can't
modify `/app` or site-packages even without them.)

### Low

**L1. Upload temp file names** came from `id(file)` and `hash(filename)` (collisions possible between concurrent
uploads). Now `uuid4`. A client disconnect or cancellation mid-upload now removes the partial file too (was only on
validation errors), and leftovers in `_incoming` are removed at startup. Test: `test_incoming_leftovers_removed_at_startup`.

**L2. Content sniffing didn't tie magic bytes to the extension** (WebM bytes accepted as `.mp4`). Now ISO-BMFF for
mp4/mov/m4v and EBML for webm. The stored name is fixed (`input.<ext>`); the display name keeps only word
characters, `.`, `-`, space (Unicode letters fine; bidi overrides, `<`, `/` are not). Tests:
`test_magic_must_match_extension`, `test_display_name_is_sanitised`.

**L3. CORS** allowed any request header. Now `Content-Type` and `Range` only, `GET`/`POST`, no credentials, and a
hosted deploy refuses to start with `A2H_CORS_ORIGINS=*`; `build_space.py --cors` accepts only https origins (or
localhost). Tests: `test_cors_allows_only_configured_origin`, `test_hosted_rejects_wildcard_cors`.

**L4. Colab runner could overwrite a repo clip.** An upload named `squat.mp4` was copied to `clips/squat.mp4`.
Now prefixed with the job's tag (`clips/web1a2b3c_squat.mp4`). Local-only runner.

**L5. Bad configuration failed late or obscurely** (`float("abc")` at import). Settings are now parsed and
validated at startup and every bad variable is reported at once; the process doesn't start. Test:
`test_config_validation_lists_every_problem`.

**L6. `.env` handling in remote runs.** The Colab path uploads the whole `.env` to the VM as its token file
(unchanged here: out of scope, and the VM is the user's own). The new FarmShare path sends only the `HF_TOKEN` line,
over stdin (never argv or logs), into a `umask 077` directory. Test: `tests/test_farmshare.py::test_secret_goes_over_stdin_only`.

### Informational (no change)

- **Path traversal on artifacts**: `runs.safe_path` resolves the path (following symlinks), requires it to stay
  inside the run folder, rejects dot-segments and serves only an allow-list of extensions. The SPA route checks the
  resolved path is under the static root. Covered by `test_artifact_serving` (test_api). Good as is.
- **Checkpoints are PyTorch pickles** (`model_*.pt`, `reward.pt`): loading one runs code. They come from our own
  runs, so serving them is fine, but users should prefer the ONNX downloads; noted for the README.
- **`/docs` (Swagger UI) is public.** It documents only public endpoints; it gets the security headers minus the CSP.
- **Rate-limit state is in memory**, per process. Correct for the single-process Space; a multi-replica deploy needs
  a shared store (see SYSTEM_DESIGN.md).
- **`A2H_PROXY_HOPS=1` on the Space** assumes one proxy hop appends the client address. If the Space logs show a
  single client pseudonym for everyone, the hop count is wrong; per-IP limits then act as global limits (fail safe,
  not open).

## Live runner: deploy-side controls (2026-10-28)

The `live` runner (`server/live.py`) turns an anonymous upload into a task environment on the Space's own CPU.
That makes untrusted video the input to ffmpeg, MediaPipe and the pipeline's native code, so the build runs
under the controls below.

| Control | How |
|---|---|
| A separate process | `python -m server.live_worker <video> <out>` runs in its own session (`start_new_session`), with stdin closed. A cancel or timeout kills the whole process group. It reports progress as JSON lines on stdout. A traceback goes to the server log under a reference id; the client sees only `internal error building the environment (ref …)`, or a short message for a clip it can't use. |
| No secrets in its environment | The worker gets a fresh environment, not a copy of the server's. It holds `PATH`, `PYTHONPATH`, thread caps (2), `MUJOCO_GL=disable`, and `A2H_ASSETS_DIR` when set. `HOME`, `TMPDIR`, the caches (`XDG_CACHE_HOME`, `MPLCONFIGDIR`) and `config.py`'s output folders all point at the job's own scratch folder, `A2H_DATA_DIR/uploads/<job>/live/scratch`. A Space secret, if one were ever added, would not reach it. |
| Resource limits | One build at a time (`A2H_LIVE_CONCURRENCY`, default 1); later jobs queue, under the existing per-client and global job caps. Only the first `A2H_LIVE_MAX_SECONDS` (20 s) of a clip is used; ffmpeg re-encodes it to at most 640 px, 30 fps, no audio or metadata. Wall clock: `A2H_LIVE_TIMEOUT_S` (240 s). `RLIMIT_CPU` is twice that. Address space: `RLIMIT_AS` = `A2H_LIVE_MEMORY_MB` (8192 MB, Linux only). No core dumps (`RLIMIT_CORE` 0). The worker runs at `nice 10`, so the API stays responsive. |
| Filesystem | The image is unchanged: uid 1000, root-owned `/app`, read-only root filesystem in CI. The pipeline's modules and assets (MediaPipe model, G1 meshes) are read-only in `/app`; the worker writes only to its scratch folder under `/tmp`. Measured in the image: a 5.6 s front-kick clip built its environment in 28 s (linux/amd64, emulated) with `RLIMIT_AS` at 8192 MB, and the container peaked at ~1 GB including the server. |
| Hosted mode | Unchanged: `A2H_HOSTED` still locks out `colab`. `live` turns on by itself only when its imports succeed (`A2H_LIVE=auto`); `A2H_LIVE=0` turns it off with a Space variable and no rebuild. CI asserts that `colab` is absent and `live` present in the image's `/api/health`. |
| Dependencies | A second hash-locked file, `server/requirements-live.lock`, is installed with `--require-hashes --no-deps` as root, like the server lock. It is resolved for the Space's platform (linux x86_64, Python 3.12). Packages shared with the server lock are constrained to that lock's versions. torch comes from the PyTorch CPU index: the hashes pin the exact `+cpu` wheels, so the extra index can't substitute another file. mediapipe's `opencv-contrib-python` is replaced by `opencv-python-headless` (`server/requirements-live-excludes.txt`). The only system packages added are `libegl1` and `libgles2`, from Debian's signed archive: the GL dispatch libraries that MediaPipe's C library links against even on its CPU path, with no drivers. |

Not covered, and accepted for a hackathon demo: the decoders in ffmpeg (the binary bundled with imageio-ffmpeg) and
MediaPipe run on attacker-supplied bytes. A decoder exploit would get the worker's uid 1000 with no secrets, a
read-only `/app`, the limits above and the Space's network. The Space has no secrets today; keep it that way.

## Dependency audit (2026-10-27)

| What | Tool | Result |
|---|---|---|
| `server/requirements.lock` (what the Space runs) | `pip-audit 2.10.1 --require-hashes` | No known vulnerabilities |
| `server/requirements-live.lock` (the live runner, 2026-10-28) | `pip-audit 2.10.1 --require-hashes` | No known vulnerabilities. pip-audit skips `torch==2.15.0+cpu` (a local version, not on PyPI), so CI audits `torch==2.15.0` separately: none. |
| `web/package-lock.json` before | `npm audit` | 5 (4 high, 1 moderate) |
| `web/package-lock.json` after | `npm audit` | 0 (react-router 7.18.4; see M3) |
| `requirements.txt` (pipeline, unpinned, not deployed) | `pip-audit` | runs in CI, non-blocking |

CI runs these audits on every PR (`audit` job, non-blocking) and Dependabot opens weekly PRs for Actions, npm and
`server/requirements*.txt`. After changing `server/requirements.txt` or `server/requirements-live.txt`, re-lock
with `make lock` (both locks).

## Secrets

- `scripts/deploy.sh` uses the `hf` and `wrangler` logins already stored on the machine; it never reads, prints or
  passes a token. Before uploading, it refuses a bundle that holds a `.env`, a key, a log or a token-shaped string.
  `.github/workflows/keepalive.yml` has `permissions: {}` and needs no secrets.
- The server needs no secrets. The Space image contains code, dependency locks, run data and the live runner's
  model files only:
  `build_space.py` copies `server/` without dotfiles, journals or tests, writes a `.dockerignore`, and never
  copies `.env`, raw clips or logs.
- The frontend bundle gets exactly one build-time value, `API_BASE`. CI fails the `web` job if the built bundle
  contains anything shaped like an HF, OpenAI or AWS key or a private key.
- Deploy credentials live only in GitHub Actions secrets (`HF_TOKEN`, `CLOUDFLARE_API_TOKEN`,
  `CLOUDFLARE_ACCOUNT_ID`), are passed to single steps via `env`, and the deploy jobs are skipped when they are
  missing. Workflows have `contents: read` only, check out with `persist-credentials: false`, and pin every
  Action to a commit SHA. If the Space ever needs a secret (it doesn't today), add it as a Space secret, which
  reaches the container as an environment variable at runtime and is never baked into the image.
- `.env` stays gitignored; log lines pass through `stages.redact` (HF/OpenAI-style tokens, `token=…`, `password=…`).

## Checklist for changes

- New endpoint taking a body: it gets the 64 kB cap automatically; add its path to `BodyLimit.upload_paths` only
  if it really needs more.
- New file type served from runs: add its extension to `runs.SERVABLE`; never serve `.log` or `.py`.
- New third-party origin in the frontend (fonts, analytics): add it to the CSP in `web/vite.config.ts`.
- New runner: make it unavailable when `settings.hosted`, like colab, unless it runs on this host's CPU under the
  live runner's controls (subprocess, fresh environment, limits).
- New module imported by `server/live_worker.py`: add it to `LIVE_MODULES` in `scripts/build_space.py`, and its
  dependencies to `server/requirements-live.txt` (then `make lock`).
