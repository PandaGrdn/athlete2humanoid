# System design: the web app and the GPU backends

How a video becomes a policy in the hosted product, what can fail on the way, and what the system does about it.
Security limits are in `docs/SECURITY.md`; deploy steps in `docs/DEPLOY.md`; the FarmShare backend in
`docs/FARMSHARE.md`.

## Components

```
 browser ──HTTPS──▶ Cloudflare Pages (web/dist: React SPA, _headers CSP)
    │
    └──HTTPS (fetch, EventSource, <video>)──▶ Hugging Face Space (Docker, CPU)
                                              uvicorn ─ FastAPI app (server/main.py)
                                              ├─ middleware: Hardening → RateLimit → BodyLimit → CORS
                                              ├─ runs.py      finished runs on disk (data/runs, read-only)
                                              ├─ jobs.py      JobStore + runners (threads)
                                              ├─ journal.py   SQLite journal (A2H_DATA_DIR/jobs.sqlite3)
                                              └─ A2H_DATA_DIR uploads (/tmp/a2h on the Space)

 local machine only (never hosted):
   colab runner ─▶ scripts/remote_run.py ─┬─ --backend colab      ─▶ Colab A100 VM (colab CLI)
                                          └─ --backend farmshare  ─▶ FarmShare Slurm, L40S (ssh/rsync)
   results land in outputs/remote_<session>/ ─▶ build_space.py ─▶ the Space's data/runs
```

| Component | State | Scales by |
|---|---|---|
| Pages frontend | none (static) | Cloudflare's CDN |
| Space API | runs (read-only files), jobs (memory + SQLite), rate-limit buckets (memory), uploads (disk) | one process; see "Scaling" |
| Demo runner | a thread per job, replaying a recorded run's log and curves on a ~40 s clock | `A2H_MAX_ACTIVE_JOBS` |
| Colab / FarmShare runner | a `remote_run.py` subprocess per job, hours long | GPU quota (Colab units; FarmShare: 4 running jobs per user) |

## Data flow

1. **Gallery.** `GET /api/runs` scans `A2H_RUNS_DIR` for folders with `eval.json` and summarises each (metrics,
   curves, clips, robustness). Artifacts are served by `GET /api/runs/{id}/files/{path}` with Range support, from an
   extension allow-list, never outside the run folder.
2. **Job.** `POST /api/jobs` with a gallery `run_id` or an uploaded video. The upload streams to
   `uploads/_incoming/<uuid>.<ext>` under the body limit, is checked (extension, declared type, magic bytes, size),
   then moved to `uploads/<job id>/input.<ext>`. The job is written to the journal and a runner thread starts.
3. **Progress.** The runner updates stages, log lines (redacted for secrets) and training curves. The client follows
   `GET /api/jobs/{id}/events` (SSE: the whole job on each change), or polls `GET /api/jobs/{id}` if SSE fails or
   the stream cap is reached. As soon as the retarget is compiled into a task spec (the "Task environment" stage,
   minutes into an hours-long job), `GET /api/jobs/{id}/task_spec` serves a sanitized view of it (key moments,
   flights, body parts, the retarget's self-check) so the page shows the RL environment long before the policy;
   finished runs serve the same view at `GET /api/runs/{id}/task_spec`.
4. **Result.** A finished job points at a run (`result_run`); the frontend opens that run's page.

## Job lifecycle

```
           create (journaled)          runner thread starts
 (none) ──────────────────▶ queued ───────────────────────▶ running ──┬──▶ succeeded
                                                                       ├──▶ failed     (error shown, details in server log)
                                                                       └──▶ cancelled  (POST /cancel)
 on restart: queued/running ──▶ failed ("the server restarted while this job was running; start it again")

 terminal + A2H_UPLOAD_TTL_H (1 h)  → upload deleted
 terminal + A2H_JOB_TTL_H (24 h)    → job deleted (memory and journal); also only the newest 200 terminal jobs are kept
```

Writes: status changes are written to the journal immediately; log/progress changes mark the job dirty and a
background thread writes dirty jobs once a second (a crash loses at most ~1 s of log lines). The same thread runs
the expiry sweep every minute and removes orphaned upload folders and stale partial uploads.

## Failure modes

| Failure | Detection | What happens | Residual risk |
|---|---|---|---|
| API process restarts (deploy, crash, Space restart) | journal at startup | Finished jobs reappear; jobs that were running are marked failed with a clear message; partial uploads removed. On the free Space `/tmp` is wiped on restart, so there the journal survives only process restarts, not container restarts (see Scaling for a persistent volume). | Demo jobs aren't resumed: they're 40 s replays, so "start it again" is cheaper than resume logic. |
| Runner bug / exception | catch-all in the runner thread | Job → failed with `internal error … (ref X)`; traceback in the server log under that ref. | |
| Client disconnects mid-upload | exception in `_save_upload` | Partial file deleted. | |
| Disk filling | upload storage cap, body limit, TTL sweep | New uploads get 503; gallery replays keep working. | |
| Abusive client | per-client rate limits, per-client job cap, stream cap | 429 / 503 with `Retry-After`. | One process's memory; resets on restart. |
| **GPU backend unavailable** (colab CLI missing, FarmShare unreachable) | `gpu_backend_status` on every request | `/api/health` lists only `demo` and reports `degraded`; a request for the GPU runner falls back to the demo runner and the job log says so. | |
| **Colab VM lost** | `remote_run.py`: status unknown 5× or "No active" | Stops the VM, recreates it once, resumes from checkpoints synced every 10 min. | A second loss fails the job. |
| **GPU budget exhausted** (Colab units gone) | `colab new` fails → `remote_run.py` exits non-zero | Job → failed with the exit code. Switch the backend: `--backend farmshare` (CLI) or `A2H_REMOTE_BACKEND=farmshare` (web app). | Not auto-detected from the error text; the operator switches. |
| **FarmShare job lost** (node failure, preemption, time limit) | `sacct` state + `status.json` still "running" | Resubmitted once from synced checkpoints. Ctrl-C → `scancel`. | |
| FarmShare SSH master expires mid-job | 5 unanswered polls (10 min) | `remote_run.py` stops with instructions; the Slurm job keeps running and its outputs stay on FarmShare. | Results fetched by hand or by rerunning. |
| **Partial results** (some policies trained, eval missing) | no `eval.json` in the output folder | The job is failed, but every file that came back stays in `outputs/remote_<session>/` (checkpoints, `job.log`). The gallery ignores folders without `eval.json`, so half-finished runs never appear publicly. | |
| Colab/FarmShare job runs forever | `A2H_JOB_TIMEOUT_H` (8 h) in the web runner; `--fs-time` in Slurm | Driver process terminated, job failed. | Terminating `remote_run.py` doesn't stop a Colab VM; `colab stop -s <session>` by hand. |
| Bad configuration | `Settings.validate()` at startup | The process doesn't start; the log lists every bad variable. | |
| Space asleep (48 h idle) | | First request wakes it (~30 s). | Before a demo, load `/readyz` once. |

## Health endpoints

| Endpoint | Meaning | Use |
|---|---|---|
| `GET /healthz` | Liveness: the process answers. Checks nothing else, so it can't cause restart loops. | Docker `HEALTHCHECK`, uptime monitors |
| `GET /readyz` | Readiness: at least one run to serve, data dir writable, journal answering. 503 with the failing checks otherwise. Also reports the GPU backend state and active jobs. | deploy workflow (waits for 200), load balancers |
| `GET /api/health` | What the frontend needs: runners available now, limits, stages, `degraded` notes. | the UI |

## Logs

One JSON object per line on stderr (`A2H_LOG_FORMAT=text` for local reading), fields: `ts`, `level`, `logger`,
`msg`, plus event fields. Every request gets an `X-Request-ID` (also in the access line) so a user-reported
`ref` can be found. Client addresses are logged as a 10-hex pseudonym (SHA-256 with a per-process random salt), never raw. Events:
`request`, `rate limited`, `job created`, `job finished` (with duration), `runner crashed` (with traceback and ref),
`jobs recovered`, `expired`, `config` warnings, `started`. Uvicorn's own access log is off in the image (ours
replaces it).

```bash
hf spaces logs <hf-user>/athlete2humanoid-api | grep '"job finished"'
```

## Configuration

Everything is an environment variable, documented in `server/config.py` and `docs/DEPLOY.md`. Values are parsed
and range-checked at startup; unknown runners or backends, non-numbers, zero limits and a wildcard CORS origin on a
hosted deploy stop startup with a list of every problem. Hosted mode (`A2H_HOSTED=1`, or a Space's `SPACE_ID`)
pins the safety-relevant choices regardless of other variables.

## Scaling notes

Today: one uvicorn process on a 2-vCPU Space, which is plenty for a demo (demo jobs are sleeps and small JSON
updates; video bytes are served by Starlette's file responses with Range support).

- **More traffic, same design:** put the Space behind Cloudflare (proxy a custom domain) so `/api/runs/*/files/*`
  (immutable videos, `Cache-Control: public, max-age=3600`) is served from the edge. Raise
  `A2H_MAX_STREAMS` and `--limit-concurrency` together.
- **Several processes or replicas:** the in-memory pieces have to move out. Rate limits and SSE fan-out → Redis;
  the job journal → Postgres (the `Journal` class is the only place to change); uploads → object storage (R2/S3)
  with presigned PUTs, so upload bytes never touch the API; runners → a queue (one worker per GPU backend) instead
  of threads in the API process.
- **Persistent job history on the Space:** attach Hugging Face persistent storage and set `A2H_DATA_DIR=/data/a2h`.
  The journal and uploads then survive container restarts as well.
- **Real GPU jobs for the public:** only with accounts and quotas. Each job costs GPU hours, so anonymous access to a
  GPU runner stays impossible by design (`hosted` mode), not by configuration.
