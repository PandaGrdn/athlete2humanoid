"""athlete2humanoid web API.

    uvicorn server.main:app --reload --port 8000

Runs (finished pipeline outputs) are read from A2H_RUNS_DIR; jobs live in memory and in a SQLite
journal under A2H_DATA_DIR. See server/config.py for every setting, docs/DEPLOY.md for hosting,
docs/SECURITY.md for the limits and docs/SYSTEM_DESIGN.md for the job lifecycle.
"""

import asyncio
import logging
import re
import secrets
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import live, logs, runs, store as persist, train
from .config import Settings
from .journal import Journal
from .jobs import TERMINAL, JobStore, job_event
from .security import BodyLimit, Hardening, RateLimit, RateLimiter, anon, client_ip
from .stages import LIVE_STAGES, STAGES

log = logging.getLogger("a2h.api")
NON_UPLOAD_BODY = 64 * 1024  # every other request body is a tiny form or nothing
MULTIPART_SLACK = 64 * 1024  # multipart framing and the small form fields around the file

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-m4v", "video/webm", "application/octet-stream", ""}


def sniff_video(head, ext=None):
    """True for ISO-BMFF (mp4/mov/m4v: 'ftyp' at byte 4) and WebM/Matroska (EBML magic), matching ext if given."""
    iso = head[4:8] in (b"ftyp", b"moov", b"wide", b"mdat", b"free")
    ebml = head[:4] == b"\x1a\x45\xdf\xa3"
    if ext is None:
        return iso or ebml
    return ebml if ext == ".webm" else iso


def gpu_backend_status(settings):
    """(available, reason) for the colab runner's GPU backend. Hosted deploys never have one."""
    if "colab" not in settings.enabled_runners:
        return False, "disabled"
    if not (settings.remote_backend == "farmshare" or shutil.which("colab")):
        return False, "colab CLI not found on PATH"
    return True, "ok"


ATTEMPT_HEADERS = {"Cache-Control": "private, max-age=86400, immutable", "X-Content-Type-Options": "nosniff"}  # the job id is the only key: no shared caches


def create_app(settings=None):
    settings = settings or Settings()
    warnings = settings.validate()  # ConfigError (every bad variable at once) stops startup here
    logs.setup(settings.log_format, settings.log_level)
    for w in warnings:
        log.warning("config", extra={"fields": {"warning": w}})
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.live_ok = live.available(settings)  # imports checked once, in the worker's Python
    journal = Journal(settings.data_dir / "jobs.sqlite3")
    store = JobStore(settings, journal)

    @asynccontextmanager
    async def lifespan(app):
        store.start_background()
        log.info("started", extra={"fields": {"runners": settings.enabled_runners, "hosted": settings.hosted,
                                              "runs": len(all_runs())}})
        yield
        store.close()

    app = FastAPI(title="athlete2humanoid", version="1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.jobs = store
    app.state.streams = 0
    # outermost first: hardening sees every response (even 413/429), then limits, then CORS
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "Range", "X-A2H-Owner"], max_age=600,
                       expose_headers=["Content-Length", "Content-Range", "Accept-Ranges", "Retry-After", "X-Request-ID"])
    app.add_middleware(BodyLimit, limit=NON_UPLOAD_BODY, upload_limit=settings.max_upload_bytes + MULTIPART_SLACK)
    app.add_middleware(RateLimit, limiter=RateLimiter(), settings=settings)
    app.add_middleware(Hardening, proxy_hops=settings.proxy_hops)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        ref = request.scope.get("state", {}).get("request_id", "?")
        log.exception("unhandled error", extra={"fields": {"request_id": ref, "path": request.url.path}})
        return JSONResponse({"detail": f"internal error (ref {ref})"}, status_code=500)

    def all_runs():
        return runs.find_runs(settings.runs_dir, settings.run_exclude)

    def demo_run():
        found = all_runs()
        if settings.demo_run in found:
            return settings.demo_run
        with_video = [r for r, d in found.items() if (d / "comparison.mp4").is_file()]
        return (with_video or list(found) or [None])[0]

    def runners_now():
        """Enabled runners whose backend is reachable right now: demo and live (this host) always, colab when
        its CLI is there, modal when its app answers (checked every few minutes)."""
        return [r for r in settings.enabled_runners if r in ("demo", "live")
                or (r == "colab" and gpu_backend_status(settings)[0]) or (r == "modal" and store.train.available()[0])]

    def train_status():
        """What the page needs to offer "Train on a GPU" (None when this host can't): the gate and the queue."""
        if "modal" not in runners_now() or not settings.train_open:
            return None
        per_day = settings.train_per_day
        return {"backend": "modal", "code_required": bool(settings.train_code) and per_day <= 0,
                "code_accepted": bool(settings.train_code), "per_day": per_day,
                "left_today": max(0, per_day - store.train.today(cached=True)) if per_day > 0 else 0,
                "iterations": settings.train_iters, "attempt_every": settings.train_attempt_every,
                "minutes": store.train.eta_min(), "busy": store.train.active_count(cached=True) >= settings.train_concurrency}

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        """Liveness: the process answers. Nothing else is checked, so a restart loop can't start here."""
        return {"ok": True}

    @app.get("/readyz", include_in_schema=False)
    def readyz():
        """Readiness: runs to show, a writable data dir and a working journal. 503 lists what failed."""
        checks = {}
        checks["runs"] = len(all_runs()) > 0
        try:
            with tempfile.NamedTemporaryFile(dir=settings.data_dir):
                pass
            checks["data_dir_writable"] = True
        except OSError:
            checks["data_dir_writable"] = False
        try:
            checks["journal"] = journal.ping()
        except Exception:
            checks["journal"] = False
        try:
            checks["store"] = store.persist.ping()
        except Exception:
            checks["store"] = False
        gpu_ok, gpu_why = gpu_backend_status(settings)
        tr_ok, tr_why = store.train.available()
        ok = all(checks.values())
        body = {"ok": ok, "checks": checks, "gpu_backend": "ok" if gpu_ok else gpu_why,
                "train_backend": "ok" if tr_ok else tr_why, "active_jobs": store.active_count(),
                "trainings": store.train.active_count(cached=True)}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/api/health")
    def health():
        gpu_ok, gpu_why = gpu_backend_status(settings)
        tr_ok, tr_why = store.train.available()
        degraded = [] if gpu_ok or gpu_why == "disabled" else [f"colab runner unavailable: {gpu_why}"]
        if not tr_ok and tr_why != "disabled":
            degraded.append(f"modal runner unavailable: {tr_why}")
        return {
            "ok": True, "runners": runners_now(), "default_runner": "demo",
            "degraded": degraded,
            "demo_run": demo_run(), "demo_seconds": settings.demo_seconds, "run_count": len(all_runs()),
            "limits": {"max_upload_mb": settings.max_upload_mb, "extensions": sorted(VIDEO_EXT)},
            "stages": [{"key": k, "label": label, "description": d} for k, label, d, _ in STAGES],
            "live_stages": [{"key": k, "label": label, "description": d} for k, label, d, _ in LIVE_STAGES],
            "live": {"max_seconds": settings.live_max_seconds, "concurrency": settings.live_concurrency,
                     "pose": settings.live_pose}  # "modal": SAM 3.1 + SAM 3D Body on a GPU; "cpu": MediaPipe here
            if settings.live_ok else None,
            "train": train_status(),  # the modal runner: a live build trained on a GPU, when this host offers it
            # where finished builds and trainings are kept (server/store.py); durable: past this host's lifetime (a
            # bucket, or local files on a machine with a disk; not /tmp on a hosted instance)
            "store": {"kind": store.persist.kind, "durable": bool(store.persist.durable or not settings.hosted)},
        }

    @app.get("/api/runs")
    def list_runs():
        return [runs.summarize(settings.runs_dir, rid, d) for rid, d in all_runs().items()]

    @app.get("/api/runs/{rid}")
    def get_run(rid: str):
        d = all_runs().get(rid)
        if d is None:
            raise HTTPException(404, "run not found")
        return runs.summarize(settings.runs_dir, rid, d, detail=True, env_cache=settings.data_dir / "env")

    @app.get("/api/runs/{rid}/env.zip")
    def run_env(rid: str):
        """The run's training environment: one zip anyone with mjlab can train on (docs/TRAINING_ENV.md)."""
        d = all_runs().get(rid)
        p = runs.env_bundle(d, rid, settings.data_dir / "env") if d else None
        if p is None:
            raise HTTPException(404, "no training environment for this run")
        return FileResponse(p, media_type="application/zip", filename=f"{rid}.a2h-env.zip",
                            headers={"Cache-Control": "public, max-age=3600", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/runs/{rid}/attempts")
    def run_attempts(rid: str):
        """The policy's attempts recorded while it trained (a2h_env/attempts.py), for the "watch it learn" view."""
        d = all_runs().get(rid)
        view = train.attempts_view(d, lambda name: runs.file_url(rid, f"{train.ATTEMPTS}/{name}")) if d else None
        if view is None:
            raise HTTPException(404, "no attempts recorded for this run")
        return view

    @app.get("/api/runs/{rid}/task_spec")
    def run_task_spec(rid: str):
        """The run's task environment: key moments, flights, body parts and the retarget's self-check."""
        d = all_runs().get(rid)
        view = runs.task_spec(d) if d else None
        if view is None:
            raise HTTPException(404, "no task spec for this run")
        return view

    @app.get("/api/runs/{rid}/files/{rel:path}")
    def run_file(rid: str, rel: str, download: bool = False):
        d = all_runs().get(rid)
        p = runs.safe_path(d, rel) if d else None
        if p is None:
            raise HTTPException(404, "file not found")
        headers = {"Cache-Control": "public, max-age=3600", "X-Content-Type-Options": "nosniff"}
        # downloads are named by run and path, so three policies' policy.onnx don't collide
        name = "-".join(["a2h", rid, *Path(rel).parts]) if download else None
        return FileResponse(p, headers=headers, filename=name,
                            content_disposition_type="attachment" if download else "inline")

    @app.post("/api/jobs", status_code=201)
    async def create_job(request: Request, file: UploadFile | None = File(None), run_id: str | None = Form(None),
                         runner: str = Form("demo")):
        owner = anon(client_ip(request.scope, settings.proxy_hops))
        owner_key = persist.owner_key(request.headers)  # the browser's own key: what "My environments" lists by
        if runner == "modal":
            raise HTTPException(400, "the modal runner trains a finished live build: upload with runner=live, then "
                                     "POST /api/jobs/{id}/train")
        if runner not in settings.enabled_runners:
            raise HTTPException(400, f"runner '{runner}' is not enabled (available: {settings.enabled_runners})")
        fell_back = None
        if runner != "demo" and runner not in runners_now():
            # graceful degradation: the GPU backend is gone, so replay a recorded run and say so
            fell_back, runner = gpu_backend_status(settings)[1], "demo"
        if store.active_count() >= settings.max_active_jobs:
            raise HTTPException(429, "too many jobs in progress; try again in a minute")
        if store.active_count(owner) >= settings.max_jobs_per_ip:
            raise HTTPException(429, "you already have jobs running; wait for them to finish or cancel one")
        if file is None and not run_id:
            raise HTTPException(400, "upload a video or pick a gallery run")
        if run_id and run_id not in all_runs():
            raise HTTPException(404, "run not found")
        if runner in ("colab", "live") and file is None:
            raise HTTPException(400, f"the {runner} runner needs an uploaded video")
        if file is not None and store.upload_bytes() > settings.max_upload_storage_mb * 1048576:
            raise HTTPException(503, "upload storage is full right now; try again later or pick a gallery run")
        upload = await _save_upload(file, store, settings) if file is not None else None
        source = run_id or demo_run()
        if runner == "demo" and source is None:
            if upload:
                upload[0].unlink(missing_ok=True)
            raise HTTPException(503, "no finished runs available to replay")
        job = store.create(runner, source_run=source if runner == "demo" else None, upload=upload and upload[1],
                           owner=owner, owner_key=owner_key)
        if upload:
            tmp, meta = upload
            dest = store.upload_dir(job.id)
            dest.mkdir(parents=True, exist_ok=True)
            tmp.rename(dest / meta["stored"])
        if fell_back:
            store.log(job, "silhouette", f"GPU backend unavailable ({fell_back}); falling back to the demo runner")
        store.start(job)
        return store.snapshot(job.id)

    @app.get("/api/jobs")
    def list_jobs():
        """Recent jobs, for local debugging. Off when hosted: ids are the only key to someone's upload."""
        if not settings.job_list_enabled:
            raise HTTPException(404, "not found")
        with store.lock:
            js = sorted(store.jobs.values(), key=lambda j: j.created_at, reverse=True)[:20]
            return [{k: v for k, v in j.public().items() if k not in ("logs", "curves", "input")} for j in js]

    @app.get("/api/my/jobs")
    def my_jobs(request: Request):
        """This visitor's builds and trainings (the X-A2H-Owner key the browser made), kept past restarts by the store."""
        key = persist.owner_key(request.headers)
        if not key:
            raise HTTPException(400, "send your key as X-A2H-Owner (32 hex characters)")
        out = []
        for r in store.mine(key)[:200]:
            jid = r["id"]
            kept = r.get("kept") or []
            env = "env.zip" in kept or runs.safe_path(store.upload_dir(jid) / live.LIVE_DIR, runs.ENV_ZIP) is not None
            tr = r.get("training") or None
            out.append({"id": jid, "runner": r.get("runner"), "status": r.get("status"), "created_at": r.get("created_at"),
                        "updated_at": r.get("updated_at"), "filename": r.get("filename"), "summary": r.get("summary"),
                        "duration_s": r.get("duration_s"), "error": r.get("error"), "kept": bool(kept),
                        "env": {"url": f"/api/jobs/{jid}/env.zip", "filename": live.env_filename(jid), "size": None} if env else None,
                        "training": {k: tr.get(k) for k in ("state", "iterations", "attempt_every", "attempts", "iteration", "reward", "checkpoint")} if tr else None})
        return {"owner": key, "jobs": out}

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        snap = store.snapshot(jid)
        if snap is None:
            raise HTTPException(404, "job not found")
        return snap

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str, request: Request):
        job = store.get(jid)
        if job is None:
            raise HTTPException(404, "job not found")
        # Only whoever started it: the browser's key when it sent one, else the same client. An id alone reads a job.
        key = store.owner_key.get(jid)
        if key:
            mine = persist.owner_key(request.headers) == key
        else:
            mine = store.owner.get(jid) == anon(client_ip(request.scope, settings.proxy_hops))
        if not mine:
            raise HTTPException(403, "only whoever started this job can cancel it")
        if job.status not in TERMINAL:
            store.cancel[jid].set()
        return store.snapshot(jid)

    @app.get("/api/jobs/{jid}/events")
    async def job_events(jid: str, request: Request):
        """Server-sent events: the full job on every change, until it finishes."""
        if await run_in_threadpool(store.get, jid) is None:  # may read the store (a kept job): off the event loop
            raise HTTPException(404, "job not found")
        if app.state.streams >= settings.max_streams:
            # the client falls back to polling GET /api/jobs/{id}
            raise HTTPException(503, "too many live streams; poll instead", headers={"Retry-After": "5"})

        async def stream():
            app.state.streams += 1
            try:
                async for chunk in _events():
                    yield chunk
            finally:
                app.state.streams -= 1

        async def _events():
            last = -1
            while True:
                job = store.peek(jid)  # in memory now (just revived if need be); never the store from the loop
                if job is None or await request.is_disconnected():
                    return
                if job.version != last:
                    last = job.version
                    yield job_event(store.snapshot(jid))
                    if job.status in TERMINAL:
                        return
                else:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.25)
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/jobs/{jid}/train")
    def train_job(jid: str, request: Request, code: str = Form("")):
        """Train a finished live build's environment on a GPU (the modal runner, server/train.py). Gated: the invite
        code (A2H_TRAIN_CODE), or one of the day's free trainings (A2H_TRAIN_PER_DAY); one at a time by default."""
        job = store.get(jid)
        if job is None or job.runner != "live":
            raise HTTPException(404, "not a live job")
        if "modal" not in settings.enabled_runners or not settings.train_open:
            raise HTTPException(503, "this host can't train policies (no GPU backend)")
        if not store.train.available()[0]:
            raise HTTPException(503, "the GPU backend isn't reachable right now; try again in a few minutes")
        code = (code or "").strip()[:200]
        with_code = bool(settings.train_code and code)
        if with_code and not secrets.compare_digest(code.encode(), settings.train_code.encode()):
            raise HTTPException(403, "that invite code isn't right")
        if not with_code and settings.train_per_day <= 0:
            raise HTTPException(403, "training needs an invite code")
        with store.train.lock:  # the checks that spend: one per job, the day's allowance, the queue, all under one lock
            state = (job.training or {}).get("state")
            if state in ("starting", "running"):
                raise HTTPException(409, "this build's policy is training already")
            if state == "done":
                raise HTTPException(409, "this build's policy is trained; download it below")
            if job.status != "succeeded" or runs.safe_path(store.live_dir(job), runs.ENV_ZIP) is None:
                raise HTTPException(409, "the environment isn't built yet; training starts from a finished build")
            if not with_code and store.train.today() >= settings.train_per_day:
                raise HTTPException(403, f"today's {settings.train_per_day} free training{'s are' if settings.train_per_day != 1 else ' is'} "
                                         f"used up" + ("; an invite code starts one" if settings.train_code else ""))
            if store.train.active_count() >= settings.train_concurrency:
                raise HTTPException(429, f"another visitor's policy is training right now (about {store.train.eta_min()} min); "
                                         f"try again when it's done", headers={"Retry-After": "600"})
            store.train.start(job)
        log.info("training started", extra={"fields": {"job": jid, "client": anon(client_ip(request.scope, settings.proxy_hops)),
                                                       "with_code": bool(code)}})
        return store.snapshot(jid)

    @app.get("/api/jobs/{jid}/attempts")
    def job_attempts(jid: str):
        """The attempts mirrored so far from the GPU (every A2H_TRAIN_ATTEMPT_EVERY iterations), newest last."""
        job = store.get(jid)
        if job is None or job.runner != "live" or job.training is None:
            raise HTTPException(404, "this job isn't training a policy")
        view = train.attempts_view(store.train.train_dir(job), lambda name: f"/api/jobs/{jid}/attempts/{name}")
        if view is None:
            view = {"iterations": job.training.get("iterations"), "attempt_every": job.training.get("attempt_every"),
                    "fps": 50, "task": None, "policy": None, "attempts": [], "curve": job.curves.get("policy", [])}
        view["training"] = job.training
        return view

    @app.get("/api/jobs/{jid}/attempts/{name}")
    def job_attempt(jid: str, name: str):
        """One attempt: iter_NNNN.npz (qpos (T, 36) float32 at 50 Hz, reward, stages, fell)."""
        job = store.get(jid)
        if job is None or job.runner != "live" or job.training is None or not train.NPZ.fullmatch(name):
            raise HTTPException(404, "no such attempt")
        p = runs.safe_path(store.train.train_dir(job) / train.ATTEMPTS, name)
        if p is None:
            raise HTTPException(404, "no such attempt")
        return FileResponse(p, media_type="application/octet-stream", headers=ATTEMPT_HEADERS)

    @app.get("/api/jobs/{jid}/policy.pt")
    def job_policy(jid: str):
        """The trained policy's checkpoint (rsl_rl format), once training is done."""
        job = store.get(jid)
        if job is None or job.runner != "live" or not (job.training or {}).get("checkpoint"):
            raise HTTPException(404, "no trained policy")
        p = runs.safe_path(store.train.train_dir(job), "policy.pt")
        if p is None:
            raise HTTPException(404, "no trained policy")
        return FileResponse(p, media_type="application/octet-stream", filename=f"live-{jid[:8]}-policy.pt",
                            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/jobs/{jid}/task_spec")
    def job_task_spec(jid: str):
        """A job's task environment, as soon as it exists (long before the policy is trained)."""
        job = store.get(jid)
        if job is None:
            raise HTTPException(404, "job not found")
        view = runs.task_spec(store.spec_folder(job))
        if view is None:
            raise HTTPException(404, "the task environment isn't compiled yet")
        return view

    def _live_folder(jid):
        job = store.get(jid)
        if job is None or job.runner != "live":
            raise HTTPException(404, "not a live job")
        return store.live_dir(job)

    @app.get("/api/jobs/{jid}/live")
    def job_live(jid: str):
        """What the live page draws (2D landmarks, the G1's bodies, timings), growing as stages finish."""
        data = live.live_data(_live_folder(jid))
        if data is None:
            raise HTTPException(404, "no pose yet")
        return data

    @app.get("/api/jobs/{jid}/live/video")
    def job_live_video(jid: str):
        """The re-encoded clip the landmarks line up with (H.264, no audio or metadata)."""
        p = runs.safe_path(_live_folder(jid), "processed.mp4")
        if p is None:
            raise HTTPException(404, "the video isn't processed yet")
        return FileResponse(p, media_type="video/mp4", headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/api/jobs/{jid}/live/mask")
    def job_live_mask(jid: str):
        """The processed clip with SAM 3.1's mask tinted over the person, when the pose ran on the GPU."""
        p = runs.safe_path(_live_folder(jid), live.MASK_VIDEO)
        if p is None:
            raise HTTPException(404, "this build has no mask video")
        return FileResponse(p, media_type="video/mp4", headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/api/jobs/{jid}/env.zip")
    def job_env(jid: str):
        """A live build's training environment, once its variants stage has written it (server/live_worker.py)."""
        p = runs.safe_path(_live_folder(jid), runs.ENV_ZIP)
        if p is None or p.stat().st_size > live.MAX_ENV_BYTES:
            raise HTTPException(404, "the training environment isn't built yet")
        return FileResponse(p, media_type="application/zip", filename=live.env_filename(jid),
                            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.get("/api/jobs/{jid}/input")
    def job_input(jid: str):
        job = store.get(jid)
        if job is None or not job.input:
            raise HTTPException(404, "no input video")
        p = store.upload_dir(jid) / job.input["stored"]
        if not p.is_file():
            raise HTTPException(404, "no input video")
        return FileResponse(p, media_type=job.input["content_type"], headers={"X-Content-Type-Options": "nosniff"})

    if settings.static_dir and Path(settings.static_dir, "index.html").is_file():
        static = Path(settings.static_dir)
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/{full:path}", include_in_schema=False)
        def spa(full: str):
            p = (static / full).resolve()
            if full and p.is_file() and static.resolve() in p.parents:
                return FileResponse(p)
            return FileResponse(static / "index.html")

    return app


async def _save_upload(file, store, settings):
    """Stream to disk, enforcing type and size. Returns (temp path, metadata)."""
    name = Path(file.filename or "").name
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(415, f"unsupported file type '{ext or '?'}'; use {', '.join(sorted(VIDEO_EXT))}")
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype not in VIDEO_TYPES and not ctype.startswith("video/"):
        raise HTTPException(415, f"unsupported content type '{ctype}'")
    tmp_dir = settings.data_dir / "uploads" / "_incoming"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{uuid.uuid4().hex}{ext}"  # never derived from the client's filename
    size, head, limit = 0, b"", settings.max_upload_bytes
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1 << 20):
                if not head:
                    head = chunk[:16]
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"file is larger than {settings.max_upload_mb:g} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "empty file")
        if not sniff_video(head, ext):
            raise HTTPException(415, "that file doesn't look like a video")
    except BaseException:  # rejected, disconnected or cancelled: never leave a partial file
        tmp.unlink(missing_ok=True)
        raise
    safe = _display_name(name)
    ctype = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}[ext]
    return tmp, {"filename": safe, "size": size, "stored": f"input{ext}", "content_type": ctype}


def _display_name(name):
    """The name shown back to the user: letters, digits and ._- only, no leading dots, bounded length."""
    # \w is Unicode-aware but excludes control and bidi-override characters
    return re.sub(r"[^\w.\- ]", "_", name).strip(" .")[:120] or "upload"


app = create_app()  # uvicorn imports server.main:app: a bad config stops startup here, listing every problem
