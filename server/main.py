"""athlete2humanoid web API.

    uvicorn server.main:app --reload --port 8000

Runs (finished pipeline outputs) are read from A2H_RUNS_DIR; jobs live in memory and in a SQLite
journal under A2H_DATA_DIR. See server/config.py for every setting, docs/DEPLOY.md for hosting,
docs/SECURITY.md for the limits and docs/SYSTEM_DESIGN.md for the job lifecycle.
"""

import asyncio
import logging
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import logs, runs
from .config import Settings
from .journal import Journal
from .jobs import TERMINAL, JobStore, job_event
from .security import BodyLimit, Hardening, RateLimit, RateLimiter, anon, client_ip
from .stages import STAGES

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
    """(available, reason) for the runner that needs a GPU backend. Hosted deploys never have one."""
    if "colab" not in settings.enabled_runners:
        return False, "disabled"
    if not (settings.remote_backend == "farmshare" or shutil.which("colab")):
        return False, "colab CLI not found on PATH"
    return True, "ok"


def create_app(settings=None):
    settings = settings or Settings()
    warnings = settings.validate()  # ConfigError (every bad variable at once) stops startup here
    logs.setup(settings.log_format, settings.log_level)
    for w in warnings:
        log.warning("config", extra={"fields": {"warning": w}})
    settings.data_dir.mkdir(parents=True, exist_ok=True)
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
                       allow_headers=["Content-Type", "Range"], max_age=600,
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
        """Enabled runners whose backend is reachable right now: demo always, colab when its CLI is there."""
        return [r for r in settings.enabled_runners if r == "demo" or gpu_backend_status(settings)[0]]

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
        gpu_ok, gpu_why = gpu_backend_status(settings)
        ok = all(checks.values())
        body = {"ok": ok, "checks": checks, "gpu_backend": "ok" if gpu_ok else gpu_why,
                "active_jobs": store.active_count()}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/api/health")
    def health():
        gpu_ok, gpu_why = gpu_backend_status(settings)
        return {
            "ok": True, "runners": runners_now(), "default_runner": "demo",
            "degraded": [] if gpu_ok or gpu_why == "disabled" else [f"colab runner unavailable: {gpu_why}"],
            "demo_run": demo_run(), "demo_seconds": settings.demo_seconds, "run_count": len(all_runs()),
            "limits": {"max_upload_mb": settings.max_upload_mb, "extensions": sorted(VIDEO_EXT)},
            "stages": [{"key": k, "label": label, "description": d} for k, label, d, _ in STAGES],
        }

    @app.get("/api/runs")
    def list_runs():
        return [runs.summarize(settings.runs_dir, rid, d) for rid, d in all_runs().items()]

    @app.get("/api/runs/{rid}")
    def get_run(rid: str):
        d = all_runs().get(rid)
        if d is None:
            raise HTTPException(404, "run not found")
        return runs.summarize(settings.runs_dir, rid, d, detail=True)

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
        if runner == "colab" and file is None:
            raise HTTPException(400, "the colab runner needs an uploaded video")
        if file is not None and store.upload_bytes() > settings.max_upload_storage_mb * 1048576:
            raise HTTPException(503, "upload storage is full right now; try again later or pick a gallery run")
        upload = await _save_upload(file, store, settings) if file is not None else None
        source = run_id or demo_run()
        if runner == "demo" and source is None:
            if upload:
                upload[0].unlink(missing_ok=True)
            raise HTTPException(503, "no finished runs available to replay")
        job = store.create(runner, source_run=source if runner == "demo" else None, upload=upload and upload[1],
                           owner=owner)
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

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        snap = store.snapshot(jid)
        if snap is None:
            raise HTTPException(404, "job not found")
        return snap

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        job = store.get(jid)
        if job is None:
            raise HTTPException(404, "job not found")
        if job.status not in TERMINAL:
            store.cancel[jid].set()
        return store.snapshot(jid)

    @app.get("/api/jobs/{jid}/events")
    async def job_events(jid: str, request: Request):
        """Server-sent events: the full job on every change, until it finishes."""
        if store.get(jid) is None:
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
                job = store.get(jid)
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
