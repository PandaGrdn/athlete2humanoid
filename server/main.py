"""athlete2humanoid web API.

    uvicorn server.main:app --reload --port 8000

Runs (finished pipeline outputs) are read from A2H_RUNS_DIR; jobs live in memory. See
server/config.py for every setting and docs/DEPLOY.md for hosting.
"""

import asyncio
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import runs
from .config import Settings
from .jobs import TERMINAL, JobStore, job_event
from .stages import STAGES

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-m4v", "video/webm", "application/octet-stream", ""}


def sniff_video(head):
    """True for ISO-BMFF (mp4/mov/m4v: 'ftyp' at byte 4) and WebM/Matroska (EBML magic)."""
    return head[4:8] in (b"ftyp", b"moov", b"wide", b"mdat", b"free") or head[:4] == b"\x1a\x45\xdf\xa3"


def create_app(settings=None):
    settings = settings or Settings()
    app = FastAPI(title="athlete2humanoid", version="1.0")
    app.state.settings = settings
    app.state.jobs = store = JobStore(settings)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"], expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"])

    def all_runs():
        return runs.find_runs(settings.runs_dir, settings.run_exclude)

    def demo_run():
        found = all_runs()
        if settings.demo_run in found:
            return settings.demo_run
        with_video = [r for r, d in found.items() if (d / "comparison.mp4").is_file()]
        return (with_video or list(found) or [None])[0]

    @app.get("/api/health")
    def health():
        return {
            "ok": True, "runners": settings.enabled_runners, "default_runner": "demo",
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
        if runner not in settings.enabled_runners:
            raise HTTPException(400, f"runner '{runner}' is not enabled (available: {settings.enabled_runners})")
        if store.active_count() >= settings.max_active_jobs:
            raise HTTPException(429, "too many jobs in progress; try again in a minute")
        if file is None and not run_id:
            raise HTTPException(400, "upload a video or pick a gallery run")
        if run_id and run_id not in all_runs():
            raise HTTPException(404, "run not found")
        if runner == "colab" and file is None:
            raise HTTPException(400, "the colab runner needs an uploaded video")
        upload = await _save_upload(file, store, settings) if file is not None else None
        source = run_id or demo_run()
        if runner == "demo" and source is None:
            raise HTTPException(503, "no finished runs available to replay")
        job = store.create(runner, source_run=source if runner == "demo" else None, upload=upload and upload[1])
        if upload:
            tmp, meta = upload
            dest = store.upload_dir(job.id)
            dest.mkdir(parents=True, exist_ok=True)
            tmp.rename(dest / meta["stored"])
        store.start(job)
        return store.snapshot(job.id)

    @app.get("/api/jobs")
    def list_jobs():
        with store.lock:
            js = sorted(store.jobs.values(), key=lambda j: j.created_at, reverse=True)[:20]
            return [{k: v for k, v in j.public().items() if k not in ("logs", "curves")} for j in js]

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

        async def stream():
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
    tmp = tmp_dir / f"{id(file):x}-{abs(hash(name)):x}{ext}"
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
        if not sniff_video(head):
            raise HTTPException(415, "that file doesn't look like a video")
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    safe = re.sub(r"[^\w.\- ]", "_", name)[:120]
    ctype = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}[ext]
    return tmp, {"filename": safe, "size": size, "stored": f"input{ext}", "content_type": ctype}


app = create_app()
