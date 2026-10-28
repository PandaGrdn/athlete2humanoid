"""Jobs: one video through the pipeline, and the runners that execute them.

demo   replays a finished run (its real log lines and training curves) on a compressed clock.
       Free hosting has no GPU, so this is what the hosted demo uses. Results are that run's.
colab  shells out to scripts/remote_run.py (a real A100 job, hours long) and follows its
       runner.log / job.log / status.json. Local only, off unless A2H_ENABLE_COLAB=1.
"""

import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from . import runs
from .config import REPO
from .stages import STAGES, STAGE_KEYS, classify, redact

log = logging.getLogger("a2h.jobs")
TERMINAL = {"succeeded", "failed", "cancelled"}
MAX_LOG = 400
KEEP_JOBS = 200  # terminal jobs kept (newest), whatever their age


def _now():
    return time.time()


@dataclass
class Stage:
    key: str
    label: str
    description: str
    status: str = "pending"  # pending | running | done | failed
    progress: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None


@dataclass
class Job:
    id: str
    runner: str
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    stage: str | None = None
    progress: float = 0.0
    stages: list = field(default_factory=list)
    logs: list = field(default_factory=list)  # [[unix time, stage, line]]
    curves: dict = field(default_factory=dict)  # policy -> [[iteration, mean reward]] seen so far
    input: dict | None = None  # {filename, size, url} of an upload
    source_run: str | None = None  # the run a demo job replays
    result_run: str | None = None  # the run to show when it's done
    error: str | None = None
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    version: int = 0

    def public(self):
        d = asdict(self)
        d["demo"] = self.runner == "demo"
        return d


def _job_from_dict(d):
    names = {f.name for f in fields(Job)}
    job = Job(**{k: v for k, v in d.items() if k in names and k != "stages"})
    job.stages = [Stage(**{k: v for k, v in st.items() if k in Stage.__dataclass_fields__}) for st in d.get("stages") or []]
    return job


class JobStore:
    """Jobs in memory, mirrored to a Journal (when given) so they survive a restart.

    Log lines arrive many times a second, so changes mark a job dirty and a background thread writes
    dirty jobs once a second; status changes (create, finish) are written at once. The same thread
    expires uploads (A2H_UPLOAD_TTL_H after their job ends) and jobs (A2H_JOB_TTL_H).
    """

    def __init__(self, settings, journal=None):
        self.settings = settings
        self.jobs = {}
        self.owner = {}  # job id -> client pseudonym (never sent to clients)
        self.lock = threading.RLock()
        self.cancel = {}
        self.journal = journal
        self.dirty = set()
        self.stop = threading.Event()
        (settings.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        if journal:
            self._recover()
        self._sweep_incoming(max_age=0)  # half-written uploads from before a restart
        self._flusher = None

    def start_background(self, interval=1.0):
        """The flush + expiry thread. Separate from __init__ so tests can drive sweep() themselves."""
        if self._flusher is None:
            self._flusher = threading.Thread(target=self._background, args=(interval,), daemon=True, name="jobs-flush")
            self._flusher.start()

    def close(self):
        self.stop.set()
        for ev in list(self.cancel.values()):
            ev.set()
        self.flush()

    # --- persistence -------------------------------------------------------------------------
    def _recover(self):
        """Load the journal. A job that was queued or running when the process died can't be resumed
        (its runner thread is gone), so it is marked failed with a reason the UI shows."""
        n = 0
        for owner, d in self.journal.load():
            try:
                job = _job_from_dict(d)
            except TypeError:
                continue
            self.jobs[job.id], self.owner[job.id] = job, owner
            self.cancel[job.id] = threading.Event()
            if job.status not in TERMINAL:
                msg = "the server restarted while this job was running; start it again"
                if job.runner == "colab":
                    msg += " (the remote GPU job may still be running: see its runner.log under outputs/)"
                self.finish(job, "failed", error=msg)
                n += 1
        if self.jobs:
            log.info("jobs recovered", extra={"fields": {"jobs": len(self.jobs), "interrupted": n}})

    def _row(self, job):
        return (job.id, job.status, job.created_at, job.updated_at, self.owner.get(job.id), asdict(job))

    def flush(self):
        if not self.journal:
            return
        with self.lock:
            rows = [self._row(self.jobs[j]) for j in self.dirty if j in self.jobs]
            self.dirty.clear()
        try:
            self.journal.save(rows)
        except Exception:
            log.exception("journal write failed")
            with self.lock:
                self.dirty.update(r[0] for r in rows)

    def _background(self, interval):
        last_sweep = 0.0
        while not self.stop.wait(interval):
            self.flush()
            if time.monotonic() - last_sweep > 60:
                last_sweep = time.monotonic()
                try:
                    self.sweep()
                except Exception:
                    log.exception("sweep failed")

    # --- expiry ------------------------------------------------------------------------------
    def sweep(self, now=None):
        """Delete expired uploads and jobs, and stray files no job owns. Returns (uploads, jobs) removed."""
        now = now or _now()
        s = self.settings
        gone_uploads, gone_jobs = 0, []
        with self.lock:
            for j in list(self.jobs.values()):
                if j.status not in TERMINAL:
                    continue
                age = now - j.updated_at
                if age > s.job_ttl_h * 3600:
                    gone_jobs.append(j.id)
                elif age > s.upload_ttl_h * 3600 and self.upload_dir(j.id).exists():
                    shutil.rmtree(self.upload_dir(j.id), ignore_errors=True)
                    gone_uploads += 1
            for jid in gone_jobs:
                self._forget(jid)
            known = set(self.jobs)
        for d in (s.data_dir / "uploads").iterdir():
            if d.is_dir() and d.name != "_incoming" and d.name not in known:
                shutil.rmtree(d, ignore_errors=True)
                gone_uploads += 1
        self._sweep_incoming(max_age=3600)
        if gone_jobs and self.journal:
            self.journal.delete(gone_jobs)
        if gone_uploads or gone_jobs:
            log.info("expired", extra={"fields": {"uploads": gone_uploads, "jobs": len(gone_jobs)}})
        return gone_uploads, len(gone_jobs)

    def _sweep_incoming(self, max_age):
        inc = self.settings.data_dir / "uploads" / "_incoming"
        if inc.is_dir():
            for f in inc.iterdir():
                try:
                    if _now() - f.stat().st_mtime >= max_age:
                        f.unlink()
                except OSError:
                    pass

    def upload_bytes(self):
        root = self.settings.data_dir / "uploads"
        return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())

    def _forget(self, jid):
        shutil.rmtree(self.upload_dir(jid), ignore_errors=True)
        self.jobs.pop(jid, None)
        self.cancel.pop(jid, None)
        self.owner.pop(jid, None)
        self.dirty.discard(jid)

    # --- bookkeeping -------------------------------------------------------------------------
    def create(self, runner, source_run=None, upload=None, owner=None):
        jid = uuid.uuid4().hex  # 128 bits: a job id is the only key to its upload
        job = Job(id=jid, runner=runner, source_run=source_run,
                  stages=[Stage(k, label, desc) for k, label, desc, _ in STAGES])
        if upload:
            job.input = {**upload, "url": f"/api/jobs/{jid}/input"}
        with self.lock:
            self.jobs[jid] = job
            self.owner[jid] = owner
            self.cancel[jid] = threading.Event()
            self._prune()
            self.dirty.add(jid)
        self.flush()
        log.info("job created", extra={"fields": {"job": jid, "runner": runner, "source_run": source_run,
                                                  "upload_bytes": upload and upload.get("size"), "client": owner}})
        return job

    def active_count(self, owner=None):
        with self.lock:
            return sum(j.status not in TERMINAL and (owner is None or self.owner.get(j.id) == owner)
                       for j in self.jobs.values())

    def _prune(self, keep=KEEP_JOBS):
        done = sorted((j for j in self.jobs.values() if j.status in TERMINAL), key=lambda j: j.created_at)
        drop = [j.id for j in done[:max(0, len(done) - keep)]]
        for jid in drop:
            self._forget(jid)
        if drop and self.journal:
            self.journal.delete(drop)

    def upload_dir(self, jid):
        return self.settings.data_dir / "uploads" / jid

    @staticmethod
    def colab_out(job):
        """Where remote_run.py writes a colab job's results (its tag is derived from the job id)."""
        return REPO / "outputs" / f"remote_latent_web{job.id[:6]}"

    def spec_folder(self, job):
        """The folder whose task_spec.json is this job's task environment, once the job has one.

        demo: the replayed run's, from the moment the replay reaches the environment stage.
        colab: the job's output folder (the spec appears there when remote_run.py brings it back).
        """
        if job.result_run:
            done = runs.find_runs(self.settings.runs_dir).get(job.result_run)
            if done is not None:
                return done
        if job.runner == "demo":
            env = next((s for s in job.stages if s.key == "environment"), None)
            if env is None or env.status == "pending":
                return None
            return runs.find_runs(self.settings.runs_dir).get(job.source_run)
        out = self.colab_out(job)
        return out if out.is_dir() else None

    def get(self, jid):
        with self.lock:
            return self.jobs.get(jid)

    def snapshot(self, jid):
        with self.lock:
            j = self.jobs.get(jid)
            return j.public() if j else None

    def update(self, job, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(job, k, v)
            self._recompute(job)
        if "status" in kw:
            self.flush()

    def log(self, job, stage, line):
        with self.lock:
            job.logs.append([_now(), stage, redact(line)])
            del job.logs[:-MAX_LOG]
            self._recompute(job)

    def set_stage(self, job, key, progress=None, status=None):
        with self.lock:
            i = STAGE_KEYS.index(key)
            for s in job.stages[:i]:  # stages are sequential: everything before is done
                if s.status != "done":
                    s.status, s.progress, s.finished_at = "done", 1.0, s.finished_at or _now()
                    s.started_at = s.started_at or s.finished_at
            s = job.stages[i]
            if s.status == "pending":
                s.status, s.started_at = "running", _now()
            if progress is not None:
                s.progress = max(s.progress, min(1.0, progress))
            # A line can arrive after its stage is done (the task spec is logged between two latent-reward
            # steps): it is logged under its own stage, but the job doesn't step back to it.
            late = s.status == "done" and status is None
            if status:
                s.status = status
                if status == "done":
                    s.progress, s.finished_at = 1.0, _now()
            if not late or job.stage is None:
                job.stage = key
            self._recompute(job)

    def _recompute(self, job):
        w = {k: share for k, _, _, share in STAGES}
        job.progress = round(sum(w[s.key] * (1.0 if s.status == "done" else s.progress) for s in job.stages), 4)
        job.updated_at = _now()
        job.version += 1
        self.dirty.add(job.id)

    def finish(self, job, status, error=None, result_run=None):
        with self.lock:
            if status == "succeeded":
                for s in job.stages:
                    if s.status != "done":
                        s.status, s.progress, s.finished_at = "done", 1.0, _now()
                        s.started_at = s.started_at or s.finished_at
            else:
                for s in job.stages:
                    if s.status == "running":
                        s.status = "failed" if status == "failed" else "pending"
            job.status, job.error = status, error
            job.result_run = result_run or job.result_run
            self._recompute(job)
        self.flush()
        log.info("job finished", extra={"fields": {"job": job.id, "runner": job.runner, "status": status,
                                                   "seconds": round(job.updated_at - job.created_at, 1)}})

    # --- running -----------------------------------------------------------------------------
    def start(self, job):
        target = {"demo": self._run_demo, "colab": self._run_colab}[job.runner]

        def go():
            try:
                self.update(job, status="running")
                target(job)
            except Exception:  # a runner bug must not leave the job "running" forever
                ref = uuid.uuid4().hex[:8]
                log.exception("runner crashed", extra={"fields": {"job": job.id, "ref": ref}})
                # the traceback (paths, internals) stays in the server log; the client gets a reference
                self.finish(job, "failed", error=f"internal error in the {job.runner} runner (ref {ref})")
        threading.Thread(target=go, daemon=True, name=f"job-{job.id}").start()

    def _sleep(self, job, seconds):
        """False when the job was cancelled meanwhile."""
        return not self.cancel[job.id].wait(max(0.0, seconds))

    def _run_demo(self, job):
        folder = runs.find_runs(self.settings.runs_dir).get(job.source_run)
        if folder is None:
            return self.finish(job, "failed", error=f"run '{job.source_run}' not found")
        lines = runs.replay_lines(folder)
        detail = runs.summarize(self.settings.runs_dir, job.source_run, folder, detail=True)
        curves = {p["key"]: p["curve"] for p in detail["policies"] if not p["previous"] and p["curve"]}
        total = self.settings.demo_seconds
        if job.input:
            self.log(job, "silhouette", f"received {job.input['filename']} ({job.input['size'] / 1e6:.1f} MB)")
        self.log(job, "silhouette", f"demo runner: replaying the recorded run '{job.source_run}' (hosted demo has no GPU)")
        for key, label, _, share in STAGES:
            budget = total * share
            fallback = "no task spec in this run (it predates the task environment)" if key == "environment" \
                and runs.task_spec(folder) is None else f"{label}: done"
            mine = [t for s, t in lines if s == key] or [fallback]
            if key == "train" and curves:
                if not self._replay_training(job, budget, mine, curves):
                    return self.finish(job, "cancelled")
                continue
            self.set_stage(job, key, 0.0)
            # Long stages (hundreds of lines) show an even sample; the last line always shows.
            step = max(1, len(mine) // 40)
            shown = mine[::step][:-1] + mine[-1:] if len(mine) > 1 else mine
            for i, text in enumerate(shown):
                if not self._sleep(job, budget / (len(shown) + 1)):
                    return self.finish(job, "cancelled")
                self.log(job, key, text)
                self.set_stage(job, key, (i + 1) / len(shown))
            self.set_stage(job, key, status="done")
        self.finish(job, "succeeded", result_run=job.source_run)

    def _replay_training(self, job, budget, lines, curves):
        self.set_stage(job, "train", 0.0)
        for text in [t for t in lines if t.startswith("training ")]:
            self.log(job, "train", text)
        ticks = max(len(c) for c in curves.values())
        last_logged = -1
        for i in range(ticks):
            if not self._sleep(job, budget / ticks):
                return False
            with self.lock:
                job.curves = {k: c[:int(round((i + 1) / ticks * len(c)))] for k, c in curves.items()}
            frac = (i + 1) / ticks
            if int(frac * 8) != last_logged:  # a status line every eighth of training
                last_logged = int(frac * 8)
                parts = [f"{k} {c[-1][1]:.2f}" for k, c in job.curves.items() if c]
                it = max((c[-1][0] for c in job.curves.values() if c), default=0)
                self.log(job, "train", f"iteration {it + 1}: mean reward " + ", ".join(parts))
            self.set_stage(job, "train", frac)
        for text in [t for t in lines if "finished with code" in t]:
            self.log(job, "train", text)
        self.set_stage(job, "train", status="done")
        return True

    def _run_colab(self, job):
        """A real run: copies the upload into clips/, launches remote_run.py, follows its files."""
        if not job.input:
            return self.finish(job, "failed", error="the colab runner needs an uploaded video")
        src = self.upload_dir(job.id) / job.input["stored"]
        tag = f"web{job.id[:6]}"
        # prefixed with the tag: an upload named squat.mp4 must not overwrite clips/squat.mp4
        stem = tag + "_" + (re.sub(r"[^a-z0-9_]", "_", Path(job.input["filename"]).stem.lower())[:40].strip("_") or "clip")
        clip = REPO / "clips" / f"{stem}{src.suffix.lower()}"
        shutil.copyfile(src, clip)
        out = self.colab_out(job)
        cmd = [sys.executable, str(REPO / "scripts" / "remote_run.py"), "latent", "--target", stem,
               "--tag", tag, "--min-clips", "1", "--iters", str(self.settings.colab_iters),
               "--backend", self.settings.remote_backend, "--args", "--policies " + " ".join(self.settings.colab_policies)]
        if self.settings.colab_init and Path(self.settings.colab_init).is_dir():
            cmd += ["--init-weights", self.settings.colab_init]
        self.log(job, "silhouette", f"{self.settings.remote_backend} runner: remote_run.py {' '.join(cmd[3:])}")
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        seen = {"runner.log": 0, "job.log": 0}
        deadline = time.monotonic() + self.settings.job_timeout_h * 3600
        while True:
            for name in seen:
                f = out / name
                if f.is_file():
                    with open(f, errors="ignore") as fh:
                        fh.seek(seen[name])
                        chunk = fh.read()
                        seen[name] = fh.tell()
                    for raw in re.split(r"[\r\n]+", chunk):
                        stage, text = classify(raw)
                        if name == "runner.log" and text:
                            self.log(job, job.stage or "silhouette", text[:300])
                        elif stage:
                            self.set_stage(job, stage)
                            if "Learning iteration" not in text:
                                self.log(job, stage, text[:300])
            if proc.poll() is not None:
                break
            if self.cancel[job.id].wait(5):
                proc.terminate()
                return self.finish(job, "cancelled")
            if time.monotonic() > deadline:
                proc.terminate()
                return self.finish(job, "failed", error=f"timed out after {self.settings.job_timeout_h:g} h")
        state = (runs.read_json(out / "status.json") or {}).get("state")
        if (out / "eval.json").is_file():
            return self.finish(job, "succeeded", result_run=runs.run_id(out))
        self.finish(job, "failed", error=f"remote_run.py exited {proc.returncode}, job state {state}")


def job_event(snapshot):
    return f"data: {json.dumps(snapshot)}\n\n"
