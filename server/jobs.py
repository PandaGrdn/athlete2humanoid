"""Jobs: one video through the pipeline, and the runners that execute them.

demo   replays a finished run (its real log lines and training curves) on a compressed clock.
       Free hosting has no GPU, so this is what the hosted demo uses. Results are that run's.
colab  shells out to scripts/remote_run.py (a real A100 job, hours long) and follows its
       runner.log / job.log / status.json. Local only, off unless A2H_ENABLE_COLAB=1.
"""

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import runs
from .config import REPO
from .stages import STAGES, STAGE_KEYS, classify, redact

TERMINAL = {"succeeded", "failed", "cancelled"}
MAX_LOG = 400


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


class JobStore:
    def __init__(self, settings):
        self.settings = settings
        self.jobs = {}
        self.lock = threading.RLock()
        self.cancel = {}
        (settings.data_dir / "uploads").mkdir(parents=True, exist_ok=True)

    # --- bookkeeping -------------------------------------------------------------------------
    def create(self, runner, source_run=None, upload=None):
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid, runner=runner, source_run=source_run,
                  stages=[Stage(k, label, desc) for k, label, desc, _ in STAGES])
        if upload:
            job.input = {**upload, "url": f"/api/jobs/{jid}/input"}
        with self.lock:
            self.jobs[jid] = job
            self.cancel[jid] = threading.Event()
            self._prune()
        return job

    def active_count(self):
        with self.lock:
            return sum(j.status not in TERMINAL for j in self.jobs.values())

    def _prune(self, keep=50):
        done = sorted((j for j in self.jobs.values() if j.status in TERMINAL), key=lambda j: j.created_at)
        for j in done[:max(0, len(self.jobs) - keep)]:
            shutil.rmtree(self.upload_dir(j.id), ignore_errors=True)
            self.jobs.pop(j.id, None)
            self.cancel.pop(j.id, None)

    def upload_dir(self, jid):
        return self.settings.data_dir / "uploads" / jid

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
            if status:
                s.status = status
                if status == "done":
                    s.progress, s.finished_at = 1.0, _now()
            job.stage = key
            self._recompute(job)

    def _recompute(self, job):
        w = {k: share for k, _, _, share in STAGES}
        job.progress = round(sum(w[s.key] * (1.0 if s.status == "done" else s.progress) for s in job.stages), 4)
        job.updated_at = _now()
        job.version += 1

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

    # --- running -----------------------------------------------------------------------------
    def start(self, job):
        target = {"demo": self._run_demo, "colab": self._run_colab}[job.runner]

        def go():
            try:
                self.update(job, status="running")
                target(job)
            except Exception as e:  # a runner bug must not leave the job "running" forever
                self.finish(job, "failed", error=f"{type(e).__name__}: {e}")
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
            mine = [t for s, t in lines if s == key] or [f"{label}: done"]
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
        stem = re.sub(r"[^a-z0-9_]", "_", Path(job.input["filename"]).stem.lower())[:40].strip("_") or "clip"
        clip = REPO / "clips" / f"{stem}{src.suffix.lower()}"
        shutil.copyfile(src, clip)
        tag = f"web{job.id[:6]}"
        out = REPO / "outputs" / f"remote_latent_{tag}"
        cmd = [sys.executable, str(REPO / "scripts" / "remote_run.py"), "latent", "--target", stem,
               "--tag", tag, "--min-clips", "1", "--iters", str(self.settings.colab_iters)]
        self.log(job, "silhouette", f"colab runner: {' '.join(cmd[1:])}")
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        seen = {"runner.log": 0, "job.log": 0}
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
        state = (runs.read_json(out / "status.json") or {}).get("state")
        if (out / "eval.json").is_file():
            return self.finish(job, "succeeded", result_run=runs.run_id(out))
        self.finish(job, "failed", error=f"remote_run.py exited {proc.returncode}, job state {state}")


def job_event(snapshot):
    return f"data: {json.dumps(snapshot)}\n\n"
