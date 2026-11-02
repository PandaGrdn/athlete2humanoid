"""The modal runner: a live build's training environment trained on a Modal GPU, with the policy's attempts
mirrored back every 50 iterations so the page shows the robot learn (docs/DEPLOY.md "Live training on a GPU").

After the live runner has built an upload's environment (uploads/<job>/live/env.zip), POST /api/jobs/{id}/train
(gated by server/main.py: the invite code A2H_TRAIN_CODE, or A2H_TRAIN_PER_DAY free trainings a day) hands the job
to TrainRunner, which:

1. uploads the build's inputs (env.zip, task_spec.json, qpos.npy, processed.mp4) to the Volume "a2h-runs" under
   runs/live_<job>/inputs/ (the same Volume and layout as scripts/modal_backend.py);
2. spawns scripts/modal_app.py's run_job with `python -m a2h_env train ... --attempt-every N` on the repository's
   A2H_TRAIN_BRANCH (main): mjlab's G1 tracking task on the reference and every variant the build kept, the policy
   the environment download trains (docs/TRAINING_ENV.md);
3. follows it the way the colab runner follows remote_run.py: the job's log on the Volume (progress, the reward
   curve), its status.json, and attempts/index.json, copying each new attempts/iter_NNNN.npz into the job's folder
   (uploads/<job>/live/train/attempts/, served by GET /api/jobs/{id}/attempts);
4. when the call ends, brings back weights.json and the final checkpoint (train/policy.pt), deletes the run's folder
   from the Volume, and finishes the job's policy stage.

No secret goes up: the training needs no Hugging Face token, and the Modal token stays in this process (the modal
client reads MODAL_TOKEN_ID / MODAL_TOKEN_SECRET itself). Billing stops when run_job returns, is cancelled
(POST /api/jobs/{id}/cancel) or hits its time limit (A2H_TRAIN_TIMEOUT_H, enforced here, and modal_app.py's own).
The backend is an object with a handful of methods (ModalBackend) so the tests drive the runner with a stand-in.
"""

import json
import logging
import re
import threading
import time
from pathlib import Path

from .stages import clean

log = logging.getLogger("a2h.train")
# scripts/modal_app.py's names (not imported: that module builds a Modal image at import and needs the repository)
APP, FUNCTION, VOLUME, MOUNT = "athlete2humanoid", "run_job", "a2h-runs", "/vol"
TRAIN_DIR = "train"  # under the live job's folder
ATTEMPTS = "attempts"
INPUTS = ("env.zip", "task_spec.json", "qpos.npy", "processed.mp4")
MAX_INDEX_BYTES = 1 << 20
MAX_ATTEMPT_BYTES = 4 << 20
MAX_LOG_BYTES = 8 << 20
NPZ = re.compile(r"iter_\d{4,6}\.npz")  # used with fullmatch: no trailing newline slips through
_ITER = re.compile(r"Learning iteration (\d+)/(\d+)")
_REWARD = re.compile(r"Mean reward:\s*(-?[\d.]+)")
_ATTEMPT = re.compile(r"^\[a2h_env\] attempt at iteration")


def configured():
    """Whether this process has a Modal token (the client reads it from the environment)."""
    import os
    return bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))


def command(session, iters, num_envs, attempt_every):
    """The job command run_job executes in the repository checkout, with the run's folder on the Volume."""
    run = f"{MOUNT}/runs/{session}"
    return (f"$A2H_MJLAB_PYTHON -m a2h_env train {run}/inputs/env.zip --iters {int(iters)} --num-envs {int(num_envs)} "
            f"--attempt-every {int(attempt_every)} --out {run}/outputs/policy --work-dir {run}/outputs/motions")


class ModalBackend:
    """The real thing: the deployed app's run_job and the Volume, through the modal client (imported on first use)."""

    def __init__(self):
        self._vol = None
        self._checked = (0.0, None)

    def _modal(self):
        import modal  # the client only; hosts without it never get here (configured() is false without a token)
        return modal

    def volume(self):
        if self._vol is None:
            self._vol = self._modal().Volume.from_name(VOLUME, create_if_missing=True)
        return self._vol

    def status(self, max_age=600.0):
        """(ok, why): whether the deployed app's run_job can be looked up, cached for max_age seconds."""
        at, res = self._checked
        if res is not None and time.monotonic() - at < max_age:
            return res
        try:
            self._modal().Function.from_name(APP, FUNCTION).hydrate()
            res = (True, "ok")
        except Exception as e:  # noqa: BLE001
            res = (False, f"Modal app {APP} not deployed or unreachable ({type(e).__name__})")
        self._checked = (time.monotonic(), res)
        return res

    def upload(self, session, files):
        """files: {name on the Volume under runs/<session>/inputs/: local Path}."""
        with self.volume().batch_upload(force=True) as b:
            for name, p in files.items():
                b.put_file(str(p), f"runs/{session}/inputs/{name}")

    def spawn(self, session, cmd, branch):
        """-> the function call's id."""
        return self._modal().Function.from_name(APP, FUNCTION).spawn(session, cmd, branch).object_id

    def _call(self, call_id):
        return self._modal().FunctionCall.from_id(call_id)

    def state(self, call_id):
        """done | failed | running | unknown (this host couldn't reach Modal; the job runs on)."""
        try:
            rc = self._call(call_id).get(timeout=0)
        except TimeoutError:
            return "running"
        except Exception as e:  # noqa: BLE001  (the container died, was cancelled, timed out)
            name = type(e).__name__
            if (isinstance(e, (ConnectionError, OSError)) or "Deadline" in str(e)
                    or any(w in name for w in ("Connection", "Unavailable", "ClientClosed", "Grpc", "Timeout", "Auth"))):
                return "unknown"  # this host's problem, not the job's: it runs on
            log.warning("modal call ended", extra={"fields": {"call": call_id, "error": f"{type(e).__name__}: {str(e)[-200:]}"}})
            return "failed"
        return "done" if rc == 0 else "failed"

    def cancel(self, call_id):
        self._call(call_id).cancel()

    def read(self, session, rel, limit=MAX_LOG_BYTES):
        """A file under runs/<session>/ as bytes, or None when it isn't there (yet)."""
        try:
            out, n = [], 0
            for chunk in self.volume().read_file(f"runs/{session}/{rel}"):
                n += len(chunk)
                if n > limit:
                    return None
                out.append(chunk)
            return b"".join(out)
        except Exception:  # noqa: BLE001  (not written yet, or not committed yet)
            return None

    def download(self, session, rel, local, limit=MAX_ATTEMPT_BYTES):
        data = self.read(session, rel, limit)
        if data is None:
            return False
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(local)
        return True

    def remove(self, session):
        try:
            self.volume().remove_file(f"runs/{session}", recursive=True)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("volume cleanup failed", extra={"fields": {"session": session, "error": repr(e)[:200]}})
            return False


class TrainRunner:
    """Starts and follows GPU trainings for live jobs (one thread each); JobStore owns one."""

    def __init__(self, store, backend=None):
        self.store, self.settings = store, store.settings
        self.backend = backend or ModalBackend()
        self.poll_s = max(0.01, float(self.settings.train_poll_s))
        self.lock = threading.Lock()

    # --- what the API asks ---------------------------------------------------------------------
    def available(self):
        """(ok, why) for /api/health and /readyz."""
        if "modal" not in self.settings.enabled_runners:
            return False, "disabled"
        return self.backend.status()

    def active(self):
        with self.store.lock:
            return [j for j in self.store.jobs.values() if j.training and j.training.get("state") in ("starting", "running")]

    def shutdown(self):
        """The process is stopping (a Cloud Run instance going away): stop the GPU jobs it can no longer follow, now,
        and say so in their records. Without this a call would run on unwatched until its own time limit."""
        for job in self.active():
            tr = job.training
            try:
                if tr.get("call_id"):
                    self.backend.cancel(tr["call_id"])
                if tr.get("session"):
                    self.backend.remove(tr["session"])
            except Exception:  # noqa: BLE001
                log.warning("could not stop a training at shutdown", extra={"fields": {"job": job.id}})
            self._end(job, "failed", "the server stopped while the policy was training; the GPU job was stopped")

    def started_since(self, since):
        with self.store.lock:
            return sum(1 for j in self.store.jobs.values() if j.training and (j.training.get("started_at") or 0) >= since)

    def today(self):
        return self.started_since(time.time() - 86400)

    def eta_min(self, job=None):
        """Minutes a training is expected to take (the last ones' mean, else a rule of thumb per 1,000 iterations)."""
        return round(self.settings.train_iters / 1000 * 30)

    def train_dir(self, job):
        return self.store.live_dir(job) / TRAIN_DIR

    # --- running -------------------------------------------------------------------------------
    def start(self, job):
        s = self.settings
        n = sum(1 for _ in (job.training or {}).get("history", [])) + (1 if job.training else 0)
        session = f"live_{job.id[:12]}" + (f"_{n}" if n else "")  # a retry gets its own folder on the Volume
        with self.store.lock:
            history = ((job.training or {}).get("history") or []) + ([{k: job.training.get(k) for k in ("state", "started_at", "finished_at", "error", "attempts")}] if job.training else [])
            job.training = {"backend": "modal", "state": "starting", "session": session, "call_id": None, "history": history,
                            "iterations": s.train_iters, "attempt_every": s.train_attempt_every, "num_envs": s.train_num_envs,
                            "started_at": time.time(), "finished_at": None, "attempts": 0, "iteration": 0,
                            "reward": None, "last": None, "checkpoint": None, "error": None}
            # the policy stage now carries most of the job's progress (it was 0: a CPU host trains nothing)
            for st in job.stages:
                st.share = 0.6 if st.key == "policy" else st.share * 0.4
            job.status, job.error = "running", None
            self.store._recompute(job)
        self.store.set_stage(job, "policy", 0.0, status="running")
        self.store.cancel[job.id].clear()  # a cancelled training's flag must not stop the next one at once
        self.store.flush()
        threading.Thread(target=self._run, args=(job,), daemon=True, name=f"train-{job.id}").start()
        return job

    def _run(self, job):
        try:
            self._work(job)
        except Exception:  # noqa: BLE001
            import uuid
            ref = uuid.uuid4().hex[:8]
            log.exception("train runner crashed", extra={"fields": {"job": job.id, "ref": ref}})
            self._end(job, "failed", f"internal error in the modal runner (ref {ref})")

    def _set(self, job, **kw):
        with self.store.lock:
            job.training.update(kw)
            self.store._recompute(job)

    def _work(self, job):
        store, s, be = self.store, self.settings, self.backend
        tr = job.training
        session = tr["session"]
        cancel = store.cancel[job.id]
        live = store.live_dir(job)
        files = {n: live / n for n in INPUTS if (live / n).is_file()}
        if "env.zip" not in files:
            return self._end(job, "failed", "this build has no training environment to train")
        store.log(job, "policy", f"modal runner: uploading the environment ({files['env.zip'].stat().st_size / 1e6:.1f} MB) "
                                 f"to a GPU; PPO in mjlab, {tr['iterations']} iterations, an attempt every {tr['attempt_every']}")
        be.upload(session, files)
        if cancel.is_set():
            be.remove(session)
            return self._end(job, "cancelled")
        cmd = command(session, tr["iterations"], tr["num_envs"], tr["attempt_every"])
        try:
            call_id = be.spawn(session, cmd, s.train_branch)
        except Exception as e:  # noqa: BLE001  (refused: the app isn't deployed, the workspace's spend limit)
            be.remove(session)
            log.error("modal spawn failed", extra={"fields": {"job": job.id, "error": repr(e)[:300]}})
            return self._end(job, "failed", "the GPU didn't take the job; try again later")
        self._set(job, state="running", call_id=call_id)
        store.flush()
        store.log(job, "policy", f"running on a Modal GPU (call {call_id}); the robot's first attempt arrives after "
                                 f"{tr['attempt_every']} iterations")
        state = "running"
        try:
            state, index_seen, job_state, got = self._follow(job, session, call_id)
        finally:
            # whatever ended the follow (a cancel, the time limit, a crash in this thread, lost contact): the GPU
            # job is stopped unless it is known to have finished, and its folder goes; it must never bill unwatched
            if state != "done":
                try:
                    be.cancel(call_id)
                except Exception:  # noqa: BLE001
                    log.warning("cancel failed", extra={"fields": {"job": job.id, "call": call_id}})
            be.remove(session)
        if state == "cancelled":
            return self._end(job, "cancelled")
        if state == "timeout":
            return self._end(job, "failed", f"the GPU job ran past {s.train_timeout_h:g} h and was stopped")
        if state == "lost":
            return self._end(job, "failed", "lost contact with the GPU job; it was stopped")
        if state == "done" and got:
            n = len(index_seen)
            last = job.training.get("last") or {}
            store.log(job, "policy", f"training finished: {n} attempt{'s' if n != 1 else ''} recorded"
                                     + (f", final reward {last['reward']:.2f}" if isinstance(last.get("reward"), (int, float)) else ""))
            return self._end(job, "succeeded")
        self._end(job, "failed", "the GPU job failed" + (f" (job state {job_state})" if job_state else "")
                  + ("" if got else "; no policy came back"))

    def _follow(self, job, session, call_id):
        """Poll the call, the log and the attempts until the call ends; -> (state, attempts seen, job state, got policy).
        state: done | failed | cancelled | timeout | lost."""
        be, s = self.backend, self.settings
        cancel = self.store.cancel[job.id]
        deadline = time.monotonic() + s.train_timeout_h * 3600
        log_off, index_seen, last_line, unknown = 0, set(), None, 0
        state = "running"
        while True:
            if cancel.wait(self.poll_s if state == "running" else 0):
                return "cancelled", index_seen, None, False
            log_off, last_line = self._follow_log(job, session, log_off, last_line)
            index_seen = self._mirror_attempts(job, session, index_seen)
            state = be.state(call_id)
            unknown = unknown + 1 if state == "unknown" else 0
            if state in ("done", "failed"):
                break
            if time.monotonic() > deadline:
                return "timeout", index_seen, None, False
            if unknown >= 20:
                return "lost", index_seen, None, False
        # the last commit may land after the call returns: one more pass for the log and the attempts
        self._follow_log(job, session, log_off, last_line)
        index_seen = self._mirror_attempts(job, session, index_seen)
        job_state = None
        raw = be.read(session, f"jobs/{session}/status.json", 1 << 16)
        try:
            job_state = json.loads(raw)["state"] if raw else None
        except (ValueError, KeyError, TypeError):
            pass
        return state, index_seen, job_state, self._fetch_results(job, session)

    def _follow_log(self, job, session, offset, last_line):
        """New lines of the job's log: a progress line every few iterations, the reward curve, the attempts' lines."""
        data = self.backend.read(session, f"jobs/{session}/log")
        if data is None or len(data) <= offset:
            return offset, last_line
        chunk = data[offset:].decode("utf-8", "ignore")
        it = total = None
        new_points = []
        for raw in re.split(r"[\r\n]+", chunk):
            line = clean(raw).strip()
            if not line:
                continue
            m = _ITER.search(line)
            if m:
                it, total = int(m.group(1)) + 1, int(m.group(2))
                continue
            m = _REWARD.search(line)
            if m and it is not None:
                new_points.append([it, float(m.group(1))])
                continue
            if _ATTEMPT.search(line) or line.startswith("[a2h_env]") or line.startswith("rendering with"):
                self.store.log(job, "policy", line[:300])
        if new_points:
            self._merge_curve(job, new_points, total)
            it, reward = job.training["iteration"], job.training["reward"]
            every = max(1, job.training["attempt_every"])
            if last_line is None or it - last_line >= every:
                last_line = it
                self.store.log(job, "policy", f"iteration {it}/{total or job.training['iterations']}: mean reward {reward:.2f}")
        return len(data), last_line

    def _merge_curve(self, job, points, total=None):
        """New [iteration, mean reward] points into the job's curve; the newest iteration drives the stage's progress."""
        with self.store.lock:
            pts = job.curves.setdefault("policy", [])
            known = {p[0] for p in pts}
            pts.extend(p for p in points if p[0] not in known)
            pts.sort()
            del pts[:-5000]
            tr = job.training
            if pts and pts[-1][0] >= (tr.get("iteration") or 0):
                tr["iteration"], tr["reward"] = pts[-1][0], pts[-1][1]
            total = total or tr["iterations"]
        self.store.set_stage(job, "policy", min(0.99, tr["iteration"] / max(1, total)))

    def _mirror_attempts(self, job, session, seen):
        """Copy attempts the trainer has written since the last pass into the job's folder; rewrite its index."""
        raw = self.backend.read(session, f"outputs/policy/{ATTEMPTS}/index.json", MAX_INDEX_BYTES)
        if not raw:
            return seen
        try:
            index = json.loads(raw)
            rows = [r for r in index.get("attempts", []) if isinstance(r, dict) and NPZ.fullmatch(str(r.get("file", "")))]
        except (ValueError, AttributeError):
            return seen
        out = self.train_dir(job) / ATTEMPTS
        got = []
        for r in rows:
            name = r["file"]
            if name in seen:
                continue
            if self.backend.download(session, f"outputs/policy/{ATTEMPTS}/{name}", out / name):
                seen = seen | {name}
                got.append(r)
        if got or index.get("curve"):
            index["attempts"] = [r for r in rows if r["file"] in seen]
            tmp = out / "index.json.tmp"
            out.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(index, separators=(",", ":")))
            tmp.replace(out / "index.json")
        # The index carries the reward curve too, and it lands on the Volume as soon as it is written, while the job's
        # log (a file kept open) can lag until the job ends: progress and the curve come from here first.
        curve = [[int(x), float(y)] for x, y in index.get("curve", []) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
        if curve:
            self._merge_curve(job, curve, index.get("iterations"))
        if got:
            last = got[-1]
            self._set(job, attempts=len(seen), last={k: last.get(k) for k in ("iteration", "reward", "stages_completed",
                                                                                 "stages_total", "success", "fell")})
            self.store.flush()
            stages = (f", {last['stages_completed']}/{last['stages_total']} key moments" if last.get("stages_total") else "")
            self.store.log(job, "policy", f"attempt at iteration {last['iteration']}: reward {float(last.get('reward') or 0):.2f}"
                                          f"{stages}{', fell' if last.get('fell') else ''}")
        return seen

    def _fetch_results(self, job, session):
        """weights.json and the final checkpoint -> train/; True when the checkpoint came back."""
        out = self.train_dir(job)
        ok = self.backend.download(session, "outputs/policy/weights.json", out / "weights.json", 1 << 16)
        ck = None
        if ok:
            try:
                ck = json.loads((out / "weights.json").read_text()).get("checkpoint")
            except (ValueError, OSError):
                ck = None
        if ck and re.fullmatch(r"[\w./-]+\.pt", str(ck)) and ".." not in str(ck):
            if self.backend.download(session, f"outputs/policy/{ck}", out / "policy.pt", 64 << 20):
                self._set(job, checkpoint="policy.pt")
                return True
        return False

    def _end(self, job, status, error=None):
        """The training ends (done | failed | cancelled). The build stays succeeded: the job's status goes back to it,
        the policy stage says how the training went, and a failed or cancelled one can be started again."""
        self.store.finish_training(job, "done" if status == "succeeded" else status, error)

    def recover(self, job):
        """A job journalled mid-training when the process died: best effort, stop its GPU job (it bills otherwise)."""
        tr = job.training or {}
        if not tr:
            return False
        call_id = tr.get("call_id")
        try:
            if call_id:
                self.backend.cancel(call_id)
            if tr.get("session"):
                self.backend.remove(tr["session"])
        except Exception:  # noqa: BLE001
            log.warning("could not stop a recovered training", extra={"fields": {"job": job.id, "call": call_id}})
        return True


def attempts_view(folder, url_of):
    """attempts/index.json under `folder` -> what the page lists (only attempts whose file is here), or None.
    url_of(name) -> where iter_NNNN.npz is served from."""
    folder = Path(folder)
    p = folder / ATTEMPTS / "index.json"
    try:
        if not p.is_file() or p.stat().st_size > MAX_INDEX_BYTES:
            return None
        index = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict) or not isinstance(index.get("attempts"), list):
        return None
    rows = []
    for r in index["attempts"]:
        if not isinstance(r, dict) or not NPZ.fullmatch(str(r.get("file", ""))) or not (folder / ATTEMPTS / r["file"]).is_file():
            continue
        rows.append({"iteration": int(r.get("iteration") or 0), "reward": r.get("reward"), "success": r.get("success"),
                     "stages_completed": r.get("stages_completed"), "stages_total": r.get("stages_total"),
                     "fell": bool(r.get("fell")), "fell_at_s": r.get("fell_at_s"), "frames": r.get("frames"),
                     "seconds": r.get("seconds"), "url": url_of(r["file"])})
    rows.sort(key=lambda r: r["iteration"])
    curve = [[int(x), float(y)] for x, y in index.get("curve", []) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    return {"iterations": index.get("iterations"), "attempt_every": index.get("attempt_every"), "fps": index.get("fps", 50),
            "task": index.get("task"), "policy": index.get("policy"), "attempts": rows, "curve": curve}
