"""The live runner: an uploaded video -> its RL task environment, for real, on the host's CPU.

No GPU needed, so it runs on the free Space too: video check, pose, retarget and the keypose task
spec take ~10-40 s (server/live_worker.py). Training a policy on that environment needs a GPU for
1-2 h; on a host without one the policy stage ends "skipped" and the page links to trained runs.

The worker is untrusted-input code (ffmpeg and MediaPipe parse the upload), so it runs as a
subprocess: its own session (killed as a group on cancel or timeout), a minimal environment (no
secrets), lowered priority, CPU-time and (Linux) address-space limits, and at most
A2H_LIVE_CONCURRENCY at a time; later jobs wait in "queued".
"""

import functools
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time

from .config import REPO

log = logging.getLogger("a2h.live")
LIVE_DIR = "live"
MAX_LIVE_BYTES = 8 * 1048576  # live.json is ~250 kB for a 6 s clip, ~1 MB at the 20 s cap
_IMPORTS = "import mediapipe, mink, mujoco, torch, cv2, scipy, mediapy, imageio_ffmpeg"


@functools.lru_cache(maxsize=4)
def _probe(python):
    try:
        r = subprocess.run([python, "-c", _IMPORTS], capture_output=True, timeout=120, env=_env(None))
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def available(settings):
    """Whether this host can build live environments: A2H_LIVE=1/0 forces it, "auto" checks the imports once."""
    if settings.live_mode in ("0", "1"):
        return settings.live_mode == "1"
    return _probe(settings.live_python)


def _env(out_dir):
    """The worker's whole environment: nothing from ours but the path and the asset location."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MUJOCO_GL": "disable",
           "GLOG_minloglevel": "2", "TF_CPP_MIN_LOG_LEVEL": "2"}
    if out_dir is not None:  # config.py makes its output folders at import: keep them in the job's own
        scratch = str(out_dir / "scratch")
        env |= {"HOME": scratch, "TMPDIR": scratch, "XDG_CACHE_HOME": scratch, "MPLCONFIGDIR": scratch,
                "A2H_OUTPUT_DIR": scratch, "A2H_CLIPS_DIR": scratch}
    if os.environ.get("A2H_ASSETS_DIR"):
        env["A2H_ASSETS_DIR"] = os.environ["A2H_ASSETS_DIR"]
    return env


def _limits(cpu_s, memory_mb):
    def apply():
        import resource
        os.nice(10)  # the API stays responsive while a build runs
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if memory_mb and sys.platform.startswith("linux"):  # macOS doesn't enforce RLIMIT_AS
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
    return apply


def live_data(folder):
    """A live job's live.json, or None before its pose stage is done."""
    p = folder / "live.json"
    try:
        if not p.is_file() or p.stat().st_size > MAX_LIVE_BYTES:
            return None
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


class LiveRunner:
    def __init__(self, store):
        self.store = store
        self.settings = store.settings
        self.slots = threading.BoundedSemaphore(max(1, self.settings.live_concurrency))

    def run(self, job):
        store = self.store
        cancel = store.cancel[job.id]
        src = store.upload_dir(job.id) / job.input["stored"]
        out = store.upload_dir(job.id) / LIVE_DIR
        (out / "scratch").mkdir(parents=True, exist_ok=True)
        store.log(job, "video", f"received {job.input['filename']} ({job.input['size'] / 1e6:.1f} MB)")
        if not self.slots.acquire(blocking=False):
            store.log(job, "video", "waiting for another visitor's environment build to finish (one at a time here)")
            while not self.slots.acquire(timeout=0.5):
                if cancel.is_set():
                    return store.finish(job, "cancelled")
        try:
            store.update(job, status="running")
            self._work(job, src, out)
        finally:
            self.slots.release()

    def _work(self, job, src, out):
        store, s = self.store, self.settings
        cancel = store.cancel[job.id]
        cmd = [s.live_python, "-m", "server.live_worker", str(src), str(out), "--max-seconds", str(s.live_max_seconds),
               "--timeout-s", str(s.live_timeout_s)]
        store.set_stage(job, "video", 0.0)
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=REPO, env=_env(out), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True,
                                preexec_fn=_limits(int(s.live_timeout_s * 2), s.live_memory_mb))
        err_tail = []
        threading.Thread(target=lambda: err_tail.extend(proc.stderr.read().splitlines()[-40:]), daemon=True).start()
        reader_done = threading.Event()
        user_error = []

        def read():
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                if "error" in msg:
                    user_error.append(str(msg["error"])[:300])
                elif msg.get("stage") in ("video", "pose", "retarget", "environment") and msg.get("note"):
                    store.log(job, msg["stage"], str(msg["note"])[:600])  # a line for the log, no status change
                elif msg.get("stage") in ("video", "pose", "retarget", "environment"):
                    if msg.get("done"):
                        store.log(job, msg["stage"], str(msg.get("log", ""))[:600])
                        store.set_stage(job, msg["stage"], status="done")
                    else:
                        p = msg.get("progress")
                        store.set_stage(job, msg["stage"], float(p) if isinstance(p, (int, float)) else None)
            reader_done.set()
        threading.Thread(target=read, daemon=True).start()

        while proc.poll() is None:
            if cancel.wait(0.25):
                self._kill(proc)
                return store.finish(job, "cancelled")
            if time.monotonic() - t0 > s.live_timeout_s:
                self._kill(proc)
                return store.finish(job, "failed", error=f"building the environment took longer than {s.live_timeout_s:g} s")
        reader_done.wait(5)
        if proc.returncode == 2 and user_error:
            return store.finish(job, "failed", error=user_error[0])
        if proc.returncode != 0:
            ref = os.urandom(4).hex()
            log.error("live worker failed", extra={"fields": {"job": job.id, "ref": ref, "code": proc.returncode,
                                                               "stderr": "\n".join(err_tail)[-4000:]}})
            return store.finish(job, "failed", error=f"internal error building the environment (ref {ref})")
        if s.live_muse_names:
            self._name(job, out)
        store.log(job, "policy", "training a policy on this environment needs a GPU (PPO in mjlab, 1-2 h); "
                                 "this host builds environments only")
        store.set_stage(job, "policy", status="skipped")
        store.finish(job, "succeeded")

    def _name(self, job, out):
        """Muse names the key moments (muse/moment_namer.py). Here, not in the worker: the worker never
        holds the key. Best effort: on any failure the measured names stay."""
        try:
            from muse.client import MuseClient
            from muse.moment_namer import name_moments
            spec = name_moments(out / "task_spec.json", MuseClient(), video=out / "processed.mp4")
            self.store.log(job, "environment", f"Muse named the key moments: {spec['summary']}"[:600])
        except Exception as e:  # noqa: BLE001
            ref = os.urandom(4).hex()
            log.warning("muse naming failed", extra={"fields": {"job": job.id, "ref": ref, "error": repr(e)[:500]}})
            self.store.log(job, "environment", f"Muse couldn't name the moments (ref {ref}); keeping the measured names")

    @staticmethod
    def _kill(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(5)
