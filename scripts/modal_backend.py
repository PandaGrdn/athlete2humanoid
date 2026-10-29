"""remote_run.py's Modal backend: upload a run's inputs to the Volume, spawn scripts/modal_app.py's
run_job, follow its status and checkpoints, fetch the results, clean up. docs/MODAL.md.

Needs `pip install modal` and a login (`modal token new`, once, in a browser). The HF token goes
to the Volume as a one-line file (never on a command line or in a log) and is deleted when the job
ends; nothing else from .env leaves this machine.
"""

import io
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from modal_app import APP, MOUNT, VOLUME  # noqa: E402


class ModalRun:
    def __init__(self, session, out, log, gpu=None):
        import modal
        self.modal, self.session, self.out, self.log, self.gpu = modal, session, Path(out), log, gpu
        self.rel = f"runs/{session}"  # in the Volume
        self.run_dir = f"{MOUNT}/{self.rel}"  # the same, as the job sees it
        self.vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
        self.call = None

    @staticmethod
    def check_login():
        r = subprocess.run([sys.executable, "-m", "modal", "token", "info"], capture_output=True, text=True)
        if r.returncode != 0:  # prints nothing secret; exit 1 without a token
            raise SystemExit("Modal isn't logged in on this machine. Run once: python -m modal token new")

    def deploy(self):
        """(Re)deploy the app; Modal reuses the cached image unless modal_app.py or requirements.txt changed."""
        self.log("deploying the Modal app (the image builds once, ~10 min; cached after)")
        env = {"A2H_MODAL_GPU": self.gpu} if self.gpu else {}
        r = subprocess.run([sys.executable, "-m", "modal", "deploy", str(REPO / "scripts" / "modal_app.py")],
                           capture_output=True, text=True, env={**__import__("os").environ, **env})
        if r.returncode != 0:
            raise RuntimeError(f"modal deploy failed: {(r.stderr or r.stdout)[-600:]}")

    def upload(self, found, inputs, env_file):
        with self.vol.batch_upload(force=True) as b:
            for p in found:
                b.put_file(str(p), f"{self.rel}/clips/{p.name}")
            if inputs:
                b.put_file(str(inputs), f"{self.rel}/inputs.tar.gz")
            lines = Path(env_file).read_text().splitlines() if Path(env_file).is_file() else []
            token = next((ln for ln in lines if ln.strip().startswith("HF_TOKEN")), None)
            if token:
                b.put_file(io.BytesIO((token.strip() + "\n").encode()), f"{self.rel}/env/HF_TOKEN")
        self.log(f"uploaded {len(found)} clip(s)" + (", inputs" if inputs else "") + (", HF token line" if token else ""))

    def submit(self, cmd):
        fn = self.modal.Function.from_name(APP, "run_job")
        self.call = fn.spawn(self.session, cmd)
        (self.out / "modal_call.json").write_text(json.dumps({"call_id": self.call.object_id, "session": self.session}))
        self.log(f"{self.session} running on Modal (call {self.call.object_id})")

    def state(self):
        """done | failed | running | unknown, from the function call (the job's own status.json says
        why). unknown: this machine couldn't reach Modal (asleep, offline); the job itself runs on."""
        try:
            rc = self.call.get(timeout=0)
        except TimeoutError:
            return "running"
        except Exception as e:  # noqa: BLE001  (the container died, was cancelled, timed out)
            if isinstance(e, (ConnectionError, OSError)) or "Connection" in type(e).__name__ or "Deadline" in str(e):
                self.log(f"can't reach Modal ({type(e).__name__}: {str(e)[-120:]}); the job runs on, polling again")
                return "unknown"
            self.log(f"Modal call ended with {type(e).__name__}: {str(e)[-300:]}")
            return "failed"
        return "done" if rc == 0 else "failed"

    def job_status(self):
        """The job's own state (scripts/job.py's status.json on the Volume), or None if unreadable."""
        try:
            raw = b"".join(self.vol.read_file(f"{self.rel}/jobs/{self.session}/status.json"))
            return json.loads(raw)["state"]
        except Exception:  # noqa: BLE001
            return None

    def download(self, remote, local, timeout=None):
        """remote: a path as the job sees it (/vol/runs/...) -> local file."""
        rel = remote.removeprefix(MOUNT + "/")
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "wb") as f:
            for chunk in self.vol.read_file(rel):
                f.write(chunk)

    def cancel(self):
        if self.call is not None:
            self.call.cancel()
            self.log("Modal call cancelled")

    def cleanup(self):
        if self.job_status() == "running":  # never delete a live job's files (a misread call state)
            self.log(f"kept {self.rel} on the Volume: its job still says running")
            return
        try:
            self.vol.remove_file(self.rel, recursive=True)
            self.log(f"removed {self.rel} from the Volume")
        except Exception as e:  # noqa: BLE001
            self.log(f"couldn't remove {self.rel} from the Volume: {e}")


def wait(run, sync, poll_s=120):
    """Poll until the call ends, syncing checkpoints between polls. -> final state."""
    while True:
        st = run.state()
        if st not in ("running", "unknown"):
            return st
        if st == "running":
            sync()
        time.sleep(poll_s)
