"""Run a long command on a Colab VM so it survives disconnects and leaves a trail on Drive.

    python3 scripts/job.py NAME -- bash -c "..."

- logs to /content/a2h/jobs/NAME/log ($A2H_JOB_DATA instead of /content/a2h when set), status in /content/a2h/jobs/NAME/status.json
  ({"state": "running" | "done" | "failed", "returncode", "started", "finished"})
- copies /content/a2h/{outputs,jobs,logs} to Drive (a2h/colab_runs/) every 5 min and at the end,
  when Drive is mounted, so results outlive the VM
- launch it detached (start_new_session) and poll status.json: unlike a bare PID check, this
  can't be fooled by a finished-but-unreaped (zombie) process
"""

import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

DATA = Path(os.environ.get("A2H_JOB_DATA", "/content/a2h"))  # FarmShare: <fs-root>/runs/<session>
DRIVE = Path("/content/drive/MyDrive/a2h/colab_runs")


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def sync():
    if not DRIVE.parent.parent.exists():  # Drive not mounted
        return
    for sub in ("outputs", "jobs", "logs"):
        if (DATA / sub).exists():
            shutil.copytree(DATA / sub, DRIVE / sub, dirs_exist_ok=True)


def main():
    name, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
    job = DATA / "jobs" / name
    job.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "cmd": cmd, "started": now()}
    (job / "status.json").write_text(json.dumps(status, indent=1))

    stop = threading.Event()
    def periodic_sync():
        while not stop.wait(300):
            try:
                sync()
            except Exception as e:  # a failed copy must never kill the job
                print(f"[job] sync failed: {e}", flush=True)
    threading.Thread(target=periodic_sync, daemon=True).start()

    with open(job / "log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    stop.set()
    status |= {"state": "done" if rc == 0 else "failed", "returncode": rc, "finished": now()}
    (job / "status.json").write_text(json.dumps(status, indent=1))
    try:
        sync()
    except Exception as e:
        print(f"[job] final sync failed: {e}", flush=True)


if __name__ == "__main__":
    main()
