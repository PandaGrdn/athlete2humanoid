"""Unattended GPU jobs on Colab, driven from this Mac with the Colab CLI.

    python3 scripts/remote_run.py probe --min-clips 7       # cross-embodiment probe
    python3 scripts/remote_run.py latent --target squat     # video -> latent reward -> policies
    python3 scripts/remote_run.py latent --smoke            # the same, tiny, no clips: plumbing test

1. waits until at least --min-clips videos sit in clips/ and stop changing (copy finished)
2. checks the HF token in .env before spending GPU time
3. creates an A100, uploads clips + token, launches the job via scripts/job.py
4. keeps the VM busy, polls status every 2 min (and pulls the job log every 10), recreates the VM
   and retries once if it's lost
5. downloads results to outputs/remote_<job>/ and stops the VM

Progress goes to outputs/remote_<job>/runner.log. Needs `colab` logged in (oauth2).
"""

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".mov", ".m4v"}
SKIP = {"bedroom"}  # test footage, not an action clip
REMOTE = "/content/a2h"
OUT = SESSION = None  # set in main()


def job_spec(a):
    """(setup flag, command run in the repo after setup, {local name: remote path} to fetch)."""
    if a.job == "probe":
        out = f"{REMOTE}/outputs/cross_embodiment"
        return ("--with-sam", f"python3 experiments/cross_embodiment_probe.py {{clips}} --out {out}",
                {"cross_embodiment.json": f"{out}/cross_embodiment.json"})
    out = f"{REMOTE}/outputs/latent_{'smoke' if a.smoke else a.target}"
    cmd = f"python3 experiments/latent_policy.py {{clips}} --target {a.target} --out {out} --iters {a.iters}"
    if a.smoke:
        return "--with-main", cmd + " --smoke", {"results.tar.gz": f"{out}/results.tar.gz"}
    return "--with-sam", cmd, {"results.tar.gz": f"{out}/results.tar.gz"}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "runner.log", "a") as f:
        f.write(line + "\n")


def colab(*args, timeout=600, check=True):
    r = subprocess.run(["colab", "--auth", "oauth2", *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"colab {' '.join(args[:2])} failed: {(r.stderr or r.stdout)[-300:]}")
    return r.stdout


def clips():
    return sorted(p for p in (REPO / "clips").iterdir()
                  if p.suffix.lower() in VIDEO_EXT and p.stem.lower() not in SKIP)


def wait_for_clips(min_clips, max_wait_h):
    deadline, last = time.time() + max_wait_h * 3600, None
    while time.time() < deadline:
        found = clips()
        sizes = tuple((p.name, p.stat().st_size) for p in found)
        if len(found) >= min_clips and sizes == last:  # unchanged for a full poll: copying is done
            return found
        if sizes != last:
            log(f"waiting for clips: {len(found)}/{min_clips} present ({', '.join(p.name for p in found)})")
        last = sizes
        time.sleep(60)
    raise SystemExit(f"gave up waiting for {min_clips} clips")


def check_token():
    code = ("import sys; sys.path.insert(0, '.'); import os; os.environ['A2H_SECRETS_DIR'] = '/nonexistent'; "
            "os.environ.pop('HF_TOKEN', None); import config; "
            "from huggingface_hub import HfApi, hf_hub_download; t = config.get_secret('HF_TOKEN'); "
            "HfApi(token=t).whoami(); hf_hub_download('facebook/sam3.1', 'config.json', token=t); print('TOKEN_OK')")
    # Run with a Python that has huggingface_hub (the project's requirements).
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    if "TOKEN_OK" not in r.stdout:
        raise SystemExit("HF token in .env is invalid or lacks SAM 3.1 access; update .env and rerun.")


def launch(found, setup, cmd):
    colab("new", "-s", SESSION, "--gpu", "A100", timeout=900)
    log("A100 ready; uploading clips")
    _exec(f"import os; [os.makedirs(d, exist_ok=True) for d in ('{REMOTE}/clips', '{REMOTE}/env')]")
    for p in found:
        colab("upload", "-s", SESSION, str(p), f"{REMOTE}/clips/{p.name}", timeout=900)
    colab("upload", "-s", SESSION, str(REPO / ".env"), f"{REMOTE}/env/HF_TOKEN")
    paths = " ".join(f"'{REMOTE}/clips/{p.name}'" for p in found)
    inner = f"cd /content/athlete2humanoid && bash scripts/setup_colab.sh {setup} && " + cmd.format(clips=paths)
    _exec(f"""
import os, subprocess
R = "/content/athlete2humanoid"
if not os.path.exists(R):
    subprocess.run(["git", "clone", "-q", "https://github.com/PandaGrdn/athlete2humanoid.git", R], check=True)
else:
    subprocess.run(["git", "-C", R, "pull", "-q"], check=True)
env = dict(os.environ, MUJOCO_GL="egl", A2H_SAM_PYTHON="/content/sam_env/bin/python",
           A2H_OUTPUT_DIR="{REMOTE}/outputs", A2H_SECRETS_DIR="{REMOTE}/env", A2H_ASSETS_DIR=R + "/assets")
subprocess.Popen(["python3", R + "/scripts/job.py", "{SESSION}", "--", "bash", "-c", {inner!r}], env=env,
                 start_new_session=True, cwd="/content", stdout=open("{REMOTE}/job_{SESSION}.out", "w"),
                 stderr=subprocess.STDOUT)
print("launched")
""")
    log(f"{SESSION} job launched")
    return keep_busy()


def keep_busy():
    """A cell that stays busy until the job ends, so Colab doesn't reclaim an "idle" VM."""
    keeper = OUT / "keeper.py"
    keeper.write_text(f"""
import json, time
f = "{REMOTE}/jobs/{SESSION}/status.json"
time.sleep(20)
while json.load(open(f))["state"] == "running":
    print(".", end="", flush=True); time.sleep(60)
""")
    return subprocess.Popen(["colab", "--auth", "oauth2", "exec", "-s", SESSION, "-f", str(keeper)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _exec(code):
    f = OUT / "_exec.py"
    f.write_text(code)
    return colab("exec", "-s", SESSION, "-f", str(f))


def status():
    try:
        colab("download", "-s", SESSION, f"{REMOTE}/jobs/{SESSION}/status.json", str(OUT / "status.json"), timeout=120)
        return json.loads((OUT / "status.json").read_text())["state"]
    except Exception as e:
        return "lost" if "No active" in str(e) or "lost" in str(e) or "404" in str(e) else "unknown"


def pull_log():
    try:
        colab("download", "-s", SESSION, f"{REMOTE}/jobs/{SESSION}/log", str(OUT / "job.log"), timeout=300)
    except Exception as e:
        log(f"could not download job.log: {e}")


def fetch_results(fetch):
    pull_log()
    for local, remote in fetch.items():
        try:
            colab("download", "-s", SESSION, remote, str(OUT / local), timeout=900)
        except Exception as e:
            log(f"could not download {local}: {e}")
            continue
        if local.endswith(".tar.gz"):
            with tarfile.open(OUT / local) as tar:
                tar.extractall(OUT, filter="data")
            log(f"unpacked {local}")


def main(a):
    global OUT, SESSION
    SESSION = a.job + ("_smoke" if a.smoke else "")
    OUT = REPO / "outputs" / f"remote_{SESSION}"
    OUT.mkdir(parents=True, exist_ok=True)
    setup, cmd, fetch = job_spec(a)
    found = [] if a.smoke else wait_for_clips(a.min_clips, a.max_wait_h)
    if found:
        log(f"clips ready: {', '.join(p.name for p in found)}")
        check_token()
        log("HF token OK")
    for attempt in (1, 2):
        keeper = launch(found, setup, cmd)
        state, unknown, polls = "running", 0, 0
        while state in ("running", "unknown"):
            time.sleep(120)
            state, polls = status(), polls + 1
            unknown = unknown + 1 if state == "unknown" else 0
            if unknown >= 5:
                state = "lost"
            if state == "running" and keeper.poll() is not None:
                keeper = keep_busy()  # the exec connection dropped; keep the VM busy again
            if state == "running" and polls % 5 == 0:
                pull_log()
        keeper.kill()
        log(f"job state: {state} (attempt {attempt})")
        if state in ("done", "failed"):
            fetch_results(fetch)
            break
        log("VM lost; retrying on a fresh one")
        colab("stop", "-s", SESSION, check=False)
    colab("stop", "-s", SESSION, check=False)
    log(f"VM stopped. Results in {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["probe", "latent"])
    ap.add_argument("--min-clips", type=int, default=6)
    ap.add_argument("--max-wait-h", type=float, default=8)
    ap.add_argument("--target", default="squat")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args())
