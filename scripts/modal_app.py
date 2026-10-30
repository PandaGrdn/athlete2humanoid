"""Modal as a GPU backend (scripts/remote_run.py --backend modal; docs/MODAL.md).

    python -m modal deploy scripts/modal_app.py      # remote_run.py does this before each run

One function, run_job, runs the same job command a Colab VM would, on a Modal GPU. The image holds
what scripts/setup_colab.sh installs (the mjlab virtualenv at /content/mjlab_env, the main
environment's packages, the G1 assets), so a run starts in seconds instead of ~10 min of setup;
Modal caches it and rebuilds only when this file or requirements.txt changes. The code itself is the
repo's main branch, cloned fresh for each run, like Colab's.

Each run's files live on the Volume "a2h-runs" under runs/<session>/: remote_run.py uploads the
inputs and the HF_TOKEN line there, the job writes outputs/, jobs/ (status) and checkpoints there,
and the Volume is committed every 2 min so remote_run.py can follow along. The token file is deleted
when the job ends; remote_run.py deletes the whole folder once it has the results.

Billing stops when the function returns: there is no VM to forget to stop. TIMEOUT_H caps a run.
"""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

APP, VOLUME, MOUNT = "athlete2humanoid", "a2h-runs", "/vol"
GPU = os.environ.get("A2H_MODAL_GPU", "L40S")  # 48 GB, ~$1.95/h; "A100-40GB" ~$2.10/h
TIMEOUT_H = 8
REPO_URL = "https://github.com/PandaGrdn/athlete2humanoid.git"
R = "/content/athlete2humanoid"
HERE = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "ffmpeg", "libegl1", "libgl1", "libglvnd0", "libgles2", "libosmesa6")
    .pip_install("uv")
    # setup_colab.sh's mjlab virtualenv, at the path the scripts expect
    .run_commands("uv venv -p 3.12 /content/mjlab_env",
                  "uv pip install -p /content/mjlab_env/bin/python mjlab==1.6.0 tensorboard")
    # the main environment: requirements.txt plus what Colab's Python already has
    .add_local_file(HERE / "requirements.txt", "/opt/a2h/requirements.txt", copy=True)
    .run_commands("uv pip install --system -r /opt/a2h/requirements.txt torch torchvision transformers matplotlib")
    .add_local_file(HERE / "scripts" / "fetch_assets.sh", "/opt/a2h/scripts/fetch_assets.sh", copy=True)
    .run_commands("bash /opt/a2h/scripts/fetch_assets.sh")  # -> /opt/a2h/assets
    .env({"A2H_ASSETS_DIR": "/opt/a2h/assets", "PYTHONUNBUFFERED": "1"})
)
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)


def _gl():
    """MuJoCo's headless rendering: the GPU through EGL when the container exposes it, else OSMesa (CPU)."""
    egl = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if not egl.exists():
        egl.parent.mkdir(parents=True, exist_ok=True)
        egl.write_text('{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}')
    probe = ("import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/></worldbody></mujoco>');"
             "r = mujoco.Renderer(m, 32, 32); r.update_scene(mujoco.MjData(m)); r.render()")
    for gl in ("egl", "osmesa"):
        env = dict(os.environ, MUJOCO_GL=gl, PYOPENGL_PLATFORM=gl)
        if subprocess.run(["python3", "-c", probe], env=env, capture_output=True, timeout=120).returncode == 0:
            return gl
    raise RuntimeError("no headless OpenGL (EGL or OSMesa) for MuJoCo's renderer")


@app.function(gpu=GPU, volumes={MOUNT: vol}, timeout=TIMEOUT_H * 3600, cpu=8, memory=65536)
def run_job(session: str, cmd: str, branch: str = "main") -> int:
    """cmd (from remote_run.job_spec) under scripts/job.py, with runs/<session>/ as its data folder;
    the code is `branch` of the repo (remote_run.py --branch), main by default."""
    remote = f"{MOUNT}/runs/{session}"
    # Modal reuses warm containers: start from a clean checkout. Retry: concurrent jobs' clones can fail
    # transiently (exit 128); git's own message is printed so a real failure says why.
    for attempt in range(4):
        shutil.rmtree(R, ignore_errors=True)
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", branch, REPO_URL, R],
                           capture_output=True, text=True)
        if r.returncode == 0:
            break
        print(f"git clone failed ({r.returncode}): {r.stderr.strip()[-500:]}", flush=True)
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"git clone of {branch} failed 4 times: {r.stderr.strip()[-500:]}")
    if os.path.lexists(f"{R}/assets"):
        os.remove(f"{R}/assets") if os.path.islink(f"{R}/assets") else shutil.rmtree(f"{R}/assets")
    os.symlink("/opt/a2h/assets", f"{R}/assets")
    if Path(f"{remote}/inputs.tar.gz").exists():
        import tarfile
        with tarfile.open(f"{remote}/inputs.tar.gz") as tar:
            tar.extractall(f"{remote}/inputs", filter="data")
    gl = _gl()
    print(f"rendering with {gl}", flush=True)
    env = dict(os.environ, MUJOCO_GL=gl, PYOPENGL_PLATFORM=gl, A2H_OUTPUT_DIR=f"{remote}/outputs",
               A2H_SECRETS_DIR=f"{remote}/env", A2H_JOB_DATA=remote, A2H_MJLAB_PYTHON="/content/mjlab_env/bin/python")
    stop = threading.Event()

    def commit():  # so remote_run.py sees status and checkpoints while the job runs
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print(f"volume commit failed: {e}", flush=True)
    threading.Thread(target=commit, daemon=True).start()
    try:
        rc = subprocess.run(["python3", f"{R}/scripts/job.py", session, "--", "bash", "-c", f"cd {R} && {cmd}"],
                            env=env, cwd=R).returncode
    finally:
        stop.set()
        Path(f"{remote}/env/HF_TOKEN").unlink(missing_ok=True)
        time.sleep(1)
        vol.commit()
    return rc
