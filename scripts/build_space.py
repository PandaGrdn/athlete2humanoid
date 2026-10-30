"""Assemble the backend + a slim copy of finished runs into a folder that is a Hugging Face Space.

    python3 scripts/build_space.py --cors https://athlete2humanoid.pages.dev \
        --onnx-python <a python with torch + onnx>        # optional: ships ONNX policies
    hf upload <user>/athlete2humanoid-api build/space . --repo-type space

Only what the web app shows is copied (videos, metrics, task specs, the latest checkpoint per
policy, motion CSVs, the reward net); training logs become a small curve.json and job.log keeps only
the lines the demo runner replays. No .env, tokens or raw clips are ever copied. See docs/DEPLOY.md.

Default runs: every outputs/ folder matching DEFAULT_RUNS that has an eval.json, so a new keypose
run joins the next bundle by itself. The live runner (server/live_worker.py: video -> task
environment on CPU) adds its pipeline modules, the MediaPipe model, the G1 model files the MJCF
references and a second hash-locked dependency set (server/requirements-live.lock); --live off
leaves all of that out.
"""

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from server import runs as R  # noqa: E402

RUN_FILES = ["eval.json", "status.json", "task_spec.json", "distill_report.json", "comparison.mp4", "reward.pt", "web.json",
             "poster.jpg"]
POLICY_FILES = ["policy.mp4", "policy_metrics.json", "policy.onnx"]
CLIP_FILES = ["meta.json", "retarget_report.json", "motion.csv", "side_by_side.mp4", "silhouette.mp4"]

# The gallery: the runs trained on the SAM 3D Body references (docs/PIPELINE_ACCURACY.md, fix 3): single-clip
# tracking on the squat jump, front kick and squat, the keypose task on jumping jacks, lunge and punches.
DEFAULT_RUNS = ["remote_latent_s3_*"]
DEFAULT_DEMO_RUN = "latent_s3_kick"  # featured on the home page: a front kick learned from its video

# What server/live_worker.py imports from the repo (no latent/, no SAM: MediaPipe on CPU), and its assets.
LIVE_MODULES = ["config.py", "pose.py", "track.py", "retarget.py", "render.py",
                "skill/__init__.py", "skill/task.py", "skill/keypose.py"]
POSE_MODEL = "assets/pose_landmarker_heavy.task"
G1_SCENE = "assets/mujoco_menagerie/unitree_g1/scene.xml"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

README = """---
title: athlete2humanoid API
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Video to humanoid policy (SAM 3.1, V-JEPA 2, mjlab) - demo API
---

Backend for athlete2humanoid: any video of a person moving in, a Unitree G1 control policy out.
This Space runs on CPU. The live runner builds a real task environment (pose, retarget, keypose
task spec) from an uploaded clip; training the policy needs a GPU, so the demo runner replays recorded
A100 runs. Endpoints: `/api/health`, `/api/runs`, `/api/runs/{id}`, `/api/jobs` (see `/docs`).
The Unitree G1 model under `assets/` is from MuJoCo Menagerie (BSD-3-Clause, its LICENSE is included).
"""

# Pinned by digest (Dependabot bumps it); deps installed from a hash-locked file as root, so the app
# user (uid 1000, which Spaces run as) can read but not modify code or packages. Only /tmp is written.
BASE_IMAGE = "python:3.12.16-slim-bookworm@sha256:6dd8b6dab6b3adc7812bab69eca5dbdc1a5cc690ad28306d234ffb4ef8ac9753"
DOCKERFILE = """FROM {base}
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY server/requirements.lock server/requirements.lock
RUN pip install --require-hashes --no-deps -r server/requirements.lock \\
 && useradd --create-home --uid 1000 --shell /usr/sbin/nologin user
{live_deps}COPY server server
{live_code}COPY data data
# A2H_HOSTED: the colab runner can't be enabled and the job list is off, whatever Space variables say.
ENV A2H_HOSTED=1 \\
    A2H_RUNS_DIR=/app/data/runs \\
    A2H_DATA_DIR=/tmp/a2h \\
    A2H_RUNNERS=demo \\
    A2H_MAX_UPLOAD_MB=100 \\
    A2H_MAX_UPLOAD_STORAGE_MB=2000 \\
    A2H_PROXY_HOPS=1 \\
    A2H_CORS_ORIGINS="{cors}"{demo_run}{live_env}
USER 1000
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \\
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4)"]
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log", "--no-server-header", \\
     "--limit-concurrency", "200", "--timeout-keep-alive", "15", "--timeout-graceful-shutdown", "10"]
"""
# CPU torch from the PyTorch index (hashes pin the exact +cpu wheels); everything else as for the server lock.
# /app/outputs and /app/clips exist (empty, root-owned) so config.py's mkdir at import is a no-op.
LIVE_DEPS = """# The live runner's pipeline: CPU torch, MediaPipe, MuJoCo + mink, headless OpenCV (server/requirements-live.txt).
# MediaPipe's C library links libEGL/libGLESv2 even on its CPU path: the GL dispatch libraries only, no drivers
# (Debian's signed archive; the digest-pinned base fixes the release).
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 && rm -rf /var/lib/apt/lists/*
COPY server/requirements-live.lock server/requirements-live.lock
RUN pip install --require-hashes --no-deps --extra-index-url {torch_index} -r server/requirements-live.lock \\
 && mkdir -p /app/outputs /app/clips
COPY assets assets
"""
LIVE_CODE = """COPY {modules} ./
COPY skill skill
"""
# server/live.py hands the worker A2H_ASSETS_DIR and points HOME, caches and config.py's output folders at
# the job's own scratch under A2H_DATA_DIR. These defaults keep anything else (the import probe, a manual
# `python -m server.live_worker`) writing under /tmp too. No MUJOCO_GL: nothing is rendered.
LIVE_ENV = """ \\
    A2H_ASSETS_DIR=/app/assets \\
    A2H_OUTPUT_DIR=/tmp/a2h-pipeline/outputs \\
    A2H_CLIPS_DIR=/tmp/a2h-pipeline/clips \\
    XDG_CACHE_HOME=/tmp/a2h-cache \\
    MPLCONFIGDIR=/tmp/a2h-cache/matplotlib \\
    TORCH_HOME=/tmp/a2h-cache/torch"""
DOCKERIGNORE = """**/__pycache__
**/*.pyc
**/.*
**/*.sqlite3*
"""


def copy(src, dst):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst.stat().st_size
    return 0


def ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def slim_run(src, dst):
    size = sum(copy(src / f, dst / f) for f in RUN_FILES)
    for pdir in sorted(src.glob("policy_*")):
        out = dst / pdir.name
        size += sum(copy(pdir / f, out / f) for f in POLICY_FILES)
        cks = sorted(pdir.glob("model_*.pt"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
        if cks:
            size += copy(cks[-1], out / cks[-1].name)
        out.mkdir(parents=True, exist_ok=True)
        (out / "curve.json").write_text(json.dumps(R.train_curve(pdir)))
    for cdir in sorted(d for d in (src / "clips").glob("*") if d.is_dir()):
        size += sum(copy(cdir / f, dst / "clips" / cdir.name / f) for f in CLIP_FILES)
    lines = R.replay_lines(src)
    if lines:
        (dst / "job.log").write_text("\n".join(t for _, t in lines) + "\n")
    return size


def poster(run_dir, exe):
    """A still from the hero clip, so gallery cards paint before their video loads."""
    target = R.summarize(run_dir.parent, R.run_id(run_dir), run_dir).get("target")
    src = run_dir / "clips" / str(target) / "side_by_side.mp4"
    if exe and src.is_file() and not (run_dir / "poster.jpg").exists():
        subprocess.run([exe, "-y", "-loglevel", "error", "-ss", "2", "-i", str(src), "-frames:v", "1", "-q:v", "3",
                        str(run_dir / "poster.jpg")], check=False)


def mjcf_files(xml, seen=None):
    """The MJCF and every file it pulls in (includes, meshes, textures): includes against the including
    file, assets against the meshdir/texturedir of the file that declares them. Right for the G1 (all its
    files sit in one folder; g1.xml sets meshdir="assets"); a model that splits across folders needs a check."""
    import xml.etree.ElementTree as ET
    xml = Path(xml)
    seen = seen if seen is not None else {}
    seen[xml] = None
    root = ET.parse(xml).getroot()
    comp = root.find("compiler")
    dirs = {"mesh": (comp.get("meshdir") if comp is not None else None) or "",
            "texture": (comp.get("texturedir") if comp is not None else None) or "",
            "hfield": (comp.get("assetdir") if comp is not None else None) or ""}
    for el in root.iter():
        f = el.get("file")
        if not f:
            continue
        if el.tag == "include":
            mjcf_files(xml.parent / f, seen)
        else:
            seen[xml.parent / dirs.get(el.tag, "") / f] = None
    return list(seen)


def live_files():
    """Repo-relative paths the live worker needs: its modules, the MediaPipe model and the G1 model."""
    g1 = [p.resolve().relative_to(REPO) for p in mjcf_files(REPO / G1_SCENE)]
    lic = REPO / Path(G1_SCENE).parent / "LICENSE"
    return [Path(m) for m in LIVE_MODULES], [Path(POSE_MODEL), *g1, *([lic.relative_to(REPO)] if lic.is_file() else [])]


def server_has_live():
    """True once server/ has the live runner (server/live.py + server/live_worker.py). It turns itself on
    (A2H_LIVE=auto) when the image's pipeline imports, so A2H_RUNNERS stays "demo"."""
    return all((REPO / "server" / f).is_file() for f in ("live.py", "live_worker.py"))


def default_runs(src_root):
    found = R.find_runs(src_root, ["*smoke*", "_*"])
    return [d for d in found.values() if any(fnmatch.fnmatch(d.name, p) for p in DEFAULT_RUNS)]


def check_origins(cors):
    """The Space's CORS list ends up in the image: https origins (or localhost) only, never '*'."""
    bad = [o for o in cors.split(",") if o == "*" or not re.fullmatch(r"https://[^/\s\"]+|http://(localhost|127\.0\.0\.1)(:\d+)?", o)]
    if bad:
        raise SystemExit(f"--cors: {bad} must be https origins (scheme://host, no path), comma-separated")


def main(a):
    check_origins(a.cors)
    src_root = REPO / "outputs"
    if a.code_only:  # CI redeploys: code, Dockerfile and README only; the Space keeps its data/ and assets/
        picked = []
    else:
        picked = [Path(p).resolve() for p in a.runs] if a.runs else default_runs(src_root)
        if not picked:
            raise SystemExit("no finished runs (folders with eval.json) to ship")
    has_live = server_has_live()
    live = a.live == "on" or (a.live == "auto" and has_live)  # ship the pipeline
    if live:
        modules, assets = live_files()
        missing = [str(p) for p in modules + ([] if a.code_only else assets) if not (REPO / p).is_file()]
        if missing:
            raise SystemExit(f"live runner: missing {missing}. Run scripts/fetch_assets.sh, or pass --live off.")
    if not has_live and a.live != "off":
        print("note: server/ has no live runner (server/live.py, server/live_worker.py)"
              + ("; pipeline shipped anyway" if live else "; building demo-only"))
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    data = out / "data" / "runs"
    total = 0
    for run in picked:
        total += slim_run(run, data / run.name)
        print(f"run    {run.name}")
    for rb in sorted({p for root in ({src_root, *(r.parent for r in picked)} if picked else ()) for p in root.glob("remote_robust*/robust.json")}):
        total += copy(rb, data / rb.parent.name / "robust.json")
        print(f"robust {rb.parent.name}")
    if a.onnx_python:
        subprocess.run([a.onnx_python, str(REPO / "scripts" / "export_onnx.py"), *[str(data / r.name) for r in picked],
                        "--verify"], check=True)
    exe = ffmpeg()
    for run in picked:
        poster(data / run.name, exe)
    # code and the dependency lock only: never dotfiles (.env), journals or tests
    shutil.copytree(REPO / "server", out / "server", ignore=shutil.ignore_patterns(
        "__pycache__", "tests", "*.pyc", ".*", "*.sqlite3*", "requirements-dev.txt"))
    code_size = 0
    if live:
        for p in modules + ([] if a.code_only else assets):
            code_size += copy(REPO / p, out / p)
        print(f"live   {len(modules)} modules" + ("" if a.code_only else f", {len(assets)} asset files"))
    ids = [R.run_id(r) for r in picked]
    # code-only keeps the full deploy's default (the server falls back to the newest run if it's missing)
    demo_run = a.demo_run or (DEFAULT_DEMO_RUN if a.code_only or DEFAULT_DEMO_RUN in ids else "")
    if demo_run and picked and demo_run not in ids:
        raise SystemExit(f"--demo-run {demo_run!r} is not in the bundle ({', '.join(ids)})")
    demo = f" \\\n    A2H_DEMO_RUN={demo_run}" if demo_run else ""
    (out / "Dockerfile").write_text(DOCKERFILE.format(
        base=BASE_IMAGE, cors=a.cors, demo_run=demo,
        live_deps=LIVE_DEPS.format(torch_index=TORCH_CPU_INDEX) if live else "",
        live_code=LIVE_CODE.format(modules=" ".join(m for m in LIVE_MODULES if "/" not in m)) if live else "",
        live_env=LIVE_ENV if live else ""))
    (out / ".dockerignore").write_text(DOCKERIGNORE)
    (out / "README.md").write_text(README)
    (out / ".gitattributes").write_text("*.mp4 filter=lfs diff=lfs merge=lfs -text\n*.pt filter=lfs diff=lfs merge=lfs -text\n"
                                       "*.onnx filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n"
                                       "*.task filter=lfs diff=lfs merge=lfs -text\n*.STL filter=lfs diff=lfs merge=lfs -text\n")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    runs_size = sum(p.stat().st_size for p in data.rglob("*") if p.is_file()) if data.is_dir() else 0
    print("\nrunners demo" + (", live (A2H_LIVE=auto: on when the pipeline imports)" if live else "")
          + (f"; demo run {demo_run}" if demo_run else ""))
    print(f"{out}: {size / 1e6:.1f} MB (runs {runs_size / 1e6:.1f} MB, live pipeline + assets {code_size / 1e6:.1f} MB)."
          f" Next: hf upload <user>/<space> {out} . --repo-type space")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run folders to ship (default: outputs/ folders matching "
                    + ", ".join(DEFAULT_RUNS) + " that have an eval.json)")
    ap.add_argument("--out", default=str(REPO / "build" / "space"))
    ap.add_argument("--cors", default="http://localhost:5173", help="the frontend's origin(s), comma-separated")
    ap.add_argument("--demo-run", default="", help=f"run id featured on the home page and replayed by the demo runner "
                    f"(default: {DEFAULT_DEMO_RUN} when bundled, else the newest)")
    ap.add_argument("--live", choices=["auto", "on", "off"], default="auto",
                    help="ship the live runner's pipeline, assets and dependencies: auto = when server/ has the runner")
    ap.add_argument("--code-only", action="store_true",
                    help="no runs: server code, Dockerfile and README only (CI redeploy onto a Space that has data/)")
    ap.add_argument("--onnx-python", help="a python with torch + onnx (+ rsl_rl to verify): exports policy.onnx")
    main(ap.parse_args())
