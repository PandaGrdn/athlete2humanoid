"""Assemble the backend + a slim copy of finished runs into a folder that is a Hugging Face Space.

    python3 scripts/build_space.py --cors https://athlete2humanoid.pages.dev \
        --onnx-python <a python with torch + onnx>        # optional: ships ONNX policies
    hf upload <user>/athlete2humanoid-api build/space . --repo-type space

Only what the web app shows is copied (videos, metrics, the latest checkpoint per policy, motion
CSVs, the reward net); training logs become a small curve.json and job.log keeps only the lines the
demo runner replays. No .env, tokens or raw clips are ever copied. See docs/DEPLOY.md.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from server import runs as R  # noqa: E402

RUN_FILES = ["eval.json", "status.json", "distill_report.json", "comparison.mp4", "reward.pt", "web.json", "poster.jpg"]
POLICY_FILES = ["policy.mp4", "policy_metrics.json", "policy.onnx"]
CLIP_FILES = ["meta.json", "retarget_report.json", "motion.csv", "side_by_side.mp4", "silhouette.mp4"]

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
This Space runs on CPU, so jobs use the demo runner, which replays recorded A100 runs. Endpoints:
`/api/health`, `/api/runs`, `/api/runs/{id}`, `/api/jobs` (see `/docs`).
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
COPY server server
COPY data data
# A2H_HOSTED: the colab runner can't be enabled and the job list is off, whatever Space variables say.
ENV A2H_HOSTED=1 \\
    A2H_RUNS_DIR=/app/data/runs \\
    A2H_DATA_DIR=/tmp/a2h \\
    A2H_RUNNERS=demo \\
    A2H_MAX_UPLOAD_MB=100 \\
    A2H_MAX_UPLOAD_STORAGE_MB=2000 \\
    A2H_PROXY_HOPS=1 \\
    A2H_CORS_ORIGINS="{cors}"{demo_run}
USER 1000
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \\
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4)"]
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log", "--no-server-header", \\
     "--limit-concurrency", "200", "--timeout-keep-alive", "15", "--timeout-graceful-shutdown", "10"]
"""
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


def check_origins(cors):
    """The Space's CORS list ends up in the image: https origins (or localhost) only, never '*'."""
    bad = [o for o in cors.split(",") if o == "*" or not re.fullmatch(r"https://[^/\s\"]+|http://(localhost|127\.0\.0\.1)(:\d+)?", o)]
    if bad:
        raise SystemExit(f"--cors: {bad} must be https origins (scheme://host, no path), comma-separated")


def main(a):
    check_origins(a.cors)
    src_root = REPO / "outputs"
    if a.code_only:  # CI redeploys: code, Dockerfile and README only; the Space keeps its data/ folder
        picked = []
    else:
        picked = [Path(p).resolve() for p in a.runs] if a.runs else list(R.find_runs(src_root, ["*smoke*", "_*"]).values())
        if not picked:
            raise SystemExit("no finished runs (folders with eval.json) to ship")
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
    demo = f" \\\n    A2H_DEMO_RUN={a.demo_run}" if a.demo_run else ""
    (out / "Dockerfile").write_text(DOCKERFILE.format(base=BASE_IMAGE, cors=a.cors, demo_run=demo))
    (out / ".dockerignore").write_text(DOCKERIGNORE)
    (out / "README.md").write_text(README)
    (out / ".gitattributes").write_text("*.mp4 filter=lfs diff=lfs merge=lfs -text\n*.pt filter=lfs diff=lfs merge=lfs -text\n"
                                       "*.onnx filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\n{out}: {size / 1e6:.1f} MB. Next: hf upload <user>/<space> {out} . --repo-type space")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run folders to ship (default: every outputs/remote_* with eval.json)")
    ap.add_argument("--out", default=str(REPO / "build" / "space"))
    ap.add_argument("--cors", default="http://localhost:5173", help="the frontend's origin(s), comma-separated")
    ap.add_argument("--demo-run", default="", help="run id the demo runner replays for uploads (default: newest)")
    ap.add_argument("--code-only", action="store_true",
                    help="no runs: server code, Dockerfile and README only (CI redeploy onto a Space that has data/)")
    ap.add_argument("--onnx-python", help="a python with torch + onnx (+ rsl_rl to verify): exports policy.onnx")
    main(ap.parse_args())
