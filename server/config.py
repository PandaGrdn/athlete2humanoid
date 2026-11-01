"""Server settings, all from environment variables (nothing secret is ever sent to the client).

A2H_RUNS_DIR        folder of finished runs, one sub-folder per run (default: <repo>/outputs)
A2H_DATA_DIR        writable scratch for uploads and job state (default: <repo>/outputs/_web)
A2H_CORS_ORIGINS    comma-separated allowed origins (default: the Vite dev server)
A2H_RUNNERS         comma-separated enabled runners (default: demo)
A2H_ENABLE_COLAB    must be 1 as well before the colab runner is offered (local only)
A2H_HOSTED          1 on a public deploy (the Space image sets it; a Space's SPACE_ID implies it): the colab
                    runner can't be enabled, the job list is hidden, CORS must name explicit origins
A2H_REMOTE_BACKEND  GPU backend the colab runner hands to remote_run.py: colab (default) or farmshare
A2H_DEMO_RUN        run the demo runner replays for uploads (default: newest finished run)
A2H_DEMO_SECONDS    wall-clock length of a demo replay (default: 40)
A2H_MAX_UPLOAD_MB   upload size limit, enforced on the raw request body while it streams (default: 200)
A2H_MAX_UPLOAD_STORAGE_MB  total size of stored uploads before new ones are refused (default: 2000)
A2H_MAX_ACTIVE_JOBS queued + running jobs allowed at once (default: 4)
A2H_MAX_JOBS_PER_IP queued + running jobs one client may have at once (default: 2)
A2H_RATE_JOBS       job creations per client per hour (default: 30)
A2H_RATE_REQUESTS   API requests per client per minute (default: 600)
A2H_MAX_STREAMS     open server-sent-event streams at once (default: 100)
A2H_PROXY_HOPS      trusted reverse proxies in front of the app; the client IP is taken that many entries
                    from the right of X-Forwarded-For (default: 0 = use the socket peer; the Space sets 1)
A2H_JOB_TTL_H       hours a finished job (and its upload) is kept (default: 24)
A2H_UPLOAD_TTL_H    hours an upload is kept after its job finished (default: 1)
A2H_JOB_TIMEOUT_H   a colab job still running after this many hours is cancelled (default: 8)
A2H_LIST_JOBS       1 to enable GET /api/jobs (default: on locally, off when hosted)
A2H_LOG_FORMAT      json (default) or text
A2H_LOG_LEVEL       default INFO
A2H_RUN_EXCLUDE     comma-separated glob patterns of run folders to hide (default: *smoke*)
A2H_STATIC_DIR      optional built frontend (web/dist) to serve at / (single-process deploys)
A2H_COLAB_ITERS     PPO iterations for colab jobs (default: 2000)
A2H_TRAIN_CODE      invite code that lets a visitor train a live build's environment on a Modal GPU (the modal
                    runner, server/train.py); with A2H_RUNNERS=demo,modal and a Modal token in the environment
A2H_TRAIN_PER_DAY   trainings a day allowed without the code (default: 0 = code only)
A2H_TRAIN_ITERS     PPO iterations per training (default: 1000, ~30 min on an L40S)
A2H_TRAIN_ATTEMPT_EVERY  iterations between the attempts the page plays (default: 50)
A2H_TRAIN_NUM_ENVS  parallel environments (default: 4096)
A2H_TRAIN_TIMEOUT_H a training still running after this many hours is cancelled (default: 3)
A2H_TRAIN_BRANCH    the repository branch the GPU job runs (default: main)
A2H_TRAIN_CONCURRENCY  trainings at once (default: 1; each one is a GPU)
A2H_RATE_TRAIN      POST /api/jobs/{id}/train per client per hour (default: 12; bounds invite-code guessing)
A2H_STORE           where finished builds and trainings are kept past a restart (server/store.py): unset = files
                    under A2H_DATA_DIR/persist; gcs://<bucket>[/<prefix>] = a Cloud Storage bucket
"""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNERS = {"demo", "colab", "modal"}
LIVE_POSE = {"cpu", "modal"}
BACKENDS = {"colab", "farmshare", "modal"}


class ConfigError(ValueError):
    """Invalid settings; the message lists every problem, by variable name."""


_errors = []  # parse errors collected while a Settings is built, reported by validate()


def _env(name, default):
    return os.environ.get(name, default)


def _list(name, default):
    return [s.strip() for s in _env(name, default).split(",") if s.strip()]


def _num(name, default, kind=float):
    raw = _env(name, str(default))
    try:
        return kind(raw)
    except ValueError:
        _errors.append(f"{name} must be {'an integer' if kind is int else 'a number'}, got {raw!r}")
        return kind(default)


def _flag(name, default=False):
    raw = _env(name, "")
    return default if raw == "" else raw.lower() in ("1", "true", "yes", "on")


def _hosted():
    return _flag("A2H_HOSTED") or bool(os.environ.get("SPACE_ID"))


@dataclass
class Settings:
    runs_dir: Path = field(default_factory=lambda: Path(_env("A2H_RUNS_DIR", REPO / "outputs")))
    data_dir: Path = field(default_factory=lambda: Path(_env("A2H_DATA_DIR", REPO / "outputs" / "_web")))
    cors_origins: list = field(default_factory=lambda: _list("A2H_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"))
    runners: list = field(default_factory=lambda: _list("A2H_RUNNERS", "demo"))
    enable_colab: bool = field(default_factory=lambda: _env("A2H_ENABLE_COLAB", "") == "1")
    hosted: bool = field(default_factory=_hosted)
    remote_backend: str = field(default_factory=lambda: _env("A2H_REMOTE_BACKEND", "colab"))
    demo_run: str = field(default_factory=lambda: _env("A2H_DEMO_RUN", ""))
    demo_seconds: float = field(default_factory=lambda: _num("A2H_DEMO_SECONDS", 40))
    max_upload_mb: float = field(default_factory=lambda: _num("A2H_MAX_UPLOAD_MB", 200))
    max_upload_storage_mb: float = field(default_factory=lambda: _num("A2H_MAX_UPLOAD_STORAGE_MB", 2000))
    max_active_jobs: int = field(default_factory=lambda: _num("A2H_MAX_ACTIVE_JOBS", 4, int))
    max_jobs_per_ip: int = field(default_factory=lambda: _num("A2H_MAX_JOBS_PER_IP", 2, int))
    rate_jobs_per_hour: float = field(default_factory=lambda: _num("A2H_RATE_JOBS", 30))
    rate_requests_per_min: float = field(default_factory=lambda: _num("A2H_RATE_REQUESTS", 600))
    max_streams: int = field(default_factory=lambda: _num("A2H_MAX_STREAMS", 100, int))
    proxy_hops: int = field(default_factory=lambda: _num("A2H_PROXY_HOPS", 0, int))
    job_ttl_h: float = field(default_factory=lambda: _num("A2H_JOB_TTL_H", 24))
    upload_ttl_h: float = field(default_factory=lambda: _num("A2H_UPLOAD_TTL_H", 1))
    job_timeout_h: float = field(default_factory=lambda: _num("A2H_JOB_TIMEOUT_H", 8))
    list_jobs: bool | None = field(default_factory=lambda: _flag("A2H_LIST_JOBS", None))
    log_format: str = field(default_factory=lambda: _env("A2H_LOG_FORMAT", "json"))
    log_level: str = field(default_factory=lambda: _env("A2H_LOG_LEVEL", "INFO").upper())
    run_exclude: list = field(default_factory=lambda: _list("A2H_RUN_EXCLUDE", "*smoke*"))
    static_dir: str = field(default_factory=lambda: _env("A2H_STATIC_DIR", ""))
    colab_iters: int = field(default_factory=lambda: _num("A2H_COLAB_ITERS", 2000, int))
    # The approach that has worked best: the task policy on the video's keypose spec (docs/KEYPOSE_SPEC.md),
    # built off a finished run's weights ("" trains from scratch).
    # The live runner (server/live.py): builds an uploaded video's task environment on this host's CPU.
    live_mode: str = field(default_factory=lambda: _env("A2H_LIVE", "auto"))  # auto | 1 | 0
    live_python: str = field(default_factory=lambda: _env("A2H_LIVE_PYTHON", sys.executable))
    live_max_seconds: float = field(default_factory=lambda: _num("A2H_LIVE_MAX_SECONDS", 20))
    live_timeout_s: float = field(default_factory=lambda: _num("A2H_LIVE_TIMEOUT_S", 240))
    live_concurrency: int = field(default_factory=lambda: _num("A2H_LIVE_CONCURRENCY", 1, int))
    live_memory_mb: int = field(default_factory=lambda: _num("A2H_LIVE_MEMORY_MB", 8192, int))
    # Muse names each live upload's key moments (muse/moment_namer.py, ~10 s and ~1 cent a clip). Off by
    # default: it needs MUSE_API_KEY in this process's environment, and spends it per visitor upload.
    live_muse_names: bool = field(default_factory=lambda: _env("A2H_MUSE_NAMES", "0") == "1")
    # Where the live runner's pose runs: "cpu" is MediaPipe here; "modal" sends the clip through SAM 3.1 and SAM 3D
    # Body on the deployed Modal GPU app (server/live_pose.py, frontends/live_pose_modal.py; MODAL_TOKEN_ID and
    # MODAL_TOKEN_SECRET must be in this process's environment; ~$0.05-0.15 a clip), MediaPipe if that fails.
    live_pose: str = field(default_factory=lambda: _env("A2H_LIVE_POSE", "cpu"))
    live_ok: bool = field(default=False, repr=False)  # set at startup by server.live.available()
    # The modal runner (server/train.py): a live build's environment trained on a Modal GPU, gated by an invite code
    # and/or a daily allowance, never on a host without a Modal token.
    train_code: str = field(default_factory=lambda: _env("A2H_TRAIN_CODE", ""), repr=False)
    train_per_day: int = field(default_factory=lambda: _num("A2H_TRAIN_PER_DAY", 0, int))
    train_iters: int = field(default_factory=lambda: _num("A2H_TRAIN_ITERS", 1000, int))
    train_attempt_every: int = field(default_factory=lambda: _num("A2H_TRAIN_ATTEMPT_EVERY", 50, int))
    train_num_envs: int = field(default_factory=lambda: _num("A2H_TRAIN_NUM_ENVS", 4096, int))
    train_timeout_h: float = field(default_factory=lambda: _num("A2H_TRAIN_TIMEOUT_H", 3))
    train_branch: str = field(default_factory=lambda: _env("A2H_TRAIN_BRANCH", "main"))
    train_concurrency: int = field(default_factory=lambda: _num("A2H_TRAIN_CONCURRENCY", 1, int))
    train_poll_s: float = field(default_factory=lambda: _num("A2H_TRAIN_POLL_S", 30))  # how often the GPU job is read
    rate_train_per_hour: float = field(default_factory=lambda: _num("A2H_RATE_TRAIN", 12))
    modal_token: bool = field(default_factory=lambda: bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")))
    train_backend: object = field(default=None, repr=False)  # tests: a stand-in for server/train.py's ModalBackend
    # Durable storage for what visitors made (server/store.py): local files by default, a GCS bucket when hosted.
    store_url: str = field(default_factory=lambda: _env("A2H_STORE", ""))
    store_backend: object = field(default=None, repr=False)  # tests: a Store to use instead
    colab_policies: list = field(default_factory=lambda: _list("A2H_COLAB_POLICIES", "task"))
    colab_init: str = field(default_factory=lambda: _env("A2H_COLAB_INIT", str(REPO / "outputs" / "remote_latent_final")))
    parse_errors: list = field(default_factory=lambda: _errors[:], repr=False)

    def __post_init__(self):
        _errors.clear()

    @property
    def enabled_runners(self):
        """demo is always available; colab only with both A2H_RUNNERS and A2H_ENABLE_COLAB, and never hosted."""
        out = ["demo"] + (["live"] if self.live_ok else [])
        if "colab" in self.runners and self.enable_colab and not self.hosted:
            out.append("colab")
        if "modal" in self.runners and self.modal_token and self.live_ok:  # trains live builds: needs both
            out.append("modal")
        return out

    @property
    def train_open(self):
        """Whether anyone can start a training: the code, or a daily allowance, is set."""
        return bool(self.train_code) or self.train_per_day > 0

    @property
    def job_list_enabled(self):
        return (not self.hosted) if self.list_jobs is None else self.list_jobs

    @property
    def max_upload_bytes(self):
        return int(self.max_upload_mb * 1024 * 1024)

    def validate(self):
        """Raise ConfigError naming every bad variable. Returns warnings (fine, but worth logging)."""
        errs, warns = list(self.parse_errors), []
        for name, v, lo in [("A2H_DEMO_SECONDS", self.demo_seconds, 0.01), ("A2H_MAX_UPLOAD_MB", self.max_upload_mb, 0.0001),
                            ("A2H_MAX_UPLOAD_STORAGE_MB", self.max_upload_storage_mb, 0),
                            ("A2H_MAX_ACTIVE_JOBS", self.max_active_jobs, 1), ("A2H_MAX_JOBS_PER_IP", self.max_jobs_per_ip, 1),
                            ("A2H_RATE_JOBS", self.rate_jobs_per_hour, 0.01), ("A2H_RATE_REQUESTS", self.rate_requests_per_min, 1),
                            ("A2H_MAX_STREAMS", self.max_streams, 1), ("A2H_PROXY_HOPS", self.proxy_hops, 0),
                            ("A2H_JOB_TTL_H", self.job_ttl_h, 0.001), ("A2H_UPLOAD_TTL_H", self.upload_ttl_h, 0.0),
                            ("A2H_JOB_TIMEOUT_H", self.job_timeout_h, 0.01), ("A2H_COLAB_ITERS", self.colab_iters, 1),
                            ("A2H_LIVE_MAX_SECONDS", self.live_max_seconds, 1), ("A2H_LIVE_TIMEOUT_S", self.live_timeout_s, 1),
                            ("A2H_LIVE_CONCURRENCY", self.live_concurrency, 1), ("A2H_LIVE_MEMORY_MB", self.live_memory_mb, 0),
                            ("A2H_TRAIN_PER_DAY", self.train_per_day, 0), ("A2H_TRAIN_ITERS", self.train_iters, 1),
                            ("A2H_TRAIN_ATTEMPT_EVERY", self.train_attempt_every, 1), ("A2H_TRAIN_NUM_ENVS", self.train_num_envs, 1),
                            ("A2H_TRAIN_TIMEOUT_H", self.train_timeout_h, 0.01), ("A2H_TRAIN_CONCURRENCY", self.train_concurrency, 1),
                            ("A2H_RATE_TRAIN", self.rate_train_per_hour, 1)]:
            if v < lo:
                errs.append(f"{name} must be >= {lo:g}, got {v:g}")
        if unknown := sorted(set(self.runners) - RUNNERS):
            errs.append(f"A2H_RUNNERS: unknown runner(s) {unknown}; choose from {sorted(RUNNERS)}")
        if self.remote_backend not in BACKENDS:
            errs.append(f"A2H_REMOTE_BACKEND must be one of {sorted(BACKENDS)}, got {self.remote_backend!r}")
        if self.live_mode not in ("auto", "0", "1"):
            errs.append(f"A2H_LIVE must be auto, 0 or 1, got {self.live_mode!r}")
        if self.live_pose not in LIVE_POSE:
            errs.append(f"A2H_LIVE_POSE must be one of {sorted(LIVE_POSE)}, got {self.live_pose!r}")
        elif self.live_pose == "modal" and not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
            warns.append("A2H_LIVE_POSE=modal but MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are not set: live builds use "
                         "MediaPipe on the CPU")
            self.live_pose = "cpu"
        if self.log_format not in ("json", "text"):
            errs.append(f"A2H_LOG_FORMAT must be json or text, got {self.log_format!r}")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            errs.append(f"A2H_LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR, got {self.log_level!r}")
        for o in self.cors_origins:
            if o == "*":
                (errs if self.hosted else warns).append("A2H_CORS_ORIGINS='*' allows any site to call the API; "
                                                        "name the frontend origin(s)")
            elif not (o.startswith("https://") or o.startswith("http://localhost") or o.startswith("http://127.0.0.1")):
                warns.append(f"A2H_CORS_ORIGINS: {o!r} is neither https nor localhost")
        if self.store_url and not re.fullmatch(r"gcs://[a-z0-9][a-z0-9._-]{1,220}[a-z0-9](/[\w./-]*)?", self.store_url):
            errs.append(f"A2H_STORE must be unset or gcs://<bucket>[/<prefix>], got {self.store_url!r}")
        if not re.fullmatch(r"[A-Za-z0-9._/-]+", self.train_branch):
            errs.append(f"A2H_TRAIN_BRANCH: odd branch name {self.train_branch!r}")
        if "modal" in self.runners:
            if not self.modal_token:
                warns.append("A2H_RUNNERS has modal but MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are not set: no GPU training")
            elif not self.train_open:
                warns.append("A2H_RUNNERS has modal but neither A2H_TRAIN_CODE nor A2H_TRAIN_PER_DAY is set: nobody can "
                             "start a training")
        if self.hosted and ("colab" in self.runners or self.enable_colab):
            warns.append("colab runner requested on a hosted deploy; it stays disabled (A2H_HOSTED / SPACE_ID)")
        if self.hosted and self.proxy_hops == 0:
            warns.append("A2H_PROXY_HOPS=0 on a hosted deploy: every client shares the proxy's IP, so per-IP limits "
                         "act as global limits")
        if errs:
            raise ConfigError("invalid configuration:\n  " + "\n  ".join(errs))
        return warns
