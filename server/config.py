"""Server settings, all from environment variables (nothing secret is ever sent to the client).

A2H_RUNS_DIR        folder of finished runs, one sub-folder per run (default: <repo>/outputs)
A2H_DATA_DIR        writable scratch for uploads and job state (default: <repo>/outputs/_web)
A2H_CORS_ORIGINS    comma-separated allowed origins (default: the Vite dev server)
A2H_RUNNERS         comma-separated enabled runners (default: demo)
A2H_ENABLE_COLAB    must be 1 as well before the colab runner is offered (local only)
A2H_DEMO_RUN        run the demo runner replays for uploads (default: newest finished run)
A2H_DEMO_SECONDS    wall-clock length of a demo replay (default: 40)
A2H_MAX_UPLOAD_MB   upload size limit (default: 200)
A2H_MAX_ACTIVE_JOBS queued + running jobs allowed at once (default: 4)
A2H_RUN_EXCLUDE     comma-separated glob patterns of run folders to hide (default: *smoke*)
A2H_STATIC_DIR      optional built frontend (web/dist) to serve at / (single-process deploys)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _list(name, default):
    return [s.strip() for s in os.environ.get(name, default).split(",") if s.strip()]


@dataclass
class Settings:
    runs_dir: Path = field(default_factory=lambda: Path(os.environ.get("A2H_RUNS_DIR", REPO / "outputs")))
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("A2H_DATA_DIR", REPO / "outputs" / "_web")))
    cors_origins: list = field(default_factory=lambda: _list("A2H_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"))
    runners: list = field(default_factory=lambda: _list("A2H_RUNNERS", "demo"))
    enable_colab: bool = field(default_factory=lambda: os.environ.get("A2H_ENABLE_COLAB") == "1")
    demo_run: str = field(default_factory=lambda: os.environ.get("A2H_DEMO_RUN", ""))
    demo_seconds: float = field(default_factory=lambda: float(os.environ.get("A2H_DEMO_SECONDS", "40")))
    max_upload_mb: float = field(default_factory=lambda: float(os.environ.get("A2H_MAX_UPLOAD_MB", "200")))
    max_active_jobs: int = field(default_factory=lambda: int(os.environ.get("A2H_MAX_ACTIVE_JOBS", "4")))
    run_exclude: list = field(default_factory=lambda: _list("A2H_RUN_EXCLUDE", "*smoke*"))
    static_dir: str = field(default_factory=lambda: os.environ.get("A2H_STATIC_DIR", ""))
    colab_iters: int = field(default_factory=lambda: int(os.environ.get("A2H_COLAB_ITERS", "2000")))

    @property
    def enabled_runners(self):
        """demo is always available; colab only with both A2H_RUNNERS and A2H_ENABLE_COLAB."""
        out = ["demo"]
        if "colab" in self.runners and self.enable_colab:
            out.append("colab")
        return out

    @property
    def max_upload_bytes(self):
        return int(self.max_upload_mb * 1024 * 1024)
