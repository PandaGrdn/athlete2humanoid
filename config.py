"""Shared paths and settings. Everything can be overridden with environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Where big outputs go. In Colab, point this at Drive so nothing is lost on disconnect.
OUTPUT_DIR = Path(os.environ.get("A2H_OUTPUT_DIR", ROOT / "outputs"))
CLIPS_DIR = Path(os.environ.get("A2H_CLIPS_DIR", ROOT / "clips"))
ASSETS_DIR = Path(os.environ.get("A2H_ASSETS_DIR", ROOT / "assets"))

# Unitree G1 from MuJoCo Menagerie (fetched by scripts/fetch_assets.sh).
G1_DIR = ASSETS_DIR / "mujoco_menagerie" / "unitree_g1"
G1_SCENE_XML = G1_DIR / "scene.xml"

# MediaPipe pose model (fetched by scripts/fetch_assets.sh).
POSE_MODEL = ASSETS_DIR / "pose_landmarker_heavy.task"

# Meta Model API (OpenAI-compatible). Confirm base URL and model name at kickoff.
META_API_KEY_ENV = "META_API_KEY"
META_API_BASE = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
MUSE_MODEL = os.environ.get("MUSE_MODEL", "muse-spark-1.1")

# SAM 3.1 open weights (gated on Hugging Face: request access at huggingface.co/facebook/sam3.1).
SAM_PROMPT = os.environ.get("A2H_SAM_PROMPT", "athlete")
# SAM 3.1 pins numpy<2, so it lives in its own Python 3.12 env. Point this at that env's python
# and pipeline.py runs sam_track.py there; leave unset to import sam3 in-process.
SAM_PYTHON = os.environ.get("A2H_SAM_PYTHON")

# Secrets: one file per secret in this folder (Drive in Colab), e.g. a2h/env/HF_TOKEN containing
# HF_TOKEN = "hf_...". Works from an IDE attached to a Colab runtime, where the Colab
# secrets panel (google.colab.userdata) isn't available.
SECRETS_DIR = Path(os.environ.get("A2H_SECRETS_DIR", "/content/drive/MyDrive/a2h/env"))


def get_secret(name, required=True):
    """Look up a secret: environment variable, then SECRETS_DIR/<name>[.txt|.rtf|.txt.rtf], then
    the repo's .env file, then Colab's secrets panel. Also exports it to os.environ so
    subprocesses (SAM's own env, huggingface_hub) see it. Never prints the value."""
    value = os.environ.get(name)
    for suffix in ("", ".txt", ".rtf", ".txt.rtf"):
        path = SECRETS_DIR / f"{name}{suffix}"
        if not value and path.is_file():
            value = _read_secret_file(path, name)
    if not value and (SECRETS_DIR / f"{name}.gdoc").exists():
        raise RuntimeError(f"{SECRETS_DIR / name} is a Google Doc, which Drive can't expose as "
                           "text. Save the secret as a plain text file instead.")
    if not value and (ROOT / ".env").is_file():
        value = _read_secret_file(ROOT / ".env", name)
    if not value:
        try:
            from google.colab import userdata
            value = userdata.get(name)
        except Exception:
            pass
    if value:
        os.environ[name] = value
    elif required:
        raise RuntimeError(f"Secret {name} not found. Put it in {SECRETS_DIR / name} "
                           f'as {name} = "..." (or set the {name} environment variable).')
    return value


_QUOTES = "\"'\u201c\u201d\u2018\u2019"  # straight and curly (TextEdit "smart quotes")


def _read_secret_file(path, name):
    """Accepts `NAME = "value"`, `NAME=value`, or just the value on the first line; plain text
    or RTF (TextEdit's default format)."""
    text = path.read_text(errors="ignore")
    if path.suffix == ".rtf":
        return _secret_from_rtf(text, name)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, rest = line.partition("=")
        if sep and key.strip() == name:
            return rest.strip().strip(_QUOTES)
        if not sep and path.name.startswith(name):
            return line.strip(_QUOTES)
    return None


def _secret_from_rtf(text, name):
    """Find NAME = value inside RTF markup, skipping quote escapes like \\'93 or \\uc0\\u8220."""
    import re
    skip = r"(?:\\'[0-9a-fA-F]{2}|\\uc\d+ ?|\\u-?\d+ ?\??|[\s\"'\u201c\u201d\u2018\u2019])*"
    m = re.search(rf"{re.escape(name)}\s*={skip}([A-Za-z0-9_\-.]{{8,}})", text)
    return m.group(1) if m else None


for _d in (OUTPUT_DIR, CLIPS_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
