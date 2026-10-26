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

for _d in (OUTPUT_DIR, CLIPS_DIR, ASSETS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
