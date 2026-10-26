"""Muse Spark via the Meta Model API: phases, coaching notes, reward weights.

The API is OpenAI-compatible, so we use the `openai` SDK with Meta's base URL.
Set META_API_KEY (Colab: the key icon in the left sidebar -> add a secret named META_API_KEY).
"""

import base64
import json
import os
from pathlib import Path

from openai import OpenAI

import config


def get_api_key():
    key = os.environ.get(config.META_API_KEY_ENV)
    if not key:
        try:  # Colab secrets
            from google.colab import userdata
            key = userdata.get(config.META_API_KEY_ENV)
        except Exception:
            pass
    if not key:
        raise RuntimeError(f"Set {config.META_API_KEY_ENV} (env var or Colab secret).")
    return key


_client = None


def client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_api_key(), base_url=config.META_API_BASE)
    return _client


def ask(prompt, system=None, images=(), json_mode=False, **kwargs):
    """One Muse Spark call. images: file paths (PNG/JPG) sent inline as base64 data URLs."""
    content = [{"type": "text", "text": prompt}]
    for img in images:
        mime = "image/png" if str(img).lower().endswith(".png") else "image/jpeg"
        b64 = base64.b64encode(Path(img).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": content if images else prompt}
    ]
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client().chat.completions.create(model=config.MUSE_MODEL, messages=messages, **kwargs)
    text = resp.choices[0].message.content
    return json.loads(text) if json_mode else text


# ---- Day 4+ ---------------------------------------------------------------------------

PHASE_SYSTEM = (
    "You are a biomechanics coach. You receive per-frame joint angles of an athlete. "
    "Return JSON: {\"phases\": [{\"name\", \"start_frame\", \"end_frame\", \"key_joints\", \"coaching\"}]}."
)


def segment_phases(joint_angle_summary: str, frames=()):
    """Day 4: label phases (approach, takeoff, flight, landing...) with frame ranges."""
    return ask(joint_angle_summary, system=PHASE_SYSTEM, images=frames, json_mode=True)


def reward_weights(phases, training_report=None):
    """Day 7/8: per-phase imitation reward weights; with a training report, propose revisions."""
    raise NotImplementedError("Day 7")


def explain_infeasibility(tracking_errors_summary: str):
    """Day 6: explain why the robot can't follow the motion (balance, timing, joint limits)."""
    raise NotImplementedError("Day 6")
