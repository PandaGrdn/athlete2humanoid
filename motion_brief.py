"""Muse Spark via the Meta Model API: a motion brief for the robot (phases, key joints, reward weights).

Everything here describes the motion for the G1 to learn; nothing addresses the person in the video.
For the full training spec from video, see muse/skill_reader.py.

The API is OpenAI-compatible, so we use the `openai` SDK with Meta's base URL.
The key comes from config.get_secret("MUSE_API_KEY") (or META_API_KEY): env var, a2h/env/ on Drive, or .env.
"""

import base64
import json
from pathlib import Path

from openai import OpenAI

import config


def get_api_key():
    from muse.client import api_key
    return api_key()


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
    "You segment a human motion into phases for a Unitree G1 humanoid robot to learn in simulation. "
    "You receive per-frame joint angles of the person in the video. Describe the motion, not the person. "
    "Return JSON: {\"phases\": [{\"name\", \"start_frame\", \"end_frame\", \"key_joints\", \"description\"}]}."
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
