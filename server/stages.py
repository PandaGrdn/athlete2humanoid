"""Pipeline stages and a classifier that maps experiments/latent_policy.py log lines onto them.

The same classifier drives both runners: the demo runner picks the meaningful lines out of a
finished run's job.log to replay, the colab runner uses it to follow a live job.
"""

import re

STAGES = [
    # key, label, what it does, share of a demo replay's wall-clock
    ("silhouette", "Silhouette", "SAM 3.1 segments and tracks the person", 0.12),
    ("pose", "Pose", "MediaPipe 3D landmarks per frame", 0.10),
    ("retarget", "Retarget", "mink IK onto the Unitree G1 (29 DoF)", 0.12),
    # minutes into a real run, the first thing a tester sees that is their action's own RL environment
    ("environment", "Task environment", "Key moments, contacts and the reward machine, compiled from the retarget", 0.06),
    ("latent", "Latent reward", "V-JEPA 2 embeds the clip; a reward net is distilled", 0.17),
    ("train", "Policy training", "PPO in mjlab, 4,096 parallel envs", 0.31),
    ("eval", "Evaluation", "Rollouts re-rendered and scored by V-JEPA 2", 0.12),
]
STAGE_KEYS = [s[0] for s in STAGES]

_PATTERNS = [
    ("silhouette", r"^SAM 3\.1|^prompt '.*':"),
    ("pose", r"^pose found"),
    ("retarget", r"^retarget error"),
    ("environment", r"^task spec\b"),
    ("latent", r"windows, \d+ dense steps|^target \w+: viewpoint|^distillation dataset|motions rendered \+ embedded"
               r"|^training the reward network|^epoch \d+:|^reward bundle ready"),
    ("train", r"^training (tracking|hybrid|latent)|finished with code|Learning iteration"),
    ("eval", r"^(retarget|tracking|hybrid|latent|prev_\w+)\s*: aligned|^done: "),
]
_COMPILED = [(k, re.compile(p)) for k, p in _PATTERNS]
_TS = re.compile(r"^\[\d\d:\d\d:\d\d\]\s*")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Anything that looks like a credential never leaves the server.
_SECRET = re.compile(r"(hf_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9_-]{16,}|(?i:(token|api[_-]?key|secret|password)\s*[=:]\s*)\S+)")


def redact(line):
    return _SECRET.sub("[redacted]", line)


def clean(line):
    return _ANSI.sub("", line).rstrip()


def classify(line):
    """(stage key or None, cleaned line without its timestamp)."""
    text = _TS.sub("", clean(line)).strip()
    for key, pat in _COMPILED:
        if pat.search(text):
            return key, redact(text)
    return None, redact(text)


def stage_index(key):
    return STAGE_KEYS.index(key)
