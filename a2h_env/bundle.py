"""Read a training environment (<id>.a2h-env.zip, or the folder it unzips to). Needs numpy only.

Layout (manifest.json lists every file):
    reference/qpos.npy        (T, 36) the retargeted G1 at reference/meta.json's fps: root xyz, root quaternion
                              (w, x, y, z), then the 29 joint angles in the G1's joint order
    reference/contacts.npy    (T, 2) bool: left and right foot on the floor, by the task spec's own rule
    task_spec.json            skill/keypose.py's spec: the key moments (Muse's names as muse_label), flights,
                              settles, tolerances and the reward machine's weights
    reward_config.json        the tracking task's reward terms, and feet_height when the run used it
    variants/train/index.json every training variant (mirror, tempo, amplitude, warp), kept or dropped and why;
    variants/train/<name>/    a kept one's qpos.npy, task_spec.json and src_frame.npy (the clip frame each of
                              its frames shows)
    variants/held_out/        the same for the held-out grid (tempos and amplitudes outside the training range)
    credit.json               the source video's attribution, when it has one
    extras/                   the SAM 3.1 silhouette and the V-JEPA 2 latent, when small
"""

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NQ = 36  # root (3 + 4) + 29 joints


@dataclass
class Variant:
    name: str
    params: dict
    ok: bool
    rules: list
    why: str
    frames: int
    summary: str | None = None
    qpos: np.ndarray | None = None  # (T, 36) at the reference's fps; None when dropped, or when the set is index-only
    spec: dict | None = None
    src_frame: np.ndarray | None = None  # (T,) the reference frame each frame of the variant shows


@dataclass
class Bundle:
    path: Path
    manifest: dict
    qpos: np.ndarray  # (T, 36)
    fps: float
    spec: dict
    reward_config: dict
    reference_meta: dict = field(default_factory=dict)
    contacts: np.ndarray | None = None  # (T, 2) bool
    importance: dict | None = None
    credit: dict | None = None
    train: list = field(default_factory=list)  # [Variant]
    held_out: list = field(default_factory=list)

    @property
    def id(self):
        return self.manifest.get("id")

    @property
    def clip(self):
        return self.manifest.get("clip")

    @property
    def in_place(self):
        return bool(self.reference_meta.get("in_place"))

    @property
    def duration_s(self):
        return len(self.qpos) / self.fps

    def kept(self, which="train"):
        """The variants that may be trained on (their own reference completes their spec and passes the physics
        filter) and come with their arrays."""
        return [v for v in (self.train if which == "train" else self.held_out) if v.ok and v.qpos is not None]

    def variant(self, name):
        for v in self.train + self.held_out:
            if v.name == name:
                return v
        raise KeyError(f"no variant named {name!r}; kept: {[v.name for v in self.kept()] + [v.name for v in self.kept('held_out')]}")

    def stages(self):
        """[(kind, label)] of the task spec's stages: Muse's label when it named the moment, else the compiler's."""
        return [(st.get("kind", "pose"), st.get("muse_label") or st.get("label") or st.get("name") or st.get("kind", "pose"))
                for st in self.spec.get("stages", []) if isinstance(st, dict)]


class _Source:
    """Files of a bundle, from a zip or a folder, by their manifest path."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            self.zip = None
            self.names = {str(p.relative_to(self.path)) for p in self.path.rglob("*") if p.is_file()}
        else:
            self.zip = zipfile.ZipFile(self.path)
            self.names = {n for n in self.zip.namelist() if not n.endswith("/")}
            if "manifest.json" not in self.names:  # zipped with a top-level folder
                tops = {n.split("/", 1)[0] for n in self.names if "/" in n}
                if len(tops) == 1 and f"{next(iter(tops))}/manifest.json" in self.names:
                    self.prefix = next(iter(tops)) + "/"
                    self.names = {n[len(self.prefix):] for n in self.names if n.startswith(self.prefix)}
                    return
        self.prefix = ""

    def __contains__(self, name):
        return name in self.names

    def bytes(self, name):
        if self.zip is None:
            return (self.path / name).read_bytes()
        return self.zip.read(self.prefix + name)

    def json(self, name, default=None):
        return json.loads(self.bytes(name).decode("utf-8")) if name in self else default

    def npy(self, name):
        return np.load(io.BytesIO(self.bytes(name)), allow_pickle=False) if name in self else None

    def close(self):
        if self.zip is not None:
            self.zip.close()


def _variants(src, which):
    index = src.json(f"variants/{which}/index.json")
    out = []
    for row in index or []:
        name = row["name"]
        base = f"variants/{which}/{name}/"
        v = Variant(name=name, params=dict(row.get("params") or {}), ok=bool(row.get("ok")), rules=list(row.get("rules") or []),
                    why=row.get("why") or "", frames=int(row.get("frames") or 0), summary=row.get("summary"))
        if v.ok and base + "qpos.npy" in src:
            v.qpos = np.asarray(src.npy(base + "qpos.npy"), np.float64)
            v.spec = src.json(base + "task_spec.json")
            v.src_frame = src.npy(base + "src_frame.npy")
            v.frames = len(v.qpos)
        out.append(v)
    return out


def load_bundle(path):
    """A Bundle from a .a2h-env.zip (or its unzipped folder): the reference, spec, variants and configs as arrays
    and dicts. Raises FileNotFoundError / ValueError for anything that isn't one."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    src = _Source(path)
    try:
        manifest = src.json("manifest.json")
        if not isinstance(manifest, dict) or manifest.get("format") != "a2h-env":
            raise ValueError(f"{path} is not an athlete2humanoid training environment (no manifest.json with format a2h-env)")
        qpos = src.npy("reference/qpos.npy")
        if qpos is None or qpos.ndim != 2 or qpos.shape[1] != NQ:
            raise ValueError(f"{path}: reference/qpos.npy must be (T, {NQ}), got {None if qpos is None else qpos.shape}")
        meta = src.json("reference/meta.json", {}) or {}
        fps = float(meta.get("fps") or manifest.get("reference", {}).get("fps") or 30.0)
        spec = src.json("task_spec.json")
        if not isinstance(spec, dict) or not isinstance(spec.get("stages"), list):
            raise ValueError(f"{path}: task_spec.json is missing or has no stages")
        contacts = src.npy("reference/contacts.npy")
        b = Bundle(path=path, manifest=manifest, qpos=np.asarray(qpos, np.float64), fps=fps, spec=spec,
                   reward_config=src.json("reward_config.json", {}) or {}, reference_meta=meta,
                   contacts=None if contacts is None else np.asarray(contacts, bool),
                   importance=src.json("importance.json"), credit=src.json("credit.json"),
                   train=_variants(src, "train"), held_out=_variants(src, "held_out"))
    finally:
        src.close()
    return b


def describe(b):
    """What a bundle holds, as `python -m a2h_env info` prints it."""
    m, ref = b.manifest, b.manifest.get("reference", {})
    pipe = m.get("pipeline", {})
    lines = [f"{m.get('id', '?')}: {m.get('title') or m.get('clip') or ''}".rstrip(": "),
             f"  file        {b.path.name}" + (f"  ({m['bytes'] / 1e6:.1f} MB)" if isinstance(m.get("bytes"), (int, float)) else ""),
             f"  reference   {len(b.qpos)} frames at {b.fps:g} fps ({b.duration_s:.1f} s), qpos (T, {NQ})"
             + (", run in place" if b.in_place else "")
             + (", foot contacts" if b.contacts is not None else ""),
             f"  retarget    {pipe.get('retarget_version') or ref.get('retarget_version') or '?'}",
             f"  pipeline    commit {pipe.get('commit') or '?'}; spec sha256 {str(m.get('spec_sha256', '?'))[:12]}"]
    stages = b.stages()
    poses = [label for kind, label in stages if kind == "pose"]
    flights = sum(kind == "flight" for kind, _ in stages)
    named = (b.spec.get("named_by") or {}).get("model") if isinstance(b.spec.get("named_by"), dict) else None
    lines.append(f"  task spec   {len(poses)} key moments" + (f", {flights} flight{'s' * (flights != 1)}" if flights else "")
                 + (f", named by {named}" if named else "") + ":")
    for i, label in enumerate(poses, 1):
        lines.append(f"              {i:2d}. {label}")
    for which, rows in (("train", b.train), ("held_out", b.held_out)):
        kept, with_arrays = [v for v in rows if v.ok], [v for v in rows if v.ok and v.qpos is not None]
        drops = {}
        for v in rows:
            for r in v.rules or ([] if v.ok else ["other"]):
                drops[r] = drops.get(r, 0) + 1
        secs = sum(v.frames for v in with_arrays) / b.fps
        lines.append(f"  {which:11s} {len(kept)} of {len(rows)} kept" + (f" ({secs:.0f} s of motion)" if with_arrays else "")
                     + (f"; dropped for {drops}" if drops else "") + ("" if with_arrays or not kept else "; verdicts only, no arrays"))
    rc = b.reward_config
    fh = rc.get("feet_height")
    lines.append(f"  reward      {rc.get('task', 'A2H-Tracking-Flat-Unitree-G1')}: mjlab's tracking terms"
                 + (f" + feet_height (weight {fh.get('weight')})" if isinstance(fh, dict) and fh.get("weight") else ""))
    if b.credit:
        lines.append(f"  source      {b.credit.get('text')}" + (f" ({b.credit['licence']})" if b.credit.get("licence") else ""))
    lines.append(f"  robot       {m.get('robot', {}).get('name', 'Unitree G1')}; env: {m.get('loader', {}).get('requires', 'mjlab')}")
    return "\n".join(lines)
