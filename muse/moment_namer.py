"""Moment namer: Muse Spark watches the clip and names the key moments the keypose compiler found.

    python -m muse.moment_namer outputs/remote_latent_kp_kick           # names its task_spec.json in place
    python -m muse.moment_namer data/uploads/<job>/live --dry-run        # a live job's spec, no API call

skill/keypose.py picks the key moments from the retarget's statistics and names each one by what
changed most ("right foot up, right foot forward"). Muse gets the video, the frame at each moment
(captioned with its number and time) and that measured change, and returns a short name and one
sentence per moment, plus the action as a whole. The names are for people: the reward never reads
them, so naming a spec doesn't change what a policy trained on it learns. The measured name is kept
as "measured"; renaming a spec again starts from it.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from muse import media  # noqa: E402
from muse.client import image_part, make_client, now, read_json, text_part, video_part, write_json  # noqa: E402
from muse.schema import INT, STR, arr, obj  # noqa: E402

NAMES_SCHEMA = obj(
    action=obj(label=STR, description=STR),
    moments=arr(obj(index=INT, name=STR, description=STR)),
)

SYSTEM = (
    "You label the key moments of a motion that a Unitree G1 humanoid robot learns in a physics simulator "
    "from a video of a person doing it. The moments were found by measuring the motion; you name them so "
    "people can follow the robot's task. Describe the motion only: never address or advise the person, "
    "never judge their technique. Answer with JSON only."
)

MAX_NAME_WORDS, MAX_NAME_CHARS, MAX_DESC_CHARS = 5, 40, 240
VIDEO_NAMES = ("processed.mp4",)  # a live job's own transcoded upload


def moments(spec):
    """The spec's pose stages, in order: the ones Muse names."""
    return [st for st in spec["stages"] if st["kind"] == "pose"]


def moment_time(st, fps):
    return st["frame"] / fps if st.get("frame") is not None else st["end_s"]


def find_video(folder, spec, video=None):
    """The clip to show Muse: --video, a live job's processed.mp4, clips/<spec name>.*, or the run's side-by-side."""
    if video:
        return Path(video)
    folder = Path(folder)
    for n in VIDEO_NAMES:
        if (folder / n).is_file():
            return folder / n
    name = spec.get("name") or ""
    for ext in (".mov", ".mp4", ".MOV", ".MP4"):
        p = config.ROOT / "clips" / f"{name}{ext}"
        if name and p.is_file():
            return p
    p = folder / "clips" / name / "side_by_side.mp4"
    return p if p.is_file() else None


def prompt(spec, fps, sample_fps, side_by_side=False):
    lines = [f"The video is sampled at {sample_fps:g} frames per second; it is {spec['duration_s']:.2f} s long."]
    if side_by_side:
        lines.append("Each frame shows the person on the left and the robot replaying their motion on the right.")
    lines += ["", "Key moments found in the motion (times from the start of the clip). The image shows the video's "
                  "frame at each one, captioned with its number. 'Measured' is what changed most since the "
                  "previous moment, by body position, in the body's own frame: left and right are the person's "
                  "own sides and forward is the way their hips face. The camera may be at an angle to the body, "
                  "so take sides and directions from 'measured', not from the image."]
    for i, st in enumerate(moments(spec), 1):
        need = "optional" if st.get("optional") else "required"
        measured = st.get("measured", st["name"]).removesuffix(" (finish)")
        lines.append(f"{i}. {moment_time(st, fps):.2f} s, {need}. Measured: {measured}.")
    flights = [st for st in spec["stages"] if st["kind"] == "flight"]
    if flights:
        lines.append("Both feet leave the ground (jumps): " + ", ".join(f"{st['start_s']:.2f}-{st['end_s']:.2f} s"
                                                                       for st in flights) + ".")
    lines += [
        "",
        "For each moment, by its number:",
        f"- name: the pose reached at that moment, 1-{MAX_NAME_WORDS - 1} words, lower case, e.g. 'knee chambered', "
        "'kick extended', 'deep crouch', 'arms overhead'. Say which side when one side moves.",
        "- description: one sentence on what the body does from the previous moment to this one.",
        "Then the action as a whole: label (short, snake_case) and description (one sentence).",
    ]
    return "\n".join(lines)


def check(value, n):
    """Checks beyond the schema -> list of problems."""
    errors = []
    idx = [m["index"] for m in value["moments"]]
    if sorted(idx) != list(range(1, n + 1)):
        errors.append(f"moments: exactly one entry per moment, index 1 to {n} (got {idx})")
    for m in value["moments"]:
        name, desc = m["name"].strip(), m["description"].strip()
        if not name or len(name.split()) > MAX_NAME_WORDS or len(name) > MAX_NAME_CHARS:
            errors.append(f"moments[{m['index']}].name: 1-{MAX_NAME_WORDS} words, at most {MAX_NAME_CHARS} characters")
        if not desc or len(desc) > MAX_DESC_CHARS:
            errors.append(f"moments[{m['index']}].description: one sentence, at most {MAX_DESC_CHARS} characters")
    texts = [value["action"]["description"]] + [m["name"] + " " + m["description"] for m in value["moments"]]
    if any(re.search(r"\b(you|your)\b", t, re.I) for t in texts):
        errors.append("describe the motion; don't address the person (no 'you'/'your')")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", value["action"]["label"]):
        errors.append("action.label: snake_case, at most 40 characters")
    return errors


def apply_names(spec, value, info):
    """Muse's answer -> the spec: each pose stage's name, label and description; the measured name kept."""
    by_index = {m["index"]: m for m in value["moments"]}
    for i, st in enumerate(moments(spec), 1):
        m = by_index[i]
        st.setdefault("measured", st["name"])
        st["name"] = st["label"] = m["name"].strip()
        st["description"] = m["description"].strip()
    if spec["stages"][-1]["kind"] == "pose":  # the end pose is the finish (keypose._compile)
        spec["stages"][-1]["name"] = spec["stages"][-1]["label"] = spec["stages"][-1]["name"] + " (finish)"
    spec["action"] = value["action"]["label"]
    spec["action_description"] = value["action"]["description"].strip()
    spec["summary"] = summarize(spec["stages"])
    spec["named_by"] = {"created": now(), **info}
    return spec


def summarize(stages):
    """skill.keypose.summarize, kept here so muse/ runs without torch (tests check they agree)."""
    poses = [s for s in stages if s["kind"] == "pose"]
    jumps = sum(s["kind"] == "flight" for s in stages)
    head = f"{len(poses)} key moment{'s' * (len(poses) != 1)}" + (f", {jumps} jump{'s' * (jumps != 1)}" if jumps else "")
    return head + ": " + " -> ".join(s["name"] for s in stages)


def name_moments(spec_path, client, video=None, sample_fps=4, max_side=480, out=None):
    """Muse names the moments of `spec_path` (a keypose task_spec.json) -> the renamed spec, written to out."""
    spec_path = Path(spec_path)
    spec = read_json(spec_path)
    if not spec or spec.get("type") != "keypose":
        raise ValueError(f"{spec_path} is not a keypose task spec")
    for st in moments(spec):  # renaming starts from the measured names
        st["name"] = st.get("measured", st["name"])
    poses = moments(spec)
    if not poses:
        raise ValueError(f"{spec_path} has no key moments to name")
    src = find_video(spec_path.parent, spec, video)
    if src is None:
        raise FileNotFoundError(f"no video for {spec_path}: pass --video")
    fps = spec["fps"]
    clip = media.prepare_video(src, spec_path.parent / "muse_names_input.mp4", max_side, required=client.live)
    sheet = media.frames_at(src, [moment_time(st, fps) for st in poses],
                            [f"{i}  {moment_time(st, fps):.2f} s" for i, st in enumerate(poses, 1)])
    content = [text_part(prompt(spec, fps, sample_fps, side_by_side=src.name == "side_by_side.mp4")),
               video_part(clip, sample_fps)]
    if sheet:
        content.append(image_part(sheet))
    value, info = client.ask_json("moment_names", SYSTEM, content, NAMES_SCHEMA,
                                  check=lambda v: check(v, len(poses)), effort="low")
    apply_names(spec, value, info)
    out = Path(out or spec_path)
    write_json(out, spec)
    if not client.live:
        client.save_requests(out.with_name("moment_names.request.json"))
    print(f"{out}: {spec['action']}: {spec['summary']} ({info['mode']})")
    return spec


def spec_file(path):
    path = Path(path)
    return path if path.suffix == ".json" else path / "task_spec.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("specs", nargs="+", help="keypose task_spec.json files, or folders holding one")
    ap.add_argument("--video", help="the clip (one spec only); default: found next to the spec or in clips/")
    ap.add_argument("--out", help="write here instead of in place (one spec only)")
    ap.add_argument("--fps", type=float, default=4, help="frames per second Muse samples")
    ap.add_argument("--max-side", type=int, default=480, help="downscale the video to this many px first")
    ap.add_argument("--dry-run", action="store_true", help="no API call: bundled fixture response")
    ap.add_argument("--fixture", help="no API call: responses from this recorded file")
    ap.add_argument("--record", help="save live exchanges here (they work as --fixture files)")
    ap.add_argument("--model")
    a = ap.parse_args(argv)
    if (a.video or a.out) and len(a.specs) > 1:
        ap.error("--video and --out take one spec")
    for s in a.specs:
        client = make_client(a.dry_run, a.fixture, "moment_names_front_kick.json", a.model, a.record)
        name_moments(spec_file(s), client, a.video, a.fps, a.max_side, a.out)


if __name__ == "__main__":
    main()
