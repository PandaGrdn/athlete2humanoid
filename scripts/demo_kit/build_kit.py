"""Cut the demo video kit: one clean 1920x1080/30 fps clip per storyboard shot, then a captioned rough cut.

    python scripts/demo_kit/build_kit.py          # every shot -> outputs/demo_kit/shotNN_*.mp4 + rough_cut.mp4
    python scripts/demo_kit/build_kit.py 2 8      # rebuild shots 2 and 8, then re-assemble the rough cut
    A2H_OUTPUTS=/path/to/outputs python scripts/demo_kit/build_kit.py   # outputs/ somewhere else

SHOTS follows docs/DEMO_SCRIPT.md; docs/demo_kit/EDIT_LIST.md lists the result. Each shot is a list of segments
(source clip, trim, optional filter) plus captions (start, end, text) relative to the shot, which are burned in only
for the rough cut. Sources: the final renders (scripts/make_final_demo.py), the retarget comparison, the goal-relevance
plot, and the screen takes from scripts/demo_kit/record_takes.mjs in outputs/demo_kit/raw. A take's trims are given
relative to the actions in its .marks.json, as (mark, offset in seconds), so a re-recorded take is cut at the same moments. Needs pillow and ffmpeg (FFMPEG, or imageio-ffmpeg's); title cards use macOS system fonts.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

M = Path(__file__).resolve().parents[2]
O = Path(os.environ.get("A2H_OUTPUTS", M / "outputs"))  # a worktree can point at the main checkout's outputs
KIT = O / "demo_kit"
RAW = KIT / "raw"
WORK = KIT / "work"


def _ffmpeg():
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


FF = _ffmpeg()
MONO = "/System/Library/Fonts/SFNSMono.ttf"
SANS = "/System/Library/Fonts/SFNS.ttf"
FIT = "scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,pad=1920:1080:-1:-1:color=black"
ENC = ["-r", "30", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-color_range", "tv", "-an"]


def ff(*args):
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)


def dur(p):
    out = subprocess.run([FF, "-hide_banner", "-i", str(p)], capture_output=True, text=True).stderr
    h, m, s = out.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def card(path, lines, sub=None, foot=None):
    """A black title card in the site's style: mono kicker, big title, small footer."""
    im = Image.new("RGB", (1920, 1080), (0, 0, 0))
    d = ImageDraw.Draw(im)
    y = 380 if sub else 430
    for text, size, font, col in lines:
        f = ImageFont.truetype(font, size)
        w = d.textlength(text, font=f)
        d.text(((1920 - w) / 2, y), text, font=f, fill=col)
        y += int(size * 1.35)
    if foot:
        f = ImageFont.truetype(MONO, 26)
        for i, t in enumerate(foot):
            w = d.textlength(t, font=f)
            d.text(((1920 - w) / 2, 900 + i * 40), t, font=f, fill=(150, 150, 150))
    im.save(path)


def still(png, out, secs, zoom=False):
    if zoom:  # slow push-in on one frame
        n = int(secs * 30)
        vf = (f"scale=1800:1000:force_original_aspect_ratio=decrease:flags=lanczos,pad=1920:1080:-1:-1:color=black,"
              f"scale=3840:2160:flags=lanczos,zoompan=z='1+0.12*on/{n}':d={n}:"
              f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30")
        ff("-i", png, "-vf", vf + ",format=yuv420p", "-frames:v", n, *ENC, out)
        return
    ff("-loop", "1", "-t", secs, "-i", png, "-vf", FIT + ",fps=30,format=yuv420p", *ENC, out)


def seg(out, src, ss=0.0, to=None, vf=FIT, speed=1.0, hold=0.0, loop=0, pre=None):
    """Trim [ss, to) of src, apply vf, change speed, freeze the last frame for `hold` seconds."""
    chain = []
    if pre:
        chain.append(pre)
    chain.append(vf)
    if speed != 1.0:
        chain.append(f"setpts=PTS/{speed}")
    chain.append("fps=30")
    if hold:
        chain.append(f"tpad=stop_mode=clone:stop_duration={hold}")
    args = []
    if loop:
        args += ["-stream_loop", loop]
    args += ["-ss", ss]
    if to is not None:
        args += ["-to", to]
    ff(*args, "-i", src, "-vf", ",".join(chain) + ",format=yuv420p", *ENC, out)


def at(src, ref):
    """A trim point: seconds, or (mark, offset) resolved against the take's <name>.marks.json."""
    if not isinstance(ref, tuple):
        return ref
    mark, off = ref
    marks = dict(json.loads(Path(src).with_suffix(".marks.json").read_text())["marks"])
    if mark not in marks:
        raise SystemExit(f"{Path(src).name}: no mark {mark!r} (has {', '.join(marks)})")
    return round(marks[mark] + off, 3)


def concat(parts, out):
    lst = WORK / (Path(out).stem + ".txt")
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    ff("-f", "concat", "-safe", "0", "-i", lst, *ENC, out)



def captioned(src, caps, out):
    """Burn captions as a lower third: dark band, mono text."""
    if not caps:
        ff("-i", src, *ENC, out)
        return
    fl = []
    for i, (t0, t1, text) in enumerate(caps):
        tf = WORK / f"cap_{Path(out).stem}_{i}.txt"
        tf.write_text(text)
        en = f"between(t\\,{t0}\\,{t1})"
        fl.append(f"drawbox=x=0:y=ih-150:w=iw:h=110:color=black@0.72:t=fill:enable='{en}'")
        fl.append(f"drawtext=fontfile={MONO}:textfile='{tf}':expansion=none:fontsize=38:fontcolor=white:x=(w-text_w)/2:y=h-95-text_h/2+0:enable='{en}'")
    ff("-i", src, "-vf", ",".join(fl), *ENC, out)


# --- sources ---------------------------------------------------------------------------------------------
FINAL = O / "demo" / "final"
GRID, FROZEN, PUSH, ROBUST = (FINAL / f"{n}.mp4" for n in ("grid", "frozen_reference", "push", "robustness"))
COMPARE = O / "retarget_compare" / "squat_jump" / "compare.mp4"
GOAL = M / "docs" / "goal_relevance" / "squat_jump_hype_importance.png"
SJ, KICK, LIVE, SHOW = (RAW / f"{n}.mp4" for n in ("build_sj", "build_kick", "live_upload", "showcase"))

def cards():
    card(WORK / "title.png", [("athlete2humanoid", 120, MONO, (255, 255, 255)),
                              ("One phone video in. A humanoid that learns the move.", 40, SANS, (200, 200, 200))],
         sub=True, foot=["Meta SAM 3.1 · SAM 3D Body · V-JEPA 2 · Muse Spark → PPO in mjlab → Unitree G1"])
    card(WORK / "end.png", [("athlete2humanoid", 120, MONO, (255, 255, 255)),
                            ("Any video of a person moving can become robot training data.", 40, SANS, (200, 200, 200)),
                            ("github.com/PandaGrdn/athlete2humanoid  ·  athlete2humanoid.pages.dev", 34, MONO, (76, 166, 255))],
         sub=True, foot=["Code and weights are open", "Built with Meta SAM 3.1 · SAM 3D Body · V-JEPA 2 · Muse Spark"])
    card(WORK / "agent.png", [("Muse Spark as the reward agent", 64, MONO, (255, 255, 255)),
                              ("Jumping jacks · 3 rounds · 900 PPO iterations", 36, SANS, (200, 200, 200)),
                              ("V-JEPA 2 likeness  0.410 ± 0.008  vs  0.388 ± 0.017 without the agent", 40, MONO, (76, 166, 255)),
                              ("start 0.357 · success and falls level · one training run per arm", 30, SANS, (150, 150, 150))],
         sub=True)


# left panels of the final renders, cropped so the % labels (shot 8's numbers) stay hidden
PUSH_LEFT = "crop=880:670:40:120"
FROZEN_LEFT = "crop=660:640:0:130"
# compare.mp4 is person | uniform | proportional; label the panels
CMP_LABELS = ",".join(
    f"drawtext=fontfile={MONO}:text='{t}':fontsize=34:fontcolor=white:x={x}-text_w/2:y=120"
    for t, x in (("person", 211), ("uniform scale", 796), ("G1 proportions", 1546)))

SHOTS = [
    dict(n=1, name="hook", secs=12, segs=[dict(src=GRID, ss=0, to=9.0, speed=0.75)],
         caps=[(0.3, 5.8, "One phone video → a Unitree G1 policy in full physics"),
               (6.2, 11.8, "9/9 stages · 0 falls · no mocap, no hand-written controller")]),
    dict(n=2, name="problem", secs=18, segs=[
        dict(still=WORK / "title.png", secs=5),
        dict(src=PUSH, ss=0, to=9.0, pre=PUSH_LEFT),
        dict(src=FROZEN, ss=0, to=4.0, pre=FROZEN_LEFT)],
        caps=[(5.2, 13.8, "A clip copier (tracking policy) shoved at 1.5 m/s"),
              (14.1, 17.9, "Same copier, clip frozen: it just stands")]),
    dict(n=3, name="pipeline_build_sj", secs=18, segs=[dict(src=SJ, ss=("replay", -0.2), to=("replay", 17.8))],
         caps=[(0.5, 4.5, "SAM 3.1 finds the person"), (4.6, 8.4, "SAM 3D Body: 17 body joints shown"),
               (8.5, 11.8, "Retargeted onto the G1's 29 joints"),
               (11.9, 17.9, "Key moments named by Muse Spark: sink into squat · jump straight up")]),
    dict(n=4, name="retarget", secs=15, segs=[
        dict(src=SJ, ss=("orbit", -0.2), to=("orbit", 5.8)),
        dict(src=COMPARE, ss=0, to=9.0, pre=None, vf=FIT + "," + CMP_LABELS)],
        caps=[(0.3, 5.8, "Rebuilt on the G1's own bone lengths · IK error 0.1 cm"),
              (6.2, 14.8, "Joint-angle error 2–10× lower (elbow 13.4° → 1.3°) · feet stay planted")]),
    dict(n=5, name="goal_relevance", secs=15, segs=[dict(still=GOAL, secs=15, zoom=True)],
         caps=[(0.5, 7.0, "V-JEPA 2 style reward, weighted by what the physics needs"),
               (7.4, 14.8, "Arm-waving: floor weight 0.1 · takeoff push and landing: 1.0")]),
    dict(n=6, name="reward_machine", secs=15, segs=[
        dict(src=SJ, ss=("chip04", -0.3), to=("chip05", 0)), dict(src=SJ, ss=("scroll_timeline", -1.1), to=("end", 0))],
        caps=[(0.3, 5.1, "Each key moment is a stage of the reward"),
              (5.5, 14.8, "Reach the stages in order, own timing · a fall voids the attempt")]),
    dict(n=7, name="results_grid", secs=20, segs=[dict(src=GRID, ss=0, to=None, loop=1, hold=1.94)],
         caps=[(0.3, 9.5, "All three policies: 9/9 stages · 0 falls"),
               (9.8, 19.8, "Distance from the clip: copier 0.11 m · task 0.41 m · task, no clip 1.06 m")]),
    dict(n=8, name="robustness", secs=32, segs=[
        dict(src=FROZEN, ss=0, to=9.0, hold=1.0),
        dict(src=PUSH, ss=0, to=9.0, hold=3.0),
        dict(src=ROBUST, ss=0, to=6.0, hold=4.0)],
        caps=[(0.3, 9.8, "Clip frozen: copier 0% · task 100% · task, no clip 100%"),
              (10.2, 21.8, "Shoved at 1.5 m/s: copier 14% · ours 81%"),
              (22.2, 31.8, "18 conditions × 64 rollouts per policy")]),
    dict(n=9, name="muse", secs=15, segs=[
        dict(src=KICK, ss=("chip04", -0.2), to=("chip04", 4.8)), dict(src=KICK, ss=("timeline", -0.2), to=("timeline", 5.8)),
        dict(still=WORK / "agent.png", secs=4)],
        caps=[(0.3, 10.8, "Muse Spark watches the clip and names its key moments"),
              (11.1, 14.9, "…then tunes the reward between training rounds")]),
    dict(n=10, name="live_and_end", secs=20, segs=[
        dict(src=LIVE, ss=("upload", -0.4), to=("built", 0.3), fit=6.0),
        dict(src=KICK, ss=("push1", -0.2), to=("push1", 4.3)),
        dict(src=SHOW, ss=("push3", -0.25), to=("push3", 4.25)),
        dict(still=WORK / "end.png", secs=5)],
        caps=[(0.2, 5.9, "Live: upload a phone clip · this CPU build took 21 s (shown 3.6×)"),
              (6.1, 14.9, "Push the trained robot yourself · recorded 1 m/s recoveries")]),
]


def build(shot):
    parts = []
    for i, s in enumerate(shot["segs"]):
        p = WORK / f"s{shot['n']:02d}_{i}.mp4"
        if "still" in s:
            still(s["still"], p, s["secs"], s.get("zoom", False))
        else:
            ss, to = at(s["src"], s.get("ss", 0)), at(s["src"], s.get("to"))
            speed = (to - ss) / s["fit"] if "fit" in s else s.get("speed", 1.0)  # fit: play the trim in this many s
            seg(p, s["src"], ss, to, vf=s.get("vf", FIT), speed=speed,
                hold=s.get("hold", 0), loop=s.get("loop", 0), pre=s.get("pre"))
        parts.append(p)
    out = KIT / f"shot{shot['n']:02d}_{shot['name']}.mp4"
    tmp = WORK / f"shot{shot['n']:02d}_cat.mp4"
    concat(parts, tmp)
    ff("-i", tmp, "-t", shot["secs"], "-vf", f"tpad=stop_mode=clone:stop_duration={shot['secs']}", *ENC, out)
    return out


if __name__ == "__main__":
    only = {int(a) for a in sys.argv[1:]}
    WORK.mkdir(parents=True, exist_ok=True)
    cards()
    report = []
    capped = []
    for sh in SHOTS:
        if only and sh["n"] not in only:
            out = KIT / f"shot{sh['n']:02d}_{sh['name']}.mp4"
        else:
            out = build(sh)
        c = WORK / f"cap{sh['n']:02d}.mp4"
        if not only or sh["n"] in only or not c.exists():
            captioned(out, sh["caps"], c)
        capped.append(c)
        report.append(dict(n=sh["n"], file=out.name, secs=round(dur(out), 2), caps=sh["caps"]))
        print(f"shot {sh['n']:2d} {out.name:32s} {dur(out):6.2f}s")
    concat(capped, KIT / "rough_cut.mp4")
    total = dur(KIT / "rough_cut.mp4")
    print(f"rough_cut.mp4 {total:.2f}s (target 180)")
    (KIT / "kit.json").write_text(json.dumps(dict(shots=report, rough_cut_s=round(total, 2)), indent=1))
