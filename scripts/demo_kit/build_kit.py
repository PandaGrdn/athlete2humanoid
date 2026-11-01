"""Cut the demo video kit: one clean 1920x1080/30 fps clip per storyboard shot, then a captioned rough cut.

    python scripts/demo_kit/build_kit.py          # every shot -> outputs/demo_kit/shotNN_*.mp4 + rough_cut.mp4
    python scripts/demo_kit/build_kit.py 2 8      # rebuild shots 2 and 8, then re-assemble the rough cut
    A2H_OUTPUTS=/path/to/outputs python scripts/demo_kit/build_kit.py   # outputs/ somewhere else

SHOTS follows docs/DEMO_SCRIPT.md; docs/demo_kit/EDIT_LIST.md lists the result. Each shot is a list of segments
(source clip, trim, optional filter) plus captions (start, end, text) relative to the shot, which are burned in only
for the rough cut. Sources: the screen takes from scripts/demo_kit/record_takes.mjs in outputs/demo_kit/raw (the build
view of the lakeside jog and the front kick, the training-environment download, a live upload), the 1080p policy renders
from scripts/demo_kit/render_policies.py in outputs/demo_kit/src, the jog's source video (clips_pro/), the adaptation
chart and cards. A take's trims are given relative to the actions in its .marks.json, as (mark, offset in seconds), so a re-recorded take is cut at the same moments. Needs pillow and ffmpeg (FFMPEG, or imageio-ffmpeg's); title cards use macOS system fonts.
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
SRC = KIT / "src"  # 1080p policy renders (scripts/demo_kit/render_policies.py)
CLIPS_PRO = Path(os.environ.get("A2H_CLIPS_PRO", O.parent / "clips_pro"))  # the downloaded source videos (gitignored)
JOG_VIDEO = CLIPS_PRO / "athletes" / "4065467-uhd_2560_1440_30fps.mp4"  # docs/real_athletes/clips.json: trim 0-6 s
JOG_CREDIT = "Video: Pavel Danilyuk, Pexels (Pexels License)"
ADAPT = M / "docs" / "results" / "adaptation" / "adaptation_curves.png"
ENV_ZIP = RAW / "latent_ra_jog.a2h-env.zip"  # saved by the env_jog take
JOG, KICK, ENV, LIVE = (RAW / f"{n}.mp4" for n in ("build_jog", "build_kick", "env_jog", "live_kick"))
WHITE, GREY, DIM, BLUE = (255, 255, 255), (200, 200, 200), (150, 150, 150), (76, 166, 255)


def term_card(path, title, lines):
    """A terminal-style card: a mono title, then left-aligned lines; '$ ' lines in white, the rest grey."""
    im = Image.new("RGB", (1920, 1080), (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((120, 110), title, font=ImageFont.truetype(MONO, 30), fill=DIM)
    f = ImageFont.truetype(MONO, 30)
    y = 190
    for line in lines:
        col = WHITE if line.startswith("$ ") else BLUE if line.startswith("  set_param") or line.startswith("  train_") else GREY
        d.text((120, y), line, font=f, fill=col)
        y += 44
    im.save(path)


def env_info():
    """`python -m a2h_env info` on the downloaded bundle: the real output, trimmed to fit a card."""
    out = subprocess.run([sys.executable, "-m", "a2h_env", "info", str(ENV_ZIP)], cwd=M, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    keep, moments = [], 0
    for line in out:
        if line.strip()[:3].rstrip(".").strip().isdigit():  # the key-moment list: the first three, then a count
            moments += 1
            if moments == 4:
                keep.append("               …")
            if moments > 3:
                continue
        keep.append(line if len(line) <= 96 else line[:95] + "…")
    return keep


def cards():
    (WORK / "jog_credit.txt").write_text(JOG_CREDIT)
    card(WORK / "problem.png", [("How a humanoid learns a new move today", 56, MONO, WHITE),
                                ("a motion-capture studio, or a hand-written reward", 40, SANS, GREY),
                                ("What everyone has: a phone video of a person", 44, MONO, BLUE)], sub=True,
         foot=["athlete2humanoid · video in, robot policy out"])
    card(WORK / "speed.png", [("18 minutes", 150, MONO, WHITE),
                              ("from a clip it had never seen to a policy that does it, on one L40S", 40, SANS, GREY)],
         sub=True, foot=["from scratch: 24 min · punches and jumping jacks: not learned in 30 min",
                         "docs/FRONTIER_ADAPTATION.md"])
    term_card(WORK / "agent_calls.png", "Muse Spark (muse-spark-1.3) between training rounds · outputs/agent_jacks/agent_log.jsonl", [
        "round 1",
        "  set_param(tracking_scale, 0.1 → 0.2)",
        "    \"video shows robot arms not fully overhead vs person\"",
        "round 2",
        "  set_param(task.jump, 2 → 4)",
        "    \"raise flight quality bonus to reward distinct jump apart\"",
        "round 3",
        "  set_param(style_weight, 1 → 2)",
        "    \"video shows narrow arms vs person wide overhead clap\"",
        "  train_next_round(watch=clean_stages)",
        "",
        "finish: \"raising tracking_scale and style_weight helped likeness;",
        "         raising the stage bonuses did not: overhead stages still skipped\""])
    card(WORK / "agent.png", [("Muse Spark as the reward agent", 64, MONO, WHITE),
                              ("Jumping jacks · 3 rounds · 900 PPO iterations, against the same run without it", 36, SANS, GREY),
                              ("V-JEPA 2 likeness  0.410 ± 0.008  vs  0.388 ± 0.017", 44, MONO, BLUE),
                              ("success and falls level · the gain came from upweighting the likeness terms", 30, SANS, DIM),
                              ("one training run per arm · $0.10 of API", 30, SANS, DIM)], sub=True)
    term_card(WORK / "terminal.png", "the downloaded training environment", [
        f"$ python -m a2h_env info {ENV_ZIP.name}", *env_info(), "",
        f"$ python -m a2h_env train {ENV_ZIP.name} --iters 1500     # with mjlab installed"])
    card(WORK / "honest.png", [("Not yet", 90, MONO, WHITE),
                               ("one policy for every move, zero-shot: at most 7% on unseen variants", 40, SANS, GREY),
                               ("the lunge: every policy stalls at a 0.32 m weight shift", 40, SANS, GREY),
                               ("simulation only: no real G1 yet", 40, SANS, GREY)], sub=True,
         foot=["so it trains on each video · docs/FRONTIER_GENERALIST.md · docs/FRONTIER_TRAVEL.md"])
    card(WORK / "end.png", [("athlete2humanoid", 120, MONO, WHITE),
                            ("Any video of a person moving can become robot training data.", 40, SANS, GREY),
                            ("athlete2humanoid.pages.dev  ·  github.com/PandaGrdn/athlete2humanoid", 34, MONO, BLUE)],
         sub=True, foot=["Code open · a training environment for every video",
                         "Built with Meta SAM 3.1 · SAM 3D Body · V-JEPA 2 · Muse Spark"])


# the hook: the stock video's runner (a 960 px crop of the 1080p frame) beside the policy's 960x1080 render
JOG_LEFT = (f"scale=1920:1080:flags=lanczos,crop=960:1080:480:0,"
            f"drawtext=fontfile={MONO}:textfile='{WORK / 'jog_credit.txt'}':expansion=none:fontsize=22:"
            f"fontcolor=white@0.85:x=24:y=h-44")  # a textfile: the credit's colon and commas would end the filter

SHOTS = [
    dict(n=1, name="hook", secs=12, segs=[
        dict(pair=(JOG_VIDEO, SRC / "jog.mp4"), ss=0, to=6.0, left=JOG_LEFT),
        dict(pair=(JOG_VIDEO, SRC / "jog.mp4"), ss=0, to=6.0, left=JOG_LEFT)],
        caps=[(0.3, 5.8, "A stock video of a stranger jogging → a G1 policy, in physics"),
              (6.2, 11.8, "100% of clean runs · 98% under 1 m/s pushes · 0 falls")]),
    dict(n=2, name="problem", secs=15, segs=[dict(still=WORK / "problem.png", secs=15, zoom=True)], caps=[]),
    dict(n=3, name="pipeline", secs=33, segs=[
        dict(src=JOG, ss=("replay", -0.2), to=("replay", 17.8)),
        dict(src=JOG, ss=("orbit", -0.2), to=("orbit", 5.8)),
        dict(src=KICK, ss=("chip04", -0.2), to=("chip04", 4.8)),
        dict(src=JOG, ss=("chip05", -0.2), to=("chip05", 3.8))],
        caps=[(0.3, 2.8, "01 · SAM 3.1 finds and tracks the person"), (2.9, 5.4, "02 · SAM 3D Body lifts them to 3D"),
              (5.5, 8.0, "03 · mink IK onto the G1's own skeleton"), (8.1, 10.6, "04 · the key moments of the action"),
              (10.7, 13.2, "05 · variants: mirrored, re-timed, rescaled"),
              (13.3, 17.9, "06 · PPO in mjlab · V-JEPA 2 judges the likeness"),
              (18.1, 23.9, "Rebuilt on the G1's bone lengths: planted feet stay planted"),
              (24.1, 28.9, "Muse Spark names each key moment: \"kick right leg forward\""),
              (29.1, 32.9, "36 of 36 variants kept · each checked against its own key moments")]),
    dict(n=4, name="speed", secs=20, segs=[
        dict(still=ADAPT, secs=7, zoom=True),
        dict(src=SRC / "discus.mp4", ss=0, to=4.37, fit=9.0),
        dict(still=WORK / "speed.png", secs=4)],
        caps=[(0.3, 6.8, "Task success vs minutes: scratch, a specialist, the 5-clip generalist"),
              (7.2, 15.8, "A 1900 Olympic discus throw: 100% clean · 88% under 1 m/s pushes")]),
    dict(n=5, name="robust", secs=25, segs=[
        dict(src=JOG, ss=("push1", -0.2), to=("push1", 4.3)),
        dict(src=KICK, ss=("push1", -0.2), to=("push1", 4.3)),
        dict(src=KICK, ss=("push2", -0.2), to=("push2", 4.3)),
        dict(src=SRC / "shot_put.mp4", ss=0, to=5.23, fit=11.5)],
        caps=[(0.3, 13.3, "Shoved, it recovers · unseen variants: squat jump 82%, kick 100%"),
              (13.6, 24.8, "Shot put: a travel reward and termination took success 8% → 100%")]),
    dict(n=6, name="agent", secs=30, segs=[
        dict(still=WORK / "agent_calls.png", secs=10),
        dict(src=SRC / "agent.mp4", ss=0, to=6.27, fit=14.0),
        dict(still=WORK / "agent.png", secs=6)],
        caps=[(0.3, 9.8, "Its tool calls, with its reasons, from the run's log"),
              (10.2, 23.8, "Same 900 PPO iterations: without the agent (middle), with it (right)")]),
    dict(n=7, name="live_env", secs=25, segs=[
        dict(src=LIVE, ss=("upload", -0.4), to=("built", 0.6), fit=8.0),
        dict(src=ENV, ss=("chip05", -0.2), to=("downloaded", 1.5), fit=8.0),
        dict(still=WORK / "terminal.png", secs=9)],
        caps=[(0.2, 7.9, "Live: a stock kick clip · built on the API's CPU in 101 s (12.7×)"),
              (8.1, 15.9, "Every build hands back a training environment: one 2 MB zip")]),
    dict(n=8, name="close", secs=20, segs=[dict(still=WORK / "honest.png", secs=8), dict(still=WORK / "end.png", secs=12)],
         caps=[]),
]


def pair(out, left, right, ss, to, lvf):
    """left | right, each 960x1080, trimmed to the same [ss, to)."""
    t = to - ss
    ff("-ss", ss, "-t", t, "-i", left, "-ss", ss, "-t", t, "-i", right, "-filter_complex",
       f"[0:v]{lvf},fps=30,setsar=1[l];[1:v]scale=960:1080,fps=30,setsar=1[r];[l][r]hstack=inputs=2,format=yuv420p",
       *ENC, out)


def build(shot):
    parts = []
    for i, s in enumerate(shot["segs"]):
        p = WORK / f"s{shot['n']:02d}_{i}.mp4"
        if "still" in s:
            still(s["still"], p, s["secs"], s.get("zoom", False))
        elif "pair" in s:
            pair(p, *s["pair"], s["ss"], s["to"], s["left"])
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
