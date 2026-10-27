#!/usr/bin/env python3
"""Build the clips docs/DEMO_SCRIPT.md calls for, from existing pipeline outputs.

Python only: video goes in and out through imageio-ffmpeg's bundled ffmpeg (no system ffmpeg, no
cv2), and text and diagrams are drawn with Pillow.

    python3 -m venv ~/.venvs/a2h-demo && ~/.venvs/a2h-demo/bin/pip install imageio-ffmpeg pillow numpy
    ~/.venvs/a2h-demo/bin/python scripts/make_demo_assets.py --src outputs/remote_latent

--render-robots first re-renders every robot motion in each run (the retarget and each policy's
rollout) at 1280x720 with MuJoCo through the repo's render.py, into <renders>/<run name>/. That
needs `pip install mujoco mediapy` and the Menagerie G1 (scripts/fetch_assets.sh, or point
A2H_ASSETS_DIR at a checkout that has it). Clips use those renders wherever they exist, and fall
back to the runs' own 320x240 videos. A second --src (e.g. outputs/remote_latent_v2) adds
grid_compare.mp4: the same policies from both runs side by side.

Every clip is 1920x1080, 30 fps, H.264, silent, written to outputs/demo/. rough_cut.mp4 strings
them together at the storyboard's timings (SHOTS) with the captions burned in, as a timing guide
for recording the voice. Metrics are read from <src>/eval.json, so a rerun after a new training
run updates the cards.
"""

import argparse
import functools
import json
import logging
import math
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
ROOT = Path(__file__).resolve().parent.parent
POLICIES = ("tracking", "hybrid", "latent")

# imageio-ffmpeg warns whenever the decode size differs from the source, which ours always does.
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

BG = (11, 15, 23)
PANEL = (24, 31, 44)
FG = (236, 240, 246)
MUTED = (140, 152, 170)
ACCENT = (34, 195, 238)   # the cyan of the labels in comparison.mp4
META = (8, 102, 255)      # Meta models
GOOD = (52, 211, 153)

# The storyboard (docs/DEMO_SCRIPT.md mirrors it): (seconds, asset, on-screen caption).
SHOTS = [
    (10, "hook", "One phone video. One humanoid policy."),
    (6, "title", ""),
    (14, "problem", "A new robot skill today: a mocap studio or a hand-written reward"),
    (15, "pipeline", "Video in, policy out, with three Meta models"),
    (10, "sam_split", "SAM 3.1 isolates the person: no suit, no markers"),
    (10, "retarget_split", "Retarget: a kinematic first guess on the Unitree G1"),
    (20, "pipeline_reward", "V-JEPA 2 knows what the action looks like; a small MLP turns that into a reward"),
    (10, "strip", "PPO in mjlab: one clip, three reward recipes"),
    (20, "grid", "Latent policy: no joint targets, still a squat"),
    (15, "metrics", "Scored by the real V-JEPA 2, not the reward net"),
    (25, "muse_loop", "Muse Spark reads the clip, designs the reward, judges the rollout"),
    (15, "impact", "Every video of a person moving is now robot training data"),
    (10, "end", ""),
]
assert sum(s for s, _, _ in SHOTS) == 180


# ---------------------------------------------------------------- drawing helpers

_FONTS = {
    False: ["/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    True: ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}


@functools.cache
def font(size, bold=False):
    for path in _FONTS[bold]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def clamp(x):
    return min(max(x, 0.0), 1.0)


def ramp(t, start, length=0.6):
    """0 -> 1, eased, over [start, start + length] seconds."""
    return 0.5 - 0.5 * math.cos(math.pi * clamp((t - start) / length))


def mix(color, alpha, under=BG):
    return tuple(int(u + (c - u) * alpha) for c, u in zip(color, under))


def canvas():
    return Image.new("RGB", (W, H), BG)


def text(d, xy, s, size, fill=FG, bold=False, anchor="la", alpha=1.0):
    d.text(xy, s, font=font(size, bold), fill=mix(fill, alpha), anchor=anchor)


def wrap(s, size, width, bold=False):
    """Split s into lines no wider than `width` pixels."""
    f, lines, line = font(size, bold), [], ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if line and f.getlength(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def chip(d, xy, s, size=30, fill=FG, bg=(0, 0, 0), anchor="la", pad=12):
    """Text on a solid rounded box, for labels over video."""
    box = d.textbbox(xy, s, font=font(size, True), anchor=anchor)
    d.rounded_rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), 10, fill=bg)
    d.text(xy, s, font=font(size, True), fill=fill, anchor=anchor)


def arrow(d, a, b, fill=MUTED, width=4, head=14):
    d.line([a, b], fill=fill, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    pts = [b] + [(b[0] - head * math.cos(ang + s * 0.45), b[1] - head * math.sin(ang + s * 0.45))
                 for s in (1, -1)]
    d.polygon(pts, fill=fill)


def fade(img, t, duration, length=0.35):
    """Fade in from and out to BG."""
    a = min(clamp(t / length), clamp((duration - t) / length))
    return img if a >= 1 else Image.blend(Image.new("RGB", img.size, BG), img, a)


def dim(img, alpha, blur=0):
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return Image.blend(Image.new("RGB", img.size, BG), img, alpha)


# ---------------------------------------------------------------- video in / out

class Clip:
    """A video decoded on demand and letterboxed to `size` (or, with fill=True, scaled to cover
    it and centre-cropped). at(t) holds the last frame past the
    end; at(t, loop=True) wraps around instead. Times must not go backwards (apart from wrapping),
    since frames are decoded in order."""

    def __init__(self, path, size, crop=None, fill=False):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        self.size = size
        w, h = size
        if fill:
            vf = f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}"
        else:
            vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                  f"pad={w}:{h}:-1:-1:color=0x{''.join(f'{c:02x}' for c in BG)}")
        if crop:
            vf = f"crop={crop}," + vf
        self._open = lambda: imageio_ffmpeg.read_frames(str(path), output_params=["-vf", vf])
        self._frames = self._open()
        meta = next(self._frames)
        self.fps, self.duration = meta["fps"], meta["duration"]
        self._i, self._frame = -1, None

    def at(self, t, loop=False):
        if loop:
            t %= self.duration
            if int(t * self.fps + 1e-6) < self._i:  # wrapped: decode from the start again
                self._frames.close()
                self._frames = self._open()
                next(self._frames)
                self._i = -1
        target = max(0, int(t * self.fps + 1e-6))
        while self._i < target:
            buf = next(self._frames, None)
            if buf is None:
                break
            self._frame, self._i = buf, self._i + 1
        return Image.frombuffer("RGB", self.size, self._frame).copy()

    def close(self):
        self._frames.close()


def render(path, seconds, frame_fn, size=(W, H)):
    """Write frame_fn(t) for t in [0, seconds) at FPS. Returns the duration written."""
    writer = imageio_ffmpeg.write_frames(
        str(path), size, fps=FPS, codec="libx264", quality=None, macro_block_size=8,
        output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"])
    writer.send(None)
    n = int(round(seconds * FPS))
    for i in range(n):
        img = frame_fn(i / FPS)
        assert img.size == size, img.size
        writer.send(img.tobytes())
    writer.close()
    return n / FPS


# ---------------------------------------------------------------- 720p robot renders

RENDER_SIZE = (1280, 720)
SKY = (17, 23, 34)
FLOOR = ((31, 40, 55), (24, 31, 43), (50, 64, 86))  # the two checker squares, the grid lines


def robot_motions(src, motion):
    """(name, qpos file, its fps) for the retarget and every policy rollout in a run."""
    found, clip = [], src / "clips" / motion
    if (clip / "qpos.npy").exists():
        meta = clip / "meta.json"
        found.append(("retarget", clip / "qpos.npy",
                      json.loads(meta.read_text()).get("fps", 30.0) if meta.exists() else 30.0))
    for k in POLICIES:
        run = src / f"policy_{k}"
        if (run / "policy_qpos.npy").exists():
            metrics = run / "policy_metrics.json"
            found.append((k, run / "policy_qpos.npy",
                          json.loads(metrics.read_text()).get("fps", 50) if metrics.exists() else 50))
    return found


def style_scene(model, mujoco):
    """Restyle the Menagerie scene in place (render.py stays untouched): a dark floor that fades
    into a matching background, one shadow-casting key light from the camera side, a soft fill."""
    sky = np.array(SKY) / 255
    model.vis.rgba.haze[:] = [*sky, 1]
    model.vis.headlight.ambient[:] = 0.22
    model.vis.headlight.diffuse[:] = 0.2
    model.vis.headlight.specular[:] = 0.05
    model.vis.quality.offsamples = 8
    model.vis.global_.offwidth, model.vis.global_.offheight = RENDER_SIZE
    key = np.array([-0.45, 0.75, -1.0])
    model.light_dir[0] = key / np.linalg.norm(key)
    model.light_diffuse[0] = 0.6
    model.light_specular[0] = 0.25
    model.light_castshadow[0] = 1
    for i in range(model.ntex):
        w, h, c, adr = model.tex_width[i], model.tex_height[i], model.tex_nchannel[i], model.tex_adr[i]
        if model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            img = np.broadcast_to(np.array(SKY, np.uint8), (h, w, 3))
        elif model.tex(i).name == "groundplane":  # a 2x2 checker with grid lines, like the original
            y, x = np.mgrid[:h, :w]
            img = np.where(((x * 2 // w + y * 2 // h) % 2)[..., None], FLOOR[0], FLOOR[1]).astype(np.uint8)
            img[(x % (w // 2) < 2) | (y % (h // 2) < 2)] = FLOOR[2]
        else:
            continue
        model.tex_data[adr:adr + w * h * c] = img[..., :c].ravel()
    model.mat_reflectance[model.mat("groundplane").id] = 0.1


def render_robots(src, motion, dest, azimuth=145.0, force=False):
    """Kinematic replay of each robot motion in `src` at 1280x720, 30 fps, into dest/<name>.mp4.
    The camera follows the pelvis on the ground plane (smoothed, at a fixed height), so squats
    read as motion instead of the camera bobbing with the robot."""
    sys.path.insert(0, str(ROOT))
    import mujoco
    import render as a2h_render  # the repo's render.py

    model, data = a2h_render.load_g1()
    style_scene(model, mujoco)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 2.4, azimuth, -6.0  # 145 matches side_by_side.mp4
    dest.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=RENDER_SIZE[1], width=RENDER_SIZE[0]) as renderer:
        for name, path, fps in robot_motions(src, motion):
            out = dest / f"{name}.mp4"
            if out.exists() and not force:
                print(f"{out}  (kept; --force-render to redo)")
                continue
            qpos = np.load(path)
            k = max(1, int(fps))  # 1 s box filter on the camera target
            pad = np.pad(qpos[:, :2], ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
            target = np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in (0, 1)], 1)

            def frame(t):
                i = min(int(round(t * fps)), len(qpos) - 1)
                data.qpos[:] = qpos[i]
                mujoco.mj_forward(model, data)
                cam.lookat[:] = [*target[i], 0.6]
                renderer.update_scene(data, camera=cam)
                return Image.fromarray(renderer.render())

            seconds = render(out, len(qpos) / fps, frame, size=RENDER_SIZE)
            print(f"{out}  {seconds:6.2f} s")


# ---------------------------------------------------------------- inputs

class Inputs:
    def __init__(self, src, raw, motion, bank_size, renders=None):
        self.src, self.motion, self.bank_size = Path(src), motion, bank_size
        self.renders = Path(renders) / self.src.name if renders else None
        clip = self.src / "clips" / motion
        self.silhouette = clip / "silhouette.mp4"
        self.side_by_side = clip / "side_by_side.mp4"
        self.comparison = self.src / "comparison.mp4"
        self.policies = [k for k in POLICIES if (self.src / f"policy_{k}").is_dir()]
        self.eval = json.loads((self.src / "eval.json").read_text())
        self.distill = json.loads((self.src / "distill_report.json").read_text())
        self.latent = np.load(clip / "video_latent.npz")["dense"]
        # side_by_side.mp4 is [human | robot] with a square robot panel, so split it by its height.
        frames = imageio_ffmpeg.read_frames(str(self.side_by_side))
        self.sbs_w, self.sbs_h = next(frames)["source_size"]
        frames.close()
        s = self.sbs_h
        self.sbs_human, self.sbs_robot = f"{self.sbs_w - s}:{s}:0:0", f"{s}:{s}:{self.sbs_w - s}:0"
        # The raw clip is sharper than side_by_side's left half; fall back to that half without it.
        self.raw = Path(raw) if raw and Path(raw).exists() else None

    def human(self, size):
        if self.raw:
            return Clip(self.raw, size)
        return Clip(self.side_by_side, size, crop=self.sbs_human)

    def rendered(self, key):
        path = self.renders / f"{key}.mp4" if self.renders else None
        return path if path and path.exists() else None

    def robot(self, key, size, fill=False):
        """The best footage of a robot motion ("retarget" or a policy): its 720p render if there
        is one, else the run's own 320x240 policy video or side_by_side.mp4's robot half."""
        if self.rendered(key):
            return Clip(self.rendered(key), size, fill=fill)
        if key == "retarget":
            return Clip(self.side_by_side, size, crop=self.sbs_robot, fill=fill)
        return Clip(self.src / f"policy_{key}" / "policy.mp4", size, fill=fill)

    def score(self, key):
        return self.eval[key]["aligned"]


# ---------------------------------------------------------------- assets

def asset_hook(inp, out):
    """The human clip next to the latent-only policy."""
    half = (W // 2, H)
    human, robot = inp.human(half), inp.robot("latent", half, fill=True)

    def frame(t):
        img = canvas()
        img.paste(human.at(t), (0, 0))
        img.paste(robot.at(t), (W // 2, 0))
        d = ImageDraw.Draw(img)
        d.line([(W // 2, 0), (W // 2, H)], fill=BG, width=6)
        chip(d, (40, 40), "phone video", fill=ACCENT)
        chip(d, (W // 2 + 40, 40), "Unitree G1 policy · reward from V-JEPA 2 only", fill=ACCENT)
        return img

    return render(out, human.duration, frame)


def asset_title(inp, out, seconds=6):
    strip = Clip(inp.comparison, (W, 384))

    def frame(t):
        img = canvas()
        img.paste(dim(strip.at(t, loop=True), 0.22, blur=2), (0, 640))
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 330), "athlete2humanoid", 140, bold=True, anchor="mm", alpha=ramp(t, 0.2, 0.8))
        text(d, (W // 2, 460), "Any video in. A humanoid policy out.", 56, MUTED, anchor="mm",
             alpha=ramp(t, 0.8))
        text(d, (W // 2, 560), "Meta SAM 3.1  ·  V-JEPA 2  ·  Muse Spark   →   PPO in mjlab  ·  Unitree G1",
             34, ACCENT, anchor="mm", alpha=ramp(t, 1.4))
        return fade(img, t, seconds)

    return render(out, seconds, frame)


def bullet_card(out, seconds, heading, bullets, punch, starts, punch_at):
    def frame(t):
        img = canvas()
        d = ImageDraw.Draw(img)
        text(d, (160, 170), heading, 48, MUTED, bold=True, alpha=ramp(t, 0.1))
        y = 290
        for s, start in zip(bullets, starts):
            a = ramp(t, start)
            d.ellipse((165, y + 16, 185, y + 36), fill=mix(ACCENT, a))
            for line in wrap(s, 50, 1500):
                text(d, (220, y), line, 50, alpha=a)
                y += 66
            y += 44
        for i, line in enumerate(wrap(punch, 60, 1600, bold=True)):
            text(d, (160, 860 + i * 76), line, 60, ACCENT, bold=True, alpha=ramp(t, punch_at))
        return fade(img, t, seconds)

    return render(out, seconds, frame)


def asset_problem(inp, out, seconds=14):
    return bullet_card(
        out, seconds, "Teaching a humanoid a new skill today",
        ["A motion-capture studio: suits, markers, cameras.",
         "Or a reward function written by hand, one skill at a time.",
         "Copy joint angles from a video and the robot replays that clip. It hasn't learned the skill."],
        "Meanwhile there are billions of hours of people moving on video.",
        starts=[0.5, 3.0, 5.5], punch_at=9.0)


def asset_impact(inp, out, seconds=15):
    return bullet_card(
        out, seconds, "What this unlocks",
        ["Any clip of a person (a dance, a chore, a rehab exercise) becomes a robot training signal.",
         "The reward comes from what the action looks like, so it transfers across bodies.",
         "Perception and reward run on open weights (SAM 3.1, V-JEPA 2); training runs in open-source mjlab."],
        "Every video of a person moving is now robot training data.",
        starts=[0.5, 3.5, 6.5], punch_at=10.0)


NODES = [
    ("Phone video", "any clip, any room", False),
    ("SAM 3.1", "segments the person", True),
    ("Retarget", "human → G1 joints", False),
    ("V-JEPA 2", "latent of the action", True),
    ("Distilled reward", "MLP: G1 body → latent", False),
    ("PPO in mjlab", "Unitree G1 policy", False),
]
NODE_W, NODE_GAP, NODE_Y = 262, 50, 250


def node_x(i):
    left = (W - len(NODES) * NODE_W - (len(NODES) - 1) * NODE_GAP) // 2
    return left + i * (NODE_W + NODE_GAP)


def latent_tile(latent, frac, size):
    """One step of the clip's dense V-JEPA latent (1024 dims shown as 32x32), z-scored per dim."""
    z = (latent - latent.mean(0)) / (latent.std(0) + 1e-6)
    a = np.clip((z[min(int(frac * len(z)), len(z) - 1)].reshape(32, 32) + 2) / 4, 0, 1)[..., None]
    rgb = np.array(BG) * (1 - a) ** 2 + np.array(META) * 2 * a * (1 - a) + np.array(ACCENT) * a ** 2
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize(size, Image.NEAREST)


@functools.cache
def mlp_glyph(size):
    img = Image.new("RGB", size, PANEL)
    d = ImageDraw.Draw(img)
    w, h = size
    layers = [6, 8, 8, 4]
    pts = [[(int(w * (0.18 + 0.64 * li / (len(layers) - 1))), int(h * (0.15 + 0.7 * (k + 0.5) / n)))
            for k in range(n)] for li, n in enumerate(layers)]
    for a, b in zip(pts, pts[1:]):
        for p in a:
            for q in b:
                d.line([p, q], fill=mix(MUTED, 0.35, PANEL), width=1)
    for layer in pts:
        for x, y in layer:
            d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=ACCENT)
    return img


def pipeline_card(inp, out, seconds, highlight=None, detail=()):
    """The six stages left to right with live thumbnails. Without `highlight` the nodes build in
    one by one and the Muse Spark bracket appears under them; with it, the other nodes are dimmed
    and the `detail` lines are shown underneath."""
    box = (NODE_W, NODE_W)
    thumbs = {0: inp.human(box), 1: Clip(inp.silhouette, box),
              2: inp.robot("retarget", box, fill=True), 5: inp.robot("latent", box, fill=True)}
    build = highlight is None
    bottom = NODE_Y + NODE_W

    def frame(t):
        img = canvas()
        d = ImageDraw.Draw(img)
        title = "The pipeline" if build else "The reward: V-JEPA 2, distilled"
        text(d, (W // 2, 130), title, 56, bold=True, anchor="mm", alpha=ramp(t, 0.1))
        for i, (name, sub, meta) in enumerate(NODES):
            a = ramp(t, 0.4 + 0.8 * i) if build else (1.0 if i in highlight else 0.3)
            if a <= 0:
                continue
            x = node_x(i)
            if i in thumbs:
                th = thumbs[i].at(t, loop=True)
            elif i == 3:
                th = latent_tile(inp.latent, (t % thumbs[1].duration) / thumbs[1].duration, box)
            else:
                th = mlp_glyph(box)
            img.paste(dim(th, a), (x, NODE_Y))
            edge = META if meta else (ACCENT if not build else PANEL)
            d.rectangle((x - 3, NODE_Y - 3, x + NODE_W + 2, bottom + 2), outline=mix(edge, a), width=4)
            text(d, (x + NODE_W // 2, bottom + 44), name, 36, bold=True, anchor="mm", alpha=a)
            text(d, (x + NODE_W // 2, bottom + 88), sub, 26, MUTED, anchor="mm", alpha=a)
            if meta:
                chip(d, (x + 14, NODE_Y + 14), "META", size=20, fill=mix(FG, a), bg=mix(META, a), pad=6)
            if i:
                y = NODE_Y + NODE_W // 2
                arrow(d, (x - NODE_GAP + 8, y), (x - 8, y), fill=mix(MUTED, a))
        if build:
            a = ramp(t, 0.4 + 0.8 * len(NODES) + 0.5)
            col, y = mix(META, a), bottom + 150
            for i in (0, 4, 5):
                cx = node_x(i) + NODE_W // 2
                d.line([(cx, y - 24), (cx, y)], fill=col, width=4)
            d.line([(node_x(0) + NODE_W // 2, y), (node_x(5) + NODE_W // 2, y)], fill=col, width=4)
            text(d, (W // 2, y + 60), "Muse Spark agent: reads the clip, designs the reward, watches the rollout, revises",
                 34, bold=True, anchor="mm", alpha=a)
        for k, line in enumerate(detail):
            text(d, (W // 2, bottom + 170 + k * 58), line, 36, FG if k < 2 else MUTED, bold=k < 2,
                 anchor="mm", alpha=ramp(t, 0.8 + 2.5 * (k // 2)))
        return fade(img, t, seconds)

    return render(out, seconds, frame)


def asset_pipeline(inp, out):
    return pipeline_card(inp, out, 15)


def asset_pipeline_reward(inp, out):
    r = inp.distill
    detail = [
        "V-JEPA 2 is too slow to run inside RL, so a small MLP learns to predict its latent",
        "from ~1 s of the robot's body positions. Reward = cosine to the human clip's latent.",
        f"Trained on {r['samples']:,} windows from {r['motions']} rendered robot motions,",
        "falls, freezes and wrong moves included, so it can tell the robot when it's wrong.",
    ]
    return pipeline_card(inp, out, 20, highlight={3, 4}, detail=detail)


def asset_sam_split(inp, out):
    half = (W // 2 - 60, H - 160)
    human, sil = inp.human(half), Clip(inp.silhouette, half)

    def frame(t):
        img = canvas()
        img.paste(human.at(t), (40, 120))
        img.paste(sil.at(t), (W // 2 + 20, 120))
        d = ImageDraw.Draw(img)
        text(d, (40 + half[0] // 2, 64), "input video", 40, bold=True, anchor="mm")
        text(d, (W // 2 + 20 + half[0] // 2, 64), "SAM 3.1 mask", 40, ACCENT, bold=True, anchor="mm")
        return img

    return render(out, human.duration, frame)


def asset_retarget_split(inp, out):
    if inp.rendered("retarget"):  # the human next to the 720p render
        human, robot = inp.human((W // 3, H)), inp.robot("retarget", (W - W // 3, H), fill=True)

        def frame(t):
            img = canvas()
            img.paste(human.at(t), (0, 0))
            img.paste(robot.at(t), (W // 3, 0))
            d = ImageDraw.Draw(img)
            d.line([(W // 3, 0), (W // 3, H)], fill=BG, width=6)
            chip(d, (24, 30), "human", fill=ACCENT)
            chip(d, (W // 3 + 24, 30), "Unitree G1 · kinematic retarget", fill=ACCENT)
            return img

        return render(out, human.duration, frame)

    sbs = Clip(inp.side_by_side, (W, H))
    scale = min(W / inp.sbs_w, H / inp.sbs_h)
    left = (W - int(inp.sbs_w * scale)) // 2
    split = left + int((inp.sbs_w - inp.sbs_h) * scale)

    def frame(t):
        img = sbs.at(t)
        d = ImageDraw.Draw(img)
        chip(d, (left + 24, 30), "human", fill=ACCENT)
        chip(d, (split + 24, 30), "Unitree G1 · kinematic retarget", fill=ACCENT)
        return img

    return render(out, sbs.duration, frame)


def asset_strip(inp, out):
    strip = Clip(inp.comparison, (W, 384))
    cols = [None, "retarget", "tracking", "hybrid", "latent"]  # comparison.mp4's column order
    notes = {"tracking": "joint targets", "hybrid": "joints + V-JEPA", "latent": "V-JEPA only"}

    def frame(t):
        img = canvas()
        img.paste(strip.at(t), (0, 300))
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 140), "What V-JEPA 2 sees: the clip's silhouette next to each robot's", 46,
             bold=True, anchor="mm")
        text(d, (W // 2, 210), "three policies trained with PPO in mjlab, each with a different reward",
             32, MUTED, anchor="mm")
        cw = W // len(cols)
        for i, key in enumerate(cols):
            if not key:
                continue
            cx = i * cw + cw // 2
            text(d, (cx, 760), f"{inp.score(key):.3f}", 56, GOOD, bold=True, anchor="mm")
            text(d, (cx, 818), "V-JEPA similarity", 26, MUTED, anchor="mm")
            if key in notes:
                text(d, (cx, 870), notes[key], 30, anchor="mm")
        return img

    return render(out, strip.duration, frame)


def asset_grid(inp, out):
    cell = (W // 2, H // 2)
    clips = [inp.human(cell), inp.robot("retarget", cell, fill=True),
             inp.robot("tracking", cell, fill=True), inp.robot("latent", cell, fill=True)]
    labels = ["human video", "retarget (kinematic)", "tracking policy · joint targets",
              "latent policy · V-JEPA reward only"]
    keys = [None, "retarget", "tracking", "latent"]

    def frame(t):
        img = canvas()
        d = ImageDraw.Draw(img)
        for k, (c, label, key) in enumerate(zip(clips, labels, keys)):
            x, y = (k % 2) * cell[0], (k // 2) * cell[1]
            img.paste(c.at(t), (x, y))
            chip(d, (x + 24, y + 24), label, size=30, fill=ACCENT)
            if key:
                falls = inp.eval[key].get("falls")
                s = f"V-JEPA {inp.score(key):.3f}" + ("" if falls is None else f" · {falls} falls")
                chip(d, (x + cell[0] - 24, y + 24), s, size=28, fill=GOOD, anchor="ra")
        d.line([(W // 2, 0), (W // 2, H)], fill=BG, width=6)
        d.line([(0, H // 2), (W, H // 2)], fill=BG, width=6)
        return img

    return render(out, max(c.duration for c in clips), frame)


def asset_grid_compare(runs, labels, out):
    """The human clip and the retarget in the first column, then one column per policy that
    every run has, one row per run (e.g. v1 over v2)."""
    first = runs[0]
    keys = [k for k in first.policies if all(k in r.policies for r in runs[1:])]
    cols, rows = 1 + len(keys), len(runs)
    cw, ch = W // cols, H // rows
    left = [first.human((cw, ch)), first.robot("retarget", (cw, ch), fill=True)]
    cells = [[r.robot(k, (cw, ch), fill=True) for k in keys] for r in runs]
    names = {"tracking": "tracking", "hybrid": "hybrid", "latent": "latent · V-JEPA only"}

    def frame(t):
        img = canvas()
        d = ImageDraw.Draw(img)
        for j in range(rows):
            if j < len(left):
                img.paste(left[j].at(t), (0, j * ch))
                chip(d, (20, j * ch + 20), ["human video", "retarget"][j], size=26, fill=ACCENT)
            for i, (key, clip) in enumerate(zip(keys, cells[j])):
                x, y = (i + 1) * cw, j * ch
                img.paste(clip.at(t), (x, y))
                chip(d, (x + 20, y + 20), f"{labels[j]} · {names[key]}", size=26, fill=ACCENT)
                e = runs[j].eval.get(key)
                if e:
                    s = f"V-JEPA {e['aligned']:.3f}" + (f" · {e['falls']} falls" if "falls" in e else "")
                    chip(d, (x + 20, y + ch - 20), s, size=24, fill=GOOD, anchor="ld")
        for i in range(1, cols):
            d.line([(i * cw, 0), (i * cw, H)], fill=BG, width=6)
        for j in range(1, rows):
            d.line([(0, j * ch), (W, j * ch)], fill=BG, width=6)
        return img

    return render(out, max(c.duration for row in cells for c in row), frame)


def asset_metrics(inp, out, seconds=15):
    rows = [("retarget", "Retarget (kinematic, not physical)"),
            ("tracking", "Tracking policy (joint targets)"),
            ("hybrid", "Hybrid policy (joints + V-JEPA reward)"),
            ("latent", "Latent policy (V-JEPA reward only)")]
    x0, x1 = 760, 1560

    def frame(t):
        img = canvas()
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 110), f"Real V-JEPA 2 similarity to the human {inp.motion} clip", 52, bold=True,
             anchor="mm", alpha=ramp(t, 0.1))
        text(d, (W // 2, 175), "aligned cosine of video embeddings · 1.0 = identical", 30, MUTED,
             anchor="mm", alpha=ramp(t, 0.3))
        for i, (key, name) in enumerate(rows):
            y, e = 300 + i * 150, inp.eval[key]
            a, grow = ramp(t, 0.8 + 0.6 * i), ramp(t, 1.0 + 0.6 * i, 1.4)
            v = e["aligned"]
            text(d, (x0 - 30, y), name, 34, bold=key == "latent", anchor="rm", alpha=a)
            d.rectangle((x0, y - 32, x1, y + 32), fill=mix(PANEL, a))
            d.rectangle((x0, y - 32, x0 + int((x1 - x0) * v * grow), y + 32),
                        fill=mix(ACCENT if key == "latent" else META, a))
            text(d, (x1 + 30, y), f"{v * grow:.3f}", 44, bold=True, anchor="lm", alpha=a)
            tags = ([f"{e['falls']} falls"] if "falls" in e else []) + \
                   [f"rank #{e['dtw_rank_vs_bank']} of {inp.bank_size} motions"]
            text(d, (x0, y + 56), "  ·  ".join(tags), 26, GOOD, anchor="lm", alpha=a)
        a = ramp(t, 5.5)
        text(d, (W // 2, 930), "Rank: each robot motion against the bank of robot motions, by closeness to the clip.",
             30, MUTED, anchor="mm", alpha=a)
        text(d, (W // 2, 980), "The latent policy never saw a joint target, and V-JEPA still picks it out as the squat.",
             30, bold=True, anchor="mm", alpha=a)
        return fade(img, t, seconds)

    return render(out, seconds, frame)


def asset_muse_loop(inp, out, seconds=25):
    """Muse Spark's two jobs drawn as one loop: the skill reader, then reward-design rounds."""
    steps = [
        ("Skill reader", "watches the clip → training spec", "(phases, nearest bank motions, weights)"),
        ("Train", "PPO in mjlab with that reward", "(3 configs in parallel)"),
        ("Judge", "real V-JEPA 2 score + falls", "+ the rollout video"),
        ("Revise", "Muse Spark edits the reward", "and termination config"),
    ]
    cx, cy, r = W // 2, 560, 330
    centres = [(cx, cy - r), (cx + r + 130, cy), (cx, cy + r), (cx - r - 130, cy)]

    def frame(t):
        img = canvas()
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 90), "The agentic loop: Muse Spark designs the reward", 52, bold=True,
             anchor="mm", alpha=ramp(t, 0.1))
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=mix(PANEL, ramp(t, 0.3)), width=6)
        if t > 3.5:  # a dot running the loop, one lap per 4 s
            ang = math.radians(-90 + (t - 3.5) / 4 * 360)
            px, py = cx + r * math.cos(ang), cy + r * math.sin(ang)
            d.ellipse((px - 14, py - 14, px + 14, py + 14), fill=ACCENT)
        a = ramp(t, 0.4)
        d.ellipse((cx - 160, cy - 75, cx + 160, cy + 75), fill=mix(META, a))
        text(d, (cx, cy - 14), "Muse Spark", 44, bold=True, anchor="mm", alpha=a)
        text(d, (cx, cy + 32), "video in · tool calls", 24, anchor="mm", alpha=a)
        for i, ((name, l1, l2), (nx, ny)) in enumerate(zip(steps, centres)):
            a = ramp(t, 0.8 + 0.6 * i)
            bw, bh = 470, 150
            bx, by = nx - bw / 2, ny - bh / 2
            d.rounded_rectangle((bx, by, bx + bw, by + bh), 16, fill=mix(PANEL, a), outline=mix(ACCENT, a), width=3)
            text(d, (nx, by + 40), name, 34, ACCENT, bold=True, anchor="mm", alpha=a)
            text(d, (nx, by + 86), l1, 24, anchor="mm", alpha=a)
            text(d, (nx, by + 118), l2, 24, MUTED, anchor="mm", alpha=a)
        text(d, (W // 2, 1030), "Scored by the real V-JEPA 2, not the distilled reward, so the agent can't game its own reward.",
             28, MUTED, anchor="mm", alpha=ramp(t, 4.0))
        return fade(img, t, seconds)

    return render(out, seconds, frame)


def asset_end(inp, out, seconds=10):
    robot = inp.robot("latent", (W, H), fill=True)

    def frame(t):
        img = dim(robot.at(t, loop=True), 0.18, blur=3)
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 380), "athlete2humanoid", 130, bold=True, anchor="mm", alpha=ramp(t, 0.2))
        text(d, (W // 2, 500), "From any video to a humanoid policy that learns the action.", 48, MUTED,
             anchor="mm", alpha=ramp(t, 0.7))
        text(d, (W // 2, 640), "github.com/PandaGrdn/athlete2humanoid", 40, ACCENT, bold=True, anchor="mm",
             alpha=ramp(t, 1.2))
        text(d, (W // 2, 760), "Built with Meta SAM 3.1  ·  V-JEPA 2  ·  Muse Spark", 34, anchor="mm",
             alpha=ramp(t, 1.6))
        return fade(img, t, seconds, length=0.6)

    return render(out, seconds, frame)


ASSETS = {
    "hook": asset_hook,
    "title": asset_title,
    "problem": asset_problem,
    "pipeline": asset_pipeline,
    "sam_split": asset_sam_split,
    "retarget_split": asset_retarget_split,
    "pipeline_reward": asset_pipeline_reward,
    "strip": asset_strip,
    "grid": asset_grid,
    "metrics": asset_metrics,
    "muse_loop": asset_muse_loop,
    "impact": asset_impact,
    "end": asset_end,
}


def rough_cut(out_dir, path):
    """All shots back to back at the SHOTS timings, captions burned in. A clip shorter than its
    shot loops."""
    plan, start = [], 0.0
    for seconds, name, cap in SHOTS:
        plan.append((start, start + seconds, name, cap))
        start += seconds
    cur = {"shot": None, "clip": None}

    def frame(t):
        s0, _, name, cap = next(p for p in plan if p[0] <= t < p[1])
        if cur["shot"] != s0:
            if cur["clip"]:
                cur["clip"].close()
            cur.update(shot=s0, clip=Clip(out_dir / f"{name}.mp4", (W, H)))
        img = cur["clip"].at(t - s0, loop=True)
        if cap:
            chip(ImageDraw.Draw(img), (W // 2, H - 70), cap, size=40, anchor="mm", pad=18)
        return img

    return render(path, start, frame)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", nargs="+", default=["outputs/remote_latent"],
                   help="experiments/latent_policy.py output dir(s); the first drives every clip, and "
                        "a second adds grid_compare.mp4")
    p.add_argument("--labels", nargs="+", help="row labels for grid_compare (default v1 v2 ...)")
    p.add_argument("--renders", default="outputs/demo/renders",
                   help="720p robot renders, one subfolder per run; used wherever they exist")
    p.add_argument("--render-robots", action="store_true", help="(re)build the renders first (needs mujoco)")
    p.add_argument("--force-render", action="store_true", help="redo renders that already exist")
    p.add_argument("--azimuth", type=float, default=145.0, help="render camera azimuth, degrees")
    p.add_argument("--raw", default="clips/squat.mov",
                   help="original clip, sharper than side_by_side's left half (skipped if missing)")
    p.add_argument("--motion", default="squat")
    p.add_argument("--bank-size", type=int, default=26, help="motions the bank rank is taken over")
    p.add_argument("--out", default="outputs/demo")
    p.add_argument("--only", nargs="+", choices=[*ASSETS, "grid_compare", "rough_cut"], help="build just these")
    p.add_argument("--no-rough-cut", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.render_robots:
        for src in map(Path, args.src):
            render_robots(src, args.motion, Path(args.renders) / src.name, args.azimuth, args.force_render)
    runs = [Inputs(src, args.raw, args.motion, args.bank_size, args.renders) for src in args.src]
    labels = args.labels or [f"v{i + 1}" for i in range(len(runs))]
    inp = runs[0]
    print(f"human clip: {inp.raw or 'side_by_side.mp4, left half'}")
    for r in runs:
        print(f"{r.src.name}: renders for {[k for k in ['retarget', *r.policies] if r.rendered(k)] or 'none'}")
    default = [*ASSETS, *(["grid_compare"] if len(runs) > 1 else []),
               *([] if args.no_rough_cut else ["rough_cut"])]
    for name in args.only or default:
        path = out / f"{name}.mp4"
        if name == "rough_cut":
            seconds = rough_cut(out, path)
        elif name == "grid_compare":
            seconds = asset_grid_compare(runs, labels, path)
        else:
            seconds = ASSETS[name](inp, path)
        print(f"{path}  {seconds:6.2f} s")


if __name__ == "__main__":
    main()
