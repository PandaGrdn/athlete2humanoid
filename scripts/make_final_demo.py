"""Demo shots for the final squat-jump run, built on scripts/make_demo_assets.py's helpers.

    python scripts/make_final_demo.py            # -> outputs/demo/final/*.mp4, robustness.png

- grid.mp4: the person | tracking | task | task (no reference), as trained, with the task stage
  the reference is in shown underneath.
- frozen_reference.mp4: the reference clip frozen at its first frame: tracking stands still, the
  task policies still squat-jump (rollouts from the robustness suite's "noref" condition).
- push.mp4: random 1.5 m/s shoves, tracking vs task.
- robustness.png / robustness.mp4: task success per condition suite.

Needs mujoco and the Menagerie G1 (scripts/fetch_assets.sh), imageio-ffmpeg and pillow. Numbers come
from outputs/remote_latent_final/eval.json and outputs/remote_robust_final/robust.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_demo_assets as mda  # noqa: E402
from make_demo_assets import ACCENT, BG, FG, GOOD, MUTED, PANEL, Clip, H, W, render, text  # noqa: E402

ROOT = mda.ROOT
LABELS = {"tracking": "Tracking (copies the clip)", "task": "Task (rewarded for the action)",
          "taskphase": "Task, never sees the clip"}
BAD = (230, 103, 103)


class Robot:
    """Kinematic replay of qpos (at fps) in the restyled scene, camera following the pelvis."""

    def __init__(self, size):
        sys.path.insert(0, str(ROOT))
        import mujoco
        import render as a2h_render
        self.mujoco = mujoco
        self.model, self.data = a2h_render.load_g1()
        mda.style_scene(self.model, mujoco)
        self.size = size
        self.renderer = mujoco.Renderer(self.model, height=size[1], width=size[0])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.6, 145.0, -6.0

    def frames(self, qpos, fps, seconds):
        """PIL frames at the video FPS for `seconds` (holding the last pose past the end)."""
        k = max(1, int(fps))
        pad = np.pad(qpos[:, :2], ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
        target = np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in (0, 1)], 1)
        out = []
        for n in range(int(round(seconds * mda.FPS))):
            i = min(int(round(n / mda.FPS * fps)), len(qpos) - 1)
            self.data.qpos[:] = qpos[i]
            self.mujoco.mj_forward(self.model, self.data)
            self.cam.lookat[:] = [*target[i], 0.6]
            self.renderer.update_scene(self.data, camera=self.cam)
            out.append(Image.fromarray(self.renderer.render()))
        return out


def label(img, s, sub=None, color=FG, sub_color=MUTED):
    d = ImageDraw.Draw(img)
    text(d, (24, 18), s, 34, fill=color, bold=True)
    if sub:
        text(d, (24, 62), sub, 26, fill=sub_color)
    return img


def stage_bar(d, spec, t, y, x0=60, x1=W - 60):
    """The task stages as a strip, the current one highlighted, a playhead at time t."""
    dur = spec["duration_s"]
    for st in spec["stages"]:
        a, b = x0 + (x1 - x0) * st["start_s"] / dur, x0 + (x1 - x0) * max(st["end_s"], st["start_s"] + 0.15) / dur
        on = st["start_s"] <= t < max(st["end_s"], st["start_s"] + 0.15)
        d.rounded_rectangle([a + 2, y, b - 2, y + 40], 8, fill=ACCENT if on else PANEL)
        if b - a > 70:
            text(d, ((a + b) / 2, y + 20), st["kind"], 22, fill=BG if on else MUTED, anchor="mm", bold=on)
    px = x0 + (x1 - x0) * min(t, dur) / dur
    d.line([px, y - 8, px, y + 48], fill=FG, width=3)


def grid(run, spec, raw, robot, out):
    ev = json.loads((run / "eval.json").read_text())
    cell = (W // 2, (H - 120) // 2)
    seconds = spec["duration_s"]
    human = Clip(raw, cell) if raw and Path(raw).exists() else Clip(run / "clips" / "squat_jump" / "side_by_side.mp4", cell)
    panels = {k: robot.frames(np.load(run / f"policy_{k}" / "policy_qpos.npy"), 50, seconds) for k in LABELS}

    def frame(t):
        img = Image.new("RGB", (W, H), BG)
        img.paste(label(human.at(t), "The video", "one phone clip"), (0, 0))
        for (k, frames), pos in zip(panels.items(), [(cell[0], 0), (0, cell[1]), (cell[0], cell[1])]):
            e = ev[k]
            sub = f"{e['task_stages']} stages · {e['falls']} falls · {e['mean_torso_position_error_m']:.2f} m from the clip"
            img.paste(label(frames[min(int(t * mda.FPS), len(frames) - 1)].copy(), LABELS[k], sub), pos)
        stage_bar(ImageDraw.Draw(img), spec, t, H - 90)
        return img

    return render(out, seconds, frame)


def pick(roll, cond, pol, want_fall):
    """A stored rollout of this condition/policy that fell (torso below 0.35 m) or didn't, if any."""
    keys = [k for k in roll.files if k.startswith(f"{cond}/{pol}/") and k.count("/") == 2]
    for k in keys:
        fell = (roll[f"{k}/body"][:, 7, 2] < 0.35).any()
        if fell == want_fall:
            return roll[k]
    return roll[keys[0]]


def columns(roll, rb, cond, pols, robot, out, title, subtitle):
    c = rb["conditions"][cond]
    seconds = rb["meta"]["clip_steps"] / rb["meta"]["step_hz"]
    want = {k: c["policies"][k]["task_success"] < 0.5 for k in pols}  # show the typical outcome
    frames = {k: robot.frames(pick(roll, cond, k, want[k]), rb["meta"]["step_hz"], seconds) for k in pols}
    cw = W // len(pols)

    def frame(t):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 46), title, 44, bold=True, anchor="mm")
        text(d, (W // 2, 96), subtitle, 28, fill=MUTED, anchor="mm")
        for i, k in enumerate(pols):
            f = frames[k][min(int(t * mda.FPS), len(frames[k]) - 1)]
            x = i * cw + (cw - robot.size[0]) // 2
            img.paste(f, (x, 150))
            s = c["policies"][k]["task_success"]
            text(d, (i * cw + cw // 2, 150 + robot.size[1] + 40), LABELS[k], 32, bold=True, anchor="mm")
            text(d, (i * cw + cw // 2, 150 + robot.size[1] + 90), f"{s:.0%} complete the squat jump", 34,
                 fill=GOOD if s >= 0.5 else BAD, anchor="mm", bold=True)
        return img

    return render(out, seconds, frame)


def chart(rb, out_png, out_mp4):
    summ, pols = rb["summary"], list(rb["meta"]["policies"])
    suites = [("clean", "As trained"), ("push", "Random shoves"), ("init", "Noisy start"), ("start", "Start pose"),
              ("speed", "Tempo"), ("dynamics", "Friction & mass"), ("noref", "Reference frozen")]
    colors = {"tracking": (57, 135, 229), "task": (201, 139, 240), "taskphase": (224, 164, 58)}
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    text(d, (W // 2, 60), "Task success under conditions never seen in training", 46, bold=True, anchor="mm")
    text(d, (W // 2, 112), f"{rb['meta']['reps']} rollouts per condition · squat, jump, land, stand · no fall", 28,
         fill=MUTED, anchor="mm")
    x0, x1, y0, y1 = 140, W - 80, 200, H - 200
    for v in (0, 0.5, 1):
        y = y1 - (y1 - y0) * v
        d.line([x0, y, x1, y], fill=PANEL, width=2)
        text(d, (x0 - 16, y), f"{v:.0%}", 24, fill=MUTED, anchor="rm")
    gw = (x1 - x0) / len(suites)
    bw = gw * 0.8 / len(pols)
    for i, (s, name) in enumerate(suites):
        for j, k in enumerate(pols):
            v = summ[k].get(s, {}).get("task_success")
            if v is None:
                continue
            a = x0 + i * gw + gw * 0.1 + j * bw
            d.rectangle([a + 3, y1 - (y1 - y0) * v, a + bw - 3, y1], fill=colors.get(k, MUTED))
            text(d, (a + bw / 2, y1 - (y1 - y0) * v - 18), f"{v:.0%}", 22, fill=FG, anchor="mm")
        text(d, (x0 + i * gw + gw / 2, y1 + 34), name, 26, fill=ACCENT if s == "noref" else FG, anchor="mm",
             bold=s == "noref")
    for j, k in enumerate(pols):
        x = x0 + j * 520
        d.rectangle([x, H - 110, x + 28, H - 82], fill=colors.get(k, MUTED))
        text(d, (x + 44, H - 96), LABELS[k], 28, anchor="lm")
    img.save(out_png)
    return render(out_mp4, 6, lambda t: img)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="outputs/remote_latent_final")
    ap.add_argument("--robust", default="outputs/remote_robust_final")
    ap.add_argument("--raw", default="clips/squat_jump.mov", help="the original phone clip, if present")
    ap.add_argument("--out", default="outputs/demo/final")
    a = ap.parse_args(argv)
    run, robust, out = ROOT / a.run, ROOT / a.robust, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((run / "task_spec.json").read_text())
    rb = json.loads((robust / "robust.json").read_text())
    roll = np.load(robust / "rollouts.npz")
    conds = {(c["suite"], c["level"]): i for i, c in enumerate(rb["conditions"])}
    print("grid", grid(run, spec, ROOT / a.raw, Robot((W // 2, (H - 120) // 2)), out / "grid.mp4"), "s")
    cols = Robot((600, 600))
    print("frozen", columns(roll, rb, conds[("noref", "frozen")], ["tracking", "task", "taskphase"], cols,
                            out / "frozen_reference.mp4", "The reference clip, frozen at its first frame",
                            "the policies get no clip to follow"), "s")
    print("push", columns(roll, rb, conds[("push", "1.5 m/s")], ["tracking", "task"], Robot((800, 640)),
                          out / "push.mp4", "Shoved at 1.5 m/s every 1–3 s", "never seen in training"), "s")
    print("chart", chart(rb, out / "robustness.png", out / "robustness.mp4"), "s")
    print(f"assets in {out}")


if __name__ == "__main__":
    main()
