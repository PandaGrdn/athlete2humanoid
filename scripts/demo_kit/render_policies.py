"""1080p renders of saved policy rollouts for the demo kit: the lakeside jog, the discus, the shot put before / after
the travel task, and the Muse reward agent's three policies.

    python scripts/demo_kit/render_policies.py            # every render -> outputs/demo_kit/src/<name>.mp4
    python scripts/demo_kit/render_policies.py shot_put   # one of them
    A2H_OUTPUTS=/path/to/outputs python scripts/demo_kit/render_policies.py

The rollouts' own policy.mp4 files are 320x240. Each run also saved policy_qpos.npy (mjlab_rollout.py: MuJoCo qpos
at the policy's 50 Hz), which this replays kinematically in the restyled scene of scripts/make_demo_assets.py, side by
side where a shot compares policies. The frontier runs' folders live in their own worktrees; copy each policy folder's
policy_qpos.npy and policy_metrics.json to the same path under outputs/ first (docs/demo_kit/EDIT_LIST.md).
Needs mujoco, numpy, pillow, imageio-ffmpeg and the G1 assets (scripts/fetch_assets.sh).
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

M = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(M / "scripts"))
sys.path.insert(0, str(M))
import make_demo_assets as mda  # noqa: E402

O = Path(os.environ.get("A2H_OUTPUTS", M / "outputs"))
SRC = O / "demo_kit" / "src"

# name: panels (policy folder under outputs/, label, sub-label), camera ("follow" the pelvis, or "fixed" on
# the whole motion so travel shows as travel)
RENDERS = {
    "jog": dict(panels=[("remote_latent_ra_jog/policy_tracking_var", "The G1, in physics", "its policy, trained on that clip")],
                camera="follow", azimuth=90, width=960),
    "discus": dict(panels=[("remote_adapt_v2_discus/policy_multi", "1900 Olympic discus, never trained on",
                            "18 min on one GPU from the 5-clip generalist")], camera="follow", azimuth=150),
    "shot_put": dict(panels=[("remote_travel_c3_shot_put/policy_base", "Before: plain tracking",
                              "73% of the glide, ends 0.42 m short · task success 8%"),
                             ("remote_travel_c3_shot_put/policy_all", "After: + travel reward and termination",
                              "96% of the glide, ends 0.05 m off · task success 100%")], camera="fixed"),
    "agent": dict(panels=[("remote_agentround_agent_jacks_confirm/policy_task_start", "Start", "1,500 iterations"),
                          ("remote_agentround_agent_jacks_confirm/policy_task_base", "+900 iterations, defaults",
                           "no agent"),
                          ("remote_agentround_agent_jacks_confirm/policy_task_agent", "+900 iterations, Muse's reward",
                           "3 agent rounds")], camera="follow", azimuth=180),
}


def load(folder):
    d = O / folder
    q = np.load(d / "policy_qpos.npy")
    fps = json.loads((d / "policy_metrics.json").read_text()).get("fps", 50)
    return q, fps


class Panel:
    """A kinematic replay in the restyled scene; one renderer per panel size."""

    def __init__(self, size):
        import mujoco
        import render as a2h_render
        self.mj = mujoco
        self.model, self.data = a2h_render.load_g1()
        mda.style_scene(self.model, mujoco)
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 1920, 1080
        self.renderer = mujoco.Renderer(self.model, height=size[1], width=size[0])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def frame(self, qpos, lookat, distance, azimuth, elevation=-8.0):
        self.data.qpos[:] = qpos
        self.mj.mj_forward(self.model, self.data)
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.renderer.update_scene(self.data, camera=self.cam)
        return Image.fromarray(self.renderer.render())


def smooth_xy(q, k):
    pad = np.pad(q[:, :2], ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in (0, 1)], 1)


def build(name, spec):
    runs = [(load(f), label, sub) for f, label, sub in spec["panels"]]
    n = len(runs)
    width = spec.get("width", 1920)  # 960: half of the frame, for a shot that puts the source video beside it
    pw = width // n
    panel = Panel((pw, 1080))
    seconds = max(len(q) / fps for (q, fps), _, _ in runs) + 1.0  # hold the last pose for a second
    allxy = np.concatenate([q[:, :2] for (q, _), _, _ in runs])
    if spec["camera"] == "fixed":  # frame the whole motion, looking across its main direction of travel
        q0 = runs[-1][0][0]
        d = q0[-1, :2] - q0[0, :2]
        az = math.degrees(math.atan2(d[1], d[0])) + 90
        centre = (allxy.min(0) + allxy.max(0)) / 2
        span = float(np.ptp(allxy, 0).max())
        fixed = dict(lookat=[*centre, 0.65], distance=2.4 + 1.3 * span, azimuth=az)
    targets = [smooth_xy(q, max(1, int(fps))) for (q, fps), _, _ in runs]
    out = SRC / f"{name}.mp4"
    SRC.mkdir(parents=True, exist_ok=True)

    def frame(t):
        img = Image.new("RGB", (width, 1080), mda.BG)
        for j, (((q, fps), label, sub), tgt) in enumerate(zip(runs, targets)):
            i = min(int(round(t * fps)), len(q) - 1)
            if spec["camera"] == "fixed":
                im = panel.frame(q[i], **fixed)
            else:
                im = panel.frame(q[i], [*tgt[i], 0.65], 2.7 if n < 3 else 3.0, spec.get("azimuth", 145))
            img.paste(im, (j * pw, 0))
            d = ImageDraw.Draw(img)
            mda.chip(d, (j * pw + 36, 48), label, size=30 if n < 3 else 26, bg=(0, 0, 0))
            mda.text(d, (j * pw + 36, 104), sub, 26 if n < 3 else 22, fill=mda.MUTED)
        d = ImageDraw.Draw(img)
        for j in range(1, n):
            d.line([(j * pw, 0), (j * pw, 1080)], fill=mda.BG, width=6)
        return img

    secs = mda.render(out, seconds, frame, size=(width, 1080))
    print(f"{out}: {secs:.2f}s, {n} panel(s)")


if __name__ == "__main__":
    names = sys.argv[1:] or list(RENDERS)
    for nm in names:
        build(nm, RENDERS[nm])
