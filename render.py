"""Load the G1 and render videos (robot-only now; side-by-side with the athlete on day 3)."""

import os

# Headless rendering in Colab. Must be set before mujoco is imported.
if "COLAB_RELEASE_TAG" in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import mediapy
import mujoco
import numpy as np

import config

if not mediapy.video_is_available():  # no system ffmpeg (e.g. a Mac without Homebrew ffmpeg)
    import imageio_ffmpeg
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


def load_g1(xml_path: Path = config.G1_SCENE_XML):
    """Return (model, data) for the G1 scene, reset to the 'stand' keyframe."""
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"{xml_path} missing. Run scripts/fetch_assets.sh first.")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    reset_to_stand(model, data)
    return model, data


def reset_to_stand(model, data):
    key = model.key("stand").id
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)


def tracking_camera(model, distance=2.5, azimuth=135.0, elevation=-15.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("pelvis").id
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def render_qpos_sequence(model, qpos_seq, fps=30, size=(480, 640), cam=None):
    """Kinematic replay: render one frame per qpos row. Used for the day-3 retarget replay."""
    data = mujoco.MjData(model)
    cam = cam or tracking_camera(model)
    frames = []
    with mujoco.Renderer(model, height=size[0], width=size[1]) as renderer:
        for qpos in qpos_seq:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
    return np.stack(frames)


def render_physics(model, data, seconds=3.0, fps=30, size=(480, 640), ctrl_fn=None, cam=None):
    """Simulate with physics and render. ctrl_fn(model, data) sets data.ctrl each step.

    Default controller holds the current joint angles, i.e. the robot tries to stand still.
    """
    hold = data.qpos[7:].copy()  # actuated joints follow the 7-dof free base
    ctrl_fn = ctrl_fn or (lambda m, d: d.ctrl.__setitem__(slice(None), hold))
    cam = cam or tracking_camera(model)
    frames = []
    with mujoco.Renderer(model, height=size[0], width=size[1]) as renderer:
        while data.time < seconds:
            ctrl_fn(model, data)
            mujoco.mj_step(model, data)
            if len(frames) < data.time * fps:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
    return np.stack(frames)


def save_video(frames, path, fps=30):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps)
    return path


def side_by_side(athlete_frames, robot_frames):
    """Stack athlete and robot frames horizontally at the robot's height."""
    h = robot_frames.shape[1]
    athlete = mediapy.resize_video(athlete_frames, (h, round(athlete_frames.shape[2] * h / athlete_frames.shape[1])))
    n = min(len(athlete), len(robot_frames))
    return np.concatenate([athlete[:n], robot_frames[:n]], axis=2)


if __name__ == "__main__":
    m, d = load_g1()
    print(f"G1 loaded: nq={m.nq}, nu={m.nu}, timestep={m.opt.timestep}")
    out = save_video(render_physics(m, d, seconds=3.0), config.OUTPUT_DIR / "g1_stand.mp4")
    print(f"Pelvis height after 3 s: {d.qpos[2]:.3f} m  ->  {out}")
