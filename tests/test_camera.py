"""Camera compensation (camera.py) and the pixel-height check (pose.py): a synthetic textured scene,
panned and zoomed by a known amount, must give back that motion; a static camera must be an exact no-op.
Needs numpy, scipy and OpenCV only (no video files, no MediaPipe model).

    python -m unittest tests.test_camera
"""

import importlib.util
import unittest

if not all(importlib.util.find_spec(m) for m in ("numpy", "scipy", "cv2")):
    raise unittest.SkipTest("needs numpy, scipy and opencv-python")

import numpy as np

try:
    import camera
    import pose
except ImportError as e:  # mediapipe missing (pose imports it)
    raise unittest.SkipTest(f"pose deps missing: {e}") from e

H, W, T = 360, 640, 24


def scene(seed=0, h=H * 2, w=W * 2):
    """A large textured background (blurred noise plus a few edges), bigger than the frame."""
    import cv2

    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 3)
    img = (img - img.min()) / np.ptp(img) * 200 + 20
    for _ in range(12):
        x0, y0 = rng.integers(0, w - 80), rng.integers(0, h - 80)
        img[y0:y0 + rng.integers(20, 80), x0:x0 + rng.integers(20, 80)] = rng.uniform(0, 255)
    return np.repeat(img[..., None], 3, -1).astype(np.uint8)


def with_athlete(frames, xs):
    """Paint a moving 'athlete' block (that should be ignored) and return frames and its masks."""
    frames = frames.copy()
    masks = np.zeros(frames.shape[:3], bool)
    for t, x in enumerate(xs):
        x = int(x)
        frames[t, 100:300, x:x + 60] = (255, 0, 0)
        masks[t, 100:300, x:x + 60] = True
    return frames, masks


class TestEstimate(unittest.TestCase):
    def test_static_camera_is_identity(self):
        big = scene()
        frames = np.repeat(big[None, :H, :W], T, 0)
        frames = np.clip(frames.astype(int) + np.random.default_rng(1).integers(-2, 3, frames.shape), 0, 255)
        frames, masks = with_athlete(frames.astype(np.uint8), np.linspace(100, 400, T))  # athlete runs
        Hs, info = camera.estimate(frames, masks)
        self.assertTrue(info["static"], info)
        np.testing.assert_array_equal(Hs, np.tile(np.eye(3), (T, 1, 1)))

    def test_recovers_pan_and_zoom(self):
        big = scene()
        tt = np.arange(T) / (T - 1)
        shifts = np.stack([-200 * tt, 30 * np.sin(np.pi * tt)], -1)  # pan right 200 px, tilt
        zooms = 1 + 0.15 * tt
        frames, Ht = camera.warp_synthetic(np.repeat(big[None], T, 0), shifts, zooms, out_shape=(H, W))
        frames, masks = with_athlete(frames, np.full(T, 300))  # stays mid-frame while the camera follows
        Hs, info = camera.estimate(frames, masks)
        self.assertFalse(info["static"])
        pts = np.tile([[W / 2, H / 2], [50, 50], [W - 50, H - 50]], (T, 1, 1)).astype(float)
        err = np.linalg.norm(camera.apply(Hs, pts) - camera.apply(Ht, pts), axis=-1)
        self.assertLess(err.max(), 2.0, err.max(axis=1))

    def test_textureless_is_unreliable_and_left_alone(self):
        # A blank wall that "moves" a few pixels: nothing registers, so the estimate is not trusted and,
        # being under a clear pan, not applied.
        frames = np.full((T, H, W, 3), 128, np.uint8)
        frames[:, :, : W // 2] = 120
        frames = np.stack([np.roll(f, t, axis=1) for t, f in enumerate(frames)])
        Hs, info = camera.estimate(frames)
        self.assertFalse(info["reliable"])
        self.assertTrue(info["static"])
        np.testing.assert_array_equal(Hs, np.tile(np.eye(3), (T, 1, 1)))

    def test_cut_detected(self):
        a, b = scene(0)[:H, :W], scene(7)[:H, :W]
        b = (b.astype(int) * [1.0, 0.5, 0.2]).astype(np.uint8)  # another shot, other colours
        frames = np.stack([a] * 5 + [b] * 5)
        self.assertEqual(camera.detect_cuts(frames), [5])
        _, info = camera.estimate(frames)
        self.assertEqual(info["cuts"], [5])

    def test_apply_identity_is_exact(self):
        pts = np.random.default_rng(0).uniform(0, 1000, (T, 33, 2))
        np.testing.assert_array_equal(camera.apply(np.tile(np.eye(3), (T, 1, 1)), pts), pts)


def synthetic_pose(T=40, pan_px=None, seed=0):
    """MediaPipe-shaped landmarks of a person standing still: world (T, 33, 3) hip-centred, and
    crop-normalised image_xy with boxes, as the camera pans by pan_px[t] pixels."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.05, (33, 3))
    base[[11, 12]] = [[0.18, -0.5, 0.0], [-0.18, -0.5, 0.02]]  # shoulders above the hips (y down)
    base[[23, 24]] = [[0.1, 0, 0.0], [-0.1, 0, 0.02]]
    base[0], base[[7, 8]] = [0.05, -0.65, -0.1], [[0.08, -0.63, 0.0], [-0.08, -0.63, 0.0]]
    base[27:33] = [[0.1, 0.85, 0], [-0.1, 0.85, 0], [0.1, 0.9, 0.05], [-0.1, 0.9, 0.05], [0.1, 0.88, -0.1], [-0.1, 0.88, -0.1]]
    world = np.repeat(base[None], T, 0)
    px_per_m = 400.0
    pan = np.zeros((T, 2)) if pan_px is None else pan_px
    px = world[..., :2] * px_per_m + np.array([600, 500]) + pan[:, None]
    side = 800.0
    boxes = np.column_stack([np.full(T, 200.0), np.full(T, 100.0), np.full(T, side)])
    image_xy = (px - boxes[:, None, :2]) / side
    return world, image_xy, boxes


class TestTrajectory(unittest.TestCase):
    def test_identity_camera_changes_nothing(self):
        world, image_xy, boxes = synthetic_pose(pan_px=np.stack([np.linspace(0, 80, 40), np.zeros(40)], -1))
        a = pose.human_trajectory(world, image_xy, boxes)
        b = pose.human_trajectory(world, image_xy, boxes, camera=np.tile(np.eye(3), (40, 1, 1)))
        np.testing.assert_array_equal(a, b)

    def test_pan_is_not_root_motion(self):
        T = 40
        pan = np.stack([np.linspace(0, -300, T), np.linspace(0, 40, T)], -1)  # camera pans; athlete still
        world, image_xy, boxes = synthetic_pose(T, pan)
        Hs = np.tile(np.eye(3), (T, 1, 1))
        Hs[:, :2, 2] = -pan  # frame t pixels -> frame 0 pixels
        raw = pose.human_trajectory(world, image_xy, boxes)
        fixed = pose.human_trajectory(world, image_xy, boxes, camera=Hs)
        drift = lambda P: np.linalg.norm(P[:, [23, 24]].mean(1) - P[0, [23, 24]].mean(0), axis=-1).max()
        self.assertGreater(drift(raw), 0.5)  # 300 px at 400 px/m: 0.75 m of phantom travel
        self.assertLess(drift(fixed), 1e-6)


class TestHeight(unittest.TestCase):
    def test_height_check(self):
        _, image_xy, boxes = synthetic_pose()
        px = pose.landmark_pixels(image_xy, boxes)
        h = pose.athlete_height_px(px)
        self.assertAlmostEqual(h, (0.9 + 0.65) * 400 * pose.HEAD_TOP, delta=2)
        self.assertTrue(pose.height_check(px)["ok"])
        small = pose.height_check(px * 0.4)
        self.assertFalse(small["ok"])
        self.assertIn("warning", small)


if __name__ == "__main__":
    unittest.main()
