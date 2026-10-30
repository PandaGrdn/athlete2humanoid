"""frontends/sam3d_body.py: SAM 3D Body's MHR70 keypoints -> the (T, 33, 3) trajectory retarget.py reads."""

import numpy as np
import pytest

pytest.importorskip("mediapipe")  # sam3d_body builds on pose.py, which imports it (CI's light deps don't have it)
from frontends import sam3d_body  # noqa: E402

# A standing person in camera axes (x right, y down, z depth), facing the camera, by MHR70 index.
STAND = {0: (0, -0.65, -0.1), 1: (0.03, -0.68, -0.08), 2: (-0.03, -0.68, -0.08), 3: (0.07, -0.66, 0.0), 4: (-0.07, -0.66, 0.0),
         5: (0.18, -0.45, 0), 6: (-0.18, -0.45, 0), 7: (0.2, -0.17, 0), 8: (-0.2, -0.17, 0), 62: (0.21, 0.08, 0), 41: (-0.21, 0.08, 0),
         9: (0.09, 0.0, 0), 10: (-0.09, 0.0, 0), 11: (0.1, 0.42, 0), 12: (-0.1, 0.42, 0), 13: (0.1, 0.82, 0), 14: (-0.1, 0.82, 0),
         17: (0.1, 0.87, 0.05), 20: (-0.1, 0.87, 0.05), 15: (0.1, 0.88, -0.12), 18: (-0.1, 0.88, -0.12)}


def fake(T=40, rise=0.3, focal=1000.0):
    kp = np.zeros((T, 70, 3))
    for i, p in STAND.items():
        kp[:, i] = p
    for j in range(21, 62):  # finger points at the wrists
        kp[:, j] = kp[:, 41 if j < 42 else 62]
    t = np.linspace(0, 1, T)
    cam_t = np.stack([np.zeros(T), -rise * np.sin(np.pi * t), np.full(T, 4.0)], -1)  # a hop (y down: up is -y)
    cam = kp + cam_t[:, None]
    kp2d = focal * cam[..., :2] / cam[..., 2:] + [640, 360]
    return {"kp3d": kp, "cam_t": cam_t, "kp2d": kp2d}


def test_mapping_covers_mediapipe_layout():
    m = sam3d_body.MHR70_TO_MP33
    assert m.shape == (33,) and m.min() >= 0 and m.max() < 70
    # the landmarks downstream code reads: shoulders, elbows, wrists, hips, knees, ankles
    assert list(m[[11, 12, 13, 14, 15, 16]]) == [5, 6, 7, 8, 62, 41]
    assert list(m[[23, 24, 25, 26, 27, 28]]) == [9, 10, 11, 12, 13, 14]


def test_both_roots_give_an_upright_trajectory_with_the_hop():
    d = fake()
    for root in ("camera", "image"):
        P = sam3d_body.human_trajectory(d, root=root)
        assert P.shape == (40, 33, 3) and np.isfinite(P).all()
        hip, ankle, head = P[:, [23, 24]].mean(1), P[:, [27, 28]].mean(1), P[:, 0]
        assert (head[:, 2] > hip[:, 2]).all() and (hip[:, 2] > ankle[:, 2]).all()  # z up
        rise = hip[:, 2].max() - hip[0, 2]
        assert 0.2 < rise < 0.4, (root, rise)  # the 0.3 m hop, metric either way


def test_landmarks_are_what_the_mediapipe_path_takes():
    import pose

    d = fake()
    world, image_xy, boxes = sam3d_body.landmarks(d)
    assert world.shape == (40, 33, 3) and image_xy.shape == (40, 33, 2) and boxes.shape == (40, 3)
    assert np.allclose(world[:, [23, 24]].mean(1), 0)  # hip-centred, like MediaPipe's world landmarks
    assert np.allclose(pose.landmark_pixels(image_xy, boxes), d["kp2d"][:, sam3d_body.MHR70_TO_MP33])


def test_mask_boxes_fill_frames_without_the_athlete():
    masks = np.zeros((3, 20, 30), bool)
    masks[0, 2:10, 5:9] = True
    masks[2, 4:12, 9:13] = True
    b = sam3d_body.mask_boxes(masks)
    assert list(b[0]) == [5, 2, 9, 10] and list(b[2]) == [9, 4, 13, 12]
    assert list(b[1]) == [7, 3, 11, 11]  # interpolated


def test_auto_falls_back_to_mediapipe_where_sam3d_body_is_not_set_up(monkeypatch):
    monkeypatch.setattr(sam3d_body, "available", lambda: False)
    assert sam3d_body.choose("auto") == "mediapipe" and sam3d_body.choose("mediapipe") == "mediapipe"
    with pytest.raises(RuntimeError, match="setup_sam3d.sh"):
        sam3d_body.choose("sam3d")
    monkeypatch.setattr(sam3d_body, "available", lambda: True)
    assert sam3d_body.choose("auto") == "sam3d"
