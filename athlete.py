"""Which person is the athlete, when the frame has several.

SAM 3.1's text prompt returns every person it finds; `track.main_track_masks` then keeps the object
with the highest summed confidence, which in a meet or a match is often a bystander. Here the caller
picks the athlete on the first frame instead, with one string (the CLI's `--athlete`):

    "640,360"           a click: the person under (or nearest to) this pixel
    "500,120,780,900"   a box x0,y0,x1,y1: the person whose mask overlaps it most
    "1"                 the index: people on the first frame counted from the left, from 0

Coordinates are full-frame pixels of the video as mediapy reads it (rotation applied).

`track_people` is the same choice without SAM (CPU, MediaPipe only): it finds the chosen person on
the first frame and follows them, re-detecting in a crop around the last position (so a small, distant
athlete is detected at a usable resolution), and returns box masks for `track.crop_to_mask`.
"""

import numpy as np

import config


def parse(spec):
    """'x,y' -> {'point'}, 'x0,y0,x1,y1' -> {'box'}, 'n' -> {'index'}; None/'' -> None."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, dict):
        return spec
    parts = [float(p) for p in str(spec).replace(" ", "").split(",")]
    if len(parts) == 1 and parts[0] == int(parts[0]) and parts[0] >= 0:
        return {"index": int(parts[0])}
    if len(parts) == 2:
        return {"point": tuple(parts)}
    if len(parts) == 4:
        x0, y0, x1, y1 = parts
        return {"box": (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))}
    raise ValueError(f"--athlete {spec!r}: expected 'x,y', 'x0,y0,x1,y1' or a person index")


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def choose(boxes, select, centres=None, inside=None):
    """Index of the chosen person among candidates on one frame. boxes: (N, 4) x0,y0,x1,y1. centres:
    (N, 2) for the click/left-to-right rules (default box centres). inside(i, x, y) -> bool says whether
    the click is on candidate i (default: inside its box)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not len(boxes):
        raise RuntimeError("no person on the first frame to choose from")
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2 if centres is None else np.asarray(centres, float)
    if "index" in select:
        order = np.argsort(centres[:, 0], kind="stable")
        if select["index"] >= len(order):
            raise RuntimeError(f"--athlete {select['index']}: only {len(order)} people on the first frame")
        return int(order[select["index"]])
    if "point" in select:
        x, y = select["point"]
        if inside is None:
            def inside(i, x, y):
                b = boxes[i]
                return b[0] <= x <= b[2] and b[1] <= y <= b[3]
        hits = [i for i in range(len(boxes)) if inside(i, x, y)]
        cand = hits or range(len(boxes))
        return int(min(cand, key=lambda i: np.hypot(*(centres[i] - (x, y)))))
    if "box" in select:
        return int(np.argmax([_iou(b, select["box"]) for b in boxes]))
    raise ValueError(f"bad selection {select!r}")


def select_object(outputs, select):
    """SAM 3.1 outputs ({frame: {out_obj_ids, out_binary_masks, ...}}) -> the chosen object id, picked
    on the earliest frame where any object is seen."""
    select = parse(select)
    for f in sorted(outputs):
        out = outputs[f]
        cands = [(int(o), np.asarray(m, bool)) for o, m in zip(out["out_obj_ids"], out["out_binary_masks"])
                 if np.asarray(m).any()]
        if cands:
            break
    else:
        raise RuntimeError("SAM found nothing. Try another prompt, e.g. 'person'.")
    ids = [o for o, _ in cands]
    masks = [m for _, m in cands]
    boxes = np.array([_bbox(m) for m in masks])
    centres = np.array([np.argwhere(m)[:, ::-1].mean(0) for m in masks])

    def on_mask(i, x, y):
        h, w = masks[i].shape
        return 0 <= int(y) < h and 0 <= int(x) < w and bool(masks[i][int(y), int(x)])

    return ids[choose(boxes, select, centres, on_mask)]


def continuity(masks, min_iou=0.2):
    """Identity-switch check on a (T, H, W) mask track: frames where the mask's box jumps (IoU with the
    previous seen frame's box below min_iou). A switch to another person shows up here."""
    seen = np.flatnonzero(masks.any((1, 2)))
    boxes = {t: _bbox(masks[t]) for t in seen}
    jumps = [int(b) for a, b in zip(seen[:-1], seen[1:]) if _iou(boxes[a], boxes[b]) < min_iou]
    return {"seen_frac": round(len(seen) / max(1, len(masks)), 3), "jumps": jumps}


# --- Without SAM: MediaPipe person tracking on CPU -----------------------------------------------------

def _landmarker(model_path, num_poses):
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
                                        running_mode=vision.RunningMode.IMAGE, num_poses=num_poses,
                                        min_pose_detection_confidence=0.3)
    return vision.PoseLandmarker.create_from_options(opts)


def _detect(lm, image, x0=0.0, y0=0.0, scale=1.0):
    """People in an image -> list of (33, 2) landmark pixels in the source frame (crop offset x0, y0,
    crop upscaled by `scale`)."""
    import mediapipe as mp

    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)))
    return [np.array([(x0 + p.x * image.shape[1] / scale, y0 + p.y * image.shape[0] / scale) for p in pl])
            for pl in res.pose_landmarks]


def _look(frame, pts):
    """Hue-saturation histogram of the torso (shoulders to hips): what the athlete is wearing."""
    import cv2

    q = pts[[11, 12, 23, 24]]
    x0, y0 = np.floor(q.min(0)).astype(int)
    x1, y1 = np.ceil(q.max(0)).astype(int)
    H, W = frame.shape[:2]
    x0, x1, y0, y1 = max(0, x0), min(W, max(x1, x0 + 4)), max(0, y0), min(H, max(y1, y0 + 4))
    if x1 <= x0 or y1 <= y0:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(frame[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).astype(np.float32)


def _same_look(a, b):
    import cv2

    return a is None or b is None or cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA) < LOOK_DIST


# Torso colour distance (Bhattacharyya) above which a detection is someone else in other clothes.
LOOK_DIST = 0.6


def _nearest(people, centre, radius, frame=None, look=None):
    """The detected person whose hips are nearest `centre`, within `radius` pixels and, given the
    athlete's look, dressed like them."""
    best, best_d = None, radius
    for p in people:
        d = np.linalg.norm(p[[23, 24]].mean(0) - centre)
        if d < best_d and (look is None or _same_look(look, _look(frame, p))):
            best, best_d = p, d
    return best


def _box(pts, pad=0.15):
    lo, hi = pts.min(0), pts.max(0)
    d = (hi - lo) * pad
    lo, hi = lo - d, hi + d
    lo[1] -= 0.08 * (hi[1] - lo[1])  # the nose is not the top of the head
    return np.concatenate([lo, hi])


def _crop(frame, centre, side, out=512):
    """Square crop of `side` around `centre` (padded), resized to at least `out` px."""
    import cv2

    H, W = frame.shape[:2]
    side = int(max(32, side))
    x0, y0 = int(round(centre[0] - side / 2)), int(round(centre[1] - side / 2))
    pad = np.full((side, side, 3), 114, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    scale = max(1.0, out / side)
    if scale > 1:
        pad = cv2.resize(pad, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return pad, x0, y0, scale


def track_people(frames, select=None, model_path=config.POSE_MODEL, max_people=6, coast=15):
    """Follow one person through the clip without SAM. select: as `parse` (None = the largest person
    on the first frame). Returns (masks (T, H, W) bool boxes, info)."""
    select = parse(select)
    T, H, W = frames.shape[:3]
    boxes = np.full((T, 4), np.nan)
    with _landmarker(model_path, max_people) as lm, _landmarker(model_path, 3) as lm_crop:
        people = _detect(lm, frames[0])
        n_first = len(people)
        if select and "box" in select:
            b = np.array(select["box"], float)
            near = _detect(lm_crop, *_crop(frames[0], (b[:2] + b[2:]) / 2, 1.6 * max(b[2:] - b[:2])))
            cand = [_box(p) for p in near] or [b]
            box = cand[int(np.argmax([_iou(c, b) for c in cand]))]
        else:
            if not people:
                raise RuntimeError("no person found on the first frame; give the athlete's box with --athlete")
            cand = np.array([_box(p) for p in people])
            if select:
                hips = np.array([p[[23, 24]].mean(0) for p in people])
                box = cand[choose(cand, select, hips)]
            else:
                box = cand[int(np.argmax((cand[:, 2] - cand[:, 0]) * (cand[:, 3] - cand[:, 1])))]
        boxes[0] = box
        mine = _nearest(_detect(lm_crop, *_crop(frames[0], (box[:2] + box[2:]) / 2, 1.6 * max(box[2:] - box[:2]))),
                        (box[:2] + box[2:]) / 2, max(box[2:] - box[:2]))
        look = None if mine is None else _look(frames[0], mine)
        vel = np.zeros(2)
        last, lost = box, 0  # the last box where the athlete was actually seen
        for t in range(1, T):
            size = max(last[2] - last[0], last[3] - last[1])
            centre = (last[:2] + last[2:]) / 2 + vel * (lost + 1)
            best = _nearest(_detect(lm_crop, *_crop(frames[t], centre, 2.5 * size)), centre, 1.0 * size, frames[t], look)
            if best is None:  # not near where expected: look over the whole frame, a little wider
                best = _nearest(_detect(lm, frames[t]), centre, (1.5 + 0.2 * lost) * size, frames[t], look)
            if best is None:
                lost += 1
                if lost <= coast:  # keep a predicted box for a short gap; past that the frame has no mask
                    boxes[t] = last + np.tile(vel * lost, 2)
                continue
            boxes[t] = _box(best)
            step = ((boxes[t, :2] + boxes[t, 2:]) - (last[:2] + last[2:])) / 2 / (lost + 1)
            vel = 0.5 * vel + 0.5 * step
            last, lost = boxes[t], 0
    masks = np.zeros((T, H, W), bool)
    for t in np.flatnonzero(~np.isnan(boxes[:, 0])):
        x0, y0 = np.clip(np.floor(boxes[t, :2]), 0, [W, H]).astype(int)
        x1, y1 = np.clip(np.ceil(boxes[t, 2:]), 0, [W, H]).astype(int)
        masks[t, y0:y1, x0:x1] = True
    return masks, {"people_first_frame": n_first, "tracked_frac": round(float(masks.any((1, 2)).mean()), 3)}
