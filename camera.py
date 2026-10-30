"""Camera motion from the background, so a pan or zoom stops becoming athlete root motion.

`pose.human_trajectory` builds the root from the 2D hip track in pixels. If the camera pans, the
hips move in the image although the athlete may stand still. Here we estimate, for every frame, the
homography that maps its pixels onto the first frame's, from feature tracks on the background only
(the athlete is masked out), and `pose.human_trajectory` maps the landmarks through it before
lifting to metres. The root then lives in the first frame's image plane, as on a static camera.

What a homography covers: a camera that rotates about its centre (pan, tilt, roll) and zooms, which is
what a tripod or a broadcast camera does. It does not cover a camera that translates (a dolly, a
tracking shot, a drone) in a scene with depth: that needs a depth model (WHAM / TRAM / GVHMR).

Static cameras are a strict no-op: if the accumulated motion never moves the athlete's position by more
than `STATIC_PX` pixels, every homography is set to the identity, so the root is bit-identical. Where the
estimate is unreliable (archive film: most new frames do not register), only a clear pan is applied.

    python camera.py clip.mp4      # prints the pan/zoom summary for a video (whole frame, no mask)
"""

import numpy as np

# Below this accumulated displacement (pixels at the full frame, at the athlete) the camera is static.
STATIC_PX = 2.0
# Work on frames downscaled to this many pixels on the long side (features are found faster and the
# estimate is as good; the homography is scaled back to full-frame pixels).
WORK_SIDE = 960
MIN_INLIERS = 20
# Frame-to-frame motion model. A camera that rotates and zooms moves the image by a homography; at the
# focal lengths sport is filmed with (a pan of a few tens of degrees) a similarity (shift, roll, zoom)
# is within a pixel or two of it, and with 4 parameters instead of 8 it stays stable on frames with
# few background features (archive film, a textureless wall). "homography" is there for wide shots.
MODEL = "similarity"
# A frame registers straight to the current keyframe while at least this fraction of the keyframe's
# features still tracks into it; below that the previous frame becomes the keyframe. Registering to a
# keyframe instead of chaining frame pairs keeps the error from adding up frame after frame (a static
# camera never changes keyframe at all).
KEEP_FRAC = 0.4
# An estimate is reliable when at least this fraction of the distinct frames registered from features;
# an unreliable one is applied only when it shows a pan of more than UNRELIABLE_PAN of the frame width.
RELIABLE_FRAC = 0.8
UNRELIABLE_PAN = 0.05
# Phase correlation (the fallback) is trusted above this peak response.
PHASE_MIN = 0.05
# A hip step above this fraction of the athlete's height in one frame is a teleport (see `teleports`).
TELEPORT_FRAC = 0.35
# Gaussian blur (work pixels) before features are found and tracked.
BLUR = 1.5
# A frame pair whose colour histograms differ by more than this (Bhattacharyya, 0-1) is a cut.
CUT_DIST = 0.5


def _gray(frame, scale):
    """Downscaled grey image with its global brightness and contrast normalised (archive film flickers)."""
    import cv2

    g = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # A light blur: film grain and compression noise make corners that do not belong to the scene.
    g = cv2.GaussianBlur(g, (0, 0), BLUR).astype(np.float32)
    return np.clip((g - g.mean()) * (48.0 / (g.std() + 1e-6)) + 128, 0, 255).astype(np.uint8)


def _background(mask, scale, shape, dilate_frac=0.04):
    """uint8 mask of where background features may be taken: not on the (dilated) athlete, not on the
    outermost 2% of the frame (black bars, letterboxing)."""
    import cv2

    h, w = shape
    keep = np.full((h, w), 255, np.uint8)
    if mask is not None:
        m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        k = max(3, int(dilate_frac * max(h, w)))
        keep[cv2.dilate(m, np.ones((k, k), np.uint8)) > 0] = 0
    b = max(1, int(0.02 * min(h, w)))
    keep[:b], keep[-b:], keep[:, :b], keep[:, -b:] = 0, 0, 0, 0
    return keep


def _corners(g, keep):
    import cv2

    p = cv2.goodFeaturesToTrack(g, maxCorners=800, qualityLevel=0.01, minDistance=8, mask=keep, blockSize=7)
    return np.zeros((0, 2), np.float32) if p is None else p[:, 0]


def _track(g0, g1, p0, guess, keep1):
    """Track points p0 of g0 into g1 with pyramidal Lucas-Kanade from the guessed positions,
    forward-backward checked; points that land on the athlete (or off the frame) in g1 are dropped.
    Returns matched (a in g0, b in g1)."""
    import cv2

    if len(p0) < MIN_INLIERS:
        return p0[:0], p0[:0]
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p0 = np.ascontiguousarray(p0, np.float32).reshape(-1, 1, 2)
    p1 = np.ascontiguousarray(guess, np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, p1, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
    ok = (st[:, 0] == 1) & (stb[:, 0] == 1) & (np.linalg.norm(pb - p0, axis=-1)[:, 0] < 0.5)
    a, b = p0[ok, 0], p1[ok, 0]
    h, w = keep1.shape
    xi, yi = np.clip(b[:, 0].astype(int), 0, w - 1), np.clip(b[:, 1].astype(int), 0, h - 1)
    inside = (b[:, 0] >= 0) & (b[:, 0] < w) & (b[:, 1] >= 0) & (b[:, 1] < h) & (keep1[yi, xi] > 0)
    return a[inside], b[inside]


def _orb(g0, g1, keep0, keep1):
    """ORB matches (ratio test) on the background: the fallback when LK loses the scene (fast pans)."""
    import cv2

    orb = cv2.ORB_create(2000)
    k0, d0 = orb.detectAndCompute(g0, keep0)
    k1, d1 = orb.detectAndCompute(g1, keep1)
    if d0 is None or d1 is None or len(k0) < MIN_INLIERS or len(k1) < MIN_INLIERS:
        return np.zeros((0, 2)), np.zeros((0, 2))
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d0, d1, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    a = np.array([k0[m.queryIdx].pt for m in good]).reshape(-1, 2)
    b = np.array([k1[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return a, b


def _phase(g0, g1, keep0, keep1):
    """Global translation g1 -> g0 by phase correlation of the backgrounds (the athlete blanked out):
    the last resort when no features track, e.g. a new exposure on archive film where grain and flicker
    change every corner but the frame as a whole has only shifted (gate weave). Returns (3x3, response)."""
    import cv2

    def prep(g, keep):
        f = g.astype(np.float32)
        f[keep == 0] = f[keep > 0].mean() if (keep > 0).any() else 128
        return f

    (dx, dy), resp = cv2.phaseCorrelate(prep(g0, keep0), prep(g1, keep1), cv2.createHanningWindow(g0.shape[::-1], cv2.CV_32F))
    return np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1.0]]), resp


def _plausible(Hm, shape, ref, max_frac=0.08):
    """A homography a camera could have produced: its corners land within max_frac of the frame diagonal
    of where `ref` (the motion predicted from the previous frames) puts them, its scale is within
    [0.7, 1.4] and it keeps the frame convex. A few bad matches on a texture-poor frame can otherwise
    give a degenerate fit (thousands of pixels of 'motion')."""
    h, w = shape
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    a, b = apply(Hm[None], c[None])[0], apply(ref[None], c[None])[0]
    if not np.all(np.isfinite(a)) or np.linalg.norm(a - b, axis=-1).max() > max_frac * np.hypot(h, w):
        return False
    e = np.roll(a, -1, 0) - a
    cross = e[:, 0] * np.roll(e, -1, 0)[:, 1] - e[:, 1] * np.roll(e, -1, 0)[:, 0]
    s = np.sqrt(abs(np.linalg.det(Hm[:2, :2])))
    return bool(0.7 < s < 1.4 and (np.all(cross > 0) or np.all(cross < 0)))


def _fit(a, b, shape, ref, model):
    """Motion b -> a (maps the later frame's pixels onto the earlier one's) as a 3x3 matrix, RANSAC;
    None if too few inliers or implausible next to `ref`."""
    import cv2

    if len(a) < MIN_INLIERS:
        return None
    a, b = a.astype(np.float32), b.astype(np.float32)
    if model == "homography":
        Hm, inl = cv2.findHomography(b, a, cv2.RANSAC, 1.5, maxIters=2000, confidence=0.995)
    else:
        M, inl = cv2.estimateAffinePartial2D(b, a, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=2000,
                                             confidence=0.995)
        Hm = None if M is None else np.vstack([M, [0, 0, 1]])
    if Hm is None or inl.sum() < MIN_INLIERS:
        return None
    Hm = Hm / Hm[2, 2]
    return Hm if _plausible(Hm, shape, ref) else None


def _hist(frame):
    import cv2

    hsv = cv2.cvtColor(cv2.resize(np.ascontiguousarray(frame), (160, 90)), cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).astype(np.float32)


def detect_cuts(frames, threshold=CUT_DIST):
    """Indices t where frame t starts a new shot (colour-histogram jump from t-1). A spliced clip (a cut
    mid-motion) cannot be fixed downstream; this is the intake check that reports it."""
    import cv2

    hs = [_hist(f) for f in frames]
    d = [cv2.compareHist(hs[t - 1], hs[t], cv2.HISTCMP_BHATTACHARYYA) for t in range(1, len(hs))]
    return [t for t, x in enumerate(d, start=1) if x > threshold]


def estimate(frames, masks=None, work_side=WORK_SIDE, static_px=STATIC_PX, anchor=None, model=MODEL):
    """Per-frame camera motion relative to frame 0.

    frames: (T, H, W, 3) uint8. masks: (T, H, W) bool athlete masks (excluded from the features), or
    None. anchor: (T, 2) full-frame pixels where the athlete is (e.g. the hips), for the static test;
    default the frame centre. model: "similarity" (pan, tilt, roll and zoom, 4 parameters) or
    "homography" (8, adds the perspective of a wide-angle pan).

    Returns (Hs, info). Hs: (T, 3, 3); Hs[t] maps frame-t pixels to frame-0 pixels (Hs[0] = I).
    info: {"static", "reliable", "max_shift_px", "max_zoom", "keyframes", "cuts", "failed", "methods"}. "methods"
    counts how each frame was registered (features, orb, phase, repeat, none). "failed" lists frames
    whose motion could not be measured (too little background texture), where the previous frame's
    motion is continued; a cut also counts as failed and is bridged with no motion across it.
    """
    T, H, W = frames.shape[:3]
    scale = min(1.0, work_side / max(H, W))
    S, Sinv = np.diag([scale, scale, 1.0]), np.diag([1 / scale, 1 / scale, 1.0])
    cuts = set(detect_cuts(frames))

    def load(t):
        g = _gray(frames[t], scale)
        return g, _background(None if masks is None else masks[t], scale, g.shape)

    Hw = np.tile(np.eye(3), (T, 1, 1))  # in work pixels: frame t -> frame 0
    failed, keyframes = [], [0]
    methods = dict.fromkeys(["features", "orb", "phase", "repeat", "none"], 0)
    key = prev = (0, *load(0))  # (index, grey, background mask)
    kp = _corners(key[1], key[2])
    shape = key[1].shape
    for t in range(1, T):
        cur = (t, *load(t))
        if t in cuts:
            Hw[t] = Hw[t - 1]
            failed.append(t)
            key = prev = cur
            kp = _corners(cur[1], cur[2])
            keyframes.append(t)
            continue
        if np.abs(frames[t].astype(np.int16) - frames[t - 1]).mean() < 0.5:  # a repeated frame (archive film)
            Hw[t] = Hw[t - 1]
            methods["repeat"] += 1
            continue
        # The previous frame's motion, continued, is the prediction for this one.
        vel = np.linalg.inv(Hw[t - 2]) @ Hw[t - 1] if t >= 2 else np.eye(3)
        rel = None
        for _attempt in range(2):
            pred = np.linalg.inv(Hw[key[0]]) @ Hw[t - 1] @ vel  # frame t -> keyframe
            how = "features"
            a, b = _track(key[1], cur[1], kp, apply(np.linalg.inv(pred)[None], kp[None])[0], cur[2])
            rel = _fit(a, b, shape, pred, model) if len(a) >= KEEP_FRAC * len(kp) or key[0] == t - 1 else None
            if rel is None and key[0] == t - 1:
                how, rel = "orb", _fit(*_orb(key[1], cur[1], key[2], cur[2]), shape, pred, model)
            if rel is None and key[0] == t - 1:
                shift, resp = _phase(key[1], cur[1], key[2], cur[2])
                if resp > PHASE_MIN and _plausible(shift, shape, pred):
                    how, rel = "phase", shift
            if rel is not None or key[0] == t - 1:
                break
            key = prev  # lost the keyframe: register to the previous frame instead, and keep it as the key
            kp = _corners(key[1], key[2])
            keyframes.append(key[0])
        if rel is None:
            failed.append(t)
            how, rel = "none", pred
        methods[how] += 1
        Hw[t] = Hw[key[0]] @ rel
        prev = cur

    Hs = Sinv @ Hw @ S
    Hs /= Hs[:, 2:3, 2:3]
    if anchor is None:
        anchor = np.tile([W / 2, H / 2], (T, 1))
    shift = np.linalg.norm(apply(Hs, anchor) - anchor, axis=-1)
    zoom = np.sqrt(np.abs(np.linalg.det(Hs[:, :2, :2])))
    # Reliable: most distinct frames registered from features. Archive film (grain, flicker, a few new
    # exposures among repeated frames) is not; there a small apparent motion is indistinguishable from
    # the estimate's own noise, so only a clear pan (UNRELIABLE_PAN of the frame width) counts.
    measured = methods["features"] + methods["orb"]
    reliable = measured >= RELIABLE_FRAC * max(1, measured + methods["phase"] + methods["none"])
    still = shift.max() < (static_px if reliable else max(static_px, UNRELIABLE_PAN * W))
    info = {"static": bool(still), "reliable": bool(reliable), "max_shift_px": float(shift.max()),
            "max_zoom": float(np.max(np.maximum(zoom, 1 / zoom))), "keyframes": keyframes,
            "cuts": sorted(cuts), "failed": failed, "methods": methods}
    if info["static"]:
        Hs = np.tile(np.eye(3), (T, 1, 1))
    return Hs, info


def apply(Hs, pts):
    """Map pixels through per-frame homographies. Hs: (T, 3, 3); pts: (T, ..., 2). Same shape back."""
    pts = np.asarray(pts, float)
    flat = pts.reshape(len(pts), -1, 2)
    h = np.concatenate([flat, np.ones(flat.shape[:2] + (1,))], -1) @ np.transpose(Hs, (0, 2, 1))
    return (h[..., :2] / h[..., 2:3]).reshape(pts.shape)


# MediaPipe skeleton: the limbs drawn as thick lines for the landmark mask.
_LIMBS = [(11, 13), (13, 15), (15, 19), (12, 14), (14, 16), (16, 20), (23, 25), (25, 27), (27, 31), (24, 26),
          (26, 28), (28, 32), (27, 29), (28, 30), (11, 12), (23, 24), (11, 23), (12, 24)]


def landmark_masks(px, shape, width=0.12):
    """(T, H, W) bool athlete silhouettes drawn from the landmarks (full-frame pixels, (T, 33, 2), NaN
    allowed): limbs as lines `width` x the body's height thick, the torso filled, the head a disc. The
    camera estimate's mask when SAM's is not there; a padded box would hide most of the background as
    soon as the arms go out (jumping jacks)."""
    import cv2

    T = len(px)
    H, W = shape
    masks = np.zeros((T, H, W), np.uint8)
    for t in range(T):
        p = px[t]
        if np.isnan(p[[0, 11, 12, 23, 24]]).any():
            continue
        size = max(np.ptp(p[~np.isnan(p).any(-1)], axis=0).max(), 20.0)
        th = max(3, int(width * size))
        for a, b in _LIMBS:
            if not np.isnan(p[[a, b]]).any():
                cv2.line(masks[t], tuple(np.round(p[a]).astype(int)), tuple(np.round(p[b]).astype(int)), 1, th)
        cv2.fillConvexPoly(masks[t], np.round(p[[11, 12, 24, 23]]).astype(np.int32), 1)
        head = np.linalg.norm(p[0] - p[[11, 12]].mean(0))
        cv2.circle(masks[t], tuple(np.round(p[0]).astype(int)), int(0.7 * head + th / 2), 1, -1)
    return masks.astype(bool)


def teleports(px, height_px, max_frac=TELEPORT_FRAC, max_gap=3):
    """Frames where the hips have moved more than max_frac of the athlete's height per frame since the
    previous detection (camera motion removed; px may be NaN where pose was not found, gaps up to
    max_gap frames are bridged): faster than any sprint (0.35 x 1.8 m in 1/25 s is 16 m/s). That is a
    splice of the same shot, which the colour-histogram cut test cannot see, or a jump to another person."""
    hip = px[:, [23, 24]].mean(1)
    ok = np.flatnonzero(~np.isnan(hip).any(-1))
    out = []
    for a, b in zip(ok[:-1], ok[1:]):
        if b - a <= max_gap and np.linalg.norm(hip[b] - hip[a]) > max_frac * max(height_px, 1) * (b - a):
            out.append(int(b))
    return out


def intake(frames, masks, image_xy, boxes, compensate=True):
    """The coverage stage between pose and `pose.human_trajectory`: the camera motion (athlete masked
    out by its mask joined with a box around its landmarks, so a whole-frame `--no-sam` mask still
    works), shot cuts and the pixel-height check. Returns (Hs or None, report); Hs goes to
    `pose.human_trajectory(camera=...)`, the report to `camera.json`."""
    import pose

    px = pose.landmark_pixels(pose.fill_and_smooth(image_xy), boxes)
    report = {"height": pose.height_check(px)}
    T, H, W = frames.shape[:3]
    fg = landmark_masks(px, (H, W))
    if masks is not None and masks.mean() < 0.9:  # a real athlete mask, not the whole frame
        fg |= masks
    Hs, info = estimate(frames, fg, anchor=px[:, [pose.L_HIP, pose.R_HIP]].mean(1))
    report["camera"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}
    report["camera"]["compensated"] = bool(compensate and not info["static"])
    report["teleports"] = teleports(apply(Hs, pose.landmark_pixels(image_xy, boxes)), report["height"]["height_px"])
    if masks is not None and masks.mean() < 0.9:
        import athlete

        report["mask"] = athlete.continuity(masks)
    warnings = [report["height"]["warning"]] if "warning" in report["height"] else []
    if info["cuts"]:
        warnings.append(f"shot cuts at frames {info['cuts']}: a clip has to be one continuous shot")
    jumps = sorted(set(report["teleports"]) | set(report.get("mask", {}).get("jumps", [])))
    if jumps:
        warnings.append(f"the athlete jumps across the frame at frames {jumps}: a splice of the same shot, or "
                        "the tracking switched person")
    if not info["reliable"]:
        warnings.append("camera motion is unreliable here (few frames register: archive film, little background); "
                        + ("a large pan was compensated anyway" if report["camera"]["compensated"] else "left uncompensated"))
    n_bad = len(set(info["failed"]) - set(info["cuts"]))
    if n_bad > 0.1 * T:
        warnings.append(f"camera motion not measurable in {n_bad}/{T} frames (no background texture)")
    report["warnings"] = warnings
    for w in warnings:
        print("intake warning:", w)
    return (Hs if report["camera"]["compensated"] else None), report


def warp_synthetic(frames, shifts, zooms=None, out_shape=None):
    """Build a synthetic moving-camera video from a static one: frame t is the static frame zoomed by
    zooms[t] about the frame centre and shifted by shifts[t] (pixels, (dx, dy): the scene moves by
    +shift in the image, i.e. the camera pans by -shift), then centre-cropped to out_shape. Returns
    (frames, Hs_true) with Hs_true[t] mapping the new frame t's pixels to the new frame 0's, the
    quantity `estimate` should recover."""
    import cv2

    T, H, W = frames.shape[:3]
    zooms = np.ones(T) if zooms is None else np.asarray(zooms, float)
    oh, ow = out_shape or (H, W)
    c = np.array([W / 2, H / 2])
    A = np.array([[[z, 0, (1 - z) * c[0] + s[0] - (W - ow) / 2],
                   [0, z, (1 - z) * c[1] + s[1] - (H - oh) / 2], [0, 0, 1]] for z, s in zip(zooms, shifts)])
    out = np.stack([cv2.warpPerspective(np.ascontiguousarray(f), M, (ow, oh), borderMode=cv2.BORDER_REFLECT)
                    for f, M in zip(frames, A)])
    return out, A[0] @ np.linalg.inv(A)


if __name__ == "__main__":
    import sys

    import mediapy

    frames = np.asarray(mediapy.read_video(sys.argv[1]))
    print(estimate(frames)[1])
