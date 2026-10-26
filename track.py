"""SAM 3.1: text-prompted tracking of the athlete -> per-frame masks and crops.

Needs a CUDA GPU (Colab L4 or A100; T4 is too old for the attention kernels) and approved
access to the gated checkpoint at huggingface.co/facebook/sam3.1 (log in with an HF token).
"""

import numpy as np

import config

_predictor = None


def predictor():
    global _predictor
    if _predictor is None:
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        # use_fa3 defaults to True, which needs FlashAttention 3 (H100 only). Colab GPUs aren't.
        _predictor = build_sam3_multiplex_video_predictor(use_fa3=False)
    return _predictor


def track(video_path, prompt=config.SAM_PROMPT, frame_index=0):
    """Return {frame_index: outputs}. outputs has out_obj_ids, out_probs, out_boxes_xywh,
    out_binary_masks (one entry per detected object)."""
    p = predictor()
    session_id = p.handle_request(dict(type="start_session", resource_path=str(video_path)))["session_id"]
    p.handle_request(dict(type="add_prompt", session_id=session_id, frame_index=frame_index, text=prompt))
    outputs = {}
    for resp in p.handle_stream_request(dict(type="propagate_in_video", session_id=session_id)):
        outputs[resp["frame_index"]] = resp["outputs"]
    p.handle_request(dict(type="close_session", session_id=session_id))
    return outputs


def main_track_masks(outputs, n_frames):
    """Pick the object seen most often with the highest confidence (the athlete), return
    (n_frames, H, W) bool masks; frames where it's missing are all False."""
    score = {}
    for out in outputs.values():
        for obj_id, prob in zip(out["out_obj_ids"], out["out_probs"]):
            score[int(obj_id)] = score.get(int(obj_id), 0.0) + float(prob)
    if not score:
        raise RuntimeError("SAM found nothing. Try another prompt, e.g. 'person'.")
    best = max(score, key=score.get)
    first = next(iter(outputs.values()))["out_binary_masks"]
    h, w = np.asarray(first).shape[-2:]
    masks = np.zeros((n_frames, h, w), bool)
    for f, out in outputs.items():
        for obj_id, m in zip(out["out_obj_ids"], out["out_binary_masks"]):
            if int(obj_id) == best:
                masks[f] = np.asarray(m, bool)
    return masks, best


def overlay(frames, masks, color=(255, 64, 64), alpha=0.5):
    """Tint the tracked athlete for the 'show the tracking' part of the demo."""
    out = frames.astype(np.float32).copy()
    out[masks] = out[masks] * (1 - alpha) + np.array(color) * alpha
    return out.astype(np.uint8)


def crop_to_mask(frames, masks, pad=0.2, background=114):
    """Square crops that follow the athlete, for MediaPipe. One crop size for the whole clip
    keeps the pixel scale constant. Everything outside the (dilated) mask is greyed out so
    MediaPipe can't latch onto another person.

    Returns crops (T, S, S, 3) and boxes (T, 3) = (x0, y0, S) in full-frame pixels; x0/y0 can
    be negative (the frame is padded)."""
    import cv2
    from scipy.ndimage import uniform_filter1d

    T, H, W = masks.shape
    seen = masks.any((1, 2))
    if not seen.any():
        raise RuntimeError("Mask is empty in every frame.")
    boxes = np.full((T, 4), np.nan)  # x_min, y_min, x_max, y_max
    for t in np.flatnonzero(seen):
        ys, xs = np.nonzero(masks[t])
        boxes[t] = xs.min(), ys.min(), xs.max(), ys.max()
    t_all = np.arange(T)
    for j in range(4):
        boxes[:, j] = np.interp(t_all, t_all[seen], boxes[seen, j])
    side = int(np.max(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])) * (1 + 2 * pad))
    centre = uniform_filter1d((boxes[:, :2] + boxes[:, 2:]) / 2, size=5, axis=0, mode="nearest")
    x0y0 = np.round(centre - side / 2).astype(int)

    kernel = np.ones((max(3, side // 30),) * 2, np.uint8)
    padded = np.full((H + 2 * side, W + 2 * side, 3), background, np.uint8)
    crops = np.empty((T, side, side, 3), np.uint8)
    for t in range(T):
        frame = frames[t]
        if seen[t]:
            keep = cv2.dilate(masks[t].astype(np.uint8), kernel).astype(bool)
            frame = np.where(keep[..., None], frame, np.uint8(background))
        padded[side:side + H, side:side + W] = frame
        x, y = x0y0[t] + side
        crops[t] = padded[y:y + side, x:x + side]
    return crops, np.column_stack([x0y0, np.full(T, side)])
