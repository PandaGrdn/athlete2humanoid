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


def crop_to_mask(frames, masks, pad=0.15):
    """Day 2: crop each frame to the athlete's (padded) bounding box for MediaPipe."""
    raise NotImplementedError("Day 2")
