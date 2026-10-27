"""Goal relevance from V-JEPA 2's predictor (Meta's video world model): which moments, and which
body parts' motion, the goal events of a clip are predicted from. A diagnostic: on squat_jump the
predictor's signal is at noise level and the encoder's is salience, not goal relevance (it rises
for injected arm-waving), so skill/importance.py doesn't mix it in by default; docs/GOAL_RELEVANCE.md.

For each goal event (squat bottom, takeoff, apex, landing; skill/importance.goal_events) the clip's
retarget is rendered as a robot silhouette (MuJoCo segmentation, the reward's own domain) over the
~1 s before the event plus the event itself. V-JEPA 2's predictor gets the encoder tokens of the
context frames and predicts the tokens of the event frames (which are blanked in the context video,
so nothing leaks through the encoder's attention); the error is L1 between layer-normed predicted and
actual tokens over the patches the body covers. Then, for every short window of the context, the
motion of the whole body or of one body group is removed (its joints interpolated straight across
the window, so the pose stays continuous), re-rendered with the baseline's framing, and predicted
again. The relative increase in error is that window's / group's relevance for the event:
occlusion attribution (Zeiler & Fergus 1311.2901) with a world model as the function, i.e. a
counterfactual "what if the arms hadn't moved then".

Cost: per event (1 + windows x 6) clips of context + target frames through ViT-L; squat_jump with
the defaults is ~200 clips of 36 frames: about a minute on an A100, ~10 min on an M-series GPU (mps).
"""

import numpy as np

from latent import silhouette
from latent.vjepa import MODEL, TUBELET
from skill.importance import EVENT_WEIGHT, GROUPS, QPOS_GROUPS

CONTEXT, TARGET, OCCLUDE = 32, 4, 8  # frames: context before the event, event frames, occlusion window
PATCH = 16
AZ_BUCKET = 15  # experiments/cross_embodiment_probe.py: renders per viewpoint bucket


def azimuth_of(clip):
    """A clip folder's video viewpoint (deg), rounded to the render bucket the reward uses."""
    import json
    from pathlib import Path
    clip = Path(clip)
    if (clip / "video_latent.npz").exists():
        az = float(np.load(clip / "video_latent.npz")["view_azimuth"])
    else:
        az = float(json.loads((clip / "meta.json").read_text()).get("view_azimuth", -150.0))
    return int(round(az / AZ_BUCKET) * AZ_BUCKET)


# ---------------------------------------------------------------- counterfactual motion + rendering

def remove_motion(qpos, a, b, cols):
    """qpos (L, 36) with columns `cols` interpolated straight from frame a - 1 to frame b across
    [a, b): the motion inside the window is gone, the pose before and after is unchanged."""
    q = qpos.copy()
    lo, hi = max(a - 1, 0), min(b, len(q) - 1)
    u = (np.arange(a, b) - lo) / max(hi - lo, 1)
    q[a:b, cols] = (1 - u[:, None]) * qpos[lo, cols] + u[:, None] * qpos[hi, cols]
    if 3 in cols:
        q[a:b, 3:7] /= np.linalg.norm(q[a:b, 3:7], axis=1, keepdims=True)
    return q


GROUP_COLS = {g: list(range(s.start, s.stop)) for g, s in QPOS_GROUPS.items()}
GROUP_COLS["all"] = list(range(0, 36))


def crop_params(masks, body_frac=0.6, feet_at=0.85):
    """latent/silhouette.normalize's framing, computed once from the baseline so every
    counterfactual is cropped identically (a changed crop would itself look like motion)."""
    from scipy.ndimage import uniform_filter1d
    T = len(masks)
    seen = masks.any((1, 2))
    box = np.full((T, 4), np.nan)
    for t in np.flatnonzero(seen):
        ys, xs = np.nonzero(masks[t])
        box[t] = xs.min(), ys.min(), xs.max(), ys.max()
    ti = np.arange(T)
    for j in range(4):
        box[:, j] = np.interp(ti, ti[seen], box[seen, j])
    side = int(np.percentile(box[seen, 3] - box[seen, 1], 90) / body_frac)
    cx = uniform_filter1d((box[:, 0] + box[:, 2]) / 2, size=31, mode="nearest")
    cy = np.full(T, np.median(box[seen, 3]) - feet_at * side + side / 2)
    return side, cx, cy


def apply_crop(masks, params, size=silhouette.SIZE):
    import cv2
    side, cx, cy = params
    T, H, W = masks.shape
    padded = np.zeros((H + 2 * side, W + 2 * side), np.uint8)
    out = np.empty((T, size, size, 3), np.uint8)
    for t in range(T):
        padded[side:side + H, side:side + W] = masks[t] * 255
        x0, y0 = int(round(cx[t] - side / 2)) + side, int(round(cy[t] - side / 2)) + side
        out[t] = cv2.resize(padded[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_AREA)[..., None]
    return out


def event_clip(qpos, frame, context=CONTEXT, target=TARGET):
    """Frames [frame - context, frame + target), edge-padded at the clip's ends."""
    idx = np.clip(np.arange(frame - context, frame + target), 0, len(qpos) - 1)
    return qpos[idx]


def counterfactuals(model, q, azimuth, context=CONTEXT, occlude=OCCLUDE, groups=("all", *GROUPS)):
    """-> (videos (1 + n, L, 256, 256, 3) uint8 [baseline first], [(window_start, window_end, group)])."""
    variants, keys = [q], []
    for a in range(0, context, occlude):
        for g in groups:
            variants.append(remove_motion(q, a, min(a + occlude, context), GROUP_COLS[g]))
            keys.append((a, min(a + occlude, context), g))
    masks = [silhouette.robot_masks(model, v, azimuth=azimuth) for v in variants]
    params = crop_params(masks[0])
    return np.stack([apply_crop(m, params) for m in masks]), keys


# ---------------------------------------------------------------- the predictor

class Predictor:
    """V-JEPA 2 encoder + predictor. `model`/`preprocess` can be injected (tests use a tiny
    randomly initialised VJEPA2Model); otherwise the pretrained MODEL is loaded."""

    def __init__(self, name=MODEL, device=None, model=None, preprocess=None, batch=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            from transformers import AutoModel, AutoVideoProcessor
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            model = AutoModel.from_pretrained(name, torch_dtype=dtype)
            proc = AutoVideoProcessor.from_pretrained(name)
            preprocess = lambda v: proc(list(v), return_tensors="pt")["pixel_values_videos"]
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.batch = batch or (8 if self.device == "cuda" else 2)
        cfg = self.model.config
        self.grid = cfg.crop_size // PATCH if isinstance(cfg.patch_size, int) else cfg.crop_size // cfg.patch_size[0]

    def _pixels(self, videos):
        """(N, L, H, W, 3) uint8 -> model input (N, L, 3, crop, crop) in the model's dtype."""
        torch = self.torch
        x = self.preprocess(torch.from_numpy(np.ascontiguousarray(videos)).permute(0, 1, 4, 2, 3))
        return x.to(self.device, dtype=next(self.model.parameters()).dtype)

    def _body_patches(self, video, context_frames):
        """The patches the body covers in the event frames (dilated by one patch), (target tokens,)."""
        from scipy.ndimage import binary_dilation
        body = video[context_frames:, :, :, 0] > 0
        h = video.shape[1] // self.grid
        cov = body.reshape(len(body), self.grid, h, self.grid, h).any((2, 4))
        cov = cov.reshape(-1, TUBELET, self.grid, self.grid).any(1)
        return np.stack([binary_dilation(c, iterations=1) for c in cov]).reshape(-1)

    def shifts(self, videos, context_frames):
        """videos (N, L, H, W, 3): row 0 the baseline. The encoder attends across the whole clip, so
        its tokens of the event frames [context_frames, L) depend on the motion before them: per
        video, the L1 distance of its (layer-normed) event tokens from the baseline's, over the
        body's patches -> ((N,), body patch count)."""
        torch = self.torch
        F = torch.nn.functional
        steps, ctx_steps, P = videos.shape[1] // TUBELET, context_frames // TUBELET, self.grid ** 2
        cov = self._body_patches(videos[0], context_frames)
        sel = torch.as_tensor(cov, device=self.device)
        toks = []
        with torch.no_grad():
            for i in range(0, len(videos), self.batch):
                h = self.model(pixel_values_videos=self._pixels(videos[i:i + self.batch]), skip_predictor=True).last_hidden_state
                h = h[:, ctx_steps * P:steps * P].float()
                toks.append(F.layer_norm(h, h.shape[-1:])[:, sel].cpu())
        toks = torch.cat(toks)
        return (toks - toks[:1]).abs().mean((1, 2)).numpy(), int(cov.sum())

    def errors(self, videos, context_frames):
        """videos (N, L, H, W, 3): row 0 the baseline. Predict the tokens of frames
        [context_frames, L) from the context frames of each video (event frames blanked) ->
        (per-video L1 error over the body's patches (N,), body patch count)."""
        torch = self.torch
        N, L = videos.shape[:2]
        steps, ctx_steps, P = L // TUBELET, context_frames // TUBELET, self.grid ** 2
        F = torch.nn.functional
        with torch.no_grad():
            tokens = self.model(pixel_values_videos=self._pixels(videos[:1]), skip_predictor=True).last_hidden_state
            tgt = tokens[0, ctx_steps * P:steps * P].float()
            tgt = F.layer_norm(tgt, tgt.shape[-1:])
            cov = self._body_patches(videos[0], context_frames)
            sel = torch.as_tensor(cov, device=self.device)
            cm = torch.arange(ctx_steps * P, device=self.device)[None]
            tm = torch.arange(ctx_steps * P, steps * P, device=self.device)[None]
            ctx = videos.copy()
            ctx[:, context_frames:] = 0
            errs = []
            for i in range(0, N, self.batch):
                x = self._pixels(ctx[i:i + self.batch])
                B = len(x)
                out = self.model(pixel_values_videos=x, context_mask=[cm.repeat(B, 1)], target_mask=[tm.repeat(B, 1)])
                pred = F.layer_norm(out.predictor_output.last_hidden_state.float(), tgt.shape[-1:])
                e = (pred - tgt[None]).abs().mean(-1)  # (B, target tokens)
                errs.append((e[:, sel].mean(1) if sel.any() else e.mean(1)).cpu().numpy())
        return np.concatenate(errs), int(cov.sum())


def scores(model, qpos, events, fps, azimuth=-150, device=None, predictor=None, context=CONTEXT,
           target=TARGET, occlude=OCCLUDE, log=print, mode="predictor"):
    """-> (scores (T, len(GROUPS)) >= 0 per frame and body group, detail dict for the npz).
    mode "predictor": each event's relative error increase when a group's motion is removed from a
    window; mode "encoder": the shift of the event's encoder tokens (Predictor.shifts), relative to
    the shift from removing all motion in the window nearest the event. Credited to that group over
    that window's frames, times the event's weight."""
    predictor = predictor or Predictor(device=device)
    T = len(qpos)
    S = np.zeros((T, len(GROUPS)))
    whole = np.zeros(T)
    seen, rows = set(), []
    for ev in events:
        if ev["kind"] == "balance" or ev["frame"] in seen:
            continue
        seen.add(ev["frame"])
        q = event_clip(qpos, ev["frame"], context, target)
        videos, keys = counterfactuals(model, q, azimuth, context, occlude)
        if mode == "predictor":
            err, n_body = predictor.errors(videos, context)
            base = max(err[0], 1e-6)
        else:
            err, n_body = predictor.shifts(videos, context)
            base = max(err[1 + keys.index((keys[-1][0], keys[-1][1], "all"))], 1e-6)
            err = base + err  # so that e / base - 1 is the shift relative to base
        start = ev["frame"] - context
        for (a, b, g), e in zip(keys, err[1:]):
            rel = max(e / base - 1.0, 0.0)
            fa, fb = max(start + a, 0), max(start + b, 0)
            if g == "all":
                whole[fa:fb] += EVENT_WEIGHT[ev["kind"]] * rel
            else:
                S[fa:fb, GROUPS.index(g)] += EVENT_WEIGHT[ev["kind"]] * rel
            rows.append((ev["frame"], start + a, start + b, ["all", *GROUPS].index(g), e / base - 1.0))
        log(f"  V-JEPA {mode}, {ev['kind']} at {ev['frame'] / fps:.2f} s: reference L1 {base:.3f} over "
            f"{n_body} body patches; largest relative increase {max(err[1:] / base - 1):+.3f}")
    return S, {"whole": whole, "rows": np.array(rows, dtype=np.float32),
               "row_columns": np.array(["event_frame", "start", "end", "group(0=all)", "rel_increase"])}
