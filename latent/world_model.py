"""V-JEPA 2 as a world model: from the video seen so far, imagine its latent 0.5 s and 1 s ahead.

latent/video_condition.py's causal latent (A2H-TrackingVJCausal) describes the ~1 s of video ending
now. The non-causal one sees ~1.5 s ahead, which is why it does better, but no robot watching a
video live has that. Here V-JEPA 2's predictor fills the gap: it was trained to predict the latents
of masked tokens from the visible ones, so given only the past it can be asked for the future.

For each V-JEPA step s (2 frames; step s ends at frame 2s, as vjepa.embed_trailing):

1. The encoder sees only the 32 frames [2s - 31, 2s] (the first frame repeated before the start):
   16 time steps x 256 patches of tokens. Nothing after frame 2s goes in.
2. The predictor gets those as context and positions for the next HORIZON steps (frames
   2s + 1 ... 2s + 30) as targets, and predicts their tokens. The encoder never sees the future
   frames, not even blanked: the predictor is called directly with the context tokens.
3. "The latent h steps ahead" is the trailing window that would end at frame 2(s + h): its seen part
   from the encoder's tokens, the rest from the predictor's (predicted_trailing). Both in the space
   V-JEPA 2 is trained in: tokens layer-normed, then averaged over space per time step.

How good the imagined future is (quality): against the true trailing latent at s + h, compared with
just copying the present (the trailing latent at s): cosine, and squared error relative to the copy.

The fallback when the predictor isn't better than the copy: LatentPredictor, ridge regression from
the past latents (now, ~0.27 s and ~0.53 s ago) to the change h steps ahead, fitted on training clips.
"""

import numpy as np

from latent.vjepa import TUBELET, WINDOW, VJepa

STEPS = WINDOW // TUBELET   # time steps in a 32-frame window
HORIZONS = (8, 15)          # steps ahead: ~0.53 s and 1.0 s at 30 fps
LAGS = (0, 4, 8)            # LatentPredictor's inputs: the latent now, 4 and 8 steps ago


class WorldModel(VJepa):
    """VJepa plus its predictor. _tokens / _predict are the two model calls (tests replace them)."""

    def _tokens(self, clips):
        """clips: list of (T, H, W, 3) uint8 -> encoder tokens (N, T / 2 * patches, D), time-major."""
        torch = self.torch
        videos = [torch.from_numpy(np.ascontiguousarray(c)).permute(0, 3, 1, 2) for c in clips]
        inputs = self.processor(videos, return_tensors="pt", device=self.model.device).to(self.model.device)
        inputs = {k: v.to(self.model.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
        with torch.no_grad():
            return self.model.get_vision_features(**inputs)

    def _predict(self, tokens, horizon, pad=None):
        """Encoder tokens of the context (N, n_ctx, D) -> predicted tokens of the next `horizon` time
        steps (N, horizon * patches, D). Only the context positions are read: the target slots of
        encoder_hidden_states (`pad`, zeros) are never gathered (the predictor takes context_mask's indices)."""
        torch = self.torch
        B, n, D = tokens.shape
        P = n // STEPS
        dev = tokens.device
        hidden = torch.cat([tokens, tokens.new_zeros(B, horizon * P, D) if pad is None else pad], 1)
        cm = torch.arange(n, device=dev)[None].repeat(B, 1)
        tm = torch.arange(n, n + horizon * P, device=dev)[None].repeat(B, 1)
        with torch.no_grad():
            return self.model.predictor(encoder_hidden_states=hidden, context_mask=[cm], target_mask=[tm]).last_hidden_state

    def imagine(self, video, horizon=max(HORIZONS), batch=8):
        """(T, H, W, 3) video -> {"raw": (S, D) the trailing latent as vjepa.embed_trailing (encoder
        tokens averaged), "tub": (S, STEPS, D) the window's layer-normed tokens averaged per time step,
        "pred": (S, horizon, D) the predicted ones of the next `horizon` steps}, S = T // 2. Step s
        reads frames [2s - 31, 2s] only."""
        F = self.torch.nn.functional
        steps = np.arange(len(video) // TUBELET)
        offs = np.arange(-WINDOW + 1, 1)
        raw, tub, pred = [], [], []
        for i in range(0, len(steps), batch):
            idx = np.clip(TUBELET * steps[i:i + batch, None] + offs, 0, len(video) - 1)
            tok = self._tokens([video[j] for j in idx])
            p = self._predict(tok, horizon)
            B, n, D = tok.shape
            ln = lambda x: F.layer_norm(x.float(), (D,))  # noqa: E731  (V-JEPA 2's target space)
            raw.append(tok.float().mean(1).cpu().numpy())
            tub.append(ln(tok).reshape(B, STEPS, -1, D).mean(2).cpu().numpy())
            pred.append(ln(p).reshape(B, horizon, -1, D).mean(2).cpu().numpy())
        return {"raw": np.concatenate(raw), "tub": np.concatenate(tub).astype(np.float16),
                "pred": np.concatenate(pred).astype(np.float16)}


def trailing(tub):
    """(S, STEPS, D) per-step tokens of each trailing window -> (S, D) its latent."""
    return tub.astype(np.float32).mean(1)


def predicted_trailing(tub, pred, h):
    """The trailing window that would end h steps after step s, as seen from s: its first STEPS - h
    time steps are the last ones the encoder saw, the other h imagined -> (S, D)."""
    tub, pred = tub.astype(np.float32), pred.astype(np.float32)
    return (tub[:, h:].sum(1) + pred[:, :h].sum(1)) / STEPS


def shifted(z, h):
    """z[s + h], held at the end: the true latent h steps after each step."""
    return z[np.minimum(np.arange(len(z)) + h, len(z) - 1)]


def quality(present, pred, actual, mean=None):
    """Imagined vs actual future latents (n, D), against copying the present. mean: the centre for
    cosines (the training latents' mean, so a constant offset doesn't count as agreement)."""
    mean = np.zeros(present.shape[1]) if mean is None else mean

    def cos(a, b):
        return float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)))
    p, a, c = pred - mean, actual - mean, present - mean
    err = lambda x: float(np.mean(np.sum((x - actual) ** 2, 1)))  # noqa: E731
    moved = np.linalg.norm(actual - present, axis=1) > 1e-6
    return {"cos_pred": cos(p, a), "cos_copy": cos(c, a), "mse_pred": err(pred), "mse_copy": err(present),
            "mse_ratio": err(pred) / max(err(present), 1e-12),
            "cos_change": cos((pred - present)[moved], (actual - present)[moved]) if moved.any() else 0.0,
            "n": len(present)}


def lagged(z, lags=LAGS):
    """(S, K) -> (S, K * len(lags)): the latent now and lags steps ago (held at the start). Causal."""
    s = np.arange(len(z))
    return np.concatenate([z[np.maximum(s - g, 0)] for g in lags], 1)


class LatentPredictor:
    """Ridge regression from lagged(z)[s] to z[s + h] - z[s], per horizon: a latent world model
    learned from the training clips' own latents (the fallback to V-JEPA's predictor). The ridge
    strength is relative (lam * the mean input variance * rows), so it doesn't depend on the latent's
    scale; select() picks it by leaving one clip out."""

    def __init__(self, horizons=HORIZONS, lags=LAGS, lam=1.0):
        self.horizons, self.lags, self.lam = tuple(horizons), tuple(lags), lam

    def fit(self, zs):
        X = np.concatenate([lagged(z, self.lags) for z in zs]).astype(np.float64)
        mu = X.mean(0)
        Xc = X - mu
        A = Xc.T @ Xc + self.lam * np.trace(Xc.T @ Xc) / X.shape[1] * np.eye(X.shape[1])
        self.mu, self.W, self.b = mu, {}, {}
        for h in self.horizons:
            Y = np.concatenate([shifted(z, h) - z for z in zs]).astype(np.float64)
            self.W[h] = np.linalg.solve(A, Xc.T @ (Y - Y.mean(0)))
            self.b[h] = Y.mean(0)
        return self

    def predict(self, z, h):
        return (z + (lagged(z, self.lags) - self.mu) @ self.W[h] + self.b[h]).astype(np.float32)

    @classmethod
    def select(cls, groups, grid=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0), **kw):
        """groups: {clip: [its latent sequences]} -> fitted on all, with the lam whose leave-one-clip-out
        error (summed over horizons, relative to copying the present) is lowest."""
        def score(lam):
            r = []
            for g in groups:
                p = cls(lam=lam, **kw).fit([z for o in groups if o != g for z in groups[o]])
                for z in groups[g]:
                    for h in p.horizons:
                        n = max(len(z) - h, 1)
                        e = np.sum((p.predict(z, h)[:n] - shifted(z, h)[:n]) ** 2)
                        r.append(e / max(np.sum((z[:n] - shifted(z, h)[:n]) ** 2), 1e-12))
            return float(np.mean(r))
        scores = {lam: score(lam) for lam in grid} if len(groups) > 1 else {}
        best = min(scores, key=scores.get) if scores else 10.0  # one clip: nothing to leave out
        p = cls(lam=best, **kw).fit([z for zs in groups.values() for z in zs])
        p.cv = scores
        return p
