"""V-JEPA 2 (Meta's video world model) as a motion encoder for silhouette videos.

embed_windows: one embedding per ~1 s window. embed_dense: one embedding per 2 frames (15/s at
30 fps), each from V-JEPA's own time step within a window, so it still carries ~1 s of context.
embed_centred: one mean-pooled embedding per 2 frames, from a window centred on that moment, so
it depends only on the 32 frames around it (what the distilled reward network sees).
"""

import numpy as np

MODEL = "facebook/vjepa2-vitl-fpc64-256"
WINDOW = 32   # frames per window: about 1 s at 30 fps
STRIDE = 8
TUBELET = 2   # V-JEPA 2 patches 2 frames x 16 x 16 pixels into one token


class VJepa:
    def __init__(self, name=MODEL, device="cuda"):
        import torch
        from transformers import AutoModel, AutoVideoProcessor
        self.torch = torch
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        self.processor = AutoVideoProcessor.from_pretrained(name)

    def embed_clips(self, clips):
        """clips: list of (T, H, W, 3) uint8 -> (N, D) float32, mean-pooled over tokens."""
        return self.embed_clips_temporal(clips).mean(1)

    def embed_clips_temporal(self, clips):
        """clips: list of (T, H, W, 3) uint8 -> (N, T / 2, D): tokens averaged over space only,
        one vector per V-JEPA time step (2 frames)."""
        torch = self.torch
        videos = [torch.from_numpy(np.ascontiguousarray(c)).permute(0, 3, 1, 2) for c in clips]
        # Resize/crop/normalise on the GPU: on the CPU this dominated the time to embed renders.
        inputs = self.processor(videos, return_tensors="pt", device=self.model.device).to(self.model.device)
        inputs = {k: v.to(self.model.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_vision_features(**inputs)  # (N, steps * patches, D), time-major
        steps = len(clips[0]) // TUBELET
        feats = feats.float().reshape(len(clips), steps, -1, feats.shape[-1])
        return feats.mean(2).cpu().numpy()

    def embed_windows(self, video, window=WINDOW, stride=STRIDE, batch=16):
        """Sliding windows over a (T, H, W, 3) video -> (embeddings (N, D), window start frames)."""
        starts = list(range(0, max(1, len(video) - window + 1), stride))
        out = []
        for i in range(0, len(starts), batch):
            out.append(self.embed_clips([video[s:s + window] for s in starts[i:i + batch]]))
        return np.concatenate(out), np.array(starts)

    def embed_dense(self, video, window=WINDOW, batch=16):
        """(T, H, W, 3) video -> (T // 2, D): one embedding per 2 frames. Windows overlap by half;
        each time step takes the window where it sits nearest the centre, so it has context on
        both sides (except at the clip's ends)."""
        steps_per_win, half = window // TUBELET, window // 2
        starts = list(range(0, max(1, len(video) - window + 1), half))
        if starts[-1] + window < len(video):
            starts.append(len(video) - window)
        out = []
        for i in range(0, len(starts), batch):
            out.append(self.embed_clips_temporal([video[s:s + window] for s in starts[i:i + batch]]))
        per_win = np.concatenate(out)  # (n_windows, steps_per_win, D)
        n_steps = len(video) // TUBELET
        dense, best = np.zeros((n_steps, per_win.shape[-1]), np.float32), np.full(n_steps, np.inf)
        for w, s in enumerate(starts):
            for k in range(steps_per_win):
                t = s // TUBELET + k
                dist = abs(k + 0.5 - steps_per_win / 2)
                if t < n_steps and dist < best[t]:
                    dense[t], best[t] = per_win[w, k], dist
        return dense

    def embed_centred(self, video, every=1, window=WINDOW, batch=16):
        """(T, H, W, 3) video -> (embeddings (n, D), steps (n,)) for dense steps 0, every, 2 * every...
        Step s covers frames 2s, 2s + 1; its window is frames [2s - 15, 2s + 16], edge-padded."""
        steps = np.arange(0, len(video) // TUBELET, every)
        offs = np.arange(-(window // 2) + 1, window // 2 + 1)
        out = []
        for i in range(0, len(steps), batch):
            idx = np.clip(TUBELET * steps[i:i + batch, None] + offs, 0, len(video) - 1)
            out.append(self.embed_clips([video[j] for j in idx]))
        return np.concatenate(out), steps
