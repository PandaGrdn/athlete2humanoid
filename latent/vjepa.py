"""V-JEPA 2 (Meta's video world model) as a motion encoder: short windows of a silhouette video
-> one embedding per window."""

import numpy as np

MODEL = "facebook/vjepa2-vitl-fpc64-256"
WINDOW = 32   # frames per window: about 1 s at 30 fps
STRIDE = 8


class VJepa:
    def __init__(self, name=MODEL, device="cuda"):
        import torch
        from transformers import AutoModel, AutoVideoProcessor
        self.torch = torch
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        self.processor = AutoVideoProcessor.from_pretrained(name)

    def embed_clips(self, clips):
        """clips: list of (T, H, W, 3) uint8 -> (N, D) float32, mean-pooled over tokens."""
        torch = self.torch
        videos = [torch.from_numpy(np.ascontiguousarray(c)).permute(0, 3, 1, 2) for c in clips]
        inputs = self.processor(videos, return_tensors="pt").to(self.model.device)
        inputs = {k: v.to(self.model.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_vision_features(**inputs)
        return feats.float().mean(1).cpu().numpy()

    def embed_windows(self, video, window=WINDOW, stride=STRIDE, batch=16):
        """Sliding windows over a (T, H, W, 3) video -> (embeddings (N, D), window start frames)."""
        starts = list(range(0, max(1, len(video) - window + 1), stride))
        out = []
        for i in range(0, len(starts), batch):
            out.append(self.embed_clips([video[s:s + window] for s in starts[i:i + batch]]))
        return np.concatenate(out), np.array(starts)
