"""The latent reward model: robot body trajectory -> predicted V-JEPA latent.

V-JEPA can't run inside RL (thousands of robots, 50 steps/s), so a small network is distilled to
predict, from ~1 s of the robot's body positions, the latent V-JEPA would give a silhouette video
of that motion. The reward is the cosine similarity between this prediction and the video's own
latent at the same moment. Depends only on torch, so mjlab's environment can load it.
"""

import torch
from torch import nn

# The bodies mjlab's G1 tracking task tracks, in its order (MuJoCo Menagerie uses the same names).
BODIES = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
          "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
          "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
          "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")
WINDOW = 32  # frames at 30 fps, matching the V-JEPA windows the latents come from
FPS = 30.0


def window_center_step(end_frame):
    """V-JEPA dense step whose context window is frames [end - 31, end]. Step s covers frames
    2s and 2s + 1, and its window is centred on it: frames [2s - 15, 2s + 16]."""
    return (end_frame - (WINDOW // 2)) / 2


class LatentRewardNet(nn.Module):
    def __init__(self, k=64, hidden=512, n_bodies=len(BODIES), window=WINDOW):
        super().__init__()
        d = window * n_bodies * 3
        self.register_buffer("in_mean", torch.zeros(d))
        self.register_buffer("in_std", torch.ones(d))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ELU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, k))

    @staticmethod
    def featurize(win):
        """(B, WINDOW, bodies, 3) world positions -> (B, D). Horizontal position relative to the
        pelvis's mean over the window (the silhouette camera follows the body sideways); height
        and the world-frame heading kept (the camera has a fixed height and azimuth)."""
        centre = win[:, :, :1, :2].mean(1, keepdim=True)
        return torch.cat([win[..., :2] - centre, win[..., 2:]], -1).flatten(1)

    def forward(self, win):
        return self.net((self.featurize(win) - self.in_mean) / self.in_std)


def load_bundle(path, device="cpu"):
    """A reward bundle (latent.distill.save_bundle): the network plus the video's latent target."""
    b = torch.load(path, map_location=device, weights_only=False)
    net = LatentRewardNet(**b["net_kwargs"]).to(device)
    net.load_state_dict(b["state_dict"])
    net.eval()
    b["cos_ref"] = torch.as_tensor(b["cos_ref"], dtype=torch.float32, device=device)
    return net, torch.as_tensor(b["target"], dtype=torch.float32, device=device), b
