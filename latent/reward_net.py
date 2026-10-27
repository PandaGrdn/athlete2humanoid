"""The latent reward model: robot body trajectory -> predicted V-JEPA latent.

V-JEPA can't run inside RL (thousands of robots, 50 steps/s), so a small network is distilled to
predict, from ~1 s of the robot's body positions, the latent V-JEPA would give a silhouette video
of that motion. The reward is the cosine similarity between this prediction and the video's own
latent at the same moment. Depends only on torch, so mjlab's environment can load it.

The network is only right near the motions it was distilled on; RL finds motions where it is
wrong in its favour (v2: the distilled score rose 0.35 -> 0.49 while real V-JEPA stayed ~0.37).
Its uncertainty there is higher, so the reward can subtract it (mean - lambda * excess spread,
Coste et al. 2310.02743): member_predictions() gives an ensemble's members, or MC-dropout samples
of a single network (Gal & Ghahramani 1506.02142), and uncertainty() their relative spread.
"""

from contextlib import contextmanager

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
    # Defaults are the first version's (bundles saved before features/depth existed load as v1).
    def __init__(self, k=64, hidden=512, depth=2, features=1, n_bodies=len(BODIES), window=WINDOW):
        super().__init__()
        self.features = features
        d = window * n_bodies * 3 * (3 if features >= 2 else 1)
        self.register_buffer("in_mean", torch.zeros(d))
        self.register_buffer("in_std", torch.ones(d))
        layers = [nn.Linear(d, hidden), nn.ELU(), nn.Dropout(0.1)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ELU()]
        self.net = nn.Sequential(*layers, nn.Linear(hidden, k))

    def featurize(self, win):
        """(B, WINDOW, bodies, 3) world positions -> (B, D). Horizontal position relative to the
        pelvis's mean over the window (the silhouette camera follows the body sideways); height
        and the world-frame heading kept (the camera has a fixed height and azimuth). v2 adds the
        pose relative to the pelvis at each frame, and velocities."""
        centre = win[:, :, :1, :2].mean(1, keepdim=True)
        f = [torch.cat([win[..., :2] - centre, win[..., 2:]], -1)]
        if self.features >= 2:
            f += [win - win[:, :, :1], torch.diff(win, dim=1, prepend=win[:, :1]) * FPS]
        return torch.cat(f, -1).flatten(1)

    def forward(self, win):
        return self.net((self.featurize(win) - self.in_mean) / self.in_std)


class LatentRewardEnsemble(nn.Module):
    """Independently trained networks (latent.distill.train_reward_net members > 1). Called like
    one network it gives their mean prediction, so everything that uses one network works."""

    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.features = nets[0].features

    def members(self, win):
        return torch.stack([n(win) for n in self.nets])

    def forward(self, win):
        return self.members(win).mean(0)


@contextmanager
def _dropout_on(net):
    drops = [m for m in net.modules() if isinstance(m, nn.Dropout)]
    for m in drops:
        m.train()
    try:
        yield len(drops)
    finally:
        for m in drops:
            m.eval()


def member_predictions(net, win, mc=8):
    """(M, B, k): an ensemble's members' predictions, or `mc` MC-dropout samples of one network
    (its dropout switched on, nothing else), or (1, B, k) when it has neither."""
    if isinstance(net, LatentRewardEnsemble):
        return net.members(win)
    with _dropout_on(net) as n:
        if n and mc > 1:
            return torch.stack([net(win) for _ in range(mc)])
    return net(win)[None]


def uncertainty(preds):
    """(M, B, k) -> (B,): spread of the predictions relative to their size (|std| / |mean|).
    On the v2 squat bundle it tells the hacked policies' windows from in-distribution ones with
    AUROC 0.88 (MC dropout, 16 samples; the spread of the cosine itself: 0.70)."""
    return preds.std(0, unbiased=False).norm(dim=-1) / preds.mean(0).norm(dim=-1).clamp(min=1e-6)


def window_index(ts, dt, L=None):
    """The 32 window frames (30 fps) ending at control step ts (step dt) as fractional indices
    into a 50 Hz sequence: (i0, i1, w), shaped (*ts.shape, WINDOW); before 0 holds the first frame."""
    back = (WINDOW - 1 - torch.arange(WINDOW, device=ts.device)) / FPS / dt
    pos = (ts[..., None].float() - back).clamp(min=0)
    i0 = pos.floor().long()
    i1 = i0 + 1 if L is None else (i0 + 1).clamp(max=L - 1)
    return i0, i1, (pos - i0)


def reference_windows(body_pos, ts, dt):
    """Reference body positions (T, bodies, 3) at step dt -> the windows (len(ts), WINDOW, bodies, 3)
    the reward sees when the robot is on the reference at steps ts."""
    i0, i1, w = window_index(ts, dt, len(body_pos))
    w = w[..., None, None]
    return body_pos[i0] * (1 - w) + body_pos[i1] * w


def load_bundle(path, device="cpu"):
    """A reward bundle (latent.distill.save_bundle): the network (an ensemble when it has
    members) plus the video's latent target."""
    b = torch.load(path, map_location=device, weights_only=False)
    sds = b.get("members") or [b["state_dict"]]
    nets = [LatentRewardNet(**b["net_kwargs"]).to(device) for _ in sds]
    for n, sd in zip(nets, sds):
        n.load_state_dict(sd)
    net = nets[0] if len(nets) == 1 else LatentRewardEnsemble(nets).to(device)
    net.eval()
    b["cos_ref"] = torch.as_tensor(b["cos_ref"], dtype=torch.float32, device=device)
    return net, torch.as_tensor(b["target"], dtype=torch.float32, device=device), b
