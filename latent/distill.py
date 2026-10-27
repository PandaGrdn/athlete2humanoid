"""Distil V-JEPA into the latent reward network (latent/reward_net.py).

Dataset: many robot motions (the IK bank, every clip's retarget, and randomised variations of
them: speed changes, mixed upper/lower bodies, joint noise, leaning, falling, freezing), each
rendered as a silhouette video from the target video's viewpoint and embedded with V-JEPA 2.
Pairs: ~1 s of the robot's body positions -> the V-JEPA latent at that moment.

The variations matter: during RL the robot is mostly *not* doing the motion well, and the
network has to say so instead of guessing.
"""

import json
import time
from multiprocessing import get_context

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from latent import silhouette
from latent.reward_net import BODIES, WINDOW, LatentRewardNet

FPS = 30
LEGS = slice(7, 19)    # qpos columns: base xyz (0:3), base quat wxyz (3:7), 12 leg joints,
UPPER = slice(19, 36)  # then waist (3) + arms (14)


# ---------------------------------------------------------------- motion variations

def _quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _tilt(roll, pitch):
    """World-frame roll/pitch rotations (T,) -> quats (T, 4), wxyz."""
    qr = np.stack([np.cos(roll / 2), np.sin(roll / 2), 0 * roll, 0 * roll], -1)
    qp = np.stack([np.cos(pitch / 2), 0 * pitch, np.sin(pitch / 2), 0 * pitch], -1)
    return _quat_mul(qp, qr)


def resample(q, n):
    """Time-resample a qpos sequence to n frames (linear; quaternions renormalised)."""
    t = np.linspace(0, len(q) - 1, n)
    out = np.stack([np.interp(t, np.arange(len(q)), q[:, j]) for j in range(q.shape[1])], 1)
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def _smooth_noise(rng, shape, sigma, amp):
    n = gaussian_filter1d(rng.standard_normal(shape), sigma, axis=0)
    return n / (n.std(0, keepdims=True) + 1e-9) * amp


def variation(motions, rng, seconds=4.0):
    """One randomised robot motion built from the base motions (name -> qpos at 30 fps)."""
    names = list(motions)
    n = int(seconds * FPS)
    base = motions[names[rng.integers(len(names))]]
    q = resample(base, max(WINDOW, int(len(base) / rng.uniform(0.6, 1.6)))) if rng.random() < 0.5 else base
    if len(q) > n:
        s = rng.integers(len(q) - n + 1)
        q = q[s:s + n]
    q = q.copy()
    T = len(q)
    if rng.random() < 0.4:  # another motion's upper body
        other = motions[names[rng.integers(len(names))]]
        q[:, UPPER] = resample(other, T)[:, UPPER]
    if rng.random() < 0.6:  # imperfect joints
        q[:, 7:] += _smooth_noise(rng, (T, 29), rng.uniform(4, 12), rng.uniform(0.05, 0.4))
    if rng.random() < 0.4:  # leaning, bobbing
        rp = _smooth_noise(rng, (T, 2), 10, rng.uniform(0.05, 0.35))
        q[:, 3:7] = _quat_mul(_tilt(rp[:, 0], rp[:, 1]), q[:, 3:7])
        q[:, 2] += _smooth_noise(rng, (T,), 10, rng.uniform(0.0, 0.08))
    if rng.random() < 0.15:  # falling over
        t0 = rng.integers(T // 2)
        a = np.clip((np.arange(T) - t0) / (0.8 * FPS), 0, 1) ** 2
        q[:, 3:7] = _quat_mul(_tilt(0 * a, a * rng.choice([-1.3, 1.3])), q[:, 3:7])
        q[:, 2] = (1 - a) * q[:, 2] + a * 0.25
    if rng.random() < 0.1:  # frozen
        q[:] = q[rng.integers(T)]
    return q


# ---------------------------------------------------------------- rendering + embedding

_G1 = None


def _init_worker():
    global _G1
    import render
    _G1 = render.load_g1()[0]


def body_positions(model, qpos):
    """(T, 36) qpos -> (T, len(BODIES), 3) world body positions (MuJoCo xpos, like mjlab's)."""
    data = mujoco.MjData(model)
    ids = [model.body(b).id for b in BODIES]
    out = np.empty((len(qpos), len(ids), 3), np.float32)
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        out[t] = data.xpos[ids]
    return out


def _render(job):
    qpos, azimuth = job
    sil = silhouette.robot_silhouettes(_G1, qpos, azimuth=azimuth)[..., 0]  # one channel over the pipe
    return sil, body_positions(_G1, qpos)


def windows(pos, n_steps):
    """Body positions (T, B, 3) -> the 32-frame window around each V-JEPA dense step (n, 32, B, 3):
    step s covers frames 2s, 2s + 1 and its window is frames [2s - 15, 2s + 16], edge-padded."""
    idx = 2 * np.arange(n_steps)[:, None] + np.arange(-(WINDOW // 2) + 1, WINDOW // 2 + 1)
    return pos[np.clip(idx, 0, len(pos) - 1)]


def build_dataset(motions, azimuth, vj, workers=8):
    """name -> qpos (30 fps) -> {X: (N, 32, B, 3) windows, Y: (N, 1024) V-JEPA latents,
    seq: (N,) motion index, names}. Rendering runs in parallel worker processes."""
    names = list(motions)
    X, Y, seq = [], [], []
    t0 = time.time()
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        jobs = ((motions[k], azimuth) for k in names)
        for i, (sil, pos) in enumerate(pool.imap(_render, jobs, chunksize=1)):
            dense = vj.embed_dense(np.repeat(sil[..., None], 3, -1))
            X.append(windows(pos, len(dense)))
            Y.append(dense.astype(np.float16))
            seq.append(np.full(len(dense), i))
            if (i + 1) % 25 == 0 or i + 1 == len(names):
                print(f"  {i + 1}/{len(names)} motions rendered + embedded ({time.time() - t0:.0f} s)", flush=True)
    return {"X": np.concatenate(X), "Y": np.concatenate(Y), "seq": np.concatenate(seq), "names": np.array(names)}


# ---------------------------------------------------------------- training

def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def train_reward_net(X, Y, seq, r_mean, k=64, epochs=80, val_frac=0.1, seed=0, device=None):
    """Fit the network on (window -> centred latent projected on its top-k principal components).
    Validation holds out whole motions. Returns (net, basis (1024, k), report)."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    motions = np.unique(seq)
    val_m = rng.choice(motions, max(1, int(len(motions) * val_frac)), replace=False)
    val = np.isin(seq, val_m)

    Yc = Y.astype(np.float32) - r_mean
    _, S, Vt = np.linalg.svd(Yc[~val] - 0.0, full_matrices=False)
    U = Vt[:k].T
    explained = float((S[:k] ** 2).sum() / (S ** 2).sum())
    Z = Yc @ U

    net = LatentRewardNet(k=k).to(device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=device)
    tr, va = torch.as_tensor(~val, device=device), torch.as_tensor(val, device=device)
    feats = LatentRewardNet.featurize(Xt[tr])
    net.in_mean.copy_(feats.mean(0))
    net.in_std.copy_(feats.std(0) + 1e-3)
    z_scale = Zt[tr].pow(2).mean()

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx_tr = torch.nonzero(tr).squeeze(1)

    def loss_fn(pred, z):
        cos = torch.nn.functional.cosine_similarity(pred, z, dim=-1)
        return (1 - cos).mean() + 0.1 * (pred - z).pow(2).mean() / z_scale, cos

    for ep in range(epochs):
        net.train()
        for b in idx_tr[torch.randperm(len(idx_tr), device=device)].split(512):
            x = Xt[b] + 0.01 * torch.randn_like(Xt[b])  # ~1 cm jitter
            loss, _ = loss_fn(net(x), Zt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            net.eval()
            with torch.no_grad():
                _, cos_v = loss_fn(net(Xt[va]), Zt[va])
                _, cos_t = loss_fn(net(Xt[tr][:20000]), Zt[tr][:20000])
            print(f"  epoch {ep + 1}: train cos {cos_t.mean():.3f}, held-out motions cos {cos_v.mean():.3f}", flush=True)
    net.eval()
    with torch.no_grad():
        _, cos_v = loss_fn(net(Xt[va]), Zt[va])
    report = {"samples": int(len(X)), "motions": int(len(motions)), "val_motions": int(len(val_m)),
              "pca_k": k, "pca_explained_variance": explained, "val_cosine_mean": float(cos_v.mean()),
              "val_cosine_p10": float(cos_v.quantile(0.1))}
    return net.cpu(), U.astype(np.float32), report


def predict(net, pos):
    """Body positions (T, B, 3) at 30 fps -> predicted latents (T // 2, k)."""
    import torch
    with torch.no_grad():
        return net(torch.as_tensor(windows(pos, len(pos) // 2), dtype=torch.float32)).numpy()


def video_target(human_dense, h_mean, U):
    """The video's latent in the reward network's space: centred on the human-domain mean."""
    return unit(human_dense - h_mean) @ U


def save_bundle(path, net, U, r_mean, h_mean, target, cos_ref, video_fps, report, azimuth):
    """Everything the RL reward needs: the network, the video's latent target in its space, and
    cos_ref, how well the retarget itself matches the video at each step (the reward's 'full
    marks' level)."""
    import torch
    torch.save({"state_dict": net.state_dict(), "net_kwargs": {"k": U.shape[1]}, "target": target.astype(np.float32),
                "cos_ref": cos_ref.astype(np.float32), "basis": U, "r_mean": r_mean, "h_mean": h_mean,
                "video_fps": float(video_fps), "azimuth": float(azimuth), "report": report}, path)
    print(f"reward bundle -> {path}: {len(target)} target steps, {json.dumps(report)}")
