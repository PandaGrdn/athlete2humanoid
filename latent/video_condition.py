"""The video's V-JEPA latent as a policy input: what the video shows now and next, in V-JEPA 2's
latent space, alongside the robot's own state and the retarget (mjlab_tasks.py video_latent).

The latent is the athlete's silhouette video embedded densely (latent/vjepa.py embed_dense: one
1024-d vector per 2 frames, each with ~1 s of context), reduced by a PCA fitted on the training
clips only, then laid out per 50 Hz motion step so it lines up with the motion file:

- a variant re-timed by skill/variations.py reads the video at its source frame (src_frame.npy),
  so a 2x variant sees the video played at 2x;
- a mirrored variant reads the latent of the mirrored (horizontally flipped) silhouette video;
- a scaled variant reads the original's latent: the video doesn't change, only the retarget does.

A new clip gets its latent the same way, from its own video, through the same PCA: the policy never
sees anything at test time that it couldn't be given for any video.
"""

import numpy as np

K = 32                    # PCA components kept
OFFSETS_S = (0.0, 0.5, 1.0)  # the policy sees the latent now and this far ahead (each ~1 s of context)
STEP_HZ = 50.0


def dense_latents(silhouette_mp4, vj):
    """A silhouette video -> (dense latent, dense latent of its mirror image), (T // 2, 1024) each."""
    import mediapy
    sil = np.asarray(mediapy.read_video(str(silhouette_mp4)))
    return vj.embed_dense(sil), vj.embed_dense(np.ascontiguousarray(sil[:, :, ::-1]))


def fit_pca(denses, k=K):
    """{"mean": (D,), "components": (k, D), "explained": float} from a list of (n, D) latents."""
    x = np.concatenate(denses).astype(np.float64)
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    var = s ** 2
    return {"mean": mean.astype(np.float32), "components": vt[:k].astype(np.float32),
            "explained": float(var[:k].sum() / var.sum())}


def project(dense, pca):
    return ((dense - pca["mean"]) @ pca["components"].T).astype(np.float32)


def at_frames(z, frames):
    """Dense latent (n, k) at fractional video frames: step s covers frames 2s and 2s + 1, so frame f
    sits at step (f - 0.5) / 2; linear in between, held at the ends."""
    pos = np.clip((np.asarray(frames, np.float64) - 0.5) / 2, 0, len(z) - 1)
    i0 = np.minimum(np.floor(pos).astype(int), max(len(z) - 2, 0))
    i1 = np.minimum(i0 + 1, len(z) - 1)
    w = (pos - i0)[:, None]
    return (z[i0] * (1 - w) + z[i1] * w).astype(np.float32)


def per_step(z, src_frame, fps, n_steps):
    """The latent for each of a motion file's n_steps (50 Hz) steps. src_frame: for each frame of the
    motion at `fps`, the video frame it shows (identity for the clip itself)."""
    t = np.arange(n_steps) / STEP_HZ * fps  # this motion's own frame at each step
    src = np.interp(t, np.arange(len(src_frame)), src_frame)
    return at_frames(z, src)
