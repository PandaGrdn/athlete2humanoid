"""latent/video_condition.py: the video's latent laid out per motion step, for A2H-TrackingVJ."""

import numpy as np

from latent import video_condition as VC


def test_pca_keeps_k_components_fitted_on_what_it_is_given():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 16)) @ rng.standard_normal((16, 64))
    pca = VC.fit_pca([x[:100], x[100:]], k=8)
    assert pca["components"].shape == (8, 64) and 0 < pca["explained"] <= 1
    z = VC.project(x, pca)
    assert z.shape == (200, 8) and abs(z.mean()) < 1e-3


def test_at_frames_reads_the_step_that_covers_a_frame():
    z = np.arange(10, dtype=np.float32)[:, None]  # step s covers frames 2s, 2s + 1
    assert VC.at_frames(z, [0.5, 2.5, 18.5])[:, 0].tolist() == [0, 1, 9]
    assert VC.at_frames(z, [-5, 100])[:, 0].tolist() == [0, 9]  # held at the ends


def test_a_retimed_variant_reads_the_video_at_its_source_frames():
    z = np.arange(50, dtype=np.float32)[:, None]  # a 100-frame video at 30 fps
    fps = 30.0
    same = VC.per_step(z, np.arange(100, dtype=float), fps, 166)  # the clip itself, at 50 Hz
    fast = VC.per_step(z, np.arange(0, 100, 2, dtype=float), fps, 83)  # 2x: every other source frame
    assert np.allclose(fast[:80, 0], same[::2, 0][:80], atol=0.05)  # (the last frames are held, differently)
    assert same[-1, 0] > 48 and fast[-1, 0] > 47  # both reach the end of the video


def test_the_causal_latent_never_reads_a_step_whose_window_ends_after_the_frame():
    z = np.arange(20, dtype=np.float32)[:, None]  # step s: the window of frames ending at frame 2s
    frames = np.linspace(0, 39, 500)
    s = VC.at_frames_causal(z, frames)[:, 0]
    assert np.all(2 * s <= frames + 1e-9)
    assert np.all(2 * (s + 1) > frames)  # ... and is the latest such step


def test_a_causal_variant_reads_only_video_it_has_already_shown():
    z = np.arange(50, dtype=np.float32)[:, None]
    src = np.linspace(0, 99, 60) ** 1.2 / 99 ** 0.2  # a non-uniform re-timing of a 100-frame video
    lat = VC.per_step(z, src, 30.0, 100, causal=True)[:, 0]
    shown = np.interp(np.arange(100) / 50 * 30, np.arange(len(src)), src)
    assert np.all(2 * lat <= shown + 1e-6)


def test_trailing_windows_end_at_their_step():
    class Fake(VJ := __import__("latent.vjepa", fromlist=["VJepa"]).VJepa):
        def __init__(self):
            pass

        def embed_clips(self, clips):  # each window -> its last frame's value
            return np.array([[c[-1, 0, 0, 0]] for c in clips], np.float32)
    video = np.arange(64, dtype=np.uint8)[:, None, None, None].repeat(3, -1)
    last = Fake().embed_trailing(video)[:, 0]
    assert last.tolist() == [2 * s for s in range(32)]  # step s's window ends at frame 2s, no later


def test_looking_back_stays_inside_the_variant():
    import torch
    from types import SimpleNamespace

    import mjlab_tasks
    seg_of = torch.tensor([0] * 10 + [1] * 10)
    cmd = SimpleNamespace(motion=SimpleNamespace(time_step_total=20), time_steps=torch.tensor([0, 5, 10, 12, 19]),
                          seg_end=torch.tensor([9] * 10 + [19] * 10), seg_start=torch.tensor([0, 10]), seg_of=seg_of)
    assert mjlab_tasks._ahead(cmd, -3).tolist() == [0, 2, 10, 10, 16]  # held at its variant's first step
    assert mjlab_tasks._ahead(cmd, 3).tolist() == [3, 8, 13, 15, 19]
    assert mjlab_tasks._ahead(cmd, 0).tolist() == [0, 5, 10, 12, 19]
