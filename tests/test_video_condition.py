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
