"""latent/world_model.py and A2H-TrackingVJPred: the imagined future latent never reads the future."""

import numpy as np
import pytest

from latent import world_model as WM

P, D = 4, 8  # patches per time step, token width (a fake model)


class Fake(WM.WorldModel):
    """Encoder: each time step's tokens from its 2 frames' pixels. Predictor: a nonlinear mix of all the
    context tokens, so anything after the context would show up in the prediction."""

    def __init__(self):
        import torch
        self.torch = torch
        g = torch.Generator().manual_seed(0)
        self.Wt = torch.randn(3, P * D, generator=g)
        self.Wp = torch.randn(D, D, generator=g)

    def _tokens(self, clips):
        torch = self.torch
        x = torch.as_tensor(np.stack(clips), dtype=torch.float32)  # (B, 32, H, W, 3)
        px = x.reshape(len(clips), WM.STEPS, -1, 3).mean(2) / 255  # (B, 16, 3): per time step
        return torch.tanh(px @ self.Wt).reshape(len(clips), WM.STEPS * P, D)

    def _predict(self, tokens, horizon, pad=None):
        torch = self.torch
        ctx = torch.tanh(tokens @ self.Wp).cumsum(1)[:, -1:]  # depends on every context token
        steps = torch.arange(1, horizon * P + 1, dtype=torch.float32)[None, :, None]
        return torch.sin(ctx * steps / 10)


def video(T=80, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (T, 6, 6, 3), dtype=np.uint8)


def test_the_imagined_future_at_step_s_reads_only_frames_up_to_2s():
    pytest.importorskip("torch")
    wm, v = Fake(), video()
    base = wm.imagine(v, horizon=15, batch=5)
    assert base["pred"].shape == (40, 15, D) and base["tub"].shape == (40, WM.STEPS, D)
    for s in (0, 7, 20, 38):
        w = v.copy()
        w[2 * s + 1:] = 255 - w[2 * s + 1:]  # change every frame after 2s
        got = wm.imagine(w, horizon=15, batch=5)
        for k in ("raw", "tub", "pred"):
            assert np.array_equal(got[k][:s + 1], base[k][:s + 1]), (k, s)  # steps <= s: unchanged
        assert not np.array_equal(got["pred"][s + 1], base["pred"][s + 1])  # (and the change is visible after)


def test_the_predictor_never_reads_the_target_slots():
    """The real call: V-JEPA 2's predictor gathers only context_mask's tokens, so the zeros standing in
    for the future in encoder_hidden_states could be anything."""
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    cfg = tr.VJEPA2Config(crop_size=32, frames_per_clip=WM.WINDOW * 2, hidden_size=32, num_hidden_layers=1,
                          num_attention_heads=2, pred_hidden_size=16, pred_num_hidden_layers=1,
                          pred_num_attention_heads=2)
    wm = WM.WorldModel.__new__(WM.WorldModel)
    wm.torch, wm.model = torch, tr.VJEPA2Model(cfg).eval()
    tok = torch.randn(2, WM.STEPS * 4, 32)
    a = wm._predict(tok, 15)
    b = wm._predict(tok, 15, pad=torch.randn(2, 15 * 4, 32) * 100)
    assert a.shape == (2, 15 * 4, 32) and torch.allclose(a, b, atol=1e-5)


def test_a_perfect_prediction_gives_the_true_future_trailing_latent():
    rng = np.random.default_rng(1)
    u = rng.standard_normal((60, D)).astype(np.float32)  # each time step's token mean
    s = np.arange(40)
    idx = np.clip(s[:, None] + np.arange(-WM.STEPS + 1, 1), 0, None)
    tub = u[idx]
    pred = u[s[:, None] + np.arange(1, 16)]
    z = WM.trailing(u[np.clip(np.arange(60)[:, None] + np.arange(-WM.STEPS + 1, 1), 0, None)])
    for h in WM.HORIZONS:
        assert np.allclose(WM.predicted_trailing(tub, pred, h), z[s + h], atol=1e-3)


def test_the_learned_predictor_is_causal_and_learns_dynamics():
    t = np.arange(400) * 0.07
    z = lambda ph: np.stack([np.sin(t + ph), np.cos(t + ph), np.sin(2 * t + ph)], 1)  # noqa: E731
    p = WM.LatentPredictor().fit([z(0.0), z(1.0)])
    test = z(2.5)
    for h in WM.HORIZONS:
        q = WM.quality(test[:-h], p.predict(test, h)[:-h], WM.shifted(test, h)[:-h])
        assert q["mse_ratio"] < 0.2 and q["cos_pred"] > q["cos_copy"]
    w = test.copy()
    w[200:] = 0
    assert np.allclose(p.predict(w, 15)[:200], p.predict(test, 15)[:200])


def test_the_policy_input_reads_no_row_after_the_current_step():
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    from types import SimpleNamespace

    import mjlab_tasks
    from latent.video_condition import K
    T = 200
    table = torch.arange(T, dtype=torch.float32)[:, None].repeat(1, 3 * K)  # row t holds t everywhere
    ts = torch.tensor([0, 10, 50, 199])
    cmd = SimpleNamespace(motion=SimpleNamespace(time_step_total=T), time_steps=ts)
    env = SimpleNamespace(a2h_video_latent=table, command_manager=SimpleNamespace(get_term=lambda n: cmd),
                          step_dt=0.02, num_envs=4, device="cpu")
    obs = mjlab_tasks.video_latent_predicted(env)
    assert obs.shape == (4, 4 * K)
    assert torch.all(obs <= ts[:, None])  # nothing from a later step
    assert torch.equal(obs[:, 2 * K:], ts[:, None].float().expand(4, 2 * K))  # the predictions made now
    assert torch.equal(obs[:, K:2 * K], ts[:, None].float().expand(4, K))  # the causal latent now
