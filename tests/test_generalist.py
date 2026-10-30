"""experiments/generalist_policy.py: the zero-shot generalist's motion pool and its trackability filter.

trackable() needs MuJoCo and the G1 assets (scripts/fetch_assets.sh).
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from experiments import generalist_policy as G  # noqa: E402

QPOS = Path(__file__).parent / "data" / "qpos_front_kick.npy"  # a real retarget, 30 fps


@pytest.fixture(scope="module")
def g1():
    from skill import variations as V
    try:
        return V.G1()
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def foot0(g1):
    return G.stand_foot_height(g1)


@pytest.fixture(scope="module")
def kick():
    """Slowed 1.25x: this copy predates retarget.py's 20 rad/s joint-speed cap (its fastest joint: 22)."""
    from skill.variations import resample
    q = np.load(QPOS).astype(np.float64)
    return resample(q, np.linspace(0, len(q) - 1, int(len(q) * 1.25)))


def test_real_clip_is_trackable(g1, foot0, kick):
    q, why = G.trackable(kick, g1, foot0)
    assert why == []
    assert q.shape == kick.shape


def test_frozen_pose_rejected(g1, foot0, kick):
    _, why = G.trackable(np.repeat(kick[:1], 60, 0), g1, foot0)
    assert "a frozen pose" in why


def test_falling_rejected(g1, foot0, kick):
    """latent/distill.py's "falling over": tipped 1.3 rad and down to 0.25 m."""
    q = kick.copy()
    a = np.clip(np.arange(len(q)) / 30 - 1, 0, 1)
    q[:, 3], q[:, 5] = np.cos(1.3 * a / 2), np.sin(1.3 * a / 2)  # pitch about y
    q[:, 4] = q[:, 6] = 0
    q[:, 2] = (1 - a) * q[:, 2] + a * 0.25
    _, why = G.trackable(q, g1, foot0)
    assert any(w.startswith("tilts") for w in why)


def test_float_rejected(g1, foot0, kick):
    """Lifted 15 cm for a second with no rise and fall a flight that long needs."""
    q = kick.copy()
    q[30:60, 2] += 0.15
    _, why = G.trackable(q, g1, foot0)
    assert any(w.startswith("floats") for w in why)


def test_sink_and_slide_rejected(g1, foot0, kick):
    sunk = kick.copy()
    sunk[:, 2] -= 0.1
    assert any("into the floor" in w for w in G.trackable(sunk, g1, foot0)[1])
    slid = kick.copy()
    slid[:, 0] += np.arange(len(slid)) / 30 * 1.0  # the whole body gliding at 1 m/s
    assert any("slides" in w for w in G.trackable(slid, g1, foot0)[1])


def test_limits_clipped(g1, foot0, kick):
    q = kick.copy()
    q[:3, 7] = g1.hi[0] + 0.5  # a few samples past a limit: kept, clipped
    out, why = G.trackable(q, g1, foot0)
    assert not any("limits" in w for w in why)
    assert out[:, 7:].max(0)[0] <= g1.hi[0] + 1e-9


def test_runs():
    assert G._runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert G._runs(np.zeros(3, bool)) == []


def test_levels_nest_and_test_clips_held_out():
    assert set(G.LEVELS["real7"]) < set(G.LEVELS["real7_bank"]) < set(G.LEVELS["all"])
    assert not set(G.REAL_TRAIN) & set(G.ZERO_SHOT)
    assert len(G.REAL_TRAIN) == 7


def test_bank_excludes_zero_shot_twins():
    pytest.importorskip("mink")
    try:
        bank = G.bank_motions()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert not set(G.BANK_EXCLUDE) & set(bank)
    assert "squat_d20_r2" in bank


def test_policies_start_from_5clip_or_scratch():
    for name, (task, start) in G.POLICIES.items():
        assert task.startswith("A2H-Tracking")
        assert start in (None, "multi", "multi_future")
        assert (start is None) == (name.endswith(("_s", "_l")))
