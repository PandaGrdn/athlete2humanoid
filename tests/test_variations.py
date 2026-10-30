"""skill/variations.py (variants of any clip) and the variant training env (mjlab_tasks.py A2H_VARIANTS).

The generator needs MuJoCo and the G1 assets (scripts/fetch_assets.sh); the env test needs mjlab.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from skill import task as T  # noqa: E402

DATA = Path(__file__).parent / "data"
QPOS = DATA / "qpos_front_kick.npy"  # front_kick's human.npy through the 20 rad/s-capped retarget.py, 30 fps


@pytest.fixture(scope="module")
def g1():
    from skill import variations as V
    try:
        return V.G1()
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def kick():
    return np.load(QPOS).astype(np.float64)


def test_mirror_is_exact(g1, kick):
    from latent.reward_net import BODIES
    from skill import variations as V
    p, pm = g1.bodies(kick), g1.bodies(V.mirror(kick, g1))
    swap = lambda n: n.replace("left", "@").replace("right", "left").replace("@", "right")  # noqa: E731
    expect = p[:, [BODIES.index(swap(b)) for b in BODIES]] * np.array([1, -1, 1])
    assert np.abs(pm - expect).max() < 1e-4
    assert np.abs(V.mirror(V.mirror(kick, g1), g1) - kick).max() < 1e-9


def test_tempo_keeps_flights_at_real_time():
    from skill import variations as V
    n = 100
    air = np.zeros(n, bool)
    air[40:60] = True
    src = V.time_map(n, [20, 70], 2.0, 0.0, air, np.random.default_rng(0))
    step = np.diff(src)  # source frames per output frame: 2 on the ground, 1 in the air
    in_air = (src[:-1] >= 40) & (src[1:] <= 59)
    assert np.allclose(step[in_air], 1.0) and np.allclose(step[(src[1:] < 39) | (src[:-1] > 60)], 2.0)
    assert len(src) == pytest.approx(1 + 79 / 2 + 19 + 1, abs=2)
    assert np.all(np.diff(src) > 0) and src[0] == 0 and src[-1] <= n - 1


def test_amplitude_scales_about_the_first_frame_and_keeps_contact(g1, kick):
    from skill import variations as V
    spec_foot0 = float(np.percentile(g1.bodies(kick)[:, [3, 6], 2].min(1), 5))
    same, clipped = V.scale(kick, 1.0, g1, spec_foot0)
    assert np.abs(same - kick).max() < 1e-6 and clipped == 0
    small, _ = V.scale(kick, 0.8, g1, spec_foot0)
    j0 = kick[0, 7:]
    assert np.allclose(small[:, 7:] - j0, 0.8 * (kick[:, 7:] - j0), atol=1e-6)
    grounded = g1.bodies(kick)[:, [3, 6], 2].min(1) - spec_foot0 < T.DOWN
    assert np.abs(g1.soles(small)[grounded] - g1.soles(kick)[grounded]).max() < 1e-6  # the planted sole stays on the floor


G = 9.81


def hop(pos_bodies, n=90, start=30, apex=0.15, slow=1.0, fps=30.0):
    """(n, bodies, 3) body positions: standing still, then one rigid hop whose feet rise `apex` m. slow > 1 stretches
    its airtime without changing its height (not a jump gravity allows)."""
    t = (np.arange(n) - start) / fps / slow
    T0 = 2 * np.sqrt(2 * apex / G)
    z = np.where((t > 0) & (t < T0), G * T0 / 2 * t - G / 2 * t ** 2, 0.0)
    out = np.repeat(pos_bodies[None], n, 0).astype(np.float64)
    out[..., 2] += z[:, None]
    return out


def test_flights_measure_ballistic_ratio_between_frames():
    """A real hop reads 1 (rise above the air threshold vs g T^2 / 8 of the time above it), whatever frame
    the threshold crossings fall between; one with twice its airtime reads 1/4."""
    from latent.reward_net import BODIES
    from skill import variations as V
    body = np.zeros((len(BODIES), 3))
    body[0, 2] = 0.7
    for start in (30, 30.4, 30.7):
        f = V.flights(hop(body, start=start), 0.0, 30.0)
        assert len(f) == 1 and abs(f[0]["ratio"] - 1) < 0.05
        assert f[0]["air_s"] == pytest.approx(2 * np.sqrt(2 * (0.15 - V.FLIGHT_AIR) / G), abs=0.01)
    assert V.flights(hop(body, slow=2.0), 0.0, 30.0)[0]["ratio"] == pytest.approx(0.25, abs=0.03)
    assert V.flights(hop(body, apex=0.09), 0.0, 30.0) == []  # never clears the air threshold


def test_time_map_holds_the_speed_cap():
    """A speed-up plays the stretches already at the cap at 1x; amplitude > 1 may slow them to 1 / alpha; flights
    stay at 1x whatever the cap says."""
    from skill import variations as V
    n, rng = 60, np.random.default_rng(0)
    air = np.zeros(n, bool)
    air[40:50] = True
    js = np.full(n - 1, 10.0)
    js[10:20] = 20.0
    js[30:35] = 22.0
    js[42:46] = 22.0
    src = V.time_map(n, [], 1.5, 0.0, air, rng, js, 20.0, 1 / 1.1)
    step = np.diff(src)
    at = lambda a, b: step[(src[:-1] >= a) & (src[1:] <= b)]  # noqa: E731
    assert np.allclose(at(0, 9), 1.5) and np.allclose(at(10, 19), 1.0)
    assert np.allclose(at(30, 34), 20 / 22) and np.allclose(at(41, 48), 1.0)


@pytest.fixture(scope="module")
def stand(g1):
    return g1.model.key("stand").qpos.astype(np.float64)


def hop_qpos(stand, apex=0.15, slow=1.0, n=90):
    from latent.reward_net import BODIES
    q = np.repeat(stand[None], n, 0)
    q[:, 2] += hop(np.zeros((len(BODIES), 3)), n=n, apex=apex, slow=slow)[:, 0, 2]
    return q


def test_amplitude_retimes_flights_to_their_new_rise(g1, stand):
    """Scaled 1.25x, a ballistic hop rises 1.25x as high in the same airtime unless re-timed; re-timed, it is
    ballistic again and longer."""
    from skill import variations as V
    q = hop_qpos(stand)
    foot0 = float(np.percentile(g1.bodies(q)[:, [3, 6], 2].min(1), 5))
    big, _ = V.scale(q, 1.25, g1, foot0)
    (unretimed,) = V.flights(g1.bodies(big), foot0, 30.0)
    assert unretimed["ratio"] > 1.2
    v, _, src = V.make(q, g1, False, 1.0, 1.25, 0.0, [], foot0, fps=30.0)
    (f,) = V.flights(g1.bodies(v), foot0, 30.0)
    assert abs(f["ratio"] - 1) < 0.1 and f["air_s"] == pytest.approx(unretimed["air_s"] * np.sqrt(unretimed["ratio"]), rel=0.05)
    assert np.all(np.diff(src) > 0) and src[0] == 0 and src[-1] > len(q) - 2


def test_open_flights_need_only_the_lower_bound():
    """A flight the clip ends in has a censored airtime: a ballistic one caught mid-air reads >= 1 (no upper
    bound applies); one that floats (hurdles' tail: the pelvis held up long after take-off) reads far below."""
    from latent.reward_net import BODIES
    from skill import variations as V
    body = np.zeros((len(BODIES), 3))
    body[0, 2] = 0.7
    for n in (38, 40, 43):  # the clip ends on the way up, near the apex, on the way down
        (f,) = V.flights(hop(body, n=n, apex=0.3), 0.0, 30.0)
        assert f["open"] and f["ratio"] >= 0.95 and V.ballistic(f)
    float_ = hop(body, n=60)
    float_[36:, :, 2] = float_[36, :, 2]  # rises, then stays up until the clip ends
    (f,) = V.flights(float_, 0.0, 30.0)
    assert f["open"] and f["ratio"] < 0.3 and not V.ballistic(f)
    (landed,) = V.flights(hop(body), 0.0, 30.0)
    assert not landed["open"]


def test_amplitude_leaves_unballistic_flights_untimed(g1, stand):
    """Re-timing squeezes a flight into the airtime its rise allows; done to one that was never ballistic it
    only speeds its joints up, so it is left as it is (and the filter reports it)."""
    from skill import variations as V
    q = hop_qpos(stand, slow=2.0)
    foot0 = float(np.percentile(g1.bodies(q)[:, [3, 6], 2].min(1), 5))
    v, _, src = V.make(q, g1, False, 1.0, 1.25, 0.0, [], foot0, fps=30.0)
    assert len(v) == len(q) and np.allclose(src, np.arange(len(q)))


def test_filter_drops_unphysical_variants(g1, stand):
    """The filter's physical rules, each on the variant that breaks it."""
    from skill import variations as V
    grid = [(False, 1.0, 1.0, 0.0), (False, 1.0, 0.6, 0.0), (False, 1.0, 1.25, 0.0)]
    vs = V.generate(hop_qpos(stand), 30.0, grid, g1=g1, log=lambda s: None)
    assert [v["ok"] for v in vs] == [True, False, True]
    assert vs[1]["rules"] == ["lost flight"]  # 0.6x of a 15 cm hop no longer clears the floor by 10 cm
    floaty = V.generate(hop_qpos(stand, slow=2.0), 30.0, grid[:1], g1=g1, log=lambda s: None)[0]
    assert "ballistic" in floaty["rules"] and "outside [0.6, 1.5]" in floaty["why"]
    fast = stand + np.zeros((60, 1))
    fast[30:, 7 + 15] += 1.0  # one joint jumps 1 rad in one frame: 30 rad/s
    v = V.generate(fast, 30.0, grid[:1], g1=g1, log=lambda s: None)[0]
    assert "speed" in v["rules"]
    sunk = stand + np.zeros((60, 1))
    sunk[20:40, 2] -= 0.05
    v = V.generate(sunk, 30.0, grid[:1], g1=g1, log=lambda s: None)[0]
    assert "penetration" in v["rules"]


def test_slow_tempo_does_not_sink_the_feet(g1, kick):
    """Blending frames linearly would put a moving planted foot through the floor between them."""
    from skill import variations as V
    t = np.arange(0, len(kick) - 1, 0.75)
    out = V.regrounded(kick, t, g1)
    assert np.abs(g1.soles(out) - np.interp(t, np.arange(len(kick)), g1.soles(kick))).max() < 1e-6


def test_variants_pass_their_own_specs(g1, kick):
    from skill import variations as V
    grid = [(False, 1.0, 1.0, 0.0), (True, 1.33, 0.85, 0.25), (False, 0.75, 1.1, 0.0), (False, 1.0, 1.6, 0.0)]
    vs = V.generate(kick, 30.0, grid, g1=g1, log=lambda s: None)
    assert [v["ok"] for v in vs] == [True, True, True, False]  # 1.6x pushes joints past their limits
    assert "limits" in vs[3]["why"]
    assert vs[1]["frames"] < len(kick) < vs[2]["frames"]
    assert vs[0]["spec"]["summary"].startswith("4 key moments")
    assert not set(map(tuple, V.TRAIN)) & set(map(tuple, V.HELD_OUT))  # held out means never trained on


@pytest.mark.skipif(importlib.util.find_spec("mjlab") is None, reason="needs mjlab (its virtualenv)")
def test_variant_env_keeps_each_env_inside_its_variant(tmp_path, monkeypatch):
    """Two variants end to end (the clip and its mirror): every env's reward machine follows the variant
    its reference is in, and no env runs from one variant into the next."""
    from experiments.robustness_eval import mirror_motion, motion_body_pos
    from skill.keypose import compile_keypose_spec
    src = DATA / "motion_squat_jump_head.npz"
    mirror_motion(src, tmp_path / "m.npz")
    parts = [np.load(src), np.load(tmp_path / "m.npz")]
    np.savez(tmp_path / "motion.npz", fps=parts[0]["fps"],
             **{k: np.concatenate([p[k] for p in parts]) for k in parts[0].files if k != "fps"})
    segs, start = [], 0
    for name, f in (("orig", src), ("mirror", tmp_path / "m.npz")):
        n = len(np.load(f)["joint_pos"])
        spec = compile_keypose_spec(motion_body_pos(f), float(np.asarray(np.load(f)["fps"]).reshape(-1)[0]), name=name)
        (tmp_path / f"{name}.json").write_text(json.dumps(spec))
        segs.append({"name": name, "start": start, "length": n, "spec": f"{name}.json"})
        start += n
    (tmp_path / "variants.json").write_text(json.dumps({"segments": segs}))
    monkeypatch.setenv("A2H_VARIANTS", str(tmp_path / "variants.json"))
    monkeypatch.setenv("A2H_TASK_SPEC", str(tmp_path / "orig.json"))
    monkeypatch.delenv("A2H_LATENT_BUNDLE", raising=False)
    monkeypatch.delenv("A2H_IMPORTANCE", raising=False)
    import mjlab_tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    cfg = mjlab_tasks.use_variants(load_env_cfg("A2H-TaskHybrid-Flat-Unitree-G1"))
    cfg.commands["motion"].motion_file = str(tmp_path / "motion.npz")
    cfg.commands["motion"].debug_vis = False
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    cmd, task = env.command_manager.get_term("motion"), env.reward_manager.get_term_cfg("task").func
    assert isinstance(cmd, mjlab_tasks.VariantMotionCommand) and task.m.V == 2
    env.reset()
    resampled, real = [], cmd._resample_command
    def record(ids):
        resampled.append(torch.as_tensor(ids).clone())
        return real(ids)
    cmd._resample_command = record
    zero = torch.zeros(8, env.action_manager.total_action_dim)
    seen, last, ends = set(), cmd.time_steps.clone(), 0
    for _ in range(200):
        resampled.clear()
        env.step(zero)
        ts = cmd.time_steps
        at_end = cmd.seg_last[last].nonzero().squeeze(1).tolist()
        done = set(torch.cat(resampled).tolist()) if resampled else set()
        assert set(at_end) <= done, "an env at the end of its variant ran on into the next"
        ends += len(at_end)
        # rewards are computed before the command advances: this step was scored at time step `last`
        assert torch.equal(task.m.var, cmd.seg_of[last]), "reward machine scoring the wrong variant"
        seen |= set(cmd.seg_of[ts].tolist())
        last = ts.clone()
    assert seen == {0, 1} and ends > 0
    env.close()
