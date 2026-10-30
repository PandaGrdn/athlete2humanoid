"""Motions that travel: travel_tasks.py's terms and start sampling, skill/travel_variants.py's hold
sync, and experiments/travel_policy.py's bookkeeping (docs/FRONTIER_TRAVEL.md).

The variant tests need MuJoCo and the G1 assets; the env test needs mjlab (its virtualenv).
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

DATA = Path(__file__).parent / "data"
HAS_MJLAB = importlib.util.find_spec("mjlab") is not None
mjlab_only = pytest.mark.skipif(not HAS_MJLAB, reason="needs mjlab (its virtualenv)")


@pytest.fixture(scope="module")
def tt():
    if not HAS_MJLAB:
        pytest.skip("needs mjlab")
    import travel_tasks
    return travel_tasks


def fake_env(ref_xy, robot_xy, ref_vel=None, robot_vel=None):
    n = len(ref_xy)
    z = torch.full((n, 1), 0.8)
    cmd = SimpleNamespace(anchor_pos_w=torch.cat([torch.as_tensor(ref_xy, dtype=torch.float32), z], -1),
                          robot_anchor_pos_w=torch.cat([torch.as_tensor(robot_xy, dtype=torch.float32), z], -1),
                          anchor_lin_vel_w=torch.zeros(n, 3) if ref_vel is None else torch.as_tensor(ref_vel),
                          robot_anchor_lin_vel_w=torch.zeros(n, 3) if robot_vel is None else torch.as_tensor(robot_vel))
    return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: cmd))


@mjlab_only
def test_root_terms_reach_metre_errors(tt):
    env = fake_env([[0, 0], [0, 0], [0, 0]], [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    coarse = tt.anchor_position_error_exp(env, std=1.0)
    fine = tt.anchor_position_error_exp(env, std=0.3)  # mjlab's: ~nothing left to gain at 1 m
    assert torch.allclose(coarse, torch.exp(-torch.tensor([0.0, 0.25, 1.0])))
    assert fine[2] < 1e-4 < coarse[2] - 0.3
    v = tt.anchor_lin_vel_xy_error_exp(fake_env([[0, 0]] * 2, [[0, 0]] * 2, ref_vel=[[1.0, 0, 0], [1.0, 0, 5.0]],
                                                robot_vel=[[0.0, 0, 0], [1.0, 0, 0]]), std=0.7)
    assert v[0] == pytest.approx(np.exp(-1 / 0.49), rel=1e-5) and v[1] == 1.0  # vertical velocity isn't travel


@mjlab_only
def test_xy_termination_and_its_curriculum(tt):
    env = fake_env([[0, 0], [0, 0]], [[0.3, 0.0], [0.0, 0.6]])
    assert tt.bad_anchor_xy(env, threshold=0.5).tolist() == [False, True]
    term = SimpleNamespace(params={"threshold": 1.0})
    cfg = SimpleNamespace(params={"term_name": "anchor_xy"})
    menv = SimpleNamespace(termination_manager=SimpleNamespace(get_term_cfg=lambda n: term), common_step_counter=0)
    cur = tt.TerminationCurriculum(cfg, menv)
    for step, want in ((0, 1.0), (500, 0.7), (1000, 0.4), (5000, 0.4)):
        menv.common_step_counter = step
        cur(menv, None, "anchor_xy", 1.0, 0.4, 1000)
        assert term.params["threshold"] == pytest.approx(want)


@mjlab_only
def test_travel_weights_ignore_sway(tt):
    t = torch.arange(200, dtype=torch.float32)
    sway = torch.stack([0.06 * torch.sin(t / 3), 0.06 * torch.cos(t / 5)], -1)  # in place: fast, but goes nowhere
    assert tt.travel_weights(sway, 50, 0.2).sum() == 0
    glide = sway.clone()
    glide[80:120, 1] += torch.linspace(0, 1.0, 40)
    glide[120:, 1] += 1.0
    w = tt.travel_weights(glide, 50, 0.2)
    assert w[:30].sum() == 0 and w[130:].sum() == 0 and w[75:110].min() > 0
    seg_end = torch.where(t < 100, 99, 199).long()  # a variant ends at step 99: windows don't cross it
    assert tt.travel_weights(glide, 50, 0.2, seg_end)[95:100].max() < w[95:100].max()


@mjlab_only
def test_travel_sampling_mixes_starts(tt):
    T = 300
    body = torch.zeros(T, 2, 3)
    body[150:200, 1, 0] = torch.linspace(0, 1.5, 50)  # the anchor (index 1) glides 1.5 m mid-clip
    body[200:, 1, 0] = 1.5
    calls = []

    def adaptive(env_ids):
        calls.append(len(env_ids))
        cmd.time_steps[env_ids] = 0
    cmd = SimpleNamespace(motion=SimpleNamespace(body_pos_w=body), motion_anchor_body_index=1, device="cpu",
                          time_steps=torch.zeros(4000, dtype=torch.long), _adaptive_sampling=adaptive)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda n: cmd), step_dt=0.02)
    s = tt.TravelSampling(SimpleNamespace(params={"mix": 0.5, "window_s": 1.0, "min_m": 0.2}), env)
    torch.manual_seed(0)
    cmd._adaptive_sampling(torch.arange(4000))
    assert calls == [4000]  # mjlab's sampling still ran for everyone
    moved = cmd.time_steps > 0
    assert 0.45 < moved.float().mean() < 0.55
    assert ((cmd.time_steps[moved] >= 100) & (cmd.time_steps[moved] < 200)).all()
    assert 0 < s.share < 0.5
    flat = SimpleNamespace(motion=SimpleNamespace(body_pos_w=torch.zeros(T, 2, 3)), motion_anchor_body_index=1,
                           device="cpu", time_steps=torch.zeros(4, dtype=torch.long), _adaptive_sampling=adaptive)
    tt.TravelSampling(SimpleNamespace(params={"mix": 0.5, "window_s": 1.0, "min_m": 0.2}),
                      SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda n: flat), step_dt=0.02))
    assert flat._adaptive_sampling is adaptive  # nothing travels: untouched


@mjlab_only
def test_registered_tasks_change_only_what_they_say(tt):
    import mjlab_tasks  # noqa: F401  (registers them)
    from mjlab.tasks.registry import load_env_cfg
    base = load_env_cfg("A2H-Tracking-Flat-Unitree-G1")
    for task, changes in tt.TASKS.items():
        cfg = load_env_cfg(task)
        extra_r = set(cfg.rewards) - set(base.rewards)
        extra_t = set(cfg.terminations) - set(base.terminations)
        extra_c = set(cfg.curriculum or {}) - set(base.curriculum or {})
        assert extra_r == ({"travel_root_pos", "travel_root_vel"} if "reward" in changes else set()), task
        assert extra_t == ({"anchor_xy"} if "term" in changes else set()), task
        assert extra_c == ({"anchor_xy"} if "term" in changes else set()) | ({"travel_sampling"} if "phase" in changes else set())
        for k, r in base.rewards.items():  # mjlab's own terms as they were
            assert (cfg.rewards[k].weight, cfg.rewards[k].params) == (r.weight, r.params), (task, k)
        assert list(cfg.observations["actor"].terms) == list(base.observations["actor"].terms)
        play, base_play = load_env_cfg(task, play=True), load_env_cfg("A2H-Tracking-Flat-Unitree-G1", play=True)
        assert set(play.rewards) == set(base_play.rewards) and set(play.terminations) == set(base_play.terminations)


@pytest.fixture(scope="module")
def g1():
    pytest.importorskip("mujoco")
    from skill import variations as V
    try:
        return V.G1()
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_synced_holds_keep_a_travelling_variant(g1):
    from skill import travel_variants as TV
    from skill import variations as V
    lunge = np.load(DATA / "qpos_lunge.npy").astype(np.float64)
    grid = [(False, 1.33, 1.0, 0.0)]  # the lunge 1.33x as fast: its reference fails its own stock spec
    stock = V.generate(lunge, 30.0, grid, 0, g1, log=lambda *a: None)[0]
    assert not stock["ok"] and "its own reference completes" in stock["why"]
    synced = TV.generate(lunge, 30.0, grid, 0, g1, log=lambda *a: None)[0]
    assert synced["ok"] and synced["synced"] and synced["spec"]["self_check"]["success"]
    a, b = stock["spec"]["stages"], synced["spec"]["stages"]
    assert [s.get("goal") for s in a] == [s.get("goal") for s in b]  # only holds change
    assert any(x.get("hold_s") != y.get("hold_s") for x, y in zip(a, b))


def test_synced_holds_leave_kept_variants_alone(g1):
    from skill import travel_variants as TV
    from skill import variations as V
    kick = np.load(DATA / "qpos_front_kick.npy").astype(np.float64)
    grid = V.TRAIN[:2]
    for s, t in zip(V.generate(kick, 30.0, grid, 0, g1, log=lambda *a: None),
                    TV.generate(kick, 30.0, grid, 0, g1, log=lambda *a: None)):
        assert s["ok"] == t["ok"] and s["spec"] == t["spec"] and not t.get("synced")


def test_policy_specs_and_travel_metrics():
    from experiments import travel_policy as TP
    assert TP.parse_policies(["base=A2H-Tracking-Flat-Unitree-G1", "all=A2H-Travel-Flat-Unitree-G1@synced"]) == {
        "base": {"task": "A2H-Tracking-Flat-Unitree-G1", "variants": "stock"},
        "all": {"task": "A2H-Travel-Flat-Unitree-G1", "variants": "synced"}}
    with pytest.raises(SystemExit):
        TP.parse_policies(["x=A2H-Tracking-Flat-Unitree-G1@loose"])
    ref = np.stack([np.linspace(0, 1, 11), np.zeros(11)], -1)
    q = lambda xy: np.concatenate([np.asarray(xy), np.zeros((len(xy), 34))], -1)  # noqa: E731
    roll = {"0/stay/0": q(np.zeros((11, 2))), "0/stay/0/start": np.array(0),
            "0/half/0": q(ref * 0.5), "0/half/0/start": np.array(0),
            "0/half/1": q(ref * 0.5 + [0, 0.1]), "0/half/1/start": np.array(5),  # mid-clip start: not counted
            "3/half/0": q(ref), "3/half/0/start": np.array(0)}  # another condition
    m = TP.travel_metrics(_Npz(roll), ref)
    assert m["stay"] == {"travel_frac": 0.0, "end_err_m": 1.0, "rollouts": 1}
    assert m["half"]["travel_frac"] == pytest.approx(0.5) and m["half"]["rollouts"] == 1


class _Npz(dict):
    @property
    def files(self):
        return list(self)
