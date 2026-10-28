"""skill/variations.py (variants of any clip) and the variant training env (mjlab_tasks.py A2H_VARIANTS).

The generator needs MuJoCo and the G1 assets (scripts/fetch_assets.sh); the env test needs mjlab.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from skill import task as T  # noqa: E402

DATA = Path(__file__).parent / "data"
QPOS = DATA / "qpos_front_kick.npy"  # outputs/remote_latent_final/clips/front_kick/qpos.npy, 30 fps


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
    lo = lambda q: g1.bodies(q)[:, [3, 6], 2].min(1)  # noqa: E731
    grounded = lo(kick) - spec_foot0 < T.DOWN
    assert np.abs(lo(small)[grounded] - lo(kick)[grounded]).max() < 1e-3  # the planted foot stays on the floor


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
