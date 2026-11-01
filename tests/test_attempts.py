"""a2h_env/attempts.py: the attempts a training writes every N iterations, without mjlab (a stand-in env and runner).

    python -m pytest -q tests/test_attempts.py
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch = pytest.importorskip("torch")
from a2h_env import attempts as A  # noqa: E402
from a2h_env import bundle as B  # noqa: E402
from a2h_env import export as X  # noqa: E402

DATA = ROOT / "tests" / "data"
QPOS = DATA / "qpos_front_kick.npy"
SPEC = DATA / "task_spec_front_kick.json"


def test_the_reference_passes_its_own_spec():
    qpos = np.load(QPOS)
    pos = X.body_positions(qpos)
    if pos is None:
        pytest.skip("mujoco or the G1 assets aren't here")
    st = A.evaluate_stages(json.loads(SPEC.read_text()), pos, 30)
    assert st["stages_completed"] == st["stages_total"] and st["success"] and st["fell_at_s"] is None
    assert A.evaluate_stages({"stages": []}, pos, 30) is None


class _Scene(dict):
    env_origins = torch.zeros(1, 3)


class FakeEnv:
    """One env whose robot stands still at the reference's first frame; `dones` after `fall_at` steps."""

    def __init__(self, qpos, fall_at=None):
        self.step_dt = 0.02
        q = torch.tensor(qpos[0], dtype=torch.float32)
        data = SimpleNamespace(root_link_pos_w=q[None, :3], root_link_quat_w=q[None, 3:7], joint_pos=q[None, 7:],
                               body_link_pos_w=torch.zeros(1, 14, 3))
        self.scene = _Scene({"robot": SimpleNamespace(data=data, body_names=list(A.BODIES))})
        self.fall_at, self.t = fall_at, 0

    def step(self, actions):
        self.t += 1
        done = torch.tensor([self.fall_at is not None and self.t >= self.fall_at])
        return {}, torch.tensor([0.5]), done, {}

    def reset(self):
        self.t = 0
        return {}, {}


class FakeRunner:
    def __init__(self):
        self.modes = []
        self.alg = SimpleNamespace(eval_mode=lambda: self.modes.append("eval"), train_mode=lambda: self.modes.append("train"),
                                   get_policy=lambda: (lambda obs: torch.zeros(1, 29)))
        self.logger = SimpleNamespace(rewbuffer=[1.0, 3.0])
        self.current_learning_iteration = 4


@pytest.fixture
def bundle(tmp_path):
    """A minimal Bundle around the front kick's reference (no mjlab needed)."""
    qpos = np.load(QPOS).astype(np.float64)
    return B.Bundle(path=tmp_path / "x.zip", manifest={"id": "kick"}, qpos=qpos, fps=30.0, spec=json.loads(SPEC.read_text()),
                    reward_config={})


def test_attempts_write_the_npz_and_the_index(tmp_path, bundle, monkeypatch):
    qpos = bundle.qpos
    out = tmp_path / "policy"
    at = A.Attempts(bundle, out, iterations=100, every=50, log=lambda *a: None)
    env = FakeEnv(qpos, fall_at=8)

    def build(self):
        self.env, self.venv = env, env
        self.body_ids, self.steps, self.fps, self.task = None, 20, 50, "Mjlab-Tracking-Flat-Unitree-G1"
    monkeypatch.setattr(A.Attempts, "_build", build)
    runner = FakeRunner()
    at.note_iteration(runner)
    row = at.record(runner, 50)
    assert runner.modes == ["eval", "train"]  # the policy is evaluated, then training mode comes back
    assert row["iteration"] == 50 and row["frames"] == 7 and row["fell"] and row["fell_at_s"] == 0.14
    assert row["reward"] == pytest.approx(0.5 * 8) and row["stages_total"] is None  # no body positions: not scored
    f = out / "attempts" / "iter_0050.npz"
    z = np.load(f)
    assert z["qpos"].shape == (7, 36) and z["qpos"].dtype == np.float32 and int(z["fps"]) == 50 and bool(z["fell"])
    np.testing.assert_allclose(z["qpos"][0], qpos[0], atol=1e-5)
    index = json.loads((out / "attempts" / "index.json").read_text())
    assert index["iterations"] == 100 and index["attempt_every"] == 50 and index["policy"] == "tracking_var"
    assert [r["file"] for r in index["attempts"]] == ["iter_0050.npz"] and index["curve"] == [[5, 2.0]]
    # the final attempt, a full episode this time
    env.fall_at = None
    at.record(runner, 100)
    index = json.loads((out / "attempts" / "index.json").read_text())
    assert [r["iteration"] for r in index["attempts"]] == [50, 100]
    assert index["attempts"][1]["frames"] == 20 and not index["attempts"][1]["fell"]
    assert sorted(p.name for p in (out / "attempts").iterdir()) == ["index.json", "iter_0050.npz", "iter_0100.npz"]


def test_the_loader_ships_the_attempts_module():
    from a2h_env import LOADER_FILES
    assert "attempts.py" in LOADER_FILES and "attempts.py" in X.LOADER_FILES
