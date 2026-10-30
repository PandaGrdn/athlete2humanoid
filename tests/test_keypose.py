"""skill/keypose.py: key moments for any action, on synthetic body trajectories (no mujoco needed)."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from skill import keypose as K  # noqa: E402
from skill import task as T  # noqa: E402

FPS = 30.0
# rest pose, mjlab's G1 tracking bodies (latent/reward_net.py BODIES), facing +x
REST = np.array([
    [0.0, 0.0, 0.78],  # pelvis
    [0.0, 0.10, 0.70], [0.02, 0.10, 0.42], [0.0, 0.10, 0.08],  # left hip, knee, ankle
    [0.0, -0.10, 0.70], [0.02, -0.10, 0.42], [0.0, -0.10, 0.08],  # right hip, knee, ankle
    [0.0, 0.0, 1.05],  # torso
    [0.0, 0.18, 1.20], [0.0, 0.20, 0.98], [0.0, 0.20, 0.78],  # left shoulder, elbow, wrist
    [0.0, -0.18, 1.20], [0.0, -0.20, 0.98], [0.0, -0.20, 0.78],  # right shoulder, elbow, wrist
], np.float32)
LEFT_LEG, RIGHT_LEG, UPPER = [1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11, 12, 13]


def bump(t, c, w):
    """Smooth 0 -> 1 -> 0 around time c, half width w."""
    x = np.clip((t - c) / w, -1, 1)
    return 0.5 * (1 + np.cos(np.pi * x))


def motion(kind, seconds=4.0, reps=2, seed=0):
    t = np.arange(int(seconds * FPS)) / FPS
    pos = np.repeat(REST[None], len(t), 0).copy()
    pos += np.random.default_rng(seed).normal(0, 0.003, pos.shape).astype(np.float32)  # pose noise
    centres = [seconds * (i + 1) / (reps + 1) for i in range(reps)]
    for c in centres:
        if kind == "squat":
            b = bump(t, c, 0.6)[:, None]
            pos[:, [0, 7, 8, 9, 10, 11, 12, 13], 2] -= 0.25 * b
            pos[:, [2, 5], 0] += 0.15 * b
        elif kind == "kick":  # the right foot forward and up, the left one planted
            b = bump(t, c, 0.45)[:, None]
            pos[:, 6, 0] += 0.55 * b[:, 0]
            pos[:, 6, 2] += 0.45 * b[:, 0]
            pos[:, 5, 0] += 0.3 * b[:, 0]
            pos[:, 5, 2] += 0.2 * b[:, 0]
        elif kind == "punch":  # the right hand straight forward
            b = bump(t, c, 0.3)[:, None]
            pos[:, 13, 0] += 0.5 * b[:, 0]
            pos[:, 13, 2] += 0.4 * b[:, 0]
            pos[:, 12, 0] += 0.25 * b[:, 0]
        elif kind == "jump":  # countermovement, then 0.3 s in the air
            dip = bump(t, c - 0.35, 0.25)[:, None]
            pos[:, [0, 7, 8, 9, 10, 11, 12, 13], 2] -= 0.2 * dip
            air = np.clip(1 - ((t - c) / 0.15) ** 2, 0, None)
            pos[:, :, 2] += 0.25 * air[:, None]
    return pos


def turned(pos, yaw, shift=(1.0, -2.0)):
    c, s = math.cos(yaw), math.sin(yaw)
    out = pos.copy()
    out[..., 0], out[..., 1] = c * pos[..., 0] - s * pos[..., 1] + shift[0], s * pos[..., 0] + c * pos[..., 1] + shift[1]
    return out


def self_spec(spec):
    return {**spec, "down": spec["down_ref"], "air": spec["air_ref"]}


@pytest.mark.parametrize("kind", ["squat", "kick", "punch", "jump"])
def test_every_action_compiles_and_its_own_motion_passes(kind):
    pos = motion(kind)
    spec = K.compile_keypose_spec(pos, FPS, name=kind)
    assert spec["type"] == "keypose" and len(spec["stages"]) <= T.MAX_STAGES
    assert any(s["kind"] == "pose" for s in spec["stages"])
    r = T.evaluate(self_spec(spec), pos, FPS)
    assert r["success"], (kind, r["stages_completed"], r["stages_total"], spec["summary"])


@pytest.mark.parametrize("kind", ["squat", "kick", "punch", "jump"])
def test_standing_still_fails_every_spec(kind):
    pos = motion(kind)
    spec = K.compile_keypose_spec(pos, FPS)
    r = T.evaluate(self_spec(spec), np.repeat(pos[:1], len(pos), 0), FPS)
    assert not r["success"] and r["stages_completed"] <= 2


def test_specs_tell_actions_apart():
    pos = {k: motion(k) for k in ("squat", "kick", "punch")}
    for k, p in pos.items():
        spec = self_spec(K.compile_keypose_spec(p, FPS))
        for other, q in pos.items():
            assert T.evaluate(spec, q, FPS)["success"] == (other == k), (k, other)


def test_the_goal_is_the_moving_part():
    spec = K.compile_keypose_spec(motion("kick"), FPS)
    first = next(s for s in spec["stages"] if s["kind"] == "pose")
    top = spec["features"][int(np.argmax(first["relevance"]))]
    assert top.startswith("right ankle") or top.startswith("right foot"), top
    assert "right foot" in first["name"]
    assert not first["optional"]
    assert first["feet_down"] == [True, False]  # the support foot stays down, the kicking one doesn't
    punch = next(s for s in K.compile_keypose_spec(motion("punch"), FPS)["stages"] if s["kind"] == "pose")
    assert "right hand" in punch["name"]


def test_timing_and_heading_are_free():
    pos = motion("kick")
    spec = self_spec(K.compile_keypose_spec(pos, FPS))
    slow = np.stack([np.interp(np.linspace(0, len(pos) - 1, int(1.5 * len(pos))), np.arange(len(pos)), pos[:, b, a])
                     for b in range(pos.shape[1]) for a in range(3)], -1).reshape(-1, pos.shape[1], 3)
    assert T.evaluate(spec, slow.astype(np.float32), FPS)["success"]
    assert T.evaluate(spec, turned(pos, 1.3), FPS)["success"]


def test_flight_is_its_own_stage():
    spec = K.compile_keypose_spec(motion("jump"), FPS)
    kinds = [s["kind"] for s in spec["stages"]]
    assert kinds.count("flight") == 2
    for i, k in enumerate(kinds):
        if k == "flight":
            assert kinds[i + 1] == "settle" and spec["stages"][i + 1]["name"] == "land"
    r = T.evaluate(self_spec(spec), motion("jump"), FPS)
    assert r["success"] and len(r["jump_quality"]) == 2


def test_optional_moments_can_be_skipped():
    spec = K.compile_keypose_spec(motion("squat"), FPS)
    poses = [i for i, s in enumerate(spec["stages"]) if s["kind"] == "pose"]
    spec["stages"][poses[0]]["optional"] = True
    # a goal nobody reaches: only skipping gets past it
    spec["stages"][poses[0]]["goal"] = [g + 5.0 for g in spec["stages"][poses[0]]["goal"]]
    r = T.evaluate(self_spec(spec), motion("squat"), FPS)
    assert r["success"] and r["stages_skipped"] >= 1
    spec["stages"][poses[0]]["optional"] = False
    assert not T.evaluate(self_spec(spec), motion("squat"), FPS)["success"]


def test_machine_interface_and_fall_refund():
    pos = torch.as_tensor(motion("squat"))
    spec = K.compile_keypose_spec(pos.numpy(), FPS)
    m = T.machine(spec, 3, dt=1 / FPS)
    assert isinstance(m, K.KeyposeMachine)
    m.reset(torch.arange(3), torch.tensor([0.0, 1.0, 2.0]), pos[[0, 30, 60]])
    assert m.obs().shape == (3, T.MAX_STAGES + 2)
    assert m.stage[2] >= m.stage[1] >= m.stage[0]  # mid-motion starts begin at the reference's stage
    assert spec["stages"][0]["name"] == "ready" and m.stage[0] == 1 and m.completed[0] == 1  # ready takes no time
    m = T.machine(spec, 2, dt=1 / FPS)
    m.reset(torch.arange(2), 0.0, pos[:1].expand(2, -1, -1))
    total = torch.zeros(2)
    for t in range(1, 75):
        total += m.step(pos[t:t + 1].expand(2, -1, -1)) * m.dt
    assert m.completed[0] >= 2 and total[0] > 0
    fell = torch.tensor([True, False])
    r = m.step(pos[75:76].expand(2, -1, -1), fell=fell) * m.dt
    assert r[0] < 0 and m.banked[0] == 0  # the attempt since the last settle is refunded


def test_importance_weights_the_approach_to_key_moments():
    spec = K.compile_keypose_spec(motion("kick"), FPS)
    imp = K.importance(spec, len(motion("kick")), FPS)
    w = np.array(imp["weight"])
    assert imp["fps"] == FPS and len(w) == len(motion("kick")) and w.min() >= 0.1 - 1e-6
    k = next(s for s in spec["stages"] if s["kind"] == "pose")["frame"]
    assert w[k] > 0.9 and w[0] < 0.2


def test_noise_alone_has_no_key_moments():
    still = np.repeat(REST[None], 120, 0) + np.random.default_rng(1).normal(0, 0.003, (120, 14, 3)).astype(np.float32)
    spec = K.compile_keypose_spec(still, FPS)
    assert not [s for s in spec["stages"] if s["kind"] == "pose" and not s.get("optional")]


def test_done_the_end_pose_is_held():
    pos = torch.as_tensor(motion("squat"))
    spec = K.compile_keypose_spec(pos.numpy(), FPS)
    m = T.machine(spec, 2, dt=1 / FPS)
    m.reset(torch.arange(2), 0.0, pos[:1].expand(2, -1, -1))
    for t in range(1, len(pos)):
        m.step(pos[t:t + 1].expand(2, -1, -1))
    assert m.done.all() and m.style_until_done
    end = pos[-1:].expand(2, -1, -1)
    held = sum(m.step(end) * m.dt for _ in range(int(FPS)))
    assert torch.allclose(held, torch.full((2,), K.WEIGHTS["hold"]), atol=1e-3)  # 1 s in the end pose
    r = m.step(end, fell=torch.tensor([True, False])) * m.dt
    assert r[0] < -K.WEIGHTS["hold"] + 0.1 and r[1] > 0  # falling after the finish gives it back


def test_loose_task_requires_few_moments():
    spec = K.compile_keypose_spec(motion("squat", reps=3), FPS)
    inner = [s for s in spec["stages"] if s["kind"] == "pose" and s["level"] < 99]  # not the clip's end
    assert len(inner) > K.REQUIRED_TOP and sum(not s["optional"] for s in inner) == K.REQUIRED_TOP


def test_a_foot_on_its_toes_is_not_required_down():
    pos = motion("squat")
    pos[:, 3, 2] += 0.08 * bump(np.arange(len(pos)) / FPS, 4.0 / 3, 0.6)  # left heel up 8 cm through the first rep
    spec = K.compile_keypose_spec(pos, FPS)
    first = next(s for s in spec["stages"] if s["kind"] == "pose")
    assert first["feet_down"] == [False, True]  # low in the reference, but not planted by the robot's rule
    assert T.evaluate(spec, pos, FPS)["success"]  # so the reference passes under the robot's own thresholds


def test_a_tuck_hop_is_not_a_jump():
    """Feet off the floor with the pelvis at standing height (the task policies' squat-jump shortcut) fails."""
    pos = motion("jump")
    spec = self_spec(K.compile_keypose_spec(pos, FPS))
    flight = next(s for s in spec["stages"] if s["kind"] == "flight")
    assert flight["pelvis_rise"] > 0.1 and flight["min_air_s"] >= 0.6 * flight["ref_air_s"] - 1e-3
    t = np.arange(len(pos)) / FPS
    tuck = pos.copy()
    for c in (4.0 / 3, 8.0 / 3):
        air = np.clip(1 - ((t - c) / 0.15) ** 2, 0, None)
        tuck[:, :, 2] -= 0.25 * air[:, None]  # undo the body's rise...
        tuck[:, [3, 6], 2] += 0.25 * air[:, None]  # ...and lift only the feet
    r = T.evaluate(spec, tuck, FPS)
    assert not r["success"] and spec["stages"][r["stages_completed"]]["kind"] == "flight", r
    hop = pos.copy()  # a real but short hop: the body rises, for a third of the airtime
    for c in (4.0 / 3, 8.0 / 3):
        full = np.clip(1 - ((t - c) / 0.15) ** 2, 0, None)
        short = np.clip(1 - ((t - c) / 0.05) ** 2, 0, None)
        hop[:, :, 2] += 0.25 * (short - full)[:, None]
    assert not T.evaluate(spec, hop, FPS)["success"]


def test_a_key_pose_must_be_held():
    """A kick flicked through its apex (the task policies' front-kick shortcut) fails; held like the clip, it passes."""
    pos = motion("kick")
    spec = self_spec(K.compile_keypose_spec(pos, FPS))
    kick = next(s for s in spec["stages"] if s["kind"] == "pose")
    assert 0 < kick["hold_s"] <= K.HOLD_MAX and kick["hold_s"] == pytest.approx(min(0.5 * kick["ref_dwell_s"], K.HOLD_MAX), abs=1e-3)
    t = np.arange(len(pos)) / FPS
    flick = np.repeat(REST[None], len(t), 0).copy()
    flick += np.random.default_rng(0).normal(0, 0.003, flick.shape).astype(np.float32)
    for c in (4.0 / 3, 8.0 / 3):
        b = bump(t, c, 0.1)
        flick[:, 6, 0] += 0.55 * b
        flick[:, 6, 2] += 0.45 * b
        flick[:, 5, 0] += 0.3 * b
        flick[:, 5, 2] += 0.2 * b
    r = T.evaluate(spec, flick, FPS)
    assert not r["success"] and r["stages_completed"] <= 1, r
    assert T.evaluate(spec, pos, FPS)["success"]


def test_one_machine_scores_each_env_against_its_own_variant():
    """A machine over several specs (skill/variations.py's variants) gives each env the same rewards and
    stage counts as a machine over that env's spec alone."""
    clips = {k: motion(k) for k in ("kick", "squat", "jump")}
    specs = [self_spec(K.compile_keypose_spec(p, FPS)) for p in clips.values()]
    n = 6  # env i: variant i % 3, replaying motion i % 3 (envs 3-5 the kick/squat/jump again)
    m = K.KeyposeMachine(specs, n, dt=1 / FPS)
    m.set_variant(torch.arange(n), torch.arange(n) % 3)
    singles = [K.KeyposeMachine(s, 1, dt=1 / FPS) for s in specs]
    pos = [torch.as_tensor(clips[k]) for k in clips]
    T_ = min(len(p) for p in pos)
    m.reset(torch.arange(n), torch.zeros(n), torch.stack([pos[i % 3][0] for i in range(n)]))
    for s, p in zip(singles, pos):
        s.reset(torch.tensor([0]), 0.0, p[:1])
    for t in range(1, T_):
        r = m.step(torch.stack([pos[i % 3][t] for i in range(n)]))
        for i, (s, p) in enumerate(zip(singles, pos)):
            ri = s.step(p[t:t + 1])
            assert torch.allclose(r[i], ri[0], atol=1e-4) and torch.allclose(r[i + 3], ri[0], atol=1e-4), (t, i)
    assert [int(c) for c in m.completed[:3]] == [int(s.completed[0]) for s in singles]
    assert m.S == max(len(s["stages"]) for s in specs) and list(m.n_stages[:3]) == [len(s["stages"]) for s in specs]


def resampled(pos, speed):
    """The motion played `speed` times as fast (linear interpolation, same frame rate)."""
    t = np.arange(0, len(pos) - 1, speed)
    i0 = np.minimum(np.floor(t).astype(int), len(pos) - 2)
    w = (t - i0)[:, None, None]
    return (pos[i0] * (1 - w) + pos[i0 + 1] * w).astype(np.float32)


def held_kick(seconds=4.0):
    """The right foot kicked up by 1.4 s, held there to 2.6 s, down by 3.0 s: a key pose with a long hold (HOLD_MAX)."""
    t = np.arange(int(seconds * FPS)) / FPS
    pos = np.repeat(REST[None], len(t), 0).copy()
    pos += np.random.default_rng(0).normal(0, 0.003, pos.shape).astype(np.float32)
    up = 0.5 * (1 - np.cos(np.pi * np.clip((t - 1.0) / 0.4, 0, 1) * np.clip((3.0 - t) / 0.4, 0, 1)))
    for b, dx, dz in ((6, 0.55, 0.45), (5, 0.3, 0.2)):
        pos[:, b, 0] += dx * up
        pos[:, b, 2] += dz * up
    return pos, up


@pytest.mark.parametrize("every", [6, 5, 2])
def test_one_step_out_of_tolerance_does_not_reset_a_hold(every):
    """A hold counts HOLD_FILL of the steps in its window: the held kick with the foot snapping back to rest for one
    frame in five or six still passes (the old counter reset on each, so it never made its 0.4 s in a row); out
    every other frame fails."""
    pos, up = held_kick()
    spec = self_spec(K.compile_keypose_spec(pos, FPS))
    assert [s["hold_s"] for s in spec["stages"] if s["kind"] == "pose"] == [K.HOLD_MAX, K.HOLD_MAX]
    flicker = pos.copy()
    flicker[np.nonzero(up > 0.3)[0][::every]] = pos[0]
    r = T.evaluate(spec, flicker, FPS)
    assert r["success"] == (every != 2), r["abandoned"]


@pytest.mark.parametrize("kind,speed", [(k, s) for k in ("squat", "kick", "punch", "jump") for s in (0.7, 1.4)]
                         + [(k, 1.8) for k in ("squat", "kick", "punch")])
def test_the_reference_faster_or_slower_passes_its_own_spec(kind, speed):
    """Holds are a share of the stage's own duration and scale with the rollout's tempo: timing is free both ways.
    (Not a jump at 1.8x: its airtime is physics, not tempo, and isn't scaled.)"""
    pos = motion(kind)
    spec = self_spec(K.compile_keypose_spec(pos, FPS))
    r = T.evaluate(spec, resampled(pos, speed), FPS)
    assert r["success"], (kind, speed, r["stages_completed"], r["abandoned"])


def test_holds_scale_with_the_rollouts_tempo(monkeypatch):
    """A hold is at most HOLD_STAGE_FRAC of its stage's reference duration, and the machine shortens it by the
    rollout's tempo (down to TEMPO_MIN of hold_s): the punches twice as fast pass, and without the tempo scaling
    they don't. (test_a_key_pose_must_be_held: a flick at the reference's own tempo still fails.)"""
    pos = motion("punch")
    spec = self_spec(K.compile_keypose_spec(pos, FPS))
    for s in spec["stages"]:
        if s["kind"] == "pose":
            assert s["hold_s"] <= K.HOLD_STAGE_FRAC * (s["end_s"] - s["start_s"]) + 1e-3
    fast = resampled(pos, 2.0)
    assert T.evaluate(spec, fast, FPS)["success"]
    monkeypatch.setattr(K, "TEMPO_MIN", 1.0)
    assert not T.evaluate(spec, fast, FPS)["success"]


def test_arriving_far_too_early_is_not_a_faster_tempo(monkeypatch):
    """Already in the key pose at the start (5x "early"), for less than its hold: not the motion at another tempo,
    so the hold counts in full and it fails (the punches policies sat in the arms clip's one key pose this way)."""
    pos, _ = held_kick()
    spec = self_spec(K.compile_keypose_spec(pos, FPS))
    kick = next(s for s in spec["stages"] if s["kind"] == "pose")
    early = np.repeat(pos[:1], len(pos), 0)
    early[:int(0.27 * FPS)] = pos[kick["frame"]]  # kicked for 0.27 s from the first frame, then standing
    assert T.evaluate(spec, early, FPS)["stages_completed"] <= 1
    monkeypatch.setattr(K, "TEMPO_CUT", 0.0)
    assert T.evaluate(spec, early, FPS)["stages_completed"] >= 2


def test_tracking_rms_and_the_calibrated_floor():
    pos = motion("punch")
    spec = K.compile_keypose_spec(pos, FPS)
    feats = spec["features"]
    assert np.allclose(K.tracking_rms(pos, pos, spec["foot0"], fps=FPS), 0, atol=1e-6)
    off = pos.copy()
    off[:, 13, 2] += 0.03  # the right hand 3 cm high throughout
    rms = K.tracking_rms(off, pos, spec["foot0"], fps=FPS)
    assert rms[feats.index("right wrist z")] == pytest.approx(0.03, abs=1e-4)
    assert np.delete(rms, feats.index("right wrist z")).max() < 1e-4
    cal = K.calibration(rms)
    floor = np.array(cal["floor_m"])
    assert floor[feats.index("right wrist z")] == pytest.approx(K.TOL_MIN[feats.index("right wrist z")].item())  # 2 x 3 cm < 10 cm
    big = K.calibration(np.full(K.D, 0.5))  # a useless "tracker": capped
    assert np.allclose(big["floor_m"], K.CALIB_MAX * K.TOL_MIN.numpy(), atol=1e-4)
    rms2 = np.zeros(K.D)
    rms2[feats.index("right wrist z")] = 0.08
    cal2 = K.calibration(rms2)
    spec2 = K.calibrate(spec, cal2)
    for a, b in zip(spec["stages"], spec2["stages"]):
        if a["kind"] == "pose":
            assert all(y >= x - 1e-6 for x, y in zip(a["tol"], b["tol"]))  # floors only raise tolerances
            i = feats.index("right wrist z")
            w = max(a["relevance"][i], 1e-3)
            assert b["tol"][i] == pytest.approx(min(max(a["tol"][i], 0.16 / np.sqrt(w)), 9.99), abs=1e-3)
            assert b["relevance"] == a["relevance"] and b["goal"] == a["goal"] and b["hold_s"] == a["hold_s"]
    assert K.calibrate(spec2, cal2)["stages"] == spec2["stages"]  # re-calibrating starts from tol_base
    assert spec2["calibration"]["floor_m"] == cal2["floor_m"]


def test_a_floor_lets_a_good_tracker_pass_and_still_rejects_a_wrong_move():
    """A tracker whose right hand sits 15 cm low throughout fails uncalibrated (beyond the punch's wrist tolerance);
    with the clip calibrated from it (floor 2 x its RMS, capped) it passes; standing still and punching with the
    other hand still fail, even at the floors' cap."""
    pos = motion("punch")
    spec = self_spec(K.compile_keypose_spec(pos, FPS))
    short = pos.copy()
    short[:, 13, 2] -= 0.15
    assert not T.evaluate(spec, short, FPS)["success"]
    cal = K.calibration(K.tracking_rms(short, pos, spec["foot0"], fps=FPS))
    assert T.evaluate(K.calibrate(spec, cal), short, FPS)["success"]
    capped = K.calibrate(spec, K.calibration(np.full(K.D, 1.0)))
    still = np.repeat(pos[:1], len(pos), 0)
    other = pos.copy()  # the left hand punches instead of the right
    other[:, 10, 0] += pos[:, 13, 0] - pos[:1, 13, 0]
    other[:, 10, 2] += pos[:, 13, 2] - pos[:1, 13, 2]
    other[:, [12, 13]] = pos[:1, [12, 13]]
    for wrong in (still, other):
        assert not T.evaluate(capped, wrong, FPS)["success"]


def test_evaluate_names_the_feature_that_failed():
    pos = motion("punch")
    spec = self_spec(K.compile_keypose_spec(pos, FPS))
    weak = pos.copy()
    weak[:, 13] = pos[:1, 13] + 0.3 * (pos[:, 13] - pos[:1, 13])  # the punch only a third of the way
    r = T.evaluate(spec, weak, FPS)
    assert not r["success"]
    last = r["abandoned"][-1]
    assert last["why"] == "clip ended" and last["kind"] == "pose" and last["stage"] == r["stages_completed"]
    assert last["feature"].startswith("right wrist") and last["error_tol"] > 1 and not last["reached"]
    assert all(a["why"] == "skipped" for a in T.evaluate(spec, pos, FPS)["abandoned"])  # only optional details


def test_upright_is_the_torso_not_the_pelvis_link():
    """Upright reads pelvis -> shoulder midpoint: 2 cm of noise on every body is not a fall (it was, on the 4.4 cm
    pelvis -> torso_link vector); an 85 deg lean is, and a squat-like 50 deg forward lean is not."""
    g1 = np.repeat(REST[None], 200, 0).copy()
    g1[:, 7] = g1[:, 0] + [0, 0, 0.044]  # the G1's torso_link sits 4.4 cm above the pelvis
    noisy = g1 + np.random.default_rng(0).normal(0, 0.02, g1.shape).astype(np.float32)
    assert not T.fallen(noisy).any()

    def lean(deg):
        a = np.radians(deg)
        out = REST.copy()
        rel = REST - REST[0]
        upper = [7, 8, 9, 10, 11, 12, 13]
        out[upper, 0] = REST[0, 0] + np.cos(a) * rel[upper, 0] + np.sin(a) * rel[upper, 2]
        out[upper, 2] = REST[0, 2] - np.sin(a) * rel[upper, 0] + np.cos(a) * rel[upper, 2]
        return out[None]
    assert T.fallen(lean(85))[0] and not T.fallen(lean(50))[0]
    assert float(T.upright_gate(T.features(lean(50), 0.0)[3])[0]) == 1.0  # a squatter's lean keeps its progress
    assert float(T.upright_gate(T.features(lean(78), 0.0)[3])[0]) == 0.0


def test_a_clip_cut_mid_motion_does_not_end_on_its_last_frame():
    """The punches clip ends as the athlete steps out of frame: its last frame is a pose nobody holds, so a tracker
    a few frames behind could never finish. The goals end before that burst; a 3-frame lag passes."""
    pos = motion("punch")
    n = 4
    step = np.linspace(0, 1, n + 1)[1:] ** 2
    pos[-n:, 6, 2] += 0.2 * step  # the right foot lifts fast over the last frames
    pos[-n:, 6, 0] += 0.25 * step
    pos[-n:, 0, 0] += 0.1 * step
    spec = self_spec(K.compile_keypose_spec(pos, FPS))
    assert spec["goals_end_s"] < spec["duration_s"] and spec["warnings"]
    assert T.evaluate(spec, pos, FPS)["success"]
    lag = np.concatenate([np.repeat(pos[:1], 3, 0), pos[:-3]])
    assert T.evaluate(spec, lag, FPS)["success"]
    calm = K.compile_keypose_spec(motion("punch"), FPS)  # a clip that ends at rest is compiled as before
    assert "goals_end_s" not in calm and not calm["warnings"]
