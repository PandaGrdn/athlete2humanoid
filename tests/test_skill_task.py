"""skill/task.py on synthetic body trajectories (no mujoco): the spec is read off the reference,
and success depends on doing the stages in order, not on copying the reference's timing or poses.

    python -m unittest tests.test_skill_task
"""

import importlib.util
import unittest

if all(importlib.util.find_spec(m) for m in ("numpy", "torch")):
    import numpy as np
    import torch

    from skill import task as T
else:  # e.g. a bare system Python: the rest of the suite still runs
    raise unittest.SkipTest("needs numpy and torch")

FPS = 30.0


def traj(segments, hands=None):
    """segments: [(seconds, pelvis_z_end, feet_z_end)] linearly interpolated from a 0.8 m stand
    -> (T, 14, 3) body positions: torso 0.3 m above the pelvis, feet at 0.05 m + clearance."""
    z, f = [0.8], [0.0]
    for sec, z1, f1 in segments:
        n = max(int(round(sec * FPS)), 1)
        z += list(np.linspace(z[-1], z1, n + 1)[1:])
        f += list(np.linspace(f[-1], f1, n + 1)[1:])
    z, f = np.array(z), np.array(f)
    pos = np.zeros((len(z), 14, 3), np.float32)
    pos[:, :, 2] = z[:, None]
    pos[:, T.PELVIS, 2] = z
    pos[:, T.TORSO, 2] = z + 0.3
    pos[:, [T.L_ANKLE, T.R_ANKLE], 2] = 0.05 + f[:, None]
    pos[:, T.L_ANKLE, 1], pos[:, T.R_ANKLE, 1] = 0.1, -0.1
    for h in (T.L_HAND, T.R_HAND):
        pos[:, h, 2] = z + (hands[: len(z)] if hands is not None else -0.1)
    return pos


STAND = [(1.0, 0.8, 0.0)]


def squat_jump(depth=0.2, clearance=0.25, lead=0.0):
    return traj([(1.0 + lead, 0.8, 0.0), (0.6, 0.8 - depth, 0.0), (0.2, 0.8 - depth, 0.0),
                 (0.2, 0.8 + clearance, clearance), (0.2, 0.82, 0.0), (0.3, 0.62, 0.0),
                 (0.6, 0.8, 0.0), (1.0, 0.8, 0.0)])


class SkillTaskTest(unittest.TestCase):
    def setUp(self):
        self.ref = squat_jump()
        self.spec = T.compile_spec(self.ref, FPS)

    def test_spec_reads_the_stages_off_the_reference(self):
        self.assertEqual([s["kind"] for s in self.spec["stages"]], ["stand", "squat", "jump", "land", "stand"])
        jump = self.spec["stages"][2]
        self.assertAlmostEqual(jump["clearance"], 0.25, delta=0.02)
        self.assertAlmostEqual(self.spec["stages"][1]["depth"], 0.2, delta=0.02)

    def test_jump_target_is_clipped_to_the_robot(self):
        spec = T.compile_spec(squat_jump(clearance=0.5), FPS)
        self.assertEqual(spec["stages"][2]["clearance"], T.MAX_CLEARANCE)
        self.assertGreater(spec["stages"][2]["ref_clearance"], 0.45)

    def test_reference_passes_its_own_spec(self):
        r = T.evaluate(self.spec, self.ref, FPS)
        self.assertTrue(r["success"], r)
        self.assertEqual(r["jump_quality"], [1.0])

    def test_success_is_semantic_not_a_copy(self):
        # Later, a shallower (but deep enough) squat, a lower jump: still the same action.
        r = T.evaluate(self.spec, squat_jump(depth=0.17, clearance=0.15, lead=0.7), FPS)
        self.assertTrue(r["success"], r)
        self.assertAlmostEqual(r["jump_quality"][0], 0.6, delta=0.08)

    def test_arm_waving_before_the_jump_does_not_matter(self):
        n = len(squat_jump())
        hands = np.where(np.arange(n) < 30, 0.4 * np.sin(np.arange(n) / 2.0) + 0.3, -0.1)  # hype the crowd
        self.assertTrue(T.evaluate(self.spec, traj([(1.0, 0.8, 0.0), (0.6, 0.6, 0.0), (0.2, 0.6, 0.0),
                                                    (0.2, 1.05, 0.25), (0.2, 0.82, 0.0), (0.3, 0.62, 0.0),
                                                    (0.6, 0.8, 0.0), (1.0, 0.8, 0.0)], hands), FPS)["success"])

    def test_skipping_the_jump_fails(self):
        r = T.evaluate(self.spec, traj([(1.0, 0.8, 0.0), (0.6, 0.6, 0.0), (0.6, 0.8, 0.0), (2.0, 0.8, 0.0)]), FPS)
        self.assertFalse(r["success"])
        self.assertEqual(r["next_stage"], "jump")
        self.assertEqual(r["stages_completed"], 2)

    def test_a_fall_fails(self):
        pos = squat_jump()
        pos[60:, :, 2] = 0.1  # collapsed after the jump
        r = T.evaluate(self.spec, pos, FPS)
        self.assertFalse(r["success"])
        self.assertIsNotNone(r["fell_at_s"])

    def test_bobbing_earns_no_more_than_one_stage_of_progress(self):
        bob = [(0.3, 0.7, 0.0), (0.3, 0.8, 0.0)] * 10  # never deep enough to complete the squat
        pos = torch.as_tensor(traj([(1.0, 0.8, 0.0)] + bob))
        m = T.StageMachine(self.spec, 1, dt=1 / FPS)
        m.reset(torch.tensor([0]), 0.0, pos[:1])
        total = sum(float(m.step(pos[t:t + 1])[0]) / FPS for t in range(1, len(pos)))
        # stand bonus + its progress, then squat progress that never accumulates past one pass
        self.assertLess(total, (m.w["stage"] + m.w["progress"]) + m.w["progress"] + 0.05)
        self.assertEqual(int(m.completed[0]), 1)

    def test_batched_resets_start_at_the_reference_stage(self):
        m = T.StageMachine(self.spec, 3, dt=0.02)
        pos = torch.as_tensor(self.ref[[0, 50, 58]])
        for i, t in enumerate((0.0, 1.7, 1.95)):
            m.reset(torch.tensor([i]), t, pos[i:i + 1])
        self.assertEqual(m.stage.tolist(), [0, 1, 2])
        self.assertEqual(tuple(m.obs().shape), (3, T.MAX_STAGES + 2))
        self.assertEqual(tuple(m.step(pos).shape), (3,))

    def test_muse_skill_labels_and_checks(self):
        skill = {"action": {"label": "squat_jump"}, "tempo": {"airborne": False, "reps": 1},
                 "phases": [{"name": "crouch", "start_s": 1.0, "end_s": 1.8, "description": "load"},
                            {"name": "takeoff_and_flight", "start_s": 1.8, "end_s": 2.2, "description": "fly"}],
                 "success_criteria": [{"name": "air", "metric": "airborne_s", "op": ">=", "value": 0.1}]}
        spec = T.compile_spec(self.ref, FPS, skill)
        self.assertEqual(spec["stages"][1]["label"], "crouch")
        self.assertEqual(spec["stages"][2]["label"], "takeoff_and_flight")
        self.assertTrue(any("grounded" in w for w in spec["warnings"]))
        crit = T.evaluate(spec, self.ref, FPS)["criteria"]
        self.assertTrue(crit[0]["pass"])


if __name__ == "__main__":
    unittest.main()
