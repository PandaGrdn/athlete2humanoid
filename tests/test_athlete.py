"""Choosing the athlete among several people (athlete.py, track.main_track_masks): the CLI spec, the
choice on SAM 3.1's per-object outputs, and the identity-switch check. Fake SAM outputs; no model.

    python -m unittest tests.test_athlete
"""

import importlib.util
import unittest

if not importlib.util.find_spec("numpy"):
    raise unittest.SkipTest("needs numpy")

import numpy as np

import athlete
import track

H, W = 120, 200


def blob(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def sam_outputs(n_frames=6, drift=0):
    """Three people: 1 on the left (seen often, high confidence: SAM's default pick), 2 in the middle,
    3 on the right; frame 0 has all three."""
    out = {}
    for f in range(n_frames):
        d = f * drift
        out[f] = {"out_obj_ids": [1, 2, 3], "out_probs": [0.99, 0.6, 0.5],
                  "out_binary_masks": [blob(10, 20, 40, 100), blob(90 + d, 20, 120 + d, 100), blob(160, 30, 190, 110)]}
    return out


class TestParse(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(athlete.parse("640,360"), {"point": (640.0, 360.0)})
        self.assertEqual(athlete.parse("700,900,500,100"), {"box": (500.0, 100.0, 700.0, 900.0)})
        self.assertEqual(athlete.parse("2"), {"index": 2})
        self.assertIsNone(athlete.parse(None))
        self.assertIsNone(athlete.parse(""))
        with self.assertRaises(ValueError):
            athlete.parse("1,2,3")


class TestSelectObject(unittest.TestCase):
    def test_default_is_most_seen(self):
        _, best = track.main_track_masks(sam_outputs(), 6)
        self.assertEqual(best, 1)

    def test_click_picks_the_person_under_it(self):
        masks, best = track.main_track_masks(sam_outputs(), 6, select="100,50")
        self.assertEqual(best, 2)
        self.assertTrue(masks[0, 50, 100] and not masks[0, 50, 20])

    def test_click_off_everyone_picks_the_nearest(self):
        self.assertEqual(athlete.select_object(sam_outputs(), "150,5"), 3)

    def test_box_and_index(self):
        self.assertEqual(athlete.select_object(sam_outputs(), "150,25,195,115"), 3)
        self.assertEqual(athlete.select_object(sam_outputs(), "0"), 1)  # counted from the left
        self.assertEqual(athlete.select_object(sam_outputs(), "1"), 2)
        with self.assertRaises(RuntimeError):
            athlete.select_object(sam_outputs(), "5")

    def test_picks_on_the_first_frame_with_anyone(self):
        out = sam_outputs()
        out[0] = {"out_obj_ids": [], "out_probs": [], "out_binary_masks": []}
        self.assertEqual(athlete.select_object(out, "100,50"), 2)


class TestContinuity(unittest.TestCase):
    def test_smooth_track_has_no_jumps(self):
        masks, _ = track.main_track_masks(sam_outputs(drift=3), 6, select="100,50")
        self.assertEqual(athlete.continuity(masks), {"seen_frac": 1.0, "jumps": []})

    def test_switch_is_flagged(self):
        masks = np.stack([blob(10, 20, 40, 100)] * 3 + [blob(150, 20, 180, 100)] * 3)
        self.assertEqual(athlete.continuity(masks)["jumps"], [3])


if __name__ == "__main__":
    unittest.main()
