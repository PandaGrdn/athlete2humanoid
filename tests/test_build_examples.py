"""scripts/build_examples.py: the committed web examples validate, a missing video fails the check, and
the numbers it quotes from docs are parsed from the docs (stdlib only; no outputs/ or ffmpeg needed).

    python -m unittest tests.test_build_examples
"""

import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_examples", REPO / "scripts" / "build_examples.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def quiet(f, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a)


class TestExamples(unittest.TestCase):
    def test_committed_examples_validate(self):
        self.assertEqual(quiet(B.check, B.OUT), 0)

    def test_missing_video_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "examples"
            shutil.copytree(B.OUT, out)
            video = next(p for p in B.referenced(out) if p.endswith(".mp4"))
            (out / video).unlink()
            self.assertEqual(quiet(B.check, out), 1)

    def test_index_matches_examples(self):
        index = json.loads((B.OUT / "index.json").read_text())
        ids = [e["id"] for e in index["examples"]]
        self.assertEqual(sorted(ids), sorted(B.BUILDERS))
        for i in ids:
            ex = json.loads((B.OUT / i / "example.json").read_text())
            if ex["status"] == "ready":
                for s in ex["sources"]:  # repo files cited as sources exist (outputs/ is local only)
                    if not s.startswith("outputs/"):
                        self.assertTrue((REPO / s).exists(), s)

    def test_semantic_result_from_doc(self):
        r = B.semantic_result(REPO / "docs" / "SEMANTIC_REWARD.md", "v1")
        self.assertEqual((r["done"], r["total"]), (5, 5))
        self.assertEqual(r["policies"], ["tracking", "hybrid", "latent"])


if __name__ == "__main__":
    unittest.main()
