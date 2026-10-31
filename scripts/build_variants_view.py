"""The variants of a finished run's clip, as the web build view shows them: <run>/variants.json.

    python scripts/build_variants_view.py outputs/remote_latent_s3_*      # (the repo's venv: numpy, scipy, mujoco)

For a run whose clip was also trained on variants (outputs/remote_variants_<tag>, e.g. s3_kick), that set: its
index.json (every variant, kept or dropped, and why) and a few kept ones' qpos as ghosts. Otherwise the same
generator (skill/variations.py, the TRAIN grid) on the run's own retarget, marked as not trained on. The API
serves the file as /api/runs/<id>/files/variants.json; scripts/build_space.py ships it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def target(run):
    spec = json.loads((run / "task_spec.json").read_text())
    name = spec.get("name")
    if not name or not (run / "clips" / name / "qpos.npy").is_file():
        raise SystemExit(f"{run}: no clip named after its task spec ({name!r})")
    return name


def build(run):
    from skill import variations
    clip = target(run)
    tag = run.name.removeprefix("remote_latent_")
    trained = run.parent / f"remote_variants_{tag}"
    index = trained / "variants" / "index.json"
    if index.is_file():
        rows = json.loads(index.read_text())
        view = variations.preview(rows, lambda n: np.load(trained / "variants" / n / "qpos.npy"))
        view |= {"source": trained.name, "trained_on": True,
                 "policies": sorted(p.name.removeprefix("policy_") for p in trained.glob("policy_*") if p.is_dir())}
    else:
        qpos = np.load(run / "clips" / clip / "qpos.npy")
        vs = variations.generate(qpos, log=lambda *a: None)
        by_name = {v["name"]: v["qpos"] for v in vs}
        view = variations.preview(vs, by_name.__getitem__) | {"source": "generated", "trained_on": False, "policies": []}
    view["clip"] = clip
    out = run / "variants.json"
    out.write_text(json.dumps(view, separators=(",", ":")))
    print(f"{out}: {view['kept']}/{view['total']} kept, {len(view['ghosts'])} ghosts, from {view['source']} "
          f"({out.stat().st_size / 1e3:.0f} kB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    for run in ap.parse_args(argv).runs:
        if (run / "task_spec.json").is_file():
            build(run)


if __name__ == "__main__":
    main()
