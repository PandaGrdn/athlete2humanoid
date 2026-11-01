"""Export each run's training environment: outputs/remote_<run>/env.zip, the file the API serves at
/api/runs/<id>/env.zip and scripts/build_space.py ships.

    python scripts/export_env.py outputs/remote_latent_s3_*        # the repo's venv: numpy, mujoco, scipy, torch
    python scripts/export_env.py outputs/remote_latent_s3_kick --out-dir build/envs --force

A run whose clip was trained on variants (outputs/remote_variants_<tag>/) takes that set; otherwise the TRAIN and
HELD_OUT grids are generated from the retarget (--no-generate: the verdicts from variants.json only). See
docs/TRAINING_ENV.md and a2h_env/export.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from a2h_env import export  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run folders (each with task_spec.json and clips/<clip>/qpos.npy)")
    ap.add_argument("--out-dir", help="write <id>.a2h-env.zip here instead of env.zip in each run folder")
    ap.add_argument("--no-generate", action="store_true", help="don't generate variants for runs that have none")
    ap.add_argument("--force", action="store_true", help="rebuild an env.zip that is already there")
    a = ap.parse_args()
    export.main(a.runs, out_dir=a.out_dir, generate=not a.no_generate, force=a.force)
