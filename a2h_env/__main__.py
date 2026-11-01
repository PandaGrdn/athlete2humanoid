"""python -m a2h_env info <bundle> | train <bundle> --iters N | export <run folder> (the last one needs the repository)."""

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m a2h_env", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("info", help="what a bundle holds (numpy only)")
    i.add_argument("bundle")
    i.add_argument("--json", action="store_true", help="print the manifest instead")
    t = sub.add_parser("train", help="PPO on the bundle with mjlab (needs mjlab)")
    t.add_argument("bundle")
    t.add_argument("--iters", type=int, default=1500)
    t.add_argument("--num-envs", type=int, default=4096)
    t.add_argument("--variant", default="train", help="train (every kept training variant, the default), reference, or a variant's name")
    t.add_argument("--out", help="output folder (default: <bundle>.train/)")
    t.add_argument("--device", help="cuda:0 (default when available) or cpu")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save-interval", type=int, default=250)
    e = sub.add_parser("export", help="a run folder of the athlete2humanoid repository -> its bundle")
    e.add_argument("runs", nargs="+")
    e.add_argument("--out-dir")
    e.add_argument("--no-generate", action="store_true")
    e.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "info":
        from .bundle import describe, load_bundle
        b = load_bundle(a.bundle)
        print(json.dumps(b.manifest, indent=1) if a.json else describe(b))
    elif a.cmd == "train":
        from pathlib import Path

        from .bundle import load_bundle
        from .train import train
        b = load_bundle(a.bundle)
        out = a.out or str(Path(a.bundle).with_name(Path(a.bundle).name.removesuffix(".zip") + ".train"))
        train(b, out, iters=a.iters, num_envs=a.num_envs, variant=a.variant, device=a.device, seed=a.seed,
              save_interval=a.save_interval)
    else:
        from .export import main as export_main
        export_main(a.runs, out_dir=a.out_dir, generate=not a.no_generate, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
