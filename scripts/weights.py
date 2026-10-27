"""Trained policy weights: find them, keep the good ones, build off them.

    python scripts/weights.py list                            # every policy in outputs/, with its scores
    python scripts/weights.py keep remote_latent_sj_task task --name sj_task_v1
    python scripts/weights.py push weights/sj_task_v1 <user>/athlete2humanoid-g1 --private   # HF Hub

`list` scans outputs/*/policy_*/ (checkpoint, weights.json from train_motion.py, eval.json scores)
and writes weights/INDEX.json. `keep` copies one policy into weights/<name>/ (checkpoint, ONNX if
exported, motion.npz, weights.json, its eval scores and the run's task_spec.json / reward bundle),
where re-running a job with the same tag can't overwrite it. weights/ is gitignored; `push` uploads a
kept folder to a Hugging Face model repo (needs `hf auth login`; it publishes, so run it yourself).

To build off kept weights: train_motion.py --init weights/<name>/model_*.pt, or experiments/
latent_policy.py --init task=weights/<name>/model_*.pt (another task's weights are adapted), or
--resume-from <run folder> to continue the same task for more iterations. On Colab:
scripts/remote_run.py latent --reuse <run> --with-weights --args "--init /content/a2h/inputs".
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS, WEIGHTS = REPO / "outputs", REPO / "weights"
SCORES = ("aligned", "dtw", "task_success", "task_stages", "falls", "mean_torso_position_error_m")


def newest(d):
    c = sorted(Path(d).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return c[-1] if c else None


def policies(root=OUTPUTS):
    rows = []
    for d in sorted(root.glob("*/policy_*")):
        ck = newest(d)
        if not ck:
            continue
        run, name = d.parent, d.name.removeprefix("policy_")
        ev = json.loads((run / "eval.json").read_text()).get(name, {}) if (run / "eval.json").exists() else {}
        meta = json.loads((d / "weights.json").read_text()) if (d / "weights.json").exists() else {}
        rows.append({"run": run.name, "policy": name, "checkpoint": str(ck.relative_to(REPO)),
                     "iterations": meta.get("iterations", int(ck.stem.split("_")[1]) + 1),
                     "task": meta.get("task"), "started_from": (meta.get("started_from") or {}).get("checkpoint"),
                     **{k: ev[k] for k in SCORES if k in ev}})
    return rows


def cmd_list(a):
    rows = policies()
    WEIGHTS.mkdir(exist_ok=True)
    (WEIGHTS / "INDEX.json").write_text(json.dumps(rows, indent=1))
    kept = sorted(p.name for p in WEIGHTS.iterdir() if p.is_dir())
    print(f"{'run':28s} {'policy':10s} {'iters':>5s} {'aligned':>7s} {'task':>6s} {'falls':>5s}  checkpoint")
    for r in rows:
        al = f"{r['aligned']:.3f}" if "aligned" in r else "-"
        print(f"{r['run']:28s} {r['policy']:10s} {r['iterations']:5d} {al:>7s} {str(r.get('task_stages', '-')):>6s} "
              f"{str(r.get('falls', '-')):>5s}  {r['checkpoint']}")
    print(f"\n{len(rows)} policies -> weights/INDEX.json; kept: {', '.join(kept) or 'none'}")


def cmd_keep(a):
    run = OUTPUTS / a.run
    d = run / f"policy_{a.policy}"
    ck = newest(d)
    if not ck:
        raise SystemExit(f"no checkpoint in {d}")
    dst = WEIGHTS / (a.name or f"{a.run}_{a.policy}")
    if dst.exists() and not a.force:
        raise SystemExit(f"{dst} exists (--force to replace)")
    dst.mkdir(parents=True, exist_ok=True)
    for f in [ck, *d.glob("*.onnx"), d / "motion.npz", d / "weights.json", d / "policy.mp4",
              d / "reward_config.json", run / "task_spec.json", run / "reward.pt"]:
        if f.exists():
            shutil.copy(f, dst / f.name)
    ev = json.loads((run / "eval.json").read_text()).get(a.policy, {}) if (run / "eval.json").exists() else {}
    card = {"source": f"outputs/{a.run}/policy_{a.policy}", "checkpoint": ck.name, "scores": ev,
            "how_to_use": "train_motion.py --init <checkpoint> (or --resume-from for the same task); "
                          "mjlab_rollout.py --checkpoint <checkpoint> --motion motion.npz --task <weights.json task>"}
    (dst / "card.json").write_text(json.dumps(card, indent=1))
    print(f"kept {a.run}/{a.policy} -> {dst.relative_to(REPO)} ({', '.join(sorted(p.name for p in dst.iterdir()))})")


def cmd_push(a):
    src = Path(a.folder)
    if not (src / "card.json").exists():
        raise SystemExit(f"{src} is not a kept weights folder (scripts/weights.py keep)")
    hf = shutil.which("hf") or sys.exit("needs the Hugging Face CLI (pip install -U huggingface_hub) and hf auth login")
    subprocess.run([hf, "repos", "create", a.repo, "--repo-type", "model", "--exist-ok"]
                   + (["--private"] if a.private else []), check=True)
    subprocess.run([hf, "upload", a.repo, str(src), src.name, "--repo-type", "model"], check=True)
    print(f"uploaded {src} to https://huggingface.co/{a.repo}/tree/main/{src.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    k = sub.add_parser("keep")
    k.add_argument("run", help="folder in outputs/, e.g. remote_latent_sj_task")
    k.add_argument("policy", help="e.g. task, hybrid, tracking")
    k.add_argument("--name")
    k.add_argument("--force", action="store_true")
    p = sub.add_parser("push")
    p.add_argument("folder")
    p.add_argument("repo", help="<user>/<model repo>")
    p.add_argument("--private", action="store_true")
    a = ap.parse_args()
    {"list": cmd_list, "keep": cmd_keep, "push": cmd_push}[a.cmd](a)
