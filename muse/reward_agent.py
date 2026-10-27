"""Reward-design agent: Muse Spark reads a finished run and proposes the next reward config.

    python -m muse.reward_agent outputs/latent_squat                    # one run
    python -m muse.reward_agent outputs/ --watch 300 --log outputs/agent_log.jsonl   # every new run, forever
    python -m muse.reward_agent outputs/latent_squat --dry-run --out-dir /tmp/agent

A run is a folder from experiments/latent_policy.py: eval.json (real V-JEPA scores and rollout
metrics per policy), distill_report.json (how good the reward network is), comparison.mp4 and
policy_<variant>/policy.mp4 (sent to Muse as frame contact sheets), and optionally
policy_<variant>/reward_config.json (the config it was trained with, written by train_motion.py
--reward-config; otherwise DEFAULTS) and a skill_spec.json from muse/skill_reader.py.

Muse answers with a diagnosis and a JSON patch (RFC 6902 "replace" ops) on the reward config; it is
checked against PROPOSAL_SCHEMA and BOUNDS, then written as reward_proposal.json and the patched
config as next_reward_config.json. Each proposal is appended to the log, which later proposals see
as history, so a loop over runs is: train with --reward-config next_reward_config.json
(experiments/latent_policy.py or train_motion.py) -> run this -> repeat.
"""

import argparse
import copy
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from muse import media  # noqa: E402
from muse.client import image_part, make_client, now, read_json, text_part, write_json  # noqa: E402
from muse.schema import BOOL, NUM, STR, arr, enum, obj  # noqa: E402

# mjlab_tasks.reward_config() defaults and experiments/latent_policy.py's --iters (tests check them).
DEFAULTS = {
    "hybrid": {"latent_weight": 3.0, "sigma": 0.1, "tracking_scale": 0.2, "iterations": 1500,
               "terminations": {"anchor_pos": 0.5, "anchor_ori": 1.2, "ee_body_pos": 0.6}},
    "latent": {"latent_weight": 3.0, "sigma": 0.1, "tracking_scale": 0.0, "iterations": 1500,
               "terminations": {"anchor_pos": 0.5, "anchor_ori": 1.2, "ee_body_pos": 0.6}},
}

# Patchable paths -> (low, high). Muse may not touch anything else.
BOUNDS = {
    "/latent_weight": (0.25, 10.0),
    "/sigma": (0.02, 0.5),
    "/tracking_scale": (0.0, 1.0),
    "/iterations": (200, 6000),
    "/terminations/anchor_pos": (0.2, 1.5),
    "/terminations/anchor_ori": (0.4, 3.2),
    "/terminations/ee_body_pos": (0.2, 1.5),
}
INTEGER_PATHS = {"/iterations"}
MAX_OPS = 3

PROPOSAL_SCHEMA = obj(
    diagnosis=STR,
    patch=arr(obj(op=enum(["replace"]), path=enum(BOUNDS), value=NUM, reason=STR)),
    expected_effect=STR,
    confidence=enum(["low", "medium", "high"]),
    stop=BOOL,
)

SYSTEM = (
    "You tune the reward of a reinforcement-learning run that teaches a Unitree G1 humanoid robot a motion "
    "from a video of a person. You get the run's evaluation, the current reward config and frames of the "
    "rollouts, and you propose the next config as a JSON patch. Change one to three settings, each for a "
    "reason you can point to in the numbers or frames. If no change is likely to help, set stop to true and "
    "leave the patch empty. Answer with JSON only."
)

CONFIG_DOC = """How the config is used (mjlab PPO, G1, 4096 parallel robots, 50 steps/s):
- latent reward per step = latent_weight * exp(-max(0, cos_ref - cos) / sigma), where cos is the reward
  network's cosine between the robot's last ~1 s of motion and the video's V-JEPA latent at that moment,
  and cos_ref is the same score for the retargeted motion. Smaller sigma = sharper, sparser reward.
- tracking_scale multiplies mjlab's six motion-tracking rewards (copy the retargeted motion);
  0 removes them (latent reward only, and the end-effector termination is dropped).
- terminations: an attempt ends when the torso drifts more than anchor_pos m or anchor_ori rad from the
  retarget, or a hand/foot more than ee_body_pos m. Loose = the robot may deviate to look more like the video.
- iterations: PPO iterations."""

EVAL_DOC = """Evaluation (real V-JEPA on silhouette renders, not the reward network; one row per policy,
plus "retarget" = the kinematic retargeted motion itself):
- aligned: cosine between the video's and the rollout's V-JEPA latents, time-aligned. Higher = looks more
  like the video. dtw / chamfer: the same with dynamic time warping / nearest-match.
- dtw_rank_vs_bank: rank of the rollout among the motion bank by similarity to the video (1 = closest).
- reward_net_aligned: the reward network's score (what the policy actually optimised).
- falls, mean_joint_error_rad, mean/max_torso_position_error_m: physics rollout vs the retarget."""


def run_arg(run_dir, flag):
    """A flag's value on the run's command line (status.json from scripts/job.py), or None."""
    status = read_json(Path(run_dir) / "status.json")
    args = [t for c in (status or {}).get("cmd", []) for t in shlex.split(c)]  # also splits `bash -c "..."`
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def default_config(variant, run_dir=None):
    """DEFAULTS[variant], with the run's --iters when its command line is known."""
    cfg = copy.deepcopy(DEFAULTS[variant])
    iters = run_arg(run_dir, "--iters") if run_dir else None
    if iters:
        cfg["iterations"] = int(iters)
    return cfg


def target_name(run_dir):
    return run_arg(run_dir, "--target")


def _ptr(path):
    return path.strip("/").split("/")


def get(cfg, path):
    for k in _ptr(path):
        cfg = cfg[k]
    return cfg


def apply_patch(cfg, patch):
    """RFC 6902 'replace' ops -> a new config."""
    cfg = copy.deepcopy(cfg)
    for op in patch:
        *parents, last = _ptr(op["path"])
        node = cfg
        for k in parents:
            node = node[k]
        if last not in node:
            raise KeyError(op["path"])
        node[last] = int(op["value"]) if op["path"] in INTEGER_PATHS else op["value"]
    return cfg


def check(proposal, cfg):
    """Checks beyond the schema: bounds, no repeats or no-ops, stop vs patch. -> list of problems."""
    errors = []
    if cfg["tracking_scale"] == 0 and any(op["path"] == "/tracking_scale" for op in proposal["patch"]):
        errors.append("/tracking_scale: the latent-only policy has no tracking rewards; it stays 0")
    patch = proposal["patch"]
    if proposal["stop"] and patch:
        errors.append("stop is true, so patch must be empty")
    if not proposal["stop"] and not 1 <= len(patch) <= MAX_OPS:
        errors.append(f"patch: 1-{MAX_OPS} ops (or stop: true)")
    paths = [op["path"] for op in patch]
    if len(set(paths)) != len(paths):
        errors.append("patch: each path at most once")
    for op in patch:
        lo, hi = BOUNDS[op["path"]]
        v = op["value"]
        if not lo <= v <= hi:
            errors.append(f"{op['path']}: {v} is outside [{lo}, {hi}]")
        if op["path"] in INTEGER_PATHS and float(v) != int(v):
            errors.append(f"{op['path']}: must be a whole number")
        if v == get(cfg, op["path"]):
            errors.append(f"{op['path']}: {v} is already the current value")
    return errors


def _bounds_text():
    return "\n".join(f"- {p}: [{lo}, {hi}]" + (" (integer)" if p in INTEGER_PATHS else "")
                     for p, (lo, hi) in BOUNDS.items())


def _skill_text(spec):
    a, t = spec["action"], spec["tempo"]
    bodies = ", ".join(f"{b['body']} {b['weight']:g}" for b in spec["key_bodies"])
    crit = "; ".join(f"{c['metric']} {c['op']} {c['value']:g}" for c in spec["success_criteria"])
    phases = "; ".join(f"{p['name']} {p['start_s']:.1f}-{p['end_s']:.1f} s ({p['match']})" for p in spec["phases"])
    return (f"Skill spec (from the video): {a['label']}: {a['description']} Tempo: {t['reps']} reps, "
            f"{t['cycle_s']:g} s each, {t['speed']}, airborne: {t['airborne']}. Phases: {phases}. "
            f"Key bodies: {bodies}. Success criteria: {crit}.")


def prompt(run, variant, cfg, history, images):
    ev = {k: v for k, v in run["eval"].items() if not k.startswith("_")}
    parts = [
        f"Run: {run['name']} (target clip: {run['target'] or 'unknown'}). Tune the '{variant}' policy.",
        CONFIG_DOC,
        "Current config:\n" + json.dumps(cfg, indent=1),
        "Allowed values:\n" + _bounds_text(),
        EVAL_DOC,
        "eval.json:\n" + json.dumps(ev, indent=1),
    ]
    if run["eval"].get("_bank_top5_dtw"):
        parts.append(f"Bank motions closest to the video by DTW: {run['eval']['_bank_top5_dtw']}")
    if run["distill"]:
        parts.append(
            "Reward network quality (distill_report.json): val_cosine_mean / val_cosine_p10 = how well it "
            "predicts V-JEPA on held-out motions; cos_ref_mean = its score for the retarget, the bar the policy "
            "must reach for full latent reward; pca_own_retarget_dtw_rank 1 = the compressed latent space still "
            "picks the clip's own retarget.\n" + json.dumps(run["distill"], indent=1))
    if run["skill"]:
        parts.append(_skill_text(run["skill"]))
    if history:
        parts.append("Earlier proposals in this loop (config -> resulting metrics):\n"
                     + "\n".join(json.dumps(h) for h in history))
    parts.append("Images: " + ("; ".join(desc for desc, _ in images) if images else "none available this time."))
    parts.append(f"Propose the next config for the '{variant}' policy.")
    return "\n\n".join(parts)


def load_run(run_dir, variant, frames=True):
    run_dir = Path(run_dir)
    target = target_name(run_dir)
    skill = read_json(run_dir / "skill_spec.json") or (read_json(run_dir / "clips" / target / "skill_spec.json")
                                                       if target else None)
    images = []
    if frames:
        for video, desc in ((run_dir / "comparison.mp4", "image {i}: rows are moments in time, top to bottom; "
                             "columns are the video's silhouette and each robot's silhouette, labelled in the corner"),
                            (run_dir / f"policy_{variant}" / "policy.mp4",
                             f"image {{i}}: the '{variant}' policy's physics rollout, moments in time top to bottom")):
            png = media.contact_sheet(video) if video.exists() else None
            if png:
                images.append((desc.format(i=len(images) + 1), png))
        if not images:
            print("no rollout frames (videos missing or OpenCV not installed); sending numbers only")
    return {"name": run_dir.name, "dir": run_dir, "target": target, "eval": read_json(run_dir / "eval.json"),
            "distill": read_json(run_dir / "distill_report.json"), "skill": skill, "images": images}


HISTORY_METRICS = ("aligned", "dtw_rank_vs_bank", "reward_net_aligned", "falls", "mean_torso_position_error_m")


def read_history(log, variant, n=8):
    if not log or not Path(log).is_file():
        return []
    rows = [json.loads(line) for line in Path(log).read_text().splitlines() if line.strip()]
    return [{k: r[k] for k in ("run", "config", "metrics", "patch")} for r in rows if r.get("variant") == variant][-n:]


def propose(run_dir, client, variant="hybrid", out_dir=None, log=None, frames=True):
    """One run -> reward_proposal.json + next_reward_config.json in out_dir (default: the run folder)."""
    run = load_run(run_dir, variant, frames)
    if variant not in run["eval"]:
        raise ValueError(f"{run_dir}/eval.json has no '{variant}' policy")
    cfg = (read_json(run["dir"] / f"policy_{variant}" / "reward_config.json")
           or read_json(run["dir"] / "reward_config.json") or default_config(variant, run["dir"]))
    if variant == "latent":
        cfg["tracking_scale"] = 0.0  # mjlab_tasks ignores the file's value for this task
    content = [text_part(prompt(run, variant, cfg, read_history(log, variant), run["images"]))]
    content += [image_part(png) for _, png in run["images"]]
    proposal, info = client.ask_json("reward_proposal", SYSTEM, content, PROPOSAL_SCHEMA,
                                     check=lambda p: check(p, cfg))
    nxt = apply_patch(cfg, proposal["patch"])
    out = Path(out_dir or run["dir"])
    out.mkdir(parents=True, exist_ok=True)
    proposal["_meta"] = {"run": str(run["dir"]), "variant": variant, "config": cfg, "created": now(), **info}
    write_json(out / "reward_proposal.json", proposal)
    write_json(out / "next_reward_config.json", nxt)
    if not client.live:
        client.save_requests(out / "reward_proposal.request.json")
    if log:
        metrics = {k: run["eval"][variant].get(k) for k in HISTORY_METRICS}
        row = {"run": run["name"], "variant": variant, "config": cfg, "metrics": metrics,
               "patch": proposal["patch"], "stop": proposal["stop"], "created": now(), "mode": info["mode"]}
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
    ops = ", ".join(f"{op['path']} {get(cfg, op['path'])} -> {op['value']}" for op in proposal["patch"]) or "no change"
    print(f"{run['name']} [{variant}]: {'stop' if proposal['stop'] else ops} ({info['mode']})")
    return proposal


def find_runs(paths):
    """Each path is a run (has eval.json) or a folder of runs."""
    runs = []
    for p in map(Path, paths):
        runs += [p] if (p / "eval.json").is_file() else sorted(d for d in p.iterdir() if (d / "eval.json").is_file())
    return runs


def process(paths, make, variant, out_dir=None, log=None, frames=True, force=False):
    """Propose for every run without a proposal yet. -> number of proposals written."""
    done = 0
    for run in find_runs(paths):
        out = Path(out_dir) / run.name if out_dir else run
        if (out / "reward_proposal.json").exists() and not force:
            continue
        propose(run, make(), variant, out, log, frames)
        done += 1
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run folders, or folders containing runs")
    ap.add_argument("--variant", choices=sorted(DEFAULTS), default="hybrid")
    ap.add_argument("--out-dir", help="write proposals to <out-dir>/<run name> instead of into each run")
    ap.add_argument("--log", default=config.OUTPUT_DIR / "agent_log.jsonl", help="proposal history (JSON lines)")
    ap.add_argument("--watch", type=float, metavar="SECONDS", help="keep checking for new runs this often")
    ap.add_argument("--no-frames", action="store_true", help="send numbers only, no rollout images")
    ap.add_argument("--dry-run", action="store_true", help="no API call: bundled fixture response")
    ap.add_argument("--fixture", help="no API call: responses from this recorded file")
    ap.add_argument("--record", help="save live exchanges here (they work as --fixture files)")
    ap.add_argument("--model")
    ap.add_argument("--force", action="store_true", help="redo runs that already have a proposal")
    a = ap.parse_args(argv)
    make = lambda: make_client(a.dry_run, a.fixture, "reward_proposal_remote_latent.json", a.model, a.record)  # noqa: E731
    kw = dict(variant=a.variant, out_dir=a.out_dir, log=a.log, frames=not a.no_frames)
    n = process(a.runs, make, force=a.force, **kw)
    if not a.watch:
        if not n:
            print("nothing to do: every run already has reward_proposal.json (--force to redo)")
        return
    print(f"watching {', '.join(a.runs)} every {a.watch:g} s (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(a.watch)
            process(a.runs, make, **kw)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
