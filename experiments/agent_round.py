"""One round of muse/agent_loop.py on the GPU: train each arm a short budget from its last checkpoint,
then score every arm the same way.

    python experiments/agent_round.py --inputs <in> --out <out> --target jumping_jacks --iters 300
    python experiments/agent_round.py --inputs <in> --out <out> --target jumping_jacks --eval-only

<in> (packed by muse/agent_loop.py; scripts/remote_run.py agentround):
    clips/<clip>/          the run's processed clips (the target's motion.csv, every clip's video_latent.npz)
    reward.pt              the latent reward bundle the policies were trained with (style term)
    task_spec.json         the task, fixed: every round is scored on it, whatever an arm trains on
    arms/<arm>/model_<n>.pt          where the arm's policy is now
    arms/<arm>/reward_config.json    its reward settings (mjlab_tasks A2H_REWARD_CONFIG), optional
    arms/<arm>/task_weights.json     its stage machine's weights, written into the spec it trains on, optional
    arms/<arm>/{motion.npz, policy_qpos.npy, policy.mp4, policy_metrics.json}  --eval-only: its rollout

Training resumes each arm's checkpoint (actor, critic, optimiser) for --iters more PPO iterations of
A2H-Task-Flat-Unitree-G1, the arms in parallel on one GPU. Scoring, identical for every arm and round:
experiments/robustness_eval.py on the fixed spec (all arms batched in one environment: same conditions
and seeds), --reps envs per condition: clean, pushes, start noise; then skill/task.py evaluate() of the
arm's deterministic rollout (which stage it gave up on, and the feature that kept it from passing), and
the last training iteration's reward terms. -> <out>/round.json and <out>/results.tar.gz.
"""

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK = "A2H-Task-Flat-Unitree-G1"
ARM_FILES = ("policy.mp4", "policy_qpos.npy", "policy_metrics.json", "weights.json", "reward_config.json",
             "train_spec.json", "motion.npz")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def newest(paths):
    paths = list(paths)
    return max(paths, key=lambda p: int(Path(p).stem.split("_")[1])) if paths else None


def policy_dir(out, arm):
    return out / f"policy_task_{arm}"  # experiments/latent_policy.TASKS names it, for robustness_eval.py


def train_cmd(inp, out, arm, a):
    """Seed the arm's checkpoint where train_motion.py resumes from, write its training spec -> the command."""
    src, d = inp / "arms" / arm, policy_dir(out, arm)
    ckpt = newest(src.glob("model_*.pt"))
    n = int(ckpt.stem.split("_")[1])
    seed = d / "logs" / "resumed" / "from" / ckpt.name
    seed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt, seed)
    spec = json.loads((out / "task_spec.json").read_text())
    if (src / "task_weights.json").exists():
        spec["weights"] = {**spec["weights"], **json.loads((src / "task_weights.json").read_text())}
    (d / "train_spec.json").write_text(json.dumps(spec, indent=1))
    cmd = [sys.executable, ROOT / "train_motion.py", "--csv", inp / "clips" / a.target / "motion.csv", "--fps", 30,
           "--out", d, "--iters", n + a.iters, "--num-envs", a.num_envs, "--task", TASK, "--bundle", out / "reward.pt",
           "--task-spec", d / "train_spec.json", "--importance", out / "importance.json", "--save-interval", a.iters]
    if (src / "reward_config.json").exists():
        rc = json.loads((src / "reward_config.json").read_text())
        rc.pop("iterations", None)  # train_motion.py would take it in place of --iters
        (d / "arm_reward_config.json").write_text(json.dumps(rc, indent=1))
        cmd += ["--reward-config", d / "arm_reward_config.json"]
    return cmd, n


def last_iteration(train_log):
    """The last 'Learning iteration' block of an mjlab training log -> {iteration, mean_reward, ...,
    rewards: {term: per-episode value}, terminations: {term: share}}."""
    text = Path(train_log).read_text(errors="replace") if Path(train_log).exists() else ""
    blocks = re.split(r"Learning iteration (\d+)/\d+", text)
    if len(blocks) < 3:
        return None
    it, body = int(blocks[-2]), blocks[-1]
    num = lambda k: (lambda m: float(m.group(1)) if m else None)(re.search(rf"{k}:\s*(-?[\d.]+)", body))  # noqa: E731
    out = {"iteration": it, "mean_reward": num("Mean reward"), "mean_episode_length": num("Mean episode length"),
           "action_std": num("Mean action std")}
    out["rewards"] = {k: float(v) for k, v in re.findall(r"Episode_Reward/(\w+):\s*(-?[\d.]+)", body)}
    out["terminations"] = {k: float(v) for k, v in re.findall(r"Episode_Termination/(\w+):\s*(-?[\d.]+)", body)}
    return out


def rollout_check(spec, qpos_file, g1):
    """skill/task.py evaluate() on the arm's deterministic rollout (50 Hz qpos)."""
    from experiments.latent_policy import to_30fps
    from latent import distill
    from skill import task as skill_task
    sem = skill_task.evaluate(spec, distill.body_positions(g1, to_30fps(np.load(qpos_file))), 30.0)
    return {k: sem[k] for k in ("stages_completed", "stages_total", "success", "fell_at_s", "next_stage",
                                "completed_at_s", "abandoned")}


ROBUST_KEYS = ("task_success", "stages_completed", "stages_frac", "fall_rate", "survival", "vjepa_aligned", "abandoned")


def robust_rows(robust, arms):
    """robust.json -> {arm: {"<suite> <level>": the condition's numbers}}."""
    rows = {arm: {} for arm in arms}
    for c in robust["conditions"]:
        for arm in arms:
            p = c["policies"].get(f"task_{arm}")
            if p:
                rows[arm][f"{c['suite']} {c['level']}"] = {k: p.get(k) for k in ROBUST_KEYS}
    return rows


def main(a):
    import render
    from skill import keypose
    inp, out = Path(a.inputs).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arms = sorted(d.name for d in (inp / "arms").iterdir() if d.is_dir())
    spec = json.loads((inp / "task_spec.json").read_text())
    (out / "task_spec.json").write_text(json.dumps(spec, indent=1))
    shutil.copy(inp / "reward.pt", out / "reward.pt")
    qpos = np.load(inp / "clips" / a.target / "qpos.npy")
    (out / "importance.json").write_text(json.dumps(keypose.importance(spec, len(qpos), 30.0)))  # as latent_policy.py
    if not (out / "clips").exists():
        (out / "clips").symlink_to(inp / "clips")
    t0, timing = time.time(), {}

    # 1. train every arm in parallel (one GPU fits three mjlab trainings)
    start_iter = {}
    if not a.eval_only:
        procs = {}
        for arm in arms:
            cmd, start_iter[arm] = train_cmd(inp, out, arm, a)
            f = open(policy_dir(out, arm) / "train.log", "w")
            procs[arm] = subprocess.Popen(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
            log(f"training {arm} from iteration {start_iter[arm]} for {a.iters}")
        for arm, p in procs.items():
            if p.wait() != 0:
                raise SystemExit(f"{arm}: train_motion.py exited with {p.returncode}; see {policy_dir(out, arm)}/train.log")
        log("training done")
    else:
        for arm in arms:
            d = policy_dir(out, arm)
            d.mkdir(exist_ok=True)
            for f in (inp / "arms" / arm).iterdir():
                shutil.copy(f, d / f.name)
    timing["train_s"] = round(time.time() - t0)

    # 2. the same scoring for every arm: robustness suites on the fixed spec, batched in one env
    t1 = time.time()
    names = [f"task_{arm}" for arm in arms]
    cmd = [sys.executable, ROOT / "experiments" / "robustness_eval.py", "--runs", out, "--out", out / "robust",
           "--target", a.target, "--policies", *names, "--reps", a.reps, "--score", a.score,
           "--push", *a.push, "--init", *a.init_noise, "--dynamics", 0, "--speed", "--tempo", "--phase-push",
           "--starts", "--no-noref"]
    subprocess.run(list(map(str, cmd)), check=True)
    robust = json.loads((out / "robust" / "robust.json").read_text())
    timing["eval_s"] = round(time.time() - t1)

    # 3. per arm: its rollout's stage check, its training log, its checkpoint
    g1 = render.load_g1()[0]
    rows = robust_rows(robust, arms)
    result = {"target": a.target, "iters": 0 if a.eval_only else a.iters, "reps": a.reps, "arms": {}}
    for arm in arms:
        d = policy_dir(out, arm)
        ckpt = newest([*d.glob("model_*.pt"), *d.glob("logs/*/*/model_*.pt")])
        result["arms"][arm] = {
            "checkpoint": ckpt.name, "started_at_iteration": start_iter.get(arm),
            "robust": rows[arm], "rollout": rollout_check(spec, d / "policy_qpos.npy", g1),
            "rollout_vjepa_aligned": (robust.get("reference_check") or {}).get(f"task_{arm}", {}).get("aligned"),
            "train": last_iteration(d / "train.log") if not a.eval_only else None,
            "policy_metrics": json.loads((d / "policy_metrics.json").read_text()) if (d / "policy_metrics.json").exists() else None,
        }
        log(f"{arm}: " + "; ".join(f"{k} success {v['task_success']:.0%}, stages {v['stages_completed']:.2f}, "
                                    f"falls {v['fall_rate']:.0%}" for k, v in rows[arm].items()))
    result["seconds"] = timing | {"total_s": round(time.time() - t0)}
    (out / "round.json").write_text(json.dumps(result, indent=1))
    with tarfile.open(out / "results.tar.gz", "w:gz") as tar:
        for f in ("round.json", "task_spec.json", "robust/robust.json", "robust/robust_summary.md"):
            if (out / f).exists():
                tar.add(out / f, f)
        for arm in arms:
            d = policy_dir(out, arm)
            for f in ARM_FILES:
                if (d / f).exists():
                    tar.add(d / f, f"{d.name}/{f}")
            ckpt = newest([*d.glob("model_*.pt"), *d.glob("logs/*/*/model_*.pt")])
            tar.add(ckpt, f"{d.name}/{ckpt.name}")
            if (d / "train.log").exists():  # the last iterations only: mjlab logs are tens of MB
                tail = (d / "train.log").read_text(errors="replace")[-200_000:]
                info = tarfile.TarInfo(f"{d.name}/train_tail.log")
                info.size = len(tail.encode())
                tar.addfile(info, io.BytesIO(tail.encode()))
    log(f"done: {out / 'round.json'} ({result['seconds']['total_s']} s)")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--iters", type=int, default=300, help="PPO iterations per arm this round")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--eval-only", action="store_true", help="score the arms' policies as given (round 0)")
    ap.add_argument("--reps", type=int, default=64, help="envs per arm per condition")
    ap.add_argument("--score", type=int, default=2, help="rollouts per arm per condition scored with V-JEPA")
    ap.add_argument("--push", type=float, nargs="+", default=[0.75], help="push strengths (m/s)")
    ap.add_argument("--init-noise", type=float, nargs="+", default=[2.0], help="start noise, x training's")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(parse())
