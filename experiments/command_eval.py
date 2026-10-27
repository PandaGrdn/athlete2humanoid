"""Commanded targets: does a policy squat as deep and jump as high as it is asked, also for targets
the clip never showed? Runs in mjlab's virtualenv; CPU is enough for a few envs per command.

    A2H_TASK_SPEC=task_spec.json $MJLAB_PY experiments/command_eval.py \\
        --motion run/policy_taskcmd/motion.npz taskcmd=run/policy_taskcmd/model_1499.pt \\
        taskphase=run/policy_taskphase/model_1499.pt --out command_eval.json

Every policy acts in the A2H-TaskCmd env (play config: no pushes or observation noise, starts at
the beginning of the clip). Each env gets one commanded (depth, clearance[, reps]) from a grid
around and outside the clip's values; a policy without the command input (TaskPhase) is loaded
through policy_weights.py, so its command columns are zero and it shows what ignoring the command
does. Per command and policy: task success under the commanded spec (skill.task.command_spec:
all stages, no fall; cmd_success also needs every jump in the quality band, q >= 0.7), and the
achieved squat depth / jump clearance per rep next to the commanded ones. Summary: success at the
clip's own targets vs at novel ones, mean |achieved - commanded|, and the slope of achieved on
commanded (1 = follows the command, 0 = ignores it).

--check-warm-start CKPT: a TaskPhase checkpoint loaded into TaskCmd must act exactly as it does in
TaskPhase (the command inputs are appended and get zero weights): prints the max action difference
over a rollout with random commands.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skill import task as skill_task  # noqa: E402

TASK, PHASE_TASK = "A2H-TaskCmd-Flat-Unitree-G1", "A2H-TaskPhase-Flat-Unitree-G1"
TASK_SPEC_ENV = "A2H_TASK_SPEC"  # mjlab_tasks.TASK_SPEC_ENV
STEP_HZ = 50.0
Q_OK = 0.7  # jump quality that counts as hitting the commanded height (skill/commands.py q_success)
NOVEL = 0.02  # m from the clip's own target: a command the clip never showed


def default_grid(spec, depths=None, clearances=None, reps=None):
    """Commands (dicts) to evaluate: the clip's own targets plus targets outside them, inside the
    G1 ranges. A squat-only clip gets no clearance command."""
    nom = skill_task.nominal_command(spec)
    lo_d, hi_d = skill_task.COMMAND_RANGES["depth"]
    lo_c, hi_c = skill_task.COMMAND_RANGES["clearance"]
    depths = depths or sorted({round(float(np.clip(d, lo_d, hi_d)), 3) for d in (nom["depth"], 0.14, 0.19, 0.30)})
    clearances = (clearances or sorted({round(nom["clearance"], 3), 0.12, 0.20})) if nom["clearance"] > 0 else [None]
    reps = reps or [None]
    return [{"depth": float(d), "clearance": None if c is None else float(np.clip(c, lo_c, hi_c)),
             "reps": r} for d in depths for c in clearances for r in reps]


def score(spec, pos, commands, fps=STEP_HZ):
    """pos: per env (T, bodies, 3) rollouts; commands: per env command dicts -> per command rows
    and a summary. Pure numpy/torch (unit-tested without mjlab)."""
    nom = skill_task.nominal_command(spec)
    rows = {}
    for p, c in zip(pos, commands):
        r = skill_task.evaluate(skill_task.command_spec(spec, **c), p, fps)
        q_ok = all(q >= Q_OK for q in r["jump_quality"]) and len(r["jump_quality"]) == sum(
            s["kind"] == "jump" for s in skill_task.command_spec(spec, **c)["stages"])
        key = json.dumps(c, sort_keys=True)
        row = rows.setdefault(key, {"command": c, "n": 0, "success": 0, "cmd_success": 0, "falls": 0,
                                    "fell_at_s": [], "stages": [], "depths": [], "clearances": []})
        row["n"] += 1
        row["success"] += r["success"]
        row["cmd_success"] += r["success"] and q_ok
        row["falls"] += r["fell_at_s"] is not None
        row["fell_at_s"] += [r["fell_at_s"]] if r["fell_at_s"] is not None else []
        row["stages"].append(f"{r['stages_completed']}/{r['stages_total']}")
        row["depths"] += r["metrics"]["squat_depths"]
        row["clearances"] += r["metrics"]["jump_clearances"]
    out, pairs = [], {"depth": [], "clearance": []}
    for row in rows.values():
        c = row["command"]
        res = {"command": c, "n": row["n"], "success": row["success"] / row["n"],
               "cmd_success": row["cmd_success"] / row["n"], "fall_rate": row["falls"] / row["n"],
               "fell_at_s": row["fell_at_s"], "stages": row["stages"],
               "novel": bool(abs(c["depth"] - nom["depth"]) > NOVEL or
                             (c["clearance"] is not None and abs(c["clearance"] - nom["clearance"]) > NOVEL)
                             or (c["reps"] is not None and c["reps"] != nom["reps"]))}
        for k, vals in (("depth", row["depths"]), ("clearance", row["clearances"])):
            if c[k] is not None and vals:
                res[f"achieved_{k}"] = round(float(np.mean(vals)), 3)
                res[f"{k}_error"] = round(float(np.mean(vals)) - c[k], 3)
                pairs[k] += [(c[k], v) for v in vals]
        out.append(res)

    def mean(xs):
        return round(float(np.mean(xs)), 3) if xs else None

    summary = {}
    for split, sel in (("clip", [r for r in out if not r["novel"]]), ("novel", [r for r in out if r["novel"]])):
        summary[split] = {"commands": len(sel), "success": mean([r["success"] for r in sel]),
                          "cmd_success": mean([r["cmd_success"] for r in sel]),
                          "fall_rate": mean([r["fall_rate"] for r in sel])}
    for k, pr in pairs.items():
        if len({c for c, _ in pr}) >= 2:
            x, y = np.array(pr).T
            summary[f"{k}_slope"] = round(float(np.polyfit(x, y, 1)[0]), 3)
            summary[f"{k}_mae"] = round(float(np.abs(y - x).mean()), 3)
    return out, summary


# ---------------------------------------------------------------- mjlab (its virtualenv)

def make_env(task, motion, num_envs, device, seed=0):
    import mjlab_tasks  # noqa: F401  (registers the A2H tasks)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    cfg = load_env_cfg(task, play=True)
    cfg.commands["motion"].motion_file = str(motion)
    cfg.commands["motion"].debug_vis = False
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}  # falls: ours
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    return env, RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)


def load_policy(venv, task, checkpoint, device):
    """A runner for `task` with `checkpoint`'s weights, adapted by policy_weights.py when the
    checkpoint was trained with fewer inputs (e.g. TaskPhase -> TaskCmd)."""
    from dataclasses import asdict

    import torch

    import policy_weights
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(venv, asdict(load_rl_cfg(task)), device=device)
    report = policy_weights.init_from(runner.alg, torch.load(checkpoint, map_location=device, weights_only=False))
    grown = sorted({h for h in report["actor"].values() if h != "copied"})
    if any(h.startswith("fresh") for h in grown):  # e.g. an A2H-Task checkpoint (it sees the retarget)
        raise SystemExit(f"{checkpoint}: its actor inputs don't fit {task} ({grown}); only TaskPhase/TaskCmd weights do")
    return runner.get_inference_policy(device=device), grown


def rollout(env, venv, policy, commands, steps=None):
    """Every env under its command from the start of the clip -> (n, T, bodies, 3) body positions."""
    import torch
    m, sampler, cmd = env.a2h_task, env.a2h_cmd, env.command_manager.get_term("motion")
    dev = env.device
    t = lambda k, dt: None if commands[0][k] is None else torch.tensor([c[k] for c in commands], dtype=dt, device=dev)
    sampler.fix(t("depth", torch.float32), t("clearance", torch.float32), t("reps", torch.long))
    T = int(cmd.motion.time_step_total) - 1 if steps is None else steps
    pos = torch.zeros(env.num_envs, T + 1, len(cmd.cfg.body_names), 3)
    with torch.inference_mode():
        obs, _ = venv.reset()
        origins = env.scene.env_origins[:, None].cpu()
        pos[:, 0] = cmd.robot_body_pos_w.cpu() - origins
        for i in range(T):
            obs, *_ = venv.step(policy(obs))
            pos[:, i + 1] = cmd.robot_body_pos_w.cpu() - origins
    assert torch.allclose(m.command[:, 0].cpu(), torch.tensor([c["depth"] for c in commands])), "commands not applied"
    return pos.numpy()


def check_warm_start(phase_ckpt, motion, n=8, steps=100, device="cpu"):
    """Max |action difference| between a TaskPhase checkpoint run in TaskPhase and the same weights
    loaded into TaskCmd, on the same observations, with random commands (should be exactly 0)."""
    import torch
    from tensordict import TensorDict
    env_c, venv_c = make_env(TASK, motion, n, device)
    env_p, venv_p = make_env(PHASE_TASK, motion, n, device)
    pol_c, grown = load_policy(venv_c, TASK, phase_ckpt, device)
    pol_p, _ = load_policy(venv_p, PHASE_TASK, phase_ckpt, device)
    k = len(skill_task.COMMANDS)
    env_c.a2h_cmd.cfg["explore"] = 1.0  # full ranges: the command columns are far from zero
    worst = 0.0
    with torch.inference_mode():
        obs, _ = venv_c.reset()
        cmd_cols = obs["actor"][:, -k:].abs().max().item()
        for _ in range(steps):
            a = pol_c(obs)
            trimmed = TensorDict({g: obs[g][:, :-k] for g in obs.keys()}, batch_size=obs.batch_size)
            worst = max(worst, (a - pol_p(trimmed)).abs().max().item())
            obs, *_ = venv_c.step(a)
    return {"max_action_diff": worst, "grown": grown, "max_command_input": cmd_cols,
            "actor_inputs": {"taskcmd": int(obs["actor"].shape[1]), "taskphase": int(obs["actor"].shape[1]) - k}}


def main(a):
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    spec = json.loads(Path(os.environ[TASK_SPEC_ENV]).read_text())
    if a.check_warm_start:
        r = check_warm_start(a.check_warm_start, a.motion, device=a.device)
        print(json.dumps(r, indent=1))
        if r["max_action_diff"] != 0.0:
            raise SystemExit("warm start changed the policy's actions")
        return
    pols = dict(s.split("=", 1) for s in a.policies)
    cmds = default_grid(spec, a.depths, a.clearances, a.reps)
    per_env = [c for c in cmds for _ in range(a.per_command)]
    env, venv = make_env(TASK, a.motion, len(per_env), a.device, a.seed)
    result = {"spec": spec.get("name"), "nominal": skill_task.nominal_command(spec), "per_command": a.per_command,
              "policies": {}}
    for name, ckpt in pols.items():
        t0 = time.time()
        policy, grown = load_policy(venv, TASK, ckpt, a.device)
        pos = rollout(env, venv, policy, per_env, a.steps)
        rows, summary = score(spec, pos, per_env)
        result["policies"][name] = {"checkpoint": str(ckpt), "adapted": grown, "summary": summary, "commands": rows}
        print(f"{name} ({time.time() - t0:.0f} s): {json.dumps(summary)}", flush=True)
        for r in rows:
            print("   ", {k: v for k, v in r.items() if k != "n"})
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("policies", nargs="*", help="name=model_*.pt")
    ap.add_argument("--motion", required=True, help="the clip's motion.npz (mjlab tracking format)")
    ap.add_argument("--depths", type=float, nargs="*")
    ap.add_argument("--clearances", type=float, nargs="*")
    ap.add_argument("--reps", type=int, nargs="*")
    ap.add_argument("--per-command", type=int, default=2, help="envs per command")
    ap.add_argument("--steps", type=int, help="default: the whole clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    ap.add_argument("--check-warm-start", metavar="TASKPHASE_CKPT")
    main(ap.parse_args())
