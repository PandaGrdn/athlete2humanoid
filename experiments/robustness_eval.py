"""Does a policy learn the action, or copy the clip? Trained policies, pushed off the clip.

    python experiments/robustness_eval.py --runs outputs/remote_latent --out outputs/robust

--runs is a latent run's results folder: policy_<name>/model_*.pt + motion.npz per policy, and
clips/<clip>/video_latent.npz (the human clips' V-JEPA latents). Two stages, split like
train_motion.py: the rollouts run in mjlab's virtualenv (A2H_MJLAB_PYTHON; this file re-runs
itself there with --mjlab), the V-JEPA scoring in the main env.

1. mjlab: every policy, batched in one environment (each policy drives its own slice of envs),
   under each condition of five suites:
   - clean:    the clip as trained, nominal friction and mass
   - push:     random horizontal shoves (root velocity kicks, every 1-3 s as in training) of
               increasing strength; training pushes reach ~0.7 m/s
   - init:     start at a random phase of the first half of the clip, from a state perturbed
               like training's reference-state init but 1x / 2x / 4x as far
   - dynamics: per-env foot friction and per-body mass randomisation, three widening ranges
               (level 1 = training's friction range)
   - speed:    the reference resampled 0.8x / 1.2x as fast (an unseen command sequence)
   All policies run in the stock tracking task's environment: the A2H tasks differ from it in
   rewards and terminations, and in the actor's observations for the phase task (which gets its
   own observation group here; everything else is checked equal at start-up). The eval replaces
   terminations with one task-independent fall test, so the relaxed tasks can't look better by
   just not resetting.
   Fall: torso below --fall-z or tilted past --fall-tilt (latched; no reset, the env stays down).
2. V-JEPA: a few rollouts per policy and condition re-rendered as silhouettes and compared with
   the human video exactly as latent_policy.py's evaluation does (aligned cosine and DTW, same
   human/robot domain means), so the numbers are comparable with its eval.json. Speed-suite
   rollouts are time-resampled back to the clip's tempo first; init rollouts are compared with
   the part of the video from their start phase on. As a check, the run's own clean rollouts
   (policy_*/policy_qpos.npy) are re-scored next to their eval.json values.

The claim being tested: tracking is best on the clean clip, but hybrid/latent hold up better
(fewer falls, smaller drop in V-JEPA score) as conditions move away from it.

robust.json:
{
  "meta": {
    "target": clip name, "view_bucket_deg": render azimuth, "policies": {name: {"checkpoint", "task"}},
    "env_task": task whose env all policies ran in, "reps": envs per policy per condition,
    "vjepa_rollouts": rendered rollouts per policy per condition, "step_hz": 50, "clip_steps",
    "fall_rule": {"torso_z_m", "tilt_rad", "ref_torso_z_min_m", "ref_tilt_max_rad"},
    "seconds": {"mjlab", "vjepa"}
  },
  "reference_check": {policy: {"aligned", "dtw", "eval_json_aligned", "eval_json_dtw"}},
  "conditions": [{
    "suite": "clean" | "push" | "init" | "dynamics" | "speed", "level": label,
    "params": {"push_mps", "init_scale", "phase_max", "friction", "mass_scale", "speed"},
    "policies": {policy: {
      "fall_rate":      fraction of envs that fell before the reference ended,
      "survival":       mean fraction of the clip completed upright,
      "torso_err_m":    mean torso (anchor) position error vs the reference while upright (m),
      "joint_err_rad":  mean |joint - reference joint| while upright,
      "vjepa_aligned":  mean over the rendered rollouts (null if none),
      "vjepa_dtw":      same, DTW,
      "vjepa_each":     [[aligned, dtw], ...] per rendered rollout
    }}
  }, ...],
  "summary": {policy: {suite: {"fall_rate", "torso_err_m", "vjepa_aligned",
                               "vjepa_retained": suite vjepa_aligned / clean vjepa_aligned}}}
}

Cost on an A100 (Colab), estimated (the full suite ran locally on CPU at 4 envs/policy): setup ~5 min;
one env build + kernel compile (~1-2 min); 13 conditions, ~6,600 steps at 3 x 64 envs (~2 min);
V-JEPA load + bank latents (~2 min); 78 rendered rollouts + 3 checks (8 render workers, ~3 min).
About 9 min of job, ~15 min of VM (~1.3 compute units). --smoke: tiny, for plumbing.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_TASK = "Mjlab-Tracking-Flat-Unitree-G1"
STEP_HZ = 50
PUSH_EVERY_S = (1.0, 3.0)  # as mjlab's training push_robot event
DYNAMICS = (((0.3, 1.2), (0.9, 1.1)),  # (foot friction range, per-body mass scale range)
            ((0.2, 1.6), (0.8, 1.2)),
            ((0.1, 2.0), (0.7, 1.3)))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def conditions(a):
    """[(suite, level label, params)]."""
    c = {"push_mps": 0.0, "init_scale": 0.0, "phase_max": 0.0, "friction": None, "mass_scale": None, "speed": 1.0}
    out = [("clean", "nominal", c)]
    out += [("push", f"{v:g} m/s", c | {"push_mps": v}) for v in a.push]
    out += [("init", f"x{s:g}", c | {"init_scale": s, "phase_max": 0.5}) for s in a.init]
    out += [("dynamics", f"level {i + 1}", c | {"friction": f, "mass_scale": m})
            for i, (f, m) in enumerate(DYNAMICS[:a.dynamics])]
    out += [("speed", f"{s:g}x", c | {"speed": s}) for s in a.speed]
    return out


# ---------------------------------------------------------------- stage 1: mjlab (its venv)

def resample_motion(src, dst, speed):
    """motion.npz played `speed` times as fast: same 50 Hz, fewer/more frames; velocities scaled."""
    d = np.load(src)
    T = len(d["joint_pos"])
    n = int(round((T - 1) / speed)) + 1
    t = np.linspace(0, T - 1, n)
    i0 = np.minimum(np.floor(t).astype(int), T - 2)
    w = t - i0

    def lerp(x):
        ww = w.reshape(-1, *[1] * (x.ndim - 1))
        return x[i0] * (1 - ww) + x[i0 + 1] * ww

    q0, q1 = d["body_quat_w"][i0], d["body_quat_w"][i0 + 1]
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)  # same hemisphere, then nlerp
    quat = q0 * (1 - w[:, None, None]) + q1 * w[:, None, None]
    out = {"fps": d["fps"], "joint_pos": lerp(d["joint_pos"]), "joint_vel": lerp(d["joint_vel"]) * speed,
           "body_pos_w": lerp(d["body_pos_w"]), "body_quat_w": quat / np.linalg.norm(quat, axis=-1, keepdims=True),
           "body_lin_vel_w": lerp(d["body_lin_vel_w"]) * speed, "body_ang_vel_w": lerp(d["body_ang_vel_w"]) * speed}
    np.savez(dst, **{k: v.astype(np.float32) for k, v in out.items()})
    return dst


def mjlab_stage(a, out):
    import warnings
    from dataclasses import asdict

    import torch

    import mjlab.tasks  # noqa: F401  (registers the tasks)
    import mjlab_tasks  # noqa: F401  (and ours: only their cfgs are read, to check they match)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.envs.mdp import dr
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.tracking.mdp.commands import MotionLoader

    warnings.filterwarnings("ignore", message="dr.body_mass")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pols = json.loads((out / "policies.json").read_text())
    names = list(pols)
    motion_file = Path(pols[names[0]]["motion"])
    for n in names[1:]:  # all policies were trained on the same clip
        assert np.allclose(np.load(pols[n]["motion"])["joint_pos"], np.load(motion_file)["joint_pos"]), n

    cfg = load_env_cfg(ENV_TASK, play=True)
    groups = {}  # policy -> its actor observation group
    for n in names:  # the A2H tasks must act the same way; a different actor input gets its own group
        c = load_env_cfg(pols[n]["task"], play=True)
        assert c.actions["joint_pos"].scale == cfg.actions["joint_pos"].scale, n
        assert list(c.observations["critic"].terms) == list(cfg.observations["critic"].terms), n
        groups[n] = "actor"
        if list(c.observations["actor"].terms) != list(cfg.observations["actor"].terms):
            groups[n] = f"actor_{n}"
            cfg.observations[groups[n]] = c.observations["actor"]
    train_cmd = load_env_cfg(ENV_TASK).commands["motion"]  # training's reference-state-init ranges
    cfg.commands["motion"].motion_file = str(motion_file)
    cfg.commands["motion"].debug_vis = False
    cfg.scene.num_envs = a.reps * len(names)
    cfg.seed = a.seed
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}  # falls: ours
    # Expands body_mass per env (a no-op at start-up); the conditions set it per run.
    cfg.events["eval_mass"] = EventTermCfg(mode="startup", func=dr.body_mass, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)), "operation": "scale", "ranges": (1.0, 1.0)})

    t0 = time.time()
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent_cfg = load_rl_cfg(ENV_TASK)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policies = []
    for n in names:
        rl = load_rl_cfg(pols[n]["task"])
        rl.obs_groups = {**rl.obs_groups, "actor": (groups[n],)}
        runner = (load_runner_cls(pols[n]["task"]) or MjlabOnPolicyRunner)(venv, asdict(rl), device=device)
        runner.load(pols[n]["checkpoint"], load_cfg={"actor": True}, strict=True, map_location=device)
        policies.append(runner.get_inference_policy(device=device))
    log(f"env with {cfg.scene.num_envs} envs + {len(names)} policies ready ({time.time() - t0:.0f} s)")

    cmd, robot = env.command_manager.get_term("motion"), env.scene["robot"]
    friction = env.event_manager.get_term_cfg("foot_friction")
    mass = env.event_manager.get_term_cfg("eval_mass")
    slices = [slice(i * a.reps, (i + 1) * a.reps) for i in range(len(names))]
    keep = torch.cat([torch.arange(s.start, s.start + a.score) for s in slices]).to(device)

    def tilt(quat):  # angle between the body's z axis and world up (quat wxyz)
        return torch.acos((1 - 2 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)).clamp(-1, 1))

    ref_z = cmd.motion.body_pos_w[:, cmd.motion_anchor_body_index, 2]
    ref_tilt = tilt(cmd.motion.body_quat_w[:, cmd.motion_anchor_body_index])
    meta = {"env_task": ENV_TASK, "reps": a.reps, "vjepa_rollouts": a.score, "step_hz": STEP_HZ,
            "clip_steps": int(cmd.motion.time_step_total),
            "fall_rule": {"torso_z_m": a.fall_z, "tilt_rad": a.fall_tilt, "ref_torso_z_min_m": ref_z.min().item(),
                          "ref_tilt_max_rad": ref_tilt.max().item()}}
    assert ref_z.min() > a.fall_z + 0.1 and ref_tilt.max() < a.fall_tilt - 0.2, f"fall rule fires on the reference: {meta}"

    results, rollouts = [], {}
    with torch.inference_mode():  # also the resets: tensors made inside can't be updated outside
        for ci, (suite, level, p) in enumerate(conditions(a)):
            tc = time.time()
            torch.manual_seed(a.seed + ci)
            mf = motion_file if p["speed"] == 1 else resample_motion(motion_file, out / f"motion_x{p['speed']:g}.npz", p["speed"])
            cmd.motion = MotionLoader(str(mf), cmd.body_indexes, device=env.device)
            T = cmd.motion.time_step_total
            s = p["init_scale"]
            cmd.cfg.pose_range = {k: (lo * s, hi * s) for k, (lo, hi) in train_cmd.pose_range.items()}
            cmd.cfg.velocity_range = {k: (lo * s, hi * s) for k, (lo, hi) in train_cmd.velocity_range.items()}
            cmd.cfg.joint_position_range = tuple(x * s for x in train_cmd.joint_position_range)
            max_start = int(p["phase_max"] * (T - 1))
            cmd.cfg.sampling_mode = "uniform"  # start phase in [0, max_start]
            cmd._uniform_sampling = lambda ids: cmd.time_steps.__setitem__(
                ids, torch.randint(0, max_start + 1, (len(ids),), device=env.device))
            f_rng = p["friction"] or (1.0, 1.0)
            dr.geom_friction(env, None, **friction.params | {"ranges": f_rng, "operation": "abs" if p["friction"] else "scale"})
            dr.body_mass(env, None, **mass.params | {"ranges": p["mass_scale"] or (1.0, 1.0)})
            env.sim.recompute_constants(RecomputeLevel.set_const)

            obs, _ = venv.reset()
            start = cmd.time_steps.clone()
            length = (T - 1) - start  # steps until the reference ends (then it wraps: robot teleported)
            steps = int(length.max()) if a.max_steps is None else min(int(length.max()), a.max_steps)
            length = length.clamp(max=steps)
            N = env.num_envs
            fallen = torch.zeros(N, dtype=torch.bool, device=device)
            upright = torch.zeros(N, device=device)
            torso_err, joint_err = torch.zeros(N, device=device), torch.zeros(N, device=device)
            next_push = torch.empty(N, device=device).uniform_(*PUSH_EVERY_S) * STEP_HZ
            qpos = torch.zeros(len(keep), steps, 7 + robot.data.joint_pos.shape[1], device=device)
            for t in range(steps):
                if p["push_mps"]:
                    ids = ((t >= next_push) & ~fallen).nonzero().squeeze(1)
                    if len(ids):
                        ang = torch.rand(len(ids), device=device) * 2 * torch.pi
                        vel = robot.data.root_link_vel_w[ids].clone()
                        vel[:, 0] += p["push_mps"] * torch.cos(ang)
                        vel[:, 1] += p["push_mps"] * torch.sin(ang)
                        robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
                        next_push[ids] += torch.empty(len(ids), device=device).uniform_(*PUSH_EVERY_S) * STEP_HZ
                obs, _, _, _ = venv.step(torch.cat([pi(obs[sl]) for pi, sl in zip(policies, slices)]))
                live = (t < length) & ~fallen
                z = cmd.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
                fallen |= live & ((z < a.fall_z) | (tilt(cmd.robot_anchor_quat_w) > a.fall_tilt))
                live &= ~fallen
                upright += live
                torso_err += live * (cmd.robot_anchor_pos_w - cmd.anchor_pos_w).norm(dim=-1)
                joint_err += live * (cmd.robot_joint_pos - cmd.joint_pos).abs().mean(-1)
                qpos[:, t] = torch.cat([robot.data.root_link_pos_w[keep] - env.scene.env_origins[keep],
                                        robot.data.root_link_quat_w[keep], robot.data.joint_pos[keep]], -1)

            res = {}
            for name, sl in zip(names, slices):
                u = upright[sl].clamp(min=1)
                res[name] = {"fall_rate": fallen[sl].float().mean().item(),
                             "survival": (upright[sl] / length[sl].clamp(min=1)).mean().item(),
                             "torso_err_m": (torso_err[sl] / u).mean().item(),
                             "joint_err_rad": (joint_err[sl] / u).mean().item()}
            for k, e in enumerate(keep.tolist()):
                name = names[e // a.reps]
                rollouts[f"{ci}/{name}/{e % a.reps}"] = qpos[k, :int(length[e])].cpu().numpy()
                rollouts[f"{ci}/{name}/{e % a.reps}/start"] = np.array(int(start[e]))
            results.append({"suite": suite, "level": level, "params": p, "policies": res})
            log(f"{suite:8s} {level:9s} ({steps} steps, {time.time() - tc:.0f} s): " + "  ".join(
                f"{n} fall {r['fall_rate']:.2f} err {r['torso_err_m']:.3f}" for n, r in res.items()))

    meta["seconds"] = {"mjlab": time.time() - t0}
    np.savez(out / "rollouts.npz", **rollouts)
    (out / "mjlab_eval.json").write_text(json.dumps({"meta": meta, "conditions": results}, indent=1))


# ---------------------------------------------------------------- stage 2: V-JEPA (main env)

def find_policies(runs, wanted):
    from experiments.latent_policy import TASKS
    pols = {}
    for name in wanted or TASKS:
        d = runs / f"policy_{name}"
        ckpts = sorted(d.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])) or \
            sorted(d.glob("logs/*/*/model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if ckpts and (d / "motion.npz").exists():
            pols[name] = {"checkpoint": str(ckpts[-1]), "task": TASKS[name], "motion": str(d / "motion.npz")}
    if not pols:
        raise SystemExit(f"no policy_*/model_*.pt + motion.npz in {runs}")
    return pols


def vjepa_stage(a, runs, out, pols):
    from multiprocessing import get_context

    import config
    import render
    from experiments.cross_embodiment_probe import AZ_BUCKET, dtw
    from experiments.latent_policy import aligned, bank_motions, eval_bank, to_30fps
    from latent import distill
    from latent.distill import resample, unit
    from latent.vjepa import VJepa

    t0 = time.time()
    ev = json.loads((out / "mjlab_eval.json").read_text())
    rollouts = np.load(out / "rollouts.npz")
    # The evaluation's reference frame, as in latent_policy.py: the target clip's dense latent minus
    # the mean over all clips; robot latents minus the bank's mean at the clip's viewpoint.
    lats = {p.parent.name: np.load(p) for p in sorted((runs / "clips").glob("*/video_latent.npz"))}
    h_mean = np.concatenate([v["dense"] for v in lats.values()]).mean(0)
    tgt = lats[a.target]
    bucket = int(round(float(tgt["view_azimuth"]) / AZ_BUCKET) * AZ_BUCKET)
    H = unit(tgt["dense"] - h_mean)
    vj = VJepa()
    bank_d = eval_bank(bank_motions(config.OUTPUT_DIR / "cache"), bucket, vj, render.load_g1()[0], out, a.smoke)
    r_mean = np.concatenate(list(bank_d.values())).mean(0)
    log(f"V-JEPA + bank latents ready ({time.time() - t0:.0f} s); view {bucket} deg")

    # (where the score goes, qpos at 30 fps and the clip's tempo, first dense step of the video it covers)
    jobs = [(("check", n), to_30fps(np.load(runs / f"policy_{n}" / "policy_qpos.npy")), 0)
            for n in pols if (runs / f"policy_{n}" / "policy_qpos.npy").exists()]
    for ci, c in enumerate(ev["conditions"]):
        speed = c["params"]["speed"]
        for k in rollouts.files:
            if k.startswith(f"{ci}/") and not k.endswith("/start"):
                q = rollouts[k] if speed == 1 else resample(rollouts[k], int(round((len(rollouts[k]) - 1) * speed)) + 1)
                start = int(rollouts[k + "/start"])  # reference step (50 Hz) -> dense step (2 frames at 30 fps)
                jobs.append(((ci, k.split("/")[1]), to_30fps(q), int(round(start * 30 / STEP_HZ / 2))))
    scores = {}
    # Silhouettes render in worker processes (latent/distill.py's renderer), V-JEPA embeds here.
    with get_context("spawn").Pool(a.workers, initializer=distill._init_worker) as pool:
        for (key, _, off), (sil, _) in zip(jobs, pool.imap(distill._render, ((q, bucket) for _, q, _ in jobs))):
            R = unit(vj.embed_dense(np.repeat(sil[..., None], 3, -1)) - r_mean)
            scores.setdefault(key, []).append((aligned(H[off:], R), float(dtw(H[off:], R))))
    log(f"{len(jobs)} rollouts rendered + scored ({time.time() - t0:.0f} s)")

    old = json.loads((runs / "eval.json").read_text()) if (runs / "eval.json").exists() else {}
    check = {}
    for name in pols:
        if ("check", name) in scores:
            (al, dt), = scores["check", name]
            check[name] = {"aligned": al, "dtw": dt, "eval_json_aligned": old.get(name, {}).get("aligned"),
                           "eval_json_dtw": old.get(name, {}).get("dtw")}
            log(f"check {name}: aligned {al:.3f} (eval.json {check[name]['eval_json_aligned']})")
    for ci, c in enumerate(ev["conditions"]):
        for name, r in c["policies"].items():
            each = scores.get((ci, name), [])
            r["vjepa_each"] = [[round(x, 4), round(y, 4)] for x, y in each]
            r["vjepa_aligned"] = float(np.mean([x for x, _ in each])) if each else None
            r["vjepa_dtw"] = float(np.mean([y for _, y in each])) if each else None

    ev["meta"] |= {"target": a.target, "view_bucket_deg": bucket,
                   "policies": {n: {"checkpoint": str(Path(p["checkpoint"]).relative_to(runs)), "task": p["task"]}
                                for n, p in pols.items()}}
    ev["meta"]["seconds"]["vjepa"] = time.time() - t0
    result = {"meta": ev["meta"], "reference_check": check, "conditions": ev["conditions"],
              "summary": summarise(ev["conditions"])}
    (out / "robust.json").write_text(json.dumps(result, indent=1))
    print_table(result)
    return result


def summarise(conds):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    out = {}
    for name in conds[0]["policies"]:
        clean = next(c for c in conds if c["suite"] == "clean")["policies"][name]["vjepa_aligned"]
        out[name] = {}
        for suite in dict.fromkeys(c["suite"] for c in conds):
            rs = [c["policies"][name] for c in conds if c["suite"] == suite]
            s = {k: mean(r.get(k) for r in rs) for k in ("fall_rate", "torso_err_m", "vjepa_aligned")}
            s["vjepa_retained"] = s["vjepa_aligned"] / clean if clean and s["vjepa_aligned"] is not None else None
            out[name][suite] = s
    return out


def print_table(result):
    def f(x):
        return "  -  " if x is None else f"{x:.3f}"

    names = list(result["summary"])
    print(f"\n{'suite':9s} {'level':9s} " + " ".join(f"{n + ' fall/vjepa':>20s}" for n in names))
    for c in result["conditions"]:
        print(f"{c['suite']:9s} {c['level']:9s} " + " ".join(
            f"{f(c['policies'][n]['fall_rate']):>9s} / {f(c['policies'][n]['vjepa_aligned']):>8s}" for n in names))
    print()


def main(a):
    runs, out = Path(a.runs).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.mjlab:
        return mjlab_stage(a, out)
    pols = find_policies(runs, a.policies)
    (out / "policies.json").write_text(json.dumps(pols, indent=1))
    log("policies: " + ", ".join(f"{n} ({Path(p['checkpoint']).name})" for n, p in pols.items()))
    if not (out / "mjlab_eval.json").exists():
        py = os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python")
        env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), WANDB_MODE="disabled")
        subprocess.run([py, __file__, "--mjlab", *sys.argv[1:]], check=True, env=env)
    vjepa_stage(a, runs, out, pols)
    log(f"done: {out / 'robust.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="a latent run's results folder")
    ap.add_argument("--out", default="outputs/robust")
    ap.add_argument("--target", default="squat", help="the clip the policies were trained on")
    ap.add_argument("--policies", nargs="+", help="default: every policy_<name> found (tracking, hybrid, ...)")
    ap.add_argument("--reps", type=int, default=64, help="envs per policy per condition")
    ap.add_argument("--score", type=int, default=2, help="rollouts per policy per condition scored with V-JEPA")
    ap.add_argument("--push", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.5], help="push strengths (m/s)")
    ap.add_argument("--init", type=float, nargs="*", default=[1, 2, 4], help="init noise, x training's")
    ap.add_argument("--dynamics", type=int, default=3, help="how many friction/mass levels")
    ap.add_argument("--speed", type=float, nargs="*", default=[0.8, 1.2])
    ap.add_argument("--fall-z", type=float, default=0.35, help="fallen: torso below this height (m)")
    ap.add_argument("--fall-tilt", type=float, default=1.4, help="... or tilted further than this (rad)")
    ap.add_argument("--workers", type=int, default=8, help="silhouette render processes")
    ap.add_argument("--max-steps", type=int, help="cap rollout length (debugging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny: plumbing test")
    ap.add_argument("--mjlab", action="store_true", help=argparse.SUPPRESS)  # internal: stage 1
    a = ap.parse_args()
    if a.smoke:
        a.reps, a.score, a.push, a.init, a.dynamics, a.speed = 2, 1, [1.0], [2], 1, [1.2]
        a.max_steps = a.max_steps or 60
    main(a)
