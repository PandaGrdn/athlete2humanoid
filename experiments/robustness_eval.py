"""Does a policy learn the action, or copy the clip? Trained policies, pushed off the clip.

    python experiments/robustness_eval.py --runs outputs/remote_latent --out outputs/robust
    python experiments/robustness_eval.py --summary outputs/remote_robust/robust.json   # -> robust_summary.md

--runs is a latent run's results folder: policy_<name>/model_*.pt + motion.npz per policy, and
clips/<clip>/video_latent.npz (the human clips' V-JEPA latents). Task-first policies (task,
taskphase: their actor observes the task's stage machine) also need the task spec they were trained
on: the folder's task_spec.json (skill/task.py), else $A2H_TASK_SPEC; without one they refuse to
load. reward.pt / importance.json, when there, are passed on as A2H_LATENT_BUNDLE / A2H_IMPORTANCE.
Runs without a spec (v1) are scored against one compiled from their motion.npz. Two stages, split
like train_motion.py: the rollouts run in mjlab's virtualenv (A2H_MJLAB_PYTHON; this file re-runs
itself there with --mjlab), the V-JEPA scoring in the main env (--no-vjepa skips it).

1. mjlab: every policy, batched in one environment (each policy drives its own slice of envs and
   reads its own actor/critic observation groups), under each condition of eight suites:
   - clean:      the clip as trained, nominal friction and mass
   - push:       random horizontal shoves (root velocity kicks, every 1-3 s as in training) of
                 increasing strength; training pushes reach ~0.7 m/s
   - init:       start at a random phase of the first half of the clip, from a state perturbed
                 like training's reference-state init but 1x / 2x / 4x as far
   - dynamics:   per-env foot friction and per-body mass randomisation, three widening ranges
                 (level 1 = training's friction range)
   - speed:      the reference resampled 0.8x / 1.2x as fast (an unseen command sequence)
   - push_phase: one shove per stage, timed by the robot's own stage machine: halfway down each
                 squat, or at each touchdown (both feet down after a jump)
   - start:      t = 0 but a different start state, at rest: crouched (the clip's deepest squat
                 pose) or standing still (the G1's default stance)
   - noref:      no reference at all: standing still at t = 0 and the reference frozen there.
                 TaskPhase never reads it, so only it can still do the action; the policies that
                 read the reference are shown a robot standing still.
   - tempo:      the reference 0.5x / 1.5x / 2x as fast: far off the trained tempo (speed's 0.8 / 1.2x
                 are mild). The task must still be done before the (shorter or longer) clip ends.
   - transfer:   a different, similar clip the policies never trained on (--transfer [name=]motion.npz,
                 e.g. a squat for squat-jump policies; --mirror: the clip's mirror image, the kick with
                 the other leg, for any clip): its reference, and a keypose spec compiled from it,
                 replace the clip's. Distance is to that clip; no V-JEPA score (the video is the
                 training clip's).
   All policies run in the stock tracking task's environment: the A2H tasks differ from it in
   rewards, terminations and the actor's (and task critics') observations. Those get their own
   observation groups here; actions are checked equal at start-up. The environment's only reward
   is the task spec's stage machine (mjlab_tasks.TaskReward, as in training: also what the task
   policies observe), and the eval replaces terminations with one task-independent fall test, so
   the relaxed tasks can't look better by just not resetting.
   Fall: torso below --fall-z or tilted past --fall-tilt (latched; no reset, the env stays down).
   Task success: every stage of the spec, in order, from the stage the clip is in at the start,
   and no fall before the clip ends. Cross-checked offline with skill/task.py evaluate() on the
   saved rollouts' body positions where episodes start at t = 0.
   Distance to the clip: joint-space DTW to the retarget (at 1x, from the start phase) of every
   rollout that stayed up, so timing changes cost nothing and only different poses count. Next to
   it, the time-locked joint error, and in meta what standing still would score.
2. V-JEPA: a few rollouts per policy and condition re-rendered as silhouettes and compared with
   the human video exactly as latent_policy.py's evaluation does (aligned cosine and DTW, same
   human/robot domain means), so the numbers are comparable with its eval.json. Speed-suite
   rollouts are time-resampled back to the clip's tempo first; init rollouts are compared with
   the part of the video from their start phase on. As a check, the run's own clean rollouts
   (policy_*/policy_qpos.npy) are re-scored next to their eval.json values.

The claim being tested: a policy that learned the action keeps succeeding at the task as conditions
move away from the clip, at a non-trivial distance from the clip; one that copies the clip is
closest to it on the clean clip and fails without it. Conditions 0-12 are those of v1's
robust.json (same order, seeds and metrics); the rest are appended.

robust.json:
{
  "meta": {
    "target": clip name, "view_bucket_deg": render azimuth, "policies": {name: {"checkpoint", "task"}},
    "env_task": task whose env all policies ran in, "obs_groups": {policy: {"actor", "critic"}},
    "reps": envs per policy per condition, "vjepa_rollouts": rendered rollouts per policy per
    condition, "step_hz": 50, "clip_steps",
    "fall_rule": {"torso_z_m", "tilt_rad", "ref_torso_z_min_m", "ref_tilt_max_rad"},
    "task_spec": {"source", "summary", "stages", "h0", "jump_clearance_m", "ref_squat_depth_m",
                  "self_check"}, "start_poses": {"crouched_frame"},
    "dtw_joint_rad_still": joint-space DTW of standing still at frame 0 to the clip (the scale),
    "seconds": {"mjlab", "vjepa"}
  },
  "reference_check": {policy: {"aligned", "dtw", "eval_json_aligned", "eval_json_dtw"}},
  "conditions": [{
    "suite": "clean" | "push" | "init" | "dynamics" | "speed" | "push_phase" | "start" | "noref",
    "level": label,
    "params": {"push_mps", "init_scale", "phase_max", "friction", "mass_scale", "speed",
               "push_at": null | "squat" | "land", "start": "reference" | "crouched" | "standing",
               "reference": "clip" | "frozen"},
    "policies": {policy: {
      "fall_rate":      fraction of envs that fell before the reference ended,
      "survival":       mean fraction of the clip completed upright,
      "torso_err_m":    mean torso (anchor) position error vs the clip while upright (m),
      "joint_err_rad":  mean |joint - clip joint| while upright, time-locked,
      "task_success":   fraction of envs that did every remaining stage in order without falling,
      "task_success_evaluate": the same by skill/task.py evaluate() on the saved rollouts (null
                        where episodes may start mid-clip),
      "stages_completed": mean stages done; "stages_frac": mean fraction of the remaining stages,
      "jumps":          mean jump stages completed, "jump_apex_m": mean highest lower-foot
                        clearance in flight (m), "jump_quality": mean skill/task.py jump quality
                        (1 = apex in [goal, 1.5 goal]) over landed jumps (null if none),
      "squat_depth_m":  mean depth of the pelvis below the spec's standing height, feet down,
                        landings excluded (m),
      "pushes":         mean shoves per env,
      "dtw_joint_rad":  mean joint-space DTW to the clip over envs that stayed up (null if none),
      "vjepa_aligned":  mean over the rendered rollouts (null if none),
      "vjepa_dtw":      same, DTW,
      "vjepa_each":     [[aligned, dtw], ...] per rendered rollout
    }}
  }, ...],
  "summary": {policy: {suite: {"fall_rate", "torso_err_m", "vjepa_aligned", "task_success",
                               "stages_frac", "dtw_joint_rad",
                               "vjepa_retained": suite vjepa_aligned / clean vjepa_aligned}}}
}
robust_summary.md: the compact version for the write-up (write_summary()).

Cost on an A100 (Colab): v1's 13 conditions x 3 policies x 64 envs took 4 min of mjlab and
16 min of V-JEPA (78 renders), ~25 min of VM. The 18 conditions here: ~5-6 min of mjlab (+ DTW,
seconds), ~22 min of V-JEPA (108 renders), ~35 min of VM (~3.1 compute units at 5.3/h).
--smoke: tiny, for plumbing.
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
TASK_SPEC_ENV, LATENT_BUNDLE_ENV, IMPORTANCE_ENV = "A2H_TASK_SPEC", "A2H_LATENT_BUNDLE", "A2H_IMPORTANCE"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def needs_spec(task):
    """Task-first tasks: the actor observes the stage machine, so the policy can't run without its spec."""
    return task.startswith("A2H-Task")


def conditions(a):
    """[(suite, level label, params)]. v1's 13 first, in v1's order (same per-condition seeds)."""
    c = {"push_mps": 0.0, "init_scale": 0.0, "phase_max": 0.0, "friction": None, "mass_scale": None, "speed": 1.0,
         "push_at": None, "start": "reference", "reference": "clip", "transfer": None}
    out = [("clean", "nominal", c)]
    out += [("push", f"{v:g} m/s", c | {"push_mps": v}) for v in a.push]
    out += [("init", f"x{s:g}", c | {"init_scale": s, "phase_max": 0.5}) for s in a.init]
    out += [("dynamics", f"level {i + 1}", c | {"friction": f, "mass_scale": m})
            for i, (f, m) in enumerate(DYNAMICS[:a.dynamics])]
    out += [("speed", f"{s:g}x", c | {"speed": s}) for s in a.speed]
    out += [("push_phase", f"{at} {v:g} m/s", c | {"push_at": at, "push_mps": v})
            for at in ("squat", "land") for v in a.phase_push]
    out += [("start", s, c | {"start": s}) for s in a.starts]
    if a.noref:
        out += [("noref", "frozen", c | {"start": "standing", "reference": "frozen"})]
    out += [("tempo", f"{s:g}x", c | {"speed": s}) for s in a.tempo]
    named = [t.split("=", 1) if "=" in t else (Path(t).stem, t) for t in a.transfer]
    out += [("transfer", "mirrored", c | {"transfer": "mirror"})] if a.mirror else []
    out += [("transfer", name, c | {"transfer": str(t)}) for name, t in named]
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


def mirror_motion(src, dst):
    """motion.npz reflected through the sagittal plane (y -> -y): left and right swap, roll and yaw joints
    change sign (the G1's left and right joint axes are the same vectors). Exact: forward kinematics of
    the mirrored joints gives the mirrored bodies (checked to ~1e-5 m on the front kick)."""
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
    m = get_spec().compile()
    joints = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    bodies = [m.body(i).name for i in range(1, m.nbody)]

    def swap(n):
        return n.replace("left", "@").replace("right", "left").replace("@", "right")
    jp, bp = [joints.index(swap(n)) for n in joints], [bodies.index(swap(n)) for n in bodies]
    sign = np.array([-1.0 if ("roll" in n or "yaw" in n) else 1.0 for n in joints], np.float32)
    d = np.load(src)
    flip, pseudo, quat = (np.array(v, np.float32) for v in ([1, -1, 1], [-1, 1, -1], [1, -1, 1, -1]))
    out = {"fps": d["fps"], "joint_pos": (d["joint_pos"] * sign)[:, jp], "joint_vel": (d["joint_vel"] * sign)[:, jp],
           "body_pos_w": d["body_pos_w"][:, bp] * flip, "body_quat_w": d["body_quat_w"][:, bp] * quat,
           "body_lin_vel_w": d["body_lin_vel_w"][:, bp] * flip, "body_ang_vel_w": d["body_ang_vel_w"][:, bp] * pseudo}
    np.savez(dst, **{k: np.asarray(v, np.float32) if k != "fps" else v for k, v in out.items()})
    return dst


def frozen_motion(src, dst):
    """motion.npz held at its first frame for its whole length, at rest: a reference with no motion in it."""
    d = np.load(src)
    out = {k: np.repeat(d[k][:1], len(d[k]), 0) * (0 if "vel" in k else 1) for k in d.files if k != "fps"}
    np.savez(dst, fps=d["fps"], **{k: v.astype(np.float32) for k, v in out.items()})
    return dst


def motion_body_pos(motion_file):
    """motion.npz -> (T, len(BODIES), 3): its body positions (all of the G1's bodies, in model order)
    picked down to mjlab's tracking bodies, the ones skill/task.py indexes."""
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    from latent.reward_net import BODIES
    m = get_spec().compile()
    names = [m.body(i).name for i in range(1, m.nbody)]  # without the world body
    return np.load(motion_file)["body_pos_w"][:, [names.index(b) for b in BODIES]]


def dtw_joint(q, nq, ref, nref):
    """Batched joint-space DTW. q: (B, T, J) rollouts of lengths nq (B,); ref: (B, U, J) clips of
    lengths nref. Step cost: mean |joint difference| (rad). Returns the path's total cost over
    (nq + nref) / 2, so the clip at its own tempo scores its time-locked joint error and timing
    differences cost nothing. Row by row, with the in-row recurrence as a cumulative min:
    D[i, j] = C[j] + min_k<=j (min(D[i-1, k], D[i-1, k-1]) - C[k-1]), C = cumsum of row i's cost."""
    import torch
    B, U = len(q), ref.shape[1]
    out = torch.full((B,), float("nan"), device=q.device)
    inf, zero = torch.full((B, 1), float("inf"), device=q.device), torch.zeros(B, 1, device=q.device)
    D = None
    for i in range(int(nq.max())):
        C = (q[:, i, None] - ref).abs().mean(-1).cumsum(1)
        if D is None:
            D = C
        else:
            A = torch.minimum(D, torch.cat([inf, D[:, :-1]], 1))
            D = C + torch.cummin(A - torch.cat([zero, C[:, :-1]], 1), 1).values
        last = (nq - 1 == i).nonzero().squeeze(1)
        out[last] = D[last, (nref[last] - 1).clamp(0, U - 1)]
    return out / ((nq + nref).float() / 2)


def task_spec_path(runs, pols):
    """The spec the task-first policies were trained on: <runs>/task_spec.json, else $A2H_TASK_SPEC.
    None for runs without task-first policies and without either (they get one compiled from the
    motion). Exits with an explanation if a task-first policy has none."""
    path = runs / "task_spec.json" if runs and (runs / "task_spec.json").exists() else os.environ.get(TASK_SPEC_ENV)
    need = [n for n, p in pols.items() if needs_spec(p["task"])]
    if need and not (path and Path(path).is_file()):
        raise SystemExit(
            f"policies {need} ({', '.join(sorted({pols[n]['task'] for n in need}))}) observe the task's stage machine "
            f"and can't run without the task spec they were trained on: put its task_spec.json (skill/task.py) in "
            f"{runs} or set {TASK_SPEC_ENV}=<task_spec.json>" + (f" (it is set to {path!r}, not a file)" if path else ""))
    return str(Path(path).resolve()) if path else None


def run_env(runs, pols):
    """Env vars the A2H tasks read, from the run folder: the task spec, the latent bundle, the importance."""
    env = {}
    spec = task_spec_path(runs, pols)
    if spec:
        env[TASK_SPEC_ENV] = spec
    for var, name in ((LATENT_BUNDLE_ENV, "reward.pt"), (IMPORTANCE_ENV, "importance.json")):
        if (runs / name).exists():
            env[var] = str((runs / name).resolve())
    return env


def mjlab_stage(a, out):
    import warnings
    from dataclasses import asdict

    import torch

    import mjlab.tasks  # noqa: F401  (registers the tasks)
    import mjlab_tasks  # (and ours: their cfgs are read for their observations, and TaskReward used)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.tracking.mdp.commands import MotionLoader
    from mjlab.utils.lab_api.math import yaw_quat

    from skill import task as skill_task

    warnings.filterwarnings("ignore", message="dr.body_mass")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pols = json.loads((out / "policies.json").read_text())
    names = list(pols)
    motion_file = Path(pols[names[0]]["motion"])
    for n in names[1:]:  # all policies were trained on the same clip
        assert np.allclose(np.load(pols[n]["motion"])["joint_pos"], np.load(motion_file)["joint_pos"], atol=1e-4), n  # (variants: float noise)

    # The task spec: the one the task-first policies were trained on, or (v1 runs) one read off the
    # reference the policies were trained on, at 50 Hz; then the retarget must pass it, as in skill/task.py.
    spec_path = task_spec_path(None, pols)
    ref_pos = motion_body_pos(motion_file)
    if spec_path:
        spec, source = json.loads(Path(spec_path).read_text()), spec_path
    else:
        spec, source = skill_task.compile_spec(ref_pos, float(STEP_HZ), name=a.target), "compiled from motion.npz"
        spec_path = str(out / "task_spec.json")
        Path(spec_path).write_text(json.dumps(spec, indent=1))
    from skill import keypose as K

    def kinds_of(sp):
        """The stage kinds the machine reports: keypose (pose / flight / settle) or stand / squat / jump / land."""
        if sp.get("type") == "keypose":
            return K.FLIGHT, K.SETTLE, K.POSE
        return skill_task.JUMP, skill_task.LAND, skill_task.SQUAT
    clip_s = len(ref_pos) / STEP_HZ
    assert abs(spec["duration_s"] - clip_s) < 0.25, f"task spec ({spec['duration_s']} s) is not of this clip ({clip_s:.2f} s)"
    self_check = skill_task.evaluate({**spec, "down": spec["down_ref"], "air": spec["air_ref"]}, ref_pos, STEP_HZ)
    os.environ[TASK_SPEC_ENV] = spec_path  # read by TaskReward when the env is built
    log(f"task spec ({source}): {spec['summary']}; the reference completes "
        f"{self_check['stages_completed']}/{self_check['stages_total']}")

    cfg, base = load_env_cfg(ENV_TASK, play=True), load_env_cfg(ENV_TASK, play=True)
    groups = {}  # policy -> its actor and critic observation groups
    for n in names:  # the A2H tasks must act the same way; a different input gets its own group
        c = load_env_cfg(pols[n]["task"], play=True)
        assert c.actions["joint_pos"].scale == cfg.actions["joint_pos"].scale, n
        groups[n] = {}
        for g in ("actor", "critic"):
            groups[n][g] = g
            if list(c.observations[g].terms) != list(base.observations[g].terms):
                groups[n][g] = f"{g}_{n}"
                cfg.observations[groups[n][g]] = c.observations[g]
    train_cmd = load_env_cfg(ENV_TASK).commands["motion"]  # training's reference-state-init ranges
    cfg.commands["motion"].motion_file = str(motion_file)
    cfg.commands["motion"].debug_vis = False
    cfg.scene.num_envs = a.reps * len(names)
    cfg.seed = a.seed
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}  # falls: ours
    cfg.rewards = {"task": RewardTermCfg(func=mjlab_tasks.TaskReward, weight=1.0)}  # the stage machine
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
        rl.obs_groups = {**rl.obs_groups, "actor": (groups[n]["actor"],), "critic": (groups[n]["critic"],)}
        runner = (load_runner_cls(pols[n]["task"]) or MjlabOnPolicyRunner)(venv, asdict(rl), device=device)
        runner.load(pols[n]["checkpoint"], load_cfg={"actor": True}, strict=True, map_location=device)
        policies.append(runner.get_inference_policy(device=device))
    log(f"env with {cfg.scene.num_envs} envs + {len(names)} policies ready ({time.time() - t0:.0f} s)")

    cmd, robot, m = env.command_manager.get_term("motion"), env.scene["robot"], env.a2h_task
    reward_term, clip_machine = env.reward_manager.get_term_cfg("task").func, env.a2h_task  # TaskReward instance
    assert np.allclose(cmd.motion.body_pos_w.cpu().numpy(), ref_pos, atol=1e-5), "motion body order"
    friction = env.event_manager.get_term_cfg("foot_friction")
    mass = env.event_manager.get_term_cfg("eval_mass")
    slices = [slice(i * a.reps, (i + 1) * a.reps) for i in range(len(names))]
    keep = torch.cat([torch.arange(s.start, s.start + a.score) for s in slices]).to(device)

    def tilt(quat):  # angle between the body's z axis and world up (quat wxyz)
        return torch.acos((1 - 2 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)).clamp(-1, 1))

    anchor = cmd.motion_anchor_body_index
    ref_z = cmd.motion.body_pos_w[:, anchor, 2]
    ref_tilt = tilt(cmd.motion.body_quat_w[:, anchor])
    clip_q = cmd.motion.joint_pos.clone()  # the retarget at 1x: what "distance to the clip" is measured to
    kinds = [s["kind"] for s in spec["stages"]]
    squats = [s for s in spec["stages"] if s["kind"] == "squat"]
    lo_f, hi_f = (int(squats[0]["start_s"] * STEP_HZ), int(squats[0]["end_s"] * STEP_HZ) + 1) if squats else (0, len(ref_z))
    crouch = lo_f + int(cmd.motion.body_pos_w[lo_f:hi_f, 0, 2].argmin())  # the clip's deepest (first) squat
    T0 = len(clip_q)
    still = dtw_joint(clip_q[:1].expand(T0, -1)[None], torch.tensor([T0], device=device), clip_q[None],
                      torch.tensor([T0], device=device))
    meta = {"env_task": ENV_TASK, "obs_groups": groups, "reps": a.reps, "vjepa_rollouts": a.score, "step_hz": STEP_HZ,
            "clip_steps": int(cmd.motion.time_step_total),
            "fall_rule": {"torso_z_m": a.fall_z, "tilt_rad": a.fall_tilt, "ref_torso_z_min_m": ref_z.min().item(),
                          "ref_tilt_max_rad": ref_tilt.max().item()},
            "task_spec": {"source": source, "summary": spec["summary"], "stages": kinds, "h0": spec["h0"],
                          "jump_clearance_m": [s["clearance"] for s in spec["stages"] if s["kind"] in ("jump", "flight")],
                          "ref_squat_depth_m": [s["depth"] for s in squats if "depth" in s],
                          "self_check": {k: self_check[k] for k in ("stages_completed", "stages_total", "success")}},
            "start_poses": {"crouched_frame": crouch},
            "dtw_joint_rad_still": still.item()}
    assert ref_z.min() > a.fall_z + 0.1 and ref_tilt.max() < a.fall_tilt - 0.2, f"fall rule fires on the reference: {meta}"

    # Start states other than the reference's: the command's reset writes them instead (at rest).
    start = {"mode": "reference"}
    write_reference = cmd._write_reference_state_to_sim

    def write_start(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        if start["mode"] != "reference":
            root_lin_vel, root_ang_vel, joint_vel = (torch.zeros_like(x) for x in (root_lin_vel, root_ang_vel, joint_vel))
            root_pos = root_pos.clone()
            if start["mode"] == "crouched":  # the clip's crouch, where the clip starts
                root_pos[:, 2] = start["root"][2] + env.scene.env_origins[env_ids, 2]
                root_ori = start["root"][3:7].expand(len(env_ids), -1).clone()
                joint_pos = start["joints"].expand(len(env_ids), -1).clone()
            else:  # standing: the G1's default stance, facing the clip's way
                root_pos[:, 2] = robot.data.default_root_state[env_ids, 2] + env.scene.env_origins[env_ids, 2]
                root_ori = yaw_quat(root_ori)
                joint_pos = robot.data.default_joint_pos[env_ids].clone()
        write_reference(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel)

    cmd._write_reference_state_to_sim = write_start
    start["root"] = torch.cat([cmd.motion.body_pos_w[crouch, 0], cmd.motion.body_quat_w[crouch, 0]])
    start["joints"] = cmd.motion.joint_pos[crouch]

    def push(ids, mps):
        ang = torch.rand(len(ids), device=device) * 2 * torch.pi
        vel = robot.data.root_link_vel_w[ids].clone()
        vel[:, 0] += mps * torch.cos(ang)
        vel[:, 1] += mps * torch.sin(ang)
        robot.write_root_link_velocity_to_sim(vel, env_ids=ids)

    results, rollouts = [], {}
    with torch.inference_mode():  # also the resets: tensors made inside can't be updated outside
        for ci, (suite, level, p) in enumerate(conditions(a)):
            tc = time.time()
            torch.manual_seed(a.seed + ci)
            mf = motion_file if p["speed"] == 1 else resample_motion(motion_file, out / f"motion_x{p['speed']:g}.npz", p["speed"])
            if p["reference"] == "frozen":
                mf = frozen_motion(motion_file, out / "motion_frozen.npz")
            if p["transfer"]:
                mf = mirror_motion(motion_file, out / "motion_mirrored.npz") if p["transfer"] == "mirror" else Path(p["transfer"])
            cmd.motion = MotionLoader(str(mf), cmd.body_indexes, device=env.device)
            # A2H-TrackingVJ's input: this motion's own video latent (<motion>.vjl.npy), else the clip's
            # (A2H_VIDEO_LATENT), re-timed to this motion's length
            vjl = Path(mf).with_suffix(".vjl.npy")
            env.a2h_video_latent = torch.as_tensor(np.load(vjl), dtype=torch.float32, device=env.device) if vjl.exists() else None
            truth = cmd.motion if p["reference"] == "clip" else MotionLoader(str(motion_file), cmd.body_indexes, device=env.device)
            # The task: the clip's spec, or for a new clip one compiled from it (the policies never saw either)
            if p["transfer"]:
                cs = K.compile_keypose_spec(motion_body_pos(mf), float(STEP_HZ), name=level)
                reward_term.m = env.a2h_task = skill_task.machine(cs, env.num_envs, env.device, env.step_dt)
                log(f"transfer to {level}: {cs['summary']}")
            else:
                cs, reward_term.m = spec, clip_machine
                env.a2h_task = clip_machine
            m = env.a2h_task
            JUMP_K, LAND_K, SQUAT_K = kinds_of(cs)
            cq = cmd.motion.joint_pos.clone() if p["transfer"] else clip_q  # what "distance to the clip" is to
            T0c = len(cq)
            start["mode"] = p["start"]
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
            start_t = cmd.time_steps.clone()
            length = (T - 1) - start_t  # steps until the reference ends (then it wraps: robot teleported)
            steps = int(length.max()) if a.max_steps is None else min(int(length.max()), a.max_steps)
            length = length.clamp(max=steps)
            N = env.num_envs
            zeros = lambda dtype=torch.float32: torch.zeros(N, dtype=dtype, device=device)  # noqa: E731
            fallen, finished, fire = zeros(torch.bool), zeros(torch.bool), zeros(torch.bool)
            upright, torso_err, joint_err, apex, jq_sum = zeros(), zeros(), zeros(), zeros(), zeros()
            done_n, jumps, jq_n, pushes, start_stage = (zeros(torch.long) for _ in range(5))
            pushed_stage = torch.full((N,), -1, dtype=torch.long, device=device)
            z_min = torch.full((N,), cs["h0"], device=device)
            next_push = torch.empty(N, device=device).uniform_(*PUSH_EVERY_S) * STEP_HZ
            qpos = torch.zeros(len(keep), steps, 7 + robot.data.joint_pos.shape[1], device=device)
            body = torch.zeros(len(keep), steps, *cmd.robot_body_pos_w.shape[1:], device=device)
            joints = torch.zeros(N, steps, robot.data.joint_pos.shape[1], device=device)
            for t in range(steps):
                if p["push_mps"] and not p["push_at"]:
                    ids = ((t >= next_push) & ~fallen).nonzero().squeeze(1)
                    if len(ids):
                        push(ids, p["push_mps"])
                        pushes[ids] += 1
                        next_push[ids] += torch.empty(len(ids), device=device).uniform_(*PUSH_EVERY_S) * STEP_HZ
                if fire.any():  # a stage-timed shove, decided after the previous step
                    ids = fire.nonzero().squeeze(1)
                    push(ids, p["push_mps"])
                    pushes[ids] += 1
                prev_stage, prev_done = m.stage.clone(), m.completed.clone()
                obs, _, _, _ = venv.step(torch.cat([pi(obs[sl]) for pi, sl in zip(policies, slices)]))
                live = (t < length) & ~fallen
                z = cmd.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
                fallen |= live & ((z < a.fall_z) | (tilt(cmd.robot_anchor_quat_w) > a.fall_tilt))
                live &= ~fallen
                upright += live
                ts = cmd.time_steps.clamp(max=T - 1)
                torso_err += live * (cmd.robot_anchor_pos_w - truth.body_pos_w[ts, anchor] - env.scene.env_origins).norm(dim=-1)
                joint_err += live * (cmd.robot_joint_pos - truth.joint_pos[ts]).abs().mean(-1)

                # The task, by the stage machine the env's reward ran this step (TaskReward).
                pos = cmd.robot_body_pos_w
                pz, lo, hi, _ = skill_task.features(pos, cs["foot0"])
                kind = m.kind[m.stage]
                if t == 0:  # the machine (re)starts on the first step, at the clip's stage then
                    start_stage = m.stage - m.completed
                else:
                    adv = live & (m.completed > prev_done)
                    jumps += adv & (m.kind[prev_stage] == JUMP_K)
                    landed = adv & (m.jump_goal[prev_stage] > 0)
                    jq_sum += landed * m.stats["jump_q"]
                    jq_n += landed
                done_n = torch.where(live, m.completed, done_n)
                finished = torch.where(live, m.done, finished)
                apex = torch.where(live & (lo > cs["air"]), torch.maximum(apex, lo), apex)
                z_min = torch.where(live & (hi < cs["down"]) & (kind != LAND_K), torch.minimum(z_min, pz), z_min)
                if p["push_at"]:
                    landing = (kind == LAND_K) & (hi < cs["down"]) & ((m.jump_goal[m.stage] > 0) if JUMP_K != skill_task.JUMP else True)
                    want = (kind == SQUAT_K) & (m.p >= 0.5) if p["push_at"] == "squat" else landing
                    fire = want & live & ~m.done & (m.stage != pushed_stage)
                    pushed_stage = torch.where(fire, m.stage, pushed_stage)

                joints[:, t] = robot.data.joint_pos
                qpos[:, t] = torch.cat([robot.data.root_link_pos_w[keep] - env.scene.env_origins[keep],
                                        robot.data.root_link_quat_w[keep], robot.data.joint_pos[keep]], -1)
                body[:, t] = pos[keep] - env.scene.env_origins[keep, None]

            # Distance to the clip (1x, from the start phase) of every rollout, by joint-space DTW.
            speed = p["speed"]
            nref = torch.minimum(T0c - start_t, (length * speed).round().long() + 1)
            idx = (start_t[:, None] + torch.arange(T0c, device=device)).clamp(max=T0c - 1)
            dtw = dtw_joint(joints, length.clamp(min=1), cq[idx], nref)
            success = finished & ~fallen

            res = {}
            for name, sl in zip(names, slices):
                u = upright[sl].clamp(min=1)
                up_ok = ~fallen[sl]

                def mean(x, mask=None):
                    x = x[sl].float() if mask is None else x[sl][mask].float()
                    return x.mean().item() if len(x) else None

                res[name] = {"fall_rate": fallen[sl].float().mean().item(),
                             "survival": (upright[sl] / length[sl].clamp(min=1)).mean().item(),
                             "torso_err_m": (torso_err[sl] / u).mean().item(),
                             "joint_err_rad": (joint_err[sl] / u).mean().item(),
                             "task_success": mean(success),
                             "task_success_evaluate": None,
                             "stages_completed": mean(done_n),
                             "stages_frac": mean(done_n / (m.S - start_stage).clamp(min=1)),
                             "jumps": mean(jumps), "jump_apex_m": mean(apex),
                             "jump_quality": jq_sum[sl].sum().item() / jq_n[sl].sum().item() if jq_n[sl].sum() else None,
                             "squat_depth_m": mean(cs["h0"] - z_min),
                             "pushes": mean(pushes),
                             "dtw_joint_rad": mean(dtw, up_ok)}
            for k, e in enumerate(keep.tolist()):
                name = names[e // a.reps]
                key = f"{ci}/{name}/{e % a.reps}"
                rollouts[key] = qpos[k, :int(length[e])].cpu().numpy()
                rollouts[key + "/start"] = np.array(int(start_t[e]))
                rollouts[key + "/body"] = body[k, :int(length[e])].cpu().numpy()
            if p["phase_max"] == 0:  # episodes start at t = 0: skill/task.py's evaluate() applies as is
                for name in names:
                    ev = [skill_task.evaluate(cs, rollouts[f"{ci}/{name}/{r}/body"], STEP_HZ)["success"]
                          for r in range(a.score)]
                    res[name]["task_success_evaluate"] = float(np.mean(ev))
            results.append({"suite": suite, "level": level, "params": p, "policies": res})
            log(f"{suite:10s} {level:14s} ({steps} steps, {time.time() - tc:.0f} s): " + "  ".join(
                f"{n} fall {r['fall_rate']:.2f} task {r['task_success']:.2f} dtw {fmt(r['dtw_joint_rad'])}"
                for n, r in res.items()))

    meta["seconds"] = {"mjlab": time.time() - t0}
    np.savez(out / "rollouts.npz", **rollouts)
    (out / "mjlab_eval.json").write_text(json.dumps({"meta": meta, "conditions": results}, indent=1))


def fmt(x, digits=3):
    return "-" if x is None else f"{x:.{digits}f}"


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


def vjepa_stage(a, runs, out, ev, pols):
    """Adds the V-JEPA scores to ev's conditions -> the reference check."""
    from multiprocessing import get_context

    import config
    import render
    from experiments.cross_embodiment_probe import AZ_BUCKET, dtw
    from experiments.latent_policy import aligned, bank_motions, eval_bank, to_30fps
    from latent import distill
    from latent.distill import resample, unit
    from latent.vjepa import VJepa

    t0 = time.time()
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
        if c["params"].get("transfer"):  # another clip: the training clip's video is the wrong reference
            continue
        for k in rollouts.files:
            if k.startswith(f"{ci}/") and k.count("/") == 2:  # ci/policy/rep: qpos (also .../start, .../body)
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
    ev["meta"]["view_bucket_deg"] = bucket
    ev["meta"]["seconds"]["vjepa"] = time.time() - t0
    return check


def summarise(conds):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    out = {}
    for name in conds[0]["policies"]:
        clean = next(c for c in conds if c["suite"] == "clean")["policies"][name].get("vjepa_aligned")
        out[name] = {}
        for suite in dict.fromkeys(c["suite"] for c in conds):
            rs = [c["policies"][name] for c in conds if c["suite"] == suite]
            s = {k: mean(r.get(k) for r in rs) for k in ("fall_rate", "torso_err_m", "vjepa_aligned", "task_success",
                                                          "stages_frac", "dtw_joint_rad")}
            s["vjepa_retained"] = s["vjepa_aligned"] / clean if clean and s["vjepa_aligned"] is not None else None
            out[name][suite] = s
    return out


def print_table(result):
    names = list(result["summary"])
    print(f"\n{'suite':10s} {'level':14s} " + " ".join(f"{n + ' fall/task/vjepa':>26s}" for n in names))
    for c in result["conditions"]:
        print(f"{c['suite']:10s} {c['level']:14s} " + " ".join(
            f"{fmt(r['fall_rate'], 2):>8s} / {fmt(r.get('task_success'), 2):>5s} / {fmt(r.get('vjepa_aligned')):>7s}"
            for r in (c["policies"][n] for n in names)))
    print()


# Headline columns of robust_summary.md: (header, suites averaged).
SUMMARY_COLUMNS = (("Clean", ("clean",)), ("Pushes", ("push",)), ("Stage-timed pushes", ("push_phase",)),
                   ("Start noise", ("init",)), ("Start pose", ("start",)), ("Speed", ("speed",)),
                   ("Tempo 0.5-2x", ("tempo",)), ("Friction/mass", ("dynamics",)), ("No reference", ("noref",)),
                   ("New similar clip", ("transfer",)))


def write_summary(result, path):
    """robust.json -> a compact markdown table for the write-up: per policy, task success clean and
    under each kind of change, and its distance to the clip. Works on v1's robust.json too (no
    task metrics: falls only)."""
    meta, conds = result["meta"], result["conditions"]
    names = list(conds[0]["policies"])
    suites = {c["suite"] for c in conds}
    cols = [(h, ss) for h, ss in SUMMARY_COLUMNS if suites & set(ss)]
    has_task = any(r.get("task_success") is not None for c in conds for r in c["policies"].values())

    def pct(x):
        return "-" if x is None else f"{100 * x:.0f}%"

    def avg(name, ss, key):
        xs = [c["policies"][name].get(key) for c in conds if c["suite"] in ss]
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    def cell(name, ss):
        succ, fall = avg(name, ss, "task_success"), avg(name, ss, "fall_rate")
        return f"{pct(succ)} ({pct(fall)} fall)" if has_task else f"{pct(fall)} fall"

    spec = meta.get("task_spec", {})
    clean = next(c for c in conds if c["suite"] == "clean")["policies"]
    lines = [f"# Robustness: {meta.get('target', '?')}", "",
             f"{meta.get('reps', '?')} envs per policy per condition, {len(conds)} conditions."
             + (f" Task: {spec['summary']} (spec: {Path(spec['source']).name if spec['source'].endswith('.json') else spec['source']})." if spec else ""),
             "", "Task success (fall rate) per suite, averaged over its conditions" if has_task else
             "Fall rate per suite (this robust.json predates the task metrics)", "",
             "| Policy | " + " | ".join(h for h, _ in cols) + " | DTW to clip | Joint err | V-JEPA |",
             "|---|" + "---|" * (len(cols) + 3)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(cell(n, ss) for _, ss in cols)
                     + f" | {fmt(clean[n].get('dtw_joint_rad'))} | {fmt(clean[n].get('joint_err_rad'))}"
                     + f" | {fmt(clean[n].get('vjepa_aligned'))} |")
    still, rule = meta.get("dtw_joint_rad_still"), meta.get("fall_rule", {"torso_z_m": "?", "tilt_rad": "?"})
    lines += [""] + ([f"- Success: every stage of the task spec done in order, from the stage the clip is in at the "
                      f"start, with no fall (torso below {rule['torso_z_m']} m or tilted past {rule['tilt_rad']} rad) "
                      "before the clip ends."] if has_task else []) + [
              "- Pushes: random shoves every 1-3 s; stage-timed pushes: one per squat (halfway down) or per touchdown; "
              "start noise: random start in the first half of the clip, 1-4x training's init noise; start pose: crouched "
              "or standing still at t = 0; no reference: standing still and the reference frozen at t = 0.",
              "- DTW to clip: joint-space DTW to the retarget on the clean clip, mean rad per matched frame, over "
              "rollouts that stayed up" + (f"; standing still scores {still:.3f}." if still else ".")
              + " Joint err: the same, time-locked. V-JEPA: aligned cosine with the video, clean clip.", "",
              "## Per condition", "",
              "| Suite | Level | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for c in conds:
        cells = []
        for n in names:
            r = c["policies"][n]
            cells.append((f"{pct(r.get('task_success'))} / " if has_task else "") + f"{pct(r['fall_rate'])} fall"
                         + (f" / dtw {fmt(r['dtw_joint_rad'])}" if r.get("dtw_joint_rad") is not None else ""))
        lines.append(f"| {c['suite']} | {c['level']} | " + " | ".join(cells) + " |")
    lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def main(a):
    if a.summary:
        f = Path(a.summary)
        log(f"wrote {write_summary(json.loads(f.read_text()), f.with_name('robust_summary.md'))}")
        return
    runs, out = Path(a.runs).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.mjlab:
        return mjlab_stage(a, out)
    pols = find_policies(runs, a.policies)
    env_vars = run_env(runs, pols)  # exits here if a task-first policy has no task spec
    (out / "policies.json").write_text(json.dumps(pols, indent=1))
    log("policies: " + ", ".join(f"{n} ({Path(p['checkpoint']).name}, {p['task']})" for n, p in pols.items())
        + "".join(f"; {k}={v}" for k, v in env_vars.items()))
    if not (out / "mjlab_eval.json").exists():
        py = os.environ.get("A2H_MJLAB_PYTHON", "/content/mjlab_env/bin/python")
        env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), WANDB_MODE="disabled", **env_vars)
        subprocess.run([py, __file__, "--mjlab", *sys.argv[1:]], check=True, env=env)
    ev = json.loads((out / "mjlab_eval.json").read_text())
    ev["meta"] |= {"target": a.target, "view_bucket_deg": None,
                   "policies": {n: {"checkpoint": str(Path(p["checkpoint"]).relative_to(runs)), "task": p["task"]}
                                for n, p in pols.items()}}
    check = {} if a.no_vjepa else vjepa_stage(a, runs, out, ev, pols)
    result = {"meta": ev["meta"], "reference_check": check, "conditions": ev["conditions"],
              "summary": summarise(ev["conditions"])}
    (out / "robust.json").write_text(json.dumps(result, indent=1))
    write_summary(result, out / "robust_summary.md")
    print_table(result)
    log(f"done: {out / 'robust.json'}, {out / 'robust_summary.md'}")


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="a latent run's results folder")
    ap.add_argument("--out", default="outputs/robust")
    ap.add_argument("--target", default="squat", help="the clip the policies were trained on")
    ap.add_argument("--policies", nargs="+", help="default: every policy_<name> found (tracking, hybrid, ...)")
    ap.add_argument("--reps", type=int, default=64, help="envs per policy per condition")
    ap.add_argument("--score", type=int, default=2, help="rollouts per policy per condition scored with V-JEPA")
    ap.add_argument("--push", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.5], help="push strengths (m/s)")
    ap.add_argument("--init", type=float, nargs="*", default=[1, 2, 4], help="init noise, x training's")
    ap.add_argument("--dynamics", type=int, default=3, help="how many friction/mass levels")
    ap.add_argument("--speed", type=float, nargs="*", default=[0.8, 1.2])
    ap.add_argument("--tempo", type=float, nargs="*", default=[0.5, 1.5, 2.0], help="far tempos (the tempo suite)")
    ap.add_argument("--transfer", nargs="*", default=[], help="[name=]motion.npz of similar clips never trained on")
    ap.add_argument("--mirror", action="store_true", help="also the clip's mirror image (left <-> right)")
    ap.add_argument("--phase-push", type=float, nargs="*", default=[0.75],
                    help="stage-timed push strengths (m/s): one per squat (halfway down) / per touchdown")
    ap.add_argument("--starts", nargs="*", default=["crouched", "standing"], choices=["crouched", "standing"],
                    help="start states at t = 0 other than the reference's")
    ap.add_argument("--no-noref", dest="noref", action="store_false", help="skip the no-reference condition")
    ap.add_argument("--fall-z", type=float, default=0.35, help="fallen: torso below this height (m)")
    ap.add_argument("--fall-tilt", type=float, default=1.4, help="... or tilted further than this (rad)")
    ap.add_argument("--workers", type=int, default=8, help="silhouette render processes")
    ap.add_argument("--max-steps", type=int, help="cap rollout length (debugging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-vjepa", action="store_true", help="mjlab stage only (no rendering; e.g. on a CPU)")
    ap.add_argument("--summary", help="only (re)write robust_summary.md next to this robust.json")
    ap.add_argument("--smoke", action="store_true", help="tiny: plumbing test")
    ap.add_argument("--mjlab", action="store_true", help=argparse.SUPPRESS)  # internal: stage 1
    a = ap.parse_args(argv)
    if not (a.runs or a.summary):
        ap.error("--runs is required (or --summary <robust.json>)")
    if a.smoke:
        a.reps, a.score, a.push, a.init, a.dynamics, a.speed, a.phase_push = 2, 1, [1.0], [2], 1, [1.2], [1.0]
        a.max_steps = a.max_steps or 60
    return a


if __name__ == "__main__":
    main(parse())
