"""PPO on a bundle with mjlab's trainer (rsl_rl), as train_motion.py runs it for the stored runs.

    python -m a2h_env train <bundle.zip> --iters 1500 --num-envs 4096 [--variant train|reference|<name>] [--out DIR]
    python -m a2h_env train <bundle.zip> --iters 1000 --attempt-every 50     # + attempts/ (a2h_env/attempts.py)

Writes <out>/logs/<timestamp>/model_<iteration>.pt (rsl_rl checkpoints, loadable by mjlab's play script or
mjlab_rollout.py) and <out>/weights.json saying what was trained on what. --attempt-every N also rolls the policy
out on the reference every N iterations and at the end, into <out>/attempts/ (the "watch it learn" view).
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from .env import default_device, env_cfg, resolve_variant, task_of


def train(bundle, out, iters=1500, num_envs=4096, variant="train", device=None, seed=0, save_interval=250, work_dir=None,
          attempt_every=0, log=print):
    """-> the newest checkpoint path. variant "train" (every kept training variant) needs the bundle to carry their
    arrays; "reference" or None trains on the clip's retarget alone. attempt_every > 0: attempts/ (a2h_env.attempts)."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    from mjlab.utils.torch import configure_torch_backends

    device = device or default_device()
    configure_torch_backends()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if variant == "train" and resolve_variant(bundle, variant) != "train":
        print("[a2h_env] no training variants with arrays in this bundle: training on the reference", flush=True)
    variant = resolve_variant(bundle, variant)
    task = task_of(variant, bundle.reward_config.get("feet_height"))
    cfg = env_cfg(bundle, variant, num_envs, play=False, work_dir=work_dir, device=device)
    cfg.seed = seed
    agent = unitree_g1_tracking_ppo_runner_cfg()
    agent.seed, agent.max_iterations, agent.save_interval = seed, iters, min(save_interval, iters)
    agent.logger, agent.experiment_name = "tensorboard", "a2h_env"
    log_dir = out / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True)
    print(f"[a2h_env] {task} on {bundle.id or bundle.path.name}: variant {variant}, {num_envs} envs, {iters} iterations, "
          f"device {device}; logs in {log_dir}", flush=True)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MotionTrackingOnPolicyRunner(venv, asdict(agent), str(log_dir), device)
    attempts = None
    if attempt_every and attempt_every > 0:
        from .attempts import Attempts
        attempts = Attempts(bundle, out, iters, attempt_every, clip_actions=agent.clip_actions, work_dir=work_dir,
                            device=device, log=log)
        logged = runner.logger.log

        def record(done):
            """An attempt; anything wrong with it (the play env, the score) is logged and turns attempts off, never
            stops the training."""
            nonlocal attempts
            try:
                attempts.record(runner, done)
            except Exception as e:  # noqa: BLE001
                log(f"[a2h_env] attempt at iteration {done} failed ({type(e).__name__}: {str(e)[:200]}); no more attempts this run")
                try:
                    attempts.close()
                except Exception:  # noqa: BLE001
                    pass
                attempts = None

        def log_and_attempt(*a, **kw):  # rsl_rl calls logger.log(it=...) at the end of every iteration
            logged(*a, **kw)
            it = kw.get("it", a[0] if a else None)
            if it is None or attempts is None:
                return
            attempts.note_iteration(runner)
            done = int(it) + 1  # updates so far
            if done % attempt_every == 0 and done < iters:
                record(done)
        runner.logger.log = log_and_attempt
    t0 = time.time()
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=True)
    took = time.time() - t0
    if attempts is not None:
        record(iters)  # the policy as trained
    if attempts is not None:
        attempts.close()
    ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:  # rsl_rl saves on its interval and at the end; make sure something is there
        runner.save(str(log_dir / f"model_{iters - 1}.pt"))
        ckpts = [log_dir / f"model_{iters - 1}.pt"]
    meta = {"task": task, "bundle": str(bundle.path), "bundle_id": bundle.id, "variant": variant,
            "checkpoint": str(ckpts[-1].relative_to(out)), "iterations": iters, "num_envs": num_envs, "device": device,
            "seed": seed, "train_s": round(took, 1), "motion_file": str(cfg.commands["motion"].motion_file),
            "feet_height": bundle.reward_config.get("feet_height"), "spec_sha256": bundle.manifest.get("spec_sha256"),
            "attempt_every": int(attempt_every or 0), "attempts": len(attempts.rows) if attempts else 0}
    (out / "weights.json").write_text(json.dumps(meta, indent=1))
    env.close()
    print(f"[a2h_env] done: {ckpts[-1]} after {took:.0f} s", flush=True)
    return ckpts[-1]
