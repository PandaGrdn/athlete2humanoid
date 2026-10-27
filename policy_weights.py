"""Start PPO from saved weights instead of from scratch. Runs inside mjlab's virtualenv (torch only).

- resume: the same task, interrupted (e.g. the Colab VM died): actor, critic, optimizer and the
  iteration counter, so training continues where it stopped.
- init: build off another policy, possibly of another task: weights whose shapes match are copied;
  an input layer whose observation grew (terms appended, e.g. Tracking -> Task adds task_stage)
  keeps the old columns and gives the new inputs zero weight, so the new policy starts out acting
  exactly like the old one; anything else keeps its fresh initialisation. Fresh optimizer, iteration 0.

mjlab_tasks.py applies it when A2H_INIT_CHECKPOINT (and A2H_INIT_MODE) are set; train_motion.py
--init sets them, and resumes by itself when an earlier attempt left checkpoints behind.
"""

import torch

INIT_ENV, MODE_ENV = "A2H_INIT_CHECKPOINT", "A2H_INIT_MODE"
NORMALIZER = ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std")


def adapt_state_dict(src, dst):
    """src state dict -> one shaped like dst. -> (state dict, {key: how it was filled})."""
    out, how = {}, {}
    for k, d in dst.items():
        s = src.get(k)
        if s is None:
            out[k], how[k] = d, "fresh (not in source)"
        elif s.shape == d.shape:
            out[k], how[k] = s.to(d), "copied"
        elif s.dim() == d.dim() and s.shape[:-1] == d.shape[:-1] and s.shape[-1] < d.shape[-1]:
            # More inputs at the end: old columns kept; new weights 0, new normaliser stats identity.
            fill = torch.ones_like(d) if k.endswith(("_var", "_std")) else torch.zeros_like(d)
            fill[..., : s.shape[-1]] = s
            out[k], how[k] = fill, f"grown {s.shape[-1]} -> {d.shape[-1]} inputs"
        else:
            out[k], how[k] = d, f"fresh (shape {tuple(s.shape)} -> {tuple(d.shape)})"
    return out, how


def init_from(alg, ckpt):
    """Load ckpt's actor and critic into alg (rsl_rl PPO), adapting shapes. -> report."""
    report = {}
    for name, module in (("actor", alg._raw_actor), ("critic", alg._raw_critic)):
        sd, how = adapt_state_dict(ckpt[f"{name}_state_dict"], module.state_dict())
        module.load_state_dict(sd)
        report[name] = how
    return report


def warm_start(runner, path, mode="init"):
    """Apply a checkpoint to a freshly built runner (before learn())."""
    ckpt = torch.load(path, map_location=runner.device, weights_only=False)
    if mode == "resume":
        runner.load(path, map_location=runner.device)
        print(f"[a2h] resumed {path} at iteration {runner.current_learning_iteration}", flush=True)
        return
    report = init_from(runner.alg, ckpt)
    fresh = [f"{n}.{k}: {h}" for n, r in report.items() for k, h in r.items() if h != "copied"]
    print(f"[a2h] initialised from {path} (trained {ckpt.get('iter', '?')} iterations); "
          + ("; ".join(fresh) if fresh else "all weights copied"), flush=True)


def patch_mjlab_train():
    """Make mjlab's train script warm-start every runner it builds from A2H_INIT_CHECKPOINT."""
    import os

    import mjlab.scripts.train as train
    from mjlab.rl import MjlabOnPolicyRunner

    path, mode = os.environ.get(INIT_ENV), os.environ.get(MODE_ENV, "init")
    if not path:
        return
    load = train.load_runner_cls

    def load_runner_cls(task_id):
        base = load(task_id) or MjlabOnPolicyRunner

        class WarmStarted(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                warm_start(self, path, mode)

        return WarmStarted

    train.load_runner_cls = load_runner_cls
