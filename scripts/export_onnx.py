"""Export trained policies (policy_*/model_*.pt, rsl_rl format) to ONNX, next to each checkpoint.

    python scripts/export_onnx.py outputs/remote_latent [more run folders...] [--verify]

The graph is the deterministic actor: obs (1 x obs_dim, the policy's observation vector in
mjlab's order) -> empirical normalisation -> MLP (ELU) -> 29 G1 joint actions. --verify rebuilds
the actor with rsl_rl's own MLPModel and checks both give the same actions. Needs torch and
onnx (the mjlab venv has both).
"""

import argparse
import re
from pathlib import Path

import torch

EPS = 1e-2  # rsl_rl EmpiricalNormalization default


class Actor(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"].float())
        self.register_buffer("std", sd["obs_normalizer._std"].float())
        idx = sorted({int(m.group(1)) for k in sd if (m := re.match(r"mlp\.(\d+)\.weight", k))})
        self.layers = torch.nn.ModuleList()
        for i in idx:
            w, b = sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            self.layers.append(lin)

    def forward(self, obs):
        x = (obs - self.mean) / (self.std + EPS)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.nn.functional.elu(x)
        return x


def verify(actor, sd):
    from rsl_rl.models.mlp_model import MLPModel
    from tensordict import TensorDict

    obs_dim, act_dim = actor.layers[0].in_features, actor.layers[-1].out_features
    hidden = [layer.out_features for layer in actor.layers[:-1]]
    td = TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1])
    ref = MLPModel(td, {"actor": ["policy"]}, "actor", act_dim, hidden, "elu", True,
                   {"class_name": "rsl_rl.modules.distribution.GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})
    ref.load_state_dict(sd)
    ref.eval()
    x = torch.randn(64, obs_dim) * 2
    with torch.no_grad():
        err = (ref(TensorDict({"policy": x}, batch_size=[64])) - actor(x)).abs().max().item()
    if err > 1e-5:
        raise SystemExit(f"ONNX actor disagrees with rsl_rl by {err}")
    return err


def export(ckpt, check=False):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
    actor = Actor(sd).eval()
    obs_dim = actor.layers[0].in_features
    out = ckpt.parent / "policy.onnx"
    torch.onnx.export(actor, (torch.zeros(1, obs_dim),), out, input_names=["obs"], output_names=["actions"],
                      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17, dynamo=False)
    note = f", matches rsl_rl (max err {verify(actor, sd):.1e})" if check else ""
    print(f"{out}: obs {obs_dim} -> actions {actor.layers[-1].out_features}{note}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    for run in a.runs:
        for pdir in sorted(run.glob("policy_*")):
            cks = sorted(pdir.glob("model_*.pt"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
            if cks:
                export(cks[-1], a.verify)
