"""Commanded targets: every episode asks for its own squat depth, jump clearance and (optionally)
number of reps, around the clip's values and inside what the G1 can do (task.COMMAND_RANGES), so
the policy learns "squat this deep, jump this high" instead of the one trajectory in the clip.

- CommandSampler.sample(env_ids): new targets for episodes that (re)start, written into the
  StageMachine (set_goals), which also gives them to the policy (StageMachine.command_obs).
- Curriculum: the ranges start narrow around the clip's values (level 0) and widen toward the full
  ranges (level 1) while the commanded task succeeds. Success is per finished episode: every stage
  done, no fall, and every jump within the quality band (q >= q_success: apex at least 0.7x the
  commanded clearance and at most 1.8x). After every `window` finished episodes the level goes up
  by `step` if the success rate is above `up`, down if it is below `down`. A fraction `explore` of
  episodes always samples the full ranges, so the policy sees them from the start.

Pure torch (no mjlab), so the curriculum is unit-tested on its own; mjlab_tasks.TaskCmdReward wires
it into the A2H-TaskCmd task.
"""

import torch

from skill.task import COMMAND_RANGES

CURRICULUM = {"start": 0.15, "step": 0.1, "up": 0.6, "down": 0.3, "window": 2048, "explore": 0.2,
              "q_success": 0.7, "level": 0.0, "reps": False}


class CommandSampler:
    def __init__(self, machine, seed=None, **cfg):
        self.m, self.cfg = machine, {**CURRICULUM, **cfg}
        unknown = set(cfg) - set(CURRICULUM) - {"ranges"}
        if unknown:
            raise ValueError(f"unknown command settings {sorted(unknown)}")
        self.ranges = {**COMMAND_RANGES, **cfg.get("ranges", {})}
        nom = machine.nominal
        # Only what the clip does is commanded: a squat-only clip gets no clearance command.
        self.keys = [k for k in ("depth", "clearance") if nom[k] > 0] + (["reps"] if self.cfg["reps"] else [])
        self.nom = {k: min(max(float(nom[k]), self.ranges[k][0]), self.ranges[k][1]) for k in ("depth", "clearance")}
        self.level = float(self.cfg["level"])
        self.n_ok = self.n_ep = 0
        self.rate = float("nan")  # success rate of the last full window
        self.fixed = None
        self.gen = torch.Generator(device="cpu")
        if seed is not None:
            self.gen.manual_seed(seed)

    def bounds(self, level=None):
        """(lo, hi) per commanded quantity at this level: start x the distance to the range's ends
        on each side of the clip's value at level 0, the whole range at level 1."""
        lv = self.level if level is None else level
        w = self.cfg["start"] + (1 - self.cfg["start"]) * min(max(lv, 0.0), 1.0)
        out = {}
        for k in ("depth", "clearance"):
            lo, hi = self.ranges[k]
            out[k] = (self.nom[k] - w * (self.nom[k] - lo), self.nom[k] + w * (hi - self.nom[k]))
        R = self.m.nominal["reps"]
        out["reps"] = (max(1, R - round(w * (R - 1))), R)
        return out

    def fix(self, depth=None, clearance=None, reps=None):
        """Evaluation: these targets (per env tensors, or None for the clip's) instead of sampling."""
        self.fixed = {"depth": depth, "clearance": clearance, "reps": reps}

    def sample(self, env_ids=None):
        dev = self.m.dev
        ids = torch.arange(self.m.n, device=dev) if env_ids is None else torch.as_tensor(env_ids, device=dev)
        if not len(ids):
            return
        if self.fixed is not None:
            f = {k: (None if v is None else torch.as_tensor(v, device=dev)[ids]) for k, v in self.fixed.items()}
            self.m.set_goals(ids, **f)
            return
        n = len(ids)
        full = (torch.rand(n, generator=self.gen) < self.cfg["explore"]).to(dev)
        b, bf = self.bounds(), self.bounds(1.0)
        vals = {}
        for k in self.keys:
            u = torch.rand(n, generator=self.gen).to(dev)
            if k == "reps":
                lo = torch.where(full, torch.tensor(bf[k][0], device=dev), torch.tensor(b[k][0], device=dev))
                vals[k] = (lo + (u * (b[k][1] - lo + 1)).floor()).long().clamp(max=b[k][1])
            else:
                lo = torch.where(full, torch.tensor(bf[k][0], device=dev), torch.tensor(b[k][0], device=dev))
                hi = torch.where(full, torch.tensor(bf[k][1], device=dev), torch.tensor(b[k][1], device=dev))
                vals[k] = lo + u * (hi - lo)
        self.m.set_goals(ids, **vals)

    def success(self, env_ids, fell):
        """Did the episodes of env_ids (about to restart) do the commanded task?"""
        m = self.m
        return m.done[env_ids] & (m.completed[env_ids] > 0) & ~fell & (m.q_min[env_ids] >= self.cfg["q_success"])

    def record(self, env_ids, fell):
        """Episodes of env_ids end (fell: per env bool). Updates the curriculum level."""
        ids = torch.as_tensor(env_ids, device=self.m.dev)
        if not len(ids):
            return
        fell = torch.as_tensor(fell, device=self.m.dev).expand(len(ids))
        ran = self.m.steps[ids] > 0  # the very first reset ends no episode
        ok = self.success(ids, fell) & ran
        self.n_ok += int(ok.sum())
        self.n_ep += int(ran.sum())
        if self.n_ep >= self.cfg["window"]:
            self.rate = self.n_ok / self.n_ep
            if self.rate > self.cfg["up"]:
                self.level = min(1.0, self.level + self.cfg["step"])
            elif self.rate < self.cfg["down"]:
                self.level = max(0.0, self.level - self.cfg["step"])
            self.n_ok = self.n_ep = 0

    def log(self):
        return {"level": self.level, "success": self.rate,
                **{f"{k}_lo": v[0] for k, v in self.bounds().items() if k in self.keys},
                **{f"{k}_hi": v[1] for k, v in self.bounds().items() if k in self.keys}}
