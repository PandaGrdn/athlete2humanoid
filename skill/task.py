"""Semantic task spec and reward: the video's action as an ordered list of stages (stand, squat,
jump, land), each with a success predicate on the robot's own body, instead of poses to copy.

    python -m skill.task outputs/remote_latent/clips/squat_jump    # -> task_spec.json + self-check

- compile_spec(): stages from the retarget's kinematics. The retarget is already G1-sized, so the
  targets (squat depth, jump clearance) are in the robot's proportions, not the person's; jump
  clearance is also clipped to what the G1 can do. Muse's skill_spec.json (muse/skill_reader.py),
  when present, names the stages and is cross-checked against the kinematics.
- StageMachine: a batched reward machine (torch). Per env: the current stage. The policy is paid
  for progress toward that stage's goal (potential-based, so going back and forth earns nothing),
  a one-off bonus when the stage's predicate holds long enough (for a jump, scaled by how close
  the apex came to the target band), and a bonus for finishing the sequence. Progress is gated on
  staying upright, and a fall refunds the attempt (everything since the last completed stand). Timing and poses are free: only the order of the stages counts. Its stage is
  also the policy's observation, so the reward stays Markov ("squat now", "jump now").
- evaluate(): the same machine over a rollout -> semantic success (stages completed in order, jump
  heights, squat depths, falls) plus Muse's success criteria where the simulator can measure them.

Body positions are mjlab's G1 tracking bodies (latent/reward_net.py BODIES), world frame, z-up.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PELVIS, L_ANKLE, R_ANKLE, TORSO, L_HAND, R_HAND = 0, 3, 6, 7, 10, 13
STAND, SQUAT, JUMP, LAND = range(4)
KINDS = ("stand", "squat", "jump", "land")
MAX_STAGES = 16  # observation size: one-hot stage, padded
MAX_CLEARANCE = 0.30  # m of foot clearance the G1 is asked for at most, whatever the person jumped
FALL_Z, FALL_TILT = 0.35, 1.4  # torso height (m) / tilt (rad): experiments/robustness_eval.py's fall rule
WEIGHTS = {"progress": 1.0, "stage": 1.0, "jump": 2.0, "done": 4.0}  # in seconds of unit reward
MIN_FLIGHT = 0.15  # s
LAND_WINDOW = 0.6  # s after touchdown in which a crouch is the landing, not a new squat
DOWN, AIR = 0.05, 0.06  # m above standing ankle height: a foot on the ground / both feet in the air
# "Standing" is back above this fraction of the retarget's standing pelvis height. The simulated G1
# stands on bent knees; the proportional retarget does too (the uniform one stood ~7 cm taller).
STAND_FRAC = 0.92
UPRIGHT = 0.85  # cos of torso lean from vertical (~32 deg) counted as upright
HOLD = {STAND: 0.3, SQUAT: 0.05, LAND: 0.15}  # s a predicate must hold (a jump's is its min flight)


def features(pos, foot0):
    """(..., bodies, 3) -> pelvis height, lower and higher foot above the ground, torso uprightness
    (cos of the pelvis->torso axis from vertical)."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    feet = pos[..., [L_ANKLE, R_ANKLE], 2] - foot0
    axis = pos[..., TORSO, :] - pos[..., PELVIS, :]
    up = axis[..., 2] / axis.norm(dim=-1).clamp(min=1e-6)
    return pos[..., PELVIS, 2], feet.min(-1).values, feet.max(-1).values, up


def _runs(labels, fps, min_s):
    """Run-length encode per-frame labels, dropping runs shorter than min_s[label] (they merge
    into their neighbours) -> [(label, start_frame, end_frame)] with neighbours of a label merged."""
    runs, i = [], 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if labels[i] != "mid" and (j - i) / fps >= min_s[labels[i]]:
            if runs and runs[-1][0] == labels[i]:
                runs[-1] = (labels[i], runs[-1][1], j)
            else:
                runs.append((labels[i], i, j))
        i = j
    return runs


def compile_spec(pos, fps, skill=None, max_clearance=MAX_CLEARANCE, name=None):
    """pos: (T, bodies, 3) retarget body positions (G1-sized) at fps -> task spec (JSON-able)."""
    pos = np.asarray(pos, np.float32)
    foot0 = float(np.percentile(pos[:, [L_ANKLE, R_ANKLE], 2].min(1), 5))
    z, lo, hi, up = (x.numpy() for x in features(pos, foot0))
    ground = lo < 0.10
    h0 = float(np.percentile(z[ground], 75))
    standing = ground & (z > 0.95 * h0)
    # The retarget's feet can hover several cm (pose noise): its own thresholds, for detecting
    # stages here and for its self-check. The simulated robot is held to the physical ones.
    down_ref = float(np.clip(np.percentile(hi[standing], 95) + 0.02, DOWN, 0.15)) if standing.any() else DOWN
    air_ref = max(AIR, down_ref + 0.01, 0.10)
    drop = h0 - z
    labels = np.where(lo > air_ref, "air", np.where(drop > 0.1 * h0, "low",
                      np.where(drop < 0.05 * h0, "stand", "mid")))
    # A flight shorter than MIN_FLIGHT is a pose glitch, not a jump (ballistic 0.15 s ~ 3 cm).
    runs = _runs(labels, fps, {"air": MIN_FLIGHT, "low": 0.15, "stand": 0.15})

    stages = []
    def add(kind, s, e, **params):
        if stages and stages[-1]["kind"] == KINDS[kind] and kind != JUMP:
            stages[-1]["end_s"] = round(e / fps, 3)
            stages[-1].update(params)
            return
        stages.append({"kind": KINDS[kind], "start_s": round(s / fps, 3), "end_s": round(e / fps, 3), **params})

    last_air = -10**9
    for lab, s, e in runs:
        if lab == "air":
            flight = (e - s) / fps
            stages.append({"kind": "jump", "start_s": round(s / fps, 3), "end_s": round(e / fps, 3),
                           "clearance": round(float(min(lo[s:e].max(), max_clearance)), 3),
                           "ref_clearance": round(float(lo[s:e].max()), 3),
                           "min_air_s": round(float(np.clip(0.5 * flight, 0.05, 0.12)), 3)})
            add(LAND, e, e)
            last_air = e
        elif lab == "low" and (s - last_air) / fps < LAND_WINDOW:  # absorbing the landing
            if stages[-1]["kind"] == "stand":  # passed through standing height on the way down
                stages.pop()
            add(LAND, s, e, absorb_z=round(float(z[s:e].min()), 3))
        elif lab == "low":
            depth = float(h0 - z[s:e].min())
            add(SQUAT, s, e, depth=round(depth, 3), z_goal=round(h0 - 0.75 * depth, 3))
        else:
            add(STAND, s, e)
    if not stages:
        raise ValueError("no stages found: the motion never leaves standing")
    if len(stages) > MAX_STAGES:
        raise ValueError(f"{len(stages)} stages; at most {MAX_STAGES}")

    spec = {"name": name, "fps": fps, "duration_s": round(len(pos) / fps, 3),
            "h0": round(h0, 3), "foot0": round(foot0, 3), "z_low": round(0.7 * h0, 3),
            "down": DOWN, "air": AIR, "down_ref": round(down_ref, 3), "air_ref": round(air_ref, 3), "weights": dict(WEIGHTS), "stages": stages,
            "summary": summarize(stages), "warnings": []}
    if skill:
        attach_skill(spec, skill)
    return spec


def summarize(stages):
    kinds = [s["kind"] for s in stages]
    jumps, squats = kinds.count("jump"), kinds.count("squat")
    return (f"{jumps} jump{'s' * (jumps != 1)}" if jumps else f"{squats} squat{'s' * (squats != 1)}") + \
        ": " + " -> ".join(kinds)


def attach_skill(spec, skill):
    """Muse's skill spec -> stage labels (the phase overlapping each stage most), the action label,
    its success criteria for evaluation, and warnings where Muse and the kinematics disagree."""
    phases = skill.get("phases", [])
    for st in spec["stages"]:
        best = max(phases, default=None, key=lambda p: min(p["end_s"], st["end_s"] + 0.2)
                   - max(p["start_s"], st["start_s"] - 0.2))
        if best:
            st["label"], st["description"] = best["name"], best.get("description", "")
    spec["action"] = skill.get("action", {}).get("label")
    spec["criteria"] = skill.get("success_criteria", [])
    tempo, kinds = skill.get("tempo", {}), [s["kind"] for s in spec["stages"]]
    if tempo.get("airborne") and "jump" not in kinds:
        spec["warnings"].append("Muse says the motion is airborne but the retarget never leaves the ground")
    if tempo.get("airborne") is False and "jump" in kinds:
        spec["warnings"].append("Muse says the motion stays grounded but the retarget has a flight phase")
    reps = kinds.count("jump") or kinds.count("squat")
    if tempo.get("reps") and tempo["reps"] != reps:
        spec["warnings"].append(f"Muse counts {tempo['reps']} reps, the kinematics {reps}")


class StageMachine:
    """Batched reward machine over a task spec. step(pos) -> reward per env (to be multiplied by
    the reward term's weight * dt, so bonuses are paid as weight-seconds of unit reward)."""

    def __init__(self, spec, num_envs, device="cpu", dt=0.02, weights=None):
        self.spec, self.n, self.dev, self.dt = spec, num_envs, device, dt
        self.w = {**WEIGHTS, **spec.get("weights", {}), **(weights or {})}
        st = spec["stages"]
        self.S = len(st)
        self.kind = torch.tensor([KINDS.index(s["kind"]) for s in st], device=device)
        h0 = spec["h0"]
        goal = [h0 * STAND_FRAC if s["kind"] == "stand" else s.get("z_goal", 0.0) if s["kind"] == "squat"
                else s.get("clearance", 0.0) for s in st]
        self.goal = torch.tensor(goal, dtype=torch.float32, device=device)
        self.hold_need = torch.tensor([s.get("min_air_s", HOLD.get(KINDS.index(s["kind"]), 0.1))
                                       for s in st], dtype=torch.float32, device=device)
        self.start = torch.tensor([s["start_s"] for s in st], dtype=torch.float32, device=device)
        # A land stage that follows a jump pays the jump's quality bonus, against that jump's goal.
        self.jump_goal = torch.tensor([st[i - 1].get("clearance", 0.0) if i and st[i - 1]["kind"] == "jump"
                                       and s["kind"] == "land" else 0.0 for i, s in enumerate(st)],
                                      dtype=torch.float32, device=device)
        z = lambda dtype=torch.float32: torch.zeros(num_envs, dtype=dtype, device=device)
        self.stage, self.completed = z(torch.long), z(torch.long)
        self.hold, self.phi, self.apex, self.z_prev, self.p, self.banked = z(), z(), z(), z(), z(), z()
        self.done = z(torch.bool)
        self.stats = {"advances": z(torch.long), "jump_q": z()}
        # Per env goals: the spec's unless set_goals() commands others (skill/commands.py).
        self.goals = self.goal.expand(num_envs, self.S).clone()
        self.jump_goals = self.jump_goal.expand(num_envs, self.S).clone()
        self.n_stages = torch.full((num_envs,), self.S, dtype=torch.long, device=device)
        self.rep_ends = rep_ends(spec)
        self.nominal = nominal_command(spec)
        self.command = torch.tensor([self.nominal[k] for k in COMMANDS], device=device).expand(num_envs, -1).clone()
        self.extra_depth = z()  # m a commanded squat goes below the spec's deepest (stability margins)
        self.q_min, self.steps = torch.ones(num_envs, device=device), z(torch.long)

    def set_goals(self, env_ids, depth=None, clearance=None, reps=None):
        """Command other targets for env_ids (tensors of len(env_ids), or None to keep): squat depth
        (m below standing; the predicate asks for 3/4 of it, as compile_spec does), jump clearance (m)
        and number of reps (the episode's sequence ends after that many). Same as command_spec()."""
        kind = self.kind
        if depth is not None:
            depth = torch.as_tensor(depth, dtype=torch.float32, device=self.dev)
            sq = (kind == SQUAT)[None]
            self.goals[env_ids] = torch.where(sq, self.spec["h0"] - 0.75 * depth[:, None], self.goals[env_ids])
            self.command[env_ids, 0] = depth
            self.extra_depth[env_ids] = (depth - self.nominal["depth_max"]).clamp(min=0)
        if clearance is not None:
            c = torch.as_tensor(clearance, dtype=torch.float32, device=self.dev)[:, None]
            self.goals[env_ids] = torch.where((kind == JUMP)[None], c, self.goals[env_ids])
            self.jump_goals[env_ids] = torch.where((self.jump_goal > 0)[None], c, self.jump_goals[env_ids])
            self.command[env_ids, 1] = c[:, 0]
        if reps is not None:
            reps = torch.as_tensor(reps, dtype=torch.long, device=self.dev).clamp(1, len(self.rep_ends))
            self.n_stages[env_ids] = torch.tensor(self.rep_ends, device=self.dev)[reps - 1]
            self.command[env_ids, 2] = reps.float()

    def _eval(self, stage, goals, z, lo, hi, up, vz):
        """Progress in [0, 1] toward, and whether the predicate holds for, each env's stage."""
        s = self.spec
        k, a = self.kind[stage], goals.gather(1, stage[:, None]).squeeze(1)
        feet_down = hi < s["down"]
        h0, zl = s["h0"], s["z_low"]
        p = torch.stack([((z - zl) / (a - zl).clamp(min=1e-3)).clamp(0, 1),
                         ((h0 - z) / (h0 - a).clamp(min=1e-3)).clamp(0, 1),
                         (lo / a.clamp(min=1e-3)).clamp(0, 1),
                         feet_down.float() * ((up - 0.6) / 0.25).clamp(0, 1)], -1)
        sat = torch.stack([(z >= a) & feet_down & (up > UPRIGHT),
                           (z <= a) & feet_down,
                           lo > s["air"],
                           feet_down & (up > UPRIGHT) & (vz.abs() < 0.5)], -1)
        idx = k[:, None]
        return p.gather(1, idx).squeeze(1), sat.gather(1, idx).squeeze(1)

    def stage_at(self, t):
        """The stage the reference is in at time t (s): where an episode started at t begins."""
        return (torch.searchsorted(self.start, t.float().contiguous(), right=True) - 1).clamp(0, self.S - 1)

    def reset(self, env_ids, t, pos):
        """Episodes of env_ids (re)start at reference time t (s) with bodies at pos (len(env_ids), B, 3)."""
        z, lo, hi, up = features(pos, self.spec["foot0"])
        stage = self.stage_at(torch.as_tensor(t, device=self.dev).expand(len(env_ids)))
        # Fewer reps commanded than the reference is into: the sequence is already over.
        n = self.n_stages[env_ids]
        over = stage >= n
        stage = torch.minimum(stage, n - 1)
        p, _ = self._eval(stage, self.goals[env_ids], z, lo, hi, up, torch.zeros_like(z))
        self.stage[env_ids], self.p[env_ids] = stage, p
        self.phi[env_ids] = torch.where(over, n.float(), stage + p)
        for buf in (self.hold, self.apex, self.completed, self.steps, self.banked):
            buf[env_ids] = 0
        self.q_min[env_ids] = 1.0
        self.done[env_ids], self.z_prev[env_ids] = over, z

    def step(self, pos, fell=None):
        """fell: (n,) bool, episodes that end this step in a fall. Potential-based shaping gives a
        terminal state potential 0, so a fall refunds what the attempt earned: everything since the
        last completed stand (a rep counts once the robot has landed and stood back up)."""
        z, lo, hi, up = features(pos, self.spec["foot0"])
        vz, self.z_prev = (z - self.z_prev) / self.dt, z
        p, sat = self._eval(self.stage, self.goals, z, lo, hi, up, vz)
        k = self.kind[self.stage]
        self.apex = torch.where((k == JUMP) | (k == LAND), torch.maximum(self.apex, lo), torch.zeros_like(lo))
        self.hold = torch.where(sat & ~self.done, self.hold + self.dt, torch.zeros_like(self.hold))
        adv = (self.hold >= self.hold_need[self.stage] - 1e-6) & ~self.done

        # Jump quality, paid when the landing after it completes: 1 inside [goal, 1.25 goal], less
        # below it, and falling to 0 at 2x the goal (no reward for launching itself out of control).
        jg = self.jump_goals.gather(1, self.stage[:, None]).squeeze(1)
        ratio = self.apex / jg.clamp(min=1e-3)
        q = torch.where(jg > 0, ratio.clamp(0, 1) - ((ratio - 1.25) / 0.75).clamp(0, 1), torch.zeros_like(ratio))
        bonus = adv * (self.w["stage"] + self.w["jump"] * q)

        self.completed += adv.long()
        new_stage = self.stage + adv.long()
        just_done = new_stage >= self.n_stages
        self.done |= just_done
        self.stage = torch.minimum(new_stage, self.n_stages - 1)
        self.p = torch.where(adv, torch.zeros_like(p), p)
        self.hold = torch.where(adv, torch.zeros_like(self.hold), self.hold)
        bonus = bonus + just_done * self.w["done"] * self.completed / self.n_stages

        phi = torch.where(self.done, self.n_stages.float(), self.stage + self.p)
        d, self.phi = phi - self.phi, phi
        gate = ((up - 0.6) / 0.3).clamp(0, 1)
        prog = torch.where(d > 0, d * gate, d) * self.w["progress"]
        self.stats["advances"] += adv.long()
        self.stats["jump_q"] = torch.where(adv & (jg > 0), q, self.stats["jump_q"])
        self.q_min = torch.where(adv & (jg > 0), torch.minimum(self.q_min, q), self.q_min)
        self.steps += 1
        r = prog + bonus
        self.banked = self.banked + r
        if fell is not None:
            r = torch.where(fell, r - self.banked, r)
            self.banked = torch.where(fell, torch.zeros_like(r), self.banked)
        rep_done = (adv & (k == STAND)) | just_done
        self.banked = torch.where(rep_done, torch.zeros_like(r), self.banked)
        return r / self.dt

    def obs(self):
        """(n, MAX_STAGES + 2): one-hot stage, progress in it, finished."""
        one = torch.nn.functional.one_hot(self.stage, MAX_STAGES).float()
        return torch.cat([one, self.p[:, None], self.done.float()[:, None]], -1)

    def command_obs(self):
        """(n, 3): the commanded depth, clearance and reps, normalised to [-1, 1] over COMMAND_RANGES
        (fixed G1 ranges, not the clip's, so the numbers mean the same thing for every clip)."""
        lo = torch.tensor([COMMAND_RANGES[k][0] for k in COMMANDS], device=self.dev)
        hi = torch.tensor([COMMAND_RANGES[k][1] for k in COMMANDS], device=self.dev)
        return (2 * (self.command - lo) / (hi - lo) - 1).clamp(-1, 1)


COMMANDS = ("depth", "clearance", "reps")
# What a G1 can be asked for (m below standing / of foot clearance / reps): the commanded-target
# task samples inside these, and they normalise the command observation.
COMMAND_RANGES = {"depth": (0.10, 0.32), "clearance": (0.08, MAX_CLEARANCE), "reps": (1, 4)}


def rep_ends(spec):
    """Number of stages that make 1, 2, ... reps: a rep ends with its landing (or, without jumps,
    its squat) and the stand after it; the last rep keeps the rest of the sequence."""
    kinds = [s["kind"] for s in spec["stages"]]
    key = "land" if "jump" in kinds else "squat"
    ends = [i + 1 + (i + 1 < len(kinds) and kinds[i + 1] == "stand") for i, k in enumerate(kinds) if k == key]
    return (ends[:-1] + [len(kinds)]) if ends else [len(kinds)]


def nominal_command(spec):
    """The clip's own targets: mean squat depth (and the deepest), mean jump clearance, its reps."""
    st = spec["stages"]
    d = [s["depth"] for s in st if s["kind"] == "squat" and "depth" in s]
    c = [s["clearance"] for s in st if s["kind"] == "jump"]
    return {"depth": float(np.mean(d)) if d else 0.0, "depth_max": max(d, default=0.0),
            "clearance": float(np.mean(c)) if c else 0.0, "reps": len(rep_ends(spec))}


def command_spec(spec, depth=None, clearance=None, reps=None):
    """The spec with commanded targets in place of the clip's (for evaluate()): every squat to
    `depth`, every jump to `clearance`, and the sequence cut after `reps` reps."""
    out = json.loads(json.dumps(spec))
    for s in out["stages"]:
        if s["kind"] == "squat" and depth is not None:
            s["depth"], s["z_goal"] = round(float(depth), 4), round(spec["h0"] - 0.75 * float(depth), 4)
        if s["kind"] == "jump" and clearance is not None:
            s["clearance"] = round(float(clearance), 4)
    if reps is not None:
        ends = rep_ends(spec)
        out["stages"] = out["stages"][: ends[int(np.clip(reps, 1, len(ends))) - 1]]
    out["summary"] = summarize(out["stages"])
    out["command"] = {"depth": depth, "clearance": clearance, "reps": reps}
    return out


def fallen(pos):
    z_t = torch.as_tensor(pos, dtype=torch.float32)[..., TORSO, 2]
    *_, up = features(pos, 0.0)
    return (z_t < FALL_Z) | (torch.acos(up.clamp(-1, 1)) > FALL_TILT)


def _longest(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def measure(spec, pos, fps):
    """Muse skill_reader METRICS on a trajectory (only those computable from body positions)."""
    z, lo, hi, up = (x.numpy() for x in features(pos, spec["foot0"]))
    pos = np.asarray(pos)
    hands = pos[:, [L_HAND, R_HAND], 2].max(1) - pos[:, PELVIS, 2]
    return {"pelvis_drop_m": float(spec["h0"] - z.min()), "pelvis_rise_m": float(z.max() - spec["h0"]),
            "airborne_s": _longest(lo > spec["air"]) / fps, "foot_height_max_m": float(hi.max()),
            "hand_height_max_m": float(hands.max()),
            "torso_pitch_max_deg": float(np.degrees(np.arccos(np.clip(up, -1, 1))).max()),
            "falls": int(fallen(torch.as_tensor(pos)).any())}


FALL_ONSET = 0.3  # s before the fall rule fires that are the collapse, not the movement


def achieved(spec, pos, trace):
    """What the rollout actually did per rep: each squat's depth (standing pelvis height minus the
    lowest pelvis while in that squat or the stage after it, so the countermovement's bottom) and
    each jump's clearance (highest lower foot during the jump and its landing). Only stages the
    rollout completed count; trace (stage per frame) stops at the fall's onset."""
    z, lo, _, _ = (x.numpy() for x in features(pos, spec["foot0"]))
    trace, kinds = np.asarray(trace), [s["kind"] for s in spec["stages"]]
    out = {"squat_depths": [], "jump_clearances": []}
    for i, k in enumerate(kinds):
        f = (trace == i) | (trace == i + 1)
        if k in ("squat", "jump") and (trace == i).any() and (trace > i).any():
            out["squat_depths" if k == "squat" else "jump_clearances"].append(
                round(float(spec["h0"] - z[f].min() if k == "squat" else lo[f].max()), 3))
    return out


def machine(spec, num_envs, device="cpu", dt=0.02, weights=None):
    """The reward machine for a spec: skill/keypose.py's for "keypose" specs (any action), else StageMachine."""
    if spec.get("type") == "keypose":
        from skill.keypose import KeyposeMachine
        return KeyposeMachine(spec, num_envs, device, dt, weights)
    return StageMachine(spec, num_envs, device, dt, weights)


def evaluate(spec, pos, fps):
    """A rollout's body positions (T, bodies, 3) from the start of the motion -> semantic success."""
    m = machine(spec, 1, dt=1.0 / fps)
    pos_t = torch.as_tensor(np.asarray(pos), dtype=torch.float32)
    m.reset(torch.tensor([0]), 0.0, pos_t[:1])
    done_at, jump_q, fell_at, trace = [], [], None, [int(m.stage[0])]
    for t in range(1, len(pos_t)):
        trace.append(int(m.stage[0]))  # the stage this frame is scored against
        before = int(m.completed[0])
        m.step(pos_t[t:t + 1])
        if int(m.completed[0]) > before:
            done_at.append(round(t / fps, 3))
            if m.jump_goal[min(before, m.S - 1)] > 0:
                jump_q.append(round(float(m.stats["jump_q"][0]), 3))
        if fallen(pos_t[t:t + 1])[0]:
            fell_at = round(t / fps, 3)
            break
    n = int(m.completed[0]) + int(getattr(m, "skipped", torch.zeros(1, dtype=torch.long))[0])  # stages passed
    met = measure(spec, pos_t[: (len(pos_t) if fell_at is None else int(fell_at * fps) + 1)].numpy(), fps)
    n_ok = len(trace) if fell_at is None else max(0, int(round((fell_at - FALL_ONSET) * fps)))
    met |= achieved(spec, pos_t[:n_ok], trace[:n_ok])
    kinds = [s["kind"] for s in spec["stages"]]
    met["reps"] = kinds[:n].count("jump" if "jump" in kinds else "squat")
    met["cycle_s"] = (len(pos_t) / fps) / max(met["reps"], 1)
    crit = [{**c, "measured": met.get(c["metric"]),
             "pass": None if met.get(c["metric"]) is None else
             (met[c["metric"]] >= c["value"] if c["op"] == ">=" else met[c["metric"]] <= c["value"])}
            for c in spec.get("criteria", [])]
    return {"stages_total": m.S, "stages_completed": n, "completed_at_s": done_at,
            "next_stage": None if n >= m.S else spec["stages"][n].get("label", spec["stages"][n].get("name", spec["stages"][n]["kind"])),
            "success": bool(m.done[0]) and fell_at is None,
            "stages_skipped": int(getattr(m, "skipped", torch.zeros(1, dtype=torch.long))[0]), "fell_at_s": fell_at, "jump_quality": jump_q,
            "metrics": met, "criteria": crit}


def body_positions_from_qpos(qpos):
    """(T, 36) G1 qpos -> (T, bodies, 3), via latent.distill (needs mujoco and the G1 assets)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import render
    from latent.distill import body_positions
    return body_positions(render.load_g1()[0], np.asarray(qpos))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip", help="a clip folder from pipeline.py (qpos.npy, meta.json; skill_spec.json optional)")
    ap.add_argument("--out", help="default <clip>/task_spec.json")
    ap.add_argument("--skill", help="Muse skill_spec.json (default <clip>/skill_spec.json if present)")
    ap.add_argument("--max-clearance", type=float, default=MAX_CLEARANCE)
    a = ap.parse_args(argv)
    clip = Path(a.clip)
    fps = json.loads((clip / "meta.json").read_text())["fps"]
    pos = body_positions_from_qpos(np.load(clip / "qpos.npy"))
    skill_path = Path(a.skill) if a.skill else clip / "skill_spec.json"
    skill = json.loads(skill_path.read_text()) if skill_path.exists() else None
    spec = compile_spec(pos, fps, skill, a.max_clearance, name=clip.name)
    # The retarget itself must pass its own spec (at its own noise level).
    spec["self_check"] = evaluate({**spec, "down": spec["down_ref"], "air": spec["air_ref"]}, pos, fps)
    out = Path(a.out or clip / "task_spec.json")
    out.write_text(json.dumps(spec, indent=1))
    sc = spec["self_check"]
    print(f"{out}: {spec['summary']}; retarget completes {sc['stages_completed']}/{sc['stages_total']}"
          + "".join(f"\n  warning: {w}" for w in spec["warnings"]))


if __name__ == "__main__":
    main()
