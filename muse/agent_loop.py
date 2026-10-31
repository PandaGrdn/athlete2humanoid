"""Reward agent loop: Muse Spark tunes a task policy's reward over short training rounds, with tool calls.

    MODAL_PROFILE=a2h python -m muse.agent_loop outputs/remote_latent_s3_jacks --target jumping_jacks \\
        --video clips/jumping_jacks.mov --out outputs/agent_jacks --rounds 3 --iters 300 --branch muse-agent-loop
    python -m muse.agent_loop ... --dry-run     # offline: fixture responses and canned rounds, no GPU, no key

Round 0 scores the run's task policy as it is. Then each round: Muse (Responses API; previous_response_id
carries its reasoning from turn to turn) gets the round's numbers and a rollout video, and calls tools:
set_param (one setting, within BOUNDS, at most MAX_CHANGES per round), get_config, and, to end its turn,
train_next_round or finish. Each round trains two arms for --iters PPO iterations from their own last
checkpoints, on Modal (experiments/agent_round.py through scripts/remote_run.py agentround): "agent" with
Muse's settings and "base" with the run's defaults, the same budget, and scores both alike on the fixed
task spec. train_next_round's tool output is the agent arm's result; its rollout video follows as a
user message. Muse never sees the base arm: it is the measurement of whether the agent helped.

What Muse can change (PARAMS): the stage machine's weights (written into the spec the agent arm trains on;
every arm is scored on the original), the reward terms' weights, the stability terms and the two
fall-termination thresholds of the task-first tasks (mjlab_tasks.reward_config / task_config).

Files in --out: state.json (resumable: rounds already scored on Modal are not re-run), agent_log.jsonl
(every request's size, response, tool call and result), rounds/r<k>/inputs/ (what each round was given),
report.json and report.md (per round, agent vs base; tokens and cost).
"""

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muse import media  # noqa: E402
from muse.client import FIXTURES, MuseClient, data_url, now, read_json, redact, write_json  # noqa: E402
from muse.reward_agent import STABILITY  # noqa: E402
from muse.schema import INT, NUM, STR, enum, obj  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# path -> (default, low, high, what it does). task.*: the stage machine's weights (skill/task.py WEIGHTS, in the
# spec); the rest: mjlab_tasks.task_config() / reward_config() for A2H-Task-Flat-Unitree-G1.
PARAMS = {
    "task.progress": (1.0, 0.25, 4.0, "potential-based progress toward the current key moment's pose, per unit of the "
                                      "distance closed (gated on staying upright)"),
    "task.stage": (1.0, 0.25, 4.0, "bonus for each key moment (stage) completed, in order"),
    "task.jump": (2.0, 0.5, 6.0, "bonus scale for jump quality on flight stages (no effect without jumps)"),
    "task.done": (4.0, 1.0, 12.0, "bonus for completing every stage, scaled by the share done"),
    "task.hold": (1.0, 0.0, 4.0, "per second of holding the end pose after finishing"),
    "task_weight": (2.0, 0.5, 6.0, "weight of the whole stage-machine reward term"),
    "style_weight": (1.0, 0.0, 3.0, "weight of the V-JEPA latent style term (looks like the video), "
                                    "time-weighted toward the key moments"),
    "tracking_scale": (0.1, 0.02, 0.5, "scale of mjlab's six retarget-tracking terms (copy the retarget), time-weighted"),
    "stability.alive": (1.0, 0.0, 5.0, "reward per second not fallen"),
    "stability.upright": (-2.0, -10.0, 0.0, "penalty per rad the torso leans past the clip's own maximum + margin"),
    "stability.feet_slip": (-0.5, -3.0, 0.0, "penalty on the squared speed of feet on the floor"),
    "stability.action_rate": (-0.1, -1.0, -0.01, "penalty on action changes between steps (smoothness)"),
    "stability.push_scale": (1.5, 1.0, 2.5, "how much stronger than mjlab's default the random training pushes grow"),
    "terminations.fall_tilt": (0.35, 0.15, 1.0, "an episode ends (fallen) when the torso leans this many rad past the "
                                                "clip's own maximum lean"),
    "terminations.fall_height": (0.2, 0.05, 0.5, "... or sinks this many m below the clip's lowest torso height"),
}
MAX_CHANGES = 3  # settings changed per round
MAX_STEPS = 6  # responses per agent turn before it must have decided
METRICS = ("clean_task_success", "clean_stages", "push_task_success", "init_task_success", "vjepa_aligned")
PRICE = {"input": 1.25, "cached": 0.15, "output": 4.25}  # $ per M tokens, muse-spark-1.3 standard

TOOLS = [
    {"type": "function", "name": "set_param", "strict": True,
     "description": "Change one reward, stability or termination setting for the next round. The value must be "
                    f"within the setting's bounds (get_config); at most {MAX_CHANGES} settings per round.",
     "parameters": obj(path=enum(PARAMS), value=NUM, reason=STR)},
    {"type": "function", "name": "get_config", "strict": True,
     "description": "The current settings, their defaults and bounds, and how many changes are left this round.",
     "parameters": obj()},
    {"type": "function", "name": "train_next_round", "strict": True,
     "description": "End this turn: train the policy for the next round with the current settings (a few hundred "
                    "PPO iterations from its last checkpoint, ~15-20 min), then score it. The result comes back as "
                    "this call's output, and the rollout video after it.",
     "parameters": obj(hypothesis=STR, watch=enum(METRICS))},
    {"type": "function", "name": "finish", "strict": True,
     "description": "Stop tuning: no further round is likely to help. Name the best round so far.",
     "parameters": obj(best_round=INT, summary=STR)},
]

SYSTEM = (
    "You tune the reward of a reinforcement-learning run that teaches a Unitree G1 humanoid robot, in a physics "
    "simulator, the motion in a video of a person. The robot's task is a sequence of key moments measured from the "
    "video (the stages); the policy is scored on a fixed copy of that task, which you cannot change. Between short "
    "training rounds you get the evaluation and a video of the robot's rollout, and you adjust settings with the "
    "tools, each change for a reason you can point to in the numbers or the video. Then call train_next_round. "
    "Describe the robot's motion only; never address or advise the person in the video."
)

OBJECTIVE = """How each round is scored (64 robots per condition: clean, random pushes, noisy starts):
- task_success: the share that finish the task in order without falling. Optional stages may be skipped and
  still count as success, so success alone can be saturated.
- stages_completed: mean stages actually done (skipped optional ones don't count), out of the stages left
  from where the episode starts; fall_rate: the share that fell.
- vjepa_aligned: V-JEPA 2 similarity between the robot's rollout and the person's video (higher is better;
  the retargeted motion itself scores about 0.72).
- single_rollout: the deterministic rollout checked offline: which stage it gave up on, the feature furthest
  outside its tolerance there (error_tol: error over tolerance, 1 = at the edge).
Objective, in order: (1) keep task success and keep falls at zero under pushes and noisy starts; (2) complete
more of the key moments, including optional ones it skips now; (3) look more like the video (vjepa_aligned).
Each round trains {iters} PPO iterations (4096 robots, 50 steps/s) from the last checkpoint; there are {rounds}
rounds. Changes take effect from the next round; the critic was trained on the old reward, so very large
changes cost part of a round re-learning values."""


def defaults(spec):
    """Every PARAMS path at its default, the stage weights from the spec itself."""
    cfg = {p: v[0] for p, v in PARAMS.items()}
    cfg |= {f"task.{k}": float(v) for k, v in spec.get("weights", {}).items() if f"task.{k}" in PARAMS}
    stab = {f"stability.{k}": float(STABILITY[k]) for k in ("alive", "upright", "feet_slip", "action_rate", "push_scale")}
    stab |= {"terminations.fall_tilt": float(STABILITY["fall_tilt"]), "terminations.fall_height": float(STABILITY["fall_height"])}
    return cfg | stab


def config_files(cfg):
    """Flat settings -> (reward_config.json for mjlab_tasks, task_weights.json for the training spec)."""
    stability = {k.split(".", 1)[1]: v for k, v in cfg.items() if k.startswith("stability.")}
    stability |= {k.split(".", 1)[1]: v for k, v in cfg.items() if k.startswith("terminations.")}
    rc = {k: cfg[k] for k in ("task_weight", "style_weight", "tracking_scale")} | {"stability": stability}
    return rc, {k.split(".", 1)[1]: v for k, v in cfg.items() if k.startswith("task.")}


# ---------------------------------------------------------------- tool calls

def function_calls(resp):
    """A Responses API response (dict) -> its function calls, [{"call_id", "name", "arguments"}], in order."""
    return [{"call_id": o["call_id"], "name": o["name"], "arguments": o.get("arguments") or "{}"}
            for o in resp.get("output", []) if o.get("type") == "function_call"]


def output_text(resp):
    return "\n".join(c.get("text", "") for o in resp.get("output", []) if o.get("type") == "message"
                     for c in o.get("content", []) if c.get("type") == "output_text").strip()


class Session:
    """The agent's settings and its changes this round; every tool call goes through call()."""

    def __init__(self, cfg):
        self.cfg, self.round_start, self.changes = dict(cfg), dict(cfg), []

    def new_round(self):
        self.round_start, self.changes = dict(self.cfg), []

    def changed(self):
        return sorted(p for p in self.cfg if self.cfg[p] != self.round_start[p])

    def call(self, name, arguments):
        """-> (tool output dict, decision or None). Bad calls get an error the agent can act on, never an exception."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"arguments are not valid JSON ({e})"}, None
        if name == "set_param":
            return self.set_param(args), None
        if name == "get_config":
            return {"ok": True, "config": self.cfg, "changed_this_round": self.changed(),
                    "changes_left": MAX_CHANGES - len(self.changed()),
                    "settings": {p: {"default": d, "bounds": [lo, hi], "what": w} for p, (d, lo, hi, w) in PARAMS.items()}}, None
        if name == "train_next_round":
            return {"ok": True}, {"tool": name, "hypothesis": str(args.get("hypothesis", "")),
                                  "watch": args.get("watch"), "changed": self.changed()}
        if name == "finish":
            br = args.get("best_round")
            if not isinstance(br, int) or isinstance(br, bool):
                return {"ok": False, "error": "best_round: a round number"}, None
            return {"ok": True}, {"tool": name, "best_round": br, "summary": str(args.get("summary", ""))}
        return {"ok": False, "error": f"unknown tool {name!r}; the tools are " + ", ".join(t["name"] for t in TOOLS)}, None

    def set_param(self, args):
        path, value = args.get("path"), args.get("value")
        if path not in PARAMS:
            return {"ok": False, "error": f"unknown setting {path!r}"}
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            return {"ok": False, "error": f"{path}: value must be a number"}
        _, lo, hi, _ = PARAMS[path]
        if not lo <= value <= hi:
            return {"ok": False, "error": f"{path}: {value} is outside [{lo}, {hi}]; unchanged ({self.cfg[path]})"}
        if value == self.cfg[path]:
            return {"ok": False, "error": f"{path} is already {value}"}
        if path not in self.changed() and len(self.changed()) >= MAX_CHANGES:
            return {"ok": False, "error": f"at most {MAX_CHANGES} settings change per round: already "
                                          f"{', '.join(self.changed())}"}
        old, self.cfg[path] = self.cfg[path], float(value)
        self.changes.append({"path": path, "old": old, "new": float(value), "reason": str(args.get("reason", ""))})
        return {"ok": True, "path": path, "old": old, "new": float(value), "changes_left": MAX_CHANGES - len(self.changed())}


# ---------------------------------------------------------------- what the agent is shown

def summary(arm):
    """agent_round.py's result for one arm -> the compact numbers the agent reads."""
    out = {}
    for cond, r in arm["robust"].items():
        out[cond] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "abandoned"}
        out[cond]["where_failed"] = [{k: a.get(k) for k in ("name", "share", "fell", "feature", "error_tol", "reached")}
                                     for a in (r.get("abandoned") or [])]
    ro = arm["rollout"]
    out["single_rollout"] = {"stages": f"{ro['stages_completed']}/{ro['stages_total']}", "success": ro["success"],
                             "fell_at_s": ro["fell_at_s"], "gave_up_on": ro["abandoned"],
                             "vjepa_aligned": arm.get("rollout_vjepa_aligned")}
    t = arm.get("train")
    if t:
        out["training_last_iteration"] = {k: t[k] for k in ("iteration", "mean_reward", "mean_episode_length", "action_std")}
        out["training_last_iteration"] |= {"reward_terms": t["rewards"], "terminations": t["terminations"]}
    return out


def headline(arm):
    """The few numbers the report compares, from one arm's result."""
    rb = arm["robust"]
    pick = lambda prefix, k: next((v[k] for c, v in rb.items() if c.startswith(prefix)), None)  # noqa: E731
    return {"clean_task_success": pick("clean", "task_success"), "clean_stages": pick("clean", "stages_completed"),
            "clean_fall_rate": pick("clean", "fall_rate"), "push_task_success": pick("push", "task_success"),
            "push_fall_rate": pick("push", "fall_rate"), "init_task_success": pick("init", "task_success"),
            "vjepa_aligned": pick("clean", "vjepa_aligned"), "rollout_stages": arm["rollout"]["stages_completed"]}


def spec_text(spec):
    from skill.spec_schema import stage_label
    lines = [f"{i}. {stage_label(st)} ({st['kind']}{', optional' if st.get('optional') else ''}, "
             f"{st['start_s']:.2f}-{st['end_s']:.2f} s)" for i, st in enumerate(spec["stages"])]
    return f"Task: {spec.get('action_description') or spec.get('name')}\nStages (scored in order):\n" + "\n".join(lines)


def video_message(text, *videos):
    content = [{"type": "input_text", "text": text}]
    content += [{"type": "input_video", "video_url": data_url(v, "video/mp4")} for v in videos if v]
    return {"role": "user", "content": content}


def small_video(src, dst, side=320):
    """An MP4 for Muse (fewer pixels, fewer tokens); None if there is no video."""
    if not src or not Path(src).exists():
        return None
    return media.prepare_video(src, dst, side, required=False)


# ---------------------------------------------------------------- rounds on Modal

def arm_dir(round_dir, arm):
    return Path(round_dir) / f"policy_task_{arm}"


def prepare_inputs(inp, run, spec_f, arms, bank=None):
    """The round's inputs folder for experiments/agent_round.py. arms: {name: (policy folder, cfg or None, eval_only)}.
    bank: an earlier round's eval_bank_dense.npz (saves re-rendering the bank)."""
    if inp.exists():
        shutil.rmtree(inp)
    (inp / "arms").mkdir(parents=True)
    shutil.copytree(run / "clips", inp / "clips")
    shutil.copy(run / "reward.pt", inp / "reward.pt")
    shutil.copy(spec_f, inp / "task_spec.json")
    if bank and Path(bank).exists():
        shutil.copy(bank, inp / "eval_bank_dense.npz")
    for name, (src, cfg, eval_only) in arms.items():
        d = inp / "arms" / name
        d.mkdir()
        ckpts = sorted(Path(src).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        shutil.copy(ckpts[-1], d / ckpts[-1].name)
        if eval_only:
            for f in ("motion.npz", "policy_qpos.npy", "policy.mp4", "policy_metrics.json"):
                shutil.copy(Path(src) / f, d / f)
        if cfg is not None:
            rc, tw = config_files(cfg)
            write_json(d / "reward_config.json", rc)
            write_json(d / "task_weights.json", tw)
    return inp


def modal_round(k, inp, tag, target, iters, branch, eval_only, extra_args=""):
    """scripts/remote_run.py agentround on Modal -> the results folder (outputs/remote_agentround_<tag>)."""
    res = REPO / "outputs" / f"remote_agentround_{tag}"
    if (res / "round.json").exists():
        print(f"round {k}: {res / 'round.json'} already scored, reusing it")
        return res
    args = ("--eval-only " if eval_only else "") + extra_args
    cmd = ["python3", REPO / "scripts" / "remote_run.py", "agentround", "--backend", "modal", "--reuse", inp,
           "--tag", tag, "--target", target, "--iters", iters, "--branch", branch, "--args", args.strip()]
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "a2h"))
    print(f"round {k}: training/scoring on Modal ({res.name})", flush=True)
    subprocess.run(list(map(str, cmd)), check=True, env=env, cwd=REPO)
    if not (res / "round.json").exists():
        raise RuntimeError(f"round {k}: no round.json in {res}; see its runner.log and job.log")
    return res


# ---------------------------------------------------------------- the loop

class Loop:
    def __init__(self, run, target, out, client, run_round, video=None, rounds=3, iters=300, effort="medium"):
        self.run, self.target, self.out = Path(run), target, Path(out)
        self.client, self.run_round, self.video = client, run_round, video
        self.rounds, self.iters, self.effort = rounds, iters, effort
        self.out.mkdir(parents=True, exist_ok=True)
        self.spec_f = self.run / "task_spec.json"
        self.spec = json.loads(self.spec_f.read_text())
        self.state = read_json(self.out / "state.json") or {"rounds": [], "prev": None, "usage": [], "decisions": []}
        self.session = Session(self.state.get("config") or defaults(self.spec))

    def save(self):
        self.state["config"] = self.session.cfg
        write_json(self.out / "state.json", self.state)

    def log(self, row):
        with open(self.out / "agent_log.jsonl", "a") as f:
            f.write(json.dumps({"t": now(), **row}) + "\n")

    def respond(self, items, rnd):
        req = {"instructions": SYSTEM, "input": items, "tools": TOOLS, "reasoning": {"effort": self.effort}}
        if self.state["prev"]:
            req["previous_response_id"] = self.state["prev"]
        for attempt in range(len(RETRY_S) + 1):
            try:
                resp = self.client.respond(**req)
                break
            except Exception as e:  # noqa: BLE001  (only transient API errors are retried)
                if not transient(e) or attempt == len(RETRY_S):
                    raise
                self.log({"round": rnd, "error": f"{type(e).__name__}: {str(e)[:300]}", "retry_in_s": RETRY_S[attempt]})
                print(f"round {rnd}: {type(e).__name__} from the API; retrying in {RETRY_S[attempt]} s", flush=True)
                time.sleep(RETRY_S[attempt])
        if resp is None:
            raise RuntimeError("no response (fixture used up)")
        self.state["prev"] = resp["id"]
        self.state["usage"].append(resp.get("usage") or {})
        self.log({"round": rnd, "request": redact(req) | {"tools": [t["name"] for t in TOOLS]}, "response_id": resp["id"],
                  "calls": function_calls(resp), "text": output_text(resp), "usage": resp.get("usage")})
        return resp

    def turn(self, items, rnd):
        """Responses until the agent calls train_next_round or finish -> (decision, its call_id, outputs still owed)."""
        for _ in range(MAX_STEPS):
            resp = self.respond(items, rnd)
            calls = function_calls(resp)
            if not calls:
                items = [{"role": "user", "content": [{"type": "input_text", "text":
                          "Use the tools: change settings with set_param if you want, then call train_next_round "
                          "(or finish)."}]}]
                continue
            # settings first, whatever order they came in; the turn ends at the first train/finish call
            calls.sort(key=lambda c: c["name"] in ("train_next_round", "finish"))
            owed, decision, decision_id = [], None, None
            for c in calls:
                if decision is not None:  # anything after the decision is answered, not run
                    owed.append(fco(c["call_id"], {"ok": False, "error": "ignored: the round had already started"}))
                    continue
                result, dec = self.session.call(c["name"], c["arguments"])
                self.log({"round": rnd, "tool": c["name"], "arguments": c["arguments"], "result": result})
                if dec:
                    decision, decision_id = dec, c["call_id"]
                else:
                    owed.append(fco(c["call_id"], result))
            if decision:
                return decision, decision_id, owed
            items = owed
        raise RuntimeError(f"round {rnd}: no train_next_round or finish after {MAX_STEPS} responses")

    def round_result(self, k, arms, eval_only):
        """Run (or reuse) round k -> (its results folder, round.json)."""
        tag = f"{self.out.name}_r{k}"
        banks = [Path(r["dir"]) / "robust" / "eval_bank_dense.npz" for r in self.state["rounds"]]
        inp = prepare_inputs(self.out / "rounds" / f"r{k}" / "inputs", self.run, self.spec_f, arms,
                             next((b for b in reversed(banks) if b.exists()), None))
        res = Path(self.run_round(k, inp, tag, eval_only))
        return res, json.loads((res / "round.json").read_text())

    def go(self):
        st = self.state
        if not st["rounds"]:  # round 0: the run's policy as it is
            res, r0 = self.round_result(0, {"start": (self.run / "policy_task", None, True)}, eval_only=True)
            st["rounds"].append({"k": 0, "dir": str(res), "arms": {"start": headline(r0["arms"]["start"])}})
            vid = small_video(arm_dir(res, "start") / "policy.mp4", self.out / "r0_rollout.mp4")
            person = small_video(self.video, self.out / "person.mp4", 480)
            st["pending"] = [video_message(
                spec_text(self.spec) + "\n\n" + OBJECTIVE.format(iters=self.iters, rounds=self.rounds)
                + "\n\nRound 0, the policy before tuning (1500 iterations with the default settings):\n"
                + json.dumps(summary(r0["arms"]["start"]), indent=1)
                + "\n\nThe videos: first the person, then the robot's rollout this round.", person, vid)]
            self.save()
        while len(st["rounds"]) <= self.rounds and not st.get("finished"):
            k = len(st["rounds"])
            planned = st.get("planned")
            if planned and planned["round"] == k:  # decided before a round that failed to run: don't ask again
                decision, call_id, owed = planned["decision"], planned["call_id"], planned["owed"]
                self.session.cfg, self.session.changes = dict(planned["config"]), planned["changes"]
            else:
                decision, call_id, owed = self.turn(st["pending"], k)
                st["decisions"].append({"round": k, **decision, "config": dict(self.session.cfg),
                                        "changes": self.session.changes})
                st["planned"] = {"round": k, "decision": decision, "call_id": call_id, "owed": owed,
                                 "config": dict(self.session.cfg), "changes": self.session.changes}
                self.save()
            if decision["tool"] == "finish":
                st["finished"] = decision
                self.save()
                break
            prev = Path(st["rounds"][-1]["dir"])
            start = prev / "policy_task_start" if k == 1 else None
            arms = {"agent": (start or arm_dir(prev, "agent"), dict(self.session.cfg), False),
                    "base": (start or arm_dir(prev, "base"), defaults(self.spec), False)}
            res, r = self.round_result(k, arms, eval_only=False)
            st["rounds"].append({"k": k, "dir": str(res), "arms": {a: headline(r["arms"][a]) for a in ("agent", "base")},
                                 "seconds": r.get("seconds")})
            last = k == self.rounds
            vid = small_video(arm_dir(res, "agent") / "policy.mp4", self.out / f"r{k}_rollout.mp4")
            note = (f"Round {k} done (the agent arm, {r['iters']} more iterations with your settings). "
                    + ("That was the last round: call finish with the best round and a short summary of what "
                       "helped and what didn't." if last else "Adjust and call train_next_round, or finish."))
            st["pending"] = owed + [fco(call_id, {"round": k, "result": summary(r["arms"]["agent"])}),
                                    video_message(note + " The robot's rollout:", vid)]
            self.session.new_round()
            self.save()
            if last:
                decision, call_id, _ = self.turn(st["pending"], k + 1)
                st["finished"] = decision
                self.save()
                break
        return self.report()

    def report(self):
        st = self.state
        cost = tokens(st["usage"])
        rows = []
        for r in st["rounds"]:
            for arm, h in r["arms"].items():
                rows.append({"round": r["k"], "arm": arm, **h})
        rep = {"target": self.target, "rounds": len(st["rounds"]) - 1, "iters_per_round": self.iters, "rows": rows,
               "decisions": st["decisions"], "finished": st.get("finished"), "tokens": cost,
               "seconds": {f"r{r['k']}": (r.get("seconds") or {}) for r in st["rounds"]}}
        write_json(self.out / "report.json", rep)
        (self.out / "report.md").write_text(report_md(rep))
        print((self.out / "report.md").read_text())
        return rep


RETRY_S = (10, 30, 60, 120)  # waits before retrying a request the API failed with 5xx, a timeout or no connection


def transient(e):
    return (getattr(e, "status_code", None) or 0) >= 500 or type(e).__name__ in (
        "APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError")


def fco(call_id, value):
    return {"type": "function_call_output", "call_id": call_id, "output": json.dumps(value)}


def tokens(usages):
    """Responses API usage dicts -> totals and their cost at PRICE."""
    t = {"input": 0, "cached": 0, "output": 0, "reasoning": 0, "responses": len(usages)}
    for u in usages:
        t["input"] += u.get("input_tokens") or 0
        t["cached"] += (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
        t["output"] += u.get("output_tokens") or 0
        t["reasoning"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
    t["usd"] = round(((t["input"] - t["cached"]) * PRICE["input"] + t["cached"] * PRICE["cached"]
                      + t["output"] * PRICE["output"]) / 1e6, 4)
    return t


def report_md(rep):
    pct = lambda x: "-" if x is None else f"{x:.0%}"  # noqa: E731
    num = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"  # noqa: E731
    lines = [f"# Reward agent loop: {rep['target']}", "",
             f"{rep['rounds']} rounds of {rep['iters_per_round']} PPO iterations per arm; both arms from the same "
             "checkpoint, scored alike (64 robots per condition, fixed task spec).", "",
             "| Round | Arm | Clean success | Clean stages | Clean falls | Push success | Push falls | Start-noise success | V-JEPA | Rollout stages |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        lines.append(f"| {r['round']} | {r['arm']} | {pct(r['clean_task_success'])} | {num(r['clean_stages'])} | "
                     f"{pct(r['clean_fall_rate'])} | {pct(r['push_task_success'])} | {pct(r['push_fall_rate'])} | "
                     f"{pct(r['init_task_success'])} | {num(r['vjepa_aligned'], 3)} | {r['rollout_stages']} |")
    lines += ["", "## What the agent did", ""]
    for d in rep["decisions"]:
        ch = "; ".join(f"`{c['path']}` {c['old']:g} -> {c['new']:g} ({c['reason']})" for c in d.get("changes", [])) or "no change"
        what = d.get("hypothesis") or d.get("summary") or ""
        lines.append(f"- Before round {d['round']}: {ch}. {d['tool']}: {what}")
    if rep.get("finished"):
        f = rep["finished"]
        lines.append(f"- Finished: best round {f.get('best_round')}: {f.get('summary', '')}")
    t = rep["tokens"]
    lines += ["", f"Muse Spark: {t['responses']} responses, {t['input']:,} input tokens ({t['cached']:,} cached), "
                  f"{t['output']:,} output ({t['reasoning']:,} reasoning): ${t['usd']:.3f}."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- offline

def canned_round(fixture):
    """A run_round for --dry-run: copies a recorded round.json (and nothing else) into the results folder."""
    def run_round(k, inp, tag, eval_only):
        res = Path(inp).parent / "results"
        res.mkdir(exist_ok=True)
        r = copy.deepcopy(read_json(fixture))
        if eval_only:
            r["arms"] = {"start": next(iter(r["arms"].values()))}
            r["iters"] = 0
        write_json(res / "round.json", r)
        for arm in r["arms"]:  # the next round starts from these
            (res / f"policy_task_{arm}").mkdir(exist_ok=True)
            (res / f"policy_task_{arm}" / f"model_{1499 + 300 * k}.pt").write_bytes(b"")
        return res
    return run_round


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a latent run's results folder with policy_task/ (model, rollout), clips/, reward.pt, "
                                "task_spec.json")
    ap.add_argument("--target", required=True, help="the clip it was trained on (clips/<target>)")
    ap.add_argument("--video", help="the person's video, shown to Muse once (default: none)")
    ap.add_argument("--out", required=True, help="the loop's folder (state.json makes it resumable)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=300, help="PPO iterations per arm per round")
    ap.add_argument("--branch", default="main", help="the repo branch the GPU jobs run (pushed)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--reps", type=int, default=64, help="robots per arm per scoring condition")
    ap.add_argument("--dry-run", action="store_true", help="fixture responses and canned rounds: no key, no GPU")
    ap.add_argument("--model")
    a = ap.parse_args(argv)
    if a.dry_run:
        client = MuseClient(fixture=FIXTURES / "agent_loop_jacks.json", model=a.model)
        run_round = canned_round(FIXTURES / "agent_round_jacks.json")
    else:
        client = MuseClient(model=a.model)
        run_round = lambda k, inp, tag, eval_only: modal_round(k, inp, tag, a.target, a.iters, a.branch, eval_only,  # noqa: E731
                                                               f"--reps {a.reps}")
    Loop(a.run, a.target, a.out, client, run_round, a.video, a.rounds, a.iters, a.effort).go()


if __name__ == "__main__":
    main()
