"""Rollout critic: Muse Spark watches the person and each trained policy, and judges them blind.

    python -m muse.rollout_critic outputs/remote_latent_kp_kick outputs/remote_latent_hyb_kick
    python -m muse.rollout_critic outputs/remote_latent_kp_kick --dry-run

The run folders (experiments/latent_policy.py) must be of the same clip; their policies are pooled,
the first folder's copy winning. Muse gets the person's video and each policy's physics rollout
(policy_<name>/policy.mp4) as "Robot 1..n" in a shuffled order, so a policy's name can't sway it, and
the clip's key moments (named by muse/moment_namer.py when the spec has been named). Per robot it
says whether the robot does the action, how much it looks like the person (0-10), whether it fell,
which key moments it reaches and how it differs; then it ranks them.

critic.json (in the first folder, or --out) holds Muse's answer by policy name, and how it agrees
with the measurements in eval.json: falls, key moments done (the stage checker) and the V-JEPA
ranking. Disagreement is the interesting part: a stage checker can pass a motion that doesn't look
like the action, and V-JEPA can prefer a motion people wouldn't.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muse import media  # noqa: E402
from muse.client import make_client, now, read_json, text_part, video_part, write_json  # noqa: E402
from muse.moment_namer import find_video, moment_time, moments  # noqa: E402
from muse.reward_agent import target_name  # noqa: E402
from muse.schema import BOOL, INT, STR, arr, enum, obj  # noqa: E402

VERDICT = ["yes", "partly", "no"]
CRITIC_SCHEMA = obj(
    robots=arr(obj(robot=INT, does_action=enum(VERDICT), likeness=INT, fell=BOOL,
                   moments=arr(obj(index=INT, reached=enum(VERDICT))), differences=arr(STR), summary=STR)),
    ranking=arr(INT),
    verdict=STR,
)

SYSTEM = (
    "You evaluate physics simulations of a Unitree G1 humanoid robot that was trained to perform the motion "
    "in a video of a person. You compare each robot's motion with the person's and report what you see. "
    "Describe motion only: never address or advise the person, never judge their technique. Answer with JSON only."
)


def policies(run_dirs, only=None):
    """{name: (run dir, policy.mp4)} pooled over the runs, first one winning."""
    found = {}
    for run in map(Path, run_dirs):
        for d in sorted(run.glob("policy_*")):
            name = d.name.removeprefix("policy_")
            if (d / "policy.mp4").is_file() and name not in found and (not only or name in only):
                found[name] = (run, d / "policy.mp4")
    return found


def prompt(spec, order, sample_fps):
    n = len(order)
    lines = [f"Video 1 is the person. Videos 2-{n + 1} are Robot 1 to Robot {n}, each a separate trained policy "
             f"replaying the same motion in simulation, sampled at {sample_fps:g} frames per second.",
             "In the robot videos the solid white robot is the policy; the translucent green figure is the reference "
             "it was shown (the person's motion retargeted to the robot). Judge the white robot against the person "
             "in video 1, not against the green figure. The robots may drift across the floor; where they end up "
             "matters less than the motion itself."]
    if spec:
        fps = spec["fps"]
        lines += ["", "The motion's key moments (times from the start; a robot may reach them a little early or late):"]
        for i, st in enumerate(moments(spec), 1):
            desc = f" ({st['description']})" if st.get("description") else ""
            lines.append(f"{i}. {moment_time(st, fps):.2f} s: {st['name']}{desc}")
    lines += [
        "",
        "For each robot, by its number:",
        "- does_action: yes / partly / no, does it perform the person's motion;",
        "- likeness: 0-10, how much its motion looks like the person's (10 = indistinguishable timing and shape, "
        "0 = unrelated or standing still);",
        "- fell: whether it falls over at any point;",
        "- moments: for each key moment by number, whether the robot reaches that pose (yes / partly / no);",
        "- differences: up to 3 short phrases on how its motion differs from the person's;",
        "- summary: one sentence.",
        "Then ranking: the robot numbers, most like the person first; and verdict: one or two sentences comparing them.",
    ]
    return "\n".join(lines)


def check(value, n, n_moments):
    errors = []
    got = sorted(r["robot"] for r in value["robots"])
    if got != list(range(1, n + 1)):
        errors.append(f"robots: exactly one entry per robot, 1 to {n} (got {got})")
    if sorted(value["ranking"]) != list(range(1, n + 1)):
        errors.append(f"ranking: each robot number 1 to {n} once")
    for r in value["robots"]:
        if not 0 <= r["likeness"] <= 10:
            errors.append(f"robots[{r['robot']}].likeness: 0 to 10")
        idx = sorted(m["index"] for m in r["moments"])
        if idx != list(range(1, n_moments + 1)):
            errors.append(f"robots[{r['robot']}].moments: one entry per key moment, 1 to {n_moments}")
        if len(r["differences"]) > 3:
            errors.append(f"robots[{r['robot']}].differences: at most 3")
    return errors


def _rank(values, reverse=True):
    order = sorted(values, key=values.get, reverse=reverse)
    return {k: i + 1 for i, k in enumerate(order)}


def agreement(by_policy, ranking, evals, n_moments):
    """Muse's answer against eval.json: falls, key moments done, and its ranking against V-JEPA's."""
    rows, pairs = {}, []
    for name, r in by_policy.items():
        ev = evals.get(name) or {}
        stages = str(ev.get("task_stages") or "")
        rows[name] = {
            "fell": {"muse": r["fell"], "measured": (ev.get("falls") or 0) > 0 if "falls" in ev else None},
            "moments_reached": {"muse": sum(m["reached"] == "yes" for m in r["moments"]), "of": n_moments,
                                "stage_checker": stages or None},
            "vjepa_aligned": ev.get("aligned"),
        }
    vj = {k: v["vjepa_aligned"] for k, v in rows.items() if isinstance(v["vjepa_aligned"], (int, float))}
    if len(vj) >= 2:
        vrank, mrank = _rank(vj), {k: ranking.index(k) + 1 for k in vj}
        names = sorted(vj)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        same = sum((vrank[a] - vrank[b]) * (mrank[a] - mrank[b]) > 0 for a, b in pairs)
    fell = [v["fell"] for v in rows.values() if v["fell"]["measured"] is not None]
    return {"by_policy": rows,
            "falls_agree": f"{sum(f['muse'] == f['measured'] for f in fell)}/{len(fell)}" if fell else None,
            "vjepa_rank": sorted(vj, key=vj.get, reverse=True) if vj else None,
            "rank_pairs_agree": f"{same}/{len(pairs)}" if pairs else None}


def critique(run_dirs, client, only=None, sample_fps=10, out=None, seed=None):
    run_dirs = [Path(d) for d in run_dirs]
    pols = policies(run_dirs, only)
    if not pols:
        raise FileNotFoundError(f"no policy_*/policy.mp4 in {', '.join(map(str, run_dirs))}")
    first = run_dirs[0]
    spec_f = next((d / "task_spec.json" for d in run_dirs if (d / "task_spec.json").is_file()), None)
    spec = read_json(spec_f) if spec_f else None
    if spec and spec.get("type") != "keypose":
        spec = None
    target = target_name(first) or (spec or {}).get("name")
    person = find_video(first, {"name": target})
    if person is None:
        raise FileNotFoundError(f"no video of the person for {first}")
    order = sorted(pols)
    random.Random(seed if seed is not None else first.name).shuffle(order)  # blind, and the same each time
    tmp = (Path(out).parent if out else first) / "muse_critic"
    tmp.mkdir(parents=True, exist_ok=True)
    content = [text_part(prompt(spec, order, sample_fps)),
               video_part(media.prepare_video(person, tmp / "person.mp4", 480, required=client.live), sample_fps)]
    for name in order:
        content.append(video_part(media.prepare_video(pols[name][1], tmp / f"{name}.mp4", 320,
                                                      required=client.live), sample_fps))
    n_moments = len(moments(spec)) if spec else 0
    value, info = client.ask_json("rollout_critic", SYSTEM, content, CRITIC_SCHEMA,
                                  check=lambda v: check(v, len(order), n_moments), effort="medium")
    by_policy = {order[r["robot"] - 1]: {k: v for k, v in r.items() if k != "robot"} for r in value["robots"]}
    ranking = [order[i - 1] for i in value["ranking"]]
    evals = {}
    for name, (run, _) in pols.items():
        evals[name] = (read_json(run / "eval.json") or {}).get(name) or {}
    result = {"clip": target, "policies": by_policy, "ranking": ranking, "verdict": value["verdict"],
              "blind_order": order, "moments": [st["name"] for st in moments(spec)] if spec else [],
              "agreement": agreement(by_policy, ranking, evals, n_moments),
              "sources": {k: str(v[1]) for k, v in pols.items()}, "person_video": str(person),
              "_meta": {"created": now(), "sample_fps": sample_fps, **info}}
    out = Path(out or first / "critic.json")
    write_json(out, result)
    if not client.live:
        client.save_requests(out.with_name("critic.request.json"))
    report(result)
    return result


def report(result):
    print(f"{result['clip']}: Muse ranks {' > '.join(result['ranking'])} ({result['_meta']['mode']})")
    a = result["agreement"]
    for name in result["ranking"]:
        p, m = result["policies"][name], a["by_policy"][name]
        vj = m["vjepa_aligned"]
        print(f"  {name:11s} action {p['does_action']:6s} likeness {p['likeness']:2d}/10  fell {p['fell']!s:5s} "
              f"(measured {m['fell']['measured']})  moments {m['moments_reached']['muse']}/{m['moments_reached']['of']} "
              f"(checker {m['moments_reached']['stage_checker']})  V-JEPA {vj if vj is None else round(vj, 3)}")
    print(f"  V-JEPA ranks {' > '.join(a['vjepa_rank'] or [])}; pairs in the same order: {a['rank_pairs_agree']}; "
          f"falls agree {a['falls_agree']}")
    print(f"  {result['verdict']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run folders of the same clip; their policies are pooled")
    ap.add_argument("--policies", nargs="+", help="only these policies")
    ap.add_argument("--out", help="default <first run>/critic.json")
    ap.add_argument("--fps", type=float, default=10, help="frames per second Muse samples (fast moves like a kick last ~0.2 s)")
    ap.add_argument("--seed", type=int, help="the blind order's shuffle (default: from the run's name)")
    ap.add_argument("--dry-run", action="store_true", help="no API call: bundled fixture response")
    ap.add_argument("--fixture", help="no API call: responses from this recorded file")
    ap.add_argument("--record", help="save live exchanges here (they work as --fixture files)")
    ap.add_argument("--model")
    a = ap.parse_args(argv)
    client = make_client(a.dry_run, a.fixture, "critic_front_kick.json", a.model, a.record)
    critique(a.runs, client, a.policies, a.fps, a.out, a.seed)


if __name__ == "__main__":
    main()
