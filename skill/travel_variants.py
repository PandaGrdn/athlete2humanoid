"""skill/variations.py's variants, with one false rejection fixed: holds out of sync with the stage
machine on stages that move the root (docs/FRONTIER_TRAVEL.md).

The compiler sets a key pose's hold from how long the reference dwells in it, with the root goal
measured from the previous key frame; the machine measures it from where the previous stage was first
satisfied, often frames earlier. On a stage that moves the root ("step forward") the reference then
dwells less than the hold asks, the stage completes only when a later repetition passes the same pose,
the machine falls behind the clip, and the variant fails its own spec. keypose.fit_holds halves holds
only at the stage the self-check stops at, which is by then a later one. sync_holds() halves the hold
of the first stage completed late instead (after its key frame + MIN_GAP_S), until the reference
completes its spec. Nothing else changes: the spec's goals and tolerances, and the joint-limit filter.

On the lunge it keeps 24 of the 36 training variants instead of 15; the clips that stay in place lose
nothing either way.

    python -m skill.travel_variants outputs/mc_inputs/clips/lunge
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from skill import task as T
from skill import variations as V
from skill.keypose import MIN_GAP_S


def completions(spec, pos, fps):
    """Replays the reference through the spec's machine -> ({stage: frame it completed or was skipped
    at}, whether the whole spec was done)."""
    m = T.machine({**spec, "down": spec["down_ref"], "air": spec["air_ref"]}, 1, dt=1.0 / fps)
    p = torch.as_tensor(np.asarray(pos), dtype=torch.float32)
    m.reset(torch.tensor([0]), 0.0, p[:1])
    done = {}
    for t in range(1, len(p)):
        before = int(m.completed[0]) + int(m.skipped[0])
        m.step(p[t:t + 1])
        for k in range(before, int(m.completed[0]) + int(m.skipped[0])):
            done[k] = t
        if bool(m.done[0]):
            break
    return done, bool(m.done[0])


def sync_holds(spec, pos, fps, tries=24):
    """Halves (then drops) the hold of the first pose stage the reference completes late, until it
    completes its own spec. -> True if it then does (spec edited in place)."""
    slack = int(round(MIN_GAP_S * fps))
    for _ in range(tries):
        done, ok = completions(spec, pos, fps)
        if ok:
            return True
        late = [s for k, s in enumerate(spec["stages"]) if s["kind"] == "pose" and s.get("hold_s")
                and (k not in done or done[k] > s["frame"] + slack)]
        if not late:
            return False
        late[0]["hold_s"] = round(late[0]["hold_s"] / 2, 3) if late[0]["hold_s"] >= 0.1 else 0.0
    return False


def generate(qpos, fps=V.FPS, grid=None, seed=0, g1=None, log=print):
    """V.generate, then every variant dropped only because its reference failed its own spec is given
    sync_holds(); one that then passes is kept ("synced": True, and its spec's self_check redone)."""
    g1 = g1 or V.G1()
    vs = V.generate(qpos, fps, grid, seed, g1, log)
    for v in vs:
        if v["ok"] or v["spec"] is None or "beyond their limits" in v["why"]:
            continue
        pos = g1.bodies(v["qpos"])
        if sync_holds(v["spec"], pos, fps):
            sc = T.evaluate({**v["spec"], "down": v["spec"]["down_ref"], "air": v["spec"]["air_ref"]}, pos, fps)
            v["spec"]["self_check"] = sc
            v["ok"], v["why"], v["synced"] = bool(sc["success"]), "" if sc["success"] else v["why"], True
            log(f"  {v['name']:22s} kept after syncing its holds" if v["ok"] else f"  {v['name']:22s} still dropped")
    return vs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip", help="a clip folder (qpos.npy, meta.json)")
    ap.add_argument("--held-out", action="store_true")
    a = ap.parse_args(argv)
    clip = Path(a.clip)
    fps = json.loads((clip / "meta.json").read_text())["fps"]
    vs = generate(np.load(clip / "qpos.npy"), fps, V.HELD_OUT if a.held_out else V.TRAIN, log=lambda *x: None)
    print(f"{clip.name}: {sum(v['ok'] for v in vs)}/{len(vs)} kept "
          f"({sum(bool(v.get('synced')) and v['ok'] for v in vs)} after syncing holds)")


if __name__ == "__main__":
    main()
