"""Record a stored run's "watch it learn" attempts on a Modal GPU: its training environment (env.zip) trained again
with an attempt every N iterations, through the same backend the website's modal runner uses (server/train.py).

    MODAL_PROFILE=a2h python3 scripts/record_attempts.py outputs/remote_latent_s3_kick --iters 1000 --attempt-every 50

-> <run>/attempts/ (index.json + iter_NNNN.npz, ~30 KB each; scripts/build_space.py ships them, and /build/<run>
plays them) and <run>/attempts/train.log. --with-policy also keeps the trained checkpoint as <run>/attempts/policy.pt.
The GPU job clones --branch (main by default) from GitHub, as every Modal job does: push first.

Needs `pip install modal` and a login (python -m modal token new, or MODAL_PROFILE). No other secret goes up.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from server import train as T  # noqa: E402

GPU_USD_PER_H = 1.95  # an L40S (scripts/modal_app.py GPU); the estimate printed at the end


def log(msg, f=None):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if f:
        f.write(line + "\n")
        f.flush()


def main(a):
    run = Path(a.run).resolve()
    env = run / "env.zip"
    if not env.is_file():
        raise SystemExit(f"{env} not found: python scripts/export_env.py {run} first")
    out = run / T.ATTEMPTS
    out.mkdir(parents=True, exist_ok=True)
    session = f"record_{re.sub(r'[^a-z0-9]+', '_', run.name.lower())[:30]}_{time.strftime('%m%d%H%M')}"
    be = T.ModalBackend()
    ok, why = be.status(0)
    if not ok:
        raise SystemExit(f"{why}: python -m modal deploy scripts/modal_app.py")
    with open(out / "train.log", "a") as f:
        log(f"{run.name}: uploading env.zip ({env.stat().st_size / 1e6:.1f} MB) as {session}", f)
        be.upload(session, {"env.zip": env})
        cmd = T.command(session, a.iters, a.num_envs, a.attempt_every)
        t0 = time.time()
        call = be.spawn(session, cmd, a.branch)
        log(f"spawned run_job (call {call}, branch {a.branch}): {cmd}", f)
        off, seen, last_it = 0, set(), 0
        try:
            while True:
                time.sleep(a.poll_s)
                data = be.read(session, f"jobs/{session}/log")
                if data and len(data) > off:
                    for raw in re.split(r"[\r\n]+", data[off:].decode("utf-8", "ignore")):
                        line = T.clean(raw).strip()
                        m = T._ITER.search(line)
                        if m:
                            last_it = int(m.group(1)) + 1
                        if line.startswith(("[a2h_env]", "rendering with", "Traceback", "git clone")) or "Error:" in line:
                            log(line[:300], f)
                    off = len(data)
                raw = be.read(session, f"outputs/policy/{T.ATTEMPTS}/index.json", T.MAX_INDEX_BYTES)
                if raw:
                    try:
                        index = json.loads(raw)
                    except ValueError:
                        index = None
                    if index:
                        for r in index.get("attempts", []):
                            name = r.get("file", "")
                            if T.NPZ.match(name) and name not in seen and be.download(session, f"outputs/policy/{T.ATTEMPTS}/{name}", out / name):
                                seen.add(name)
                                log(f"attempt {r['iteration']}: reward {r.get('reward')}, stages {r.get('stages_completed')}/{r.get('stages_total')}"
                                    f"{', fell' if r.get('fell') else ''} -> {name}", f)
                        index["attempts"] = [r for r in index["attempts"] if r.get("file") in seen]
                        (out / "index.json").write_text(json.dumps(index, separators=(",", ":")))
                state = be.state(call)
                if state in ("done", "failed"):
                    break
                if state == "running" and last_it:
                    print(f"\r  iteration {last_it}/{a.iters}, {len(seen)} attempts, {(time.time() - t0) / 60:.0f} min", end="", flush=True)
        except KeyboardInterrupt:
            be.cancel(call)
            log("cancelled", f)
            be.remove(session)
            raise
        print()
        took = time.time() - t0
        log(f"job state: {state} after {took / 60:.1f} min (~${took / 3600 * GPU_USD_PER_H:.2f} of L40S time)", f)
        if a.with_policy and be.download(session, "outputs/policy/weights.json", out / "weights.json", 1 << 16):
            ck = json.loads((out / "weights.json").read_text()).get("checkpoint")
            if ck and be.download(session, f"outputs/policy/{ck}", out / "policy.pt", 64 << 20):
                log(f"policy checkpoint -> {out / 'policy.pt'}", f)
        full = be.read(session, f"jobs/{session}/log")
        if full:
            (out / "job.log").write_bytes(full)
        be.remove(session)
        log(f"{len(seen)} attempts in {out}; the Volume folder is gone", f)
    return 0 if state == "done" else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a run folder with env.zip (scripts/export_env.py)")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--attempt-every", type=int, default=50)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--branch", default="main", help="the repository branch the GPU job runs (pushed to GitHub)")
    ap.add_argument("--poll-s", type=float, default=30)
    ap.add_argument("--with-policy", action="store_true", help="keep the trained checkpoint too (<run>/attempts/policy.pt)")
    sys.exit(main(ap.parse_args()))
