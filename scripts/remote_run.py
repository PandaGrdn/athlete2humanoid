"""Unattended GPU jobs on Colab, driven from this Mac with the Colab CLI.

    python3 scripts/remote_run.py probe --min-clips 7       # cross-embodiment probe
    python3 scripts/remote_run.py latent --target squat     # video -> latent reward -> policies
    python3 scripts/remote_run.py latent --smoke            # the same, tiny, no clips: plumbing test
    python3 scripts/remote_run.py latent --tag v2 --reuse outputs/remote_latent   # a follow-up run:
        # an earlier run's processed clips (no SAM) and its policies' rollouts (DAgger data)
    python3 scripts/remote_run.py robust --reuse outputs/remote_latent   # that run's policies under
        # perturbation (experiments/robustness_eval.py) -> outputs/remote_robust/robust.json
    python3 scripts/remote_run.py pro --iters 0 --backend modal --clips-dir clips_athletes --min-clips 1 --in-place sprint
        # new videos processed on Modal (SAM 3.1 + SAM 3D Body installed at job start), nothing trained
    python3 scripts/remote_run.py athletes --backend modal --reuse outputs/athletes_inputs --with-weights --iters 600
        # each processed clip -> a variants tracker + its held-out variants (scripts/real_athletes.sh)
    python3 scripts/remote_run.py relevance --reuse outputs/remote_latent_v2 --target squat_jump --iters 1000
        # goal relevance (experiments/goal_relevance.py): V-JEPA importance + with/without --importance

1. waits until at least --min-clips videos sit in clips/ and stop changing (copy finished)
2. checks the HF token in .env before spending GPU time
3. creates an A100, uploads clips + token, launches the job via scripts/job.py
4. keeps the VM busy, polls status every 2 min (and pulls the job log every 10), recreates the VM
   and retries once if it's lost
5. downloads results to outputs/remote_<job>/ and stops the VM

Progress goes to outputs/remote_<job>/runner.log. Needs `colab` logged in (oauth2).

--backend farmshare runs the same job on Stanford FarmShare through Slurm instead (scripts/farmshare.py,
docs/FARMSHARE.md): for when Colab compute units run out. Colab stays the default.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".mov", ".m4v"}
SKIP = {"bedroom"}  # test footage, not an action clip
REMOTE = "/content/a2h"  # on FarmShare: <fs-root>/runs/<session>, set in main()
OUT = SESSION = None  # set in main()
CLIPS_DIR = REPO / "clips"  # --clips-dir
BRANCH = "main"  # --branch: the code the GPU job runs


def latent_out(a):
    kind = a.job if a.job in ("variants", "multiclip") else "latent"
    return f"{REMOTE}/outputs/{kind}_{'smoke' if a.smoke else a.tag or a.target}"


def job_spec(a):
    """(setup flag, command run in the repo after setup, {local name: remote path} to fetch)."""
    if a.job == "pro":  # scripts/pro_clips.sh: new videos -> SAM (installed at job start) -> trained, tested policies
        if not a.reuse and a.iters:  # --iters 0: process only (the way to process new videos on Modal)
            raise SystemExit("pro needs --reuse <a folder with policy_*/model_*.pt and vjl_pca.npz>")
        pols = " ".join(x for x in a.args.split() if x.isidentifier()) or "tracking_var multi multi_vjc"
        if not re.fullmatch(r"[A-Za-z0-9_,.-]*", a.in_place):
            raise SystemExit(f"odd --in-place: {a.in_place}")
        return ("--with-main", f"POLICIES='{pols}' IN_PLACE='{a.in_place}' bash scripts/pro_clips.sh {a.iters} {{clips}}",
                {"results.tar.gz": f"{REMOTE}/outputs/pro/results.tar.gz"})
    if a.job == "athletes":  # scripts/real_athletes.sh: processed clips -> a variants tracker each, in parallel, + short eval
        if not a.reuse:
            raise SystemExit("athletes needs --reuse <a folder with clips/<name>/ and policy_tracking_var/model_*.pt> --with-weights")
        return ("--with-main", f"bash scripts/real_athletes.sh {a.iters} {a.args}",
                {"results.tar.gz": f"{REMOTE}/outputs/athletes/results.tar.gz"})
    if a.job == "process":  # experiments/process_clips.py: new videos -> clip folders (SAM, retarget, V-JEPA), no training
        out = f"{REMOTE}/outputs/process" + (f"_{a.tag}" if a.tag else "")
        return ("--with-sam", f"python3 experiments/process_clips.py {{clips}} --out {out}",
                {"results.tar.gz": f"{out}/results.tar.gz"})
    if a.job == "agentround":  # experiments/agent_round.py: one round of muse/agent_loop.py (--reuse: its inputs folder)
        if not a.reuse:
            raise SystemExit("agentround needs --reuse <the round's inputs folder, from muse/agent_loop.py>")
        out, inp = f"{REMOTE}/outputs/agentround" + (f"_{a.tag}" if a.tag else ""), f"{REMOTE}/inputs"
        return ("--with-main", f"python3 experiments/agent_round.py --inputs {inp} --out {out} --target {a.target} "
                f"--iters {a.iters} {a.args}", {"results.tar.gz": f"{out}/results.tar.gz"})
    if a.job == "probe":
        out = f"{REMOTE}/outputs/cross_embodiment"
        return ("--with-sam", f"python3 experiments/cross_embodiment_probe.py {{clips}} --out {out}",
                {"cross_embodiment.json": f"{out}/cross_embodiment.json"})
    if a.job == "robust":
        out, inp = f"{REMOTE}/outputs/robust" + (f"_{a.tag}" if a.tag else ""), f"{REMOTE}/inputs"
        return ("--with-main", f"python3 experiments/robustness_eval.py --runs {inp} --out {out} --target {a.target} {a.args}",
                {"robust.json": f"{out}/robust.json", "robust_summary.md": f"{out}/robust_summary.md",
                 "rollouts.npz": f"{out}/rollouts.npz"})
    if a.job == "pushes":  # experiments/push_showcase.py: the site's recorded pushes of one policy
        out, inp = f"{REMOTE}/outputs/pushes" + (f"_{a.tag}" if a.tag else ""), f"{REMOTE}/inputs"
        # The job clones main: until the script is there, it rides along in the inputs (pack_reuse)
        return ("--with-main", f"test -f experiments/push_showcase.py || cp {inp}/push_showcase.py experiments/; "
                f"$A2H_MJLAB_PYTHON experiments/push_showcase.py --runs {inp} --out {out} {a.args}",
                {"push_showcase.npz": f"{out}/push_showcase.npz", "push_showcase.json": f"{out}/push_showcase.json"})
    if a.job == "relevance":  # experiments/goal_relevance.py: V-JEPA importance + the counterfactual policies
        out, inp = f"{REMOTE}/outputs/relevance" + (f"_{a.tag}" if a.tag else ""), f"{REMOTE}/inputs"
        return ("--with-main", f"python3 experiments/goal_relevance.py --inputs {inp} --target {a.target} --out {out} "
                f"--iters {a.iters} {a.args}", {"results.tar.gz": f"{out}/results.tar.gz"})
    if a.job == "variants":  # experiments/variant_policy.py: train on variants of the clip, hold others out
        if not a.reuse:
            raise SystemExit("variants needs --reuse <a latent run's results folder> (its clips, and weights to start from)")
        out, inp = latent_out(a), f"{REMOTE}/inputs"
        return ("--with-main", f"python3 experiments/variant_policy.py {inp}/clips/{a.target} --out {out} "
                f"--iters {a.iters} {a.args}", {"results.tar.gz": f"{out}/results.tar.gz"})
    if a.job == "multiclip":  # experiments/multiclip_policy.py: one policy over several clips, +/- the V-JEPA input
        if not a.reuse:
            raise SystemExit("multiclip needs --reuse <a folder with clips/ (and policy_*/model_*.pt to --init from)>")
        out, inp = latent_out(a), f"{REMOTE}/inputs"
        return ("--with-main", f"python3 experiments/multiclip_policy.py {inp}/clips --out {out} --iters {a.iters} {a.args}",
                {"results.tar.gz": f"{out}/results.tar.gz"})
    out = latent_out(a)
    cmd = f"python3 experiments/latent_policy.py {{clips}} --target {a.target} --out {out} --iters {a.iters} {a.args}"
    if a.init_weights and not a.reuse:  # a new video, its policies built off an earlier run's (pack_weights)
        cmd += f" --init {REMOTE}/inputs"
    if a.reuse:
        inp = f"{REMOTE}/inputs"
        # clips/*/: directories only, so a config file next to them isn't read as a clip
        cmd = cmd.replace("{clips}", f"{inp}/clips/*/") + f" --extra-motions {inp}/policy_*/policy_qpos.npy"
        return "--with-main", cmd, {"results.tar.gz": f"{out}/results.tar.gz"}
    if a.smoke:
        return "--with-main", cmd + " --smoke", {"results.tar.gz": f"{out}/results.tar.gz"}
    return "--with-sam", cmd, {"results.tar.gz": f"{out}/results.tar.gz"}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "runner.log", "a") as f:
        f.write(line + "\n")


def colab(*args, timeout=600, check=True):
    r = subprocess.run(["colab", "--auth", "oauth2", *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"colab {' '.join(args[:2])} failed: {(r.stderr or r.stdout)[-300:]}")
    return r.stdout


def clips():
    return sorted(p for p in CLIPS_DIR.iterdir()
                  if p.suffix.lower() in VIDEO_EXT and p.stem.lower() not in SKIP)


def wait_for_clips(min_clips, max_wait_h):
    deadline, last = time.time() + max_wait_h * 3600, None
    while time.time() < deadline:
        found = clips()
        sizes = tuple((p.name, p.stat().st_size) for p in found)
        if len(found) >= min_clips and sizes == last:  # unchanged for a full poll: copying is done
            return found
        if sizes != last:
            log(f"waiting for clips: {len(found)}/{min_clips} present ({', '.join(p.name for p in found)})")
        last = sizes
        time.sleep(60)
    raise SystemExit(f"gave up waiting for {min_clips} clips")


def check_token():
    code = ("import sys; sys.path.insert(0, '.'); import os; os.environ['A2H_SECRETS_DIR'] = '/nonexistent'; "
            "os.environ.pop('HF_TOKEN', None); import config; "
            "from huggingface_hub import HfApi, hf_hub_download; t = config.get_secret('HF_TOKEN'); "
            "HfApi(token=t).whoami(); hf_hub_download('facebook/sam3.1', 'config.json', token=t); print('TOKEN_OK')")
    # Run with a Python that has huggingface_hub (the project's requirements).
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    if "TOKEN_OK" not in r.stdout:
        raise SystemExit("HF token in .env is invalid or lacks SAM 3.1 access; update .env and rerun.")


def pack_reuse(src, policies=False, checkpoints=None, script=None):
    """An earlier run's processed clips and policy rollouts, as one archive to upload. policies: its
    trained weights too (robust job; latent jobs with --with-weights, for --init/--resume-from).
    checkpoints: this run's own synced checkpoints (retry after a lost VM) -> inputs/checkpoints.
    script: a file to put at the archive's top (the pushes job's own script)."""
    f = OUT / "inputs.tar.gz"
    with tarfile.open(f, "w:gz") as tar:
        if checkpoints:
            for q in Path(checkpoints).glob("policy_*/model_*.pt"):
                tar.add(q, f"checkpoints/{q.parent.name}/{q.name}")
        if not src:
            return f
        src = Path(src)
        tar.add(src / "clips", "clips")
        for q in src.glob("policy_*/policy_qpos.npy"):
            tar.add(q, f"{q.parent.name}/{q.name}")
        for name in ("reward.pt", "distill_report.json", "vjl_pca.npz"):  # --args "--reward ..."; multiclip --pca
            if (src / name).exists():
                tar.add(src / name, name)
        if script:
            tar.add(script, script.name)
        if (src / "transfer").is_dir():  # robust job: clips the policies never trained on (--transfer)
            tar.add(src / "transfer", "transfer")
        if policies:  # the robust job: the trained policies themselves, and their eval
            for q in [*src.glob("policy_*/model_*.pt"), *src.glob("policy_*/motion.npz")]:
                tar.add(q, f"{q.parent.name}/{q.name}")
            for name in ("eval.json", "task_spec.json", "importance.json"):  # task-first policies need the spec
                if (src / name).exists():
                    tar.add(src / name, name)
    return f


def pack_dir(src):
    """A folder as is (agentround: muse/agent_loop.py lays out the round's inputs itself)."""
    f = OUT / "inputs.tar.gz"
    with tarfile.open(f, "w:gz") as tar:
        for q in sorted(Path(src).iterdir()):
            tar.add(q, q.name)
    return f


def pack_weights(src):
    """An earlier run's newest checkpoint per policy, alone (no clips): --init-weights for a new video."""
    f = OUT / "inputs.tar.gz"
    with tarfile.open(f, "w:gz") as tar:
        for d in sorted(Path(src).glob("policy_*")):
            ck = sorted(d.glob("model_*.pt"), key=lambda q: int(q.stem.split("_")[1]))
            if ck:
                tar.add(ck[-1], f"{d.name}/{ck[-1].name}")
    return f


def launch(found, setup, cmd, inputs=None):
    colab("new", "-s", SESSION, "--gpu", "A100", timeout=900)
    log("A100 ready; uploading inputs")
    _exec(f"import os; [os.makedirs(d, exist_ok=True) for d in ('{REMOTE}/clips', '{REMOTE}/env')]")
    for p in found:
        colab("upload", "-s", SESSION, str(p), f"{REMOTE}/clips/{p.name}", timeout=900)
    if inputs:
        colab("upload", "-s", SESSION, str(inputs), f"{REMOTE}/inputs.tar.gz", timeout=900)
        _exec(f"import tarfile, os; os.makedirs('{REMOTE}/inputs', exist_ok=True); "
              f"tarfile.open('{REMOTE}/inputs.tar.gz').extractall('{REMOTE}/inputs')")
    colab("upload", "-s", SESSION, str(REPO / ".env"), f"{REMOTE}/env/HF_TOKEN")
    paths = " ".join(f"'{REMOTE}/clips/{p.name}'" for p in found)
    inner = f"cd /content/athlete2humanoid && bash scripts/setup_colab.sh {setup} && " + cmd.format(clips=paths)
    _exec(f"""
import os, subprocess
R = "/content/athlete2humanoid"
if not os.path.exists(R):
    subprocess.run(["git", "clone", "-q", "--branch", "{BRANCH}", "https://github.com/PandaGrdn/athlete2humanoid.git", R], check=True)
else:
    subprocess.run(["git", "-C", R, "fetch", "-q", "origin", "{BRANCH}"], check=True)
    subprocess.run(["git", "-C", R, "checkout", "-q", "-B", "{BRANCH}", "FETCH_HEAD"], check=True)
env = dict(os.environ, MUJOCO_GL="egl", A2H_SAM_PYTHON="/content/sam_env/bin/python",
           A2H_SAM3D_PYTHON="/content/sam3d_env/bin/python", A2H_SAM3D_DIR="/content/sam-3d-body",
           A2H_OUTPUT_DIR="{REMOTE}/outputs", A2H_SECRETS_DIR="{REMOTE}/env", A2H_ASSETS_DIR=R + "/assets")
subprocess.Popen(["python3", R + "/scripts/job.py", "{SESSION}", "--", "bash", "-c", {inner!r}], env=env,
                 start_new_session=True, cwd="/content", stdout=open("{REMOTE}/job_{SESSION}.out", "w"),
                 stderr=subprocess.STDOUT)
print("launched")
""")
    log(f"{SESSION} job launched")
    return keep_busy()


def keep_busy():
    """A cell that stays busy until the job ends, so Colab doesn't reclaim an "idle" VM."""
    keeper = OUT / "keeper.py"
    keeper.write_text(f"""
import json, time
f = "{REMOTE}/jobs/{SESSION}/status.json"
time.sleep(20)
while json.load(open(f))["state"] == "running":
    print(".", end="", flush=True); time.sleep(60)
""")
    return subprocess.Popen(["colab", "--auth", "oauth2", "exec", "-s", SESSION, "-f", str(keeper)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _exec(code):
    f = OUT / "_exec.py"
    f.write_text(code)
    return colab("exec", "-s", SESSION, "-f", str(f))


def status():
    try:
        colab("download", "-s", SESSION, f"{REMOTE}/jobs/{SESSION}/status.json", str(OUT / "status.json"), timeout=120)
        return json.loads((OUT / "status.json").read_text())["state"]
    except Exception as e:
        return "lost" if "No active" in str(e) or "lost" in str(e) or "404" in str(e) else "unknown"


def pull_log():
    try:
        colab("download", "-s", SESSION, f"{REMOTE}/jobs/{SESSION}/log", str(OUT / "job.log"), timeout=300)
    except Exception as e:
        log(f"could not download job.log: {e}")


def colab_get(remote, local, timeout=600):
    colab("download", "-s", SESSION, remote, str(local), timeout=timeout)


def sync_checkpoints(a, get=None):
    """Newest checkpoint of every policy still training -> OUT/checkpoints/policy_*/ (a few MB each),
    so a lost VM costs at most the last sync interval of training, not the run. Plain downloads of
    the paths experiments/latent_policy.py lists in checkpoints.json: the kernel is busy (keeper)."""
    if a.job not in ("latent", "variants", "multiclip"):
        return
    out = latent_out(a)
    get = get or colab_get
    if not (OUT / "task_spec.json").exists():  # the task environment, for the web app before training ends
        try:
            get(f"{out}/task_spec.json", OUT / "task_spec.json", timeout=120)
            log("fetched task_spec.json")
        except Exception:
            pass  # not compiled yet
    try:
        get(f"{out}/checkpoints.json", OUT / "checkpoints.json", timeout=120)
        got = []
        for name, rel in json.loads((OUT / "checkpoints.json").read_text()).items():
            dst = OUT / "checkpoints" / name / Path(rel).name
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = dst.with_suffix(".part")
            get(f"{out}/{rel}", part, timeout=600)
            for old in dst.parent.glob("model_*.pt"):
                old.unlink()
            part.rename(dst)
            got.append(f"{name}/{dst.name}")
        if got:
            log("synced checkpoints: " + ", ".join(got))
    except Exception as e:
        log(f"checkpoint sync skipped: {str(e)[-200:]}")


def fetch_results(fetch):
    pull_log()
    for local, remote in fetch.items():
        try:
            colab("download", "-s", SESSION, remote, str(OUT / local), timeout=900)
        except Exception as e:
            log(f"could not download {local}: {e}")
            continue
        if local.endswith(".tar.gz"):
            with tarfile.open(OUT / local) as tar:
                tar.extractall(OUT, filter="data")
            log(f"unpacked {local}")


def unpack(path):
    with tarfile.open(path) as tar:
        tar.extractall(OUT, filter="data")
    log(f"unpacked {path.name}")


def run_modal(a, mr, found, setup, cmd, fetch, inputs):
    """The Colab loop on Modal: one spawn; the container (and billing) ends with the job. scripts/modal_backend.py."""
    from modal_backend import wait
    if setup == "--with-sam":
        raise SystemExit("the Modal image has no SAM 3.1 (new videos): run this job on Colab, or --reuse processed clips")
    mr.deploy()
    mr.upload(found, inputs, REPO / ".env")
    try:
        mr.submit(cmd.format(clips=" ".join(f"'{REMOTE}/clips/{p.name}'" for p in found)), a.branch)
    except BaseException:  # refused (e.g. the workspace's spend limit): the uploaded token must not stay on the Volume
        mr.cleanup()
        raise
    try:
        state = wait(mr, lambda: sync_checkpoints(a, mr.download))
    except KeyboardInterrupt:
        mr.cancel()
        raise
    log(f"job state: {state}")
    for local, remote in {"job.log": f"{REMOTE}/jobs/{SESSION}/log", **fetch}.items():
        try:
            mr.download(remote, OUT / local)
        except Exception as e:  # noqa: BLE001
            log(f"could not download {local}: {str(e)[-200:]}")
            continue
        if local.endswith(".tar.gz"):
            unpack(OUT / local)
    mr.cleanup()
    log(f"done. Results in {OUT}")


def run_farmshare(a, fs, found, setup, cmd, fetch, inputs):
    """The Colab loop's twin on Slurm: one submit, one resubmit (resuming from checkpoints) if it's lost."""
    fs.push_secret(REPO / ".env")
    fs.setup(setup)
    for attempt in (1, 2):
        fs.upload(found, inputs)
        fs.submit(cmd.format(clips=" ".join(f"'{REMOTE}/clips/{p.name}'" for p in found)))
        try:
            state = fs.wait(latent_out(a) if a.job in ("latent", "variants", "multiclip") else None)
        except KeyboardInterrupt:
            fs.cancel()
            raise
        log(f"job state: {state} (attempt {attempt})")
        if state in ("done", "failed"):
            fs.fetch(fetch, unpack)
            break
        log("Slurm job lost (node failure, preemption or time limit); resubmitting")
        if any((OUT / "checkpoints").glob("policy_*/model_*.pt")):
            inputs = pack_reuse(a.reuse, policies=a.job == "robust" or a.with_weights, checkpoints=OUT / "checkpoints")
            if "--resume-from" not in cmd:
                cmd += f" --resume-from {REMOTE}/inputs/checkpoints"
            log("resubmit resumes from the synced checkpoints")
    log(f"done. Results in {OUT}")


def main(a):
    global OUT, SESSION, REMOTE, CLIPS_DIR, BRANCH
    CLIPS_DIR = Path(a.clips_dir).resolve()
    BRANCH = a.branch
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", BRANCH):
        raise SystemExit(f"odd branch name: {BRANCH}")
    pushed = subprocess.run(["git", "ls-remote", "--heads", "origin", BRANCH], cwd=REPO, capture_output=True, text=True)
    if BRANCH not in pushed.stdout:
        raise SystemExit(f"branch {BRANCH} isn't on GitHub: push it first (the GPU job clones it from there)")
    SESSION = a.job + ("_smoke" if a.smoke else "") + (f"_{a.tag}" if a.tag else "")
    OUT = REPO / "outputs" / f"remote_{SESSION}"
    OUT.mkdir(parents=True, exist_ok=True)
    fs = mr = None
    if a.backend == "modal":
        from modal_backend import ModalRun  # scripts/ is on sys.path when this file runs
        ModalRun.check_login()
        mr = ModalRun(SESSION, OUT, log, a.modal_gpu)
        REMOTE = mr.run_dir
    if a.backend == "farmshare":
        from farmshare import FarmShare  # scripts/ is on sys.path when this file runs
        fs = FarmShare.from_args(a, OUT, SESSION, log)
        log(f"FarmShare via {a.fs_host}: {fs.check_connection()} connection")
        fs.resolve()
        REMOTE = fs.run_dir
    setup, cmd, fetch = job_spec(a)
    found = [] if a.smoke or (a.reuse and a.job != "pro") else wait_for_clips(a.min_clips, a.max_wait_h)
    if a.job in ("robust", "relevance", "pushes") and not a.reuse:
        raise SystemExit(f"{a.job} needs --reuse <a latent run's results folder>")
    inputs = (pack_dir(a.reuse) if a.job == "agentround" else
              pack_reuse(a.reuse, policies=a.job in ("robust", "pushes") or a.with_weights,
                         script=REPO / "experiments/push_showcase.py" if a.job == "pushes" else None) if a.reuse
              else pack_weights(a.init_weights) if a.init_weights else None)
    if found:
        log(f"clips ready: {', '.join(p.name for p in found)}")
        check_token()
        log("HF token OK")
    if fs:
        return run_farmshare(a, fs, found, setup, cmd, fetch, inputs)
    if mr:
        return run_modal(a, mr, found, setup, cmd, fetch, inputs)
    for attempt in (1, 2):
        try:
            keeper = launch(found, setup, cmd, inputs)
        except BaseException:  # a VM that never got its job would sit idle, billed, until Colab reclaims it
            colab("stop", "-s", SESSION, check=False)
            log("launch failed; VM stopped")
            raise
        state, unknown, polls = "running", 0, 0
        while state in ("running", "unknown"):
            time.sleep(120)
            state, polls = status(), polls + 1
            unknown = unknown + 1 if state == "unknown" else 0
            if unknown >= 5:
                state = "lost"
            if state == "running" and keeper.poll() is not None:
                keeper = keep_busy()  # the exec connection dropped; keep the VM busy again
            if state == "running" and polls % 5 == 0:
                pull_log()
                sync_checkpoints(a)
        keeper.kill()
        log(f"job state: {state} (attempt {attempt})")
        if state in ("done", "failed"):
            fetch_results(fetch)
            break
        log("VM lost; retrying on a fresh one")
        colab("stop", "-s", SESSION, check=False)
        if any((OUT / "checkpoints").glob("policy_*/model_*.pt")):  # carry on from the last sync
            inputs = pack_reuse(a.reuse, policies=a.job == "robust" or a.with_weights, checkpoints=OUT / "checkpoints")
            if "--resume-from" not in cmd:
                cmd += f" --resume-from {REMOTE}/inputs/checkpoints"
            log("retry resumes from the synced checkpoints")
    colab("stop", "-s", SESSION, check=False)
    log(f"VM stopped. Results in {OUT}")


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["probe", "latent", "robust", "relevance", "variants", "multiclip", "process", "pro", "pushes",
                                        "agentround", "athletes"])
    ap.add_argument("--min-clips", type=int, default=6)
    ap.add_argument("--clips-dir", default=str(REPO / "clips"), help="where the videos to upload are (default clips/)")
    ap.add_argument("--max-wait-h", type=float, default=8)
    ap.add_argument("--target", default="squat")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="", help="names the run (outputs/remote_<job>_<tag>)")
    ap.add_argument("--branch", default="main", help="the repo branch the GPU job runs (pushed to GitHub); Modal and Colab")
    ap.add_argument("--reuse", help="an earlier latent run's results folder: its clips and rollouts")
    ap.add_argument("--with-weights", action="store_true",
                    help="with --reuse: upload its trained policies too (for --args '--init/--resume-from /content/a2h/inputs')")
    ap.add_argument("--init-weights", help="an earlier run's results folder whose policies this run's start from "
                                           "(a new video: its weights only; with --reuse use --with-weights)")
    ap.add_argument("--in-place", default="", help="pro job: comma-separated video stems to run in place "
                    "(pipeline.py --in-place: a sprint as a gait on the spot)")
    ap.add_argument("--args", default="", help="extra flags for the job's script (latent_policy.py, robustness_eval.py)")
    fs = ap.add_argument_group("FarmShare (--backend farmshare; docs/FARMSHARE.md)")
    ap.add_argument("--backend", choices=["colab", "farmshare", "modal"], default="colab")
    ap.add_argument("--modal-gpu", default=os.environ.get("A2H_MODAL_GPU"),
                    help="Modal GPU type (default L40S; e.g. A100-40GB): docs/MODAL.md")
    fs.add_argument("--fs-host", default=os.environ.get("A2H_FS_HOST", "farmshare"), help="ssh host alias")
    fs.add_argument("--fs-root", default=os.environ.get("A2H_FS_ROOT", "a2h"),
                    help="remote folder for the repo, venvs and runs (relative to home, or e.g. /scratch/users/$USER/a2h)")
    fs.add_argument("--fs-partition", default=os.environ.get("A2H_FS_PARTITION", "gpu"))
    fs.add_argument("--fs-qos", default=os.environ.get("A2H_FS_QOS", "gpu"))
    fs.add_argument("--fs-gres", default=os.environ.get("A2H_FS_GRES", "gpu:1"))
    fs.add_argument("--fs-cpus", type=int, default=8)
    fs.add_argument("--fs-mem", default="64G")
    fs.add_argument("--fs-time", default=os.environ.get("A2H_FS_TIME", "12:00:00"), help="Slurm --time")
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
