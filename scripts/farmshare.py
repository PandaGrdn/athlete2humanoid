"""Stanford FarmShare as a GPU backend for scripts/remote_run.py (--backend farmshare).

    python3 scripts/remote_run.py latent --target squat --backend farmshare

Same jobs as the Colab path, on a FarmShare L40S through Slurm:

1. reuses the SSH connection the user opened (ControlMaster; Duo is done once, by them). Every command
   runs with BatchMode=yes, so nothing here ever waits on a password or Duo prompt
2. copies the HF token line from .env into <root>/env/HF_TOKEN (mode 600, over stdin: never in argv
   or logs), rsyncs clips and the reuse archive to <root>/runs/<session>/
3. updates the repo clone and runs scripts/setup_farmshare.sh on the login node (venvs, assets; idempotent)
4. submits a batch script with sbatch, polls squeue/sacct every 2 min, rsyncs job.log and the newest
   checkpoints every 10, and fetches results with rsync when the job ends
5. a job that dies without finishing (node failure, preemption, time limit) is "lost": remote_run.py
   resubmits once, resuming from the synced checkpoints, exactly as it does for a lost Colab VM

See docs/FARMSHARE.md for the one-time setup.
"""

import json
import re
import shlex
import subprocess
import time
from pathlib import Path

RUNNING = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING", "STAGE_OUT"}
LOST = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "TIMEOUT", "DEADLINE"}
SAFE_ROOT = re.compile(r"^[\w./~$-]+$")  # $USER is fine, $( ) ` ; " are not


class FarmShareError(RuntimeError):
    pass


class FarmShare:
    def __init__(self, host, root, out, session, log, partition="gpu", qos="gpu", gres="gpu:1", cpus=8, mem="64G",
                 time_limit="12:00:00", runner=subprocess.run, sleep=time.sleep, poll_s=120):
        if not SAFE_ROOT.match(root):
            raise FarmShareError(f"--fs-root {root!r}: letters, digits, . / ~ $ _ - only")
        self.host, self.root_spec, self.out, self.session, self.log = host, root, Path(out), session, log
        self.slurm = {"partition": partition, "qos": qos, "gres": gres, "cpus-per-task": cpus, "mem": mem,
                      "time": time_limit}
        self.run, self.sleep, self.poll_s = runner, sleep, poll_s
        self.root = None  # absolute, once resolve() ran
        self.job_id = None

    @classmethod
    def from_args(cls, a, out, session, log, **kw):
        return cls(a.fs_host, a.fs_root, out, session, log, partition=a.fs_partition, qos=a.fs_qos, gres=a.fs_gres,
                   cpus=a.fs_cpus, mem=a.fs_mem, time_limit=a.fs_time, **kw)

    # --- command construction (pure; unit-tested) ------------------------------------------------
    def ssh(self, remote_cmd):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.host, remote_cmd]

    def rsync(self, src, dst, *extra):
        return ["rsync", "-az", "--partial", *extra, "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20", src, dst]

    def remote(self, path):
        """host:path for rsync, the path quoted for the remote shell."""
        return f"{self.host}:{shlex.quote(str(path))}"

    @property
    def run_dir(self):
        return f"{self.root}/runs/{self.session}"

    @property
    def repo(self):
        return f"{self.root}/athlete2humanoid"

    def resolve_cmd(self):
        root = self.root_spec[2:] if self.root_spec.startswith("~/") else self.root_spec
        # double quotes: $USER expands, nothing splits or substitutes (SAFE_ROOT rules out $( and `)
        return f'mkdir -p "{root}" && cd "{root}" && pwd -P'

    def sbatch_script(self, cmd):
        q = shlex.quote
        lines = ["#!/bin/bash", f"#SBATCH --job-name=a2h-{self.session}",
                 *[f"#SBATCH --{k}={v}" for k, v in self.slurm.items() if v not in (None, "")],
                 f"#SBATCH --output={self.run_dir}/slurm-%j.out", "set -euo pipefail",
                 f"export MUJOCO_GL=egl A2H_OUTPUT_DIR={q(self.run_dir + '/outputs')} A2H_SECRETS_DIR={q(self.root + '/env')}",
                 f"export A2H_ASSETS_DIR={q(self.repo + '/assets')} A2H_JOB_DATA={q(self.run_dir)} HF_HOME={q(self.root + '/hf')}",
                 f"export A2H_MJLAB_PYTHON={q(self.root + '/envs/mjlab/bin/python')} A2H_SAM_PYTHON={q(self.root + '/envs/sam/bin/python')}",
                 f"export A2H_SAM3D_PYTHON={q(self.root + '/sam3d_env/bin/python')} A2H_SAM3D_DIR={q(self.root + '/sam-3d-body')}",
                 f"source {q(self.root + '/envs/main/bin/activate')}",
                 f"cd {q(self.repo)}",
                 "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
                 f"python3 scripts/job.py {q(self.session)} -- bash -c {q(cmd)}", ""]
        return "\n".join(lines)

    def state_cmd(self, job_id):
        j = shlex.quote(str(job_id))
        return f"squeue -h -j {j} -o %T 2>/dev/null; echo ---; sacct -n -X -P -j {j} -o State 2>/dev/null"

    def setup_cmd(self, setup_flag):
        q = shlex.quote
        clone = (f"if [ -d {q(self.repo)}/.git ]; then git -C {q(self.repo)} pull -q --ff-only; "
                 f"else git clone -q https://github.com/PandaGrdn/athlete2humanoid.git {q(self.repo)}; fi")
        return f"{clone} && bash {q(self.repo + '/scripts/setup_farmshare.sh')} {q(self.root)} {q(setup_flag)}"

    # --- remote operations ----------------------------------------------------------------------
    def _ok(self, args, timeout=600, input=None, what=None):
        r = self.run(args, capture_output=True, text=True, timeout=timeout, input=input)
        if r.returncode != 0:
            raise FarmShareError(f"{what or args[0]} failed: {(r.stderr or r.stdout or '').strip()[-300:]}")
        return r.stdout

    def check_connection(self):
        """The user's open ControlMaster (or keys/Kerberos). Never prompts: BatchMode."""
        master = self.run(["ssh", "-O", "check", self.host], capture_output=True, text=True, timeout=30)
        if master.returncode == 0:
            return "master"
        probe = self.run(self.ssh("true"), capture_output=True, text=True, timeout=60)
        if probe.returncode == 0:
            return "direct"
        raise FarmShareError(
            f"no usable SSH connection to '{self.host}'. Open one first (password + Duo, once), then rerun:\n"
            f"    ssh {self.host}\n(see docs/FARMSHARE.md for the ControlMaster config that keeps it open)")

    def resolve(self):
        self.root = self._ok(self.ssh(self.resolve_cmd()), what="resolving --fs-root").strip().splitlines()[-1]
        return self.root

    def push_secret(self, env_file):
        """The HF_TOKEN line of .env -> <root>/env/HF_TOKEN, 600, through stdin."""
        line = next((ln for ln in Path(env_file).read_text().splitlines()
                     if re.match(r"\s*(export\s+)?HF_TOKEN\s*[=:]", ln)), None) if Path(env_file).is_file() else None
        if not line:
            self.log("no HF_TOKEN in .env; the job will use whatever is already in <root>/env on FarmShare")
            return False
        env = shlex.quote(self.root + "/env")
        self._ok(self.ssh(f"umask 077 && mkdir -p {env} && chmod 700 {env} && cat > {env}/HF_TOKEN"),
                 input=line + "\n", what="copying the HF token")
        return True

    def upload(self, clips, inputs):
        q = shlex.quote
        self._ok(self.ssh(f"mkdir -p {q(self.run_dir + '/clips')} {q(self.run_dir + '/outputs')}"), what="mkdir")
        if clips:
            self._ok(self.rsync(*[str(p) for p in clips], self.remote(self.run_dir + "/clips/")), timeout=1800,
                     what="rsync clips")
        if inputs:
            self._ok(self.rsync(str(inputs), self.remote(self.run_dir + "/inputs.tar.gz")), timeout=1800,
                     what="rsync inputs")
            inp = q(self.run_dir + "/inputs")
            self._ok(self.ssh(f"rm -rf {inp} && mkdir -p {inp} && tar -xzf {q(self.run_dir + '/inputs.tar.gz')} -C {inp}"),
                     what="unpacking inputs")

    def setup(self, setup_flag):
        self.log(f"updating the repo and environments on {self.host} (first time: ~20 min)")
        self._ok(self.ssh(self.setup_cmd(setup_flag)), timeout=3 * 3600, what="setup_farmshare.sh")

    def submit(self, cmd):
        script = self.run_dir + "/job.sbatch"
        self._ok(self.ssh(f"cat > {shlex.quote(script)}"), input=self.sbatch_script(cmd), what="writing job.sbatch")
        out = self._ok(self.ssh(f"sbatch --parsable {shlex.quote(script)}"), what="sbatch")
        m = re.match(r"\s*(\d+)", out)
        if not m:
            raise FarmShareError(f"sbatch said {out.strip()[:200]!r}")
        self.job_id = m.group(1)
        self.log(f"submitted Slurm job {self.job_id} ({self.slurm['partition']}/{self.slurm['qos']}, {self.slurm['gres']})")
        return self.job_id

    def slurm_state(self):
        r = self.run(self.ssh(self.state_cmd(self.job_id)), capture_output=True, text=True, timeout=120)
        if r.returncode != 0 and "---" not in (r.stdout or ""):
            return "UNKNOWN"
        queued, _, acct = (r.stdout or "").partition("---")
        states = [s.strip().split()[0] for s in (queued.strip().splitlines() or acct.strip().splitlines()) if s.strip()]
        return states[0].rstrip("+") if states else "UNKNOWN"

    def job_status(self):
        """status.json's state as written by scripts/job.py: running | done | failed, or None."""
        path = f"{self.run_dir}/jobs/{self.session}/status.json"
        r = self.run(self.rsync(self.remote(path), str(self.out / "status.json")), capture_output=True, text=True,
                     timeout=120)
        if r.returncode != 0:
            return None
        try:
            return json.loads((self.out / "status.json").read_text()).get("state")
        except (OSError, ValueError):
            return None

    def state(self):
        """running | done | failed | lost | unknown, from Slurm first and status.json once it has left the queue."""
        s = self.slurm_state()
        if s in RUNNING:
            return "running"
        if s == "UNKNOWN":
            return "unknown"
        js = self.job_status()
        if js in ("done", "failed"):
            return js
        if s in LOST or js == "running":  # died mid-job: resubmit and resume
            return "lost"
        return "failed"  # never got as far as job.py (bad script, env): look at slurm-<id>.out

    def pull_log(self):
        r = self.run(self.rsync(self.remote(f"{self.run_dir}/jobs/{self.session}/log"), str(self.out / "job.log")),
                     capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            self.log("could not fetch job.log yet")

    def sync_checkpoints(self, outputs_dir):
        """Newest model_*.pt of each policy -> OUT/checkpoints/policy_*/ (what a resubmit resumes from)."""
        code = ("import glob, os, sys\nfor d in glob.glob(sys.argv[1] + '/policy_*'):\n"
                "    c = sorted(glob.glob(d + '/logs/*/*/model_*.pt'), key=lambda p: int(p.rsplit('_', 1)[1][:-3]))\n"
                "    if c: print(os.path.relpath(c[-1], sys.argv[1]))\n")
        r = self.run(self.ssh(f"python3 -c {shlex.quote(code)} {shlex.quote(outputs_dir)}"), capture_output=True,
                     text=True, timeout=120)
        names = [n for n in (r.stdout or "").split() if n.endswith(".pt")] if r.returncode == 0 else []
        for rel in names:
            policy = rel.split("/", 1)[0]
            dest = self.out / "checkpoints" / policy
            dest.mkdir(parents=True, exist_ok=True)
            got = self.run(self.rsync(self.remote(f"{outputs_dir}/{rel}"), str(dest / Path(rel).name)),
                           capture_output=True, text=True, timeout=600)
            if got.returncode == 0:
                for old in dest.glob("model_*.pt"):
                    if old.name != Path(rel).name:
                        old.unlink()
        if names:
            self.log("synced checkpoints: " + ", ".join(names))

    def fetch(self, fetch, unpack):
        """fetch: {local name: remote path} from job_spec; unpack(path) for .tar.gz results."""
        self.pull_log()
        if self.job_id:
            self.run(self.rsync(self.remote(f"{self.run_dir}/slurm-{self.job_id}.out"), str(self.out / "slurm.out")),
                     capture_output=True, text=True, timeout=120)
        for local, remote in fetch.items():
            r = self.run(self.rsync(self.remote(remote), str(self.out / local)), capture_output=True, text=True,
                         timeout=1800)
            if r.returncode != 0:
                self.log(f"could not download {local}: {(r.stderr or '').strip()[-200:]}")
                continue
            if local.endswith(".tar.gz"):
                unpack(self.out / local)

    def cancel(self):
        if self.job_id:
            self.run(self.ssh(f"scancel {shlex.quote(self.job_id)}"), capture_output=True, text=True, timeout=60)
            self.log(f"cancelled Slurm job {self.job_id}")

    def wait(self, ckpt_dir=None):
        """Poll until the job leaves the queue. Returns done | failed | lost."""
        polls = unknown = 0
        while True:
            self.sleep(self.poll_s)
            state, polls = self.state(), polls + 1
            unknown = unknown + 1 if state == "unknown" else 0
            if state == "unknown" and unknown < 5:
                continue
            if state == "unknown":  # 10 min without an answer: the connection is gone
                raise FarmShareError(f"lost contact with {self.host}; reopen `ssh {self.host}` and rerun "
                                     f"(Slurm job {self.job_id} keeps running; see docs/FARMSHARE.md)")
            if state != "running":
                return state
            if polls % 5 == 0:
                self.pull_log()
                if ckpt_dir:
                    self.sync_checkpoints(ckpt_dir)
