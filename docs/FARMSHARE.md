# Stanford FarmShare as a GPU backend

When Colab compute units run out, the same jobs can run on FarmShare, Stanford's free cluster for coursework and
unsponsored research:

```bash
python3 scripts/remote_run.py latent --target squat --backend farmshare
```

Colab stays the default; nothing changes for `remote_run.py` without `--backend farmshare`.

## What FarmShare offers (checked 2026-10-27)

- **GPUs:** "oat" nodes. The December 2025 expansion added 6 GPU nodes with 4 NVIDIA L40S (48 GB) each
  ([FarmShare docs, home page](https://docs.farmshare.stanford.edu/)). An L40S is roughly an A100-40GB for this
  workload: mjlab's 4,096 environments and V-JEPA 2 ViT-L both fit.
- **Scheduler:** Slurm 25.11 ([docs](https://docs.farmshare.stanford.edu/)). GPU jobs use the `gpu` QoS: at least 1
  GPU per job, at most 6 GPUs and 4 running jobs per user; the `normal` QoS allows 3 GPUs per user, and `dev` 1 GPU
  for up to 8 h ([Running Jobs](https://docs.farmshare.stanford.edu/slurm)). The current docs list the partitions
  `normal`, `interactive` and `bigmem` and don't name a GPU partition; older FarmShare guides used
  `--partition=gpu --qos=gpu --gres=gpu:1`. That is the default here, and step 4 below checks it.
- **Login:** `login.farmshare.stanford.edu`, load-balanced over the "rice" login nodes; password + Duo
  ([Connecting](https://docs.farmshare.stanford.edu/connecting/)). Needs a full-service SUNetID.
- **Storage:** `$HOME` and `/scratch` are on an all-NVMe appliance visible from every node; quotas aren't
  published, so large outputs are better under `/scratch/users/$USER`
  ([Software](https://docs.farmshare.stanford.edu/software/), [docs](https://docs.farmshare.stanford.edu/)).
- **Software:** Lmod modules (Python 3.10–3.13, CUDA 11.4, micromamba, Apptainer). This setup doesn't use them:
  `uv` brings its own Python 3.12 and the CUDA wheels bring their own runtime; only the NVIDIA driver comes from the node.
- **Policy:** coursework and unsponsored research; no high-risk data
  ([Policy](https://docs.farmshare.stanford.edu/policy/)). Public athlete videos are fine; don't upload footage of
  people who haven't agreed to it.

## How it works

`scripts/farmshare.py`, called by `remote_run.py`:

1. Reuses your open SSH connection (ControlMaster). Every command runs with `BatchMode=yes`, so the script never
   waits on a password or Duo prompt: if the connection is closed, it stops and tells you to run `ssh farmshare`.
2. Copies only the `HF_TOKEN` line of `.env` to `<root>/env/HF_TOKEN` (mode 600, over stdin: never on a command
   line or in a log), and rsyncs the clips (and `--reuse` inputs) to `<root>/runs/<session>/`.
3. On the login node: clones or fast-forwards the repo under `<root>/athlete2humanoid` (from GitHub, like Colab:
   push your changes first), then runs `scripts/setup_farmshare.sh`, which builds three uv environments
   (`mjlab`, `main`, `sam`), fetches the G1 model and pre-downloads the gated V-JEPA 2 and SAM 3.1 weights into
   `<root>/hf` (in case compute nodes can't reach the Hub). The first run takes about 20 minutes; later runs, seconds.
4. Writes `job.sbatch` and submits it. The job runs `scripts/job.py`, which writes `status.json` and `log` exactly as
   on Colab.
5. Every 2 minutes polls `squeue` (then `sacct` once the job has left the queue); every 10 minutes copies `job.log`
   and the newest checkpoint of each policy back to `outputs/remote_<session>/`.
6. When the job ends, rsyncs the results (and `slurm.out`) into `outputs/remote_<session>/`, the same layout as a
   Colab run, so the web app's gallery picks it up.
7. A job that dies without finishing (node failure, preemption, or the `--fs-time` limit) is resubmitted once,
   resuming from the synced checkpoints. Ctrl-C cancels the Slurm job (`scancel`).

## One-time setup (you)

### 1. SSH config with a shared connection

Add to `~/.ssh/config` on the Mac (replace `SUNETID`):

```
Host farmshare
    HostName login.farmshare.stanford.edu
    User SUNETID
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 12h
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

`ControlPersist 12h` keeps the authenticated connection open in the background for 12 hours after the last
session closes, so Duo is needed once per day, not once per `ssh`/`rsync` call. `login.farmshare.stanford.edu`
picks a login node per connection; the shared connection pins you to one, which is what the job needs.

### 2. First login (password + Duo)

```bash
ssh farmshare
```

Check the host key against the published fingerprint, ED25519
`SHA256:bKb1Znir/1tOg+TMyALDYWeK0lclsulriDN8aOvWteU`
([Connecting](https://docs.farmshare.stanford.edu/connecting/)), then approve the Duo push. Your first login
also creates your Slurm account. Leave this terminal open, or close it: the master connection stays up either way.
To check it from another terminal:

```bash
ssh -O check farmshare
```

### 3. Pick where runs live (optional)

The default is `~/a2h`. To use scratch instead (more room), set this in the shell you launch jobs from:

```bash
export A2H_FS_ROOT='/scratch/users/$USER/a2h'
```

### 4. Confirm the GPU partition and QoS (once)

```bash
ssh farmshare 'sinfo -o "%P %G %D %N" | grep -i gpu; sacctmgr -n show assoc user=$USER format=partition,qos%40'
```

If the L40S nodes sit in a partition other than `gpu` (for example `normal`), pass it on every run with
`--fs-partition normal`, or `export A2H_FS_PARTITION=normal`. An empty value (`--fs-partition ''`) leaves the
partition to Slurm's default. A two-minute test that the request is accepted and a GPU shows up:

```bash
ssh farmshare 'srun --partition=gpu --qos=gpu --gres=gpu:1 --time=00:02:00 nvidia-smi -L'
```

### 5. Environment setup (optional ahead of time)

`remote_run.py` runs this before every job anyway, but running it once by hand shows any problem early. It
needs the repo pushed to GitHub:

```bash
ssh farmshare 'git clone -q https://github.com/PandaGrdn/athlete2humanoid.git ~/a2h/athlete2humanoid; bash ~/a2h/athlete2humanoid/scripts/setup_farmshare.sh ~/a2h --with-sam'
```

It ends with `SETUP_DONE`. If `uv` isn't on FarmShare yet, the script installs it into `~/.local/bin` with the
official installer.

## Running jobs

Every `remote_run.py` job works; add `--backend farmshare`:

```bash
python3 scripts/remote_run.py latent --target squat --tag fs1 --backend farmshare
```

```bash
python3 scripts/remote_run.py robust --reuse outputs/remote_latent --backend farmshare --fs-time 04:00:00
```

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--fs-host` | `A2H_FS_HOST` | `farmshare` | SSH host alias from step 1 |
| `--fs-root` | `A2H_FS_ROOT` | `a2h` (in home) | Repo, venvs, weights cache, runs |
| `--fs-partition` | `A2H_FS_PARTITION` | `gpu` | Slurm partition (step 4) |
| `--fs-qos` | `A2H_FS_QOS` | `gpu` | Slurm QoS |
| `--fs-gres` | `A2H_FS_GRES` | `gpu:1` | GPUs per job |
| `--fs-time` | `A2H_FS_TIME` | `12:00:00` | Wall-clock limit; a timed-out job is resubmitted once from its checkpoints |
| `--fs-cpus`, `--fs-mem` | | `8`, `64G` | CPU cores and memory |

The web app's GPU runner can use FarmShare too: start the API locally with `A2H_RUNNERS=demo,colab
A2H_ENABLE_COLAB=1 A2H_REMOTE_BACKEND=farmshare` (never on the hosted Space; see `docs/SECURITY.md`).

Progress: `outputs/remote_<session>/runner.log` (this script), `job.log` (the job), `slurm.out` (Slurm's own
output, fetched at the end; look here first if a job fails before it starts). On FarmShare:
`squeue -u $USER`, and `sacct -j <id>` after it ends.

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `no usable SSH connection to 'farmshare'` | the master connection expired or the Mac slept | `ssh farmshare` (Duo), rerun |
| `lost contact with farmshare` mid-job | as above, during polling | The Slurm job keeps running. Reopen `ssh farmshare`. Then wait for it (`squeue -u $USER`) and rsync `<root>/runs/<session>/outputs/` back by hand, or rerun the same command with a new `--tag` |
| `sbatch: error: Invalid partition/qos` | the defaults don't match FarmShare's current setup | step 4, then `--fs-partition` / `--fs-qos` |
| Job pending for hours | the L40S nodes are busy (shared cluster; 24 GPUs total) | wait, or use `--fs-qos dev --fs-time 08:00:00` for short runs |
| `EGL` / rendering errors in `job.log` | the node's NVIDIA EGL vendor file is missing | ask srcc-support (subject "FarmShare"); short-term, set `MUJOCO_GL=osmesa` in `job.sbatch` (slow) |
| `ModuleNotFoundError` in `job.log` | Colab preinstalls packages the main env lacks | add the package to the `--with-main` line of `scripts/setup_farmshare.sh` |
| CUDA "no kernel image" or driver errors | the CUDA wheels need a newer driver than the node has | `export A2H_TORCH_BACKEND=cu121` in the setup command (step 5) |

Not tested end to end yet: this was built without a FarmShare account in the loop. The command construction,
Slurm state handling, resubmission and secret handling are unit-tested with mocks (`tests/test_farmshare.py`);
the first real run will show whether the default partition and the environments need the adjustments above.
