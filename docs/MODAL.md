# Modal as a GPU backend

Modal runs the same GPU jobs as Colab, billed per second, with no VM to stop. The free Starter plan includes $30 of
compute a month. An L40S costs about $1.95 an hour, so that covers about 15 GPU-hours: a training run is about an
hour, a robustness eval about 45 minutes.

```bash
python3 scripts/remote_run.py variants --tag var_sj --target squat_jump --reuse outputs/gen_sj --with-weights \
    --backend modal --args "--policies taskhybrid_var tracking_var --init /vol/runs/variants_var_sj/inputs"
```

`--args` paths that point at the job's own inputs use `/vol/runs/<job>_<tag>/inputs` on Modal, not Colab's
`/content/a2h/inputs`. Colab stays the default.

## One-time setup (you)

1. Make an account at [modal.com](https://modal.com) (the Starter plan is free).
2. Optionally, set a spending limit in the Modal dashboard (Settings → Usage and billing) so a run can never go past the free credits.
3. On this machine: `pip install modal`, then `python -m modal token new`. That opens a browser to log in, and the token is stored in `~/.modal.toml`. Don't paste it anywhere.

The first run builds the image (about 10 min, once). Later runs start in seconds.

## How it works

- **`scripts/modal_app.py`** is the Modal app.
  - **The image** holds what `scripts/setup_colab.sh` installs:
    - the mjlab virtualenv at `/content/mjlab_env`;
    - the main environment (`requirements.txt` plus torch, torchvision, transformers and matplotlib, which Colab's Python already has);
    - the G1 assets.
  - **`run_job`** clones the repo's `main` branch, unpacks the inputs, and runs the job command under `scripts/job.py`, as on Colab.
  - **Rendering:** MuJoCo renders headless through EGL on the GPU, or OSMesa on the CPU if the container doesn't expose EGL.
  - **Defaults:** GPU `L40S` (`--modal-gpu A100-40GB` or `A2H_MODAL_GPU` to change it); time limit 8 h.
- **`scripts/modal_backend.py`** is the client.
  1. Deploys the app (cached).
  2. Uploads the inputs and only the `HF_TOKEN` line of `.env` to the Volume `a2h-runs` under `runs/<session>/`.
  3. Spawns `run_job`, then polls every 2 min, syncing checkpoints as Colab does.
  4. Downloads `job.log` and the results, then deletes the run's folder from the Volume. The token file is deleted when the job ends.
- **Billing** stops when `run_job` returns or fails.
  - `Ctrl-C` in `remote_run.py` cancels the call.
  - The image has no SAM 3.1, so new videos (`--with-sam` jobs) still go to Colab. Reprocessed clips (`--reuse`) work.
