"""experiments/worldmodel_policy.py on a Modal GPU, through scripts/remote_run.py's Modal path (deploy, upload,
spawn on a pushed branch, follow, fetch, clean up) with this experiment's own inputs and command.

    python experiments/worldmodel_remote.py --tag wm_smoke --stage all --smoke --branch <branch>
    python experiments/worldmodel_remote.py --tag wm_lat1 --stage latents --branch <branch>
    python experiments/worldmodel_remote.py --tag wm_vjp1 --stage train --latents outputs/remote_wm_lat1/wm --branch <branch>

Inputs (from the main checkout's outputs/): the eight clips (five trained on, jumping jacks, punches and
discus held out), the warm start (tracking_var) and the policies compared against (multi, multi_vj,
multi_vjc) with their runs' PCAs, and stage 1's latents for stage 2. Results: outputs/remote_<tag>/.
"""

import argparse
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import remote_run as R  # noqa: E402

MAIN = Path("/Users/ethan/Documents/Github/athlete2humanoid")  # the main checkout: outputs/ and .env
O = MAIN / "outputs"
CLIPS = {**{c: O / "mc_inputs" / "clips" / c for c in ("squat_jump", "front_kick", "squat", "arms", "lunge",
                                                       "jumping_jacks", "punches")},
         "discus": O / "remote_pro_v1" / "process" / "clips" / "discus"}
CLIP_FILES = ("qpos.npy", "meta.json", "motion.csv", "silhouette.mp4", "video_latent.npz")
POLICIES = {"policy_tracking_var": O / "mc_inputs" / "policy_tracking_var" / "model_1499.pt",
            "policy_multi": O / "mc_adapt_inputs" / "policy_multi" / "model_1999.pt",
            "policy_multi_vj": O / "mc_adapt_inputs" / "policy_multi_vj" / "model_1999.pt",
            "policy_multi_vjc": O / "remote_multiclip_vjc1" / "policy_multi_vjc" / "model_1999.pt"}
PCAS = {"pca_vj.npz": O / "mc_adapt_inputs" / "vjl_pca.npz", "pca_vjc.npz": O / "remote_multiclip_vjc1" / "vjl_pca.npz"}


def pack(dst, latents=None):
    with tarfile.open(dst, "w:gz") as tar:
        for c, d in CLIPS.items():
            for f in CLIP_FILES:
                tar.add(d / f, f"clips/{c}/{f}")
        for name, f in {**PCAS, **{f"{k}/{v.name}": v for k, v in POLICIES.items()}}.items():
            tar.add(f, name)
        for f in sorted(Path(latents).glob("*.npz")) if latents else []:
            tar.add(f, f"wm/{f.name}")
    return dst


def main(a):
    R.REPO, R.BRANCH = MAIN, a.branch  # the HF token line comes from the main checkout's .env, as remote_run.py's
    R.SESSION = a.tag
    R.OUT = O / f"remote_{a.tag}"
    R.OUT.mkdir(parents=True, exist_ok=True)
    from modal_backend import ModalRun
    ModalRun.check_login()
    mr = ModalRun(R.SESSION, R.OUT, R.log, a.modal_gpu)
    R.REMOTE = inp = mr.run_dir
    # remote_run.sync_checkpoints follows <run>/outputs/multiclip_<tag>: the job writes there
    ra = R.parser().parse_args(["multiclip", "--backend", "modal", "--branch", a.branch, "--tag", a.tag])
    out = R.latent_out(ra)
    i = f"{inp}/inputs"
    cmd = (f"python3 experiments/worldmodel_policy.py {i}/clips --out {out} --stage {a.stage} --iters {a.iters} "
           f"--predictors {' '.join(a.predictors)} --pca-vjc {i}/pca_vjc.npz --pca-vj {i}/pca_vj.npz "
           f"--init {i}/policy_tracking_var/model_1499.pt --init-dir {i}")
    if a.latents:
        cmd += f" --latents {i}/wm"
    if a.stage != "train":  # the real predictor call, on a tiny model, before any GPU time goes to it
        cmd = (f"python3 -c 'import sys, types; sys.modules[\"pytest\"] = types.SimpleNamespace(importorskip=__import__); "
               f"import tests.test_world_model as t; t.test_the_predictor_never_reads_the_target_slots(); "
               f"print(\"predictor test passed\")' && " + cmd)
    if a.smoke:
        cmd += " --smoke --num-envs 256 --reps 4"
    cmd += f" {a.args}"
    R.log(f"{a.tag}: {cmd}")
    R.run_modal(ra, mr, [], "--with-main", cmd, {"results.tar.gz": f"{out}/results.tar.gz"},
                pack(R.OUT / "inputs.tar.gz", a.latents))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True, help="wm_<name>: the run's name (outputs/remote_<tag>)")
    ap.add_argument("--branch", required=True)
    ap.add_argument("--stage", choices=["latents", "train", "all"], default="train")
    ap.add_argument("--latents", help="stage 1's wm/ folder (local) for --stage train")
    ap.add_argument("--predictors", nargs="+", default=["vjepa", "learned"])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--modal-gpu", default=None)
    ap.add_argument("--args", default="", help="more flags for worldmodel_policy.py")
    main(ap.parse_args())
