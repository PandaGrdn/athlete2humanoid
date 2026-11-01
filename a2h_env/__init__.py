"""athlete2humanoid training environments: one zip per video that anyone with mjlab can train on.

    bundle = a2h_env.load_bundle("latent_s3_kick.a2h-env.zip")   # numpy only
    env = a2h_env.make_env(bundle, variant="train", num_envs=4096)  # mjlab's G1 tracking env on the variants
    python -m a2h_env info  <bundle.zip>
    python -m a2h_env train <bundle.zip> --iters 1500

The zip holds the retargeted Unitree G1 reference, its keypose task spec, every training and held-out variant with
the physics filter's verdict, the reward config, the source credit, a manifest and this package, so it works from a
clean mjlab install with nothing else of the athlete2humanoid repository on the path (docs/TRAINING_ENV.md).
Dependencies: numpy for reading; mjlab (which brings torch, mujoco, mujoco-warp and rsl-rl) for environments.
"""

from .bundle import Bundle, Variant, describe, load_bundle  # noqa: F401

VERSION = "1"  # bundle format version (manifest.json "format_version")
LOADER_FILES = ("__init__.py", "__main__.py", "bundle.py", "env.py", "train.py")  # vendored into every bundle


def make_env(bundle, variant=None, num_envs=4096, device=None, play=False, work_dir=None):
    """mjlab's Unitree G1 motion-tracking environment on this bundle's reference (variant None), one variant
    by name, or every kept training variant laid end to end ("train"). See a2h_env.env.make_env."""
    from .env import make_env as _make
    return _make(bundle, variant=variant, num_envs=num_envs, device=device, play=play, work_dir=work_dir)


def train(bundle, out, iters=1500, num_envs=4096, variant="train", device=None, seed=0, save_interval=250):
    """PPO on the bundle, as train_motion.py / experiments/variant_policy.py train the stored runs. See a2h_env.train."""
    from .train import train as _train
    return _train(bundle, out, iters=iters, num_envs=num_envs, variant=variant, device=device, seed=seed,
                  save_interval=save_interval)
