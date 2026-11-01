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

VERSION = "1"  # bundle format version (manifest.json "format_version")
LOADER_FILES = ("__init__.py", "__main__.py", "bundle.py", "env.py", "train.py", "attempts.py")  # vendored into every bundle

# Everything is imported on first use: the API server imports a2h_env.export with nothing but the standard library.


def load_bundle(path):
    """A Bundle from a .a2h-env.zip or its unzipped folder (numpy only). See a2h_env.bundle."""
    from .bundle import load_bundle as _load
    return _load(path)


def describe(bundle):
    """What a bundle holds, as `python -m a2h_env info` prints it."""
    from .bundle import describe as _describe
    return _describe(bundle)


def __getattr__(name):
    if name in ("Bundle", "Variant"):
        from . import bundle
        return getattr(bundle, name)
    raise AttributeError(name)


def make_env(bundle, variant="train", num_envs=4096, device=None, play=False, work_dir=None):
    """mjlab's Unitree G1 motion-tracking environment on every kept training variant laid end to end ("train", the
    default, as `train` and the CLI; the reference when the bundle has none), the reference ("reference" or None), or
    one variant by name. See a2h_env.env.make_env."""
    from .env import make_env as _make
    return _make(bundle, variant=variant, num_envs=num_envs, device=device, play=play, work_dir=work_dir)


def train(bundle, out, iters=1500, num_envs=4096, variant="train", device=None, seed=0, save_interval=250,
          work_dir=None, attempt_every=0):
    """PPO on the bundle, as train_motion.py / experiments/variant_policy.py train the stored runs; attempt_every > 0 also
    records the policy's attempts on the reference (a2h_env.attempts). See a2h_env.train."""
    from .train import train as _train
    return _train(bundle, out, iters=iters, num_envs=num_envs, variant=variant, device=device, seed=seed,
                  save_interval=save_interval, work_dir=work_dir, attempt_every=attempt_every)
