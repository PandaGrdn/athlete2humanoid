"""scripts/modal_backend.py without a Modal account: a stand-in Volume and function call."""

import io
import sys
from pathlib import Path

import pytest

pytest.importorskip("modal")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import modal_backend as MB  # noqa: E402


class FakeBatch:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def put_file(self, src, dst):
        self.store[dst] = src.read() if isinstance(src, io.BytesIO) else Path(src).read_bytes()


class FakeVolume:
    def __init__(self):
        self.files, self.removed = {}, []

    def batch_upload(self, force=False):
        return FakeBatch(self.files)

    def read_file(self, rel):
        data = self.files[rel]
        yield data[:3]
        yield data[3:]

    def remove_file(self, rel, recursive=False):
        self.removed.append(rel)


class FakeCall:
    object_id = "fc-123"

    def __init__(self, results):
        self.results = list(results)

    def get(self, timeout=0):
        r = self.results.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def make(tmp_path):
    r = MB.ModalRun("variants_x", tmp_path, lambda m: None)
    r.vol = FakeVolume()
    return r


def test_uploads_only_the_hf_token_line(tmp_path):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN = hf_" + "a" * 30 + "\n\nMUSE_API_KEY = secret-muse\n")
    clip = tmp_path / "kick.mov"
    clip.write_bytes(b"video")
    inputs = tmp_path / "inputs.tar.gz"
    inputs.write_bytes(b"tar")
    r = make(tmp_path)
    r.upload([clip], inputs, env)
    f = r.vol.files
    assert f["runs/variants_x/clips/kick.mov"] == b"video" and f["runs/variants_x/inputs.tar.gz"] == b"tar"
    assert f["runs/variants_x/env/HF_TOKEN"].decode().startswith("HF_TOKEN = hf_")
    assert not any(b"MUSE" in v for v in f.values())  # nothing else from .env leaves the machine
    assert r.run_dir == "/vol/runs/variants_x"


def test_download_maps_the_job_path_to_the_volume(tmp_path):
    r = make(tmp_path)
    r.vol.files["runs/variants_x/outputs/results.tar.gz"] = b"abcdef"
    r.download("/vol/runs/variants_x/outputs/results.tar.gz", tmp_path / "got" / "results.tar.gz")
    assert (tmp_path / "got" / "results.tar.gz").read_bytes() == b"abcdef"


def test_state_and_wait(tmp_path):
    r = make(tmp_path)
    r.call = FakeCall([TimeoutError(), TimeoutError(), 0])
    synced = []
    assert MB.wait(r, lambda: synced.append(1), poll_s=0) == "done" and len(synced) == 2
    r.call = FakeCall([1])
    assert r.state() == "failed"
    r.call = FakeCall([RuntimeError("container OOM")])
    assert r.state() == "failed"


def test_remote_run_routes_paths_to_the_volume(tmp_path, monkeypatch):
    import remote_run
    a = remote_run.parser().parse_args(["variants", "--tag", "x", "--target", "squat_jump", "--reuse", "outputs/gen_sj",
                                        "--backend", "modal"])
    monkeypatch.setattr(remote_run, "REMOTE", "/vol/runs/variants_x")
    setup, cmd, fetch = remote_run.job_spec(a)
    assert setup == "--with-main" and "/vol/runs/variants_x/inputs/clips/squat_jump" in cmd
    assert fetch == {"results.tar.gz": "/vol/runs/variants_x/outputs/variants_x/results.tar.gz"}
