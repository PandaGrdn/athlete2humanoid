"""remote_run.py --backend farmshare: ssh / rsync / sbatch command construction and the Slurm state
machine, with subprocess mocked (no network, no FarmShare account needed).

    python -m unittest tests.test_farmshare
"""

import argparse
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import farmshare as F  # noqa: E402
import remote_run as RR  # noqa: E402


class FakeRunner:
    """Records every command; answers from (predicate, CompletedProcess) rules, first match wins."""

    def __init__(self, rules=()):
        self.calls, self.rules = [], list(rules)

    def __call__(self, args, **kw):
        self.calls.append((args, kw))
        for pred, out in self.rules:
            if pred(args):
                return out(args, kw) if callable(out) else out
        return subprocess.CompletedProcess(args, 0, "", "")

    def remote_cmds(self):
        return [a[-1] for a, _ in self.calls if a[0] == "ssh" and a[1] != "-O"]


def done(stdout="", code=0, stderr=""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def ssh_has(text):
    return lambda a: a[0] == "ssh" and text in a[-1]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.logs = []

    def fs(self, runner, **kw):
        f = F.FarmShare("farmshare", "a2h", self.tmp, "latent_v3", self.logs.append, runner=runner,
                        sleep=lambda s: None, **kw)
        f.root = "/home/users/me/a2h"
        return f


class CommandConstruction(Base):
    def test_ssh_never_prompts(self):
        cmd = self.fs(FakeRunner()).ssh("true")
        self.assertEqual(cmd[:2], ["ssh", "-o"])
        self.assertIn("BatchMode=yes", cmd)
        self.assertEqual(cmd[-2:], ["farmshare", "true"])

    def test_rsync_goes_through_batch_ssh(self):
        cmd = self.fs(FakeRunner()).rsync("a.mp4", "farmshare:/x/")
        self.assertEqual(cmd[0], "rsync")
        self.assertIn("-az", cmd)
        self.assertIn("BatchMode=yes", cmd[cmd.index("-e") + 1])
        self.assertEqual(cmd[-2:], ["a.mp4", "farmshare:/x/"])

    def test_remote_paths_are_quoted(self):
        f = self.fs(FakeRunner())
        self.assertEqual(f.remote("/a b/c"), "farmshare:'/a b/c'")
        self.assertEqual(f.run_dir, "/home/users/me/a2h/runs/latent_v3")

    def test_root_spec(self):
        self.assertEqual(F.FarmShare("h", "~/a2h", self.tmp, "s", print).resolve_cmd(),
                         'mkdir -p "a2h" && cd "a2h" && pwd -P')
        self.assertIn('"/scratch/users/$USER/a2h"', F.FarmShare("h", "/scratch/users/$USER/a2h", self.tmp, "s", print).resolve_cmd())
        for bad in ("a2h; rm -rf ~", "$(whoami)", "`id`", 'a"b'):
            with self.assertRaises(F.FarmShareError):
                F.FarmShare("h", bad, self.tmp, "s", print)

    def test_sbatch_script(self):
        f = self.fs(FakeRunner(), partition="gpu", qos="gpu", gres="gpu:1", time_limit="06:00:00")
        s = f.sbatch_script("python3 experiments/latent_policy.py 'x y' --iters 10")
        lines = s.splitlines()
        self.assertEqual(lines[0], "#!/bin/bash")
        for d in ("--partition=gpu", "--qos=gpu", "--gres=gpu:1", "--time=06:00:00", "--cpus-per-task=8", "--mem=64G",
                  "--job-name=a2h-latent_v3", "--output=/home/users/me/a2h/runs/latent_v3/slurm-%j.out"):
            self.assertIn(f"#SBATCH {d}", lines)
        self.assertIn("A2H_JOB_DATA=/home/users/me/a2h/runs/latent_v3", s)
        self.assertIn("A2H_MJLAB_PYTHON=/home/users/me/a2h/envs/mjlab/bin/python", s)
        self.assertIn("source /home/users/me/a2h/envs/main/bin/activate", s)
        last = [ln for ln in lines if ln.startswith("python3 scripts/job.py")][0]
        # the job command survives the shell round trip intact
        self.assertEqual(shlex.split(last)[-1], "python3 experiments/latent_policy.py 'x y' --iters 10")

    def test_empty_partition_is_left_out(self):
        s = self.fs(FakeRunner(), partition="").sbatch_script("true")
        self.assertNotIn("--partition", s)

    def test_setup_cmd_clones_or_pulls(self):
        c = self.fs(FakeRunner()).setup_cmd("--with-sam")
        self.assertIn("git -C /home/users/me/a2h/athlete2humanoid pull -q --ff-only", c)
        self.assertIn("git clone -q https://github.com/PandaGrdn/athlete2humanoid.git", c)
        self.assertTrue(c.endswith("setup_farmshare.sh /home/users/me/a2h --with-sam"))


class Operations(Base):
    def test_connection_prefers_the_control_master(self):
        r = FakeRunner()
        self.assertEqual(self.fs(r).check_connection(), "master")
        self.assertEqual(r.calls[0][0], ["ssh", "-O", "check", "farmshare"])

    def test_no_connection_explains_what_to_do(self):
        r = FakeRunner([(lambda a: True, done(code=255, stderr="Permission denied"))])
        with self.assertRaises(F.FarmShareError) as e:
            self.fs(r).check_connection()
        self.assertIn("ssh farmshare", str(e.exception))

    def test_resolve(self):
        r = FakeRunner([(ssh_has("pwd -P"), done("/home/users/me/a2h\n"))])
        f = F.FarmShare("farmshare", "a2h", self.tmp, "s", print, runner=r)
        self.assertEqual(f.resolve(), "/home/users/me/a2h")

    def test_secret_goes_over_stdin_only(self):
        token = "hf_" + "Z" * 34
        env = self.tmp / ".env"
        env.write_text(f'OTHER_KEY="nope"\nHF_TOKEN = "{token}"\n')
        r = FakeRunner()
        self.assertTrue(self.fs(r).push_secret(env))
        (args, kw), = r.calls
        self.assertNotIn(token, " ".join(args))
        self.assertEqual(kw["input"], f'HF_TOKEN = "{token}"\n')  # only that line, not OTHER_KEY
        self.assertIn("umask 077", args[-1])
        self.assertIn("cat > /home/users/me/a2h/env/HF_TOKEN", args[-1])

    def test_submit_parses_job_id(self):
        r = FakeRunner([(ssh_has("sbatch --parsable"), done("4242;farmshare\n"))])
        f = self.fs(r)
        self.assertEqual(f.submit("true"), "4242")
        write, submit = r.calls
        self.assertEqual(write[0][-1], "cat > /home/users/me/a2h/runs/latent_v3/job.sbatch")
        self.assertTrue(write[1]["input"].startswith("#!/bin/bash"))

    def test_sbatch_error_surfaces(self):
        r = FakeRunner([(ssh_has("sbatch"), done(code=1, stderr="sbatch: error: Invalid qos specification"))])
        with self.assertRaisesRegex(F.FarmShareError, "Invalid qos"):
            self.fs(r).submit("true")

    def test_upload(self):
        r = FakeRunner()
        clip = self.tmp / "squat.mp4"
        clip.write_bytes(b"x")
        self.fs(r).upload([clip], self.tmp / "inputs.tar.gz")
        rsyncs = [a for a, _ in r.calls if a[0] == "rsync"]
        self.assertEqual(rsyncs[0][-2:], [str(clip), "farmshare:/home/users/me/a2h/runs/latent_v3/clips/"])
        self.assertEqual(rsyncs[1][-1], "farmshare:/home/users/me/a2h/runs/latent_v3/inputs.tar.gz")
        self.assertIn("tar -xzf", r.remote_cmds()[-1])


class SlurmStates(Base):
    def run_with(self, squeue, sacct="", status=None):
        def rsync_status(args, kw):
            if status is None:
                return done(code=23)
            Path(args[-1]).write_text(f'{{"state": "{status}"}}')
            return done()
        return FakeRunner([(ssh_has("squeue"), done(f"{squeue}\n---\n{sacct}\n")),
                           (lambda a: a[0] == "rsync" and a[-1].endswith("status.json"), rsync_status)])

    def state(self, *a, **k):
        f = self.fs(self.run_with(*a, **k))
        f.job_id = "7"
        return f.state()

    def test_states(self):
        self.assertEqual(self.state("PENDING"), "running")
        self.assertEqual(self.state("RUNNING"), "running")
        self.assertEqual(self.state("", "COMPLETED", status="done"), "done")
        self.assertEqual(self.state("", "COMPLETED", status="failed"), "failed")
        self.assertEqual(self.state("", "FAILED"), "failed")             # never reached job.py
        self.assertEqual(self.state("", "NODE_FAIL", status="running"), "lost")
        self.assertEqual(self.state("", "TIMEOUT", status="running"), "lost")  # resubmit resumes
        self.assertEqual(self.state("", "CANCELLED+"), "failed")

    def test_unknown_when_ssh_fails(self):
        f = self.fs(FakeRunner([(lambda a: True, done(code=255))]))
        f.job_id = "7"
        self.assertEqual(f.state(), "unknown")

    def test_wait_gives_up_after_repeated_silence(self):
        f = self.fs(FakeRunner([(lambda a: True, done(code=255))]))
        f.job_id = "7"
        with self.assertRaisesRegex(F.FarmShareError, "lost contact"):
            f.wait()

    def test_wait_syncs_while_running(self):
        seq = iter(["RUNNING"] * 5 + [""])

        def squeue(args, kw):
            s = next(seq)
            return done(f"{s}\n---\n{'COMPLETED' if not s else ''}\n")
        r = FakeRunner([(ssh_has("squeue"), squeue), (ssh_has("python3 -c"), done("policy_latent/logs/a/b/model_99.pt\n"))])
        f = self.fs(r)
        f.job_id = "7"
        self.assertEqual(f.wait("/home/users/me/a2h/runs/latent_v3/outputs/latent_v3"), "failed")
        fetched = [a[-2] for a, _ in r.calls if a[0] == "rsync"]
        self.assertTrue(any(p.endswith("/jobs/latent_v3/log") for p in fetched))
        self.assertTrue(any(p.endswith("policy_latent/logs/a/b/model_99.pt") for p in fetched))


class RemoteRunDispatch(unittest.TestCase):
    def args(self, **kw):
        base = dict(job="latent", min_clips=1, max_wait_h=0, target="squat", iters=10, smoke=False, tag="t",
                    reuse=None, with_weights=False, args="", backend="farmshare", fs_host="farmshare", fs_root="a2h",
                    fs_partition="gpu", fs_qos="gpu", fs_gres="gpu:1", fs_cpus=8, fs_mem="64G", fs_time="01:00:00")
        base.update(kw)
        return argparse.Namespace(**base)

    def test_farmshare_paths_replace_content(self):
        """job_spec sees FarmShare's run folder instead of /content/a2h; colab is untouched."""
        with mock.patch.object(RR, "REMOTE", "/home/users/me/a2h/runs/latent_t"):
            setup, cmd, fetch = RR.job_spec(self.args())
        self.assertIn("--out /home/users/me/a2h/runs/latent_t/outputs/latent_t", cmd)
        self.assertEqual(fetch["results.tar.gz"], "/home/users/me/a2h/runs/latent_t/outputs/latent_t/results.tar.gz")
        self.assertEqual(RR.REMOTE, "/content/a2h")

    def test_colab_is_the_default(self):
        a = RR.parser().parse_args(["latent"])
        self.assertEqual((a.backend, a.fs_host, a.fs_qos, a.fs_gres), ("colab", "farmshare", "gpu", "gpu:1"))
        self.assertEqual(RR.parser().parse_args(["latent", "--backend", "farmshare", "--fs-time", "2:00:00"]).fs_time,
                         "2:00:00")

    def test_lost_job_is_resubmitted_from_checkpoints(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        ck = tmp / "checkpoints" / "policy_latent"
        ck.mkdir(parents=True)
        (ck / "model_500.pt").write_bytes(b"pt")
        fs = mock.Mock()
        fs.wait.side_effect = ["lost", "done"]
        remote = "/home/users/me/a2h/runs/latent_t"
        with mock.patch.object(RR, "OUT", tmp), mock.patch.object(RR, "REMOTE", remote):
            RR.run_farmshare(self.args(), fs, [], "--with-main", "python3 x.py {clips}", {"r.tar.gz": "/r"}, None)
        self.assertEqual(fs.submit.call_count, 2)
        self.assertNotIn("--resume-from", fs.submit.call_args_list[0].args[0])
        self.assertIn(f"--resume-from {remote}/inputs/checkpoints", fs.submit.call_args_list[1].args[0])
        second_inputs = fs.upload.call_args_list[1].args[1]
        import tarfile
        with tarfile.open(second_inputs) as t:
            self.assertIn("checkpoints/policy_latent/model_500.pt", t.getnames())
        fs.fetch.assert_called_once()
        fs.setup.assert_called_once_with("--with-main")

    def test_ctrl_c_cancels_the_slurm_job(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        fs = mock.Mock()
        fs.wait.side_effect = KeyboardInterrupt
        with mock.patch.object(RR, "OUT", tmp), self.assertRaises(KeyboardInterrupt):
            RR.run_farmshare(self.args(), fs, [], "", "true", {}, None)
        fs.cancel.assert_called_once()

if __name__ == "__main__":
    unittest.main()
