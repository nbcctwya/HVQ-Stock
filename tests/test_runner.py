"""Unit tests for the Phase 2 executor (experiments/runner.py).

All tests use temporary fake artifacts; no real experiment artifacts,
training, or git operations are involved.
"""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve().parent.parent / "experiments" / "runner.py"
spec = importlib.util.spec_from_file_location("runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def make_repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "artifacts").mkdir(parents=True)
    return repo


def make_ckpt(run_dir: Path, name: str, content: bytes = b"x" * 16) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / name
    p.write_bytes(content)
    return p


class Stage1MarkerRestoreTest(unittest.TestCase):
    """Fix 1: resume must use the exact checkpoint recorded in .stage1.done."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        self.run_dir = self.repo / "artifacts" / "009" / "run"
        self.run_dir.mkdir(parents=True)

    def test_marker_restores_exact_ckpt_despite_lower_val_loss_stage2_ckpt(self):
        s1 = make_ckpt(self.run_dir, "infu_x-epoch=10-val_loss=0.9000.ckpt")
        # A stage2 checkpoint with a much lower val_loss sits in the same dir.
        make_ckpt(self.run_dir, "0_VQK512_x-epoch=3-val_loss=0.1000.ckpt")
        (self.run_dir / ".stage1.done").write_text(f"best={s1.name}\nval_loss=0.9\n")

        ckpt, prov = runner.stage1(self.repo, self.run_dir, {"id": "009"})

        self.assertEqual(ckpt, s1.resolve())
        self.assertEqual(prov["mode"], "self")

    def test_missing_marker_file_falls_back_to_rescan(self):
        # No marker: resolve_marker_ckpt has nothing to read; a fresh scan
        # (as used right after a real stage1 run) still works.
        make_ckpt(self.run_dir, "infu_x-epoch=2-val_loss=0.7000.ckpt")
        best = runner.find_best_stage1_ckpt(self.run_dir)
        self.assertEqual(best[1].name, "infu_x-epoch=2-val_loss=0.7000.ckpt")

    def test_marker_pointing_to_missing_file_is_invalid(self):
        (self.run_dir / ".stage1.done").write_text("best=gone-epoch=1-val_loss=0.5.ckpt\n")
        info = runner.read_stage1_marker(self.run_dir / ".stage1.done")
        self.assertIsNone(runner.resolve_marker_ckpt(self.repo, self.run_dir, info))

    def test_marker_pointing_to_empty_file_is_invalid(self):
        make_ckpt(self.run_dir, "empty-epoch=1-val_loss=0.5.ckpt", content=b"")
        (self.run_dir / ".stage1.done").write_text("best=empty-epoch=1-val_loss=0.5.ckpt\n")
        info = runner.read_stage1_marker(self.run_dir / ".stage1.done")
        self.assertIsNone(runner.resolve_marker_ckpt(self.repo, self.run_dir, info))

    def test_marker_pointing_outside_artifacts_is_invalid(self):
        outside = self.tmp / "evil.ckpt"
        outside.write_bytes(b"x" * 16)
        info = {"best": str(outside)}
        self.assertIsNone(runner.resolve_marker_ckpt(self.repo, self.run_dir, info))


class Stage1SourceReuseTest(unittest.TestCase):
    """Fix 2: stage1_source reuses another experiment's stage1 checkpoint."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        self.src_run = self.repo / "artifacts" / "001" / "run"
        self.src_ckpt = make_ckpt(
            self.src_run, "hvq-epoch=5-val_loss=0.4592.ckpt"
        )
        (self.src_run / ".stage1.done").write_text(
            f"best={self.src_ckpt.name}\nval_loss=0.4592\n"
        )
        self.run_dir = self.repo / "artifacts" / "010" / "run"
        self.run_dir.mkdir(parents=True)

    def test_reuse_skips_training_and_returns_source_ckpt(self):
        exp = {"id": "010", "stage1_source": "001"}
        ckpt, prov = runner.stage1(self.repo, self.run_dir, exp)

        self.assertEqual(ckpt, self.src_ckpt.resolve())
        self.assertEqual(prov["mode"], "reused")
        self.assertEqual(prov["source_id"], "001")

        # Own marker records the reuse precisely, so a later resume resolves
        # the same source checkpoint without rescanning.
        info = runner.read_stage1_marker(self.run_dir / ".stage1.done")
        self.assertEqual(info["reused"], "true")
        self.assertEqual(info["source"], "001")
        ckpt2, prov2 = runner.stage1(self.repo, self.run_dir, exp)
        self.assertEqual(ckpt2, self.src_ckpt.resolve())
        self.assertEqual(prov2["mode"], "reused")

    def test_missing_source_marker_raises_instead_of_retraining(self):
        (self.src_run / ".stage1.done").unlink()
        with self.assertRaises(runner.StageError):
            runner.stage1(self.repo, self.run_dir, {"id": "010", "stage1_source": "001"})
        # No marker written, no training attempted.
        self.assertFalse((self.run_dir / ".stage1.done").exists())

    def test_corrupt_source_ckpt_raises_instead_of_retraining(self):
        self.src_ckpt.write_bytes(b"")
        with self.assertRaises(runner.StageError):
            runner.stage1(self.repo, self.run_dir, {"id": "010", "stage1_source": "001"})
        self.assertFalse((self.run_dir / ".stage1.done").exists())

    def test_self_source_is_default_when_field_absent(self):
        # Marker from a previous self-trained run: default (no field) resumes it.
        own = make_ckpt(self.run_dir, "infu_y-epoch=1-val_loss=0.8000.ckpt")
        (self.run_dir / ".stage1.done").write_text(f"best={own.name}\n")
        ckpt, prov = runner.stage1(self.repo, self.run_dir, {"id": "010"})
        self.assertEqual(ckpt, own.resolve())
        self.assertEqual(prov["mode"], "self")


class SelectExperimentsTest(unittest.TestCase):
    """Fix 3: default selection consumes pending + running."""

    ENTRIES = [
        {"id": "001", "status": "done"},
        {"id": "002", "status": "pending"},
        {"id": "003", "status": "running"},
        {"id": "004", "status": "failed"},
    ]

    def test_default_selects_pending_and_running(self):
        selected = runner.select_experiments(self.ENTRIES)
        self.assertEqual([e["id"] for e in selected], ["002", "003"])

    def test_done_is_never_selected(self):
        selected = runner.select_experiments(self.ENTRIES, only=["001"])
        self.assertEqual(selected, [])

    def test_failed_only_via_explicit_only(self):
        self.assertNotIn("004", [e["id"] for e in runner.select_experiments(self.ENTRIES)])
        selected = runner.select_experiments(self.ENTRIES, only=["004"])
        self.assertEqual([e["id"] for e in selected], ["004"])


class Stage1MarkerSourceConsistencyTest(unittest.TestCase):
    """Fix 1: marker provenance must match the queue's current stage1_source."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        self.ck001 = self._make_source("001")
        self.ck003 = self._make_source("003")
        self.run_dir = self.repo / "artifacts" / "010" / "run"
        self.run_dir.mkdir(parents=True)

    def _make_source(self, sid):
        src_run = self.repo / "artifacts" / sid / "run"
        ck = make_ckpt(src_run, f"hvq_{sid}-epoch=5-val_loss=0.4592.ckpt")
        (src_run / ".stage1.done").write_text(f"best={ck.name}\nval_loss=0.4592\n")
        return ck.resolve()

    def _mark_downstream_done(self):
        (self.run_dir / ".stage2.done").write_text("res=x\n")
        (self.run_dir / ".backtest.done").write_text("metric=y\n")

    def test_source_changed_001_to_003_rejects_old_marker(self):
        (self.run_dir / ".stage1.done").write_text(
            f"best={self.ck001}\nsource=001\nreused=true\n"
        )
        self._mark_downstream_done()

        ckpt, prov = runner.stage1(
            self.repo, self.run_dir, {"id": "010", "stage1_source": "003"}
        )

        self.assertEqual(ckpt, self.ck003)
        self.assertEqual(prov["source_id"], "003")
        # Downstream markers invalidated; own marker re-recorded for 003.
        self.assertFalse((self.run_dir / ".stage2.done").exists())
        self.assertFalse((self.run_dir / ".backtest.done").exists())
        info = runner.read_stage1_marker(self.run_dir / ".stage1.done")
        self.assertEqual(info["source"], "003")
        self.assertEqual(info["reused"], "true")

    def test_self_marker_rejected_when_source_set_to_001(self):
        own = make_ckpt(self.run_dir, "infu_y-epoch=1-val_loss=0.8000.ckpt")
        (self.run_dir / ".stage1.done").write_text(f"best={own.name}\nval_loss=0.8\n")
        self._mark_downstream_done()

        ckpt, prov = runner.stage1(
            self.repo, self.run_dir, {"id": "010", "stage1_source": "001"}
        )

        self.assertEqual(ckpt, self.ck001)
        self.assertEqual(prov["mode"], "reused")
        self.assertFalse((self.run_dir / ".stage2.done").exists())
        self.assertFalse((self.run_dir / ".backtest.done").exists())

    def test_matching_source_resumes_normally(self):
        (self.run_dir / ".stage1.done").write_text(
            f"best={self.ck001}\nsource=001\nreused=true\n"
        )
        self._mark_downstream_done()

        ckpt, prov = runner.stage1(
            self.repo, self.run_dir, {"id": "010", "stage1_source": "001"}
        )

        self.assertEqual(ckpt, self.ck001)
        self.assertEqual(prov["mode"], "reused")
        # Downstream markers untouched on a consistent resume.
        self.assertTrue((self.run_dir / ".stage2.done").exists())
        self.assertTrue((self.run_dir / ".backtest.done").exists())


class Stage2MarkerConsistencyTest(unittest.TestCase):
    """Fix 2: .stage2.done is only valid for the stage1 checkpoint it used."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        self.run_dir = self.repo / "artifacts" / "010" / "run"
        self.ckpt_a = make_ckpt(self.run_dir, "a-epoch=1-val_loss=0.5000.ckpt").resolve()
        self.ckpt_b = make_ckpt(self.run_dir, "b-epoch=1-val_loss=0.4000.ckpt").resolve()
        self.res = self.run_dir / "res" / "fake_run"
        self.res.mkdir(parents=True)
        (self.res / "0_best.pkl").write_bytes(b"pred")
        (self.res / "0_metric.csv").write_text(",values\nIC,0.01\n")

    def test_same_ckpt_skips_stage2(self):
        (self.run_dir / ".stage2.done").write_text(
            f"res=fake_run\nckpt={self.ckpt_a}\nseed=0\n"
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("stage2 must not run")

        orig = runner.run_with_retries
        runner.run_with_retries = forbidden
        try:
            res = runner.stage2(self.repo, self.run_dir, self.ckpt_a)
        finally:
            runner.run_with_retries = orig
        self.assertEqual(res, self.res)

    def test_relative_and_absolute_refs_compare_equal(self):
        # Marker stores the bare filename; current ckpt arrives as an
        # absolute path — they must be recognized as the same checkpoint.
        (self.run_dir / ".stage2.done").write_text(
            f"res=fake_run\nckpt={self.ckpt_a.name}\nseed=0\n"
        )
        res = runner.stage2(self.repo, self.run_dir, self.ckpt_a)
        self.assertEqual(res, self.res)

    def test_changed_ckpt_reruns_stage2_and_invalidates_backtest(self):
        (self.run_dir / ".stage2.done").write_text(
            f"res=fake_run\nckpt={self.ckpt_a}\nseed=0\n"
        )
        (self.run_dir / ".backtest.done").write_text("metric=x\n")

        calls = []
        orig = runner.run_with_retries

        def fake_run(cmd, log_path, cwd, attempts):
            calls.append(cmd)
            # Simulated stage2 leaves fresh prediction/metric outputs.

        runner.run_with_retries = fake_run
        try:
            res = runner.stage2(self.repo, self.run_dir, self.ckpt_b)
        finally:
            runner.run_with_retries = orig

        self.assertEqual(len(calls), 1)
        self.assertEqual(res, self.res)
        self.assertFalse((self.run_dir / ".backtest.done").exists())
        info = runner.read_stage1_marker(self.run_dir / ".stage2.done")
        self.assertEqual(info["ckpt"], str(self.ckpt_b))


class ColdResumeTest(unittest.TestCase):
    """Fix 3 (cold resume): the runner must read the canonical queue on main,
    even when a previous interrupted run left an experiment branch checked out.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self._git("init")
        # git 2.25 has no `init -b`; point HEAD at main before the first commit.
        self._git("symbolic-ref", "HEAD", "refs/heads/main")
        (self.repo / "experiments").mkdir()
        # Old queue without 003; the experiment branch is created from here.
        (self.repo / "experiments" / "queue.yaml").write_text(
            'experiments:\n  - id: "001"\n    status: done\n'
        )
        self._git("add", ".")
        self._git("commit", "-m", "old queue")
        self._git("branch", "exp/003-old")
        # main advances: 003 is now running (interrupted mid-execution).
        (self.repo / "experiments" / "queue.yaml").write_text(
            'experiments:\n'
            '  - id: "001"\n    status: done\n'
            '  - id: "003"\n    status: running\n'
        )
        self._git("add", ".")
        self._git("commit", "-m", "003 running")
        # Simulate the interrupted state: stuck on the stale exp branch.
        self._git("checkout", "exp/003-old")

    def _git(self, *args):
        subprocess.run(
            ["git", "-C", str(self.repo),
             "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
            check=True, capture_output=True,
        )

    def _current_branch(self):
        return runner.git(self.repo, "branch", "--show-current").stdout.strip()

    def test_restart_from_exp_branch_reads_canonical_queue(self):
        # The stale branch queue does not contain 003 at all.
        stale = runner.load_queue(self.repo)
        self.assertNotIn("003", [e["id"] for e in stale["experiments"]])

        runner.ensure_control_branch(self.repo)

        self.assertEqual(self._current_branch(), "main")
        queue = runner.load_queue(self.repo)
        selected = runner.select_experiments(queue["experiments"])
        self.assertEqual([e["id"] for e in selected], ["003"])

    def test_start_from_main_is_unaffected(self):
        self._git("checkout", "main")
        runner.ensure_control_branch(self.repo)
        self.assertEqual(self._current_branch(), "main")
        queue = runner.load_queue(self.repo)
        selected = runner.select_experiments(queue["experiments"])
        self.assertEqual([e["id"] for e in selected], ["003"])

    def test_dirty_tracked_changes_are_not_overwritten(self):
        (self.repo / "experiments" / "queue.yaml").write_text("dirty edit\n")
        with self.assertRaises(runner.StageError):
            runner.ensure_control_branch(self.repo)
        # Still on the experiment branch; the dirty file is untouched.
        self.assertEqual(self._current_branch(), "exp/003-old")
        self.assertEqual(
            (self.repo / "experiments" / "queue.yaml").read_text(), "dirty edit\n"
        )


class CommitPinningTest(unittest.TestCase):
    """A queued experiment executes its pinned commit, never the branch HEAD.

    Regression: advancing the experiment branch after queueing must not
    silently change what the queued experiment runs.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self._git("init")
        self._git("symbolic-ref", "HEAD", "refs/heads/main")
        (self.repo / "code.py").write_text("VERSION = 1\n")
        self._git("add", ".")
        self._git("commit", "-m", "base")
        self._git("branch", "exp/004-x")
        # The experiment's final code at queue time (pinned).
        self._git("checkout", "exp/004-x")
        (self.repo / "code.py").write_text("VERSION = 2\n")
        self._git("add", ".")
        self._git("commit", "-m", "experiment code")
        self.pinned = self._git("rev-parse", "HEAD").stdout.strip()
        # The branch advances AFTER queueing (must not affect the experiment).
        (self.repo / "code.py").write_text("VERSION = 3\n")
        self._git("add", ".")
        self._git("commit", "-m", "post-queue change")
        self.later = self._git("rev-parse", "HEAD").stdout.strip()
        self._git("checkout", "main")

    def _git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.repo),
             "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
            check=True, capture_output=True, text=True,
        )

    def test_executes_pinned_commit_not_branch_head(self):
        exp = {"id": "004", "branch": "exp/004-x", "commit": self.pinned}
        commit = runner.resolve_pinned_commit(self.repo, exp)
        self.assertEqual(commit, self.pinned)

        runner.checkout_experiment(self.repo, "exp/004-x", commit)

        head = runner.git(self.repo, "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(head, self.pinned)
        self.assertEqual((self.repo / "code.py").read_text(), "VERSION = 2\n")
        # The branch ref itself is untouched and still points at the later commit.
        branch_head = runner.git(self.repo, "rev-parse", "exp/004-x").stdout.strip()
        self.assertEqual(branch_head, self.later)

    def test_short_or_full_pin_resolves_to_full_sha(self):
        exp = {"id": "004", "commit": self.pinned[:7]}
        self.assertEqual(runner.resolve_pinned_commit(self.repo, exp), self.pinned)

    def test_missing_commit_is_rejected(self):
        with self.assertRaises(runner.StageError):
            runner.resolve_pinned_commit(self.repo, {"id": "004"})

    def test_unknown_commit_is_rejected(self):
        with self.assertRaises(runner.StageError):
            runner.resolve_pinned_commit(
                self.repo, {"id": "004", "commit": "0" * 40}
            )


class MarkerCommitProvenanceTest(unittest.TestCase):
    """Markers recorded under a different pinned commit must not be skipped."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        self.src_run = self.repo / "artifacts" / "001" / "run"
        self.src_ckpt = make_ckpt(self.src_run, "hvq-epoch=5-val_loss=0.4592.ckpt")
        (self.src_run / ".stage1.done").write_text(
            f"best={self.src_ckpt.name}\nval_loss=0.4592\ncommit=COMMIT_A\n"
        )
        self.run_dir = self.repo / "artifacts" / "010" / "run"
        self.run_dir.mkdir(parents=True)
        self.exp = {
            "id": "010", "commit": "COMMIT_B", "stage1_source": "001",
        }

    def test_stage1_marker_with_old_commit_is_invalidated(self):
        (self.run_dir / ".stage1.done").write_text(
            f"best={self.src_ckpt.resolve()}\nsource=001\nreused=true\ncommit=COMMIT_A\n"
        )
        (self.run_dir / ".stage2.done").write_text("res=x\n")
        (self.run_dir / ".backtest.done").write_text("metric=y\n")

        ckpt, prov = runner.stage1(self.repo, self.run_dir, self.exp)

        self.assertEqual(ckpt, self.src_ckpt.resolve())
        self.assertEqual(prov["mode"], "reused")
        # Downstream markers invalidated; own marker re-recorded with the
        # current pinned commit.
        self.assertFalse((self.run_dir / ".stage2.done").exists())
        self.assertFalse((self.run_dir / ".backtest.done").exists())
        info = runner.read_stage1_marker(self.run_dir / ".stage1.done")
        self.assertEqual(info["commit"], "COMMIT_B")

    def test_stage1_marker_with_matching_commit_resumes(self):
        (self.run_dir / ".stage1.done").write_text(
            f"best={self.src_ckpt.resolve()}\nsource=001\nreused=true\ncommit=COMMIT_B\n"
        )
        ckpt, prov = runner.stage1(self.repo, self.run_dir, self.exp)
        self.assertEqual(ckpt, self.src_ckpt.resolve())
        self.assertEqual(prov["mode"], "reused")

    def _make_stage2_outputs(self):
        res = self.run_dir / "res" / "fake_run"
        res.mkdir(parents=True, exist_ok=True)
        (res / "0_best.pkl").write_bytes(b"pred")
        (res / "0_metric.csv").write_text(",values\nIC,0.01\n")
        return res

    def test_stage2_marker_with_old_commit_reruns(self):
        res = self._make_stage2_outputs()
        ckpt = self.src_ckpt.resolve()
        (self.run_dir / ".stage2.done").write_text(
            f"res=fake_run\nckpt={ckpt}\nseed=0\ncommit=COMMIT_A\n"
        )
        (self.run_dir / ".backtest.done").write_text("metric=y\n")

        calls = []
        orig = runner.run_with_retries
        runner.run_with_retries = lambda cmd, log_path, cwd, attempts: calls.append(cmd)
        try:
            out = runner.stage2(self.repo, self.run_dir, ckpt, "COMMIT_B")
        finally:
            runner.run_with_retries = orig

        self.assertEqual(len(calls), 1)
        self.assertEqual(out, res)
        self.assertFalse((self.run_dir / ".backtest.done").exists())
        info = runner.read_stage1_marker(self.run_dir / ".stage2.done")
        self.assertEqual(info["commit"], "COMMIT_B")
        self.assertEqual(info["seed"], "0")

    def test_stage2_marker_with_matching_commit_skips(self):
        res = self._make_stage2_outputs()
        ckpt = self.src_ckpt.resolve()
        (self.run_dir / ".stage2.done").write_text(
            f"res=fake_run\nckpt={ckpt}\nseed=0\ncommit=COMMIT_B\n"
        )
        orig = runner.run_with_retries
        runner.run_with_retries = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("stage2 must not run")
        )
        try:
            out = runner.stage2(self.repo, self.run_dir, ckpt, "COMMIT_B")
        finally:
            runner.run_with_retries = orig
        self.assertEqual(out, res)


class BacktestProtocolProvenanceTest(unittest.TestCase):
    """A .backtest.done marker is valid only for the protocol that produced it."""

    UNIVERSE = "csi300"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        self.run_dir = self.repo / "artifacts" / "010" / "run"
        self.res = self.run_dir / "res" / "fake_run"
        bt = self.res / "backtest" / "seed0_top30_drop5"
        bt.mkdir(parents=True)
        (bt / "portfolio_metric.csv").write_text(
            ",project_portfolio,project_benchmark\nMDD,-0.1,-0.2\n"
        )
        self.metric = bt / "portfolio_metric.csv"
        self.sig = runner.backtest_protocol_sig(self.UNIVERSE)

    def test_matching_protocol_and_commit_skips(self):
        (self.run_dir / ".backtest.done").write_text(
            f"metric=res/fake_run/backtest/seed0_top30_drop5/portfolio_metric.csv\n"
            f"protocol={self.sig}\ncommit=COMMIT_B\n"
        )
        orig = runner.run_with_retries
        runner.run_with_retries = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("backtest must not run")
        )
        try:
            out = runner.backtest(
                self.repo, self.run_dir, self.res, self.UNIVERSE, "COMMIT_B"
            )
        finally:
            runner.run_with_retries = orig
        self.assertEqual(out, self.metric)

    def test_changed_protocol_reruns_backtest(self):
        (self.run_dir / ".backtest.done").write_text(
            "metric=res/fake_run/backtest/seed0_top30_drop5/portfolio_metric.csv\n"
            "protocol=universe=csi300;seed=0;args=--topk 50\ncommit=COMMIT_B\n"
        )
        calls = []
        orig = runner.run_with_retries
        runner.run_with_retries = lambda cmd, log_path, cwd, attempts: calls.append(cmd)
        try:
            out = runner.backtest(
                self.repo, self.run_dir, self.res, self.UNIVERSE, "COMMIT_B"
            )
        finally:
            runner.run_with_retries = orig
        self.assertEqual(len(calls), 1)
        self.assertEqual(out, self.metric)
        info = runner.read_stage1_marker(self.run_dir / ".backtest.done")
        self.assertEqual(info["protocol"], self.sig)
        self.assertEqual(info["commit"], "COMMIT_B")

    def test_changed_commit_reruns_backtest(self):
        (self.run_dir / ".backtest.done").write_text(
            "metric=res/fake_run/backtest/seed0_top30_drop5/portfolio_metric.csv\n"
            f"protocol={self.sig}\ncommit=COMMIT_A\n"
        )
        calls = []
        orig = runner.run_with_retries
        runner.run_with_retries = lambda cmd, log_path, cwd, attempts: calls.append(cmd)
        try:
            runner.backtest(self.repo, self.run_dir, self.res, self.UNIVERSE, "COMMIT_B")
        finally:
            runner.run_with_retries = orig
        self.assertEqual(len(calls), 1)


class ExternalStage1Test(unittest.TestCase):
    """stage1_source: external reuses an exact baseline/external checkpoint."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        # The external checkpoint deliberately lives OUTSIDE repo/artifacts
        # (e.g. the PRISM-VQ baseline checkpoint in a sibling repo).
        self.ext = self.tmp / "external" / "baseline-epoch=7-val_loss=0.5712.ckpt"
        self.ext.parent.mkdir(parents=True)
        self.ext.write_bytes(b"x" * 16)
        self.run_dir = self.repo / "artifacts" / "011" / "run"
        self.run_dir.mkdir(parents=True)
        self.exp = {
            "id": "011", "commit": "COMMIT_B",
            "stage1_source": "external", "stage1_ckpt": str(self.ext),
        }

    def test_external_reuse_and_resume(self):
        ckpt, prov = runner.stage1(self.repo, self.run_dir, self.exp)

        self.assertEqual(ckpt, self.ext.resolve())
        self.assertEqual(prov["mode"], "reused")
        self.assertEqual(prov["source_id"], "external")

        info = runner.read_stage1_marker(self.run_dir / ".stage1.done")
        self.assertEqual(info["reused"], "true")
        self.assertEqual(info["source"], "external")
        self.assertEqual(info["commit"], "COMMIT_B")
        self.assertEqual(info["best"], str(self.ext.resolve()))

        # Resume resolves the same external checkpoint without any training.
        ckpt2, prov2 = runner.stage1(self.repo, self.run_dir, self.exp)
        self.assertEqual(ckpt2, self.ext.resolve())
        self.assertEqual(prov2["mode"], "reused")

    def test_missing_external_ckpt_raises_instead_of_retraining(self):
        exp = dict(self.exp, stage1_ckpt=str(self.tmp / "gone.ckpt"))
        with self.assertRaises(runner.StageError):
            runner.stage1(self.repo, self.run_dir, exp)
        self.assertFalse((self.run_dir / ".stage1.done").exists())

    def test_missing_stage1_ckpt_field_raises(self):
        exp = {"id": "011", "stage1_source": "external"}
        with self.assertRaises(runner.StageError):
            runner.stage1(self.repo, self.run_dir, exp)

    def test_marker_pointing_elsewhere_is_invalid_and_reresolved(self):
        # A stale/tampered marker records source=external but points at a
        # different (artifacts-internal) checkpoint: it must not be trusted.
        other = make_ckpt(self.run_dir, "other-epoch=1-val_loss=0.5000.ckpt")
        (self.run_dir / ".stage1.done").write_text(
            f"best={other.resolve()}\nsource=external\nreused=true\ncommit=COMMIT_B\n"
        )
        (self.run_dir / ".stage2.done").write_text("res=x\n")

        ckpt, prov = runner.stage1(self.repo, self.run_dir, self.exp)

        self.assertEqual(ckpt, self.ext.resolve())
        self.assertFalse((self.run_dir / ".stage2.done").exists())
        info = runner.read_stage1_marker(self.run_dir / ".stage1.done")
        self.assertEqual(info["best"], str(self.ext.resolve()))

    def test_marker_pointing_to_deleted_external_file_reresolves_then_raises(self):
        (self.run_dir / ".stage1.done").write_text(
            f"best={self.ext.resolve()}\nsource=external\nreused=true\ncommit=COMMIT_B\n"
        )
        self.ext.unlink()
        with self.assertRaises(runner.StageError):
            runner.stage1(self.repo, self.run_dir, self.exp)

    def test_relative_stage1_ckpt_resolves_against_repo(self):
        rel_ckpt = self.repo / "artifacts" / "baseline.ckpt"
        rel_ckpt.write_bytes(b"x" * 16)
        exp = dict(self.exp, stage1_ckpt="artifacts/baseline.ckpt")
        ckpt, prov = runner.stage1(self.repo, self.run_dir, exp)
        self.assertEqual(ckpt, rel_ckpt.resolve())
        self.assertEqual(prov["source_id"], "external")


class Stage1SourceCommitValidationTest(unittest.TestCase):
    """experiment→experiment reuse: the source stage1 marker's commit must
    equal the source experiment's canonical queue pinned commit.

    Queue metadata is passed in by the caller (run_experiment uses the
    canonical queue read on main); a missing/mismatching commit or a
    missing source entry must loud-fail — no reuse, no silent retrain.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = make_repo(self.tmp)
        self.src_run = self.repo / "artifacts" / "001" / "run"
        self.src_ckpt = make_ckpt(self.src_run, "hvq-epoch=5-val_loss=0.4592.ckpt")
        self.run_dir = self.repo / "artifacts" / "010" / "run"
        self.run_dir.mkdir(parents=True)
        self.exp = {"id": "010", "stage1_source": "001"}
        self.queue_by_id = {"001": {"id": "001", "commit": "SRC_FINAL_SHA"}}
        # Any training attempt is a test failure.
        orig = runner.run_with_retries
        runner.run_with_retries = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("stage1 must never retrain for a reuse experiment")
        )
        self.addCleanup(setattr, runner, "run_with_retries", orig)

    def _write_src_marker(self, text):
        (self.src_run / ".stage1.done").write_text(text)

    def test_matching_source_commit_reuses(self):
        self._write_src_marker(
            f"best={self.src_ckpt.name}\nval_loss=0.4592\ncommit=SRC_FINAL_SHA\n"
        )
        ckpt, prov = runner.stage1(self.repo, self.run_dir, self.exp, self.queue_by_id)
        self.assertEqual(ckpt, self.src_ckpt.resolve())
        self.assertEqual(prov["mode"], "reused")
        self.assertEqual(prov["source_id"], "001")

    def test_source_marker_missing_commit_loud_fails(self):
        self._write_src_marker(f"best={self.src_ckpt.name}\nval_loss=0.4592\n")
        with self.assertRaises(runner.StageError):
            runner.stage1(self.repo, self.run_dir, self.exp, self.queue_by_id)
        self.assertFalse((self.run_dir / ".stage1.done").exists())

    def test_source_marker_commit_mismatch_loud_fails(self):
        self._write_src_marker(
            f"best={self.src_ckpt.name}\nval_loss=0.4592\ncommit=OTHER_SHA\n"
        )
        with self.assertRaises(runner.StageError):
            runner.stage1(self.repo, self.run_dir, self.exp, self.queue_by_id)
        self.assertFalse((self.run_dir / ".stage1.done").exists())

    def test_source_queue_entry_missing_commit_loud_fails(self):
        self._write_src_marker(
            f"best={self.src_ckpt.name}\nval_loss=0.4592\ncommit=SRC_FINAL_SHA\n"
        )
        with self.assertRaises(runner.StageError):
            runner.stage1(
                self.repo, self.run_dir, self.exp, {"001": {"id": "001"}}
            )
        self.assertFalse((self.run_dir / ".stage1.done").exists())

    def test_source_missing_from_queue_metadata_loud_fails(self):
        self._write_src_marker(
            f"best={self.src_ckpt.name}\nval_loss=0.4592\ncommit=SRC_FINAL_SHA\n"
        )
        with self.assertRaises(runner.StageError):
            runner.stage1(self.repo, self.run_dir, self.exp, {})
        self.assertFalse((self.run_dir / ".stage1.done").exists())


if __name__ == "__main__":
    unittest.main()
