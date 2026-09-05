"""Unit tests for the Phase 2 executor (experiments/runner.py).

All tests use temporary fake artifacts; no real experiment artifacts,
training, or git operations are involved.
"""

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
