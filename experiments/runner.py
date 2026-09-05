#!/usr/bin/env python3
"""Phase 2 executor: consume pending experiments from experiments/queue.yaml.

Fixed pipeline per experiment (see experiments/templates/RULES.md):

    pending -> checkout exp branch -> stage1 -> best stage1 checkpoint
            -> stage2 (seed 0) -> prediction/metrics -> qlib backtest
            -> collect metrics -> back to main -> update record + queue

All formal artifacts live under ``artifacts/<ID>/run/`` via the
``artifact_root`` execution-layer override. The runner is deterministic and
resumable: a stage is re-executed only when its completion marker or its
expected artifacts are missing, or when the marker's recorded provenance
(pinned commit, stage1 checkpoint/source, backtest protocol) no longer
matches the queue entry. Stage 1 resume reads the exact checkpoint recorded
in ``.stage1.done`` instead of rescanning the shared checkpoint directory.

A queued experiment is IMMUTABLE: the queue entry's ``commit`` field pins
the exact code version, and Phase 2 checks out that pinned commit (detached
HEAD), never the experiment branch's latest HEAD. If the experiment logic
changes, a new experiment id must be created instead of mutating the entry.

By default the runner consumes ``pending`` and ``running`` entries (a
``running`` entry means a previous execution was interrupted; it resumes
from the existing markers/artifacts). ``done`` is always skipped; ``failed``
is retried only via an explicit ``--only``.

A queue entry declares its stage1 provenance via ``stage1_source``:

- ``self`` (default): train its own stage1;
- ``"<id>"``: reuse that experiment's exact formal stage1 best checkpoint;
- ``external`` together with ``stage1_ckpt: <path>``: reuse an exact
  baseline/external checkpoint (relative paths resolve against the repo
  root).

The source is declared by the experiment author in Phase 1; the runner never
infers it.

The runner only invokes the existing entry points (stage1.py, stage2.py,
backtest_qlib.py) and never touches experiment logic, hyperparameters, data
splits or the backtest protocol.

Usage (run a copy of this file from outside the git tree, because the runner
checks out experiment branches where this file does not exist):

    python /path/to/copy/runner.py --repo /path/to/HVQ-Stock [--only 002 003]
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

QUEUE_PATH = Path("experiments") / "queue.yaml"
RECORDS_DIR = Path("experiments") / "records"

STAGE1_MAX_ATTEMPTS = 2
STAGE2_MAX_ATTEMPTS = 2
BACKTEST_MAX_ATTEMPTS = 2

# Fixed protocols (see experiment records; must not change per experiment).
STAGE2_SEED = 0
BACKTEST_ARGS = [
    "--start_time", "2023-01-01",
    "--end_time", "2025-12-31",
    "--topk", "30",
    "--drop", "5",
    "--open_cost", "0.0005",
    "--close_cost", "0.0015",
    "--min_cost", "0",
]

BEST_CKPT_RE = re.compile(r"-epoch=(\d+)-val_loss=([0-9.]+?)(?:-v\d+)?\.ckpt$")


class StageError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check, capture_output=True, text=True,
    )


def load_queue(repo: Path) -> dict:
    with open(repo / QUEUE_PATH) as f:
        return yaml.safe_load(f)


def set_queue_status(repo: Path, exp_id: str, status: str) -> None:
    """Rewrite the status line of one queue entry, preserving comments."""
    path = repo / QUEUE_PATH
    text = path.read_text()
    block_re = re.compile(
        r'(- id: "%s"\n(?:    .*\n)*?)    status: \w+' % re.escape(exp_id)
    )
    new_text, n = block_re.subn(r"\g<1>    status: " + status, text)
    if n != 1:
        raise StageError(f"queue entry for id {exp_id} not found or ambiguous")
    path.write_text(new_text)


def commit_main(repo: Path, message: str) -> None:
    git(repo, "add", str(QUEUE_PATH), str(RECORDS_DIR))
    diff = git(repo, "diff", "--cached", "--quiet", check=False)
    if diff.returncode != 0:
        git(repo, "commit", "-m", message)


def ensure_clean(repo: Path) -> None:
    # Only tracked modifications block a checkout; untracked paths such as the
    # dataset/data symlink or ignored outputs/ survive branch switches fine.
    status = git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    if status:
        raise StageError(f"git worktree is dirty, refusing to checkout:\n{status}")


def checkout(repo: Path, branch: str) -> None:
    ensure_clean(repo)
    git(repo, "checkout", branch)
    log(f"checked out {branch}")


def resolve_pinned_commit(repo: Path, exp: dict) -> str:
    """Return the full sha of the experiment's pinned commit.

    A queued experiment is immutable: the queue entry's ``commit`` field is
    the exact code version Phase 2 must execute. The field is required.
    """
    commit = str(exp.get("commit") or "").strip()
    if not commit:
        raise StageError(
            f"experiment {exp.get('id')}: queue entry is missing the pinned "
            f"'commit' field; a queued experiment must pin its exact code version"
        )
    res = git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}", check=False)
    if res.returncode != 0:
        raise StageError(
            f"experiment {exp.get('id')}: pinned commit {commit!r} not found in the repo"
        )
    return res.stdout.strip()


def checkout_experiment(repo: Path, branch: str, commit: str) -> None:
    """Check out the experiment's pinned commit (detached HEAD).

    The branch HEAD is intentionally NOT used: even if the branch advances
    after the experiment entered the queue, the queued experiment still
    executes exactly the pinned commit.
    """
    head = git(repo, "rev-parse", branch, check=False)
    if head.returncode == 0 and head.stdout.strip() != commit:
        log(
            f"WARNING: branch {branch} HEAD ({head.stdout.strip()[:12]}) differs from "
            f"the pinned commit ({commit[:12]}); executing the pinned commit"
        )
    checkout(repo, commit)
    actual = git(repo, "rev-parse", "HEAD").stdout.strip()
    if actual != commit:
        raise StageError(
            f"checkout of pinned commit failed: HEAD is {actual}, expected {commit}"
        )


def ensure_control_branch(repo: Path) -> None:
    """The queue/records on main are the canonical control state.

    A previous run may have been interrupted (kill, Ctrl+C, crash) while an
    experiment branch was checked out; that branch's queue.yaml can be a
    stale copy. Always return to main before reading the queue. The dirty
    worktree check in checkout() still applies, so tracked user changes are
    never silently overwritten.
    """
    current = git(repo, "branch", "--show-current").stdout.strip()
    if current != "main":
        log(f"repo is on {current or 'detached HEAD'}; returning to main before reading the queue")
        checkout(repo, "main")


def run_dir_for(repo: Path, exp_id: str) -> Path:
    return repo / "artifacts" / exp_id / "run"


def find_best_stage1_ckpt(run_dir: Path, since: float = None):
    """Return (val_loss, path) of the best stage1 checkpoint, or None.

    ``since`` restricts the scan to checkpoints modified at/after that
    timestamp; a fresh scan right after a stage1 run then cannot pick up
    stale stage2 checkpoints sharing the same directory.
    """
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    cands = []
    for p in ckpt_dir.glob("*.ckpt"):
        m = BEST_CKPT_RE.search(p.name)
        if not m:
            continue
        st = p.stat()
        if st.st_size == 0:
            continue
        if since is not None and st.st_mtime < since:
            continue
        cands.append((float(m.group(2)), p))
    if not cands:
        return None
    return min(cands, key=lambda t: t[0])


def read_stage1_marker(marker_path: Path) -> dict:
    info = {}
    for line in marker_path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
    return info


def resolve_marker_ckpt(repo: Path, run_dir: Path, info: dict, allow: Path = None):
    """Resolve the checkpoint recorded in a .stage1.done marker and validate it.

    A bare filename resolves against ``run_dir/checkpoints`` (self-trained);
    an absolute path is used as-is (reused from another experiment's
    artifacts). Either way the file must exist, be non-empty, and live under
    ``repo/artifacts`` — unless it is exactly ``allow``, the queue-declared
    external checkpoint (``stage1_source: external``), which may live
    outside the repo.
    """
    best = info.get("best")
    if not best:
        return None
    p = _normalize_ckpt_ref(run_dir, best)
    if allow is not None and p == allow.resolve():
        pass
    elif not p.is_relative_to((repo / "artifacts").resolve()):
        return None
    if not p.is_file() or p.stat().st_size == 0:
        return None
    return p


def _normalize_ckpt_ref(run_dir: Path, ref: str) -> Path:
    """Normalize a checkpoint reference (bare filename or absolute path) to a
    resolved absolute path, so equivalent references compare equal."""
    p = Path(ref)
    if not p.is_absolute():
        p = run_dir / "checkpoints" / p.name
    return p.resolve()


def marker_commit_matches(info: dict, commit: str) -> bool:
    """A marker is valid only for the pinned commit that produced it.

    Markers written before commit pinning (no ``commit=`` line) do not match
    a pinned experiment and are correctly invalidated. An empty ``commit``
    (callers without pinning, e.g. legacy/unit-test paths) skips the check.
    """
    if not commit:
        return True
    return info.get("commit") == commit


def marker_matches_source(info: dict, source: str) -> bool:
    """Check .stage1.done provenance against the queue's current stage1_source.

    ``self`` accepts only self-trained markers; ``external`` accepts only
    markers recorded as reused from an external checkpoint; ``<ID>`` accepts
    only markers recorded as reused from exactly that source experiment.
    """
    if source == "self":
        return info.get("reused") != "true"
    return info.get("reused") == "true" and info.get("source") == source


def external_stage1_ckpt(repo: Path, exp: dict) -> Path:
    """Resolve and validate the exact external/baseline stage1 checkpoint
    declared by ``stage1_source: external`` + ``stage1_ckpt: <path>``.

    Relative paths resolve against the repo root. The file must exist and be
    non-empty; the runner never silently retrains stage1 for a stage2-only
    experiment.
    """
    ref = str(exp.get("stage1_ckpt") or "").strip()
    if not ref:
        raise StageError(
            f"experiment {exp['id']}: stage1_source=external requires a "
            f"'stage1_ckpt' path in the queue entry"
        )
    p = Path(ref)
    if not p.is_absolute():
        p = repo / p
    p = p.resolve()
    if not p.is_file() or p.stat().st_size == 0:
        raise StageError(
            f"experiment {exp['id']}: external stage1 checkpoint missing or "
            f"empty: {p}"
        )
    return p


def backtest_protocol_sig(universe: str) -> str:
    """Signature of the fixed backtest protocol a .backtest.done marker is
    valid for. Any change to the universe or BACKTEST_ARGS invalidates old
    markers (string comparison; no hashing)."""
    return f"universe={universe};seed={STAGE2_SEED};args={' '.join(BACKTEST_ARGS)}"


def find_stage2_outputs(run_dir: Path):
    """Return the res subdir containing 0_best.pkl + 0_metric.csv (unique)."""
    res_dir = run_dir / "res"
    if not res_dir.is_dir():
        return None
    hits = [
        p.parent for p in res_dir.glob("*/0_best.pkl")
        if (p.parent / "0_metric.csv").exists() and p.stat().st_size > 0
    ]
    if len(hits) != 1:
        return None
    return hits[0]


def find_backtest_metric(res_subdir: Path):
    hits = list(res_subdir.glob("backtest/*/portfolio_metric.csv"))
    if len(hits) != 1:
        return None
    return hits[0]


def run_cmd(cmd, log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"RUN: {' '.join(str(c) for c in cmd)}")
    log(f"  log -> {log_path}")
    # No wandb API key is configured on this machine; keep wandb in offline
    # mode (same as the recorded smoke runs) so runs stay local under the
    # artifact checkpoint dir.
    env = dict(os.environ, WANDB_MODE="offline")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise StageError(
            f"command failed with exit code {proc.returncode}; see {log_path}"
        )


def run_with_retries(cmd, log_path: Path, cwd: Path, attempts: int) -> None:
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            run_cmd(cmd, log_path, cwd)
            return
        except StageError as e:
            last_err = e
            log(f"attempt {attempt}/{attempts} failed: {e}")
    raise last_err


def read_branch_universe(repo: Path) -> str:
    with open(repo / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["data"]["universe"]


def read_metric_csv(path: Path) -> dict:
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    # stage2 format: header ",values" then "<name>,<value>" rows
    header = rows[0]
    value_col = header.index("values") if "values" in header else 1
    return {r[0]: float(r[value_col]) for r in rows[1:] if len(r) > value_col and r[1]}


def read_portfolio_metric_csv(path: Path) -> dict:
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    header = rows[0]
    col = header.index("project_portfolio")
    bench_col = header.index("project_benchmark")
    out, bench = {}, {}
    for r in rows[1:]:
        if len(r) > col and r[col]:
            out[r[0]] = float(r[col])
        if len(r) > bench_col and r[bench_col]:
            bench[r[0]] = float(r[bench_col])
    return out, bench


def fmt4(x) -> str:
    return f"{x:.4f}"


def update_record(repo: Path, exp: dict, result_text: str, conclusion: str) -> None:
    path = repo / RECORDS_DIR / f"{exp['id']}-{exp['name']}.md"
    text = path.read_text()
    new_tail = f"## Result\n\n{result_text}\n\n## Conclusion\n\n{conclusion}\n"
    new_text, n = re.subn(
        r"## Result\n.*?\n## Conclusion\n.*\Z", new_tail, text, flags=re.DOTALL
    )
    if n != 1:
        raise StageError(f"record {path} does not match expected Result/Conclusion layout")
    path.write_text(new_text)


def reuse_stage1_ckpt(repo: Path, source_id: str, exp_id: str) -> Path:
    """Load the exact stage1 best checkpoint of a finished source experiment.

    Fails loudly when the source artifacts are missing or corrupt; the runner
    must never silently retrain stage1 for a stage2-only experiment.
    """
    if source_id == str(exp_id):
        raise StageError(
            f"stage1_source={source_id} must differ from the experiment's own id"
        )
    src_run = run_dir_for(repo, source_id)
    src_marker = src_run / ".stage1.done"
    if not src_marker.is_file():
        raise StageError(
            f"stage1_source={source_id}: source stage1 marker not found "
            f"({src_marker}); refusing to retrain stage1 for this experiment"
        )
    info = read_stage1_marker(src_marker)
    ckpt = resolve_marker_ckpt(repo, src_run, info)
    if ckpt is None:
        raise StageError(
            f"stage1_source={source_id}: source checkpoint missing or invalid "
            f"(marker best={info.get('best')!r}); refusing to retrain stage1"
        )
    return ckpt


def stage1(repo: Path, run_dir: Path, exp: dict):
    """Return (ckpt_path, provenance) for the experiment's stage1 checkpoint."""
    marker = run_dir / ".stage1.done"
    source = str(exp.get("stage1_source") or "self")
    commit = str(exp.get("commit") or "")

    # Fail fast on a missing/invalid external checkpoint, even when a marker
    # exists: without the file the marker can never validate anyway.
    external = external_stage1_ckpt(repo, exp) if source == "external" else None

    # Resume path: trust the exact checkpoint recorded in the marker instead
    # of rescanning the shared checkpoint directory. The marker is only valid
    # when its provenance (pinned commit + stage1 source + checkpoint) matches
    # the queue's current declaration.
    if marker.exists():
        info = read_stage1_marker(marker)
        ckpt = resolve_marker_ckpt(repo, run_dir, info, allow=external)
        if (
            ckpt is not None
            and marker_commit_matches(info, commit)
            and marker_matches_source(info, source)
            and (source != "external" or ckpt == external)
        ):
            mode = "reused" if info.get("reused") == "true" else "self"
            log(f"stage1 already complete (best={ckpt.name}, mode={mode}), skipping")
            return ckpt, {"mode": mode, "source_id": info.get("source"), "ckpt": str(ckpt)}
        reasons = []
        if ckpt is None:
            reasons.append("checkpoint missing/invalid")
        if not marker_commit_matches(info, commit):
            reasons.append(f"marker commit {info.get('commit')!r} != pinned {commit!r}")
        if not marker_matches_source(info, source):
            reasons.append(
                f"marker provenance (reused={info.get('reused')}, "
                f"source={info.get('source')}) != stage1_source={source}"
            )
        if source == "external" and ckpt is not None and ckpt != external:
            reasons.append(f"marker ckpt {ckpt} != declared external {external}")
        log(f"stage1 marker invalid ({'; '.join(reasons)}); re-resolving stage1")
        # The stage1 input may change: downstream results are no longer valid.
        marker.unlink(missing_ok=True)
        for downstream in (".stage2.done", ".backtest.done"):
            (run_dir / downstream).unlink(missing_ok=True)

    if source == "external":
        marker.write_text(f"best={external}\nsource=external\nreused=true\ncommit={commit}\n")
        log(f"stage1 reused external checkpoint: {external}")
        return external, {"mode": "reused", "source_id": "external", "ckpt": str(external)}

    if source != "self":
        ckpt = reuse_stage1_ckpt(repo, source, exp["id"])
        marker.write_text(f"best={ckpt}\nsource={source}\nreused=true\ncommit={commit}\n")
        log(f"stage1 reused from experiment {source}: {ckpt}")
        return ckpt, {"mode": "reused", "source_id": source, "ckpt": str(ckpt)}

    # Self-trained stage1. Actually re-running stage1 invalidates downstream
    # stage2/backtest markers, since their inputs change.
    marker.unlink(missing_ok=True)
    for downstream in (".stage2.done", ".backtest.done"):
        (run_dir / downstream).unlink(missing_ok=True)
    artifact_root = run_dir.relative_to(repo)
    start = time.time()
    run_with_retries(
        [sys.executable, "stage1.py", f"artifact_root={artifact_root}"],
        run_dir / "stage1.log", repo, STAGE1_MAX_ATTEMPTS,
    )
    best = find_best_stage1_ckpt(run_dir, since=start)
    if best is None:
        raise StageError("stage1 finished but no best checkpoint found")
    marker.write_text(f"best={best[1].name}\nval_loss={best[0]}\ncommit={commit}\n")
    log(f"stage1 done: best={best[1].name} val_loss={best[0]:.4f}")
    return best[1], {"mode": "self", "source_id": None, "ckpt": str(best[1])}


def stage2(repo: Path, run_dir: Path, ckpt: Path, commit: str = "") -> Path:
    marker = run_dir / ".stage2.done"
    res_subdir = find_stage2_outputs(run_dir)
    if marker.exists() and res_subdir is not None:
        # Only skip when the recorded stage2 run used exactly the current
        # stage1 checkpoint (normalized absolute-path comparison), the fixed
        # seed, and the current pinned commit.
        info = read_stage1_marker(marker)
        recorded = info.get("ckpt")
        ckpt_ok = bool(recorded) and _normalize_ckpt_ref(run_dir, recorded) == Path(ckpt).resolve()
        seed_ok = info.get("seed") == str(STAGE2_SEED)
        if ckpt_ok and seed_ok and marker_commit_matches(info, commit):
            log(f"stage2 already complete (res={res_subdir.name}), skipping")
            return res_subdir
        log(
            f"stage2 marker stale (ckpt_ok={ckpt_ok}, seed_ok={seed_ok}, "
            f"commit {info.get('commit')!r} vs pinned {commit!r}); rerunning stage2"
        )
    marker.unlink(missing_ok=True)
    (run_dir / ".backtest.done").unlink(missing_ok=True)
    artifact_root = run_dir.relative_to(repo)
    # A checkpoint reused from another experiment lives outside this
    # experiment's checkpoint dir; pass it as an absolute path (stage2
    # resolves only relative saved_model values against the checkpoint dir).
    own_ckpt = ckpt.parent.resolve() == (run_dir / "checkpoints").resolve()
    saved_model = ckpt.name if own_ckpt else str(ckpt)
    run_with_retries(
        [
            sys.executable, "stage2.py",
            f"train.seed={STAGE2_SEED}",
            f"artifact_root={artifact_root}",
            f'predictor.saved_model="{saved_model}"',
        ],
        run_dir / "stage2.log", repo, STAGE2_MAX_ATTEMPTS,
    )
    res_subdir = find_stage2_outputs(run_dir)
    if res_subdir is None:
        raise StageError("stage2 finished but prediction/metric artifacts not found")
    marker.write_text(
        f"res={res_subdir.name}\nckpt={Path(ckpt).resolve()}\n"
        f"seed={STAGE2_SEED}\ncommit={commit}\n"
    )
    log(f"stage2 done: res={res_subdir.name}")
    return res_subdir


def backtest(repo: Path, run_dir: Path, res_subdir: Path, universe: str, commit: str = "") -> Path:
    marker = run_dir / ".backtest.done"
    metric_path = find_backtest_metric(res_subdir)
    if marker.exists() and metric_path is not None:
        info = read_stage1_marker(marker)
        if info.get("protocol") == backtest_protocol_sig(universe) and marker_commit_matches(info, commit):
            log(f"backtest already complete ({metric_path.parent.name}), skipping")
            return metric_path
        log(
            f"backtest marker stale (protocol {info.get('protocol')!r} vs "
            f"{backtest_protocol_sig(universe)!r}, commit {info.get('commit')!r} vs "
            f"pinned {commit!r}); rerunning backtest"
        )
    marker.unlink(missing_ok=True)
    pred_path = res_subdir / f"{STAGE2_SEED}_best.pkl"
    run_with_retries(
        [
            sys.executable, "backtest_qlib.py",
            "--pred_path", str(pred_path),
            "--universe", universe,
            *BACKTEST_ARGS,
        ],
        run_dir / "backtest.log", repo, BACKTEST_MAX_ATTEMPTS,
    )
    metric_path = find_backtest_metric(res_subdir)
    if metric_path is None:
        raise StageError("backtest finished but portfolio_metric.csv not found")
    marker.write_text(
        f"metric={metric_path.relative_to(run_dir)}\n"
        f"protocol={backtest_protocol_sig(universe)}\ncommit={commit}\n"
    )
    log(f"backtest done: {metric_path.parent.name}")
    return metric_path


def build_result_text(metrics: dict, port: dict, bench: dict) -> str:
    ar = port["Annualized Return"]
    ar_b = bench.get("Annualized Return")
    excess = ar - ar_b if ar_b is not None else None
    ar_line = f"Annual Return: {ar * 100:.2f}%"
    if ar_b is not None:
        ar_line += f"（基准 {ar_b * 100:.2f}%，超额 {excess * 100:.2f}%）"
    lines = [
        "Status: DONE（test 区间 2023-01-01 – 2025-12-31）",
        "",
        f"IC: {fmt4(metrics['IC'])}",
        f"ICIR: {fmt4(metrics['ICIR'])}",
        f"RankIC: {fmt4(metrics['RankIC'])}",
        f"RankICIR: {fmt4(metrics['RankICIR'])}",
        "",
        ar_line,
        f"Sharpe: {fmt4(port['Sharpe Ratio'])}",
        f"Sortino: {fmt4(port['Sortino Ratio'])}",
        f"MDD: {port['MDD'] * 100:.2f}%",
        f"Calmar: {fmt4(port['Calmar Ratio'])}",
        f"Turnover: {fmt4(port['Turnover'])}",
    ]
    return "\n".join(lines)


def run_experiment(repo: Path, exp: dict, valid_ids) -> bool:
    exp_id, name, branch = exp["id"], exp["name"], exp["branch"]
    log(f"===== experiment {exp_id} ({name}) on {branch} =====")
    run_dir = run_dir_for(repo, exp_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_queue_status(repo, exp_id, "running")
    commit_main(repo, f"Phase 2: start {exp_id}-{name} (running)")
    try:
        commit = resolve_pinned_commit(repo, exp)
        source = str(exp.get("stage1_source") or "self")
        if source not in ("self", "external") and source not in valid_ids:
            raise StageError(f"stage1_source={source}: no such experiment in the queue")
        if source == "external":
            # Fail fast before checking out / training anything.
            external_stage1_ckpt(repo, exp)
        checkout_experiment(repo, branch, commit)
        universe = read_branch_universe(repo)
        exp = dict(exp, commit=commit)
        ckpt, stage1_prov = stage1(repo, run_dir, exp)
        res_subdir = stage2(repo, run_dir, ckpt, commit)
        metric_path = backtest(repo, run_dir, res_subdir, universe, commit)

        metrics = read_metric_csv(res_subdir / f"{STAGE2_SEED}_metric.csv")
        port, bench = read_portfolio_metric_csv(metric_path)
        summary = {
            "id": exp_id, "name": name, "branch": branch, "commit": commit,
            "stage1_best_ckpt": ckpt.name,
            "stage1_source": stage1_prov,
            "res_dir": res_subdir.name,
            "stage2_metrics": metrics,
            "portfolio": port,
            "benchmark": bench,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        result_text = build_result_text(metrics, port, bench)
        if stage1_prov["mode"] == "reused":
            if stage1_prov["source_id"] == "external":
                stage1_line = (
                    f"Stage 1 复用外部 exact checkpoint："
                    f"`{stage1_prov['ckpt']}`（本实验未重新训练 Stage 1）"
                )
            else:
                stage1_line = (
                    f"Stage 1 复用实验 {stage1_prov['source_id']} 的正式 checkpoint："
                    f"`{stage1_prov['ckpt']}`（本实验未重新训练 Stage 1）"
                )
        else:
            stage1_line = f"Stage 1 best checkpoint：`{ckpt.name}`（self-trained）"
        conclusion = (
            f"Phase 2 固定执行器完成正式训练、预测与回测"
            f"（pinned commit {commit}）。"
            f"{stage1_line}；"
            f"Stage 2 seed {STAGE2_SEED}。\n\n"
            f"产物：`artifacts/{exp_id}/run/`（checkpoints/、res/、"
            f"stage1.log、stage2.log、backtest.log、summary.json）。"
        )
        status, ok = "done", True
    except Exception as e:  # noqa: BLE001 - record any failure and move on
        log(f"experiment {exp_id} FAILED: {e}")
        result_text = f"Status: FAILED\n\n失败原因：{e}"
        conclusion = (
            f"Phase 2 执行失败：{e}\n\n"
            f"全部 artifact 与日志保留在 `artifacts/{exp_id}/run/`。"
        )
        status, ok = "failed", False
    finally:
        checkout(repo, "main")

    update_record(repo, exp, result_text, conclusion)
    set_queue_status(repo, exp_id, status)
    commit_main(repo, f"Phase 2: {exp_id}-{name} {status}")
    log(f"===== experiment {exp_id} -> {status} =====")
    return ok


def select_experiments(experiments, only=None):
    """Pick which queue entries to run.

    Default: pending + running (a running entry means a previous execution
    was interrupted; resume from its artifact markers). done is always
    skipped; failed is retried only via an explicit --only selection.
    """
    if only is not None:
        wanted = set(only)
        return [
            e for e in experiments
            if e["id"] in wanted and e["status"] in ("pending", "running", "failed")
        ]
    return [e for e in experiments if e["status"] in ("pending", "running")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Path to the HVQ-Stock repo root.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only run these experiment ids (also selects failed entries for retry).")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    ensure_control_branch(repo)
    queue = load_queue(repo)
    experiments = sorted(queue["experiments"], key=lambda e: e["id"])

    selected = select_experiments(experiments, args.only)

    if not selected:
        log("no experiments to run")
        return 0

    log(f"experiments to run: {[e['id'] for e in selected]}")
    valid_ids = {e["id"] for e in experiments}
    results = {}
    for exp in selected:
        results[exp["id"]] = run_experiment(repo, exp, valid_ids)

    log(f"all done: {results}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
