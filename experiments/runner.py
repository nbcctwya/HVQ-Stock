#!/usr/bin/env python3
"""Phase 2 executor: consume pending experiments from experiments/queue.yaml.

Fixed pipeline per experiment (see experiments/templates/RULES.md):

    pending -> checkout exp branch -> stage1 -> best stage1 checkpoint
            -> stage2 (seed 0) -> prediction/metrics -> qlib backtest
            -> collect metrics -> back to main -> update record + queue

All formal artifacts live under ``artifacts/<ID>/run/`` via the
``artifact_root`` execution-layer override. The runner is deterministic and
resumable: a stage is re-executed only when its completion marker or its
expected artifacts are missing.

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


def run_dir_for(repo: Path, exp_id: str) -> Path:
    return repo / "artifacts" / exp_id / "run"


def find_best_stage1_ckpt(run_dir: Path):
    """Return (val_loss, path) of the best stage1 checkpoint, or None."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    cands = []
    for p in ckpt_dir.glob("*.ckpt"):
        m = BEST_CKPT_RE.search(p.name)
        if m and p.stat().st_size > 0:
            cands.append((float(m.group(2)), p))
    if not cands:
        return None
    return min(cands, key=lambda t: t[0])


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


def stage1(repo: Path, run_dir: Path) -> Path:
    marker = run_dir / ".stage1.done"
    best = find_best_stage1_ckpt(run_dir)
    if marker.exists() and best is not None:
        log(f"stage1 already complete (best={best[1].name}, val_loss={best[0]:.4f}), skipping")
        return best[1]
    marker.unlink(missing_ok=True)
    artifact_root = run_dir.relative_to(repo)
    run_with_retries(
        [sys.executable, "stage1.py", f"artifact_root={artifact_root}"],
        run_dir / "stage1.log", repo, STAGE1_MAX_ATTEMPTS,
    )
    best = find_best_stage1_ckpt(run_dir)
    if best is None:
        raise StageError("stage1 finished but no best checkpoint found")
    marker.write_text(f"best={best[1].name}\nval_loss={best[0]}\n")
    log(f"stage1 done: best={best[1].name} val_loss={best[0]:.4f}")
    return best[1]


def stage2(repo: Path, run_dir: Path, ckpt: Path) -> Path:
    marker = run_dir / ".stage2.done"
    res_subdir = find_stage2_outputs(run_dir)
    if marker.exists() and res_subdir is not None:
        log(f"stage2 already complete (res={res_subdir.name}), skipping")
        return res_subdir
    marker.unlink(missing_ok=True)
    artifact_root = run_dir.relative_to(repo)
    run_with_retries(
        [
            sys.executable, "stage2.py",
            f"train.seed={STAGE2_SEED}",
            f"artifact_root={artifact_root}",
            f"predictor.saved_model={ckpt.name}",
        ],
        run_dir / "stage2.log", repo, STAGE2_MAX_ATTEMPTS,
    )
    res_subdir = find_stage2_outputs(run_dir)
    if res_subdir is None:
        raise StageError("stage2 finished but prediction/metric artifacts not found")
    marker.write_text(f"res={res_subdir.name}\nckpt={ckpt.name}\n")
    log(f"stage2 done: res={res_subdir.name}")
    return res_subdir


def backtest(repo: Path, run_dir: Path, res_subdir: Path, universe: str) -> Path:
    marker = run_dir / ".backtest.done"
    metric_path = find_backtest_metric(res_subdir)
    if marker.exists() and metric_path is not None:
        log(f"backtest already complete ({metric_path.parent.name}), skipping")
        return metric_path
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
    marker.write_text(f"metric={metric_path.relative_to(run_dir)}\n")
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


def run_experiment(repo: Path, exp: dict) -> bool:
    exp_id, name, branch = exp["id"], exp["name"], exp["branch"]
    log(f"===== experiment {exp_id} ({name}) on {branch} =====")
    run_dir = run_dir_for(repo, exp_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_queue_status(repo, exp_id, "running")
    commit_main(repo, f"Phase 2: start {exp_id}-{name} (running)")
    try:
        checkout(repo, branch)
        universe = read_branch_universe(repo)
        ckpt = stage1(repo, run_dir)
        res_subdir = stage2(repo, run_dir, ckpt)
        metric_path = backtest(repo, run_dir, res_subdir, universe)

        metrics = read_metric_csv(res_subdir / f"{STAGE2_SEED}_metric.csv")
        port, bench = read_portfolio_metric_csv(metric_path)
        summary = {
            "id": exp_id, "name": name, "branch": branch,
            "stage1_best_ckpt": ckpt.name,
            "res_dir": res_subdir.name,
            "stage2_metrics": metrics,
            "portfolio": port,
            "benchmark": bench,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        result_text = build_result_text(metrics, port, bench)
        conclusion = (
            f"Phase 2 固定执行器完成正式训练、预测与回测。"
            f"Stage 1 best checkpoint：`{ckpt.name}`；"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Path to the HVQ-Stock repo root.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only run these experiment ids (also selects failed entries for retry).")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    queue = load_queue(repo)
    experiments = sorted(queue["experiments"], key=lambda e: e["id"])

    selected = []
    for exp in experiments:
        if args.only is not None:
            if exp["id"] in args.only and exp["status"] in ("pending", "failed", "running"):
                selected.append(exp)
        elif exp["status"] == "pending":
            selected.append(exp)

    if not selected:
        log("no experiments to run")
        return 0

    log(f"experiments to run: {[e['id'] for e in selected]}")
    results = {}
    for exp in selected:
        results[exp["id"]] = run_experiment(repo, exp)

    log(f"all done: {results}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
