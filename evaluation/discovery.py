"""Discovery of formal, accepted prediction artifacts.

Seed 0 comes from the Phase2 formal run at ``artifacts/<experiment>/run``.
Seeds >= 1 come from Phase3 batches: only receipts under
``artifacts/_phase3/batches/<batch>/receipts/`` with ``accepted == true``
are formal. Anything under incoming/, staging/, diagnostics/, smoke/ or a
batch without accepted receipts is never used.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .protocol import REPO_ROOT

PHASE3_BATCHES_DIR = REPO_ROOT / "artifacts" / "_phase3" / "batches"

_SEED_DIR_RE = re.compile(r"/seed(\d+)(?:/|$)")
_PRED_FILE_RE = re.compile(r"^(\d+)_.*\.pkl$")


@dataclass(frozen=True)
class FormalPrediction:
    experiment: str
    seed: int
    path: Path  # absolute
    source: str  # "phase2_run" or "phase3:<batch>"

    @property
    def rel_path(self) -> str:
        return str(self.path.relative_to(REPO_ROOT))


class DiscoveryError(RuntimeError):
    pass


def _find_seed0(experiment: str) -> FormalPrediction:
    run_dir = REPO_ROOT / "artifacts" / experiment / "run"
    candidates = sorted(run_dir.glob("res/*/0_best.pkl"))
    if not candidates:
        raise DiscoveryError(
            f"No formal Phase2 seed0 prediction found for experiment '{experiment}' "
            f"under {run_dir}/res/*/0_best.pkl"
        )
    if len(candidates) > 1:
        raise DiscoveryError(
            f"Ambiguous seed0 prediction for experiment '{experiment}': "
            + ", ".join(str(c.relative_to(REPO_ROOT)) for c in candidates)
        )
    return FormalPrediction(experiment, 0, candidates[0], "phase2_run")


def _iter_accepted_receipts() -> List[dict]:
    receipts = []
    if not PHASE3_BATCHES_DIR.is_dir():
        return receipts
    for batch_dir in sorted(PHASE3_BATCHES_DIR.iterdir()):
        receipts_dir = batch_dir / "receipts"
        if not receipts_dir.is_dir():
            continue
        for receipt_path in sorted(receipts_dir.glob("*.json")):
            with open(receipt_path) as f:
                receipt = json.load(f)
            if receipt.get("accepted") is not True:
                continue
            receipt["_batch"] = batch_dir.name
            receipts.append(receipt)
    return receipts


def _receipt_to_prediction(receipt: dict) -> FormalPrediction:
    code = receipt.get("task", {}).get("code", "")
    experiment = code.split("/", 1)[-1] if code.startswith("code/") else code
    archive = Path(receipt["archive"])
    seed_match = _SEED_DIR_RE.search(str(archive).replace("\\", "/") + "/")
    if not seed_match:
        raise DiscoveryError(f"Cannot parse seed from archive path: {archive}")
    seed = int(seed_match.group(1))

    pred_rel = receipt.get("validation", {}).get("prediction")
    if pred_rel:
        pred_path = archive / pred_rel
    else:
        candidates = sorted(archive.glob(f"res/*/{seed}_best.pkl"))
        if len(candidates) != 1:
            raise DiscoveryError(
                f"Cannot locate prediction inside archive {archive} "
                f"(validation.prediction missing, glob found {len(candidates)})"
            )
        pred_path = candidates[0]

    if not pred_path.is_file():
        raise DiscoveryError(f"Accepted prediction file missing on disk: {pred_path}")
    try:
        pred_path.relative_to(REPO_ROOT)
    except ValueError:
        raise DiscoveryError(f"Archive path is outside the repository: {pred_path}")
    return FormalPrediction(experiment, seed, pred_path, f"phase3:{receipt['_batch']}")


def discover(experiments: List[str], seeds: List[int]) -> Dict[tuple, FormalPrediction]:
    """Return {(experiment, seed): FormalPrediction} for every requested pair.

    Missing formal artifacts are a hard error — never skipped silently.
    """
    found: Dict[tuple, FormalPrediction] = {}

    if 0 in seeds:
        for experiment in experiments:
            fp = _find_seed0(experiment)
            found[(experiment, 0)] = fp

    if any(s > 0 for s in seeds):
        by_pair: Dict[tuple, List[FormalPrediction]] = {}
        for receipt in _iter_accepted_receipts():
            fp = _receipt_to_prediction(receipt)
            by_pair.setdefault((fp.experiment, fp.seed), []).append(fp)
        for experiment in experiments:
            for seed in seeds:
                if seed == 0:
                    continue
                cands = by_pair.get((experiment, seed), [])
                if not cands:
                    raise DiscoveryError(
                        f"No accepted Phase3 receipt for experiment '{experiment}' seed {seed}. "
                        "Formal evaluation cannot proceed with a missing artifact."
                    )
                batches = {c.source for c in cands}
                if len(batches) > 1:
                    raise DiscoveryError(
                        f"Conflicting accepted receipts for experiment '{experiment}' seed {seed} "
                        f"from batches {sorted(batches)}. Resolve the intended batch explicitly."
                    )
                found[(experiment, seed)] = cands[0]

    missing = [(e, s) for e in experiments for s in seeds if (e, s) not in found]
    if missing:
        raise DiscoveryError(f"Missing formal predictions: {missing}")
    return found
