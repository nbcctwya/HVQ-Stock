"""Repository-derived protocol configuration."""

from pathlib import Path

BASELINE_ID = "prism_vq"
MODEL_ID = "prism_vq"
SEEDS = [0, 1, 2, 3, 4]
TEST = ["2023-01-01", "2025-12-31"]
SPLITS = {"train": ["2009-01-01", "2020-12-31"], "valid": ["2021-01-01", "2022-12-31"], "test": TEST}
RUNS = {
    "csi300": {
        "run_dir": "res/VQK512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3",
        "provider_uri": "~/.qlib/qlib_data/cn_data", "region": "cn", "instruments": "csi300",
        "benchmark": "SH000300", "limit_rule": "Qlib REG_CN default limit_threshold=0.095; suspension and tradability enforced by Exchange",
    },
    "sp500": {
        "run_dir": "res/VQK512_sp500_mo8_k4_mh64_md0.1_dm64_nh4_l1_d0.1_au0.001_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3",
        "provider_uri": "~/.qlib/qlib_data/us_data", "region": "us", "instruments": "sp500",
        "benchmark": "^gspc", "limit_rule": "Qlib REG_US default limit_threshold=None; suspension and tradability enforced by Exchange",
    },
}


def prediction_path(root: Path, market: str, seed: int) -> Path:
    return root / RUNS[market]["run_dir"] / f"{seed}_best.pkl"
