# HVQ canonical dataset

Main has exactly one schema for **CSI300 and SP500**:

**158 stock + 13 prior + 63 market + 10 future returns = 244 dimensions.**

| Qlib group | Slice | Parsed shape |
| --- | --- | --- |
| `feature` (stock) | `0:158` | `[N,T,158]` |
| `prior` | `158:171` | `[N,13]` |
| `market` | `171:234` | `[N,T,63]` |
| `label` (future returns) | `234:244` | `[N,10]` |

The generated datasets use **T=20**. `num_features` remains **158**.
`schema.py` centrally defines the dimensions, group order, slices and filename
convention. All six trainers and `utils/test.py` call `unpack_batch(batch)`.
It accepts only `[N,T,244]`, preserves singleton stock batches, and raises
`ValueError` for any other width (including 181). It never guesses a schema
and always returns all four groups. `parts.target(5)` reads only the fifth
column of `future_returns`, with integer horizon bounds checked. DataLoader
creation also validates the sample width before entering a model loop.

Experiments **001–006** retain their own historical conventions. Reproduction
belongs to their frozen branches: check out the corresponding branch and
regenerate data using its code. Main does not implement their data format.

## Direct generation from raw data

`get_dataset.py` is the single preprocessing entrypoint:

```text
local Qlib OHLCV -> inherited HVQ Alpha158 ------------------ feature (158)
raw JKP daily CSV -> prior rolling returns + normalization - prior (13)
Qlib Mask expressions -> market normalization -------------- market (63)
Qlib forward close expressions + label processors ---------- label (10)
                                  |
                       four-group Qlib DataFrame
                                  |
                       CanonicalSampler, step_len=20
                                  |
                       canonical train/valid/test pickle
```

No input pickle is read. The generator inherits the existing Qlib Alpha158
feature configuration without replacing or extending stock formulas, including
HVQ's existing US stock feature handling. JKP definitions also stay unchanged:
select the country's `vw_cap`, `daily` rows, pivot factors by date/name, reindex
to the Qlib calendar, shift by one trading day, then compute
`expm1(rolling_20_sum(log1p(ret)))`. Exactly 13 prior columns are required.
Stock and prior use their existing RobustZScoreNorm/Fillna processors.

The ten labels keep HVQ's existing expressions
`Ref($close, -h) / Ref($close, -1) - 1`, for `h=1..10`, named `RET_1D` through
`RET_10D` (including the existing zero-valued raw first horizon). Train/valid
use `DK_L`, with DropnaLabel and CSRankNorm; test uses `DK_I`.

`Alpha158WithJKP` produces the ordered Qlib groups `feature`, `prior`, `market`,
`label`. `CanonicalSampler` accepts only this DataFrame layout, retains actual
column names, and serializes all 244 columns. Stock/prior/label windows use
Qlib's existing `ffill+bfill` behavior. Market windows use their own trading
calendar by endpoint date, so suspended or newly listed stocks share exactly
the same market sequence as other stocks on that date. Missing market dates,
insufficient history, incorrect formula ordering or non-finite market values
fail explicitly.

## CN and US market63

| Universe | Indices, in feature order |
| --- | --- |
| CSI300 | `sh000300`, **`sh000852`**, `sh000905` |
| SP500 | `^gspc`, `^dji`, `^ndx` |

`market.py` follows AlphaMaster's
`src/alphamaster/dataset.py::marketDataHandler.get_feature_config` and the
explicit index lists in `configs/master_csi300.yaml` / `master_sp500.yaml`.
For each index, first emit `$close/Ref($close,1)-1`, then the following four
expressions for each window `w=5,10,20,30,60`, in this order:

1. `Mean($close/Ref($close,1)-1,w)`
2. `Std($close/Ref($close,1)-1,w)`
3. `Mean($volume,w)/$volume`
4. `Std($volume,w)/$volume`

Every expression uses Qlib `Mask(expression, "index")` unchanged: **21 × 3 = 63**.
Market preprocessing uses a separate group with AlphaMaster's universe/date
weighting, RobustZScoreNorm fitted on 2009–2020, clipping and Fillna. Entirely
missing index expressions fail before Fillna can hide the missing feed.
Processed market rows are checked for equality across stocks on each date.

**The baseline currently ignores market63.** No Gate, market encoder, market
conditioning or fusion mechanism is added. Later experiments may investigate
market-aware mechanisms.

## Commands and filenames

From the HVQ repository root, in an environment with Qlib and PyTorch
(locally `prism-vq`):

```bash
python -m dataset.get_dataset --universe csi300
python -m dataset.get_dataset --universe sp500
```

`python dataset/get_dataset.py` accepts the same arguments. Inputs are the
Qlib providers configured in `dataset/2025_{csi300,sp500}.yaml` and the raw
JKP CSVs under `dataset/jkpdata/`:

- `[chn]_[all_themes]_[daily]_[vw_cap].csv`
- `[usa]_[all_themes]_[daily]_[vw_cap].csv`

Override these with `--jkp-path` and `--data-handler-config` if necessary.
The generator reads the default output directory from `data.data_path` in
`configs/config.yaml`, currently `dataset/processed`; `--output-dir` can
choose a different directory. Outputs are exclusively created and never
silently overwritten:

```text
dataset/processed/CN/csi300_20_h10_train.pkl
dataset/processed/CN/csi300_20_h10_valid.pkl
dataset/processed/CN/csi300_20_h10_test.pkl
dataset/processed/US/sp500_20_h10_train.pkl
dataset/processed/US/sp500_20_h10_valid.pkl
dataset/processed/US/sp500_20_h10_test.pkl
```

Stage1 and Stage2 read only these names under `data.data_path/{CN,US}`.
There is no alternate path, version switch or filename fallback. If generating
elsewhere, set `data.data_path` to that output directory when running the model.
The separate output directory keeps generated pickles away from the raw JKP
inputs in `dataset/jkpdata/`. Pickles are reproducible artifacts and ignored by
Git. Older artifact files are neither read nor maintained by main.

## Validate without training

```bash
python -m unittest discover -s tests -v
python scripts/smoke_dataset.py
```

The smoke script **generates from actual raw Qlib and JKP inputs**, with a
small 2020–2021 date range and three days per split. It disables `pickle.load`
during generation to catch accidental serialized-data dependencies, then
reloads its newly generated outputs to check all four groups, shape, horizon
5, same-date market sequences, finite information groups and a real DataLoader batch for
both markets. It uses temporary files and removes them on completion; pass
`--output-dir /path/to/fresh/smoke-dir` to retain smoke outputs. Smoke files
are small validation fixtures and are not production train/valid/test splits.
Missing future labels in DK_I remain unchanged (for example around stock suspensions).
No model training, inference export or formal backtest is performed.
