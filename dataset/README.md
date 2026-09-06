# HVQ dataset schema

Schema v1: **158 stock + 13 prior + 10 future returns = 181**. Used by
experiments **001–006**, with their original `*_dl2_{train,valid,test}.pkl` data.
Keep those files and experiment branches for reproduction.

Schema v2: **158 stock + 13 prior + 63 market + 10 future returns = 244**.
Used by **main and future experiments**. Only CSI300 and SP500 are upgraded.

| Group | v2 slice | Parsed shape |
| --- | --- | --- |
| `feature` (stock) | `0:158` | `[N,20,158]` |
| `prior` | `158:171` | `[N,13]` |
| `market` | `171:234` | `[N,20,63]` |
| `label` (future returns) | `234:244` | `[N,10]` |

`schema.py` owns the widths, offsets, versions and filenames. All six trainers
and `utils/test.py` use `unpack_model_batch`, which calls `unpack_batch`.
The parser accepts v1 and v2, rejects unknown widths, retains singleton stock
batches, and returns `market_feature=None` for v1. `parts.target(5)` selects
the fifth column of **future_returns**, with horizon bounds checked. Never
slice returns from `num_features + num_prior_factors` to the end of a v2 batch.

## Market63

| Universe | Region | Indices, in feature order |
| --- | --- | --- |
| CSI300 | CN | `sh000300`, **`sh000852`**, `sh000905` |
| SP500 | US | `^gspc`, `^dji`, `^ndx` |

The formulas and ordering come from AlphaMaster's
`src/alphamaster/dataset.py::marketDataHandler.get_feature_config`, with the
explicit index lists and normalization in `configs/master_csi300.yaml` and
`configs/master_sp500.yaml`. Do not use AlphaMaster's historical default
`sh000903`; the active CN config uses `sh000852`.

For each index, the 21 expressions are the one-day return
`$close/Ref($close,1)-1`, followed by these four expressions for each window
`w = 5,10,20,30,60`, in order:

1. `Mean($close/Ref($close,1)-1,w)`
2. `Std($close/Ref($close,1)-1,w)`
3. `Mean($volume,w)/$volume`
4. `Std($volume,w)/$volume`

Every expression is evaluated by Qlib as `Mask(expression, "index")`, preserving
Qlib's reference, rolling-window, volume and mask semantics. The separate
`MarketDataHandler` applies RobustZScoreNorm (fit 2009–2020, clip outliers) and
Fillna to **market**, using the same universe/date weighting as AlphaMaster.
Entirely missing market expressions fail before Fillna can hide a missing
index. Processed rows must be identical across stocks for each date and finite.

`MarketWindowSampler` embeds the unchanged v1 TSDataSampler and adds market
windows by endpoint trading date. Market uses its own calendar so stock
suspensions or listing gaps cannot change the shared market history. The
complete 20-day market window is retained; missing dates/history fail explicitly.
The wrapper exposes the four column groups and the existing `get_index`,
`config`, integer/vectorized/tuple sampling interfaces used by HVQ DataLoaders.
It stores one market row per date, avoiding duplicating market storage for
every stock. The serialized sampler is self-contained: it does not read the
old pickle at runtime. Stock, prior, labels, fill rules and sample order are
preserved, including existing US stock feature handling and return definitions.

**The current baseline does not use market63.** No Gate, market encoder, or
market-aware MoE is implemented. This is infrastructure for future
market-aware experiments; model inputs and checkpoint state dictionaries stay
unchanged.

## Generate v2 data

Use the environment containing Qlib and PyTorch (locally `prism-vq`) from the
HVQ repository root. This migration requires the existing v1 samplers and
local Qlib CN/US providers. It deliberately does not recompute stock features,
JKP priors, or labels from raw data, preserving exact v1 compatibility.

```bash
python -m dataset.prepare_schema_v2 --universe csi300
python -m dataset.prepare_schema_v2 --universe sp500
```

The existing `python dataset/get_dataset.py --universe csi300` / `sp500`
entrypoints now dispatch to the same migration. The historical CSI500 path
is unchanged. For custom locations, pass `--source-dir`, `--output-dir`, and
optionally `--data-handler-config` to the module entrypoint
(`--data_handler_config` on the old entrypoint).

Input defaults to `dataset/data/{CN,US}`. Output defaults to the separate
`dataset/schema_v2_data/{CN,US}` directory inside HVQ, because local
`dataset/data` may be a symlink shared with other repositories. All outputs
use exclusive creation; existing v1 **and v2** files are never overwritten.
There are no new unversioned dataframe pickle outputs.

```text
dataset/schema_v2_data/CN/csi300_20_h10_schema_v2_train.pkl
dataset/schema_v2_data/CN/csi300_20_h10_schema_v2_valid.pkl
dataset/schema_v2_data/CN/csi300_20_h10_schema_v2_test.pkl
dataset/schema_v2_data/US/sp500_20_h10_schema_v2_train.pkl
dataset/schema_v2_data/US/sp500_20_h10_schema_v2_valid.pkl
dataset/schema_v2_data/US/sp500_20_h10_schema_v2_test.pkl
```

Stage1/Stage2 default to `data.schema_version=2` and
`data.schema_v2_path=dataset/schema_v2_data`. An explicit
`data.schema_version=1` selects legacy filenames under `data.data_path` for
main's compatibility checks. Experiments 001–006 retain their original code
and data configuration. New datasets are ignored by Git and must not be committed.

## Validate without training

```bash
python -m unittest discover -s tests -v
python scripts/smoke_schema_v2.py
# Optional: full Stage2 checkpoint with architecture matching main/config.yaml
python scripts/smoke_schema_v2.py --checkpoint /path/to/existing-stage2.ckpt
```

The smoke check covers both universes and all three splits: every legacy
storage row, sample identities, parsed shapes, return isolation, horizon 5,
shared market windows, and a real DataLoader batch. The optional checkpoint
is loaded strictly on CPU, used in eval/no-grad mode, and requires **exact**
v1/v2 prediction equality. It performs no optimizer step, formal inference
export, training, or backtest.
