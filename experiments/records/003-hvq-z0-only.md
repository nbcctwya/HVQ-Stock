# 003 — hvq-z0-only

## Idea

001 使用两级 Residual HVQ：`z = z0 + z1`，其中 `z0` 是第 0 级 VQ 输出，
`z1` 是对残差 `h - z0` 进行第 1 级 VQ 得到的输出。003 做消融：Stage 1
保持与 001 完全相同的两级 Residual HVQ 训练方式，但 Stage 2 收益预测阶段
只使用第一级量化表示 `z0`，而不是 `z0 + z1`。

## Motivation

验证 Residual HVQ 的第二级残差量化 `z1` 是否真的为最终股票收益预测提供了
有效信息。若 003 与 001 表现持平，说明第二级残差量化对下游预测没有实质
贡献；若 003 显著变差，则说明 `z1` 携带了有效信息。

## Modification

- `module/quantise_hvq.py::ResidualVectorQuantiser` 新增
  `forward_level0(h_batch)`：只运行第 0 级量化，返回
  `(z_q0, loss_0, ([ppl_0], [min_enc_0], [idx_0]))`，结构与 `forward` 同构；
  默认 `forward`（`z0 + z1`）行为不变，Stage 1 不受影响。
- `trainer/train_ypred.py::GenerateReturn`：新增 `predictor.z0_only` 开关
  （默认 False 保持 001 行为；为 True 时要求量化器为
  `ResidualVectorQuantiser` 否则报错），`z0_only=True` 时 forward 调用
  `quantizer.forward_level0(h_batch)` 取 `z0` 作为 Stage 2 输入。
- `configs/config.yaml`：`predictor.z0_only: true`——默认配置即 003 实验
  本身，无需实验特有 CLI override；Stage 2 `train.seed: 0` 不变。
- 新增 `tests/test_z0_only.py`（6 个用例：z0 数值等于第 0 级 codebook
  查表、z0 排除第二级、返回接口与 forward 同构、STE 梯度回传、默认
  forward 行为不变、默认 config 已启用 z0_only）。
- 分支根目录 `README.md` 改写为 003 实验说明。

## Constraints

- Stage 1 两级 HVQ 结构、训练目标、训练流程与 001 完全一致（固定 seed 42，
  `stage1.py` 与量化器默认 `forward` 均未改动）
- 数据划分：train 2009–2020，valid 2021–2022，test 2023–2025（CSI300）
- 训练协议：两 stage 均 max 70 epoch，early stop patience 15；Stage 2 seed 0
- 回测协议：Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，
  close 成交，CN limit=None（本阶段不执行）
- 唯一核心实验变量：Stage 2 量化表示输入由 `z0 + z1` 改为 `z0`；
  MoE n_expert=2（k=1）、use_prior=True、target_day=5 等均与 001 一致

## Git

Base: exp/001-hvq-residual-2level
Branch: exp/003-hvq-z0-only
Commit: 057a722
Queue pinned commit: b84ba53d7af54adb35c42974b96ab81ea826feec
（实验代码 057a722 + 公共 correctness fix b84ba53；Phase 2 只执行该 pinned commit）
Stage 1 provenance: 复用实验 001 的 exact Stage 1（queue `stage1_source: "001"`）

## Smoke Test

Status: PASS
Notes: 单元测试 6/6（`tests.test_z0_only`）+ 14/14（`tests.test_hvq` 与
`tests.test_protocol_metrics` 回归）通过（conda `prism-vq`）。
Stage 1 smoke：`stage1.py train.num_epochs=1 train.run_name=z0only_smoke
artifact_root=artifacts/003/smoke`，1 epoch 完成，
val_loss=0.6443——与 001 当时的 Stage 1 smoke checkpoint
（`hvq_smoke-epoch=0-val_loss=0.6443.ckpt`）val_loss 完全一致，
确认 Stage 1 行为与 001 相同。
Stage 2 smoke：`stage2.py train.num_epochs=1 train.seed=0
artifact_root=artifacts/003/smoke
predictor.saved_model="z0only_smoke-epoch=0-val_loss=0.6443.ckpt"`，
两级 HVQ checkpoint 以 strict 方式加载（missing=0/unexpected=0），
训练 + valid/test 推理全流程完成（Test RankIC 0.0417，仅供流程验证）。
另以临时脚本实证（留存于 `artifacts/003/smoke/verify_z0_smoke.py`）：
z0_only=True 时 Stage 2 forward 使用的 z_q 与 `forward_level0` 输出
完全一致（max diff 0），且不等于 `z0+z1`；z0_only=False 时与 `z0+z1`
完全一致，001 默认行为未破坏。
全部产物（stage1/stage2 checkpoint、wandb 文件、prediction、metric、日志）
均落在 `artifacts/003/smoke/` 下，公共 `checkpoints/` 与 `res/` 经前后
diff 确认无任何写入。

## Result

Status: DONE — Corrected Protocol seed0（test 区间 2023-01-01 – 2025-12-31）

协议：corrected Stage 2 freeze（encoder/quantizer/RevIN 全程 eval）+ corrected MDD（初始 NAV=1 计入 peak）；Stage 2 seed 0；回测 Top30/Drop5，open 0.0005 / close 0.0015，min_cost 0，close 成交，CN limit=None。

IC: 0.0333
ICIR: 0.1810
RankIC: 0.0533
RankICIR: 0.2830

Annual Return: 11.23%（基准 6.40%，超额 4.83%，AR 差值口径）
Sharpe: 0.7651
Sortino: 1.1160
MDD: -12.64%
Calmar: 0.8886
Turnover: 0.3275

Stage 1 checkpoint provenance：reused 自实验 001（`stage1_source: "001"`，结构与数值均验证一致）：`artifacts/001/run/checkpoints/hvq_csi300_full-epoch=5-val_loss=0.4592.ckpt`，本实验未重新训练 Stage 1。
Stage 2 seed：0
产物：`artifacts/003/run/`（checkpoints/、res/、backtest、summary.json、stage1/stage2/backtest 日志）。

代码版本：实验代码 057a722 + 公共 correctness fix b84ba53（同 001 的 freeze 扩展修复；不改变实验 Idea、结构、超参、数据或协议）。

### Historical（不再作为正式比较依据）

pre-fix seed0（quantizer-only freeze override + 旧 MDD 实现，旧产物备份于 `artifacts/003/run/pre_fix/`）：IC 0.0333 / ICIR 0.1810 / RankIC 0.0533 / RankICIR 0.2830 / AR 11.23% / Sharpe 0.7651 / MDD -12.64% / Calmar 0.8886 / Turnover 0.3275。

### Corrected 与 Historical 的关系

corrected 重跑与 historical 逐位一致，原因同 001。此外 corrected 轮的 Stage 1 由自有 checkpoint 切换为显式复用 001 的正式 checkpoint（数值等价，provenance 更清晰）。

## Conclusion

Corrected Protocol 下，z0-only（003）与完整 z0+z1（001）使用同一份
Stage 1 checkpoint（显式复用）：003 RankIC 0.0533 高于 001 的 0.0506，
IC 0.0333 略低于 0.0352，Sharpe 0.7651 vs 0.7591 持平略高，
MDD -12.64% 明显优于 001 的 -18.49%，AR 11.23% 低于 13.69%。
结论：第二级残差量化 z1 对下游收益预测没有稳定贡献——去掉 z1 后 RankIC、
Sharpe、MDD 均不差甚至更优，仅 AR 与 IC 略降。此前"z1 可能主要携带
reconstruction/detail 信息、对收益预测帮助有限"的信号在 corrected
protocol 下仍然成立。
