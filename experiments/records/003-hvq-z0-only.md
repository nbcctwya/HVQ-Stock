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

Status: PENDING

IC:
ICIR:
RankIC:
RankICIR:

Annual Return:
Sharpe:
Sortino:
MDD:
Calmar:
Turnover:

## Conclusion

正式实验完成后填写。
