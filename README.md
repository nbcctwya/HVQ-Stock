# exp/003 — hvq-z0-only

- Base: `exp/001-hvq-residual-2level`
- Branch: `exp/003-hvq-z0-only`

## Idea / Motivation

001 使用两级 Residual HVQ：最终量化表示为 `z = z0 + z1`，其中 `z0` 是第 0 级
VQ 输出，`z1` 是对残差 `h - z0` 进行第 1 级 VQ 得到的输出。

003 是一个消融实验：**Stage 1 保持与 001 完全相同的两级 Residual HVQ 训练方式，
但 Stage 2 收益预测阶段只使用第一级量化表示 `z0`，而不是 `z0 + z1`**。

目的：验证 Residual HVQ 的第二级残差量化 `z1` 是否真的为最终股票收益预测
提供了有效信息。对比：

- 001：Stage 2 input = `z0 + z1`
- 003：Stage 2 input = `z0`

若 003 与 001 的预测/回测表现持平，说明第二级残差量化对下游预测没有实质
贡献；若 003 显著变差，则说明 `z1` 携带了有效信息。

## 核心修改（相对 base 001）

- `module/quantise_hvq.py::ResidualVectorQuantiser` 新增 `forward_level0(h_batch)`
  方法：只运行第 0 级量化，返回 `(z_q0, loss_0, ([ppl_0], [min_enc_0], [idx_0]))`，
  返回结构与 `forward` 同构。默认 `forward` 的 `z0 + z1` 行为完全不变，
  Stage 1 训练不受影响。
- `trainer/train_ypred.py::GenerateReturn`：
  - `__init__` 读取 `config['predictor'].get('z0_only', False)`；`z0_only=True`
    时要求量化器为 `ResidualVectorQuantiser`，否则报错。
  - `forward` 中 `z0_only=True` 时调用 `self.quantizer.forward_level0(h_batch)`
    取 `z0`（随后照常 `.detach()`，送入 loading generator / latent value head /
    return predictor）；否则走原 `self.quantizer(h_batch)` 路径（`z0 + z1`）。
- `configs/config.yaml`：`predictor.z0_only: true`——默认配置即为 003 实验本身，
  不需要任何实验特有 CLI override；`train.seed: 0`（Stage 2）保持不变。
- 新增 `tests/test_z0_only.py`（6 个用例：z0 数值等于第 0 级 codebook 查表、
  z0 排除第二级、返回接口与 forward 同构、STE 梯度回传、默认 forward 行为不变、
  默认 config 已启用 z0_only）。

## 与 base（001）的区别

唯一核心实验变量：Stage 2 的量化表示输入由 `z0 + z1` 改为 `z0`。

保持不变：

- Stage 1 两级 HVQ 结构、训练目标（recon + vq + pred）与训练流程（固定
  seed 42）与 001 完全一致——`forward` 默认行为未动，`stage1.py` 未改动。
- 数据划分（train 2009–2020 / valid 2021–2022 / test 2023–2025，CSI300）、
  训练协议（max 70 epoch、early stop patience 15）、回测协议
  （Top30/Drop5，open 0.0005 / close 0.0015）、MoE/prior 等全部参数。
- Stage 2 默认 `train.seed: 0`。

## 运行

默认配置即为本实验（z0-only），无需额外 override 启用核心改动：

```bash
# Stage 1（两级 Residual HVQ，与 001 相同；固定 seed 42）
conda run -n prism-vq python stage1.py data.universe=csi300

# Stage 2（z0-only；configs/config.yaml 的 stage2_presets.csi300.predictor.saved_model
# 需填入 Stage 1 生成的 checkpoint 文件名）
conda run -n prism-vq python stage2.py data.universe=csi300
```

CLI override 仅用于统一执行层参数，例如：

```bash
# artifact 隔离：checkpoints/res/wandb 落到 artifacts/003/run/ 下，
# Stage 2 也会从该目录的 checkpoints/ 加载 Stage 1 checkpoint
python stage1.py data.universe=csi300 artifact_root=artifacts/003/run
python stage2.py data.universe=csi300 artifact_root=artifacts/003/run \
  predictor.saved_model="<stage1_ckpt_name>.ckpt"
```

## 单元测试

```bash
conda run -n prism-vq python -m unittest tests.test_z0_only -v
# 回归：001 的 HVQ 测试不受影响
conda run -n prism-vq python -m unittest tests.test_hvq -v
```

## Smoke 状态

PASS — 单元测试 6/6（z0-only）+ 14/14（001 HVQ 回归）通过；Stage 1 与
Stage 2 各 1 epoch smoke 跑通（`artifact_root=artifacts/003/smoke`），Stage 2
正确从 smoke artifact 目录加载两级 HVQ checkpoint 且实际使用 `z0`；全部产物
落在 `artifacts/003/smoke/`，未写入公共 `checkpoints/` / `res/`。详见 `main`
分支 `experiments/records/003-hvq-z0-only.md`。

---

本仓库以 [PRISM-VQ](../PRISM-VQ)（IJCAI-ECAI 2026）为底；两级 Residual HVQ 的
完整实现说明见 `exp/001-hvq-residual-2level` 分支 README。原始 PRISM-VQ 项目
说明见 `main` 分支的 README。
