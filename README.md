# HVQ-Stock

层次化向量量化（Hierarchical VQ / Residual VQ）股票预测实验仓库。代码以
[PRISM-VQ](../PRISM-VQ)（IJCAI-ECAI 2026）为底，在其两阶段框架上将 Stage 1 的
单层 `VectorQuantiser` 扩展为**残差式多级量化器 `ResidualVectorQuantiser`**，
用于对比离散表示容量对收益预测的影响。实验市场：CSI300。

## 与上游 PRISM-VQ 的关系

- 全部训练/评估代码拷贝自 `../PRISM-VQ`，仅做最小改动（见下文"改动点"）。
- 量化器接口保持兼容：`forward(h_batch)` 输入 `(N_t, 128)`，返回
  `(z_q, vq_loss, (perplexity, min_encodings, encoding_indices))`，z_q 走 STE。
- 默认配置 `vqvae.quantizer.type: 'single'` 时行为与上游完全一致。
- 不复制上游的大制品：数据集（`dataset/data/`，9GB）、checkpoints、结果文件。
  预处理 pickle 默认直接读取 `../PRISM-VQ/dataset/data`（见"数据路径配置"）。

## 目录结构

```text
stage1.py               Stage 1：训练 VQ-VAE 表征模型（固定 seed 42）
stage2.py               Stage 2：加载 Stage 1 checkpoint，训练收益预测模型
module/
  quantise.py           上游单层 VectorQuantiser（未改动）
  quantise_hvq.py       新增：ResidualVectorQuantiser + create_quantizer 工厂
  autoencoder.py        VQVAE（量化器实例化改为走 create_quantizer）
trainer/
  train_vqvae.py        Stage 1 Lightning 模块
  train_ypred.py        Stage 2 Lightning 模块（工厂分支 + codebook 冻结修复）
  train_ypred_*.py      上游消融变体（未改动，不支持 hvq 开关）
configs/config.yaml     全部实验配置
dataset/                数据预处理脚本与 universe 配置（不含 data/）
tests/                  单元测试
scripts/ utils/         上游工具脚本
```

## 残差式层次化量化器（HVQ）

`module/quantise_hvq.py::ResidualVectorQuantiser` 用 `nn.ModuleList` 堆叠
`num_levels` 个上游 `VectorQuantiser`：

- 第 0 级量化编码器输出 `h`；第 l 级量化前级残差 `r = h - sum(z_q_j)`。
- 最终 `z_q = sum(z_q_l)`，整体 STE：`z_q_output = h + (z_q - h).detach()`。
- 总 vq_loss 为各级 loss 之和（各级内部已含 beta commitment + codebook loss；
  `contras_loss` 逐层按各自残差计算）。
- 每一级有独立的 codebook 与 FeaturePool，dead-code 重初始化逻辑（training mode
  下自动生效）逐层独立保留。
- 返回签名与单层一致，但 perplexity / encodings / indices 为长度为
  `num_levels` 的按级 list（Stage 2 forward 只用 z_q，不受影响；Stage 1 的
  codebook 利用率统计取第 0 级）。

## 配置开关

`configs/config.yaml`：

```yaml
vqvae:
  quantizer:
    type: 'single'            # 'single'（默认，原行为）或 'hvq'
    num_levels: 2             # hvq 量化级数
    level_num_embed: [256, 256]  # 各级 codebook 大小，长度须等于 num_levels
```

Stage 1 与 Stage 2 通过同一个 `create_quantizer` 工厂实例化量化器，两阶段共用
一份 config 即可保证 checkpoint 参数命名（`quantizer.` / `quantizer.levels.*`）
严格匹配。

## 数据路径配置

预处理 pickle 目录优先读 `data.pickle_dir`（默认 `../PRISM-VQ/dataset/data`，
即复用 PRISM-VQ 已生成的数据），缺省回退 `data.data_path`；文件名拼接逻辑不变
（`<universe>_<window>_h<pred_len>_dl2_{train,valid,test}.pkl`，位于
`<pickle_dir>/<region>/` 下）。如需自行生成数据，用 `dataset/get_dataset.py`
生成后修改 `data.pickle_dir` 指向即可。

## 运行

```bash
# Stage 1（HVQ 版本）
conda run -n prism-vq python stage1.py data.universe=csi300 \
  vqvae.quantizer.type=hvq

# Stage 2：先在 configs/config.yaml 的 stage2_presets.<universe>.predictor.saved_model
# 填入 Stage 1 生成的 checkpoint 文件名，然后：
conda run -n prism-vq python stage2.py data.universe=csi300 \
  vqvae.quantizer.type=hvq train.seed=0
```

多 seed sweep：`python stage2.py -m data.universe=csi300 vqvae.quantizer.type=hvq train.seed=0,1,2,3,4`

## 单元测试

```bash
conda run -n prism-vq python -m unittest tests.test_hvq -v
```

## 改动点（相对上游 PRISM-VQ）

- 新增 `module/quantise_hvq.py`：`ResidualVectorQuantiser` 与 `create_quantizer`。
- `module/autoencoder.py`、`trainer/train_ypred.py`：量化器实例化改为工厂分支。
- `trainer/train_vqvae.py`：codebook 利用率统计兼容 hvq 的按级 indices list（取第 0 级）。
- `trainer/train_ypred.py::GenerateReturn`：覆写 `train()`，强制 quantizer 保持 eval，
  修复 Lightning 递归 `.train()` 导致已冻结 codebook 在 training mode 下被 `.data`
  改写（dead-code 重初始化）的漏洞。
- `stage1.py` / `stage2.py`：pickle 目录支持 `data.pickle_dir`。
- `configs/config.yaml`：新增 quantizer `type` / `num_levels` / `level_num_embed`、
  `data.pickle_dir`；`stage2_presets` 的 `saved_model` 置空待训练后填写。
- 新增 `tests/test_hvq.py`。
