# Lab History

本目录记录 HVQ-Stock 每次实验的完整档案。每次实验一个文件，命名规则：

```text
NNNN-简短描述-市场-seed.md   # 例如 0001-hvq-residual-2level-csi300-seed0.md
```

## 每条记录必须包含的字段

- **Git 版本**：训练时代码对应的 commit hash（`git rev-parse HEAD`；若训练后有未入库改动需注明）
- **创新点 / 魔改点**：相对上游 PRISM-VQ 或上一次实验的改动
- **关键配置**：quantizer 类型、codebook 规模、universe、seed、epoch 等
- **训练耗时**：Stage 1 / Stage 2 各自的墙钟时间（单 seed）
- **预测指标**：test 集的 IC、ICIR、RankIC、RankICIR
- **回测指标**：年化收益、Sharpe、MDD、累计收益、换手率等（注明回测口径）
- **产物路径**：checkpoint、预测 pkl、回测目录、日志

新建记录时复制 `TEMPLATE.md`。
