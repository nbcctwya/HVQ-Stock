# Scripts 使用说明

本目录提供两个基于 `rsync + ssh` 的同步脚本：

```text
rsync_push.sh  # 本地同步到远程
rsync_pull.sh  # 远程同步到本地
```

另有一个实验结果处理脚本：

```text
ensemble_predictions.py  # 将多个 seed 的 stage2 预测分数平均成 ensemble 预测文件
```

它们主要用于 AutoDL 服务器和本地工作区之间同步代码、运行结果和少量必要数据。代码协作仍建议优先使用 Git；`rsync` 更适合传实验结果、checkpoint、临时数据文件。

## 默认配置

两个脚本默认使用同一组远程配置：

```text
REMOTE=autodl-4090d-1
REMOTE_PATH=/root/autodl-tmp/PRISM-VQ
SSH_PORT=from SSH config
```

也就是说，默认依赖本机 `~/.ssh/config` 中的 SSH alias，例如：

```sshconfig
Host autodl-4090d-1
    HostName connect.cqa1.seetacloud.com
    User root
    Port 16352
```

如果要切换服务器，可以直接传 SSH alias：

```bash
scripts/rsync_push.sh autodl-2080ti-1
scripts/rsync_pull.sh autodl-2080ti-1
```

也可以传完整地址和端口：

```bash
scripts/rsync_push.sh root@connect.example.com:12345
scripts/rsync_pull.sh root@connect.example.com:12345
```

## 推送本地代码到服务器

日常推送代码：

```bash
scripts/rsync_push.sh
```

默认不会同步这些大文件或生成目录：

```text
checkpoints/
outputs/
res/
dataset/data/
```

如果要推送到另一台服务器：

```bash
scripts/rsync_push.sh autodl-2080ti-1
```

如果远程路径不是默认路径：

```bash
scripts/rsync_push.sh autodl-2080ti-1 /root/autodl-tmp/PRISM-VQ
```

如果担心覆盖服务器上更新过的文件：

```bash
scripts/rsync_push.sh --skip-newer
```

## 从服务器拉取运行结果

日常拉取服务器运行结果：

```bash
scripts/rsync_pull.sh
```

默认只拉这些目录：

```text
checkpoints/
outputs/
res/
```

这样不会默认把服务器上的代码覆盖回本地。

如果要从另一台服务器拉结果：

```bash
scripts/rsync_pull.sh autodl-2080ti-1
```

如果担心覆盖本地更新过的结果文件：

```bash
scripts/rsync_pull.sh --skip-newer
```

## 拉取服务器代码

如果服务器上也改了代码，需要显式使用 `--code`：

```bash
scripts/rsync_pull.sh --code
```

这个模式会拉取远程代码和配置，但仍默认排除：

```text
checkpoints/
outputs/
res/
dataset/data/
```

如果本地也可能有更新，建议先预览：

```bash
scripts/rsync_pull.sh --code --dry-run
```

再执行：

```bash
scripts/rsync_pull.sh --code --skip-newer
```

## 拉取完整项目

如果确实要拉取远程完整项目：

```bash
scripts/rsync_pull.sh --all
```

这个模式风险更高，可能覆盖本地代码和结果。建议先使用：

```bash
scripts/rsync_pull.sh --all --dry-run
```

## 同步 dataset/data

`dataset/data/` 默认不会被同步，因为其中通常包含生成好的训练 pickle、JKP CSV 或其他本地数据。

如果确实需要拉取远程 `dataset/data/`：

```bash
scripts/rsync_pull.sh --include-data
```

如果需要推送本地 `dataset/data/`、`checkpoints/`、`outputs/` 和 `res/`：

```bash
scripts/rsync_push.sh --include-data
```

注意：`--include-data` 会同步所有默认排除的大目录，不只是 JKP CSV。只同步少量指定文件时，建议直接使用 `rsync` 命令，例如：

```bash
rsync -avh dataset/data/[chn]_[all_themes]_[daily]_[vw_cap].csv \
  autodl-2080ti-1:/root/autodl-tmp/PRISM-VQ/dataset/data/
```

## 常用参数

```text
-n, --dry-run       预览将要同步的文件，不实际修改
--delete            删除目标端多余文件，谨慎使用
--skip-newer        跳过目标端更新时间更晚的文件
--include-data      同步 dataset/data 以及生成结果相关目录
-p, --port PORT     临时指定 SSH 端口
-h, --help          查看帮助
```

`rsync_pull.sh` 额外支持：

```text
--code              从远程拉取代码，而不是只拉结果
--all               从远程拉取完整项目
```

## 环境变量

也可以使用环境变量覆盖默认值：

```bash
REMOTE=autodl-2080ti-1 scripts/rsync_push.sh
REMOTE_PATH=/root/autodl-tmp/PRISM-VQ scripts/rsync_push.sh
SSH_PORT=20785 scripts/rsync_push.sh
```

高级 SSH 或 rsync 参数：

```bash
SSH_OPTS="-o StrictHostKeyChecking=accept-new" scripts/rsync_push.sh
RSYNC_EXTRA_OPTS="--itemize-changes" scripts/rsync_pull.sh --dry-run
```

## 安全建议

- 第一次同步前先使用 `--dry-run`。
- 不确定时不要加 `--delete`。
- 代码长期协作优先使用 Git。
- 运行结果建议放入唯一目录，避免不同服务器或本地实验互相覆盖。
- 同步大文件前确认目标服务器磁盘空间。

## 生成多 seed ensemble 预测

`ensemble_predictions.py` 用于把多个 stage2 `*_best.pkl` 的 `score` 按
`(datetime, instrument)` 对齐后取平均。输出仍然是一个可直接传给
`backtest_qlib.py` 的 pickle 文件。

以 csi300 五个 seed 为例：

```bash
conda run -n prism-vq python scripts/ensemble_predictions.py \
  --run-dir res/VQK512_csi300_mo2_k1_mh64_md0.1_dm64_nh2_l1_d0.1_au0.01_1h2_1emb128_1dl2p10_1l2_p20_ai3_ks3 \
  --seeds 0 1 2 3 4
```

默认输出：

```text
res/.../ensemble_5seed_best.pkl
```

然后可以直接回测：

```bash
conda run -n prism-vq python backtest_qlib.py \
  --universe csi300 \
  --pred_path res/.../ensemble_5seed_best.pkl \
  --output_dir res/.../backtest/ensemble_5seed_top30_drop5
```

也可以显式指定输入文件和输出路径：

```bash
conda run -n prism-vq python scripts/ensemble_predictions.py \
  --pred res/run/0_best.pkl res/run/1_best.pkl res/run/2_best.pkl \
  --output res/run/ensemble_3seed_best.pkl
```
