#!/usr/bin/env bash
# preflight.sh — 启动实验队列前的环境自检。只检查，不修改任何系统配置。
#
# 用法：bash runner/preflight.sh
# 退出码：0 = 全部关键检查通过；1 = 存在失败项（警告不阻塞）。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-prism-vq}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/../PRISM-VQ/dataset/data/CN}"

PASS=0; FAIL=0; WARN=0
ok()   { echo "  [OK]   $*"; PASS=$((PASS+1)); }
fail() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $*"; WARN=$((WARN+1)); }
info() { echo "  [INFO] $*"; }

echo "== preflight @ $REPO_ROOT =="

# --- 操作系统 ---
echo "-- 操作系统"
if [ "$(uname -s)" = "Linux" ]; then
  ok "Linux"
  grep -qi microsoft /proc/version 2>/dev/null && info "检测到 WSL 环境（符合预期）"
else
  fail "非 Linux 环境: $(uname -s)"
fi

# --- 基础工具 ---
echo "-- 基础工具"
for tool in git kimi jq conda; do
  if command -v "$tool" >/dev/null 2>&1; then ok "$tool -> $(command -v "$tool")"
  else fail "$tool 不在 PATH"; fi
done

# --- conda 环境与 Python / CUDA ---
echo "-- conda 环境: $CONDA_ENV"
if command -v conda >/dev/null 2>&1; then
  if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    ok "conda env '$CONDA_ENV' 存在"
    if conda run -n "$CONDA_ENV" python --version >/dev/null 2>&1; then
      ok "python 可运行: $(conda run -n "$CONDA_ENV" python --version 2>&1)"
    else
      fail "conda env '$CONDA_ENV' 内 python 不可运行"
    fi
    if conda run -n "$CONDA_ENV" python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      ok "PyTorch CUDA 可用: $(conda run -n "$CONDA_ENV" python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
    else
      fail "PyTorch CUDA 不可用（env: $CONDA_ENV）"
    fi
    if conda run -n "$CONDA_ENV" python -c "import qlib" 2>/dev/null; then
      ok "qlib 可导入"
    else
      fail "qlib 不可导入（env: $CONDA_ENV）"
    fi
  else
    fail "conda env '$CONDA_ENV' 不存在"
  fi
fi

# --- 数据路径 ---
echo "-- 数据"
if [ -d "$DATA_DIR" ]; then
  ok "预处理数据目录存在: $DATA_DIR"
  for f in csi300_20_h10_dl2_train.pkl csi300_20_h10_dl2_valid.pkl csi300_20_h10_dl2_test.pkl; do
    [ -f "$DATA_DIR/$f" ] && ok "  $f" || fail "  缺少 $f"
  done
else
  fail "预处理数据目录不存在: $DATA_DIR"
fi
if [ -d "$HOME/.qlib/qlib_data/cn_data" ]; then
  ok "Qlib cn_data 存在: $HOME/.qlib/qlib_data/cn_data"
else
  fail "Qlib cn_data 不存在: $HOME/.qlib/qlib_data/cn_data"
fi

# --- Git 状态 ---
echo "-- Git"
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  ok "git 仓库正常（分支: $(git -C "$REPO_ROOT" branch --show-current)）"
  dirty="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no)"
  if [ -z "$dirty" ]; then
    ok "无未提交的已跟踪文件修改"
  else
    fail "存在未提交的已跟踪文件修改（先 commit 或 stash）："
    echo "$dirty" | sed 's/^/         /'
  fi
  untracked="$(git -C "$REPO_ROOT" status --porcelain | grep -c '^??' || true)"
  [ "$untracked" -gt 0 ] && warn "$untracked 个未跟踪文件（不阻塞，但注意别误提交大文件）"
else
  fail "$REPO_ROOT 不是 git 仓库"
fi

# --- 磁盘 ---
echo "-- 磁盘"
avail_gb="$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')"
if [ "${avail_gb:-0}" -ge "$MIN_FREE_GB" ]; then
  ok "可用磁盘 ${avail_gb}GB >= ${MIN_FREE_GB}GB"
else
  fail "可用磁盘 ${avail_gb:-0}GB < ${MIN_FREE_GB}GB"
fi

# --- 队列配置 ---
echo "-- 队列配置"
if command -v jq >/dev/null 2>&1 && [ -f "$REPO_ROOT/experiments/queue.json" ]; then
  if jq empty "$REPO_ROOT/experiments/queue.json" 2>/dev/null; then
    ok "queue.json 是合法 JSON"
    jq -r '.experiments[] | "         \(.id) enabled=\(.enabled) seeds=\(.seeds|tojson) spec=\(.spec)"' \
      "$REPO_ROOT/experiments/queue.json" | while read -r line; do info "$line"; done
    # screening 阶段保护：任何实验 seeds 含非 0 值即报错
    bad="$(jq '[.experiments[] | select(.enabled) | .seeds[] | select(. != 0)] | length' \
      "$REPO_ROOT/experiments/queue.json")"
    if [ "$bad" -gt 0 ]; then
      fail "Idea Screening 阶段只允许 seed 0，但 queue.json 中出现了其他 seed"
    else
      ok "所有 enabled 实验 seeds 均为 [0]"
    fi
  else
    fail "queue.json 不是合法 JSON"
  fi
fi

echo
echo "== 结果: $PASS 通过 / $WARN 警告 / $FAIL 失败 =="
[ "$FAIL" -eq 0 ]
