#!/usr/bin/env bash
# smoke_test_scheduler.sh — 调度器端到端 smoke 测试（不触发任何真实训练/AI 调用）
#
# 做法：把仓库克隆到临时目录，用 stub 版 `kimi`（shell 脚本模拟
# Implement/Review Agent 的交付行为）跑通完整队列流程，验证：
#   worktree 创建 -> implement -> run.sh 执行 -> review -> lab 记录同步 main
#   -> 状态机 DONE -> 重启后 DONE 跳过
#
# 用法：bash runner/tests/smoke_test_scheduler.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "[SMOKE-FAIL] $*" >&2; exit 1; }
ok()   { echo "[SMOKE-OK] $*"; }

# --- 1. 克隆仓库到临时目录 ---
git clone -q "$REPO_ROOT" "$TMP/repo"
cd "$TMP/repo"
git config user.email smoke@test && git config user.name smoke
ok "仓库已克隆到 $TMP/repo"

# --- 2. 造一个假的 smoke 实验 ---
mkdir -p experiments/specs
cat > experiments/specs/H999-smoke.md <<'EOF'
# H999 — smoke test
调度器自检测试，不做真实训练。
EOF
cat > experiments/queue.json <<'EOF'
{
  "version": 1,
  "experiments": [
    {"id": "H999", "name": "smoke", "spec": "experiments/specs/H999-smoke.md",
     "base_ref": "main", "enabled": true, "seeds": [0]}
  ]
}
EOF
git add -A && git commit -qm "smoke: fake experiment"

# --- 3. stub kimi：模拟 Implement / Review Agent ---
mkdir -p "$TMP/bin"
cat > "$TMP/bin/kimi" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
prompt="${@: -1}"
id="$(grep -oP '实验 ID：`\K[^`]+' <<<"$prompt" | head -1)"
[ -n "$id" ] || { echo "stub: 无法从 prompt 解析实验 ID" >&2; exit 1; }

if grep -q 'Implement Agent' <<<"$prompt"; then
  branch="$(git branch --show-current)"
  mkdir -p "runner/jobs/$id"
  cat > "runner/jobs/$id/run.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
cd "\$(dirname "\${BASH_SOURCE[0]}")/../../.."
mkdir -p results/$id
echo '{"IC": 0.01}' > results/$id/metrics.json
EOF2
  echo '{}' > "runner/jobs/$id/manifest.json"
  git add -A && git commit -qm "$id: implement"
  jq -n --arg id "$id" --arg br "$branch" --arg c "$(git rev-parse HEAD)" \
    '{experiment_id:$id, branch:$br, implementation_commit:$c, base_ref:"main",
      requires_stage1:false, stage1_checkpoint:null, seeds:[0],
      result_paths:{metrics:("results/"+$id+"/metrics.json")}, log_paths:[]}' \
    > "runner/jobs/$id/manifest.json"
  git add -A && git commit -qm "$id: manifest"
  echo "stub implement done"
elif grep -q 'Review Agent' <<<"$prompt"; then
  rec="lab-history/0999-smoke-csi300-seed0.md"
  echo "# 0999 smoke 验收记录" > "$rec"
  jq -n --arg id "$id" --arg rec "$rec" \
    '{experiment_id:$id, verdict:"success", failure_reason:null,
      lab_record:$rec, metrics:{test:{IC:0.01}}, anomalies:[], reviewed_at:"now"}' \
    > "runner/jobs/$id/review.json"
  git add -A && git commit -qm "$id: review"
  echo "stub review done"
else
  echo "stub: 未知 prompt 类型" >&2; exit 1
fi
STUB
chmod +x "$TMP/bin/kimi"
export PATH="$TMP/bin:$PATH"
export KIMI_BIN=kimi
ok "stub kimi 就绪"

# --- 4. 首次运行：完整走通 ---
bash runner/run_queue.sh > "$TMP/run1.log" 2>&1 || { cat "$TMP/run1.log"; fail "首次 run_queue 非零退出"; }

status="$(jq -r '.H999.status' runner/state.json)"
[ "$status" = "DONE" ] || fail "期望 DONE，实际 $status"
ok "状态机: H999 -> DONE"

[ -f lab-history/0999-smoke-csi300-seed0.md ] || fail "lab 记录未同步到 main"
ok "lab 记录已同步 main"

[ -f runner/jobs/H999/review.json ] || fail "review.json 未同步到 main"
ok "review.json 已同步 main"

git show-ref --verify --quiet refs/heads/exp/H999-smoke || fail "实验分支未创建"
ok "实验分支 exp/H999-smoke 存在"

[ -f runner/worktrees/H999/results/H999/metrics.json ] || fail "run.sh 产物不存在"
ok "run.sh 在 worktree 内执行并产出结果"

git log --oneline main | grep -q "H999: lab record" || fail "main 上没有验收同步 commit"
ok "main 上存在验收同步 commit"

# --- 5. 重启续跑：DONE 必须跳过 ---
bash runner/run_queue.sh > "$TMP/run2.log" 2>&1 || fail "二次运行非零退出"
grep -q "已 DONE，跳过" "$TMP/run2.log" || fail "二次运行未跳过 DONE 实验"
ok "重启续跑: DONE 实验被跳过"

# --- 6. dry-run 不执行 ---
cat > experiments/queue.json <<'EOF'
{
  "version": 1,
  "experiments": [
    {"id": "H998", "name": "dry", "spec": "experiments/specs/H999-smoke.md",
     "base_ref": "main", "enabled": true, "seeds": [0]}
  ]
}
EOF
bash runner/run_queue.sh --dry-run > "$TMP/run3.log" 2>&1 || fail "dry-run 非零退出"
grep -q "\[dry-run\] \[H998\]" "$TMP/run3.log" || fail "dry-run 未列出 H998"
[ ! -d runner/worktrees/H998 ] || fail "dry-run 不应创建 worktree"
ok "dry-run 只打印计划不执行"

echo
echo "== SMOKE TEST PASSED =="
