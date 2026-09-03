#!/usr/bin/env bash
# run_queue.sh — HVQ-Stock 自动科研实验调度器
#
# 流程（每个实验独立循环）：
#   新 Kimi 上下文实现实验 -> Bash 无人值守运行 run.sh -> 新 Kimi 上下文验收
#   -> lab-history 记录同步回 main -> 下一个实验
#
# 用法：
#   bash runner/run_queue.sh                  # 跑整个 enabled 队列
#   bash runner/run_queue.sh --only H002      # 只跑指定实验
#   bash runner/run_queue.sh --retry-failed   # 重试此前失败的实验
#   bash runner/run_queue.sh --dry-run        # 只打印计划，不执行
#
# 调度器不理解任何模型细节；具体实验命令由各实验自己的
# runner/jobs/<ID>/run.sh 负责。AI 长期状态只依赖 Git / spec / manifest /
# state.json / lab-history。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_JSON="$REPO_ROOT/experiments/queue.json"
PROMPTS_DIR="$REPO_ROOT/experiments/prompts"
STATE_JSON="$REPO_ROOT/runner/state.json"
LOG_DIR="$REPO_ROOT/runner/logs"
WORKTREE_BASE="$REPO_ROOT/runner/worktrees"

KIMI_BIN="${KIMI_BIN:-kimi}"
KIMI_FLAGS="${KIMI_FLAGS:---yolo}"   # 非交互自动批准工具调用；可用环境变量覆盖

ONLY_ID=""
RETRY_FAILED=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY_ID="$2"; shift 2 ;;
    --retry-failed) RETRY_FAILED=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------- state ----
init_state() {
  [ -f "$STATE_JSON" ] || echo '{}' > "$STATE_JSON"
  jq empty "$STATE_JSON" 2>/dev/null || echo '{}' > "$STATE_JSON"
}

state_get() {  # state_get <id> -> status 或空
  jq -r --arg id "$1" '.[$id].status // empty' "$STATE_JSON"
}

state_set() {  # state_set <id> <status> [reason]
  local id="$1" status="$2" reason="${3:-null}"
  local tmp; tmp="$(mktemp)"
  jq --arg id "$id" --arg st "$status" --arg rs "$reason" --arg ts "$(date -Is)" '
    .[$id] = ((.[$id] // {}) + {
      status: $st,
      updated_at: $ts,
      reason: (if $rs == "null" then null else $rs end),
      attempts: ((.[$id].attempts // 0) + (if $st == "IMPLEMENTING" or $st == "RUNNING" or $st == "REVIEWING" then 1 else 0 end))
    })' "$STATE_JSON" > "$tmp" && mv "$tmp" "$STATE_JSON"
  log "STATE $id -> $status${reason:+ ($reason)}"
}

# --------------------------------------------------------------- prompt ----
fill_prompt() {  # fill_prompt <template> <out> <id> <name> <spec> <branch> <base_ref> <seeds> <worktree>
  local tpl="$1" out="$2" id="$3" name="$4" spec="$5" branch="$6" base_ref="$7" seeds="$8" wt="$9"
  sed -e "s|{{EXP_ID}}|$id|g" \
      -e "s|{{EXP_NAME}}|$name|g" \
      -e "s|{{SPEC_PATH}}|$spec|g" \
      -e "s|{{BRANCH}}|$branch|g" \
      -e "s|{{BASE_REF}}|$base_ref|g" \
      -e "s|{{SEEDS}}|$seeds|g" \
      -e "s|{{WORKTREE}}|$wt|g" \
      -e "s|{{MAIN_REPO}}|$REPO_ROOT|g" \
      "$tpl" > "$out"
}

# ------------------------------------------------------------- worktree ----
ensure_worktree() {  # ensure_worktree <id> <branch> <base_ref> -> 0/1
  local id="$1" branch="$2" base_ref="$3" wt="$WORKTREE_BASE/$1"
  if [ -d "$wt" ]; then
    return 0
  fi
  if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
    log "复用已有分支 $branch 创建 worktree"
    git -C "$REPO_ROOT" worktree add "$wt" "$branch" >>"$LOG_DIR/worktree.log" 2>&1
  else
    log "从 $base_ref 创建实验分支 $branch"
    git -C "$REPO_ROOT" worktree add "$wt" -b "$branch" "$base_ref" >>"$LOG_DIR/worktree.log" 2>&1
  fi
}

# -------------------------------------------------------------- phases ----
run_kimi() {  # run_kimi <prompt_file> <log_file> <worktree>
  local prompt_file="$1" log_file="$2" wt="$3"
  ( cd "$wt" && "$KIMI_BIN" $KIMI_FLAGS -p "$(cat "$prompt_file")" ) >"$log_file" 2>&1
}

phase_implement() {  # -> 0 成功(READY) / 1 失败
  local id="$1" name="$2" spec="$3" branch="$4" base_ref="$5" seeds="$6" wt="$7"
  local prompt_file="$LOG_DIR/${id}/prompt_implement.md"
  local kimi_log="$LOG_DIR/${id}/implement_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$LOG_DIR/$id"

  ensure_worktree "$id" "$branch" "$base_ref" || { state_set "$id" IMPLEMENT_FAILED "worktree 创建失败"; return 1; }
  fill_prompt "$PROMPTS_DIR/implement.md" "$prompt_file" "$id" "$name" "$spec" "$branch" "$base_ref" "$seeds" "$wt"

  state_set "$id" IMPLEMENTING
  log "[$id] 启动 Implement Kimi（全新上下文，日志: $kimi_log）"
  if ! run_kimi "$prompt_file" "$kimi_log" "$wt"; then
    state_set "$id" IMPLEMENT_FAILED "kimi implement 非零退出，见 $(basename "$kimi_log")"
    return 1
  fi
  if [ ! -f "$wt/runner/jobs/$id/run.sh" ] || [ ! -f "$wt/runner/jobs/$id/manifest.json" ]; then
    state_set "$id" IMPLEMENT_FAILED "缺少 runner/jobs/$id/run.sh 或 manifest.json"
    return 1
  fi
  state_set "$id" READY
}

phase_run() {  # -> 0 成功 / 1 失败
  local id="$1" wt="$2"
  local run_log="$LOG_DIR/${id}/run_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$LOG_DIR/$id"

  state_set "$id" RUNNING
  log "[$id] Bash 无人值守执行 runner/jobs/$id/run.sh（日志: $run_log）"
  if ! ( cd "$wt" && bash "runner/jobs/$id/run.sh" ) >"$run_log" 2>&1; then
    state_set "$id" RUN_FAILED "run.sh 非零退出，见 $(basename "$run_log")"
    return 1
  fi
  log "[$id] 训练/回测完成"
}

phase_review() {  # -> 0 成功(DONE) / 1 失败
  local id="$1" name="$2" spec="$3" branch="$4" base_ref="$5" seeds="$6" wt="$7"
  local prompt_file="$LOG_DIR/${id}/prompt_review.md"
  local kimi_log="$LOG_DIR/${id}/review_$(date +%Y%m%d_%H%M%S).log"
  local review_json="$wt/runner/jobs/$id/review.json"
  mkdir -p "$LOG_DIR/$id"

  fill_prompt "$PROMPTS_DIR/review.md" "$prompt_file" "$id" "$name" "$spec" "$branch" "$base_ref" "$seeds" "$wt"

  state_set "$id" REVIEWING
  log "[$id] 启动 Review Kimi（全新上下文，日志: $kimi_log）"
  if ! run_kimi "$prompt_file" "$kimi_log" "$wt"; then
    state_set "$id" REVIEW_FAILED "kimi review 非零退出，见 $(basename "$kimi_log")"
    return 1
  fi
  if [ ! -f "$review_json" ]; then
    state_set "$id" REVIEW_FAILED "缺少 review.json"
    return 1
  fi

  # 把 lab 记录与机器可读结论同步回 main（只拷小文件，不 merge 实验代码）
  local lab_rel verdict
  lab_rel="$(jq -r '.lab_record // empty' "$review_json")"
  verdict="$(jq -r '.verdict // "unknown"' "$review_json")"
  mkdir -p "$REPO_ROOT/runner/jobs/$id"
  cp "$wt/runner/jobs/$id/manifest.json" "$review_json" "$REPO_ROOT/runner/jobs/$id/" 2>/dev/null || true
  if [ -n "$lab_rel" ] && [ -f "$wt/$lab_rel" ]; then
    cp "$wt/$lab_rel" "$REPO_ROOT/$lab_rel"
    git -C "$REPO_ROOT" add "$lab_rel" "runner/jobs/$id/manifest.json" "runner/jobs/$id/review.json"
    if ! git -C "$REPO_ROOT" diff --cached --quiet; then
      git -C "$REPO_ROOT" commit -m "$id: lab record + review verdict ($verdict)" >>"$LOG_DIR/$id/git_sync.log" 2>&1 \
        || log "[$id] 警告：main 上 commit 失败，见 runner/logs/$id/git_sync.log"
    fi
  else
    log "[$id] 警告：review.json 未给出有效 lab_record，跳过 main 同步"
  fi

  state_set "$id" DONE
  jq --arg id "$id" --arg v "$verdict" --arg ts "$(date -Is)" \
     '.[$id].verdict = $v | .[$id].updated_at = $ts' "$STATE_JSON" > "$STATE_JSON.tmp" \
     && mv "$STATE_JSON.tmp" "$STATE_JSON"
  log "[$id] DONE (verdict=$verdict)"
}

# ---------------------------------------------------------------- main ----
main() {
  init_state
  mkdir -p "$LOG_DIR" "$WORKTREE_BASE"

  local n
  n="$(jq '.experiments | length' "$QUEUE_JSON")"
  log "队列共 $n 个实验（dry_run=$DRY_RUN, retry_failed=$RETRY_FAILED, only=${ONLY_ID:-全部}）"

  local i
  for ((i = 0; i < n; i++)); do
    local exp id name spec base_ref enabled seeds branch wt status
    exp="$(jq ".experiments[$i]" "$QUEUE_JSON")"
    id="$(jq -r '.id' <<<"$exp")"
    name="$(jq -r '.name' <<<"$exp")"
    spec="$(jq -r '.spec' <<<"$exp")"
    base_ref="$(jq -r ".experiments[$i].base_ref // \"main\"" "$QUEUE_JSON")"
    enabled="$(jq -r ".experiments[$i].enabled" "$QUEUE_JSON")"
    seeds="$(jq -c ".experiments[$i].seeds // [0]" "$QUEUE_JSON")"
    branch="exp/${id}-${name}"
    wt="$WORKTREE_BASE/$id"

    [ "$enabled" = "true" ] || { log "[$id] disabled，跳过"; continue; }
    [ -z "$ONLY_ID" ] || [ "$ONLY_ID" = "$id" ] || continue
    if [ ! -f "$REPO_ROOT/$spec" ]; then
      log "[$id] spec 文件不存在: $spec，跳过"
      [ "$DRY_RUN" -eq 1 ] || state_set "$id" IMPLEMENT_FAILED "spec 文件不存在: $spec"
      continue
    fi

    status="$(state_get "$id")"
    status="${status:-PENDING}"

    if [ "$status" = "DONE" ]; then
      log "[$id] 已 DONE，跳过"
      continue
    fi
    if [[ "$status" == *_FAILED ]] && [ "$RETRY_FAILED" -ne 1 ]; then
      log "[$id] 状态 $status，跳过（--retry-failed 可重试）"
      continue
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
      log "[dry-run] [$id] 状态=$status 分支=$branch base=$base_ref seeds=$seeds"
      log "[dry-run] [$id]   worktree=$wt spec=$spec"
      [ "$status" = "READY" ] || log "[dry-run] [$id]   将执行: implement (kimi -p, 全新上下文)"
      log "[dry-run] [$id]   将执行: bash runner/jobs/$id/run.sh（在 worktree 内）"
      log "[dry-run] [$id]   将执行: review (kimi -p, 全新上下文) + lab 记录同步 main"
      continue
    fi

    log "===== [$id] $name（状态: $status）====="

    # Implement 阶段：READY 及以后说明已实现过，直接复用
    case "$status" in
      READY|RUNNING|REVIEWING) : ;;
      *)
        phase_implement "$id" "$name" "$spec" "$branch" "$base_ref" "$seeds" "$wt" || continue
        ;;
    esac

    # Run 阶段（REVIEWING 说明 run 已完成过）
    if [ "$(state_get "$id")" != "REVIEWING" ]; then
      phase_run "$id" "$wt" || continue
    fi

    # Review 阶段
    phase_review "$id" "$name" "$spec" "$branch" "$base_ref" "$seeds" "$wt" || continue
  done

  log "队列处理完毕。状态总览："
  jq -r 'to_entries[] | "  \(.key): \(.value.status)\(if .value.verdict then " (verdict=\(.value.verdict))" else "" end)\(if .value.reason then " -- \(.value.reason)" else "" end)"' "$STATE_JSON"
}

main "$@"
