#!/usr/bin/env bash
# JingXin 夜跑外壳:到点停 + 配额重试 + 日志留痕。
#
# 用法:
#   tools/jx_night_run.sh [结束时刻,默认 06:00] [最大重试次数,默认 3]
#
# 例:
#   tools/jx_night_run.sh 06:00 3
#   JX_PROMPT_FILE=/tmp/my_prompt.md tools/jx_night_run.sh 05:30
#
# 设计取舍(都对应一次真实事故,见 docs/夜跑提示词.md B 节):
#   * 提示词从 docs/夜跑提示词.md 的 **A 节**(第一个 ``` 代码块)抽,
#     单一来源 —— 改提示词只改那一个文件。
#   * 配额 403 用 `-c` 续上一次会话重试,不重开(2026-10-24 04:37 停了两小时二十三分钟)。
#   * 到点停只负责"停",**收口(提交/推 git)由会话内自己做** ——
#     只有它知道这轮改了什么、门过没过。
#   * 本脚本**不做任何 git 操作**。
set -uo pipefail

STOP_AT="${1:-06:00}"
MAX_RETRY="${2:-3}"
REPO="${JX_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${JX_LOG_DIR:-$REPO/logs}"
PROMPT_FILE="${JX_PROMPT_FILE:-$REPO/docs/夜跑提示词.md}"

mkdir -p "$LOG_DIR"

# 从提示词文档里抽出 A 节(第一个 ``` 代码块)作为实际提示词。
if [ ! -f "$PROMPT_FILE" ]; then
  echo "❌ 找不到提示词文件:$PROMPT_FILE" >&2
  exit 1
fi
PROMPT="$(awk '/^```/{n++; next} n==1' "$PROMPT_FILE")"
if [ -z "$PROMPT" ]; then
  echo "❌ 没从 $PROMPT_FILE 抽出提示词(A 节应为第一个代码块)" >&2
  exit 1
fi

LOG="$LOG_DIR/night_run_$(date +%Y%m%d).log"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

# 到点了吗?跨零点也算(例如现在 23:40、结束时刻 06:00)。
past_stop() {
  local now stop
  now=$(date +%H%M)
  stop="${STOP_AT//:/}"
  case "$stop" in
    ''|*[!0-9]*) say "⚠️ 结束时刻格式不对:$STOP_AT(应为 HH:MM),按 06:00 处理"; stop=0600 ;;
  esac
  # 去掉前导 0 再比较,避免 08xx 被当成八进制。
  now=$((10#$now)); stop=$((10#$stop))
  if [ "$stop" -lt 1200 ]; then          # 结束时刻在凌晨
    [ "$now" -ge "$stop" ] && [ "$now" -lt 1200 ]
  else                                    # 结束时刻在白天/晚上
    [ "$now" -ge "$stop" ]
  fi
}

say "夜跑开始 · 结束时刻 $STOP_AT · 最多重试 $MAX_RETRY 次"
say "提示词:$PROMPT_FILE"
say "日志:$LOG"

attempt=0
SESSION_ARGS=()
while :; do
  if past_stop; then
    say "⏰ 已到 $STOP_AT,停止。"
    exit 0
  fi

  say "第 $((attempt + 1)) 轮启动${SESSION_ARGS[*]:+ (续上一次会话)}"
  out="$(claude --permission-mode auto "${SESSION_ARGS[@]}" -p "$PROMPT" 2>&1)"
  rc=$?
  printf '%s\n' "$out" | tee -a "$LOG"

  # 配额不足:等一会儿再续,不重开。
  if printf '%s' "$out" | grep -q 'Subscription quota is insufficient'; then
    attempt=$((attempt + 1))
    if [ "$attempt" -ge "$MAX_RETRY" ]; then
      say "❌ 配额重试 $MAX_RETRY 次仍失败,退出。"
      exit 2
    fi
    wait_min=$((attempt * 10))
    say "⚠️ 配额不足,${wait_min} 分钟后重试(第 $attempt 次)。"
    sleep $((wait_min * 60))
    SESSION_ARGS=(-c)          # 续上最近一次会话,不丢上下文
    continue
  fi

  if [ "$rc" -eq 0 ]; then
    say "✅ 本轮正常结束(rc=0)。"
    exit 0
  fi

  # 其他错误:不盲目重试,留痕退出让人看。
  say "❌ 异常退出 rc=$rc,见日志。"
  exit "$rc"
done
