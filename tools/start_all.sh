#!/usr/bin/env bash
# 一键起全套:三个分析服务 + Flask 面板 + 前端 dev server。
#
# 跑法:  bash tools/start_all.sh
#
# 幂等:已经在听的端口会跳过,不会重复起。
# 起完在**Windows 的浏览器**里打开 http://localhost:5173
#   —— WSL2 会把 localhost 转发给 Windows,而在 Windows 浏览器里摄像头才拿得到。
#      (WSL 里的浏览器拿不到摄像头)

set -u
source "$(dirname "${BASH_SOURCE[0]}")/jx_env.sh"
PY="$JX_PY"
UP=()   ; SKIP=()

up() { (echo > /dev/tcp/127.0.0.1/$1) 2>/dev/null; }

start() {  # start <port> <desc> <cwd> <cmd...>
  local port=$1 desc=$2 cwd=$3; shift 3
  if up "$port"; then SKIP+=("$desc(:$port)"); return; fi
  # ★ 必须 `setsid`:否则这些服务留在**调用方的进程组**里 —— 终端一 Ctrl-Z、
  #   或被父进程清理时,它们会变成 T 态(端口在听却不回话,看着像崩溃)甚至直接死。
  #   2026-09-27 实测:不带 setsid 时,起完的五个服务在父 shell 退出后全部消失。
  ( cd "$cwd" && setsid "$@" > "/tmp/jx_$port.log" 2>&1 < /dev/null & )
  UP+=("$desc(:$port)")
}

echo "── 启动 ──"
start 8001 "voice   " "$REPO" $PY -m voice_interaction.api.app
start 8000 "face    " "$REPO" $PY -m face_expression.api.app
start 8002 "gesture " "$REPO" $PY -m gesture_analysis.api.app
start 5000 "面板    " "$REPO" $PY app.py
start 5173 "前端    " "${JX_FRONTEND:-$HOME/JingXin-frontend}" npm run dev

for s in "${SKIP[@]:-}"; do [ -n "$s" ] && echo "  已在跑,跳过: $s"; done
for s in "${UP[@]:-}";   do [ -n "$s" ] && echo "  已启动:       $s"; done

echo
echo "── 等就绪(gesture 要 ~11s:加载两个模型)──"
$PY - <<'POLL'
import socket, time
need = {8000:"face", 8001:"voice", 8002:"gesture", 5000:"面板", 5173:"前端"}
up, t0 = set(), time.time()
while time.time()-t0 < 180 and len(up) < len(need):
    for p, n in need.items():
        if n in up: continue
        try:
            socket.create_connection(("127.0.0.1", p), 0.3).close()
            up.add(n); print(f"  ✓ {n:8s} :{p}")
        except OSError: pass
    time.sleep(0.3)
bad = {n for p, n in need.items() if n not in up}
if bad: print(f"  ✗ 没起来: {bad} —— 看 /tmp/jx_<端口>.log")
POLL

cat <<'EOF'

── 打开这个(用 Windows 的浏览器,不是 WSL 里的)──
     http://localhost:5173
   WSL2 会把 localhost 转发给 Windows。摄像头必须在 Windows 浏览器里才有。

── 各服务的日志 ──
     /tmp/jx_8000.log  face      /tmp/jx_8001.log  voice
     /tmp/jx_8002.log  gesture   /tmp/jx_5000.log  面板
     /tmp/jx_5173.log  前端

── 全停 ──
     for p in 8000 8001 8002 5000 5173; do fuser -k $p/tcp; done
EOF
