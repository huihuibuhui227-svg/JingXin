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

# ── 绑定与 CORS ───────────────────────────────────────────────────────
# 默认只绑本机(安全)。要**让别的电脑访问**,设 JX_BIND=0.0.0.0:
#   JX_BIND=0.0.0.0 bash tools/start_all.sh
# 设了之后这个脚本会一并做三件事,少任何一件远端浏览器都连不上:
#   ① 面板绑 0.0.0.0   ② 前端 dev server 加 --host   ③ 把本机 IP 的 Origin
#   加进四个服务的 CORS 白名单(它们都读 CORS_ORIGINS;默认只放行 localhost)
JX_BIND="${JX_BIND:-127.0.0.1}"
CORS_DEFAULT="http://127.0.0.1:5000,http://localhost:5000,http://localhost:5173,http://127.0.0.1:5173"
if [ "$JX_BIND" = "0.0.0.0" ]; then
  # ★ 用**所有**本机 IP,不只第一个 —— 多网卡机器上取第一个可能拿到 Tailscale/VPN,
  #   真正访问用的 LAN IP 反而没放行(实测踩到)。
  LAN_IP=""
  for ip in $(hostname -I 2>/dev/null || true); do
    case "$ip" in *:*) continue;; esac
    LAN_IP="${LAN_IP:+$LAN_IP,}http://$ip:5173,http://$ip:5000"
  done
  # Tailscale 域名(经 `tailscale serve` 访问时,页面的 Origin 是
# `https://<主机>.<tailnet>.ts.net:<port>`)。自动探测,不写死 —— 换 tailnet、
# 换主机名都不用改这个脚本。取不到就跳过(不影响本地/隧道那两种用法)。
TS_DNS="$(tailscale status --json 2>/dev/null \
  | python3 -c "import json,sys;print(((json.load(sys.stdin).get('Self') or {}).get('DNSName') or '').rstrip('.'))" 2>/dev/null || true)"
  [ -n "$TS_DNS" ] && LAN_IP="${LAN_IP:+$LAN_IP,}https://$TS_DNS:5173,https://$TS_DNS:5000"
  if [ -n "$LAN_IP" ]; then
    export CORS_ORIGINS="${CORS_ORIGINS:-$CORS_DEFAULT},$LAN_IP"
    export PANEL_HOST=0.0.0.0
    FE_ARGS=(-- --host 0.0.0.0)
    echo "  绑定 0.0.0.0 ⟹ CORS 已放行: $LAN_IP"
  else
    echo "  ⚠ JX_BIND=0.0.0.0 但取不到本机 IP(hostname -I 为空),CORS 只放行 localhost"
    export PANEL_HOST=0.0.0.0
    FE_ARGS=()
  fi
else
  export CORS_ORIGINS="${CORS_ORIGINS:-$CORS_DEFAULT}"
  export PANEL_HOST="${PANEL_HOST:-127.0.0.1}"
  FE_ARGS=()
fi

echo "── 启动 ──"
start 8001 "voice   " "$REPO" $PY -m voice_interaction.api.app
start 8000 "face    " "$REPO" $PY -m face_expression.api.app
start 8002 "gesture " "$REPO" $PY -m gesture_analysis.api.app
start 5000 "面板    " "$REPO" $PY app.py
start 5173 "前端    " "${JX_FRONTEND:-$HOME/JingXin-frontend}" npm run dev "${FE_ARGS[@]:-}"

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
