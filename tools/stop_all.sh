#!/usr/bin/env bash
# 一键优雅停止《衡才智析》五个核心服务（face/voice/gesture/panel/frontend）。
# 若由 systemd --user 托管则优先通过 systemctl 停止，避免自动拉起；否则按端口安全发送 SIGTERM。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jx_env.sh
source "$SCRIPT_DIR/jx_env.sh"

UNITS=(jingxin-frontend jingxin-panel jingxin-gesture jingxin-voice jingxin-face)
PORTS=(5173 5000 8002 8001 8000)

echo "── 停止《衡才智析》服务 ──"

if command -v systemctl >/dev/null 2>&1; then
  for unit in "${UNITS[@]}"; do
    if systemctl --user is-active --quiet "$unit" 2>/dev/null; then
      echo "  停止 systemd 用户服务: $unit"
      systemctl --user stop "$unit"
    fi
  done
fi

for port in "${PORTS[@]}"; do
  if command -v fuser >/dev/null 2>&1 && fuser "$port/tcp" >/dev/null 2>&1; then
    echo "  停止端口 :$port 监听进程"
    fuser -k -TERM "$port/tcp" >/dev/null 2>&1 || true
  fi
done

sleep 1

STILL_UP=()
for port in "${PORTS[@]}"; do
  if (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    STILL_UP+=(":$port")
  fi
done

if [ "${#STILL_UP[@]}" -eq 0 ]; then
  echo "  ✓ 全部 5 个服务端口已释放"
else
  echo "  ⚠ 仍有端口监听: ${STILL_UP[*]}（可按需人工检查）"
fi
