#!/usr/bin/env bash
# 从**要录制的那台电脑**开一条 SSH 隧道到服务器,然后浏览器用 http://localhost:5173 访问。
#
# ── 为什么必须这样,不能直接开 http://<服务器IP>:5173 ─────────────────────
# 浏览器规定:getUserMedia / navigator.mediaDevices **只在「安全上下文」里存在**,
# 也就是 https:// 或 http://localhost。http://<IP> **两条都不占** ⟹
# `navigator.mediaDevices === undefined` ⟹ 摄像头和麦克风**在浏览器层面就不存在**。
#
# 2026-09-27 在同一台机器、同一个页面上实测(仅地址不同):
#     http://192.168.72.30:5173   isSecureContext=False  mediaDevices=undefined
#     http://localhost:15173      isSecureContext=True   mediaDevices=object
#
# 这不是配置问题,没有开关可开。三条路:① 上 HTTPS(自签证书 + 五个服务全得 HTTPS,
# 否则页面是 https 而接口是 http,混合内容会被浏览器拦)② 隧道(本脚本,零证书)
# ③ 把浏览器跑在服务器上(可服务器没摄像头)。
#
# 隧道还顺带解决两件:页面 host 推出来是 localhost ⟹ 后端也走隧道;
# 而 `http://localhost:5173` 本来就在四个服务的 CORS 默认白名单里。
#
# 用法:
#     bash tools/tunnel_to_server.sh                 # 前台跑,Ctrl-C 断
#     JX_SERVER=zgy@192.168.72.30 bash tools/tunnel_to_server.sh
#
# ⚠️ 本地这五个端口必须空着 —— 如果你本机也跑着 JingXin,先停掉:
#     for p in 8000 8001 8002 5000 5173; do fuser -k $p/tcp; done
set -eu
SRV="${JX_SERVER:-zgy@192.168.72.30}"
PORTS=(5173 5000 8000 8001 8002)

for p in "${PORTS[@]}"; do
  if (echo > "/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
    echo "✗ 本地 $p 已被占用 —— 先停掉本机的 JingXin:"
    echo "    for p in 8000 8001 8002 5000 5173; do fuser -k \$p/tcp; done"
    exit 1
  fi
done

ARGS=(); for p in "${PORTS[@]}"; do ARGS+=(-L "$p:localhost:$p"); done

echo "隧道:localhost:{$(IFS=,; echo "${PORTS[*]}")} → $SRV"
echo "开好之后浏览器打开:  http://localhost:5173"
echo "(Ctrl-C 断开)"
exec ssh -N -o ExitOnForwardFailure=yes "${ARGS[@]}" "$SRV"
