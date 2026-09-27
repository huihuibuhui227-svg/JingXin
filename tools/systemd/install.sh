#!/usr/bin/env bash
# 把 JingXin 的五个服务装成 **systemd 用户服务** —— 关机重启后自动起,不占终端。
#
# 照搬这台机器上 FunASR 的做法(`~/.config/systemd/user/funasr-streaming.service`,
# 已连跑 5 天)。为什么不照 start_all.sh 那样用 setsid:那样只防终端关闭,
# **重启机器就没了**;systemd 才是真常驻。
#
# 用法(在服务器上跑):
#     bash tools/systemd/install.sh          # 装 + 起 + 开机自启
#     bash tools/systemd/install.sh --dry    # 只看会写什么,不落盘
#
# 卸载:
#     systemctl --user disable --now jingxin-{voice,face,gesture,panel,frontend}
#     rm ~/.config/systemd/user/jingxin-*.service
#
# ⚠️ 要让服务在**你没登录时也活着**,需要一次 linger(要 sudo):
#     sudo loginctl enable-linger $USER
#   没开 linger 的话,你退出最后一个会话后 systemd --user 会被回收,服务跟着停。
set -eu

DRY=0; [ "${1:-}" = "--dry" ] && DRY=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
PY="${JX_PY:-$HOME/anaconda3/envs/jingxin/bin/python}"
[ -x "$PY" ] || PY="$HOME/miniconda3/envs/jingxin/bin/python"
REC="${JX_RECORDINGS:-$HOME/JingXin/recordings}"
FE="${JX_FRONTEND:-$HOME/JingXin/frontend}"

# ★ 用**所有**本机 IP,不只第一个 —— 多网卡机器上 `awk '{print $1}'` 可能取到
#   Tailscale/VPN 那个,于是真正用来访问的 LAN IP 反而没放行(实测踩到)。
LAN_IPS="$(hostname -I 2>/dev/null || true)"
CORS="http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:5000,http://localhost:5000"
for ip in $LAN_IPS; do
  case "$ip" in *:*) continue;; esac          # 跳过 IPv6
  CORS="$CORS,http://$ip:5173,http://$ip:5000"
done

# Tailscale 域名(经 `tailscale serve` 访问时,页面的 Origin 是
# `https://<主机>.<tailnet>.ts.net:<port>`)。自动探测,不写死 —— 换 tailnet、
# 换主机名都不用改这个脚本。取不到就跳过(不影响本地/隧道那两种用法)。
TS_DNS="$(tailscale status --json 2>/dev/null \
  | python3 -c "import json,sys;print(((json.load(sys.stdin).get('Self') or {}).get('DNSName') or '').rstrip('.'))" 2>/dev/null || true)"
if [ -n "$TS_DNS" ]; then
  CORS="$CORS,https://$TS_DNS:5173,https://$TS_DNS:5000"
  echo "tailscale : $TS_DNS(已加进 CORS)"
fi

# 前端那个 unit 要 node。systemd 的 PATH 很干净,必须显式给。
NODE_BIN="$(command -v node || true)"
if [ -z "$NODE_BIN" ]; then
  for c in "$HOME"/.nvm/versions/node/*/bin/node; do [ -x "$c" ] && NODE_BIN="$c"; done
fi
NODE_DIR="$(dirname "${NODE_BIN:-/usr/bin/node}")"
# ⚠️ 前端要起的是 **npm run dev**,不是 `node run dev` —— `run` 是 npm 的子命令。
#    2026-09-27 实测踩到:用 node 起 ⟹ Node 去找一个叫 `run` 的模块 ⟹ MODULE_NOT_FOUND,
#    服务无限重启(状态停在 activating/auto-restart),端口 5173 从来没起来过。
NPM_BIN="$NODE_DIR/npm"
[ -x "$NPM_BIN" ] || NPM_BIN="$(command -v npm || echo "")"

echo "仓库    : $REPO"
echo "解释器  : $PY"
echo "素材    : $REC"
echo "前端    : $FE"
echo "node    : ${NODE_BIN:-★ 没找到}"
echo "npm     : ${NPM_BIN:-★ 没找到}"
echo "本机 IP  : ${LAN_IPS:-<取不到>}(都加进 CORS 了)"
echo

emit() {  # emit <名字> <描述> <工作目录> <可执行> <参数...>
  local name=$1 desc=$2 wd=$3; shift 3
  cat <<EOF
[Unit]
Description=$desc
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$wd
Environment=PYTHONUNBUFFERED=1
Environment=CORS_ORIGINS=$CORS
# ★ 绑 **127.0.0.1** 而不是 0.0.0.0 —— 经 `tailscale serve` 发布时这是必须的:
#   服务若占了 tailscale IP 上的同一个端口,tailscaled 的 TLS 监听器会
#   `bind: address already in use`(**只在它自己的 journal 里报,静默重试**),
#   于是 https://<host>.ts.net:<port> 永远通不了。2026-09-27 实测踩到。
#   改成 loopback 之后:tailscale serve 转发到 127.0.0.1 ✓,
#   SSH 隧道也打 127.0.0.1 ✓,两条路都成立。
Environment=JX_BIND=127.0.0.1
Environment=PANEL_HOST=127.0.0.1
Environment=JX_RECORDINGS=$REC
Environment=PATH=$NODE_DIR:/usr/local/bin:/usr/bin:/bin
ExecStart=$*
Restart=on-failure
RestartSec=5
TimeoutStopSec=10

[Install]
WantedBy=default.target
EOF
}

write_unit() {
  local name=$1 content=$2
  if [ "$DRY" = "1" ]; then
    echo "── 会写 $UNIT_DIR/$name ──"; echo "$content"; echo
  else
    printf '%s' "$content" > "$UNIT_DIR/$name"
    echo "  ✓ 写好 $name"
  fi
}

# 前端要 node_modules 才能起。没装就**先说清楚**,别装出一个必然重启的服务。
if [ ! -d "$FE/node_modules" ]; then
  echo "⚠️ $FE/node_modules 不存在 —— 前端服务起来会立刻失败。"
  echo "   先跑:  cd $FE && npm install"
  echo "   (以下仍会继续装 unit,但你 npm install 之前它是起不来的)"
  echo
fi

[ "$DRY" = "0" ] && mkdir -p "$UNIT_DIR"

write_unit jingxin-voice.service    "$(emit jingxin-voice    "JingXin 语音服务 (:8001)"     "$REPO" "$PY" -m voice_interaction.api.app)"
write_unit jingxin-face.service     "$(emit jingxin-face     "JingXin 面部服务 (:8000)"     "$REPO" "$PY" -m face_expression.api.app)"
write_unit jingxin-gesture.service  "$(emit jingxin-gesture  "JingXin 手势服务 (:8002)"     "$REPO" "$PY" -m gesture_analysis.api.app)"
write_unit jingxin-panel.service    "$(emit jingxin-panel    "JingXin 报告面板 (:5000)"     "$REPO" "$PY" app.py)"
write_unit jingxin-frontend.service "$(emit jingxin-frontend "JingXin 前端 dev server (:5173)" "$FE" "$NPM_BIN" run dev -- --host 127.0.0.1)"

if [ "$DRY" = "1" ]; then echo "(dry run,未落盘)"; exit 0; fi

echo
echo "── 启用并启动 ──"
# ★ 必须 `enable` + **`restart`**,不能只 `enable --now`。
#   `--now` 对**已经在跑**的服务是 no-op(只设开机自启,不重启进程)⟹
#   改过的 `Environment=` 永远不会被读进去,而 `systemctl status` 照样显示 running ——
#   2026-09-27 实测踩到:改绑 127.0.0.1 之后 `ss -ltn` 里仍是 0.0.0.0,查了半天。
systemctl --user daemon-reload
for s in voice face gesture panel frontend; do
  systemctl --user enable "jingxin-$s.service"
  systemctl --user restart "jingxin-$s.service"
done
sleep 3
echo
systemctl --user --no-pager list-units 'jingxin-*' --all | sed 's/^/  /'
echo
echo
echo "ℹ️ 五个服务现在只绑 127.0.0.1 ⟹ 对外**不需要**放行 8000/8001/8002/5000/5173,"
echo "   走 tailscale serve 或 SSH 隧道即可(两者都转发到本机 loopback)。"
echo "   之前为直连 IP 加的那几条 ufw 规则可以撤:"
echo "     sudo ufw status numbered   # 找到 JingXin 那几条,按编号 sudo ufw delete N"
echo
echo "── 自检:五个端口现在绑在哪(应当全是 127.0.0.1)──"
sleep 3
ss -ltn 2>/dev/null | grep -E ':(5173|5000|8000|8001|8002)\b' | sed 's/^/  /'
if ss -ltn 2>/dev/null | grep -E ':(5173|5000|8000|8001|8002)\b' | grep -q '0.0.0.0'; then
  echo "  ✗ 还有 0.0.0.0 —— 环境变量没生效,检查上面的 restart 有没有报错"
else
  echo "  ✓ 全部绑在 127.0.0.1"
fi
echo
echo "看日志:  journalctl --user -u jingxin-voice -f"
echo "★ 别忘了(要 sudo,只做一次): sudo loginctl enable-linger $USER"
echo "  否则你退出登录后,这些服务会跟着 systemd --user 一起被回收。"
