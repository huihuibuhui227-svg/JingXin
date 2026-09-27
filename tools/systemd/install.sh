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

# 前端那个 unit 要 node。systemd 的 PATH 很干净,必须显式给。
NODE_BIN="$(command -v node || true)"
if [ -z "$NODE_BIN" ]; then
  for c in "$HOME"/.nvm/versions/node/*/bin/node; do [ -x "$c" ] && NODE_BIN="$c"; done
fi
NODE_DIR="$(dirname "${NODE_BIN:-/usr/bin/node}")"

echo "仓库    : $REPO"
echo "解释器  : $PY"
echo "素材    : $REC"
echo "前端    : $FE"
echo "node    : ${NODE_BIN:-★ 没找到}"
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
Environment=PANEL_HOST=0.0.0.0
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

[ "$DRY" = "0" ] && mkdir -p "$UNIT_DIR"

write_unit jingxin-voice.service    "$(emit jingxin-voice    "JingXin 语音服务 (:8001)"     "$REPO" "$PY" -m voice_interaction.api.app)"
write_unit jingxin-face.service     "$(emit jingxin-face     "JingXin 面部服务 (:8000)"     "$REPO" "$PY" -m face_expression.api.app)"
write_unit jingxin-gesture.service  "$(emit jingxin-gesture  "JingXin 手势服务 (:8002)"     "$REPO" "$PY" -m gesture_analysis.api.app)"
write_unit jingxin-panel.service    "$(emit jingxin-panel    "JingXin 报告面板 (:5000)"     "$REPO" "$PY" app.py)"
write_unit jingxin-frontend.service "$(emit jingxin-frontend "JingXin 前端 dev server (:5173)" "$FE" "$NODE_BIN" run dev -- --host 0.0.0.0)"

if [ "$DRY" = "1" ]; then echo "(dry run,未落盘)"; exit 0; fi

echo
echo "── 启用并启动 ──"
for s in voice face gesture panel frontend; do
  systemctl --user daemon-reload
  systemctl --user enable --now "jingxin-$s.service"
done
sleep 3
echo
systemctl --user --no-pager list-units 'jingxin-*' --all | sed 's/^/  /'
echo
echo "看日志:  journalctl --user -u jingxin-voice -f"
echo "★ 别忘了(要 sudo,只做一次): sudo loginctl enable-linger $USER"
echo "  否则你退出登录后,这些服务会跟着 systemd --user 一起被回收。"
