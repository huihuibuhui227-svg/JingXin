#!/usr/bin/env bash
# JingXin 的运行环境解析 —— 被 tools/*.sh 与 tools/jx_*.py **共用**。
#
# 存在的理由:这些脚本原先写死了本机路径(`~/jingxin`、`~/shared/ans`、
# `~/shared/mp_frames/frames/*.png`、`~/miniconda3/envs/jingxin/bin/python`),
# 换一台机器就全废。现在一律走这里解析,每项都可用环境变量覆盖。
#
# 覆盖方式(例):
#   JX_PY=/usr/bin/python3 JX_RECORDINGS=/data/jx bash tools/start_all.sh
#
# ── 变量 ────────────────────────────────────────────────────────────────
#   JX_PY            本仓解释器(自动探 ~/anaconda3/envs/jingxin 与 ~/miniconda3/envs/jingxin)
#   JX_FRONTEND      前端仓目录(默认探 ~/JingXin/frontend 与 ~/JingXin-frontend)
#   JX_RECORDINGS    录制根(默认 ~/shared/jingxin_recordings)
#   JX_FACE_FRAME    face 验收用的图片(**默认从最近的会话取,不往仓库塞真人帧**)
#   JX_GESTURE_FRAME gesture 验收用的图片(同上)
#   JX_AUDIO_SAMPLE  音频验收用的 wav(同上,取会话的 16k converted wav)
#   JX_AUDIO_SAMPLES 同上,但是**多段**(空格/换行分隔;t7 要 6 段内容不同的回答)
#
# ⚠️ 为什么样本不放进仓库:本仓的原则是**原始媒体不进 git**(见 .gitignore 与
#    docs/下一步.md)。验收脚本要的是"一张真脸/一只手/一段真人语音",
#    那就从**已经录好的会话目录**里取 —— 换机器时带一目录素材即可。

# ── 解释器:两个常见位置都试 ────────────────────────────────────────────
# ⚠️ 此前默认写死 `~/miniconda3/envs/jingxin/bin/python`,而**服务器上那个 env 在
#    `~/anaconda3/`** ⟹ `tools/start_all.sh` 在这台机器上五个服务一个都起不来
#    (命令不存在,`setsid` 起的进程当场死,而端口探测只看"有没有在听")。
#    2026-09-27 实测踩到。这里按 install.sh 的老办法:优先 anaconda3,回退 miniconda3,
#    两个都没有就保持原值(让调用方自己报错,不静默换一个不存在的路径)。
if [ -z "${JX_PY:-}" ]; then
  for __cand in "$HOME/anaconda3/envs/jingxin/bin/python" \
                "$HOME/miniconda3/envs/jingxin/bin/python"; do
    [ -x "$__cand" ] && { JX_PY="$__cand"; break; }
  done
  JX_PY="${JX_PY:-$HOME/miniconda3/envs/jingxin/bin/python}"
  unset __cand
fi

# ── 录制根:先看仓库里那个,再回退老的 ~/shared ────────────────────────
# 服务器上素材落在 `~/JingXin/recordings/`(systemd unit 里也是这么设的);
# 而老笔记本上还在 `~/shared/jingxin_recordings/`。两个都探,别让"路径变了"变成静默空目录。
if [ -z "${JX_RECORDINGS:-}" ]; then
  if [ -d "$HOME/JingXin/recordings" ]; then
    JX_RECORDINGS="$HOME/JingXin/recordings"
  else
    JX_RECORDINGS="$HOME/shared/jingxin_recordings"
  fi
fi
# REPO:优先用调用方已算好的,否则从本文件位置推(本文件在 <repo>/tools/ 下)
if [ -z "${REPO:-}" ]; then
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# ── 前端仓:本仓**不含**前端,它是隔壁一个独立仓 ────────────────────────
# ⚠️ 此前只有 start_all.sh 里写死 `$HOME/JingXin-frontend`,而服务器上前端在
#    `~/JingXin/frontend` ⟹ 那一行 `cd` 失败,前端静默起不来。定义收到这里,
#    与 systemd/install.sh 的 `$HOME/JingXin/frontend` 对齐。
if [ -z "${JX_FRONTEND:-}" ]; then
  for __cand in "$HOME/JingXin/frontend" "$HOME/JingXin-frontend"; do
    [ -d "$__cand" ] && { JX_FRONTEND="$__cand"; break; }
  done
  JX_FRONTEND="${JX_FRONTEND:-$HOME/JingXin/frontend}"
  unset __cand
fi

export REPO JX_PY JX_RECORDINGS JX_FRONTEND

__jx_latest_session() {
  # 取 JX_RECORDINGS 下**名字最新**的会话目录(排除 NONE 与 _试跑)
  ls -1d "$JX_RECORDINGS"/2026* 2>/dev/null | grep -v '/_试跑' | sort | tail -1
}

__jx_sample() {   # __jx_sample <env名> <相对会话目录的 glob>
  local var=$1 glob=$2 cur sid
  eval "cur=\${$var:-}"
  [ -n "$cur" ] && { echo "$cur"; return; }
  # ★ 按**内容**找,不是按目录名取最新 —— 验收脚本自己会铸号并建出一个**空壳**会话目录,
  #   而它永远比真素材新 ⟹ "取最新"会稳定地选到那个空目录(2026-09-27 实测踩到)。
  #   这里从新到旧扫,返回**第一场真的有该素材**的会话。
  while IFS= read -r sid; do
    local hit
    hit="$(ls -1 "$sid"/$glob 2>/dev/null | head -1)"
    [ -n "$hit" ] && { echo "$hit"; return; }
  done < <(ls -1d "$JX_RECORDINGS"/2026* 2>/dev/null | grep -v '/_试跑' | sort -r)
}

JX_FACE_FRAME="${JX_FACE_FRAME:-$(__jx_sample JX_FACE_FRAME 'media/face/*.jpg')}"
JX_GESTURE_FRAME="${JX_GESTURE_FRAME:-$(__jx_sample JX_GESTURE_FRAME 'media/gesture/*.jpg')}"
JX_AUDIO_SAMPLE="${JX_AUDIO_SAMPLE:-$(__jx_sample JX_AUDIO_SAMPLE 'media/audio/*_converted.wav')}"
export JX_FACE_FRAME JX_GESTURE_FRAME JX_AUDIO_SAMPLE

# 多段音频(默认取最新会话里前 N 段**内容各不相同**的真回答)。
# t7 用它验「连接词密度随回答而变」—— 拿同一段重复提交会让 G2 判「本次会话内无变化」。
__jx_audios() {
  local n=${1:-6} sid
  if [ -n "${JX_AUDIO_SAMPLES:-}" ]; then echo "$JX_AUDIO_SAMPLES"; return; fi
  while IFS= read -r sid; do
    local hit
    hit="$(ls -1 "$sid"/media/audio/*_converted.wav 2>/dev/null | head -"$n")"
    [ -n "$hit" ] && { echo "$hit"; return; }
  done < <(ls -1d "$JX_RECORDINGS"/2026* 2>/dev/null | grep -v '/_试跑' | sort -r)
}
JX_AUDIO_SAMPLES="${JX_AUDIO_SAMPLES:-$(__jx_audios 6)}"
export JX_AUDIO_SAMPLES

jx_require() {   # 缺文件就**响亮地**失败,别静默跳过(本仓栽过"静默跳过")
  local var=$1 desc=$2 v
  eval "v=\${$var:-}"
  if [ -z "$v" ] || [ ! -f "$v" ]; then
    echo "✗ $var 没解析到$desc。"
    echo "  取值优先级:环境变量 > $JX_RECORDINGS 下最新会话的素材。"
    echo "  显式指定:  $var=/path/to/file bash $0"
    exit 1
  fi
}

jx_require_list() {   # 多行列表版:每个元素都得是真文件,且**个数要够**
  local var=$1 desc=$2 need=$3 v n missing=0
  eval "v=\${$var:-}"
  n=0
  while IFS= read -r one; do
    [ -z "$one" ] && continue
    n=$((n+1))
    [ -f "$one" ] || { echo "  ✗ $var 第 $n 项不是文件:$one"; missing=1; }
  done <<< "$v"
  if [ "$n" -lt "$need" ] || [ "$missing" = 1 ]; then
    echo "✗ $var 不合格(需要 ≥$need 个真文件,实得 $n$([ $missing = 1 ] && echo ",且有缺失"))。"
    echo "  取值优先级:环境变量 > \$JX_RECORDINGS 下**最近一场有素材**的会话。"
    echo "  显式指定:  $var=\"a.wav b.wav ...\" bash $0"
    exit 1
  fi
}
