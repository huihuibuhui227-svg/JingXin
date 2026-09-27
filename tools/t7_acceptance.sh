#!/usr/bin/env bash
# T7 验收跑一次 —— M1 + M1.5 的端到端验收门。
#
# 跑法:  bash tools/t7_acceptance.sh
#
# 它做什么:起三个服务 → /interview/start 铸号 → 提交 6 段回答 → face/gesture 各一帧
#          → 核四条(落盘 / 三份日志对上号 / 原句不进仓库 / 报告)。
#
# 刻意**不** set -e:某一步失败时要继续跑完,把问题一次看全,而不是跑一半就停。
# 服务跑完**不停**,方便你排查;收尾命令在最后打印。

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || { echo "找不到仓库根"; exit 1; }
source "$(dirname "${BASH_SOURCE[0]}")/jx_env.sh"
PY="$JX_PY"
mapfile -t ANSARR <<< "$JX_AUDIO_SAMPLES"
ANS="${ANSARR[0]%/*}"
LOGS=data/logs

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1"; }
head_() { printf '\n\033[1m── %s ──\033[0m\n' "$1"; }

# ── 前置:回答音频齐了没 ────────────────────────────────────────────────
head_ "前置检查"
miss=0
for i in 1 2 3 4 5 6; do
  s="${ANSARR[$((i-1))]:-}"
  [ -n "$s" ] && [ -f "$s" ] || { bad "缺第 $i 段音频:$s"; miss=1; }
done
[ $miss -eq 0 ] && ok "六段回答音频齐全(取自会话素材)" \
  || { echo "样本没凑齐。显式指定:JX_AUDIO_SAMPLES=\"a.wav b.wav ...\" bash $0"; exit 1; }
$PY - "$ANS" <<'PYCHK' || exit 1
import sys, wave, glob
bad = []
for p in sorted(glob.glob(f"{sys.argv[1]}/ans*.wav")):
    with wave.open(p) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 16000):
            bad.append(p)
print("  音频格式全部合规 (1ch/16bit/16000Hz)") if not bad else print("  不合规:", bad)
sys.exit(1 if bad else 0)
PYCHK

# ── 0) 清掉可能残留的占用者(让重跑安全)────────────────────────────────
head_ "清理端口"
for p in 8000 8001 8002; do fuser -k $p/tcp 2>/dev/null; done
sleep 1
ok "8000/8001/8002 已清空"

# ── 1) 起三个服务 ───────────────────────────────────────────────────────
head_ "启动三个服务"
$PY -m voice_interaction.api.app  > /tmp/t7_voice.log   2>&1 &
$PY -m face_expression.api.app    > /tmp/t7_face.log    2>&1 &
$PY -m gesture_analysis.api.app   > /tmp/t7_gesture.log 2>&1 &
$PY - <<'POLL'
import socket, time
need = {8000: "face", 8001: "voice", 8002: "gesture"}; up = set(); t0 = time.time()
while time.time() - t0 < 180 and len(up) < 3:
    for p, n in need.items():
        if n in up: continue
        try:
            socket.create_connection(("127.0.0.1", p), 0.3).close()
            up.add(n); print(f"  ✓ {n:8s} 就绪 {time.time()-t0:5.1f}s")
        except OSError:
            pass
    time.sleep(0.3)
missing = {n for p, n in need.items() if n not in up}
if missing:
    print(f"  ✗ 180s 内没起来: {missing} —— 看 /tmp/t7_*.log")
    raise SystemExit(1)
POLL
[ $? -ne 0 ] && { echo "服务没起全,先看日志再重跑"; exit 1; }

# ── 2) 铸号 ─────────────────────────────────────────────────────────────
head_ "铸号"
SID=$(curl -s -X POST http://127.0.0.1:8001/interview/start \
      | $PY -c "import json,sys;print(json.load(sys.stdin).get('session_id',''))")
if [ -z "$SID" ]; then bad "/interview/start 没给出 session_id"; tail -20 /tmp/t7_voice.log; exit 1; fi
ok "session_id = $SID"
echo "$SID" > /tmp/t7_sid.txt

# ── 3) 六段回答(第 1 段走**表单字段**,其余走 query)──────────────────
head_ "提交六段回答(走 FunASR,每段约 5–15 秒)"
echo "  ans1 ← 表单字段;ans2–6 ← query(样本 ${#ANSARR[@]} 段,取自会话素材)"
curl -s -X POST http://127.0.0.1:8001/interview/answer_audio \
     -F "audio=@${ANSARR[0]}" -F "session_id=$SID" -o /tmp/t7_a1.json
$PY -c "
import json; d=json.load(open('/tmp/t7_a1.json'))
print(f'  ans1  密度={d.get(\"connective_density\")}  sid={d.get(\"session_id\")}')" 2>/dev/null \
  || bad "ans1 返回异常:$(head -c 200 /tmp/t7_a1.json)"
for i in 2 3 4 5 6; do
  curl -s -X POST "http://127.0.0.1:8001/interview/answer_audio?session_id=$SID" \
       -F "audio=@${ANSARR[$((i-1))]}" -o /tmp/t7_a$i.json
  $PY -c "
import json; d=json.load(open('/tmp/t7_a$i.json'))
print(f'  ans$i  密度={d.get(\"connective_density\")}')" 2>/dev/null \
    || bad "ans$i 返回异常:$(head -c 200 /tmp/t7_a$i.json)"
done
echo "  (样本已换成真会话的 6 段回答,故密度值与旧稿子的设计值不再可比 —— 这里只看有没有返回)"

# ── 4) face + gesture 各一帧(face 走表单字段)──────────────────────────
head_ "收帧"
curl -s -o /tmp/t7_face.json -w "  face    HTTP %{http_code}\n" -X POST http://127.0.0.1:8000/analyze \
  -F "file=@$JX_FACE_FRAME" -F "session_id=$SID"
curl -s -o /tmp/t7_gesture.json -w "  gesture HTTP %{http_code}\n" -X POST http://127.0.0.1:8002/analyze \
  -F "file=@$JX_GESTURE_FRAME" -F "session_id=$SID"

# ── 核一:落盘 ──────────────────────────────────────────────────────────
head_ "核一 · 落盘"
if [ -d "$HOME/shared/jingxin_recordings/$SID" ]; then
  ls -l "$HOME/shared/jingxin_recordings/$SID/" | sed 's/^/    /'
  for f in session.json transcript.json; do
    [ -f "$HOME/shared/jingxin_recordings/$SID/$f" ] && ok "$f 在" || bad "$f 缺"
  done
else
  bad "会话目录不存在:$HOME/shared/jingxin_recordings/$SID"
fi

# ── 核二:三份日志对上号 ────────────────────────────────────────────────
head_ "核二 · 三份日志对上号(文件名 + 首列)"
for m in face_au_log interview_emotion_log gesture_emotion_log; do
  f="$LOGS/${m}_${SID}.csv"
  if [ -f "$f" ]; then
    first=$(sed -n '2p' "$f" | cut -d, -f1)
    [ "$first" = "$SID" ] && ok "$(basename $f)  首列=$first" || bad "$(basename $f) 首列=$first(**不是 $SID**)"
  else
    bad "缺 $f"
  fi
done

# ── 核三:原句不进仓库 ──────────────────────────────────────────────────
head_ "核三 · 原句不进仓库"
PHRASE=$($PY - "$SID" <<'PYPHRASE'
import json, sys, os
sid = sys.argv[1]
rec = os.environ.get("JX_RECORDINGS") or os.path.expanduser("~/shared/jingxin_recordings")
p = os.path.join(rec, sid, "transcript.json")
def first_text(o):
    if isinstance(o, dict):
        for k in ("utterances", "segments", "answers", "entries"):
            v = o.get(k)
            if isinstance(v, list) and v:
                for it in v:
                    t = first_text(it)
                    if t: return t
        for k in ("text", "sentence", "content", "raw", "transcript"):
            if isinstance(o.get(k), str) and len(o[k].strip()) > 12: return o[k].strip()
    return None
try:
    t = first_text(json.load(open(p, encoding="utf-8"))) or ""
except Exception:
    t = ""
# 取中段 12 字,避开开头常见词,让 grep 更有区分度
print(t[8:20] if len(t) >= 20 else t[:12])
PYPHRASE
)
if [ -z "$PHRASE" ]; then
  bad "拿不到原句 ⟹ 核三**跑不了**(不是"通过")。看 $JX_RECORDINGS/$SID/transcript.json 是否真的有文本"
  MISS=$((MISS+1))
else
  n=$(grep -rl -- "$PHRASE" "$REPO" 2>/dev/null | grep -v '/\.git/' | wc -l)
  [ "$n" -eq 0 ] && ok "原句「$PHRASE」在仓库里命中 0 个文件" \
                 || { bad "命中 $n 个文件(必须 0):"; grep -rl -- "$PHRASE" "$REPO" 2>/dev/null | grep -v '/\.git/' | sed 's/^/      /'; }
fi

# ── 核四:报告 ──────────────────────────────────────────────────────────
head_ "核四 · 生成报告"
$PY -m report_frontend.report_generator 2>&1 | tail -3 | sed 's/^/    /'
REPORT=$(ls -t data/output/Research_Assessment_Report_*.html 2>/dev/null | head -1)
if [ -z "$REPORT" ]; then
  bad "没生成报告"
else
  ok "报告:$REPORT"
  $PY - "$REPORT" <<'PYREP'
import re, sys
h = open(sys.argv[1], encoding="utf-8").read()
print(f"  「连接词密度」出现 {h.count('连接词密度')} 次")
print(f"  「本场会话：」(I4 的表头披露)出现 {h.count('本场会话：')} 次")
assert "本场会话：" in h, "报告头没有点名本场是哪一场(I4 的披露丢了)"
m = re.search(r"本场会话：.{0,600}?</div>", h, re.S)
if m:
    print("  表头披露原文:", re.sub(r"<[^>]+>", "", m.group(0)).strip()[:220])
for line in re.findall(r"<p[^>]*>[^<]{0,120}</p>", h):
    t = re.sub(r"<[^>]+>", "", line)
    if "评分" in t or "评级" in t:
        print(f"  含「评分/评级」的段落(自己看是不是在**解释**而不是在**给分**): {t.strip()[:110]}")
PYREP
fi

# ── 收尾 ────────────────────────────────────────────────────────────────
head_ "完成"
cat <<EOF
  服务仍在后台跑(方便你排查)。
  停掉:   for p in 8000 8001 8002; do fuser -k \$p/tcp; done
  看日志: /tmp/t7_voice.log  /tmp/t7_face.log  /tmp/t7_gesture.log
  本次 SID: $SID   (存在 /tmp/t7_sid.txt)
EOF
