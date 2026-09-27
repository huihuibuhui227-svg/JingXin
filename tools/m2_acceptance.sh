#!/usr/bin/env bash
# M2 端到端验收 —— 按**前端的调用顺序**模拟一场真会话。
#
# 跑法:  bash tools/m2_acceptance.sh
#
# 它验的是 M2 的成功定义:一场会话 → 三份日志**文件名与首列都等于服务端铸的号**
#   → 报告覆盖 **不是 0**、且报告头点名的那一场就是这个号。
#
# 为什么按前端的顺序而不是随便发:前端原来自己造 UUID、且 answer_audio 不带 id,
# 这两条都会让日志落错地方 —— 本脚本复刻"修好之后"的顺序,所以它绿 = 那条链通了。

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
source "$(dirname "${BASH_SOURCE[0]}")/jx_env.sh"
PY="$JX_PY"

ok()  { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad() { printf '  \033[31m✗\033[0m %s\n' "$1"; }
h()   { printf '\n\033[1m── %s ──\033[0m\n' "$1"; }

jx_require JX_FACE_FRAME "验收用的面部帧"; jx_require JX_GESTURE_FRAME "验收用的手势帧"

h "起服务"
bash tools/start_all.sh 2>&1 | grep -E "✓|✗|已在跑" | sed 's/^/  /'

h "① 铸号(前端 interview.start 的那一步)"
SID=$(curl -s -X POST http://127.0.0.1:8001/interview/start \
      | $PY -c "import json,sys;print(json.load(sys.stdin).get('session_id',''))")
[ -z "$SID" ] && { bad "没拿到铸号"; exit 1; }
ok "服务端铸号 = $SID"
case "$SID" in *-*-*-*) bad "拿到的是 UUID 形态 —— 铸号没生效"; exit 1;; esac

h "② 五段回答(带同一个号;前端重修后就是这个形状)"
jx_require_list JX_AUDIO_SAMPLES "验收用的音频" 5
mapfile -t ANSARR <<< "$JX_AUDIO_SAMPLES"
echo "  样本 ${#ANSARR[@]} 段(取自会话素材;同一段重复提交会让 G2 判「无变化」)"
for i in 1 2 3 4 5; do
  curl -s -X POST "http://127.0.0.1:8001/interview/answer_audio?session_id=$SID" \
       -F "audio=@${ANSARR[$((i-1))]}" -o /tmp/m2_a$i.json
  $PY -c "
import json;d=json.load(open('/tmp/m2_a$i.json'))
print(f'  ans$i  密度={d.get(\"connective_density\")}  回传 sid={d.get(\"session_id\")}')" 2>/dev/null \
   || bad "ans$i 异常:$(head -c 150 /tmp/m2_a$i.json)"
done

h "③ 收帧(带同一个号)"
curl -s -o /dev/null -w "  face    HTTP %{http_code}\n" -X POST \
  "http://127.0.0.1:8000/analyze?session_id=$SID" \
  -F "file=@$JX_FACE_FRAME"
curl -s -o /dev/null -w "  gesture HTTP %{http_code}\n" -X POST \
  "http://127.0.0.1:8002/analyze?session_id=$SID" \
  -F "file=@$JX_GESTURE_FRAME"

h "④ 三份日志:文件名与首列都必须是这个号(不是 UUID)"
for m in face_au_log interview_emotion_log gesture_emotion_log; do
  f="data/logs/${m}_${SID}.csv"
  if [ -f "$f" ]; then
    first=$(sed -n '2p' "$f" | cut -d, -f1)
    [ "$first" = "$SID" ] && ok "$(basename $f)  首列=$first" \
                          || bad "$(basename $f) 首列=$first(**不是 $SID**)"
  else
    bad "缺 $f"
  fi
done

h "⑤ 出报告,并断言覆盖 ≠ 0、且点名的是这一场"
$PY -m report_frontend.report_generator --session-id "$SID" 2>&1 | grep -E "目标会话|📥|⚠️" | sed 's/^/  /'
REPORT=$(ls -t data/output/Research_Assessment_Report_*.html | head -1)
$PY - "$REPORT" "$SID" <<'PY'
import re, sys
html = open(sys.argv[1], encoding="utf-8").read()
sid = sys.argv[2]
cov = re.search(r'cover-number">([^<]+)<', html)
cov = cov.group(1).strip() if cov else "?"
print(f"  报告:{sys.argv[1].split('/')[-1]}")
print(f"  覆盖:{cov}")
ok = True
if cov.startswith("0 /"):
    print("  ✗ 覆盖仍是 0 —— M2 的成功定义没达成"); ok = False
else:
    print(f"  ✓ 覆盖不是 0({cov})")
i = html.find("本场会话")
head = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html[i:i+300])) if i >= 0 else ""
print(f"  报告头:{head[:200]}")
if sid not in head:
    print(f"  ✗ 报告头没点名 {sid}"); ok = False
else:
    print(f"  ✓ 报告头点名了 {sid}")
sys.exit(0 if ok else 1)
PY
RC=$?

h "完成"
echo "  本次 SID: $SID"
echo "  停服务:   for p in 8000 8001 8002 5000 5173; do fuser -k \$p/tcp; done"
echo "  ⚠️ 这个脚本验的是**契约层**(按前端的调用顺序)。真正点界面那半要你在浏览器里做。"
exit $RC
