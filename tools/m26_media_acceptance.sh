#!/usr/bin/env bash
# M2.6 前端腿·服务端半的冒烟验收:两个新端点 + 收尾对账。
# 用法: bash tools/m26_media_acceptance.sh
#
# 它与 spec §7.7 那份清单的关系:§7.7 是给**真会话**(N3)用的,还要求 camera.webm
# 能解出音轨(要等前端腿)。这里是它的**服务端冒烟版** —— 用 curl 造一场假会话,
# 把 POST /session/{sid}/media、POST /session/{sid}/question 与收尾对账串起来跑通。
# **它不替代 §7.7。**
set -uo pipefail
PY=~/miniconda3/envs/jingxin/bin/python
WORK=$(mktemp -d); PORT=8093
export JINGXIN_RECORDINGS_DIR="$WORK/rec" JINGXIN_DATA_DIR="$WORK/data"
mkdir -p "$JINGXIN_RECORDINGS_DIR" "$JINGXIN_DATA_DIR"
cd ~/jingxin
$PY -m uvicorn voice_interaction.api.app:app --host 127.0.0.1 --port $PORT \
  > "$WORK/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; echo "(服务已停;临时目录 $WORK 留着给排查)"' EXIT
for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 0.5; done

SID=$(curl -s -X POST "http://127.0.0.1:$PORT/interview/start" \
      | $PY -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
echo "会话 = $SID"

echo "① 原生视频上传"
head -c 40000 /dev/urandom > "$WORK/camera.webm"
curl -s -X POST "http://127.0.0.1:$PORT/session/$SID/media" \
     -F "file=@$WORK/camera.webm" | tee "$WORK/media.json"; echo
grep -q '"stored": *true' "$WORK/media.json" || { echo "❌ 没存下"; exit 1; }
cmp -s "$WORK/camera.webm" "$JINGXIN_RECORDINGS_DIR/$SID/media/camera.webm" \
  && echo "   ✓ 盘上字节与上传逐字节相同" || { echo "❌ 字节不一致"; exit 1; }
[ -d "$JINGXIN_RECORDINGS_DIR/$SID/media/camera" ] \
  && { echo "❌ 建出了空的 media/camera/"; exit 1; } || echo "   ✓ 没有多余的空目录"

echo "② 题目时刻上报(两题)"
for i in 0 1; do
  curl -s -X POST "http://127.0.0.1:$PORT/session/$SID/question" \
    -H 'Content-Type: application/json' \
    -d "{\"qid\":\"题目${i}\",\"index\":$i,\"ask_start\":$((1758824000+i*10)).0,\"ask_end\":$((1758824006+i*10)).0}" \
    >/dev/null
done
echo "   questions.jsonl 行数 = $(wc -l < "$JINGXIN_RECORDINGS_DIR/$SID/questions.jsonl")"

echo "③ 同题重复上报 → 只留后一个"
curl -s -X POST "http://127.0.0.1:$PORT/session/$SID/question" \
  -H 'Content-Type: application/json' \
  -d '{"qid":"题目0","index":0,"ask_start":1758824000.0,"ask_end":1758824999.0}' >/dev/null
$PY - "$JINGXIN_RECORDINGS_DIR/$SID/questions.jsonl" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
q0 = [r for r in rows if r["qid"] == "题目0"]
assert len(rows) == 2, f"应当仍是 2 行,得到 {len(rows)}"
assert len(q0) == 1, f"题目0 留了 {len(q0)} 行"
assert q0[0]["ask_end"] == 1758824999.0, q0[0]
print("   ✓ 仍是 2 行,且题目0 取的是后报的值")
PYEOF

echo "④ 倒挂窗口 → 400(不是 500)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  "http://127.0.0.1:$PORT/session/$SID/question" -H 'Content-Type: application/json' \
  -d '{"qid":"坏","index":2,"ask_start":9.0,"ask_end":1.0}')
[ "$code" = "400" ] && echo "   ✓ 400" || { echo "❌ 得到 $code"; exit 1; }

echo "⑤ 路径穿越 → 400 或 404,且不落盘"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  "http://127.0.0.1:$PORT/session/..%2Fescape/media" -F "file=@$WORK/camera.webm")
echo "   HTTP $code"
[ -e "$JINGXIN_RECORDINGS_DIR/../escape" ] && { echo "❌ 越界目录出现了"; exit 1; }
echo "   ✓ 盘上没有越界目录"

echo "⑥ 收尾对账(meta 还没填,应当点出来)"
$PY -m session_meta --write-template "$SID" >/dev/null
$PY -m session_meta --check-session "$SID" --expected-questions 3
rc=$?
echo "   (退出码 $rc —— 1 = 有缺项,正是期望)"

echo
echo "全部冒烟判据通过。真会话的 §7.7 清单在录制时另跑。"
