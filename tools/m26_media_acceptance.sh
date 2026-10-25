#!/usr/bin/env bash
# M2.6 前端腿·服务端半的冒烟验收:两个新端点 + 收尾对账。
# 用法: bash tools/m26_media_acceptance.sh
#
# 它与 spec §7.7 那份清单的关系:§7.7 是给**真会话**(N3)用的,还要求 camera.webm
# 能解出音轨(要等前端腿)。这里是它的**服务端冒烟版** —— 用 curl 造一场假会话,
# 把 POST /session/{sid}/media、POST /session/{sid}/question 与收尾对账串起来跑通。
# **它不替代 §7.7。**
set -uo pipefail
# ⚠️ 解释器与仓库根**都从 `jx_env.sh` 解析**,不在这里写死。
#    2026-10-25 实测:此前写的是 `PY=~/miniconda3/envs/jingxin/bin/python` +
#    `cd ~/jingxin`,而服务器上前者是 `~/anaconda3/...`、后者是 `~/JingXin/jingxin`
#    ⟹ 脚本第一行就废,而报错落在「① 原生视频上传 ❌ 没存下」上,看着像上传坏了。
#    钉子:`tests/test_tools_resolve_paths_by_machine.py`。
source "$(dirname "${BASH_SOURCE[0]}")/jx_env.sh"
PY="$JX_PY"
WORK=$(mktemp -d); PORT=8093
export JINGXIN_RECORDINGS_DIR="$WORK/rec" JINGXIN_DATA_DIR="$WORK/data"
mkdir -p "$JINGXIN_RECORDINGS_DIR" "$JINGXIN_DATA_DIR"
cd "$REPO"
$PY -m uvicorn voice_interaction.api.app:app --host 127.0.0.1 --port $PORT \
  > "$WORK/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; echo "(服务已停;临时目录 $WORK 留着给排查)"' EXIT
for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 0.5; done

# ── 判红要真的判红(2026-10-25)─────────────────────────────────────────
# 旧版:第 ②③ 步没有 `|| exit`,而结尾是**无条件** `echo "全部冒烟判据通过"`。
# 实测那天第 ② 步 400、第 ③ 步抛 `FileNotFoundError`,结尾**照样说通过**。
# 本仓记过的「静默失败」至少是没说话;这一条是**说了反话**。
FAIL=0
bad() { echo "   ❌ $1"; FAIL=1; }
ok()  { echo "   ✓ $1"; }

SID=$(curl -s -X POST "http://127.0.0.1:$PORT/interview/start" \
      | $PY -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
echo "会话 = $SID"

echo "① 原生视频上传"
head -c 40000 /dev/urandom > "$WORK/camera.webm"
curl -s -X POST "http://127.0.0.1:$PORT/session/$SID/media" \
     -F "file=@$WORK/camera.webm" | tee "$WORK/media.json"; echo
grep -q '"stored": *true' "$WORK/media.json" || bad "没存下"
cmp -s "$WORK/camera.webm" "$JINGXIN_RECORDINGS_DIR/$SID/media/camera.webm" \
  && ok "盘上字节与上传逐字节相同" || bad "字节不一致"
[ -d "$JINGXIN_RECORDINGS_DIR/$SID/media/camera" ] \
  && bad "建出了空的 media/camera/" || ok "没有多余的空目录"

# ⚠️ 时间戳**当场算**,不写死。写死的 `1758824000` 是 2025-09 的秒数,而
#    `session_meta._validate_window` 有一条量程闸:离此刻超过 1 天 ⟹ 400。
#    ⟹ 写死的那天起这一步就静默不执行了。钉子:`test_acceptance_scripts_do_not_lie.py`。
NOW=$(date +%s)
echo "② 题目时刻上报(两题,基时刻=$NOW)"
for i in 0 1; do
  curl -s -X POST "http://127.0.0.1:$PORT/session/$SID/question" \
    -H 'Content-Type: application/json' \
    -d "{\"qid\":\"题目${i}\",\"index\":$i,\"ask_start\":$((NOW+i*10)).0,\"ask_end\":$((NOW+6+i*10)).0}" \
    -o "$WORK/q$i.json"
  grep -q '"status": *"success"' "$WORK/q$i.json" \
    || bad "题目${i} 上报失败:$(cat "$WORK/q$i.json")"
done
[ -f "$JINGXIN_RECORDINGS_DIR/$SID/questions.jsonl" ] \
  && ok "questions.jsonl 行数 = $(wc -l < "$JINGXIN_RECORDINGS_DIR/$SID/questions.jsonl")" \
  || bad "questions.jsonl 没有落盘(上报被拒了?见上一条)"

echo "③ 同题重复上报 → 只留后一个"
curl -s -X POST "http://127.0.0.1:$PORT/session/$SID/question" \
  -H 'Content-Type: application/json' \
  -d "{\"qid\":\"题目0\",\"index\":0,\"ask_start\":${NOW}.0,\"ask_end\":$((NOW+999)).0}" >/dev/null
$PY - "$JINGXIN_RECORDINGS_DIR/$SID/questions.jsonl" "$((NOW+999))" <<'PYEOF' || bad "重复上报的断言没过"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
q0 = [r for r in rows if r["qid"] == "题目0"]
assert len(rows) == 2, f"应当仍是 2 行,得到 {len(rows)}"
assert len(q0) == 1, f"题目0 留了 {len(q0)} 行"
assert q0[0]["ask_end"] == float(sys.argv[2]), q0[0]
print("   ✓ 仍是 2 行,且题目0 取的是后报的值")
PYEOF

echo "④ 倒挂窗口 → 400(不是 500)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  "http://127.0.0.1:$PORT/session/$SID/question" -H 'Content-Type: application/json' \
  -d "{\"qid\":\"坏\",\"index\":2,\"ask_start\":${NOW}.0,\"ask_end\":1.0}")
[ "$code" = "400" ] && ok "400" || bad "得到 $code"

echo "⑤ 路径穿越 → 400 或 404,且不落盘"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  "http://127.0.0.1:$PORT/session/..%2Fescape/media" -F "file=@$WORK/camera.webm")
echo "   HTTP $code"
[ -e "$JINGXIN_RECORDINGS_DIR/../escape" ] && bad "越界目录出现了" || ok "盘上没有越界目录"

echo "⑥ 收尾对账(meta 还没填,应当点出来)"
$PY -m session_meta --write-template "$SID" >/dev/null
$PY -m session_meta --check-session "$SID" --expected-questions 3
rc=$?
# 这一条**期望非零**(1 = 有缺项)。所以它不许进 FAIL —— 但也不许静默:
# 退出码 0 说明"该报的缺项没报",那是反方向的坏。
[ "$rc" = "1" ] && ok "退出码 1(有缺项,正是期望)" || bad "退出码 $rc,期望 1"

echo
if [ "$FAIL" = 0 ]; then
  echo "全部冒烟判据通过。真会话的 §7.7 清单在录制时另跑。"
else
  echo "★ 有判据没过 —— 见上面带 ❌ 的几行。"
fi
exit $FAIL
