#!/usr/bin/env bash
# M2.1 验收(第 19 条):科研评估自成一场,不再把科研回答写进面试会话目录。
#
# 只依赖语音服务(:8001)与局域网 FunASR —— 铸号、转写、录制目录三件事都在它手里。
# 语音样本默认取最近一场会话的 16k converted wav(见 tools/jx_env.sh;可用 $JX_AUDIO_SAMPLE 覆盖)。
# (合成的 espeak 假语音 ASR 转不出字,实测返回空串,不可用 —— 见账本。)
set -u
V=http://127.0.0.1:8001
REC=${JINGXIN_RECORDINGS_DIR:-$HOME/shared/jingxin_recordings}
source "$(dirname "${BASH_SOURCE[0]}")/jx_env.sh"
PY="$JX_PY"
FAIL=0
bad() { echo "❌ $1"; FAIL=1; }
ok()  { echo "✅ $1"; }

sid_of() { $PY -c 'import sys,json;print(json.load(sys.stdin).get("session_id",""))'; }

ISID=$(curl -s -m 20 -X POST "$V/interview/start" | sid_of)
[ -n "$ISID" ] || { echo "语音服务没起(:8001)?先跑 tools/start_all.sh,或 PY -m voice_interaction.api.app"; exit 1; }
RSID=$(curl -s -m 20 -X POST "$V/research/start" | sid_of)
echo "面试会话 = $ISID"
echo "科研会话 = $RSID"
[ -n "$RSID" ] || bad "科研入口没铸号(第 19 条未修)"
[ "$ISID" != "$RSID" ] || bad "科研与面试拿到同一个号"

ffmpeg -y -loglevel error -ss 5 -t 12 -i "$JX_AUDIO_SAMPLE" \
       -ar 16000 -ac 1 -c:a pcm_s16le /tmp/m21_ans.wav || bad "取音频样本失败"

echo "--- ① 用**科研自己的号**提交语音回答(修好之后前端的行为)---"
curl -s -m 120 -X POST "$V/research/answer_audio?session_id=$RSID" \
     -F "audio=@/tmp/m21_ans.wav" | head -c 200; echo
[ -f "$REC/$RSID/transcript.json" ] && ok "科研回答落在科研号的目录" || bad "科研回答没落到 $RSID"
[ -f "$REC/$ISID/transcript.json" ] && bad "面试目录被写入了 —— 张冠李戴仍在" || ok "面试目录没被写入"

echo "--- ③ 科研会话要产出**报告侧读得到**的特征行(I2 收口)---"
ROW="$REPO/data/logs/research_emotion_log_$RSID.csv"
[ -f "$ROW" ] && ok "科研特征行落在 $RSID 名下" || bad "科研会话没有产出特征行:$ROW 不存在"
if [ -f "$ROW" ]; then
  head -1 "$ROW" | grep -q "connective_density" && ok "特征行含连接词密度列" || bad "特征行没有密度列"
  grep -q "$RSID" "$ROW" && ok "首列是本次会话号" || bad "首列不是本次会话号"
fi

echo "--- ④ 报告侧要能把它读成「语音（科研）· 已读入」---"
# ⚠️ 必须在仓库根下跑:`report_frontend` 是从**当前目录**导入的。
# 2026-09-25 踩过:从 ~/shared 直接跑这个脚本时,这一步 ModuleNotFoundError,
# 而前面几步都是纯 curl/bash 所以照常绿 —— 整条验收看起来"只是第④步失败",
# 真因是自己的工作目录不对。用子 shell cd 掉,与调用者的 cwd 无关。
( cd "$REPO" && $PY - <<PYEOF
from report_frontend.data_loader import LogDataLoader
l = LogDataLoader()
l.get_fused_latest_data("$RSID")
st = l.selected_sessions.get("voice_research", {})
print("   voice_research →", st)
assert st.get("status") == "loaded" and st.get("rows") == 1, f"科研模态没被读成 loaded:{st}"
assert st.get("session_id") == "$RSID", st
PYEOF
)
[ $? -eq 0 ] && ok "报告侧把科研模态读成活的了" || bad "报告侧仍读不出科研模态"

echo "--- ② 反向复现:把**上一场面试的号**喂给科研回答(修之前前端的行为)---"
curl -s -m 120 -X POST "$V/research/answer_audio?session_id=$ISID" \
     -F "audio=@/tmp/m21_ans.wav" | head -c 200; echo
if [ -f "$REC/$ISID/transcript.json" ]; then
  ok "复现成功:带旧号时科研回答确实会写进面试目录(这就是修掉的那条路)"
else
  bad "复现不出来 —— 说明这条验证没压住那个失效模式,别当成已验"
fi

[ "$FAIL" = 0 ] && echo "M2.1 验收通过" || echo "M2.1 验收失败"
exit $FAIL
