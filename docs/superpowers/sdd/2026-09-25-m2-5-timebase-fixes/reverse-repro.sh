#!/usr/bin/env bash
# M2.5 Task 9 Step 1:反向复现(账本 §4.2 的标准动作)。
# 每条:撤掉一处生产改动 → 对应测试必须**红** → 恢复。红错了地方同样是失败。
set -u
cd /home/huihuibuhui/jingxin
PY=~/miniconda3/envs/jingxin/bin/python
FAIL=0

patch() {  # patch <file> <old> <new>
  $PY - "$1" "$2" "$3" <<'PYEOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
t = p.read_text()
old, new = sys.argv[2], sys.argv[3]
if old not in t:
    print(f"  !! 锚点没找到:{p}"); sys.exit(3)
p.write_text(t.replace(old, new, 1))
PYEOF
}

check() {  # check <名字> <期望红的测试> <文件> <old> <new>
  name=$1; test=$2; file=$3; old=$4; new=$5
  patch "$file" "$old" "$new" || { echo "❌ $name:补丁没打上"; FAIL=1; return; }
  out=$($PY -m pytest "$test" -q 2>&1 | tail -1)
  if echo "$out" | grep -q "failed"; then
    echo "✅ $name:撤掉后**红了** ($out)"
  else
    echo "❌ $name:撤掉后**没红** —— 该测试没约束力 ($out)"; FAIL=1
  fi
  git checkout -- "$file"
}

echo "── ① SessionClock 的「抬到 +1」──"
check "严格递增" tests/test_session_clock.py::test_stamp_is_strictly_increasing_even_within_one_millisecond \
  session_clock.py '            elapsed_ms = self._last_ms + 1' '            pass  # 反向复现:去掉抬升'

echo "── ② 探测器的回退守卫 ──"
check "时间戳回退抛错" tests/test_detector_contract.py::test_detector_rejects_a_timestamp_that_goes_backwards \
  face_expression/pipeline/detector.py '        if self._last_ts is not None and timestamp_ms < self._last_ts:' '        if False:'

echo "── ③ eye_closed_sec 的真实间隔 ──"
check "闭眼时长" tests/test_video_pipeline_timebase.py::test_eye_closed_seconds_uses_the_real_gap_between_frames \
  face_expression/pipeline/video_pipeline.py \
  '                self.eye_closed_duration += (timestamp_ms - self.history_last_ms) / 1000.0' \
  '                self.eye_closed_duration += 1 / 30.0'

echo "── ④ duration_sec 由时间戳导出 ──"
check "会话时长" tests/test_video_pipeline_timebase.py::test_duration_comes_from_timestamps_not_frame_count \
  face_expression/pipeline/video_pipeline.py \
  '        duration_sec = (last - first) / 1000.0' \
  '        duration_sec = len(self.au_history) / 30.0'

echo
echo "=== 恢复后全量套件 ==="
$PY -m pytest -q 2>&1 | tail -2
echo "=== 工作树干净? ==="
git status --short | grep -v '^??' || echo "(tracked 文件无残留改动)"
exit $FAIL
