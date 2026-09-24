### Task 2: face 适配

**Files:**
- Modify: `face_expression/pipeline/video_pipeline.py:24,31-42,44-52`
- Modify: `face_expression/api/app.py:116-153`(创建/回收/重置),`:369-392`(`__main__` 自检探针)
- Test: `tests/test_face_detector_wiring.py`

**Interfaces:**
- Consumes: `FaceDetector`、`verify_models`(T1)
- Produces: `VideoPipeline(fps=, session_id=, save_landmarks=, detector=None)` —— 新增 `detector` 参数(测试注入用);`VideoPipeline.reset()`;`session_pipelines[sid]` 的值仍是 `(pipeline, last_used)` 二元组(**不变**,避免动到别处)

- [ ] **Step 1: 写失败测试**

创建 `tests/test_face_detector_wiring.py`:

```python
# tests/test_face_detector_wiring.py
"""M1.5 T2:face 侧的接线。Review Focus 2/3/4。

不 import mediapipe:探测器通过 `detector=` 注入假件。
"""

import numpy as np
import pytest

from face_expression.pipeline.video_pipeline import VideoPipeline


class _FakeDetector:
    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.closed = False

    def detect(self, image_rgb):
        self.calls += 1
        return None            # "没检出" —— 走 process_frame 的 no_face 分支,不碰几何层

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


_FRAME = np.zeros((48, 48, 3), dtype=np.uint8)


def test_pipeline_uses_the_injected_detector():
    """接线的底线:process_frame 走注入的探测器,而不是自己造一个。

    红法:保留旧的惰性 `face_mesh` 属性(它会去 import mediapipe.solutions 而炸)。
    """
    d = _FakeDetector()
    p = VideoPipeline(fps=30, session_id="s", detector=d)
    p.process_frame(_FRAME)

    assert d.calls == 1


def test_reset_forwards_to_the_detector():
    """spec §6.4 / Review Focus 2:`/session/{sid}/reset` 必须把帧计数一起归零。

    红法:`VideoPipeline.reset()` 只清自己的历史、不转发给探测器 —— 之后同一会话
    再来一帧,时间戳回退,mediapipe 抛错。
    """
    d = _FakeDetector()
    p = VideoPipeline(fps=30, session_id="s", detector=d)
    p.process_frame(_FRAME)
    p.reset()

    assert d.resets == 1


def test_close_releases_the_detector():
    """spec §6.3 / Review Focus 3:TTL 回收时必须 close(否则泄漏 native 句柄)。

    红法:不给 VideoPipeline 提供 close(),或写了但不转发。
    """
    d = _FakeDetector()
    p = VideoPipeline(fps=30, session_id="s", detector=d)
    p.close()

    assert d.closed is True
```

- [ ] **Step 2: 跑测试,确认它红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_face_detector_wiring.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'detector'`

- [ ] **Step 3: 改 `video_pipeline.py`**

把第 24 行的 `self._face_mesh = None`、第 31-42 行的整个 `face_mesh` 属性**删掉**,`__init__` 改成:

```python
    def __init__(self, fps=30, session_id="default", save_landmarks=False, detector=None):
        self.fps = fps
        self.session_id = session_id
        self.blink_times = []
        self.eye_closed_duration = 0
        self.last_blink_time = 0
        self.EAR_THRESHOLD = 0.21

        # 探测器**按会话**注入(不是惰性属性,也不是模块级单例):tasks 的 VIDEO 模式
        # 把跟踪状态挂在实例上,单例会让并发会话互相污染跟踪(spec §3.5 / D3)。
        # 缺省自己造一个真的;测试注入假件。
        if detector is None:
            from .detector import FaceDetector
            from ..config import FACE_MODEL
            detector = FaceDetector(FACE_MODEL, fps=fps)
        self.detector = detector

        self.feature_calculator = AUFeatureCalculator(save_landmarks=save_landmarks)
        self.micro_detector = MicroExpressionDetector(fps=fps)
        self.tension_engine = TensionEngine()
        self.emotion_engine = EmotionEngine()
        self.au_history = collections.deque(maxlen=int(3 * fps))
```

`process_frame` 的开头**替换**掉下面这 4 行(原文第 45、46、48、49、51、52 行):

```python
        results = self.face_mesh.process(image_rgb)      # ← 删
        h, w = image_rgb.shape[:2]                       # ← 保留
        if not results.multi_face_landmarks:             # ← 删
            return None, None, {"emotion": "no_face"}    # ← 保留(条件变了)
        lm = results.multi_face_landmarks[0].landmark    # ← 删
        landmarks_norm = [(pt.x, pt.y) for pt in lm]     # ← 删
```

替换成:

```python
    def process_frame(self, image_rgb):
        h, w = image_rgb.shape[:2]
        landmarks_norm = self.detector.detect(image_rgb)

        if not landmarks_norm:
            return None, None, {"emotion": "no_face"}
```

**从 `nose_tip = np.array(landmarks_norm[1])` 那一行起,一个字都不改** —— 几何层整段原样保留。

在类里加两个方法:

```python
    def reset(self):
        """`/session/{sid}/reset` 调:帧计数一起归零,否则下一帧时间戳回退(spec §6.4)。"""
        self.detector.reset()

    def close(self):
        """TTL 回收时调:释放探测器的 native 句柄(spec §6.3)。"""
        self.detector.close()
```

- [ ] **Step 4: 跑测试,确认它绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_face_detector_wiring.py -q`
Expected: PASS

- [ ] **Step 5: 改 `face_expression/api/app.py` 的三处**

(a) TTL 回收(`:125-128`)—— 删会话时把探测器一起关掉:

```python
    for sid in expired_sessions:
        pipeline, _ = session_pipelines.pop(sid)
        pipeline.close()          # 必须显式关:tasks 探测器持 native 句柄(spec §6.3)
        session_loggers.pop(sid, None)
        logger.info(f"清理过期会话: {sid}")
```

(b) `/session/{sid}/reset`(`:336` 附近)—— 找到现有的重置逻辑,在删掉会话之前加一行转发。**先读那一段确认现有形状**;若它是 `del session_pipelines[session_id]`,改成:

```python
            pipeline, _ = session_pipelines.pop(session_id)
            pipeline.close()
```

并在同一函数里、删除**之前**调用 `pipeline.reset()`(若该端点语义是"重置状态但保留会话")。**实施者按读到的实际语义二选一,并在任务报告里说明选了哪个、依据是什么** —— 这是本任务唯一需要判断的地方。

(c) `__main__` 自检探针(`:352-379`)—— 把那段 `mp.solutions` 探针整体换成:

```python
        from face_expression.pipeline.detector import verify_models
        verify_models()
        logger.info("人脸模型自检通过")
```

并**删掉** `:20` 的模块级 `import mediapipe as mp`(它现在没人用了)。`verify_models()` 抛错时**不要**吞 —— 让它拦住启动(spec §8)。所以这个 `__main__` 块里的 `try/except` 要去掉包裹,或改成 `except Exception: logger.exception(...); raise`。

- [ ] **Step 6: 跑全量**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest -q`
Expected: 0 红

- [ ] **Step 7: 实跑一次(本任务自己的验收,不等最终审查)**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m face_expression.api.app &
sleep 8
SID=$($HOME/miniconda3/envs/jingxin/bin/python -c "import datetime;print('t_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_aaaa')")
curl -s -o /tmp/face_out.json -w "%{http_code}\n" -X POST http://127.0.0.1:8000/analyze \
  -F "file=@$HOME/shared/mp_frames/frames/frame_0001.png" -F "session_id=$SID"
ls -l ~/jingxin/data/logs/face_au_log_$SID.csv
kill %1
```

Expected: HTTP **200**;日志文件存在且首列是 `$SID`。**这是 face 迁移前做不到的事** —— 迁移前这里必然 500 且没有文件。

- [ ] **Step 8: 提交**

```bash
git add face_expression/pipeline/video_pipeline.py face_expression/api/app.py \
        tests/test_face_detector_wiring.py
git commit -m "feat(m1.5): face 接入 tasks 探测器(按会话 + 帧计数 + TTL 释放)"
```

---

