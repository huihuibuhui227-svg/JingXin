# M2.5 时间基修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把系统里每一处「时间」改成由调用方显式给出、且与真实时间一致 —— 实时路径喂服务端实测流逝毫秒,离线路径喂 `帧序号 × FRAME_SKIP / src_fps`。

**Architecture:** 时间戳从「探测器/管线内部用 `帧序号 ÷ fps` 推算」改为**显式入参**沿调用链下传:调用方 → `VideoPipeline.process_frame(frame, timestamp_ms)` → `detector.detect(frame, timestamp_ms)` / `micro_detector.detect(au, timestamp_ms)`。所有由 `fps` 派生的时长(3 秒历史窗、眨眼去抖、`eye_closed_sec`、`duration_sec`)一律改由时间戳差值导出;`fps` 降级为元信息里的**滑动实测值**。三条已有契约测钉住的是**旧公式**,必须按新契约重写。

**Tech Stack:** Python 3 / pytest / FastAPI / mediapipe `tasks`(VIDEO 模式)。解释器固定 `~/miniconda3/envs/jingxin/bin/python`。

**Spec:** `docs/superpowers/specs/2026-09-25-m2-5-timebase-fixes-design.md`

## Global Constraints

- **解释器必须是** `~/miniconda3/envs/jingxin/bin/python`(不可用 `~/huihui/bin/python` —— 缺 websockets 等)。
- **`code_data_supplement/` 一行都不改。** 它是 CueCoP 的去品牌化快照,不在上线路径上(spec §1.2)。
- **前端一行都不改。** `?fps=30` 会继续被发过来,服务端**忽略**它并记一次 warning(spec §5.4)。
- **`measured_fps` 不进 CSV 列**,只进服务日志与 `session.json`(spec §5.5)。
- **`timestamp_ms` 必须是 `int`,且严格递增。**
- 每个提交只 `git add` 本任务列出的文件;本仓有未跟踪的 `experiments/` 等杂物,**不要 `git add -A`**。

## Review Focus

按 spec 隐含、但没有任何任务天然会覆盖的失效形态,从最可能咬人排起。每条都在它归属的任务里加了钉子。

1. **同一毫秒内的连续两帧** —— 实时 1 fps 下也可能(测试里几乎必然)。时间戳**相等**会让 mediapipe 的 VIDEO 模式出问题,而旧契约测断言的是**严格**递增(`ts == sorted(set(ts))`)。→ 时钟必须保证**严格**递增,不是「非递减」。钉子:Task 1 Step 1 第 2 条。
2. **`/reset` 之后时钟没跟着归零** —— 管线重建了、时钟没重建,同一 id 的第二段会话时间戳会接着上一段涨。→ 钉子:Task 5 Step 1 第 3 条。
3. **`no_face` 帧不推进历史** —— `process_frame` 在没检出脸时提前返回,3 秒时间窗里可能一帧都没有。→ 时序统计必须对空窗给出 0/空,不许抛。钉子:Task 4 Step 1 第 4 条。
4. **离线 `src_fps` 为 0 / NaN** —— 除零或 NaN 时间戳污染整列。→ 钉子:Task 7 Step 1 第 2 条。
5. **单帧会话** —— 只有 1 帧时 `duration_sec` 必须是 0、时间窗只有 1 帧,不许抛、不许把 NaN 写进 CSV。钉子:Task 4 Step 1 第 5 条。

---

## Task 1: `session_clock.py` —— 会话时钟(新文件)

**Files:**
- Create: `session_clock.py`
- Test: `tests/test_session_clock.py`

**Interfaces:**
- Consumes: 无
- Produces: `SessionClock(now: Callable[[], float] = time.monotonic)`,方法 `stamp_ms() -> int`、`reset() -> None`。Task 5 / Task 6 使用。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_session_clock.py
"""M2.5:会话时钟。spec §4 / §5.1 / Review Focus ①。

本文件不 import mediapipe、不起服务 —— 时钟的全部行为都能用一个假 `now` 验完。
"""

import pytest

from session_clock import SessionClock


def test_stamp_is_relative_to_construction_in_milliseconds():
    """会话开始 = 0 ms,之后按真实流逝走。

    红法:把 `stamp_ms()` 写成 `int(time.time() * 1000)`(绝对墙钟)——
    第一条断言立刻红(那是 1.7e12 量级,不是 0)。
    """
    t = [100.0]
    c = SessionClock(now=lambda: t[0])
    assert c.stamp_ms() == 0           # 建好即 0

    t[0] = 100.5
    assert c.stamp_ms() == 500         # 500 ms 后

    t[0] = 101.25
    assert c.stamp_ms() == 1250


def test_stamp_is_strictly_increasing_even_within_one_millisecond():
    """★ Review Focus ①:`now()` 不动(同一毫秒内两次调用)时也必须**严格**递增。

    红法:直接 `return int((self._now() - self._start) * 1000)` ——
    第二帧与第一帧同值,mediapipe 的 VIDEO 模式要求时间戳单调递增,旧契约测
    `ts == sorted(set(ts))` 断言的也正是「严格」(不是「非递减」)。
    """
    t = [100.0]
    c = SessionClock(now=lambda: t[0])
    a = c.stamp_ms()
    b = c.stamp_ms()                   # now() 没动
    c_ms = c.stamp_ms()

    assert a == 0, a
    assert b == 1 and c_ms == 2, (a, b, c_ms)
    assert a < b < c_ms, "必须严格递增,否则 mediapipe 会抛"


def test_reset_rewinds_to_zero():
    """`/reset` 之后新一段从 0 起(Review Focus ② 的一半:时钟这一半)。

    红法:把 `reset()` 写成空函数 —— 第二段的时间戳接着第一段涨,
    `/reset` 那条路径(管线归零)就只归零了一半。
    """
    t = [100.0]
    c = SessionClock(now=lambda: t[0])
    t[0] = 105.0
    assert c.stamp_ms() == 5000

    c.reset()
    assert c.stamp_ms() == 0

    t[0] = 106.0
    assert c.stamp_ms() == 1000
```

- [ ] **Step 2: 跑测试,确认它失败**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_clock.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'session_clock'`

- [ ] **Step 3: 写实现**

```python
# session_clock.py
"""会话相对时钟:把「距会话开始的毫秒数」做成一个显式对象。

为什么要有这个文件(M2.5 spec §4 / D2):
  * 实时路径**没有视频文件**,时间只能实测 —— 墙钟在实时流里就是真实时间;
  * 而 mediapipe 的 VIDEO 模式要求时间戳**严格递增**,否则抛错;
  * 于是「实测」不能是 `int((now - start) * 1000)` 就完事:同一毫秒内的两帧会撞值。
    本类把那 1 ms 的抖动吃在内部,对外保证严格递增。

离线批量路径**不用**这个类 —— 那里时间由 `帧序号 × FRAME_SKIP / src_fps` 决定,
是确定性的,不该混进墙钟(spec §4 表)。

与 `logging_config.py` 同层放在仓库根:face / gesture 两个服务都以 `-m <模块>.api.app`
启动,仓库根在 sys.path 上,所以两边都 import 得到。
"""

import time
from typing import Callable


class SessionClock:
    """距会话开始的毫秒数,**严格递增**。"""

    def __init__(self, now: Callable[[], float] = time.monotonic):
        # 用 monotonic 而不是 time():后者会被 NTP 回拨,回拨即时间戳回退。
        self._now = now
        self.reset()

    def reset(self) -> None:
        """`/reset` 与新会话都走这里。"""
        self._start = self._now()
        self._last_ms = -1

    def stamp_ms(self) -> int:
        """本帧的时间戳。保证 `> 上一次返回值`(不是 `>=`)。"""
        elapsed_ms = int((self._now() - self._start) * 1000)
        if elapsed_ms <= self._last_ms:
            # 同一毫秒内的两帧 —— 抬到上一次 +1,而不是放行相等值。
            elapsed_ms = self._last_ms + 1
        self._last_ms = elapsed_ms
        return elapsed_ms
```

- [ ] **Step 4: 跑测试,确认它通过**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_clock.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
cd ~/jingxin
git add session_clock.py tests/test_session_clock.py
git commit -m "feat(m2.5): 会话相对时钟(严格递增) —— spec §4"
```

---

## Task 2: 探测器契约改成显式时间戳(face + gesture)

**Files:**
- Modify: `face_expression/pipeline/detector.py:81-109`(`__init__` 去掉 `fps`、`detect` 收 `timestamp_ms`、`reset` 清 `_last_ts`)
- Modify: `gesture_analysis/core/detectors.py:88-135`(`_Base` 同上;`_timestamp_ms()` 删除)
- Test: `tests/test_detector_contract.py`(改 3 条、新增 3 条)

**Interfaces:**
- Consumes: 无
- Produces: `FaceDetector.detect(image_rgb, timestamp_ms: int)`、`HandDetector.detect(...)`、`PoseDetector.detect(...)` —— 均由 Task 4 / Task 5 / Task 6 / Task 7 调用。三者构造签名变为 `(model_path, *, factory=None, **build_kwargs)`(**不再有 `fps`**)。

- [ ] **Step 1: 改测试(旧公式的三条按新契约重写 + 新增三条)**

把 `tests/test_detector_contract.py` 里的 `test_timestamps_are_strictly_increasing_within_a_session` 与 `test_reset_rewinds_the_frame_counter` 整体替换为:

```python
def test_detector_passes_the_caller_timestamp_straight_through():
    """M2.5 契约:时间戳由调用方给,探测器**原样**转交,不再自己算。

    红法:把 `detect()` 改回 `int(self._frame_index * 1000 / self.fps)` ——
    下面三个值立刻变成 [0, 100, 200],红。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=1))
    for ts in (0, 7, 1234):
        d.detect(_FRAME, ts)

    assert h["l"].timestamps == [0, 7, 1234], h["l"].timestamps


def test_detector_rejects_a_timestamp_that_goes_backwards():
    """★ 时间戳回退必须**抛**,不许静默转交 —— mediapipe 会拿着乱序时间戳继续算。

    红法:去掉 `if timestamp_ms < self._last_ts: raise`。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=1))
    d.detect(_FRAME, 100)

    with pytest.raises(ValueError) as ei:
        d.detect(_FRAME, 99)
    assert "100" in str(ei.value) and "99" in str(ei.value), (
        f"错误信息要同时带上一次和本次的值,否则排障时不知道谁回退了:{ei.value}")


def test_equal_timestamps_are_allowed_but_only_because_the_clock_prevents_them():
    """相等**不**抛(严格递增由 `SessionClock` 保证,见 Task 1),但也不许被改写成别的值。

    这条刻意把责任划清:探测器只管"不许回退",「严格递增」是时钟的契约。
    红法:在探测器里自作主张 `timestamp_ms = self._last_ts + 1` —— 转交的值就变了。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=1))
    d.detect(_FRAME, 50)
    d.detect(_FRAME, 50)

    assert h["l"].timestamps == [50, 50], h["l"].timestamps


def test_reset_allows_the_next_session_to_start_from_zero_again():
    """`/reset` 之后新一段可以从 0 起(不抛),但**不做任何改写**。

    红法:把 `reset()` 写成空函数 —— 第二段的 `d.detect(_FRAME, 0)` 会抛 ValueError。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=1))
    d.detect(_FRAME, 5000)
    d.reset()
    d.detect(_FRAME, 0)

    assert h["l"].timestamps == [5000, 0], h["l"].timestamps
```

再把该文件里其余用到旧签名的调用补上第二个参数(全部是位置参数 `0`):
`test_face_detector_flattens_to_xy_pairs`(:73)、`test_gesture_detectors_keep_x_and_y_attributes`(:94、:103)、`test_close_releases_the_underlying_landmarker`(:144)、`test_two_sessions_get_different_detectors`(:158、:159),并把它们构造里的 `fps=10` 删掉(参数已不存在)。

- [ ] **Step 2: 跑测试,确认它失败**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_detector_contract.py -v`
Expected: FAIL —— `TypeError: detect() takes 2 positional arguments but 3 were given`(以及 `unexpected keyword argument 'fps'`)

- [ ] **Step 3: 实现 —— face 侧**

`face_expression/pipeline/detector.py`,把 `__init__` 与 `detect`/`reset` 改成:

```python
    def __init__(self, model_path: Path, *, num_faces: int = 1,
                 min_detection_confidence: float = 0.8,
                 min_tracking_confidence: float = 0.8,
                 factory: Callable | None = None):
        self.model_path = Path(model_path)
        self._frame_index = 0
        self._last_ts: int | None = None
        self._factory = factory or _default_factory
        self._landmarker = self._factory(
            self.model_path, num_faces=num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def detect(self, image_rgb: np.ndarray, timestamp_ms: int):
        """VIDEO 模式:时间戳由**调用方**给(M2.5 spec §5.1)。

        `fps` 参数已删除 —— 探测器不再自己推算时间。实测它推错 30 倍:
        客户端 `?fps=30` 而实发 1 帧/秒,`frame_index * 1000 / 30` 于是把
        1 秒当成 33 ms(spec §3.1)。
        """
        import mediapipe as mp

        if self._last_ts is not None and timestamp_ms < self._last_ts:
            raise ValueError(
                f"timestamp_ms 回退:上一次 {self._last_ts},本次 {timestamp_ms} —— "
                f"mediapipe 的 VIDEO 模式要求时间戳严格递增(spec §5.1)")
        self._last_ts = timestamp_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        self._frame_index += 1
        if not result.face_landmarks:
            return None
        return [(p.x, p.y) for p in result.face_landmarks[0]]

    def reset(self) -> None:
        """`/session/{sid}/reset` 要调:帧计数与时间戳基线都归零。

        否则同一 id 的第二段会话第一帧就是回退值,mediapipe 抛错。
        """
        self._frame_index = 0
        self._last_ts = None
```

- [ ] **Step 4: 实现 —— gesture 侧**

`gesture_analysis/core/detectors.py`,`_Base` 改成:

```python
    def __init__(self, model_path: Path, *, factory: Callable | None = None,
                 **build_kwargs):
        self.model_path = Path(model_path)
        self._frame_index = 0
        self._last_ts: int | None = None
        self._landmarker = (factory or self._default_factory)(self.model_path, **build_kwargs)

    def detect(self, image_rgb: np.ndarray, timestamp_ms: int):
        """时间戳由调用方给。`_timestamp_ms()` 已删除(它按 `fps` 推算,错得和 face 一样)。"""
        import mediapipe as mp

        if self._last_ts is not None and timestamp_ms < self._last_ts:
            raise ValueError(
                f"timestamp_ms 回退:上一次 {self._last_ts},本次 {timestamp_ms} —— "
                f"mediapipe 的 VIDEO 模式要求时间戳严格递增(spec §5.1)")
        self._last_ts = timestamp_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        self._frame_index += 1
        return self._groups(result)

    def reset(self) -> None:
        """`/reset` 要调:帧计数与时间戳基线都归零(spec §6.4)。"""
        self._frame_index = 0
        self._last_ts = None
```

同时删掉 `_timestamp_ms()` 与 `_bump()`(后者已并入 `detect`)。
⚠️ 实测 `gesture_analysis/config.py:73-85` 的 `MEDIAPIPE_CONFIG` 里**没有** `fps`,所以删掉这个参数不会把它漏进 `**build_kwargs`。

- [ ] **Step 5: 跑测试,确认它通过**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_detector_contract.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
cd ~/jingxin
git add face_expression/pipeline/detector.py gesture_analysis/core/detectors.py tests/test_detector_contract.py
git commit -m "feat(m2.5): 探测器时间戳改为显式入参;删掉按 fps 推算的 _timestamp_ms"
```

---

## Task 3: 微表情:帧数窗 → 1.5 秒时间窗

**Files:**
- Modify: `face_expression/core/analysis/micro_expression.py`(整个文件)
- Test: `tests/test_micro_expression_window.py`(新建)

**Interfaces:**
- Consumes: 无
- Produces: `MicroExpressionDetector(window_ms: int = 1500)`,方法 `detect(current_au_values, timestamp_ms: int) -> MicroExpressionResult`。Task 4 调用。构造**不再收 `fps`**。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_micro_expression_window.py
"""M2.5:微表情窗按**时间**而不是帧数(spec §5.3 —— 本设计对 spec 措辞的唯一偏离)。"""

from types import SimpleNamespace

from face_expression.core.analysis.micro_expression import MicroExpressionDetector


def _au(**kw):
    """`detect` 只从对象上按名字取那几个 AU,`SimpleNamespace` 就够。"""
    base = dict(au4_frown=0.0, au7_eye_squeeze=0.0, au15_mouth_down=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_window_is_pruned_by_time_not_by_frame_count():
    """★ 窗长是 1.5 **秒**,不是 15 帧。

    造法:先喂 20 帧密集帧(每 10 ms),再跳到 5 秒后喂 1 帧 ——
    如果窗是按帧数(15)而不是按时间,那 5 秒前那些帧还会留在窗里。

    红法:把 `maxlen=15` 的旧实现拿回来 —— 断言 `len(history)` 会 >= 15 而不是 1。
    """
    d = MicroExpressionDetector()
    for i in range(20):
        d.detect(_au(au4_frown=1.0), timestamp_ms=i * 10)
    assert len(d._window('au4_frown')) > 1, "前提不成立:密集帧应当留在窗里"

    d.detect(_au(au4_frown=1.0), timestamp_ms=5000)
    assert len(d._window('au4_frown')) == 1, (
        f"5 秒前的帧还在窗里 —— 窗是按帧数而不是按时间的:{len(d._window('au4_frown'))}")


def test_same_frame_count_different_spacing_gives_different_windows():
    """同样的**帧数**、不同的**间隔**,窗里剩的帧数必须不同。

    这是「帧数窗」和「时间窗」的分水岭:旧实现在两种情形下窗里都是 15 帧。

    红法:实现里不读 `timestamp_ms`。
    """
    dense = MicroExpressionDetector()
    for i in range(20):
        dense.detect(_au(au4_frown=1.0), timestamp_ms=i * 10)      # 200 ms 内 20 帧
    assert len(dense._window('au4_frown')) == 20

    sparse = MicroExpressionDetector()
    for i in range(20):
        sparse.detect(_au(au4_frown=1.0), timestamp_ms=i * 1000)   # 20 秒内 20 帧
    assert len(sparse._window('au4_frown')) == 2, (
        f"1.5 秒窗里应当只剩最后一帧和它 1 秒前那帧:{len(sparse._window('au4_frown'))}")
```

- [ ] **Step 2: 跑测试,确认它失败**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_micro_expression_window.py -v`
Expected: FAIL —— `TypeError: detect() takes 2 positional arguments but 3 were given`

- [ ] **Step 3: 写实现**

把 `face_expression/core/analysis/micro_expression.py` 改为:

```python
from collections import deque
import numpy as np
from ...models.features import AUFeatures, MicroExpressionResult

# 窗长按**秒**定义,不按帧数定义(M2.5 spec §5.3)。
#
# 为什么必须这样:实时路径的 fps 是实测且逐会话可变的。旧的 `maxlen=15` 在 1 fps 下
# 是 15 秒、在 30 fps 下是 0.5 秒 —— 同一个常数两种含义。时间窗在两条路径下语义唯一。
#
# ⚠️ 副作用(账本已记):实时 1 fps 时 1.5 秒窗只有 ~2 帧,于是 `_MIN_FRAMES` 那道门
# 永远过不去 —— 微表情在实时路径**不再产出**。那不是本改动的 bug,而是采集率太低
# 被暴露出来。提高采集率见 spec §8.1 / §10。
_WINDOW_MS = 1500

# 窗内至少要有这么多帧才评估。保持与改动前同量级(旧实现要求 10 帧)。
_MIN_FRAMES = 10

# 校准期:丢掉最前面这么多帧(数据还没稳)。按时间给,理由同上。
_CALIBRATION_MS = 1500


class MicroExpressionDetector:
    def __init__(self, window_ms: int = _WINDOW_MS):
        self.window_ms = window_ms
        # 值为 (timestamp_ms, au_value);按时间修剪,所以 deque 不设 maxlen。
        self.au_history = {au: deque() for au in [
            'au4_frown', 'au7_eye_squeeze', 'au15_mouth_down'
        ]}
        self._first_ts = None

    def _window(self, au_name: str) -> list:
        """窗内这一路的 AU 取值(按时间升序)。测试直接用,故公开下划线名。"""
        return [v for _, v in self.au_history[au_name]]

    def _prune(self, timestamp_ms: int) -> None:
        cutoff = timestamp_ms - self.window_ms
        for dq in self.au_history.values():
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def detect(self, current_au_values: AUFeatures, timestamp_ms: int) -> MicroExpressionResult:
        if self._first_ts is None:
            self._first_ts = timestamp_ms

        au_dict = {k: v for k, v in current_au_values.__dict__.items() if k in self.au_history}

        self._prune(timestamp_ms)
        for au_name in self.au_history:
            if au_name in au_dict:
                self.au_history[au_name].append((timestamp_ms, au_dict[au_name]))

        if timestamp_ms - self._first_ts < _CALIBRATION_MS:
            return MicroExpressionResult(data={})

        micro_exps = {}
        for au_name in self.au_history:
            series = self._window(au_name)
            if len(series) < _MIN_FRAMES:
                continue

            baseline = np.mean(series[:-5])
            std_dev = np.std(series[:-5]) or 0.01
            current_val = series[-1]

            dynamic_threshold = baseline + 1.5 * std_dev
            min_activation = 0.1

            if current_val > dynamic_threshold and current_val > min_activation:
                recent_series = series[-8:]
                duration = sum(1 for v in recent_series if v > dynamic_threshold)
                if 2 <= duration <= 8:
                    micro_exps[au_name] = {
                        'intensity': round(float(current_val), 3),
                        'duration_frames': duration,
                        'onset_frame': len(series) - duration
                    }

        return MicroExpressionResult(data=micro_exps)
```

⚠️ 只把**窗口**改成时间制;窗内的 `series[:-5]` / `series[-8:]` / `2 <= duration <= 8` 仍是**帧数**,刻意不动 —— 那属于 M3 的逐列重构(spec §1.2)。这里要的就是「窗是 1.5 秒」这一个事实成立。

- [ ] **Step 4: 跑测试,确认它通过**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_micro_expression_window.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
cd ~/jingxin
git add face_expression/core/analysis/micro_expression.py tests/test_micro_expression_window.py
git commit -m "feat(m2.5): 微表情窗改 1.5 秒时间窗(帧数窗在实时路径语义不唯一)"
```

---

## Task 4: `VideoPipeline` 时间基(核心)

**Files:**
- Modify: `face_expression/pipeline/video_pipeline.py`(多处,见下)
- Test: `tests/test_video_pipeline_timebase.py`(新建)
- Modify: `tests/test_face_detector_wiring.py:19,39,52,65`(替身与构造签名)

**Interfaces:**
- Consumes: Task 2 的 `detector.detect(image, timestamp_ms)`;Task 3 的 `micro_detector.detect(au, timestamp_ms)`
- Produces: `VideoPipeline(session_id="default", save_landmarks=False, detector=None)`(**无 `fps`**);`process_frame(image_rgb, timestamp_ms: int)`;`_update_blink_state(ear: float, timestamp_ms: int) -> tuple[bool, int, float]`;`_measured_fps() -> float`。Task 5 / 6 / 7 调用。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_video_pipeline_timebase.py
"""M2.5:管线的每一处时间都由 `timestamp_ms` 导出,且 `is_blink` 真的进序列化。

不 import mediapipe:探测器注入假件,替身交 `None` 走 no_face 分支,于是不必造 468 个 landmark
就能验时间逻辑 —— 时间逻辑被抽成了可直接调用的 `_update_blink_state`。
"""

import numpy as np
import pytest

from face_expression.pipeline.video_pipeline import VideoPipeline

_FRAME = np.zeros((48, 48, 3), dtype=np.uint8)


class _FakeDetector:
    def __init__(self):
        self.seen = []
        self.resets = 0
        self.closed = False

    def detect(self, image_rgb, timestamp_ms):
        self.seen.append(timestamp_ms)
        return None                    # 没检出脸 —— 走 no_face 分支

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


def test_pipeline_forwards_the_timestamp_to_the_detector():
    """接线:调用方给的时间戳必须原样到达探测器。

    红法:`process_frame` 里自己造一个时间戳(或忽略入参)。
    """
    d = _FakeDetector()
    p = VideoPipeline(session_id="s", detector=d)
    for ts in (0, 1000, 2000):
        p.process_frame(_FRAME, ts)

    assert d.seen == [0, 1000, 2000], d.seen


def test_blink_count_uses_real_seconds_not_the_wall_clock():
    """★ `blink_rate_per_min`(实为「60 秒窗内的次数」)必须按**真实时间**算。

    造法:0/500/1000/1500 ms 四帧都闭眼(ear < 0.21),去抖 0.3 s 放行每一次 ——
    相邻间隔都是 0.5 s > 0.3 s,所以四次全部计入,期望 [1, 2, 3, 4]。

    红法:把去抖与 60 秒窗改回 `time.time()`(见 `video_pipeline.py:59,62,66`)——
    四次调用在挂钟上只隔几微秒,`now - last_blink_time > 0.3` 只有第一次成立,
    于是结果退化成 [1, 1, 1, 1],红。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    counts = [p._update_blink_state(0.10, ts)[1] for ts in (0, 500, 1000, 1500)]

    assert counts == [1, 2, 3, 4], f"眨眼计数没按真实时间窗算:{counts}"


def test_eye_closed_seconds_uses_the_real_gap_between_frames():
    """★ `eye_closed_sec` 累加的是**相邻帧的真实间隔**,不是 `1/fps`。

    造法:间隔刻意不均匀(300 / 600 / 100 ms)。

    红法:`self.eye_closed_duration += 1 / self.fps`(见 `video_pipeline.py:70`)——
    三次合计会是 3/30 = 0.1 s 而不是 1.0 s。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    p._update_blink_state(0.10, 0)          # 第 1 帧:起点,不计时长
    p._update_blink_state(0.10, 300)        # 闭眼,记 0.3
    p._update_blink_state(0.10, 900)        # 闭眼,记 0.6
    _, _, eye_closed = p._update_blink_state(0.10, 1000)   # 闭眼,记 0.1

    assert abs(eye_closed - 1.0) < 1e-6, f"应当是 1.0 秒(0.3+0.6+0.1),实际 {eye_closed}"


def test_no_history_yet_gives_zero_not_an_exception():
    """★ Review Focus ③:`no_face` 帧不推进历史 —— 一帧都没进过时 `get_summary` 不许抛。

    红法:在 `get_summary` 里无条件读 `self.au_history[0]`。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    p.process_frame(_FRAME, 0)              # 没检出脸 → au_history 仍为空

    s = p.get_summary()
    assert s["frame_count"] == 0
    assert s["duration_sec"] == 0


def test_single_frame_session_has_zero_duration():
    """★ Review Focus ⑤:只有一帧时 `duration_sec` = 0,不许抛、不许 NaN。"""
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    p.au_history.append(_stub_au(timestamp_ms=1234))

    s = p.get_summary()
    assert s["duration_sec"] == 0, s["duration_sec"]
    assert s["frame_count"] == 1


def test_duration_comes_from_timestamps_not_frame_count():
    """`duration_sec` = 末帧 − 首帧(秒),不是 `帧数/fps`。

    红法:把它改回 `frame_count / self.fps` —— 三帧会给出 0.1 而不是 9.5。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector())
    for ts in (0, 4000, 9500):
        p.au_history.append(_stub_au(timestamp_ms=ts))

    assert p.get_summary()["duration_sec"] == 9.5


def _stub_au(timestamp_ms: int):
    """历史帧的最小替身:`get_summary` 与时间窗只用这两样。"""
    from types import SimpleNamespace
    return SimpleNamespace(timestamp_ms=timestamp_ms, focus_score=0.5)
```

- [ ] **Step 2: 跑测试,确认它失败**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_video_pipeline_timebase.py -v`
Expected: FAIL —— `TypeError: process_frame() takes 2 positional arguments` / `AttributeError: '_update_blink_state'`

- [ ] **Step 3: 改 `__init__`(删 `fps`,历史窗改时间制)**

`face_expression/pipeline/video_pipeline.py:15-37` 替换为:

```python
class VideoPipeline:
    def __init__(self, session_id="default", save_landmarks=False, detector=None):
        # `fps` 参数已删除(M2.5 spec §5.1):它以前同时是「元数据」和「计时依据」,
        # 而那个计时依据实测错了 30 倍(§3.1)。现在时间只剩一个来源 —— 入参 timestamp_ms。
        # 元信息里仍报 fps,但那是**滑动实测值**(见 `_measured_fps`)。
        self.session_id = session_id
        self.blink_times = []          # 单位:秒(会话相对),不是挂钟
        self.eye_closed_duration = 0.0
        self.last_blink_time = None
        self.EAR_THRESHOLD = 0.21

        # 历史窗按**秒**定,不按帧数定(旧的是 `int(3 * fps)` = 90 帧)。
        # 实时 1 fps 下 90 帧是 90 秒,与「最近 3 秒」差了 30 倍(spec §3.3)。
        self.history_window_ms = 3000
        self.n_submitted = 0
        self.first_ts = None
        self.last_ts = None

        if detector is None:
            from .detector import FaceDetector
            from ..config import FACE_MODEL
            detector = FaceDetector(FACE_MODEL)
        self.detector = detector

        self.feature_calculator = AUFeatureCalculator(save_landmarks=save_landmarks)
        self.micro_detector = MicroExpressionDetector()
        self.tension_engine = TensionEngine()
        self.emotion_engine = EmotionEngine()
        self.au_history = collections.deque()      # 按时间修剪,不设 maxlen
```

- [ ] **Step 4: 改 `process_frame` 签名与时间逻辑**

`process_frame(self, image_rgb)` → `process_frame(self, image_rgb, timestamp_ms: int)`,并把开头的探测改成:

```python
    def process_frame(self, image_rgb, timestamp_ms: int):
        if self.first_ts is None:
            self.first_ts = timestamp_ms
        self.last_ts = timestamp_ms
        self.n_submitted += 1

        h, w = image_rgb.shape[:2]
        landmarks_norm = self.detector.detect(image_rgb, timestamp_ms)
```

把 `:57-78` 那段眨眼/闭眼/深拷贝逻辑整体替换为对一个新方法的调用:

```python
        ear = current_au.avg_ear
        is_blink, blink_count, eye_closed_sec = self._update_blink_state(ear, timestamp_ms)

        au_for_history = copy.deepcopy(current_au)
        au_for_history.is_blink = is_blink
        au_for_history.blink_rate_per_min = blink_count
        au_for_history.eye_closed_sec = eye_closed_sec
        # 历史修剪要靠它,所以挂一个不在 `__annotations__` 里的属性 ——
        # 于是它不会被当成特征字段卷进时序统计(:107 那个列表按 `__annotations__` 取)。
        au_for_history.timestamp_ms = timestamp_ms

        # ★ M2.5 修复:序列化用的是 `current_au`(:172 `au_features=current_au`),
        # 而这三个字段以前只写在深拷贝上 → CSV 里 is_blink 三列恒 0(spec §3.4)。
        current_au.is_blink = is_blink
        current_au.blink_rate_per_min = blink_count
        current_au.eye_closed_sec = eye_closed_sec
```

紧随其后加时间窗修剪(放在 `self.au_history.append(au_for_history)` 之前):

```python
        cutoff = timestamp_ms - self.history_window_ms
        while self.au_history and getattr(self.au_history[0], "timestamp_ms", 0) < cutoff:
            self.au_history.popleft()
```

把 `:155-164` 的微表情调用改成 `self.micro_detector.detect(current_au, timestamp_ms)`。

把 `:170` 的 `timestamp=current_time` 改成 `timestamp=timestamp_ms / 1000.0`(单位仍是**秒**,与旧列同量纲,只是基准变成会话相对时间 —— 这是 M2.5 唯一改变 `timestamp` 列语义的地方,下游影响已核,见 spec §3.6)。

新增这个纯函数方法(可被 Step 1 的测试直接调用):

```python
    def _update_blink_state(self, ear: float, timestamp_ms: int):
        """眨眼/闭眼记账。全部按 `timestamp_ms` 算 —— 不再碰挂钟。

        为什么必须抽出来:旧代码把这件事和 `time.time()` 缠在 `process_frame` 里,
        离线批处理下那个挂钟是**处理时间**(实测中位是真时长的 2.32 倍),
        于是「每分钟眨眼次数」算的是处理时间里的次数(spec §3.3)。
        """
        now_sec = timestamp_ms / 1000.0
        is_blink = ear < self.EAR_THRESHOLD

        if is_blink and (self.last_blink_time is None
                         or (now_sec - self.last_blink_time) > 0.3):
            self.blink_times.append(now_sec)
            self.last_blink_time = now_sec

        one_minute_ago = now_sec - 60.0
        blink_count = sum(1 for t in self.blink_times if t > one_minute_ago)

        if ear < 0.18:
            # 真实间隔 = 与上一帧的时间差;第一帧没有上一帧,记 0
            if self.history_last_ms is not None:
                self.eye_closed_duration += (timestamp_ms - self.history_last_ms) / 1000.0
        else:
            self.eye_closed_duration = 0.0

        self.history_last_ms = timestamp_ms
        return is_blink, blink_count, self.eye_closed_duration
```

并在 `__init__` 里补 `self.history_last_ms = None`。

- [ ] **Step 5: 改 `get_summary` 的两处时长 + `fps` 元信息**

`:258-265` 替换为:

```python
        # 时长一律由时间戳导出,不再 `帧数 / fps`(spec §3.3)。
        first = getattr(self.au_history[0], "timestamp_ms", 0)
        last = getattr(self.au_history[-1], "timestamp_ms", 0)
        duration_sec = (last - first) / 1000.0
        duration_min = duration_sec / 60.0
        recent_blinks = sum(1 for t in self.blink_times
                            if (last / 1000.0 - t) < 60.0)
        avg_blink_rate = (recent_blinks / max(duration_min, 0.01)
                          if duration_min >= 0.1 else recent_blinks / 1.0)
```

把 `:207`、`:274` 的 `"fps": self.fps` 改成 `"fps": self._measured_fps()`,并新增:

```python
    def _measured_fps(self) -> float:
        """到当前为止的**滑动实测**帧率 —— 元信息用,不是特征列(spec §5.5)。"""
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        elapsed_sec = (self.last_ts - self.first_ts) / 1000.0
        if elapsed_sec <= 0:
            return 0.0
        return round(self.n_submitted / elapsed_sec, 3)
```

- [ ] **Step 6: 改 `tests/test_face_detector_wiring.py`**

替身 `_FakeDetector.detect(self, image_rgb)` → `detect(self, image_rgb, timestamp_ms)`;三处 `VideoPipeline(fps=30, session_id="s", detector=d)` → `VideoPipeline(session_id="s", detector=d)`;两处 `p.process_frame(_FRAME)` → `p.process_frame(_FRAME, 0)`(第二条改成 `p.process_frame(_FRAME, 1000)` 更能守住时间戳被透传)。

- [ ] **Step 7: 跑测试,确认通过**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_video_pipeline_timebase.py tests/test_face_detector_wiring.py -v`
Expected: PASS

- [ ] **Step 8: 提交**

```bash
cd ~/jingxin
git add face_expression/pipeline/video_pipeline.py tests/test_video_pipeline_timebase.py tests/test_face_detector_wiring.py
git commit -m "feat(m2.5): VideoPipeline 时间戳改入参;眨眼/闭眼/时长全按真实时间;is_blink 写回序列化对象"
```

---

## Task 5: face 服务接线(会话时钟 + 忽略 `?fps=`)

**Files:**
- Modify: `face_expression/api/app.py:119-140`(`get_or_create_pipeline`)、`:180-230`(端点)
- Test: `tests/test_analyze_session_fallback.py:115-138`(替身跟着真接口长)

**Interfaces:**
- Consumes: Task 1 的 `SessionClock`;Task 4 的 `VideoPipeline(session_id=..., detector=None)` 与 `process_frame(image, timestamp_ms)`
- Produces: `get_or_create_pipeline(session_id: str) -> VideoPipeline`(**不再收 `fps`**)

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_face_session_clock.py
"""M2.5:face 服务的时钟接线。spec §5.4 / Review Focus ②。"""

import inspect

from face_expression.api import app as face_app


def test_query_fps_is_no_longer_accepted_by_the_pipeline_factory():
    """★ spec §5.4:`?fps=` 不再进管线 —— 它是错的 30 倍的那个数(spec §3.1)。

    红法:把 `fps` 参数加回 `get_or_create_pipeline`。
    """
    params = inspect.signature(face_app.get_or_create_pipeline).parameters
    assert "fps" not in params, (
        f"get_or_create_pipeline 又收 fps 了 —— 那会重新变成计时依据:{list(params)}")


def test_each_session_gets_its_own_clock():
    """★ Review Focus ②:时钟必须**按会话**,不能是模块级单例。

    否则两个会话共享一个起点,各自的 `timestamp_ms` 会互相错位。
    红法:把 `session_clocks` 改成模块级一个 `SessionClock()`。
    """
    face_app.session_clocks.clear()
    a = face_app._clock_for("20260925_120000_aaaa")
    b = face_app._clock_for("20260925_120000_bbbb")

    assert a is not b
    assert face_app._clock_for("20260925_120000_aaaa") is a, "同一会话应复用同一个时钟"


def test_reset_rewinds_the_clock_with_the_pipeline():
    """★ Review Focus ②(核心):`/reset` 必须把时钟和管线**一起**归零。

    只归零管线不归零时钟,同一 id 的第二段会话时间戳会接着上一段涨。
    红法:`reset` 只调 `pipeline.reset()`。
    """
    face_app.session_clocks.clear()
    face_app.session_pipelines.clear()
    sid = "20260925_120000_cccc"

    clock = face_app._clock_for(sid)
    clock.stamp_ms()
    clock.stamp_ms()
    assert clock.stamp_ms() > 0, "前提不成立:时钟应当已经走过一段"

    face_app._reset_session(sid)
    assert clock.stamp_ms() == 0, "reset 之后时钟没归零"
```

- [ ] **Step 2: 跑测试,确认它失败**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_face_session_clock.py -v`
Expected: FAIL —— `AttributeError: module ... has no attribute 'session_clocks'`

- [ ] **Step 3: 实现**

`face_expression/api/app.py`:在 `session_pipelines = {}` / `session_loggers = {}`(:47-48)旁边加:

```python
from session_clock import SessionClock

session_clocks = {}   # 每个会话一个相对时钟(M2.5 spec §4)


def _clock_for(session_id: str) -> SessionClock:
    """会话时钟。与 pipeline 同生命周期 —— 但**分开存**,因为 TTL 回收时两者都要动。"""
    if session_id not in session_clocks:
        session_clocks[session_id] = SessionClock()
    return session_clocks[session_id]
```

`get_or_create_pipeline` 改成:

```python
def get_or_create_pipeline(session_id: str) -> VideoPipeline:
    """获取或创建 VideoPipeline 实例。

    `fps` 参数已删除(M2.5 spec §5.4):它以前来自客户端 `?fps=30`,而客户端实际
    只发 1 帧/秒 —— 那个数把时间量整体抬了 30 倍。现在时间由调用方的会话时钟给。
    """
```

函数体里 `pipeline = VideoPipeline(fps=fps, session_id=session_id)` → `VideoPipeline(session_id=session_id)`;TTL 回收那段里 `session_loggers.pop(sid, None)` 旁边加 `session_clocks.pop(sid, None)`。

端点里把 `fps: int = 30` 形参**保留**(前端还在发,删了会 422)但标记为忽略:

```python
        fps: int = 30
):
    """
    ...
        fps: **已忽略**(保留形参只为兼容旧客户端)。实时时间由服务端实测,
             因为客户端申报的 30 与实发的 1 帧/秒差了 30 倍(spec §5.4)。
    """
    if fps != 30:
        logger.warning("收到 ?fps=%s —— 已忽略。时间由服务端实测(spec §5.4)", fps)
```

并把 `:227-230` 改成:

```python
            pipeline = get_or_create_pipeline(session_id)
            timestamp_ms = _clock_for(session_id).stamp_ms()

            # 处理帧
            result_obj, mesh_results, features_dict = pipeline.process_frame(
                image_rgb, timestamp_ms)
```

新增 `_reset_session(sid)`(把 `/reset` 端点里重复的"归零管线"那段收成一处,时钟一起归零):

```python
def _reset_session(session_id: str) -> None:
    """`/reset`:管线与时钟**必须一起**归零,否则第二段会话第一帧就是回退值。"""
    entry = session_pipelines.get(session_id)
    if entry is not None:
        pipeline, _ = entry
        pipeline.reset()
    _clock_for(session_id).reset()
```

把 `/reset` 端点里原来直接调 `pipeline.reset()` 的地方改为调 `_reset_session(session_id)`。

- [ ] **Step 3b: 记账 `measured_fps`(进服务日志)**

在 TTL 回收那段(`session_loggers.pop(sid, None)` 旁)与 `_reset_session()` 里,回收/归零**之前**各记一行:

```python
        logger.info("会话 %s 收尾:实测 fps=%.3f(spec §5.5)", sid, pipeline._measured_fps())
```

⚠️ **`session.json` 那一半本轮不做**(这是对 spec §5.5 的收紧,已在 spec 里注明):清单由 `voice_interaction/asr/transcript_store.py:186` 的 `ensure_manifest` 独占,而它的语义是「会话开始写一次、已存在即不动」——face/gesture 要写进去就得引入跨模块 import 并动 `refresh_manifest`,而后者至今**零生产调用方**(账本 §3 第 3 条,M2 刻意没碰)。为一个本轮不进特征列的协变量付这个耦合不划算。留到 M3(协变量升为特征列时一并处理)。

- [ ] **Step 4: 改 `tests/test_analyze_session_fallback.py` 的替身**

`:121-128` 的 `_FakeFacePipeline` 改成(它的 docstring 自己就写着「替身必须跟着真接口长」):

```python
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id
        self.closed = False

    def process_frame(self, _image_rgb, timestamp_ms):
        return object(), None, {"timestamp": timestamp_ms / 1000.0, "focus_score": 0.5,
                                "dominant_emotion": "neutral", "confidence": 0.5}
```

- [ ] **Step 5: 跑测试,确认通过**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_face_session_clock.py tests/test_analyze_session_fallback.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
cd ~/jingxin
git add face_expression/api/app.py tests/test_face_session_clock.py tests/test_analyze_session_fallback.py
git commit -m "feat(m2.5): face 服务按会话持有相对时钟;忽略 ?fps= 并记警告"
```

---

## Task 6: gesture 服务接线

**Files:**
- Modify: `gesture_analysis/api/app.py`(`session_clocks` + `_clock_for` + `_reset_session`;`:263`、`:274` 两处 `detect` 调用)

**Interfaces:**
- Consumes: Task 1 的 `SessionClock`;Task 2 的 `HandDetector.detect(image, timestamp_ms)` / `PoseDetector.detect(image, timestamp_ms)`
- Produces: 无(端点契约不变)

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_gesture_session_clock.py
"""M2.5:gesture 服务的时钟接线。gesture 连 `?fps=` 参数都没有(spec §3.5),所以这里只验时钟。

⚠️ 「时间戳真的传进了探测器」这一条**刻意不写在这里** —— 它由既有的
`tests/test_analyze_session_fallback.py` 用假探测器真跑端点来压(见本任务 Step 4),
比字符串比对源码结实得多:那句断言在改动换行/重命名变量时就会碎,而它想守的东西
其实一点没变(账本 §4.1「测试通过 ≠ 有约束力」的同一类毛病)。
"""

from gesture_analysis.api import app as gesture_app


def test_each_session_gets_its_own_clock():
    """按会话,不是单例。红法:改成模块级一个 `SessionClock()`。"""
    gesture_app.session_clocks.clear()
    a = gesture_app._clock_for("20260925_120000_aaaa")
    b = gesture_app._clock_for("20260925_120000_bbbb")

    assert a is not b
    assert gesture_app._clock_for("20260925_120000_aaaa") is a, "同一会话应复用同一个时钟"
```

- [ ] **Step 2: 跑测试,确认它失败**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_gesture_session_clock.py -v`
Expected: FAIL —— `AttributeError: module ... has no attribute 'session_clocks'`

- [ ] **Step 3: 实现**

按 Task 5 的同一形状:加 `from session_clock import SessionClock`、`session_clocks = {}`、`_clock_for()`、`_reset_session()`;在 `:256` 附近取 `timestamp_ms = _clock_for(session_id).stamp_ms()`;把 `:263`、`:274` 改成 `detect(image_rgb, timestamp_ms)`;TTL / `/reset` 回收处一并 `session_clocks.pop(sid, None)` / `_clock_for(sid).reset()`。

⚠️ gesture 的 `get_or_create_detectors`(`:180-190`)构造探测器时**没有**传 fps(它从 `MEDIAPIPE_CONFIG` 展开),所以删掉 `fps` 形参这里无需改动 —— 但 Step 6 的全量测试会证明这一点。

- [ ] **Step 4: 让既有端点测试的假探测器真压住「时间戳递到了」**

`tests/test_analyze_session_fallback.py` 的 `_FakeGestureDetector`(约 `:140`)改成:构造里加 `self.timestamps = []`,并在 `detect` 开头记一笔、新增第二个位置参数:

```python
    def __init__(self, *args, **kwargs):
        self.timestamps = []
        # …原文件里其余的初始化原样保留

    def detect(self, image_rgb, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return <原文件本来的返回逻辑,原样保留,形状一个字都不改>
```

⚠️ **返回形状必须原样保住**(`HandDetector.detect` → `[]`,`PoseDetector.detect` → `None`)—— 那是这个文件自己的 docstring 反复强调的前提;shape 一改,端点后半段的真实逻辑就不跑了,这个测试也就不再测它要测的接线。

再在该文件里加一条断言(用既有的端点调用脚手架):两帧之后,所有假探测器看到的 `timestamps` **严格递增**。

红法:把 `gesture_analysis/api/app.py:263`/`:274` 的第二个实参去掉 —— 端点 TypeError。

- [ ] **Step 5: 跑测试,确认通过**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_gesture_session_clock.py tests/test_gesture_detector_wiring.py tests/test_analyze_session_fallback.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
cd ~/jingxin
git add gesture_analysis/api/app.py tests/test_gesture_session_clock.py
git commit -m "feat(m2.5): gesture 服务按会话持有相对时钟并把时间戳传进探测器"
```

---

## Task 7: 离线 `extract_features.py` 对齐

**Files:**
- Modify: `experiments/extract_features.py`(`:84`、`:86-88`、`:130`、`:251`、`:294-300`、`:309`)
- Test: `tests/test_offline_timestamp.py`(新建)

**Interfaces:**
- Consumes: Task 4 的 `VideoPipeline(session_id=..., detector=None)` 与 `process_frame(image, timestamp_ms)`
- Produces: `offline_timestamp_ms(k: int, frame_skip: int, src_fps: float) -> int`(纯函数,可单测)

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_offline_timestamp.py
"""M2.5:离线路径的时间戳 = `帧序号 × FRAME_SKIP / src_fps`(spec §4 表)。"""

import pytest

from experiments.extract_features import offline_timestamp_ms


def test_timestamp_is_frame_index_times_skip_over_src_fps():
    """30 fps 源、每 3 帧取 1 帧 → 每个提交帧相隔 100 ms。

    红法:传 `src_fps` 而不是 `src_fps / FRAME_SKIP`(等于沿用旧行为)——
    第 3 帧会给出 100 而不是 300。
    """
    assert offline_timestamp_ms(0, 3, 30.0) == 0
    assert offline_timestamp_ms(1, 3, 30.0) == 100
    assert offline_timestamp_ms(3, 3, 30.0) == 300


def test_zero_or_nan_src_fps_falls_back_instead_of_dividing_by_zero():
    """★ Review Focus ④:容器读不出帧率时不许除零、不许产 NaN。

    红法:去掉守卫 —— 第一个断言会 ZeroDivisionError,
    第二个会把 `nan` 一路写进 CSV。
    """
    assert offline_timestamp_ms(1, 3, 0.0) == 100        # 回退 30 fps
    assert offline_timestamp_ms(1, 3, float("nan")) == 100


def test_returns_int():
    """时间戳必须是 int(mediapipe 吃 int;float 会在某些帧率下退化成同一个值)。"""
    assert isinstance(offline_timestamp_ms(7, 3, 29.97), int)
```

- [ ] **Step 2: 跑测试,确认它失败**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_offline_timestamp.py -v`
Expected: FAIL —— `ImportError: cannot import name 'offline_timestamp_ms'`

- [ ] **Step 3: 实现**

`experiments/extract_features.py`,在 `FRAME_SKIP = 3`(:45)下方加:

```python
def offline_timestamp_ms(k: int, frame_skip: int, src_fps: float) -> int:
    """离线路径的时间戳:第 k 个**提交**帧距视频开头的毫秒数。

    为什么是 `frame_skip / src_fps` 而不是 `1 / src_fps`:提交的是每 `frame_skip`
    帧里的第 1 帧,所以相邻两个提交帧在**视频时间**上相隔 `frame_skip / src_fps` 秒。
    旧代码把 `fps=30` 直接传给管线,等于声称每个提交帧相隔 33 ms,真实是 100 ms
    —— 时间量整体错 3 倍(M2.5 spec §3.2)。
    """
    if not src_fps or src_fps != src_fps:      # 0 / NaN 都回退
        src_fps = 30.0
    return int(round(k * frame_skip * 1000.0 / src_fps))
```

并把两处 `process_frame` 调用改成带时间戳:两个 Extractor 类(`:86-88` 与 `:130`)的 `process_frame(self, rgb)` 改成 `process_frame(self, rgb, timestamp_ms)`,各自把 `timestamp_ms` 转交给内部的 `VideoPipeline` / gesture 分析器;主循环 `:294-310` 维护一个只对**提交帧**自增的计数器 `k`:

```python
            if frame_idx % FRAME_SKIP == 0:
                ts_ms = offline_timestamp_ms(k, FRAME_SKIP, src_fps)
                k += 1
                fr = face_ext.process_frame(rgb, ts_ms)
                ...
                gs = ges_ext.process_frame(rgb, ts_ms)
```

同时把 `:251` 的 `FaceExtractor(fps=30)` 改成 `FaceExtractor()`(fps 参数已随 Task 4 消失),并让 `:84` 的 `VideoPipeline(fps=fps, session_id="batch", ...)` 变成 `VideoPipeline(session_id="batch", ...)`。

- [ ] **Step 4: 跑测试,确认通过**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_offline_timestamp.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
cd ~/jingxin
git add experiments/extract_features.py tests/test_offline_timestamp.py
git commit -m "feat(m2.5): 离线提取改按 帧序号×FRAME_SKIP/src_fps 给时间戳"
```

---

## Task 8: 两个 example 调用点跟上

**Files:**
- Modify: `face_expression/examples/run_video_analyzer.py:110`
- Modify: `gesture_analysis/examples/main_integrator.py:208`

**Interfaces:**
- Consumes: Task 4 的 `process_frame(image, timestamp_ms)`
- Produces: 无

- [ ] **Step 1: 确认它们是死的但确实会坏**

Run: `cd ~/jingxin && grep -rn "run_video_analyzer\|main_integrator" --include='*.py' . | grep -v examples/ | grep -v code_data_supplement`
Expected: **无输出** —— 没人 import 它们。但它们仍是仓库里的可运行入口,签名一变就 TypeError,所以一并修。

- [ ] **Step 2: 改**

两处都是"读视频文件逐帧分析"的循环,各自维护自己的提交帧计数 `k` 与视频 fps,按 Task 7 的公式给时间戳:

```python
    # 这两个示例读的是**视频文件**,所以走离线公式(与 experiments/extract_features.py 同源)。
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    k = 0
    ...
        result_obj, results, features = analyzer.process_frame(frame_rgb, int(round(k * 1000.0 / src_fps)))
        k += 1
```

(两处都不做 `FRAME_SKIP` 抽稀,所以 `frame_skip = 1`。)

- [ ] **Step 3: 确认没把别的调用点漏掉**

Run: `cd ~/jingxin && grep -rn "process_frame(" --include='*.py' . | grep -v __pycache__ | grep -v code_data_supplement`
Expected: 每个调用点都带第二个参数。

- [ ] **Step 4: 提交**

```bash
cd ~/jingxin
git add face_expression/examples/run_video_analyzer.py gesture_analysis/examples/main_integrator.py
git commit -m "chore(m2.5): 两个 example 的 process_frame 调用跟上新签名"
```

---

## Task 9: 反向复现 + 端到端验收 + 账本

**Files:**
- Modify: `docs/下一步.md`(§0 / §3 第 15、16 条的依赖 / §6 里程碑表)
- Create: `docs/superpowers/sdd/2026-09-25-m2-5-timebase-fixes/progress.md`

**Interfaces:**
- Consumes: Task 1–8 的全部产出
- Produces: 账本(下一个会话的入口)

- [ ] **Step 1: 反向复现(spec §7.2,账本 §4.2 的标准动作)**

逐条撤掉生产改动,确认对应测试**红在对的地方**,再恢复。至少做这四条,每条把命令与原始输出抄进账本:

| 撤掉什么 | 哪个测试必须红 |
|---|---|
| `SessionClock.stamp_ms()` 的「抬到 `_last_ms + 1`」两行 | `test_stamp_is_strictly_increasing_even_within_one_millisecond` |
| `detector.detect()` 里的回退 `raise` | `test_detector_rejects_a_timestamp_that_goes_backwards` |
| `_update_blink_state` 换成旧的 `1 / self.fps` 累加 | `test_eye_closed_seconds_uses_the_real_gap_between_frames` |
| `get_summary` 的 `duration_sec` 换回 `frame_count / 30` | `test_duration_comes_from_timestamps_not_frame_count` |

- [ ] **Step 2: 全量测试 + 合并门**

```bash
cd ~/jingxin
~/miniconda3/envs/jingxin/bin/python -m pytest -q
cd experiments/duration_audit
PY=~/miniconda3/envs/jingxin/bin/python
$PY reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_m25
$PY reaggregate_normalized.py --verify-legacy /tmp/legacy_m25
```

Expected: 全部通过;合并门 **0 / 2835510**。
⚠️ 合并门这条**不是本设计的证据**(它复现已有 CSV,采集代码怎么改都绿)—— 记账时照 spec §7.4 写明它只是"确认没有误伤"。

- [ ] **Step 3: 端到端(真会话,spec §7.3)**

按 `docs/下一步.md` §2 的步骤起三个服务,用前端或 `~/shared/start_all.sh` 跑一场真会话(前端 1 帧/秒),然后:

```bash
~/miniconda3/envs/jingxin/bin/python - <<'EOF'
import pandas as pd, glob
f = sorted(glob.glob('/home/huihuibuhui/jingxin/data/logs/face_au_log_*.csv'))[-1]
df = pd.read_csv(f)
span = df['timestamp'].max() - df['timestamp'].min()
print(f"{f}\n  行数={len(df)}  时间戳跨度={span:.2f}s")
print(f"  is_blink 非零={int((df['is_blink'] != 0).sum())} / {len(df)}")
EOF
```

Expected:
- 时间戳跨度与墙钟实测跨度差 **< 10%**;
- `measured_fps`(服务日志里)落在 **[0.5, 2.0]**;
- **`is_blink` 列不再是恒 0**(§3.4 的修复生效)。若仍然恒 0,说明闭眼阈值在这段素材上没触发 —— 那不是回归,但要在账本里说清是哪种情形。

- [ ] **Step 4: 写账本**

建 `docs/superpowers/sdd/2026-09-25-m2-5-timebase-fixes/progress.md`,记:逐任务裁决、反向复现的原始输出、端到端三项数字、以及 deferred minor 清单。
⚠️ 目录**直接建在 git 跟踪的路径下**(吸取 M1.5「复审报告落在 gitignore 目录里差点丢」的教训)。

- [ ] **Step 5: 改账本 `docs/下一步.md`**

- §3 第 15 条(VIDEO 等价性门)与第 16 条(阈值/量程重登记):补一句「**基线已移动** —— M2.5 改了喂给 mediapipe 的时间戳,`2026-09-25-m2-5-*` 之前量的位移表已过期,必须在新时间基上重测」。
- §6 里程碑表:`M2.5 时间基修复 ← ✅ 已并入 main`,并把 M3 那行的依赖写成「M2.5」。
- §0:一句话现状更新到"M2.5 完成,下一步 M3"。
- **§4.6 的规矩照旧:历史记录不改,只加。**

- [ ] **Step 6: 提交**

```bash
cd ~/jingxin
git add docs/下一步.md docs/superpowers/sdd/2026-09-25-m2-5-timebase-fixes/progress.md
git commit -m "docs(m2.5): 账本、反向复现记录与下一步口径"
git push origin main
```

---

## 完成后

M2.5 落地后,**M3(L0 逐列重构)才能开始** —— 它要的就是一个可信的「秒」。M3 的 spec 需另开一轮 brainstorming(本计划不含 M3 的任何内容)。

**本计划不解决、也不假装解决的**(spec §8,记账时必须一起带上):

1. **实时 1 fps 下时序族更退化** —— 时间修对了,同时暴露采集率太低。提高采集率是前端改动,不在本计划。
2. **VIDEO 模式等价性门仍未验**,且本计划让它的基线再次移动。
3. 前端仍在发 `?fps=30`(一个不再被相信的谎)。
