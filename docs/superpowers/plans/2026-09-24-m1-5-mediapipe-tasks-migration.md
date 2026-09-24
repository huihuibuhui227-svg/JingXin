# M1.5 face/gesture 迁 mediapipe tasks API —— 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `face_expression` 与 `gesture_analysis` 两个服务在 mediapipe 1.0.0 上重新可用(现在一个每帧静默 500、一个进程起不来),并把换探测器的数值位移量清楚、把受影响的门槛逐条指名。

**Architecture:** 两个模块**各自**新增一个探测器封装(不共享 —— 理由见 spec §4),把 `mp.solutions` 换成语义等价的 `tasks` 探测器;**gesture 的探测器从模块级单例改为按会话**(spec D3),与既有的 `session_analyzers` 同生命周期、同 TTL 回收。几何层(~3,002 行,零 mediapipe 引用)一行不动。

**Tech Stack:** Python 3.11(`~/miniconda3/envs/jingxin`)、mediapipe 1.0.0 `tasks` API、FastAPI/uvicorn、pytest。

**Spec:** `docs/superpowers/specs/2026-09-24-m1-5-mediapipe-tasks-migration-design.md`(实施者**必须同时读它**;本计划是它的落地,不重复它的论证)

## Global Constraints

- **解释器固定** `~/miniconda3/envs/jingxin/bin/python`。**不可**用 `~/huihui/bin/python`(缺 websockets 等)。
- **模型文件**:`<repo>/models/mediapipe/{face_landmarker,hand_landmarker,pose_landmarker_full}.task`,路径由 config 读取,**不硬编码**在探测器代码里。
- **档位必须对齐**:生产 `model_complexity=1` ↔ `pose_landmarker_full.task`(spec D5;用 lite 会得出 42 px / 545 px 的假象)。
- **版本钉法**:URL 用的是 `…/float16/latest/…`,`latest` 会漂移 —— 这是**已知未验证项**(spec §6.1),不许在代码注释里写成"已钉版本"。
- **不动的**:几何层(`au_calculator` / `landmarks` / `*_analyzer` / `*_feature_extractor` / `emotion_engine` / `tension_engine` / `micro_expression`);`examples/*`、`utils/visualization.py`、`pipeline/gesture_pipeline.py`、`experiments/extract_features.py`;`session_id` 的写侧契约。
- **`hand_id == 0 → left_hand` 的位置判断不修**(spec §9 第 7 条)—— 修它会混进迁移、让 delta 无法归因。
- **每个测试都要能说出"哪个生产改动会让它变红"**;说不出来就是没约束力(仓库教训,见 spec 上游 `docs/下一步.md` §4)。
- **提交前必须跑全量**:`~/miniconda3/envs/jingxin/bin/python -m pytest -q`(基线 177 passed)。

## Review Focus

以下是 spec 蕴含、但**没有任何任务的测试会覆盖**、且最可能咬到真实使用者的五类情形。每条都在它归属的任务里加了钉住它的测试:

1. **同会话连发两帧 → 时间戳必须严格递增。** 实现者很可能每帧从 0 起算或复用同一个 ts → mediapipe 抛错或跟踪错乱。归属 T2/T3。
2. **`/reset` 之后同一会话再来一帧 → 不许因为时间戳回退而抛错。** 帧计数器必须随 `/reset` 归零。归属 T3(face 的 `/session/{sid}/reset` 归 T2)。
3. **会话过 TTL 回收时探测器必须被 `close()`。** 只从 dict 里删掉会泄漏 native 资源 —— 而**现有 TTL 逻辑只删分析器**。归属 T2/T3。
4. **两个会话并发时拿到的是不同的探测器对象。** gesture 现在这条会红(模块级单例);这是 D3 的全部意义。归属 T3。
5. **模型文件缺失时必须启动失败,而不是启动成功、第一次请求才炸。** 这正是 face 现在"哑死"的形态。归属 T1。
6. ★ **把 gesture 的 landmarks 摊平成元组 → 分析器静默回落到默认分。** 实测(2026-09-24,`HandAnalyzer` / `ArmAnalyzer`):喂元组**不抛异常**,只是 `is_valid=False`、分数变成默认的 `50.0`;喂带 `.x/.y` 的对象才 `is_valid=True` 并给出真值(51.47 / 90.0)。**这是"报告里印着一个数、而它什么都不代表"最纯粹的形态** —— 归属 T1(封装)与 T3(消费点),两处都要有钉子。

---

## 文件结构

| 文件 | 动作 | 职责 |
|---|---|---|
| `face_expression/pipeline/detector.py` | **新建** | `FaceDetector`(按会话实例、VIDEO 模式、自持帧计数器)+ `verify_models()`。**模块级不 import mediapipe**(延迟到工厂调用) |
| `gesture_analysis/core/detectors.py` | **新建** | `HandDetector` / `PoseDetector` + `verify_models()`,同上 |
| `face_expression/config.py` | 改 | 加 `MEDIAPIPE_MODELS_DIR`;**删掉死代码 `MEDIAPIPE_CONFIG`**(spec §1) |
| `gesture_analysis/config.py` | 改 | `MEDIAPIPE_CONFIG` 按 spec §6.2 改写 + 模型路径键 |
| `face_expression/pipeline/video_pipeline.py` | 改 | 删惰性 `face_mesh` 属性,改为构造注入 `FaceDetector`;`process_frame` 换 `detect()` |
| `face_expression/api/app.py` | 改 | `get_or_create_pipeline` 里建探测器 + TTL 回收一起 `close()`;`/session/{sid}/reset` 重置;`__main__` 自检探针改指 `verify_models()` |
| `gesture_analysis/api/app.py` | 改 | 删模块级 `mp.solutions`/`Hands`/`Pose`;`detectors[sid]` 表;TTL 与 `/reset` 一起回收;两处 `process()` 换 `detect()` |
| `tests/test_detector_contract.py` | **新建** | T1 的契约测(不依赖真 mediapipe) |
| `tests/test_face_detector_wiring.py` | **新建** | T2 |
| `tests/test_gesture_detector_wiring.py` | **新建** | T3 |
| `tests/test_analyze_session_fallback.py` | 改 | 替身从 `mp.solutions` 改为**不再需要**(探测器不再在 import 期构造);加 `import mediapipe` 守卫 |
| `requirements.txt` / `requirements-full.txt` | 改 | `mediapipe>=0.8.0` → `mediapipe==1.0.0` |
| `.gitignore` | 改 | 加 `models/mediapipe/` |

---

### Task 1: 契约层(探测器封装 + 配置 + 钉版本)

**这是 T2/T3 的前置,必须先落地并冻结。**

**Files:**
- Create: `face_expression/pipeline/detector.py`
- Create: `gesture_analysis/core/detectors.py`
- Modify: `face_expression/config.py`(加模型目录;删 `MEDIAPIPE_CONFIG`)
- Modify: `gesture_analysis/config.py:56-69`(`MEDIAPIPE_CONFIG` 改写)
- Modify: `requirements.txt:13`、`requirements-full.txt:14`、`.gitignore`
- Test: `tests/test_detector_contract.py`

**Interfaces:**
- Consumes: 无(本任务是底座)
- Produces:
  - `face_expression.config.MEDIAPIPE_MODELS_DIR: Path`、`face_expression.config.FACE_MODEL: Path`
  - `face_expression.pipeline.detector.FaceDetector(model_path, *, fps=30, num_faces=1, min_detection_confidence=0.8, min_tracking_confidence=0.8, factory=None)`
    - `.detect(image_rgb: np.ndarray) -> list[tuple[float, float]] | None`
    - `.reset() -> None`、`.close() -> None`
  - `face_expression.pipeline.detector.verify_models(model_path=None, factory=None) -> None`(缺失/加载失败 → `RuntimeError`)
  - `gesture_analysis.config.MEDIAPIPE_MODELS_DIR: Path`、`HAND_MODEL: Path`、`POSE_MODEL: Path`
  - `gesture_analysis.core.detectors.HandDetector(...)` / `PoseDetector(...)` / `verify_models(...)` —— 与 face 同形,但 **`.detect()` 交出的是 mediapipe 的 landmark 对象本身,不是元组**:
    - `HandDetector.detect(image_rgb) -> list[list[NormalizedLandmark]]`(0–2 只手)
    - `PoseDetector.detect(image_rgb) -> list[NormalizedLandmark] | None`
    - ⚠️ **这个不对称是有实测依据的,不是随手**:face 的下游 `au_calculator` 吃 `(x, y)` 元组(`video_pipeline.py:52` 现在就摊平);gesture 的下游分析器吃**带 `.x/.y` 的对象**。把 gesture 也摊平成元组会**静默失效** —— 见下面那条实测。

- [ ] **Step 1: 让 `.gitignore` 与 requirements 先落地(便宜且无依赖)**

编辑 `.gitignore`,追加一行(放在文件末尾):

```
models/mediapipe/
```

编辑 `requirements.txt` 第 13 行与 `requirements-full.txt` 第 14 行,把 `mediapipe>=0.8.0` 改成:

```
mediapipe==1.0.0
```

- [ ] **Step 2: 写失败测试(探测器封装的契约)**

创建 `tests/test_detector_contract.py`:

```python
# tests/test_detector_contract.py
"""M1.5 契约测:探测器封装。spec §6.3/§6.4/§8。

本文件**不 import mediapipe** —— 封装的全部行为都能用一个注入的假工厂验完。
这既是"模块级不 import mediapipe"这条设计的证明,也让这些测试跑得飞快。
"""

from pathlib import Path

import numpy as np
import pytest

from face_expression.pipeline.detector import FaceDetector, verify_models
from gesture_analysis.core.detectors import HandDetector, PoseDetector

_FRAME = np.zeros((48, 48, 3), dtype=np.uint8)


class _FakeLandmarker:
    """假的 tasks 探测器:记下收到的每个 timestamp_ms,返回可配置的结果。"""

    def __init__(self, points_per_group=0, groups=1):
        self.timestamps = []
        self.closed = False
        self._points = points_per_group
        self._groups = groups

    def _result(self):
        pt = type("P", (), {"x": 0.5, "y": 0.5})
        return [pt() for _ in range(self._points)]

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        groups = [self._result() for _ in range(self._groups)] if self._points else []
        return type("R", (), {
            "face_landmarks": groups, "hand_landmarks": groups,
            "pose_landmarks": groups})()

    def close(self):
        self.closed = True


def _factory(holder, **kwargs):
    """返回一个 (factory, holder) —— holder['l'] 拿到被造出来的假探测器。"""
    def build(**opts):
        holder["opts"] = opts
        holder["l"] = _FakeLandmarker(**kwargs)
        return holder["l"]
    return build


def test_face_detector_flattens_to_xy_pairs():
    """face 的契约:`.detect()` 交出 `[(x, y)]` —— 它的下游 `au_calculator` 吃元组
    (`video_pipeline.py:52` 现在就摊平)。

    红法:让它返回点对象(`au_calculator` 的下标运算会 `TypeError`)。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=3))
    out = d.detect(_FRAME)

    assert out == [(0.5, 0.5)] * 3, out


def test_gesture_detectors_keep_x_and_y_attributes():
    """★ gesture 的契约**与 face 相反**:交出 landmark 对象,不摊平。

    这是 2026-09-24 实测出来的静默失效,不是口味问题:分析器收到 `(x, y)` 元组
    **不抛异常**,只是 `is_valid=False`、分数回落到默认的 `50.0` —— 报告里于是印着
    一个数,而它什么都不代表。实测对照:

        HandAnalyzer <- 元组 (x,y)    : is_valid=False  resilience=50.0
        HandAnalyzer <- 带 .x/.y 对象 : is_valid=True   resilience=51.466…
        ArmAnalyzer  <- 元组          : is_valid=False  arm_score=50.0
        ArmAnalyzer  <- 对象          : is_valid=True   arm_score=90.0

    红法:把 `_Base._groups` 改成 `[[(p.x, p.y) for p in g] for g in groups]`。
    """
    h = {}
    d = HandDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=21))
    groups = d.detect(_FRAME)

    assert groups, "前提不成立:假探测器应当返回一组点"
    assert hasattr(groups[0][0], "x") and hasattr(groups[0][0], "y"), (
        "landmarks 被摊平成了元组 —— 分析器会静默回落到默认分"
        "(实测 is_valid=False, score=50.0,且不抛异常、不留日志)")

    h2 = {}
    p = PoseDetector(Path("/nonexistent.task"), factory=_factory(h2, points_per_group=33))
    pose = p.detect(_FRAME)
    assert pose and hasattr(pose[0], "x"), "姿态的 landmarks 也被摊平了"


def test_timestamps_are_strictly_increasing_within_a_session():
    """spec §6.4 / Review Focus 1:同一会话连发两帧,ts 必须严格递增。

    红法:每帧都从 0 起算(或复用同一个值)—— mediapipe 的 VIDEO 模式会抛错。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), fps=10, factory=_factory(h, points_per_group=1))
    for _ in range(3):
        d.detect(_FRAME)

    ts = h["l"].timestamps
    assert ts == sorted(set(ts)), f"时间戳非严格递增:{ts}"
    assert ts[0] == 0 and ts[1] == 100, f"应按 1000/fps 步进(10 fps → 100ms):{ts}"


def test_reset_rewinds_the_frame_counter():
    """spec §6.4 / Review Focus 2:`reset()` 之后计数归零,且**不抛错**。

    红法:把 `reset()` 写成空函数 —— 摄像头重启后 ts 回退,mediapipe 抛错。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), fps=10, factory=_factory(h, points_per_group=1))
    d.detect(_FRAME)
    d.detect(_FRAME)
    d.reset()
    d.detect(_FRAME)

    assert h["l"].timestamps == [0, 100, 0], h["l"].timestamps


def test_close_releases_the_underlying_landmarker():
    """spec §6.3 / Review Focus 3:`close()` 必须真的转发到探测器(native 资源)。

    红法:把 `close()` 写成空函数 —— TTL 回收时泄漏 native 句柄。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=1))
    d.detect(_FRAME)
    d.close()

    assert h["l"].closed is True


def test_两会话拿到不同的探测器对象(tmp_path):
    """spec D3 / Review Focus 4:按会话 —— 两个实例必须各自持有探测器。

    红法:把探测器做成模块级单例(这正是 gesture 迁移前的形态)。
    """
    h1, h2 = {}, {}
    a = HandDetector(Path("/nonexistent.task"), factory=_factory(h1, points_per_group=1))
    b = HandDetector(Path("/nonexistent.task"), factory=_factory(h2, points_per_group=1))
    a.detect(_FRAME)
    b.detect(_FRAME)

    assert h1["l"] is not h2["l"], "两个会话共用了同一个探测器实例"


def test_verify_models_fails_loudly_when_the_file_is_missing(tmp_path):
    """spec §8 / Review Focus 5:模型缺失 → **启动即失败**,不是推迟到第一次请求。

    红法:把 `verify_models()` 写成空函数(或只打日志)—— 那就回到 face 现在
    "启动正常、每帧 500"的哑死法。
    """
    missing = tmp_path / "not_here.task"
    with pytest.raises(RuntimeError) as ei:
        verify_models(missing, factory=_factory({}, points_per_group=1))

    assert str(missing) in str(ei.value), "错误信息里必须含期望路径,否则排障时不知道去哪找"


def test_verify_models_accepts_a_present_file(tmp_path):
    """另一侧:`verify_models` 能通过时不许抛(否则服务永远起不来)。

    红法:让 `verify_models` 无条件抛。
    """
    real = tmp_path / "ok.task"
    real.write_bytes(b"stub")
    verify_models(real, factory=_factory({}, points_per_group=1))    # 不抛即通过
```

- [ ] **Step 3: 跑测试,确认它红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_detector_contract.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'face_expression.pipeline.detector'`(feature missing)

- [ ] **Step 4: 写 `face_expression/pipeline/detector.py`**

```python
"""人脸探测器封装:mediapipe `tasks` 的 FaceLandmarker(spec §4/§6.3/§6.4)。

**模块级不 import mediapipe** —— 延迟到工厂函数里。两个理由:
  1. `gesture_analysis/api/app.py` 与 `face_expression/api/app.py` 在 import 期会被
     测试加载,而测试不该因为机器上没装 mediapipe 就整片死掉(见 spec §10.2);
  2. 让"探测器怎么造"成为**可注入的一等参数**,契约测不需要 mediapipe 也不需要模型。

为什么按会话实例(而不是像 gesture 迁移前那样做模块级单例):`tasks` 的 VIDEO 模式把
跟踪状态挂在探测器实例上 —— 单例会让两个并发会话互相污染跟踪(spec §3.5)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def _default_factory(model_path: Path, *, num_faces: int,
                     min_detection_confidence: float, min_tracking_confidence: float):
    """真的造一个 FaceLandmarker。**mediapipe 只在这里被 import。**"""
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence))


class FaceDetector:
    """一个会话一个人脸探测器。`detect()` 交出 `[(x, y)]` 或 `None`(没检出)。"""

    def __init__(self, model_path: Path, *, fps: int = 30, num_faces: int = 1,
                 min_detection_confidence: float = 0.8,
                 min_tracking_confidence: float = 0.8,
                 factory: Callable | None = None):
        self.model_path = Path(model_path)
        self.fps = fps
        self._frame_index = 0
        self._factory = factory or _default_factory
        self._landmarker = self._factory(
            self.model_path, num_faces=num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def detect(self, image_rgb: np.ndarray):
        """VIDEO 模式:时间戳由本会话的帧计数导出,天然单调(spec §6.4)。"""
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect_for_video(
            image, int(self._frame_index * 1000 / self.fps))
        self._frame_index += 1
        if not result.face_landmarks:
            return None
        return [(p.x, p.y) for p in result.face_landmarks[0]]

    def reset(self) -> None:
        """`/session/{sid}/reset` 要调:帧计数归零,否则下一个会话的时间戳会回退。"""
        self._frame_index = 0

    def close(self) -> None:
        """释放 native 句柄。TTL 回收时必须调,否则泄漏(spec §6.3)。"""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def verify_models(model_path: Path | None = None,
                  factory: Callable | None = None) -> None:
    """启动自检:模型文件在不在、能不能加载。**缺失即抛**(spec §8)。

    为什么必须是"启动即失败":face 现在的哑死法正是"启动看着正常、每帧 500"——
    故障被推迟到第一次请求,而那时客户端只看到一个 500,排障的人不知道去哪找。
    """
    from ..config import FACE_MODEL

    path = Path(model_path) if model_path is not None else FACE_MODEL
    if not path.exists():
        raise RuntimeError(
            f"人脸模型文件不存在:{path}\n"
            f"  期望位置是 <repo>/models/mediapipe/,下载方式见 spec §6.1。")

    build = factory or _default_factory
    landmarker = build(path, num_faces=1,
                       min_detection_confidence=0.8, min_tracking_confidence=0.8)
    landmarker.close()
```

- [ ] **Step 5: 跑测试,确认人脸那几条绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_detector_contract.py -q`
Expected: 人脸相关的 4 条 PASS;gesture 那几条仍 FAIL(`ModuleNotFoundError: gesture_analysis.core.detectors`)

- [ ] **Step 6: 写 `gesture_analysis/core/detectors.py`**

```python
"""手势/姿态探测器封装:mediapipe `tasks` 的 HandLandmarker / PoseLandmarker。

与 `face_expression/pipeline/detector.py` **同形但各持一份**,刻意不共享 ——
理由与 Ruling M1-2 同:三个包各自持有守卫副本是有意为之,跨包 import 会把整条依赖链
拉起来并让依赖方向反转。这里重复的代价(~15 行)小于耦合的代价(spec §4)。

**模块级不 import mediapipe** —— 同 face 那份的理由,见它的模块 docstring。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def _default_hand_factory(model_path: Path, *, num_hands: int,
                          min_detection_confidence: float, min_tracking_confidence: float):
    import mediapipe as mp  # noqa: F401  (与下面 create_from_options 同批 import)
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence))


def _default_pose_factory(model_path: Path, *, num_poses: int,
                          min_detection_confidence: float, min_tracking_confidence: float):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence))


class _Base:
    """两个封装的共同部分(fps / 帧计数 / close)。子类只需给 `_factory` 与 `_build_kwargs`。"""

    _result_attr = ""

    def __init__(self, model_path: Path, *, fps: int = 30, factory: Callable | None = None,
                 **build_kwargs):
        self.model_path = Path(model_path)
        self.fps = fps
        self._frame_index = 0
        self._landmarker = (factory or self._default_factory)(self.model_path, **build_kwargs)

    def _timestamp_ms(self) -> int:
        return int(self._frame_index * 1000 / self.fps)

    def _bump(self) -> None:
        self._frame_index += 1

    def _groups(self, result):
        """**原样交出 landmark 对象,不做任何转换。**

        ⚠️ 别"顺手"摊平成 `(x, y)` 元组。2026-09-24 实测:分析器收到元组**不抛异常**,
        只是 `is_valid=False`、分数回落到默认的 `50.0` —— 报告里于是印着一个数,而它
        什么都不代表。这是本项目一路在杀的那种静默失效,连日志都不会留。

        `tasks.NormalizedLandmark` 与 `solutions` 的点对象一样暴露 `.x/.y/.z/.visibility`,
        所以下游四个分析器 + 四个特征抽取器**一行都不用改**(spec §3.3)。
        """
        return getattr(result, self._result_attr) or []

    def reset(self) -> None:
        """`/reset` 要调:帧计数归零(spec §6.4)。"""
        self._frame_index = 0

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


class HandDetector(_Base):
    """手部探测器。`detect()` 交出 landmark **对象**分组(0–2 只手,每只 21 个点)。

    为什么不是 `[(x, y)]`:分析器要 `.x/.y` 属性,元组会静默失效 —— 见 `_groups` 的说明。
    """

    _result_attr = "hand_landmarks"
    _default_factory = staticmethod(_default_hand_factory)

    def __init__(self, model_path: Path, *, fps: int = 30, num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 factory: Callable | None = None):
        super().__init__(model_path, fps=fps, factory=factory, num_hands=num_hands,
                         min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence)

    def detect(self, image_rgb: np.ndarray):
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect_for_video(image, self._timestamp_ms())
        self._bump()
        return self._groups(result)


class PoseDetector(_Base):
    """姿态探测器。`detect()` 交出 landmark **对象**列表(33 个点)或 `None`。"""

    _result_attr = "pose_landmarks"
    _default_factory = staticmethod(_default_pose_factory)

    def __init__(self, model_path: Path, *, fps: int = 30, num_poses: int = 1,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.6,
                 factory: Callable | None = None):
        super().__init__(model_path, fps=fps, factory=factory, num_poses=num_poses,
                         min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence)

    def detect(self, image_rgb: np.ndarray):
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect_for_video(image, self._timestamp_ms())
        self._bump()
        groups = self._groups(result)
        return groups[0] if groups else None


def verify_models(hand_path: Path | None = None, pose_path: Path | None = None,
                  factory: Callable | None = None) -> None:
    """启动自检(spec §8)。两个模型都要能加载,否则抛。"""
    from ..config import HAND_MODEL, POSE_MODEL

    hp = Path(hand_path) if hand_path is not None else HAND_MODEL
    pp = Path(pose_path) if pose_path is not None else POSE_MODEL
    for path in (hp, pp):
        if not path.exists():
            raise RuntimeError(
                f"手势/姿态模型文件不存在:{path}\n"
                f"  期望位置是 <repo>/models/mediapipe/,下载方式见 spec §6.1。")

    build = factory
    if build is None:
        _default_hand_factory(hp, num_hands=2, min_detection_confidence=0.7,
                              min_tracking_confidence=0.5).close()
        _default_pose_factory(pp, num_poses=1, min_detection_confidence=0.6,
                              min_tracking_confidence=0.6).close()
    else:
        build(hp, num_hands=2, min_detection_confidence=0.7,
              min_tracking_confidence=0.5).close()
```

> ⚠️ 实施者注意:`tests/test_detector_contract.py` 里那条 `test_verify_models_fails_loudly_when_the_file_is_missing` 用的是 `verify_models(missing, factory=...)` —— face 那个签名是 `(model_path, factory)`,gesture 的是 `(hand_path, pose_path, factory)`。**两条 `verify_models` 签名不同是有意的**(face 一个模型、gesture 两个)。

- [ ] **Step 7: 改两个 config**

`face_expression/config.py`:在 `MEDIAPIPE_CONFIG` 那段(第 27 行起)整体**删除**,替换为:

```python
# 模型文件目录。路径在这里定义、由探测器读取,**不硬编码在抽取代码里**
# (spec §6.1):M3 重排特征时不该动到探测器路径。
MEDIAPIPE_MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'mediapipe')
FACE_MODEL = os.path.join(MEDIAPIPE_MODELS_DIR, 'face_landmarker.task')
```

同时删掉第 69 行那条注释掉的 `MEDIAPIPE_CONFIG[...]` 覆盖(它引用的键已经不存在了)。

`gesture_analysis/config.py`:把第 56-69 行的 `MEDIAPIPE_CONFIG` 整体替换为:

```python
# 模型文件目录(与 face 同一棵树,但两个模块各持一份路径常量 —— spec §4)
MEDIAPIPE_MODELS_DIR = PROJECT_ROOT / "models" / "mediapipe"
HAND_MODEL = MEDIAPIPE_MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MEDIAPIPE_MODELS_DIR / "pose_landmarker_full.task"

# ⚠️ 与旧 config 的差别**不是改名,是换概念**(spec §6.2):
#   * static_image_mode(布尔)→ running_mode(三态,封装内部固定 VIDEO)
#   * max_num_hands        → num_hands
#   * model_complexity     → **由加载哪个 .task 文件决定**:0/1/2 ↔ lite/full/heavy。
#     生产用的是 1,所以 POSE_MODEL 指向 full。用 lite 会和旧实现差出 42px 的假象
#     (spec D5、§3.4 的实测教训)。
MEDIAPIPE_CONFIG: Dict[str, Dict[str, Any]] = {
    'hands': {
        'num_hands': 2,
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.5
    },
    'pose': {
        'tier': 'full',
        'num_poses': 1,
        'min_detection_confidence': 0.6,
        'min_tracking_confidence': 0.6
    }
}
```

- [ ] **Step 8: 跑契约测与全量**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_detector_contract.py -q`
Expected: 全 PASS

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest -q`
Expected: 177 + 新增条数 passed,**0 红**。若有红,先查是不是别的模块 import 了被删掉的 `MEDIAPIPE_CONFIG`:

```bash
grep -rn "MEDIAPIPE_CONFIG" --include="*.py" face_expression/ gesture_analysis/ | grep -v examples/
```

- [ ] **Step 9: 提交**

```bash
git add .gitignore requirements.txt requirements-full.txt \
        face_expression/config.py face_expression/pipeline/detector.py \
        gesture_analysis/config.py gesture_analysis/core/detectors.py \
        tests/test_detector_contract.py
git commit -m "feat(m1.5): 探测器封装与配置契约(模型路径/按会话/帧计数)"
```

---

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

`process_frame` 的头两行改成(其余**一行不动**):

```python
    def process_frame(self, image_rgb):
        h, w = image_rgb.shape[:2]
        landmarks_norm = self.detector.detect(image_rgb)

        if not landmarks_norm:
            return None, None, {"emotion": "no_face"}
```

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

### Task 3: gesture 适配

**Files:**
- Modify: `gesture_analysis/api/app.py:22,42-46,49,121-159,230-255,433-457`
- Modify: `tests/test_analyze_session_fallback.py:48-80`
- Test: `tests/test_gesture_detector_wiring.py`

**Interfaces:**
- Consumes: `HandDetector` / `PoseDetector` / `verify_models`(T1)
- Produces: `gesture_analysis.api.app.get_or_create_detectors(session_id) -> dict[str, Any]`,键为 `'hands'` / `'pose'`;`detectors` 表与 `session_analyzers` 同 TTL

- [ ] **Step 1: 写失败测试**

创建 `tests/test_gesture_detector_wiring.py`。**这个文件要 import 真 app 模块**(这是 D3 的直接后果:探测器不再在 import 期构造,所以 import 不再需要 mediapipe):

```python
# tests/test_gesture_detector_wiring.py
"""M1.5 T3:gesture 侧的接线。Review Focus 3/4 + D3 的全部意义。

注意本文件**能 import 真 app 模块且不需要 mediapipe** —— 这正是 D3 的副产品:
探测器从模块级单例改成按会话之后,import 期不再构造任何探测器。
如果哪天这条 import 又开始要 mediapipe,说明有人把探测器挪回模块级了。
"""

import importlib

import numpy as np
import pytest


@pytest.fixture
def gapp():
    return importlib.import_module("gesture_analysis.api.app")


def test_importing_the_app_does_not_touch_mediapipe(gapp):
    """D3 的副产品也是它的证明:import 期不许构造探测器。

    红法:把 `hands = HandLandmarker(...)` 挪回模块级 —— 在没装 mediapipe(或
    mediapipe 1.0 无 solutions)的环境里,这个 import 会直接炸,本测试先红。
    """
    assert not hasattr(gapp, "hands"), "模块级探测器回来了 —— 那正是并发污染的形态"
    assert not hasattr(gapp, "pose"), "模块级探测器回来了"


def test_two_sessions_get_different_detectors(gapp):
    """spec D3 / Review Focus 4:两会话必须是不同探测器对象。

    红法:探测器做成模块级单例(迁移前的形态)—— VIDEO 模式会让两会话互相污染跟踪。
    """
    gapp.session_analyzers.clear()
    gapp.detectors.clear()
    a = gapp.get_or_create_detectors("t_a")
    b = gapp.get_or_create_detectors("t_b")

    assert a["hands"] is not b["hands"]
    assert a["pose"] is not b["pose"]


def test_expired_sessions_close_their_detectors(gapp, monkeypatch):
    """spec §6.3 / Review Focus 3:TTL 回收时必须 close 探测器,不然泄漏 native 句柄。

    红法:回收循环只 `del session_analyzers[sid]`(迁移前的样子),不碰探测器。
    """
    gapp.session_analyzers.clear()
    gapp.detectors.clear()
    dets = gapp.get_or_create_detectors("t_old")
    closed = []
    dets["hands"].close = lambda: closed.append("hands")
    dets["pose"].close = lambda: closed.append("pose")
    # 把这个会话的最后使用时间推到 TTL 之外
    import time
    analyzers, _ = gapp.session_analyzers["t_old"]
    gapp.session_analyzers["t_old"] = (analyzers, time.time() - gapp.SESSION_TIMEOUT - 1)

    gapp.get_or_create_detectors("t_new")

    assert sorted(closed) == ["hands", "pose"], f"探测器没被释放:{closed}"
```

- [ ] **Step 2: 跑测试,确认它红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_gesture_detector_wiring.py -q`
Expected: FAIL / ERROR at import — `AttributeError: module 'mediapipe' has no attribute 'solutions'`(这正是要消灭的那个崩溃)

- [ ] **Step 3: 改 `gesture_analysis/api/app.py`**

删掉第 22 行的 `import mediapipe as mp`,把第 42-46 行整段删掉,在 `SESSION_TIMEOUT = 300` 之后加:

```python
# 按会话的探测器表(与 session_analyzers 同生命周期、同 TTL 回收 —— spec §6.3)。
# 迁移前 `hands`/`pose` 是模块级单例,所有客户端共用;tasks 的 VIDEO 模式把跟踪状态
# 挂在探测器实例上,单例会让并发会话互相污染跟踪(spec §3.5)。
detectors: dict = {}
```

在 `get_or_create_analyzers` 里,TTL 回收段(`:124-131`)改成:

```python
    for sid in expired_sessions:
        del session_analyzers[sid]
        session_loggers.pop(sid, None)
        dets = detectors.pop(sid, None)
        if dets is not None:
            for d in dets.values():
                d.close()          # 必须显式关:持 native 句柄(spec §6.3)
```

在同一函数末尾(`return` 之前)加一行 `get_or_create_detectors(session_id)`,或让 `_analyze` 各调一次 —— **实施者选后者更直白**(见下一步)。

新增函数(放在 `get_or_create_analyzers` 之后):

```python
def get_or_create_detectors(session_id: str):
    """取或建本会话的探测器。VIDEO 模式的跟踪状态挂在实例上,所以必须按会话(spec D3)。"""
    current_time = time.time()

    expired = [sid for sid, (_, last) in session_analyzers.items()
               if current_time - last > SESSION_TIMEOUT]
    for sid in expired:
        dets = detectors.pop(sid, None)
        if dets is not None:
            for d in dets.values():
                d.close()

    if session_id not in detectors:
        from gesture_analysis.core.detectors import HandDetector, PoseDetector
        from gesture_analysis.config import HAND_MODEL, POSE_MODEL
        detectors[session_id] = {
            'hands': HandDetector(HAND_MODEL, **{
                k: v for k, v in MEDIAPIPE_CONFIG['hands'].items()}),
            'pose': PoseDetector(POSE_MODEL, **{
                k: v for k, v in MEDIAPIPE_CONFIG['pose'].items() if k != 'tier'}),
        }
        logger.info(f"创建手势探测器会话: {session_id}")
    return detectors[session_id]
```

`/analyze` 里两处调用(th `:230`、`:243`)改成:

```python
        dets = get_or_create_detectors(session_id)

        hand_groups = dets['hands'].detect(image_rgb)
        detected_hands = 0
        hand_scores = []

        for hand_id, landmarks in enumerate(hand_groups):
            if hand_id >= 2: break
            analyzer_key = 'left_hand' if hand_id == 0 else 'right_hand'
            analyzers[analyzer_key].update(landmarks)
            hand_scores.append(analyzers[analyzer_key].get_results()['resilience_score'])
            detected_hands += 1

        pose_landmarks = dets['pose'].detect(image_rgb)
        shoulder_score = 50.0

        if pose_landmarks:
            analyzers['shoulder'].update(pose_landmarks)
            shoulder_score = analyzers['shoulder'].get_results()['shoulder_score']

        left_arm_score = 50.0
        right_arm_score = 50.0
        if pose_landmarks:
            analyzers['left_arm'].update(pose_landmarks)
            analyzers['right_arm'].update(pose_landmarks)
```

> **这里 `update()` 收到的必须是 landmark 对象(带 `.x/.y`),不是元组。**
> 2026-09-24 实测:tasks 的 `NormalizedLandmark` 与 `solutions` 的点对象暴露同样的
> `.x/.y/.z/.visibility`,所以**四个分析器 + 四个特征抽取器一行都不用改**(spec §3.3)。
> 而"顺手摊平成元组"会让它们**静默回落到默认分 50.0**(实测 `is_valid=False`)。
> T1 的封装已经按这条实现(§T1 Step 6 的 `_groups`),T3 这边只要**别再加转换**。

`/reset`(`:433-457`)改成:

```python
        if session_id:
            dets = detectors.pop(session_id, None)
            if dets is not None:
                for d in dets.values():
                    d.close()
            if session_id in session_analyzers:
                del session_analyzers[session_id]
                session_loggers.pop(session_id, None)
                return {"status": "success", "message": f"会话 {session_id} 已重置"}
            return {"status": "not_found", "message": f"会话 {session_id} 不存在"}
        else:
            for dets in detectors.values():
                for d in dets.values():
                    d.close()
            detectors.clear()
            session_analyzers.clear()
            session_loggers.clear()
            return {"status": "success", "message": "所有会话已重置"}
```

`__main__` 里在 `uvicorn.run(...)` 之前加启动自检:

```python
    from gesture_analysis.core.detectors import verify_models
    verify_models()
```

- [ ] **Step 4: 补一条消费侧的钉子(把"静默回落"这条路堵死)**

在 `tests/test_gesture_detector_wiring.py` 追加:

```python
def test_analyzer_actually_consumes_what_the_detector_produces():
    """★ 消费侧钉子:封装交出的东西必须让分析器给出 `is_valid=True`。

    T1 那条契约测钉的是"封装不摊平";这条钉的是"**摊平之后真的会坏**"——
    两条合起来才能防住"有人觉得元组更干净就改回去"。

    红法:把封装的 `_groups` 改成摊平元组 → `is_valid` 变 False、分数回落 50.0
    → 断言失败。(实测:元组不抛异常,所以**没有这条断言就没有任何东西会红**。)
    """
    from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer

    class Pt:
        def __init__(self, x, y):
            self.x, self.y = x, y

    a = HandAnalyzer(hand_id=0)
    a.update([Pt(0.3 + 0.02 * i, 0.4 + 0.01 * i) for i in range(21)])
    r = a.get_results()

    assert r["is_valid"] is True, (
        "分析器没吃下探测器给的 landmarks —— 分数会静默回落成默认的 50.0")
    assert r["resilience_score"] != 50.0, "分数恰好是默认值,很可能就是静默回落"
```

- [ ] **Step 5: 改测试替身**

`tests/test_analyze_session_fallback.py:48-80` 的 `_install_env_shims()`:**整段 `mp.solutions` 替身删掉**(探测器不再在 import 期构造)。文件顶部第 64 行的裸 `import mediapipe as mp` 一并删。若文件里还有别处依赖替身,在任务报告里列出。

同时给 `python_multipart` 那段保留(它与 mediapipe 无关,是 FastAPI 定义 `File(...)` 路由时需要的)。

- [ ] **Step 6: 跑全量**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest -q`
Expected: 0 红

- [ ] **Step 7: 实跑一次**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m gesture_analysis.api.app &
sleep 8
SID=$($HOME/miniconda3/envs/jingxin/bin/python -c "import datetime;print('t_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_bbbb')")
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8002/analyze \
  -F "file=@$HOME/shared/mp_frames/frames/frame_0009.png" -F "session_id=$SID"
ls -l ~/jingxin/data/logs/gesture_emotion_log_$SID.csv
kill %1
```

Expected: HTTP **200**;日志文件存在且首列是 `$SID`。**迁移前这个服务连进程都起不来。**

- [ ] **Step 8: 提交**

```bash
git add gesture_analysis/api/app.py tests/test_analyze_session_fallback.py \
        tests/test_gesture_detector_wiring.py
git commit -m "feat(m1.5): gesture 接入 tasks 探测器(按会话 + TTL 释放 + 启动自检)"
```

---

## 最终全分支审查(不逐任务派审查,只做这一次)

**必须是全新视角的审查者,而且判据里写死"必须实跑"** —— 不许只读 diff。理由见 spec §11:M1 的 C1 活过了六轮任务级审查,只有最终审查用**真加载器**跑一遍才抓到。

审查者必须完成:

- [ ] **金标比对(spec §10.3)**:用 `~/shared/mp_frames/` 下现有的 `probe.py` / `probe_gesture.py` 重跑新实现,**判据是逐点一致**:

```bash
cd ~/shared/mp_frames
# 先把现有基线备份,再用迁移后的代码重跑(probe.py 直接用仓库的 FaceDetector 而不是自己造)
~/miniconda3/envs/jingxin/bin/python probe.py --engine tasks
~/miniconda3/envs/jingxin/bin/python compare.py
```

> 实施者注意:`probe.py` 现在**自己构造** `FaceLandmarker`(见它的 `_detector_tasks`)。迁移后它应当改成**直接用仓库的 `FaceDetector`**,这样这条判据验的才是**适配层**(而不是探针自己的实现)。这是最终审查前的必做改动,记在任务报告里。

- [ ] **两个服务实跑**:起 8000/8002,各发一帧真图,确认 200 + 落出带 sid 的日志(Step 7 的命令)。
- [ ] **`grep -rn "mp.solutions" --include="*.py" face_expression/ gesture_analysis/ | grep -v examples/`** → 活路径为空。
- [ ] **`grep -rn "MEDIAPIPE_CONFIG" --include="*.py" face_expression/`** → 为空(死代码已删)。
- [ ] **合并门**:`cd ~/jingxin/experiments/duration_audit && $PY reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_probe_m15 && $PY reaggregate_normalized.py --verify-legacy /tmp/legacy_probe_m15` → **0 / 2835510**。
- [ ] **全量测试** green;`requirements*.txt` 已钉 `==1.0.0`。
- [ ] Review Focus 五条**逐条**对着测试确认有钉子。
- [ ] 最后更新 `docs/下一步.md`(§0 现状、§2 的 T7 前置、§3 跟进项)并把本任务的账本落进 `docs/superpowers/sdd/`。

---

## Self-Review

**1. Spec 覆盖**:spec §1 的 8 项在范围内事项 → ① 探测器 T2/T3;② 同;③ D3 → T3;④ 模型路径 → T1;⑤ config → T1;⑥ requirements → T1;⑦ 测试替身 → T3;⑧ 阈值重登记 → **spec §0 第 3 条已收窄成"指名 + 依据",具体改数留 M3**,本计划不实现数值改动,由最终审查核对 §3.4 的数据被引用。spec §10.1 的五条契约测 → T1(4 条)+ T3(1 条)。§10.4 的 VIDEO 门 → **明确留到迁移后**(plan 不实现,已在 spec 记档)。

**2. 占位符扫描**:无 TBD/TODO。T2 Step 5(b) 留了一处**要求实施者读代码后二选一并说明依据**的判断点 —— 那处的现有语义我只读到行号、没读到正文,所以给了两条具体候选代码,不是"自行处理"。(T3 原来也有一处,已在下面被实测消解。)

**3. 类型一致性**:`FaceDetector.detect -> list[tuple[float,float]] | None`;`HandDetector.detect -> list[list[NormalizedLandmark]]`;`PoseDetector.detect -> list[NormalizedLandmark] | None`。**这个不对称是实测结论,不是笔误**(见 Review Focus 6)。三处消费点(T2 Step 3、T3 Step 3)按此写。`verify_models` 两个签名不同,已在 T1 Step 6 末尾显式警告。

**4. 写这份计划时改掉的一个真 bug(记档)**:初稿里让 gesture 的封装也把 landmarks 摊平成 `(x, y)` 元组 —— 那对 face 是对的(`au_calculator` 吃元组),对 gesture 会**静默失效**。实测:分析器收元组不抛异常,只是 `is_valid=False`、分数回落成默认 `50.0`。已改成原样交出 landmark 对象,并在 T1(封装不摊平)与 T3(消费侧 `is_valid=True`)两处各加了一条钉子。**这个 bug 原本会以"计划里一句风险提示"的形式交给实施者去猜 —— 不留这种债。**
