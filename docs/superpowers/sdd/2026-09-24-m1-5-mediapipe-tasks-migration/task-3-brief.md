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


@pytest.fixture
def fake_detectors(monkeypatch):
    """把真探测器换成假的。

    为什么必须有这个 fixture:`get_or_create_detectors` 会构造真的 HandDetector /
    PoseDetector —— 那要加载 21 MB 的 `.task` 模型(慢),而且会让**没下模型的环境整片红**。
    测试不该依赖那 21 MB 的存在。
    """
    import gesture_analysis.core.detectors as det

    class _Fake:
        def __init__(self, *a, **k):
            self.closed = False

        def detect(self, image_rgb):
            return []

        def reset(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(det, "HandDetector", _Fake)
    monkeypatch.setattr(det, "PoseDetector", _Fake)
    return _Fake


def test_importing_the_app_does_not_touch_mediapipe(gapp):
    """D3 的副产品也是它的证明:import 期不许构造探测器。

    红法:把 `hands = HandLandmarker(...)` 挪回模块级 —— 在没装 mediapipe(或
    mediapipe 1.0 无 solutions)的环境里,这个 import 会直接炸,本测试先红。
    """
    assert not hasattr(gapp, "hands"), "模块级探测器回来了 —— 那正是并发污染的形态"
    assert not hasattr(gapp, "pose"), "模块级探测器回来了"


def test_two_sessions_get_different_detectors(gapp, fake_detectors):
    """spec D3 / Review Focus 4:两会话必须是不同探测器对象。

    红法:探测器做成模块级单例(迁移前的形态)—— VIDEO 模式会让两会话互相污染跟踪。
    """
    gapp.session_analyzers.clear()
    gapp.detectors.clear()
    a = gapp.get_or_create_detectors("t_a")
    b = gapp.get_or_create_detectors("t_b")

    assert a["hands"] is not b["hands"]
    assert a["pose"] is not b["pose"]


def test_expired_sessions_close_their_detectors(gapp, fake_detectors):
    """spec §6.3 / Review Focus 3:TTL 回收时必须 close 探测器,不然泄漏 native 句柄。

    红法:回收循环只 `del session_analyzers[sid]`(迁移前的样子),不碰探测器
    → `closed` 为空 → 断言失败。

    注意:`fake_detectors` 造出来的假件自带 `closed` 标志,所以这里**不用**再 monkeypatch
    `.close` —— 直接读标志更接近真实(也避免测到"我替换掉的那个方法")。
    """
    import time

    gapp.session_analyzers.clear()
    gapp.detectors.clear()
    dets = gapp.get_or_create_detectors("t_old")
    hands, pose = dets["hands"], dets["pose"]

    # 先给它建一条分析器记录,再把最后使用时间推到 TTL 之外(回收是挂在分析器表上的)
    gapp.get_or_create_analyzers("t_old")
    analyzers, _ = gapp.session_analyzers["t_old"]
    gapp.session_analyzers["t_old"] = (analyzers, time.time() - gapp.SESSION_TIMEOUT - 1)

    gapp.get_or_create_detectors("t_new")

    assert hands.closed and pose.closed, "过期会话的探测器没被 close —— native 句柄泄漏"
    assert "t_old" not in gapp.detectors, "过期会话没从探测器表里移除"
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
