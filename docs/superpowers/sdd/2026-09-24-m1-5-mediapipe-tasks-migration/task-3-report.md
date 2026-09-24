# Task 3 报告:gesture 适配(mediapipe `tasks`)

**状态**:DONE_WITH_CONCERNS
**提交**:`8cbe149` — `feat(m1.5): gesture 接入 tasks 探测器(按会话 + TTL 释放 + 启动自检)`
**改动文件(严格 = 任务书 Files 列表,3 个)**:
`gesture_analysis/api/app.py`(改)、`tests/test_analyze_session_fallback.py`(改)、
`tests/test_gesture_detector_wiring.py`(新增)
**`face_expression/**` 未动一个字节。**

---

## 0. 起点基线(改动前实跑)

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest -q
189 passed in 10.72s
```

`mediapipe` 版本与崩溃面(实测,不是推的):

```
$ ~/miniconda3/envs/jingxin/bin/python -c "import mediapipe as mp; print(mp.__version__, hasattr(mp,'solutions'))"
1.0.0 False
```

环境前提也都实查过,Step 7 有真东西可跑:`models/mediapipe/{hand_landmarker,pose_landmarker_full}.task`
(7.8 MB + 9.4 MB)在,`~/shared/mp_frames/frames/frame_0009.png`(370 KB)在,8002 端口空闲,
`python_multipart` **0.0.32 已装**(所以真实 multipart POST 跑得起来,不用绕)。

---

## 1. Step 1:写失败测试

创建 `tests/test_gesture_detector_wiring.py`,逐字按任务书(含 Step 4 那条第 5 个测试)。

## 2. Step 2:真的跑出红

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_gesture_detector_wiring.py -q
>   mp_hands = mp.solutions.hands
               ^^^^^^^^^^^^
E   AttributeError: module 'mediapipe' has no attribute 'solutions'
gesture_analysis/api/app.py:42: AttributeError
ERROR tests/...::test_importing_the_app_does_not_touch_mediapipe
ERROR tests/...::test_two_sessions_get_different_detectors
ERROR tests/...::test_expired_sessions_close_their_detectors
1 passed, 3 errors in 0.54s
```

红在任务书预言的那一行(`app.py:42`),错误类型逐字一致。**这就是迁移前"服务连进程都起不来"的同一个崩溃。**
第 4 个测试(消费侧钉子)此刻是绿的 —— 符合预期:它是防回归的钉子,不是新功能测试,
红法在"封装被改成摊平元组"那天(见 §4 的实测证据)。

## 3. Step 3:改 `gesture_analysis/api/app.py`

按任务书逐段落地:

| 位置 | 改动 |
|---|---|
| `:22` | 删 `import mediapipe as mp` |
| 原 `:42-46` | 删模块级 `mp_hands/mp_pose/hands/pose` 整段 |
| `SESSION_TIMEOUT = 300` 后 | 新增 `detectors: dict = {}`(带 spec §6.3/§3.5 的理由注释) |
| `get_or_create_analyzers` TTL 段 | pop 并 `close()` 探测器 |
| 新增函数 | `get_or_create_detectors(session_id) -> dict`(键 `'hands'`/`'pose'`) |
| `/analyze` | 两处调用改成 `dets = get_or_create_detectors(session_id)` + `dets['hands'].detect` / `dets['pose'].detect` |
| `/reset` | 单会话与全清两条路都 `close()` 探测器 |
| `__main__` | `verify_models()` 自检,位置在 `uvicorn.run` 之前 |

任务书给了**二选一**的那处("同函数末尾加一行,或让 `_analyze` 各调一次"),我选了后者
(`/analyze` 内显式调用)—— 即任务书自己标为"更直白"的那条。
`'tier'` 按任务书要求用 `if k != 'tier'` 显式过滤(它不是 `PoseDetector` 的构造参数)。
`get_or_create_detectors` 内是**函数体延迟 import**,`tests/` 的 monkeypatch 才拦得住(见 §5)。

★ `detect()` 交出的东西**没有任何形状转换** —— 保留 landmark 对象。这条是硬约束,
§4 有实测证据证明"顺手摊平"会静默失效。

## 4. Step 4:消费侧钉子 —— 真的跑了,并实测了"摊平真的会坏"

任务书给的钉子测试直接跑(见 §6 的绿)。为确认这条断言**有牙**(不是恒真),
我按任务书 docstring 里写的手法实测了一次真数字:

```
objects -> {'is_valid': True,  'resilience_score': 51.46625258399798}
tuples  -> {'is_valid': False, 'resilience_score': 50.0}
raises on tuples? no - silent fallback
```

所以"喂元组不抛异常、只静默回落成默认 50.0"这条**是实测事实,不是推测**;
而钉子的 `!= 50.0` 断言在真数据下确有区分力(51.47 vs 50.0)。

> 精确说清这条钉子的射程:T1 的契约测钉"封装不摊平",这条钉"分析器吃对象"。
> 它本身不经过 `_groups`,所以**单改 `_groups` 不会让这条红**;两条合起来才闭环。
> 任务书把它放在"消费侧"是准确的。

## 5. Step 5:改测试替身 —— 以及一个任务书没预见到的问题

### 5.1 删了什么(逐项)

`tests/test_analyze_session_fallback.py`:

1. `_install_env_shims()` 里 `import mediapipe as mp` **整行删**;
2. 其后的 `if not hasattr(mp, "solutions"): ... mp.solutions = types.SimpleNamespace(...)`
   **整段删**(含 `_process`/`_ctor` 两个内嵌函数与 `Hands`/`Pose` 两个假类);
3. 函数 docstring 从"补两个缝"改成"补一个缝",并写明 gesture 那个缝**已经消失**、
   少一个缝本身就是 D3 生效的证据。

保留了 `python_multipart` 那段(与 mediapipe 无关)。`types` 的 import **保留**
(`_FakeRequest` 之外,`_freeze_clock` 仍用 `types.SimpleNamespace`)。

### 5.2 Task 2 的 3 行:**保住了**

`_FakeFacePipeline` 的 `closed` 标志 + `close()` 全部原地未动:

```
154:        self.closed = False      # Task 2 加的
163:        self.closed = True       # Task 2 加的(close())
```

(124/137 是我新加的 `_FakeGestureDetector`,不冲突。)删除范围严格限定在
`_install_env_shims()` 内的 mediapipe 段,没有整段重写该文件。

### 5.3 ★ 任务书没预见到的问题:删完替身后,4 条 gesture 测试开始**加载 21 MB 真模型**

任务书 Step 5 只要求删替身,理由是"探测器不再在 import 期构造"。**这个理由成立,
但结论不完整**:`analyze_image` 现在会在**调用期**构造真探测器,而本文件是**直接调
`analyze_image`** 的 —— 于是 `gesture_env` 下真的 `HandDetector(HAND_MODEL, ...)` /
`PoseDetector(POSE_MODEL, ...)` 被构造出来。实跑证据:

```
$ pytest tests/test_analyze_session_fallback.py -q
19 passed in 32.44s          <-- 基线整个 suite 才 10.72s
Exception ignored in: <function HandLandmarker.__del__ ...>
Exception ignored in: <function PoseLandmarker.__del__ ...>
```

两件事同时暴露:(a) 本文件开始**依赖那两个 `.task` 存在** —— 而任务书自己给的
`fake_detectors` fixture docstring 明写"测试不该依赖那 21 MB 的存在",没下模型的
环境会整片红;(b) 单文件 32s,且真探测器泄漏到解释器退出才 GC(`__del__` 报错噪音)。

**处理**:替身点从"import 期"移到"构造处" —— 在 `gesture_env` 里新增
`_FakeGestureDetector`(交出"这一帧没手也没姿态"的空结果,与迁移前替身形状一致),
并 `monkeypatch.setattr(det, "HandDetector"/"PoseDetector", ...)`;同时补
`gesture_app.detectors.clear()`(与 `session_analyzers` 同级的模块级状态,原 fixture 漏了它)。

这是**在任务书列出的文件内**做的最小改动,守住了它原本要守的东西(本文件测的是
"会话 → 日志文件"接线,不是视觉)。替身真有牙 —— 反向复现(把假 `detect` 改成 raise):

```
$ pytest tests/test_analyze_session_fallback.py -q -k gesture
4 failed, 4 passed, 11 deselected     # 4 条 gesture 全红 => 端点确实走这个假件
# 还原后:
19 passed in 1.67s                     # 32.44s -> 1.67s,且不再加载真模型
```

`mp`/`mediapipe` 字样如今在该文件里**只剩注释散文**,无可执行引用。

## 6. Step 6:全量

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest -q
193 passed in 10.50s
```

基线 189 → 193(+4 = 新文件),**耗时未涨**(10.72s → 10.50s,§5.3 那个 32s 已消掉)。

## 7. Step 7:实跑(真模型、真 HTTP、真落盘)

服务起来时的启动日志确认**自检真的执行了**(两个模型被 TFLite 加载):

```
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 ... inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference.
...
INFO:     Uvicorn running on http://0.0.0.0:8002
```

(服务用 `python -m gesture_analysis.api.app` 启动,故 `__main__` 里的 `verify_models()`
走到了;先起在 `/health` 上确认 200 再发真请求。因为前台 `sleep` 被环境禁用,
就绪等待用 `curl --retry 40 --retry-delay 1 --retry-connrefused`,等价于任务书的 `sleep 8`。)

**真实结果**:

- **HTTP 状态码:`200`**
- **落盘文件名**:`data/logs/gesture_emotion_log_t_20260924_211735_bbbb.csv`(1232 B)
- **首列 = `t_20260924_211735_bbbb`**(CSV 首行表头 `session_id,...`,数据行首列即 SID)

★ 最有价值的一条:响应里 `"detected_hands": 2`,且两个手部 / 肩部 / 双臂都是
`"is_valid": true`,分数是 **58.43 / 53.84 / 70.0 / 90.0** —— **不是**那个静默回落的 50.0。
这是"真 `tasks` landmark → 真分析器"整条链的端到端证明,也正是 §4 那条钉子要防的东西。

额外补的实时 D3 证据(3 帧 / 2 会话,`创建手势探测器会话` 日志行):

```
创建手势探测器会话: t_20260924_211735_bbbb     <-- 第 1 帧
创建手势探测器会话: t_other_bbbb               <-- 另一个会话,自己一份
grep -c => 2                                   <-- 同会话第 2 帧没重建 => 按会话,不是按帧
同会话 CSV: 3 行(1 表头 + 2 数据行)=> 两帧落同一个文件
```

`/reset` 两条新路径也真跑了:`?session_id=<SID>` → `{"status":"success",...}` [200];
`?session_id=t_nope` → `{"status":"not_found",...}` [200]。

服务已停(`kill`)。落盘 CSV 在 `data/logs/` 下,该目录被 `.gitignore:45 **/data/logs/` 排除,未污染提交。

## 8. Step 8:提交

```
$ git add gesture_analysis/api/app.py tests/test_analyze_session_fallback.py tests/test_gesture_detector_wiring.py
$ git commit -m "feat(m1.5): gesture 接入 tasks 探测器(按会话 + TTL 释放 + 启动自检)"
[feat/m1-asr-session-id 8cbe149] 3 files changed, 223 insertions(+), 47 deletions(-)
```

提交里**只有这 3 个文件**。`docs/superpowers/plans/2026-09-24-m1-5-mediapipe-tasks-migration.md`
在我开工前就是 modified 状态(不是我改的),**没有**被我带进提交,仍是 modified。

---

## 9. 可疑但我没动的东西(交给最终审查)

1. **`gesture_analysis/utils/visualization.py:110-136` 仍引用 `mp.solutions`** ——
   `draw_hand_landmarks` / `draw_pose_landmarks` 里用了 `mp.solutions.drawing_utils` /
   `.hands` / `.pose` / `.POSE_CONNECTIONS` / `.HAND_CONNECTIONS`。
   **这是全仓 `grep -rn "mp.solutions" --include="*.py" face_expression/ gesture_analysis/ | grep -v examples/`
   下唯一剩下的可执行命中** —— 最终审查那条"活路径为空"的判据**按字面并不成立**。
   缓解事实(实测):这两个方法**全仓无任何调用者**(只有定义,以及 `code_data_supplement/`
   下一份未跟踪的陈旧副本),且都用 `try/except Exception` 包着(`mediapipe` 在方法体内
   延迟 import,所以 import 期不炸;真被调到也只是打印一条警告后返回原图)。
   定性:死代码 + 优雅降级,**不是**活路径。
   没动的理由:任务书 Files 只列了 `api/app.py`;而且改它要不要顺手迁到 tasks 的
   drawing 工具是**另一个决定**(tasks 的 `drawing_utils` 与 `solutions` 的形状不同)。
   建议最终审查把这条判据的措辞收窄成"可达路径",或单开一条清理项。

2. `face_expression/api/app.py:373`、`face_expression/pipeline/video_pipeline.py:179`
   的 `mp.solutions` 命中**是注释散文**(解释迁移的由来),不是代码。grep 会命中,
   但不构成"活路径"。

3. `code_data_supplement/modules/gesture_analysis/api/app.py` 是迁移前的**一份副本**
   (未跟踪,`?? code_data_supplement/`),里面还是 `mp.solutions` 的老写法 + 模块级
   `session_analyzers`。它不在任何包的 import 路径上,也不在我的改动范围。提一句免得
   最终审查的 grep 扫到它时误判。

4. `MEDIAPIPE_CONFIG` 在 `face_expression/` 下的命中只剩 `config.py:27` 的**一行注释**
   (记录"原来那份是死代码")—— T1 的删除是干净的,那条审查判据实质成立。

## 10. 逐条自检

- [x] 严格按任务书步骤;先写红测并**真跑出红**(非"应该会红")
- [x] 只改任务书列出的 3 个文件;`face_expression/**` 未碰
- [x] 未派任何子智能体
- [x] Step 4 那条钉子实跑,并额外做了"摊平真的会坏"的实测取证
- [x] 如实记录:红了说红(§2),异常说异常(§5.3 的 32s/真模型,已修且有偿)
- [x] Task 2 的 3 行保留(§5.2,附行号)
