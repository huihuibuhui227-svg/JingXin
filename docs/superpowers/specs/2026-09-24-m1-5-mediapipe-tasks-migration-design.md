# M1.5 设计:face / gesture 迁 mediapipe `tasks` API

- 日期:2026-09-24
- 状态:**待使用者审阅**(审阅通过后才写实施计划)
- 上游文件:`docs/下一步.md`(§0/§2 定义本任务;§6 把它列为 M1.5)、`docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md`(§4 特征体系,L0 逐列重构消费本任务的产物)、`docs/superpowers/specs/2026-09-24-m1-asr-session-id-design.md`(`session_id` 写侧契约,本任务不动它)
- 实测支撑:`~/shared/mp_frames/`(探针与比对脚本 + `{solutions,tasks,gesture_*}.json` + `out/` 下的标注图)

---

## 0. 目标(一句话)

**让 face 与 gesture 两个服务重新能跑,并且把"换探测器"带来的数值位移量清楚、把受影响的门槛重新登记** —— 而不是无声地换掉测量仪器。

**成功的定义**(三条都要):

1. `face_expression` 的 `/analyze` 与 `gesture_analysis` 的 `/analyze` 返 200,**能落出带 `session_id` 的日志**(现在 face 永远落不了,gesture 连进程都起不来);
2. 迁移的数值影响**有实测记录**(同一批帧、新旧两个实现的逐部位位移表),不是"看着对";
3. 受影响的阈值/量程**被逐条指名并给出依据** —— 哪些需要重登记、为什么、用哪个数。**具体把数值改进 `evidence_thresholds.json` 留给 M3**(理由见 §12 第 1 条:那要与逐列重构一起做);本任务的交付是**判定**,不是改数。

第 2、3 条是本设计的重点。**第 1 条只是让服务活过来;如果只做第 1 条,报告会照旧拿新探测器的数字去撞按旧探测器分布定的阈值,判定会整体偏移而无人知晓。**

---

## 1. 范围

**在范围内**

1. face 探测器:`mp.solutions.face_mesh.FaceMesh` → `vision.FaceLandmarker`
2. gesture 探测器:`mp.solutions.hands.Hands` / `mp.solutions.pose.Pose` → `vision.HandLandmarker` / `vision.PoseLandmarker`
3. **gesture 的探测器改为按会话**(现在 `hands`/`pose` 是模块级单例,所有客户端共用 —— 见 §3.5)
4. 模型文件的路径契约与加载封装
5. `gesture_analysis/config.py` 的 `MEDIAPIPE_CONFIG` 改写(键名语义变了,见 §6.2)
6. `requirements.txt` / `requirements-full.txt` 钉 `mediapipe==1.0.0`
7. `tests/test_analyze_session_fallback.py` 的环境替身重指(它是**唯一** import mediapipe 的测试文件)
8. 基于 §3.4 实测的阈值/量程重登记

**不在范围内(明确记下,别顺手扩)**

- `face_expression/examples/*`、`gesture_analysis/examples/*`、`gesture_analysis/utils/visualization.py`、`gesture_analysis/pipeline/gesture_pipeline.py` —— 都**不在活路径**;`visualization.py` 的 `solutions` 引用本来就包在 `try/except` 里,在 1.0 上只是静默不画图。**留它们坏着,单独记账。**
- `experiments/extract_features.py` —— **论文线**,按"完善系统不是重跑论文"不动(它用了 `multi_handedness`,迁移后那一处的形状会变;记在 §12)。
- `face_expression/config.py` 的 `MEDIAPIPE_CONFIG` —— **死代码,删而不是迁**(证据:`video_pipeline.py:35-41` 硬编码全部 5 个 kwarg,从不 import 它)。
- **按 `session_id` 选日志** —— 那是 M2。
- VIDEO 模式的行为等价性**不在本设计的验收里**,单列一道门(§10.4)。

---

## 2. 决策记录

| # | 决策 | 依据 / 出处 |
|---|---|---|
| **D1** | 走 `tasks` API,**不降级** mediapipe | 用户 2026-09-24 裁定(降级只是把问题推后) |
| **D2** | **要量数值位移**(方案 A),不满足于"能跑" | 用户 2026-09-24 裁定;理由见 §0 |
| **D3** | gesture 探测器**改为按会话** | 用户 2026-09-24 裁定;理由见 §3.5 与 §6.3 |
| **D4** | face 的 `MEDIAPIPE_CONFIG` **删**,不迁 | 死代码,见 §1 |
| **D5** | 姿态档位**必须对齐**:`model_complexity=1` ↔ `pose_landmarker_full` | 实测教训,见 §3.4 |
| **D6** | 金标环境**不新建**,用现成的旧版解释器 | 原计划新建抛荒 venv;实际核查环境图谱后发现旧版环境现成,最终用 `~/mp-ref`(见 §3.3) |
| **D7** | 位移结论**只写"模型包差异",不写"API 差异"** | 姿态那一栏的实测反证,见 §3.4 |

---

## 3. 实测事实(2026-09-24 探测,是设计的地基)

### 3.1 mediapipe 1.0.0 删掉的是整个 `solutions` 命名空间

```python
import mediapipe as mp
mp.__version__        # '1.0.0'
hasattr(mp, 'solutions')   # False   —— 只剩 mediapipe.tasks
pkgutil.iter_modules(mediapipe.__path__)  # ['tasks']
```

`vision.FaceLandmarker / HandLandmarker / PoseLandmarker` 都在。**模型文件不随 pip 包发布**(`find site-packages/mediapipe -name "*.task"` → 空)。

### 3.2 两个崩溃点,坏法不同

| 服务 | 位置 | 坏法 |
|---|---|---|
| gesture | `gesture_analysis/api/app.py:42-46` | **模块导入期**取 `mp.solutions.hands/hands` + 构造 `Hands()`/`Pose()` → `AttributeError`,进程在 uvicorn 绑定前就死 |
| face | `face_expression/pipeline/video_pipeline.py:33-41` | `face_mesh` 是**惰性属性** → 启动不碰它、`/health` 与 `/session/{sid}/summary` 正常,而**每一帧** `/analyze` 抛 `AttributeError` → 500 |

后果:`data/logs/` 里**永远不会出现** `face_au_log_<sid>.csv`(历史 11 个 face 日志全是旧形态、且全部产自 2026-03~04,即 WSL 迁移之前)。T7 的「face/voice 两份日志」判据在现环境**不可能通过** —— 这是 `docs/下一步.md` §2 的由来。

### 3.3 几何层是 API 无关的

**~3,002 行零 mediapipe 引用**(face 806 + gesture 2,196):`au_calculator` / `landmarks` / `emotion_engine` / `tension_engine` / `micro_expression` / 四个 analyzer / 四个 feature extractor,只 import `numpy` / `scipy.spatial.distance` / `dataclasses`。

消费方接口恰好对得上两代 API:

- **face**:`video_pipeline.py:51-52` 把结果**摊平成 `[(pt.x, pt.y)]` 元组**,下游只吃浮点数;`results` 原始对象在 `api/app.py:226` 绑给 `mesh_results` 后**再没被读过**。
- **gesture**:分析器拿**裸 landmark 列表**、用 `.x/.y` **属性访问**(~112 处数字下标)—— `tasks.NormalizedLandmark` 暴露同样的 `.x/.y/.z/.visibility`。**四个分析器 + 四个特征抽取器零改动。**

  ⚠️ **这条"零改动"有前提,而且踩错是静默的**(2026-09-24 实测):前提是**把 landmark 对象原样交下去**。若"顺手"摊平成 `(x, y)` 元组(face 那侧就是这么做的,容易顺手照搬),分析器**不抛异常**,只是 `is_valid=False`、分数回落到默认的 `50.0`:

  | 喂进去的形状 | `HandAnalyzer` | `ArmAnalyzer` |
  |---|---|---|
  | `(x, y)` 元组 | `is_valid=False`,`resilience=50.0` | `is_valid=False`,`arm_score=50.0` |
  | 带 `.x/.y` 的对象 | `is_valid=True`,`resilience=51.47` | `is_valid=True`,`arm_score=90.0` |

  也就是说:**报告里会印着一个数,而它什么都不代表,且不留任何日志**。所以两个封装的返回形状**刻意不对称** —— face 摊平成元组(`au_calculator` 吃元组),gesture 原样交对象。这条不变量要有测试钉住(见实施计划 T1 / T3)。

**478 点拓扑保留**:`tasks.FaceLandmarker` 默认输出含虹膜的 478 点,`au_calculator.py:220-221` 读的 `468-476` **一个下标都不用改**。

**金标侧**:`~/mp-ref`(由项目环境 python 建的 venv,`mediapipe==0.10.5`,`solutions` 在)下,**现行 `video_pipeline.py` 原样可跑** —— 也就是说"旧实现"不需要另写,它就是仓库里现在这份代码。

### 3.4 ★ 数值位移实测(本设计的核心数据)

**方法**:28 张同一个人的摄像头实拍帧(640×480,`~/shared/mp_frames/frames/`),喂给两个实现,**IMAGE 模式**(每帧独立、无跟踪,量的是探测器本身),检测置信度按生产配置对齐。

| 部位 | 旧 → 新 | 拓扑 | 平均位移 | p95 | 最大 | 检出一致性 |
|---|---|---|---|---|---|---|
| 人脸 | `FaceMesh` → `FaceLandmarker` | 478 点 | **0.0036(≈2.0 px)** | 0.0092 | 0.0239(≈13 px) | **28/28,0 分歧** |
| 手部 | `Hands` → `HandLandmarker` | 21 点 | **0.0109(≈6.1 px)** | 0.0229 | 0.1242(≈70 px) | **28/28,0 分歧** |
| 姿态 | `Pose(complexity=1)` → `PoseLandmarker full` | 33 点 | **0.00073(≈0.41 px)** | 0.0035 | 0.0230(≈13 px) | **28/28,0 分歧** |

**三条结论:**

1. **检出层零退化** —— 三个部位都是 28/28 数目一致、0 帧分歧。没有"一边有一边没有"的形态。
2. **姿态几乎逐点重合(0.41 px)**,而人脸 2 px、手部 6 px。**姿态那一栏是反证**:当底层模型对得上时,API 换代本身在数值上近乎中性。**所以脸与手的位移来自 `.task` 模型包与 `solutions` 内置模型的差异,不是 API 换法不对**(D7)。
   - **这条决定了修法**:位移**不能靠改代码修回来**,只能接受它、并把它记进阈值依据。
   - **它也暴露了一个陷阱**:第一次测姿态时新旧档位不一致(`complexity=1` ↔ 当时手头的 `lite`),得出 42 px、最大 545 px 的假象 —— **全部由档位差异造成**。档位对齐后降到 0.41 px。(D5)
3. **下游特征位移不可忽略**:face 的 29 个数值特征里 **11 个相对位移 >10%,4 个 >25%**(`au23_lip_compression` 55.8%、`au25_mouth_open` 46.4%、`gaze_direction_x` 28.0%、`au26_jaw_drop` 25.4%),**且方向一致偏小**(au23/25/26/12/20/4/2/1/10/15/7 全是新的比旧的小)—— 是**系统性偏置**,不是噪声。
   - **机理**:AU 由 landmark 之间的**微小距离差**算出(眼睑开合、唇缝高度),分母小、放大倍数大 —— 2 px 落到"嘴唇开合度"上就是几十个百分点。
   - **相对位移 ≠ 绝对影响**:`au23` 的 55.8% 对应绝对值只有 0.068。用时看绝对量。

**边界(必须随数据一起引用,别脱离条件引用)**

- 这是 **IMAGE 模式**;生产用 **VIDEO 模式**(带跟踪),那是另一件事(§10.4)。
- 20 个检出帧是**同一个人、同一台摄像头、同一间屋子、同一段光照**。"系统性偏小"**只在这套条件下成立**,不是"新探测器一律偏小"。
- **gesture 只有 landmark 层,没有特征层**:gesture 的分析器是**有状态**的(deque 滚动历史 + 时序统计),对单张静止帧算出的值没有意义。手势特征位移要等迁移后在**真实序列**上量(§12)。

### 3.5 gesture 的探测器归属是错的

| | face | gesture |
|---|---|---|
| 探测器 | **按会话** —— `session_pipelines[sid]` → 每个 `VideoPipeline` 一个自己的 `_face_mesh` | ⚠️ **模块级单例** —— `hands`/`pose` 在 `app.py:45-46` 全局构造一次 |
| 分析器 | 随探测器同实例 | 按会话(`session_analyzers[sid]`) |
| `/analyze` 用哪个 | 会话自己的 | **所有会话共用同一个** `hands.process()` / `pose.process()`(`:230`,`:243`) |

`solutions` 的 `process()` 基本无状态,所以现在这点只表现为"统计状态串味"。**`tasks` 的 VIDEO 模式把跟踪状态挂在探测器实例上** —— 迁过去之后,两会话并发会**互相污染跟踪**(追踪的是别人)。D3 据此裁定:改为按会话。

---

## 4. 架构

```
face_expression
  POST /analyze
    api/app.py: session_pipelines[sid] -> VideoPipeline -> face_mesh
    (face_mesh 从"惰性属性"改为"构造时注入的探测器")
                                                   -> FaceLandmarker   [按会话]

gesture_analysis
  POST /analyze
    api/app.py: session_analyzers[sid] -> analyzers
                detectors[sid]         -> HandLandmarker / PoseLandmarker   [按会话,新增]

两个模块各自持有探测器加载封装,不共享(理由见下)。
```

**探测器加载封装不共享,两份。** 理由与 Ruling M1-2 同:face/gesture/voice 三个包各自持有守卫副本是**有意为之**,跨包 import 会把整条依赖链拉起来并让依赖方向反转。这次的两份封装各约 15 行,重复的代价小于耦合的代价。**这条要写进实施计划,免得实施者顺手"去重"。**

**模型文件路径**:`<repo>/models/mediapipe/`,路径**由 config 读取,不硬编码在抽取代码里**(M3 的逐列重构会重排特征,但探测器路径不该跟着动)。

---

## 5. 组件(必须改的 5 个)

| 文件 | 改什么 | 为什么只有它 |
|---|---|---|
| `face_expression/pipeline/video_pipeline.py` | `face_mesh` 属性(:33-41)改为构造时注入的 `FaceLandmarker`;`process_frame`(:44-52)换成 `detect_for_video(mp.Image, ts_ms)` + 结果形状(`result.face_landmarks[0]`);下游 `landmarks_norm` 那行**不动** | 唯一活路径的 face 崩溃点 |
| `gesture_analysis/api/app.py` | 删掉模块级 `mp.solutions` 与 `Hands()`/`Pose()`(:42-46);新增**按会话**探测器表 + TTL 回收;两处 `process()`(:230,:243)换成 `detect_for_video` + 形状映射 | 唯一活路径的 gesture 崩溃点 |
| `gesture_analysis/config.py` | `MEDIAPIPE_CONFIG`(:56-69)按 §6.2 改写;新增模型文件路径键 | 唯一真被传给探测器的 config |
| `tests/test_analyze_session_fallback.py` | `_install_env_shims`(:48-80)的替身从 `mp.solutions` 重指到 `mp.tasks`;**并给裸 `import mediapipe` 加守卫** | 唯一 import mediapipe 的测试 |
| `requirements.txt` / `requirements-full.txt` | `mediapipe>=0.8.0` → **`mediapipe==1.0.0`** | 这个范围**允许 1.0 进来**,所以这次"服务全死"没有任何一处报警 |

**附带(顺手,不单列任务)**:`face_expression/config.py` 的死 `MEDIAPIPE_CONFIG` 删除;`face_expression/api/app.py:369-392` 的 `__main__` 自检探针改指 `FaceLandmarker`(它包在 `try/except` 里,不拦启动,但会一直误报"MediaPipe检查失败",干扰排障)。

---

## 6. 接口契约

### 6.1 模型文件

| 文件 | 用途 | 实测体积 | 来源(URL 已验证 200) |
|---|---|---|---|
| `face_landmarker.task` | face 478 点 | 3758596 | `…/face_landmarker/face_landmarker/float16/latest/face_landmarker.task` |
| `hand_landmarker.task` | gesture 手部 21 点 | 7819105 | `…/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task` |
| `pose_landmarker_full.task` | gesture 姿态 33 点 | 9398198 | `…/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task` |

- 存放:`<repo>/models/mediapipe/`,**进 `.gitignore`**(合计 ~21 MB,而 `.git` 已是 1.37 GiB 全松散对象)。
- **版本钉法未验证**:URL 用的是 `…/float16/latest/…`;要可复现应钉数字版本段,但**该写法我没验过**,实施时要么验证要么明确接受 `latest` 漂移的风险。**不要当成已确认的事实。**

### 6.2 `MEDIAPIPE_CONFIG` 新形状(键名语义变了)

| 旧键 | 新键 | 说明 |
|---|---|---|
| `static_image_mode: False` | `running_mode: VIDEO` | **不是改名,是换概念**:`static_image_mode` 是布尔,`running_mode` 是三态(IMAGE/VIDEO/LIVE_STREAM) |
| `max_num_hands: 2` | `num_hands: 2` | HandLandmarkerOptions |
| `model_complexity: 1` (pose) | **由加载哪个 `.task` 文件决定** | 0/1/2 ↔ lite/full/heavy。**配置里要显式写出档位名** |
| `min_detection_confidence` | `min_face_detection_confidence` / `min_hand_detection_confidence` / `min_pose_detection_confidence` | 各自命名 |
| `min_tracking_confidence` | `min_tracking_confidence` | 同名保留 |
| — | `num_faces` / `num_poses` | 新键,显式写 |

### 6.3 探测器生命周期(按会话 + TTL)

- 两个模块各持一张 `detectors[sid]` 表,**与既有的 `session_pipelines(session)/session_analyzers(detectors)` 同生命周期**:建会话时一起建,TTL 回收时**一起回收**。
- **回收必须显式 `close()`** —— `FaceLandmarker`/`HandLandmarker`/`PoseLandmarker` 持有 native 资源,只从 dict 里删掉会泄漏。
- 这条是**新的集成缝**:现有 TTL 回收逻辑(`face api/app.py:122-126`、`gesture api/app.py:127-131`)只删分析器,迁移后要连探测器一起删。**实施计划里要点名。**

### 6.4 时间戳契约(VIDEO 模式要求单调递增)

- `detect_for_video(image, timestamp_ms)` 的 `timestamp_ms` **必须单调递增**,否则 mediapipe 抛错。
- **face**:`VideoPipeline` 已有 `self.fps`,用**每会话的帧计数器** `frame_index * (1000/fps)` 即可,天然单调。
- **gesture**:现在的模块级探测器**没有任何帧计数状态**,要新增 —— 且因为探测器改成按会话(§6.3),计数器也**按会话**。
- ⚠️ **`/reset` 必须一起重置计数器**(`gesture api/app.py` 的 reset 与 `face api/app.py:336` 的 `/session/{sid}/reset`),否则摄像头重启后时间戳回退 → mediapipe 抛错/跟踪错乱。**这条要写进实施计划。**

### 6.5 结果形状映射

| | 旧 | 新 |
|---|---|---|
| face | `results.multi_face_landmarks[0].landmark` | `result.face_landmarks[0]` |
| gesture 手 | `hand_results.multi_hand_landmarks` → `.landmark` | `result.hand_landmarks` |
| gesture 姿态 | `pose_results.pose_landmarks.landmark` | `result.pose_landmarks[0]` |
| 输入 | 直接传 RGB `ndarray` | 需包成 `mp.Image(image_format=SRGB, data=…)` |

**点对象不用适配**:`.x/.y` 属性访问在两代都成立(§3.3)。

---

## 7. 数据流(一帧)

```
POST /analyze?session_id=SID
  └─ _resolve_session_id            (不动;D3 of M1 已把空串口径收口)
  └─ get_or_create_pipeline/analyzers(SID)
       └─ 同时取/建本会话的探测器  ← 新增
  └─ BGR → RGB(contiguous)
  └─ mp.Image(…, data=rgb)          ← 新增
  └─ detect_for_video(img, ts_ms)   ← frame_counter 自增;ts 单调
  └─ 形状映射(§6.5)
  └─ 几何层:AU / EAR / 情绪 / 张力 / 微表情 / 手-肩-臂分析   ← 一行不改
  └─ logger.log(...)(首列写 session_id)                      ← 一行不改
```

---

## 8. 错误处理(写死,不留含糊)

| 情形 | 行为 |
|---|---|
| 模型文件缺失 | **启动即失败**并说清缺哪个文件、期望路径。**不许静默降级、不许推迟到第一次请求** —— face 现在"启动正常、每帧 500"的哑死法正是要消灭的形态 |
| 模型加载失败(损坏/版本不符) | 同上:启动失败 + 原样上抛 |
| `detect_for_video` 的时间戳非单调 | **原样上抛**(500)。这是编程错误,不是坏输入 |
| 一帧内检不到任何目标 | 现有行为(返回 `no_face` / 空结果),**不算错误**,不动 |
| 探测器 TTL 回收 | 显式 `close()`,失败记日志但不拦请求 |

---

## 9. 已知未满足项与风险

1. **位移是模型包差异,改代码修不回来**(§3.4 结论 2)。本设计能做的是**度量它并重新登记阈值**,不是消除它。
2. **gesture 没有特征层位移数据**(§3.4 边界)。手势的下游影响目前**只有 landmark 层的 6 px** 作为间接证据,真实序列上的特征位移要等迁移后量。
3. **实测样本局限**:1 个人 / 1 台摄像头 / 1 间屋子 / 20 个检出帧。要更强的结论得多拍几种条件;本设计**不把这 20 帧的分布当成总体**。
4. **VIDEO 模式未验**(§10.4)。生产用带跟踪的模式,其行为与 IMAGE 模式不同,本次所有数值都来自 IMAGE 模式。
5. **姿态 lite/full 的档位语义要落到配置里**,否则下一个人会重犯 §3.4 那个假象(42 px / 545 px 的教训)。
6. **`multi_handedness` 形状变了**(`handedness[i][0].category_name`),活路径不用它(用的是位置判断),但**论文线与 examples 用了**,迁移后那几处会坏 —— 见 §12。
7. **`gesture_analysis/api/app.py:237` 的位置判断 `hand_id == 0 → left_hand` 是潜在左右手反了的 bug**。迁移会碰到这一行;**修不修要单独决定**,不要顺手改(会混进"迁移"这件事里,让 delta 无法归因)。

---

## 10. 测试与验收

### 10.1 契约测(新增)

- 探测器加载:模型缺失 → 启动失败且错误信息含路径(§8 第 1 行)。
- 按会话隔离:`sid_A` 与 `sid_B` 拿到的探测器**不是同一个对象**(§6.3;gesture 现在这条会红)。
- TTL 回收:会话过期后探测器被 `close()` 且从表里移除。
- 时间戳单调:同一会话连续两帧的 `ts_ms` 严格递增;`/reset` 后从 0 重新开始且不抛错。
- 形状映射:用一个假的结果对象验证 §6.5 的四个映射(不依赖真 mediapipe)。

### 10.2 测试替身必须重指

`tests/test_analyze_session_fallback.py:48-80` 现在替的是 `mp.solutions`。迁移后:

- 替身要改指 `mp.tasks.python.vision.{HandLandmarker,PoseLandmarker}`,`detect_for_video` 返回 `hand_landmarks=None, pose_landmarks=None` 的空结果。
- **`import mediapipe as mp`(:64)是裸的,必须加守卫** —— 否则 mediapipe 一旦缺失,这个文件里 **20+ 条测试全部 ImportError 死掉**,不只 gesture 那些。

### 10.3 金标比对(本设计的验收要件)

`~/shared/mp_frames/` 下的四个脚本(`probe.py` / `probe_gesture.py` / `compare.py` / `compare_gesture.py`)是一次性测量工具,**不进仓库**,但要保留并在迁移后重跑。

**判据只有一条,不许放松**:**迁移后的实现跑出的 landmarks,与 `tasks.json` / `gesture_tasks.json` 里已记录的那份逐点一致**(同一批帧、同一 IMAGE 模式、同一模型文件)。

为什么是"逐点一致"而不是"同数量级":那两份 JSON 记的是**同一个底层实现**(mediapipe 1.0.0 + 同样的 `.task` 文件 + 同样的 IMAGE 模式)的产出。适配层写对了就应当**完全相同**;只要有一点差异,就说明适配层改变了输入(颜色通道顺序、`mp.Image` 的 data 形状、置信度参数、结果索引方式)。

⚠️ 注意区分:这条判据验的是"**适配层没写错**",**不是**"迁移无位移"。位移(§3.4)是另一个实现与旧 API 比出来的,与这条无关。

### 10.4 VIDEO 模式的门(单独一道,不在本次验收内)

在**真实帧序列**上(摄像头连续采一段,而不是独立静帧)跑一次新旧对比,确认带跟踪模式下行为等价。这道门的意义是:IMAGE 模式证不了生产行为。**允许留到迁移后单独做,但必须记档,不许当成"已经验过"。**

### 10.5 T7 的前置关系

本任务完成后,T7(§2 of `docs/下一步.md`)的「face/voice 两份日志」判据才可能通过 —— 但**要跑完整 T7 还需要 gesture 也活着**(判据写的是三份日志)。T7 的其它前置(≥5 段内容各异的回答、只用铸造 id、原句不出仓库)不变。

---

## 11. 任务切分与评审安排

**规模判断**:必须改的 5 个文件,其中真正是"探测器适配"的只有**几十行**;几何层 3,002 行一行不动。**因此不照搬 M1 的 6 任务规模** —— 那会把几十行摊成 6 份报告,协调成本大于收益。

| 任务 | 内容 | 依赖 |
|---|---|---|
| **T1 契约层** | 模型路径/加载封装(两份)+ `gesture_analysis/config.py` 改写 + `requirements` 钉版本 + **`models/mediapipe/` 加进 `.gitignore`** + 删死 config | **无(是 T2/T3 的前置,必须先落地并冻结)** |
| **T2 face 适配** | `video_pipeline.py` 探测器 + `mp.Image` + 时间戳 + 形状映射;`api/app.py:369-392` 自检探针 | T1 |
| **T3 gesture 适配** | `api/app.py` 探测器**按会话** + TTL 回收 + 时间戳/`reset` + 形状映射;`test_analyze_session_fallback.py` 替身重指 + 加 import 守卫 | T1 |

**评审**:每任务一次审查 + **一次最终全分支审查**。这不是形式主义 —— M1 那次最严重的 C1(评估产物遮蔽会话日志)**活过了六轮任务级审查**,只有最终全分支审查用**真加载器**跑一遍才抓到,因为任务级审查在结构上看不到集成缝。本任务同样有集成缝(`§6.3` 探测器与 TTL 的交互、`§6.4` 时间戳与 `/reset` 的交互),值得同样的待遇。

**最终审查必须实跑的东西**(不许只看 diff):起两个服务、发一帧、确认 200 且落出带 sid 的日志;跑 §10.3 的金标比对。

---

## 12. 待办(由本设计引出,不属 M1.5)

1. **阈值/量程重登记的落地**:§3.4 给了 face 的 11 个 >10% 特征,但"哪个阈值该改成多少"需要结合 `evidence_thresholds.json` 逐条判断 —— 归到 M3 的逐列重构一起做,本任务只负责**把数据交出来**。
2. **gesture 特征层的位移**:迁移后在真实序列上量。
3. **`multi_handedness` 形状变更的下游**:`gesture_analysis/pipeline/gesture_pipeline.py:113-138`、`gesture_analysis/examples/*`、`experiments/extract_features.py:138-148`。前两个不在活路径(留坏 + 记账),第三个是论文线。
4. **`hand_id == 0 → left_hand` 的位置判断**(§9 第 7 条):单独决定修不修。
5. **`examples/*` 与 `visualization.py` 的整体处置**:它们是**已经坏了**的代码(mediapipe 1.0 + 一个 import 不存在的模块 `face_expression.analyzers.image_analyzer`)。是修、是删、还是标记为冻结快照,单独决定。
6. **`code_data_supplement/modules/face_expression/pipeline/video_pipeline.py` 与活文件逐字节相同** —— 迁移后必然漂移。要显式决定:更新它,还是标记为投稿时的冻结快照(**倾向后者**,它是匿名投稿包)。
7. **`models/mediapipe/` 的版本钉法**(§6.1):验证 `…/float16/<n>/…` 写法,或明确接受 `latest` 漂移。
