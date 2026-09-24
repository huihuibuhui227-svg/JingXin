# SDD ledger — plan: docs/superpowers/plans/2026-09-24-m1-5-mediapipe-tasks-migration.md

Spec: `docs/superpowers/specs/2026-09-24-m1-5-mediapipe-tasks-migration-design.md`(可达,绑定权威)

## 开工前的裁决(执行方式)

- **Ruling: T2/T3 串行执行,不并行派发。** — 理由:同一个工作树里两个实现者并发
  `git add`/`git commit` 会抢 `index.lock`,更坏的是任何 `commit -a`/`add -A` 式的操作
  会把**对方正在改的文件**卷进自己的提交(而这两个任务的提交范围恰好互为"无关文件",
  一旦卷进去就很难发现)。代价:墙钟时间多几分钟;可回退(真并行也能重来)。
- **Ruling: 不用 git worktree。** — 理由:模型文件在 `<repo>/models/mediapipe/` 且被
  `.gitignore` 排除,worktree 里**没有**它们 → 探测器找不到模型,测试与实跑全废。
  代价:没有隔离;但本就在 `feat/m1-asr-session-id` 上(非 main),且任务面窄。
- **Ruling: T1 由控制器在主会话亲自做。** — 使用者确认;理由:T1 是契约层,写它的人
  = 写计划的人,上下文最全;且它是 T2/T3 的前置,派出去只会多一轮对齐。代价:
  控制器上下文被 T1 的实现细节占用。

## 预检冲突扫描(开工前,对着计划文本逐对逐任务查)

| 行 | 查什么 | 结论 |
|---|---|---|
| T1↔T2(共享接口) | T1 产出 `FaceDetector` / `verify_models(model_path, factory)` / `FACE_MODEL`;T2 消费这三样 | ✅ 名字与签名一致 |
| T1↔T3(共享接口) | T1 产出 `HandDetector` / `PoseDetector` / `verify_models(hand,pose,factory)` / `HAND_MODEL` / `POSE_MODEL`;T3 用 `**MEDIAPIPE_CONFIG['hands']` 展开构造 | ✅ 配置键(`num_hands`/`min_detection_confidence`/`min_tracking_confidence`)与 `HandDetector.__init__` 的形参逐字一致;pose 那侧多一个 `tier`,T3 已显式过滤 |
| T2↔T3(共享文件) | T2 改 `face_expression/**`;T3 改 `gesture_analysis/**` + `tests/test_analyze_session_fallback.py` | ✅ **零重叠**(这也是原计划里"可并行"的依据;并行被 R1 否掉,与本行无关) |
| T1(自洽) | 测试里定义的假工厂 vs 封装实际调用工厂的方式 | ❌ **缺陷 1**:假工厂写成 `def build(**opts)`,而封装把 `model_path` **位置传参** → 测试会 ERROR 而不是按预期红。**已修**(改成 `def build(*args, **opts)`,并写明为什么) |
| T2(自洽) | Step 3 的替换指令 vs `process_frame` 原文行号 | ❌ **缺陷 2**:"头两行"的说法与实际要删的 6 行不符,实施者可能删错。**已修**(逐行标出删/留) |
| T3(自洽) | 新测试 vs 真探测器构造 | ❌ **缺陷 3**:`get_or_create_detectors` 会构造**真**探测器 → 测试要加载 21 MB 模型、且**没下模型的环境整片红**。**已修**(加 `fake_detectors` fixture,monkeypatch 掉两个探测器类) |
| T1/T2/T3(约束) | 是否有任务要求"计划明文规定但评审视为缺陷"的东西 | ✅ 无(没有空断言测试,几何层重复被明确禁止) |

三条缺陷都在计划文本里就地修掉后再开工;**没有留成"实施者自行判断"**。

## 进度

### Task 1: 契约层(控制器亲自做,`a29d249`)

- 新增 `face_expression/pipeline/detector.py`(FaceDetector + verify_models)、
  `gesture_analysis/core/detectors.py`(HandDetector + PoseDetector + verify_models)
- 两个 config:gesture 的 `MEDIAPIPE_CONFIG` 按语义改写;face 的同名配置**删掉**
  (死代码,删除后全量测试零回归 —— 反过来证明它确实没人用)
- `requirements*.txt` 钉 `mediapipe==1.0.0`;`.gitignore` 排除 `models/mediapipe/`
- 测试 177 → 186(+9);契约测**不 import mediapipe**、不依赖 21 MB 模型

**执行中发现并修掉的两个缺陷**(都不在计划文本里,是实现期暴露的):

- **缺陷 1(契约测漏掉的真实 bug)**:`FACE_MODEL` 是 `os.path.join` 出来的 **str**,
  而 `verify_models()` 不带参数时直接对它 `.exists()` → `AttributeError`。
  契约测全绿是因为它们**都显式传 `Path`** —— 参数的形状被测试定死了,而生产路径给的不是
  那个形状。这正是"测试通过 ≠ 有约束力"的又一例。
  **Ruling: 补一条刻意传 `str` 的回归测,先看它红、再修实现。** — 代价:多一条测试;
  不这么做的代价:这个 bug 会活到 T2 的实跑才暴露。
- **缺陷 2(计划文本自身)**:T1 的假工厂写成 `def build(**opts)`,而封装把 `model_path`
  位置传参 → 测试会 ERROR 而非按预期红。已在开工前的预检里修掉。

**反向复现**(两条都做过,证明测试有牙齿):
- 把 `_groups` 改成摊平元组 → `test_gesture_detectors_keep_x_and_y_attributes` 精准红
  (`False = hasattr((0.5, 0.5), 'x')`),恢复后 9 条全绿
- `str` 路径那条:先红(`AttributeError: 'str' object has no attribute 'exists'`)后修

**T1 自己的实跑验收**:真加载三个模型 → `face_verify()` OK / gesture `verify_models()` OK;
真跑一帧 → `FaceDetector 478 点`、`PoseDetector 33 点`、`HandDetector 0 只`(该帧本无手),
`pose[0]` 带 `.x/.y` = True。

### Task 1: complete (commits 234f817..a29d249, 控制器自验 + 反向复现)

**交给 T2/T3 的接口(冻结)**:
- `face_expression.config.FACE_MODEL`(**str**;`MEDIAPIPE_MODELS_DIR` 同风格)
- `face_expression.pipeline.detector.FaceDetector(model_path, *, fps=30, num_faces=1,
  min_detection_confidence=0.8, min_tracking_confidence=0.8, factory=None)`
  → `.detect(np.ndarray) -> list[tuple[float,float]] | None`、`.reset()`、`.close()`
- `face_expression.pipeline.detector.verify_models(model_path=None, factory=None)` → `RuntimeError`
- `gesture_analysis.config.{MEDIAPIPE_MODELS_DIR, HAND_MODEL, POSE_MODEL}` +
  `MEDIAPIPE_CONFIG['hands'] = {num_hands, min_detection_confidence, min_tracking_confidence}`、
  `MEDIAPIPE_CONFIG['pose'] = {tier, num_poses, min_detection_confidence, min_tracking_confidence}`
- `gesture_analysis.core.detectors.{HandDetector, PoseDetector, verify_models}`
  → `.detect(np.ndarray) -> list[list[Landmark]]`(手)/ `list[Landmark] | None`(姿态)
  —— **交出对象,不是元组**(见缺陷 1 的同类:摊平会静默回落默认分)

### Task 2: face 适配(子智能体,`db3cca1`,DONE_WITH_CONCERNS)

4 文件 +120/−54;全量 189 passed(控制器自己复跑,不信报告);实跑 8000 端口 **HTTP 200**
且落盘 `face_au_log_<SID>.csv`(迁移前必然 500 且无文件)。

**执行者提的两条偏离,控制器裁决如下:**

- **Ruling: 接受它在 `tests/test_analyze_session_fallback.py` 加 3 行替身(`close()`)—— 虽超出任务书文件清单。**
  — 理由:T2 给 TTL 回收加了 `pipeline.close()`,而那个文件里的 `_FakeFacePipeline` 没有
  `close()`,且它冻住的时钟每步 +1000s > `SESSION_TIMEOUT=300`,第二帧必然走 TTL 那条路
  → 真实 AttributeError。**替身必须跟着真接口长,否则测的就不是接线本身。**
  它明确拒绝了 `getattr(pipeline,'close',lambda:None)()` 那种静默兜底 —— 判断正确。
  代价:又一个文件被两个任务碰(见下面的扫描更正)。
- **Ruling: 接受它把 `video_pipeline.py:187` 的 `return result, results, ...` 改成
  `return result, landmarks_norm, ...`。** — 理由:任务书说"从 `nose_tip` 起一个字都不改"
  是**计划自身的 bug** —— `results` 正是被删掉的那个局部变量,照抄会 NameError。
  第二个槽的三处消费点**全都不读它**(`api/app.py:226` 绑成 `mesh_results` 后再没出现;
  两个 `examples/` 脚本在本环境本来就跑不起来;`experiments/` 那处直接 `_` 丢弃)。
  为什么选 `landmarks_norm` 而不是 `None`:给 `None` 的话未来消费方写 `if raw:` 会**静默跳过**,
  而给 landmarks 是"用错了就 AttributeError" —— **响亮失败 > 静默跳过**,与本仓一贯取向一致。
  **代价**:若真有未知消费方依赖"第二个槽是原始 mp 对象",它会 AttributeError(响亮,可查)。

- **⚠️ 控制器自我更正:预检扫描漏了一对。** 扫描表里写的「T2↔T3 共享文件:**零重叠**」是**错的** ——
  `tests/test_analyze_session_fallback.py` 同时在 T3 的文件清单里。**Ruling: 串行执行(R1)已经
  挡住这次冲突;T3 的派发词里必须显式点名这件事,要求它在 T2 的改动之上做删除、不许整段覆盖。**
  — 代价:若 T3 整段重写该文件,T2 那 3 行替身会被抹掉 → 189 里会掉几条红(可发现,非静默)。

### Task 2: complete (commits a29d249..db3cca1, 控制器独立核验:全量 189 passed + 实跑 200)

### Task 3: gesture 适配(子智能体,`8cbe149`,DONE_WITH_CONCERNS)

3 文件 +223/−47;全量 193 passed(控制器复跑);实跑 8002 端口 **HTTP 200**,
落盘 `gesture_emotion_log_<SID>.csv`,响应 `detected_hands: 2` 且四个分析器
`is_valid: true`、分数 58.43/53.84/70.0/90.0 —— **不是静默回落的 50.0**(这是反摊平那条
不变量在真实服务上的端到端证明)。另实时证了 D3:3 帧 / 2 会话只产生 2 条"创建探测器会话"
日志(同会话第 2 帧没重建)。

**执行者提的两条顾虑,控制器裁决如下:**

- **Ruling: 接受它把测试替身点从 import 期移到构造处(仍在同一个文件内,未出清单)。**
  — 理由:任务书 Step 5 只写了"删掉 `mp.solutions` 替身",但删完那 4 条 gesture 测试就开始
  **构造真探测器、加载 21 MB 模型** —— 单文件 32.44s、依赖模型存在、并伴随 `__del__` 泄漏噪音。
  而**任务书自己的 fixture docstring 就写着"测试不该依赖那 21 MB"**。它把替身点移到
  `get_or_create_detectors` 的构造处并反向复现过(假件 `detect` 改成 raise → 4 条全红)。
  控制器复验:**32.44s → 1.61s**。代价:替身从"import 期形状"改成"构造处行为",更贴近真实。
- **Ruling: 采纳它的意见 —— 最终审查那条 grep 判据原措辞不成立,已改成按"可达性"判。**
  — 理由:`grep -v examples/` 会假红。逐条查过:face 那两处是**注释散文**;
  `gesture_analysis/utils/visualization.py:111-136` 是真代码,但那两个方法
  (`draw_hand_landmarks`/`draw_pose_landmarks`)**全仓零调用者**,且在 `try/except` 内、
  方法体内延迟 import。**`visualization.py` 本就是 spec §1 划出范围的**(留坏 + 记账)。
  代价:若将来有人给那两个方法加了调用者,它们会静默不画图 —— 所以它进 deferred 清单,
  由最终审查triage。

### Task 3: deferred minor: `gesture_analysis/utils/visualization.py:111-136` 的
`mp.solutions` 引用(死代码 + 优雅降级)与 `examples/*`(例程)一并留待单独清理。

### Task 3: complete (commits db3cca1..8cbe149, 控制器独立核验:全量 193 passed + 实跑 200)

### 最终全分支审查:complete(`final-review-report.md`,审查者全新视角、未派子智能体)

产出 4 条 Important(I1 close() 阻塞 / I2 下一步未更新 / I3 两处钉子只钉一半 /
I4 logging_config 未跟踪)+ 9 条 Minor,无 Critical。裁决:全部并入**修复波次 1**。

### 修复波次 1:complete(`fix-wave-1-report.md`,提交 `4609d8b`)

**4 条 Important 全修,无遗漏、无超范围改动;TDD 三条代码/测试改动逐条反向复现过。**
全量 193 → **198 passed**;合并门 0 / 2835510。

| 条 | 落地 |
|---|---|
| I1 | 两个模块各加模块级 `close_detached()`(刻意不共享);**6 处调用点**改走它(TTL ×3、`/reset` ×3)。**实测**:`/reset`(无 id,2 会话)20.04s → **0.01s**;期间并发 `/health` max 20.03s → **0.00s**(基线 1.05ms) |
| I2 | `docs/下一步.md` §0/§2/§3/§5/§6 按报告清单改;账本 + 任务报告 + 最终审查报告归档进**被跟踪的** `docs/superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/` |
| I3(a) | gesture `verify_models` 补 2 条(缺 hand / 缺 pose 各一半 + 文件在则不抛) |
| I3(b) | 补"同会话复用同一探测器对象"1 条(红法:去掉缓存 → 该条红,其余 193 条全绿) |
| I4 | `logging_config.py` 入 git(`git check-ignore` exit 1 = 未被忽略,无需加例外) |

**执行期的两处判断(记档):**

- **Ruling: `test_expired_sessions_close_their_detectors` 改成等条件落地。**
  — 理由:I1 把 app 侧的 close 变成异步之后,原地读 `hands.closed` 量到的是"还没跑"
  而不是"没跑" —— 那是**假红**。改成轮询(超时 5s 仍 False 才红)后语义没放宽:
  close 一次不调照样红。改前 5/5 红、改后 3/3 绿。代价:该测试多花最多 5s(实测毫秒级)。
- **Ruling: gesture 的 `api/app.py` 只按名字 import `close_detached`,不把
  `HandDetector`/`PoseDetector` 提到模块级。** — 理由:那两个名字必须留在
  `get_or_create_detectors` 体内按调用时解析,否则 `test_analyze_session_fallback.py`
  对 `detectors.HandDetector` 的 monkeypatch 会静默失效(模块级 import 绑死补丁前的值)。
  代价:同一模块里两种 import 风格并存(已就地写注释说明)。

**未做(按裁决不属本轮)**:spec §10.4 的 VIDEO 模式等价性门(登记进 `docs/下一步.md` §3
第 15 条)、spec §12.1 的 M3 阈值重登记(第 16 条)、Minor M1/M2/M5/M6/M7/M8 均未动。

### 控制器执行期发现(记档,不属于 M1.5 范围)

- **deferred: gesture 的分数哨兵值 50.0 与真实值域撞车。** 四个分析器的 `reset()` 都把
  `*_score` 初始化成 `50.0` + `is_valid=False`,上层 `emotion_inferencer.py:63-66`
  也是 `if valid else 50.0`;但**真实公式也会产出 50.0**(`hand_analyzer._compute_resilience_score`
  的基准是 70,抖动扣 20 且不握拳、无张开加成时正好 50)。所以**光看数字分不出"中性"
  还是"没测到",唯一区分是 `is_valid` 字段** —— 而消费方(报告层)目前不读它。
  更糟:无效路径的警告是**被注释掉的**(`hand_analyzer.py:91-92`),完全静默。
  **这决定了本任务的验收证据口径:必须看 `is_valid` + 分数是否 ≠50,不能只看"有响应"。**
  — 属原有设计问题,不在 M1.5 范围;`visualization.py` 之外单列。代价:若报告层将来
  直接读分数,会把"没测到"当成"中性"印出去(本项目一路在杀的就是这个)。
