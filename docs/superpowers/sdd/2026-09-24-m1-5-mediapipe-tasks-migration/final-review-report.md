# M1.5 最终全分支审查报告

- 审查范围:`234f817..8cbe149`(7 个提交),diff 取自 `.superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/review-234f817..8cbe149.diff`
- 权威文档:spec `docs/superpowers/specs/2026-09-24-m1-5-mediapipe-tasks-migration-design.md`、计划 `docs/superpowers/plans/2026-09-24-m1-5-mediapipe-tasks-migration.md`、账本 `.superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/progress.md`
- 审查者:最终全分支审查(全新视角,只报告不改动;未派任何子智能体)
- 环境:`~/miniconda3/envs/jingxin/bin/python`(mediapipe 1.0.0,`hasattr(mp,'solutions') == False`)

**一句话判定:spec §1 的 8 项在范围内事项**全部落地**(第 8 项按 spec 自己收窄后的口径:判定已交、改数按 §12.1 归 M3);**6 项必跑实跑全通过**,两个适配层金标比对逐点 100% 一致。**没有 Critical 级缺陷**;4 条 Important 全部是「合并前必须登记/决定」类(其中最硬的一条是实测出来的 `close()` 阻塞),不是错值或数据丢失。

---

## 0. Declined to judge(考虑过但明确不判的,逐条一行)

1. **本区间内的 M1 收尾提交 `54ce69e`**(`load_markers` 深拷贝、`log_prosody` 改为抛出、`sources_disclosure`、空串口径三副本、`report_frontend/*`)—— 属 M1 自己的最终审查修复波次(D1/D2/D3/I3/I4),账本已按其裁决记档;我只做了集成面核对(见下 §5 的 R1),不重新审它。
2. `face_expression/examples/*`、`gesture_analysis/examples/*` 的 `mp.solutions` —— spec §1 明确划出范围;实测不在 import/调用路径。
3. `gesture_analysis/utils/visualization.py:111-136` —— spec §1 明确划出范围;实测**全仓零调用者**(见 §2 项 3)。
4. `gesture_analysis/examples/main_integrator.py:67-68` 用旧键名展开 `MEDIAPIPE_CONFIG`(迁移后会 `TypeError`)—— 与第 2 条同一处置。
5. `experiments/extract_features.py` 的 `multi_handedness` 形状变更 —— spec §1 列为论文线,不动。
6. `gesture_analysis/api/app.py:261` 的 `hand_id == 0 → left_hand` 位置判断 —— spec §9 第 7 条明令不修;确认这一行没被顺手改动。
7. `code_data_supplement/modules/.../video_pipeline.py` 的陈旧副本 —— 不在 import 路径(见 Minor M7)。
8. `validate_session_id` / `normalize_session_id` 在 voice/face/gesture 三处各持一份 —— Ruling M1-2 的有意为之,且有源码级同一性测试压着,不判为重复。
9. 50.0 哨兵值与真实值域撞车(账本「控制器执行期发现」)—— 原有设计问题,不在 M1.5 范围;我只据它确定验收口径(必须看 `is_valid` + 分数 ≠50)。
10. 本 diff 未新增任何外部输入面(没有新端点、新文件写路径、新反序列化点),故未做安全面专项审查。

---

## 1. spec §1 逐条合规判定(8 项在范围内事项)

| # | spec §1 要求 | 实现了吗 / 在哪 | 判定 |
|---|---|---|---|
| 1 | face 探测器 `mp.solutions.face_mesh.FaceMesh` → `vision.FaceLandmarker` | ✅ `face_expression/pipeline/detector.py:29-77`(`FaceDetector`,`RunningMode.VIDEO`,`detect_for_video`);消费点 `face_expression/pipeline/video_pipeline.py:31,39-43`(惰性属性已删) | 落地 |
| 2 | gesture 探测器 `Hands`/`Pose` → `HandLandmarker`/`PoseLandmarker` | ✅ `gesture_analysis/core/detectors.py:76-141`(`HandDetector`/`PoseDetector`);消费点 `gesture_analysis/api/app.py:255-279` | 落地 |
| 3 | gesture 探测器改为**按会话**(D3) | ✅ `gesture_analysis/api/app.py:48`(`detectors` 表)、`:163-185`(`get_or_create_detectors`);模块级 `hands`/`pose` 已删,`api/app.py` 无模块级 `import mediapipe`。实跑证:3 帧 / 2 会话只产生 2 条「创建手势探测器会话」(同会话第 2 帧复用),两会话对象不同(§2 项 2/6) | 落地 |
| 4 | 模型文件的路径契约与加载封装 | ✅ `face_expression/config.py:37-38`(`MEDIAPIPE_MODELS_DIR`/`FACE_MODEL`)、`gesture_analysis/config.py:61-63`(`HAND_MODEL`/`POSE_MODEL`);探测器代码里**零硬编码路径**(路径全部由构造参数传入);`.gitignore` 末行 `models/mediapipe/`,且 `requirements` 未把模型塞进包 | 落地 |
| 5 | `gesture_analysis/config.py` 的 `MEDIAPIPE_CONFIG` 改写 | ✅ `gesture_analysis/config.py:73-84`:`max_num_hands→num_hands`、`static_image_mode` 消失(`running_mode` 固定 VIDEO,进不了配置)、`model_complexity` 换成显式 `tier: 'full'` + `num_poses`;`FaceLandmarkerOptions` 的置信度键名映射由封装负责,配置保留中性键名 | 落地 |
| 6 | `requirements*.txt` 钉 `mediapipe==1.0.0` | ✅ `requirements.txt:15`、`requirements-full.txt:16`(均带「>=0.8.0 允许 1.0 进来」的成因注释) | 落地 |
| 7 | `tests/test_analyze_session_fallback.py` 替身重指 | ✅ 该文件**已不再 import mediapipe**(替身整段删除,`:48-61`);genuine 需要的那处替身改到**构造处**(`_FakeGestureDetector` `:140`,fixture `:260-261` monkeypatch `detectors.HandDetector/PoseDetector`);`python_multipart` 替身保留。**偏离**:计划写的是「给裸 `import mediapipe` 加守卫」,实施改成**整段删除** —— 更强(没得坏就没得防),且账本已裁决。另:替身从 import 期移到构造处也是账本裁决过的(单文件 32.44s → 1.61s) | 落地(两处偏离已裁决) |
| 8 | 基于 §3.4 实测的阈值/量程重登记 | ⚠️ **按 spec 自己收窄后的口径**:spec §0 第 3 条把交付定为「**判定**」而非改数,数值改动按 §12.1 归 M3。已交的判定 = spec §3.4 结论 3(29 个 face 特征里 11 个相对位移 >10%、4 个 >25%,`au23_lip_compression` 55.8% / `au25_mouth_open` 46.4% / `gaze_direction_x` 28.0% / `au26_jaw_drop` 25.4%,**方向一致偏小**=系统性偏置)+ §9.1/§12.1(位移是模型包差异,改代码修不回来,只能接受并记进阈值依据)。**未做的部分**:没有把「受影响的是 `evidence_thresholds.json` 的哪几条」点名到条目上(该文件只有 9 个族级键:`blink/gaze/au/head_pose/pause/pitch/energy/speech/density`,位移特征属于 `au` 族 + `scale_factors`),而 spec §12.1 说这一步要与逐列重构一起做 | 部分(见 Minor M5) |

**补充判定(spec §0 的三条成功定义)**:①两个服务返 200 并落出带 `session_id` 的日志 —— ✅ 实跑证(§2 项 1);②数值影响有实测记录 —— ✅ spec §3.4 的逐部位位移表(28 帧、IMAGE 模式、三个部位);③受影响阈值被指名并给出依据 —— 见上表第 8 项,判到「特征级指名」为止。

---

## 2. 必跑 6 项的实跑结果(全部跑了,下面每条都是实际命令与输出)

### 项 1:两个服务各起一次、各发一帧真图 → **通过**

`~/shared/mp_frames/frames/frame_0001.png`(face)/ `frame_0009.png`(gesture),640×480 真摄像头帧。

```
face   (:8000)  POST /analyze  sid=m15rev_face_a  -> HTTP=200
                POST /analyze  sid=m15rev_face_b  -> HTTP=200
                POST /analyze  sid=m15rev_face_a(同会话第 2 帧) -> HTTP=200
                POST /session/m15rev_face_b/reset -> 200 {"status":"success",...}
                再发一帧 m15rev_face_b -> HTTP=200
   落盘:data/logs/face_au_log_m15rev_face_a.csv / _face_b.csv,表头 session_id,...,
        数据行首列 = m15rev_face_a / m15rev_face_b(1019 / 1010 字节)
gesture(:8002)  POST /analyze  sid=m15rev_g_a -> 200;  sid=m15rev_g_b -> 200
                同会话第 2 帧 -> 200;  /reset?session_id=m15rev_g_b -> 200 后再发 -> 200
                /reset(无 id)-> 服务端 200(客户端 curl 因 --max-time 20 超时,见 I1)
   落盘:data/logs/gesture_emotion_log_m15rev_g_{a,b,c}.csv,首列 = 该会话 id
```

**迁移前不可能做到**:face 迁移前每帧 `AttributeError → 500` 且永不落 face 日志;gesture 迁移前 import 期即死。两者现在都返回 200 且日志带 `session_id`。

**顺带在真服务上验了反摊平不变量(比 200 更重要)**:gesture 响应 `detected_hands: 2`,四个分析器
`is_valid: true`,分数 **58.43 / 53.84 / 70.0 / 90.0** —— **不是**静默回落的 50.0。这是 spec §3.3 那条静默失效在真实服务上的端到端反证。

### 项 2:金标比对(spec §10.3,判据 = 逐点一致)→ **通过(适配层 100% 逐点一致)**

`probe.py` 的 `_detector_tasks` 原本**自己构造** `FaceLandmarker`,那样量的是探针而不是适配层。我把
`probe.py` / `probe_gesture.py` 复制到 `/tmp/m15review/probe/`(**不碰仓库,也不碰 `~/shared/mp_frames/` 下的基线**),在副本里改成直接用仓库封装:

- `face_expression.pipeline.detector.FaceDetector`
- `gesture_analysis.core.detectors.{HandDetector,PoseDetector}`

**这里有一条必须写清的判据问题(重要发现,见 Minor M9 的由来)**:仓库封装的 `running_mode` **固定在 VIDEO**(spec 决定),而 `tasks.json` / `gesture_tasks.json` 基线是 **IMAGE 模式**录的。所以「原样用 `FaceDetector` 去比 IMAGE 基线」**在数学上不可能逐点一致** —— 差的不是适配层,是模式。我跑了两条臂:

| 臂 | 命令 | 结果 |
|---|---|---|
| **A. 原样用仓库封装(VIDEO)** | `probe_adapter.py --engine tasks` | face:1425/9082 相等,**2 帧检出形态分歧**(`frame_0008` 旧无/新有、`frame_0009` 旧有/新无);gesture:152/1056 相等,**6 处形态分歧**(手 1↔2 组、姿态 0↔1 组) |
| **B. 同一个适配层 + IMAGE 模式内层替身**(`--engine tasks-image`:把内层 landmarker 建在 `RunningMode.IMAGE`,用 `factory=` 注入一个把 `detect_for_video` 转发到 `detect`、忽略时间戳的壳) | `probe_adapter.py --engine tasks-image` / `probe_gesture_adapter.py --engine tasks-image --pose-model full` | **face 9560/9560 点完全相等(100.000000%),0 帧形态分歧**;**gesture 1119/1119 点完全相等(100.000000%),0 处形态分歧** |

→ **结论:适配层写对了。** B 臂逐点 100% 相等,说明颜色通道、`mp.Image(SRGB, ascontiguousarray)`、置信度参数、结果索引(`face_landmarks[0]` / `hand_landmarks` / `pose_landmarks[0]`)、摊平口径**一处都没改变输入**。A 臂的差**全部可由模式差异解释**(别处也自证:`gesture` IMAGE 检出 25 帧姿态 / VIDEO 检出 28 帧,人手 11 vs 10 —— 跟踪确实在起作用),而这正是 spec §10.4 单列的那道门,不是适配层缺陷。

文件:`/tmp/m15review/probe/{probe_adapter.py,probe_gesture_adapter.py,exact_compare.py}`,产出 `face_adapter_{video,image}.json` / `gesture_adapter_{video,image}.json`,基线副本 `baseline_tasks_face.json` / `baseline_gesture_tasks.json`(md5 `718b85551cecf042661e0ab7fc4b09b2`)。

### 项 3:`mp.solutions` 在活路径上为空 → **通过(按可达性逐条查过)**

```
$ grep -rn "mp\.solutions|mediapipe\.solutions|solutions\.hands|solutions\.pose|solutions\.face_mesh" --include="*.py" face_expression/ gesture_analysis/
```
逐条可达性判定:

| 命中 | 性质 | 判定 |
|---|---|---|
| `face_expression/api/app.py:373` | **注释散文**(讲旧探针为何被换掉) | 非代码 |
| `face_expression/pipeline/video_pipeline.py:179` | **注释散文**(讲第二槽的来历) | 非代码 |
| `face_expression/examples/run_video_analyzer.py:92-99`、`gesture_analysis/examples/{run_realtime_analyzer.py:64,65,141,182,183,213; main_integrator.py:67,68,131,254,255,308,336,338}` | 例程,不在服务路径;`main_integrator.py:67-68` 还传旧键名 | spec §1 划出范围 |
| `gesture_analysis/utils/visualization.py:111,112,135,136` | **真代码,但零调用者** | 见下 |

`visualization.py` 那两条独立核实过:
- `grep -rn "draw_hand_landmarks\|draw_pose_landmarks"` → 全仓**只有定义、零调用**(`gesture_analysis/utils/__init__.py:2` 只 import 了 `Visualizer` 类本身,两个 examples 也只 import 类);
- 两处都在 `try/except Exception` **内**、且 `import mediapipe as mp` 在**方法体内**(import 期不炸,真调到只打印警告并原样返回图像)。

→ **活路径为空成立。** `visualization.py` 是 spec §1 明确划出范围的死代码 + 优雅降级,**不算活路径违规**,triage 见 §4。

附带亦为空:`grep -rn "MEDIAPIPE_CONFIG" --include="*.py" face_expression/` → 只有 `config.py:27` 的一条注释(死代码已删,删除后全量零回归,反证它确实没人用)。

### 项 4:合并门 → **通过**

```
cd ~/jingxin/experiments/duration_audit
~/miniconda3/envs/jingxin/bin/python reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_probe_review
~/miniconda3/envs/jingxin/bin/python reaggregate_normalized.py --verify-legacy /tmp/legacy_probe_review
```
输出要点:`2011 视频 × 1410 维 (face=1260 gesture=140 voice=10),死特征 169`;`特征名集合完全一致`;
`最大绝对差 1.886e-19`;`相对差 > 1e-4 的格子: 0 / 2835510`;`targets 最大绝对差 0.000e+00`;
`✅ 回归验证通过`。→ **0 / 2835510** ✅

### 项 5:全量测试 → **通过**

```
~/miniconda3/envs/jingxin/bin/python -m pytest -q
-> 193 passed in 10.83s
```
与账本记的 193 一致(计划里写的基线 177 是 T1 之前的数)。`requirements*.txt` 已钉 `==1.0.0`(已复核文件内容)。

### 项 6:Review Focus 六条逐条对测试确认有钉子 → **六条都有钉子;其中两条只钉了一半**

| # | Review Focus | 钉子(测试 + 位置) | 我的独立复现 |
|---|---|---|---|
| ① 同会话 ts 严格递增 | `tests/test_detector_contract.py:103`(face,断言 `ts == [0,100,200]` 形状) | ✅ **真探测器实测**:`FaceDetector(FACE_MODEL, fps=10)` 连发 3 帧,交给 mediapipe 的 ts = `[0, 100, 200]` 严格递增;**牙齿检查**:重复 ts 与回退 ts 都真的抛 `ValueError: Input timestamp must be monotonically increasing.` → 「非单调会抛错」成立,所以这个计数器不是形式主义 |
| ② `/reset` 后不回退 | `tests/test_detector_contract.py:118`(容器层)+ `tests/test_face_detector_wiring.py:52`(`VideoPipeline.reset()` 转发) | ✅ **真对象实测**:gesture `/reset?session_id=…`(命中)与 `/reset`(无 id)两条路径都把会话从探测器表移除、**并把 native 句柄置 None**;face `/session/{sid}/reset` 同。实跑 HTTP:reset 之后同会话再来一帧 **200**(face 与 gesture 都有)。⚠️ 注意实现选择:两个端点的语义都是**销毁会话**(不是保留会话),所以帧计数归零是靠「重建探测器」而非 `reset()` —— 见 Minor M1 |
| ③ TTL 回收 `close()` | `tests/test_detector_contract.py:133`(封装)、`tests/test_gesture_detector_wiring.py:87`(gesture app 表)、`tests/test_face_detector_wiring.py:59`(face pipeline) | ✅ **真 native 句柄实测**(账本说「任务级审查看不到的那种缝」):把会话推过 `SESSION_TIMEOUT` 再请求,过期会话**从表里移除**且 `dets['hands']._landmarker is None` / `_pose… is None`(face 侧 `pipeline.detector._landmarker is None`)—— 真关到了 native,不只是从 dict 删掉 |
| ④ 两会话拿到不同探测器 | `tests/test_detector_contract.py:146`、`tests/test_gesture_detector_wiring.py:60` | ✅ **真对象实测**:`get_or_create_detectors('sA')` vs `('sB')` → hands/pose 都是不同对象;`import` 期不构造(测试 + 实跑:三次请求只 2 条创建日志) |
| ⑤ 模型缺失启动即失败 | `tests/test_detector_contract.py:160`(**只有 face 的 `verify_models`**) | ✅ **真启动实测**:把 config 的路径改成不存在后 `runpy.run_module(..., run_name="__main__")` —— face:`RuntimeError('人脸模型文件不存在:/nonexistent/mp/face_landmarker.task …')`;gesture:`RuntimeError('手势/姿态模型文件不存在:…')`。异常未被吞,`uvicorn.run` 到不了,等于**启动即失败**。⚠️ **gesture 那条 `verify_models` 没有单测**(见 I3) |
| ⑥ ★ 摊平成元组 → 静默回落 50.0 | `tests/test_detector_contract.py:74`(`hasattr(groups[0][0],'x')`)、`tests/test_gesture_detector_wiring.py:98`(`is_valid is True` + 分数 ≠50) | ✅ **独立复现,与文档字字对上**:`HandAnalyzer` ← 带 `.x/.y` 对象 = `is_valid=True, resilience_score=51.4663`;← `(x,y)` 元组 = `is_valid=False, resilience_score=50.0000`,**不抛异常、不打印任何东西**。`ArmAnalyzer` 同形:`is_valid=True, arm_score=90.0` vs `is_valid=False, arm_score=50.0`。→ 两条钉子都有牙齿:摊平后 `hasattr` 为 False、`is_valid` 为 False,两条断言必红。**真服务侧也验了**(分数 58.43/53.84/70/90 + `is_valid: true`) |

---

## 3. Findings

### Critical(必须修)

**无。** 我逐条找过 C1 那一类「产物遮蔽 / 静默错值 / 数据丢失」形态:两个服务的输出值经金标比对与真跑双重确认(适配层逐点 100%、真服务 `is_valid=true` 且分数非哨兵)、日志首列与文件名一致、`session_id` 口径三副本同一、探测器的生命周期(建/复用/回收/重置)在真 native 对象上逐条验过。没有发现会产生错值或静默丢数据的地方。

### Important(应当修 / 合并前必须有决定)

---

**I1. `close()` 每次 5.00 秒且串行发生在请求路径上 → TTL 回收与 `/reset` 会把整个服务冻住 5–20 秒(实测,已复现)**

- 位置:`gesture_analysis/api/app.py:127-133`(TTL 回收)、`:169-173`(`get_or_create_detectors` 里**第二遍**回收)、`:470-483`(`/reset` 两条路径)、`face_expression/api/app.py:125-126`(TTL)、`:361-362`(`/session/{sid}/reset`);被调的 `close()` 在 `face_expression/pipeline/detector.py:72-77`、`gesture_analysis/core/detectors.py:139-143`。
- 失效场景:任何一次回收都要同步关掉 2 个(gesture 每会话)或 1 个(face)native landmarker,而**每次 `close()` 实测 5.00 秒**、调用点全在 `async def` 端点里同步执行 → 阻塞事件循环。
- 实测数据:
  - 直接计时:`HandDetector 构造 0.37s / close **5.01s**`,`PoseDetector 构造 0.08s / close **5.00s**`,`FaceDetector 构造 0.03s / close **5.00s**`(构造便宜、close 恒定 5 秒,像是库内部等一个 join/超时,不是我们的代码)。
  - 进程内真对象:TLL 回收一个过期会话(2 个探测器)在请求路径上阻塞 **10.12s**;face 一个会话 **5.02s**。
  - **真服务并发实验**(2 个活跃会话,基线 `/health` = 0.0019s):`POST /reset`(无 id)耗时 **20.04s**,**并发**的 `/health` 耗时 **19.73s**(之后恢复 0.002s 级)—— 即整个单 worker uvicorn 在此期间不可用。
  - 同一现象在后面那次 curl 上也复现过:`--max-time 20` 的 `/reset` 拿到 `HTTP=000`(客户端超时),而服务端日志最终记 `200 OK`。
- 影响面:活跃面试期间每帧都刷新 `last_used`,不会触发回收;**代价出现在「静默 >5 分钟后回来的那一帧」**(face 5s / gesture 10s 额外延迟)与**候选人之间的 `/reset`**(O(会话数) × 10s,无上限)。`/reset` 无 id 那条尤其危险:它会关掉**所有**会话的探测器,所以在 T7 的 curl 流程或前端里点一次「重置」= 服务冻结若干秒。
- 归因(重要):**`close()` 本身是 spec §6.3 强制要求的**(「回收必须显式 close(),只从 dict 里删掉会泄漏」),spec 只是没考虑它的代价。所以这既是实现的问题也是 spec 的遗留 —— 我没有把它记成 Critical,因为它**不产生错值、不丢数据、不崩**;但它是一个**新引入**的可用性回归(迁移前 gesture 是模块级单例、从不 close;face 也从不 close),因此在合并前**必须有决定**:要么这一轮就把回收挪出事件循环(如 `await asyncio.to_thread(...)`,或在后台任务里做),要么在账本里登记成明确的下轮首项。
- 复现素材:`/tmp/m15review/{seams.py,blocking.sh}`,日志 `/tmp/m15review/gesture_block.log`。

---

**I2. `docs/下一步.md` 没更新,而它正是两条 deferral 的唯一登记处 → 两个「已知未验」项会丢(计划自己把这一步列为本轮必需输出)**

- 位置:`docs/下一步.md:3`(`更新日期…下一步 = face/gesture 迁 mediapipe tasks API`)、`:19`(`spec 待写。修完再跑 T7`)、`:84`(跟进项第 1 条「face/gesture 迁 tasks API — 下一步主线,spec 待写」)、`:156`(`M1.5 … ← 当前,spec 待写`)、`:139`(`CV 服务(待迁 tasks API)`);§2 整节仍把两个服务描述成「一个每帧静默 500、一个进程直接死」。`git log -1 -- docs/下一步.md` = `d0c6c3b`,即**在 M1.5 动工之前**改的。
- 失效场景:下一个会话按本文件是唯一入口(「看完 §0 与 §2 就能接着干」),它会读到「M1.5 待做、spec 待写」而**重做一遍已完成的工作**;同时两件真正欠着的事**在任何跟进清单里都不存在**:
  1. **spec §10.4 的 VIDEO 模式门**(spec 允许留到迁移后,但要求「必须记档,不许当成已经验过」)—— 只写在 M1.5 spec 内部,`docs/下一步.md` §3 的 14 条里没有它;
  2. **spec §12.1 的 M3 阈值/量程重登记**(§3.4 的数据要接进 `evidence_thresholds.json`)—— 同样只在 M1.5 spec 内部。
- **而且这道 VIDEO 门不是形式主义**:我项 2 的 A 臂给出实测 —— 同一批帧、同一模型,IMAGE 与 VIDEO 的**检出形态都不同**(gesture 姿态 25 帧 → 28 帧、手 1 组 → 2 组;face `frame_0008/0009` 一个有脸一个没脸)。也就是说 spec §3.4 那张位移表(IMAGE 模式)与生产(VIDEO 模式)**不是同一件事**,这个差别现在有数了,更应该进跟进清单。
- 另:计划的「最终全分支审查」清单把「更新 `docs/下一步.md`」写成本轮必需输出(计划 `:1180`),账本也没有任何一条说它已完成。

---

**I3. 两处钉子只钉了一半(测试缺口,不是缺陷)**

- (a) **gesture 的 `verify_models` 没有测试**。`tests/test_detector_contract.py:13` 只 import 了 face 的 `verify_models`;`tests/test_detector_contract.py:160,173,183` 三条都只打 face 那一个。而 spec §8 / Review Focus ⑤ 对**两个服务**都成立,且两者的签名不同(`(model_path, factory)` vs `(hand_path, pose_path, factory)`)—— 正因签名不同才更容易漏。行为本身我已实测正确(§2 项 6 的启动失败run),但「缺断言 = 没有约束力」是这轮自己定的规矩。
  - 补法:`verify_models(hand_path=missing, pose_path=ok, factory=…)` → 期望 `RuntimeError` 且消息含该路径(两半:hand 缺、pose 缺)。
- (b) **没有任何测试钉住「同一会话必须复用同一个探测器/管线」**。扫过全部四个测试文件,`detectors` 表相关的断言只有 `a["hands"] is not b["hands"]`(不同会话**不同**)与 `"t_old" not in gapp.detectors`(回收**移除**);**没有**「同 sid 第二次调用是同一对象」。`tests/test_analyze_session_fallback.py:284`(两帧进同一文件)钉的是**日志文件**不换名,日志器与探测器**各自**存在 `session_loggers` / `detectors` 里,所以它挡不住「探测器每请求重建」。
  - 失效场景(这决定了它值得记):把 `get_or_create_detectors` 改成无条件新建 → **193 条测试全绿**,而生产上同一会话第二帧的 ts 恒为 0 → `ValueError: Input timestamp must be monotonically increasing.` 每帧 500,并且**每帧泄漏一个 native landmarker**。这是「静默通过/响亮失败」的混合体:测试侧静默、生产侧响亮 —— 按本仓的规矩该由测试侧兜住。
  - 补法:一行断言 `assert gapp.get_or_create_detectors("s") is gapp.get_or_create_detectors("s")`;face 侧同理数 `VideoPipeline` 的构造次数。
- 说明:我把这两条判为 Important 而不是 Minor,因为 ⑥ 那类「没有断言就没有任何东西会红」正是本仓一路在杀的形态,而这两条恰好落在 Review Focus ⑤ 与 D3 的**核心不变量**上。

---

**I4. `logging_config.py` 未跟踪,却被两个服务的活代码 import(已登记的地雷,M1.5 让它变成验收路径上的地雷)**

- 位置:`face_expression/api/app.py:11`、`gesture_analysis/api/app.py:11`(以及 voice)都 `from logging_config import setup_logging`;`git status` 显示 `?? logging_config.py`。
- 失效场景:任何 `git clean -fdx`、任何全新 clone/CI、任何在别的机器上 checkout 这个分支 → 两个服务**都起不来**(ImportError),而 T7 的验收判据现在要求 face + gesture + voice 三份日志。`docs/下一步.md:135` 已把它登记为「必须入 git 的地雷」,但**没有做**。
- 与本 diff 的关系:不是本次引入,但本次把这两个服务从「死的」变成「T7 验收必须活的」,所以它的紧迫性由本次改动提升。合并前 `git add logging_config.py` 是一行的事。

---

### Minor(Nice to Have)

**M1. `VideoPipeline.reset()` 没有生产调用者** —— `face_expression/pipeline/video_pipeline.py:184-186` 定义了 `reset()`,但有契约测钉着、**没有任何端点调**;face 的 `/session/{sid}/reset` 选择销毁会话(`face_expression/api/app.py:352-359` 的 docstring 明确写了为什么)。Review Focus ② 的目标由「销毁 + 重建」达成,所以不是缺陷;但一个只在测试里活着的公开方法,要么接上、要么在 docstring 里说清它留给谁。

**M2. TTL 回收逻辑写了两遍,都在关探测器** —— `gesture_analysis/api/app.py:127-133`(`get_or_create_analyzers` 的回收段)与 `:169-173`(`get_or_create_detectors` 的回收段)各 pop + close 一次。第二遍在现有调用顺序下是防御性的(第一遍已经 pop 过),但两处必须同步维护:改一处漏一处不会红,而后果是泄漏 native 句柄。

**M3. 计划的修正版没提交,工作树是脏的** —— 工作树的 `docs/superpowers/plans/2026-09-24-m1-5-mediapipe-tasks-migration.md` 相对 `8cbe149` 有 +81/−13,里面装着**本轮审查判据的修正版**(`:1161-1175` 把「照字面 `grep -v examples/`」改成按**可达性**判,正是账本记的那条)与开工前预检修掉的三条缺陷;而**已提交的版本 `8cbe149` 仍写着旧的、会假红的判据**(`git show 8cbe149:…:1107`)。不提交 → 换机器/合并后这套修正就没了。附带同一文件 `:1179` 仍写「Review Focus **五条**」而它自己的 Review Focus 节列了 **6 条**。

**M4. 账本落在被 gitignore 的目录里,M1.5 的证据进不了仓库** —— `.superpowers/sdd/.gitignore` 是 `*`(`git check-ignore` 已确认),所以 `.superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/progress.md`(146 行,含逐任务裁决与反向复现记录)**不在版本控制里**;计划 `:1180` 要求把它落进 `docs/superpowers/sdd/`(M1 的账本就落在那里,是跟踪的),`docs/superpowers/sdd/` 下**没有** M1.5 目录。

**M5. §1 第 8 项的「指名」停在特征名,没落到阈值条目** —— 已交的判定在 spec §3.4 结论 3 / §12.1;但 `report_frontend/evidence_thresholds.json` 只有 9 个族级键(`au` 是其中一族)与 `scale_factors`,spec 没有点名「要重登记的是 `thresholds.au` 与 AU 的 `scale_factors`,依据是 11 个 >10% 的系统性偏小」这一句。spec §12.1 把落地推给 M3 是对的,但一句条目级指名能让 M3 直接接上(spec §0 说第 2、3 条是本设计的重点)。

**M6. 三个 `.task` 只记了体积,没有内容哈希** —— spec §6.1 记的 3758596 / 7819105 / 9398198 与磁盘**完全一致**(我复核过);但 §10.3 的判据是「逐点一致」,它成立的前提是**同一份模型字节**,而 §6.1 同时说版本钉法未验证(URL 走 `…/float16/latest/…`)。一旦有人重下 `latest`,基线 JSON、§3.4 的位移表、乃至「逐点一致」的结论都会静默失效,而仓库里没有任何东西能发现这件事。建议把三个文件的 sha256 记进 spec §6.1(一行一个)。

**M7. `code_data_supplement/modules/face_expression/pipeline/video_pipeline.py` 已经与活文件漂移** —— 迁移前逐字节相同,现在 `diff` 报不同(spec §12.6 预言了这件事,并要求「显式决定:更新它,还是标记为投稿时的冻结快照(倾向后者)」)。**没有任何地方记下这个决定**。一行注释或一行账本即可关闭它。

**M8. 启动自检只在 `__main__` 里,换一种起法就绕过** —— `face_expression/api/app.py:377-378`、`gesture_analysis/api/app.py:495` 在 `if __name__ == "__main__"` 块内调 `verify_models()`。按文档的 `python -m …` 起法没问题(已实测);但 `uvicorn gesture_analysis.api.app:app` 这类 ASGI 直接起法**不会**执行它,于是又回到 spec §8 要消灭的「启动看着正常、第一帧才炸」。修法要权衡:挪到 import 期会破坏 D3 想要的「测试能在没 mediapipe/没模型的环境里 import 这个模块」,所以更合适的是 FastAPI 的 lifespan/startup 钩子。

**M9. 计划的最终审查判据与仓库封装的实际模式不匹配(文档口径)** —— 计划 `:1153` 要求「`probe.py` 改成用仓库的 `FaceDetector`,判据是逐点一致」,但仓库封装的 `running_mode` 固定在 **VIDEO**、基线 JSON 是 **IMAGE** 录的,所以照字面做**必然不一致**(我实测得 1425/9082)。建议把那条判据改写清楚:「用适配层 + IMAGE 模式内层(或先跑 §10.4 的 VIDEO 基线)」,否则下一个人会照字面跑一次、得到一个假的❌。

---

## 4. deferred 清单 triage(合并前修不修)

| deferred 项 | triage | 依据 |
|---|---|---|
| `gesture_analysis/utils/visualization.py:111-136` 的 `mp.solutions` | **不修** | spec §1 明确划出范围;**实测零调用者**、在 `try/except` 内、方法体内延迟 import → 不在活路径。唯一残余风险是「将来有人给它加调用者,会静默不画图」(只打印一句警告后原样返回图像),已记档即可 |
| `gesture_analysis/examples/*`、`face_expression/examples/*` | **不修** | 例程,不在服务路径;`main_integrator.py:67-68` 还传旧键名(`max_num_hands`/`model_complexity`),本来就已经坏了。spec §12.5 要求单独决定「修/删/标冻结」,已记 |
| 50.0 哨兵值与真实值域撞车(账本末段) | **不修,但口径必须带上** | 原有设计问题、不在 M1.5 范围。它的后果是直接的:**光看分数分不出「中性」与「没测到」,唯一区分是 `is_valid`**,而消费方(报告层)不读它。这决定了验收口径(账本已写,我的实跑也照它做的:看 `is_valid=true` 且分数 ≠50) |
| `experiments/extract_features.py` 的 `multi_handedness` | **不修** | 论文线,spec §1 已划出 |
| `code_data_supplement/…/video_pipeline.py` 漂移 | **不修,但要一行决定** | 见 M7:spec §12.6 要求显式决定(倾向「冻结快照」),现在漂移已经发生而决定缺席 |
| `logging_config.py` 未跟踪 | **合并前修(一行 `git add`)** | 见 I4:两个服务的 import 依赖它,而 T7 的验收判据现在要求这两个服务活着 |

---

## 5. 其他核对(不是 finding,但审查要求我确认过)

- **R1. 本区间内 M1 收尾提交的集成面**:`log_prosody` 改为抛出后,调用点 `voice_interaction/api/app.py:380-384` 的异常确实被 `:393-396` 映射成 `HTTPException(500)`(不是被吞成 200),D2 的修复真的到得了客户端。同文件 `:387-390` 的 `try: save_log() except Exception: pass` 仍在 —— 是 `下一步.md` §3 第 8 条已登记的「另一处同形吞异常,不在裁定范围内」。
- **R2. 第二槽改动的消费面**:`process_frame` 的第二槽从 mediapipe 原始 `results` 换成 `landmarks_norm`,全仓消费点只有 `face_expression/api/app.py:226`(绑成 `mesh_results`,**之后一次都没读**)、`experiments/extract_features.py:88`(丢弃)、两个 examples(本环境跑不起来)。账本对这条的裁决与我一查一致。
- **R3. 分析器/特征抽取器一行未改**:`grep "\.visibility\|\.presence" gesture_analysis/core/` → **零命中**(只有探测器 docstring 里的文字),face `core/` 亦零命中。所以 spec §3.3 的「零改动」不依赖 tasks 与 solutions 在这些属性上的差异 —— 它是真的不需要它们。这也是真服务那次 `is_valid=true` 成立的原因之一。
- **R4. 模型文件与 spec §6.1 的一致性**:磁盘上三个文件体积与 spec 记的 3758596 / 7819105 / 9398198 逐一相符。本地多出的 `pose_landmarker_lite.task` 未进 `.gitignore` 之外任何清单,与本次改动无关(它就是 §3.4 那个 42px 假象的来源,留作对照)。
- **R5. 未破坏的分支状态**:全程未改动工作树/索引/HEAD(唯一的 `M` 是我到达前就存在的计划文件;`?? logging_config.py` 与 `code_data_supplement/`、`experiments/*` 也都在我到达前如此)。所有实跑用到的改动副本都在 `/tmp/m15review/` 下,基线与帧目录未被覆盖。

---

## 6. Strengths(做对的地方,具体)

1. **两个封装的返回形状刻意相反,而且两侧都有钉子**:face 摊平成 `(x, y)`(`face_expression/pipeline/detector.py:66-68`,下游 `au_calculator` 吃元组)、gesture 原样交对象(`gesture_analysis/core/detectors.py:70-82`,分析器靠 `.x/.y`)。**钉子不是靠注释而是靠断言**:封装侧 `hasattr(groups[0][0],'x')`、消费侧 `is_valid is True` + `分数 != 50.0`。我独立复现了失效形态(元组→`is_valid=False`、`50.0000`、**不抛异常、不打印**),确认这两条断言真的会在摊平时红。**这是本 diff 最有价值的一处** —— 它挡住的是「报告里印着一个什么都不代表的数」。
2. **真 native 对象的生命周期是完整的**:建(按会话、不共享)→ 复用(同会话第二次是同对象)→ 回收(TTL 与 `/reset` 都 close 且置 None)→ 重建(reset 后可用)。四项我都在**真句柄**上验过,不是靠替身推的。
3. **`close()` 被写成幂等**(`if self._landmarker is not None: … = None`),所以 TTL 与 `/reset` 撞车、或 `pipeline.close()` 被调两次都不会二次释放 —— 这在 native 资源上本来是最容易 double-free 的地方。
4. **启动自检不吞异常**:`face_expression/api/app.py:371-378` 明确删掉了旧的 `except Exception` 包裹并写了理由(旧探针「吞掉一切 → 启动看着正常、每帧静默 500」),实测两个服务都能拦住启动并报出期望路径。
5. **契约测不 import mediapipe、不依赖 21 MB 模型**(`tests/test_detector_contract.py` 只用注入的假工厂),而且**反向复现过**(账本:`_groups` 摊平 → 精准红;`str` 路径 → 先红后修)。测试替身从 import 期移到构造处之后,单文件从 32.44s 回到 1.61s —— 测试的**代价**也被当成了设计约束。
6. **`_Base` 用 `_result_attr` + `staticmethod(_default_hand_factory)` 消掉了重复**,而不是把两个封装合并(合并会反转依赖方向)—— 与 spec §4 的取舍一致。
7. **`normalize_session_id` 的收口有源码级同一性测试**(`tests/test_session_id_normalization.py:86-124` 抽三份副本的函数源逐字比),把「改一处漏两处」这种静默分叉变成了会红的测试。
8. **两个 config 的注释把「不是改名,是换概念」写清了**,并显式记下 `model_complexity=1 ↔ pose_landmarker_full` 与 42px 假象的教训 —— 这正是 spec §9.5 要防的「下一个人重犯」。

---

## 7. Recommendations

1. **合并前**:`git add logging_config.py`(I4);`git add` 计划的修正版(I3/M3);在 `docs/下一步.md` §3 补两条(VIDEO 门 §10.4、M3 阈值重登记 §12.1)并把 §0/§2/§5/§6 的 M1.5 状态改成「已实现、待验收」(I2),把账本落到 `docs/superpowers/sdd/2026-09-24-m1-5-mediapipe-tasks-migration/`(M4)。
2. **对 I1 做一个明确决定**(我倾向这一轮就做,因为代价是两行):把 TTL 回收与 `/reset` 里的 `close()` 挪出事件循环(例如收集待关闭的探测器、`await asyncio.to_thread(lambda: [d.close() for d in victims])`),这样 `/health` 与 `/analyze` 在回收期间不再被冻。若决定推到下轮,请在账本里写成带数字的首项(「每次 close 5.00s,`/reset` 无 id = 10s × 会话数,期间服务不可用」)。
3. **给 I3 的两条缺口各补一条测试**(都只有一两行),并把「同会话复用同一探测器」写成注释里的不变量。
4. **把 M9 的口径写进计划**:金标比对的判据必须是「适配层 + 与基线相同的 running_mode」,否则下一个人会照字面跑出假的❌。
5. **把三个 `.task` 的 sha256 记进 spec §6.1**(M6),让「逐点一致」这条判据有可执行的内容前提。
6. 后续轮次可以考虑把 §2 项 2 的 A 臂结果(IMAGE vs VIDEO 的形态差)作为 §10.4 那道门的**起点数据**:它已经说明跟踪会在只有 25 帧检出姿态的序列上补出 28 帧。

---

## 8. Assessment

**Ready to merge?** **With fixes** —— 就 M1.5 的**代码范围**而言是 **Yes**(六项实跑全过,适配层逐点 100%,无 Critical、无错值、无数据丢失);但**合并这一批之前要先补上 4 件记账/一行级的事**:`logging_config.py` 入 git(I4)、`docs/下一步.md` 与账本落位(I2/M4)、提交计划的修正版(M3)、对 `close()` 阻塞(I1)给出「这一轮修」或「下轮首项」的明确决定。

**Reasoning:** 两个服务从「一个每帧静默 500、一个进程起不来」变成实跑 200、落盘带 `session_id`、并且在我独立跑的金标比对里**同一个适配层逐点 100% 等于基线**(face 9560/9560、gesture 1119/1119),gesture 那条最危险的静默失效(摊平成元组→`is_valid=False`+50.0)在封装侧、消费侧、真服务三处都有证据。剩下的问题都不是「结论错了」而是「代价与记账」:唯一有真实用户影响的实测项是 `close()` 每次 5 秒把事件循环冻住(TTL 回收 10s、`/reset` 无 id 20s、期间 `/health` 19.73s),它源自 spec §6.3 自己的强制要求,所以是设计层面的遗留,不该拦这次迁移,但不该没有决定。
