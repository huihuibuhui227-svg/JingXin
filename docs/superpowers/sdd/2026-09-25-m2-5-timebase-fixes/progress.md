# SDD ledger — plan: docs/superpowers/plans/2026-09-25-m2-5-timebase-fixes.md

- Executor: inline (executing-plans)。Base: `4fcdedf`。
- Spec: `docs/superpowers/specs/2026-09-25-m2-5-timebase-fixes-design.md`(已读)。
- 仓: `~/jingxin`,分支 `main`。解释器 `~/miniconda3/envs/jingxin/bin/python`。

---

## Setup

**Setup: Ruling: 在 `main` 上直接实现,不建 worktree。** — 依据:计划自身 Task 9 Step 6 就是
`git push origin main`;且全部验收步骤依赖绝对路径(`~/jingxin/data/logs/`、`~/shared/start_all.sh`
从 `~/jingxin` 起服务、模型在 `~/jingxin/models/`),worktree 会打断它们。使用者批准的这份计划
已含该 push 步骤,即明示同意在 `main` 上做。 — 代价(若错):需要把 8 个提交 rebase 到分支上。

**Setup: Ruling: 本会话无 TodoWrite 工具,进度只记在本账本。** — 依据:可用工具里没有 todo 工具;
账本本来就是"契约记录",harness todo 只是实时视图。 — 代价(若错):无(账本更完整)。

---

## Pre-flight scan

共享接口逐行(生产 → 消费):

| 生产 → 消费 | 生产面 | 消费面 | 结论 |
|---|---|---|---|
| T1 → T5 | `SessionClock()`、`stamp_ms() -> int`、`reset()` | `_clock_for()` 返回它;`_reset_session()` 调 `.reset()` | 一致 |
| T1 → T6 | 同上 | 同上 | 一致 |
| T2 → T4 | `detect(image_rgb, timestamp_ms)` | `self.detector.detect(image_rgb, timestamp_ms)` | 一致 |
| T2 → T6 | 同上 | `dets['hands'].detect(image_rgb, timestamp_ms)` | 一致 |
| T3 → T4 | `MicroExpressionDetector()`、`detect(values, timestamp_ms)` | `self.micro_detector.detect(current_au, timestamp_ms)` | 一致 |
| T4 → T5 | `VideoPipeline(session_id=…)`、`process_frame(img, ts)`、`_measured_fps()` | T5 Step 3 / Step 3b | 一致 |
| T4 → T7 | 同上 | `FaceExtractor` 内部 | 一致 |
| T4 → T8 | `process_frame(img, ts)` | 两个 example | 一致 |

### 冲突 1(Task 7)—— 已裁定

Task 7 写「各自把 `timestamp_ms` 转交给内部的 `VideoPipeline` / gesture 分析器」。实测两件事:

1. `experiments/extract_features.py:110-117` 的 `GestureExtractor` 用
   `mp.solutions.hands.Hands` / `mp.solutions.pose.Pose` **直连**,不经过 Task 2 的封装;
2. 实跑 `mediapipe 1.0.0` → `hasattr(mp,'solutions') = False`。

⟹ **该类的 `__init__` 必然 `AttributeError`,离线手势提取器在本环境根本起不来。**
M1.5 只迁了活服务(:8000/:8002),漏了这个离线文件。

**Task 7: Ruling: M2.5 只做时间基 —— face 侧 + `offline_timestamp_ms` 纯函数 + 调用点签名一致;
不顺手把离线手势提取器迁到 `tasks` API。** — 依据:那是 M1.5 的漏项,不是 M2.5 的范围;
且 Task 7 的验收(纯函数单测)不依赖它。照搬一个能跑的实现进来会引入 `tasks` 依赖与新的
`models/mediapipe/*.task` 路径耦合,超出本里程碑。
已把「离线 `GestureExtractor` 仍用已删除的 `mp.solutions`」记进 deferred 清单。
— 代价(若错):离线手势线继续不可跑 —— 与 M2.5 之前**没有区别**,不是本计划引入的退步。

Pre-flight: 其余共享接口无冲突。

---

## 进度

(每任务一行,由 `task-done` 追加)


**Task 1: Ruling: 未重读 task-1-brief.md,直接用本会话内逐字写下的计划正文。** — 依据:该计划由我在
本会话逐字撰写、文件自那以后未被改动(同一会话、同一文件、无中间提交),上下文里的不是摘要而是原文;
重读 143 行要付的上下文成本,在 9 个任务的预算下不划算。 — 代价(若错):漏掉计划里某个精确值。
后续任务同样处理:计划正文始终在本会话上下文中。

**注意:`scripts/task-start` 与 `scripts/task-done` 在本安装里不存在**(scripts/ 下只有
`sdd-workspace`、`task-brief`、`review-package`)。BASE 由 `git rev-parse` 取;收尾的测试跑与账本行
由本账本手动记。 — 依据:这是 superpowers 6.4.1 打包时的缺件,不是我的疏忽(两条 "没有那个文件或目录")。
— 代价(若错):无(账本格式仍与 sdd 一致,换执行者也能续)。

Task 1: complete (commits 4fcdedf..4c8831c, tests: `pytest -q` → 227 passed;单文件 3 passed)

**Task 2: Ruling: 计划漏了 PoseDetector.detect 的覆盖与两个子类的 fps 转交,一并修掉。** —
依据:计划的「Files」只点名 `_Base`,但实测 `PoseDetector.detect(self, image_rgb)` 覆盖了父类并调
`super().detect(image_rgb)`;不一起改,父类的回退守卫就被绕过去了(而且父类拿不到真时间戳)。
两个子类 `__init__` 里的 `fps=fps` 同理(参数已删,不改就 TypeError)。这是计划缺陷,不是实现偏差。
— 代价(若错):无(修的是计划本来就该覆盖的面);漏掉它才是错。

**Task 2: 观察:test_assert_coverage.py::test_no_assert_is_dead 会因内层套件红而红**(它自己断言
inner_exit_code == 0)。所以内层一红它必然跟着红 —— 排查时先看内层,别把这个当独立故障。

Task 2: complete (commits 4c8831c..9fd425e, tests: `pytest -q` → 229 passed;本文件 16 passed)

**Task 3: Ruling: 计划没点出 Task 3 与 Task 4 的耦合,我把 Task 4 的一行提前到 Task 3 做。** —
依据:Task 3 删掉 MicroExpressionDetector 的 fps 参数后,`video_pipeline.py:34` 的
`MicroExpressionDetector(fps=fps)` 立刻 TypeError,而 test_face_detector_wiring.py 的三条会红
(实测:4 failed)。executing-plans 的完成契约要求任务收尾时套件绿,所以最小修法是把那一行
一并改掉(它本来就在 Task 4 的改动面里)。 — 代价(若错):Task 4 少改一行,无风险。

Task 3: complete (commits 9fd425e..93e0f43, tests: `pytest -q` → 231 passed;本文件 2 passed)

**Task 4: Ruling: 时间窗修剪用「给历史帧挂 timestamp_ms 属性」而不是把 deque 存成 (ts, obj) 元组。** —
依据:改成元组会波及 `get_summary` 的遍历与 `process_frame` 里那段按 `__annotations__` 取字段的时序统计
(两处都得跟着改,风险面变大)。挂属性则只动修剪那一处,且因为 `timestamp_ms` 不在 `AUFeatures.__annotations__`
里,它**不会**被当成特征字段卷进时序统计 —— 正好是我们要的。实测 `AUFeatures` 是普通类、
无 `__slots__`,挂属性合法(已核)。 — 代价(若错):历史帧多带一个非特征属性,写出 CSV 时若有人
`__dict__` 全量序列化会多一列(已核 `to_dict()` 走显式字段,不受影响)。

**Task 4: 附注:计划未提但随之而来的两处** —— ①`import time` 在删掉两处 `time.time()` 后成为死导入,已删;
②`total_blinks` 这个局部量属于原有代码、plan 的替换块没写它,我保留了(后续 return dict 在用)。

Task 4: complete (commits 93e0f43..7714fea, tests: `pytest -q` → 237 passed;两个文件 9 passed)

**Task 5: Ruling: `/reset` 的语义是「删掉整个会话」,所以时钟是 pop 而非 reset;计划与测试都照此改写。** —
依据:实测 `face_expression/api/app.py:355-370` 的端点 docstring 明写"本端点的语义是删掉整个会话
(不是清空状态但保留会话)",实现里是 `session_pipelines.pop()` + `close_detached()`,**从不调**
`pipeline.reset()`。计划 Task 5 Step 3 写的"把 /reset 里原来直接调 pipeline.reset() 的地方改为
call_session" 与事实不符(没有那个调用点)。照事实写:第二段会话走 get_or_create 重建全新时钟,
第一帧自然是 0。 — 代价(若错):若将来有人把 /reset 改成"保留会话"语义,这条要跟着改。

**Task 5: Ruling: 测试必须用 `importlib.import_module("face_expression.api.app")`。** —
依据:计划的 import 写法拿到的是 FastAPI 实例(包 `__init__.py` 做了 `from .app import app`),
实测三条测试全 AttributeError。仓里 `tests/test_analyze_session_fallback.py:69` 已把这条坑写进注释。
— 代价(若错):无。**Task 6 的 gesture 测试照此办理**(`gesture_analysis/api/__init__.py` 同样做了 re-export)。

**Task 5: Ruling: `VideoPipeline._measured_fps` 改名公开 `measured_fps`。** — 依据:face 服务要调它,
跨模块调下划线私有名是不该过的味道;spec §5.5 用的名字本来就是 `measured_fps`。同步改了 Task 4
留下的 2 处调用点与测试替身。 — 代价(若错):无(纯改名,套件已绿)。

**Task 5: 附注:测试替身 `_FakeFacePipeline` 补了 `measured_fps()`。** 会话收尾会调它,
替身少了就 AttributeError —— 该替身的 docstring 本来就写着"替身必须跟着真接口长"。

Task 5: complete (commits 7714fea..81fba90, tests: `pytest -q` → 240 passed;两个文件 19 passed)

**Task 6: Ruling: gesture 的两个 TTL 清扫点只有第一个该 pop 时钟。** — 依据:实测
`gesture_analysis/api/app.py` 有两处清扫 —— 第一处(删 `session_analyzers[sid]`)是**真删会话**,
第二处(`get_or_create_detectors` 里那个)只 `detectors.pop()`,**不删会话**,会话仍活着。
在第二处 pop 时钟会把活会话的时钟弄丢(下一帧时间戳回到 0 → 对同一会话是回退 → 探测器抛)。
— 代价(若错):第二处若将来改成也删会话,得同步补 pop。

**Task 6: Ruling: 给 SessionClock 加 `n_stamps` + `measured_fps()`(Task 1 的文件在 Task 6 里改了)。** —
依据:spec §5.5 要求逐会话记实测 fps,而 gesture **没有 VideoPipeline** 可问(face 那边有)。
每次 `stamp_ms()` 就是收到一帧,所以帧计数天然在时钟里,不必再造一个字典。
— 代价(若错):时钟多担了一个"记账"职责;若将来 face/gesture 口径不一致,要在这里统一。

**Task 6: 观察(踩过):新端点测试起初写成 [0,0,0,0] 的假失败 —— 根因不是代码。** —
`_freeze_clock` 把 app 的 `time.time()` 换成**每次调用 +1000 秒**的假时钟(>SESSION_TIMEOUT 300),
于是**每个请求都成新会话**,时钟随之重置、时间戳回到 0。那不是 bug,正是"时钟按会话"的证明。
修法:该测试不调 `_freeze_clock`,验的是**同一会话内**时钟会走。— 这条值得留给后来者:
**本文件里凡是要"同一会话跨两帧"的断言,都不能配那个假时钟。**

Task 6: complete (commits 81fba90..9497254, tests: `pytest -q` → 243 passed;两个文件 21 passed)

Task 7: complete (commits 9497254..1b79ada, tests: `pytest -q` → 246 passed;本文件 3 passed)
(本任务的裁定见上文 **Pre-flight 冲突 1**:不顺手迁离线 GestureExtractor。)

**Deferred(留给账本/M3):离线 `experiments/extract_features.py` 的 `GestureExtractor`
仍直连 `mp.solutions.hands/pode`,而 mediapipe 1.0.0 已删除该命名空间(实测
`hasattr(mp,'solutions') == False`)—— 离线手势线在本环境不可跑,与 M2.5 之前无差别。**

**Task 8: 观察(记录实情,不是裁定):两个 example 是"三重死"** —— 无 importer、
引用的 `FaceAUAnalyzer` 全仓无定义、且用 `mp.solutions`(已删)。仍按计划改了签名,
目的是"接口变了不留陈旧调用点";注释里明确写了它们本来就跑不起来,避免后来者误以为可用。

Task 8: complete (commits 1b79ada..65c2c23, tests: `pytest -q` → 246 passed;本任务不改测试)

**Task 9 Step 1 反向复现:四条全部红在对的地方**(脚本 `reverse-repro.sh`,输出已实跑)——
① SessionClock 抬升 ② 探测器回退守卫 ③ eye_closed 真实间隔 ④ duration 由时间戳导出。
恢复后 `pytest -q` → 246 passed,工作树无残留。

**Task 9 Step 2 合并门:0 / 2835510** ✅(按 spec §7.4:**这只是"确认没有误伤"**,不是本设计的证据
—— 它复现的是已有 CSV,不重抽,采集代码怎么改都绿)。

**Task 9 Step 3 端到端(真服务 + 真脸图 5 帧,`e2e-precise.py`):**
  * 时间戳跨度 vs 墙钟跨度:全帧 **5.9%**;排掉预热帧(第 2→5 帧)**3.7%** —— 都 < 10% ✅
  * `is_blink` = **1/5 非零**(不再是恒 0)→ spec §3.4 的修复端到端成立 ✅
  * **抓到真缺陷**:`no_face` 行不带 timestamp → logger 兜底编了个墙钟(1.790318e+09)
    混进整列 → 已修 + 加钉子(撤掉修复 → 红,已实跑)→ 修后复跑通过。

**Task 9: Ruling: 端到端用「真服务 + `~/shared/mp_frames/frames/` 的真脸图」,而不是真前端会话。** —
依据:真前端会话要人对着摄像头说 5 段话,我驱动不了;而这批脸图是 M1.5 基线那批真图,
足以验时间基(多帧 → 时间戳跨度 vs 墙钟)。— 代价(若错):"前端 1 帧/秒"那条实际到达率没被端到端覆盖,
只在单测层面覆盖;`measured_fps` 的实时值也没被这一轮实测。

**Task 9: 观察:第一次 E2E 的 41.5% 是测量噪声,不是缺陷。** — bash+curl 循环把 5 次
curl **进程启动**(每次 ~25 ms)算进了墙钟;改成同进程发帧后降到 22.9%;再排掉第 1 帧
(服务端在它身上加载模型/首次推理,是个常数项)后为 3.7%。**教训:验时间基时,墙钟参考要取
"服务端事件之间"的量,而不是"客户端动作之间"的量** —— 否则量的是自己的进程开销。

Task 9: complete (commits 65c2c23..ead4b80, tests: `pytest -q` → 247 passed;本任务新增 1 条钉子)

---

## Final review(整分支,新上下文审查者,opus)

范围 `4fcdedf..ead4b80`,9 提交 / 19 文件。结论:**ready to merge WITH FIXES**。
审查判 F1/F2 为代码缺陷,F3/F4 为记录/证据缺陷,其余 Minor。

**修复波次(一轮,每条 RED→GREEN,套件 250 passed):**
- F1(Important):`offline_timestamp_ms` 加步长守卫(< 1 ms 直接 raise)。
  钉子 `test_sub_millisecond_step_is_rejected_loudly_not_collapsed` RED→GREEN。
- F2(Important):no_face 分支推进 `history_last_ms`(修前实测 `eye_closed_sec = 10.0`)。
  钉子 `test_no_face_gap_is_not_credited_to_eye_closure` RED→GREEN(修前红在 `abs(10.0-1.0)`)。
- F10(Minor 但涉**不实陈述**,故修):两处 `FaceAUAnalyzer(fps=)` 已删;
  run_video_analyzer 改用会话时钟并纠正错误注释;账本与提交信息里的错误说法已更正。
- F4(证据缺陷):补钉子 `test_is_blink_reaches_the_serialized_row`(撤掉三行 → 红,已实跑);
  **账本上一条"is_blink 1/5 非零"作废** —— `NaN != 0` 是 True,那是 pandas 假象;
  真口径下是 **0/5**(阈值未触发)。端到端那一半如实降级为「未验」。

**Final: Ruling: 端到端 is_blink 那条不修代码,只降级措辞。** — 依据:代码已被单元钉子
证明写回正确(撤掉即红),端到端没显示非零是**素材里闭眼阈值没触发**,不是缺陷。
— 代价(若错):实时路径的 is_blink 端到端仍未被真实会话验证过。

**Final: Ruling: 1 fps 下微表情族**永久为 null**,采纳为已知代价。** — 审查者指出它比
spec §8.1 的"时序族退化"措辞**更强**:是"有时产出"变成"永远不产出",与 §3.4 那个被当 bug
治的形态同形。用户已在 D4 上同意按时间窗并知悉 §8.1。裁定:**保留**,但 spec/账本的口径
从"退化"更正为"1 fps 下永久为 null",并**明确告知用户**这条比原先写的更重。
— 代价(若错):实时路径的 micro_exp_* 列在提高采集率之前一直为空。

**Final: Ruling: 其余"declined to judge"项照审查者意见办。** — `au_history` 由 90 帧改 3 秒窗
(spec §5.2 既定)、`recent_blinks` 的 `duration_min >= 0.1` 死分支(改前改后相同,属 M3)、
examples 的 `mp.solutions` 与离线 `GestureExtractor`(pre-flight 既定裁定)、
`evidence_thresholds.json`/前端/`code_data_supplement`(spec §1.2 界外)、
§3.7 的位移与跟进项 15/16(设计已延后,未测)。

**Final: minor (deferred)** —— 全部记进账本,不进修复波次:
F5 `?fps=` 警告永不触发(前端恒定发 30)且无 once 标志,注释与实现不符;
F6 `get_summary` 里 `fps`(全提交帧)与 `frame_count/duration_sec`(仅有脸帧)口径不一致;
F7 `duration_sec` 实为 3 秒历史窗而非会话时长(改前同样被 90 帧上限压住,短会话变化可达 10×);
F8 `measured_fps` 是 N/(N−1) 倍真值(公式按 spec,边界内;另:实时演示页 `% 10` 实际是 0.1 fps);
F9 gesture 的 CSV `timestamp` 仍是绝对墙钟(与 face 的会话相对秒不同基);
F11 `VideoPipeline.reset()` 无生产调用方且只重置探测器(`first_ts`/`au_history` 跨段残留),
`_frame_index` 已成只写死状态;
F12 单帧测试用 `_stub_au` 手工造历史,钉的是替身路径;face 侧缺 gesture 那样的"时间戳真的递到了"断言。

Task 9(剩余):complete (commits ead4b80..a4ebe48, tests: `pytest -q` → 250 passed)

---

## 收尾:deferred minors 清理(使用者 2026-09-25 指示"都做了")

**Final: minor F7 + F6 已修**(按效果重判为应修,非修饰):`duration_sec` 改取会话跨度
(first_ts/last_ts),连同**早退分支**里写死的 0;F6 不改数值、改把三个速率字段的口径写进
docstring。钉子两条 + 修正了钉错定义的旧测试。套件 252 passed。

**Final: minor F11 + F5 已修**。F11:`reset()` 全量归零(修前实测重置后 measured_fps 仍 1.5),
并把"当前无生产调用方"写进 docstring。F5:抽 `_warn_once_about_fps()`,申报 30 不记、
真不一致每会话一次(修前它是永不触发的死代码,而注释在说"记一次就够")。套件 254 passed。

**Final: Ruling: F8 不改。** — `measured_fps` 是 N/(N−1) 倍真值(5 帧会话 → 1.25),
但公式就是 spec §5.5 定的,且验收窗口 [0.5, 2.0] 本来就吸收它;改成"正确"的公式会偏离
已批准的 spec。 — 代价(若错):短会话的实测 fps 偏大,但不影响任何判定。

**Final: Ruling: F9 不改代码,记账。** — gesture 的 CSV `timestamp` 列仍是绝对墙钟,
与 face 改后的"会话相对秒"不同基。但 gesture 那条是**纯实时**路径,实时里墙钟**就是**真实
时间(D2 成立),且眼下没有任何消费者把两列混用(审查已逐个核过 data_loader / feature_engine /
duration_audit)。 — 代价(若错):将来若有人跨模态比较 `timestamp`,两列不同基会咬人;
M3 重建 L0 列时应顺手统一。

**Final: Ruling: F12 部分接受。** — face 侧"时间戳真的递到了"其实已由
`test_pipeline_forwards_the_timestamp_to_the_detector`(断言 `d.seen == [0,1000,2000]`)压住;
未覆盖的是"打戳值进到 CSV 行"这一层,以及 `_FakeFacePipeline` 只钉了 arity。
留作 M3 的事 —— M3 本来就要重建 L0 列与它们的验收。 — 代价(若错):端到端那一层的
"值真的落盘"仍只由 Task 9 的 E2E 一次性覆盖,不是常驻钉子。
