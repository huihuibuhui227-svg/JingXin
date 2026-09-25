# SDD ledger — plan: docs/superpowers/plans/2026-09-25-m2-6-media-retention-server-side.md

- Executor: inline (executing-plans)。分支 `m2.6-media-retention`,基线 `e958150`(main 顶端)。
- Spec: `docs/superpowers/specs/2026-09-25-m2-6-raw-media-retention-design.md`(已读)。
- 仓: `~/jingxin`。解释器 `~/miniconda3/envs/jingxin/bin/python`。测试基线 254 passed。

---

## Setup

**Setup: Ruling: 用分支而不是 worktree。** —— 依据:本 skill 要求"不在 main 上直接实现",
而计划的验收步骤全部依赖绝对路径(`~/shared/jingxin_recordings`、`data/logs/`、模型在
`~/jingxin/models/`),worktree 会打断它们(M2.5 账本记过同一条)。分支同目录、路径不变,
且 main 保持干净。 — 代价(若错):收尾时多一步 merge 需要使用者点头。

**Setup: Ruling: `scripts/task-start` / `task-done` 在本安装里不存在(只有
`sdd-workspace` / `task-brief` / `review-package`)。** — 依据:M2.5 账本已记录这是
superpowers 6.4.1 的打包缺件。故 BASE 用 `git rev-parse` 取,收尾的测试跑与账本行手动记。
— 代价(若错):无(账本格式仍与 sdd 一致)。

---

## Pre-flight scan

共享接口逐行(生产 → 消费):

| 生产 → 消费 | 生产面 | 消费面 | 结论 |
|---|---|---|---|
| T1 → T2 | `_prepare` / `_session_lock` / `_bump_seq` / `_current_seq` / `recording_dir` / `enabled` / `MEDIA_SUBDIR` / `validate_modality` | T2 的 `retain_frame` 实现 | 一致 |
| T1 → T3 | 同上 | T3 的 `retain_audio` 实现 | 一致 |
| T2 → T3 | `_append_jsonl` / `_record(sid, kind, modality, seq, rel, data, declared_ts, source)` | T3 以 8 个位置参数调用 | 一致(参数个数与次序逐一对过) |
| T1/T2 → T5/T6 | `retain_frame(session_id, modality, data, declared_ts=, source=)` | face `:287` 后 / gesture `:297` 后各一行 | 一致 |
| T1/T3 → T7 | `retain_audio(session_id, kind, data, source=)` | `/asr` ×2、`/interview/answer_audio`、`/research/answer_audio` | 一致 |
| T2/T3 → T8 | 账本行字段 `kind` / `modality` / `seq` / `file` / `declared_ts`;帧名 `media/<modality>/%06d.jpg` | `load_frames` 读 `rec["kind"]`、`rec["modality"]`、`rec["file"]`、`rec["seq"]`、`rec.get("declared_ts")` | 一致 |

### 冲突 1(Task 1)—— 已裁定,计划已就地修好

Task 1 的骨架把 `retain_frame` / `retain_audio` 写成**整函数** `raise NotImplementedError`,
但 Task 1 自己的两条预检测试(`test_preflight_is_loud_…`、`test_preflight_failure_is_only_loud_once`)
断言的是 **`RuntimeError`** —— `NotImplementedError` **不是**它的子类 ⟹ 那两条在 Task 1
必然红,Task 1 的 "Expected: 6 passed" 不可达。

**Task 1: Ruling: 骨架改为"先走完守卫 + 预检,只在真正落盘那一步抛 NotImplementedError"。**
— 依据:Task 1 的验收对象是**预检这一层**,它不该被"落盘还没写"挡住;而 NotImplementedError
留在落盘之后,仍能保证 Task 1 结束时没人误以为留存已经能用。
— 代价(若错):无(修的是计划缺陷,不是实现偏差)。

### 冲突 2(Task 1/2)—— 已裁定,计划已就地修好

`validate_modality` 原文安排在 **Task 2** 新增,但 Task 1 的骨架与预检路径**先要用它**。

**Task 1: Ruling: `validate_modality` 提前到 Task 1 与 `validate_session_id` 放一起。**
— 依据:它是**守卫**,和 id 守卫是同一层的东西;放在用它的落盘任务里会让 T1 引用一个
尚不存在的名字。 — 代价(若错):无(纯位置调整)。

Pre-flight: 其余共享接口无冲突。

Task 1: complete (commits e958150..9949432, tests: `pytest -q` → 260 passed;本文件 6 passed)

**Task 2: Ruling: 计划漏了建 `media/<模态>/` 子目录,补一个 `media_dir()`。** —
依据:计划里写的是 `(recording_dir(sid) / rel).write_bytes(data)`,而 `write_bytes` 不替调用方
建父目录 —— 实测第一帧就 `FileNotFoundError`(pathlib.py:931)。修法是新增
`media_dir(session_id, *parts)`,它建目录并返回路径;Task 3 的音频同样要用。
— 代价(若错):无(补的是计划本来就该有的东西)。

**Task 2: Ruling: 预检探到 `media/` 而不是只探会话目录。** — 依据:原 `_prepare` 只探
`recording_dir(sid)`;"会话目录建得出、里面子目录建不出"这种情形会漏过去,变成首帧
不响、后续每帧都 degraded。改成探真正要写的那个目录。 — 代价(若错):无。

Task 2: complete (commits 9949432..a65683f, tests: `pytest -q` → 266 passed;本文件 12 passed)

Task 3: complete (commits a65683f..e84b9a1, tests: `pytest -q` → 270 passed;本文件 16 passed)

**Task 4: Ruling: 计划里那条中途失败测试用 `monkeypatch.undo()` 是危险的,改用
`pytest.MonkeyPatch.context()`。** — 依据:`undo()` 会把 autouse fixture 设的
`JINGXIN_RECORDINGS_DIR` **一起撤掉**,于是该测试最后那句 `retain_frame(...)` 会走
`root()` 的默认值,真的写进使用者的 `~/shared/jingxin_recordings/`。
`MonkeyPatch.context()` 只撤它自己那一层。**已实测核过**:跑完后
`~/shared/jingxin_recordings/` 下没有 `s1` / `never-seen`。
— 代价(若错):无(否则就是测试污染使用者的真实数据目录)。

Task 4: complete (commits e84b9a1..28ea16f, tests: `pytest -q` → 273 passed;本文件 19 passed)
(更正:上面这行先被我误写成 274,实跑输出是 273 —— 账本与实跑不一致会误导换执行者,故改。)

**Task 5: Ruling: 计划里的 `python_multipart` 替身必须删掉 —— 真包装着。** —
依据:计划从 `test_analyze_session_fallback.py` 抄了那段 shim,而那条测试的注释
("本环境连 python-multipart 都没装")**已经过期**:实测 `find_spec('python_multipart')`
指到 `site-packages/python_multipart/__init__.py`。装残缺替身(startlette 要从中
import `__all__`)的后果是 **本文件单独跑就崩**,只在全量套件里靠"别的测试先 import 过
真包"侥幸通过 —— 顺序依赖的假绿。 — 代价(若错):无。

**Task 5: Ruling: 替身 `cv2.imread` 要返回带 `.shape` 的对象。** — 依据:计划的替身返回
`object()`,而端点 `:277` 会读 `image.shape` 写日志 ⟹ 测试红在 **AttributeError**,
不是我要的"盘上没有文件"。按 TDD"红在错处就修测试",加一个 `_FakeImage`。
— 代价(若错):无。

**Task 5: 观察: `test_retention_off_leaves_no_files_and_still_analyzes` 在 RED 阶段就是绿的。**
这不是"测试没约束力":它压的是**关掉那条路**,接线后若有人把 `enabled()` 判断去掉,
盘上就会出现文件 → 它会红。已实测接线后仍绿。

Task 5: complete (commits 28ea16f..e64be55, tests: `pytest -q` → 276 passed;本文件 3 passed)

**Task 6: Ruling: 情绪替身要返回**五个**键,不是计划里写的一个。** — 依据:端点 `:427-433`
逐个读 `overall_score` / `emotion_state` / `emoji` / `feedback` / `used_features`;
`analyzers` 也要给全 6 个键(left_hand/right_hand/shoulder/left_arm/right_arm/emotion)。
替身不够时测试红在 **KeyError** 而不是我要的"盘上没有文件" —— 那是测试自己的毛病,不是产品的。
已按 TDD"红在错处就修测试"补齐,并核过最终 RED 是干净的(端点跑到底、日志有"分析完成",只是没文件)。
— 代价(若错):无。

Task 6: complete (commits e64be55..8535fc5, tests: `pytest -q` → 278 passed;本文件 2 passed)

**Task 7: 附注:计划的 voice 测试只覆盖了面试端点,我补了一条科研端点的。** —
依据:计划自己的文件标题写的是"voice 三个端点",而两条 `answer_audio` 是**对称**的;
M2.1 第 19 条的教训正是"先面试再科研时被写成上一场的 id" —— 只测一条会漏掉一半素材。
— 代价(若错):无(多一条测试)。

Task 7: complete (commits 8535fc5..98819dc, tests: `pytest -q` → 281 passed;本文件 3 passed)

Task 8: 代码与单测 complete (commits 98819dc..aaba302, tests: `pytest -q` → 285 passed;本文件 4 passed)
(Step 7 的端到端验收要一场真会话,待使用者执行 —— 未完成,见下。)

### Task 8 Step 7 的**可自验部分已由执行者实跑**(2026-09-25 20:35)

一场真服务冒烟(不是替身):铸号 `20260925_203509_6603` → face 收 2 张真脸图 →
gesture 收同 2 张 → voice 收真 16k 单声道 WAV(三个都回 200)。实测结果:

| 判据(spec §7) | 实测 |
|---|---|
| 素材落盘 | `media/face/000001-2.jpg`、`media/gesture/000001-2.jpg`、`media/audio/0001.wav` 全部落 `~/shared` |
| sha256 对账(§7.2) | **5/5 一致**,`bytes` 与文件大小也一致 |
| `declared_ts` 是会话相对时钟 | face 0 / 273;gesture 0 / 239;音频 null —— 不是墙钟 |
| **重抽等价性(§7.1)** | `replay_retained.py` 重放 2 帧 → 与当场活跑的 `face_au_log_20260925_203509_6603.csv` 逐格比:**39 列 × 2 行,差 > 1e-9 的格子 0 / 78,最大绝对差 0.000e+00** |
| 合并门(§7.8) | **0 / 2835510** |

**仍然待使用者做的**:一场**真人**会话(≥5 段不同回答、手入镜、Windows 端浏览器),
用来验多帧节奏与真实采集路径 —— 上面那次用的是静止图,只有 2 帧。

---

## Final review(整分支,新上下文审查者,opus)

范围 `e958150..aaba302`,8 提交 / 10 文件。**结论:Not ready to merge** —— 但两条 Critical
都是"修法很便宜"的类型。审查者另在**使用者的真会话**上独立复算了 sha256(231 条行全对)、
数了 JPEG 完整头尾(374 个零截断),并做了跨进程复现实验。

### 修复波次(一轮,每条 RED→GREEN,套件 285 → 300 passed)

- **C1(账本跨进程写碎)** —— 已修。`retention.<writer>.jsonl` 每写入者一个文件 + 单次
  `os.write` 的 O_APPEND。钉子 `tests/test_media_retention_concurrency.py`:
  起**三个真子进程**各写 120 件,断言每个文件的每一行都能解析、行数一件不少。
  依据:实测那场真会话 308 行 / **77 行碎**;审查者的 9p 复现(两进程 297 行 / 76 碎,
  单进程 400 行 / 0 碎)把根因钉在"跨进程共写同一文件",不是并发帧。
  **不用 flock** —— 审查者指出 9p 没有锁协议,Linux 侧会退化成本地生效。

- **C2(序号是进程内状态 → 同名覆盖)** —— 已修。`_bump_seq` 从盘上最大编号推
  (本进程首次用到该 (会话,类别) 时扫一次目录,之后走内存高水位)。
  钉子三条(`test_seq_survives_a_process_restart` / `…_for_the_none_bucket_too` /
  `test_audio_seq_also_survives_restart`)。
  **反向复现已实跑**:把 `hi = 0 if seen else _highest_seq_on_disk(...)` 改回 `hi = 0`
  → 三条全红;恢复 → 全绿(并清了 `__pycache__`)。
  依据:实盘 `20260924_153012_9f3c` 账本 52 行只对 12 个文件、`NONE/` 54 行只对 12 个。

- **I1(预检探错目录 + 0 字节探针)** —— 已修。探到真正要写的 `media/<writer>/`,
  探针写 4 KiB 非空块,文件名带 pid + 线程号(`_probe_name`)。
  钉子:`test_preflight_probes_the_modality_dir_not_just_media`、
  `test_probe_writes_non_zero_bytes`、`test_probe_name_is_unique_per_process_and_thread`。

- **I2(降级只活在内存,账本从不写)** —— 已修。`_mark_degraded` 除内存外**往账本追加一行**
  `{"kind":"degraded","reason":…}`(尽力而为:磁盘坏时它自己也会失败,吞掉);
  `degraded_reasons` **合并**内存与账本两处(所以能活过重启)。
  钉子:`test_midway_failure_is_recorded_into_the_ledger`、`test_degraded_lines_survive_a_restart`。
  **顺带修掉计划里那条空检查**(详见下"计划自身缺陷")。

- **I3(重抽工具以账本为准)** —— 已修。`load_frames` 改为**以盘上文件建列表**,
  账本只用来取 `declared_ts`;取不到的帧进 `missing_ts` **报数**而不是静默跳过;
  碎行计数进 `torn_lines`;`main` 在有缺口时**退出码 2**。
  钉子:`test_frame_on_disk_without_a_ledger_line_is_REPORTED_not_silently_dropped`
  (含义相对旧版**反转**了 —— 旧版正是那个 bug)。

- **I4(raw↔converted 靠"此刻值"配对)** —— 已修。`retain_audio(..., seq=None)`;
  `/asr` 把 raw 那一行的 `seq` 传下去。**并补上审查指出的测试缺口**:
  `/asr` 是唯一同时产出 raw 与 converted 的端点,此前**零端点级测试**,
  现加 `test_asr_retains_raw_and_converted_under_the_same_seq`。

### 修复期**新发现并修掉**的一处(审查没提,我自己踩的)

一条新测试(**没设** `JINGXIN_RECORDINGS_DIR`)通过 `ledger_path` → `recording_dir` →
`root()` 的默认值,**往使用者真实的 `~/shared/jingxin_recordings/s1/` 写了文件**
(已删)。修法有两层:① 该测试改成直接往 `tmp_path` 写;② **在 `tests/conftest.py` 里
把默认落点兜住**(照它已有的 `LOGS_DIR` 同款做法)—— 单测各自 `setenv` 只保护
「记得设」的那几个,兜不住后来新加的。实测跑完 300 passed 后真目录干净。

### Final: Ruling: 审查者"Declined to judge"的每一条,采纳其处理。 — 依据:逐条复核后我同意。

- **前端腿与元数据层切出本计划**:采纳。plan 文件头与"完成之后"已明写是**记录在案的
  范围裁定**,不是漏做。
- **`preflight()` 折进首件 `retain_frame` / `retain_uploaded_video` 不存在**:采纳。
  行为等价,且 plan 的接口清单已如此。
- **`_PREFLIGHTED` 未持 `_STATE_GUARD`(而 `_LOUD_DONE`/`_DEGRADED` 持了)**:采纳
  "不判缺陷"。说不出可触发时序,但记在这里以正锁纪律。
- **`_append_jsonl` 不 fsync**:采纳。设计只宣称"进程被杀",不宣称断电。
- **`~/shared/m2_acceptance.sh` 会往旧 sid 反复发帧**:采纳,记 minor(见下)。
— 代价(若错):无。

### Final: Ruling: 更正本账本里一条**错误记录**。
账本开头 Setup 段写「`scripts/task-start` / `task-done` 在本安装里不存在」——**这是错的**。
审查者指出两者都装着,只是在 `superpowers/6.4.1/skills/executing-plans/scripts/` 下,
而不是 `subagent-driven-development/scripts/` 下(后者只有 `sdd-workspace` / `task-brief` /
`review-package`)。我照 M2.5 账本的一句旧话抄了下来,没有自己核。
— 代价(若错):无(本次 BASE 与测试数都手工记了,格式自洽);但**下一任执行者若信这句话
会重复手工活** —— 故在此更正,原话保留不改(§4.6:账本的历史记录不改)。

### Final: minor (deferred) —— 全部记进账本,不进修复波次

1. **账本字段名 `source` 与 spec §5.2 的 `source_endpoint` 不符**(`media_retention.py:214`)。
   审查者认为"账本是要长期读的数据契约,现在改比录完 3–5 场再改便宜"。
   我**不擅自改**:使用者的真实会话已经写进了用 `source` 的账本,改名会让新老账本两种口径;
   要改得连恢复脚本一起定。**请使用者裁定**(下一轮第一件事)。
2. **`NONE` 桶的进程内状态跨场次共用**:第一场把 `NONE` 记进 `_PREFLIGHTED` 后,
   后一场(另一个人)即使落点已坏,首件也不会再大声失败。只影响不带 id 的客户端。
3. **CV 侧只留"能解码的"帧,voice 侧留"所有上传"**(face `:274-288` 在 `imread` 成功之后;
   voice `:371`/`:539` 在格式校验之前)。spec §4 的顺序是"①落盘 ②分析" ——
   一份截断/传坏的帧在盘上不留痕迹,而"客户端到底发了什么"正是这类证据最值钱的地方。
4. **`_DEGRADED` 按失败**件数**增长**(坏掉的一场 5 fps × 10 min = 3000 条原因串),
   且没有任何会话收尾的清理入口。5–50 场规模下无妨。
5. **`~/shared/m2_acceptance.sh` 会往旧 sid 反复发帧** —— C2 的**暴露路径**
   (审查者实测那两处 52 行/12 文件、54 行/12 文件就是这么来的)。脚本不在本 diff 内。

Task 8 Step 7 的**使用者真会话**仍未完成(见下节)。

---

## Task 8 Step 7:使用者真会话的验收(2026-09-25 20:38–20:41,`20260925_203826_2449`)

**这是本里程碑第一次拿到多帧真会话** —— 也是它第一次暴露 Critical 1。

| 判据 | 实测 |
|---|---|
| 素材落盘 | face **187** 帧 / gesture **187** 帧 / audio **16** 件(8 raw webm + 8 converted wav),合计 39 MB |
| 素材完整性 | 374 个 JPEG **零截断、零空文件**;16 个音频文件头全对 |
| **重抽等价性(§7.1)** | 重放 **187/187** 帧 → 与当场活跑的 `face_au_log_…csv` 逐格比:**39 列 × 187 行 = 7293 格,差 > 1e-9 的 0 格,最大绝对差 0.000e+00** |
| 会话时长 | CSV 的 `timestamp` 0 → 181.614 s |
| 合并门(§7.8) | 0 / 2835510 |

### 账本恢复(那次损坏的善后,脚本 `recover_session.py` 留在本工作区)

素材是全的,坏的只有元数据。恢复原则:**只恢复能证明的,不编。**

- **face 187 行,永久丢失 0**。105 行取自碎账本里没碎的那些;**82 行由活跑 CSV 的 `timestamp`
  (会话相对秒 × 1000)补齐**。这条补齐**经过交叉验证**:两边都有的 105 帧
  **105 一致 / 0 不一致** —— 且它又被最终的重抽等价性(7293 格全等)二次证实。
- **gesture 187 行,永久丢失 77**。gesture 的 CSV `timestamp` 是**绝对墙钟**(已知问题 F9),
  与 `declared_ts` 的差是 16–36 ms **且不是常数偏移** ⟹ **不可重建**。
  那 77 帧的 `declared_ts` 写 **null**(不是近似值 —— 拿近似值冒充就等于伪造"逐格相等"),
  重抽工具会把它们报成缺口。**这是 Critical 1 的真实代价,如实记。**
- **audio 16 行,全部从盘上精确算出**(它的 `declared_ts` 本来就是 null)。
- 原文件改名 `retention.jsonl.corrupt` 留证,不删。

### 恢复期的一个自踩 bug(记下来免得再犯)

恢复脚本第一版用 `name.split("_")[0]` 取序号 —— 而 `0001.webm` 里**没有下划线**,
拿回的是整个 `"0001.webm"`,`isdigit()` 为假 ⟹ **8 个 raw 音频全被静默跳过**,
只恢复出 8 个 converted。改用 `re.match(r"(\d+)", name)`。
(`media_retention._highest_seq_on_disk` 里那处写法带了 `.split(".")[0]`,**没有这个 bug**,
已核。)

### Final: Ruling: 77 个 gesture 帧的时间戳**不弥补**。 — 依据:两条可走的路都不诚实 ——
① 用墙钟近似(误差 16–36 ms,非恒定)会改变 mediapipe 的 VIDEO 模式跟踪 ⟹ 重抽值不等于
当场值,却仍宣称"逐格相等";② 编一个"看起来合理"的值同理。spec §7.1 要的就是**当时那个值**。
所以标 `null` + 让工具报缺口。 — 代价(若错):这 77 帧的 gesture 素材**在时间基上不可重抽**
(像素还在,时间戳没了)。要拿回它们只能重新采集。

### 真会话暴露的**下游**缺口(不属 M2.6,但必须记下来,否则下次录制还是白录)

那场会话报告 **0 / 20**(T7 时是 1/20)。逐层查清:

1. **语音族整族空** —— 语音服务日志显示前端实际调的是
   `POST /interview/answer` ×8(文字路)、`POST /asr` ×8(转文字)、
   `POST /interview/answer_audio` **×1**。
   而 `log_prosody` 只在 `answer_audio` 里(`voice_interaction/api/app.py:397`)——
   **文字回答那条路一个语音特征行都不写**,`interview_emotion_log_<sid>.csv` 于是只有表头,
   报告侧读成"文件在但读不出来"。
   根因在前端 `~/JingXin-frontend/src/hooks/useAssessment.ts:71-77`:
   `if (audioFile) submitAudioAnswer(...)` —— **只有带 audioFile 才走音频路**;
   而 `AssessmentPage.tsx:197` 的"语音转文字"拿到文本后**把音频 blob 丢掉了**,
   于是 8 段里只有 1 次带了音频。
   ⟹ **不改这里,下一次录制照样没有语音特征**(而语音是 L0 里唯一能出率类指标的族)。
2. **面部/手势族全部"本轮停用"** —— ① 的封停名单,等 M3。范围之内,预期行为。
3. **手势 187 帧里手一次没检出**(`left/right_hand_is_valid` 全 False)——
   取景/姿态问题,不是探测器坏(同一天用 `~/shared/mp_frames` 的静止图能检出 2 只手)。

**另**:`/interview/answer`(文字路)其实**拿得到文本**,而连接词密度只需要文本
(`connective_density(text)`)—— 那条路顺手记一行是**很小的改动**,能把 T7 那个 1/20 找回来。
但它是独立改动,不在 M2.6 范围内,故只记不做。
