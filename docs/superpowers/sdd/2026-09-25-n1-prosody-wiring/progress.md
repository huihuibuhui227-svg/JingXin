# N1 语调接线 —— 账本

**日期**:2026-09-25 深夜
**依据**:`docs/下一步.md` §0.1「第 1 件:语调接线」
**状态**:已实现 + 已验(单测 / 反向复现 / 真会话现场端到端 / 报告层句子对照)。
**已提交并推送**(代码 `ed834fd`,文档与账本 `71c5fee`;三条挂账的裁定见 §10)。

---

## 1. 问题(接线前)

`/interview/answer_audio` 与 `/research/answer_audio` 都调 `log_prosody({}, …)`。
`log_prosody` 对缺的键一律 `.get(..., 0)`,于是 11 列全 0,而 `is_valid` 取默认值 `True`
—— **一行零值被标成「有效」**。`ProsodyFeatureExtractor` 在仓库里躺着,
只被两个 `examples/` 引用,活路径一次都没算过。

## 2. 改了什么

只动一个文件:`voice_interaction/api/app.py`(+64 行)

| 改动 | 说明 |
|---|---|
| `import numpy as np` + `ProsodyFeatureExtractor` | 启动开销 **0**:实测两者本就在 `sys.modules` 里(`assessment_pipeline → voice_pipeline → prosody_extractor`),app.py 只是从没**用**过 |
| 模块级 `prosody_extractor` | 无状态,可跨请求共享 |
| **`_EXTRACTOR_TO_LOG_COLUMNS`** | 提取器产出名 → 日志列名。**见 §4,这是本次最容易静默出错的地方** |
| `prosody_features_from_pcm(pcm) -> dict` | 纯函数:PCM → 特征字典(含改名)。已知缺陷在 docstring 里照实登记 |
| `_prosody_async(pcm)` | `asyncio.to_thread` 缝,与 `_transcribe_async` 同一理由(见 §5) |
| 两个端点 | `log_prosody({} …)` → `log_prosody(await _prosody_async(audio_data) …)`。放在**文本闸之后**,识别不出文字的请求已经 400,不必白算一次 pyin |

**没有改**的东西(以及为什么):

- `is_valid` —— **见 §6,这是本轮唯一一个"看着该改、我判定不该改"的点。**
- 提取器内部(能量门限 / `voiced_prob` / 真 VAD)—— §0.1 明说「接线 ≠ 定义」,归 M3。
- `speech_ratio` 这一列 —— spec §4.3 要删,但删列要动 CSV 头契约,归 M3。本轮**照实登记**(§6)。
- 报告层任何东西。

## 3. 验证

### 3.1 单测 `tests/test_prosody_wiring.py`(6 条,全绿;全量套件 **305 → 311 passed**)

两条一组,每组只钉一种坏法,免得红了看不出是哪一处:

| 测试 | 唯一的红法 |
|---|---|
| `test_interview_answer_audio_logs_real_prosody` | 面试端点退回 `{}` |
| `test_research_answer_audio_also_logs_real_prosody` | 科研端点退回 `{}` |
| `test_prosody_is_computed_on_the_webm_path_the_browser_actually_uses` | 面试端点退回 `{}` |
| `test_extractor_std_columns_are_renamed_to_the_log_column_names` | `_EXTRACTOR_TO_LOG_COLUMNS` 改空 |
| `test_rename_map_has_no_silent_gap_between_extractor_and_logger` | 提取器产出日志层不认识、也没登记改名的键 |
| `test_extraction_runs_off_the_event_loop` | 去掉 `asyncio.to_thread` |

### 3.2 反向复现(逐项实跑,均先清 `__pycache__`)

```
变异 1  改名表改空        → 2 failed(改名那两条) 4 passed
变异 2  面试端点退回 {}   → 4 failed(面试/ webm / 改名 / 线程) 2 passed
变异 3  科研端点退回 {}   → 1 failed(test_research_…) 5 passed   ← 精准
变异 4  去掉 to_thread    → 1 failed(test_extraction_runs_off…) 5 passed  ← 精准
```

### 3.3 测试夹信号的坑(写测试时踩到,记下来)

停顿族**验得到验不到,取决于静音段有多长**。`librosa.feature.rms` 窗长 2048
(0.128 s @16k),0.2 s 的静音只有 3 个「整窗都安静」的帧 →
`(i-pause_start) * duration/len(rms)` = 0.095 s,过不了提取器自己 `pause_duration > 0.1`
的闸 → **停顿列仍是 0,测试会假绿**。夹信号里那段静音因此定为 **0.5 s**。

### 3.4 真会话现场端到端(真实 webm,不是构造)

起真服务(`--port 8091`,日志与留存都改到 `/tmp`,**没碰仓库 `data/`**),
用 `/interview/start` 铸号,把使用者第一场真会话里那段 **`0001.webm`(472 KB,浏览器录的)**
POST 给 `/interview/answer_audio`:

- HTTP **200**,ASR 回真文本(那场录的是"所以现在我的情绪力…"),`connective_density=2.2222`
- 落盘那一行:**11 列全部从 0 变成真值**(`pitch_mean 254.54 / pitch_variation 221.0 /
  pitch_trend 162.99 / pitch_direction 上扬 / energy_mean 0.0243 / energy_variation 0.0362 /
  speech_ratio 0.99 / duration_sec 29.52 / pause_duration_mean 0.88 / pause_duration_max 4.32 /
  pause_frequency 32.52`)—— **没有任何一列仍然恒为 0**
- M2.6 留存腿未受影响(`0001.webm` + `0001_converted.wav` + 三份账本照常落盘)
- 该场 8 段真回答逐一过 `prosody_features_from_pcm`,**8/8 全列非 0**

### 3.5 事件循环没被卡住(实跑)

提取是**同步重活**:实测 3 s 音频 ≈ 0.3 s、29.5 s ≈ **3.7 s**(ASR 约 1.75 s)。
在一次 ~10 s 的 `answer_audio` 请求**飞行期间**连打 8 次 `/health`:
**0.00–0.01 s 全部立即返回**。(去掉 `to_thread` 那条会被 `test_extraction_runs_off…` 钉住。)

### 3.6 报告层**句子**的前后对照(这一条最能说明问题)

同一场(8 段**内容不同**的真回答),唯一变量 = 语音列是 0 还是真值:

| 报告里那句话 | 改动前 | 改动后 |
|---|---|---|
| 语调变化 | **本次会话内无变化** | **有效样本不足** |
| 语音能量 | **本次会话内无变化** | **有效样本不足** |
| 有效说话占比 / 平均停顿时长 | **本次会话内无变化** | **该指标本轮停用** |

改动前那句是**假话**:它说"本次会话内无变化",读者会理解成"量过了,是常量",
而真相是**根本没量过**。改动后两句都是真的:「有效样本不足」= 量到了、样本量不够;
「本轮停用」= 该列已按 G4 封停。

特征引擎层的门判定(8 段真回答,n=8,G2 有戏):

| 列 | 改动前 | 改动后 |
|---|---|---|
| `pitch_mean` | 0.0 门=不过 | 159.79 **门=过** |
| `pitch_trend` | 0.0 门=不过 | 77.54 **门=过** |
| `energy_variation` | 0.0 门=不过 | 0.0455 **门=过** |
| `pause_frequency` | 0.0 门=不过 | 29.57 **门=过** |
| `pitch_variation` / `energy_mean` / `speech_ratio` / `pause_duration_mean` | 0.0 不过 | 真值,仍**封停**(G4) |

**注意**:上面四个"门=过"只表示**该列自身**过得了闸,不代表它被挂到了某个构念上
(挂不挂由 `research_mapper` 决定,本轮没动)。报告句子的实际变化见上表。

## 4. `pitch_std` / `energy_std` → `pitch_variation` / `energy_variation`(本轮最大的静默陷阱)

**提取器吐的键和日志列名不一样**:

| 提取器产出 | 日志列(`VoiceLogger.fieldnames`) |
|---|---|
| `pitch_std` | `pitch_variation` |
| `energy_std` | `energy_variation` |

改名在仓库里**只发生在** `ProsodyAnalyzer.analyze_pitch/analyze_energy`
(那里写的就是 `"pitch_variation": pitch_std`),而活路径直接吃提取器返回值。
**少了映射表,只有这两列静默留 0,其余每一列都对** —— 所以单列一条测试钉它。

## 5. 为什么必须 `asyncio.to_thread`

`librosa.pyin` 是 CPU 同步重活。端点全是 `async def`,在协程里直接算 = 卡住整个事件循环。
这一条**本文件顶上已经为 ASR 记过**(`_transcribe_async` 的注释:内部 `asyncio.run`
不能在有循环的线程里调),语调要走同一条。实测 `/health` 在长请求飞行期间 0.01 s 应答。

## 6. 已知缺陷(照实登记,不在本轮修)

1. **`speech_ratio` 自指阈值** —— 实测 8 段真回答里 **7 段恰为 1.0**、一段 0.99,
   与文档里 MIT 那批「87.9% 恰为 1.0」完全一致。spec §4.3 要删这一列。
   报告层已封停(G4:「自指阈值,87.9% 恰为 1.0」),出不了分,**但它现在真的在产出了**。
2. **停顿检测的窗边长边** —— 每段停顿两侧各被吃掉最多一个窗(见 §3.3 实测),
   另有已登记的「尾部静默被丢弃」。
3. **`is_valid` 仍是恒 1** —— 详见 §7 的裁定。
4. **pyin 的 f0 轨迹在真录音上不干净(本轮实测发现,新证据)** ——
   对使用者第一场那段 29.5 s 的回答:`pyin` 给的有声帧中位数 **127 Hz**(合理男声),
   但 **179/514 帧挤在 400–500 Hz 那一桶**(比任何单桶都多),`pitch_std` 因此被抬到
   **221 Hz** —— 人类语音的 f0 标准差通常在 20–60 Hz,**这个数不可能是真的**。
   关键:相邻有声帧里比值 ≈2 / ≈0.5 的只占 **1.0%**,所以**不是逐帧八度跳变**,
   而是**整段整块**偏高 —— 简单中值滤波修不掉。
   同一批里 `0005`(中位数 132 Hz,5–95% [73,172])与 `0008`(中位数 97 Hz,[82,126])
   是**干净的单峰**,说明不是整批坏,是**部分段坏**。
   → 归 M3(§0.1 明说 M3 重写能量门限 / `voiced_prob` / 真 VAD)。
   **在此之前,`pitch_mean` / `pitch_trend` / `energy_variation` / `pause_frequency`
   这四个现在能过门的列,值是真的"提取器的输出",但还不是可信的"韵律测量"。**

## 7. 裁定:`is_valid` 本轮**不改**(已随"按你推荐的来"一并获批,2026-09-25 深夜 2)

`evidence_gate.py` 已登记 `"is_valid": Quarantine("恒 1 且下游从未使用", "M3 改真掩码")`。

我**没有**把它改成派生值,理由是一条实测结论:
本路径上 **ASR 有文本才走得到 `log_prosody`**(`if not text: raise 400` 在它前面),
所以音频必然非空,`extract_all_features` 也必然返回非空字典
→ 派生出来的 `is_valid` **仍然是恒 1**。那是"看着像修了"的假改动,
却会让那条已登记的封停理由(「恒 1」)变成**假话**。
真掩码是**逐槽位**的东西,归 M3。

**若使用者认为该改**,改法是把判据从"提取器跑没跑"换成"有没有人声"
(`pitch_direction == "无法判断"`),但那要同时决定 `connective_density` 这类
**同一行里本就算得出来的**值算不算被这条标记连坐 —— 那是定义问题,不是接线问题。

## 8. 顺带查出来的两个报告层问题(本轮**没动**,报给使用者裁定)

### 8.1 `pitch_variation` 那条封停理由与代码**净行为不符**(已实测)

`evidence_gate.py:80` 写的是 `Quarantine("实际取的是 pitch_mean,名字与计算不符", "改名后")`。
实测**净结果取的是正解**:

- 构造:`pitch_mean` 列恒 111、`pitch_variation` 列恒 222 → 特征值 **222.0**
- 真会话:该场 `pitch_mean=254.54`、`pitch_variation=221.0` → 特征值 **221.0**

原因是 `feature_engine.py:292-293` 那条别名**被后面的通用数值扫描覆盖掉了**,
而真实 CSV 的列序恰好是 `pitch_mean` 在前、`pitch_variation` 在后 —— 所以别名先写、
扫描后写,净结果是正解。诊断当时是对着**那一行**下的,没看净效果。

### 8.2 但那条别名本身是颗雷(已实测,建议归 M3)

同一份数据,**只把两列的先后换一下**:

```
真实 CSV 的列序(pitch_mean 在前)   → 222.0  正解
交换后的列序(pitch_variation 在前)  → 111.0  **取错成 pitch_mean**
```

即:今天是对的**纯属列序碰巧**。任何一次列序调整都会让它静默变成
"名字叫 pitch_variation、值是 pitch_mean"—— 而报告层**没有任何测试**盯这件事。

## 9. 本轮**没做**的(免得下轮当成已做)

- 没写 spec / 计划(§0.1「第 1 件」本身就是依据;本条若需要独立 spec 请说)
- 没动前端(—— 但本轮**另**合了前端那个 WIP 分支,见 §10)
- 没动报告层
- 没动 `ProsodyAnalyzer._calculate_overall_score` 那套同形态的无出处阈值(归 M3,见 §10)

**已做**:提交并推送 origin —— 代码 `ed834fd`、文档与账本 `71c5fee`。
合并门已跑:**0 / 2835510**(与基线逐位一致)。

---

## 10. 使用者裁定的三条挂账(2026-09-25 深夜 2)

使用者裁定:「按你推荐的来,没用就删掉,尽快把系统打通」。详版见 `docs/下一步.md` §0.4。

### ① 前端 WIP 分支 —— 合并 → 然后删掉(不是"没用",是**很有用**)

`wip/frontend-2026-08-06-sanitized` 相对 main 只有一个提交 `9b7821d`
(另一个是已 cherry-pick 过的音频修复)。里面有浏览器 TTS、`ErrorBoundary`、
代码分割、以及**把两个评估页收敛成一个**。

**决定现在就合的关键理由**:N2 要动评估流程;main 上原本是两个评估页,
不合就得做两遍。合完只剩一个 `AssessmentPage`,**N2 只做一遍**。

合并前核过:两处「编数据」已清干净(`setEvaluationResult` 零调用方);
`9b7821d` 没碰 `api.ts`/`ReportPage.tsx` ⟹ M2.1 的 `withSession`(10 处)
与披露卡原样在;`ReportPage` 的 store 短路是 main 本就有的;
**隔离快照 `tsc --noEmit` + `vite build` 三份(main / WIP / 合并后)全过**。
**已执行**:合并 `2dd713c` → 推送 → 删分支(本地 + 远端)。

### ② 后端无出处阈值测评 —— 删阈值,**不删端点**

它是"炸不了"的:活路径 `add_answer` 只收字符串 ⟹ `qa.prosody_analysis` 恒 `None`。
但 **`add_qa_pair()` 接受 prosody** —— **实测引爆成功**:喂
`pitch_variation=45 / speech_ratio=0.8 / energy_mean=0.5`,立刻吐
「语调起伏大,富有表现力;表达流畅;音量适中」。三个阈值都没依据
(`speech_ratio` 自指 ⟹ 恒判「表达流畅」;`energy_mean` 真值 0.02–0.06 对阈值 0.5–0.8
⟹ 恒判「声音偏轻」)。**已删两个类各一份阈值判语**;三个活端点
(`/interview|research/evaluation`、`/session/{sid}/summary`)实测均 **HTTP 200**。
钉子 `tests/test_no_baseless_prosody_verdicts.py`(6 条),反向复现红 4 条。

### ③ `ProsodyAnalyzer` 那套同形态阈值 —— **本轮不动,归 M3**

唯一上游是 `VoiceProcessingPipeline`,而它只被两个 `examples/` 构造。
模块会被传递加载(`pipeline/__init__.py`),但**没有活代码路径调用**。

