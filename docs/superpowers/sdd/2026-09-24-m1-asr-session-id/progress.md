# SDD ledger — plan: docs/superpowers/plans/2026-09-24-m1-asr-session-id.md

- 分支:`feat/m1-asr-session-id`(自 `main` 起,BASE = `1e53c131`)
- 解释器:`~/miniconda3/envs/jingxin/bin/python`
- Spec(权威):`docs/superpowers/specs/2026-09-24-m1-asr-session-id-design.md`
- 使用者指令:一个模块一个子智能体,相互配合;验收门由使用者亲验(≥2 段回答)

## 预检冲突扫描(dispatch Task 1 之前)

### 共享文件 / 接口的任务对

| 任务对 | 共享物 | 一方产出 vs 另一方消费 | 发现 |
|---|---|---|---|
| T1 → T2 | `funasr_engine.count_cjk_chars` / `_config` | T1 产,T2 消费 | ✅ 名字与签名一致 |
| T1 → T4 | `AsrUtterance` / `FunASREngine` / `session` / `transcript_store` | T1 产,T4 消费 | ✅ 一致 |
| T2 → T4 | `connective_density()` 的 None 语义 | T2 产,T4/T6 消费 | ✅ "过短不出值,不写 0"三处一致 |
| T3 → T4 | `VoiceLogger` 的 session_id 关键字参数 + 三个新列 | T3 产,T4 写 | ✅ 新列名与 `log_prosody` 新 kwargs 同名 |
| T3 → T5 | `NONE_SESSION` 字面量 | T3 产,T5 消费 | ⚠️ **计划让 T5 从 voice 包导入** → 方向错误且拉起重依赖 → 见 Ruling M1-2 |
| T1 → T3 | voice logger 需要 `NONE_SESSION` | T1 产,T3 消费 | ⚠️ 从 `..asr.session` 导入会触发 `voice_interaction/__init__` 的重导入 → 见 Ruling M1-1 |
| T4 ↔ T6 | 日志列 `connective_density` → 特征键 `voice_*_connective_density_mean/_std` | T4 产,T6 消费 | ✅ 走 `_extract_numeric_stats` 通用数值路径,列名一致即可 |
| T6 ↔ ① | `tests/test_report_layer.py` 的 `_MIXED` 等 fixture 用的是**旧键** `logic_keyword_density` | T6 改名后旧键不再被映射 | ⚠️ **必须同步改 ① 的 fixture**,否则 ① 的测试会红 → 见 Ruling M1-3 |
| T6 内部 | mapper 关键字 `connective_density` vs `evidence_thresholds.json` 的 `density` 族 | 子串匹配仍然命中新键 | ✅ 但量程必须随新定义更新(计划已含) |

### 每个任务的自我一致性

| 任务 | 自己的测试 vs 自己的代码 | 发现 |
|---|---|---|
| T1 | 5 条引擎测试对应 AsrUtterance 的四个字段 + raw 时间戳 | ✅ 一致(测试断言与实现字段同名) |
| T2 | 6 条密度测试对应 `connective_density` 的 None 语义与字数口径 | ✅ |
| T3 | 文件名 / 首列 / NONE 分支 / 跨模块同值 | ⚠️ 一致性测试需从"3 个文件"改为"4 个文件"(加 `asr/session.py`)→ 见 Ruling M1-2 |
| T4 | 只测 `_transcribe` 转发;vosk 归零靠 grep | ✅ 端点层不做导入级测试(已写进 Global Constraints) |
| T5 | 文本级断言(不得再有 `uuid4()`) | ✅ 与实现一致 |
| T6 | 三条:改名 / 无第二处定义 / 新键能过门出分 | ⚠️ 还需第四条:① 的 fixture 同步(见 Ruling M1-3) |
| T7 | 纯运行手册,无代码 | ✅ 无需 |

## Rulings(开工前)

**Ruling M1-1(必须做,否则新测试跑不起来):** `voice_interaction/__init__.py` 目前在 import 时**急切构造** `TTSPipeline` 与(现)`SpeechRecognitionPipeline`(后者还要 vosk 模型)。而 T1–T3 的测试要 `import voice_interaction.asr.*` —— 那会连带跑完整个包的重导入。**裁定:把 `__init__.py` 改为惰性导出(PEP 562 module `__getattr__` + 保留 `__all__`)**,现有 `from voice_interaction import X` 全部照旧可用,但不再在 import 时构造管线。 — 若判断错,代价 = 惰性属性语义与急切导入有细微差别(用一条断言公开名字仍可解析的测试兜住);反向代价是所有新测试都拖着 TTS/vosk 导入,慢且可能因无音频设备而失败。

**Ruling M1-2(`NONE_SESSION` 放哪):** 字面量在 **4 处本地定义**(`voice_interaction/asr/session.py` + 三个 `utils/logger.py`),由**文本级一致性测试**守住同值;face/gesture **不**从 voice 包导入任何东西(方向依赖错,且会拉起重依赖)。计划里 T3 的测试改为扫这 4 个文件;T5 的导入改为 `from ..utils.logger import NONE_SESSION`(本包内)。 — 若判断错,代价 = 改一个常量要动 4 处(测试会立刻发现漏改);反向代价 = face/gesture 服务启动时加载 TTS 引擎。

**Ruling M1-3(T6 必须带上的连带修改):** ① 的 `tests/test_report_layer.py` 里 `_MIXED` 等 fixture 用旧键 `logic_keyword_density`;T6 把映射关键字改成 `connective_density` 后,那些测试会因"没有任何指标过门"而变红。**裁定:T6 的派发词里必须写明同步更新 ① 的 fixture 键名**(改为 `connective_density_mean` + `connective_density_std`),且 T6 的验收里包含"整套 53 个测试仍全绿"。 — 若判断错,代价 = T6 完成后套件变红,需一次修复轮。

**Ruling M1-4(分支而非 worktree):** 与 ① 一致 —— 使用者未要求 worktree(harness 规定 EnterWorktree 仅在明确要求时使用)。分支 `feat/m1-asr-session-id` 已建,BASE = `1e53c131`。 — 若判断错,代价 = 使用者想要 worktree 时需重来一次 setup,无代码损失。

## 进度

- setup 完成:工作区、账本、预检扫描、4 条开工前裁决。

**Task 1 实现完成**(`34064d9` + `a76874c` + `4447593`)—— 报 **DONE_WITH_CONCERNS**;整套 **74 passed**(53 旧 + 21 新);vendor 客户端 md5 未改;测试从未写真实录制目录。

实现者按"真实 API 为准"改了计划的两处:
- 计划里 `recognize_pcm(pcm, host=...)` 漏了 `port`/`timeout`,而真实签名是 `arecognize_pcm(pcm, host=, port=, mode=, on_partial=, realtime=, timeout=)`(**kwarg 名是 `timeout` 不是 `timeout_s`**)→ 已转发,并加测试。
- 计划 Step 4 的代码块漏了它自己 *Produces* 段承诺的 `append_utterance` → 实现者补齐了。

**控制器裁定的两处:**
- **Ruling M1-5(否决 JSONL,按 spec §6.4 实现):** 实现者给 `append_utterance` 自创了 `transcript_{session_id}.jsonl`(每行一个 JSON)。**spec §6.4 明确规定的是单个 `transcript.json`**,含 `session_id` / `recorded_at` / `asr` / `merged`(重算 `n_chars`、`n_segments`、`vad_split`)/ `segments`(带 `index`,累积)。下游(M3 重算、验收核对)读的是 spec 的形状,故**以 spec 为准,JSONL 作废**。 — 若判断错,代价 = 多一次读-改-写循环,但契约与所有人读到的文档一致。
- **Ruling M1-6(计划缺陷,必须在 T4 的派发词里带上):** 实现者发现 `recognize_pcm` 内部用 `asyncio.run` —— **在运行中的事件循环里调用会直接抛错**。而 T4 要把它接进 FastAPI 的 async 端点。→ **T4 必须用客户端的异步 API(`arecognize_pcm`)或 `asyncio.to_thread` 包装**,不得在事件循环里直接调同步版。 — 若判断错,代价 = T4 的端点在真链路上一调就 500。

**Task 1 也已记的次要项:** ① 计划自带的 `test_multi_segment_merge_is_kept` **对它的名字不可证伪**(只断言段数/文本,把时间戳来源改错仍绿),实现者另加了一条用真实 `ASRResult` + `_merge_finals` 的测试才钉住 —— 后续任务的 brief 可能有同类弱点;② 空结果会退化成"1 个空段"(`n_segments=1`),T2/T7 不得用 `n_segments==0` 表示"没识别到";③ 我先前的遗留已清:`report_frontend/data/output/` 下 7 个被跟踪的旧图表删除已单独提交(与 M1 无关)。

**Task 1 fix 轮 1/5(1 addressed,0 open; commits `4447593..d6d3754`)**

- Ruling M1-5 落地:实现者**直接按 spec §6.4 写**(而不是照我消息里的片段),因此多做了三件我片段里省掉的:从 config 取 `endpoint` 拼 `ws://host:port`、`models` 块、以及 segment 的**六个**确切键。5 条新断言改前 RED(理由都对:jsonl 路径 / `KeyError: 'asr'` / `JSONDecodeError: Extra data` / 裸 isoformat 无时区),改后 GREEN;15 个变异全被杀。
- **实现者揪出我计划里的一个真缺陷(重要):** `test_new_session_id_is_unique` 断言"50 次抽样互不相同" —— 4 位十六进制只有 65536 个值,**按生日问题有 1.86% 概率假红**(实测 8 次全套跑中 1 次红,20 万次蒙特卡洛吻合),而且它一红会**连带**把断言覆盖检查器弄红(`inner_exit_code == 0` 那条门),一次抖动看起来像两个失败。已改为确定性钉法(`token_hex` 打桩)+ 一条文档化的宽松冒烟(≥48)。 → **后续任务不要再用"抽样互不相同"作唯一性断言。**
- 次要项(deferred):`asr.models` 现为 `{}`(服务不返回模型名,实现者拒绝猜测 —— 正确);`indent=2` 让 `timestamps_ms` 每字约 6 行(与 `session.json` 风格一致,仅体积);`merged.vad_split` 语义钉为"**任一段裂过即 true**"(O R over calls,与 spec §6.4 一致)。

**控制器从服务端取回的真实模型名(供 T4 填 `asr_config.json` 的 `models` 块,依据 = 读自 `~/huihuibui/logs/server.err.log` 的 `Loading ckpt` 行,不是 API 返回):**
```
iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online     (流式)
iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch  (2pass 离线)
iic/speech_fsmn_vad_zh-cn-16k-common-pytorch                              (VAD)
iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727             (标点)
iic/speech_campplus_sv_zh-cn_16k-common                                   (声纹验证 → spk_score 的来源)
```
**Task 1 审查已派**(复审包按路径排除我那次无关的图表删除提交,从 34 MB 降到 50 KB)。

**Task 1 审查结果**:Spec ❌(窄)/ Task quality **Needs fixes**;0 Critical、**1 Important**、10 Minor。

审查者逐条核过核心交付物并给了证据:vendor md5 字节一致、**惰性导出直接跑解释器验证**(15 个公开名全部解析、`v.NoSuchName` 正确抛 AttributeError、`import voice_interaction` 不再把 vosk 拉进 `sys.modules`)、§6.4 的键序/段白名单/`ts_origin`/置信度 provenance 全部与 spec 对上、`report_frontend/` 零改动、写盘原子(`tmp` + `os.replace`)。

- **Important #1:`tests/test_transcript_store.py:40` 名字里那条不变量根本没被测。** 它传显式 `root=tmp_path` 且只断言"等于该 root 且存在" —— 永远不可能因"原文被写进仓库"而变红;而没有任何测试用 `root=None` 调过 store。**把默认根改成仓库内路径,该文件 10 条测试全绿。** 这压着的正是 brief 里被称作绝对约束的那一条。
- 控制器自己核实了审查者标注"无法从 diff 验证"的两项:**78 passed 属实**;5 个提交各有一个 parent(无 amend/rebase),文件面与提交主题相符。
- **Ruling M1-7(`vad_split` 保持逐次 OR,不改):** spec §6.4 原文是"只要任一段裂过即为 true"。OR 正是用来区分"**一次回答被 VAD 切开**"与"**两次完整回答**";"总段数 > 1"会把后者误算成裂过 —— 那正是这个字段存在的意义。**不改,也不加新断言。**
- **Ruling M1-8(把 3 条 Minor 折进本轮,理由:同文件 + 后两条挡着 T2):** ① `funasr_engine.py:7` 模块 docstring 仍写着实现者自称已纠正的两句(照字面读会以为卡住的 socket 有界);② `asr_config.json` 的 `models` 为空,而 spec §6.4 契约有四个键(下游索引会 `KeyError`)→ 用控制器**从服务端日志取回的真实模型名**填上,依据注明来源;③ 跨模块导入私有名 `_config` → 提升为公开访问器。 — 若判断错,代价 = 多改两三行;反向代价 = T2 建在私有名与空 `models` 上。
- **deferred minor(记档不修):** manifest 与 transcript 的 provenance 重复;`ensure_manifest`/`refresh_manifest` 非原子写;`refresh_manifest` docstring 与实现不符;store 内 duck-typing 不一致;`secrets` 全局打桩;`test_empty_text_...` 名实不符;以及 **并发 read-modify-write 无锁**(已要求实现者评估其在 T4 接进端点后的真实风险,结论带进 T4 派发词)。

**Task 1: fix 轮 1/5 已发**(Important #1 + 折入的 3 条 Minor;commits 待到 `d6d3754` 之后)。

**Task 1 fix 轮结果**(`a600e13`,父 `d6d3754`,4 files +65/−9):**82 passed**(连跑 4 次),5 个变异全部被杀 —— 包括**精确复现我那条 Important**:把 `DEFAULT_ROOT` 改成 `Path("data/transcripts")`(即把原句写进仓库)时,**另外 14 条测试照旧全绿,只有新加的那条守卫变红**,且失败信息点名后果("面试原句会被写进 git")。实现者还多加了 `is_absolute()` 断言,让"相对路径默认根"在任何 CWD 下都能被抓到。

四处修复:① 该测试改名 + 两条不带 `root` 的新测试(默认根不在仓库内 / `JINGXIN_RECORDINGS_DIR` 覆盖生效);② 模块 docstring 第 4 条改为事实陈述(异常在本路径上上抛、`timeout` 只兜收尾 drain、**没有中途看门狗**);③ `models` 填上四个契约键(值来自控制器从服务端日志取回的**真实模型名**);④ `_config` → 公开 `load_config()`(就地做,未新建 `config_loader.py` 模块,理由是保持"配置路径常量只有一处")。

- **实现者的一处诚实值得记:** 它**拒绝声称自己读过服务端日志**(它没有那台机器的 ssh 授权),所以 `models_basis` 只写"部署方在服务端读取",不暗示自己核过。四个名字与两份独立记录吻合(它的探测笔记 + 本账本)。
- **Ruling M1-9(不把声纹模型提升为第五个键):** 它问要不要把 `iic/speech_campplus_sv_zh-cn_16k-common` 也作为 `models` 的键(那是 spec §6.4 契约的变更)。**裁定:不加** —— 契约是四个键,下游按那个形状读;SPK 模型及其与 `spk_score` 的关系记在 `models_basis` 文本里即可。 — 若判断错,代价 = 想按模型名筛选会话时要再改一次契约。
- **并发风险(实现者的判断,带进 T4 派发词):** spec §7 的逐段顺序流里不会触发,但 **T4 接进端点后就现实** —— 客户端重试/重复提交(ASR 要 ~1.75 s,重试窗口大)、预取下一题、或 `/asr` 与 `/answer_audio` 竞态。失败形态是**丢更新而非文件损坏**(`os.replace` 已让读者安全),而且它**静默**丢掉一条回答的审计轨迹、数字报告却看不出异常 —— 所以能不被发现。建议 T4 加**按 `session_id` 的 `threading.Lock`**(单进程)或 `fcntl.flock`(多 worker);`ensure_manifest` 同形但幂等,故只有 transcript 是有损的那个。

**Task 1 范围复审已派**(`d6d3754..a600e13`,1 提交,12 KB)。

**Task 1 范围复审结果**:四条 finding + `vad_split` 裁决 **全部 ADDRESSED,无新增 Critical/Important**。

复审者逐条独立验证(不照信报告):① **在内存里复现变异**(不动文件)把 `DEFAULT_ROOT` 改成 `Path("data/transcripts")` → 新守卫抛出「默认落盘位置落在仓库内,面试原句会被写进 git」,且确认**只有它一条会红**;测试文件 11 → 15 条,与 78 → 82 的增量吻合;② docstring 修复对着 vendor 源码逐行核(push 循环确实无超时;`timeout` 只出现在收尾 `ws.recv()`;吞异常只在 `ASRSession._run/_main`);③ `models` 四键的值与控制器取回的清单**逐字符一致**,且守卫读的是**真实配置文件**而非 fixture;④ `_config(` 全仓零调用点,守卫会因"绕过访问器"或"改回私有"而红(**不是静默失败**)。

- **new Minor(deferred,交最终审查):** `tests/test_transcript_store.py:56` 的 `assert default_root.is_absolute()` **是恒真断言** —— 它在 `.resolve()` 之后求值,而 `resolve()` 永远返回绝对路径,所以那行永远不会触发。后果两件:① 它正是 `test_assert_coverage.py` 自己文档化的盲区(恒真断言抓不到),没有东西能抓它;② 报告里"这一行让判定不依赖 CWD"的说法是**错的** —— 真正能触发的只有 `:57` 的包含检查,而它是 CWD 相对的。在规范调用下(`cd repo && pytest`)守卫确实会红,所以 Important #1 仍判 ADDRESSED。一行修法:在 `.resolve()` **之前**断言 `is_absolute()`。
- **out-of-scope,但必须带进后续派发词:** 计划与 T1 brief 的代码片段里仍写 `_config()`,而它已公开为 `load_config()` —— **照抄片段的任务会直接 `AttributeError`**。T2/T3 的派发词必须带上新名字。

**Task 1: complete (commits `1e53c131..a600e13`, review clean)** —— 12 个提交(含 1 个与本任务无关的控制器 housekeeping `5bf06a3`)。

→ 派 Task 2(连接词密度 + 标记表)。按 Model Selection,**这是"计划文本含完整代码"的抄写+测试型任务,用最便宜一档**(haiku),派发词里点名 `_config` → `load_config` 的修正。

## Task 3 派发前的修正(T3 brief 已生成,108 行)

T3 brief 里有两处必须改,已记下待派发时带上:

1. **一致性测试要扫 4 个文件,不是 3 个**(Ruling M1-2):brief 的 `SOURCES` 只列了三个 logger,而 `NONE_SESSION` 的字面量在 **4 处**本地定义 —— 加上 `voice_interaction/asr/session.py`。四个文件各持一份 `NONE_SESSION = "NONE"`(**不做任何跨模块导入**,包括 voice 内部),由那条文本级测试守住同值。
   - 附带效果:brief 的 "Consumes: `voice_interaction.asr.session.NONE_SESSION`" 一行作废 —— 三个 logger 各写各的,不 import。
   - T4 的 voice `api/app.py` 从 `..utils.logger` 或 `asr.session` 取都可以(值由测试守住);T5 的两个 `api/app.py` 一律从**本包**的 `utils.logger` 取(不得依赖 voice 包)。
2. **重依赖的应急方案 brief 里已有**(若 import `face_expression.utils.logger` 拉起 mediapipe/cv2 → 改用 `importlib.util.spec_from_file_location` 或降级为文本级断言,行为断言集中在 voice logger),保持不变。

T3 还要记得:voice logger 的 `fieldnames` **末尾追加三列**(`connective_density` / `connective_density_std` / `n_rows`),`log_prosody` 末尾加对应的三个关键字参数(默认 `None`)—— 那是 T4 写密度时的落点。

**Task 2 实现完成**(`dbbe1c4`,3 files +98)—— 报 **DONE_WITH_CONCERNS**;**90 passed**(82 旧 + 8 新),`asr_config.json` 未动。

**实现者的自查发现(值得记):** brief 的 6 条测试**全部显式传 `min_chars=10`**,所以它验证出"下限来自配置""标记表带 `_provisional`/`basis`"这两条全局约束**没有任何守卫**(把下限写死、或删掉那两个字段,6 条测试照旧全绿)—— 它自己补了 2 条守卫。这正是"测试通过 ≠ 有约束力"的又一次现身,而这次是**实现者自己**抓到的。

**控制器的三条裁定:**
- **Ruling M1-10(折进预审修复):让"空文本 → None"无条件成立。** 现在 guard 是 `chars < floor`,所以 `connective_density("", min_chars=0)` 会 `ZeroDivisionError`。生产够不到(配置下限 10),但这是公开纯函数、契约写着"空文本返回 None",而 M3 之后的实验很可能传别的下限 → 改为 `if not chars or chars < floor`,并补一条 `min_chars=0` 的测试(会因 `ZeroDivisionError` 而红)。 — 若判断错,代价 = 多一个分支;反向代价 = 未来一次难查的崩溃。
- **Ruling M1-11(折进预审修复):给标记表加"无子串包含"不变量守卫。** 实现者指出 v1 的 17 个词恰好没有子串关系,但**那是表的性质、不是代码的性质** —— 将来加一个 `为` 就会静默重复计数。补一条遍历式测试(双向断言无包含),**故意在表被改坏时红**。 — 若判断错,代价 = 表里真出现子串关系时(比如有意加入短标记)得先改测试。
- **Ruling M1-12(不改):`100` 保持内联。** 它是"每百字"的**单位**,不是可调阈值;放进数据文件会制造"这个数字可以调"的错觉。要求 docstring 说明它是单位。

**Task 2 预审修复已发**。

**Task 2 预审修复结果**(`b0b7ab6`):**92 passed**(90 → 92,两条新测试就是全部增量)。两条折叠修复的 RED 都"红得对":
- 空文本:改前 `chars == 0` 且 `floor == 0` 时穿过 guard,死在 `hits / chars`(`ZeroDivisionError`,1 failed/9 passed)→ 改 `if not chars or chars < floor` 后绿;这条测试**唯一的红法**就是去掉 `not chars or`。
- 标记表不变量:**红来自破坏数据文件**,不是改代码 —— 加一个 `为`(`因为` 的子串)立刻变红,且**生产代码一行未改**时同一句的密度从 `2.0` 涨到 `4.0`,别的测试全绿 → 正是这条守卫要拦的静默膨胀。表已还原,`git diff` 干净,`connective_markers.json` 本轮**故意未改**(17 词 / 4 键原样)。
- 附带:`100` 保持内联,docstring 已写明它是"每百字"的**单位**、不是可调项。
- 实现者另发现:新那条零下限测试**顺带守住了另一侧** —— 重跑 M4 变异(`return None` → `return 0.0`)现在会让 **4 条**测试红(此前 3 条),即 None/0.0 的边界在两个下限上都闭合了。

**Task 2 任务审查已派**(`a600e13..b0b7ab6`,2 提交,7 KB)。

**Task 2 审查结果**:Spec ✅ 合规 / Task quality **Needs fixes**;0 Critical、**1 Important**、6 Minor。

审查者逐条核过四条强制项并给证据:brief 六条测试**逐字保留**、标记表 17 词/4 键与 brief 一致、`asr_config.json` **不在 diff 里**、LF 全净、stdlib-only、范围未越界;三条裁决各自落地且 RED 都真实(零下限那条的红是真的 `ZeroDivisionError` 回溯,不是断言失败;表不变量那条的红来自**数据**变异且给出了静默膨胀的数字 2.0 → 4.0)。

- **Important #1:那条"每百字"的**分母**没有被任何测试守住。** 所有"有命中"的 fixture **恰好 100 字** → `hits / chars * 100` 退化成 `hits`。审查者模拟验证(未改仓库):把归一化整个删掉(`return round(hits, 4)`)**10 条测试全绿**;`round(hits / 100 * 100, 4)` 也全绿。**而分母正是这个指标能跨句长比较的原因,也是 T4 要写进真实日志的值**(真实回答永远不是 100 字)。报告 §3 把该测试称作"每百字定义的唯一守卫",实际只覆盖单位因子、没覆盖分母 —— **这是本项目"改前改后都通过"的第 N 次现身,而且这次是审查者抓的。**
- 控制器核实了审查者标"无法从 diff 验证"的两项:**两个提交的作者身份正确、各一个 parent(无 amend/rebase)**;**92 passed 属实**。→ ⚠️ 两项均已消解。
- **Ruling M1-13(折进本轮 3 条 Minor):** ① 下限**边界**未钉(把 `<` 改成 `<=` 仍全绿,而 spec §6.5 是 `n_chars < floor`,相等要出值);② "同一标记只计一次"只有 docstring、无测试(改成 `text.count` 仍全绿);③ **表里重复标记会双计,而两条表守卫都拦不住**(`a != b` 按构造排除了相同串这一对)—— 同一条表测试里加 `len(ms) == len(set(ms))` 即可。三条都是"加断言",与 Important 同一目的:**把定义钉住**。 — 若判断错,代价 = 测试变多;反向代价 = 指标定义可被静默改掉。
- **deferred(记档不修):** `load_markers()` 的 `@lru_cache` **把可变对象直接发出去**(调用方 append/sort 会污染本进程后续所有调用),且带 `path` 调用会挤掉默认缓存项(`maxsize=1`)——**首要 deferred**;`test_floor_is_read_from_config_not_hardcoded` 单边;spec §10 的"全标记"用例缺口。

**Task 2: fix 轮 1/5 已发。**(T3 实现者同时在跑,文件不重叠。)

**Task 2 fix 轮 1/5 结果**(`bf1758a`,**仅测试 +29 行**,13/13):

- 实现者**先自己复现审查者的 Important 主张**(不照信):`return round(hits, 4)` 与 `round(hits / 100 * 100, 4)` 确实都让原来 10 条全绿 —— **确认无误**,并把自己报告 §3 那句"唯一守卫"**就地更正**(留指针到 §F)。修后这两个变异各让 3 条红。
- 三条折叠 Minor 各"恰好一条测试红":`chars <= floor` → 只有边界那条;`text.count(m)` → 只有重复标记那条。**第 4 条的验证最有价值:** 把 `然后` 复制进表时,值断言**碰巧**也能抓到(6 条红);但把 **`例如`**(任何 fixture 里都不出现)复制进表时,**12 条值断言全绿,只有那条去重断言红** —— 证明那行不是摆设,没有它这张坏表会**完全静默**。
- **实现者自查出的错误(记 §F.4):** 它第一版重复标记 fixture 写成 `"然后然后三个" + "字"*47` 并断言 `== 2.0`,立刻红在 `1.8868` —— 因为 `"然后"*3` 是 **6 个字不是 3 个**,长度是 53。**实现是对的,它的期望值错了。**
- **并发干扰的处置(值得记):** T3 的两个**未跟踪**测试文件在它工作期间出现且正在被编辑,导致整套数字在动(10 → 19 条之间跳)、并连带弄红 `test_assert_coverage`。实现者**逐一归因、没碰 T3 的文件、也没调整测试选择让套件显得绿**,并用断言覆盖检查器(只读调用)证明 `inner_exit_code == 0`、`unexecuted = {}`、`never_called = []` —— 即**它加的每条断言与套件里每条断言都确实执行**。

**Task 2 范围复审已派**(`b0b7ab6..bf1758a`,1 提交 2.8 KB;派发词已告知 T3 并发导致的红属归因干扰)。

**Task 2 范围复审结果**:四条 finding **全部 ADDRESSED,无新增 Critical/Important**。

复审者**独立复现了实现者变异表的每一格**(在生产函数上内存变异,不动仓库):`÷ chars` 去掉 → **3 条红**(且是 `AssertionError` 不是异常);`chars <= floor` → **恰好 1 条**;`text.count` → **恰好 1 条**;表里加 `例如` → **恰好 1 条**(并独立算术确认 `例如` 不出现在任何 fixture 里,所以那条去重断言是**唯一**捕获者)。它还额外验证了"常数分母"在整个区间上被钉住(常数 50 → `:9`/`:87` 红;常数 10 → `:81` 红),以及新 fixture 的期望值**是从公式推的**(50 字 1 命中 → 2.0),不是照着实现输出填的。fix diff **纯增加**、无删除行、无生产代码改动。

**Task 2: complete (commits `a600e13..bf1758a`, review clean)**

**Task 2 deferred minor(交最终 triage):**
1. `load_markers()` 的 `@lru_cache(maxsize=1)` **把可变 dict/list 直接发出去** —— 任何调用方 append/sort 会污染本进程后续所有调用;且带 `path` 调用会挤掉默认缓存项。**首要 deferred。**
2. `test_floor_is_read_from_config_not_hardcoded` 单边:只断言 `is None`,写死 101 也能过。
3. spec §10 的"全标记"用例缺口(覆盖面,非正确性)。
4. 两处 cosmetic:测试 docstring 里 finding 编号与复审编号不一致;去重断言住在一个名字讲"子串包含"的测试里。

→ **T3 实现者仍在跑**(三个 logger)。它回来我派 T3 的任务审查。进度:T1 ✅ / T2 ✅ / T3 🔄 / T4–T7 待。

**Task 3 实现完成**(`6d4bfb5`,5 文件 = 三个 logger + 两个新测试文件;父 `bf1758a`,单父)—— 报 **DONE_WITH_CONCERNS**;**104 passed**(连跑 3 次),9 条新测试(voice 4 / face 2 / gesture 2 / 契约 1),**10 个变异全杀且每个只红在预期测试上**。

落地要点:三处各写本地 `NONE_SESSION = "NONE"`(**零跨模块导入**);契约测试按修正扫 **4 个文件**;文件名 `{prefix}_{session_id}.csv`、缺省 `{prefix}_NONE_{ts}.csv`(与 T1 的 `transcript_store.LOG_PREFIXES` 期望格式一致);`session_id` 为**首列**(face 的 video/static 两个 `fieldnames` 分支都插;gesture 的 `fieldnames` 与写出的 `data` 同步插);voice 末尾追加 `connective_density/_std/n_rows` 并给 `log_prosody` 加同名 kwarg(**默认 None,不写 0**)。**未走重依赖应急路线** —— 实测两个 import 各 0.93s / 0.37s 且成功,故三个 logger 都做了行为断言。

实现者的 5 条 concern,其中三条重要:

1. **超 brief 的一处生产改动:face 的 `log()` 在目标文件不存在时先补表头。** 依据:brief 自带的 face 测试(覆盖 `log_file` 到新路径后直接 `DictReader`)**不补表头必然 IndexError**,而修正项又要求"覆盖模式继续可用"。**这其实修掉一个潜伏 bug** —— 原来表头只在**构造时**写到**构造路径**,调用方一覆盖路径,写出的就是无表头 CSV。M6 变异证明这条断言正好钉住它。→ **待审查者判**,我倾向接受。
2. **测试卫生:它把 face/gesture logger 模块的 `LOGS_DIR` monkeypatch 到 tmp_path** —— 否则 brief 的 face 测试会往**仓库 `data/logs/`** 写真实文件。提交后 `git status data/` 为 0 条,零污染。
3. **brief 对 gesture 没有任何行为测试**(4 条只覆盖 voice×2 + face×1 + 契约×1),其改动原本无守卫 → 它补了 2 条。

**T5 必须知道的一条:** face 的 `api/app.py` **仍在用 `face_au_log_{session_ts}` 覆盖 `log_file`** —— 即 face 那条日志的文件名**暂时仍不含 session id**,要到 T5 改调用方才闭环(本任务按范围没碰 `api/app.py`)。**写进 T5 的派发词。**

**并发:** 分支在它工作期间被推进(`b0b7ab6` → `bf1758a`,另一个会话的 T2 测试补充),且 `tests/test_connective_density.py` 在它第一次跑基线时**正被写入**(同一文件前后两次一红一绿)。它归因正确、未触碰该文件、提交叠在 `bf1758a` 之上。

**Task 3 任务审查已派**(复审包按 pathspec 限定在**本任务的 5 个文件**,把范围里那个无关的 T2 测试提交排除;EOL 自检:diff 里仅 **18 行删除**,不是整文件行尾重写)。

**Task 3 审查结果**:Spec ✅ 合规 / Task quality **Approved**;**0 Critical、0 Important**、7 Minor。

审查者逐条核过**并自己重跑了相关测试**(`9 passed`):修正 1(契约测试**真的能分辨四个文件** —— 它在内存里逐个改那四份字面量,每一个都红;把 face 的字面量换成 import 也红)、修正 2(voice 21 列、三个新列两侧都钉:写路径 `"3.5"` 与"不是 `0.0`")、首列契约(video/static/voice/gesture 四个分支实测)、`NONE` 文件名三个模块实测、元数与类型契约(face 2 元组 / gesture 3 元组;`str` / `Path`)、**提交内 blob 无 CR**(voice 工作区是 CRLF 但提交是归一化,不是 churn)、`git status data/` 干净。

它也替 face 那处超 brief 的表头回填**给了结构性论证**(不是采信实现者的说法):**face 是三者里唯一"调用方在构造后覆盖 `log_file`"的**,而覆盖用的时间戳来自 `current_time` 而非构造时的 `datetime.now()`,所以跨过时间戳边界时**静默产出无表头文件、第一条数据行被当成表头**;gesture 无此问题是因为调用方在构造时就传 `log_file_path`(所以它的表头本来就落在最终路径上)—— **这个不对称由调用方形态决定,不是遗漏**。

**Task 3: complete (commits `b0b7ab6..6d4bfb5`, review clean)**

**Task 3 deferred minor(交最终 triage):**
1. face 的 **static 分支**没有测试(改动正确但无人守,mutation 只回退 static 列表不会被发现)。
2. **`session_id` 未经校验拼进文件名** —— `"../../etc/x"` 让 voice 构造函数抛未捕获 `FileNotFoundError`(崩溃非穿越);**今天够不到,T5 接前端 id 后变活**(→ 已写进 T5 派发词)。
3. **gesture 没有表头回填** —— 今天安全(调用方构造时给路径),但将来若有调用方像 face 那样覆盖 `log_file`,会重演刚修掉的 bug。
4. face 的表头回填**失败是静默的**(`_write_header` 吞异常、`log()` 仍返回 True),且 `exists()`→`open('w')` 有 TOCTOU 截断窗口。
5. 报告两处记账不准(`M9` 理由夸大;§1/§4.2 说 face 旧文件"本来就无表头"—— 实测每个旧文件都有表头,所以那处是**结构性**修复而非经验性;gesture 列数是 55 不是 56/57)。
6. `test_session_logging.py:52-63` 名字比断言强(它与 `:37` 合起来才严密)。
7. **T5 的验收要点名** `face_expression/api/app.py:71` 与 `gesture_analysis/api/app.py:81` 两行 —— 它们仍按秒命名,同一秒两个会话会共用一个文件。

**Task 4 实现完成**(`6377c70`,8 精确路径 + `rm -rf vosk-model-cn-0.22/`)—— 报 **DONE_WITH_CONCERNS**;**109 passed**(104 + 5 新);4 个变异各把对应测试打红;**并发丢更新测试在无锁时确定性丢失 7/8 次回答**;验收命令 `vosk|KaldiRecognizer` = **0**、`import voice_interaction` = **ok**;四处识别点统一 `await _transcribe_async(...)`(内部 `asyncio.to_thread`,因 vendored 客户端是 `asyncio.run`)。

**实现者的一处设计判断比 brief 更好:** 并发锁**放在 store 里**(按 `session_id` 的 get-or-create dict),而不是放在 app —— 理由是"保护所有调用方,且在不 import app 的前提下可测"。已采纳。

**控制器的四条裁定:**
- **Ruling M1-15(`question_index` 一律 0 基):** 实现者取 0 基(对齐 `save_log` 的 enumerate)。**钉死为 0 基**,写进 T5 派发词 —— 否则 T5 若取 1 基,三份日志的对齐会差一位。 — 若判断错,代价 = 按题号对齐的分析全体偏移 1。
- **Ruling M1-16(`session_id` 校验放中心,不放各端点):** 实现者上报客户端可控的 id 经 `recording_dir()` 可**越出录制根目录**(`"../../x"`),并问归属(T4 没双写)。**裁定:校验放在 `transcript_store.recording_dir()` 一处**(拒绝不合 `[A-Za-z0-9_-]+` 的 id),这样**所有调用方**——含将来的——都被保护,也顺带消掉 T3 那条"id 未校验就拼进文件名"的 minor;三个端点只需在文档里写明该契约。→ **带进 T5 派发词。** — 若判断错,代价 = 未来某个调用方绕过 store 直接拼路径时无保护。
- **Ruling M1-17(`requirements.txt` 必须改):** 它仍声明 `vosk>=0.3.45` 而**没有 `websockets`** —— 而 websockets 现在是运行必需(引擎走 vendored 客户端)。这是**真实缺陷**(照它重装会装错),虽属 root 文件也一并修。→ 带进 T5 派发词。 — 若判断错,代价 = 多改两行。
- **Ruling M1-18(`listen_for_speech` 的行为变更接受):** 它是**唯一无测试**的路径且只被 `examples/` 用(API 不走 pipeline):静音停机改用签名里从未生效的 `pause_threshold`(1.2 s),不再有逐块 partial 提示。**接受**,但要求 docstring 写明新行为,T7 的验收**不得依赖**它。 — 若判断错,代价 = 例子里的麦克风体验变了,需单独回看。

**deferred(记档):** ① `log_prosody({}, ...)` 让语音行的 prosody 列恒为 0,**与"真测出 0"不可分** —— M3 前需要一个"未测"标记(密度列是真的);② `/asr` 响应新增 `session_id`(纯加键)→ **T7 的断言不得要求逐字相等**;③ 根 `README.md` 仍提到 vosk(文档);④ 2.0 GB 磁盘要**在 Windows 侧 compact** 才真正回收。

**Task 4 任务审查已派**(opus,本计划风险最高的一步)。

**Task 4 审查结果**(opus):Spec ❌ / Task quality **Needs fixes**;**1 Critical**、2 Important、12 Minor。控制器自己核了 `109 passed` 属实。

### Critical —— 而且它打在我的裁决上

**`listen_for_speech` 的新静音规则是死代码:`pause_threshold` 永远不可能触发,每次录音都跑满 `timeout`。**
`silence_sec` 被**任何队列项**重置,而 `sd.RawInputStream` 每 0.5 s 就送一块(**不管有没有人说话**;静音也是数据,**全零的 `bytes` 是真值**),于是它永远到不了 1.2。审查者用**假设备实测**:`listen_for_speech(timeout=5, pause_threshold=1.2)` → **5.0 s 后才返回**,把 5.0 s 音频全喂给引擎。后果:①三个示例从"说完就停"变成**每答录满 30 s**;②新 docstring 与报告 §9.3 **断言的与事实相反**;③`pause_threshold` < ~0.5 s 会**句中截断**。而 **brief 的 Step 3 只要求换识别器,没要求重写停录规则**;这条路径**没有任何测试**,所以没有任何东西会红。

- **Ruling M1-18 更正(我错了):** 我上一轮裁"接受该行为变更"是**错的 —— 我采信了实现者的框架、没验证它是否真的工作**。更正为:**它是缺陷**。修复轮要求:① 把停录判定**抽成纯函数并测它**(这条路径至今不可测);② 用**真实信号能量**判静音,**能量下限进 `asr_config.json`** 带 `_provisional` 与依据;③ 让 `pause_threshold` 真正生效,测试覆盖"到阈值即停 / 短停顿不停 / 一路有声由 timeout 收";④ 同步更正文档与报告里那句相反的说法。 — 若判断错,代价 = 例子里的麦克风交互与实现者的设计意图不符(可回看)。

### Important

- **Ruling M1-16 修订(归属改判):** 客户端可控的 `session_id` 直接拼进路径,"原句绝不进仓库"这条不变量**可被客户端打穿** —— 审查者验证 `.../jingxin_recordings/../../jingxin/LEAKDIR` 解析到录制根之外,`curl '/asr?session_id=../../jingxin/x'` 能在任意可写位置建目录并写入含原句的 `transcript.json`。**我原派给 T5,现改判在 T4 修** —— 因为 **T4 正是把该参数变成客户端可控的那一步**。修法:`recording_dir()` 加 `^[A-Za-z0-9_-]{1,128}$` 守卫(放行 `"NONE"`)+ 两条测试。
- **Ruling M1-19(新增):三个端点的 `session_id` 必须"query 或表单"都接受。** 它现在只是 **query** 参数,而请求体是 multipart —— 前端若当表单字段提交会拿到 200 并**静默归到 `NONE`**,整个 M1 的"对上号"目标无声失效。**T7 的脚本要用表单字段提交来验证**。

### 已消解的 ⚠️ / 已确认

- **M1-15 确认(0 基正确):** 审查者对着 `assessment_pipeline.py:44` 与 `save_log` 的 `enumerate(qa_pairs)` 核过 —— `len(qa_pairs)` 在 `add_answer` 之前取的正是同一个下标。
- **T7 必须知道:** 打桩钩子是 `voice_interaction.asr.transcribe.asr_engine`,**不是** `app.asr_engine`(brief 的 Interfaces 行只满足了一半);`/asr` 响应新增了 `session_id` 键 → **T7 断言不得要求逐字相等**。
- `requirements.txt`(vosk 未删、websockets 未加)→ 已裁 M1-17,带进 T5。

### Minor(12 条,全部记档不修)

`_asr_meta()` 在 `api/app.py` 与 `transcript_store.py` **各存一份**(同形同源,暂无漂移风险);`_LOCKS` 从不清理(每会话一把锁,进程生命周期);**`log_prosody` 吞异常且返回值被丢掉** → CSV 写失败仍返回 `{"status":"success"}`、密度行**无声消失**(本任务正是第一个依赖该行落盘的调用方);prosody 列恒 0 与"真测出 0"不可分(→ M3 需要"未测"标记);客户端 id 写进 CSV 列而文件名是 `/interview/start` 上次建的那个(**id 与文件名不一致时行会落进错误文件**);`/research/start` 不发号 → 研究回答只能带客户端给的 id,与面试回答**混进同一个 transcript.json 且无法区分**;`tests/test_lazy_exports.py:18` 的 `HEAVY` 仍列 `"vosk"`(该元素现已永远空转);`code_data_supplement/` 那份陈旧副本里还有 11 处 vosk(未跟踪、不在导入路径、在验收 grep 范围之外);`/asr` 里的 `subprocess.run(ffmpeg, timeout=30)` **仍阻塞事件循环**(预先存在,但属同一类);两个 `answer_audio` 的 WAV 校验与空文本检查**重复**(本任务又各加了 3 行);`tests/test_transcript_store.py:75` 打桩了全局 `json.loads`(若有 xdist 会脆);一条低价值测试(守着一个没人提议的行为)。

**Task 4: fix 轮 1/5 已发**(Critical + 2 Important;Minor 不修)。**实现者的三点亮点也记了:** import 期崩溃真的消失、四处调用点收敛到一个 3 行 helper(属性不会漂移)、`except HTTPException: raise` 修掉了**预先存在**的"400 被重新包成 500"的 bug(否则新版"空识别 → 400"根本到不了)。

## Task 5 派发前准备(brief 已生成,55 行;**等 T4 修复落地再派**,避免两个实现者并行 + 避免套件被并发编辑弄红)

已核实三个落点:`face_expression/api/app.py:70-73`(按 `session_ts` 建路径并覆盖 `log_file`)、`:134`(`uuid4()` mint)、`gesture_analysis/api/app.py:80-82`(按 `session_ts` 建路径、构造时传 `log_file_path`)、`requirements.txt:24`(`vosk>=0.3.45`,且无 `websockets`)。

**派发词要带的四条修正:**

1. **无 id 一律写 `NONE`,不再 `uuid4()`。** 现状是**每个请求 mint 一个新 uuid** → 每帧一个新会话、日志文件爆炸。改为 `session_id = session_id or NONE_SESSION`(各自从**本包**的 `utils.logger` 取,不依赖 voice 包 —— Ruling M1-2)。
2. **文件名必须带 session id**:face → `face_au_log_{session_id}.csv`(保留"构造后覆盖 `log_file`"的既有模式;T3 已让 `log()` 在文件不存在时补表头,所以覆盖后仍会有表头);gesture → `gesture_emotion_log_{session_id}.csv`,经构造时的 `log_file_path` 传入(其构造时即写表头,无需回填)。
   → **这两行正是 T3 审查点名"截至 T5 之前,同一秒的两个会话会共用一个文件"的地方**,验收要点名它们。
3. **`session_id` 必须在边界校验(与 Ruling M1-16 同一把尺)** —— face/gesture **不写 transcript**,所以中心守卫(在 `transcript_store`)保护不到它们:客户端传 `../../x` 会让 `face_au_log_../../x.csv` 越出日志目录。要求:**不合 `^[A-Za-z0-9_-]{1,128}$` 且不等于 `NONE` 的 id → 400**,并给出清晰消息(而不是静默改写成 NONE —— 畸形输入不该被悄悄吞掉,这是本项目一贯的立场)。正则与 `NONE` 同处本包,不跨包共享。
4. **`requirements.txt` 修成事实**(Ruling M1-17):删 `vosk>=0.3.45`,加 `websockets`(当前环境 17.0.1);顺带核一眼 voice 模块真实依赖(`sounddevice`/`librosa` 等在不在),**只做与本次替换直接相关的改动**。

**测试要求:** brief 那条是文本级断言("不得再有 `uuid4()`");**另加行为测试** —— 沿用 T3 已验证可行的办法(把本包 logger 模块的 `LOGS_DIR` monkeypatch 到 `tmp_path`,import 成本实测 0.93s/0.37s),断言:无 id 时文件名含 `NONE` 且**两次调用落同一个文件**、带 id 时文件名含该 id、畸形 id 被拒(400)。

**T7 必须知道(累积):** 打桩钩子是 `voice_interaction.asr.transcribe.asr_engine`(不是 `app.asr_engine`);`/asr` 响应新增了 `session_id` 键 → 断言不得要求逐字相等;`session_id` 在 voice 三端点上**query 与表单都要能收**(Ruling M1-19),T7 的脚本用**表单字段**提交来验证这一条。

- **Ruling M1-20(破例:两个实现者并行):** 使用者连续两次要求"T5 该派就派",而 T4 的**修复者仍在跑**(实现者)。**按规矩这是不被允许的**("绝不并行派实现者")。控制器裁量:**本次破例,理由是零文件重叠** —— T4 修复者只动 `voice_interaction/`(pipeline / transcript_store / api/app),T5 只动 `face_expression/api/app.py`、`gesture_analysis/api/app.py`、`requirements.txt` 与 `tests/`;冲突的全部理由(同文件互相覆盖)在此不成立。**残余风险是"套件被并发编辑弄红"** —— 已写进 T5 派发词:看到 voice 相关测试红,**归因给 T4 的修复者,不要去追、不要顺手修**。 — 若判断错,代价 = T5 实现者花时间做归因(无数据损失);反向代价 = 空等 T4 修复(使用者已两次要求推进)。

**Task 4 修复轮结果**(`a4e14b0` + `7ffd9fd`;**130 passed**,连跑 4 次全绿)—— 报 DONE_WITH_CONCERNS。

- **Critical 自证**:同一复现,改前 **2.91 s / 缓冲 3.00 s**(跑满 timeout),改后 **2.01 s / 2.50 s**(在 1.5 s 末尾静音处停)。停录判定改为**真实信号能量**,能量下限 `silence_energy_floor` 进 `asr_config.json`。
- Important 1:守卫落在 store(`validate_session_id` + `recording_dir` 拦),端点在更前面校验并回 400。
- Important 2:`_resolve_session_id` 让 id 从 **query 或表单**都认。
- 那条 Minor 的作答(裁定记档):**建议"服务端为权威,id 与当前会话不匹配回 409"**。
- **方法学发现(全项目适用,已记为规则):** 变异脚本"写入→跑→`git checkout` 还原"在**同一秒内**会留下过期 `__pycache__`,**跑的是变异字节码、曾造成假红**;改为每步清缓存 + 还原后必须绿。**今后任何变异验证都必须清缓存。**
- 未解决/移交 T7:`python-multipart` 未装 → **本环境所有 `File(...)` 端点起不来**;`silence_energy_floor=0.01` 是**定义性下限、未本机实测**(现场增益低则"跑满 timeout"复发,**第一个要调的数是它**);修后仍把约 `pause_threshold`(2.5 s)的尾随静音喂给引擎(切干净需 VAD/M3)。

**Task 5 实现完成**(`b50e481`,4 路径;**130 passed**)—— 它的新测试**抓到 brief 未提的真缺陷**:gesture 只传 `log_file_path`、没传 `session_id` → CSV 首列 `NONE` 而文件名是真 id。`requirements.txt` 已按修正 4 修(vosk 删、`websockets>=12.0` 加)。**它报的一个前提我要 T5 审查者重点判:** 两个 app 模块在本环境**无法 import**(face 缺 `python-multipart`;gesture 被 mediapipe 1.0 移除 `mp.solutions` 打死),测试用"shapes-only shims" —— **这可能变成"在测桩"**,已点名要求判定。

**控制器自己核实的环境事实:**
| 检查 | 结果 |
|---|---|
| face 启动 | ❌ 缺 `python-multipart`(所有 `File(...)` 端点起不来)|
| gesture 启动 | ❌ mediapipe **1.0.0** 已无 `mp.solutions` |
| voice 启动 | ✅ |

- **Ruling M1-21(T7 验收范围收窄为 voice + face):** gesture 的启动问题**预先存在**(mediapipe 大版本 API 移除),不是 M1 引入;其 `session_id` 改动由测试覆盖。→ **T7 只跑 voice + face**;gesture 的启动修复另立跟进项。**T7 之前必须装 `python-multipart`(由使用者执行,不代装)。** — 若判断错,代价 = 验收覆盖面少一路(可后续补一场)。
- **Ruling M1-22(归属不一致的契约,记档不实现):** 采用"**服务端为权威,id 与当前会话不匹配回 409**"作为约定;M1 单会话设计下不会发生,T7 也跑单会话 → **留到多会话支持时实现**,写进 `docs/下一步.md`。 — 若判断错,代价 = 客户端传错 id 时行落进错误文件且无声(现状)。
- **Ruling M1-23(变异验证纪律):** 见上文方法学发现 —— **任何变异验证必须在每步之间清 `__pycache__`**,否则可能跑的是过期字节码(曾致假红)。 — 若判断错,代价 = 无(纯纪律);不记则未来再次把污染当证据。

**两个审查并行在跑:** T4 范围复审(opus;已把"范围混入兄弟任务提交"更正为只含 T4 两提交的 diff)+ T5 任务审查(sonnet,重点判"是否在测桩")。

**Task 4 范围复审结果**:**全部 ADDRESSED,无新增 Critical/Important**。

复审者的独立验证(不照信报告):**自己造了假设备复现**(2 个说话块 + 尾随静音,`timeout=3`、`pause_threshold=1.2`)→ `elapsed=2.01s buffered=2.50s chunks=5`,与实现者声称**逐字节一致**;并用**比实现者更广的变异**验证停录机制(改回"任何块=仍在说话" → 4 红;"非空=说话" → 3 红;去掉"还没开口"前置 → 4 红;写死能量下限 → 1 红),其中前两种下**端到端测试会跑满 15 s timeout** —— 即它们量的正是要量的维度。守卫那条它做了**端点层 22/22**(拒 `../../jingxin/LEAKDIR`/`..`/`a/../../b`/`.`/`""`/含空格/129 字符;`NONE` 与铸造型通过;**query 与表单两条路都 400**);query/表单双认它用 stubbed `Request` 验证是真第二来源(表单-only 通过、query 优先、空→NONE、非法→400)。

**Task 4: complete (commits `6377c70..7ffd9fd`, review clean)**

**Task 4 遗留(记档,交最终 triage):** ① `tests/test_speech_recognition_pipeline.py:43` **硬编码 `FLOOR = 0.01`**,与配置值重复 —— 若将来配置下限高于该 fixture 的语音 RMS,单元测试仍绿而生产会把真实说话当静音;**非阻塞但是配置耦合的味道**;② 报告里 M5–M7 的变异**计数不可复现**(方向一致,都是 RED);③ `130 passed` 没有贴原始输出块(复审者用聚焦运行确认了 27 passed / e2e 5 green);④ `assert elapsed < 5.0` **仍是墙钟断言**(本项目被它咬过两次;此处余量 ~100×,真实信号由块数断言承担);⑤ `_fake_mic` 打桩的是**进程全局** `queue.Queue`;⑥ **T7 前置:`pip install python-multipart`(使用者执行)**。

**T5 审查结果**:Spec ✅ / Task quality **Approved**;0 Critical、1 条件性 Important(已裁定要修:face/gesture 的 id 只从 query 取,与 voice 的"query/表单双认"不对称 → 按 voice 照搬)、1 条 Minor 折入、其余记档。审查者的关键判定:**T5 的测试跑的是真代码、不是在测桩**(两个 shim 只装在 `python_multipart.__version__` 与 `mp.solutions` 替身上;它还用 `git archive` 导旧树做了 10 红对照 + 只删一行 `session_id=session_id` 的承重验证)。**T5: fix 轮 1/5 已发。**

**T5 遗留(记档):** face 的覆盖 `log_file` 现为**空操作**;空串在端点视为"无 id"、辅助函数视为非法(两层口径不同);无"同一秒两个会话"的直接测试;400 文案里的"1–128"与正则分开硬编码;`/reset` 与 `/session/{id}/summary` 收未校验 id(只作字典键)。

- **Ruling M1-26(转给 T6):** `NONE` 桶(`face_au_log_NONE.csv`)会**永久追加**、跨天累积,而所有无 id 客户端**共享一个 pipeline**。→ T6 要**查清报告侧选日志文件的方式**(按文件名新鲜度?按时间戳?),并**报告**跨会话混行风险;**只做小而明确的修复,否则只报告**。 — 若判断错,代价 = 报告把不同场次的行混进同一聚合。
- **Ruling M1-27(`requirements-full.txt`):** 它仍声明 vosk(另一份 requirements)。→ 并入 T6 的一行修改(vosk 删、websockets 加),并在 T6 的范围说明里显式放行该 root 文件。

**Task 5 fix 轮 1/5 结果**(`5ca37bc`,父 `b50e481`;测试文件 10 → 17 条;**全量 137 passed**):

- Important 照 voice 形状照搬:两模块各加 `_resolve_session_id(request, session_id)` + 端点加 `request: Request`;**校验在 query 与表单字段合并之后**;`except` 兜底**承重**(本环境没装 python-multipart,`form()` 直接抛,少了它每个无 query id 的请求都会 500)。
- 折入的 Minor:`test_app_modules_never_pull_in_the_voice_package` 用 **ast 源码级**检查(收所有 import 含延迟导入 —— 刻意不用裸文本,因为两个 app 的注释里就有 `voice_interaction/...` 字样,文本匹配会误判)+ **子进程 `sys.modules` 检查**(同进程里 voice 早被别的测试导入,恒真)。**它生来即绿,是回归闸,实现者明说没假装它先红。**
- **区分度实验(提交状态上的最小破坏,每次还原、收尾 `git diff` 为空):** 去掉表单回退 → **恰 4 红**(2 条表单字段 + 2 条 malformed 的子例);`except Exception` → `except StopIteration` → **恰 2 红**;face 注解改 `dict` → **恰 1 条形状级红、行为全绿**(证明那正是行为测试看不见的维度);gesture 插一行延迟 `import voice_interaction.asr.session` → **恰 1 红**,红在 ast 断言上。
- **实现者主动交代的 TDD 妥协:** 本轮升级了行为测试的驱动方式后,生产代码加 `request` 之前它们一律 `TypeError`(红不在要守的点上),故**先实现、再以破坏实验证明区分度** —— 与审查者验证 gesture 承重性同法。

**T7 需补的端到端证据(累积):** 用**表单字段**给 face/gesture 各提交一次,断言回显 id 与落盘文件名都带它(真实 multipart POST 在本环境跑不起来 —— 实现者没有为此装包或改环境,正确)。

**Task 5 范围复审结果**:**全部 ADDRESSED,无新增 Critical/Important**。

复审者在 `/tmp/rev5`(树副本,**从未动仓库**)把四个区分度实验**全部复现**,数字逐条吻合:去掉表单回退 → 4 红 13 绿;`except Exception` 收窄 → 2 红 15 绿;face 注解改 `dict` → 1 红(只形状级);gesture 插延迟 voice import → 1 红(红在 AST 断言)。它并核了算术自洽:文件 17 条 = 10 + 7;`--collect-only` = **137 = 130 + 7**。它还验证了一个关键前提:**"不会双重消费表单"在装着的版本上成立**(fastapi 0.141.1 / starlette 1.4.1 —— `get_request_handler` 已在依赖解析阶段 `await request.form()` 并缓存 `request._form`,端点里的调用读的是缓存)—— **这正是 T7 端到端所依赖的**。

**Task 5: complete (commits `a4e14b0..5ca37bc`, review clean)**

**Task 5 遗留(记档,交最终 triage):**
1. **守护残余缺口:** 在 `face_expression/utils/logger.py`(AST 检查**不扫**的文件)里写**延迟** import 能同时绕过两道检查 —— 而那个文件正是 face 取 `NONE_SESSION` 的地方,**最可能的未来重引入点**。一行扩展 AST 扫描即可闭合。
2. `tests/test_analyze_session_fallback.py:442` 断言 `params["request"].annotation is Request` —— 若任一 app 加 `from __future__ import annotations` 会变成**假红**(字符串注解);用 `typing.get_type_hints` 更稳。失败形态是假阳性、不是假绿。
3. `except Exception` 也会吞 starlette 对畸形 multipart 体抛的 `HTTPException(400)`(此处实际够不到:FastAPI 在端点前就解析表单;且与已裁定可接受的 voice 端点同形)。
4. **`requirements.txt` 仍缺 `python-multipart`** → 照它重装连 `face_expression/api/app.py` 都 import 不了(FastAPI 定义 `File(...)` 路由需要它)→ **已捎给正在跑的 T6 一并补上**。
5. 报告用散文报计数(17 / 137)而无粘贴输出(数字已被独立佐证)。

**Task 6 实现完成**(`c9570d8`,7 文件 +238/−71;**142 passed** = 137 + 5 新;禁止词扫描仍绿)—— 报 DONE_WITH_CONCERNS。

**Correction 6 的调查结论(裁定 M1-26 的答案,很有价值):** 报告侧选日志**按文件名里嵌入的时间戳正则**(`^<模态>_<描述>_log_YYYYMMDD_HHMMSS\.csv$`,每模态取最新),**不按 mtime、不看 `session_id` 列**。故 `face_au_log_NONE.csv` / `gesture_emotion_log_NONE.csv` **匹配不上 → 不可见**(实测:即使放入更晚的 NONE 语音文件,加载器仍选中会话文件)。→ **`NONE` 桶跨天累积是磁盘留存问题,不是报告正确性问题**;**未改代码**(明确的修法都在报告层之外,报告侧那些选项属会话匹配的策略决定)。

**但报告暴露两条残余混行路径:**
- (a) 语音会话文件里可以有**列写着 `NONE`** 的行(logger 的 `session_id` 随请求变、文件名固定),而报告**从不检查该列**;
- (b) **三个模态各自独立取"最新那个文件"** → 批量报告**仍按设计("方案 A")把不同场次拼在一起** —— 即"5 月的脸 + 2 月的语音"在**报告装配**这一层依然存在,**M1 只是把 id 写进了行、还没让装配用它**。

**⚠️ 一条直接影响 T7 验收门的发现(待我核实后裁定):** 实现者报 **G3 的 `_default_n_valid=10`** —— 若 `_n_rows` 就是语音日志的行数(= 回答段数),那么**只有 2 段回答的会话会被 G3 判「有效样本不足」**,我计划里写的"≥2 段回答"**不足以让该槽过门**。

**其余 concern:** ① `basis_kind` 保持 `"definitional"`,而 `10.0` 是**选定的参考值**(依据文本已明说) —— 接受,已文档化;② 名称沿用控制器裁定的「连接词密度」(旧文档 §3.4 的「话语标记使用率」若改回需动 ~6 处断言);③ 删分支 C 同时删掉了 `text_avg_length` 指标的唯一产出方 —— 但它**本来就不可匹配**(没有任何 logger 写 `text` 列),无功能损失;④ **真实扁平化键是双前缀**(`voice_research_research_connective_density_mean`),需求里写的是单前缀;fixture 用控制器拼法、新端到端测试钉真实键(子串匹配使两者都通);⑤ `get_live_data` **不携带** connective_density → **实时报告会丢这个指标**(批量路径不受影响);⑥ `requirements-full.txt` 此前**未被跟踪**,因本轮要改它而**新入 git**;⑦ `python-multipart` 已按我的补充加进两份 requirements。

## ⚠️ 控制器核实:一个必须修的设计错配(直接决定 T7 能否达成)

T6 报 `_default_n_valid = 10`;控制器**逐项核实**:
- `evidence_thresholds.json`:`_default_n_valid` = **10**,`thresholds` 里**没有 `density` 条目** → 「连接词密度」继承默认 10;
- `feature_engine.py:68-70`:`_n_rows = float(len(self.data[key]))` = **语音日志行数 = 回答段数**;
- **面试题库每场只有 8 题**(`assessment_pipeline.py:74` 与 `:330`,各 8 条,AST 计数确认)。

⇒ **答满全场也只有 8 < 10 → 该槽永远被 G3 拦成「有效样本不足」,验收门那条核心判据("槽过门出分")在当前门槛下不可达。** 这是**单位错配**:默认 10 是当初为**帧级列**(n_rows ≈ 上千)设的,被一个**每回答一个样本**的指标继承了 —— 不是实现错误,是设计遗留。

- **Ruling M1-28(并入 T6 修复轮):** 在 `evidence_thresholds.json` 为 **`density` 族登记自己的 n_valid 阈值**:值 **5**,带 `_provisional: true`,依据写明"该指标每段回答一个样本;默认 10 为帧级列所设;一场面试仅 8 题,10 段不可达;M5 标定前暂用 5"。**理由:** 门槛本就该按指标单位登记(spec §5.1),这是把它修正到正确的单位,不是放宽诚实门 —— 一个永不达标的门槛等价于该指标永久不可用,而报告已经会诚实地按门槛拦。 — 若判断错,代价 = 该指标在只有 5–9 段的会话上出值(而 M5 标定会重新定这个数);反向代价 = M1 的核心验收判据永不可达。
- **T7 的验收口径随之修正:** 从「**≥2 段回答**」改为「**答满 ≥5 段**」(题库 8 题,建议答满 8 题;G2 仍要求会话内有变异,≥2 段即可满足)。

**并发说明:** T6 已进入任务审查;M1-28 会与审查结论**一并**作为 T6 的修复轮发出(避免在审查进行中改动被审的提交)。

**Task 6 审查结果**:Spec ✅ / Task quality **Approved**;**0 Critical、2 Important**(两条都**不是该 diff 的缺陷**,而是"调查没挖到底"与"里程碑级门槛错配")、多条 Minor。审查者独立复现了实现者的全部变异计数(在 `git archive` 出的临时树里,仓库未被触碰),并独立复现了 pandas 3 那条(字符串列的 `dtype == object` 为 False → 被删的兜底分支确实是死代码)。

### Important 1 —— **M1 之后基于文件的报告路径加载不到任何文件**(会让 T7 白跑)

`report_frontend/data_loader.py:48` 的选取正则要求文件名以**纯时间戳**结尾(`^(face|gesture|interview|research)_(.+?)_log_(\d{8})_(\d{6})\.csv$`),而 T3 起文件名是 `..._log_<session_id>.csv`(`YYYYMMDD_HHMMSS_xxxx`,逐字来自 `session.py:10-13`)。审查者**用真加载器实测**:临时目录里放齐三份 M1 命名日志 → 「未找到相关日志文件」、`LOADED MODALITIES: []`;唯一能匹配的纯时间戳形态**已无任何产出方**。

- **Ruling M1-29(必须修,并入 T6 修复轮):** 让选取**同时接受**旧形态与 M1 形态,**并显式排除 `NONE` 形态** —— ⚠️ `..._log_NONE_YYYYMMDD_HHMMSS.csv` 的尾部**恰好也是时间戳**,若把正则放宽成"时间戳可选后缀",**NONE 桶会重新可见并跨天混行**(上一轮调查结论正依赖它不可见)。要求:四形态的临时目录测试(旧形态取到、M1 形态取到、NONE 拒绝、模态齐全时不再为空),并把"三份 M1 日志 → 空"的复现**反过来**作为直接证据。 — 若判断错,代价 = NONE 桶复活并混行(这正是要防的);或放宽过头把别的文件也吞进来。
- **这是控制器(我的)计划缺陷:** T3 改文件名时,**我没把 `data_loader.py` 的选取正则列进改动面** —— 而两者是同一件事的两半。已记。

### Important 2 —— 门槛错配(已裁 M1-28,审查者独立确认)

审查者独立跑出:5 段回答时 `score=None`、缺口含「连接词密度: 有效样本不足」,并指出 T7 Step 3/4 的措辞与之一致性冲突。→ **M1-28 成立**(density 族单独登记 n_valid = 5)。

### Minor(记档不修)

`research_mapper.py:175-176` 的"仍有真实消费者(`face_eye_contact_ratio`)"是**合成 fixture**(没有生产键"有 `_std` 却无 `_mean`");fixture 用单前缀拼法(**松,但不空** —— 子串匹配使然,且已有真实键的端到端测试钉住);`basis_kind: "definitional"` 的分类拉伸(brief 字面要求,已披露);显示名「连接词密度」与 `docs/下一步.md:59`、心理测量学审查 `:250` 的「话语标记使用率」冲突 → **收尾时统一到「连接词密度」**(我在最终文档整理里改);`text_avg_length` 无产出方(删分支后仍不可达,非回归);实时路径该槽恒空(旧的"密度"在实时路径本就是**假值** —— `data_loader` 合成 `answer` 列,删掉是修正);`requirements-full.txt` 整份新入 git(43 行,此前未跟踪)。

**Task 6: fix 轮 1/5 已发**(M1-29 + M1-28)。

**Task 6 fix 轮 1/5 结果**(`ccddb82`,5 文件 +197/−7;**150 passed** = 142 + 8):

- **I1(M1 命名日志加载不到):** 先复现(三份 M1 命名日志 → `LOADED MODALITIES: []`),再修:正则把 M1 形态做成**必需时间戳 + 可选 `_[0-9a-f]{4}` 后缀**;NONE **结构上**排除(**`_log_` 后面那段必须是数字** —— 实现者**刻意没加**那句够不到的 `if "NONE" in name`)。两个变异钉住:加 `(?:NONE_)?` → 2 红;**把时间戳改成可选 → 同样那 2 红**(正是我警告的"放宽过头")。新增 `tests/test_log_file_selection.py`(7 例:反向复现、四形态选择矩阵、**M1 形态胜过更新的 NONE 文件**、会话号上报、四种 NONE 形态逐一拒绝)。
- **I2(M1-28):** `thresholds.density = 5` + 紧邻的 `_thresholds_basis.density`(依据写明单位错配、题库 8 题、10 段不可达、5 为暂定)。测试**先钉消费路径**(`_threshold_for(<density key>) == 5`,红法是 `10 == 5` → `KeyError: 'density'` → 值),再钉登记表,最后跑一个 **8 段回答的端到端**必须过 G3。变异:删条目 → 1 红。
- **已按要求记录不改:** `face_energy`/`face_energy_std` 是**合成 fixture**(没有生产键"有 `_std` 却无 `_mean`"),说明只加在测试文件里;`…_rejected_one_by_one` 那条**改前也是绿的**(它守的是**将来的**放宽,不是这次修复)—— 实现者把它写进 docstring 与报告,不让它冒充红过。
- **残余(已文档化,未改):** 同一模态若**同一秒**同时存在旧形态与 M1 形态文件,`max(timestamp_val)` 打平 → 由 rglob 顺序决定胜者;只有手工混放目录才会出现,"哪个会话该赢"是策略决定。

**Task 6 范围复审已派**(`c9570d8..ccddb82`,1 提交 19 KB)。

## T7 验收清单(控制器据全部累积发现改写;等 T6 复审判 clean 后交给使用者)

**前置**
1. `~/miniconda3/envs/jingxin/bin/python -m pip install python-multipart`(使用者执行 —— 没有它 voice/face 的 `File(...)` 端点起不来)
2. 起服务:voice(:8001)+ face(:8000)。**gesture 起不来**(mediapipe 1.0 无 `mp.solutions`,预先存在,已裁 M1-21 只跑 voice+face)

**录制**
3. `POST /interview/start` → 取 `session_id`
4. 答 **≥5 题(题库 8 题,建议答满 8 题)**(G3 新门槛 = 5;G2 需要会话内有变异,≥2 段即满足)。**其中至少一次把 `session_id` 作为表单字段提交**(验证 M1-19;voice 与 face 各验一次)
5. 顺带让 face 收几帧(表单字段带 id),验证 face 的文件名与首列

**核对(四条)**
- a. `~/shared/jingxin_recordings/{sid}/` 下有 `session.json` 与 `transcript.json`,后者含**逐字时间戳**
- b. `data/logs/` 下 face/voice 两份日志**文件名含 sid、首列是 sid**
- c. **仓库内无原句**:从 transcript 里挑一句独特短语,`grep -rn "<短语>" ~/jingxin` → 0 命中
- d. `~/miniconda3/envs/jingxin/bin/python -m report_frontend.report_generator` → 报告里「**话语结构特征**」维度出现「**连接词密度**」行并**过门出分**(带本场区间与有效样本量),而不是「未采集到对应数据」;其余维度仍应为「证据不足」

**预期形态(供对照):** 覆盖计数(如 `1 / 20` 或按比例)、该维显示「过门指标 1/4;置信度:低。(该维度目前无独立效标,仅供行为描述)」、表格列 `指标 / 原始值 / 本场会话内观测区间 / 有效样本量`、**全篇无分数与评级**。

**Task 6 范围复审结果**:**全部 ADDRESSED,无新增 Critical/Important**。

复审者独立复跑的证据:反向复现(把新测试对着 `c9570d8` 的旧加载器跑 → **3 红 4 绿**:`test_m1_session_named_logs_are_loadable`、`…_legacy_and_m1_forms_but_never_none`、`…reported_session_is_the_filename_session`);**放宽方式的穷举**(真实 pytest、临时副本):`(?:NONE_)?` → 2 红;时间戳前放通配 → **4 红**;把时间戳整个去掉 → **7 红** —— **每一种真能让 NONE 命中的放宽都被抓到**;真实目录对照:`data/logs/` 里 66 个 CSV,新旧正则都匹配 62 个,**新正则拒绝全部 4 个真实 NONE 文件**;I2 的消费路径**确实先钉**(真实双前缀键 → `_threshold_for == 5`,删掉登记 → `assert 10 == 5` 一条红)。

**Task 6: complete (commits `5ca37bc..ccddb82`, review clean)** → **M1 的六个代码任务(T1–T6)全部完成**,只余 T7(使用者亲验)。

**Task 6 遗留(记档,交最终 triage):**
1. 报告的变异计数**低报**(真实是 4 红;按字面读 7 绿) —— 方向是安全的:**守卫比声称的更强,没有静默 NONE 洞**。
2. `test_reported_session_is_the_filename_session` 的第二条断言**近乎恒真**(`session_id` 只由正则的日期/时间组构造,字面 `NONE` 永远到不了汇总)。
3. 三种 bare-NONE 参数化用例里有两个作为**守卫**是冗余的(它们只是记录了三种真实产出形态,留着便宜)。
4. **客户端自定义的 session id 仍然不可见**:face/gesture 接受 `[A-Za-z0-9_-]{1,128}`,所以 `face_au_log_my-run-1.csv` **永远不会被加载** —— 与刚修掉的 bug **同一失效形态**,只是范围缩到"非铸造 id"。预先存在、不在该 finding 的要求范围内,但**T7 必须避开**:只能用 `/interview/start` 铸造的 id。
5. 一处注释的因果措辞不准(该形态其实来自 logger 的 **falsy-id** 分支,而非"解析成字面量 NONE")。

**⚠️ 给 T7 的一条实操警告(控制器补充):** 8 段回答**必须是内容不同的**。若用同一段音频重复提交,每行密度相同 → `_std = 0` → **G2 会(按设计)判「本次会话内无变化」** 而拦下该槽 —— 别把这条路走成自己的坑。

## 补记:漏账的一条裁决

- **Ruling M1-14(下达于 T4 派发词,当时未落账,2026-09-24 补记):** brief 让把转写接缝 `_transcribe` + 模块级 `asr_engine` 放进 `voice_interaction/api/app.py`,而它的测试要 `importlib.import_module("voice_interaction.api.app")` —— **那会在 import 时构造 TTS 与评估管线**。裁定:**接缝放进轻模块 `voice_interaction/asr/transcribe.py`**,测试只 import 它。 — 若判断错,代价 = 多一个小文件;反向代价 = 那条测试拖着重依赖跑,且可能因无音频设备而失败。(已由 T4 落地:`api/app.py:33` 以 `from voice_interaction.asr.transcribe import transcribe as _transcribe` 引入。)

**裁决编号说明:** M1-1 … M1-29 中,**M1-24、M1-25 未使用**(编号在写 T5/T6 派发前被 M1-26/M1-27 占用,故跳号)。清单以本账本中实际出现者为准。

## 最终全分支审查(opus,`1e53c131..d70cf46`,18 提交)—— 判 **With fixes**

审查者自己跑了:整套 `150 passed` 复核、**两处 loader→engine→mapper 复现**、一处产物选择复现、以及 vosk/文本落盘路径的 grep 扫查。

### C1(Critical,必须修)—— **报告会加载一个"只有表头"的旧产物,而不是 M1 的会话日志 → T7 的核心判据不可能通过**

`assessment_pipeline.save_log()` 在**每次** `/interview/answer_audio` 都写 `data/logs/interview/interview_emotion_log_<时间戳>.csv`,而**经 API 走时它永远是空表头**(只有两个 `examples/` 脚本才填 prosody);它的时间戳是**回答时刻**,必然晚于会话开始 → 在"每模态取最新"里**永远胜出** → 加载器丢掉空帧 → `voice_interview` 根本到不了 feature_engine → 「连接词密度」渲染成「未采集到对应数据」。

**两个方向都复现过:** 带上该产物 → `MODALITIES LOADED: {'face'}`;去掉 → `连接词密度 3.6667 → 0.37, n_valid 6, interval [2.59, 4.75], coverage 1/20`(正是 T7 期望的形态)。**根因预先存在**(同样的"遮蔽"形态早就有),但 T6 那轮只修了正则那一半;T6 复审的反向复现把会话日志**单独**放进临时目录,所以这个碰撞**看不见** —— **这是唯一一个任务级审查结构上看不到的集成缝。** 注意:该产物**每次调用都会重建**,删文件不是修法。

### Important(全部并入修复波次)

- **I1 有效样本量夸大了它所标注的值**:`n_valid` 对所有指标都取模态行数;密度列里"回答过短 → 空值 → 被 dropna"的行**照样计入**。复现:20 行只有 3 行有值 → 报告印 **有效样本量 20** 而均值只来自 3(且一个几乎全空的日志能靠"不携带任何值的行"越过 G3 的 5)。→ 改为优先取该指标自己的 `<base>_sample_size`。**这正是 M1 要修的诚实轴,而且在 M1 自己造的那个槽上。**
- **I2 `/asr` 把识别出的句子写进了仓库**:`logger.info(f"识别结果: '{text}'")`,`logging_config.py` 把 handler 装到**仓库内**的 `data/logs/jingxin.log`(INFO,10 MB × 5)。→ M1 刚搬出仓库的文本又落回来了,而 T7 的核对正是 `grep -rn "<短语>" ~/jingxin → 0`。→ 改为记录长度/哈希,并全包扫一遍其他文本落盘点。
- **I3 文档之间就验收门互相矛盾**:spec §10/§9.4 与计划 T7 仍写「≥2 段」,而门槛已是 **5**(M1-28)。→ 改文档,并把门槛理由移到 spec §6.5 的 `min_chars_for_density` 旁边。
- **I4 报告从不说明它描述的是哪一场会话**:表头只有生成时间,三个模态各自按文件名时间戳选 —— gesture 服务已死(正是验收的配置)时,报告会把**本场的 voice+face 与上一场的 gesture 配在一起**,而读者毫无提示。裁定 M1-22 已把**选择策略**推给 M2,但**披露**只有一行:把三个被选中的 session id 打进报告表头(并标明是否一致)。

### 审查者对"三条核心主张是否有测试守得住"的结论

- **(i) 三份日志按 id 可 join —— 部分**:首列/文件名/NONE 常量/四形态都有测试,但**"join 本身"没有测试**(spec §10 自己列了这条契约测试),且**没有任何消费者会去读那一列**;所以套件是绿的,而 (i) 目前只等于"id 被写下来了"。
- **(ii) 原句不进仓库 —— 大体成立,但有 I2 这个洞**:生产默认路径的不变量有测试且经变异证明,穿越守卫有端点级 22/22;但**没有任何测试看着 logging 那条路**。
- **(iii) 报告里的密度是测出来的 —— 在它测的那一层里确实强**(静态扫描 + 行为断言 + 真实列名端到端 + 量程 + 门槛消费路径都钉住了);**结构性盲点是:所有密度测试都从 `LogDataLoader` 之后开始** —— 而那正是 C1 所在之处。
- **共因**:没有任何测试 import `voice_interaction.api.app`(计划的有意决定),所以四处识别点、`_resolve_session_id`、`logger` 交换、`answer_audio` 里的密度写入**在本环境完全没有自动化覆盖** —— 这正是 C1 与 I2 能活过六轮任务审查的原因。验收门是它们唯一的验证。

### triage(审查者给的)

- **合并前**:C1、I1、I3;外加 `load_markers()` 交出的可变缓存、`log_prosody` 吞异常且返回值被丢(这两条标"验收前")。
- **可现在就做(便宜、无策略成分)**:I4 的表头披露。
- **M2/M3**:选择策略本身、409 契约、gesture 启动、NONE 轮转、空串口径、`text_avg_length` 无产出方、`_LOCKS` 不清理、ffmpeg 阻塞循环、face static 分支无测试、AST 守卫不扫 `face_expression/utils/logger.py`。
- **人类裁量**:`requirements-full.txt` 新入 git;名称统一(旧文档的「话语标记使用率」)。
- **建议 #5(已并入 C1 的修法)**:补一条**从 `LogDataLoader` 开始的真实 `data/logs/` 端到端测试**(会话日志 + 遮蔽产物 + NONE + 旧形态)—— "这条分支的四次大意外里有三次会被它抓到"。

### 审查者的验收判据(我据此更新 T7 清单)

(a) 跑报告时**加载器打印的是会话日志**(`[INTERVIEW] 选中最新文件：interview/interview_emotion_log_<SID>.csv`),**不是** `data/logs/interview/…` 那个产物;(b)「话语结构特征」渲染出「连接词密度」+ 原始值/本场区间/有效样本量并**过门**(覆盖 1/20、无分数无评级);(c) `grep -rn "<transcript 里的独特短语>" ~/jingxin` → **0**(而且若也碰过 `/asr`,那边也要 0);(d) 三份日志首列都等于 `$SID`。

**最终修复波次已派(一次派发,含 C1+I1+I2+I3+I4 + 两条折叠的 deferred + 空串口径)**。

## 修复波次的落地与收尾(2026-09-24 晚)

派发后先落了 3 条(各带新测试文件):C1 → `6adda51`(+`tests/test_artifact_shadowing.py`,含建议 #5
那条真实 `data/logs/` 端到端);I1 → `4106bab`(+`test_sample_size_honesty.py`);I2 → `234f817`
(+`test_asr_log_privacy.py`)。**I3 只改了一半且未提交,I4 与两条 deferred 未做** —— 收尾如下。

**使用者裁定:** ① D2(`log_prosody` 写失败)**报错**,不静默;② face 与 gesture **一起修**;
③ M2 取**最小闭环**(按 `session_id` 选 + 披露 + 给 `refresh_manifest` 真消费者)。

| 项 | 落地 | 测试 | 反向复现 |
|---|---|---|---|
| **I3** | 文档口径收尾:`specs/…design.md` §10 与 `plans/…m1-asr-session-id.md` T7 的「≥2 段」→「≥5 段」,并把 G2/G3 两个门槛分开写清。**账本里那几处历史记录不改**(那是 M1-28 裁决与更正的痕迹,改了等于篡改账本) | 无(docs) | grep 残留只剩"解释性"与账本历史 |
| **I4** | `data_loader` 交出 `selected_sessions`(与 `data_frames` 同步重置,只记**真正读出来**的模态);`report_generator.sources_disclosure()` 逐模态点名 + 标明是否同场;两个 `generate_report*` 都传进表头。实时路径的来源=请求的那个 id(它按 id 取内存,不存在跨场拼接) | `tests/test_report_sources.py`(6) | 抽掉两行接线 → **4 红**(纯函数 2 条仍绿,符合预期) |
| **D1** | `load_markers()` 交出**深拷贝**,缓存本体不再外流(私有 `_load_markers_cached`) | `tests/test_marker_table_immutable.py`(2) | 修复前实测 2 红 |
| **D2** | `log_prosody` 的 `except` 保留打印但改为 `raise`(返回值不再承担成功/失败语义);端点原有的 `except Exception → 500` 接住 | `tests/test_prosody_write_failure.py`(2) | 修复前实测 `DID NOT RAISE` |
| **D3** | 新具名函数 `normalize_session_id`(三份副本):**`None`=没给→`NONE`;给了但空/仅空白=坏输入→400**。方向取"响亮失败"那一侧,依据是 face/gesture 守卫文档里自己写的原则 | `tests/test_session_id_normalization.py`(5)+ 端点级 2 | 撤掉两个 app 的改动 → **2 红**,日志打出 `session=NONE` + 落盘 `*_NONE.csv` |

**D3 的方向不是"都归 NONE"**:空串与"没给"在客户端那里是两件事(后者没接会话,前者参数拼错)。
静默归 NONE 之后客户端拿到 200 和一份看着正常的响应,问题只在报告里以"数据对不上"浮出来 ——
那正是守卫文档里已经写下的反对理由。

**测试账**:160 → **177 passed**(+17)。合并门复核:**0 / 2835510**(与改动前基线一致)。

**本轮新发现(不在原 triage 里,记档)**:

1. **`face_expression` 与 gesture 一样被 mediapipe 1.0 打死,但坏法不同** —— gesture 在
   `gesture_analysis/api/app.py:42-46` **模块导入时**取 `mp.solutions` → 进程直接死;
   face 的 `VideoPipeline.face_mesh` 是**惰性属性**(`pipeline/video_pipeline.py:33-41`),
   启动不碰它 → 服务"起来了"、`/health` 正常,而**每个 `/analyze` 静默 500**。
   实测:`hasattr(mp,'solutions') → False`;直接调 `VideoPipeline.process_frame()` →
   `AttributeError`。后果:`data/logs/` 里**永远不会出现** `face_au_log_<sid>.csv`。
   **spec §2 的 T7 前置只提了 gesture,把 face 当成可用的 —— 这一条要改,T7 的判据
   「face/voice 两份日志」在现环境不可能通过。** 使用者已裁定:face 与 gesture **一起修**。
2. **voice 服务是进程级单例,不是按会话**(`voice_interaction/api/app.py:57-59` 的
   `interview_assessment` / `research_assessment` / `voice_logger` 都不带 sid,
   `:307` 处被整体覆盖);`/session/{sid}/summary` 自己在 `:426-428` 承认。两会话并发会
   互相污染 —— 比 §3 第 9 项那个"409 契约"严重一档,单列。
3. **`session_id` 的读侧仍不存在**:`grep session_id` 在 `feature_engine` / `research_mapper`
   / `visualizer` / `evidence_gate` → **零命中**;`data_loader.py` 那个 sid 是从**文件名正则
   反推**的(不读 CSV 列、不读 `session.json`);`refresh_manifest`(`transcript_store.py:185`)
   是唯一会做 `expected → present/missing` 对账的代码,**零生产调用方** —— `session.json`
   的 `expected_file` 至今只是"声称"。I4 把"选了谁"披露出来了,但**按 id 选**仍是 M2。
4. **`api/app.py:386-389` 还有一处同类吞异常**:`try: interview_assessment.save_log()
   except Exception: pass` —— 与本轮 D2 同形,**未在裁定范围内**,未改,记档。
5. 本轮我自己踩的一个测试设计坑:`I4` 的集成断言最初用裸词「不同」判跨场,而报告别处本来就有
   「不同指标的量纲无法折算为同一尺度」→ 近乎恒真。改用只属于披露段的句子「并非同一场面试」。
   (同一家族:断言必须能说出"哪个生产改动会让它变红"。)
