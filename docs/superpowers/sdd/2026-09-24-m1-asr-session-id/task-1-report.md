# Task 1 报告:ASR 适配层(session id + 转写落盘 + 引擎)

- 分支:`feat/m1-asr-session-id`,BASE `1e53c131`
- 解释器:`~/miniconda3/envs/jingxin/bin/python`(3.11.15)
- 测试现状:新增 21 个测试 **全绿**;未被我改动的 4 个既有测试文件 53 passed;**全套 74 passed**(无 skip、无 xfail、无 warning)
- 提交(3 个,均为精确路径 add):

| SHA | subject |
| --- | --- |
| `34064d9` | feat(m1): ASR 适配层与转写落盘(session id / 清单 / 引擎规整) |
| `a76874c` | refactor(m1): voice_interaction 改惰性导出,PEP 562(Ruling M1-1) |
| `4447593` | feat(m1): 补齐 transcript_store.append_utterance(计划 Produces 承诺,Step 4 代码块遗漏) |

`git diff --stat 1e53c131..HEAD` = 10 files changed, 859 insertions(+), 20 deletions(-) —— 与下面"文件清单"逐条对应,无夹带。

---

## 1. 逐文件说明

| 文件 | 内容 | 来源 |
| --- | --- | --- |
| `voice_interaction/asr/__init__.py` | 空(0 字节) | 计划 Step 1 verbatim |
| `voice_interaction/asr/funasr_client.py` | vendor 副本,382 行 | `cp ~/asr-test/funasr_client.py`,**md5 双向核验一致**:`8b04b19990ad4f9e2e2d3f4be5f23222`(磁盘文件与 git blob 同值);`grep -c _merge_finals` = 2;未改一个字节 |
| `voice_interaction/asr/session.py` | `NONE_SESSION = "NONE"`;`new_session_id(now=None)` = `%Y%m%d_%H%M%S` + `secrets.token_hex(2)` | Step 4 verbatim |
| `voice_interaction/asr/transcript_store.py` | `DEFAULT_ROOT`(= `~/shared/jingxin_recordings`,Windows 侧 `D:\Shared\jingxin_recordings`)、`JINGXIN_RECORDINGS_DIR` 覆盖、`recording_dir`、`ensure_manifest`(幂等,不覆盖既有)、`refresh_manifest`;外加 `_transcript_path` + **`append_utterance`**(见 §5 偏离 3) | Step 4 verbatim + 补一个 Produces 承诺的函数 |
| `voice_interaction/asr/funasr_engine.py` | `count_cjk_chars`、`AsrUtterance`、`FunASREngine`;配置全部读 `asr_config.json` | Step 8 + 一处必需偏离(§5 偏离 1) |
| `voice_interaction/asr/asr_config.json` | host/port/timeout + `min_chars_for_density{value:10,_provisional,basis}` | Step 4 verbatim(阈值在数据文件,非代码裸常量) |
| `voice_interaction/__init__.py` | eager → PEP 562 惰性导出(Ruling M1-1) | 计划外,M1-1 要求 |
| `tests/test_transcript_store.py` | 计划 4 个 + 我加 3 个 = 7 | Step 2 verbatim + 补测 |
| `tests/test_asr_engine.py` | 计划 5 个 + 我加 4 个 = 9 | Step 6 verbatim + 补测 |
| `tests/test_lazy_exports.py` | 5 个(M1-1) | 新增 |

「原文只进仓库外」已核验:测试全程显式传 `root=tmp_path`;跑完 `ls ~/shared/jingxin_recordings/` = **目录根本不存在**,即测试没有写进真实目录。仓库内无任何原句落盘。

---

## 2. vendored 客户端的**真实** API(逐条核实,非照抄计划)

打开 `~/asr-test/funasr_client.py` 逐行读后的实测结论:

| 项 | 真实情况 | 与计划 snippet 的关系 |
| --- | --- | --- |
| 同步入口 | `recognize_pcm(pcm: bytes, **kw) -> ASRResult`,内部 `asyncio.run(arecognize_pcm(pcm, **kw))` | 计划只调 `recognize_pcm(pcm, host=...)`,这部分对 |
| 关键字名 | `arecognize_pcm(pcm, host=, port=, mode=, on_partial=, realtime=, timeout=)`;`timeout` 名**不是** `timeout_s` | 计划没传,所以谈不上对错;见 §5 偏离 1 |
| `ASRResult` 字段 | `text`、`is_final`、`mode`、`raw: dict`、`segments: list[ASRResult]`,另有 property `is_partial` | 计划读 `.text`/`.segments`/`.raw` **正确** |
| 逐字时间戳位置 | `raw["timestamp"]`,形如 `[[start_ms, end_ms], ...]`,**段内相对** | 计划判断正确(测试名 `test_uses_raw_timestamp_not_attribute`) |
| `raw["punc_array"]` | 存在(list) | 正确 |
| 置信度 | **完全没有**。`raw` 里没有 score/confidence,`ASRResult` 上也没有 | 与计划一致(计划没碰它);后续任务若要 ASR 置信度,只能自算 |
| 多段合并 | `_merge_finals(finals)`(定义 + 调用共 2 处)把多段拼成一条:合并结果的 `raw["timestamp"]` 是**逐段偏移接续的拼接链**,`raw["punc_array"]` 拼接,`raw["segment_count"] = len(segments)`,而 `res.segments[i].raw["timestamp"]` 仍是**段内相对**值 | 计划在这一点上判断正确并标记 `ts_origin="segment_relative"` |
| 空结果 | 无 final 时 `_merge_finals([])` 返回 `ASRResult()`,即 `text=""`、`segments=[]`、`raw={}` | 计划的 `_to_utterance` 会退化成 1 个空段,见 §6 关注点 5 |
| 异常 | 客户端 `arecognize_pcm` 本身不吞异常(异常从 `asyncio.run` 冒出来);`ASRSession` 那条线才吞(`except Exception: pass`) | 计划的"异常上抛"成立,且 `ASRSession` 不在本任务范围内 |
| 无本地看门狗 | `timeout` 只用于"推完音频后收尾 `ws.recv()`"的等待上限;推流中途挂住不设上限 | 计划的 docstring 若读作"每次调用加超时"会略微夸大,我按实际语义改了措辞(见 §5 偏离 1) |

**结论:计划 snippet 对真实 API 的判断基本正确,唯一硬伤是没把 `port`/`timeout` 传下去(§5 偏离 1)。**

---

## 3. RED/GREEN 证据(命令 + 实测输出)

### A 组:`tests/test_transcript_store.py`(计划 Step 2–5)

```
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_transcript_store.py -q
```

RED(实现前):
```
tests/test_transcript_store.py:4: in <module>
    from voice_interaction.asr import session, transcript_store
E   ImportError: cannot import name 'session' from 'voice_interaction.asr' (.../voice_interaction/asr/__init__.py)
1 error in 0.22s
```
> 与计划的预期差异:计划写"Expected: FAIL(`ModuleNotFoundError`)"。实测是 `ImportError`,因为 Step 1 已经先建好了 `voice_interaction/asr/__init__.py`(包存在、模块不存在),Python 报的是 `cannot import name` 而非 `No module named`。根因同一个(被测模块不存在),异常类别不同,记录备查。

GREEN(Step 4 实现后):`4 passed in 0.15s`;补 `append_utterance`/`refresh_manifest` 测试后:`7 passed in 0.05s`。

### B 组:`tests/test_asr_engine.py`(计划 Step 6–9)

RED:
```
tests/test_asr_engine.py:5: in <module>
    from voice_interaction.asr import funasr_engine
E   ImportError: cannot import name 'funasr_engine' from 'voice_interaction.asr'
1 error in 0.19s
```
> 计划给的另一个候选 `TypeError: __init__() got an unexpected keyword argument 'client'` 在测试模块 import 阶段就短路了,不可能出现;实际绿/红分界是"模块存在与否"。

GREEN(两文件一起跑,Step 9 要求):`12 passed in 0.15s`(计划预期 9,我多了 3 个);最终 `test_asr_engine.py` 单独 `9 passed`。

### C 组:`tests/test_lazy_exports.py`(Ruling M1-1)

RED(改 `__init__.py` 之前):
```
FAILED tests/test_lazy_exports.py::test_lightweight_import_pulls_no_heavy_pipeline[import voice_interaction]
FAILED tests/test_lazy_exports.py::test_lightweight_import_pulls_no_heavy_pipeline[import voice_interaction.asr.session ...]
E  AssertionError: 轻量导入被拖进了:vosk,sounddevice,librosa
2 failed, 3 passed in 0.49s
```

GREEN(惰性化之后):`5 passed in 0.17s`。

### D 组:`append_utterance`(偏离项 3)

RED:
```
E   AttributeError: module 'voice_interaction.asr.transcript_store' has no attribute 'append_utterance'
2 failed, 5 passed in 0.09s
```
GREEN:`7 passed in 0.05s`。

### 怎么证明这些测试不是"改前改后都绿"

对每个新行为做了**变异审计**(脚本 `/tmp/mutation_check.py`:把实现改回错误写法 → 跑对应测试 → 从内存还原;还原后复跑确认全绿)。结果:

| 变异 | 被杀掉的测试 | 结果 |
| --- | --- | --- |
| M1 引擎只传 `host`(计划原样) | `test_configured_host_port_timeout_reach_the_client` | **RED** ✔ |
| M2 `_config()` 改成硬编码常量 | `test_defaults_are_read_from_asr_config_json` | **RED** ✔ |
| M3 `n_chars = len(text)`(把标点算进去) | `test_count_cjk_chars_ignores_punctuation_and_ascii` | **RED** ✔ |
| M4 每段都读合并结果的 `raw["timestamp"]` | `test_against_vendored_asrresult_and_merge_finals_no_network` | **RED** ✔ |
| M4(同上) | `test_multi_segment_merge_is_kept`(计划自带) | **GREEN —— 杀不掉** ⚠ |
| M5 惰性映射表漏一个 `__all__` 名 | `test_every_name_in_all_resolves` | **RED** ✔ |
| M6 `__getattr__` 未知名返回 `None` | `test_unknown_attribute_raises_attributeerror` | **RED** ✔ |
| M7 `recording_dir` 不 mkdir | `test_recording_dir_is_outside_repo` | **RED** ✔ |
| M8 `__init__.py` 还原成 BASE 的 eager 版 | `test_lightweight_import_pulls_no_heavy_pipeline` | **RED** ✔ |
| M9 `append_utterance` 用 `"w"` 覆盖写 | `test_append_utterance_appends_newline_per_call` | **RED** ✔ |
| M10 `refresh_manifest` 算完不回写 | `test_refresh_manifest_flags_present_and_missing` | **RED** ✔ |

M4 那行是本任务最有价值的发现,写进 §6 关注点 3。

我保留的 3 个"改前也绿"的测试是**有意的护栏**,不谎称是驱动测试,已在文件 docstring 里写明:`test_every_name_in_all_resolves`、`test_from_import_forms_still_work`(改前 eager import 也解析得出来,但它们能抓住惰性映射表写错)、`test_refresh_manifest_flags_present_and_missing`(守的是 Step 4 已实现代码的回归)。

---

## 4. `voice_interaction/__init__.py` 的改动(Ruling M1-1)

- 删掉 6 行 eager `from .core…/from .pipeline…` import,换成 `_LAZY_EXPORTS: dict[name -> (子模块绝对路径, 属性名)]`,共 15 条,与 `__all__` 一一对应(映射表的每一项都有测试兜底)。
- 新增 PEP 562 `def __getattr__(name)`:首次访问时才 `importlib.import_module(...)`,取到值后写回 `globals()[name]` 缓存;未知名字 `raise AttributeError(...) from None`(不静默返回 `None`)。
- 新增 `__dir__()`,让 `dir(voice_interaction)` 仍列全 15 个公开名。
- `__all__`、`__version__`、`__author__`、模块 docstring 全部保持不变,**其他包的 `__init__.py` 一个没碰**。

实测语义(非"永不 import",而是"按需"):
```
after import, vosk present? False      # import voice_interaction 不再拖 vosk
name -> TTSPipeline                    # 访问时导入成功
after access, vosk present? True       # 只有真的用到重管线才付代价
cached in globals? True                # 第二次访问不再走 __getattr__
```

证明公开名字仍可用的测试:`tests/test_lazy_exports.py`
- `test_lightweight_import_pulls_no_heavy_pipeline`(参数化 2 例,子进程跑,`vosk`/`sounddevice`/`librosa` 必须一个都不在 `sys.modules`)—— 驱动测试(M8 证明可被还原杀掉)。
- `test_every_name_in_all_resolves` 遍历 `__all__` 逐个取值,漏一个就红(mapping 覆盖度)。
- `test_from_import_forms_still_work`:`from voice_interaction import SpeechRecognitionResult, ProsodyFeatureExtractor, TTSPipeline, InterviewAssessmentPipeline` 全部解析 —— 老写法零改动继续可用。
- 另核对了全仓引用:`experiments/`、`code_data_supplement/`、根 `app.py`、`report_frontend/` 里对 `voice_interaction` 的引用**全部是子模块直连**(`from voice_interaction.pipeline.X import Y`),不受父包 `__init__` 影响;全套测试 74 passed 也印证没有回归。

> 注意:`numpy` 仍会被拖进来,这是 vendored `funasr_client.py` 自己要的,属预期,所以断言名单里不含 numpy。

---

## 5. 偏离计划之处(全部为"计划不知道真实 API/自相矛盾"所迫)

1. **引擎把 `port` 与 `timeout` 真的传给客户端(偏离 Step 8 snippet 的一行)。**
   计划写 `res = self.client.recognize_pcm(pcm, host=self.host)`;那样 `__init__` 存的 `self.port` / `self.timeout_s` 是**死参数**,而模块 docstring 第 4 条却声称"给每次调用加超时"——代码与自己的注释矛盾。真实客户端签名接受 `port=` 与 `timeout=`(见 §2),故改为
   `self.client.recognize_pcm(pcm, host=self.host, port=self.port, timeout=self.timeout_s)`,并把 docstring 措辞改成实话:"`timeout` 是客户端自己的收尾等待上限;不是本地看门狗"。
   新增 `test_configured_host_port_timeout_reach_the_client` 守住它(M1 变异杀得掉)。
   未做:没有自己再加一层 watchdog 或线程超时 —— 计划没要求,属 YAGNI。

2. **`transcript_store.append_utterance` 计划里根本没有。** 计划的 `Interfaces / Produces(后续任务依赖的确切名字)` 明确承诺
   `transcript_store.append_utterance(session_id: str, utt, recorded_at: str | None = None, root=None) -> Path`,
   但 Step 4 给的 `transcript_store.py` 代码块里**没有这个函数**,Step 2 的测试也没覆盖它。这是计划自相矛盾:task 标题就叫"转写落盘",而落盘函数缺失。处理:按承诺的签名实现 + 补 2 个测试(见 §3 D 组)。
   **格式是我定的**(计划未规定):`{root}/{session_id}/transcript_{session_id}.jsonl`,一行一条 JSON,字段 `recorded_at / session_id / text / n_chars / n_segments / vad_split / segments`,以 `"a"` 追加(崩溃不丢已识别内容)。若后续任务的 brief 钉了别的文件名/schema,改这里即可,调用方签名不受影响。

3. **RED 的异常类别与计划预期不同**(A/B 两组,详见 §3):`ImportError: cannot import name` 而非 `ModuleNotFoundError` / `TypeError`。不是实现问题,是 Step 1 先建包导致的。

4. 其余各处均按计划 verbatim 落盘,包括 `refresh_manifest`(实现但无调用方),`test_transcript_store.py` 里保留计划自带的 `from pathlib import Path`(实际未使用)与行内 `__import__("json")`(我在文件顶部另加了 `import json` 供新测试用)。

---

## 6. 自审发现 / 关注点

1. **[重要] 计划自带的 `test_multi_segment_merge_is_kept` 并不能守住它名字声称的东西。** 变异 M4 把 `_to_utterance` 改成"每段都读合并结果的 `raw["timestamp"]`",该测试**照样绿**(它只断言段数与段文本);被杀掉的是我加的 `test_against_vendored_asrresult_and_merge_finals_no_network`(用 vendored 自己的 `ASRResult` + `_merge_finals` 造数据,不走网络,断言 `segments[1]["timestamps_ms"] == [[0, 50]]` 而非被累加成 `[[200, 250]]`)。这正是本项目"测试改前改后都绿"的老毛病的一例,建议后续任务照此补测。
2. **`append_utterance` 的文件名与 schema 无上游依据**(格式自拟,§5 偏离 2)。这是我最想请控制端确认的一条。
3. **`_provisional` 阈值**:`min_chars_for_density=10` 照计划进了 `asr_config.json` 且带 `basis`,但本任务的引擎还没有任何代码读它(Task 2/3 才用)。即该字段目前是"数据已就位、消费方未到",不是死代码。
4. **`timeout` 不是看门狗**:客户端只在"推完音频后的收尾 recv"上用 `timeout`;若 WebSocket 在推流中途挂住,本适配层不会自己中断。计划 docstring 原先的"给每次调用加超时"读起来像硬保护,已在措辞上改准。若 Task 2 需要硬超时,应在调用侧加线程/watchdog(本任务按 YAGNI 未加)。
5. **空结果会退化成"1 段"**:客户端无 final 时返回 `ASRResult()`(`segments=[]`、`raw={}`),`_to_utterance` 的 `segs = [res]` 会给出 `n_segments=1, vad_split=False` 的空段。计划的测试只断言 `n_chars==0`,所以这条路径被固化成了现状。Task 2 若用 `n_segments==0` 表示"整段没识别到"会踩坑,建议那时再加一个明确断言。
6. **`recognize_pcm` 内部是 `asyncio.run`**:在已在运行的事件循环里(例如 async 端点内直接调用)会抛 `RuntimeError`,必须放到工作线程。这是 vendored 客户端的既有事实,计划没提,Task 2 接 voice 的 FastAPI 时要当心。
7. **`refresh_manifest` 无任何消费方**,计划 Produces 列表里也没有它;按 Step 4 verbatim 实现,并补了 1 个测试兜住"真回写磁盘"。
8. **与我无关的既存工作区改动**:`report_frontend/data/output/*.html` 有 10 个文件在**我开始之前**就处于工作区删除状态(且 `.gitignore` 忽略 `**/data/output/`)。我没有碰、没有 add、没有 commit,提交里不含它们。

## 7. 自查清单

- Step 1–9 逐条走完,顺序即 TDD(先写测试 → 看它因正确原因失败 → 实现 → 看它变绿 → 提交)。
- 提交只含精确路径(共 10 个文件),`experiments/` 下 ~1 万未跟踪文件一个没进;无 amend、无 rebase;提交身份用 inline `-c user.name/-c user.email`。
- 行尾全 LF(`.gitattributes` 的 `* text=auto eol=lf`);vendored 文件 md5 与源文件一致。
- 无新依赖(只用 stdlib + 已在环境里的 websockets/numpy,vendored 客户端自带的)。
- 仓库内只有数字/路径,无任何原始转写文本;测试从不写真实 recordings 目录。

---

# 修复报告:Ruling M1-5(`transcript.json` 按 spec §6.4)

> **本节推翻上面 §5 偏离 2 与 §6 关注点 2 里关于落盘格式的内容。** 上文描述的
> `transcript_{session_id}.jsonl`(每次调用一行 JSON)已作废,以本节为准;
> 其余章节(引擎、session、惰性导出、`_merge_finals`、审计结论)不受影响。

**提交:`42165c2` fix(m1): transcript.json 按 spec 6.4 落盘(单文件累积写 / merged 重算 / provenance 块)**

我按裁定去读了 `docs/superpowers/specs/2026-09-24-m1-asr-session-id-design.md` §6.4 原文(而不是只按消息里的片段做),spec 比片段多给了三件事:`endpoint` 的来源(`ws://host:port`)、`models` 四键的存在、以及段内 6 个键的完整清单 —— 都已照做。

## 改动前后对比

| | 改前(已作废) | 改后(spec §6.4) |
| --- | --- | --- |
| 文件 | `transcript_{session_id}.jsonl`(每会话一文件,若干行 JSON) | **`transcript.json`**(每会话一个文件,一个 JSON 文档) |
| 写语义 | 每次调用**追加一行**,彼此独立 | 每次调用把段**追加进同一 `segments` 数组**(`index` 连续),`merged` 全段**重算** |
| `recorded_at` | 每次调用各写各的 | **只有第一次写入**决定,后续追加不刷新 |
| provenance | 无 | `asr` 块:`engine`/`endpoint`/`models` + `asr_confidence: null` + `asr_confidence_source: "unavailable"` |
| 段形状 | 原样转存引擎的段字典 | 白名单 6 键(`index`/`text`/`n_chars`/`timestamps_ms`/`punc_array`/`ts_origin`),`ts_origin` 恒 `segment_relative` |
| 写盘 | `open("a")` 直接追加 | 临时文件 + `os.replace` 原子替换(read-modify-write 中途崩掉不会毁掉整场既有段) |

真实产物(两次调用:1 段 + 2 段,已实测打印,键序与 spec 一致):

```json
{
  "session_id": "20260924_153012_9f3c",
  "recorded_at": "2026-09-24T15:30:12+08:00",
  "asr": {
    "engine": "funasr",
    "endpoint": "ws://192.168.72.30:10095",
    "models": {},
    "asr_confidence": null,
    "asr_confidence_source": "unavailable"
  },
  "merged": {"text": "我叫王国梁。前半段后半段", "n_chars": 11, "n_segments": 3, "vad_split": true},
  "segments": [
    {"index": 0, "text": "我叫王国梁。", "n_chars": 5, "timestamps_ms": [[210, 450]], "punc_array": [1], "ts_origin": "segment_relative"},
    {"index": 1, "text": "前半段", "n_chars": 3, "timestamps_ms": [[0, 100]], "punc_array": [2], "ts_origin": "segment_relative"},
    {"index": 2, "text": "后半段", "n_chars": 3, "timestamps_ms": [[0, 50]], "punc_array": [3], "ts_origin": "segment_relative"}
  ]
}
```

（上面为可读性把段内数组折成一行;落盘用 `indent=2`,与既有 `session.json` 一致。）

## RED / GREEN

命令:`cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_transcript_store.py -q`

**RED(改测试、未实现时,5 failed / 5 passed)** —— 五条新断言各自因为正确的原因红:

```
assert p == tmp_path / sid / "transcript.json"
E  AssertionError: assert PosixPath('.../transcript_20260924_153012_9f3c.jsonl') == ((PosixPath('...')
     / '20260924_153012_9f3c') / 'transcript.json')

tests/test_transcript_store.py:...: in test_asr_block_follows_asr_config
E  KeyError: 'asr'

E  json.decoder.JSONDecodeError: Extra data: line 2 column 1 (char 287)      ← 旧 jsonl 根本不是单个 JSON
E  json.decoder.JSONDecodeError: Extra data: line 2 column 1 (char 279)

assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}", recorded_at)
E  AssertionError: 2026-09-24T12:35:14                                        ← 裸 isoformat,无时区偏移
5 failed, 5 passed in 0.10s
```

**GREEN(实现后):**`10 passed in 0.06s`;全套 `77 passed in 0.99s`。

## 新增/改写的测试(共 5 条 + 1 条守卫)

| 测试 | 钉住什么 |
| --- | --- |
| `test_append_utterance_writes_single_transcript_json` | 文件名/顶层 5 键/asr 块 5 键(含两个恒定的置信度键)/merged/段 6 键;并断言目录里**只有** `transcript.json`(排除 jsonl);输入段里塞了一个 `engine_future_field`,契约不得被污染 |
| `test_asr_block_follows_asr_config` | `endpoint`/`models` 只能来自 `asr_config.json`(monkeypatch `_CONFIG_PATH` 成 10.1.2.3:4321);代码里再存一份 host/port 即红 |
| `test_append_utterance_accumulates_across_calls` | 跨调用累积:段追加、`index` 连续、不覆盖既有段、`merged` 重算、`recorded_at` 保持第一次(第二次显式传了另一个时间也不许刷新);文本带标点使 `n_chars 求和 ≠ len(text)`,并断言两次都没裂过时 `vad_split` 仍是 `False` |
| `test_merged_vad_split_is_true_once_any_call_was_split` | 任一次被裂过则整场 `True`;跨调用 index 仍连续;第二段的段内相对时间戳原样保留、不被累加 |
| `test_recorded_at_default_is_iso_with_offset` | 默认值必须带时区偏移(spec 示例 `+08:00`),裸 `isoformat()` 即红 |

**变异审计**(照旧:改回错误写法 → 跑对应测试 → 从内存还原;脚本 `/tmp/mutation_check_m15.py`)。**15 个变异全部被杀死**:

| 变异 | 杀它的测试 |
| --- | --- |
| N1 路径改回 jsonl | shape |
| N2 `recorded_at` 每次刷新 | accumulation |
| N3 段不累积(覆盖) | accumulation / vad_split |
| N4 `index` 每次从 0 重来 | accumulation / vad_split |
| N5 `merged` 不从累积段算(用空表) | accumulation / vad_split |
| N5b `merged.n_chars = len(text)` 而非各段求和 | accumulation(⚠ 见下) |
| N5c `merged.text` 只取最后一段 | accumulation / vad_split |
| N6 `vad_split = (总段数 > 1)` 而非累积 OR | accumulation |
| N7 省掉两个置信度键 | shape |
| N8 `endpoint` 硬编码 | config |
| N9 `ts_origin` 写成 `absolute` | shape / vad_split |
| N10 段白名单退化成整字典转存 | shape(⚠ 见下) |
| N11 默认 `recorded_at` 无时区 | recorded_at |
| N12 `models` 不读配置 | config |
| N13 `merged.n_chars` 只取第一段 | accumulation |

⚠ 审计逮到我自己两条**本来杀不掉**的断言,已当场补强(不是改测试去迁就实现,而是让测试真的能区分):

1. **N5b**:我最初用纯中文文本(`"第一句"`),`sum(n_chars)` 与 `len(text)` 恰好相等 → "数字符"也能通过。⇒ 改成带标点的文本(`"第一句。"`/`"第二句!"`,引擎 `count_cjk_chars` 去标点),并加 `merged.n_chars != len(merged.text)` 断言。helper 也改用 `funasr_engine.count_cjk_chars` 算 `n_chars`,与引擎口径一致。
2. **N10**:白名单被换成 `dict(seg)` 时输出对"规范输入"完全一致 → 不可能被杀。⇒ 在 shape 测试的输入段里加一个契约外字段 `engine_future_field`,现在漏进契约即红。

> 顺带说明一个**看起来是**漏网、实际不是的项:审计初版有个 N5 变异 `"n_segments": 0 if len(segments) < 2 else len(segments)`,它对 `len==2` 是恒等变换(no-op),所以"没被杀"是我的变异写坏了;把同一变异拿去对 1 段的 shape 测试跑,是 **RED**,已补验。

## 需要控制端确认的两处判断(我做了选择并写进测试)

1. **`merged.vad_split` 的语义 = "任一次识别被 VAD 裂过"的累积 OR,不是"总段数 > 1"。** 依据:字段名对应引擎的 utterance 级 `AsrUtterance.vad_split`,`"任一段裂过"` 读作"任一次回答裂过";而且按"总段数>1"解释的话,任何多次作答的会话都恒为 `true`、几乎不携带信息。两种解释在"单次回答即多段"时一致,只在"多次单段作答"时分歧 —— 而 `test_append_utterance_accumulates_across_calls` 正好钉住这个分歧点(2 段、`vad_split` 为 `False`)。若裁定应是"总段数>1",改一行 + 改一条断言即可。
2. **`models` 为空字典,理由写进 `asr_config.json` 的 `models_basis`。** spec §6.4 示例里模型名是带省略号的(`"…vocab8404-online"`),我**核实了服务端根本不回传模型名** —— 实测 raw 键只有 `is_final / mode / punc_array / segment_count / spk_name / spk_score / text / timestamp / wav_name`(见 `~/asr-test/zijijieshao_v2.json`),`/home/huihuibuhui/asr-test/` 与仓库里也没有任何地方记录过完整模型名。所以我没有把省略号"补全"成猜的模型名 —— 那等于伪造 provenance。现在做成**配置驱动**:`asr_config.json` 里填上四个键,`transcript.json` 立刻跟着变,不用改代码(`test_asr_block_follows_asr_config` 就是这个行为的守卫)。若需要真值,得由部署方在 `192.168.72.30` 上取 FunASR 服务配置。

## 其他

- 未碰 `report_frontend/data/output` 下的旧图表删除(你的 `5bf06a3`),我的 3 个提交里不含它们。
- 改动的文件仅 3 个:`voice_interaction/asr/transcript_store.py`、`voice_interaction/asr/asr_config.json`(新增 `models: {}` + `models_basis`)、`tests/test_transcript_store.py`。
- 测试仍然从不写真实 recordings 目录:跑完全套后 `~/shared/jingxin_recordings/` 依旧不存在。
- 小节:`indent=2` 下 `timestamps_ms` 会被展开成每数字一行,一次几百字的作答该数组会有数千行。形状与 spec 完全一致,只是文件偏"胖"。已与 `session.json` 的写法保持一致(未擅自改成紧凑编码);若下游觉得体积碍事,可单独裁定改成紧凑分隔符。

---

# 追加发现:计划自带的 session id 测试有 **1.86% 的假红率**(已修)

**提交:`d6d3754` test(m1): 修掉 session id 唯一性断言的 1.86% 假红(生日问题)**

## 怎么发现的

M1-5 改完后我连跑全套做终检,8 次里红了 1 次:

```
run 5: 1 failed, 76 passed in 1.02s
```

失败**不是**我这次改的东西,而是:

```
tests/test_transcript_store.py:19: AssertionError
E   AssertionError: assert 49 == 50
E    +  where 49 = len({'20260924_123733_00f5', '20260924_123733_01a7', ...})
```

即计划 Step 2 自带、我一字未改抄进来的 `test_new_session_id_is_unique`:`assert len(ids) == 50`。

## 根因(不是随机噪声,是断言本身错了)

`new_session_id` 的随机段是 `secrets.token_hex(2)` = **4 位十六进制 = 65536 种取值**。
50 次抽样按生日问题**必然**有约 `C(50,2)/65536 ≈ 1.86%` 的概率撞一次 —— 也就是说这条断言
在**实现完全正确**的前提下,每跑一次就有 1.86% 的概率报红。200k 次模拟实测:

```
P(50 个里有重复) = 1.8595%      ← 原断言(==50)的假红率
P(distinct < 49) = 2.05e-04
P(distinct < 48) = 0.00e+00     (解析近似 ~1.1e-6;200k 次模拟最小去重值 = 48)
```

**放大效应:**它还会连带打红既有的 `tests/test_assert_coverage.py::test_no_assert_is_dead` ——
那个检查器有一条"内层套件不绿则结论不可信"的闸门(`inner_exit_code == 0`),所以这一条假红
会让**两条**测试一起红,看起来像两个问题。

## 改法(先证伪、再实现)

把"50 个必须两两不同"换成两条:

| 测试 | 性质 | 说明 |
| --- | --- | --- |
| `test_new_session_id_uses_two_bytes_of_secrets_hex`(新) | **确定性** | 打桩 `session.secrets.token_hex`,断言 `new_session_id(now=固定时刻)` 恰等于 `20260924_153012_abab` —— 随机段的**来源与宽度**都被钉死 |
| `test_new_session_id_suffix_is_varied`(新) | 统计冒烟 | `len(ids) >= 48`,假红率约 1e-6,挡"常量随机段" |

变异审计(改 `session.py` → 跑这两条 → 内存还原):

| 变异 | 结果 |
| --- | --- |
| P1 随机段改成 1 字节 `token_hex(1)` | 确定性那条 **RED** ✔ |
| P2 随机段写死 `"0000"` | 两条都 **RED** ✔ |
| P3 随机段改成计数器 | 确定性那条 **RED** ✔ |

**诚实交代这条冒烟测试的功力边界**:我算了 1 字节随机段时的分布 —— 50 次抽样去重期望 45.5,
但 `>= 48` 只能挡住其中 **86%**(还有 14% 概率蒙混过去)。所以挡 `token_hex(1)` 靠的是那条
**确定性**测试,不是这条统计测试;统计测试只负责"真实(未打桩)路径确实在变"。

## 回归验证

```
run 1..12: 78 passed in 0.95~0.97s      ← 12 次连跑 0 红(改前 8 次红 1 次)
```

改前每跑一次有 1.86% 假红 ⇒ 12 次全绿在改前的概率只有约 80%,配合上面的解析值可以认定假红已消除
(不是"这次运气好")。

> 需要控制端注意:这条断言是**计划 brief 原文**(Step 2 逐字给出的),不是我在偏离里自创的。
> 我按"先让它红得可解释、再换掉"的方式处理了,若你希望保留原样(接受 1.86% 假红),回退
> `d6d3754` 即可,不影响 `42165c2` 的功能改动。

---

# 审查修复报告(1 Important + 3 Minor)

**提交:`a600e13` fix(m1): 审查修复(默认落盘路径不变量 / load_config 公开化 / 模块 docstring / models 四键)**

改动 4 个文件、+65/−9。每条修复对应的文件:

| # | 级别 | 修复 | 文件 |
| --- | --- | --- | --- |
| 1 | Important | 默认落盘路径不在仓库内 —— 现在真的被测 | `tests/test_transcript_store.py` |
| 2 | Minor | 模块 docstring 第 4 条改成事实陈述 | `voice_interaction/asr/funasr_engine.py` |
| 3 | Minor | `models` 填上四个契约键 + 依据 | `voice_interaction/asr/asr_config.json` |
| 4 | Minor | `_config` → 公开 `load_config()` | `funasr_engine.py` + `transcript_store.py` + 测试 |

---

## 1(Important)默认落盘路径不变量:改前覆盖率为零,现在真的能红

**审查的判断我复核属实。** 按你给的场景原样复现(`DEFAULT_ROOT = Path("data/transcripts")`,即把面试原句写进仓库):

```
=== Q1: DEFAULT_ROOT = Path("data/transcripts")  — 跑该文件全部测试 ===
FAILED tests/test_transcript_store.py::test_default_root_is_outside_the_repo
1 failed, 14 passed in 0.08s

--- 红色那条的报错 ---
E  AssertionError: 默认落盘位置落在仓库内,面试原句会被写进 git:/home/huihuibuhui/jingxin/data/transcripts
E  assert PosixPath('/home/huihuibuhui/jingxin') not in (PosixPath('.../jingxin/data/transcripts'),
     PosixPath('.../jingxin/data'), PosixPath('.../jingxin'), PosixPath('/home/huihuibuhui'), ...)
```

注意这张图的形状:**旧的那 14 条一条都不红**,唯一红的是新加的那条 —— 与审查说的"这 10 条测试全绿"一致,只是现在多了一个会喊的人。报错信息直接点出后果("会被写进 git"),不是只报两个路径不相等。

**改法(只动测试,生产代码未改):**

| 测试 | 覆盖的不传-root 路径 |
| --- | --- |
| `test_recording_dir_honours_explicit_root`(原 `..._is_outside_repo` 改名) | 显式 root(测试隔离用),名字现在与断言一致 |
| `test_default_root_is_outside_the_repo`(新) | **`recording_dir(sid)` / `append_utterance(…, root=None)` 的生产路径**:`DEFAULT_ROOT` 解析后必须绝对、且仓库根不在它的自身/父链上 |
| `test_recording_dir_default_root_honours_env_override`(新) | `JINGXIN_RECORDINGS_DIR` 覆盖(之前同样零覆盖) |

比你的建议多了一行 `assert default_root.is_absolute()`:相对路径(正是 Q1 那种写法)会随 CWD 漂,这一行让判定不依赖"pytest 恰好从仓库根跑",CWD 换到别处也照样红。

变异验证(改 `transcript_store.py` → 跑测试 → 内存还原):

| 变异 | 目标测试 | 结果 |
| --- | --- | --- |
| Q1 `DEFAULT_ROOT = Path("data/transcripts")` | `test_default_root_is_outside_the_repo` | **RED** ✔(已含上面 14 绿/1 红的复现) |
| Q2 `_root` 不再读 `JINGXIN_RECORDINGS_DIR` | `test_recording_dir_default_root_honours_env_override` | **RED** ✔ |

> 说明:审计脚本里我把 `test_asr_models...` 和 `test_default_root...` 也当对照组跑了一遍(Q1 下 models 绿、Q2 下 outside 绿)—— 这两行**本就不该红**(不变量不同),列出来是为了说明这份表不是"挑好看的行报",不是漏网。

## 2(Minor)模块 docstring 第 4 条:改成与实测一致的陈述

原来(模块级,你指出的漏网):

```
  4. 客户端会吞异常/卡住 —— 这里把异常上抛、给每次调用加超时
```

现在:

```
  4. 这条同步路径(`recognize_pcm` → `arecognize_pcm`)本身不吞异常(吞异常的是
     `ASRSession` 那条线),这里同样不吞、直接上抛;构造函数里的 `timeout_s` 只作为
     客户端的 `timeout` 传下去,而它**只兜住推完音频后收尾 drain 的等待** ——
     不是"中途卡死的看门狗":推流过程中 socket 若挂住,本适配层不会自己中断
```

两句都是我回读 vendored 源码逐行核实的:异常路径上 `arecognize_pcm` 只捕获 `asyncio.TimeoutError`(推流间隙的 0.001 s 与非阻塞收尾),真正的错误直接从 `asyncio.run` 冒出来;`timeout` 参数只出现在推完音频之后那个 `wait_for(ws.recv(), timeout=timeout)` 里;推流循环 `for block in _iter_blocks(pcm): await ws.send(block)` **没有任何超时**。

**没有为这条加测试**:docstring 是散文、没有行为可断言,给它写"断言 docstring 文本"的测试只会得到一条脆断言。这条靠 diff 复核,我在上面把依据摆全。

## 3(Minor)`models` 四个契约键:填的是服务端日志取回的真名,不是 API

`asr_config.json` 的 `models` 现为:

```json
"asr_online":  "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
"asr_offline": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
"vad":         "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
"punc":        "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
```

`models_basis` 已按你的要求写明**来源是服务端日志、不是 API 返回**,并附上 API 侧的否证(本部署 raw 键只有 `is_final/mode/punc_array/segment_count/spk_name/spk_score/text/timestamp/wav_name`)。

**我做了独立核对,并把来源坐标补准了:** 你给的路径 `~/huihuibui/logs/server.err.log` 在 WSL 侧不存在 —— 因为那是**服务端** `192.168.72.30` 上 `zgy` 用户家目录里的路径(我早前那次实测的笔记里记的原话是 `tail -f /home/zgy/huihuibui/logs/server.log  # 应用日志;.err.log 里有模型加载详情`)。我没有 ssh 到那台机器的授权,所以**没法在本机复读那份日志**;但四个名字与我那次实测留下的笔记("四个模型全部实测在跑")逐条一致,也与 spec §6.4 里被省略号遮住的形态(`"…vocab8404-online"`、`"…fsmn_vad…"`、`"…punc_ct-transformer…"`)一致。据此我把 `models_basis` 写成"部署方在服务端读取",而不是含糊地宣称我自己读了 —— 谁读的、从哪读的,写清楚。

**新增守卫**(RED→GREEN,见下):`test_asr_models_records_the_four_contract_keys` 断言键集恰为四键且四个值都非空 —— 它挡的正是你担心的 `KeyError: 'vad'`,也挡住"又退回空字典"。

**关于第五个键(声纹 `iic/speech_campplus_sv_zh-cn_16k-common`):我判断不记入 `models`。** 理由:spec §6.4 的契约就是四键,下游/验收按那个形状读,擅自加键是改契约;而 `spk_score` 的真实来源这件事本身有价值,所以我把它写进了 `models_basis` 的文字里(既不丢信息,也不动契约)。若你希望它成为正式键,那是一处契约变更,应由你裁定后我再改。

## 4(Minor)`_config` → 公开 `load_config()`

选了"原地提升为公开访问器"而不是新建 `config_loader.py`:配置只有一个路径常量(`_CONFIG_PATH`),新建模块会把这份常量搬来搬去,收益只是换个位置;真正的目标是"跨模块不再依赖私有名",一行改名就达成。改动:

- `funasr_engine.load_config()`(公开,带 docstring 说明"store 与 Task 2 端点都从这里取");
- `FunASREngine.__init__` 改调 `load_config()`;
- `transcript_store._asr_meta` 改 `from .funasr_engine import load_config`;
- 全仓已无 `_config(` 调用(仅 `_CONFIG_PATH` 这个模块内路径常量保留为私有)。

**新增守卫** `test_store_reads_config_through_the_public_accessor`:断言 `load_config` 可调用,并**打桩这个公开名**后看 store 落盘是否跟着变 —— 也就是"store 真的走公开访问器",而不只是"某个函数存在"。

## RED / GREEN 与变异验证

命令:`~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_transcript_store.py -q`

RED(先写/改测试,未实现时):

```
E  AssertionError: assert set() == {'asr_offline', ..., 'punc', 'vad'}
E    Extra items in the right set: 'asr_online', 'asr_offline', 'punc', 'vad'
E  AttributeError: module 'voice_interaction.asr.funasr_engine' has no attribute 'load_config'
2 failed, 13 passed in 0.11s
```

GREEN(实现后):`15 passed`;全套 **82 passed**(连跑 4 次均绿)。

| 变异 | 目标测试 | 结果 |
| --- | --- | --- |
| Q1 `DEFAULT_ROOT` 移进仓库 | `test_default_root_is_outside_the_repo` | **RED** ✔ |
| Q2 `_root` 不读环境变量 | `test_recording_dir_default_root_honours_env_override` | **RED** ✔ |
| Q3 store 绕开访问器自己读配置 | `test_store_reads_config_through_the_public_accessor` | **RED** ✔ |
| Q4 访问器改回私有名 `_config`(三处同改,套件本是绿的) | 同上 | **RED** ✔ |
| Q5 `models` 退回空字典 | `test_asr_models_records_the_four_contract_keys` | **RED** ✔(RED 段已示) |

Q4 是对你那条顾虑的直接验证:把公开名改回私有名后,套件其余部分照常能跑,唯一喊停的是这条测试 —— 也就是"改动会在测试里响亮失败",而不是静默断在写盘那一刻。

---

## 你点名要我判断的那条:并发 read-modify-write 无锁(Task 4 会不会真发生)

**结论:按 spec §7 的顺序流程不会发生;但它需要一个触发条件,而那个条件在接进端点后是现实的 —— 建议在 Task 4 加锁,而不是现在改 Task 1。**

- **失败形态是"丢更新",不是"坏文件"。** Task 1 里我已用"临时文件 + `os.replace`"落盘,所以读者永远看不到半个 JSON;真正的窗口只剩 read→replace 之间:两个并发调用各自基于同一份旧内容算出新内容,后写者覆盖前者 → **某一段回答静默消失**。
- **要触发它,需要同一个 `session_id` 上有两个在飞的 `answer_audio`。** spec §7 的流程是一问一答(同一会话串行),所以正常路径够不着。现实的触发点是:客户端**超时重试/双击重复提交**(ASR 端到端约 1.75 s,重试窗口很大)、前端在上一段还在转写时就上传下一段、以及 `/asr` 与 `/interview/answer_audio` 同时写同一会话。FastAPI 的 `def` 端点在 threadpool 里跑,两个请求是真并行线程,不会有 GIL 意外替你兜住。
- **影响面比看起来小、但更隐蔽:** 数字报告不受影响(连接词密度在采集时由引擎文本算出,不读 `transcript.json`),所以丢的是**审计/验收链**,不会让指标算错 —— 也就意味着可能直到有人拿转写对音频复核时才发现。若走 `NONE` 会话(所有无 id 的请求共用一个目录)则更容易撞。
- **给 Task 4 的处置建议:** 单进程 uvicorn 下,一个按 `session_id` 键控的模块级 `threading.Lock` 就够;若多 worker,则进程内锁不够,需要 `fcntl.flock` 锁文件或收敛为单写者。顺带:`ensure_manifest` 是同一形状,但它是"存在即返回"的幂等写,最坏是多写一次,不是丢数据 —— 唯一有损的就是 transcript。

## 未动的部分(按你的裁定)

`merged.vad_split` 保持逐次调用 OR 的语义 —— **没改,也没加新断言**(我原先那条 `test_append_utterance_accumulates_across_calls` 里"两次都没裂过 → False"的断言保留,它正是区分两种读法的那条,不是新加的)。其余 7 条 Minor(manifest 与 transcript 的 provenance 重复、`ensure_manifest`/`refresh_manifest` 非原子写、docstring 与实现对 `expected/present` 的描述不符、duck-typing 不一致、并发无锁、`secrets` 全局打桩、`test_empty_text_...` 名字与断言不符)**一条没修**。
