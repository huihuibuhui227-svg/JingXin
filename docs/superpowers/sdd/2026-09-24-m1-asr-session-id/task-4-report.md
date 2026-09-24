# Task 4 报告:voice 端点接线 + 拆掉 vosk

分支 `feat/m1-asr-session-id`,提交 **6377c70**(`6377c70f61a4e8371cb310668166233444a5ab80`)。
作者行 `huihuibuhui227 <huihuibuhui227@gmail.com>`(inline `-c`,未改全局配置)。无 amend、无 rebase、未新建分支。

提交内容 = 8 个精确路径 + 一次 `rm -rf vosk-model-cn-0.22/`(2.0 GB,未被跟踪、且被 `.gitignore:55` 忽略,`git ls-files` = 0,所以删除在 git 里不留任何记录)。

---

## 1. 四处识别点

控制器要求"verify 一下四处位置"——**已核实**:改动前四处 `KaldiRecognizer(vosk_model, SAMPLE_RATE)` 正是
`api/app.py:144 / 203 / 281 / 414`,与 brief 给的坐标一致(brief 的数字是准的)。

| # | 位置(改前 → 改后) | 所在端点 | 替换成 |
|---|---|---|---|
| 1 | `api/app.py:144` → `:177` | `POST /asr` 直通 WAV 分支 | `utt = await _transcribe_async(audio_data); text = utt.text.strip(); if text: transcript_store.append_utterance(sid, utt)` |
| 2 | `api/app.py:203` → `:236` | `POST /asr` ffmpeg 转换后分支 | 同上 |
| 3 | `api/app.py:281` → `:326` | `POST /interview/answer_audio` | `utt = await _transcribe_async(audio_data); text = utt.text.strip()` + 空判 400 → 落转写 → 算密度 → 写语音日志行 |
| 4 | `api/app.py:414` → `:475` | `POST /research/answer_audio` | `utt = await _transcribe_async(audio_data); text = utt.text.strip()` + 空判 400 → 落转写 |

四处都走**同一个**入口 `_transcribe_async`(结构性保证"四个调用点用同一条异步路线",不是四处各写一遍)。
`grep -c "await _transcribe_async(audio_data)"` = 4。

响应键:四处**原有键一个没删**,只加不换 ——

| 端点 | 原有键(保留) | 新增(纯加) |
|---|---|---|
| `/asr` | `text` | `session_id` |
| `/interview/start` | `status`, `question` | `session_id` |
| `/interview/answer_audio` | `status`, `recognized_text` | `session_id`, `connective_density` |
| `/research/answer_audio` | `status`, `recognized_text` | `session_id` |

### 异步路线:选 `await asyncio.to_thread(_transcribe, pcm)`(Ruling M1-6)

为什么 `to_thread` 是 loop-safe 的:vendored 客户端的同步包装是
`funasr_client.py:217-219` 的 `recognize_pcm = asyncio.run(arecognize_pcm(...))`,而
`asyncio.run` 一进已有事件循环就 `RuntimeError`。FastAPI 的端点都是 async,直接调必炸。
`asyncio.to_thread` 把同步调用扔进默认 ThreadPoolExecutor,`asyncio.run` 在那个线程里自建事件循环
—— 既绕过"运行中的事件循环"这条限制,又顺势把 ~1.75 s 的阻塞识别挪出事件循环(服务期间还能接别的请求)。

**实测证据**(不是推理,见 §5 的接线核验脚本):

```
PASS  同步缝在事件循环里直接调会炸(所以端点不能直调)     # RuntimeError: ... running event loop
PASS  _transcribe_async 在事件循环里可用
PASS  四处识别点都用了它                                  # 源码里 4 处 await _transcribe_async(audio_data)
```

没选"用客户端 async API"的理由:`FunASREngine.transcribe_pcm` / `asr/transcribe.py` 的缝 / 管道两处都是
**同步**的,改 async 要么给引擎加第二条并行路径(两条路各自演进),要么把管道(脚本调用、没有事件循环)
也拖进 async。`to_thread` 让全仓只有一条转写代码路径。

### 缝的位置(Ruling M1-14)

`voice_interaction/asr/transcribe.py`(新,轻模块):

```python
asr_engine = FunASREngine()                 # 只读 asr_config.json,不连网
def transcribe(pcm: bytes) -> AsrUtterance:
    return asr_engine.transcribe_pcm(pcm)   # 不 strip、不补占位、不吞异常
```

`api/app.py` 用 `from voice_interaction.asr.transcribe import transcribe as _transcribe` 取那一层
(brief 的端点体因此逐字成立:只有 `_transcribe` / `_transcribe_async` 这两个名字是新的)。
`api/app.py` 里**没有** `FunASREngine` 构造、没有 `asr_engine`,端点不直接摸引擎。
测试只 import 轻模块并 monkeypatch **它的** `asr_engine`,全程不 import `api/app.py`。

### 附带修掉的两个真缺陷

1. **`/interview/start` 的 500 被自己吞掉**:原来 `except Exception` 会把内层
   `HTTPException(500, "无法获取问题")` 再包一层,detail 变成 `启动面试失败: 500: 无法获取问题`。
   按 brief 加了 `except HTTPException: raise`。
2. **两个 `answer_audio` 的 400 一直回的是 500**:`HTTPException` 是 `Exception` 的子类,原来
   `except Exception` 把内层 400(格式不对、未识别到有效语音)全包成 500。这在本任务里**必然显形** ——
   "空识别回 400"是 brief 新加的路径,不修就永远是 500。两个端点都加了 `except HTTPException: raise`
   (实测:"空识别抛 400" PASS)。

---

## 2. 并发锁

**形状**:`transcript_store.py:19-38`,模块级 `_LOCKS: dict[str, threading.Lock]` + 一把
`_LOCKS_GUARD` 保护字典自身,`_session_lock(session_id)` 做 get-or-create(双检:先查再建,建也在
guard 内)。`append_utterance` 的**整段读-改-写**包在 `with _session_lock(session_id):` 里(`:117`)。

**保护什么**:`append_utterance` 是「读 transcript.json → 追加本次段 → `os.replace` 原子替换」。
两个同会话的并发请求(客户端重试 / 重复提交 / `/asr` 与 `/interview/answer_audio` 撞车)各自读到同一份
旧 `segments`,后写的把先写的**整个盖掉**。丢的是**整次回答的审计轨迹**,而且静默 —— `n_segments`
只是少了一段,不报错、不变负数,数字报告看不出来。

**粒度是会话**:不同 `session_id` 之间没有共享状态(路径不同),不必互相等。锁表随会话数增长,单进程
服务下可忽略(一场会话一把);多 worker/多机部署要换成文件锁 —— 已写在 `_session_lock` 的 docstring 里。

**没加锁的两处**:`ensure_manifest`(会话开始写一次、已存在即返回,幂等)与 `refresh_manifest`(会话结束
写一次),都不在"同会话并发写"的路径上。这也是控制器给的判断,与 Task 1 实现者的结论一致
(`progress.md:96`)。

**为什么锁放在 store 而不是 `api/app.py`**:三个理由 —— (a) 被守护的是 store 自己的不变量,放在写侧
则**任何调用方**都受保护(以后 face/gesture 若写 transcript 也自动安全);(b) 测试禁用 import `api/app.py`,
锁若写在 app 里就是**没有测试的并发守卫**;(c) 已实测:去掉锁、store 测试立刻红(§5)。

---

## 3. vosk 拆除:三条链

| 链 | 关闭方式 |
|---|---|
| `voice_interaction/api/app.py` | 删 `from vosk import Model, KaldiRecognizer`、`MODEL_PATH`、`if not os.path.exists(...) raise RuntimeError`、模块级 `Model(MODEL_PATH)`;四处识别点换成缝(§1)。`SAMPLE_RATE = 16000` **保留**(四处 wave 格式校验与 ffmpeg 都还在用它)。顺带删掉随之变成未使用的 `import json` 与 (本就是死 import 的) `import threading`,改为 `import asyncio` |
| `voice_interaction/pipeline/speech_recognition_pipeline.py` | 删 `from vosk import ...`、`ROOT_DIR`/`MODEL_PATH`、**import 期的 `RuntimeError`**;`self.model = Model(...)` → `self.engine = FunASREngine()`;两处识别(`recognize_from_audio` 的识别器、`listen_for_speech` 的流式循环)都改走 `self.engine.transcribe_pcm(...)`。**类名、`__init__(self, sample_rate: int = 16000)`、`SpeechRecognitionResult` 返回类型、`listen_for_speech` 的 `(result, ndarray)` 返回形状全部保持** |
| `voice_interaction/__init__.py` | Task 1 的惰性导出(PEP 562)不动,只改 docstring 里那处 `TTS / vosk / librosa` → `TTS / sounddevice / librosa`(否则 ⑧ 的文本守卫会红)。`from voice_interaction import SpeechRecognitionPipeline` **实测可用**(§5) |

另外顺手改了 `voice_interaction/README.md:63` 的 `pip install ... vosk ...` → `websockets`
(装了 2 GB 模型已删的 vosk 是条死指示;`websockets` 是 vendored 客户端的真依赖,原来漏了)。

### 两条验收命令(删除模型**之前**跑,输出原文)

```
$ grep -rn "vosk\|KaldiRecognizer" voice_interaction/ report_frontend/ --include="*.py" | wc -l
0
$ ~/miniconda3/envs/jingxin/bin/python -c "import voice_interaction; print('ok')"
ok
```

删除后复跑同两条,输出不变(0 / ok)。**守卫**:`tests/test_voice_transcribe_helper.py::
test_no_vosk_reference_left_in_voice_package` 把这条文本检查固化成常驻测试,扫描范围与验收命令**同一范围**
(`voice_interaction/` + `report_frontend/`),并钉住"扫描集必须覆盖两个识别点所在文件"(防 glob 写歪后
因"一个文件都没扫到"而假绿)。

### 模型删除

- `rm -rf vosk-model-cn-0.22/`,删前 `du -sh` = **2.0G**;删前确认 `git ls-files` = 0、`git check-ignore` 命中
  `.gitignore:55`(即工作树里没有它的任何 git 记录,删除不可逆但**不影响仓库**)。
- 删后 `ls -d vosk-model-cn-0.22` → 不存在。
- ⚠️ **WSL 内删除不会自动还 Windows 侧磁盘**(`docs/下一步.md:62` 已记):`D:\WSL\ext4.vhdx` 只增不减,
  要回收需在 Windows 侧 `wsl --manage <发行版> --set-sparse true` 或 diskpart compact。**这一步我没做**(在
  Windows 侧,且属用户环境操作),留给你/用户。

### 还没清的 vosk 残留(**不在本任务 scope,故只报不改**)

scope 是 `voice_interaction/` + `tests/`,下面这些在 scope 外:

| 文件 | 内容 |
|---|---|
| `requirements.txt:24` | `vosk>=0.3.45` —— **活依赖声明**,留着会被重新装上;且没列 `websockets`(FunASR 客户端的真依赖) |
| `requirements-full.txt:33` | 同上(该文件**未被跟踪**) |
| `README.md:87,429,490` | 仓库结构图里的 `vosk-model-cn-0.22/` 目录、技术栈里的 "Vosk: 语音识别" / "Vosk" |
| `.gitignore:2,53,55` | Vosk 模型的忽略规则 —— **建议保留**(模型若被重新拷入,忽略规则反而是保护) |
| `docs/下一步.md:47,62` | 计划文档本身,不用改 |

`code_data_supplement/modules/voice_interaction/**` 是另一份副本(自带 vosk 引用),不在活代码路径上。

---

## 4. `examples/` 里发现了什么

**没有任何 vosk 引用**(三个示例文件全文 grep 为空),所以没有一条要改。但有三条值得记的事实:

1. 三个示例(`run_speech_recognition.py` / `run_interview_assessment_voice.py` /
   `run_research_assessment_voice.py`)都 `from voice_interaction.pipeline.speech_recognition_pipeline
   import SpeechRecognitionPipeline`,然后 `SpeechRecognitionPipeline()` + `listen_for_speech(timeout=30,
   pause_threshold=1.2)`,只用 `result.text`。**我保持的正是这一组接口**(类名 / 构造签名 / 返回类型 / 返回
   元组形状),所以它们逐字仍能跑。
2. 两条**行为**变化(它们会感知到):
   - `listen_for_speech` 不再打逐块 `📝 你说的是: '...'`(FunASR 这条链路是整段识别,没有 partial);
   - 静音停机判据从"约 10 s 没出现 partial"变成 `pause_threshold`(示例显式传 1.2 s),`pause_threshold`
     这个签名参数**第一次真正生效**。理由与取舍见下面的 concerns。
     **⚠️ 这一句是错的,已被 §10.1 更正**:那一版的 `pause_threshold` 是**死代码**、永远不触发,实际
     行为是"每次录音都录满 `timeout`"(示例于是每个回答录满 30 s)。审查用假设备证明后,本任务已改
     为按真实能量判静音,`pause_threshold` 现在才真的生效。
3. 三个示例都**绕开了 M1 的 session 契约**:它们直连管道,没有 session_id、不落 transcript、不写连接词密度行。
   它们仍是"能跑的老脚本",不属于 M1 的验收路径(Task 7 走 HTTP 端点)。

---

## 5. RED/GREEN 证据(每条测试:谁把它变红)

新测试 5 条 = `tests/test_voice_transcribe_helper.py` 4 条 + `tests/test_transcript_store.py` 1 条。
基线(改动前)全套 `104 passed`;改动后 `109 passed`(+5)。

### 逐条 RED 归因

| 测试 | RED 现场(实测) | 哪一处生产改动把它变红 |
|---|---|---|
| `test_transcribe_helper_delegates_to_engine_verbatim` | 先:`ImportError: cannot import name 'transcribe' from 'voice_interaction.asr'`(整个模块收集失败)。**变异检验 M1**(缝里 `AsrUtterance(text=u.text.strip(), n_chars=u.n_chars)` 重建对象)→ `1 failed` | ① `asr/transcribe.py` 不存在;或 ② 它的 `transcribe` 做了任何后处理:重建对象(`out is utt` 红)、strip(带空白的原文对不上)、不透传 pcm(`engine.seen` 红) |
| `test_transcribe_helper_does_not_swallow_engine_errors` | **变异检验 M2**(缝里 `try/except Exception: return AsrUtterance(text='')`)→ `1 failed` | 缝里包 `try/except` / 任何"失败就返回空识别"的兜底 |
| `test_transcribe_helper_never_turns_empty_text_into_a_placeholder` | **变异检验 M3**(缝里 `if not u.text: u.text = '0'`)→ `1 failed` | 缝里给空文本补占位(`text or "0"`、`"0"` 默认值之类) |
| `test_no_vosk_reference_left_in_voice_package` | 改动中:创建 `transcribe.py` 后、拆 vosk 前,该测试**单独红**,命中 3 个文件:`{'__init__.py': [25], 'api/app.py': [16,55,56,58,60,142,143,144,201,202,203,281,414], 'pipeline/speech_recognition_pipeline.py': [13,25,28,43,84,105,149]}`(验收命令那条大小写敏感 grep 当时是 12 行) | voice 包内任何 vosk/KaldiRecognizer 文本(含注释与 docstring)残留 |
| `test_concurrent_appends_for_one_session_lose_nothing` | 改动前(store 无锁):`AssertionError: assert ['回答7', '种子'] == ['回答0', ..., '种子']` —— 8 次并发只剩 2 段(丢 7 次回答,静默)。**变异检验 M4**(把 `with _session_lock(...)` 换成 `if True:`)→ `1 failed` | `append_utterance` 里那把 `with _session_lock(session_id):` 被去掉 |

**怎么让并发测试的红是确定的**(不是概率性假红):种子那一次先串行写入(首次写入不读文件,撑不开窗口),
再把 store 读侧的 `json.loads` monkeypatch 成"解析完再睡 50 ms",于是 8 个线程必然都在任何 `os.replace`
之前读完 —— 无锁时"全部读到同一份旧状态"成为必然,而不是巧合。这是对 Task 1 那次"抽样互不相同"1.86% 假红
教训的正面应用(`progress.md:65`)。

### 变异检验汇总(证明测试不空转)

```
RED  ✔  M1 缝里重建对象并 strip(不再是纯转发)        1 failed, 3 passed
RED  ✔  M2 缝里吞掉异常(伪装成空识别)                1 failed, 3 passed
RED  ✔  M3 缝里给空文本补占位 0                       1 failed, 3 passed
RED  ✔  M4 去掉会话锁(读-改-写重新变成裸奔)          1 failed, 15 deselected
变异后工作树已还原: clean
```

### GREEN

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -q
109 passed in 2.32s
```

中间态提醒:拆 vosk 未完成的那一轮,`tests/test_assert_coverage.py::test_no_assert_is_dead` 会**连带**红
(它跑内层 pytest 并要求 `inner_exit_code == 0`)—— 这是该文件自己写明的耦合,不是缺陷;随 vosk 测试转绿
自动恢复。

### 端点层:没写测试,但人工核过一遍(brief 要求"说清而非加重型测试")

端点正确性的正式验收是 Task 7 的真会话。但 `api/app.py` 在本任务前**完全没有测试覆盖**,而这次改动把它
从"import 即崩(缺模型)"改成"import 即接线",所以我在 `/tmp` 写了一次性核验脚本(`/tmp/verify_task4_wiring.py`,
**未进仓库**):桩掉 `python_multipart`、把 `JINGXIN_RECORDINGS_DIR` 指到临时目录、把 `voice_logger` 的
log_dir 指到临时目录、`tts_engine.speak` 置空、`asr/transcribe.py` 的 `asr_engine` 换成假引擎,然后**真调**
四个端点函数。22 项断言全 PASS,摘要:

```
== 缝的接线 ==        app._transcribe 是 asr/transcribe.py 那个函数 / 经缝拿到的就是引擎的输出
== /interview/start == 响应带 session_id+status+question / session.json 落仓库外 /
                      provenance 五键齐 / voice logger 换成带 id 的实例(interview_emotion_log_<sid>.csv)
== /interview/answer_audio == 响应四键齐 / 密度出值 / 原句进仓库外 transcript.json /
                      仓库工作树里没有该文本 / voice 日志首列=session_id /
                      密度+标准差+n_rows 入行 / question_index 0 基
== /asr ==            响应 {text, session_id} / 两次识别累积(段数 2)/ 不给 id 落 NONE
== 空识别 ==          抛 400(不是 500)
== 引擎故障 ==        抛 500(不伪装成"没听清")
== 事件循环 ==        同步缝在循环里直调会炸 / _transcribe_async 可用 / 四处识别点都用了它
== research 侧 ==     响应含 session_id 且原句落盘
FAILURES: 无
```

跑完清掉了它顺手在 `data/logs/` 生成的 7 个**只有表头**的 `interview_emotion_log_NONE_*.csv`
(是 import `api/app.py` 时模块级 `VoiceLogger(...)` 建的,属既有行为;该目录被 gitignore)。

---

## 6. 与 brief 的偏离(逐条给理由)

1. **缝不在 `api/app.py`,在 `asr/transcribe.py`** —— 控制器 Ruling M1-14 直接要求,已照做。
2. **端点是 `await _transcribe_async(audio_data)` 而不是 `utt = _transcribe(audio_data)`** —— Ruling M1-6
   要求;为了不让四处各写一遍 `asyncio.to_thread`,把"loop-safe 的那一次调用"抽成 app 内的
   `_transcribe_async`(3 行),四处都调它。缝本身仍是同步的 `transcribe`。
3. **`_asr_meta()` 从 `load_config()` 取值,不用 `asr_engine.host/.port`**(brief 写的是后者)。理由:
   (a) 与 `transcript_store._asr_meta()` **同源** —— 两处都现读 asr_config.json,session.json 的 asr 块与
   transcript.json 的 asr 块因此不可能给出不同的值(brief 那一版会走引擎属性,理论上可与 store 那份不一致);
   (b) **抗引擎打桩**:Task 7 的验收若把 `asr_engine` 换成假引擎来免起 FunASR 服务,读 `asr_engine.host` 的
   版本会 `AttributeError` → `/interview/start` 直接 500,而这一版照常工作。代价:若将来有人构造
   `FunASREngine(host=...)` 指向别处,manifest 记的仍是配置里的地址(今天两者必然相同)。
4. **`/interview/start` 里 `global voice_logger` + 重建 logger**(brief 只写 `voice_logger.session_id = sid`):
   文件名是 `VoiceLogger.__init__` 算的,事后改 `.session_id` **只换列、不换文件名**;而
   `session.json` 的 `expected_file` 是 `interview_emotion_log_<sid>.csv`(`transcript_store.LOG_PREFIXES`,
   `progress.md:177` 也是这个格式)。不重建的话 `refresh_manifest` 会把 voice 日志标成 `present: false`。
   所以:会话开始重建(文件名带 id),`answer_audio` 里仍保留 brief 那一行 `voice_logger.session_id = sid`(首列随会话)。
5. **`/asr` 的响应加了 `session_id`** —— brief 只说加参数、落转写。加了是"纯加"键:调用方给了 id,应当知道
   这次文本被归到哪一场(尤其缺省落 NONE 时)。若 Task 7 期望 `/asr` 的响应**逐字**只有 `text`,这一键要去掉。
6. **空识别不落盘**(`/asr` 的 `if text:` 守卫;两个 `answer_audio` 是先 400 再落):避免给一场会话留下
   一个 `segments: []` 的空 transcript 文件。`answer_audio` 的落盘时机与 brief 一致(先判空)。
7. **`/research/answer_audio` 也加了 `session_id` + 落转写**(brief 只写"替换识别点"):"原句只出仓库"这条
   不变量对科研侧的音频回答同样成立。但**没给它写密度行** —— `LOG_PREFIXES` 里没有科研侧前缀,密度按
   spec §6.5 只进语音侧那一条日志;科研侧若也要密度,是一个后续决定(见 concerns)。
8. **`question_index` 取 `len(interview_assessment.qa_pairs)`(0 基)** —— `assessment_pipeline.save_log`
   自己就是用 `enumerate(self.qa_pairs)` 写 `question_index` 的(`assessment_pipeline.py:138/394`),所以取
   同一口径。这是跨模块接口,见 concerns。
9. **`listen_for_speech` 的静音判据改成 `pause_threshold`** —— 见 concerns 第 3 条(唯一一处交互行为变化)。

---

## 7. 改动文件

```
 tests/test_transcript_store.py                           |  56 ++++   (加并发丢更新测试)
 tests/test_voice_transcribe_helper.py                    | 121 ++++   (新:缝 3 条 + vosk 守卫 1 条)
 voice_interaction/README.md                              |   2 +-   (依赖行 vosk → websockets)
 voice_interaction/__init__.py                            |   2 +-   (docstring)
 voice_interaction/api/app.py                             | 145 +++--- (四处识别点 + 发号 + 密度 + 缝)
 voice_interaction/asr/transcribe.py                      |  32 ++++  (新:轻缝模块)
 voice_interaction/asr/transcript_store.py                |  83 +++--- (按会话的锁)
 voice_interaction/pipeline/speech_recognition_pipeline.py | 116 ++--- (vosk → FunASR)
 8 files changed, 421 insertions(+), 136 deletions(-)
 (另外:rm -rf vosk-model-cn-0.22/,2.0 GB,未被跟踪)
```

未触碰:`face_expression/`、`gesture_analysis/`、`report_frontend/`、`templates/`、根 `app.py`、`experiments/`、
`asr_config.json`(Task 1 的测试钉着它,一字未动)。没有新装任何包。所有改动文件 LF(逐个验过,含
`tests/` 下两个)。

---

## 8. 自查(逐项)

- [x] 四处识别点全部替换(`grep -n "_transcribe_async(audio_data)"` = 4 行:`177/236/326/475`)
- [x] 全仓无 vosk 残留(验收命令 = 0;文本守卫测试 = 绿)
- [x] 端点原有响应键一字未删:`text` / `status` / `recognized_text` / `question` 都在,新增全是"加键"
- [x] 测试输出 pristine(109 passed,无 warning 转红;**只**有 `LoopBoundEngine` 那条来自我核验脚本的
      `RuntimeWarning: coroutine ... never awaited`,那是脚本故意直调同步缝的副作用,**不在测试套件里**)
- [x] 无空转断言:4 个变异检验逐个把对应测试打红(§5)
- [x] 原句不落仓库:核验脚本专门断言"仓库工作树里没有该文本";落盘一律经 `transcript_store`(默认根
      `~/shared/jingxin_recordings`,即 `D:\Shared\...`,仓库外)
- [x] 阈值/常量仍只在 `asr_config.json`(本次没新增任何阈值;`SAMPLE_RATE` 是音频格式,且原本就在 app 里)
- [x] 精确路径提交、无 amend/rebase、未新建分支;提交后 `git status --short voice_interaction/ tests/` 干净

## 9. Concerns

1. **`/asr` 响应多了 `session_id`** —— 若 Task 7 的验收对 `/asr` 响应做**逐字**相等断言,去掉这一键即可
   (一行)。我判断"给了 id 就回 id"更合理,故留着。
2. **`voice_logger` 是模块级单例、会话开始时被重建** —— 与 `interview_assessment` 一样,整套设计是
   **一次只跑一场面试**。若两场会话交错(先 start A,再 start B,然后 A 的回答迟到),A 的行会写进 B 的
   文件(列仍是 A,文件名是 B)。这是既有设计限制,不是本任务引入的;真正的修法是 logger 按会话现取,
   属 M1 之外。
3. **`listen_for_speech` 的两处行为变化**(交互路径,无测试覆盖):不再有逐块 partial 提示;静音停机从
   "约 10 s"变成 `pause_threshold`(默认 1.2 s)。我没引入任何新阈值 —— 用的正是签名里本来就有的、
   此前从未生效的 `pause_threshold`,并加了"必须已经收到过音频块"守卫,免得开口前就被判"说完了"。
   若要保留 ~10 s 的旧手感,把调用方的 `pause_threshold` 传 10.0 即可(接口没变)。**这是我对唯一一处
   无测试路径做的行为判断,请重点看这一条。**
   **⚠️ 更正(§10.1)**:这一条的**前提就是错的** —— 我写的 `pause_threshold` 判据是死代码,实际没有
   任何行为变化,唯一行为是"录满 timeout";我把"我以为写了的规则"当成了"已生效的行为"来报告。
   现已按能量重写并补了这条路径的第一批测试。本条其余部分(不加新阈值、保留旧手感靠调用方传参)
   在修好后仍然成立。
4. **`question_index` 的 0 基口径必须和 face/gesture(Task 5)对齐** —— 三份日志在报告侧是按
   `question_index` 对齐的。我按 `assessment_pipeline.save_log` 的 `enumerate` 取了 0 基;若 Task 5 取 1 基,
   跨日志对齐会整体错位一位。建议控制器在 T5 派发词里把口径钉死。
5. **`voice_logger.log_prosody({}, ...)` 的 prosody 列全是 0**(brief 指定 `{}`):本任务只接**文本层**的
   连接词密度,音频层的 prosody 特征提取器(`core/feature_extraction/prosody_extractor.py`)没接。后果:
   语音日志每行 `pitch_mean`/`energy_mean`… 都写 0,与"真的测出 0"在数据上不可分。报告层若对语音侧
   prosody 做聚合,这些 0 会把均值拉低 —— **M3 之前需要一条"本行 prosody 未测"的标记**(或干脆不写这些列)。
   密度列(`connective_density` / `_std` / `n_rows`)是**真的**,`None` 与 0.0 也分得清(过短 → 空)。
6. **`session_id` 未经校验就拼进路径**(plan 已记,`progress.md:201` 说要交 T5):本任务把
   `session_id` 变成**客户端可控参数**(`/asr`、两个 `answer_audio`),它们经 `recording_dir()` 拼到
   `~/shared/jingxin_recordings/{sid}/`,即 `"../../x"` 这类值可越出根目录建目录/写文件(是**穿越**,
   不只是崩溃)。`/interview/start` 那条路安全(id 是本机 mint 的)。我**没修**(控制器把这条派给了 T5,
   双写反而容易出现"一边以为另一边修了");最小修法是 `transcript_store.recording_dir` 里加一道
   "id 只能含字母数字下划线连字符"的 2 行守卫 + 1 条测试。**请控制器明确这条归谁。**
   **⚠️ 已闭合(§10.2)**:控制器改判归本任务,守卫已按上述最小修法加上(`validate_session_id`),
   并顺带在三个端点先校验以便回 400 而不是 500。**T5 不必再重复这一条。**
7. **`requirements.txt` 仍声明 `vosk>=0.3.45` 且没有 `websockets`**(根文件,scope 外,只报不改,§3)。
   在有人改它之前,`pip install -r requirements.txt` 会把已删的依赖装回来。
8. **2.0 GB 的磁盘空间在 WSL 里还没真正还回来**(`docs/下一步.md:62`):需在 Windows 侧
   `wsl --manage <发行版> --set-sparse true` 或 diskpart compact。
9. **`api/app.py` 仍无自动化测试** —— 这是控制器的明确取舍(Task 7 验收真会话)。我以一次性核验脚本
   兜了 22 项断言,但它**不进仓库、不会在 CI 里复跑**:`_asr_meta()`、session 发号、密度入行这些接线在
   今后改动中若被弄坏,只有 Task 7 的端到端才能发现。
10. **`_asr_meta()` 的 provenance 块在两处各写一份**(app 与 store)。两者的**取值来源**已被我统一成
    `load_config()`,所以不会给出不同的值;但"契约形状"仍是两份代码。彻底消重需把 store 的私有
    `_asr_meta` 提升为公开访问器(改 Task 1 已审模块),我判断不值得在本任务做 —— `progress.md:86` 也把
    这条记为 deferred minor。

---

# 10. 审查修复(第二轮)

审查结论 **Needs fixes**:1 Critical(打在我的 Ruling M1-18 上 —— 我上一轮"接受 `listen_for_speech`
行为变更"的裁决是错的)+ 2 Important(其中 Important 1 由控制器改判归本任务)+ 12 Minor(记档不修,
其中一条在本节回答)。

提交 **`a4e14b0`**(`fix(m1): 审查修复 —— listen_for_speech 停录判定按能量(原规则是死代码)、
session_id 路径守卫与 query/表单双认`),6 文件 +426/-35;外加 **`7ffd9fd`**
(`test(m1): 假麦克风改消费驱动,停录断言按块数钉死 —— 修掉我自己引入的墙钟假红`,见 §10.5 末)。
测试 **130 passed**(修复前 109;130 = 我的 11 条新增 + Task 5 并发加入的 10 条)。

## 10.1 Critical:`listen_for_speech` 的停录判定是死代码

### 先自己复现(改前 `6377c70`,审查给的假设备复现法)

我的第一个复现**没能测出问题**,因为我把 `pause_threshold` 传成了 0.5 s:块间那 ~0.5 s 的空轮询
恰好攒够 0.5,旧代码就停了。审查的复现用**默认的 1.2 s**,块间空档攒不够 → 永远不停。改成默认值后:

```
当前代码:`listen_for_speech(timeout=3, pause_threshold=1.2)`(真设备节奏 0.5 s/块,说 1.0 s 后静音)
  耗时 2.91s(≈跑满 3 s 的 timeout),缓冲 3.00s 音频(6 块),引擎收到 [96000]
```

审查的判断完全成立:**末尾静音 3.0 s 已远超 1.2 s 阈值,却一次都没停**。

### 根因

旧循环 `if data: silence_sec = 0.0`(`data` 是回调给的块,`sd.RawInputStream` 每 `blocksize=8000`
帧 = 16 kHz 下 0.5 s **必送一块**,静音块是 `b"\x00"*16000`,**非空 → 真值**)。于是 `silence_sec`
在 0 ↔ ~0.5 之间振荡,永远到不了 1.2;唯一的收尾是 `total_time > timeout`。
三个后果(审查列的①②③)全部成立:示例每个回答录满 30 s;`:115-118` 的 docstring 与报告 §9.3 说的是反话;
`pause_threshold` 低于 ~0.5 s 时会截断 —— 而这三条**都没有任何测试**能发现,因为这条路径此前零覆盖。

### 修法(按审查指定的形状)

1. **停录判定抽成纯函数**(`voice_interaction/pipeline/speech_recognition_pipeline.py`):
   ```python
   def chunk_energy(chunk: bytes) -> float        # 归一化 RMS(0..1),空块 → 0.0
   def should_stop(chunk_energies, pause_threshold_s, chunk_seconds, energy_floor) -> bool
   ```
   `should_stop` 的语义:静音 = 能量 **不高于** `energy_floor`;只数**末尾**连续静音段;还没开口
   (没有任何块高于下限)恒为 False(留给 `timeout` 兜底,否则一进循环就返回空);粒度是**一块**
   (`chunk_seconds`),阈值低于一块时等同于一块(**这就是"低于 0.5 s 会截断"的边界**,已写进 docstring
   并由 `test_should_stop_threshold_is_inclusive_and_granular_to_one_chunk` 钉住)。
2. **按真实信号能量判静音**(RMS),不再用"队列有没有数据"。
3. **能量下限进 `voice_interaction/asr/asr_config.json`**:新增 `silence_energy_floor`
   (`value: 0.01` = 归一化 RMS 的 -40 dBFS,`_provisional: true`,basis 写明是**定义性下限、未经本机
   实测**(开发机无麦克风)、现场按实测改这一个数)。代码里没有裸常量:pipeline 在 `__init__` 里
   `float(load_config()["silence_energy_floor"]["value"])`,缺键则在构造时响亮地炸。
   `block_size` 仍留作模块级具名常量 `BLOCKSIZE = 8000`(设备参数而非阈值,`chunk_seconds` 由它算出)。
4. **`pause_threshold` 真正生效**:`time.monotonic()` 计真实时间管 `timeout`,能量判据管静音。
5. **文档同步更正**:`:115-118` 的 docstring 现在写的是**真实语义**(能量判据 + 为什么旧的
   "有数据即有声"是错的 + 一块的粒度/截断边界),报告 §9.3 与本文件 §4 里那句"第一次真正生效"已就地
   标注"这句话是错的,见 §10.1"。

### 修后同一复现

```
改后:`listen_for_speech(timeout=3, pause_threshold=1.2)`(同一段音频)
  耗时 2.01s,缓冲 2.50s 音频(5 块),引擎收到 [80000]     ← 在 1.5 s 末尾静音处停下
```
⚠️ 一个诚实的说明:修后**仍缓冲了 2.5 s**、而不是"说到哪停到哪(1.0 s)",因为静音判定要等
`pause_threshold` 那一整段静音**先进缓冲**才成立 —— 这段静音在音频里是尾随的,引擎也会收到它。
要连这段也切掉,需要 VAD 或引擎侧裁剪,属 M3 范围。

## 10.2 Important 1:客户端可控 `session_id` 的路径穿越(控制器改判归本任务)

`transcript_store.py` 新增公开守卫:

```python
SESSION_ID_PAT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
def validate_session_id(session_id: str) -> str:   # 不匹配抛 ValueError(消息含 session_id 与允许集)
```

`recording_dir()` 里的 `_root(root) / session_id` → `_root(root) / validate_session_id(session_id)`,
所以 `append_utterance` / `ensure_manifest` / `refresh_manifest` 全部自动受保护(守卫放在 store 而不是
端点:任何调用方都覆盖)。三个端点**先**校验再干别的,于是非法 id 回 **400** 而不是等落盘时炸成 500,
而且**不会为它跑一次识别**(核验脚本专门断言了这两点)。`"NONE"` 与 mint 的形状照常放行。

核验(§10.5 脚本):`../../jingxin/LEAKDIR`、`..`、`a/../../b`、`.`、`""` 全部被拒且**什么都没建**;
`NONE`、`20260924_153012_9f3c` 放行。

## 10.3 Important 2:`session_id` 改认 query **或** multipart 表单字段

原先只声明为 query 参数(`:146/311/461`),而请求体是 `multipart/form-data` —— 前端把它当表单字段提交会
**拿 200 并把回答静默归到 `NONE`**。新增 `_resolve_session_id(request, session_id)`:

- 顺序:**query 优先**,没有(或空串)再看表单字段 `session_id`;两者都没有 → `NONE`;
- 取到后立刻 `validate_session_id`(§10.2),非法 → **400**;
- `await request.form()` **不会二次消费请求体**:路由声明了 `File(...)`,FastAPI 在进端点前已解析并缓存
  在同一个 Request 上,这里只是读缓存(`except` 只是兜底非表单请求);
- 三个端点签名加 `request: Request`(FastAPI 特殊参数,不进 OpenAPI 的 query 列表),docstring 写明
  "query 或表单字段都能给"。

**核验方式与它的边界(请转给 T7)**:

- 我在核验脚本里用**桩 Request**(`async def form()` 返回给定表单)直接调 `_resolve_session_id`,覆盖:
  query 采纳 / **表单采纳** / 两者都给时 query 优先 / 都没有 → NONE / 空串 → NONE / `form()` 抛异常 →
  退回 NONE / 非法 id(query 与表单两条路各一)→ 400;并把四个端点用 query 与表单两种位置各真调了一遍。
- ⚠️ **真正的 multipart 解析在本环境里跑不了**:`python-multipart` 没装,Starlette 1.4.1 的
  `MultiPartParser` 直接断言失败(这也是"整个 voice API 在本环境起不来"的既有原因,与本次改动无关;
  我按约束没装任何包)。**T7 做真会话验收前必须先 `pip install python-multipart`**,否则所有
  `File(...)` 端点(不止我改的这三个)都会在路由注册期就 RuntimeError。
- 因此 Important 2 的"端到端表单字段"证据**只能由 T7 提供**;我这边给到的是"解析器逻辑 + 桩表单"这一层。

## 10.4 Minor 的回答:`api/app.py` 把客户端 id 写进 CSV 列,而文件名是 `/interview/start` 定的

**现在还剩多少不一致**:加了 §10.2 的守卫之后,穿越/越界**没有了**;但**归属不一致仍在**,一点没变:

- 列里的 id 来自**本次请求**(`voice_logger.session_id = sid`),文件却是 `/interview/start` 那一刻用
  **当时 mint 的 id** 建的(`interview_emotion_log_<那个id>.csv`,构造函数算好后不再变)。
- 两者不同时(客户端传了一个合法但陈旧的 id、或 `NONE`,或者在两次 `/interview/start` 之间混用 id):
  行会落进**别人名字的文件**里;而 `session.json` 的 `expected_file` 认的是文件名 → `refresh_manifest`
  会把该会话的 voice 日志报成 `present: false`(数据其实在,只是在另一个文件里);
  `transcript.json` 却按**客户端 id** 落进另一个目录 → 同一场会话的两份产物会分家。
- 报告层按**首列**归堆(§T1/§T3 的约定),所以数字不会串到别的会话上;串掉的是"文件级完整性校验 +
  审计轨迹的可核对性",而 M1 的目标正是"三模块对上号"。

**要不要与 `/interview/start` 绑定:要。** 判据是"id 该由谁说了算" —— 今天有**两个**真相来源(客户端传的
id,与进程里那个单例 logger/session 状态),守卫只堵了安全,没堵归属。建议(按推荐度):

1. **服务端为权威**:把当前会话 id 存成显式状态(`/interview/start` 写入),`answer_audio` 只在
   客户端 id **等于**当前会话时接受,不等则 **409**(M1 的契约本来就是"一次一场面试"—— 全局单例
   `interview_assessment` 已经这么假设了);等于把"对上号"变成服务端可证的,而不是客户端自称的。
2. 或者反过来:接受客户端 id 并**按该 id 现取/现建 logger**(文件与列必然一致,manifest 也按该 id 现写),
   代价是"一场会话一个文件"要改成"按 id 分文件 + 首次见到即建档"。
3. 维持现状 + 把不一致写进文档(不可取:它是静默的)。

我**没有动手改**(按指令只给判断)。另外提醒:**T7 的验收大概率不会碰到它**(用一致 id 跑就不会),
所以它不会在验收里自己暴露。

## 10.5 RED/GREEN 与变异证据(本轮新增 8 条测试)

新增 `tests/test_speech_recognition_pipeline.py`(8 条)+ `tests/test_transcript_store.py`(3 条守卫)。
RED 现场的原文(改前 `6377c70`,逐条对应"哪一处生产改动把它变红"):

| 测试 | RED 现场 | 被改坏的那一处 |
|---|---|---|
| `test_chunk_energy_measures_signal_level_not_presence` | `AttributeError: ... has no attribute 'chunk_energy'`;**变异 M6**(`return 1.0 非空即有声`)→ 红 | 新增的 `chunk_energy`(按 RMS 而不是"块非空") |
| `test_should_stop_when_trailing_silence_reaches_threshold` | `AttributeError: ... 'should_stop'`;**变异 M5**(退回"收到块就算还在说",即 `return False`)→ 红 | 末尾连续静音时长的计算 |
| `test_should_stop_ignores_a_brief_pause_inside_speech` | 同上(未实现);**变异**:去掉阈值比较(有静音块就停)→ 红 | 只数**末尾**一段、且要与阈值比较 |
| `test_should_stop_threshold_is_inclusive_and_granular_to_one_chunk` | 同上(未实现) | ≥ 而非 >;粒度为一块 |
| `test_should_stop_never_fires_before_the_user_has_spoken` | 同上(未实现);**变异 M7**(删掉"曾经有声"前置判断)→ 红 | `should_stop` 的前置判断 |
| `test_listen_for_speech_stops_on_trailing_silence_instead_of_running_to_timeout` | `AssertionError: 没在静音处停下,录了 2.35s(≈timeout)` —— **旧实现真跑满 timeout**(审查 Critical 的直接复现) | 循环:用 `chunk_energy`+`should_stop` 判停,而不是"队列里有数据就重置计时" |
| `test_listen_for_speech_quiet_room_hiss_does_not_count_as_speech` | `AssertionError: 底噪被当成说话,录了 1.81s` —— 旧实现把低幅块当有声;**变异 M6** → 红 | 能量下限(下限之下的块算静音) |
| `test_silence_energy_floor_comes_from_asr_config` | `AttributeError: ... has no attribute 'load_config'`(**当时 pipeline 没接配置**);**变异 M8**(`self.silence_energy_floor = 0.01` 写死)→ 红 | 从 `asr_config.json` 读下限(而非裸常量) |
| `test_recording_dir_rejects_ids_that_escape_the_root` | `Failed: DID NOT RAISE ValueError`(守卫未实现 → 真建了越界目录) | `recording_dir` 的 `validate_session_id` |
| `test_append_utterance_rejects_traversal_id` | 同上 —— **端点真正走到的那条路** | 同上 |
| `test_none_session_id_is_accepted` | `AttributeError: ... 'validate_session_id'` | 守卫允许 `NONE`/mint 形状 |

变异检验(**v2 方法**:每步 purge `__pycache__`,并在还原后再跑一次要求绿 —— 见下面的教训):

```
RED  ✔ M5 should_stop 退回『收到块就算还在说』      → 2 failed      | 还原后绿 ✔
RED  ✔ M6 chunk_energy 退回『非空即有声』          → 2 failed      | 还原后绿 ✔
RED  ✔ M7 去掉『还没开口不停』的前置判断            → 1 failed      | 还原后绿 ✔
RED  ✔ M8 能量下限写死(不读 asr_config)           → 1 failed      | 还原后绿 ✔
RED  ✔ M9 去掉 recording_dir 的 session_id 守卫     → 2 failed      | 还原后绿 ✔
(复检上一轮的 M1–M4:同样 RED + 还原后绿,结论不变)
```

### ⚠️ 方法学教训(值得进项目的账本):`git checkout` 还原会留下过期 `.pyc`

本轮我第一次跑完全套后,**全绿变 120 passed** 之前出现过一次假红:两个守卫测试报
`DID NOT RAISE ValueError`,而 `validate_session_id("..")` 明明抛了。查明原因**不是代码**:
变异脚本"写入变异 → 跑测试 → `git checkout` 还原"发生在**同一秒**内,`__pycache__` 里那份
**变异版**字节码的 `(mtime, size)` 与被还原的源文件对得上,于是 Python 认为缓存有效、继续用
**变异后的 `recording_dir`**。purge 掉 `voice_interaction/**/__pycache__` 与 `tests/**/__pycache__`
后,守卫立刻生效、全套 120 passed,`git status` 干净。

影响面与处置:(a) 结论是"还原没生效",**不是**提交内容有问题(HEAD 与工作树都带守卫,已核对);
(b) 我因此把**上一轮的 M1–M4 也按 v2 方法重跑了一遍**,结论不变(见上);
(c) 变异脚本从此**每步 purge + 还原后必须绿**,否则不许采信;
(d) 这也解释了为什么"单独跑 store 测试绿、跑全套红"—— 与本仓库源码无关,是缓存。
**给后续任务的建议**:任何"编辑/还原后立刻跑测试"的自动化(包括本仓的变异脚本)都应在跑之前
purge `__pycache__`,或直接 `PYTHONDONTWRITEBYTECODE=1`。

### ⚠️ 方法学教训 2(我自己的测试曾经是**假红源**):墙钟断言 + 全套重跑

上一条提交(`a4e14b0`)里那两条 `listen_for_speech` 测试,我用的是"假设备按 0.2 s 节奏喂块,
断言 `elapsed < 1.5 / 2.0`"。在**全套**里跑时**偶发假红** —— 实测两次全套红一次
(`test_listen_for_speech_quiet_room_hiss_...`),而且是 `test_assert_coverage` 的内层
`settrace` 重跑 + 并发负载(当时 Task 5 的 agent 也在跑测试)把它拖慢所致。这正是本项目
已经栽过一次的那类断言(4 位 id 的生日问题)。

修法(`7ffd9fd`)把断言从"时间"换成"块数":

- 假设备改成**消费驱动**(队列空才补下一块,`TracingQueue` 记下队列实例),于是"缓冲了几块"
  成为与机器快慢**无关**的确定量;
- 断言钉成窄区间(修好后 5–7 块 / 4–6 块;旧实现是 32 / 21 块),墙钟只留一条 **15 s timeout
  下的宽松上界 `elapsed < 5.0`**(期望值 ~0.1 s,50× 余量);
- `timeout` 从 3/2 放大到 15,专门为这条上界留余量。

复跑证据:全套连跑 4 次全绿(4.86–5.07 s);变异 M5/M6 在新断言下仍 RED(见下表);
一套 call 级 `settrace` 压力下 8 条仍全绿(141k 次调用被追踪)。**副作用:全套从 7.3–9.9 s 降到 4.9 s**
(去掉了喂块的节拍 sleep)。

## 10.6 端点层核验(未进仓库,与上一轮同样的方式)

`/tmp/verify_task4_fixes.py`,桩掉 `python_multipart`、把 `JINGXIN_RECORDINGS_DIR` 与 logger 的
`log_dir` 指到临时目录、假引擎、桩 Request,**32/32 PASS**(0 FAIL):

- ① 两个位置 10 条(含"query 优先""非法 id → 400"×4)
- ② 守卫 8 条(5 个非法值被拒 + 没建越界目录 + `NONE`/mint 放行)
- ③ 端点仍照常 7 条(发号、query 给 id、**表单**给 id、两次回答落同一 transcript、`/asr` 两种位置、
  research 侧)
- ④ 非法 id 时端点 400 且**没跑识别、没建目录** 3 条
- ⑤ 回归 3 条(空识别 400 / 引擎故障 500 / `_transcribe_async` 在事件循环里可用)

跑完清掉了 `data/logs/` 下脚本产生的空日志文件(该目录被 gitignore),临时目录已删。

## 10.7 本轮改动文件

```
 tests/test_speech_recognition_pipeline.py          | 234 ++++++++++++++++++  (新:8 条)
 tests/test_transcript_store.py                     |  40 ++++              (3 条守卫)
 voice_interaction/api/app.py                       |  64 +++++-            (_resolve_session_id + 三端点)
 voice_interaction/asr/asr_config.json              |   5 +                (silence_energy_floor)
 voice_interaction/asr/transcript_store.py          |  24 ++-               (validate_session_id + 守卫)
 voice_interaction/pipeline/speech_recognition_pipeline.py | 94 +++++++--   (chunk_energy + should_stop + 循环)
```

## 10.8 本轮新增/变化的 concerns

1. **修好后仍会把尾随静音喂给引擎**(§10.1):缓冲里含 `pause_threshold` 那一整段静音。要切干净得靠
   VAD 或引擎侧裁剪,属 M3;本任务只保证"不再录满 timeout"。
2. **`silence_energy_floor = 0.01` 未经本机实测**(开发机无麦克风):标了 `_provisional` 并写明依据。
   现场若录音增益偏低(说话 RMS 也低于 0.01),表现是"录满 timeout"这一症状**复发** —— 第一个要查的
   就是这一个数(`asr_config.json`,不要在代码里改)。
3. **`python-multipart` 未安装**(§10.3):整个 voice API 在本环境无法处理 multipart 请求,
   **T7 验收前必须装**(我不能装)。这也意味着 Important 2 的端到端证据要 T7 补。
4. **归属不一致仍在**(§10.4):建议按方案 1(服务端为权威,不匹配回 409)在 T5/T7 之前定下来。
5. ~~**`test_speech_recognition_pipeline.py` 有两条测试按时间跑**(各 ~1 s):全套从 2.3 s 涨到 ~7.3 s~~
   **已解决(§10.5 末)**:改成消费驱动后不再依赖墙钟,全套 4.9 s,且不再偶发假红。
