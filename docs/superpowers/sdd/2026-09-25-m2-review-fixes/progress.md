# SDD ledger — plan: docs/superpowers/plans/2026-09-25-m2-review-fixes.md

> 执行方式:inline(使用者 2026-09-25 裁定)。工作区 `.superpowers/sdd/2026-09-25-m2-review-fixes/`
> 里的 `progress.md` 是本文件(软链)—— 账本本体放在 **git 跟踪**的目录里,承 M2 的教训:
> 上一轮复审报告落在 gitignore 的目录里差点丢。

## Pre-flight scan(2026-09-25,开工前)

**共享接口(Interfaces 块)**:

| 行 | 生产者 → 消费者 | 结论 |
|---|---|---|
| 1 | Task 1 `LogDataLoader.target_session` → Task 2 `loader.target_session` | 名字一致,无冲突 |
| 2 | Task 1 `sources_disclosure(sources, target=None)` → 无跨任务消费者(Task 2 把 `selected_sessions` 原样交给前端,前端自己渲染标签) | 无冲突 |
| 3 | 前端 `src/services/api.ts` 被 Task 2 与 Task 3 **同一个文件**改到 | 非接口冲突,是**顺序约束**:Task 3 必须排在 Task 2 之后(基于 Task 2 的版本改) |

**结论**:1 处共享接口 + 1 处顺序约束,无冲突,可开工。

## 环境事实(实测,供后续任务引用)

- 解释器固定 `~/miniconda3/envs/jingxin/bin/python`;开工基线 `main` = **242f512**(§8 提交后),开工前 **204 测试全过**。
- `from voice_interaction.api import app` → 拿到的是 **FastAPI 实例**(包 `__init__.py` 遮蔽了子模块属性);要拿模块得用 `importlib.import_module("voice_interaction.api.app")`。
- `/research/start` 现状(2026-09-25 12:35 实测):`{'status': 'started', 'question': '请描述一个你深入研究过的技术…'}`,**没有 session_id** —— 第 19 条当场复现。
- 空数据下渲染链不崩(实测 `features={}` → 覆盖 `0/20`、五维全 `None`、雷达+5 张证据图照常生成)。
- 验收语音样本:`~/asr-test/zijijieshao.wav`(16k/单声道/16bit),截 12 秒能转出真中文;`espeak-ng` 合成的假语音 ASR 结果为 `''`,不可用。

## 任务进度

### Task 1(第 17 条 / 复审 I2)

- Task 1: Ruling: 计划 §Task1 Step1 的断言消息里内嵌了双引号(`"只有 NONE 桶时没说清"本场根本没有日志""`)—— 那是 **SyntaxError**,收集阶段就会炸。改用中文引号「」。 — cost if wrong: 仅测试措辞,零。
- Task 1: Ruling: 夹具 `files=[]` 时临时日志目录根本没被创建,而加载器**按设计要求**要求 `data/logs` 存在 → 测的成了"目录不存在"而不是"一份日志都没有"。在夹具里无条件 `mkdir`(断言强度不变)。 — cost if wrong: 会把"目录缺失"误当成"本场没有日志"。
- Task 1: Ruling: 顺手更正两处**陈旧注解** `Dict[str, str]` → `Dict[str, Dict]`(`data_loader.selected_sessions` 与 `_build_html_report` 的 `sources` 形参)。M2 把值从 `str` 改成三态 `dict` 时没跟上注解,新装的 pyright 插件当场标出。 — cost if wrong: 零(纯注解)。
- Task 1: Ruling: 计划 Expected 写「6 passed」,实际 **7** —— 自查阶段我加的 `test_report_is_still_generated_when_there_are_no_logs_at_all` 没同步改数字。以实际为准。 — cost if wrong: 零。
- Task 1: 反向复现(两次,均逐条核对):
  - 变异 1:恢复 `if not data or 'face' not in data: raise ValueError("无面部数据")` → **恰好 3 条红**(`..._only_the_none_bucket_exists` / `..._there_are_no_logs_at_all` / `..._the_named_session_has_no_logs`),其余 4 条仍绿,报错文本 `❌ 错误：无面部数据`。
  - 变异 2:`main` 恒返回 0 → **恰好 1 条红**(`test_cli_exits_nonzero_when_no_report_was_written`,`where 0 = main([])`),其余 6 条绿。
  - 两处均已恢复,恢复后 7 passed + 报告侧既有 44 passed;`find … -name __pycache__ -exec rm -rf` 在每次变异前都跑过。
Task 1: complete (commits 242f512..6e63202, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest -q → 211 passed in 25.28s)

### Task 2(第 18 条 / 复审 I3)

- Task 2: Ruling: 计划里写的 Flask 夹具补丁点是 `report_frontend.data_loader.LogDataLoader`(端点函数体内 `from … import`,每次调用重查模块属性)—— 实测这条路对,补丁打在 `app` 上无效。已在计划文本与测试里都写明。 — cost if wrong: 端点会去读真实的 `data/logs`,测试不隔离。
- Task 2: Ruling: 夹具里 `log_dir.mkdir(parents=True, exist_ok=True)` 无条件建目录(承 Task 1 同一裁决:加载器要求目录存在)。 — cost if wrong: 同 Task 1。
- Task 2: Ruling: 我自己新写的 `_row(sid: str, density=2.5)` 被 pyright 标「None 不能赋给 float」(测试里确实传了 `density=None`)。注解改 `float | None`。 — cost if wrong: 零。
- Task 2: 反向复现(两次,逐条核对):
  - 变异 1:删掉 JSON 里的 `"session_id": loader.target_session` → **3 条全红**(三条都断言返回体里有 `session_id`)。
  - 变异 2:恢复 `if not data: return jsonify({"status":"error",…})` 早退 → **恰好 2 条红**(`..._the_target_has_no_logs` / `..._only_the_none_bucket_says_so`),`..._names_the_session_it_describes` **仍绿** —— 正是"空数据不再报错"这条契约把二者分开的证据。
  - 恢复后 3 passed;每次变异前清过 `__pycache__`。
- Task 2: 前端构建门 `npm run build` → ✓ built in 13.87s(只有既有的 `GazeHeatmap` 4.7 MB chunk 警告)。**构建产物核过**:`dist/assets/api-021b99b1.js` 里 `session_id=` 在、`report/structured`,<override> 的拼参在;`dist/assets/ReportPage-*.js` 里有「本场会话」。
Task 2: complete (commits 6e63202..88a1934, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest -q → 214 passed in 20.64s)

### Task 3(第 19 条 / 复审 I4)

- Task 3: Ruling: **推翻 Ruling M1-14**(测试不许 import `voice_interaction.api.app`)—— 该裁定的前提"一 import 就因无关原因失败"在 2026-09-25 实测**不成立**(import 0.4 s 成功,TTS 后台线程正常起)。使用者审阅计划时已知悉并认可。代价:若判断有误,测试进程会多起一个 TTS 后台线程。 — cost if wrong: 测试进程里多一个 TTS 线程/听不到发声(本任务已把 `tts_engine.speak` 换掉)。
- Task 3: Ruling: 一个 `pgrep -f "voice_interaction.api.app" | xargs kill` 把**执行该命令的 shell 自己**也匹配上了(命令行文本里就含这个模式),自杀 exit 144。改用 `pgrep -af` 查看 + 在 Python 侧做变异。与代码无关,记下以免后人重踩。 — cost if wrong: 零。
- Task 3: 自动化测试红→绿:实现前 3 条全红,红因与 brief 的 Expected 逐字一致(`AssertionError: 科研评估没有铸号 —— 返回体是 {'status': 'started', 'question': '…'}` + 两处 `KeyError: 'session_id'`);实现后 3 passed。
- Task 3: 反向复现(实施后):把 `/research/start` 退回「不铸号」(精确匹配科研那一处,不碰 `/interview/start` 的同形代码)→ **3 条全红**,报错文本与修前逐字相同;恢复后 `grep -c "session_mod.new_session_id()"` = 2 且全量绿。
- Task 3: **端到端验收(HTTP 层唯一的强证据)**——`bash ~/shared/m21_acceptance.sh`,真服务(:8001)+ 局域网 FunASR + 真录音 `~/asr-test/zijijieshao.wav` 截 12 秒。原始输出:
  ```
  面试会话 = 20260925_124346_ff93
  科研会话 = 20260925_124346_6e4a
  --- ① 用**科研自己的号**提交语音回答(修好之后前端的行为)---
  {"status":"success","session_id":"20260925_124346_6e4a","recognized_text":"获取大数据的大三本科生。…"}
  ✅ 科研回答落在科研号的目录
  ✅ 面试目录没被写入
  --- ② 反向复现:把**上一场面试的号**喂给科研回答(修之前前端的行为)---
  {"status":"success","session_id":"20260925_124346_ff93","recognized_text":"获取大数据的大三本科生。…"}
  ✅ 复现成功:带旧号时科研回答确实会写进面试目录(这就是修掉的那条路)
  M2.1 验收通过
  ```
  结论:失效场景 B **真实存在且已被修掉**——同一个音频,带科研号只落科研目录,带旧面试号就写进面试目录。
- Task 3: 前端构建门 `npm run build` → ✓ built in 12.94s;产物核验 `dist/assets/api-*.js` 里有 `research/start`);return rt(((t=e.data)==null?void 0:t.session_id)??null),e.data}` —— 存号那行进了 bundle。
Task 3: complete (commits 88a1934..c17c0ac, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest -q → 217 passed in 18.89s)

### Task 4(收尾)

- Task 4: Ruling: 计划 Expected 又算错一次(「204 + 12 = 216」,实际新增 **13** 条 → 217)。以实际为准。 — cost if wrong: 零。
- Task 4: **跨任务集成点自查**(本项目第 5 条教训:T3 改文件名没把加载器正则列进改动面,报告路径什么都加载不到)。逐条核过,**无断链**:
  - `sources_disclosure` 生产调用点只有 **1 处**(`report_generator.py:345` 的 `_build_html_report` 内),批处理与实时两条路都经它,两处都传了 `target`。
  - `selected_sessions` 消费点 = `app.py:248`(新增,值是纯 `str/int` 的 dict,JSON 可序列化 ✓)+ `report_generator.py:145` ✓。
  - `report_generator` 的 CLI 被两处跑:`app.py:120` 经 `module_map["report"]`(看退出码 ✓)、`~/shared/t7_acceptance.sh:162` 与 `m2_acceptance.sh:62`。**旧脚本不会被我改的退出码误伤**:`t7_acceptance.sh:9` 明确不 `set -e`,且两处都管道给 `tail`/`grep`(退出码被管道末位吞掉)✓。
  - 前端:`getStructuredReport` 消费点 `ReportPage.tsx:43`(已传路由 id)、`runModule` 消费点 `useAssessment.ts:19/24/114`(经 `withSession` 缺省自动带当前号)✓。`useAssessment.ts:29` 的 `voiceApiService.start()` 对两种 type 都成立,科研那半靠 `api.ts` 里存号 ✓。

### 本轮明确未动(使用者已在计划评审时知悉)

- **前端 `ReportPage.tsx:92-95` 仍渲染 `total_score` + 档位标签**(`getLevelLabel`)—— 报告层按 spec §5.4/§5.6 早已停止渲染这两个东西(HTML 报告里不印),前端还在印。这是"把未标定标尺上的复合点分当对候选人的评定",与"让系统停止说谎"同轴,但属前端呈现层的独立决定(与 §3 第 21–23 条同族)。**已写进计划「本轮明确不做」,交使用者裁定。**
- `voice_interaction/api/app.py` 的一批**既有陈旧注解**(`session_id: str = None` 等 5 处、`log_assessment(evaluation_result=str)` 2 处)—— 新装的 pyright 插件标出,但与本轮三条裁定无关,未动(与 Task 1 里我只修自己碰到的那两处注解同一标准)。
- M2 复审的 Minor:`_none_bucket_rows` 每出一次报告全量读 NONE 文件、`resolve_target_session` 同秒并列取定不确定、`sources_disclosure` 的"混合 id"断言、`~/shared/m2_acceptance.sh` 两个卡死 bug、`socket.io-client` 死依赖、报告入口收口、OpenAPI→TS 类型。
- §3 第 4/5/6/9/11/12/14/15/16 条(进程级单例、409 契约、实时路径丢指标、静音下限实测、NONE 桶轮转、客户端自定义 id 不可见、`text_avg_length` 无产出方、VIDEO 等价性门、M3 阈值重登记)。

## 全量门

### 本轮覆盖力的**已知限制**(写给复审者与实际使用者)

- **前端没有任何自动化测试基础设施**(实测 `~/JingXin-frontend/package.json`:scripts 只有 `dev`/`build`/`preview`,依赖里无 vitest/jest/testing-library/playwright/cypress)。因此:
  - Task 2 的**前端**半边(两条读路径带 id、披露卡)只有 `npm run build`(= `tsc && vite build`,能挡类型错)+ **构建产物核验**(`dist/assets/api-*.js` 里确认拼参与 `session_id` 在)两道,**没有运行期行为的自动化覆盖**。
  - Task 3 的**前端**半边(`research.start()` 存号)额外被端到端验收**间接**压住 —— 验收脚本模拟的正是修好之后前端的行为(拿科研号提交)。但"前端确实把号存下来了"这件事本身只有构建产物核验。
  - 后端半边都有 pytest 覆盖(Task 1: 7 条;Task 2: 3 条;Task 3: 3 条)。
- 浏览器那半(真点界面)本轮**没有跑** —— 与 M2 复审时同样的处境(那一轮也把"浏览器那半只能你做"写进了报告)。要补的话,`example-skills:webapp-testing`(Playwright)可以驱动,但那需要另起全套 5 个服务 + 浏览器,超出本轮计划范围,交使用者决定。

- 全量门①:`~/miniconda3/envs/jingxin/bin/python -m pytest -q` → **217 passed in 18.26s**(开工前 204;新增 13 条:Task1 七条 + Task2 三条 + Task3 三条)。
- 全量门②(合并门):`reaggregate_normalized.py --stats legacy` + `--verify-legacy /tmp/legacy_probe`,原始输出:
  ```
  ✅ 特征名集合完全一致(原 1410 / 新 1410)
  ✅ 视频集合一致 (2011 个)
  ✅ 有限性完全一致(无 NaN/Inf 分歧)
  最大绝对差 1.886e-19   最大相对差 1.886e-19
  相对差 > 1e-4 的格子: 0 / 2835510
  targets 最大绝对差 0.000e+00
  ✅ 回归验证通过:legacy 模式能逐格复现现有特征矩阵
  GATE_EXIT=0
  ```
  (与 2026-09-21 那次基线的最大绝对差 `1.886e-19` **同一个数** —— 逐位一致。)
- 验收留下的现场(供你复核,未清理):仓库外 `~/shared/jingxin_recordings/20260925_124346_6e4a/`(科研,含 transcript.json)、`..._ff93/`(面试,**被反向复现故意污染过**的那一份);仓库内 `data/logs/interview_emotion_log_20260925_124346_ff93.csv`(292 B,只有表头)—— 后者由 `.gitignore:45` 的 `**/data/logs/` 兜住,不进仓库。
Task 4: complete (commits c17c0ac..5faba37, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest -q → 217 passed in 18.85s)

## 最终独立评审(2026-09-25,最强模型,单一 fresh context)

评审者真跑了:全量 `pytest` **217 passed**、合并门 `0 / 2835510`、前端 `tsc --noEmit` exit 0,并写了四个只读探针把 Review Focus 五条逐条实测。判 **Critical 1 / Important 3 / Minor 9**,结论「With fixes」。原始报告已由本会话留存。

### 重新定级(按"合理使用者会得到什么",不按 spec 有没有点名)

- **C1(前端 `total_score=null` 渲染期抛异常)= Critical,确认,已修。** 我自己独立复核过前提:盘上「最新一场」`20260925_124346_ff93` → `n_passed: 0` → `total_score: None`(经真 HTTP 从面板取出)。**红/绿都是真浏览器实跑**,不是推理:
  - 红:web-access 驱动真 Edge 打开 `http://localhost:5173/report/latest` → 页面文本 `页面出现错误 / Cannot read properties of null (reading 'toFixed')`;
  - 绿:同一 tab 重载 → `— / 未产出综合评分（证据不足）` + 披露卡（`本场会话：20260925_124346_ff93` / 四模态三态 / `另有 NONE 桶 224 行`）正常渲染,雷达与分维度段都在。
  - ⚠️ **前端没有测试设施**,所以这次修复**没有单测** —— 证据是真浏览器实跑 + `tsc --noEmit` exit 0 + `npm run build` ✓。这是本轮唯一一处"修复未经单测"的,记在这里不藏。
- **I1(store 短路让披露卡说谎)= Important,已修。** 但评审者把它算成"本轮未记录的设计决定"**是错的**:那段 `if (storeResult) { setReport(storeResult); return; }` 在我改之前就在(`ReportPage.tsx:23-27`),我没动它 —— 我给他的是**修正**而非承认。真正的缺陷是我加的卡片在那条路上会印出「（无 —— 本场没有任何日志）**而报告就在屏上**;修法是只在服务端真回答了来源时才渲染它(`sources !== null`)。 — cost if wrong: store 路径下用户看不到"这是哪一场"(而那是诚实:前端确实不知道)。
- **I2(科研会话遮蔽面试会话)= Important,裁决:本轮不修,交使用者。** 它是**方案 A 已知悉的后果**(选择 A 时我明确写过"科研期间摄头帧归到科研号 → 缺省取最新会落到科研那场"),**但评审者指出的一半是新的**:科研会话**永远读不出语音**(`research_logger = VoiceLogger(log_type='research')` 不带 session_id,且科研回答不调 `log_prosody`)→ 那份报告恒定"语音两栏全缺"。 — cost if wrong:使用者「先面试 → 再科研 → 看报告」拿到的是科研会话那份偏空的报告,而面试那份完整报告不被提及(报告头点名了是哪一场,不是静默)。**建议收口(评审者方案 a):给科研的 VoiceLogger 带上 session_id**,顺带让 `voice_research` 这个恒 missing 的模态有产出方。
- **I3(实时报告路径「没产出也报成功」)= Important,已修。** 与第 17 条逐字同族的隔壁一条路,一行 + 3 条测试。
- **Minor 1 / 3 / 4 / 5 已修**(docstring 里多余参数名;补"不许回退拼别的场次"的钉子;**该钉子做了反向复现**:让 `resolve_target_session` 回退到最新一场 → 新测试红(`assert '20260924_230914_262f' in html`)、而**旧的那条仍是绿的** —— 正是评审者说的"点了名没压住";补 `n_slots == 20`;`webbrowser.open` 移出主 `try`,它失败不许把已落盘的报告变成「任务失败」)。
- **Minor 2 / 6 / 9 未修**(none_bucket 的 `status: "present"` 是枚举外第 4 值;`_none_bucket_rows` 全量读;只有表头的 NONE 文件不被提),列进 deferred。
- **Minor 7 / 8 已处理**:账本末行已提交;`docs/下一步.md` 里"还没推 origin"已按实际改写。
- **评审者的 Declined to judge 各条**:我逐条认领,见文末「裁决清单」。

### 修复波次后的门

- 全量 `pytest -q` → **222 passed**(217 + 新增 5 条:3 条实时报告任务状态 + 2 条报告层)。
- 前端 `tsc --noEmit` exit 0;`npm run build` ✓ 14.19s。
- 合并门重跑(修复波次动了生产代码,故必须重跑):结果见下。
- **反向复现(修复波次的两处新增)**:①I3 先退回"无条件 success" → 红(`没产出却报成功:{'status': 'success', ...}`);②`webbrowser.open` 失败 → 红(`报告已经落盘,却因为打不开浏览器被判成了失败`);③RF3 的新钉子 → 见上。

### 环境事实(使用者 2026-09-25 现场指出,供后人)

- **WSL 侧的浏览器调摄像头不正常 → 凡是要真发帧的验收(face/gesture 帧、"先面试再科研"的真实场景)都要在 Windows 端浏览器里做。** 这也解释了本轮浏览器验证里 `面部 · 缺失 / 手势 · 缺失` 的现象:那是在 WSL 浏览器会话里跑的,**不该期待有帧**。我先前根据 `sources` 里有 `face`/`gesture` 两个**键**就断言"科研会话确实拿到了摄像头帧"—— 那句是错的(键恒存在,missing 也在),此处更正。

- 合并门重跑(修复波次之后):
  ```
  ✅ 特征名集合完全一致 / ✅ 视频集合一致 (2011 个) / ✅ 有限性完全一致
  最大绝对差 1.886e-19   最大相对差 1.886e-19
  相对差 > 1e-4 的格子: 0 / 2835510
  ✅ 回归验证通过:legacy 模式能逐格复现现有特征矩阵
  GATE_EXIT=0
  ```
- 提交与推送:`3fb01a0`(后端修复波次)/ `16964fe`(前端修复波次);两仓 `main` 均已与 `origin/main` 同步。

- 复审结论已归档到本目录的 `final-review-report.md`(要点版;逐字报告只在会话记录里)。承 M2 的教训:报告放 **git 跟踪**的目录,不放 gitignore 的工作区。
- 本计划的工作区 `.superpowers/sdd/2026-09-25-m2-review-fixes/` 已删除(git 历史即记录;两份 diff 可由 `git diff 242f512..5faba37` 与前端两个提交复现)。

---

## 追补(使用者 2026-09-25 授权「按你的倾向推荐和最优化方案进行操作」)

### ① I2 收口:科研会话变成**读侧可描述**的一场

**改动**(全在 `voice_interaction/api/app.py`):
- 模块级 `research_logger = VoiceLogger(log_type='research')`(与面试侧对称);
- `/research/start` 铸号后**带号重建**它(文件名在构造函数里定 —— 事后改 `.session_id` 只换列不换名,M1 记过这个坑);
- `/research/answer_audio` 与面试侧**同口径**写一条 L0 特征行(`connective_density`,过短/空 → None 不写 0);
- `/research/evaluation` 改用那个带号的,不再现造一个不带号的(否则评估行落 `..._NONE_<时间戳>.csv`,与特征行分家)。

**测试**:新建 `tests/test_research_session_logging.py` 2 条。红因逐字:
`AssertionError: 科研会话没有产出特征行 —— 盘上是 []`、
`AssertionError: 评估行没落到本会话文件:…/research_emotion_log_NONE_20260925_132919.csv`。

**反向复现**:退回「不调 log_prosody + logger 不带号」→ **2 条全红**,与上面同形;恢复后 2 passed。
(变异时往**真实** `data/logs/` 写了一个空文件 —— 模块级 logger 在 import 时就绑定了真目录。已删;这也正是下面那条测试卫生问题的活样本。)

**端到端验收**(`~/shared/m21_acceptance.sh` 扩了 ③④,真服务 + 局域网 FunASR + 真录音):
```
✅ 科研特征行落在 20260925_133215_ba6c 名下   ✅ 特征行含连接词密度列   ✅ 首列是本次会话号
   voice_research → {'session_id': '20260925_133215_ba6c', 'status': 'loaded', 'rows': 1}
✅ 报告侧把科研模态读成活的了
M2.1 验收通过
```
修之前这在**结构上不可能**:科研特征行恒落 `research_emotion_log_NONE_<时间戳>.csv`,匹配不上任何 session id。
顺带**修了验收脚本自己的 bug**:第 ④ 步用 `$PY - <<PYEOF` 跑 `from report_frontend…`,**依赖 cwd 是仓库根** —— 从 `~/shared` 直接跑就 `ModuleNotFoundError`,而前几步是纯 curl 所以照常绿,"看起来只是第④步失败"。改成子 shell `cd` 掉,并**从一个非仓库目录重跑**证明与 cwd 无关。

### ② 前端不再印「综合评分 + 档位」

- `ReportPage.tsx`:去掉 `total_score` / `getLevelLabel` 渲染,改为「**N / N 个指标槽通过证据门** + 置信度上限 + **综合总结**」——与 HTML 报告同一口径(spec §5.4/§5.6),也不再与载荷自述的「未产出综合评分,也不给评级」自相矛盾。
- `RadarChart.tsx`:**删掉写死的「常模基准」**(`r: [60,60,60,60,60,60]`,五个维度全是常量 60,而本系统**没有任何常模样本**)—— 把它画成"基准"是让人以为存在人群参照(spec §5.5)。
- 真浏览器复验(真 Edge + 真面板 + 真 vite):页面文本含 `0 / 20 个指标槽通过证据门`、`📝 综合总结`、`未产出综合评分,也不给评级…`;`document.body.innerText.includes("常模")` = **false**、`includes("综合评分:")` = **false**。`tsc --noEmit` exit 0、`npm run build` ✓。

### 测试卫生(顺手根治):pytest 不再往仓库 `data/logs/` 写空文件

根因:`VoiceLogger.__init__` **立刻**写 CSV 表头,而 `voice_interaction.api.app` 在**模块级**构造 logger → 每 import 一次(即每跑一次 pytest)就在仓库 `data/logs/` 留下**两个只有表头的 `*_log_NONE_<时间戳>.csv`**(面试一个、科研一个;盘上 9/24 那批先于本轮,是本轮之前的同一现象)。
修法:`tests/conftest.py` 会话级把 `voice_interaction.utils.logger.LOGS_DIR` 指到 tmp(纯测试侧,不动生产代码)。复验:跑完全量后仓库里**无今天新增的 NONE 空文件** ✅。我这轮造的那批已删;9/24 那批不是我的,**没动**。

### 本轮**未做**(记下来,别当成已做)

1. **`dimensions.*.score` 仍被画成雷达图的「候选人得分」**(0–100 轴)—— 与 §5.4「不渲染未标定标尺上的点分」同轴,但改它属于**前端报告页整体收口**(§3 第 21–23 条:OpenAPI→TS 类型 + 报告入口收口),不在本次两条授权内。
2. **`VoiceLogger` 构造即建文件**建议改惰性(删 `__init__` 里那行、改为首次写入前建):可一并消掉**服务启动**时的空文件,不止测试期。属既有行为,未动。
3. `ReportViewer` 的 `dimensions.*.narrative` 在真实载荷里**不存在**(页面是照 mock 形状写的);`眼动行为分析` 里的坐标也是写死的示例数据。同属第 1 条那一摊。

### 操作事故(第二次踩,记牢)

`pkill -f "vite"` / `pgrep -f "voice_interaction.api.app"` **会匹配到执行该命令的 shell 自己**(命令文本里就含那个词)→ 自杀 `exit 144`。第一次已记过一次,这次用 `[v]ite` 括号技巧**仍然失败**,因为我的命令里还写了 `vite=$(pgrep …)` 这个字面量。**可靠做法:按端口杀** —— `fuser -k -n tcp <port>`,或 `ss -ltnp` 找 PID。以后不要用进程名 pkill/pgrep 去找自己启动的服务。

### 提交范围的事后审计(必须交代)

`5aa63c0`(Task 2 前端提交)对 `src/pages/ReportPage.tsx` 的改动是 **112 insertions / 93 deletions** —— 远超我计划里的增补。核对后确认:该文件当时带着**使用者未提交的整页重写**(旧版 108 行的简单页 → 带 `useParams`/`useEffect`/`store`/`Result`/`Loading` 的新版),被我**一并提交并推送**了。
- 我能守的约束是"只 `git add` 我要提交的文件"—— 这一条**守住了**;但**我要改的那个文件里本来就装着使用者的未提交工作**,git 按文件暂存,绕不开。
- `src/services/api.ts` 同一次提交是 12+/7−,与计划一致,**没有**夹带。
- 正确做法是**提交前**就挑明并请使用者裁定(例如:先只提交不含该文件的部分、或让使用者先提交自己的版本)。这次是事后才发现并报告。
- 后续(16964fe / 本次)对该文件的改动都发生在一个**已经包含使用者版本**的基线上,不再新增他人工作。

### 追补之后的全量门

- `pytest -q` → **224 passed**(本轮新增 6 条:Task3 三条 + 实时报告三条… 实为 Task3 三条 + 修复波次五条 + 追补两条 = 224)。逐次记录以各节为准。
- 合并门**第三次跑**(追补动了 `voice_interaction/api/app.py`,故必须重跑):
  ```
  最大绝对差 1.886e-19   相对差 > 1e-4 的格子: 0 / 2835510
  ✅ 回归验证通过:legacy 模式能逐格复现现有特征矩阵
  GATE_EXIT=0
  ```
- 前端:`tsc --noEmit` exit 0;`npm run build` ✓ 12.71s。
- 提交:`cce3bdb`(后端追补)/ `f11acb3`(前端追补);两仓 `main` 已推 origin。
