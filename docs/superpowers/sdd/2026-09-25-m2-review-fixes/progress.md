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
