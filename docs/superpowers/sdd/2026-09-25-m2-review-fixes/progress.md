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
