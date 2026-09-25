# M2.1 最终独立复审报告(归档)

- 日期:2026-09-25
- 评审者:独立子代理(最强可用模型,**单一 fresh context**,按 `superpowers:requesting-code-review` 的 code-reviewer 模板派发)
- 范围:后端 `242f512..5faba37`(4 提交)+ 前端 `~/JingXin-frontend` 的 `5aa63c0` / `4232928`(只含 `src/services/api.ts`、`src/pages/ReportPage.tsx`)
- 评审包:`.superpowers/sdd/2026-09-25-m2-review-fixes/review-242f512..5faba37.diff` + `review-frontend.diff`(工作区已删,两者可由 git 复现)
- **本文件是评审结论的要点归档;逐字报告只存在于 2026-09-25 那次会话的记录里。** 每条判定的落地见 `progress.md`。

## 评审者真跑了什么

全量 `pytest` **217 passed in 18.69s**;合并门 `--verify-legacy` **0 / 2835510**、最大绝对差 `1.886e-19`(与 2026-09-21 基线逐位一致);前端 `npx tsc --noEmit` **exit 0**(刻意不跑 `npm run build`,免得往工作树写 `dist/`)。另写四个**只读**探针,把 Review Focus 五条逐条实测(真加载器 + 真渲染 + 真 Flask),并用**真子进程**验了退出码传播。

## 结论

**Ready to merge? With fixes.** 后端三条修复经其逐条实跑:Review Focus 后端四条达标,RF5(退出码传播)有真进程证据;**卡在前端**(C1 + I1)。判 **Critical 1 / Important 3 / Minor 9**。

## Strengths(要点)

- 五条 Review Focus 后端四条达标,RF5 用真进程验过(不是纸面推理)。
- `NONE` 桶统计提到早退之前,是修 spec §6 行 3 的最小正确改法;钉子测试压住。
- 跨模块接缝干净:`sources_disclosure` 的 `target` 是带缺省的追加参数,生产调用点只剩 1 处;`app.py` 回传的 `sources` 端到端验过 JSON 真能序列化。
- 三态披露在"部分模态"场景(只有手势 / 只有语音 / 面部空表头+有手势)下也正确,`df_face=None` 扛住了。
- Task 3 与 `/interview/start` 真对称;端到端验收脚本第 ② 步(旧号污染面试目录)是这条修复**唯一的强证据形态**,做法对。
- 账本诚实:自陈覆盖力限制、每条裁定带 `cost if wrong`、变异复现逐条记录。

## Issues

### Critical

**C1. 前端报告页在"证据不足"这一常态下渲染期抛异常 —— 页面变「页面出现错误」。**
`~/JingXin-frontend/src/pages/ReportPage.tsx:107` 的 `report.total_score.toFixed(1)`;而 `/api/report/structured` 的 `result.total_score` 在**没有任何维度通过证据门**时是 `null`(`report_frontend/research_mapper.py`)。实测:`total_score=None` 不是空数据的孤例 —— 只有面部 30 帧的会话同样 `n_passed: 0`。改动前空数据回 `error` → 前端显示「暂无报告数据」(优雅);Task 2 放行空数据为 `success` 后,前端走渲染 → `TypeError` → 被 `ErrorBoundary` 接住 → **Task 2 交付的披露卡根本到不了屏**。
→ 处置:**已修**(真浏览器 RED→GREEN),见 `progress.md`。

### Important

**I1. `ReportPage.tsx:35-38` 的「先信 store、否则取 API」短路。** 一旦 `evaluationResult` 非空就不发请求 → `describedSession`/`sources` 恒空 → 披露卡印出「（无 —— 本场没有任何日志）」而报告就在屏上;第 18 条的 id 与披露在这条分支上完全不生效。评审者判它是"本轮唯一未记录的设计决定"(**这一句不成立**:该短路在改动前就在,非本轮引入 —— 见账本的更正);但**卡片会说假话**确是我这轮引入的。
→ 处置:**已修**(卡片只在服务端真回答了来源时渲染)。

**I2. 科研铸号后,科研会话会遮蔽面试会话 —— 且被铸出来的科研会话在读侧永远不可描述。** 机制:摄像头对 interview/research 共用、不按类型 gate,且 `handleStart` 先 `await start()` 再置 `started` → 帧带**科研号**;于是所有**不带 id** 的消费者(`~/shared/t7_acceptance.sh`、`/api/run/report` 不带 query、`GET /api/report/structured` 不带参数)在"先面试再科研"之后都改去描述科研会话。且 `voice_interaction/api/app.py` 的 `research_logger = VoiceLogger(log_type='research')` **不带 session_id**,科研回答也不调 `log_prosody` → 盘上**不存在**任何带科研号的特征行,`ensure_manifest` 声明的四个 `expected_file` 一个都不可能被满足。
净效果:「先面试 → 再科研 → 看报告」拿到的是**科研会话那份(语音两栏全缺)**,而面试那份完整报告不被提及(报告头点名了是哪一场,不是静默)。
→ 处置:**裁决不修,交使用者**。它是方案 A 已知悉的后果;评审者新指出的是"科研会话永远读不出语音"这半边。建议收口(a):给科研的 `VoiceLogger` 带上 `session_id`,顺带让 `voice_research` 这个恒 `missing` 的模态有产出方。

**I3. 面板实时报告路径仍「没产出也报成功」。** `app.py` 的 `_run_live_report` 拿到 `generate_report_live()` 的 `""` 后**无条件**写 `"status": "success"` —— 与第 17 条逐字同族,只在隔壁那条路上(前端无消费者,故非 Critical)。
→ 处置:**已修**(1 行 + 3 条测试)。

### Minor

1. `/api/report/structured` 的 docstring 宣传了一个它不读的 `type` 参数 → **已修**。
2. `none_bucket` 的 `status: "present"` 是三态枚举之外的第 4 值(未文档化) → deferred。
3. RF3「不许回退去拼别的场次」旧测试压不住(目录里没有第二场可回退) → **已修**,并做了反向复现。
4. RF4 只断 `n_passed == 0`,没断分母 → **已修**(补 `n_slots == 20`)。
5. `webbrowser.open` 在主 `try` 内且在 `write` 之后 → 打开浏览器失败会把**已落盘**的报告变成「任务失败」(新退出码放大的假警报) → **已修**。
6. `_none_bucket_rows()` 每出一次报告全量读 NONE 文件 → deferred(承 M2)。
7. 账本最后一行未提交 → **已处理**。
8. `docs/下一步.md` 说提交未推(当时为真,现已过期) → **已按实际改写**。
9. `none_bucket` 只在 `rows > 0` 时出现(只有表头的 NONE 文件不被提) → deferred。

## Declined to judge(评审者明示设而不判的,执行者逐条认领)

`ReportPage` 该不该渲染 `total_score` + 档位(**政策裁定,留给使用者**);计划「本轮明确不做」其余各项;logger 与 data_loader 的既有陈旧注解;Ruling M1-14 的推翻(评审者补了独立证据:217 条 18.69s 跑完、无测试期 TTS 线程残留);`module_map` 指错模块;`ReportsList` 恒空;`getSessionId` 死代码;中文文案口味;浏览器那半(未跑,故 I2 是**代码路径 + 合成复现**级证据,非 vivo 实测);`ErrorBoundary` 等 UI 分支取舍;文件末尾缺换行。
