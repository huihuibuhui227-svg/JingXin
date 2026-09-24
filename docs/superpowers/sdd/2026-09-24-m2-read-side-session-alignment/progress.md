# SDD ledger — plan: docs/superpowers/plans/2026-09-24-m2-read-side-session-alignment.md

Spec: `docs/superpowers/specs/2026-09-24-m2-read-side-session-alignment-design.md`(可达,绑定权威)

**执行方式(使用者确认)**:spec/plan/实现由控制器亲自做,**只派一个独立审查者**(opus),
且它的验收必须实跑。理由(M1.5 的实证):逐任务审查没做也没损失;唯一被证明有用的是
"最后那一次独立端到端审查" —— 它抓到了 I1(20 秒冻服务),那是任务级审查结构上看不到的。

## 开工前的裁决

- **Ruling: 账本放 `docs/superpowers/sdd/`(被 git 跟踪),不放 `.superpowers/sdd/`(被 gitignore)。**
  — 理由:上一轮复审报告落在 gitignore 目录里,差点丢(`docs/下一步.md` §3 第 7 条就是这个坑)。
  代价:无。

## 进度

### Task 1: 读侧按 id 选(控制器亲做,`6f90f1a`)

- `data_loader`:新增 `resolve_target_session`(显式 id / 缺省取最新一场 / 都没有则 `None`);
  `get_fused_latest_data(session_id)` 改成"先定目标、再按 id 取每个模态"。
- `selected_sessions` 值 `str` → 三态 `dict`(`loaded`/`unreadable`/`missing`)+ `none_bucket`
  —— **破坏性变更**,消费点一起改。
- `report_generator`:`sources_disclosure` 改四态;取消"同场/不同场"那句;`generate_report` 真用 `session_id`。
- `app.py`:两条报告路由透传 `session_id`;`report_generator.__main__` 加 `--session-id`。
- 测试 199 → 202(新增 7;`test_report_sources.py` 的 4 条单元级冗余覆盖移交新文件,
  它保留独有的**完整 generate_report 集成路径**)。
- **反向复现**:把选取退回"每模态各取最新" → `test_selection_is_by_session_id_not_by_newest_timestamp`
  精准红(读到 1 行而非目标场的 2 行)。
- **真实数据验收**:指定 T7 那场 → 面部 1 行 / 手势 1 行 / 语音 6 行 / **覆盖 1/20**(此前恒 0/20)。

### Task 2: 前端改用服务端铸号(控制器亲做,前端仓库 `4b07efb`)

- `src/services/api.ts`:`SESSION_ID`(模块级 const + 自造 UUID)→ 可写的 `sessionId` +
  `getSessionId()`/`setSessionId()`;**启动时清掉** localStorage 里旧版残留的 UUID
  (不清的话老用户永远用旧值 —— spec §5.3 点名的坑)。
- `withSession(url)` 统一拼 id;**还没开始会话时不拼**(而不是带一个编出来的 id)。
- `interview.start()` 存下服务端返回的 `session_id`。
- **顺带修一个独立的漏(不在原计划里,是读代码时发现的)**:`submitAudioAnswer` /
  `research.submitAudioAnswer` / `/asr` / gesture 的 `/reset` **从来没带过 `session_id`**
  —— 回答会落进 `NONE` 桶,而报告侧整体排除 NONE。也就是**前端那条路"录了、也识别了,
  报告里什么都没有"**。这条与根因独立,修完前端才算真的通。
  **Ruling: 一并修,不另开任务。** — 理由:它与契约修复同源(都是 id 没传到位),
  分开做会留下"id 修好了但回答还是落 NONE"的半修状态,比不修更难查。代价:前端改动面扩大一点。
- `npm run build` 通过。
- ⚠️ 该仓库工作树里另有 **40+ 个使用者先前的改动**(未提交)—— **一个都没碰**,只提交了 `api.ts`。

### Task 3: 端到端验收(控制器亲做)

- 脚本 `~/shared/m2_acceptance.sh`,**按前端的调用顺序**模拟:铸号 → 5 段回答(带号)→
  face/gesture 各一帧(带号)→ 生成报告。
- 结果:`20260924_225422_45db`;**三份日志文件名与首列都等于铸号**(不再是 UUID);
  **覆盖 1 / 20**;报告头点名了该会话。
- 全量 202 passed;合并门 **0 / 2835510**。

- **Ruling:`RealtimeAnalysis` 页"自行起会话"这条**不做**,改成"有会话就带、没有就不带"。**
  — spec §5.3 写的是"先调 `/interview/start` 再开摄像头"。实施时发现:让一个"实时演示"页
  去**铸造一场真会话**(建目录、铸号)是会改变其语义的行为,而且它可能与该页作为独立
  演示入口的用法冲突。现在的形态是:`withSession` 在没会话时不带 id → 帧落 `NONE`
  → **报告头会显式点出** "另有 NONE 桶 N 行"。也就是说**这件事现在是可见的,不是静默的**。
  代价:从实时页直接开摄像头时,那些帧不归入任何会话(要归入就得先走一次面试开始流程)。
  **这一条请复审者与使用者一并裁决** —— 若认为该改,它是个独立的小改动。

## 待复审者核的事

1. `selected_sessions` 破坏性变更的**消费点有没有漏**(全仓 grep;漏了会静默拿到 dict)。
2. `NONE` 桶:进披露但**不进聚合**(既有不变量)。
3. 只有 `NONE` 桶时 `resolve_target_session` 返回 `None`(不崩、不把 NONE 当一场)。
4. 前端:**老 localStorage 清没清干净**;`withSession` 在无会话时的降级是否正确。
5. `app.py` 那条 spawn 路径(`/api/run/report --session-id`)真的能透传(实跑一次)。
6. 实跑 `~/shared/m2_acceptance.sh` 复核。
