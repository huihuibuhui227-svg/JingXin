# M2 读侧按 session_id 对齐 —— 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans(本计划由控制器亲自实现)。Steps use checkbox (`- [ ]`) syntax.

**Goal:** 让一场真实会话在报告里对上号 —— 现在恒为「本次观测覆盖 0 / 20」。

**Architecture:** 读侧不再"每模态按文件名时间戳取最新",改成**先定目标 `session_id`、再按 id 取**;前端不再自造 UUID,改用 `/interview/start` 铸的号(`SESSION_ID` 从模块级 const 改成运行时可写)。

**Tech Stack:** Python 3.11(`~/miniconda3/envs/jingxin`)、pytest;React 18 + Vite + axios(前端在 `~/JingXin-frontend`)。

**Spec:** `docs/superpowers/specs/2026-09-24-m2-read-side-session-alignment-design.md`(权威,冲突以它为准)

## Global Constraints

- 解释器固定 `~/miniconda3/envs/jingxin/bin/python`;基线测试 **199 passed**;合并门必须保持 **0 / 2835510**。
- **前端改动不写 TS 单测**(项目没有前端测试设施,不新增)——由 §验收的浏览器步骤覆盖。
- 每条测试必须能说出"哪个生产改动会让它变红"。
- 不动:`face_expression/**`、`gesture_analysis/**`、`voice_interaction/**` 的实现(只动报告侧与前端)。
- `NONE` 桶仍不进聚合(既有不变量),但它**必须出现在披露里**。

---

### Task 1: 读侧按 id 选(后端)

**Files:** Modify `report_frontend/data_loader.py:104-129`(+ `selected_sessions` 形状)、`report_frontend/report_generator.py:40` 与 `sources_disclosure`、`app.py`(`/api/report/structured`)。Test: `tests/test_read_side_session_selection.py`(新建)

**Interfaces:**
- `LogDataLoader.get_fused_latest_data(session_id: str | None = None) -> Dict[str, DataFrame]`
- `LogDataLoader.selected_sessions: Dict[str, Dict]` —— 值从 `str` 变成 `{"session_id": str, "status": "loaded" | "unreadable" | "missing", "rows": int}`(**这是破坏性变更**,所有消费点要一起改)
- `LogDataLoader.resolve_target_session(session_id: str | None) -> str | None` —— 不给就用"最新一场"
- `report_generator.sources_disclosure(sources) -> str` —— 改渲染四态

- [ ] **Step 1**:写 `tests/test_read_side_session_selection.py`,覆盖 spec §5.4 前三条(按 id 选 / 未读到的模态出现在披露里 / 缺省取最新且写明是谁)。**先跑出红**。
- [ ] **Step 2**:改 `data_loader`:`resolve_target_session` + 按 id 分组选取 + 三态 `selected_sessions`。
- [ ] **Step 3**:改 `report_generator`:`generate_report(session_id=...)` 真的透传;`sources_disclosure` 渲四态;取消"同场/不同场"那句(按 id 选之后不存在该状态)。
- [ ] **Step 4**:改 `app.py` 的 `/api/report/structured` 与 `/api/run/report`,接受并透传 `session_id`。
- [ ] **Step 5**:改既有测试里对 `selected_sessions` 旧形状的引用(最少改动,不重写)。
- [ ] **Step 6**:反向复现:把选取退回"按时间戳取最新" → 第一条测试必须红。
- [ ] **Step 7**:全量 + 提交。

### Task 2: 前端用服务端铸号

**Files:** Modify `~/JingXin-frontend/src/services/api.ts:9-23` 与 `:98`、调用方 `src/pages/AssessmentPage.tsx` / `src/pages/RealtimeAnalysis.tsx`

- [ ] **Step 1**:`SESSION_ID` 改成 **let + getter/setter**(`getSessionId()` / `setSessionId(id)`);**启动时清掉** `assessment_session_id` / `face_session_id` 里的旧 UUID(spec §5.3)。face/gesture 的 URL 用 `getSessionId()`。
- [ ] **Step 2**:`interview.start()` 把返回的 `session_id` 存进去(并 `console.log` 出来便于验收时肉眼确认)。
- [ ] **Step 3**:调用方改成**先起会话再开摄像头**(spec §5.3 选的顺序);若某页确实要先看实时画面,则不传 id(落 `NONE`,报告侧会点出来)。
- [ ] **Step 4**:`npm run build` 必须过(TS 编译通过),并确认 dev server 热更新无报错。
- [ ] **Step 5**:提交(前端仓库 `~/JingXin-frontend` 是**独立 git 仓库**,单独提交)。

### Task 3: 端到端验收

- [ ] **Step 1**:`bash ~/shared/start_all.sh` 起全套;用 curl **按前端的确切调用顺序**模拟一场(start → 带铸号发帧 → 5 段回答)→ 断言报告覆盖率 **≠ 0** 且来源等于该 id。(自动化部分。)
- [ ] **Step 2**:给使用者一份三行浏览器清单(点界面那半只能他做)。
- [ ] **Step 3**:全量测试 + 合并门。

---

## Review Focus

spec 蕴含、但任务测试覆盖不到、最可能咬人的五条(每条都要有钉子或明确记档):

1. **`selected_sessions` 是破坏性变更** —— 任何漏改的消费点会静默拿到 dict 而不是 str(比如拼字符串时变成 `{'session_id': ...}`)。钉子:全仓 grep 消费点 + 一条断言 `sources_disclosure` 拿到 dict 时渲染正确。
2. **旧前端缓存** —— 老用户 localStorage 里是 UUID,不清掉就永远用旧值。钉子:清缓存那行有测试/至少在验收清单里点名。
3. **`NONE` 桶** —— 必须出现在披露里,但不能进聚合(既有不变量)。钉子:造一份 `..._NONE_...csv`,断言披露里有它、且它没参与特征计算。
4. **不给 id 时的"最新一场"** —— 若 `data/logs` 里只有 `NONE` 桶,不能崩,也不能把它当一场。钉子:只有 NONE 时返回 `None`。
5. **前端 build 必须过** —— TS 编译错误在 dev 模式可能被绕过。钉子:`npm run build` 进验收。
