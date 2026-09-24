# M2 读侧按 `session_id` 对齐 —— 独立端到端复审报告

- 日期:2026-09-24
- 审查者:独立复审(无实施方参与;未派子智能体;未改动被审代码)
- 审查对象:后端 `1af7990..e96102b`(`6f90f1a` 为唯一实现提交)+ 前端 `~/JingXin-frontend@4b07efb`(只改了 `src/services/api.ts`)
- 权威文档:spec `docs/superpowers/specs/2026-09-24-m2-read-side-session-alignment-design.md`、plan `docs/superpowers/plans/2026-09-24-m2-read-side-session-alignment.md`、账本 `docs/superpowers/sdd/2026-09-24-m2-read-side-session-alignment/progress.md`
- 本报告所在目录**被 git 跟踪**(承账本里的 Ruling:上一轮复审报告落在 gitignore 目录里差点丢)

---

## 0. 一句话判定

**主目标达成、可以合并,但先修 2 处**:①`generate_report_live` 没跟上 `sources` 的破坏性变更(实时报告路径**回归**成静默失败);②`_none_bucket_rows` 的 glob 漏掉 face/gesture 的 NONE 形态,导致 224 行真实 NONE 数据永远不会出现在披露里 —— 而账本给第 6 点(RealtimeAnalysis 裁决)的**理由正建立在这条披露上**。两处都是一行级修复。

---

## 1. spec §1 在范围内四项的合规判定

| spec §1 项 | 判定 | 实现位置 / 证据 |
|---|---|---|
| 1. 读侧按 `session_id` 选日志 | **实现** ✓ | `LogDataLoader.resolve_target_session`(`report_frontend/data_loader.py:127-144`);按 id 逐模态取 `report_frontend/data_loader.py:183`;不再有"每模态按时间戳取最新"的拼接 |
| 2. 报告生成接受并**真的使用** `session_id` | **实现** ✓(有一处旁支未跟上) | `report_generator.py:90-106` 真透传;`--session-id`(`report_generator.py:362-370`);`app.py:138-144`(`/api/run/report`)、`app.py:228-231`(`/api/report/structured`)。旁支 `generate_report_live` 未改 → **Critical 1** |
| 3. 前端改用服务端铸的号 | **部分实现**(面试链路 ✓;科研链路 ✗) | `~/JingXin-frontend/src/services/api.ts:20-35`(`let sessionId` + get/set + `withSession`)、`:110-116`(`interview.start()` 存服务端号)、`:23-24`(启动清旧 UUID 键,构建产物 `dist/assets/api-54f199cd.js` 里确认这两行在)。科研:`:153-156` **不** setSessionId,服务端 `/research/start`(`voice_interaction/api/app.py:460-472`)根本不铸号 → **Important 4** |
| 4. 披露补齐(选了但读不出来的模态必须显示「未读到数据」) | **实现** ✓(两处不完整) | 四态渲染 `report_generator.py:34-72`;`unreadable` 会在**完整 `generate_report` 路径**上出现(实跑确认,见 §4.6)。不完整:`§6 行1/2` 未实现(**Important 2**)、NONE 披露对 face/gesture 形态失效(**Important 1**) |

不在范围内的四项(VIDEO 等价性门、M3 阈值重登记、哨兵值歧义、前端状态机字段)按 spec §1 未审,见 §6。

---

## 2. 我实跑的东西(命令与输出要点)

**没跑的一律标"未跑";下面全部真跑过。**

### 2.1 `bash ~/shared/m2_acceptance.sh`(要求 1)

**结论:全部断言通过、退出码 0;但脚本在我这里卡死了两次,需人工解围(见 Minor 6)。**

- 第 1 次卡死:`── 起服务 ──` 之后无输出 >7 分钟。原因:`start_all.sh` 用 `( cd … && "$@" > log 2>&1 & )` 起 `npm run dev`(5173 当时没在跑),该后台任务是 `start_all.sh` 的子进程,于是 `start_all.sh` 一直停在 `do_wait`;它的 stdout 是被 `grep|sed` 接走的管道,父脚本永远等不到 EOF。我杀掉自己那次启动的 vite 后它才继续。
- 第 2 次卡死:`── ⑤ 出报告 ──` 之后 >7 分钟无输出。原因:`generate_report` 里的 `webbrowser.open` → `xdg-open` → 新起一个 msedge,三者**继承了 greps 的管道写端**,报告早已写好但 `grep` 等不到 EOF。清掉 `xdg-open` 后脚本走完。

实际输出(我逐值独立复核,见 2.2):

```
── ① 铸号 ──   ✓ 服务端铸号 = 20260924_230914_262f
── ② 五段回答 ── ans1 密度=0.0 ans2=2.1978 ans3=5.2632 ans4=0.885 ans5=3.8835
                每一段回传 sid=20260924_230914_262f
── ③ 收帧 ──    face HTTP 200 / gesture HTTP 200
── ④ 三份日志 ── ✓ face_au_log_<SID>.csv 首列=<SID>
                 ✓ interview_emotion_log_<SID>.csv 首列=<SID>
                 ✓ gesture_emotion_log_<SID>.csv 首列=<SID>
── ⑤ 出报告 ──  目标会话:20260924_230914_262f
                 面部 1 行 / 手势 1 行 / 语音（面试）5 行 / 语音（科研）本场没有
                 覆盖:1 / 20   ✓ 报告头点名 20260924_230914_262f
SCRIPT_EXIT=0
```

### 2.2 我**不**采信脚本的 ✓,自己复核的值

- 三份日志的文件名与首列: `awk -F, 'NR==2{print $1}'` 独立读 → 三份都等于 `20260924_230914_262f` ✓
- 报告:`data/output/Research_Assessment_Report_20260924_230942.html`,我自己解析 → `cover-number` = **1 / 20**;`本场会话：20260924_230914_262f`,面部/手势/语音（面试）· 已读入、语音（科研）· 缺失 ✓
- 旧 UUID 命名的日志(`face_au_log_70efe7e5-….csv`)仍在盘上 → 对加载器不可见(正则实测 `False`),不影响本轮;留着是故障历史的物证。

### 2.3 合并门(要求 2)

```
cd experiments/duration_audit
python reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_rev
python reaggregate_normalized.py --verify-legacy /tmp/legacy_rev
```
输出要点:`聚合完成 124s:成功 2011,排除 0` / `[legacy] 2011 视频 × 1410 维 (face=1260 gesture=140 voice=10), 死特征 169` / **`相对差 > 1e-4 的格子: 0 / 2835510`** / `✅ 回归验证通过`。
**结果:0 / 2835510 ✓(与账本一致)**

### 2.4 全量测试(要求 3)

`~/miniconda3/envs/jingxin/bin/python -m pytest -q` → **`202 passed in 16.23s`**,退出码 0 ✓(与账本一致;基线 199 + 新增 7 − 删除 4 = 202,数目自洽)

### 2.5 前端 `npm run build`(要求 4)

`cd ~/JingXin-frontend && npm run build` → **退出码 0**、`✓ built in 16.50s` ✓
(提醒:build 编译的是**整个工作树**,含 40+ 个未提交的使用者改动 —— 它验的是"当前工作树能编译",不等于"4b07efb 单独可编译"。)

### 2.6 面板 spawn 路径真的透传 `session_id`(要求 5)—— **实跑,并用能区分真假的构造**

目的是排除"读代码觉得对":只断言"报告能出来"是**无区分度**的(缺省取最新一场也可能恰好是它)。我的构造是让"显式 id"与"缺省最新"**分歧**:

1. 先 `POST /interview/start` 铸一个新会话 `20260924_231815_3ceb`(它比目标场更新,且**是空的** —— `/interview/start` 当场就落一份只有表头的日志)
2. 再 `POST http://127.0.0.1:5000/api/run/report?session_id=20260924_230914_262f`(**更旧但有三模态数据**的那场)
3. 轮询 `/api/task/<id>` 读子进程 stdout

结果:

```
status = success      ← 注意这一点(Critical 1 与 Important 2 的同一个坑)
── 子进程 stdout ──
   目标会话:20260924_230914_262f          ← 显式 id 到了子进程
   📥 [FACE] 1 行 / [GESTURE] 1 行 / [VOICE_INTERVIEW] 5 行
   ⚠️ [VOICE_RESEARCH] 本场没有这个模态
```
产出报告 `Research_Assessment_Report_20260924_231816.html`:含 `20260924_230914_262f`、**不含**更新的 `20260924_231815_3ceb`。
**判定:透传成立,且"子进程收到的参数"与"产出报告描述的那一场"是同一场 ✓**(若缺省生效,目标会是那场空会话 → `无面部数据` → 不会有报告)

### 2.7 我额外做的两项独立验证

- **反向复现**(核账本"精准红"的说法):把仓库副本拷到 `/tmp/m2rev`,把 `mine = [f for f in files if f["session_id"] == target]` 退回 `mine = list(files)`(即"每模态各取最新"),跑新契约测 → **`1 failed, 6 passed`,红的正是 `test_selection_is_by_session_id_not_by_newest_timestamp`(读到 1 行而非 2 行)**。账本该说法**属实**。
- **spec §6 错误处理四例**(临时目录 + monkeypatch 掉 `LogDataLoader`/`webbrowser.open`,不动仓库):

| 场景 | 实测结果 | spec §6 要求 |
|---|---|---|
| ① 只有 NONE 桶 | `generate_report` 返回 `''`,**没有报告**,`selected_sessions` 连 `none_bucket` 都没有 | 生成报告 + 显式说明本场没有日志 → **违背** |
| ② 显式给了不存在的 id | 四模态全 `missing` → 返回 `''`,**没有报告**,也不"逐条列出缺什么" | 报告仍生成、头里逐条列出缺什么、不抛 → **违背** |
| ③ 目标场 face 有、语音文件空 | 报告生成 ✓,披露含「未读到数据」「缺失」 | 符合 ✓ |
| ④ 完全没有日志 | 同 ① | 违背 |

---

## 3. Findings

### Critical(必须修)

#### C1. `generate_report_live` 没跟上 `sources` 的破坏性变更 —— 实时报告路径**回归**成静默失败

- `report_frontend/report_generator.py:167`:`sources={k: session_id for k in data}` 仍然是**旧形状 `Dict[str, str]`**,而 `sources_disclosure` 现在要 `Dict[str, Dict]`
- 崩在 `report_frontend/report_generator.py:56`:`target = next(iter(items.values()))["session_id"]` → `TypeError: string indices must be integers, not 'str'`
- **失效场景**:任何调用 `generate_report_live`(面板 `POST /api/run/report_live` → `app.py:184-203`)。异常被 `generate_report_live` 自己的 `except`(`:180-182`)吞掉 → 返回 `''` → **没有任何报告**;而 `_run_live_report` 把任务状态写成 `success`、`message="实时报告生成完成！"`,`logs` 只有一个"生成失败"。
- **已复现(端到端,经面板)**:
  ```
  POST /api/run/report_live {"session_id":"20260924_230914_262f"}
  → status = success   message = 实时报告生成完成！   logs = 生成失败
  ```
  并单独复现异常原文(指向 `report_generator.py:56`)。
- **为什么算 Critical**:这是**本次改动引入的功能回归**(M2 之前 `sources_disclosure` 恰好接受 `str`,这条路径是通的);它正是 Review Focus 第 1 条点名的那类"漏改消费点";而且失败是**静默的**(面板说成功)。
- **可达性(请连同判定一起看)**:面板 dashboard 模板里没有 report_live 按钮(我 grep 过 `templates/` 与前端 `src/`),所以它目前只对直接打 API 的人可达 —— 若你按"死代码"降级为 Important 我不反对,但**修法是一行**:
  ```python
  sources={k: {"session_id": session_id, "status": "loaded", "rows": len(data[k])} for k in data}
  ```
  顺带:`report_generator.py:249` 的类型标注还写着 `sources: Optional[Dict[str, str]]`、`data_loader.py:50` 还写着 `selected_sessions: Dict[str, str]` —— 两处**过时标注**正是这个 bug 能溜过去的原因,建议一并改。

### Important(应该修)

#### I1. `_none_bucket_rows` 的 glob 漏掉 face/gesture 的 NONE 形态 → 224 行真实 NONE 数据**永远不会**出现在披露里

- `report_frontend/data_loader.py:119`:`self.log_dir.rglob("*_log_NONE_*.csv")`
- face/gesture 的 NONE 形态是**单文件无时间戳**的 `face_au_log_NONE.csv` / `gesture_emotion_log_NONE.csv`(`face_expression/utils/logger.py:36`:只有 session_id 为**假值**时才拼时间戳;而 API 把缺省 id 规范化成**真值字符串** `"NONE"`,于是走单文件分支)。glob 要求 `NONE` 后面还有 `_`,**匹配不上**。
- **实测(真实 data/logs)**:
  ```
  _none_bucket_rows() = 0
     face_au_log_NONE.csv                                  112 行   ← 漏
     gesture_emotion_log_NONE.csv                          112 行   ← 漏
     interview_emotion_log_NONE_20260924_*.csv (7 份)        0 行
  真实合计 = 224
  ```
  这 224 行的时间戳落在 **23:00:35–23:02:37**,正是我把前端 dev server 起起来之后、浏览器里那个页面在**没有会话**时发的帧 —— 也就是 spec §5.3 / 裁决第 6 点说的那个场景,真金白银地在盘上。
- **后果**:该报告两处都不提 NONE(我实测生成的 HTML 里 `"NONE" in html == False`)→ "有帧没归入本场而无人知晓",**这正是 D4/spec §6 行3 与 plan Global Constraints 要防的静默缺席**。
- **并请核对账本第 6 点**:那条给 RealtimeAnalysis 裁决的理由是"帧落 `NONE` → **报告头会显式点出**『另有 NONE 桶 N 行』"。就 face/gesture 而言**这句不成立**(只有 interview 的 NONE 形态能被数到,而那 7 份全是 0 行)。裁决本身我赞同(见 §5),但**它的论据要换成"要么先修 glob,要么承认 NONE 帧目前是静默的"**。
- **修法**:glob 收两种形态,例如
  ```python
  for p in list(self.log_dir.rglob("*_log_NONE_*.csv")) + list(self.log_dir.rglob("*_log_NONE.csv")):
  ```
  并补一条钉子测试(现在 `tests/test_log_file_selection.py` 只把这两种形态用在"正则必须拒绝"上,**没有任何测试断言它们能被计入 NONE 桶**)。
- 附带:`data_loader.py:61-64` 的注释断言"NONE 落盘为 `..._log_NONE_{YYYYMMDD}_{HHMMSS}.csv`(三个 API 都…)"对 face/gesture **是错的**;这个错误事实正是这个 bug 的来源,建议一并改注释。

#### I2. spec §6 行1/行2 未实现:**目标场没有任何日志时,一份报告都没有**,而面板报"成功"

- `report_frontend/report_generator.py:107`:`if not data or 'face' not in data: raise ValueError("无面部数据")`
- **失效场景 A(只有 NONE 桶 / 完全没有日志)**:`resolve_target_session` 返回 `None` → `data_loader.py:160-164` 直接返回 `{}`(注意此时**连 `none_bucket` 都没进 `selected_sessions`**)→ 抛 → 报告没生成。spec §6 要求"生成报告并显式说明本场没有任何日志"。
- **失效场景 B(显式指了一个没有日志的 id)**:四模态全 `missing`,同样没有报告;spec §6 行1 要求"报告仍生成,头里**逐条列出缺什么**;不抛"。
- **已复现**:§2.7 的 ①②④ 三例,全部返回 `''`。
- **额外症状**:`app.py:53` 只判 `returncode == 0`,子进程"没产出但 exit 0" → `/api/task/<id>` 报 `status=success`、`message="任务完成！"`。而我实测 `POST /api/run/report`(不带 id)+ `/api/report/structured`(不带 id)**都会命中这条路**:因为 `/interview/start` 会当场落一份空日志,于是"最新一场"可能是一场还没有任何数据的会话 —— 实测:
  ```
  GET /api/report/structured                    → {"status":"error","message":"未找到评估日志数据"}
  GET /api/report/structured?session_id=<有数据那场> → {"status":"success", "coverage":…}
  ```
  即:**前端报告页会显示「暂无报告数据」,而盘上明明有一场三模态齐全的会话。**
- 修法(供参考):把 `raise ValueError` 换成"无 face 也渲染一份报告",把 `selected_sessions` 的四态照常渲进去;`none_bucket` 在 `target is None` 时也要填(这样"只有 NONE"才说得出话)。

#### I3. 前端的**读**路径从不带 id,而结构化 JSON 里**没有任何"这是哪一场"的信息** → 前端看不到 M2 的成果

- `~/JingXin-frontend/src/services/api.ts:202-205`:`getStructuredReport()` 不带 `session_id`(且是本次提交**新加**的函数),它唯一的消费者 `src/pages/ReportPage.tsx:30` 也没传;`:192-195 runModule('report')` 同样不带(消费者 `src/hooks/useAssessment.ts:114`)
- `getSessionId()` 被导出了,但全仓 grep **没有任何消费者** → 前端拿到了铸号却不用在读侧
- 后端 `/api/report/structured`(`app.py:218-246`)返回的 JSON 里**没有 session_id / 没有披露文本**:实测带 id 成功时 `result` 只有 `coverage, dimensions, evidence_gaps, model_metadata, summary_narrative, total_level, total_score`,`20260924_230914_262f`、`session_id`、`本场会话` 一个都不在 JSON 里
- **失效场景**:①两个会话并存(第二场刚开始、或另一个标签页/窗口起了新会话)时,前端报告页显示的是**另一场**(实测见 I2:直接报"未找到评估日志数据");②即便命中正确,读报告的人也**没有任何办法从界面确认这是哪一场** —— spec D2 的"缺省取最新一场,**并在报告头写明是哪一场**"在**前端这条路上没有落地**(只在 HTML 报告里落地)
- **已复现**:上面那对 A/B curl 就是。
- 修法:`getStructuredReport(sessionId?)` / `runModule('report', sessionId?)` 拼上 `getSessionId()`,并让 `/api/report/structured` 把 `loader.selected_sessions`(披露文本或结构化字段)一起返回。

#### I4. 科研流程仍然没有可用铸号,且**新增**了"科研回答记到上一场面试名下"的失败模式

- `~/JingXin-frontend/src/services/api.ts:153-156`:`research.start()` **不** `setSessionId`;服务端 `voice_interaction/api/app.py:460-472` 的 `/research/start` 也**不铸号**(只返回 `{"status":"started","question":…}`)
- **失效场景 A(新页面直接做科研评估)**:`sessionId === null` → `withSession` 不加参数(:34-35)→ `_resolve_session_id` → `NONE`。前端提交里声称"`research.submitAudioAnswer` 从来没带过 `session_id`"这条已经修好,**实际上科研链路仍然落 NONE**。
- **失效场景 B(先面试、再科研,同一个页面)**:`sessionId` 还留着**上一场面试的铸号** → 科研回答会被写成那个面试会话的:
  - `voice_interaction/api/app.py:527` `transcript_store.append_utterance(sid, utt)` → 该候选人的**科研回答原句写进面试会话的录制目录**。这是**本次改动新引入**的失败模式(M2 之前科研一律落 NONE,不会张冠李戴),虽然落在仓库外(`~/shared/jingxin_recordings/`)、不影响报告 CSV,但它是"id 归堆"这条轴上的反向污染。
- **未复现**(需要走完整科研交互;我按代码路径判定)。仓库内的 CSV 不受影响:research 侧没有自己的 `LOG_PREFIXES`,不走 `interview_emotion_log`,所以报告数字不受污染。
- 修法:要么服务端 `/research/start` 也铸号(与 M1 对称),要么前端科研流程显式 `setSessionId(null)`,别让上一场的号顺延。

### Minor(可改可不改)

1. **`_none_bucket_rows` 每出一份报告就全量读一遍 NONE 文件**(`data_loader.py:112-125`)。NONE 是**单文件且跨天累积**(`face_au_log_NONE.csv` 已经是 112 行),随使用增长;`rglob` + `pandas.read_csv` 全表只为拿行数。修法:只数行(`sum(1 for _ in open(...))`)或缓存。
2. **`test_log_file_selection.py:74-75` 同一句注释被复制了两遍**;同文件 `:65-68` 的 docstring 仍在讲"修复前 M1 形态不被识别",而 fixture 里已经没有任何 M1 形态的文件 —— 留下会误导后来人。
3. **`resolve_target_session` 的"最新一场"在同秒并列时不确定**:`max(..., key=timestamp_val)` 在 `YYYYMMDD_HHMMSS` 相同、只差 4 位随机段的两场之间,胜者取决于 `rglob` 的遍历顺序(`data_loader.py:140-144`)。spec §5.1 本身只按时间戳定义,所以这不算违约,但"缺省取最新"因此有一个不确定窗口。修法:并列时按 `session_id` 字符串次序取定。
4. **`sources_disclosure` 从"第一个 item"推 target**(`report_generator.py:56`):函数是公开的,拿到混合 id 的 dict 会**静默**取第一个(实时路径正是这么崩的)。加一句 `assert len({v["session_id"] for v in items.values()}) == 1` 或让它接受 target 参数,能把 C1 变成显式契约违反。
5. **前端把"保存到 localStorage"改成了只在内存里存**(spec §5.3 字面要求"保存到 localStorage",实现见 `api.ts:20`)。我判**这是更好的选择**:持久化会把上一场的 id 带进"下一次会话开始之前"的窗口,反而制造跨场污染(与 I4-B 同型);当前降级是"没会话就不带参数"(正确)。**但它是一处 spec 偏离,建议把理由写进 spec 或账本,别只留在代码注释里。**
6. **验收脚本在两种环境下会无限卡住**(都不是产品 bug,但你要求只跑这一个脚本,所以我记下来):(a) `start_all.sh` 起 `npm run dev` 时会等在这个后台子进程上(5173 未先起跑时)—— `bash ~/shared/m2_acceptance.sh` 因此**永远走不到断言**;(b) 第 ⑤ 步 `webbrowser.open` 起的浏览器继承 stdout,`grep` 永远等不到 EOF(报告其实已经写好)。账本里那次"绿"是在这两个坑都恰好没触发的状态下拿到的。修法:把 `bash ~/shared/start_all.sh | grep …` 那行加 `</dev/null` 或改直写日志;报告那步加 `>/dev/null 2>&1` / `BROWSER=true`。
7. **前端提交里多了一个 `getStructuredReport`**(`api.ts:202-205`),超出 plan Task 2 声明的改动面 —— 它是使用者未提交的 `ReportPage.tsx` 的依赖(不加则工作树编译不过)。只记档,不判违规;但它正是 I3 里"该带 id 却没带"的那个新函数。
8. **前端 `dist` 里 `GazeHeatmap` chunk 4.7 MB** 触发 build 警告 —— 既有问题,与 M2 无关。

---

## 4. 审查者被点名要盯的六处 —— 逐条回答

1. **`selected_sessions` 破坏性变更的消费点有没有漏?** —— **漏了一个**。全仓(不限扩展名,排除 `.git`/`node_modules`)grep `selected_sessions`:消费点只有 `report_generator.py:121`(已改对)与各 `__main__` 演示;但 `sources_disclosure` 有**第三个**调用点 `report_generator.py:167`(`generate_report_live`),它仍传旧形状 → 实跑 TypeError。既有测试**没有**覆盖到:全仓没有任何测试碰 `generate_report_live`(grep 确认)。→ **C1**
2. **`NONE` 桶进披露但不进聚合?** —— 不进聚合 ✓(正则实测 `False`,`tests/test_log_file_selection.py` 的逐形态参数化测试钉住;"放宽正则会让 NONE 重新可见"这个反向也还在测)。**进披露 ✗**:只对 interview 的 NONE 形态成立,face/gesture 的形态被 glob 漏掉,现实里 224 行全漏。→ **I1**
3. **只有 `NONE` 桶时 `resolve_target_session` 返回 `None`、不崩、不把 NONE 当一场?** —— ✓ 全部成立(实跑:返回 `None`、`get_fused_latest_data()` → `{}`、`selected_sessions` → `{}`、无异常,并有 `test_none_bucket_is_never_treated_as_a_session` 钉住)。但**下游**没接住这个 `None`(不生成报告,spec §6 行2 要求生成)。→ **I2**
4. **前端旧 localStorage 清干净了没;`withSession` 无会话时的降级对不对?** —— 清干净了 ✓(`api.ts:23-24` 两条 `removeItem`,清的是**旧版仅有的两个键**;我另 grep 了整个 `src/`,没有第三个会话相关键;构建产物 `api-54f199cd.js` 里确认这两行进了 bundle)。降级 ✓:`withSession` 在 `sessionId === null` 时**原样返回 URL**(不带编造的 id),`console.log` 也点明了"发帧会落进 NONE 桶"。**未执行**:这两点我是**读代码 + 查 bundle**判定的,没有跑起来点界面发帧验证(浏览器那半只能你做)。
5. **`tests/test_log_file_selection.py` 的两条 fixture 改动是否削弱了覆盖?** —— **原有四件事都还在,没有净削弱**:①旧形态可读 ✓(改后 fixture 就是一场旧形态会话,三条断言);②M1 形态可读 ✓(仍在 `test_m1_session_named_logs_are_loadable`,fixture 改成三模态同一个 id —— 这个改动是**必需**的,因为按 id 选之后"三个模态各一个 id"必然只能取到一场);③NONE 永不可见 ✓(既有"名字更新的 NONE 也不许赢"的断言,加上 `test_none_shaped_filenames_are_rejected_one_by_one` 的逐形态参数化);④非日志文件被拒 ✓(`notes.csv` / `random_thing_…`)。**真正丢掉的**只有"同一模态两种命名形态取时间戳更新的那份"这条 —— 按 id 选之后它退化成"同 id 下唯一文件"的退化情形,丢了可以接受(且没有别的测试覆盖它)。另有**增量缺口**:新契约下"`selected_sessions` 标 `loaded` 的键 ⟺ `data` 的键"这条不变量没有任何断言(旧测试的 `set(selected_sessions) == set(data)` 被有意删除 —— 那条不变量已反转,删得对,但反向的新不变量没补)。→ Minor
6. **`RealtimeAnalysis` 不自行起会话那条裁决** —— 见 §5。

---

## 5. 对账本那条待裁决 deviation 的判断(第 6 点)

**判断:按现状(可见但不归属于任何会话)可以接受,不必改;但 spec §5.3 应改,且"可见"这半必须先把 I1 修掉才算真。**

理由(按分量排序):

1. **让一个"实时演示"页去铸真会话,会主动破坏缺省路径。** `/interview/start` 是**当场落一份日志文件**的(实测:我铸号就得到一份只有表头的 `interview_emotion_log_20260924_231815_3ceb.csv`)。若 RealtimeAnalysis 也起会话,那么"随手打开实时页"就会造出一场**更新的空会话**,于是任何"不带 id"的读路径(面板的 `runModule('report')`、前端报告页的 `getStructuredReport` —— 二者现在都**没带 id**,见 I3)都会改去描述那场空会话。我实测过这个后果:`GET /api/report/structured` 返回 `未找到评估日志数据`,而盘上有一场三模态齐全的会话。**即:spec §5.3 原样实现会让今天能用的读路径更容易坏。**
2. **它不违反 M2 的成功定义。** 成功定义是"在前端跑一场真会话 → 覆盖 ≠ 0 且报告头点名那一场";实时页的帧不属于任何一场,本来就不该进那一场。落 NONE 恰好是"没给 id"的既定语义(spec §6 行3、§8.3),不是新状态。
3. **"不许静默"这条约束我判它成立,但只在你补上 I1 之后。** 裁决的理由("报告头会显式点出『另有 NONE 桶 N 行』")目前对 face/gesture **事实不成立** —— 那 224 行是隐形 的。也就是说:**现状下这条裁决是"静默的",而不是它声称的"可见的"**。所以我的结论是"裁决可以照旧,但先修 I1,否则裁决的理由不成立"。
4. **代价可见且有限**:从实时页直接开摄像头,那些帧不归入任何一场(要归入就先走一次开始面试的流程)。这个代价裁决里已经写明,我认可。

**建议**:把裁决写进 spec §5.3(现在只在账本里),把 §5.3 的"取后者(先调 start 再开摄像头)"改成"`RealtimeAnalysis` 不自行起会话;无会话时帧落 `NONE`,并由披露点名";同时把"`NONE` 必须出现在披露里"这条与 I1 一起钉一条测试。

---

## 6. Declined to judge(considered and set aside)

以下行为我考虑过并**明确不判**(不是遗漏,逐条给理由):

1. **409 归属契约** —— spec D5/§8.2 明确本轮不做,且是 D1 之后才谈得上的下一步。
2. **`refresh_manifest` 的真消费者** —— spec §8.1 明确本轮不做(需新增跨目录依赖;披露已用更直接的方式覆盖同一需求)。
3. **VIDEO 模式等价性门 / M3 阈值重登记** —— spec §1 明确不在范围内。
4. **哨兵值歧义(语音能量 50.0、抖动 0.0、专注度三档)** —— spec §8.4 单列;它会让"未读到数据"与真实值长得一样,但改动点在分析器与前端显示。
5. **前端"一直显示在录音"的状态机、无产出方的字段(语音流畅度)** —— spec §8.5 单列的前端独立问题。
6. **`examples/*`、`visualization.py` 的清理** —— 承 M1.5 的 deferred。
7. **`app.py:119` `module_map["voice"]` 指向不存在的 `run_interview_assessment_voice`** —— 确认为既有缺陷(spec §4 说"本轮只做不改,除非它挡住验收");它没挡住验收(我没有通过面板跑 voice 模块)。**记录在此,不计入本轮 findings。**
8. **`NONE` 桶跨天累积 / 无 id 客户端共享一个 pipeline** —— spec §8.3 的 deferred;我只判"是否被披露",不判轮转策略。
9. **`data/output/` 里历史报告累积** —— spec §8.6。
10. **报告 HTML 的注入可能性** —— 我查过:`sources_disclosure` 把 `target` 直接插进 HTML,而 `target` 可能来自 query 参数;但只有当该 id 对应的文件**真的存在**时才会渲染披露,而文件名受正则约束(`\d{8}_\d{6}(_[0-9a-f]{4})?`),攻击者无法仅靠 query 参数把 `<script>` 送进 HTML。**判定不成立,不作为 finding。**
11. **前端 `dist` 体积警告 / plotly chunk** —— 既有,与 M2 无关。
12. **M1/M1.5 的既有行为**(写侧铸号、`_resolve_session_id` 双位置取参、`normalize_session_id` 空串判非法等)—— 属上一轮区间,本轮只在其上做集成面核对。

---

## 7. Assessment

**Ready to merge? —— With fixes(先修 C1,并建议同时修 I1)**

**Reasoning**:M2 的核心目标达成了,而且是可验证地达成:按 id 选日志(反向复现精准红)、报告真的用 `session_id`、面板那条 spawn 路径**实测**把 id 送到了子进程并产出同一场的报告、三份日志的文件名与首列都等于铸号、覆盖从恒 `0 / 20` 变成 `1 / 20`、202 passed、合并门 `0 / 2835510`、前端 build 通过。同时它没碰它承诺不碰的东西(face/gesture/voice 实现零改动,`module_map["voice"]` 的既有缺陷按 spec 只记不改),`NONE` 不进聚合这条不变量也在正则层被钉住。

必须处理的两点:**C1** 是本次改动引入的功能回归(实时报告路径崩在 `sources_disclosure`,面板还报 success),一行可修;**I1** 让"`NONE` 必须出现在披露里"这条 Global Constraint 在现实数据上**完全不成立**(224 行隐形),而账本第 6 点裁决的理由正建立在它上面 —— 修它也是一行 glob 加一条测试。I2/I3 是 spec 明文(§6 行1/2、D2 的"写明是哪一场")在前端读路径上的缺口,不修不影响"一场会话对上号",但会让"报告是谁的"在界面上仍然说不清。

---

## 附:我的实跑对环境的副作用(需要你知道)

1. **我杀掉了自己这次启动的 `vite`(5173)与 `xdg-open`**:为解开 §2.1 的两处卡死。跑之前 5173 是空闲的,现在也回到空闲(状态与跑之前一致);8000/8001/8002/5000 我**没动**,仍在跑。
2. **`webbrowser.open` 起了一个 msedge**(23:09 那次 `webbrowser.open` 的直接后果),可能还开在屏幕上,窗口里是那份报告。
3. **我多造了一场空会话**:面板透传测试需要一场"更新但为空"的会话,我用 `POST /interview/start` 铸了 `20260924_231815_3ceb`,它当场落下一份只有表头的日志。**它现在是盘上"最新一场"**,所以前端那条**不带 id** 的报告路径会因此报"未找到评估日志数据"(`/api/report/structured` 实测如此),直到你跑一场新会话为止。要清掉它:
   ```bash
   rm -f ~/jingxin/data/logs/interview_emotion_log_20260924_231815_3ceb.csv
   ```
   (我没有替你删 —— 你要求不动任何东西。)
4. 验收脚本与面板测试在 `data/logs`、`data/output` 下留下的新文件都在 gitignore 里(`**/data/logs/`、`**/data/output/`),**不污染工作树**。
5. 本报告是那个目录里新增的**未跟踪**文件;提交与否听你的。
