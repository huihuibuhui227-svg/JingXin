# SDD ledger — plan: docs/superpowers/plans/2026-09-25-m2-6-media-upload-and-metadata.md

执行者:inline(使用者 2026-09-25 选 native)。baseline 全量套件 **317 passed**。

## Setup 的裁定

**Setup: Ruling: 账本落在仓库跟踪的 `docs/superpowers/sdd/…/progress.md`,而不是技能默认的
gitignore 区 `.superpowers/sdd/…`** — 项目 §5 明写"M2 账本直接建在 git 跟踪的目录 ——
吸取『复审报告落在 gitignore 目录里差点丢』的教训",且 `6c20a46` 已把 M2.6 账本从
草稿区归档进仓库。做法:技能脚本写的 `<ws>/progress.md` 用软链接到跟踪的那份,
脚本与项目约定都不用改 — 代价:若有人 `git clean -fdx`,软链与草稿区一起没了,
但**跟踪的那份在 git 里**,重建软链即可。

**Setup: Ruling: 在 `main` 上直接做,不新开分支/worktree** — 使用者本会话的既定工作流
就是 main(`N1` 的两个提交 `ed834fd`/`71c5fee` 即落在 main 并被要求推送),且他们刚批准的
这份计划的提交步骤就是在当前分支。**推送另说 —— 全部做完、终局复核过之后再问。**
代价:若中途要放弃,得手工 reset;本地提交可回退,不影响远端。

## Pre-flight 扫描:跨任务接口

| 生产者 → 消费者 | 生产的东西 | 消费者用它做什么 | 核对结果 |
|---|---|---|---|
| T1 → T2 | `retain_uploaded_video(sid, data, source="")`、`LEDGER_WRITERS += "camera"` | 端点调它落盘;读返回行的 `bytes`/`sha256`/`file` | 一致(位置参数 + `source=`,与 `retain_audio` 同形) |
| T1 → T6 | 同上 + `ledger_files()` 能捞出 `retention.camera.jsonl` | `_media_counts` 按账本行数数件数;`video` = `counts["camera"]>0` | 一致(`_record(sid,"video","camera",…)` ⟹ `modality="camera"`) |
| T3 → T4 | `upsert_question(sid, *, qid, index, ask_start, ask_end, source="")` | 端点把请求体转成这五个关键字参数 | 一致(全关键字调用) |
| T3 → T6 | `read_questions()`、`_session_dir(create=False)` | 对账读题号;读侧不建目录 | 一致 |
| T5 → T6 | `META_REQUIRED`、`missing_meta_fields()`、`meta_path()`、`write_template()` | 对账点名缺项;测试里铺模板 | 一致 |
| T5 ↔ T3 | 都往会话根写,不同文件(`meta.json` / `questions.jsonl`) | — | **无冲突**(各自一把会话锁、各写各的文件;`write_template` 已存在则抛,不会覆盖) |

**一处计划内部张力(已裁定)**:T3 的 `_is_blank` 把 `False` 也算"没填",而 T5 的模板
`consent.archived` 默认就是 `false` —— 两者是**配套**的(否则没做知情同意的场次会过校验),
T5/T6 的测试都按"`True` 才算填了"写。见下。

**Pre-flight: T5/T6 的 `_fill_meta` 必须把 `consent.archived` 置 `True`**,否则
`test_everything_present_reports_nothing_missing` 会红 —— 计划里已如此写,记录备查。

---

## 任务进度

Task 1: complete (commits 1dbbd76..3b1b4a9, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention_video.py tests/test_media_retention.py tests/test_media_retention_concurrency.py -q → 29 passed in 0.11s)
Task 2: complete (commits 3b1b4a9..e24a969, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_media_endpoint.py tests/test_media_retention_video.py -q → 13 passed in 0.41s)
Task 3: complete (commits e24a969..0680c96, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_meta_questions.py -q → 12 passed in 0.02s)

Task 4: Ruling: 计划的测试里 `_post()` 少了一层 `asyncio.run` —— 端点是 `async def`,
  不跑协程的话 `_post()` 返回协程对象,`r["status"]` 抛 TypeError 而
  `pytest.raises(HTTPException)` 永远等不到 —— **红得不对,那条测试什么也没验到**。
  改法:`_post` 内 `return asyncio.run(...)`,`test_illegal_session_id_is_400` 同样包一层,
  并加 `import asyncio`。这是最小的满足 spec 的改动(测试必须真的执行端点行为)。
  — 代价若判错:无(纯测试写法;生产代码不受影响)
Task 4: complete (commits 0680c96..c97b67a, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_question_endpoint.py tests/test_session_meta_questions.py tests/test_session_media_endpoint.py -q → 21 passed in 0.42s)

Task 5: Ruling: 计划的 `META_REQUIRED` **漏了 `questions`**(题目 ID + 难度)——
  录制需求 §3.2 那张表明列它为必填("审查称最明显的遗漏:数据集里 74 个不同题目,
  你的协变量里一个都没有"),而计划只列了 candidate/capture/ratings/consent 四块。
  改法:把 `"questions"` 加进 `META_REQUIRED`(空列表 = 缺 ⟹ 拦住"整个题库块忘了填"
  这个真实失败形态)。**逐条 `difficulty` 的空白不在这里判** —— 本场实问题数服务端
  不知道(题目可中途结束),逐题覆盖由 Task 6 的题号对账负责;这一点写进了代码注释。
  — 代价若判错:多报一条"questions 没填"(假阳性),不会漏报;反向的风险
  (整个题库块被漏填而校验说"齐了")比它大得多
Task 5: complete (commits c97b67a..7f19404, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_meta_template.py tests/test_session_meta_questions.py -q → 17 passed in 0.03s)

Task 6: Ruling: 计划里 Task 6 的 `_fill_meta()` **没有填 `questions`** —— 这在计划写下时
  是对的(那时 META_REQUIRED 没有它),但 Task 5 的裁定把它补进必填之后,
  `test_everything_present_reports_nothing_missing` 的 `missing_meta == []` 必然红。
  改法:`_fill_meta` 补 `m["questions"] = [{"qid":…,"difficulty":…}, …]`。
  这正是 pre-flight 表里那行「T5 → T6 校验口径」的实际落点,记录备查。
  — 代价若判错:无(测试夹具;生产代码不受影响)
Task 6: complete (commits 7f19404..b3a20b5, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_closeout.py tests/test_session_meta_template.py tests/test_session_meta_questions.py -q → 23 passed in 0.05s)
Task 7: complete (commits b3a20b5..fd6e54d, tests: /home/huihuibuhui/miniconda3/envs/jingxin/bin/python -m pytest -q → 356 passed, 5 warnings in 26.78s)

---

## 终局独立复核(opus,fresh context,18 个变异)

判 **Critical 0 / Important 7 / Minor 9**;复核者自己复跑了全量套件、合并门与真 HTTP。
原话里最重的一句:*"16 of 18 guards genuinely bind — the two that don't are exactly the
two introduced by late Rulings"*(空钉子恰恰出在后期裁定带进来的那两处)。

### Final: fixed(七条 Important,一次修复波次)

- **Final: fixed I1 量程闸** — `test_implausible_wall_clock_windows_are_rejected`(毫秒/
  相对时钟/30 天前 3 例 + 正常窗口的对照例)RED→GREEN。套件 367/367。
  顺带实测出计划的"墙钟样例" `1758824000` **离现在一年**(真实留存的
  `received_at_wall` 是 `1790340000` 量级),既有测试里的玩具时间戳一并改正。
- **Final: fixed I2 碎行只报不抛** — `test_torn_ledger_line_is_reported_not_raised`
  RED→GREEN(好的行照数、碎行计数报出)。新增 `unreadable_lines` 字段。
- **Final: fixed I3 空体** — `test_empty_upload_is_rejected_instead_of_reported_as_stored`
  + `test_zero_byte_camera_row_is_not_counted_as_video` RED→GREEN。
  ⚠️ 只拒**空体**,**不拒"认不出的容器"**:浏览器换个容器(mp4)是合法录像,
  拒了就是**丢真素材** —— 这比多存几个怪字节糟得多。复核建议里"reject non-EBML"
  这部分我**只采纳一半**,理由如上。
- **Final: fixed I4 假绿灯** — `test_verdict_is_never_green_with_no_reported_questions`
  + `test_verdict_reports_each_gap_separately` RED→GREEN;抽出
  `closeout_verdict(result, expected_questions) -> (ok, reasons)`。
- **Final: fixed I5 两处空钉子** — `test_required_list_covers_the_recording_requirements_table`
  (补 `questions`)与 `test_degraded_rows_are_never_counted_as_material` RED→GREEN。
  第二条**特意用"带字节数的降级行"**去钉:原先靠"降级行恰好 bytes=0"被 I3 那条
  顺手滤掉,不靠那个偶然。
- **Final: fixed I6 纸面归档** — 计划与账本原先**都没进 git**(我的 Setup 裁定
  断言"跟踪的那份在 git 里",执行时却没 `git add` —— 裁定与执行不符,已更正);
  `docs/下一步.md` 的第 2 件与 §5 索引补齐。见下一个 docs 提交。
- **Final: fixed I7 上传上限** — `test_oversized_upload_is_413_and_writes_nothing`
  RED→GREEN;`MAX_UPLOAD_BYTES = 512 MB` + 分块读。

### Final: Ruling:复核"Declined to judge"那 12 条 —— 逐条维持,不改

1. spec §5.1 的 `preflight()` vs 私有的 `_prepare()` —— 落在**上一个已交付的腿**里,不在本区间。维持。代价若判错:文档与实现命名漂移。
2. 音频账本把 `raw` + `converted` 数成两件 —— 上一腿的既定语义,盘上确实是两个文件。维持。代价:件数比"回答数"大,已用模态名区分。
3. 前端半 —— 本计划明确不含。维持。代价:无。
4. `m26_media_acceptance.sh` 留在 `~/shared` 不进仓库 —— 计划如此规定,且它只往 mktemp 目录写。其步骤⑤ 因 URL 编码的 `%2F` 匹配不到路由而返回 404,**没真的验到守卫**;但 pytest 里那条钉住了,脚本另行断言"盘上无越界目录"。维持。代价:脚本那一步偏弱。
5. 要不要现在录 3–5 场 —— 是 N3/使用者的范围。维持。代价:无。
6. `questions.jsonl` 存题目文本 vs M1 的"原句不进仓库" —— 那条边界管**候选人说的话**;题库是系统固定输入,且文件在仓库外;受伦理前置的自评量表明确不在内。维持。代价:若将来对"文本"收紧,这是一处要复议的地方。
7. 媒体轮转 —— spec §8.4 已明说本轮不做。维持。代价:磁盘只增不减(127 GB 可用)。
8. 不复用 `refresh_manifest` —— spec §8.5 明确允许"接不上就另写"。维持。代价:多一处对账入口。
9. 合法但不存在的 `session_id` 会建目录 —— 本项目**每一个**留存路径都这样,不是新决定。维持。代价:空会话目录(与 Minor ② 相关)。
10. 畸形 JSON 体回 422 而非 400 —— FastAPI 约定;真正要紧的 400 情形都显式处理了。维持。代价:客户端要区分两种 4xx。
11. `_session_lock` 在本仓有三份拷贝 —— 计划如此,且只有正则那份有契约测试。维持。代价:改锁语义要改三处。
12. 没验 `camera.webm` 的音轨可用 —— spec §7.7 把这条派给人工;它就是 §8.2 的 R5 风险本身。维持。代价:要等前端半的实测。

### Final: minor (deferred)—— 9 条,登记不修(复核判 Minor,不进修复波次)

1. `--write-template` 的 help 写"已存在则不动"而实现是抛(CLI 已转成友好提示,help 未改)。
2. `check_session` 读侧会建会话目录(`ledger_files → recording_dir` 总 `mkdir`),与"读侧不该有副作用"相悖。
3. **实际帧率/音频采样率没有机器记录方**:spec §5.5 把这两项列在"机器记",而 `retention.jsonl` 里没有这俩字段 ⟹ 录制需求 §3.4 清单第 3 条目前**无从核验**。(复核建议把它们加进 `META_REQUIRED`,但那与 §5.5 的"人填/机器记"划分冲突 —— 机器记的东西不该让人填,所以正确修法是给账本加字段,归 N2 前端半或 M3。)
4. 模板里 `questions: []` 没有形状提示。
5. 两个端点对 `JINGXIN_RETAIN_MEDIA=0` 的处理不同(`/question` 照写 —— 语义正确)但没写下来。
6. 重复预检失败会把原因写两遍。
7. 端点级没测"中途写失败 → `stored:false`"那条分支(功能已验证,只是没钉)。
8. `questions.jsonl.tmp` 固定名 + 单进程前提未声明。
9. `async def` 里的阻塞文件 I/O(与既有留存钩子同形,非新增)。
