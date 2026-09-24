# SDD ledger — plan: docs/superpowers/plans/2026-09-22-report-layer-evidence-gate.md

- 分支:`fix/report-layer-evidence-gate`
- MERGE_BASE / BASE: `07ed73d`
- 解释器:`~/miniconda3/envs/jingxin/bin/python`
- Spec(权威):`docs/superpowers/specs/2026-09-22-jingxin-report-layer-evidence-gate-design.md`

## 前置状态

- [ ] **pytest 未安装** —— Task 1 Step 4 起全部依赖它。使用者命令:
      `~/miniconda3/envs/jingxin/bin/python -m pip install pytest`
- 说明:报告层 `__init__.py` 会连带导入 `visualizer`(依赖 plotly),故**不能用** `~/huihui/bin/python`。

## 预检冲突扫描(dispatch Task 1 之前)

### 共享文件 / 接口的任务对

| 任务对 | 共享物 | 一方产出 vs 另一方消费 | 发现 |
|---|---|---|---|
| T2 → T3 | `gate` / `confidence_from` / `Check` / `check_g2_from_std` | T2 产,T3 消费 | ✅ 签名一致 |
| T2 → T3 | `QUARANTINE` 名单 | T2 产,T3 经 `gate` 间接消费 | ✅ |
| T3 → T4 | `feature_engine.py` | T3 改 `extract_all_features`(加 `__n_rows`);T4 改 `_extract_numeric_stats`(跳过规则) | ✅ 不同函数,无冲突 |
| T3 → T5 | `research_mapper.py` | T3 删 `self.baselines` **及**其使用点;T5 原本也要删同一使用点 | ⚠️ **冲突** → 已修(见 Ruling 1) |
| T3 → T7 | `research_mapper.py` | T3 改证据门与权重;T7 改 `mapping_rules` 的 name/description、删 `_generate_deep_inference` | ✅ 不同区域 |
| T3 → T6 | `score=None` / `confidence="无"` / `evidence_gaps` | T3 产,T6 消费 | ✅ |
| T3/T5/T6/T7 | `tests/test_report_layer.py` | 四个任务各自**追加**测试 | ✅ 追加式,不冲突 |
| T5 → T7 | `visualizer.py` | T5 删 `baselines`、拆 `_build_radar_figure`;T7 改 `create_gaze_plot_from_df`、删死代码 | ✅ 不同函数 |
| T6 → T7 | `BANNED` 常量 | T6 测试内定义;T7 测试内**重新定义** | ✅ 已修(避免跨任务引用) |
| T6 → T7 | 禁止词约束 | T6 的 `BANNED` 含"压力";T7 若把维度改名含"压力"会被 T7 自己的扫描打回 | ⚠️ **冲突** → 已修(见 Ruling 3) |
| T1 → 全部 | `tests/conftest.py` | T1 产 sys.path 注入,全部测试消费 | ✅ |
| T8 | 独立文档 | 无代码接口 | ✅ |

### 每个任务的自我一致性

| 任务 | 自己的测试 vs 自己的代码 | 发现 |
|---|---|---|
| T1 | 冒烟测试只断言包可导入 | ✅ |
| T2 | 覆盖 G1 / G2(序列) / G2(std) / G3 / G4 / 置信度 / 阈值登记 / 关卡顺序;实现含全部 | ✅ |
| T3 | `test_dimension_weights_sum_to_one` 要求每维和 = 1.0;Step 4 删 `fluency_proxy`(0.4)后 comm = 0.4+0.3+0.2+0.1 = 1.0 | ✅ 一致 |
| T3 | 代码块中 `gaps_for_this_dim` **未定义**、`evidence_gaps` 作用域错(方法级 vs 维度级) | ⚠️ **缺陷** → 已修(见 Ruling 2) |
| T3 | Step 3 删 `self.baselines` 但 `map_features_to_scores` 里的使用点未删 → 运行即 `AttributeError` | ⚠️ **缺陷** → 已修(见 Ruling 1) |
| T4 | 两个测试对应两个改动(question_index / 死分支) | ✅ |
| T5 | `test_radar_has_no_norm_baseline` 依赖新函数 `_build_radar_figure`,同任务内定义 | ✅ |
| T6 | 三个测试对应三类删除;`_CONF_ORDER` / `_render_dimension_block` 同任务内定义 | ✅ |
| T7 | 扫描测试扫全仓字符串字面量;T7 自己要写的新维度名不含禁止词 | ✅ 已修 |
| T7 | `create_gaze_plot_from_df` 改为 `return None` → `generate_all_charts` 走 else 分支打印警告,不崩 | ✅ |
| T8 | 纯文档,无测试 | ✅ 无需 |

## Rulings(开工前)

**Ruling 1:** 把"删百分位计算"从 Task 5 移到 Task 3 Step 3 — 因为 Task 3 删掉 `self.baselines` 之后,`map_features_to_scores` 里的 `bl = self.baselines.get(...)` 会让程序一运行就 `AttributeError`,这不是风格问题是硬缺陷。Task 5 相应缩为"确认无残留"。 — 若判断错,代价 = 一次 fix round(改动量约 10 行)。

**Ruling 2:** `evidence_gaps` 拆成两层 —— `all_evidence_gaps`(方法开头声明一次,进返回值)+ `dim_gaps`(维度循环内声明,进该维度条目)。原计划只写了一个 `evidence_gaps` 且放在方法级,会让 5 个维度共用一个缺口列表,缺口串味。 — 若判断错,代价 = 各维度证据缺口显示错误,影响报告可信度;返工约 5 行。

**Ruling 3:** Task 7 的维度名定为「情境行为稳定性」而非「压力情境下的行为稳定性」 —— "压力"在 spec §5.6 的禁止词表里,用了会被 Task 7 自己的 `test_no_banned_words_in_output_strings` 打回,形成自相矛盾。 — 若判断错,代价 = 命名与使用者预期不符,改一个字符串。

**Ruling 4:** 用**分支**而非 worktree 做隔离 —— 使用者未要求 worktree(harness 规定 EnterWorktree 仅在明确要求时使用),且计划本身写的就是 `git checkout -b fix/report-layer-evidence-gate`。分支已建,MERGE_BASE = `07ed73d`。 — 若判断错,代价 = 使用者想要 worktree 时需重来一次 setup,无代码损失。

**Ruling 5:** T3 与 T4 都改 `feature_engine.py` 但落在不同函数(`extract_all_features` / `_extract_numeric_stats`),判定**无冲突,不调整任务顺序**。 — 若判断错,代价 = T4 的 implementer 需重读 T3 的改动,一次 fix round。

**Ruling 6:** Task 8(修订设计文档三处前提)保留在计划内、排在代码任务之后 —— 它是纯文档且与 ① 无代码依赖,但它关闭 spec §1 的三个前提,不做会让设计文档与增补自相矛盾。 — 若判断错,代价 = 文档不一致,后续读者困惑。

### Ruling 1 的完整性核验(开工前已实测)

被删属性的全部使用点已用 grep 数清,**无遗漏**:

| 属性 | 出现处 | 由谁删除 |
|---|---|---|
| `self.baselines` | `:40` 定义 | T3 Step 3 |
| | `:204` 兜底 BASELINE_FILL | T3 Step 6 |
| | `:234` 百分位 | T3 Step 3(本 Ruling 移入) |
| `self.demo_text_data` | `:19` 定义 | T3 Step 3 |
| | `:135` 假简历 fallback | T3 Step 6 |

3/3 与 2/2 全覆盖 → Task 3 完成后 `research_mapper.py` 不会再引用已删属性。

## 使用者指令(2026-09-22,开工时)

**指令 A:pytest 已装** —— `pytest 9.1.1`(conda env `jingxin`)。阻塞解除。

**指令 B:语音模块的 vosk 要换成 FunASR。**
- ⚠️ **不属于本计划(①)。** ① 范围仅 `report_frontend/` + `templates/`,已实测确认报告层**完全不碰 ASR**(`grep vosk|asr report_frontend/ templates/` = 0 命中)。
- vosk 实际位置:`voice_interaction/api/app.py`、`voice_interaction/pipeline/speech_recognition_pipeline.py`。
- **归属:M1(ASR 落盘)** —— 即把 `logic_keyword_density` 从假简历常量变成真测量的那一步。到 M1 时必须带上本条。
- 已有可用信息(来自长期记忆,需在 M1 时复核):FunASR 走 `192.168.72.30` 直连(无需隧道);systemd 服务名 `funasr-streaming`;逐字毫秒时间戳可用。这三项正是 M1 需要的。

**指令 C:每个模块/任务完成后，使用者要先单独测试再继续。**
- 与 SDD 默认("任务间不停下来")冲突 → **以使用者指令为准**。
- 落法:每完成一个文件的相关改动,停下来给出「一条可跑命令 + 该看什么」,等使用者确认后再进下一个。
- 测试门划分:门1 = 证据门(T2);门2 = research_mapper(T3+T4);门3 = feature_engine(T4);门4 = visualizer + report_generator(T5+T6+T7)。

**指令 D:使用者建议"4 个子智能体并行改 4 个模块"。**
- **Ruling 7:不采用并行。** 理由:`research_mapper.py` 被 T3/T5/T7 改,`tests/test_report_layer.py` 被 T3–T7 追加 —— 并行会写同一文件互相覆盖;且 T3 删 `self.baselines` 是 T5/T7 的前提,顺序是依赖不是洁癖。SDD 流程亦明文禁止并行派发实现者。
- 满足使用者真实诉求("每个模块单独测")的方式:**串行派发 + 每文件收尾设测试门**(见指令 C)。
- 到 **M3**(面部/手势/语音三模块 L0 重写)时,模块真正独立,**那时并行才成立** —— 已在设计文档中写明会主动提出。
- 若判断错,代价 = 使用者想要并行而未得,多花几次串行等待;无代码损失。

## 进度

- setup 完成:分支 `fix/report-layer-evidence-gate`(base `07ed73d`)、工作区、账本、预检扫描、Ruling 1 完整性核验。
- 阻塞已解除(pytest 9.1.1 就位)。
## 仓库状态事件(阻塞 → 已解决)

**发现**:Task 1 的 implementer 报回"工作区本来就脏"。实测确认了两件独立的问题,均**与 Task 1 无关**:

1. **行尾不一致** —— `report_frontend/*.py`、`templates/*.html` 在磁盘上是 CRLF、HEAD 里是 LF,`core.autocrlf` 与 `.gitattributes` 都没有。后果:`git diff` 把整文件显示为改写(visualizer.py 592 行、research_mapper.py 820 行),**审查环节会失效**。
2. **大量未提交工作** —— 12 个文件、787 增 / 220 删,是使用者的 API 层 + 实时报告 + 视频管线工作。

过程记录:`git diff --name-only` 曾误导我(报 76 个实质改动),实际用 `--numstat` 过滤后才看清**真正有内容改动的只有 12 个**,其余 ~180 个是纯行尾。

**使用者决定(2026-09-22)**:①未提交工作单独提交;②加 `.gitattributes` + 归一化。

**已执行**:
- `f506240` — `chore: 新增 .gitattributes 统一行尾为 LF`(`* text=auto eol=lf` + 二进制例外)
- `8d156f4` — `feat: 三个模块 API 服务与报告层实时模式`(使用者的 12 个文件,787+/220−)
- **效果**:已跟踪改动 197 → 剩 64 个 `.pyc`(编译产物),**代码文件 0 改动**
- **关键**:`text=auto` 让 git 读取工作区时自动归一化,**~180 个纯行尾文件无需任何提交就变干净** —— 不需要一次性大提交
- **Ruling 8:** 提交信息里显式标注"既有工作,非本次任务产出",避免历史归属错误 — 若判断错,代价 = 提交信息措辞需修订(仅文案)。

**遗留(未处理,等使用者决定)**:64 个 `.pyc` 仍被 git 跟踪且已从磁盘删除。建议加 `.gitignore` + `git rm --cached`,但那是仓库卫生改动,未获明示授权,不擅动。**不影响我们的任务 diff**(review package 走 `git diff BASE..HEAD`,只看已提交范围)。

## 派发记录

**Task 1 的 5 条 concern 处置(控制器裁决)**

- **Ruling 9:** 仓库根目录跑裸 `pytest` 会收集到 4 个既有 scratch 脚本(`experiments/test_gpt56.py` 等,其中一个硬编码 `D:/jingxin/...`)而报错。计划里所有命令都已限定 `tests/`,故不阻塞;但为防后患,**把"新建 `pytest.ini`(`testpaths = tests`)"折进 Task 2 的派发** —— 不为它单开一轮。 — 若判断错,代价 = Task 2 多改一个 2 行文件。
- **Ruling 10:** 仓库没有 git 身份配置。**继续用 `-c user.name/-c user.email` 每次提交带参**(取自仓库历史的 `huihuibuhui227 <huihuibuhui227@gmail.com>`),**不写任何配置文件**。 — 若判断错,代价 = 使用者需自己设一次 local config,无代码损失。
- **Ruling 11:** smoke test 里 `assert X is not None` 在 import 成功后是同义反复。**标为 plan-mandated minor,记入 deferred,不进修复循环**(断言确实无用,但 import 本身是真测试,无正确性风险)。 — 若判断错,代价 = 最终审查需决定是否加强该测试。
- concern 4(`.pyc` 附带物已清)与 concern 5(已并入 Ruling 11)不需动作。
- concern 2(工作区脏)已按上方"仓库状态事件"解决。

**Task 1(测试脚手架)** — **complete**(commits `0362b98`,审查干净)
- 审查结果:Spec ✅ 合规 / Task quality **Approved**;0 Critical、0 Important、3 Minor
- ⚠️ 项闭环:①环境可跑(控制器已亲验 `import report_frontend` 成功);②"182 个改动文件"为真,已在"仓库状态事件"中解决;③rootdir 表述不精确属报告措辞,不影响代码
- Minor 处置:`is not None` 冗余(审查者订正为无害,属性访问有约束力)→ 关闭;`.pytest_cache/` 未 gitignore → 并入卫生清理;**deferred minor**:`.pytest_cache/` 与 `__pycache__/` 一并加 `.gitignore`(待使用者批准)

**Task 2 审查结果**:Spec ✅ 合规 / Task quality **Approved**;0 Critical、2 Important、6 Minor。审查者逐行比对 brief,确认三个文件字节一致、五项裁决全部落地、12 个接口签名齐备、21 条断言手工推演全部成立、禁止词 0 命中。处置:

- **Ruling 16:** Important#1 成立 —— `evidence_gate.py:102` 的 `return 10` 是**裸常量**,而计划的 Global Constraints 与 spec §5.1 都明禁此写法。**这是计划自身的缺陷,spec 是权威 → 修**。默认值移入 JSON 的 `_default_n_valid`,并顺带加严 `load_thresholds` 守卫(原守卫只查键存在,`_provisional: false` 也能通过)。 — 若判断错,代价 = 多一个 JSON 字段与一次读盘。
- **Ruling 17:** Important#2 —— 审查者按 Minor 报,但指出"请确认 Task 3 不会直接渲染 `reason`"。**控制器核查计划后确认 Task 3 会**:`dim_gaps.append(f"{human_name}: {chk.reason}")`,而 Task 6 把这些 gap 渲染进报告"证据缺口"一节 → **用户会看到"已封停:可由同 block jitter 精确重构(解封:M3)"**。**升为 Important 并本轮修**(跨两个任务,现在修远比到 Task 3 再修便宜)。新增 `user_message(check)`,报告层只准用它;`Check.reason` 保持原样供维护者。 — 若判断错,代价 = 多一个 6 行函数与 2 条测试。
- **Minor 全部不进循环**(6 条):G2 无 std 时 fail-open(brief 有意,已有测试钉住,→ 在 Task 3 派发时点名要求必传 `_std`)、`test_first_failure_wins` 断言可更严、未用 import 噪音、`_threshold_for` 每次读盘、pause 族 G3 阈值由被遮蔽的 30 变为登记值 2(**这是 Ruling 14 修 bug 的预期结果**,Task 3 需知悉)。

**处置动作**:计划修正 6 处 → 重新生成 brief(451 行)→ SendMessage 恢复原实现者,fix 轮 1/5。预期 25 项全过。

## 门 1(使用者亲验,2026-09-22)— 通过

使用者跑了两条命令,结果与预期一致:

- `pytest tests/test_evidence_gate.py` → **24 passed**(单文件);裸 `pytest` → **25 passed**(含 Task 1 smoke)
  - ⚠️ 控制器在门 1 说明里把"预期 25"错写成单文件的预期值;已向使用者更正(24 是单文件,25 是含 smoke 的全量)
- 敌对探针:恒定值(std=0)→ `False`/G2/「本次会话内无变化」;封停列 → `False`/G4/「该指标本轮停用」—— 语义正确,且内部文案未外泄

**控制器补做的核对(使用者要求)**:
- 封停名单 28 条对 spec §5.2 逐行核对,唯一缺口是面部 CSV 的 `confidence` 列未列入;实测该列**不会变成特征键**(272 个真实特征键中含 confidence 者 0 个),缺口无害
- **证据门在真实数据上的咬合力:272 个真实特征键中 110 个(40%)被封停名单拦下**

**Task 2: complete (commits `8d156f4..e249450`, review clean)**
复审判定:两条 Important 全部 ADDRESSED,无新增 Critical/Important。复审者做了独立探针(JSON 默认值改 7 验证来源;逐关卡验证文案不泄漏;确认 `Check.reason` 仍完整 —— 是"加一层"非"替换")。

**deferred minor(交最终审查分诊,不进修复循环)**

- `evidence_gate.py:88` — `_default_n_valid` 只查存在、不查类型/符号。`-5` 能通过守卫并让 G3 对所有未登记指标永不失败;字符串会晚到 `_threshold_for` 才抛。同类问题也存在于 `thresholds` 块(计划未要求校验)。
- `evidence_gate.py:193` — G4 对**永久**封停项也显示"该指标本轮停用",暗示会回归,而它不会。
- `tests/test_evidence_gate.py:2,5` — `import pytest` 与 `Check` 未使用(计划原样带入的噪音)。
- `evidence_gate.py:107` — `_threshold_for` 每次调用都重读并解析 JSON(Task 3 在指标循环里调用)。
- `user_message(Check(True))` 走回退返回"证据不足" —— 当前不可达(Task 3 只在 `if not chk.ok` 下调用),但该守卫一旦去掉就会显形。

**供 M5 参考(超出本任务)**:`load_thresholds` 的 `_provisional is True` 是严格判定 —— M5 换成已标定文件时若丢掉该标志会硬失败。失败即响是刻意的,但替换时需**有意的代码改动**,不是纯 JSON 替换。

**Task 2 实现记录(`08e31fe` + fixes)** — 已并入上文

- fix 轮 1 结果:两个独立提交(`fa44d99` 默认阈值入 JSON / `e249450` 新增 `user_message`),25 项全过
- 实现者的**反作弊验证**(值得记):只在 JSON 里把默认值改为 7、不动代码 → `_threshold_for("unregistered_xyz") == 7`,证明默认值真来自配置;单靠"断言键存在"测不出这点
- RED 恰好 3 条失败(2 ImportError + 1 JSON 缺键),无第 4 条 → 改动未波及既有行为
- 复审包:`review-08e31fe..e249450.diff`(2 commits, 7.5KB),已派 sonnet 做**限定范围复审**

**Task 3(research_mapper 接入证据门)** — 实现完成(`36f1f87`),fix 轮 1 进行中

实现者报 **DONE_WITH_CONCERNS**,并在 brief 里找出 **7 处真实缺陷**(D1–D7),全部已在实现时按最小方式处置。控制器逐条复核后**全部采纳**:

| # | brief 缺陷 | 严重性 |
|---|---|---|
| D1 | Step 5 产出 `__n_rows` 但 Step 6 查询 `__n_rows` —— 扁平化后是 `face___n_rows`(3 下划线)vs 查询 `face__n_rows`(2 个)→ G3 全盘拦下;**而 5 条测试仍全绿故被掩盖** | 高 |
| D2 | `split("_",1)[0]` 对 `voice_research_*` 得 `'voice'` → 语音指标 n_valid 恒 0 | 高 |
| D3 | 删百分位后 `_generate_deep_inference` 在 try 块外 KeyError。处置用了**留空**而非 `.get()` —— 理由:`.get()` 会让"视线稳定性正常/肢体略有紧张"这类**无证据断言**分支从不可达变可达,**在以"删除伪造"为目标的改动里反向打开伪造** | 中 |
| D4 | `score=None` 的两处未防护比较(`_get_level(None)`、`_summary_narrative` 的 `> 0`) | 中 |
| D5 | Step 5 无保护访问:面部缺失时 KeyError → 落进 `extract_all_features` 的宽 except → **返回 `{}`,连手势与语音一起丢**。纯附加项放大成整模块失效 | 中 |
| D6 | Step 11 的 `git add` 漏 `feature_engine.py` → 提交消费者不提交生产者,n_valid 恒 0 | 中 |
| D7 | 死变量 `matched_count`/`found_valid_evidence`;`evidence_gaps` 未覆盖出分维度 | 低 |

**控制器裁决(2 条修复,已发 fix 轮 1)**:

- **Ruling 18:** concern 1 **成立,是计划级缺口** —— spec §5.3 的处置是**槽位级**、§5.2 是**列级**,那 7 个槽(`gaze_stability`/`au4_freq`/`au7_freq`/`jitter`/`speech_ratio`/`eye_contact`/`fluency_score`)没有任何任务翻译进 `QUARANTINE`。实测 4 个绕过 G4、其中 3 个真的出了分。**已核 `jitter` 误伤风险:安全** —— `mapping_rules` 里只有 `jitter` 这一个槽读 jitter 键;且 design doc §4.2 说 jitter 要的是 **M3 归一化(÷肩宽÷时间)** 而非永久删除,故 `unblock` 写明。补 7 条 + 一条槽位级覆盖测试 + 要求贴出逐维度证据缺口清单。 — 若判断错,代价 = 误封停一个真实测量(但可 M3 解封)。
- **Ruling 19:** concern 2 —— Task 6 确实覆盖,但它还在三步之后,而**中间几道门使用者会自己跑报告**;`generate_report` 的宽 except 会吞成空报告,是最易被误判为"改坏了"的状态。**裁决现在加最小防护**,并在注释里显式标注"Task 6 会整段重写本函数"。 — 若判断错,代价 = Task 6 会覆盖这几行,无残留。

**计划已按 D1/D2/D5/D6 + D3/D4 处置说明修正**(5 处编辑)。

**fix 轮 1 结果**(`6a80a10`,未 amend):31 项全过(原 30 + 槽位级覆盖测试)。

- **新测试非恒真**:实现者载入旧名单(`git show 36f1f87:...`)实测,6 个键 6/6 由"有"变"无",修复前必失败。既有 25 项 Task 2 测试用键与新 7 键无子串交叠,全部保持通过。
- **修复 1 效果**:sessA 端到端从"3 维出分、总分 64.98"→ **21/21 槽被拦下的诚实空报告**(与 spec §7.3 预期一致)。逐槽:`gaze_stability`/`au4_freq`/`au7_freq`/`eye_contact` 由 **G4** 拦;`jitter`/`speech_ratio` 由 **G2** 拦;`fluency_score` 无键。
- **修复 2 效果**:零证据会话上 `_generate_deep_text_analysis` 与 `_build_html_report` 均正常返回(HTML 7964 字符);此前 TypeError 被宽 except 吞成空报告。
- **Step 10 重跑**:`conf=高` 0 次、总分 None、5 维全"证据不足"。

**deferred minor(新,交最终审查分诊)**:关卡顺序是先失败先返回,故"既恒定又封停"的槽(`jitter`/`speech_ratio`)由 **G2 先响**,用户看到「本次会话内无变化」而非「该指标本轮停用」。两条文案都成立且不泄漏内部信息,但后者更准确 —— 指标其实已退役,"无变化"会让人以为它在用。**控制器判 Minor、不进循环**:指标两种情况都被正确排除,只是解释措辞;改它需重排关卡顺序,而 Task 2 的 `test_first_failure_wins` 正钉着当前顺序。**已向使用者明示:若产品上要"封停优先",那是他的决定。**

**任务级 deferred minor(累计,交最终审查)**:`_default_n_valid` 只查存在不查类型/符号;G4 对永久封停也显示"本轮停用";未使用 import;`_threshold_for` 每次读盘;`user_message(Check(True))` 回退。

**fix 轮 1 复审**(判定:**两条全部 ADDRESSED,无新增 Critical/Important**)。复审者独立验证:模拟旧名单确认新测试 6/6 有约束力、逐槽枚举确认 spec §5.3 的 **13 个封停槽全覆盖**(其中 6 个原已被 §5.2 列级条目捕获)、逐 `mapping_rules` 关键词扫描确认**无子串误伤**、直接调用守卫验证 all-None/混合/全出分三种输入。

**⚠️ Ruling 19 复盘(控制器错误,已纠正)**:我让实现者给 `report_generator` 加崩溃防护,结果**把"没有报告"换成了"有报告但在撒谎"** —— 加防护后零证据会话走通到第 4 节,打印「候选人在**证据不足**维度表现最为突出,显示出良好的**科研天赋**……具备从事科研工作的**基本心理素质**」。**这是零证据下的才能断言,正是本轮要消灭的伪造。** 教训:过渡性"让链路跑通"的改动,必须同时检查跑通之后**输出的是什么**。

- **Ruling 20:** 加"全无证据短接"。依据:经 Task 3 + 槽位级封停后,**真实会话 5 维全部 0/4**,"无任何维度出分"是当前**唯一可达**情形,故短接覆盖今天所有可能会话。 — 若判断错,代价 = 重写函数时多删几行。
- **Ruling 21:** 分数卡片不渲染 `None`(现在会印「综合科研潜力评分为 **None** 分」)。
- **Ruling 22:** 补一条测试把 7 个槽位级封停键**逐个**钉住(原测试只测真实键,`fluency_score` 无活键故未被钉,且槽位关键词改名会静默退化)。
- **判给 Task 6 的四条**(已写进 Task 6 计划并标行号,均在该任务要整段重写的函数内、短接后今天不可达):①`:165,176,194,215` 四处 `['percentile']` 默认 50 → **恒定 50 的杜撰常模,是 Task 3 引入的新形态**;②`:137` 键名不匹配致「未检测到显著的焦虑特征」无条件打印;③`:129,174` 默认值驱动的情绪控制力断言;④`research_mapper.py` 残留未使用 import。

**处置**:计划新增 **Task 3b** 一节(短接 + 卡片 + 两条测试),Task 6 计划补四条行号;fix 轮 2 已发。预期 33 项。

**fix 轮 2 结果**(`d0e7240`,未 amend):33 项全过(原 31 + 2)。

- **RED 证据(两条新测试均有约束力)**:短接测试把 `6a80a10` 的 `report_generator.py` 载入为**临时探针模块**跑零证据输入,输出与预期**逐字一致**(「候选人在 证据不足 维度表现最为突出,显示出良好的科研天赋……」),4/4 断言必失败;钉键测试对 `36f1f87` 旧名单逐键调用,7/7 由"无"变"有"(含无活键的 `fluency_score`)。
- **零证据端到端**:deep=541 字符(诚实空报告 + 缺口清单);HTML=6246 字符,`含 "None": False`、分数卡片为 `—`;四条才能断言全 False。
- **有证据路径未误伤(关键)**:构造仍未被封停的槽(`logic_keyword_density`,n_rows=100/std=0.01)→ 总分 100.0、`logical_thinking` 1/4 conf=低、deep=2248 字符走原段落,第 4 节正常。**短接只在"无任何维度出分"时生效。**
- 实现者的一处判断说明:`:171` 的 `{result['total_score']}` 未动,理由是 `total_score is None` ⟺ 无维度出分 ⟺ 已被顶部短接拦下,故短接 + 分数卡片一处即覆盖全部 None 路径。**已要求复审者独立核这条蕴含关系。**

**fix 轮 2 复审**:**三条全部 ADDRESSED,无新增 Critical/Important,无遗留 finding。** 独立核了两件关键事:

- 短接条件与 `research_mapper.py:246` 的 `valid` **逐字符相同**(不是 `total_level`/`evidence_gaps`/特征计数之类的代理);被投诉的全部段落都在 return 之后
- 实现者"`:171` 不可达 None"的蕴含关系**成立**:`total_score is None` ⟺ `valid` 空 ⟺ 短接先响;且验证无反向路径 —— 五个 `dimension_weights` 键齐全且和为 1.0 → `den > 0`,`score` 因 G1 拒 NaN + `min/max` 钳位必为有限值 → `round(num/den,2)` 必为数值。全树仅 3 处渲染 `total_score`,第三处(`research_mapper.py:356`)由 `if not valid_dims` 早退保护

**Task 3: complete (commits `e249450..d0e7240`, review clean)**

- **Ruling 20 复盘已闭环**:我造成的"把崩溃换成撒谎"的回归已消除。教训写入账本:**过渡性"让链路跑通"的改动,必须同时检查跑通之后输出的是什么。**
- 本轮 deferred minor(交最终审查):两条新测试**未钉住 `_score_display`**,日后若回退 `:278` 会静默;短接只覆盖"全无出分",**当任一维度出分时其余四维仍走默认值段落**(该残余面已判给 Task 6,见下)。

**累计 deferred minor(交最终审查分诊,均不进修复循环)**

| 来源 | 条目 |
|---|---|
| T2 | `_default_n_valid` 只查存在不查类型/符号(`-5` 可通过并让 G3 对未登记指标永不失败);`_threshold_for` 每次调用重读 JSON;未使用 import(`pytest`/`Check`);`user_message(Check(True))` 回退返回"证据不足"(当前不可达) |
| T2 | G4 对**永久**封停项也显示"该指标本轮停用",暗示会回归 |
| T3 | 关卡顺序使"既恒定又封停"的槽由 G2 先响,文案为「本次会话内无变化」而非「该指标本轮停用」(更准确但需重排关卡;**已向使用者明示,是产品决定**) |
| T3 | 新测试未钉 `_score_display` |
| T3 | 封停名单未含面部 CSV 的 `confidence` 列 —— 实测该列不会成为特征键(272 个真实键中含 confidence 者 0 个),缺口无害 |

**判给 Task 6 的条目(已写入 Task 6 计划并标行号)**:①`:165,176,194,215` 四处 `['percentile']` 默认 50 → 恒定 50 的杜撰常模;②`:137` 键名不匹配致「未检测到显著的焦虑特征」无条件打印;③`:129,174` 默认值驱动的情绪控制力断言;④`research_mapper.py` 残留未使用 import;⑤短接未覆盖的"部分出分"残余面。

## 门 2(使用者亲验,2026-09-22)— 通过

使用者跑了 `pytest -q`(33 passed)与两条验收脚本,结果与预期一致:总分 `None`、5 维全 `score=None`/`conf=无`、缺口清单正常。

**使用者追问「为啥验证结果变成无或者证据不足了」—— 控制器实证作答(这是本轮最有说服力的一次演示):**

- 用 `git show e249450:report_frontend/research_mapper.py` 载入**改动前**代码,在**同一场会话**上并排对照:

| | 改动前 | 改动后 |
|---|---|---|
| 总分 | **55.45** | **None** |
| 逻辑思维 / 抗压 / 沟通 / 自信 | 46.43 / 66.18 / 56.59 / 61.68,**全部 conf=高** | 全证据不足 |
| 认知负荷 | 44.83 · 中 | 证据不足 |
| 结语 | 「**核心优势在于抗压与情绪稳定性**」 | 「未采集到足以支撑评估的有效证据」 |

- **根因**:`get_fused_latest_data()` 挑中的"最新"文件是 **face 4 行 / gesture 4 行 / voice 8 段** 的存根会话 —— **4 帧数据**。改动前它据此给出 55.45 分与 4 个"高置信度"。
- 21 条缺口按门分布:G1 ×7、G2 ×5、G3 ×8、G4 ×1,全部拦对。

**⛔ 新观察(不属本轮,建议归 M1)**:`get_fused_latest_data()` 会挑到 4 帧存根文件。若这在真实流程中常见,是 `data_loader` 的选档问题 —— 与 session_id 链路同属 M1 范围。**已向使用者明示,待其定夺是否记录。**

**Task 4(删死分支与恢复 question_index)** — 已派发,等待回报
- BASE(dispatch 前):`d0e7240`
- brief:`task-4-brief.md`(79 行)
- model:haiku(小机械改动,计划含完整代码)
- **⚠️ brief 行号已漂**:Task 3 往 `feature_engine.py` 加过代码,计划引用的 `:117`/`:289-293` 现为 **`:128`/`:300`**。已在派发词中明确要求"按代码文本定位,不要按行号",并告知冲突时以代码为准。

**Task 4: complete (commits `d0e7240..b5f2bba`, review clean)**

审查结果:Spec ✅ 合规 / Task quality **Approved**;0 Critical、0 Important、4 Minor。审查者另做两项**跨 diff 的风险核查**:①扫 `data/` 下**每个 CSV 的表头**,确认无任何列名同时含 `speech_ratio` 与 `mean` → 删除死分支不可能改变任何生产者输出;②扫 `report_frontend/` 与 `app.py` 的 `fluency_score`/`fluency_proxy` 消费者,剩余命中均为声明性,且那些键从无数据集产出过 → 无新增 KeyError 风险。

控制器复验(门 3 命令已跑通):`question_index` 现产出 6 个键、`fluency` 残留 0 个、`voice_research` 特征数 65 → 71(+6 吻合)。

**本轮 deferred minor**:①`feature_engine.py:305` 的替代注释写作「# 4. 停顿映射(续)」,但被删的是流畅度代理分支,标签名不副实(纯文案);②测试第二条里 `speech_ratio` 那轮迭代恒为 no-op(brief 要求保留,已在 docstring 说明);③规则 B 会放行 `*question*index*`(如假想的 `questionnaire_index`),今日安全;④**`question_id` 与 `is_valid` 仍被规则 A 的 `'id'` 子串静默丢弃** —— 与本任务刚修的是**同一类 bug**,归入后续协变量审计(实现者的 concern 3 亦提此)。 — 若判断错,代价 = 将来要用这两个协变量时需再改一次。

## 门 3(使用者亲验)— 通过

`pytest -q` → 35 passed;协变量验证脚本输出与控制器一致(`question_index` 6 个键、`fluency` 残留 0、`voice_research` 65→71)。

**使用者追问「gesture 为什么变成 182 维」—— 控制器实测作答(供 M3 参考)**:

```
gesture CSV 源列 54 → 跳过非数值/含 id·timestamp 的约 22 列 → 数值基名 32
每个基名无差别派生 6 个统计量(mean/std/min/max/sum/trend)
→ 30 × 6 + 2 = 182
```

两个成因:
1. **三个模态过滤规则不统一** —— face 有 `target_keywords` 白名单(故仅 28 键)、voice 按列名映射,**只有 gesture 是"全量数值列 × 6 统计量"**,无任何过滤。此不对称是历史遗留。
2. 6 个统计量含 `_trend`,而设计文档 §1.2 实测(面部)该类二阶派生列与视频时长 |r| 达 0.77–0.84。⚠️ **该数值是面部列的实测,gesture 的 `_trend` 未单独测过**,结构上同类但不可直接套用。

归 M3:设计文档 §4.2 给 gesture 的目标是 **≈13 列**。

**Task 5(删杜撰常模与百分位)** — 已派发,等待回报
- BASE(dispatch 前):`b5f2bba`
- brief:`task-5-brief.md`(78 行)
- model:sonnet
- **派发时澄清的关键事实**:brief 的 Step 3(research_mapper 百分位)在 Task 3 已按 Ruling 1 完成 —— 控制器已 grep 验证只剩一行说明注释,**故 Step 3 是确认步骤而非删除步骤**,明确要求实现者不要去"找代码删"。真正的工作是 visualizer 的 `[60]*5` 与 `'常模基准'` 图例。
- 派发时点名:`theta` 与 `r` 长度必须仍相等(移除 baselines 后若留下 `categories += [categories[0]]` 而无对应 `scores += [...]` 会破坏 plotly 几何);以及**测试必须有约束力**(Task 4 刚出过"测试改动前后都通过"的问题)。

**Task 5(删杜撰常模与百分位)** — 实现完成(`29ab81b`),待审查

- 37 项全过(35 + 2);brief 预测 36 是**控制器算错**(brief 放了两个测试函数)
- **实现者的 TDD 纪律值得记录**:第一次 RED 因方法不存在而以 `AttributeError` 失败 —— 它判定**这种失败没有约束力**,于是**先做保持行为的重构**(抽出 `_build_radar_figure` 但保留常模线),重跑拿到真正的行为性 RED(`assert '常模基准' not in ['候选人得分','常模基准']`),**再**删线得 GREEN。这正是"测试必须有约束力"的正确做法,且是它自发识别出第一次 RED 不算数的。
- 诚实披露:第二条新测试 `test_no_fabricated_percentile` **在本次改动前就通过**(百分位是 Task 3 删的),标为回归守卫而非本任务 RED 证据。
- Step 3 确认为确认步骤:grep 5 处命中全为注释,`research_mapper.py` **未修改** ✓
- 保留 `categories/scores += [x[0]]`(候选分多边形闭合,与 baselines 无关);运行期验证 `len(r)==len(theta)==6`(空输入与有输入两种),plotly 几何完好。
- `generate_all_charts` 端到端在临时目录跑通(雷达图 + 5 张证据图),临时目录已清。
- **deferred minor**:`self.colors['baseline']`(#6c757d)现为死调色板键,实现者按 brief 范围未删,留待审查判断。

**Task 5 审查结果**:Spec ✅ 合规 / Task quality **Approved**;0 Critical、1 Important、4 Minor。

审查者确认的关键点:删除精确(正确识别 `categories += [categories[0]]`/`scores += [scores[0]]` 属于**候选分多边形**而非 baselines —— 那是本任务的陷阱);`theta`/`r` 在所有路径长度对齐(含空 dimensions 与缺键分支);公开方法行为与唯一调用点保持;10 条既有测试纯追加未被扰动;实现者的 RED 纪律被认定"真实而非走过场"。

- **Ruling 24(Important,plan-mandated):** `test_no_fabricated_percentile` **是空断言**。审查者实测其输入让 5 个维度全部 `score=None, evidence_chain=[]` → 嵌套循环**零断言** → 对任何实现都通过,**包括把百分位加回来的实现**。实现者披露了"改动前就通过"但**未披露"它守不住任何东西"**。修法:换一个能过证据门的 fixture(`logic_keyword_density` + `_std` + `_n_rows`,控制器已实测输出 1 条证据链、`logical_thinking` 100 分),并**先断言 `total_evidence > 0`**。 — 若判断错,代价 = 该测试继续形同虚设。
- Minor 顺带修:雷达测试只做否定断言(一张零轨迹的图也能过)→ 加正向断言 `names == ["候选人得分"]`;`_build_radar_figure` 补 `-> go.Figure`。
- 判为**不修**:`self.colors['baseline']` 死键(审查者认定留着不算缺陷,记入 deferred);报告贴的 grep 证据不可复现(结论经独立验证正确,提示措辞即可)。
- **回答了审查者的控制器疑问**:`report_generator.py` 的四处 `['percentile']` 默认 50 由 **Task 6** 负责 —— 已连同行号写进 Task 6 的计划与 brief,**无覆盖缺口**。

### ⚠️ 本计划反复出现的失效模式(第 4 次):测试通过 ≠ 测试有约束力

| # | 任务 | 形态 | 谁抓到 |
|---|---|---|---|
| 1 | T2 | `test_at_threshold_passes` 用了被封停的键,断言与实现矛盾 | 实现者(派发前) |
| 2 | T3 | D1 `__n_rows` 下划线:5 条测试全绿,却掩藏了 **G3 全盘失效** | 实现者 |
| 3 | T4 | brief 的测试第二条**改动前后都通过**,零约束力 | 实现者 |
| 4 | T5 | `test_no_fabricated_percentile` 的循环**一次断言都没执行** | **审查者** |

前三次靠实现者自律,第四次靠审查。**这是"每任务必审"挣到的直接证据。** 后续任务的派发词已固定加入一条要求:实现者须自证"测试在对应修复被回退时会失败"。

**Task 5 fix 轮 1 结果**(`c169789`,未 amend `29ab81b`):37 项全过。

**实现者的"有约束力"证明(本计划迄今最强的一次)**:在 `evidence_item` 临时注入 `"percentile": 88` 探针下 ——
1. **修好的测试变红**:`AssertionError: ... or 88 is None`
2. **同一探针下,旧的测试体通过,且"执行断言数 = 0"** —— **直接测出旧断言恒真**

这不是声称而是测量。探针已 `git checkout --` 还原,`research_mapper.py` 不在提交内。

- 三条 finding 全部处置:vacuous 测试换用门内 fixture + `total > 0` 前置断言;雷达测试补正向断言 `names == ["候选人得分"]`;`_build_radar_figure` 补 `-> go.Figure`。
- 按指示**未删** `colors['baseline']`,记入 deferred。
- **主动纠正了第一轮报告的证据瑕疵**:承认贴出的"5 处命中"不是该命令的输出(实际 1 处),并三种 grep 实现交叉核验;结论不受影响。
- 确认未触碰 `report_generator.py`(归 Task 6)。

**Task 5: complete (commits `b5f2bba..c169789`, review clean)**

复审判定:三条 finding 全部 ADDRESSED,无新增 Critical/Important。复审者在**仓库外**(`/tmp` 拷贝)独立注入 `"percentile": 88` 探针复现了两半证据:修好的测试体 `AssertionError`,旧 fixture + 旧测试体**执行 0 条断言且通过** —— 确认不是"两边都通过"的测试。

### 控制器系统扫描:空断言类缺陷的穷举(第 5、6 例)

复审者在 Out-of-Scope 里报出**第 5、第 6 例**(`test_no_fake_resume_fallback`、`test_quarantined_columns_are_rejected`)。控制器**没有逐个等发现,而是做了穷举**:用 `sys.settrace` 记录每个测试实际执行的行,与 AST 里的 `assert` 行求差。

**结果:`tests/test_report_layer.py` 12 个测试中恰好 2 个的断言行从未执行**,与复审者所指完全一致;其余 10 个全部断言都执行。

| 测试 | 断言行 | 从未执行 |
|---|---|---|
| `test_no_fake_resume_fallback` | 3 | **3** |
| `test_quarantined_columns_are_rejected` | 2 | **2** |

根因相同:fixture 让 `evidence_chain` 为空 → 遍历它的循环零次执行。**而这两条正守着 Task 3 的核心主张** —— "假简历兜底回归""封停列重新进链"都不会被任何测试拦住。

- **Ruling 25:** 这两条**不延 Task 5 的修复循环**(复审已判 Task 5 clean),而是**折进 Task 6 的 Step 0** —— 同一测试文件,Task 6 本就要改。替换方案已由控制器**实测验证**:喂一个只让 1 槽过门的输入 → `matched=1/4`、缺口 3、`focus_score` 不在链中。**`matched == 1/4` 就是约束力**:若 BASELINE_FILL 回归,缺失 3 槽被填空 → 变 `4/4` → 测试变红。同时要求改完后**重跑同一扫描脚本,断言"存在空断言的测试数 = 0"**。 — 若判断错,代价 = 这两条继续形同虚设(但它们现在至少被记录在案)。
- **deferred(建议交最终审查考虑)**:把上述 `sys.settrace` 扫描**固化成常驻测试**,让"空断言"这类缺陷以后自动现形,而不是靠人逐个发现。

## 使用者提问「数据集太少要不要现在录」—— 控制器实测作答

盘查结果:**55 个日志文件,但真正的会话只有 2 场**(`face_au_log_20260315_103759.csv` 14208 行、`gesture_emotion_log_20260314_090227.csv` 9050 行),其余最大 230 行、约 40 个是 1 行空壳。**`data/input/` 为空,全仓无 `*.mp4`/`*.wav` —— 一份原始媒体都没留。**

**关键论证(用实测数据,不用判断)**:

| 输入 | 结果 |
|---|---|
| 4 帧空壳会话 | 21/21 槽被拦 |
| **14208 行完整会话** | **21/21 槽被拦** |

**数据从 4 行涨到 14208 行,结果完全相同** → 拦截来自**封停名单与恒定**,不来自样本量 → **录更多数据不会改变 ① 的输出**。

**产出**:`docs/superpowers/specs/2026-09-22-jingxin-recording-requirements.md`(197 行)。要点:
- 分三阶段(阶段 A 验证 M2/M3 需 3–5 场 / 阶段 B 标定 / 阶段 C 雇主侧 ≥200–300 每群),**本文件只规定阶段 A**
- **三个硬前提**:①M1 的 `session_id` 链路先修好(否则数据对不上号,实测曾拼出"5 月的脸 + 2 月的语音");②**必须留原始媒体**(现有缺陷会永久烘进数据,有原始媒体才能 M3 后重抽);③**M3 前不采自评量表**(科技伦理审查是强制项)
- 必采元数据含审查点名的遗漏项:提问时刻、题目 ID/难度、**候选人生理属性**(`pitch_mean` ICC .723 = 解剖常量,可能成为性别探测器)、设备取景参数、**面试官结构化评分**
- 路径规定:**原始媒体一律仓库外**(`~/shared/jingxin_recordings/{session_id}/` ↔ `D:\Shared\...`),特征留仓库 `data/logs/`,标注留 `sessions/{session_id}/labels.json`
- 末尾指出两条**不需要新录数据**的更便宜路径(用已有数据集做 ICC 方差分解;MIT 的 pre/post 结构)

**Task 6(重写叙事层)** — 实现完成(`ea65285` Step 0 + `e3ed5ae` 重写),报 NEEDS_CONTEXT,fix 轮 1 进行中

实现成果(审查前):Step 0 扫描 **2 → 0**、套件 **39 passed**、`_generate_deep_text_analysis` 整段重写(**−141 行**)、复审判给它的四条经 grep 实测**真删**、Step 6 端到端四条核对全过(证据不足出现 6 次、无 `None`、三句硬编码消失)。

**唯一阻塞(实现者上报,控制器核实后确认成立):** brief 的 `test_deep_analysis_has_no_banned_words` 在 Task 6 内**结构性不可满足**。报告里的禁止词有**两个来源**,而叙事层只是其中之一:

| 来源 | 命中 | 归属 |
|---|---|---|
| `research_mapper` 的 `display_name`「抗压与情绪稳定性」 | 抗压、情绪稳定 | Task 7 Step 3 |
| `research_mapper` 的 `human_name`「面部紧张度」 | 紧张 | ⚠️ **Task 7 原改名表未覆盖** |

控制器**独立全量扫描确认**:`research_mapper` 只有这两处命中,其余标签干净。

**关键点(实现者发现,比表面更重要)**:Task 7 的改名表只覆盖 `display_name`,**不含 `human_name`** —— 所以**即使 Task 7 按原计划做完,该断言仍会红**。实现者拒绝替使用者拍名、也拒绝加命名映射层(会与 brief 给定代码冲突且属"加功能"),改为移出断言 + 原位留中文注释并上报。**正确处置。**

- **Ruling 26:** 采纳其方案,但要求**把收窄版提交进去而非移出** —— 不收窄会卡住,不提交则叙事层自身输出无人看守,两者都要避免。收窄方式是**精确字符串替换**剔除两个已知标签(叙事层若在别处写出禁止词仍会被抓到,约束力保留)。同时:
  - **Task 7 Step 3** 改名表**新增 `human_name` 一行**:「面部紧张度」→「**眉间收缩与唇部压缩**」(控制器默认,使用者可否决;spec §5.6 建议的"面部紧张相关动作单元活动率"仍含"紧张"不可用),并要求改完**全量扫一次** `name`/`description`/`human_name`。
  - **Task 7 新增 Step 3a**:改名后**删掉剔除逻辑、恢复全量扫描**,验收为"恢复全量后仍 PASS"。
  - 派发词中点名要求实现者自证"收窄版断言**仍有约束力**"(临时在叙事层塞一个禁止词验证会红)—— **本计划已栽过六次空断言,不得有第七次**。
  - 若判断错,代价 = 收窄版遗漏非标签来源的禁止词(但 Task 7 Step 3a 会恢复全量兜住)。

**Task 6 fix 轮结果**(`38360a4`,未 amend `e3ed5ae`;`report_generator.py` 零改动):**40 passed**(39 是上一轮移出断言时的数字,加回后 +1);`-W error` 下同样 40 passed 无告警。

- **断言扫描:0 个测试有未执行断言(共 15 个)** —— Step 0 修复保持有效
- **收窄版约束力已自证**:实现者临时在**叙事层自己写的字符串**里注入「紧张」,且**刻意选在剔除逻辑够不着的位置**(`_render_dimension_block` 的"证据不足"段落,不是那两个被剔除的标签)→ `pytest -k banned` → `AssertionError: 叙事层出现禁止词:紧张`。**RED 成立**,注入已 `git checkout` 还原(未提交、工作区干净)。
- 实现者两处小更正:①brief 的 docstring 误写"Task 7 Step 3b",恢复全量扫描实为 **Step 3a**(3b 是头部字符串清理),它按消息口径写成 3a 以免指错后续实现者;②brief Step 1 的 `import re` 三个测试都没用到,**未加死导入**。
- **遗留**:`_get_percentile_badge` 现为无调用方的死代码(在 brief 替换范围之外,未擅删,已标记)→ 交 Task 7 或最终审查。

**审查状态**:Task 6 因中途 NEEDS_CONTEXT 而**从未经过任务审查** —— 现按**全量任务审查**派发(`c169789..38360a4`,3 commits,22KB),非限定复审。审查者被点名重点核两件:那两条修复后的测试是否仍有约束力、以及 15 个测试的空断言扫描是否属实。

**Task 6 任务审查结果**:Spec ✅ 合规 / Task quality **Needs fixes**;0 Critical、**2 Important(均 plan-mandated)**、7 Minor。

审查者的独立验证(值得记录):①自己重写 `sys.settrace` 扫描脚本,得到 `未执行断言的测试数 = 0(共 15 个)`,与实现者声称一致;②**在仓库外执行了那条无测试覆盖的"出分路径"**(混合 fixture → `total_score=100.0`),确认表格渲染正常、`evidence_chain` 提供四个键、**无潜在 `KeyError`/`max` 崩溃**;③确认四条遗留项是**真删**(grep 文件文本而非绕过);④非破坏性复现了收窄断言的约束力(`post-injection post-strip hits: ['紧张']`)。

⚠️ 审查者还指出一处**不能被记录为已完成**的:"报告已合规" —— 生成的 HTML 里仍有 抗压×2 / 情绪稳定×2 / 紧张×2,全部来自那两个 mapper 标签,即 Task 7 的范围。**Step 6 的核对清单没含这些词,所以别把"报告合规"当成 ① 的成果。**

- **Ruling 27(Important#1):** 成立 —— `test_no_hardcoded_gaze_claim`、`test_deep_analysis_has_no_banned_words`、`test_deep_analysis_handles_none_scores` 三条**用了零证据 fixture**,而禁止词与硬编码句住在**出分路径**上,故在 base 与 head 上都通过。**本计划同类失效的第 7、8 次。** 统一修法:改用混合 fixture(`logic_keyword_density` + `_std` + `_n_rows`)。控制器已实测:该 fixture 下 `logical_thinking` **出分**(真的走进被删段落),当前实现下硬编码句与禁止词命中均为空。 — 若判断错,代价 = 三条测试继续恒真。
- **Ruling 28(Important#2):** 成立 —— `symmetry_score` 是 `stress_resilience` 的槽,而测试只查 `logical_thinking`;`grep -rn symmetry tests/` 显示该行是唯一出现处,即**它哪儿都没被钉住**。⚠️ **关键判断:"补 `stress_resilience` 的链断言"是无效修法** —— 该维四槽全部封停、链必为空,断言仍恒真。**正确修法是断它"以缺口形式出现"**(若删掉封停条目并喂入该列,它会进链、缺口便不再有它 → 测试变红)。控制器实测:链=0、缺口=4、缺口含「面部对称性」。 — 若判断错,代价 = 该断言继续恒真。
- **顺带修两条 Minor**:(e) 缺口**重复渲染**(实测 19 条缺口渲染两遍;出分路径删顶层段,短接路径保留);(f) 删死代码 `_get_percentile_badge`(本任务删掉了它唯一调用方,spec §5.5 要求不留百分位机器)。
- **判 deferred 不修**:未使用参数 `dim_key`/`features`;`test_no_fake_resume_fallback` 缺链守卫(扫描确认其循环确实执行,且首条断言已隐含拒绝空链)。

**处置**:计划新增 **Step 8**(六处改动 + 四项验收),brief 重新生成(312 行),fix 轮 1 已发。

**Task 4 实现记录(承接上文 DONE_WITH_CONCERNS 段)**

实现者报 **DONE_WITH_CONCERNS**,35 项全过(33 + 2)。三条 concern:

- **Ruling 23(控制器 brief 缺陷,实现者已修正,予以采纳):** brief 的第二条测试 `test_dead_fluency_branch_removed` **是空断言** —— 它喂的 df 列名是 `speech_ratio`,而死分支只有在列名**同时**含 `speech_ratio` 与 `mean` 时才触发,故该测试**改动前后都通过**,brief 写的"Expected: FAIL"从不成立。实现者加进唯一能触发分支的形状 `speech_ratio_mean`(同一条测试内),实测 `'speech_ratio' -> []` vs `'speech_ratio_mean' -> ['research_fluency_score_mean','research_fluency_proxy']`。**这是正确的偏离** —— 我的测试错了,它修对了。计划已按此更正。 — 若判断错,代价 = 该测试保留零约束力,死分支回归无人拦。
- concern 2:`_SKIP_COLS` 提到模块级(brief 是内联)—— 语义等价,避免每列重建元组。**接受**。
- concern 3(未处置的观察):`question_id` 仍被既有 `id` 规则丢弃。真实日志表头无此列、spec 也未要求,故未扩白名单。**记为同类的静默丢弃风险**,供后续协变量审计。 — 若判断错,代价 = 将来真要 `question_id` 时需再改一次。

**Task 3 原派发记录**
- BASE(dispatch 前):`e249450`
- brief:`task-3-brief.md`(258 行;由 Task 2 复审者确认已含 `user_message` 接线)
- report:`task-3-report.md`
- model:sonnet(最大的集成型改动:改 410 行文件、删四类伪造、接证据门)
- 派发时补充:Task 2 的三条交接项 + git 身份 + 分支已建 + `pytest.ini` 已存在 + CRLF 警告属预期
- 派发时点名的自检项:`demo_text_data`/`self.baselines` 的**每一处使用**是否都删净(遗留一处即 `AttributeError`)、三条硬编码代理、`BASELINE_FILL` 是否不可达、`matched`/`dim_gaps` 作用域、comm 权重和是否 = 1.0

**Task 3 交接项(由 Task 2 实现者写入其报告 9.8 节,已带入派发)**

1. `_std` **必传** —— G2 在 `std=None` 时是 fail-open(Task 2 有意为之,已有测试钉住)。Task 3 若不传 `_std`,常量列会静默通过。
2. 报告层**只准用 `user_message(chk)`**,不得渲染 `chk.reason`(后者含维护者文案)。
3. pause 族 G3 阈值现为登记的 **2**(此前被 `au` 遮蔽为 30)—— 行为变化,Ruling 14 修 bug 的预期结果。
- 第 2 轮回报 DONE:4 个显式路径、337 插入、无 amend/rebase
- 5 处裁决全部落地;独立复验探针:`permanent` 计数=8 且 `blink_rate` 非永久、`pause_duration_mean` 30→2、`au12_smile_mean` 仍 30(未误伤)、默认 10 仍在、`confidence_from` 3600 组返"高" 0 次
- 三个文件与重新生成的 brief 代码块 `diff` 逐字节一致;全 LF
- 旧实现下跑出 RED 恰 2 条(permanent=False 与 pause 阈值 30),证实新测试有约束力
- BASE(dispatch 前):`8d156f4`
- brief:`.superpowers/sdd/2026-09-22-report-layer-evidence-gate/task-2-brief.md`(367 行)
- report:`.superpowers/sdd/2026-09-22-report-layer-evidence-gate/task-2-report.md`
- model:sonnet(不用最便宜一档的理由:该模块被 Task 3–7 全部依赖,且含 TDD 的 RED/GREEN 纪律与 20 条断言;一次失败的派发比省下的模型差价更贵)
- 派发时补充的四条 brief 无法知道的事:**Ruling 9** 的 `pytest.ini` 折入;git 无身份需 `-c` 参数;分支已建勿再建;`.gitattributes` 已生效(新文件用 LF)
- 派发时点名的自检项:`is_quarantined` 最长匹配、`gate` 先失败先返回、`confidence_from` 永不返"高"

**Task 2 回报 NEEDS_CONTEXT(零提交)—— 实现者发现 brief 自相矛盾 3 处,拒绝猜测,正确升级。**
它点名的三项自检**全部实测通过**(最长匹配、关卡顺序、`confidence_from` 穷举 3600 组返"高"次数 = 0)。处置:

- **Ruling 12:** 矛盾 1、2 是**我的测试写错**,封停名单不动(spec §5.2 明写 blink_rate 封停到 M2)。`test_at_threshold_passes` 的 `blink_rate` → `pitch_median`(阈值 10);`test_clean_column_passes` 的 `interview_pause_duration_mean` → `interview_duration_mean`。实现者另外指出"阈值 20 只由 `blink` 产出,而所有含 blink 的键都被封停,该边界用任何未被封停的键都测不到" —— 正确,已写进计划注释。 — 若判断错,代价 = 封停名单被错误放松,坏列重新进入打分。
- **Ruling 13:** 矛盾 3 是**我的实现真漏了** —— 8 条 `unblock="永不"` 的条目无一设 `permanent=True`。**显式设 `permanent=True`,不用 `__post_init__` 从字符串反推**(实现者建议了后者)。理由:`unblock` 是显示层文本,用显示字符串驱动布尔逻辑是隐式耦合,改措辞会静默改行为。 — 若判断错,代价 = 8 行冗余字段;用 `__post_init__` 的代价则是行为随文本漂移。
- **Ruling 14:** 接受实现者的自检发现 (A):阈值表里 `"au"` 早于 `"pause"` **且是其子串**(p-**au**-se),导致 `pause_*` 键静默拿到 30 而非登记的 2;且同模块内 `is_quarantined` 用最长匹配、`_threshold_for` 用首个匹配 —— **两套规则是缺陷**。裁决 `_threshold_for` 改最长匹配,并加 `test_longest_match_wins` 钉住。 — 若判断错,代价 = pause 类指标阈值从 30 放宽到 2,门槛放松(但方向是修 bug)。
- **Ruling 15:** 实现者的自检发现 (B):`Check` 字段序在 brief 接口段(`ok, reason, gate_name`)与实现(`ok, gate_name, reason`)不一致。**实现赢**(下游全用属性访问,无运行时风险),改计划接口段文字。 — 若判断错,代价 = 文档与代码不符,已注明"勿用位置参数"。

**处置动作**:计划修正 5 处后**重新生成 brief**(392 行),用 SendMessage 恢复原实现者(非新派发,其上下文完整)。

**deferred minor(来自 Task 2 自检)**:`evidence_thresholds.json` 的 `"pause": 2` 在 Ruling 14 后生效(此前是死条目)。
- BASE(dispatch 前):`07ed73d`
- brief:`.superpowers/sdd/2026-09-22-report-layer-evidence-gate/task-1-brief.md`
- report:`.superpowers/sdd/2026-09-22-report-layer-evidence-gate/task-1-report.md`
- model:haiku(计划里代码完整,属抄写+验证)
- 派发时补充的三条 brief 无法知道的事:①分支已建,跳过 brief 的 Step 1;②pytest 9.1.1 已装;③解释器与"只加精确路径"两条全局约束

## Task 6 fix 轮 1 —— 会话中断后的恢复(2026-09-24,新会话)

**上一会话在 fix 轮 1(Step 8,针对 Ruling 27/28)派发后结束,实现者未提交、未回报。** 控制器实测的现场状态:

- HEAD = `38360a4`(与审查所见一致,即 fix 轮的 BASE)
- **工作区有未提交改动**:`report_frontend/report_generator.py`、`tests/test_report_layer.py`
- 内容与计划 Step 8 六项 (a)–(f) **逐条对应**:(a)(b)(c) 三条叙事层测试换 `_MIXED` 混合 fixture 并各加"fixture 未出分"守卫;(d) `test_quarantined_columns_are_rejected` 改断 `stress_resilience` 缺口含「对称」;(e) 删顶层「证据缺口」汇总段、逐维块自带上缺口列表;(f) 删死代码 `_get_percentile_badge`
- `pytest -q` = **40 passed**
- **未完成**:四项验收中的第 2、3、4 项(注入式 RED 自证、空断言扫描重跑、生成报告确认缺口不重复)

- **Ruling 29:** 按 SDD「rounds 1–3 续话失败则派新实现者」处理 —— **派新实现者承接这份未提交改动**,不 revert 不重做,补齐验收后提交。BASE 仍为 `38360a4`。 — 若判断错,代价 = 若那份改动其实不完整,新实现者会在核对 brief 时发现并补齐(已要求逐项核对)。
- **Ruling 30(安全约束,已写进派发词):** **禁止用 `git checkout -- <file>` 或 `git stash` 还原注入实验** —— 这两处改动尚未提交,还原即永久丢失。要求实现者先 `cp` 到 `/tmp` 备份,或用精确 Edit 反向撤销。 — 若判断错,代价 = 无(纯约束,不改变行为)。

**Task 6 fix 轮 1 结果**(`ae85bfc`,父 `38360a4`,no amend;仅两个文件 staged):**40 passed**,`-W error` 下同样 40 passed,输出干净。

实现者核对后确认:六项 (a)–(f) **在我查到的未提交改动里已全部存在且正确**,它没有重做,只补齐了缺失的三项验收证据:

- **证据 2(注入式 RED,三条全部是 `AssertionError` 打在受守字符串上,不是 AttributeError/KeyError)**:注入硬编码句 → `test_report_layer.py:254 AssertionError`;注入「抗压」→ `AssertionError: 叙事层出现禁止词:抗压`;(c) 两路:恢复零证据 fixture → `fixture 未出分…`;丢掉 None 维 → `assert '证据不足' in html`。**同批注入在旧零证据 fixture 下两条都 PASS(未被发现)** —— 这正是 Ruling 27 的要点。文件用 `cp` 备份还原,还原前后 `git diff` 的 sha256 一致(`35ed7176…`)。
- **证据 3**:`/tmp/jx_sweep.py` → **15/15 测试的全部断言都执行,未执行断言的测试数 = 0**。
- **证据 4**:真实报告(短接路径)**20/20 条缺口各出现恰好一次**;把旧顶层段加回去则变成 `{2: 20}`;「面部对称性: 有效样本不足」现在 =1、改前 =2,且带维度归属。另为 (d) 补跑了一条 RED:删掉 `symmetry_score` 的封停条目 → `AssertionError: 封停列进了链`(随后还原 `evidence_gate.py`,未提交)。

- **Ruling 31(采纳实现者的两处偏离,均优于 brief 字面):**
  1. (a) 顺带把 `_MIXED` 也作为 `features` 传入(brief 说保持 `{}`)。该参数在本函数里未被使用,传入只会让测试更贴近真实调用,严格更强。**采纳。**
  2. (e) **无条件**删掉顶层缺口段,而非 brief 说的"只在出分路径下删"。**实现者实测证明 brief 的前提是错的**:控制器已独立复核 `_generate_deep_text_analysis` 的源码 —— 它对 5 个维度**无条件**渲染,那条 `total is None` 只决定是否加一句摘要,**没有任何早退**;所以按 brief 字面改,重复会原样留在真实流程实际走的那条路径上。**采纳。** — 若判断错,代价 = 若将来某条路径真的不渲染任何维度,该路径的缺口会随之消失(但实测真实报告 20/20 缺口仍在,且带维度归属)。
- **deferred minor(实现者上报,记入最终审查)**:①(e) 无测试钉住,只有脚本计数与生成报告两份证据 —— 顶层段若回归不会变红;②`result["evidence_gaps"]` 现在在报告层已无消费方;③真实数据从不走出分路径的报告渲染,那条路径只有测试覆盖。

## Task 7 派发前的控制器核查(2026-09-24)—— Ruling 32

派发前对 brief 逐条比对了当前代码,**发现 6 处 brief 无法知道或已陈旧之处**,一并裁入派发词:

1. **Step 8 的 `git add` 清单漏了 `report_frontend/report_generator.py`**,而 Step 3b 明确要改它 → 必须一并 add,否则改动游离在审查范围外。
2. **brief 里所有行号均已失效**(`:301/:302/:155/:377/:360-382/:141-164` 是 Task 6 之前的行号,Task 6 删了 141 行)→ 一律**按字符串定位**。
3. **Step 4 的连带影响 brief 未写**:`report_generator.py:195` 的「🧠 判推」框消费 `dim['narrative']`,而 Step 4 要删掉产出它的 `_generate_deep_inference`。裁决:**删掉该框**,并让 `research_mapper` 停止产出 `narrative` / `simple_narrative`(`simple_narrative` 经 grep 确认**零消费方**;templates/ 与 app.py 均不引用)。留框即 `KeyError`,而 spec §5.4 本就要求不解读不推断。
4. **`_generate_summary_narrative`(`research_mapper.py:350`)+ `report_generator.py:242`「综合总结」卡**:brief 未列,但它是报告里**仅存的解释性生成器**(「核心优势在于…建议关注…的提升」)。裁决:**一并中性化**为陈述覆盖与置信度的事实句。同批中性化报告头部与模块描述里的「科研能力/心理」字样(`:204` title、`:226` h1、`:227` 副标题、`:236`「综合科研潜力评分」、`visualizer.py:93` 雷达图标题、以及 `report_generator.py:20/31/69`、`research_mapper.py:17/96`、`__init__.py:3`、`visualizer.py:21` 的模块描述)。已 grep 确认 **tests/ 不依赖这些字符串**。`total_level`(`_get_level` 的分数档位标签)保留 —— 它是分数自己的档,不是对人的推断。
5. **`feature_engine.py:470/477` 的 print 串里有禁止词「紧张」**,而 Step 1 的 AST 扫描覆盖 `report_frontend/**/*.py` → 这两条会被它抓到,**brief 的 Step 2 预期失败清单与 Step 7 的 grep 预期都漏了它们**。裁决:一并改写成中性表述(控制台输出亦属输出字符串)。
6. **Step 7 的预期是陈旧的**:实测 `evidence_gate.py` 与 `evidence_thresholds.json` 的禁止词命中**均为 0**(其封停理由文案不含禁止词)。改完后剩下的只应是**注释**(`research_mapper.py:22/184`、`visualizer.py:85/86`)与测试里的 `BANNED` 断言 —— 后两者按 Global Constraints 豁免。

**Scope 边界(写进派发词)**:`.pyc` 卫生(下一步文档 §5 第 3 项)、`evidence_gaps` 无消费方等 deferred minor **不在 Task 7 范围内** —— 前者交最终审查,后者由最终审查 triage。

— 若 Ruling 32 判断错,代价 = Task 7 改了 brief 未列的字样(第 3/4/5 项),使用者可在**门 4 亲验报告**时逐条否决命名;反向代价(不改)则是扫描测试必然红、或报告出现 `KeyError`/口头断言。

**Task 6 复审结果(修复轮)**:四条 finding **全部 ADDRESSED,无新增 Critical/Important 破坏**。

复审者做了独立复核(**不是读报告**)而非仅核对 diff:①自行在内存里跑 `_MIXED` → `total_score=100.0`、`logical_thinking=100.0`、其余四维 `None`,渲染出的 html 含出分分支标记「综合行为观测摘要」→ 三条测试的断言确实扫在出分路径上;②**自行做了两个 falsifiability 实验** —— pop `QUARANTINE["symmetry_score"]` → 链变成 `['face_symmetry_score_mean']` 且缺口里「对称」消失,**两条断言同时变红**;pop `focus_score` → 该列进链(证明保留的那条断言也可证伪);③独立跑了 4 条受改测试(`4 passed, 11 deselected`)。它同时确认 (e) 的偏离是对的:`result["evidence_gaps"]` 是各维缺口的**纯并集**,短接路径渲染 5 个维度块、20 条缺口各一次、无顶层 `<h3>证据缺口</h3>`。

**Task 6: fix round 1/5 (4 addressed, 0 open; commits `38360a4..ae85bfc`)**
**Task 6: complete (commits `c169789..ae85bfc`, review clean)**

**Task 6 deferred minor(复审 Out-of-Scope,交最终审查 triage):**
1. `result["evidence_gaps"]`(顶层并集)在报告层已无消费方(`research_mapper.py:261` 产出,仅 `tests/test_report_layer.py:13` 断言它)。
2. **(e) 无 pytest 守卫** —— 把顶层缺口段加回去不会让任何测试变红。brief 本就只要求"生成报告核对",故复审判其非 finding,但该性质**无回归保护**。
3. 禁止词仍会经 mapper 的 `display_name` / `human_name` / 描述与 `inference_template`(`research_mapper.py:39-41`)进入完整报告 —— 归 Task 7;(b) 的剔除清单只对深度文本面完整。
4. 真实数据只走短接路径;出分路径的报告渲染仅由 `_MIXED` 级 fixture 覆盖。

## 控制器独立核验:门 4 报告产物(2026-09-24)

Task 7 实现者留下的 gate 产物,我**没有只看它的描述,而是自己打开读了正文**(剥标签后 53 行文本):

| 检查项 | 实测 |
|---|---|
| 头部措辞 | 「🔬 JingXin 面试行为观测报告」/「基于多模态行为量的结构化观测」 ✓ |
| 分数卡 | 分数位 `—`、标签「综合行为观测评分」、档位「证据不足」,全篇无 `None` ✓ |
| 综合总结 | 「本次会话 20 个指标槽中 0 个通过证据门,未产出综合分。」—— 事实句,无「优势/建议/潜力」 ✓ |
| 逐维块 | 5 维 × 4 槽 = **20 条缺口,逐条带维度归属,无重复** ✓ |
| 判推框 / 眼动图 | 0 个 / 0 个(雷达图「📊 五维行为观测」仍在) ✓ |
| 禁止词 | 0 命中 ✓ |

**控制器另做的三项自测(不在任何报告里)**:①直接调 mapper + `_MIXED` → `summary_narrative` = 「…20 个指标槽中 1 个通过证据门,综合行为观测评分 100.0(置信度上限:低)」,与深度文本层一致;②`mapping_rules` 槽位数实测 **20**(5 维 × 4);③6 张图表 iframe 的目标文件都在 `data/output/` 且非空(plotly,4.8 MB/张),图内数据为空(`x=[''] y=['']`)—— 全槽被封停时无物可画,属**诚实空图**,不是伪造。

- **deferred(spec/文档陈旧,交最终审查与文档修订)**:spec §5.3 / §7.3 与 `docs/下一步.md` 都写「**21 个**指标槽」,但代码实测:merge base `07ed73d` = 21,`36f1f87`(Task 3 首个提交)起 = **20**。差额正是 Task 3 删掉的那个**重复计数槽**(comm 维度权重和 1.4 → 1.0 的那一槽)。**报告输出的「20」与代码一致 —— 是文档陈旧,不是代码缺陷。**
- **Ruling 33(我的派发缺陷,已就地修):** 我让实现者把报告**只拷 HTML 本身**到 `~/shared/`,而报告里 6 张图是**相对路径 iframe** —— 直接打开那份拷贝会看到 5 个空图卡。控制器已把报告连同 6 个图表文件打成 `~/shared/jingxin_gate4/`(28 MB),**门 4 要打开的是 `D:\Shared\jingxin_gate4\jingxin_gate4_report.html`**。 — 若判断错,代价 = 无(多拷 28 MB);不修则使用者在门 4 看到空图,误以为图表被删坏。

**Task 7 审查结果**:Spec ✅ 合规 / Task quality **Approved**;0 Critical、**0 Important**、7 Minor。

审查者的独立复核(不是读报告):①**重跑了全文件类型的禁止词扫描**(含 `.json`/`.html`)—— 只剩注释与测试自身的 `BANNED`;②**独立核验 gate 产物**的字节数与 sha256(与实现者声称一致),并单独数了 11 个禁止词、`判推`、`眼动`、`gaze`、`科研能力`、`心理` 的出现次数(均 0);③逐个追了 **5 条具名风险** —— 被删符号是否还有调用方、`make_subplots` 导入删了是否仍被使用、被删的 CSS 类(`grid-2`/`narrative-box`)是否仍被引用、`total_score is None` 时新综合总结里那句「0 个通过证据门」会不会变成假话(读了 `final_total` 的算法,证明该路径上必为 0)、`algorithm`/`description` 是否真的无处渲染 —— **全部干净**;④**确认扫描测试不空转**:文件数守卫在 offenders 断言**之前**触发,实测扫到 9 个 `.py`。
它还点出实现者报告里一处**不准确的自我辩解**(§8.2 称三处「心理/科研」挂在类名上 —— 实际类名是 ASCII `PsychologicalFeatureEngine`,那三处是 docstring 文本),并明确要求**不得以该理由关掉该项**。

**Task 7: complete (commits `ae85bfc..7a4a543`, review clean)**

**Task 7 deferred minor(交最终审查 triage):**
1. `feature_engine.py:17` 类 docstring 仍写「心理特征提取引擎 (最终增强版·科研数据专用)」—— 全量扫描里唯一非注释的「科研/心理」自述。不违约束(不进输出、无禁止词),但可改。
2. **`templates/dashboard.html:65-66` 的按钮仍叫「📑 生成综合判推报告」/「整合所有数据,生成 HTML 深度评估文书」** —— 与它现在真正生成的东西(无判推、无评估)矛盾。`templates/` 在范围内,审查者判 Minor 并交控制器定夺。
3. 扫描只 glob `*.py`,`templates/*.html` 与 `evidence_thresholds.json` 里的字符串**无法让它变红**(今天靠人工 grep 兜着)。
4. 反空转守卫只看总量(`>= 5`,实际 9),单个根目录整体消失仍可能过关。
5. `research_mapper.py` 的覆盖率计数靠 `int()` 解析展示串 `matched_indicators`,计算耦合了展示格式。
6. 本任务删除动作留下的死状态:`visualizer.py` 的 `charts['gaze']` 无消费方;`positive_factors`/`negative_factors` 失去唯一读者。
7. `visualizer.py:142` `create_gaze_plot_from_df(self, df_face: pd.DataFrame)` 忽略入参、且可能收到 `None`,标注不准。

## 控制器:收尾第 4 步 —— spec §6 七条验证方式逐条对照(2026-09-24)

| spec §6 | 对应测试/步骤 | 状态 |
|---|---|---|
| 1 零输入回归 | `test_zero_input_produces_no_scores` | ✅ 自动化 |
| 2 真实会话回归(不出现「置信度:高」) | 单元层:`test_confidence_can_be_none_and_low` + `TestConfidence::test_high_is_unreachable_this_round`(穷举 3600 组返「高」0 次);**真实日志端到端只做过人工核对,未固化成测试** | ⚠️ **半自动** → 记为 deferred |
| 3 封停名单单测 | `TestG4NotProxy`(3)+ `test_quarantined_columns_are_rejected` + `test_slot_level_quarantine_covers_spec_5_3` + `test_all_slot_level_quarantine_keys_are_pinned` | ✅ |
| 4 兜底单测 | `test_no_fake_resume_fallback` + `test_dead_fluency_branch_removed`(两条的空断言由 Task 6 Step 0 修掉) | ✅ |
| 5 置信度可达性 | `test_confidence_can_be_none_and_low` + `TestConfidence::test_zero_is_wu` / `test_minority_is_di` | ✅ |
| 6 措辞扫描 | `test_no_banned_words_in_output_strings`(Task 7,全 AST 扫描)+ `test_deep_analysis_has_no_banned_words` | ✅ |
| 7 回归保护(采集层逐格复现) | 见 Ruling 34 的**两步**命令,控制器正在跑 | 🔄 |

现共 **41 个测试**。

- **Ruling 34(计划里的命令是错的,已修正):** 计划与 `docs/下一步.md` 写的 `reaggregate_normalized.py --stats legacy --verify-legacy`(**无参数**)**跑不起来** —— argparse 定义的是 `--verify-legacy VERIFY_LEGACY`,需要一个**目录**。脚本自身 docstring 的正确用法是两步:
  ```
  $PY reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_probe
  $PY reaggregate_normalized.py --verify-legacy /tmp/legacy_probe
  ```
  合并门必须用这两步,否则会拿一个 usage error 当成"防線通过"。 — 若判断错,代价 = 用了错误的验证路径。
- **Ruling 35(我自己的操作失误,记为流程教训):** 我第一次跑该防线时写成 `... | tail -20`,**管道把非零退出码吞掉了** —— 命令实际报 `usage error`,而 `$?` 显示 0。**凡是用退出码判定"防线是否通过"的命令,一律不得接管道取码**;要看输出就重定向到文件再读。 — 若判断错,代价 = 无(纯纪律);不记则下一次仍可能把失败读成通过。

**Task 8(修订设计文档三处前提)** — 实现完成(`61a1548`),待审查

产物:§6.3 阶段 1 改为「**不上线**,仅作离线研究结论」+ 表格上方许可警示块 + 增补 §1.1 的两条路径 (a)/(b);§4.4 理由 1 改为「跨语言可迁移性不对称(仅适用于非 f0 类特征)」并点明普通话声调例外;§6.2 增补《科技伦理审查办法(试行)》**强制前置项**。

⚠️ **该设计文档此前未被 git 跟踪**,所以这个提交把整个文件(570 行)一次性纳入版本控制 —— 不是本任务"新增"了 570 行。

- **Ruling 36(实现者上报的 brief 与增补冲突,裁定按 brief):** 增补 §1.2 的字面写法是「理由替换为**剩下的三条**」(可读作:删掉理由 1),而 brief Step 2 要求"删掉理由 1 里的**该前提**、替换为修正版,保留其余三条"。**裁定以 brief 为准** —— 理由:①修正后的理由 1 保留了仍然成立且未夸大的论据(非 f0 类声学特征的可迁移性不对称),增补的压缩写法会把它连同错误前提一起丢掉;②增补 §1.2 的实质主张是"那个**前提**对 f0 类特征是错的",而修正版正是照此写的并显式引用增补 §1.2。 — 若判断错,代价 = 增补 §1.2 的"三条"措辞与设计文档的"四条"对不上,改一句话即可;反向代价则是设计文档永久丢掉 f0 例外这条信息(它同时是 M1 的输入约束)。
- **deferred(实现者上报,未改,交最终审查 triage):** ①§6.3 标题「三级回退(**不阻塞上线**)」在阶段 1 离开上线路径后读起来偏松(正文已消歧,标题未动);②§11 待定项第 2 条只列"版本与授权",未含现在已成强制项的伦理审查;③§8 第 5 项与 §9 的 M5 行同样未含许可/伦理前置。

## 收尾第 2 步:采集层回归防线(合并门)—— **通过**(2026-09-24)

用 Ruling 34 修正后的**两步**命令(`/tmp/legacy_probe`):

| 检查 | 结果 |
|---|---|
| step1 `--stats legacy --out-dir` | exit=0(2011 个视频,126 秒) |
| 特征名 | 原 1410 / 新 1410,**集合完全一致** |
| 视频集合 | 2011 个一致 |
| 有限性 | 完全一致(无 NaN/Inf 分歧) |
| **格子差异** | **0 / 2835510**(最大绝对差 1.886e-19,最大相对差 1.886e-19) |
| targets | 最大绝对差 0.000e+00 |

✅ **`① 只改报告层`的范围约束成立** —— 采集层特征矩阵逐格未被触动。这是合并前必须过的那道门,已过。

- minor(预先存在,与本轮无关):脚本第 665 行 `rel = np.where(...)` 抛 `RuntimeWarning: invalid value encountered in divide`,被 `np.where` 兜住,不影响结果。

**Task 8 审查结果**:Spec ✅ 合规 / Task quality **Approved**;**2 Important**、4 Minor。

审查者**逐字核过三处修订本体**并判为忠实:§6.3 警告块逐字且位置正确(表格正上方、保留「详见增补 §1.1」)、表格行已去「先上线」、增补 §1.1 的两条路径**含限制子句无损**(「不得进产品、不得进销售材料」)、§4.4 理由 1 与 brief 替换文逐字且 **f0 例外未夸大**(限定词进了加粗标题、只说「直接搬运」、全文无别处再主张声学可跨语言迁移)、§6.2 是**强制**(「强制前置项/必须通过/不得省略」,无建议类措辞)。另独立复核了 LF、弯引号 0、表格列数。

**两条 Important 都不在三处修订本体上,而在"被这次修订弄陈旧"的旧文字上:**
1. `:473` —— §6.6 的**每场会话输出示例 JSON** 里 `"calibration_stage": 1`:**线上会话报告带着数据集阶段标定字段**,正是 §6.3 现在禁止的组合。它是 **M4 要照着实现的接口形状**,照抄就会在线上产出数据集系数驱动的估计。**实现者上一轮全篇扫描没扫到它。**
2. `:432` —— §6.3 标题「三级回退(**不阻塞上线**)」与正文 `:447` 及增补 §1.1 路径 (a) 矛盾。

审查者另点出:实现者上轮报告 §五.4 那句"除三处外无别处仍主张先上线"的**结论不成立** —— `:473` 与 `:145` 就是另外两处。

- **Ruling 37(把 Minor 折进本轮修复):** 审查者把 `:145`(架构图)、`:556`(§11 待定项)、`:532`(§9 M5 行)、`:520`(§8 第 5 项)判为 Minor 交最终审查。我裁定**连同 `:145` 一并折进本轮修复轮** —— 同一文件、同一类"陈旧前置条件"问题,分两轮改同一份文档不划算;并**明确要求实现者重扫全文、逐条列出所改与判为不改的理由,禁止再下"没有别处"的结论性判断**(上一轮的正是这种断言失准)。 — 若判断错,代价 = 文档 diff 比计划列的多几行(3–4 行文字)。

**Task 8 fix 轮 1/5 已发**(resume 原实现者,findings 逐条附上)。

**Task 8 fix 轮 1 结果**(`298220f`,父 `61a1548`,no amend;单文件 8 增 6 删):六处全部落地 —— §6.6 示例改为 `"n_labels_used": 128, "calibration_stage": 2` **并新增一条规则段落**(阶段只能取 2/3;`n_labels_used` 是自有标注会话数,阶段 2 起必须 > 0);§6.3 标题改为「三级回退(**阶段 1 限离线**)」;架构图加「(数据集线仅离线)」;§11.2 / §8.5 / §9 M5 三处补齐伦理审查与"不依赖数据集系数"前置。实现者另**逐字段重读了整个 §6.6 示例**(确认只有那两个字段不一致,`capabilities` 块正确地不含标定字段),并**书面撤回**上一轮"没有别处"的结论,改为逐条列出重扫结果(类 (i) 修 4/判可 15、类 (ii) 修 5/判可 3、类 (iii) 修 1/判可 2,各附理由)。

- **Ruling 38(实现者上报的许可范围问题 —— 记录为未决合规项,不在本任务修):** 它问:数据集派生的 **L1 阈值**(§5.1 规则 1/5、§5.2 的 τ(x)、`:174` 的 `au26` p1/p99、尤其 `:295`「`au12` 的 p90 取自**常模人群**,上线时固定使用」)是否落入许可里「**models trained on it**」的范围。**裁定:按保守读法,是的** —— 在数据集人群分布上**拟合出来的参数**就是"在其上训练的模型"的一部分,上线固定使用等于部署了数据集派生物;而许可的勾选项明写禁止把数据集**及其上训练的模型**用于真实招聘/筛选/心理画像。**但动作不属于 ①**:① 不发布任何东西,§6.3 也已禁止数据集系数上线;这条改变的是 **M5 的阈值设计**(阈值要么改用自采数据/可辩护的物理下限,要么明确只用于离线)。**因此:记入未决清单,由使用者做 M5 范围决策** —— 保守读法错时的代价 = M5 多一层阈值来源约束;宽松读法错时的代价 = 触犯许可(两个方向的代价不对称)。 — 若判断错,代价 = 见上,取决于使用者选择。
- **Ruling 39(实现者 concern 3 的一半是误解):** 它把 §6.6 示例里的 `"logical_thinking"` 当作"改名前的旧键"。**不是** —— Task 7 改的是 `mapping_rules` 的 `name`(展示名),**内部键名一直没动**,`logical_thinking` 至今是稳定标识符(控制器已核 §6.6 的用法:它是输出映射的**键**)。该条**不改**。但同一 concern 的另一半成立:该示例的 `basis` 列了 `response_latency_sec`,而 §11.1 写明它到 M1 才可用 —— 示例用了尚未可用的输入,记 deferred。 — 若判断错,代价 = 若将来真要改键名,示例需同步。
- **deferred(实现者上报,未改):** ①§10 缺一条"许可/伦理限制"条目,§1.3 的离线实验未标许可 —— 属完整性缺口,非本次修订弄陈旧;②§6.6 示例 `basis` 含 M1 前不可用的 `response_latency_sec`。两条已并入 `deferred-minors.md` 由最终审查 triage。

**Task 8 fix 轮复审结果**:六条 finding **全部 ADDRESSED,无新增 Critical/Important**。

复审者**检验了重扫清单而不是照信**:逐条重读 §六(i) 的"判可"条目并回到文档核对(`:174`/`:291`/`:295`/`:303`/`:307`/`:161`/`:141` 全部存在且性质判断正确);独立跑 `许可|伦理|授权|法规|规章` 得**恰好** `:428/:429/:434/:558` 四处,与清单一致;独立跑 `跨语言|声学|迁移` 得命中集合与清单一致;并确认新 `:484` 段落是**规范化约束句**、就贴在接口示例旁(M4 照 §6.6 实现时会直接读到)。它还自查了格式(LF 572、无 CRLF、尾换行、179 个 ASCII 引号、0 弯引号、表格列数不变、§6.6 的 JSON 仍可解析且字段互相自洽)。

**残余(非阻塞,归 Ruling 38 同一族):** `:462`「百分位只能来自真实常模」与 `:273`「每一列都要在标定里被证明有用」未被列入重扫清单 —— 它们没写明来源,属同一族"来源含混"的问题,与该实现者已上报的 `:295` 同类。

**Task 8: fix round 1/5 (6 addressed, 0 open; commits `61a1548..298220f`)**
**Task 8: complete (commits `7a4a543..298220f`, review clean)**

→ **全部 8 个任务完成。** 进最终全分支审查(`07ed73d..HEAD`)。

## 控制器探针:出分路径的报告文本(真实数据从不走的那条,对应 deferred #3)

用 `_MIXED`(1 槽过门)跑完整出分路径,剥标签后逐行读全文:

- 过门维度**正确渲染证据链**(逻辑关键词密度 0.05 → 归一值 1.0 → 权重 0.4),并**同时列出该维未过门的 3 个槽**;
- 其余 4 维均为「证据不足」+ 缺口清单(带维度归属);
- 分数卡:`100.0` / 档位「卓越」 / 综合总结「本次会话 20 个指标槽中 1 个通过证据门,综合行为观测评分 100.0(置信度上限:低)」;
- **无硬编码评语、无判推框、无禁止词、无 `None`**。

- **观察(deferred,交使用者做设计决策,不阻塞合并):** 1/20 覆盖率下仍产出 `total_score=100.0` 与档位「**卓越**」。这**不算伪造** —— 分数、其依据(那 1 个槽的原始值与权重)与置信度上限「低」都在同屏可见,且 **spec 对"总分是否有覆盖率门槛"没有规定**。但「卓越」在中文里读起来像对**人**的总体评价,而证据只覆盖 1/20 个槽。**spec 沉默,故不判为本轮缺陷**;建议在 M 系列补一条"覆盖率低于 X% 不出总分、降级为覆盖陈述"的规则,阈值同样须走 `evidence_thresholds.json` 的 `_provisional` 登记,不得写成代码里的裸常量。

## 最终全分支审查(`07ed73d..298220f`,19 提交,opus)—— 1 Critical / 5 Important / 多条 Minor

审查者**自己重跑了流水线与套件**(只读),并复核了三件事,结论值得留档:
- **真会话回归(§6.2)独立复现 → 通过**:走 `get_fused_latest_data → feature_engine → mapper`,5 个维度全部 `score=None / confidence=无`、`total_score=None`、「20 个指标槽中 0 个通过证据门」,全篇无「高」。**这就是 deferred #12 缺的那份人工证据,已由它补上。**
- **35 条封停名单逐条对 spec §5.2 核过,且确认无过度封停**(`au12_smile`/`pitch_median`/`interview_duration_mean`/`gaze_direction_x`/`au26_jaw_drop` 均干净通过)。
- **空断言那一类是真修了**:逐条核了 `matched == "1/4"` 钉法、`symmetry_score` 走缺口、三条叙事测试的 `_MIXED` 前置断言等。

### Critical(必须修)

**C1 —— 报告仍会输出无依据的结论:单指标 + 任意归一化 → 点分 + 档位标签。** 审查者**在真实 live 路径上复现了两次**:连接词丰富的转写 → `logical_thinking=100.0/卓越`,总分 `100.0/卓越`;真实 ASR 转写 → `logical_thinking=0.0/待提升`,报告头渲染「0.0 / 综合行为观测评分 / **待提升**」。两个成因:①`_dynamic_normalize`(`research_mapper.py:262-285`)是一组**没有登记的裸常量**(`density×50`/`jitter×5`/`energy×250`/`length/30`/`pitch/80` + pause 分箱),这就是一个未登记的常模替身;②`weighted_sum/total_weight` 的重归一让**1 个**过门槽产出与 4 个槽同样的 0–100 刻度,再由 `_get_level` 变成对人的评语。

**控制器核对 spec 原文,确认这是真实偏离**:§5.4 `:157-158`「综合科研潜力评分为 X 分,评级为 Y」的处置明写「**改为区间 + 置信度**」;§5.6「科研潜力评分 X 分」→「**区间 + 置信度 + 依据**」、「逻辑思维 58 分」→「…(**该维度目前无独立效标,仅供行为描述**)」;§5.4 末的「整节的正确形态」= **每维输出 `值 + 有效样本量 + 置信度 + evidence_gaps`**。

- **Ruling 40(裁定按 spec 实现区间/描述形态,不放宽 spec):** spec 是权威、计划是它的论证 —— 计划 8 个任务都没做这条presentation 改动(Ruling 32 还以"档位是分数自己的档"为由保留了 `total_level`,**该理由现被推翻**:1/20 覆盖下的五档评语就是对人的评定)。**裁定:实现 spec 的形态**;1 槽过门仍出「100.0/卓越」或「0.0/待提升」,与本分支"不再输出无证据的分数与结论"的主张直接冲突。 — 若判断错,代价 = 报告呈现形态改动超出计划范围(但 spec 明写,且使用者可在门 4 否决具体措辞);反向代价是分支主张不成立。
- **Ruling 41(修复波次 = C1 + I1 + I2 + I3 + I4):** 一次派发、一个修复者。**I5(下面)与 4 条"使用者裁量"项不进波次。**
  - I1:`test_confidence_can_be_none_and_low` **从不断言「低」**(fixture 只能产出「无」)→ 属空断言家族**第 9 个成员**,且 `sys.settrace` 扫描**抓不到它**(它是"缺断言"而非"未执行断言")—— 这条元发现一并记档。
  - I2:`_dynamic_normalize` 的裸常量违反 Global Constraints(临时阈值必须进 `evidence_thresholds.json` 且带 `_provisional`)。
  - I3:已出分维度旁缺 §5.6 要求的「无独立效标」提示句。
  - I4:`templates/dashboard.html:65-66` 的「生成综合判推报告/深度评估文书」是**用户可见的过度承诺**,而扫描只 glob `*.py` 抓不到它 → 改文案 + 把 `*.html` 与阈值 JSON 纳入扫描。
- **I5(不进波次,交使用者做合并决策):** `8d156f4` 把**使用者此前的未提交工作**(`app.py`、三个模块的 `api/app.py`、`video_pipeline.py`、`requirements.txt` 等 ~600 行)与仓库级 `.gitattributes` 一并纳入本分支。它已在提交信息里声明、也是使用者 2026-09-22 的决定,但**合并即把未经本轮审查的采集层改动带上 main**,而 ① 的安全论证正是"只改报告层"。→ 见最终汇报里的合并选项。

## 最终修复波次 —— 范围复审结果:`328bb4a`(1 提交,7 文件 +669/−135)**C1/I1–I4 全部 ADDRESSED,无新增 Critical/Important**

复审者独立核验(在**交付的产物**上数词,而不是读实现者的自述):

- `卓越|优秀|良好|合格|待提升|综合行为观测评分|综合科研潜力|判推|深度评估|归一值|权重|百分位|优于|Top|分位|None` —— **两份报告命中全为 0**;`本次观测覆盖`/`置信度上限`/`有效样本量`/`无独立效标` 在出分那份里齐备。
- 雷达图实测 `r=[1,0,0,0,0,1]`、trace 名「通过证据门的指标槽数」;0 槽会话为全 0 多边形(不会被读成低分)。
- I2 的 12 族折算因子**逐族重推与原 if 链等价**,并**逐个 patch 生产调用路径**验证(把某族写回裸常量 → 该族那一行变红);`_validate_scale_factors` 对缺 `basis`/非法 `basis_kind` 直接硬失败。
- I1 的 RED 理由正确(P7 `assert '中' == '低'`);I3 钉住「该维度目前无独立效标」且断言 `归一值` 不出现;I4 的扫描逐根设下限(P9b 证明整根消失会红),且**扫描是"因正确原因而绿"**(范围内剩下的命中只有 `.py` 注释里的「常模」,AST 路径豁免)。
- 独立复跑相关测试 13 passed、收集 51 tests,与实现者声称一致;注入探针日志 `/tmp/jx_probes_final.txt`。
- **前端那条查清了**:`/api/report/structured` 确实序列化 `total_score/total_level`,但**仓内无任何调用方**;唯一可能显示它的路径是**死代码**(`templates/dashboard.html` 把 `folderMap['report']` 设为 `''`,紧接着 `if (!folder) return;`)—— 且 `ALLOWED_FOLDERS` 也不含 `''`。**没有任何活路径把分数显示给人,C1 不存在"半修"。**

**两条 Low(非阻塞,记 M 系列):** ①新 `_std` 查找让单样本 ASR 键拿到 G2 的「本次会话内无变化」而非更准的「有效样本不足」(槽照样被拦,只是理由串欠准);②`scale_factor_for` 对等长匹配(`ratio`/`pause` 均 5 字符)靠 **JSON 插入顺序**决胜,重排序会静默改行为(已用测试钉住)。

**⚠️ 复审发现的用户可见遗留(交使用者决定):** 根 `app.py` 的 `/output/<path:filename>` 路由能直接提供 `data/output/` 下 **~378 份修复前生成的旧报告**,内容仍是「综合行为观测评分」+ 评级。UI 不链接它(需手输文件名),但**文件在盘上、路由可达**。

**① 全部完成:** 20 个提交(`07ed73d..328bb4a`),51 个测试,防线 0/2835510,最终审查 1 Critical + 5 Important 全部关闭并过复审。**合并等使用者确认。**
