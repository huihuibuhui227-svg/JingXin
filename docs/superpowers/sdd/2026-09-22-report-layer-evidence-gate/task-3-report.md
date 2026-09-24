# Task 3 报告:research_mapper 接入证据门

- 状态:**DONE_WITH_CONCERNS**(实现与测试全绿;发现 3 处 brief/计划缺陷与 1 处计划级缺口,已按下文最小方式处置,均需控制器知悉)
- 提交:`36f1f87`(单提交,3 个显式路径)
- 分支:`fix/report-layer-evidence-gate`(开工前已确认,未新建)
- BASE(dispatch 前):`e249450`
- 解释器:`~/miniconda3/envs/jingxin/bin/python`
- 未使用 amend/rebase,未 `git add -A`,未碰 `face_expression/` `gesture_analysis/` `voice_interaction/` `app.py`

---

## 1. 做了什么

按 brief 的 11 步顺序执行。改动落在两个文件:

**`report_frontend/research_mapper.py`**

| 删除 | 位置 | 依据 |
|---|---|---|
| `self.demo_text_data`(内置演示文本,16 句) | 原 `:19-38` | spec §5.3,它让 `logic_keyword_density`(权重 0.4,全模型最高)在无文本列时凭空得到常量 |
| `self.baselines`(6 条无出处 `{mean,std}`) | 原 `:40-47` | spec §5.5 |
| 第 2 步 文本 Fallback(假简历 → density) | 原 `:133-140` | spec §5.3 |
| 第 3 步 智能降级(三条硬编码代理:`tension_score ← 1−symmetry`、`jitter ← 1−hand_score/100`、`au7_freq ← au4`) | 原 `:182-200` | spec §5.3 |
| 第 4 步 兜底 `BASELINE_FILL` | 原 `:202-208` | spec §5.3 |
| 百分位计算(`math.erf`)与 `evidence_item["percentile"]` | 原 `:234-243`、`:254` | spec §5.5(Ruling 1 移入本任务) |
| `fluency_proxy` 指标槽(0.4,永久封停项) | `mapping_rules` | spec §5.3;权重和 1.4 → 1.0 |
| `"高" if found_valid_evidence >= 2 else "中"` | 原 `:268` | spec §5.1,置信度三值化 |

| 新增 | 说明 |
|---|---|
| `from .evidence_gate import confidence_from, gate, user_message` | 模块级 import |
| `matched` / `dim_gaps`(**维度循环内**声明)、`all_evidence_gaps`(**方法开头**声明一次) | Ruling 2 的两层缺口 |
| 指标准入循环:逐个指标过 `gate(found_key, found_val, n_valid=, std=)`,失败写 `dim_gaps.append(f"{human_name}: {user_message(chk)}")` | **只渲染 `user_message`,不碰 `chk.reason`** |
| `_fuzzy_match(keyword, all_features)` | 只保留原第 1 步精确子串与第 2 步去 `_mean/_std/_sum` 宽松匹配,**不含降级与兜底** |
| 无过门指标时的维度条目:`score=None` / `level="证据不足"` / `confidence="无"` / `evidence_gaps=dim_gaps` | 每个维度都有 `evidence_gaps`(含出分维度) |
| 有证据时:`confidence = confidence_from(len(matched), len(indicators))` | 结构性不可达"高" |
| 顶层 `evidence_gaps`(全维度缺口汇总)、`total_score` 可为 `None` | Interfaces 段要求 |

**`report_frontend/feature_engine.py`**:`extract_all_features()` 里新增第 4 步,给每个模态写入 `_n_rows`(扁平化后为 `<模态>__n_rows`),供 G3 取真实样本量。`__n_rows` 不参与指标匹配(所有关键词都不含 `n_rows`,已实测)。

**`tests/test_report_layer.py`**:brief Step 1 的两段代码块逐字照抄,共 5 条测试。

### 交接项的三条执行情况

1. **`_std` 必传 —— 已执行**:`std_key = found_key[:-5] + "_std"`,`std = all_features.get(std_key)`。真实数据上生效:`face_focus_score_std=0.0259`、`face_gaze_deviation_std=0.2195`、`gesture_left_hand_jitter_std=0.0`(后者正是被 G2 拦下的例子)。
2. **不渲染 `chk.reason` —— 已执行**:全仓 `grep chk.reason` 只出现在注释里;`dim_gaps` 只拼接 `user_message(chk)`。实测缺口文案为"未采集到对应数据 / 本次会话内无变化 / 有效样本不足 / 该指标本轮停用",无"封停""M3""精确重构"等维护者文案。
3. **pause 族 G3 阈值 = 2 —— 已按新行为使用**;真实语音指标 `pause_duration` 由 G4 拦下(封停),未受阈值影响。

---

## 2. 测试与结果

### TDD 证据

**RED(实现之前)**

```
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -v
→ 5 failed in 0.25s
```

```
E   assert 47.08 is None                                    # test_zero_input_produces_no_scores
E   AssertionError: 假简历兜底仍在
E   assert 'logic_keyword_density' != 'logic_keyword_density'   # test_no_fake_resume_fallback
E   AssertionError: assert 'focus_score' not in 'face_focus_score_mean'  # test_quarantined_columns_are_rejected
E   AssertionError: assert '无' in {'中'}                    # test_confidence_can_be_none_and_low
E   AssertionError: communication_fluency 权重和为 1.4000000000000001   # test_dimension_weights_sum_to_one
```

5 条全部失败,**且失败原因与 brief Step 2 的预期逐条对应**(47.08、假简历兜底、封停列进了证据链、"无"不可达、comm 权重和 1.4)—— 5/5 失败而非部分失败,说明测试对现状有约束力,不是恒真断言。

**GREEN(实现之后)**

```
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py tests/test_evidence_gate.py -v
→ 29 passed in 0.26s
```

```
~/miniconda3/envs/jingxin/bin/python -m pytest
→ 30 passed in 0.25s        # 25(Task 1+2)+ 5(本任务)
```

输出无 warning、无乱码、无 stray 打印(pytest 只显示统计行)。count 从实现的 25 → 30,增量恰为 5,未波及既有行为。

---

## 3. 回归证据(Step 10)

### 3.1 brief 指定的命令(逐字执行)

```
总分: None
  logical_thinking: score=None conf=无
  stress_resilience: score=None conf=无
  communication_fluency: score=None conf=无
  confidence_level: score=None conf=无
  cognitive_efficiency: score=None conf=无
gaps: ['逻辑关键词密度: 未采集到对应数据', '面部专注度: 本次会话内无变化', '视线稳定性: 有效样本不足', '困惑微表情 (皱眉): 未采集到对应数据', '面部紧张度: 有效样本不足']
```

`conf=高` 出现 **0 次**(基线 4/5 维为"高")✓。此会话 5 维全部"证据不足",原因留在 §5 的 concern G:
`get_fused_latest_data()` 选中的"最新"面部日志是 `face_au_log_20260526_002446.csv` —— **整份只有 4 行数据**(`wc -l = 5`,含表头),于是 n_valid=4、几乎所有 G3 阈值(30/20/10)都过不去。这不是实现问题,是这条会话本身是残缺桩数据。

### 3.2 补充回归:一条**完整**会话(更能说明问题)

为了让回归落在有真实样本量的会话上,我把 `data/logs/` 里最大的面部日志(14207 行)与最新的手势/语音日志链接成临时日志目录 `/tmp/sessA` 跑同一条链路。**改前/改后对照**:

| 维度 | 改前(HEAD) | 改后 |
|---|---|---|
| logical_thinking | 44.07 **高** 4/4 | 74.17 中 2/4 |
| stress_resilience | 63.21 **高** 4/4 | **None 无** 0/4 |
| communication_fluency | 56.59 **高** 3/5 | **None 无** 0/4 |
| confidence_level | 69.95 **高** 4/4 | 72.94 低 1/4 |
| cognitive_efficiency | 42.35 中 1/4 | 41.72 低 1/4 |
| 总分 | 54.98 | 64.98 |

- **改前 4/5 维"高"**,与 spec §0 的基线表完全一致(45/63/57/70/42)→ 这份对照复现了 spec 的基线。
- **改后"高"出现 0 次**;两个维度整维不出分(它们此前 4/4、3/5 全是兜底与封停列在支撑)。
- 手势模态 n_rows=4(手势日志只有 4 行)→ `hand_score`/`shoulder_score`/`jitter` 都被 G2 以"本次会话内无变化"拦下 —— 这是 std 伴随列真正生效的直接证据。

对照用的改前版本取自 `git show HEAD:report_frontend/research_mapper.py`,未改动工作区。

### 3.3 下游消费者探针(brief 未要求,补做)

- `ReportVisualizer.generate_all_charts(result, df_face)`:`score=None` 不崩(plotly 接受 None),7 张图正常产出 → 无新 breakage。
- `ReportGenerator._build_html_report(...)`:**抛 `TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'`**(`report_generator.py:237`),见 concern B。
- `app.py:227` 只 `jsonify` 结果 → `null` 可序列化,不受影响;`templates/dashboard.html` 不读 `score`。

---

## 4. 与 brief 不一致处 / 被迫偏离(逐条给证据)

brief 的代码块本身有 3 处缺陷,均在**实现前**用算术或探针确认过;我没有猜语义,而是按"让 brief 声明的意图成立"的最小改动处置。

### D1(重要)`__n_rows` 少一个下划线 —— 照抄会让 G3 全盘失效

- Step 5 写 `self.features['face']['__n_rows']`,Step 6 却用 `all_features.get(f"{modality}__n_rows", 0)` 读取。
- 扁平化是既有的 `full_key = f"{modality}_{k}" if not k.startswith(modality) else k`。代入 `k='__n_rows'`:`'face' + '_' + '__n_rows'` = **`face___n_rows`(3 个下划线)**,而查询键是 `face__n_rows`(2 个)→ 永远取到默认值 0。
- 若照抄,所有模态 n_valid=0,`_threshold_for` 的最小登记值是 2 → **全部指标被 G3 拦下**。5 条 brief 测试**仍会全绿**(零输入本来就该全无证据),Step 10 也会"通过"(本来就没有"高")—— 也就是缺陷会被测试掩盖。
- 处置:**改生产者一个字符**,`['_n_rows']`。此时扁平化结果恰好是 `face__n_rows` / `voice_research__n_rows`,与计划 line 697 的契约"产出 `<模态>__n_rows`"和 Step 6 的查询式**逐字一致**;feature_engine 侧注释已写明这个推导。
- 实测:`n_rows_by_modality = {'face': 14207.0, 'gesture': 4.0, 'voice_research': 8.0}`,G3 正常咬合(见 §3.2 表中 G3 拦下的 pitch/energy)。

### D2(重要)`split("_", 1)[0]` 对 voice 系模态取不到行数

- Step 6 的 `modality = found_key.split("_", 1)[0]`,对 `voice_research_research_pitch_variation_mean` 得到 **`'voice'`**,于是查询 `voice__n_rows` → 0 → 所有语音指标被 G3 无差别拦下(与 D1 叠加)。
- 处置:改成按**实际存在的前缀**比对 ——
  `modality = next((m for m in n_rows_by_modality if found_key.startswith(m + "_")), None)`。
  好处是不必在报告层硬编码模态名单,模态由 feature_engine 实际产出的键决定。
- 实测:`voice_research` 的 n=8 被正确读取(`pitch_variation` 因 n=8 < 阈值 10 被 G3 拦下)。

### D3 `_generate_deep_inference` 在百分位删除后会崩(brief 未提及)

- Step 3 要求删掉百分位、`inference_data` 只留 `{"val": ...}`。但 `_generate_deep_inference` 有 3 处读 `[...]['percentile']`,其中 `p = data['gaze_stability']['percentile']` **在 try 块之外**(原 `:316`),`data['jitter']['percentile']` / `data['au7_freq']['percentile']` 在布尔表达式里,都不受 `except KeyError` 保护 → 真实会话上会 KeyError 崩溃。
- 处置:三个槽位直接留空(`gaze_info = jitter_info = au_info = ""`)而不是 `.get("percentile")`。理由:`.get()` 会落到"视线稳定性正常 / 肢体略有紧张 / 面临一定认知挑战"这类**无证据断言**分支(此前因百分位总是算得出而不可达,`.get()` 会把它变成可达)—— 在一个以"删除伪造"为目标的改动里重新打开无证据断言是反向的。
- 代价(已知):模板句会出现"结合其，显示出…"的双逗号。该函数已被计划 Task 7 Step 4 整段删除,属过渡态;见 concern C。

### D4 `score=None` 的两处未防护比较(brief 未提及)

- `"total_level": self._get_level(final_total)` → `None >= 90` 会 `TypeError`。
  处置:`self._get_level(final_total) if final_total is not None else "证据不足"`(选用"证据不足"是为了让 `report_generator.py:149` 的 `result['total_level'].split()[0]` 继续可用)。
- `_generate_summary_narrative` 里 `if v['score'] > 0` → `None > 0` 会 `TypeError`。
  处置:改为 `is not None`,与 Step 8 的 `valid` 定义保持一致(否则两处对"有效维度"的判定不同)。

### D5 Step 5 的无保护访问会把"缺模态"变成"全空"

- 照抄的 `self.features['face']['_n_rows'] = ...` 在面部数据缺失时 `self.features['face']` 不存在 → KeyError → 落到 `extract_all_features` 的宽 `except Exception` → **返回 `{}`,连手势与语音特征一起丢掉**(既有代码不丢)。可见行数对每个模态都是纯附加项,不该有这种放大效应。
- 处置:守卫条件用 `in self.features`(features 有键 ⟺ 该模态提取成功),而非 `in self.data`。

### D6 Step 11 的提交命令漏了 `feature_engine.py`

- Step 5 明确要改 `feature_engine.py`,但 Step 11 的 `git add` 只列了 `research_mapper.py` 与 `tests/test_report_layer.py`,计划顶部的 Files 段也漏列了它。只按 Step 11 提交会让**消费者进仓库、生产者不进**,提交后的版本 n_valid 恒为 0。
- 处置:三个路径显式提交(`36f1f87` 含 `report_frontend/feature_engine.py`),提交信息里加了一行说明。

### D7 其他小项

- `evidence_gaps` 也写进了**出分维度**的条目:Interfaces 段要求"每个 dimension 增加 `evidence_gaps`",Step 7 只给了无证据维度的 dict。Task 6 的 `_render_dimension_block` 用 `dim.get("evidence_gaps", [])`,两者兼容。
- 删掉死变量 `matched_count` 与 `found_valid_evidence`(新逻辑用 `len(matched)`,二者已无引用;已 `grep` 确认全仓无引用)。

---

## 5. 自检结果(对照派发时点名的清单)

| 自检项 | 结果 |
|---|---|
| `demo_text_data` 及**每一处**使用都删净 | ✅ 全仓只剩 `__init__` 里的删除说明注释 |
| `self.baselines` 及**每一处**使用都删净(定义/兜底/百分位 3 处) | ✅ 同上;运行期无 `AttributeError`(真实会话已跑通) |
| 三条硬编码代理(tension←symmetry / jitter←hand_score / au7←au4) | ✅ 全仓无 `" (代理)"` 拼接(只剩 docstring 的历史说明) |
| `BASELINE_FILL` 是否仍可产出 | ✅ 不可达(grep 无赋值点) |
| `matched` / `dim_gaps` 在维度循环内、`all_evidence_gaps` 在方法级 | ✅ 实测 5 个维度缺口互不串味(18 条 = 2+4+4+4+4) |
| comm 维度权重和 = 1.0 | ✅ 0.4+0.3+0.2+0.1;测试 `test_dimension_weights_sum_to_one` 通过 |
| 测试输出是否纯净 | ✅ 无 warning / 无 stray 输出 |
| 缺口文案是否泄漏内部信息 | ✅ 只用 `user_message`;4 类门分别显示"未采集到对应数据/本次会话内无变化/有效样本不足/该指标本轮停用" |

---

## 6. Concerns(需控制器分诊)

**A.(计划级缺口,最重要)封停名单(§5.2)缺了 §5.3 明确要求"封停"的 7 个指标。**
spec §5.3 的处置列逐槽写了"封停":`gaze_stability`、`au4_freq`、`au7_freq`、`jitter`、`fluency_score`、`speech_ratio`、`eye_contact`;但 Task 2 实现、并经复审判 clean 的 `QUARANTINE` 只覆盖 spec §5.2 的表,不含这 7 个 key。计划全文也没有任何后续任务扩表(`grep QUARANTINE` 只命中 Task 2)。

后果(§3.2 会话实测,19 个指标槽逐个过门):

| 槽 | n_valid | std | 结果 | 真实键 |
|---|---|---|---|---|
| gaze_stability | 14207 | None | **PASS** | `face_gaze_stability_mean` |
| au4_freq | 14207 | None | **PASS** | `face_micro_exp_micro_exp_au_name_au4_freq` |
| eye_contact | 14207 | None | **PASS** | `face_eye_contact_ratio` |
| au7_freq | 14207 | None | **PASS** | `face_micro_exp_micro_exp_au_name_au7_freq` |
| jitter | 4 | 0.0 | BLOCK G2 | `gesture_left_hand_jitter_mean` |
| speech_ratio | 8 | 0.0 | BLOCK G2 | `voice_research_research_speech_ratio_mean` |
| fluency_score | — | — | 无键 | 等 M1 |

即:**4 个过门键全部是 §5.3 判为"封停"的列**,其中 3 个(§3.2 的 74.17 / 72.94 / 41.72)真的产出了分数。它们能过 G2 是因为 feature_engine 把它们写成**派生标量**,没有伴随 `_std` 列 → 交接项 1 的"必传 `_std`"对这几个键**无处可传**,G2 只能 fail-open。这不是本任务能修的(要么扩 `QUARANTINE`,要么给这几列补 std,两者都超出 brief 与 1~7 任务的分工)。
建议:控制器裁决是"§5.2 表权威、§5.3 的封停字样指别的机制",还是"派一个增量任务扩表"。**在扩表前,`gaze_stability` 与 `eye_contact` 仍会在任何样本量足够的会话上出分。**

**B.`report_generator.py` 在零证据会话上会崩(过渡窗口,Task 6 已覆盖)。**
`_generate_deep_text_analysis`(`:237`)的 `sorted(result['dimensions'].items(), key=lambda x: x[1]['score'])` 在 `score=None` 时 `TypeError`;`generate_report()` 的宽 `except` 会把它吞成"❌ 错误"并返回空串 —— 即**报告生成静默失败**。这不在本任务范围(计划 Task 6 重写该函数,且它的测试 `test_deep_analysis_handles_none_scores` 正是钉这一条)。提请注意:在 Task 6 落地前,这条链路上是坏的。

**C.过渡态文案:** `_generate_deep_inference` 的三个槽位留空,致模板句出现"结合其，显示出…"。该函数由 Task 7 Step 4 整段删除。若控制器认为中间态也不能出现病句,请裁决改法(我未自行发明文案)。

**D.测试只覆盖零证据路径。** brief 给的 5 条测试全部走"无证据"分支,**出分路径没有单测**(我手工构造过门输入验证:`gaze_stability_mean=0.8/std=0.05/n_rows=100` → 出分 80.0、`conf=中`、`matched=2/4`、`total_score=80.0`、`level=优秀`、每维 `evidence_gaps` 齐全)。计划里 Task 6/7 会往同一文件追加测试;是否补一条"有证据仍能出分"的守护测试,请控制器定。

**E.遗留死 import:** `math` 的最后一个使用点(百分位)被本任务删除,现已无用;`np`/`json` 在我改之前就无用(既有噪音)。我**没有**动 import —— 只清 `math` 而不清 `np`/`json` 是一次不完整的清理,且不在 brief 的删除清单里。列为 deferred minor。

**F.`model_metadata.version` 仍是 `"JingXin-Mapper-v10.1-FixedStats"`。**打分语义已实质改变,但 brief/计划均未要求改版本号(设计文档 §5.1 规矩 5 的"替换即新模型版本"针对的是阈值文件)。未自行改动,列为待裁决。

**G.`Step 10` 那条会话本身是残缺桩:** 被选中的最新面部/手势日志只有 4 行,所以 5 维全空。真正的"进步"看 §3.2 的完整会话对照。若控制器希望回归脚本更稳定,可考虑固定一个会话而不是"取最新"。

**H.行为变化提示:** 本任务后,维度条目里 `score` 可能为 `None`,`level` 可能是"证据不足" —— Task 6/Task 5 的消费者需按 Interfaces 段处理;`app.py` 与 `templates/` 已探针确认兼容。

---

## 7. 文件清单(本任务)

| 文件 | 状态 |
|---|---|
| `report_frontend/research_mapper.py` | 修改(161 插入 / 194 删除) |
| `report_frontend/feature_engine.py` | 修改(+11,Step 5;brief Step 11 漏列,见 D6) |
| `tests/test_report_layer.py` | 新建(+54,逐字照抄 Step 1) |
| `/home/huihuibuhui/jingxin/.superpowers/sdd/2026-09-22-report-layer-evidence-gate/task-3-report.md` | 本报告 |

未触碰:`face_expression/`、`gesture_analysis/`、`voice_interaction/`、`app.py`、`report_frontend/evidence_gate.py`(Task 2 交付物)、`report_frontend/report_generator.py`、`report_frontend/visualizer.py`、`templates/`。工作区里被 tracked 的 `.pyc` 大量 dirty 是运行测试产生的,未进提交。

---

## 9. 修复轮(裁决后,提交 `6a80a10`)

控制器采纳 D1–D7 全部,并下发 2 条修复。**未 amend `36f1f87`**,修复轮是独立提交。

### 9.1 修复 1:补 7 键槽位级封停(`report_frontend/evidence_gate.py`)

在 `QUARANTINE` 的 `"fist_status"` 之后插入控制器给定的 7 条,逐字照抄,并保留其注释。

**RED 证据(新测试对旧 QUARANTINE 的实测,用 `git show 36f1f87:report_frontend/evidence_gate.py` 载入为 `old_gate`):**

| 键 | 修复前 `is_quarantined` | 修复后 |
|---|---|---|
| `face_gaze_stability_mean` | **无** | 有 |
| `face_micro_exp_au_name_au4_freq` | **无** | 有 |
| `face_micro_exp_au_name_au7_freq` | **无** | 有 |
| `gesture_left_hand_jitter_mean` | **无** | 有 |
| `voice_research_research_speech_ratio_mean` | **无** | 有 |
| `face_eye_contact_ratio` | **无** | 有 |

6/6 由"无"变"有" → 新测试 `test_slot_level_quarantine_covers_spec_5_3` 在修复前必失败,不是恒真断言。

**验证命令与输出:**

```
~/miniconda3/envs/jingxin/bin/python -m pytest -v
→ 31 passed in 0.23s        # 原 30 + 新测试 1
```

新测试通过(`tests/test_report_layer.py::test_slot_level_quarantine_covers_spec_5_3 PASSED`)。

**误伤核对(实测,非推断):** 既有 `tests/test_evidence_gate.py` 用到的键(`pitch_median`、`face_focus_score_mean`、`overall_score`、`interview_duration_mean`、`blink_rate`、`pause_duration_mean`、`au12_smile_mean`)与新增 7 键**均无子串交叠**,25 项 Task 2 测试全部保持通过。`jitter` 的误伤风险按控制器核过的结论处理(`mapping_rules` 里只有 jitter 一个槽读 jitter 键)。

### 9.2 修复 2:`report_generator` 过渡防护(`report_frontend/report_generator.py:236-246`)

原 `:237-239` 三行替换为控制器的 `scored` 版本,注释里显式写明"Task 6 会整段重写本函数"。

**验证(零证据会话,即 Step 10 那条):**

```
FIX2 OK: HTML 生成成功, len = 7964
FIX2 OK: _generate_deep_text_analysis 不崩, len = 2256
```

修复前同一条链路:`TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'` → 被 `generate_report` 的宽 except 吞成空报告。

### 9.3 重跑:pytest(全量)

```
~/miniconda3/envs/jingxin/bin/python -m pytest
→ 31 passed in 0.23s
```

### 9.4 重跑:brief Step 10 真实会话回归

命令同 §3.1(逐字)。输出:

```
总分: None
  logical_thinking: score=None conf=无
  stress_resilience: score=None conf=无
  communication_fluency: score=None conf=无
  confidence_level: score=None conf=无
  cognitive_efficiency: score=None conf=无
gaps: ['逻辑关键词密度: 未采集到对应数据', '面部专注度: 本次会话内无变化', '视线稳定性: 有效样本不足', '困惑微表情 (皱眉): 未采集到对应数据', '面部紧张度: 有效样本不足']
```

`conf=高` **0 次** ✓。

### 9.5 逐维度证据缺口清单(控制器点名要求)

**会话 1 = Step 10 那条(最新):n_rows = face 4 / gesture 4 / voice_research 8。**
注意关卡顺序是"先失败先返回"(G1→G2→G3→G4),所以某键的实际拦截门取决于哪一关先响;下表右列是该键**实际触发的**门。

| 维度 | 槽 | 拦截门 | 缺口文案 | 真实键 / n_valid |
|---|---|---|---|---|
| logical_thinking | 逻辑关键词密度 | 无键 | 未采集到对应数据 | — |
| logical_thinking | 面部专注度 | G2 | 本次会话内无变化 | `face_focus_score_mean` / 4 |
| logical_thinking | 视线稳定性 | G3 | 有效样本不足 | `face_gaze_stability_mean` / 4 |
| logical_thinking | 困惑微表情 (皱眉) | 无键 | 未采集到对应数据 | — |
| stress_resilience | 面部紧张度 | G3 | 有效样本不足 | `face_tension_score_mean` / 4 |
| stress_resilience | 肢体抖动 | G2 | 本次会话内无变化 | `gesture_left_hand_jitter_mean` / 4 |
| stress_resilience | 视线偏差 | G3 | 有效样本不足 | `face_gaze_deviation_mean` / 4 |
| stress_resilience | 面部对称性 | G3 | 有效样本不足 | `face_symmetry_score_mean` / 4 |
| communication_fluency | 语音流畅度 | 无键 | 未采集到对应数据 | — |
| communication_fluency | 有效说话占比 | G2 | 本次会话内无变化 | `voice_research_research_speech_ratio_mean` / 8 |
| communication_fluency | 语调变化 | G3 | 有效样本不足 | `voice_research_research_pitch_variation_mean` / 8 |
| communication_fluency | 平均停顿时长 | **G4** | 该指标本轮停用 | `voice_research_research_pause_duration_mean_mean` / 8 |
| confidence_level | 手势自信分 | G2 | 本次会话内无变化 | `gesture_left_hand_score_mean` / 4 |
| confidence_level | 肩部放松度 | G2 | 本次会话内无变化 | `gesture_shoulder_score_mean` / 4 |
| confidence_level | 眼神接触比例 | G3 | 有效样本不足 | `face_eye_contact_ratio` / 4 |
| confidence_level | 语音能量 | G3 | 有效样本不足 | `voice_research_research_energy_mean_mean` / 8 |
| cognitive_efficiency | 眼部挤压 (费力) / 眨眼频率 / 回答详尽度 / 反应延迟 | 无键 | 未采集到对应数据 | — |

**会话 2 = §3.2 的完整会话 sessA:face 14207 / gesture 4 / voice_research 8。**这条是修复 1 的**主证据**:修复前 4 个键 PASS 并产出 3 个维度的分数,修复后全部被 G4 拦下。

| 维度 | 槽 | 修复前 | 修复后 | 缺口文案 | 真实键 / n_valid |
|---|---|---|---|---|---|
| logical_thinking | 逻辑关键词密度 | 无键 | 无键 | 未采集到对应数据 | — |
| logical_thinking | 面部专注度 | G4 | G4 | 该指标本轮停用 | `face_focus_score_mean` / 14207 |
| logical_thinking | 视线稳定性 | **PASS → 出分** | **G4** | 该指标本轮停用 | `face_gaze_stability_mean` / 14207 |
| logical_thinking | 困惑微表情 (皱眉) | **PASS → 出分** | **G4** | 该指标本轮停用 | `face_micro_exp_micro_exp_au_name_au4_freq` / 14207 |
| stress_resilience | 面部紧张度 | G4 | G4 | 该指标本轮停用 | `face_tension_score_mean` / 14207 |
| stress_resilience | 肢体抖动 | G2 | G2 | 本次会话内无变化 | `gesture_left_hand_jitter_mean` / 4 |
| stress_resilience | 视线偏差 | G4 | G4 | 该指标本轮停用 | `face_gaze_deviation_mean` / 14207 |
| stress_resilience | 面部对称性 | G4 | G4 | 该指标本轮停用 | `face_symmetry_score_mean` / 14207 |
| communication_fluency | 语音流畅度 | 无键 | 无键 | 未采集到对应数据 | — |
| communication_fluency | 有效说话占比 | G2 | G2 | 本次会话内无变化 | `voice_research_research_speech_ratio_mean` / 8 |
| communication_fluency | 语调变化 | G3 | G3 | 有效样本不足 | `voice_research_research_pitch_variation_mean` / 8 |
| communication_fluency | 平均停顿时长 | G4 | G4 | 该指标本轮停用 | `voice_research_research_pause_duration_mean_mean` / 8 |
| confidence_level | 手势自信分 | G2 | G2 | 本次会话内无变化 | `gesture_left_hand_score_mean` / 4 |
| confidence_level | 肩部放松度 | G2 | G2 | 本次会话内无变化 | `gesture_shoulder_score_mean` / 4 |
| confidence_level | 眼神接触比例 | **PASS → 出分** | **G4** | 该指标本轮停用 | `face_eye_contact_ratio` / 14207 |
| confidence_level | 语音能量 | G3 | G3 | 有效样本不足 | `voice_research_research_energy_mean_mean` / 8 |
| cognitive_efficiency | 眼部挤压 (费力) | **PASS → 出分** | **G4** | 该指标本轮停用 | `face_micro_exp_micro_exp_au_name_au7_freq` / 14207 |
| cognitive_efficiency | 眨眼频率 / 回答详尽度 / 反应延迟 | 无键 | 无键 | 未采集到对应数据 | — |

**7 个目标槽的落点小结:**

| 目标槽 | 落在哪一维 | 修复后拦截门 | 说明 |
|---|---|---|---|
| `gaze_stability` | logical_thinking | **G4**(sessA) / G3(会话 1) | sessA 上 G3 先过(14207≥30),由新增封停拦下 |
| `au4_freq` | logical_thinking | **G4**(sessA);会话 1 无键 | 同上 |
| `au7_freq` | cognitive_efficiency | **G4**(sessA);会话 1 无键 | 同上 |
| `jitter` | stress_resilience | G2(两条会话) | 手势日志只有 4 行且 jitter 恒定 → G2 先响;G4 是第二道 |
| `speech_ratio` | communication_fluency | G2(两条会话) | 同类,std=0 先响 |
| `eye_contact` | confidence_level | **G4**(sessA) / G3(会话 1) | sessA 上由新增封停拦下 |
| `fluency_score` | communication_fluency | 无键 | 生产分支是死代码,从来没有键 |

**sessA 端到端结果(修复后):**

```
sessA 总分: None | level: 证据不足
  logical_thinking:      score=None conf=无 0/4
  stress_resilience:     score=None conf=无 0/4
  communication_fluency: score=None conf=无 0/4
  confidence_level:      score=None conf=无 0/4
  cognitive_efficiency:  score=None conf=无 0/4
```

即:修复前 sessA 上"3 个维度出分、0 次高置信度"(§3.2),修复后是**整份诚实空报告**(21/21 槽被拦,其中 4 个"无键")。这正是 spec §7.3 预期的状态 —— ① 与 M1/M2 同轮交付前,报告就是空的;**区别在于现在空得有理由,而不是靠坏列凑出 64.98 分**。

### 9.6 修复轮遗留

- §6 的 concern A/B 关闭;concern C(过渡态双逗号文案)、D(出分路径无单测)、E(死 import)、F(版本号)、G(最新会话是桩数据)保持原status,待控制器分诊。
- 新增观察:`jitter` / `speech_ratio` 这类槽在 G2 先响时,用户看到的是"本次会话内无变化"而不是"该指标本轮停用"。语义上**都对**(两条都成立),且不泄漏内部文案;但若产品上希望"封停"优先于"常量",需要调整关卡顺序(G4 提到 G2 前)—— 这属于 spec §5.1 的规则设计,未自行改动。

---

## 10. 修复轮 2(复审后,提交 `d0e7240`)

复审结论:§9 的两条修复 ADDRESSED,无新增 Critical/Important;但复审抓到**我在 §9.2 的防护本身造成了反向后果** —— 加防护前零证据会话崩成空报告,加防护后它开始说假话(零证据下的才能断言)。控制器裁决:全无证据即短接。**未 amend `6a80a10`**。

### 10.1 修复 1:全无证据短接(`report_generator.py:125-136`)

在 `_generate_deep_text_analysis` 顶部(`html_parts` 之前)插入控制器给定的短接块,逐字照抄。

**原第 4 节里 §9 加的 `scored = [...]` 已由这处承接**,第 4 节改回排序三行(不再重复声明):

```python
        # scored 由函数顶部的过渡短接处提供(全无证据已在上面 return,此处 scored 必非空)
        sorted_dims = sorted(scored, key=lambda x: x[1]['score'], reverse=True)
        top_dim = sorted_dims[0][1]['display_name']
        bottom_dim = sorted_dims[-1][1]['display_name']
```

**RED 证据(新测试对修复前版本无约束力?—— 非也,已实测):** 把 `6a80a10` 的 `report_generator.py` 载入为 `report_frontend._old_rg_probe`(探针文件用后即删),跑同一条零证据输入:

```
                                   ↓ 修复前(6a80a10)的输出
        <h3>4. 综合结论与发展建议</h3>
        <p>综上所述，候选人在 <strong>证据不足</strong> 维度表现最为突出，显示出良好的科研天赋。 然而，在 <strong>证据不足</strong> 维度上得分相对较低，是主要的短板所在。 建议后续针对该短板进行专项训练（如模拟高压面试、正念冥想等）。总体而言，该候选人具备从事科研工作的基本心理素质。</p>

  断言 '科研天赋' 出现: True  -> RED(失败)
  断言 '心理素质' 出现: True  -> RED(失败)
  断言 '最为突出' 出现: True  -> RED(失败)
  断言 '表现最为' 出现: True  -> RED(失败)
```

与复审的引文逐字一致。4/4 断言在修复前必失败 → 新测试有约束力。

### 10.2 修复 2:分数卡片不渲染 None(`report_generator.py:277-278, 328`)

```python
        # 无证据时 total_score 为 None,直接插值会印出"None 分"
        _score_display = result['total_score'] if result['total_score'] is not None else "—"
```

分数卡片 `<div class="score-number">{_score_display}</div>`。

**为何只改这一处**(`:171` 那处 `{result['total_score']}` 未动,符合裁决给的单一改点):`:171` 在 `_generate_deep_text_analysis` 内部,而 `total_score is None` ⟺ 无任何维度出分 ⟺ 已被顶部短接拦下;**到达 `:171` 时 `total_score` 必为实数**。即短接与这一处覆盖了全部 `None` 取值路径。

### 10.3 修复 3:两条测试

按裁决逐字加入 `tests/test_report_layer.py`。

**第二条(`test_all_slot_level_quarantine_keys_are_pinned`)的 RED 已实测** —— 对 `36f1f87` 的旧名单逐键调用 `is_quarantined`:

```
  gaze_stability   修复前=无 现在=有  -> 新测试 RED
  au4_freq         修复前=无 现在=有  -> 新测试 RED
  au7_freq         修复前=无 现在=有  -> 新测试 RED
  jitter           修复前=无 现在=有  -> 新测试 RED
  speech_ratio     修复前=无 现在=有  -> 新测试 RED
  eye_contact      修复前=无 现在=有  -> 新测试 RED
  fluency_score    修复前=无 现在=有  -> 新测试 RED
```

7/7 由"无"变"有" —— 含 `fluency_score` 这条**今天没有活的键**的槽位关键词,正是复审点名要补钉的漏洞。

### 10.4 命令与输出

```
~/miniconda3/envs/jingxin/bin/python -m pytest -q
→ 33 passed in 0.24s        # 31 + 2
```

**零证据真实会话端到端(brief Step 10 那条链路):**

```
deep 长度: 541
<h3>行为观测摘要</h3><p><strong>本次会话未采集到足以支撑评估的有效证据。</strong></p><ul><li>逻辑关键词密度: 未采集到对应数据</li><li>面部专注度: 本次会话内无变化</li><li>视线稳定性: 有效样本不足</li>…</ul>

HTML 长度: 6246
含 "None": False | 含 score-number>—: True
  断言 '科研天赋' 出现: False
  断言 '心理素质' 出现: False
  断言 '最为突出' 出现: False
  断言 '表现最为' 出现: False
```

**有证据路径未受影响(短接不得误伤):** 构造一个仍未被封停的槽(`logic_keyword_density`,n_rows=100、std=0.01):

```
总分: 100.0 | level: 卓越
  logical_thinking: score=100.0 conf=低 1/4
  stress_resilience / communication_fluency / confidence_level / cognitive_efficiency: score=None conf=无
deep 长度: 2248 (非短接, 走了原段落)
第4节存在: True
第4节文本: 综上所述，候选人在 <strong>逻辑思维与专注度</strong> 维度表现最为突出，…
```

即:短接只在"无任何维度出分"时生效;混合情形(部分维度出分)仍走原段落,`scored` 在函数级作用域可用,排序正常。

### 10.5 遗留(已判给 Task 6,本轮未动)

复审报的四条 out-of-scope 由控制器判给 Task 6,本轮未触碰:`:165/176/194/215` 四处 `['percentile']` 默认 50 导致恒定"超越 50% 人群";`:137` 查错键名(`face_micro_exp_au_name_au4_freq` vs 实际 `face_micro_exp_micro_exp_au_name_au4_freq`);`:129/174` 默认值驱动的"极佳的情绪控制力";`research_mapper.py:5,8` 的未使用 import。四条都在 Task 6 要整段重写的函数内,且被 §10.1 的短接在今天的会话上挡在门外。

其中第 1 条复审特别指出"这是你的改动引入的新形态(Task 3 之前至少是算出来的,现在是恒定 50)" —— 记录在案。它的成因是 §9 的 Step 3 按 brief 删除 `inference_data["percentile"]`,而 `report_generator` 侧用 `.get('percentile', 50)` 兜住 → 恒 50。**删除百分位的决定来自 brief Step 3,不是本任务的自选动作**;该调用点属 Task 6 的范围。
