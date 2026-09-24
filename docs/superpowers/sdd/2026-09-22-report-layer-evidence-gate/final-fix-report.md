# 最终全分支审查修复报告(C1 + I1 + I2 + I3 + I4)

- 分支:`fix/report-layer-evidence-gate`
- 修复提交:**`328bb4a`** — `fix: 报告层停止渲染评分与评级,改按区间+置信度+依据呈现(spec §5.4/§5.6)`
  (7 文件,669 插入 / 135 删除;父提交 `298220f`;无 amend / rebase / force)
- 解释器:`~/miniconda3/envs/jingxin/bin/python`(pytest 9.1.1)
- 测试:**51 passed**(修复前 41;新增 10 个,amend 3 个);`-W error` 下同样 51 passed
- 范围:`report_frontend/` + `templates/` + `tests/`(未碰 `face_expression/`、`gesture_analysis/`、
  `voice_interaction/`、根 `app.py`、`experiments/`);未新增依赖;提交内容全 LF

---

## 一、逐条处置

### C1(Critical)—— 报告不再输出无依据的结论

**改了什么**

| 位置 | 之前 | 现在 |
|---|---|---|
| 报告头分数卡 `report_generator.py` | `{total_score}` + 「综合行为观测评分」+ `{total_level}`(实测印出「0.0 / 综合行为观测评分 / **待提升**」) | 「📊 本次观测覆盖」`{n_passed} / {n_slots}` + 「个指标槽通过证据门」+ 「置信度上限:{cap}」 |
| 深度分析摘要 `_generate_deep_text_analysis` | 「综合行为观测摘要:**{total_level}**(置信度上限:…)」 | 整段删除;聚合呈现移到报告头部 |
| 维度块 `_render_dimension_block` | 「依据 N/M 个指标;置信度:低」+ 表(指标/原始值/**归一值**/权重) | 「过门指标 N/M;置信度:低。(该维度目前无独立效标,仅供行为描述)」+ 表(指标/**原始值**/**本场会话内观测区间**/**有效样本量**) |
| 头部依据块(新增) | —— | 「依据:通过证据门的指标」逐条 `维度 · 指标:原始值 X;本场会话内观测区间 lo–hi(与原始值同量纲);有效样本量 n` |
| 雷达图 `visualizer._build_radar_figure` | 半径 = `dim['score']`(0–100,轨迹名「候选人得分」,标题「五维行为观测」) | 半径 = 过门槽数(0–4,轨迹名「通过证据门的指标槽数」,标题「五维证据覆盖(通过证据门的指标槽数)」) |
| 控制台 `map_features_to_scores` | `得分：100.0 (卓越)` | `过门指标 1/4，置信度：低` |

`total_score` / `total_level`(以及每维 `level`)**按派发允许保留在 payload 里**(API 稳定),
但报告层一个字都不渲染 —— 已在 `research_mapper.py` 返回处写明这一点。

**满足的 spec 行**

- §5.4 `:157-158`:「综合科研潜力评分为 X 分,评级为 Y」→ **改为区间 + 置信度**。
  点分与五档评语(含 `_get_level` 的输出)在报告与控制台均不再出现。
- §5.4 末「**整节的正确形态**」:每维输出 `值 + 有效样本量 + 置信度 + evidence_gaps`
  —— 维度块的四列与缺口清单就是这四项。
- §5.6:「科研潜力评分 X 分」→ **区间 + 置信度 + 依据**;聚合呈现 = 头部覆盖卡(覆盖事实 + 置信度上限)
  + 依据块(逐指标区间)。「逻辑思维 58 分」→ 加「(**该维度目前无独立效标,仅供行为描述**)」。
- §5.5:**区间只由本场会话自己的测量构成** —— `原始值 ± 会话内标准差`,`coverage.passed_slots`
  与维度表用的是同一个数;没有任何百分位、「优于 X%」、人群位置表述(有测试反向扫这些词)。

**为什么聚合给的是"覆盖事实 + 逐指标区间"而不是一个综合区间(我的裁定,写在报告里备否决)**

§5.6 的示例(「处于标定样本第 60–75 分位区间」)以**标定样本存在**为前提,而 ① 没有
(§1.1 的许可 + §1.3 的伦理审查卡住 M5),§5.5 又明禁暗示本系统并不拥有的人群参照。
所以聚合层的可选形态只有两种:

1. 用本场数据造一个"综合区间"—— 但过门指标的量纲互不相同(density 无量纲、pause 秒、
   energy 幅度),在没有标定样本把量纲折到同一尺度之前,**任何跨指标区间都是假的**;
   而这正是 C1 要拿掉的那个 0–100 复合标尺。故**不做**。
2. 照 §5.4 的兜底:**陈述覆盖事实(多少槽过门、置信度上限),不印那个数字** —— 采用这条。
   区间以**每个过门指标各自的实测区间**给出(同在报告头部的依据块与各维度明细里),
   它只依赖本场测量,可辩护。

所以头部读起来是:「1 / 20 个指标槽通过证据门 · 置信度上限:低」+「未产出综合评分,也不给评级:
本系统尚无标定样本,不同指标的量纲无法折算为同一尺度。」+ 逐指标实测区间。

**守护测试**

- `test_report_renders_no_candidate_rating`(高分端,`_MIXED` → 旧实现 100.0/卓越):
  断言五个档位词、`综合行为观测评分`、`str(total_score)` 都不在渲染出的 HTML 里;
  **并正向断言**「观测区间」「置信度上限」「逻辑关键词密度(依据)」必须在 —— 否则"整段删空"也能过。
- `test_zero_value_scored_session_renders_no_verdict`(低分端,`_LOW` → 旧实现 0.0/待提升,
  即审查者在真实 ASR 转写上复现的那条)。只测高分端会漏掉低分端,故两端都钉。
- `test_deep_analysis_summary_has_no_point_score_or_level`(深度分析那一面)。
- `test_radar_plots_coverage_not_scores` + `test_radar_has_no_norm_baseline`(图表那一面)。
- 三条测试都以 `assert result["total_level"] in _LEVEL_WORDS` 开场 —— **先证明 fixture 真的走进了
  被删掉的那条路径**,否则零证据 fixture 下档位是「证据不足」,断言恒真(本计划栽过九次的那种)。

### I1(Important)—— 修好那个测不出「低」的测试

`test_confidence_can_be_none_and_low` 重写:去掉只能产出「无」的
`interview_pause_duration_mean` fixture,改为

```python
zero = mapper.map_features_to_scores({})["dimensions"]["logical_thinking"]["confidence"]
assert zero == "无"
low = mapper.map_features_to_scores(_MIXED)["dimensions"]["logical_thinking"]["confidence"]
assert low == "低"          # ← confidence_from 不再返回「低」时本行变红
assert tuple(confidence_from(n, 4) for n in range(5)) == ("无","低","中","中","中")
```

满足 spec §6.5(「无」与「低」都必须可达)。`_MIXED`(1/4 槽过门)提供「低」——
**RED 证据见 P7**。

### I2(Important)—— 折算因子登记进 JSON

- `report_frontend/evidence_thresholds.json` 升到 `0.2.0-provisional`,新增
  `_default_scale_factor` 与 `scale_factors`(12 族),每族带 `kind` + `basis_kind` + `basis`;
  `_scale_factors_note` / `_scale_factors_basis_note` 说明 `legacy_arbitrary` 的诚实读法。
- `evidence_gate.py` 新增 `scale_factor_for()`(与阈值、封停名单同一套**最长子串**匹配)、
  `normalize_value()`(kind 分派:identity / identity_or_percent / full_scale / pause_bins),
  以及 `load_thresholds` 里的 `_validate_scale_factors`(缺 `basis` 或 `basis_kind` 非法即硬失败)。
- `research_mapper._dynamic_normalize` 变成一行委托;**函数体内不再有任何数值常量**
  (0.0/1.0 只是 0–1 值域边界,不是标尺参数)。
- 行为**逐族保持不变**(含经 `_score`/`identity` 分支的旧语义),这也是测试的期望值来源。

**依据强度(按派发要求如实分类,没有替哪一族编依据)**

| 族 | basis_kind | 依据 |
|---|---|---|
| `score`(÷100)、`freq`/`ratio`/`stability`/`contact`(恒等) | `definitional` | 0–100 分制字段;比率/频率/占比按定义落在 0–1 |
| `density` `jitter` `deviation` `pause` `length` `energy` `pitch_variation` + 默认族 | `legacy_arbitrary` | **没有物理或定义依据**,是历史遗留的满量程/分箱 |

`basis_kind: legacy_arbitrary` 是**允许的显式取值**(不是"没填"):登记的目的是让这些因子
可被审查与替换,不是给它们编一个出处。可辩护的替代(建议交 M4/M5 定夺,本轮未擅自改):

1. `density`:密度**按定义已在 [0,1]**,旧实现的 `×50` 等于把已归一的量再放大 50 倍 →
   建议改按 `ratio` 族(恒等)。
2. `jitter` / `deviation` / `energy` / `pitch_variation` / `pause`:对应槽当前**全部被 G4 封停**,
   因子不可达 → 建议 M3/M4 重定义时一并替换或删除,而不是现在换个数。
3. `length`(30 字满量程):"回答详尽度"没有天然满量程 → 建议等 M1 真回答长度落地后用**会话内**
   分位差或直接不折。

**守护测试**:`test_scale_factors_are_registered_not_hardcoded`(逐族把 JSON 里的登记值改掉,
断言输出随之改变 —— 任何一族被写回裸常量,该族那一行就拿到旧值)、
`test_every_registered_scale_factor_declares_a_basis`(登记完整性)。
**RED 证据见 P5 / P6。**

### I3(Important)—— 补上 §5.6 的提示句

`_render_dimension_block` 现在对**出分维度**渲染
「(该维度目前无独立效标,仅供行为描述)」,与 spec §5.6 的替换表逐字一致;未出分维度保持
§5.4 的「证据不足 —— 本次未采集到足以评估该行为线索的有效样本。」不变。
守护测试:`test_scored_dimension_follows_spec_5_4_shape_and_carries_5_6_caveat`
(同时钉四列形态与 `归一值` 不再出现)。**RED 证据见 P2。**

### I4(Important)—— 用户可见的过度承诺 + 扫描盲区

- `templates/dashboard.html:65-66`:
  「📑 生成综合判推报告」→「📑 **生成行为观测报告**」;
  「整合所有数据,生成 HTML 深度评估文书。」→「**汇总各模态通过证据门的指标与证据缺口,
  生成 HTML 行为观测报告。**」
  守护测试:`test_dashboard_report_card_describes_what_it_produces`(断言 `判推`/`深度评估`/`评估文书`
  均不在文中,且 `行为观测报告` 在文中 —— 正反两个方向都可证伪)。**RED 证据见 P8。**
- 扫描面:`SCOPE` 的 glob 从 `*.py` 扩到 **`*.py` + `*.html` + `*.json`**(同一批根目录);
  `.py` 仍走 AST(注释豁免),`.html`/`.json` 按行读文本;反空断言下限改为**逐根**
  `report_frontend ≥ 5`、`templates ≥ 2`(实测 9 / 2)。
  排除目录 `data/`(运行时产物:`report_frontend/data/output` 下 66 MB × 17 张 plotly 图表;
  其字符串来自本扫描已覆盖的 visualizer/report_generator,扫它们只把测试绑在生成物上)——
  这是**唯一**的排除,已写在 `_SKIP_DIR_NAMES` 与注释里。
  **RED 证据见 P9(注入禁止词)与 P9b(整份模板消失 → 逐根下限报红,而 offenders 为空)。**

---

## 二、RED/GREEN 证据(全部在**已提交状态** `328bb4a` 上重跑)

方法:逐条把被守护的改动**反向注入**(精确字符串替换),跑对应测试,再用 `/tmp` 备份还原,
比对 sha256。脚本 `/tmp/jx_probes.py`、`/tmp/jx_probes2.py`,原始输出 `/tmp/jx_probes_final.txt`、
`/tmp/jx_probes2_final.txt`。**14 个探针全部 `pytest rc=1` 且还原后 sha256 逐字节一致。**

| 探针 | 注入(反向) | 变红的测试 | 失败原因(节选) | 还原校验 |
|---|---|---|---|---|
| P1 C1-头部 | 覆盖卡退回 `{total_score}`/`{total_level}` | `test_report_renders_no_candidate_rating`、`test_zero_value_...` | `AssertionError: 报告仍把档位评语渲染成对人的评级:卓越` / `:待提升` | `11c36cc3…` ✅ |
| P2 I3/§5.4 | 删提示句 + 表头退回归一值/权重 | `test_scored_dimension_follows_spec_5_4_shape_and_carries_5_6_caveat` | `AssertionError: 出分维度缺 无独立效标` | `11c36cc3…` ✅ |
| P3 C1-深度摘要 | 加回「综合行为观测摘要:{档位}」 | `test_deep_analysis_summary_has_no_point_score_or_level` | `AssertionError: 深度分析摘要仍印档位:卓越`(命中 `<p>综合行为观测摘要：卓越（…）</p>`) | `11c36cc3…` ✅ |
| P4 C1-雷达图 | 半径退回 `dim['score']` | `test_radar_plots_coverage_not_scores` | `AssertionError: 半径里出现空值 …:[100.0, None, None, None, None, 100.0]` | `6f77e10e…` ✅ |
| P4b 雷达图名 | 轨迹名退回「候选人得分」 | `test_radar_has_no_norm_baseline` | `assert ['候选人得分'] == ['通过证据门的指标槽数']` | `6f77e10e…` ✅ |
| P5 I2 | `normalize_value` 整段退回旧的裸常量实现 | `test_scale_factors_are_registered_not_hardcoded` | `AssertionError: logic_keyword_density 的归一化没跟着登记值变:期望 0.05,实际 1.0` | `02b44379…` ✅ |
| P6 I2-登记完整性 | 删掉 `energy` 族的 `basis` | `test_every_registered_scale_factor_declares_a_basis` | `ValueError: scale_factors[energy] 未声明 basis(spec §5.1)` | `f361c8d9…` ✅ |
| P7 I1 | `confidence_from` 的「低」改成「中」 | `test_confidence_can_be_none_and_low` | `AssertionError: 1/4 个槽过门必须落在「低」… assert '中' == '低'` | `02b44379…` ✅ |
| P8 I4-文案 | 按钮退回「判推 / 深度评估」 | `test_dashboard_report_card_describes_what_it_produces` | `AssertionError: 总控台仍宣称产出「判推」` | `63703918…` ✅ |
| P9 I4-扫描面 | 往 `templates/dashboard.html` 注入「焦虑」 | `test_no_banned_words_in_output_strings` | `AssertionError: 输出字符串含禁止词: templates/dashboard.html:66 焦虑` | `63703918…` ✅ |
| P9b 反空断言 | 整份 `templates/dashboard.html` 改名消失 | 同上 | `AssertionError: templates 只扫到 1 个文件(<2),本测试退化为空断言`(**offenders 为空** —— 只有逐根下限能抓到) | `63703918…` ✅ |
| P10 §5.6-区间 | `_std` 取法退回只认 `_mean` 后缀 | `test_observed_interval_is_session_derived` | `AssertionError: 区间未由本场会话的均值/标准差构成:None` | `2d96f648…` ✅ |
| P11 §8.2#2 | JSON 里把 `pause` 族挪到 `ratio` 之前 | `test_pause_duration_is_shadowed_by_the_ratio_family` | `assert 'pause_bins' == 'identity'` | `f361c8d9…` ✅ |

三个样本探针的完整输出形态(供核对格式):

```
### P5 I2：normalize_value 整段退回裸常量实现
  文件 report_frontend/evidence_gate.py  sha256 还原前=还原后 02b44379cffc8a8d  ✅字节一致
  pytest rc=1  ['1 failed in 0.27s']
    E  AssertionError: logic_keyword_density 的归一化没跟着登记值变:期望 0.05,实际 1.0 —— 该族的因子可能被写回了代码里的裸常量
### P7 I1：confidence_from 永不返回「低」
  pytest rc=1  ['1 failed in 0.27s']
    E  AssertionError: 1/4 个槽过门必须落在「低」—— 若 confidence_from 不再返回「低」,本断言变红
    E  assert '中' == '低'
### P9b I4-扫描面：整份 templates/dashboard.html 消失
  pytest rc=1  ['1 failed in 0.30s']
    E  AssertionError: templates 只扫到 1 个文件(<2),本测试退化为空断言
    E  assert 1 >= 2
```

探测结束后工作区对已跟踪文件**零改动**(`git status` 只剩既有 `.pyc` 与既有未跟踪产物),
且 `evidence_gate.py`/`report_generator.py`/`templates/dashboard.html` 的 sha256 与提交内一致
(`1d86f6b9…` / `53001b4a…` / `4cc48255…`)。

GREEN:`~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -q` → **51 passed**;
`-W error` 下同样 51 passed。

---

## 三、交付物:两条路径的渲染文本

重新生成(提交后):
`python -m report_frontend.report_generator`(真实批次路径)+ 同一套代码渲染 `_MIXED` 出分会话;
两份报告与其 6+6 张图表(相对路径 iframe)同目录放在 `~/shared/jingxin_gate4/`:

- `jingxin_gate4_report.html`(0 / 20,真实批次路径)← `data/output/Research_Assessment_Report_20260924_114218.html`,
  sha256 `35a850decb2a1043…`
- `jingxin_gate4_report_scored.html`(1 / 20,出分路径)← `data/output/Research_Assessment_Report_SCORED_20260924_114217.html`,
  sha256 `177992c13d4dceb0…`
- 机器检查:两份都是 **禁止词 0 命中 / 档位词 0 命中 / 「综合行为观测评分·科研潜力·判推·深度评估」0 命中 / `None` 不出现**,
  iframe 6/6 都能在同目录找到。

### 3.1 真实批次路径(今天唯一可达:20 槽全部没过门)

```
JingXin 面试行为观测报告
🔬 JingXin 面试行为观测报告
基于多模态行为量的结构化观测
2026-09-24 11:42 | JingXin-Mapper-v11-EvidenceGate
📊 本次观测覆盖
0 / 20
个指标槽通过证据门
置信度上限：无
📝 综合总结
未产出综合评分，也不给评级：本系统尚无标定样本，不同指标的量纲无法折算为同一尺度。
依据：通过证据门的指标
本次会话没有指标通过证据门。
📑 行为指标观测明细
话语结构特征
证据不足
—— 本次未采集到足以评估该行为线索的有效样本。
逻辑关键词密度: 未采集到对应数据
面部专注度: 本次会话内无变化
视线稳定性: 有效样本不足
困惑微表情 (皱眉): 未采集到对应数据
情境行为稳定性
证据不足
—— 本次未采集到足以评估该行为线索的有效样本。
眉间收缩与唇部压缩: 有效样本不足
肢体抖动: 本次会话内无变化
视线偏差: 有效样本不足
面部对称性: 有效样本不足
沟通表达流畅度
证据不足
—— 本次未采集到足以评估该行为线索的有效样本。
语音流畅度: 未采集到对应数据
有效说话占比: 本次会话内无变化
语调变化: 有效样本不足
平均停顿时长: 该指标本轮停用
行为表现活跃度
证据不足
—— 本次未采集到足以评估该行为线索的有效样本。
手势自信分: 本次会话内无变化
肩部放松度: 本次会话内无变化
眼神接触比例: 有效样本不足
语音能量: 有效样本不足
言语流畅特征
证据不足
—— 本次未采集到足以评估该行为线索的有效样本。
眼部挤压 (费力): 未采集到对应数据
眨眼频率: 未采集到对应数据
回答详尽度: 未采集到对应数据
反应延迟: 未采集到对应数据
📊 五维证据覆盖
🔍 分维度证据链
🔍 话语结构特征 - 证据链
🔍 情境行为稳定性 - 证据链
🔍 沟通表达流畅度 - 证据链
🔍 行为表现活跃度 - 证据链
🔍 言语流畅特征 - 证据链
JingXin Multi-modal Assessment System | Auto-Generated Report
```

**读法**:旧的这一页会印「**—** / 综合行为观测评分 / 证据不足」并只给一句摘要;现在这一页
把"覆盖到哪、为什么没分、缺口各是什么"讲完:0/20 与置信度上限「无」是**覆盖事实**(§5.4 兜底),
20 条缺口逐条带维度归属(§5.4 的 evidence_gaps),没有任何档位词、没有任何分数、没有 `None`。
`📊 五维证据覆盖` 图在这种会话上是一张**空的覆盖图**(0/4 全在圆心)——诚实空图,不是伪造;
若要更醒目可在 M 系列决定是否干脆不画。

### 3.2 出分路径(`_MIXED`,1/20 槽过门 —— 真实数据从不走的那条,也是 C1 的案发现场)

```
JingXin 面试行为观测报告
🔬 JingXin 面试行为观测报告
基于多模态行为量的结构化观测
2026-09-24 11:42 | JingXin-Mapper-v11-EvidenceGate
📊 本次观测覆盖
1 / 20
个指标槽通过证据门
置信度上限：低
📝 综合总结
未产出综合评分，也不给评级：本系统尚无标定样本，不同指标的量纲无法折算为同一尺度。
依据：通过证据门的指标
话语结构特征 · 逻辑关键词密度：原始值 0.05；本场会话内观测区间 0.04 – 0.06（与原始值同量纲）；有效样本量 100
📑 行为指标观测明细
话语结构特征
过门指标 1/4；置信度：
低
。
（该维度目前无独立效标，仅供行为描述）
指标
原始值
本场会话内观测区间
有效样本量
逻辑关键词密度
0.05
0.04 – 0.06
100
未过门的指标：
面部专注度: 未采集到对应数据
视线稳定性: 未采集到对应数据
困惑微表情 (皱眉): 未采集到对应数据
情境行为稳定性
证据不足
—— 本次未采集到足以评估该行为线索的有效样本。
眉间收缩与唇部压缩: 未采集到对应数据
肢体抖动: 未采集到对应数据
视线偏差: 未采集到对应数据
面部对称性: 未采集到对应数据
沟通表达流畅度
证据不足
—— 本次未采集到足以评估该行为线索的有效样本。
语音流畅度: 未采集到对应数据
有效说话占比: 未采集到对应数据
语调变化: 未采集到对应数据
平均停顿时长: 未采集到对应数据
行为表现活跃度
证据不足
—— 本次未采集到足以评估该行为线索的有效样本。
手势自信分: 未采集到对应数据
肩部放松度: 未采集到对应数据
眼神接触比例: 未采集到对应数据
语音能量: 未采集到对应数据
言语流畅特征
证据不足
—— 本次未采集到足以评估该行为线索的有效样本。
眼部挤压 (费力): 未采集到对应数据
眨眼频率: 未采集到对应数据
回答详尽度: 未采集到对应数据
反应延迟: 未采集到对应数据
📊 五维证据覆盖
🔍 分维度证据链
🔍 话语结构特征 - 证据链
🔍 情境行为稳定性 - 证据链
🔍 沟通表达流畅度 - 证据链
🔍 行为表现活跃度 - 证据链
🔍 言语流畅特征 - 证据链
JingXin Multi-modal Assessment System | Auto-Generated Report
```

**读法(对照 §5.4/§5.6)**:同一个会话在旧渲染下印「1 / 综合行为观测评分 / **卓越**」
(低分版印 0.0 / 待提升);现在同一屏给出的是 —— 覆盖(1/20)、置信度上限(低)、
依据(哪一个指标过门)、**区间**(0.04–0.06,由本场均值 0.05 ± 会话内标准差 0.01 构成)、
**有效样本量**(100),以及那句"不出综合分、不给评级,因为量纲无法折算"。
维度块按 §5.4 的四项成形并带 §5.6 的「无独立效标,仅供行为描述」;归一值列已去除
(它是未标定标尺上的点分)。全篇无档位词、无百分位、无「优于 X%」、无 `None`。
`📊 五维证据覆盖` 图上「话语结构特征」轴 = 1,其余 = 0 —— 画的是**证据在哪**,不是**人怎么样**。

---

## 四、变更文件

| 文件 | 增/删 | sha256(提交后) |
|---|---|---|
| `report_frontend/evidence_gate.py` | +81 / −2 | `1d86f6b9be362179…` |
| `report_frontend/evidence_thresholds.json` | +82 / −1 | `f361c8d944b5f4ea…` |
| `report_frontend/report_generator.py` | +60 / −33 | `53001b4a834dbcde…` |
| `report_frontend/research_mapper.py` | +110 / −59 | `76764668e3179078…` |
| `report_frontend/visualizer.py` | +25 / −9 | `6f77e10e97ca9502…` |
| `templates/dashboard.html` | +2 / −2 | `4cc48255a9e48e19…` |
| `tests/test_report_layer.py` | +309 / −29 | `6a5198f8ca2e704a…` |

提交:`git -c user.name=… -c user.email=… commit`(精确路径,7 个),**未包含**任何 `.pyc`/
未跟踪产物 → 仓库剩余的已跟踪改动仍只有那 64 个被删的 `.pyc`(按派发,交使用者单独处理)。

---

## 五、自审发现 / 关切 / 我认为需要使用者或审查者定夺的地方

1. **雷达图我改了,这超出四个被点名的面(判断项,可否决)**。理由:C1 要求"报告不得把
   未标定标尺上的点分与档位当对人的评定渲染",而雷达图当时画的正是 5 个 `dim['score']`
   (0–100),轨迹名「候选人得分」—— 1/20 覆盖下会画出一根指到 100 的轴,读起来就是"这一维满分",
   等于把被删掉的东西换个地方画出来。改为覆盖图(0–4 槽)后,图上只剩"证据在哪"。
   如果使用者认为图表该整体撤掉或换别的形态,改 `_build_radar_figure` 一处即可。
   (同类但**未改**:`create_evidence_bar_chart` 画的是维度内各指标的**贡献值** —— 主体是
   指标之间的相对影响,不是对人的评定,也没有档位/人群位置,故留下;若要求更严可一并改。)
2. **`/api/report/structured`(根 `app.py:206-229`,本次范围外)仍把整个 payload 序列化给前端**,
   其中含 `total_score` / `total_level` / 每维 `level` / `normalized_score`。派发允许保留这些键
   (API 稳定),报告层也确实一个字都不渲染,但**任何基于该接口自建 UI 的人会重新拿到那套评级**。
   建议:要么使用者裁定把该接口的这几个字段标记为 deprecated(需要改根 `app.py`,本次未动),
   要么在 M 系列前明确"该接口不是报告面"。
3. **spec §8.2 第 2 条是对的、但结论写反了**(我认为审查/spec 在这里有误,附实测):
   原文说"`pause_duration` 不含 `ratio`,走的是 `pause` 分支,故劫持未能复现"。
   实测 `'ratio' in 'pause_duration'` **为真**(`pause_duration` 里的 `duration` 含 `ratio`),
   而旧实现的恒等族排在 pause 之前 → 它走的是**恒等**分支,0.5–3.0s 的分箱**从未执行**:
   `0.2/0.3/0.8/2.0/4.0/9.0 → 0.2/0.3/0.8/1.0/1.0/1.0`(用 `git show 298220f:…` 直接跑旧函数实测)。
   本轮**保持**该 legacy 归因(最长匹配 + 同长取先登记者),并用
   `test_pause_duration_is_shadowed_by_the_ratio_family` 把"pause 分箱目前不可达"钉住,
   在 `scale_factor_for` 的 docstring 与 JSON 里写明。要不要让它"转正"(pause 族优先)是
   设计决定 —— 它只影响 payload 的复合分与证据图,而该槽当前被 G4 封停,报告面看不到差别。
4. **顺带修了一处未被点名的静默失效**(属 C1 要求"区间"的必要条件):`_std` 此前只在键名以
   `_mean` 结尾时才去找,于是 ASR 派生键(如 `logic_keyword_density`)拿不到标准差 →
   G2 静默退回 fail-open、区间拿不到变异信息。现两种键名都试。方向是**更严**(常量列会真的被拦下),
   但这是行为变化,已单独写明并配 `test_observed_interval_is_session_derived`。
5. **`total_score`/`total_level`/`level`/`normalized_score`/`status`/`positive_factors`/
   `negative_factors` 仍在 payload 里且无渲染消费方**。按派发保留(API 稳定),但它们是"结论形状"
   的残余字段(其中 `positive_factors`/`negative_factors`/`status` 本计划早已判为 deferred)。
   建议 M 系列清一次:要么删,要么在接口层显式标注"未标定、不得对外展示"。
6. **`~/shared/jingxin_gate4/` 现在是 1.3 GB / 271 个文件**:这是派发给的拷贝命令
   (`cp data/output/evidence_*.html data/output/radar_chart_*.html`)把 `data/output` 里**历次**
   累积的图表全扫进来了(两份报告真正引用的只有 12 张:6 + 6,都是 `*_1790221337.html` /
   `*_1790221338.html`)。我在第二次生成时改为只拷被引用的那 12 张,但**没有删除**第一次多拷进去的
   历史文件 —— 删除操作在共享目录上属于范围外动作(权限分类器也拦下了),请使用者自行清理:
   除 `jingxin_gate4_report.html`(`177992c13d4dceb0…`)、`jingxin_gate4_report_scored.html`
   与上述 12 张图表外,其余都可删。
7. **未处置的既有 deferred(与本波无关,未动)**:`np`/`json`/`px`/`List`/`Tuple` 等未使用 import;
   `_threshold_for`/`scale_factor_for` 每次调用重读 JSON;`result["evidence_gaps"]` 无消费方;
   `_render_dimension_block` 的未用参数 `dim_key`;`templates/dashboard.html` 文件本身是个
   嵌套 HTML 的坏结构(内容整段塞在 `<title>` 里)**且未含 `.pyc`/`.gitattributes` 卫生项** ——
   按派发,这些交使用者或最终审查分诊。
   唯一一处**顺手但可见**的改动:`templates/dashboard.html` 此前在磁盘上是 CRLF(HEAD blob 是 LF),
   我编辑的两行是 LF → 工作区变成混合行尾。已把该文件整体归一为 LF —— 因为 `.gitattributes`
   的 `text=auto eol=lf` 会让 git 双向归一化,归一前后 `git diff --numstat` 恒为 `2 2`
   (只有我改的那两行),故**没有引入任何额外汇总行改动**;提交内 blob 已核实为 LF。
   该文件只剩一处两行改动,内容零漂移。
8. **合规边界(提醒,不在本波)**:本波改的全部是呈现层,没有任何事项改变 spec §1.1 的许可结论
   或 §1.3 的伦理审查前置;报告面现在不含人群位置,但这不等于"可以上线到招聘流程"——
   那仍由 §1.1/§1.3/§6.3 决定。

---

## 六、结论

- C1 / I1 / I2 / I3 / I4 全部落地,每条都有**可证伪**的守护测试(14 个探针在提交状态上逐个 RED,
  还原后 sha256 逐字节一致)。
- 报告现在渲染的是:覆盖事实 + 置信度上限 + 逐指标**本场实测区间** + 值/有效样本量/缺口 +
  「无独立效标,仅供行为描述」;点分与五档评语在报告与控制台都不再出现。
- 51 项测试全绿;交付物两条路径都已重新生成并放在 `~/shared/jingxin_gate4/`。
