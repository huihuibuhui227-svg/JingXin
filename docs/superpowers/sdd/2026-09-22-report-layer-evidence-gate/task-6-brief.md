## Task 6: 重写硬编码叙事层

**Files:**
- Modify: `report_frontend/report_generator.py:123-250`
- Modify: `tests/test_report_layer.py`

**Interfaces:**
- Consumes: Task 3 的 `score=None` / `confidence="无"` / `evidence_gaps`
- Produces: `_generate_deep_text_analysis` 只输出真实数值 + 证据状态,不含任何解读

- [ ] **Step 0: 修复 Task 3 遗留的两条空断言(同一测试文件)**

**背景**:控制器对 `tests/test_report_layer.py` 做了一次**系统扫描**(`sys.settrace` 记录每个测试实际执行的行,与 AST 里的 `assert` 行求差),结果:**12 个测试里恰好 2 个的断言行从未执行**。这两条正好守着 Task 3 的核心主张 —— 意味着"假简历兜底回归""封停列重新进链"都不会被任何测试拦住。

扫描结果(其余 10 个测试的断言全部执行):

| 测试 | 断言行 | 从未执行 |
|---|---|---|
| `test_no_fake_resume_fallback` | 3 | **3** |
| `test_quarantined_columns_are_rejected` | 2 | **2** |

两处根因相同:fixture 让证据链为空 → 遍历 `evidence_chain` 的循环**零次执行**。

把 `tests/test_report_layer.py` 里这两个函数整体替换为:

```python
def test_no_fake_resume_fallback():
    """spec §5.3:假简历兜底与 BASELINE_FILL 必须不可达。

    ⚠️ 原版用零输入 fixture,证据链为空 → 循环零断言,对任何实现都通过。
    改为喂一个只让 1 个槽过门的输入,断言 matched 恰为 1/4 且缺口为 3。
    若 BASELINE_FILL 回归,缺失的 3 个槽会被填空 → matched 变 4/4 → 本测试变红。
    控制器已实测该 fixture 输出:score=100.0 / conf=低 / matched=1/4 / 缺口 3。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]

    assert dim["matched_indicators"] == "1/4", "缺失槽被填充(BASELINE_FILL 回归?)"
    assert len(dim["evidence_gaps"]) == 3
    for ev in dim["evidence_chain"]:
        # 裸键名是假简历兜底的签名;带模态前缀才是真测量
        assert ev["feature"] != "logic_keyword_density", "裸键名 = 假简历兜底的签名"
        assert "BASELINE" not in ev["feature"]
        assert "代理" not in ev["feature"]


def test_quarantined_columns_are_rejected():
    """spec §5.2:封停列即使有值也不得进入证据链。

    ⚠️ 原版只喂封停列 → 全部被拒 → 链为空 → 循环零断言。
    改为同时喂一个干净列让链非空,再断言封停列不在其中。
    若 G4 被移除,focus_score 会进链 → 本测试变红。
    控制器已实测:链为 [voice_research_logic_keyword_density],focus_score 不在其中。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0},
             "face": {"face_focus_score_mean": 0.3,
                      "face_focus_score_std": 0.05,
                      "_n_rows": 100.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]

    assert dim["evidence_chain"], "证据链为空 —— 本测试退化为空断言"
    for ev in dim["evidence_chain"]:
        assert "focus_score" not in ev["feature"]
        assert "symmetry_score" not in ev["feature"]
```

**验收要求**:改完后**重跑一遍上述扫描**(同一段 `sys.settrace` 脚本),断言"存在未执行断言的测试数 = 0"。这是本轮唯一能证明修复到位的办法 —— 光看代码看不出循环跑没跑。

- [ ] **Step 1: 写失败的测试**

```python
import re

BANNED = ["焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
          "录用", "人格", "心理画像", "常模"]


def test_deep_analysis_has_no_banned_words():
    """spec §5.4 + §5.6:叙事层不得含情绪/心理/诚信构念。

    ⚠️ 只覆盖**叙事层自己写的句子**。报告里出现的禁止词有另一个来源:
    `research_mapper` 提供的两个标签 —— `display_name`「抗压与情绪稳定性」
    (含 抗压、情绪稳定)与 `human_name`「面部紧张度」(含 紧张)。
    实测确认全 mapper 只有这两处命中。

    它们属 **Task 7** 的改名范围(Task 7 改完 `display_name` 与 `human_name`
    后,**须恢复全量扫描** —— 见计划 Task 7 Step 3b)。
    本测试用剔除这两个标签的方式,把叙事层自己的输出隔离出来测。
    剔除是精确字符串替换,所以叙事层**自己在别处**写出的禁止词仍会被抓到。
    """
    from report_frontend.report_generator import ReportGenerator

    feats = {"face": {"face_tension_score_mean": 0.5},
             "gesture": {"gesture_left_hand_jitter_mean": 0.02}}
    result = ResearchCapabilityMapper().map_features_to_scores(feats)
    html = ReportGenerator()._generate_deep_text_analysis(feats, result)

    # 剔除 mapper 提供的标签(Task 7 修复后本段移除,恢复全量)
    for label in ("抗压与情绪稳定性", "面部紧张度"):
        html = html.replace(label, "")

    for word in BANNED:
        assert word not in html, f"叙事层出现禁止词:{word}"


def test_deep_analysis_handles_none_scores():
    """零证据时不得崩溃(score 为 None,不能参与排序)。"""
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores({})
    html = ReportGenerator()._generate_deep_text_analysis({}, result)
    assert "证据不足" in html


def test_no_hardcoded_gaze_claim():
    """spec §5.4:那句'未出现异常的回避行为'是纯硬编码。"""
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores({})
    html = ReportGenerator()._generate_deep_text_analysis({}, result)
    assert "未出现异常的回避行为" not in html
    assert "如外科医生般" not in html
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k "banned or none_scores or hardcoded_gaze" -v`
Expected: FAIL

- [ ] **Step 3: 重写 `_generate_deep_text_analysis`**

⚠️ **这四处必须一并清掉**(Task 3 复审发现,均在本函数内;Task 3b 的短接让它们在今天不可达,但重写时必须真正删除,不能只是被跳过):

| 位置 | 问题 |
|---|---|
| 原 `:165,176,194,215` | 四处 `['percentile']` 读的是已删除的字段,默认值 `50` → 每场会话都印「优于 **50%** 的受试者」「处于 **50%** 的水平」「超越了 **50%** 的人群」—— **四条恒定在 50 的杜撰常模**,比改动前更假(Task 3 之前至少还是算出来的) |
| 原 `:137` | 查 `face_micro_exp_au_name_au4_freq`,而实际产出的键是 `face_micro_exp_micro_exp_au_name_au4_freq` → 永远取默认 0 → `:178-181` 那句「微表情监测**未检测到显著的焦虑特征**」**无条件打印**(spec §5.4 点名必删) |
| 原 `:129,174` | `face_feats.get(..., 0)` 默认值驱动「展现了极佳的**情绪控制力**和**心理稳定性**」 |
| 原 `:157` | 「综合科研潜力评分为 X 分」+「毫秒级量化分析」+「常模参照模型」 |

⚠️ **必须保留 Task 3b 建立的两条性质**,否则重写会打破现已通过的测试:

1. **全无证据时不得进入任何基于默认值的段落。** 现有测试 `test_zero_evidence_report_makes_no_claims` 钉住这条(零证据下不得出现「科研天赋」「心理素质」「最为突出」)。重写后的函数在 `scored` 为空时仍须直接返回诚实的证据缺口摘要。
2. **分数不得渲染为 `None`。** `_build_html_report` 里的 `_score_display` 兜底须保留。

这两条是 Task 3b 的产物,`grep -n "_score_display\|scored = "` 可定位现状。

整段(原 `:123-250`)替换为只陈述测量事实的版本。每个维度输出:

```python
def _render_dimension_block(self, dim_key: str, dim: Dict[str, Any]) -> str:
    """渲染单个维度的证据状态。不解读,不推断,不加形容词。"""
    if dim["score"] is None:
        gaps = "".join(f"<li>{g}</li>" for g in dim.get("evidence_gaps", []))
        return f"""
        <h3>{dim['display_name']}</h3>
        <p><strong>证据不足</strong> —— 本次未采集到足以评估该行为线索的有效样本。</p>
        <ul>{gaps}</ul>
        """

    rows = "".join(
        f"<tr><td>{e['human_name']}</td><td>{e['raw_value']}</td>"
        f"<td>{e['normalized_score']}</td><td>{e['weight']}</td></tr>"
        for e in dim["evidence_chain"]
    )
    return f"""
    <h3>{dim['display_name']}</h3>
    <p>依据 {dim['matched_indicators']} 个指标；置信度：<strong>{dim['confidence']}</strong>。</p>
    <table><thead><tr><th>指标</th><th>原始值</th><th>归一值</th><th>权重</th></tr></thead>
    <tbody>{rows}</tbody></table>
    """
```

顶层函数:

```python
def _generate_deep_text_analysis(self, features, result) -> str:
    parts = []
    total = result["total_score"]
    if total is None:
        parts.append("<p>本次会话未采集到足以支撑评估的有效证据。</p>")
    else:
        parts.append(
            f"<p>综合行为观测摘要：{result['total_level']}"
            f"(置信度上限：{max((d['confidence'] for d in result['dimensions'].values()), key=_CONF_ORDER)})</p>"
        )
    for dim_key, dim in result["dimensions"].items():
        parts.append(self._render_dimension_block(dim_key, dim))
    if result.get("evidence_gaps"):
        parts.append("<h3>证据缺口</h3><ul>"
                     + "".join(f"<li>{g}</li>" for g in result["evidence_gaps"])
                     + "</ul>")
    return "".join(parts)
```

并在模块顶部:

```python
_CONF_ORDER = {"无": 0, "低": 1, "中": 2, "高": 3}
```

- [ ] **Step 4: 同步修 `_build_html_report`**

`report_generator.py:310` 的 `{result['total_score']}` 与 `:312` 的 `{result['total_level']}` 在零证据时须显示"证据不足",不能显示 `None`:

```python
_score_display = result["total_score"] if result["total_score"] is not None else "—"
```

- [ ] **Step 5: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 6: 端到端验证**

```bash
~/miniconda3/envs/jingxin/bin/python -c "
import sys; sys.path.insert(0, '.')
from report_frontend import ReportGenerator
p = ReportGenerator(output_dir='/tmp/jxreport').generate_report()
print('报告:', p)
" 2>&1 | tail -5
```

然后人工打开生成的 HTML,确认:
- 出现"证据不足"
- 无"情绪状态与抗压能力深度剖析"整节
- 无"未检测到焦虑特征"
- 无"如外科医生般"

- [ ] **Step 7: 提交**

```bash
git add report_frontend/report_generator.py tests/test_report_layer.py
git commit -m "refactor: 重写报告叙事层,删除硬编码解读"
```

---

- [ ] **Step 8: 审查修复轮(2 Important + 2 顺手项)**

审查判定 **Needs fixes**。两条 Important **都是同一类**:新测试用了**零证据 fixture**,而禁止词与硬编码句住在**出分路径**上 —— 于是两条测试在改前改后都通过,守不住它们声称要守的东西。(这是本计划同类失效的**第 7、8 次**。)

**统一修法:把这三条测试的 fixture 换成"混合 fixture",让它走通出分路径。** 控制器已实测该 fixture 会使 `logical_thinking` 出分(即进入被删的那些段落),而当前实现下硬编码句与禁止词命中均为**空** —— 所以修完后测试有约束力:

```python
_MIXED = {"voice_research": {"logic_keyword_density": 0.05,
                             "logic_keyword_density_std": 0.01,
                             "_n_rows": 100.0}}
```

**(a) `test_no_hardcoded_gaze_claim`** —— 把 `ResearchCapabilityMapper().map_features_to_scores({})` 改为 `map_features_to_scores(_MIXED)`,`_generate_deep_text_analysis({}, result)` 保持。原版因零证据短接,永远到不了那三句。

**(b) `test_deep_analysis_has_no_banned_words`** —— 同样换成 `_MIXED`。否则它只扫"无证据"那几句,而禁止词原本住在出分路径。

**(c) `test_deep_analysis_handles_none_scores`** —— 换成 `_MIXED`,使其 docstring 声称的"`score` 为 `None` 不能参与排序"真的被走到(零证据输入根本到不了 `max(..., key=_CONF_ORDER.get)`)。

**(d) `test_quarantined_columns_are_rejected` 的 `symmetry_score` 断言结构性不可证伪** —— `symmetry_score` 是 `stress_resilience` 的槽,而该测试只查 `logical_thinking`;且新 fixture 根本没喂对称性列。`grep -rn symmetry tests/` 显示这一行是它唯一出现处,即它**哪儿都没被钉住**。

⚠️ 注意:补 `stress_resilience` 的链断言**无效** —— 该维四个槽(`tension_score`/`jitter`/`gaze_deviation`/`symmetry_score`)**全部封停**,链必为空,断言仍然恒真。**正确修法是断言"进不了链、且以缺口形式出现"**:

```python
def test_quarantined_columns_are_rejected():
    """spec §5.2:封停列即使有值也不得进入证据链。

    ⚠️ 原版只喂封停列 → 全部被拒 → 链为空 → 循环零断言。
    改为同时喂一个干净列让链非空,再断言封停列不在其中。
    若 G4 被移除,focus_score 会进链 → 本测试变红。

    symmetry_score 属 stress_resilience 维度,而该维四个槽全部封停、链必为空,
    故不能靠"链里没有它"来钉(那恒真)。改为断它**出现在证据缺口里** ——
    若删掉它的封停条目且喂入该列,它会进链、缺口里便不再有它 → 本测试变红。
    控制器实测:stress_resilience 链=0、缺口=4、缺口含「面部对称性」。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0},
             "face": {"face_focus_score_mean": 0.3,
                      "face_focus_score_std": 0.05,
                      "face_symmetry_score_mean": 0.98,
                      "face_symmetry_score_std": 0.02,
                      "_n_rows": 100.0}}
    dims = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]

    lt = dims["logical_thinking"]
    assert lt["evidence_chain"], "证据链为空 —— 本测试退化为空断言"
    for ev in lt["evidence_chain"]:
        assert "focus_score" not in ev["feature"]

    sr = dims["stress_resilience"]
    assert sr["evidence_chain"] == [], "封停列进了链"
    assert any("对称" in g for g in sr["evidence_gaps"]), "对称性槽未被处理"
```

**(e) 缺口重复渲染**(Minor,顺手) —— `_render_dimension_block` 的逐维 `<ul>` 与顶层「证据缺口」段打印的是同一批字符串(`all_evidence_gaps` 就是各维缺口的并集),实测 19 条缺口被渲染两遍。**出分路径下删掉顶层那一段**(保留逐维列表,信息更全);**短接路径仍需要它**(那时不渲染任何维度)。

**(f) 删 `_get_percentile_badge`**(Minor) —— 本任务删掉了它唯一的调用方,grep 显示只剩定义。spec §5.5 要求不留任何百分位机器。**顺手删掉。**

**验收**(修复后必须全部做到):

1. `pytest -q` 全绿
2. 对 (a)(b)(c) 三条,证明"若把对应改动回退,测试会失败" —— 本计划已栽过八次空断言
3. 重跑空断言扫描(§Step 0 的脚本),断言"存在未执行断言的测试数 = 0"
4. 生成一份报告,确认缺口不再重复出现

---

