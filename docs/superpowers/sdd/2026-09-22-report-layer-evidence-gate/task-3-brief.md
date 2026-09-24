## Task 3: research_mapper 接入证据门

**Files:**
- Modify: `report_frontend/research_mapper.py`
- Create: `tests/test_report_layer.py`

**Interfaces:**
- Consumes: Task 2 的 `gate`, `confidence_from`, `Check`
- Produces:
  - `map_features_to_scores(features)` 返回的每个 dimension 增加 `evidence_gaps: list[str]`;无有效证据时 `score is None`、`confidence == "无"`
  - 顶层返回增加 `total_score`(可能为 `None`)、`evidence_gaps: list[str]`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_report_layer.py
from report_frontend.research_mapper import ResearchCapabilityMapper


def test_zero_input_produces_no_scores():
    """spec §6.1:零输入必须不再产出分数。基线是 47.08。"""
    r = ResearchCapabilityMapper().map_features_to_scores({})

    assert r["total_score"] is None
    for key, dim in r["dimensions"].items():
        assert dim["score"] is None, f"{key} 零输入却仍出分"
        assert dim["confidence"] == "无", f"{key} 置信度应为无"
    assert r["evidence_gaps"], "必须给出 evidence_gaps"


def test_no_fake_resume_fallback():
    """spec §5.3:假简历兜底必须删除。"""
    r = ResearchCapabilityMapper().map_features_to_scores({})
    for dim in r["dimensions"].values():
        for ev in dim["evidence_chain"]:
            assert ev["feature"] != "logic_keyword_density", "假简历兜底仍在"
            assert "BASELINE" not in ev["feature"], "BASELINE_FILL 仍可达"
            assert "代理" not in ev["feature"], "硬编码代理仍在"


def test_quarantined_columns_are_rejected():
    """spec §5.2:封停列即使有值也不得进入证据链。"""
    feats = {"face": {"face_focus_score_mean": 0.3,
                      "face_symmetry_score_mean": 0.98}}
    r = ResearchCapabilityMapper().map_features_to_scores(feats)
    for dim in r["dimensions"].values():
        for ev in dim["evidence_chain"]:
            assert "focus_score" not in ev["feature"]
            assert "symmetry_score" not in ev["feature"]


def test_confidence_can_be_none_and_low():
    """spec §6.5:置信度必须能取到'无'与'低'。"""
    seen = set()
    seen.add(ResearchCapabilityMapper().map_features_to_scores({})
             ["dimensions"]["logical_thinking"]["confidence"])
    feats = {"voice_research": {"interview_pause_duration_mean": 0.8}}
    r = ResearchCapabilityMapper().map_features_to_scores(feats)
    for dim in r["dimensions"].values():
        seen.add(dim["confidence"])
    assert "无" in seen


def test_dimension_weights_sum_to_one():
    """spec §5.3:comm 维度权重和曾为 1.4。"""
    m = ResearchCapabilityMapper()
    for dim_key, rule in m.mapping_rules.items():
        total = sum(w for _, w, _, _, _ in rule["indicators"])
        assert abs(total - 1.0) < 1e-9, f"{dim_key} 权重和为 {total}"
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -v`
Expected: FAIL(零输入仍返回 47.08、假简历兜底仍在、comm 权重和 1.4)

- [ ] **Step 3: 删除假简历兜底数据与 baselines(两处必须同时删)**

删除 `research_mapper.py:19-38` 的 `self.demo_text_data`,以及 `:40-47` 的 `self.baselines`。`__init__` 仅保留 `self.mapping_rules` 与 `self.dimension_weights`。

⚠️ **删 `self.baselines` 的同时必须删掉它在 `map_features_to_scores` 里的唯一使用点**,否则运行即 `AttributeError`。删除这段(原 `:234-243`):

```python
bl = self.baselines.get(keyword, {})
percentile = None
if bl:
    std_dev = bl["std"] if bl["std"] > 0 else 0.001
    z = (matched_val - bl["mean"]) / std_dev
    try:
        percentile = int(0.5 * (1 + math.erf(z / math.sqrt(2))) * 100)
    except:
        percentile = 50
    inference_data[keyword] = {"val": matched_val, "percentile": percentile}
```

并把 `evidence_item` 里的 `"percentile": percentile` 一并删除。此后 `inference_data` 只保留 `{"val": ...}`(供调试),不再含百分位。

- [ ] **Step 4: 修正 `communication_fluency` 的权重和**

```python
"communication_fluency": {
    ...
    "indicators": [
        ("fluency_score", 0.4, True, "语音流畅度", "core"),
        ("fluency_proxy", 0.4, True, "流畅度代理指标", "core"),
        ("speech_ratio", 0.3, True, "有效说话占比", "core"),
        ("pitch_variation", 0.2, True, "语调变化", "support"),
        ("pause_duration", 0.1, False, "平均停顿时长", "support"),
    ]
},
```

改为(总权重 1.0;`fluency_proxy` 是永久封停项,直接删除):

```python
"communication_fluency": {
    ...
    "indicators": [
        ("fluency_score", 0.4, True, "语音流畅度", "core"),
        ("speech_ratio", 0.3, True, "有效说话占比", "core"),
        ("pitch_variation", 0.2, True, "语调变化", "support"),
        ("pause_duration", 0.1, False, "平均停顿时长", "support"),
    ]
},
```

- [ ] **Step 5: 前置 —— feature_engine 产出每个模态的有效行数**

G3 需要一个真实的 `n_valid`。报告层拿到的全是聚合标量,**没有 `n_valid_frames`**(那是 M4 的 L1 才有的)。当前唯一真实可用的量是**每个模态源 DataFrame 的行数** —— 它决定了一个均值是否有意义。

在 `feature_engine.py` 的 `extract_all_features` 里,每个模态结束处加一行,产出 `<模态>__n_rows`:

```python
# extract_all_features(),每个模态提取完之后
self.features['face']['__n_rows'] = float(len(self.data['face']))
self.features['gesture']['__n_rows'] = float(len(self.data['gesture']))
for key in ('voice_interview', 'voice_research'):
    if key in self.data:
        self.features[key]['__n_rows'] = float(len(self.data[key]))
```

⚠️ `__n_rows` 以双下划线开头,`_normalize_keys` 会保留它;它**不参与指标匹配**(所有 `mapping_rules` 关键词都不含 `n_rows`),只被 G3 读取。

- [ ] **Step 6: 删除步骤 2/3/4 的兜底逻辑,改为证据门**

把 `map_features_to_scores` 里第 2 步(假简历 fallback)、第 3 步(智能降级三条硬编码代理)、第 4 步(兜底 BASELINE_FILL)整段删除,替换为:

```python
from .evidence_gate import confidence_from, gate, user_message

# ...(在方法内,替换原第 2~4 步)

# 顶层缺口累加器(方法开始处初始化一次,供返回值的 evidence_gaps 用)
all_evidence_gaps = []

# ...进维度循环后,每个维度内:
matched = []          # [(keyword, weight, is_positive, human_name, raw_value, matched_key)]
dim_gaps = []         # 本维度的缺口,二者都要声明在维度循环内

for keyword, weight, is_positive, human_name, importance in indicators:
    found_key, found_val = self._fuzzy_match(keyword, all_features)
    if found_key is None:
        dim_gaps.append(f"{human_name}: 未采集到对应数据")
        continue

    # 伴随的 _std 用于 G2(常量判定);模态行数用于 G3(样本量)
    std_key = (found_key[:-len("_mean")] + "_std") if found_key.endswith("_mean") else None
    std = all_features.get(std_key) if std_key else None
    modality = found_key.split("_", 1)[0]
    n_valid = int(all_features.get(f"{modality}__n_rows", 0))

    chk = gate(found_key, found_val, n_valid=n_valid, std=std)
    if not chk.ok:
        # ⚠️ 只准用 user_message。chk.reason 含维护者文案(封停理由等),
        # 而 dim_gaps 会被 Task 6 渲染进报告的"证据缺口"一节 —— 直接用会外泄内部信息。
        dim_gaps.append(f"{human_name}: {user_message(chk)}")
        continue
    matched.append((keyword, weight, is_positive, human_name, found_val, found_key))

# 维度循环末尾,把本维缺口并入顶层:
all_evidence_gaps.extend(dim_gaps)
```

⚠️ **作用域是这里的关键**:`matched` / `dim_gaps` 必须在**维度循环内**声明(原代码的 `evidence_chain = []` 就在这个位置),`all_evidence_gaps` 在**方法开头**声明一次。写错作用域会让所有维度共用一个缺口列表。

其中 `_fuzzy_match(keyword, all_features) -> (key, value) | (None, None)` 抽出原第 1/2 步的匹配逻辑(只保留精确匹配与 `_mean/_std/_sum` 宽松匹配,**不含降级与兜底**)。

- [ ] **Step 7: 分数只在有证据时产出**

```python
if not matched:
    dimension_results[dim_key] = {
        "display_name": rule_config["name"],
        "score": None,
        "level": "证据不足",
        "narrative": "本次未采集到足以评估该行为线索的有效样本。",
        "evidence_chain": [],
        "evidence_gaps": dim_gaps,
        "confidence": "无",
        "matched_indicators": f"0/{len(indicators)}",
        "stats": {},
    }
    continue
```

有证据时:

```python
confidence = confidence_from(len(matched), len(indicators))
score = round(max(0.0, min(100.0, weighted_sum / total_weight * 100)), 2)
```

- [ ] **Step 8: 汇总总分(可能为 None)**

```python
valid = [d for d in dimension_results.values() if d["score"] is not None]
if not valid:
    final_total = None
else:
    num = sum(d["score"] * self.dimension_weights[k]
              for k, d in dimension_results.items() if d["score"] is not None)
    den = sum(self.dimension_weights[k]
              for k, d in dimension_results.items() if d["score"] is not None)
    final_total = round(num / den, 2)
```

- [ ] **Step 9: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py tests/test_evidence_gate.py -v`
Expected: 全部 PASS

- [ ] **Step 10: 真实会话回归**

```bash
~/miniconda3/envs/jingxin/bin/python -c "
import sys; sys.path.insert(0, '.')
from report_frontend import LogDataLoader, PsychologicalFeatureEngine, ResearchCapabilityMapper
d = LogDataLoader().get_fused_latest_data()
f = PsychologicalFeatureEngine(d).extract_all_features()
r = ResearchCapabilityMapper().map_features_to_scores(f)
print('总分:', r['total_score'])
for k, v in r['dimensions'].items():
    print(f\"  {k}: score={v['score']} conf={v['confidence']}\")
print('gaps:', r['evidence_gaps'][:5])
"
```

Expected:**不再出现 `conf=高`**(spec §6.2 基线是 4/5 维为"高")。

- [ ] **Step 11: 提交**

```bash
git add report_frontend/research_mapper.py tests/test_report_layer.py
git commit -m "feat: research_mapper 接入证据门,删除兜底与硬编码代理"
```

---

