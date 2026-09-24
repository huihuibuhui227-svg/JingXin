### Task 6: 报告侧同步(删两份定义 + 改名 + 量程)

**Files:**
- Modify: `report_frontend/feature_engine.py:310-346`、`report_frontend/research_mapper.py:65`、`report_frontend/evidence_thresholds.json:52-57`
- Test: `tests/test_feature_key_rename.py`

**Interfaces:**
- Consumes: 语音日志列 `connective_density` / `connective_density_std`(Task 3)
- Produces: 特征键 `voice_research_connective_density_mean` / `_std`;映射元组关键字 `connective_density`,显示名「连接词密度」

- [ ] **Step 1: 写失败测试**

```python
# tests/test_feature_key_rename.py
from pathlib import Path

from report_frontend.research_mapper import ResearchCapabilityMapper

ROOT = Path(__file__).resolve().parent.parent


def test_mapper_indicator_is_renamed_and_matches_new_key():
    rule = ResearchCapabilityMapper().mapping_rules["logical_thinking"]
    keywords = [ind[0] for ind in rule["indicators"]]
    names = [ind[3] for ind in rule["indicators"]]
    assert "connective_density" in keywords
    assert "logic_keyword_density" not in keywords
    assert "连接词密度" in names


def test_feature_engine_has_no_second_definition_of_the_metric():
    """同名指标只能有一处定义 —— feature_engine 里的两张关键词表必须删净。"""
    text = (ROOT / "report_frontend/feature_engine.py").read_text(encoding="utf-8")
    assert "logic_keyword" not in text
    assert "logic_keywords" not in text


def test_new_key_can_pass_the_gate_and_score():
    feats = {"voice_research": {"connective_density_mean": 3.5,
                                "connective_density_std": 0.4,
                                "_n_rows": 4.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]
    assert dim["evidence_chain"], "证据链为空 —— 新键没被映射到"
    assert any("连接词密度" in ev.get("human_name", "") or "连接词密度" in str(ev) 
               for ev in dim["evidence_chain"])
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_feature_key_rename.py -q`
Expected: FAIL(`logic_keyword` 仍在 feature_engine;映射元组仍是旧关键字)

- [ ] **Step 3: 三处修改**

`feature_engine.py`:**整段删除** `:310-346` 的文本扫描与两份关键词表(以及 `:509` 的自检行),让新列走通用数值路径(`_extract_numeric_stats` 自动产出 `_mean`/`_std`)。

`research_mapper.py:65`:
```python
                ("logic_keyword_density", 0.4, True, "逻辑关键词密度", "core"),
→               ("connective_density", 0.4, True, "连接词密度", "core"),
```

`evidence_thresholds.json` 的 `density` 族:量程随新定义更新(旧 `full_scale: 0.02` 对应"命中数÷字符数";新定义是每百字,典型 0~10):
```json
    "density": {
      "kind": "full_scale",
      "full_scale": 10.0,
      "basis_kind": "definitional",
      "basis": "每百字 10 个连接词记为满量程。旧值 0.02 对应'命中数÷字符数',与新的每百字定义相差约 100 倍,故必须改。M5 标定到位后替换。",
      "_provisional": true
    },
```
并把文件顶部 `_version` 提升一位。

- [ ] **Step 4: 运行,确认通过 + 全套测试 + 禁止词扫描仍绿**

```bash
~/miniconda3/envs/jingxin/bin/python -m pytest -q                 # 期望全绿
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k banned -q
```

- [ ] **Step 5: 提交**

```bash
git add report_frontend/feature_engine.py report_frontend/research_mapper.py report_frontend/evidence_thresholds.json tests/test_feature_key_rename.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "refactor(m1): 连接词密度只留一处定义;映射与折算量程随新定义更新"
```

---

