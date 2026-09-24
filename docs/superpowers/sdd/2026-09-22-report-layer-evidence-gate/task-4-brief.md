## Task 4: 删死分支与恢复 question_index

**Files:**
- Modify: `report_frontend/feature_engine.py`
- Modify: `tests/test_report_layer.py`

**Interfaces:**
- Consumes: 无
- Produces: `feature_engine` 不再产出 `fluency_score`/`fluency_proxy`(死分支);`question_index` 作为协变量保留

- [ ] **Step 1: 写失败的测试**

```python
def test_question_index_is_not_dropped():
    """spec §8-3:question_index 曾被 'index' 跳过规则静默丢弃。"""
    import pandas as pd

    from report_frontend.feature_engine import PsychologicalFeatureEngine

    df = pd.DataFrame({"question_index": [0, 1, 2], "pitch_mean": [100.0, 120.0, 110.0]})
    feats = PsychologicalFeatureEngine({"voice_research": df}).extract_all_features()
    keys = " ".join(feats.get("voice_research", {}).keys())
    assert "question_index" in keys, "question_index 仍被丢弃"


def test_dead_fluency_branch_removed():
    """spec §5.3:该分支要求列名同时含 speech_ratio 与 mean,故为死代码。"""
    import pandas as pd

    from report_frontend.feature_engine import PsychologicalFeatureEngine

    df = pd.DataFrame({"speech_ratio": [0.9, 0.8, 0.95]})
    feats = PsychologicalFeatureEngine({"voice_research": df}).extract_all_features()
    keys = " ".join(feats.get("voice_research", {}).keys())
    assert "fluency_score" not in keys
    assert "fluency_proxy" not in keys
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k "question_index or dead_fluency" -v`
Expected: FAIL

- [ ] **Step 3: 修跳过规则(`feature_engine.py:117`)**

```python
# 改前
if any(x in col.lower() for x in ['id', 'index', 'unnamed', 'timestamp']):
    continue

# 改后:放行 question_index 作为协变量
_SKIP_COLS = ('id', 'unnamed', 'timestamp')

if any(x in col.lower() for x in _SKIP_COLS):
    continue
if 'index' in col.lower() and 'question' not in col.lower():
    continue
```

- [ ] **Step 4: 删死分支(`feature_engine.py:289-293`)**

删除 `if 'speech_ratio' in col_lower and 'mean' in col_lower:` 整个分支。

- [ ] **Step 5: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add report_frontend/feature_engine.py tests/test_report_layer.py
git commit -m "fix: 保留 question_index 协变量,删除 fluency 死分支"
```

---

# 提交 B:内容(叙事层 + 常模 + 措辞)

