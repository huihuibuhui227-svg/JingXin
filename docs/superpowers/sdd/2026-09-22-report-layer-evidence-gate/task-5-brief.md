## Task 5: 删杜撰常模与百分位

**Files:**
- Modify: `report_frontend/visualizer.py`
- Modify: `report_frontend/research_mapper.py`
- Modify: `tests/test_report_layer.py`

**Interfaces:**
- Consumes: 无
- Produces: 雷达图无"常模基准"trace;证据链无 `percentile` 字段

- [ ] **Step 1: 写失败的测试**

```python
def test_no_fabricated_percentile():
    """spec §5.5:百分位只能来自真实常模。

    ⚠️ 必须喂**能过证据门**的输入。否则 evidence_chain 全为空、嵌套循环零断言,
    本测试对**任何**实现都通过 —— 包括把百分位加回来的实现。
    Task 5 复审实测:原输入 `interview_pause_duration_mean` 让 5 个维度全部
    score=None / chain=[],是彻底的空断言("回归守卫"的标签夸大了它)。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0}}
    r = ResearchCapabilityMapper().map_features_to_scores(feats)

    total = sum(len(d["evidence_chain"]) for d in r["dimensions"].values())
    assert total > 0, "输入未过证据门,本测试退化为空断言"
    for dim in r["dimensions"].values():
        for ev in dim["evidence_chain"]:
            assert "percentile" not in ev or ev["percentile"] is None


def test_radar_has_no_norm_baseline():
    """spec §5.5:[60]*5 与 '常模基准' 图例必须删除。"""
    from report_frontend.visualizer import ReportVisualizer

    result = ResearchCapabilityMapper().map_features_to_scores({})
    fig = ReportVisualizer(output_dir="/tmp")._build_radar_figure(result)
    names = [t.name for t in fig.data]
    assert "常模基准" not in names
    # 正向断言:否则"一张轨迹都没有的图"也会通过,候选分轨迹的存续无人钉住
    assert names == ["候选人得分"]
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k "percentile or radar" -v`
Expected: FAIL

- [ ] **Step 3: 确认 research_mapper 已无常模残留**

百分位计算已在 **Task 3 Step 3** 随 `self.baselines` 一并删除(那里删 `self.baselines` 就必须同步删使用点,否则 `AttributeError`)。这里只做确认:

```bash
grep -n "baselines\|percentile" report_frontend/research_mapper.py
```

Expected:0 命中(若 Task 3 已正确完成)。若有残留,就地删除。

- [ ] **Step 4: 删 visualizer 的常模线**

删除 `create_capability_radar` 里的 `baselines = [60, 60, 60, 60, 60]`、`baselines += [baselines[0]]`、以及 `fig.add_trace(go.Scatterpolar(r=baselines, ..., name='常模基准', ...))`。

同时把 `create_capability_radar` 中"画图 + `_save_fig`"拆出纯函数,便于测试:

```python
def _build_radar_figure(self, result: Dict[str, Any]):
    """构造雷达图,不落盘 —— 便于测试(spec §6)。"""
    ...  # 原 create_capability_radar 的 fig 构造部分,去掉 baselines

def create_capability_radar(self, result: Dict[str, Any]) -> str:
    fig = self._build_radar_figure(result)
    return self._save_fig(fig, "capability_radar")
```

- [ ] **Step 5: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add report_frontend/visualizer.py report_frontend/research_mapper.py tests/test_report_layer.py
git commit -m "fix: 删除杜撰常模与基于它的百分位"
```

---

