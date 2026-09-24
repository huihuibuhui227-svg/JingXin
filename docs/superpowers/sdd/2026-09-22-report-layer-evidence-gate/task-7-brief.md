## Task 7: 措辞替换与眼动图处置

**Files:**
- Modify: `report_frontend/research_mapper.py`(`:377` 硬编码注入、维度 display_name / description)
- Modify: `report_frontend/visualizer.py`(眼动图 + 死代码)
- Modify: `tests/test_report_layer.py`

**Interfaces:**
- Consumes: 无
- Produces: 全仓库 `report_frontend/` + `templates/` 禁止词 0 命中

- [ ] **Step 1: 写禁止词全仓扫描测试**

```python
import ast
from pathlib import Path

SCOPE = [Path("report_frontend"), Path("templates")]

# 禁止词表(spec §5.6)。此处在 Task 7 内重新定义,不依赖 Task 6 —— 读者可能乱序阅读。
BANNED = ["焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
          "录用", "人格", "心理画像", "常模"]


def test_no_banned_words_in_output_strings():
    """spec §5.6:禁止词不得出现在任何字符串字面量里(含 docstring)。

    注释不算 AST Constant,所以说明性注释不受影响。
    """
    offenders = []
    for root in SCOPE:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for word in BANNED:
                        if word in node.value:
                            offenders.append(f"{path}:{node.lineno} {word}")
    assert not offenders, "字符串字面量含禁止词:\n" + "\n".join(offenders)
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k banned_words_in_output -v`
Expected: FAIL,列出 `research_mapper.py` 的 `"抗压与情绪稳定性"`、`"自信度"`、`pitch_info="语调丰富"` 等

- [ ] **Step 3: 改维度名与描述**

`research_mapper.py` 的 `mapping_rules`:

| 原 | 改为 |
|---|---|
| `"name": "抗压与情绪稳定性"` | `"name": "情境行为稳定性"` |
| `"description": "评估高压下的情绪控制力、生理指标平稳度及焦虑水平。"` | `"description": "观测会话中的可测行为量。"` |
| `"name": "自信度"` | `"name": "行为表现活跃度"` |
| `"description": "评估自我效能感、肢体开放度及眼神交流质量。"` | `"description": "观测肢体与注视相关的可测量。"` |
| `"name": "逻辑思维与专注度"` | `"name": "话语结构特征"` |
| `"name": "认知负荷效率"` | `"name": "言语流畅特征"`(审查 Q5:"效率"= 产出/投入,系统没有"产出") |
| **`"human_name": "面部紧张度"`** | **`"human_name": "眉间收缩与唇部压缩"`** |
| `inference_template` 里的"抗压能力""科研自信心" | 删除模板机制(见 Step 4) |

⚠️ **`human_name` 这一行是 Task 6 上报的缺口** —— 原改名表只覆盖 `display_name`,而实测 `human_name` 里也有一个含禁止词的:「面部紧张度」(含 **紧张**),它由 `_render_dimension_block` 渲染进报告的指标表与证据缺口清单。**不改它,Task 6 那条禁止词测试即使到 Task 7 之后仍然会红。**

新名「眉间收缩与唇部压缩」**是控制器给的默认**(该槽实际覆盖 au4 眉间收缩与 au23 唇部压缩);spec §5.6 建议的"面部紧张相关动作单元活动率"仍含"紧张",不可用。**使用者可否决。**

改完后请**再全量扫一次** `research_mapper.py` 的 `name` / `description` / `human_name`,确认 0 命中(不能只改表里列的这两个)。

**命名已由使用者于 2026-09-22 拍板,采用上表默认值。** spec §7.1 的待定项关闭。

⚠️ 唯一约束:新名字里**不得含禁止词**。特别地,**不能用"压力情境下的行为稳定性"** —— "压力"在 §5.6 的禁止词表里,用了会被本任务 Step 1 的扫描测试打回。

- [ ] **Step 3a: 恢复禁止词全量扫描(Task 6 收窄的那条)**

Task 6 的 `test_deep_analysis_has_no_banned_words` 曾收窄为"剔除 `抗压与情绪稳定性` 与 `面部紧张度` 两个 mapper 标签后扫描",原因是那两个字符串属 Task 7 范围。

**Step 3 改名完成后,把那段剔除逻辑删掉,恢复全量:**

```python
    html = ReportGenerator()._generate_deep_text_analysis(feats, result)

    # Task 7 已修完 display_name 与 human_name,恢复全量扫描(不再剔除标签)
    for word in BANNED:
        assert word not in html, f"叙事层出现禁止词:{word}"
```

**验收**:恢复全量后该测试仍须 PASS。若红,说明 mapper 还有未改净的标签 —— 回到 Step 3 的"全量扫一次"。

- [ ] **Step 3b: 清理报告头部与标题字符串**

`report_generator.py:301-302` 的 `"🔬 JingXin 科研能力评估报告"` / `"基于多模态心理特征的深度分析与判推"`,以及 `:155` 的 `"常模参照模型"`:

| 位置 | 原 | 改为 |
|---|---|---|
| `:301` | `JingXin 科研能力评估报告` | `JingXin 面试行为观测报告` |
| `:302` | `基于多模态心理特征的深度分析与判推` | `基于多模态行为量的结构化观测` |
| `:155` | `并通过常模参照模型进行了深度判推` | 删 |
| `:155` | `进行了毫秒级量化分析` | 删(采样率 ≈10fps,不是毫秒级) |

同时把 `report_generator.py` 模块顶部注释与 `visualizer.py` 的 `create_capability_radar` 标题 `"📊 科研能力五维模型评估"` 改为 `"📊 五维行为观测"`。

- [ ] **Step 4: 删推理模板与硬编码注入**

删除 `_generate_deep_inference`(`:360-382`)整段,连同 `mapping_rules` 里的 `inference_template` 字段。`narrative` 改由 Task 6 的证据状态渲染产生,不再有模板填空。

这同时删掉 `:377` 的 `pitch_info="语调丰富", eye_info="眼神交流充分", hand_info="手势自然"`。

- [ ] **Step 5: 眼动图按 spec §5.5 处置**

删除 `create_gaze_plot_from_df` 里的两张图与 `add_shape(rect, x0=-1, y0=-1, x1=1, y1=1)`;函数体改为:

```python
def create_gaze_plot_from_df(self, df_face) -> Optional[str]:
    """M3 之前不生成眼动图。

    gaze_direction_y 是解剖常量、iris_x/y 是图像归一化坐标(编码人脸位置),
    两者都不能支撑"注视热力图"这个标题。见 spec §5.5。
    """
    return None
```

同时删除死代码 `visualizer.py:141-164`(函数体为 `return None` 的模拟眼动版本)。

- [ ] **Step 6: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 7: 全仓人工复核**

```bash
grep -rn "焦虑\|紧张\|压力\|抗压\|情绪稳定\|说谎\|诚信\|录用\|人格\|心理画像\|常模" report_frontend/ templates/ --include="*.py" --include="*.html"
```

Expected:仅剩注释与 `evidence_gate.py` 的封停理由说明(那是给维护者看的,不进入输出)。

- [ ] **Step 8: 提交**

```bash
git add report_frontend/research_mapper.py report_frontend/visualizer.py tests/test_report_layer.py
git commit -m "refactor: 中性化维度命名与措辞,停用眼动图与死代码"
```

---

---

