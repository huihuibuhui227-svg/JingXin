# Task 7 报告:措辞替换与眼动图处置

- BASE(dispatch 前):`ae85bfc`(Task 6 fix 轮 1 完成点)
- Commit:`7a4a543` — `refactor: 中性化维度命名与措辞,停用眼动图与死代码`
- 解释器:`~/miniconda3/envs/jingxin/bin/python`
- 测试:**41 passed**(`tests/ -v`,含 `-W error` 亦 41 passed)
- 门 4 交付物:`~/shared/jingxin_gate4_report.html`(7035 字节,sha256 `56c49b338e2af95e7ad30baf775f8fc5ceee661fa1b634a9df27aa562eb08a23`)

---

## 1. Step 1/2 — 禁止词全仓扫描测试(RED)

新增测试 `test_no_banned_words_in_output_strings`(`tests/test_report_layer.py`):

- `SCOPE` 以**测试文件位置**为基准(`_REPO_ROOT = Path(__file__).resolve().parents[1]`),不用裸相对路径 —— 否则扫描会静默依赖 cwd。
- **复用**模块级 `BANNED`(Task 6 引入),未另立同名常量。已逐字核对它与 spec §5.6 的 11 个词一致:`焦虑 紧张 压力 抗压 情绪稳定 说谎 诚信 录用 人格 心理画像 常模`(计数 = 11),无需更正。
- 加了一条**反空断言守卫**:`scanned >= _MIN_SCANNED_FILES`(5)。SCOPE 写错时 `rglob` 返回空、offenders 恒空、测试会假装通过 —— 本计划已栽过八次空断言,故把它做成显式失败。
- 只扫 `*.py` + `ast.Constant`,即**字符串字面量**(含 docstring);注释不是 Constant,故维护者注释不受约束(Global Constraints 的豁免范围)。

### RED 输出(改动前,`pytest tests/test_report_layer.py -k banned_words_in_output`)

```
E       AssertionError: 字符串字面量含禁止词:
E         /home/huihuibuhui/jingxin/report_frontend/feature_engine.py:470 紧张
E         /home/huihuibuhui/jingxin/report_frontend/feature_engine.py:477 紧张
E         /home/huihuibuhui/jingxin/report_frontend/research_mapper.py:38 抗压
E         /home/huihuibuhui/jingxin/report_frontend/research_mapper.py:38 情绪稳定
E         /home/huihuibuhui/jingxin/report_frontend/research_mapper.py:39 焦虑
E         /home/huihuibuhui/jingxin/report_frontend/research_mapper.py:41 压力
E         /home/huihuibuhui/jingxin/report_frontend/research_mapper.py:41 抗压
E         /home/huihuibuhui/jingxin/report_frontend/research_mapper.py:43 紧张
```

8 处命中,**全部**在改动前被修掉 —— 包括 brief 的 Step 2 预期清单漏掉的 `feature_engine.py` 两条 print(与控制器第 5 条一致)。扫描首次运行即失败(非恒真)。

---

## 2. 逐字符串改动清单

### 2.1 `report_frontend/research_mapper.py`

| 位置 | 原 | 改为 |
|---|---|---|
| 类 docstring | `科研能力映射引擎 (最终修复版 - 包含 stats 字段)` | `行为指标映射引擎 (最终修复版 - 包含 stats 字段)` |
| `logical_thinking.name` | `逻辑思维与专注度` | `话语结构特征` |
| `logical_thinking.description` | `评估思维严密性、语言逻辑结构及视觉注意力集中程度。` | `观测文本结构与面部动作单元相关的可测量。` |
| `stress_resilience.name` | `抗压与情绪稳定性` | `情境行为稳定性` |
| `stress_resilience.description` | `评估高压下的情绪控制力、生理指标平稳度及焦虑水平。` | `观测会话中的可测行为量。` |
| `stress_resilience` 指标 `human_name` | `面部紧张度` | `眉间收缩与唇部压缩` |
| `communication_fluency.description` | `评估语言组织能力、语调丰富度及表达连贯性。` | `观测语音韵律与停顿相关的可测量。` |
| `confidence_level.name` | `自信度` | `行为表现活跃度` |
| `confidence_level.description` | `评估自我效能感、肢体开放度及眼神交流质量。` | `观测肢体与注视相关的可测量。` |
| `cognitive_efficiency.name` | `认知负荷效率` | `言语流畅特征` |
| `cognitive_efficiency.description` | `评估处理复杂信息时的脑力消耗效率。` | `观测面部动作单元频率与回答长度的可测量。` |
| 5 × `inference_template` | (各含 抗压/压力/自信 等) | **整字段删除** |
| 控制台 print | `⚖️ 正在执行深度映射与心理科研能力判推...` | `⚖️ 正在执行行为指标映射...` |
| 无证据分支 | `"narrative": "本次未采集到足以评估该行为线索的有效样本。"` | 键删除 |
| 出分分支 | `"narrative"` / `"simple_narrative"` 两键 | 键删除 |
| `_generate_simple_narrative` | 整方法 | 删除(唯一调用方是 `_generate_deep_inference`) |
| `_generate_deep_inference` | 整方法(含 `pitch_info="语调丰富", eye_info="眼神交流充分", hand_info="手势自然"` 硬编码注入) | 删除 |
| `_generate_summary_narrative` | `综合科研潜力评分：**X** (档位)。核心优势在于**A**；建议关注**B**的提升。` | 见 §2.5 |
| 新增模块常量 | — | `CONF_ORDER = {"无":0,"低":1,"中":2,"高":3}`(唯一一份,`report_generator` 导入) |

**三处未被 brief 列出的 description 也一并中性化**(disclosed):改完表里那两个之后,`话语结构特征` 底下写着「评估思维严密性、视觉注意力」、`言语流畅特征` 底下写着「评估脑力消耗效率」,自相矛盾。审慎的最小一致性处理:按 brief 自己的替换风格(「观测…的可测量。」)改齐 5 个 description。`algorithm` 字段(如 `认知负荷推断`、`响应延迟回归`)**未动**,见 §8 观察项。

### 2.2 `report_frontend/report_generator.py`

| 原 | 改为 |
|---|---|
| 类 docstring `科研能力评估报告生成器 (终极丰满版)` | `行为观测报告生成器` + 三行「只陈述实测,不解读不推断」说明 |
| 本地 `_CONF_ORDER` 字典 | 删除;改 `from .research_mapper import CONF_ORDER` |
| print `🚀 启动 JingXin 科研能力评估报告生成系统 (批量模式)` | `🚀 启动 JingXin 面试行为观测报告生成系统 (批量模式)` |
| print `…(实时模式)` | 同上(实时模式) |
| 「🧠 判推」框 `<div class="narrative-box">…{…['narrative']}</div>` | **整框删除**(消费的 `narrative` 已不存在,留着即 `KeyError`;spec §5.4 本就要求不判推) |
| CSS `.narrative-box` | 删除(随框成为死样式) |
| `<title>JingXin 科研能力深度评估报告</title>` | `<title>JingXin 面试行为观测报告</title>` |
| `<h1>🔬 JingXin 科研能力评估报告</h1>` | `<h1>🔬 JingXin 面试行为观测报告</h1>` |
| 副标题 `基于多模态心理特征的深度分析与判推` | `基于多模态行为量的结构化观测` |
| `<h2>📑 深度心理特征分析报告</h2>` | `<h2>📑 行为指标观测明细</h2>` |
| 分数卡标签 `综合科研潜力评分` | `综合行为观测评分` |
| `<h3>📊 五维能力模型</h3>` | `<h3>📊 五维行为观测</h3>` |
| 眼动卡 `<h3>👁️ 眼动行为分析</h3>` + gaze iframe | **整卡删除**,连 `.grid-2` 包裹与对应 CSS |
| `_generate_gaze_insight`(全仓无调用方) | 删除 |

`total_level` / `_get_level` 按控制器指示**保留**(那是分数自己的档位标签,不是对人的推断)。

**眼动卡删除的额外理由**(brief 未明说,disclosed):Step 5 让 `create_gaze_plot_from_df` 恒返回 `None` 后,该卡会渲染成 `图表缺失` 占位框 —— 标题「眼动行为分析」+ 一个缺失占位,比不渲染更糟。验收清单要求「无眼动图」,故整卡移除。

### 2.3 `report_frontend/visualizer.py`

| 原 | 改为 |
|---|---|
| 类 docstring `科研能力评估报告可视化引擎 (终极版：含眼动轨迹 + 自动保存)` + 【功能升级】三条 | `行为观测报告可视化引擎 (雷达图 + 逐维证据图 + 自动保存)` + 【说明】三条(M3 前不生成眼动图) |
| 雷达图标题 `📊 科研能力五维模型评估` | `📊 五维行为观测` |
| `create_gaze_trajectory_heatmap`(函数体 `return None`,无调用方) | 删除 |
| `create_gaze_plot_from_df` 全体(两张图 + `add_shape(rect, x0=-1,y0=-1,x1=1,y1=1)`) | `return None` + 记录删除理由的 docstring |
| `generate_all_charts` 眼动分支(4 条 print:`👁️ 已保存：眼动轨迹图` / `⚠️ 未找到眼动数据，跳过眼动图生成。` / `⚠️ 未传入面部原始数据…`) | 单条 `create_gaze_plot_from_df(df_face)` 调用 + `ℹ️ 眼动图:M3 前不生成(现有坐标不能支撑'注视'构念,spec §5.5)。` |
| `from plotly.subplots import make_subplots` | 删除(唯一调用点在被删的眼动图里,本次改动使它成为死导入) |
| `__main__` print `=== 测试 Visualizer (含眼动 + 自动保存) ===` | `=== 测试 Visualizer (雷达图 + 证据图 + 自动保存) ===` |

`create_gaze_plot_from_df` 的**签名与调用点都保留**(函数体只 `return None`),M3 换眼内相对坐标后可直接续写;`charts['gaze']` 键同样保留。

### 2.4 `report_frontend/feature_engine.py`(控制器第 5 条)

`inspect_gaze_features` 的 10 行控制台小结整段改写(两条含「紧张」的必改,同块内其余同类解读句一并改,避免同一块里一半中性一半评判):

| 原 | 改为 |
|---|---|
| `💡 智能分析结论:` | `💡 量表概览:` |
| `✅ 被测者视线非常稳定，显示出极高的专注度。` | `✅ 视线稳定性指标高于 0.8。` |
| `➖ 被测者视线稳定性适中。` | `➖ 视线稳定性指标介于 0.5 与 0.8 之间。` |
| `⚠️ 被测者视线波动较大，可能注意力分散或紧张。` | `⚠️ 视线稳定性指标低于 0.5。` |
| `✅ 被测者眼神接触良好，表现出较强的自信心。` | `✅ 眼神接触比例高于 0.7。` |
| `➖ 被测者眼神接触一般。` | `➖ 眼神接触比例介于 0.4 与 0.7 之间。` |
| `⚠️ 被测者眼神接触较少，可能存在回避或紧张情绪。` | `⚠️ 眼神接触比例低于 0.4。` |

**分档阈值一字未动**(0.8/0.5/0.7/0.4 与 `stability_val`/`contact_val` 的取用关系保持原样)。

### 2.5 `_generate_summary_narrative` 中性化

报告里**仅存的解释性生成器**(「核心优势在于 X;建议关注 Y 的提升」)已改为只报事实:

```python
passed, slots = Σ int(dim["matched_indicators"] 的两个数字)
total_score is None → "本次会话 {slots} 个指标槽中 0 个通过证据门，未产出综合分。"
否则            → "本次会话 {slots} 个指标槽中 {passed} 个通过证据门，综合行为观测评分 {score}（置信度上限：{cap}）。"
```

- 槽数直接取各维已渲染的 `matched_indicators`,与报告正文同源(不再另算一遍)。
- 加了个顺带修正:原句的 `**X**` markdown 星号在 HTML 里是**字面量**(卡片是纯文本插值),会印成 `**100.0**`;新句不含星号。
- 置信度上限用 `CONF_ORDER`;该常量的**唯一定义**移到 `research_mapper`(产出置信度的一方),`report_generator` 改为导入 —— 避免两个模块各存一份顺序表而漂移。

### 2.6 `report_frontend/__init__.py`

`用于基于历史日志生成科研能力心理评估报告的前端分析模块。` → `用于基于历史日志生成面试行为观测报告的前端分析模块。`

### 2.7 `tests/test_report_layer.py`

- 顶部加 `import ast` / `from pathlib import Path`。
- 新增 `SCOPE`、`_MIN_SCANNED_FILES`、`test_no_banned_words_in_output_strings`。
- **Step 3a**:删除 `test_deep_analysis_has_no_banned_words` 里那段「精确剔除两个 mapper 标签」的逻辑,恢复全量扫描;BANNED 循环现覆盖**叙事层自己的句子 + mapper 的 `display_name`/`human_name`** 两个来源。docstring 重写并写明「不要再把标签剔除加回来」。
- `test_deep_analysis_handles_none_scores` 的 docstring 里 `_CONF_ORDER` → `CONF_ORDER`(常量改名后的陈旧引用)。

**执行顺序**(遵守交接要求):先做 Step 3 改名,再删剔除逻辑 —— 剔除逻辑在改名落地前是承重的。

---

## 3. Step 6 / Step 3a — GREEN

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v
tests/test_evidence_gate.py ......... 24 items 全部 PASSED
tests/test_report_layer.py ......... 16 items 全部 PASSED
tests/test_smoke.py::test_package_imports PASSED
============================== 41 passed in 0.33s ==============================
```

`-W error` 下同样 `41 passed`,输出干净无告警。(Task 6 结束时是 40;本任务 +1 = 新的全仓扫描测试。)

---

## 4. 约束力证明(三次注入,均以字节级还原收尾)

还原策略遵守 Ruling 30:**先 `cp` 到 `/tmp` 备份,再反向还原,并比对 sha256**(未提交改动不能用 `git checkout --`)。

### 4.1 注入 A:把两个 mapper 标签退回旧值(Task 6 剔除逻辑覆盖的正是这两个)

```
$ python -c "replace('情境行为稳定性'→'抗压与情绪稳定性', '眉间收缩与唇部压缩'→'面部紧张度')"
$ pytest tests/test_report_layer.py -k banned -q
E       AssertionError: 字符串字面量含禁止词:
E         .../research_mapper.py:42 抗压
E         .../research_mapper.py:42 情绪稳定
E         .../research_mapper.py:46 紧张
E           AssertionError: 叙事层出现禁止词:紧张
E           assert '紧张' not in '<p>综合行为观测摘要...'
E             '紧张' is contained here:
E               <ul><li>面部紧张度: 未采集到对应数据</li>...
FAILED tests/test_report_layer.py::test_deep_analysis_has_no_banned_words
FAILED tests/test_report_layer.py::test_no_banned_words_in_output_strings
2 failed, 14 deselected in 0.29s
```

**这是 Step 3a 的关键证据**:失效打在叙事层 HTML 上(`面部紧张度` 经 `_render_dimension_block` 进缺口清单),而这正是 Task 6 剔除逻辑遮住的那条路 —— 恢复全量后它真的会红。

还原:`cp /tmp/t7_mapper_backup.py` 回写,sha256 `e20b982f0e2f93efa4751e742df6cd3a20f97801738ae8f48113ae122133756d` 与注入前**逐字节相同**,随后 `pytest tests/` = 41 passed。

### 4.2 注入 B:往叙事层**自己写的**字符串里塞禁止词(非 mapper 标签)

```
$ python -c "replace('<p><strong>证据不足</strong> —— 本次未采集到足以评估该行为线索的有效样本。</p>'
                   → '<p><strong>证据不足</strong> —— 紧张。</p>')"
$ pytest tests/test_report_layer.py -k banned -q
E           AssertionError: 叙事层出现禁止词:紧张
E             '紧张' is contained here:
E               trong> —— 紧张。</p>
E       AssertionError: 字符串字面量含禁止词:      ← 扫描测试同样变红
2 failed, 14 deselected
```

还原:sha256 `6df6b6708ddd917a00a952c1c4140c9d9f2997bd859de692c18afaa0393121cb` 与注入前相同;`pytest tests/` = 41 passed。

### 4.3 注入 C:打坏 `SCOPE`,验证反空断言守卫(tests/ 内部)

```
$ python -c "SCOPE 的 report_frontend → report_frontend_missing"
$ pytest tests/test_report_layer.py -k banned_words_in_output -q
E       AssertionError: 只扫到 1 个文件,本测试退化为空断言
E       assert 1 >= 5
1 failed, 15 deselected in 0.25s
```

**证明这条扫描测试不会因为 SCOPE 打错而静默通过**(1 = 剩下的 `templates/__init__.py`)。
还原:sha256 `9e030292221689e9754b7dd3d83a09876c3f353e7a8a86ed288903854d65c3fa` 与注入前相同;`pytest tests/` = 41 passed。

---

## 5. Step 7 — 全仓人工复核

```bash
$ grep -rn "焦虑\|紧张\|压力\|抗压\|情绪稳定\|说谎\|诚信\|录用\|人格\|心理画像\|常模" \
      report_frontend/ templates/ --include="*.py" --include="*.html"
report_frontend/research_mapper.py:27:        # 常模均值/标准差)已删除 —— ...
report_frontend/research_mapper.py:184:                # 没有真实常模就不输出百分位)
report_frontend/visualizer.py:85:        # 说明:曾有的 '常模基准' 虚线来自硬编码的 [60]*5,...
report_frontend/visualizer.py:86:        # 却以权威对比的形式呈现,故一并删除(spec §5.5:没有真实常模就不画常模线)。
```

**分类:4 处命中,100% 是注释(`#`),0 处字符串字面量。** 4 处都是「为什么删掉某个东西」的维护者说明,按 Global Constraints 豁免(注释不入输出,且 Step 1 的 AST 扫描只读 Constant,不受影响)。

另两处豁免项:

- `tests/test_report_layer.py:186-187` 的 `BANNED` 列表(测试定义本身,Global Constraints 明示豁免);
- `tests/test_report_layer.py:209` 的 docstring 提到旧标签名(记录历史,`tests/` 不在扫描范围内 —— 若把 `tests/` 纳入 SCOPE,`BANNED` 自身会立刻自伤)。

`templates/dashboard.html` 0 命中(该文件另有 grep 覆盖,未被测试覆盖,见 §8)。

**与控制器第 6 条一致**:`report_frontend/evidence_gate.py` 与 `evidence_thresholds.json` 实测**0 命中**(`grep` 退出码 1),封停理由文案里没有禁止词 —— brief 的 Step 7 预期确实陈旧。证据门本次**未做任何改动**。

---

## 6. 门 4 交付物:真实报告

```
$ ~/miniconda3/envs/jingxin/bin/python -m report_frontend.report_generator
   📥 [FACE] 4 行 / [GESTURE] 4 行 / [VOICE_RESEARCH] 8 行
   📈 已保存:雷达图 + 5 张证据图
   ℹ️ 眼动图:M3 前不生成(现有坐标不能支撑'注视'构念,spec §5.5)。
✅ 报告已生成并打开：data/output/Research_Assessment_Report_20260924_111121.html

$ cp "$(ls -t data/output/*.html | head -1)" ~/shared/jingxin_gate4_report.html
```

- **路径**:`~/shared/jingxin_gate4_report.html`(= Windows `D:\Shared\jingxin_gate4_report.html`)
- **大小**:7035 字节;sha256 `56c49b338e2af95e7ad30baf775f8fc5ceee661fa1b634a9df27aa562eb08a23`
- **未手工编辑**,直接来自 `report_generator` 的输出。报告由**本任务最终代码**重新生成(注入实验之后),与提交内容一致。

### 我逐项核对的结果

| 验收项 | 结果 |
|---|---|
| 标题中性 | `<title>JingXin 面试行为观测报告</title>`、`<h1>🔬 JingXin 面试行为观测报告</h1>`、副标题「基于多模态行为量的结构化观测」 |
| 无判推框 | `判推` 出现 **0** 次;证据卡只剩 `🔍 {维度} - 证据链` + 图 |
| 无眼动图 | `眼动` 0 次、`gaze` 0 次、无 `图表缺失` 占位 |
| 逐维缺口无重复 | 20 条 `<li>` 缺口,**20 条互不相同**(`Counter` 无 >1 项),每条带维度归属 |
| 诚实「证据不足」 | 出现 6 次(分数卡档位 1 + 5 个维度块各 1);摘要行「本次会话未采集到足以支撑评估的有效证据。」 |
| 无 `None` | `"None" in html` = False;分数卡显示 `—` |
| 无禁止词 | 11 个禁止词在报告里命中 **0** |
| 无「科研能力 / 心理」 | 两串均 0 次 |

渲染出的正文(节选):

```
🔬 JingXin 面试行为观测报告 / 基于多模态行为量的结构化观测 / 2026-09-24 11:11 | JingXin-Mapper-v10.1-FixedStats
— / 综合行为观测评分 / 证据不足
📝 综合总结   本次会话 20 个指标槽中 0 个通过证据门，未产出综合分。
📑 行为指标观测明细   本次会话未采集到足以支撑评估的有效证据。
  话语结构特征  证据不足 —— 本次未采集到足以评估该行为线索的有效样本。
    逻辑关键词密度: 未采集到对应数据 / 面部专注度: 本次会话内无变化 /
    视线稳定性: 有效样本不足 / 困惑微表情 (皱眉): 未采集到对应数据
  情境行为稳定性 …眉间收缩与唇部压缩: 有效样本不足 / 肢体抖动: 本次会话内无变化 /
    视线偏差: 有效样本不足 / 面部对称性: 有效样本不足
  沟通表达流畅度 …平均停顿时长: 该指标本轮停用
  行为表现活跃度 …手势自信分/肩部放松度: 本次会话内无变化
  言语流畅特征 …眼部挤压/眨眼频率/回答详尽度/反应延迟: 未采集到对应数据
📊 五维行为观测     🔍 分维度证据链 ×5
```

即真实数据下走的仍是 Task 3 建立的**诚实短接**路径(21 槽全被拦 / 20 槽在 5 维映射规则下),报告只报事实。

---

## 7. 提交

```
$ git add report_frontend/research_mapper.py report_frontend/visualizer.py \
          report_frontend/report_generator.py report_frontend/feature_engine.py \
          report_frontend/__init__.py tests/test_report_layer.py
$ git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit
[fix/report-layer-evidence-gate 7a4a543] refactor: 中性化维度命名与措辞,停用眼动图与死代码
 6 files changed, 143 insertions(+), 216 deletions(-)
```

- **精确路径**,未用 `-A`;提交内**无 `.pyc`、无未跟踪文件**(仓库仍剩 ~369 个未跟踪文件,原样未动)。
- add 清单 = brief 的清单 + `report_generator.py`(控制器第 2 条)+ **`feature_engine.py` 与 `__init__.py`**(控制器第 4/5 条要求改这两个文件,漏掉它们会把改动孤悬在提交之外 —— 与 `report_generator.py` 同一类错误)。
- 无 amend、无 rebase、无 force。
- staged blob 已验证为纯 LF(`grep -c $'\r'` = 0);磁盘上 `feature_engine.py`/`__init__.py` 仍是 CRLF(仓库既有状态,`.gitattributes` 的 `text=auto` 在提交时归一化),提交未引入行尾噪音(`--numstat` 只有真实改动行)。

**改动文件**:`report_frontend/research_mapper.py`、`report_frontend/report_generator.py`、`report_frontend/visualizer.py`、`report_frontend/feature_engine.py`、`report_frontend/__init__.py`、`tests/test_report_layer.py`(未动 `evidence_gate.py`、`data_loader.py`、`templates/`)。

---

## 8. 自查发现与遗留(交最终审查分诊,均未擅自扩范围)

1. **`algorithm` 字段未动,但已有三项与代码不符**:`加权线性组合 + 认知负荷推断`(认知负荷推断已删)、`眼动 - 肢体多模态耦合模型`(3/4 槽被封停)、`微表情频率分析与响应延迟回归`(无回归、reaction_time 封停)。这些是**方法标签**不是对人的断言、也不渲染进报告,故按「description 属 brief 替换表覆盖范围、algorithm 从未被提及」的界线留给审查判断。
2. **`PsychologicalFeatureEngine` 类名仍含「心理」**:feature_engine.py 的三处「心理/科研」都挂在类名上(类名本身被 report_generator/tests/app.py 引用),改名会动到公共 API,超出本任务。
3. **`research_mapper.py` 的未使用 import 仍在**(`numpy`/`math`/`json`/`List`/`Optional`)。这是账本里判给 **Task 6** 的第 ④ 项,Task 6 未执行,本次未代做(避免跨任务扩范围)。
4. **两处删除没有 pytest 守卫**:①眼动卡与 `.grid-2` CSS 的移除;②`charts['gaze']` 键现在在报告层无消费方(保留是为 M3 续写)。两者的证据只有本报告的 HTML 核对。
5. **无证据会话的雷达图是空多边形**(5 维 score 全 `None` 传给 plotly)。不构成任何断言,但读者可能问「为什么图是空的」;是否加一句占位说明属产品决定。
6. **`tests/` 不在扫描范围、`templates/*.html` 只被 Step 7 的 grep 覆盖**(AST 扫描按 brief 只读 `*.py`)。若日后 `dashboard.html` 引入禁止词,测试不会变红 —— 已实测它今天 0 命中。
7. **`positive_factors` / `negative_factors` / `evidence_item["status"]` 现无消费方**(它们的唯一读者 `_generate_deep_inference` 已删)。属于打分循环内的数据字段、不构成输出断言,未删。
