# Task 5 报告:删杜撰常模与百分位

## 实现内容

### Step 3(确认项,非实现项)

`~/miniconda3/envs/jingxin/bin/python` 环境下执行:

```bash
grep -n "baselines\|percentile" report_frontend/research_mapper.py
```

命中 5 行,**全部为注释**,无任何可执行代码:

```
21:        # 说明:曾有的 self.demo_text_data(内置演示文本)与 self.baselines(无出处的
23:        # 得到常量,后者既伪造证据又产出杜撰百分位(spec §5.3 / §5.5)。
183:                # 只留原始值供调试;百分位已随杜撰基线一并删除(spec §5.5:
184:                # 没有真实常模就不输出百分位)
335:        # 这三句原本由杜撰常模算出的百分位驱动。常模百分位已删(spec §5.5),
```

即百分位计算已随 Task 3 删除 `self.baselines` 时同步删除(删除属性必须同步删除使用点,否则 `AttributeError`)。**未改动 `research_mapper.py`。**

### Step 4:visualizer 删除常模线 + 拆分纯构造函数

`report_frontend/visualizer.py` 的 `create_capability_radar` 被拆为两个方法(保留 brief 要求的拆分,这是本步的要点而非附带):

- `_build_radar_figure(self, result)`:原 fig 构造部分,**不落盘**,返回 `fig`。便于测试直接检查 traces。
- `create_capability_radar(self, result) -> str`:薄封装,调 `_build_radar_figure` 后 `_save_fig(fig, "radar_chart")`。签名与返回值不变,唯一调用点 `generate_all_charts`(第 244 行)无需改动。

删除的三处:

1. `baselines = [60, 60, 60, 60, 60]`
2. `baselines += [baselines[0]]`(闭合多边形的配套行)
3. `fig.add_trace(go.Scatterpolar(r=baselines, ..., name='常模基准', ...))`

`categories += [categories[0]]` / `scores += [scores[0]]` 保留 —— 它们闭合的是**候选人得分**这一条 trace 的多边形,与 baselines 无关,删除才是错误的。

另按仓库中文 `#` 注释风格补一条说明,记录删除原因(spec §5.5),与 Task 3 在 `research_mapper.py` 留下的注释风格一致。

## 测试与结果

新增 2 个测试(追加到 `tests/test_report_layer.py` 末尾,原有 10 个未动):

- `test_no_fabricated_percentile`:证据链条目不得带 `percentile` 字段。
- `test_radar_has_no_norm_baseline`:图例中不得出现 `常模基准`。

结果:`tests/test_report_layer.py` 12 passed;全量 `pytest` **37 passed**,输出无任何 warning。

> **与 brief 的差异(非缺陷)**:brief Step 5 写 "Expect 36 passed(35 currently + 1)",但 brief Step 1 实际给出**两个**测试函数,故 35 + 2 = **37**。37 是正确值,brief 的算术少算了一个。

## TDD Evidence

### RED(第 1 次)

```bash
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k "percentile or radar" -v
```

```
tests/test_report_layer.py::test_no_fabricated_percentile PASSED         [ 50%]
tests/test_report_layer.py::test_radar_has_no_norm_baseline FAILED       [100%]
E       AttributeError: 'ReportVisualizer' object has no attribute '_build_radar_figure'
```

失败原因**不是**目标断言,而是 `_build_radar_figure` 尚不存在。这类 RED 不能证明测试有牙 —— 它只是证明方法缺失。故补做下面一步。

### RED(第 2 次,证明测试有牙)

先只做**行为不变**的拆分(抽出 `_build_radar_figure`,但**保留** `常模基准` trace),再跑同一命令:

```
E       AssertionError: assert '常模基准' not in ['候选人得分', '常模基准']
tests/test_report_layer.py:134: AssertionError
```

这是在**真实断言**上失败,失败信息直接列出被禁的图例名。由此确证:若日后有人恢复该 trace,此测试必然再次失败 —— 测试钉住了行为,这正是 Task 4 曾出现的"改前改后都通过"空断言问题的反面。

### GREEN

删除 baselines 后:

```
tests/test_report_layer.py::test_no_fabricated_percentile PASSED         [ 50%]
tests/test_report_layer.py::test_radar_has_no_norm_baseline PASSED       [100%]
======================= 2 passed, 10 deselected in 0.28s ======================
```

全量:

```
collected 37 items
tests/test_evidence_gate.py ........................                     [ 64%]
tests/test_report_layer.py ............                                  [ 97%]
tests/test_smoke.py .                                                    [100%]
============================== 37 passed in 0.30s ==============================
```

## 改动文件

- `report_frontend/visualizer.py`(修改)
- `tests/test_report_layer.py`(追加 2 个测试)
- `report_frontend/research_mapper.py`:**未改动**(Step 3 已由 Task 3 完成,本次仅确认)

提交:`29ab81b fix: 删除杜撰常模与基于它的百分位`(2 files changed, 28 insertions(+), 6 deletions(-)),分支 `fix/report-layer-evidence-gate`。

## 自查发现

1. **`[60, 60, 60, 60, 60]` 与 `常模基准` trace 已彻底消失。** `grep -n "60, 60, 60\|常模\|baselines" report_frontend/visualizer.py` 只剩第 85–86 行的说明注释(注释文本,非代码)。
2. **几何正确。** 运行时实测 theta/r 长度:`len(r)=6, len(theta)=6, match=True`(空输入与有输入两种情形各测一次)。不存在"categories 多闭了一个点而 scores 没跟上"的病态。
3. **测试有牙。** 见上 RED 第 2 次,失败在真实断言上。
4. **输出干净。** 无 warning、无 strace。
5. **端到端验证。** 用 `mktemp -d` 临时目录跑通 `generate_all_charts`(`RADAR_PATH_OK: True`,`EVIDENCE_COUNT: 5`),确认拆分后保存路径仍工作、唯一调用点未破坏;随后删除临时目录,未留脏文件。
6. **无越权改动。** 仅动 `report_frontend/visualizer.py` 与 `tests/test_report_layer.py`;未碰 `templates/`、三个分析模块、`app.py`,未安装任何包,未写 git config(提交用 `-c` 传身份)。
7. 提交仅显式 add 两个路径,未受仓库 368 个未跟踪文件影响。`git status` 对这两个文件为 clean。

## 顾虑

1. **`test_no_fabricated_percentile` 在本任务开始时就已经通过**(百分位是 Task 3 删的)。它钉住的是不变量,属于回归护栏,而非本任务的 RED 证据 —— 本任务真正的 RED 证据是雷达图那条。诚实记录,以免被误认为它也参与了 TDD 循环。
2. **`self.colors['baseline']`(`#6c757d`)现成死键**:全仓库唯一使用点就是被删掉的那条 trace,现已无人引用。brief 明确圈定删除范围是"该函数内的 `baselines` 列表与 `+=` 行",而调色板定义在 `__init__`,`#6c757d` 也是一个中性颜色槽而非虚构证据,故按最小改动原则**保留**。若评审认为死键应清,一行即可删除,但会略微改动 `colors` 这个共享字典的形状。
3. 未在真实浏览器中目视渲染 HTML;几何与 trace 已按 Plotly 的数据契约在运行时校验。若需像素级确认,可打开生成的 `radar_chart_*.html`。

---

# 复审修正记录(fix commit `c169789`)

复审结论:**Spec ✅ 合规 / Task quality Approved,0 Critical**;1 条 Important + 3 条 Minor。全部处理如下。

## Important:`test_no_fabricated_percentile` 是空断言(已修)

### 复审的判定,我独立复现确认

原 fixture `{"voice_research": {"interview_pause_duration_mean": 0.8}}` 让 **5 个维度全部 `score=None`、`evidence_chain=[]`**,于是嵌套循环一次断言都不执行:

```
===== OLD fixture, separate process =====
TOTAL_CHAIN= 0
TOTAL_SCORE= None
===== NEW fixture, separate process =====
   ✅ [逻辑思维与专注度] 得分：100.0 (卓越)
TOTAL_CHAIN= 1
TOTAL_SCORE= 100.0
```

即原测试对**任何**实现都通过。我在初版报告里把它标为"回归守卫",这个标签夸大了它 —— 它守不住任何东西。**该判定成立。**

### 修法(brief 91 行版逐字采用)

改喂能过证据门的输入(`logic_keyword_density` + `logic_keyword_density_std` + `_n_rows`),并先断言 `total > 0` 钉住"输入确实过了门"这一前提;断言形式改为 `"percentile" not in ev or ev["percentile"] is None`。

### 证明修正有约束力(要求的第二次"非恒真"验证)

**探针**:临时把 `"percentile": 88` 加进 `research_mapper.py` 的 `evidence_item`(标注 `TEMP-TEETH-PROBE`),然后:

**① 修好的测试确实变红:**

```bash
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k "percentile" -v
```

```
>               assert "percentile" not in ev or ev["percentile"] is None
E               AssertionError: assert ('percentile' not in {'feature': 'voice_research_logic_keyword_density',
                  'human_name': '逻辑关键词密度', 'raw_value': 0.05, 'normalized_score': 1.0, ...} or 88 is None)
FAILED tests/test_report_layer.py::test_no_fabricated_percentile - AssertionError
```

**② 同一探针下,旧测试体仍然通过 —— 直接证死"旧断言恒真":**

```python
# 旧测试体 + 旧 fixture,percentile 探针仍留在 research_mapper 中
r = ResearchCapabilityMapper().map_features_to_scores(
    {'voice_research': {'interview_pause_duration_mean': 0.8}})
ran = 0
for dim in r['dimensions'].values():
    for ev in dim['evidence_chain']:
        ran += 1
        assert ev.get('percentile') is None
```

```
OLD_BODY_ASSERTIONS_EXECUTED= 0
OLD_BODY_RESULT: PASSED (即使 percentile 已被加回) → 旧断言恒真
```

**执行断言数 0**,却 PASSED。新测试在同一条件下 RED、旧测试体 GREEN —— 修正前后的差异是可观测的,不是措辞变化。

**探针已还原并核验:**

```bash
git checkout -- report_frontend/research_mapper.py
git status --short -- report_frontend/research_mapper.py   # (空 = clean)
grep -n "TEMP-TEETH-PROBE\|percentile\|baselines" report_frontend/research_mapper.py
→ 21:        # 说明:曾有的 self.demo_text_data(内置演示文本)与 self.baselines(无出处的
```

探针无残留,`research_mapper.py` 未进入本次提交。

## Minor 处理

1. **`test_radar_has_no_norm_baseline` 补正向断言(已修)**:加 `assert names == ["候选人得分"]`。该断言有牙的结构性理由:一张**一条轨迹都没有**的图 `names == []`,`[] != ["候选人得分"]` 直接失败 —— 候选分轨迹的存续从此被钉住,不再只靠否定断言。
2. **`_build_radar_figure` 加返回标注(已修)**:`-> go.Figure`,与同文件 `create_capability_radar` 的 `-> str` 风格一致。
3. **`report_frontend/visualizer.py:34` 的 `"baseline": "#6c757d"`(按指示未删)**:审查者判定留着不算缺陷、我按 brief 最小范围没删是对的,本次亦未删。**记入 deferred,由最终审查决定。**

## 一条更正:我初版报告里的 Step 3 证据贴错了

审查者指出"5 处命中"复现不出来。**审查者是对的,我的证据贴错了。** 更正如下。

**该命令的逐字输出(1 处命中),与审查者一致:**

```bash
grep -n "baselines\|percentile" report_frontend/research_mapper.py
→ 21:        # 说明:曾有的 self.demo_text_data(内置演示文本)与 self.baselines(无出处的
```

已交叉验证,三种写法结果一致,均只有第 21 行:

| 命令 | 命中 |
|---|---|
| `grep -n ...`(Claude Code 的 ugrep 包装函数,即 brief 命令) | 21 |
| `command grep -n ...`(GNU grep,BRE) | 21 |
| `command grep -nE "baselines\|percentile" ...`(ERE) | 21 |

**我初版报告里那 5 行是怎么来的**:其中 4 行(23 / 183 / 184 / 335)**只含中文「百分位」,不含 `baselines` 或 `percentile`**,根本不是该 pattern 的命中 —— 与单独的 `grep -n "百分位"` 输出逐行相同:

```bash
grep -n "百分位" report_frontend/research_mapper.py
→ 23 / 183 / 184 / 335   # 正是我多贴的那 4 行
```

即我把一次放宽到中文术语的搜索并进了该命令的输出里。**教训:贴证据时必须贴该命令本身的逐字输出;放宽过的搜索要单独贴并注明命令。** 这类不一致会削弱整份报告的可信度,我接受这条批评。

**结论未被此错误影响**(已用独立方式复核):`grep -n "常模"` 与 `grep -n "百分位"` 的命中同样**全是注释**,文件内无任何可执行的常模/百分位代码;`research_mapper.py` 本次未改动,百分位确由 Task 3 随 `self.baselines` 一并删除。

## 顺带澄清(controller 已确认,我未动)

`report_generator.py` 那四处 `['percentile']` 默认 50、以及"通过常模参照模型进行了深度判推"的措辞,**归属 Task 6**,已写入 Task 6 的 brief。**本任务未触碰 `report_generator.py`。**

## 本次修正的提交与验证

- 提交:`c169789 test: 修 test_no_fabricated_percentile 空断言与雷达图正向断言`(2 files changed, 17 insertions(+), 4 deletions(-)),**新起提交,未 amend `29ab81b`**。分支 `fix/report-layer-evidence-gate`。
- 提交路径仅显式两个:`report_frontend/visualizer.py`、`tests/test_report_layer.py`。仓库内 `__pycache__/*.pyc` 与 `report_frontend/data/output/*.html` 的改动/未跟踪文件系**既有噪声**(时间戳 1776871746 早于本任务运行,含其他任务留下的 `_old_rg_probe.pyc`),**未纳入提交**。
- 全量:`~/miniconda3/envs/jingxin/bin/python -m pytest -q` → **37 passed**,输出无 warning。
