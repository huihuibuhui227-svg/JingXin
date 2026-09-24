# Task 6 报告:重写硬编码叙事层

状态:**NEEDS_CONTEXT** —— 实现全部完成且测试全绿,但计划里的一条测试断言在 Task 6 内**不可满足**,
需要控制器裁决(详见「需要裁决的一处」)。

---

## 1. Step 0:修复两条空断言 + 扫描验收

### 扫描脚本

计划未随附扫描脚本,我照实际约定重写了一份(`/tmp/jx_sweep.py`,不入库):
`sys.settrace` 记录每个测试函数**实际执行到的行号**,与 AST 里该函数的 `assert` 行求差。
脚本另打印每个测试的 `已执行/总数`,便于区分"全部执行"与"零执行"。

### 修复前(基线,与控制器实测一致)

```
❌ test_no_fake_resume_fallback: 3/3 条断言从未执行,行号 [21, 22, 23]
❌ test_quarantined_columns_are_rejected: 2/2 条断言从未执行,行号 [33, 34]
✅ 其余 10 个测试全部执行

存在未执行断言的测试数 = 2 (共 12 个测试)
```

### 修复后

两函数整体替换为 brief 给出的版本(fixture 改成能过门的
`logic_keyword_density` + `_n_rows=100`),实测 `matched_indicators == "1/4"`、
3 条缺口、链中只有干净键,与控制器实测一致。

```
✅ test_no_fake_resume_fallback: 5/5 条断言已执行
✅ test_quarantined_columns_are_rejected: 3/3 条断言已执行

存在未执行断言的测试数 = 0 (共 12 个测试)
```

**Step 0 验收达成:未执行断言数 = 0。**

提交:`ea65285 test: 修复两条空断言(假简历兜底/封停列),并给出可过门的 fixture`
(与 Task 5 的 `c169789` 同例,Step 0 单独成一个提交)

---

## 2. TDD 证据

### RED(Step 2)

命令:
```
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py \
  -k "banned or none_scores or hardcoded_gaze" -v
```

输出:
```
FAILED tests/test_report_layer.py::test_deep_analysis_has_no_banned_words - AssertionError: 叙事层出现禁止词:紧张
FAILED tests/test_report_layer.py::test_deep_analysis_handles_none_scores - AssertionError: assert '证据不足' in '<h3>行为观测摘要</h3>...'
2 failed, 1 passed, 12 deselected in 0.26s
```

**为什么这个失败是预期的**:
- `test_deep_analysis_has_no_banned_words`:旧函数无条件拼出
  「该数值处于较低水平,展现了极佳的**情绪控制力**和**心理稳定性**」等硬编码评语,
  且 `:195`「微表情监测未检测到显著的焦虑特征」因查了拼错的键
  (`face_micro_exp_au_name_au4_freq`,实际键名带 `micro_exp_` 前缀)而**无条件打印** —— 命中「紧张」「焦虑」。
- `test_deep_analysis_handles_none_scores`:旧函数零证据时走的是 Task 3b 的过渡短接,
  返回「行为观测摘要 / 本次会话未采集到足以支撑评估的有效证据」,**不含字面量「证据不足」**。

`test_no_hardcoded_gaze_claim` 在 RED 阶段即通过 —— 因为 Task 3b 的短接恰好绕开了那两句硬编码。
它不是空断言(2/2 断言执行),作为回归守卫保留。

### GREEN(Step 5 / Step 7)

```
~/miniconda3/envs/jingxin/bin/python -m pytest -q
39 passed in 0.31s

~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -q
14 passed in 0.28s
```

`-W error::DeprecationWarning -W error::UserWarning` 下同样 39 passed,无告警。

---

## 3. 实现内容

### `report_frontend/report_generator.py`

1. 模块顶部新增 `_CONF_ORDER = {"无": 0, "低": 1, "中": 2, "高": 3}`
   (已核对 `evidence_gate.confidence_from` 只产出这四档)。
2. 新增 `_render_dimension_block(dim_key, dim)`:逐维度输出证据状态 ——
   有分则列出 `human_name / raw_value / normalized_score / weight` 表 + `matched_indicators` + `confidence`;
   无分则**证据不足** + 该维度缺口列表。
3. `_generate_deep_text_analysis` 整段重写(原 143 行 → 28 行):
   总分 `None` 时只给一句诚实空报告摘要;否则给 `total_level` + 置信度上限;
   随后逐维度渲染 + 顶层证据缺口。**不含任何阈值评语、形容词、推断或默认值兜底。**
4. `_score_display` 兜底保留(Task 3b 产物,未动)。

### `tests/test_report_layer.py`

新增 2 个测试(`test_deep_analysis_handles_none_scores`、`test_no_hardcoded_gaze_claim`)。
计划的第三个测试 `test_deep_analysis_has_no_banned_words` **暂未落盘**,原因见下。

---

## 4. 需要裁决的一处(唯一的阻塞项)

### 现象

加上 brief 的 `test_deep_analysis_has_no_banned_words` 后,全套件 `1 failed, 39 passed`:

```
AssertionError: 叙事层出现禁止词:紧张
```

### 根因(实测,非推断)

命中片段只有三处,全部来自 `research_mapper.py` **自己的标签字符串**:

```
[紧张]     <li>面部紧张度: 有效样本不足</li>                    ← research_mapper.py:43  human_name
[抗压]     <h3>抗压与情绪稳定性</h3>                             ← research_mapper.py:38  display_name
[情绪稳定] <h3>抗压与情绪稳定性</h3>                             ← 同上
```

对照组实验:**只**把这两个 mapper 标签换成占位符、叙事层代码一字不改:

```
替换 mapper 标签后命中: 0 命中
```

即**重写后的叙事层自身 0 命中**(AST 扫描 `report_generator.py` 全部字符串字面量,含 docstring,
同样 0 命中 —— 我把初稿 docstring 里的「常模」也改掉了,那条会被 Task 7 Step 1 的扫描打回)。

### 为什么 Task 6 内不可满足

- `_render_dimension_block` 按 brief 的规定渲染 `dim['display_name']`,而 5 个维度**无论是否有分都会被渲染**,
  所以 `<h3>抗压与情绪稳定性</h3>` 恒定出现。这不是 fixture 选得不好,是结构性的。
- 这两个字符串住在 `research_mapper.py`:`display_name` 是 Task 7 Step 3「改维度名与描述」的范围
  (spec §7.1 已由使用者拍板改成「情境行为稳定性」);Task 6 的 Files 只列了
  `report_generator.py` 与 `tests/test_report_layer.py`,Step 7 也只允许 `git add` 这两个路径。
- `human_name`「面部紧张度」**在 spec §7.1 与 Task 7 的改名表里都没有被拍板过新名字**
  —— 注意 Task 7 只改 `name` 与 `description`,不含 `human_name`。
  也就是说**即使 Task 7 按计划做完,这条断言仍然红**(Task 7 自己的 Step 1 扫描也会打到它)。

给出新名字属于替使用者拍板一项未决命名,故我**没有**这么做,也没在 `report_generator.py` 里
另加一层命名映射(那会与 brief 给定的 `_render_dimension_block` 冲突,并制造第二个命名单一真相源,
同时违反「本任务只删不加」)。

### 我的处置

把这条断言**移出本次提交**,并在测试文件原位留了一段中文注释说明它为何不在、什么时候补回
(不是静默删除;断言的完整原文见下)。取此处置是为了让本次提交**测试全绿**
(计划的 Step 5 验收是「全部 PASS」),而不是交付一个已知变红的套件。

### 请裁决(三选一)

1. **改名提前**:在 Task 6 内一并改 `research_mapper.py` 的 `display_name` 与 `human_name`
   (需先给「面部紧张度」拍一个不含禁止词的新名,例如「面部肌张力」;这会动到 Task 6 未列出的文件,
   并使 Task 7 Step 2 的 RED 部分提前变绿)。
2. **断言收窄**:把断言限定为"叙事层自己生成的句子"不含禁止词,允许渲染 mapper 的标签
   (需你确认 spec §5.6「禁止词不得出现在任何输出字符串」在 Task 6 阶段不覆盖标签)。
3. **接受延后**:断言等 Task 7 补回 —— 但需**同时**把 `human_name` 纳入 Task 7 的改名表,否则仍然红。

被移出的断言原文(裁决后原样补回即可):

```python
BANNED = ["焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
          "录用", "人格", "心理画像", "常模"]


def test_deep_analysis_has_no_banned_words():
    """spec §5.4 + §5.6:叙事层不得含情绪/心理/诚信构念。"""
    from report_frontend.report_generator import ReportGenerator

    feats = {"face": {"face_tension_score_mean": 0.5},
             "gesture": {"gesture_left_hand_jitter_mean": 0.02}}
    result = ResearchCapabilityMapper().map_features_to_scores(feats)
    html = ReportGenerator()._generate_deep_text_analysis(feats, result)

    for word in BANNED:
        assert word not in html, f"叙事层出现禁止词:{word}"
```

---

## 5. Step 6 端到端验证

```
~/miniconda3/envs/jingxin/bin/python -c "
import sys; sys.path.insert(0, '.')
from report_frontend import ReportGenerator
p = ReportGenerator(output_dir='/tmp/jxreport').generate_report()
print('报告:', p)
"
```

报告生成成功:`/tmp/jxreport/Research_Assessment_Report_20260922_230956.html`(7347 字节)。

> 注:该命令在本机**会挂住** —— `generate_report()` 末尾的 `webbrowser.open` 在 WSL 下不返回。
> 报告文件在 `webbrowser.open` **之前**就已写盘,所以产物完整;我用 `timeout` + 后台化处理,
> 未改任何代码。这是既有行为,与本次改动无关。

对生成 HTML 的机器校验(计划 Step 6 的四条人工核对,逐条脚本化):

| 核对项 | 结果 |
|---|---|
| 出现"证据不足" | ✅ 出现(共 6 次:5 个维度块 + 总分档位) |
| 无"情绪状态与抗压能力深度剖析"整节 | ✅ 未出现 |
| 无"未检测到焦虑特征" | ✅ 未出现 |
| 无"如外科医生般" | ✅ 未出现 |
| (附加)无"未出现异常的回避行为" | ✅ 未出现 |
| (附加)无"常模参照模型" / "毫秒级" | ✅ 均未出现 |
| (附加)无"最为突出" | ✅ 未出现 |
| (附加)分数未渲染为 `None` | ✅ 出现的是 `—` 与"证据不足" |

补充一条非验收项的观察:报告里仍有"综合科研潜力评分"这个**分数卡标签**
(在 `_build_html_report` 内,不在被重写的函数里)。它不含禁止词,
且计划把它归给 Task 7 Step 3b 的头部/标题字符串清理,故本次未动。

---

## 6. 自查

- **每个新测试都执行断言了吗?** 是。扫描:`存在未执行断言的测试数 = 0 (共 14 个测试)`;
  新增的 `test_deep_analysis_handles_none_scores` 1/1、`test_no_hardcoded_gaze_claim` 2/2 全部执行。
  计划里另有六条"通过但没钉住任何东西"的先例,本次未添第七条。
- **四条遗留项是否真的删除,而非只是不可达?** 逐条 grep 实测:
  | 项 | 结果 |
  |---|---|
  | 四处 `['percentile']` 默认 50 的杜撰常模 | ✅ 无任何 percentile 字段读取 |
  | 拼错的 `face_micro_exp_au_name_au4_freq` | ✅ 0 处 |
  | `.get(..., 0)` 驱动的「情绪控制力 / 心理稳定性」 | ✅ 0 处(连同"科研天赋""心理素质""最为突出"一起消失) |
  | 未使用的 import | ✅ AST 实测"未使用: 无" |
  这些是**整段删除**(`git diff` 为 -141 行),不是被短接跳过的。
- **零证据仍短接吗?** 是,且比原来更彻底:旧版靠函数顶部的 `if not scored: return` 过渡短接,
  新版**没有任何基于默认值的段落存在**,所以零证据下必然只输出诚实摘要 + 各维度"证据不足"。
  `test_zero_evidence_report_makes_no_claims` 通过(且该测试 2/2 断言执行)。
  说明:brief 给的顶层函数并未"提前 return",而是先压一句诚实摘要再逐维度渲染 ——
  这是 brief 指定的形状,性质 1 依然成立,故未偏离。
- **分数不再渲染 `None`?** `_score_display` 兜底保留(`:189` / `:239`),端到端 HTML 实测无 `>None<`。
- **测试输出干净?** `39 passed`,无 warnings summary;`-W error` 下同样通过。
- **注释风格**:中文 `#` 散文,与仓库一致。
- **提交卫生**:只 `git add` 了 brief 点名的两个路径;仓库里被解释器改动的 `__pycache__/*.pyc`
  与生成的 `report_frontend/data/output/*.html` 均**未入库**。

### 遗留(供控制器决定,未擅自处理)

`_get_percentile_badge`(`:116-125`)在重写后**已无任何调用方**,成为死代码。
它本身只格式化调用方给入的整数,不读任何被删字段,所以不是伪造源;但既然真实常模不存在
(spec §5.5),它目前没有正确用法。它在 brief 指定的替换范围(原 `:123-250`)之外,
且 Task 7 Step 5 有处理 `visualizer.py` 死代码的先例,故我保留并在此标记,未擅自删除。

---

## 7. 改动的文件

| 文件 | 改动 |
|---|---|
| `report_frontend/report_generator.py` | +50 / -141:新增 `_CONF_ORDER`、`_render_dimension_block`;整段重写 `_generate_deep_text_analysis` |
| `tests/test_report_layer.py` | Step 0 替换 2 个测试体;新增 2 个测试 + 一段说明注释(第 3 个测试待裁决) |

提交:
- `ea65285 test: 修复两条空断言(假简历兜底/封停列),并给出可过门的 fixture`
- `e3ed5ae refactor: 重写报告叙事层,删除硬编码解读`

---

# 附录:fix 记录(Ruling 26)

控制器裁决 **Ruling 26**:收窄版禁止词断言**采纳但须提交进去**(而不是像上一轮那样移出提交)。
理由:不收窄 = 卡住;不提交 = 叙事层自己的输出无人看守。

## 改了什么

- `tests/test_report_layer.py`:加回 `BANNED` 词表与收窄版
  `test_deep_analysis_has_no_banned_words`,替换掉上一轮那段"暂未落盘"的说明注释。
  收窄方式 = 在断言前对 HTML 做**精确字符串替换**,剔除 mapper 提供的两个已知标签
  (`"抗压与情绪稳定性"`、`"面部紧张度"`),从而把**叙事层自己写的句子**隔离出来测。
- `report_frontend/report_generator.py`:**未改**(本次提交只含测试文件)。

一处对 brief 的技术性更正:brief 的 docstring 写"见计划 Task 7 **Step 3b**",
但计划里 `Step 3b` 是"清理报告头部与标题字符串",恢复全量扫描实际是 **Step 3a**
(`plan:1429`)。我按控制器的口径写成 **Step 3a**,以免把后续实现者指到错的位置。

`import re` 仍未加 —— brief 的 Step 1 里有它但三个测试都没用到,加进去就是死导入
(本轮自查项明确要求清理未使用导入)。

## 覆盖测试与命令

```
~/miniconda3/envs/jingxin/bin/python -m pytest -q
```

输出:
```
40 passed in 0.30s
```

> 与控制器预期的 39 差 1:39 是上一轮**把该断言移出**时的数字;断言加回后总数 +1,故为 40。

断言执行情况(同一 `sys.settrace` 扫描):
```
✅ test_deep_analysis_has_no_banned_words: 1/1 条断言已执行
✅ test_deep_analysis_handles_none_scores: 1/1 条断言已执行
✅ test_no_hardcoded_gaze_claim: 2/2 条断言已执行
存在未执行断言的测试数 = 0 (共 15 个测试)
```

## 约束力验证(关键 —— 本计划已栽过六次)

"收窄"最大的风险是它悄悄退化成"任何实现都通过"。为证明它**仍有约束力**,
我临时在叙事层**自己写的字符串**里注入一个禁止词,而且刻意选在
**剔除逻辑够不着的位置**(不是那两个被剔除的标签,而是 `_render_dimension_block`
的"证据不足"段落):

```python
# report_frontend/report_generator.py,临时注入
<p><strong>证据不足</strong> —— 本次未采集到足以评估该行为线索的有效样本(紧张)。</p>
```

命令与输出:
```
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k banned -v
```
```
tests/test_report_layer.py::test_deep_analysis_has_no_banned_words FAILED [100%]
E           AssertionError: 叙事层出现禁止词:紧张
FAILED tests/test_report_layer.py::test_deep_analysis_has_no_banned_words - AssertionError: 叙事层出现禁止词:紧张
1 failed, 14 deselected in 0.25s
```

**RED 成立** —— 该断言确实会因为叙事层**自己在别处**写出的禁止词而变红,
剔除逻辑只抹掉了两个已知标签,没有削弱约束力。

随后 `git checkout -- report_frontend/report_generator.py` 还原注入,并复核:
- 注入词 0 残存(`grep -c "有效样本(紧张)"` = 0)
- `report_frontend/report_generator.py` 工作区干净(无未提交改动)
- 全套件回到 `40 passed`

**注入未被提交**,`report_generator.py` 在本次提交里零改动。

## 提交

`38360a4 test: 补回收窄版禁止词断言,隔离 mapper 标签`(1 file changed,+30 / -7)

未 amend `e3ed5ae` —— 按裁决另起 fix 提交。

## 遗留(未变)

`_get_percentile_badge` 仍是死代码,标记见正文 §6。
Task 7 的待办(控制器已同步进计划,Task 6 不用管):
Step 3 改名表新增 `human_name`「面部紧张度」;Step 3a 改名完成后**删掉剔除逻辑、恢复全量扫描**,
验收"恢复全量后该测试仍须 PASS"。

---

## fix 轮 1(Step 8)

**状态:DONE** —— 工作区里上一轮未提交的 Step 8 改动逐条对齐 brief 后**全部保留**,
补齐验收证据 2/3/4,提交 `ae85bfc`(父提交 `38360a4`)。

### 1. 逐条核对 (a)–(f):已在 vs 需补

| 项 | 工作区现状 | 我的处置 |
|---|---|---|
| (a) `test_no_hardcoded_gaze_claim` 换 `_MIXED` | 已在 | 保留(一处偏离,见下) |
| (b) `test_deep_analysis_has_no_banned_words` 换 `_MIXED` | 已在 | 保留 |
| (c) `test_deep_analysis_handles_none_scores` 换 `_MIXED` | 已在 | 保留 |
| (d) `test_quarantined_columns_are_rejected` 改断「对称性进缺口」 | 已在,与 brief 逐字一致 | 保留,另做一次可证伪实验(§5) |
| (e) 缺口重复渲染 | 已在,但与 brief 字面做法**不同** | 保留(一处偏离,见下) |
| (f) 删 `_get_percentile_badge` | 已在 | 保留;全仓 grep 只剩 docs 引用,无调用方 |

**没有发现缺失项,没有回退重做。** 三条 `_MIXED` 前置断言
(`assert result["total_score"] is not None, "fixture 未出分…"`)是这轮修复的核心:
它们把"fixture 悄悄退化回零证据"从静默变成响亮的红。

#### 偏离 1:(a) 的 `features` 实参由 `{}` 改成 `_MIXED`

brief 写"`_generate_deep_text_analysis({}, result)` 保持",工作区改成了 `_MIXED`。
Step 8 开头把修法称作"**统一**修法:把这三条测试的 fixture 换成混合 fixture",
且当前实现完全不读 `features`(docstring 已注明"参数 features 保留是为了调用方签名稳定"),
故二者当前等价;若将来实现重新从原始特征取值,传 `_MIXED` 只会更严不会更松。按"统一"口径保留。

#### 偏离 2:(e) 的短接路径 —— brief 的前提与代码不符

brief 写:出分路径删顶层段,**短接路径仍需要它(那时不渲染任何维度)**。工作区把顶层段**无条件删除**,
改为**给出分维度也渲染自己的缺口**。我按代码核实了那个前提:

- `_generate_deep_text_analysis` 对 `result["dimensions"]` **无条件遍历**,短接路径下 5 个维度块**照样渲染**
  (`score is None` 分支本身就带 `<ul>{gaps}</ul>`)。"那时不渲染任何维度"不成立。
- 按 brief 字面做,短接路径会**保留重复**(逐维 None 分支 + 顶层段各打一遍),而真实报告走的正是短接路径
  (§4 实测:5 维全部 `证据不足`)—— 等于只修了不重复的那条路径,留下重复的那条。
- 信息不丢:`all_evidence_gaps` 就是各维缺口的拼接(`research_mapper.py:159` 逐维 `extend`),
  实测 `并集 == 各维拼接: True`。

故保留工作区做法。**这是本轮唯一一处未按 brief 字面执行的地方**,理由如上,请复审确认。

### 2. 验收证据 2:三条注入式 RED(实测,非声称)

三条测试守着的缺陷都住在**出分路径**;零证据 fixture 在到达那些句子之前就被短接。
做法:临时把被禁的东西**注回出分路径** → 跑测试 → 确认按**正确理由**变红(AssertionError,非 AttributeError/KeyError)→ 还原。

备份与还原(两文件均 uncommitted,全程只用 `cp`,**未用** `git checkout/stash/restore`):
```
cp report_frontend/report_generator.py /tmp/rg_baseline.py    # sha256 c3f137b7...
cp tests/test_report_layer.py          /tmp/tl_baseline.py    # sha256 16af785a...
还原判据:git diff <两文件> | sha256sum == 35ed7176...  (实验前快照)
```

#### (a) `test_no_hardcoded_gaze_claim`

注入(出分分支内):
```python
parts.append("<p>微表情监测未检测到显著特征,未出现异常的回避行为,如外科医生般精准。</p>")
```
```
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k hardcoded_gaze -q
```
```
>       assert "未出现异常的回避行为" not in html
tests/test_report_layer.py:254: AssertionError
1 failed, 14 deselected in 0.30s
```
**理由正确**:命中的正是被守的那条断言(第 254 行),不是 AttributeError/KeyError;
且它上一行的 `assert result["total_score"] is not None` 已先通过 —— 证明 fixture 确实走在出分路径上。

**Ruling 27 对照(旧 fixture 抓不到同一个注入)**:同一份被注入的源码,用旧版零证据测试体:
```
旧 fixture 结果: PASS —— 注入的硬编码句【未被抓到】(total_score=None)
```

#### (b) `test_deep_analysis_has_no_banned_words`

注入(出分分支内,刻意用**逐字新句子**,不在被剔除的两个 mapper 标签里):
```python
parts.append("<p>该维度体现求职者的抗压能力。</p>")
```
```
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k banned -q
```
```
>           assert word not in html, f"叙事层出现禁止词:{word}"
E           AssertionError: 叙事层出现禁止词:抗压
E             '抗压' is contained here:
E               <p>综合行为观测摘要：卓越(置信度上限：低)</p><p>该维度体现求职者的抗压能力。</p>
1 failed, 14 deselected in 0.30s
```
**理由正确**:AssertionError 带的就是"叙事层出现禁止词",且剔除逻辑(精确字符串替换两个 mapper 标签)
没能抹掉它 —— 收窄后的断言对"叙事层自己在别处写的禁止词"仍有约束力。
旧 fixture 对照:同样 `PASS —— 注入的禁止词【未被抓到】 | total_score = None`。

#### (c) `test_deep_analysis_handles_none_scores`

本条守的是"`score` 为 `None` 的维度不得参与排序、且须渲染成证据不足",而
`max(..., key=_CONF_ORDER.get)` 那一行只有出分路径才走到。两个注入各证一半:

**(c-i) 回退 fixture(即回退本轮改动)** —— 把测试体里的 `_MIXED` 还原成旧版 `{}`:
```
>       assert result["total_score"] is not None, "fixture 未出分,本测试守不住 conf_cap 那一行"
E       AssertionError: fixture 未出分,本测试守不住 conf_cap 那一行
E       assert None is not None
tests/test_report_layer.py:237: AssertionError
1 failed, 14 deselected in 0.29s
```
**理由正确**:AssertionError 指名"fixture 未出分" —— 回退本轮改动即变红,直接满足验收要求 2。

**(c-ii) 把被禁行为注回出分路径** —— 让 `None` 维度不参与渲染(即"不渲染成证据不足"):
```python
for dim_key, dim in result["dimensions"].items():
    if dim["score"] is None:
        continue          # 临时注入
    parts.append(self._render_dimension_block(dim_key, dim))
```
```
>       assert "证据不足" in html
E       AssertionError: assert '证据不足' in '<p>综合行为观测摘要：卓越(置信度上限：低)</p>...'
tests/test_report_layer.py:240: AssertionError
1 failed, 14 deselected in 0.35s
```
**理由正确**:AssertionError(**不是** TypeError/AttributeError/KeyError),
同时证明该断言在 `_MIXED` 下确实活到出分路径。

#### 还原验证(三次实验共用)

```
rg   sha: c3f137b7bedb1dbd732626ab77d1207a7e404efa8f500a502a01a5ffcbe6893f  ← 与备份一致
tl   sha: 16af785a72ec8941a68e4aabbc7493e2e7a263b01cd045203c489ccf0d7c5811  ← 与备份一致
diff sha: 35ed7176ae83b30115fa0a4b235e085cc14f46a7f8b4884e3a8d1e769a2a788b  ← 与实验前快照一致
```
三个注入字符串在最终提交里 0 残留(`grep` 无命中)。

### 3. 验收证据 3:空断言扫描(Step 0 脚本,重跑)

脚本 `/tmp/jx_sweep.py`(不入库;与 §1 同一段口径:`sys.settrace` 记录实际执行行 ∩ AST `assert` 行):
```
~/miniconda3/envs/jingxin/bin/python /tmp/jx_sweep.py
```
```
✅ test_zero_input_produces_no_scores: 4/4 条断言已执行
✅ test_no_fake_resume_fallback: 5/5 条断言已执行
✅ test_quarantined_columns_are_rejected: 4/4 条断言已执行    ← (d) 后由 3 条变 4 条
✅ test_confidence_can_be_none_and_low: 1/1 条断言已执行
✅ test_dimension_weights_sum_to_one: 1/1 条断言已执行
✅ test_slot_level_quarantine_covers_spec_5_3: 1/1 条断言已执行
✅ test_zero_evidence_report_makes_no_claims: 2/2 条断言已执行
✅ test_all_slot_level_quarantine_keys_are_pinned: 1/1 条断言已执行
✅ test_question_index_is_not_dropped: 1/1 条断言已执行
✅ test_dead_fluency_branch_removed: 2/2 条断言已执行
✅ test_no_fabricated_percentile: 2/2 条断言已执行
✅ test_radar_has_no_norm_baseline: 2/2 条断言已执行
✅ test_deep_analysis_has_no_banned_words: 2/2 条断言已执行   ← (b) 新增 fixture 前置断言
✅ test_deep_analysis_handles_none_scores: 3/3 条断言已执行   ← (c) 新增两条前置断言
✅ test_no_hardcoded_gaze_claim: 3/3 条断言已执行             ← (a) 新增 fixture 前置断言

存在未执行断言的测试数 = 0 (共 15 个测试)
```
**Step 8 验收 3 达成:存在未执行断言的测试数 = 0。**

### 4. 验收证据 4:缺口不再重复渲染

真实报告(实际走的是**短接路径**,该 session 5 维全部 `score=None`):
```
~/miniconda3/envs/jingxin/bin/python -c "... ReportGenerator(output_dir='/tmp/jxreport8').generate_report() ..."
→ /tmp/jxreport8/Research_Assessment_Report_20260924_110101.html
```
在同一份 `result` 上脚本化计数(非目测):
```
result['evidence_gaps'] 总条数 = 20  (各维缺口的并集)
并集 == 各维拼接: True
[now]    20/20 条缺口在叙事层 HTML 里恰好出现 1 次; 非 1 次的 = {}
[before] 把已删除的顶层「证据缺口」段原样拼回: 出现次数分布 = {2: 20}   ← 全 2 次 = 重复

含「对称」的缺口 = ['面部对称性: 有效样本不足']
  now=1  before=2
  上下文: ...<h3>抗压与情绪稳定性</h3> <p><strong>证据不足</strong> …</p> <ul>…<li>面部对称性: 有效样本不足</li></ul>

真实报告文件: 含「对称」缺口 1 次; '<h3>证据缺口</h3>' 标题 0 次; 每条缺口出现次数全为 1: True
```
出分路径另测(brief 实测的 19 条)也每条 1 次,并带维度归属:
```
出分路径(_MIXED): 缺口总条数 = 19 | 每条出现次数 = 全 1
  缺口块归属维度 h3: 逻辑思维与专注度      ← 出分维度的 3 条缺口挂在它自己的标题下
```
**验收 4 达成:同一批字符串不再出现两遍,且缺口现在带维度归属。**

### 5. 额外:item (d) 的可证伪实验(该项没有别的验证渠道)

(d) 修的是"结构上不可证伪"的那条断言,故单独跑一次 RED:
临时删掉 `evidence_gate.py` 的 `"symmetry_score"` 封停条目(该文件先 `cp` 备份到 `/tmp/eg_baseline.py`):
```
>       assert sr["evidence_chain"] == [], "封停列进了链"
E       AssertionError: 封停列进了链
E       assert [{'feature': 'face_symmetry_score_mean', ...}] == []
tests/test_report_layer.py:66: AssertionError
1 failed, 14 deselected in 0.27s
```
还原:`evidence_gate.py` sha256 `aca19233...` 与备份一致,`git status --short` 对该文件无输出(**未进提交**)。

### 6. 覆盖测试与命令

```
~/miniconda3/envs/jingxin/bin/python -m pytest -q            →  40 passed in 0.29s
~/miniconda3/envs/jingxin/bin/python -m pytest -q -W error   →  40 passed in 0.30s
```
输出干净:无 warning、无 skip、无 xfail。

### 7. 改动的文件与提交

```
report_frontend/report_generator.py | 28 +++++++---------
tests/test_report_layer.py          | 64 ++++++++++++++++++++++++++++---------
2 files changed, 61 insertions(+), 31 deletions(-)
```
`ae85bfc test: Step 8 修复三条空断言 fixture 与缺口重复渲染,删除死代码 _get_percentile_badge`
父提交 `38360a4`,**未 amend、未 rebase**;只 `git add` 了这两个路径 ——
仓库里 ~64 个 tracked-but-deleted `.pyc`、被测试运行刷新的 `.pyc`、`code_data_supplement/`、`data/` 等噪音一律未动。

### 8. 自查与遗留

- **自查**:三处注入 0 残留;两文件 `py_compile` 通过;`import re` 未加(三个测试都不用它,加了就是死导入);
  LF 行尾保持;未新增依赖;未把第三个文件带进提交(`evidence_gate.py` 实验后已还原并留证)。
- **遗留 1(观察)**:`_get_percentile_badge` 已删且无调用方(全仓 grep 只剩 `docs/` 里的引用文字)。
- **遗留 2(观察,未处理)**:(e) **没有测试钉住** —— 它的证据是 §4 的脚本化计数与报告生成(验收 4),
  没有任何一条 pytest 会在"顶层段被加回来"时变红。brief 把 (e) 列为 Minor 且未要求补测,
  我未擅自加测(Task 6 约束是"只删不加");若控制器认为该性质必须常驻,建议另开一条断言:
  数某条缺口在 `_generate_deep_text_analysis` 输出里的出现次数 == 1。
- **遗留 3(观察,未处理)**:`result["evidence_gaps"]`(顶层并集)在报告层**已无消费者**
  (全仓 grep:只剩 `research_mapper` 产它、`test_zero_input_produces_no_scores` 断它非空)。
  这是 (e) 的自然结果,不在本轮范围。
- **遗留 4(观察,未处理)**:真实报告走的是短接路径(该 session 5 维全 `证据不足`),
  出分路径的端到端报告仍只由 `_MIXED` 级 fixture 覆盖 —— 没有真实数据走过出分路径的报告渲染。
