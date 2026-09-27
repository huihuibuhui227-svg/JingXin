# M3.0 L0 列表 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把「≈54 列」从一个文档里的估数,变成一张**能开工、且能被机器检查**的 L0 列清单,并顺手补 3 个文档要求却零实现的新列、删掉 4 个死代码文件。

**Architecture:** 清单落成仓库根的**数据文件** `l0_columns.json`(与 `media_retention.py` / `session_meta.py` 同层,三服务 + 报告层共用的横切事实),配一个极薄加载器 `l0_columns.py` 与一条**双向覆盖钉子**。钉子把"日志产出的列"与"表里的行"焊在一起:任一边多一列、少一列都红。逐列定义**按模态分任务**填表。新列一律加在**日志列序末尾**。

**Tech Stack:** Python 3 / pytest(`pytest.ini` 只设了 `testpaths = tests`)/ 三份 CSV logger(face `DataLogger`、gesture `GestureLogger`、voice `VoiceLogger`)/ librosa(pyin)。

**Spec:** `docs/superpowers/specs/2026-09-26-l0-column-table-design.md` —— 本计划实现它,执行者**两份都要读**。

## Global Constraints

- **解释器固定**:`~/miniconda3/envs/jingxin/bin/python`(不可用 `~/huihui/bin/python`,缺 websockets 等)。
- **合并门必须仍然过**:`0 / 2835510`。2026-09-26 实测确认:`最大绝对差 1.886e-19`,`相对差 > 1e-4 的格子: 0 / 2835510`。任何动了采集层的任务跑完都要复跑(命令见 Task 10)。
- **全量套件不减少**:基线 **410 passed**。
- **不许编数据**。没测到就写**空**,不写 `0`、不写 `50`、不写默认值 —— 这是本项目反复栽的形态。日志侧已是这个约定(`_safe_get` 的 `obj is None → ""`)。
- **新列必须加在列序末尾**。插在中间会让老 CSV 文件的列序对不上(`tests/test_face_blendshapes.py:185-186` 明确钉过这件事)。
- **每个测试都要能说出「哪个生产改动会让它变红」**,说不出来就是没约束力(`docs/下一步.md` §4.1)。**反向复现是标准动作**:写完把生产改动撤掉,确认新测试真的红,再恢复;变异验证**必须清 `__pycache__`**(`§4.3`)。
- **阈值/常量一律进数据文件**(带 `_provisional` + 依据),不写死在代码里。
- **提交纪律**:只 `git add` 本任务的文件,**不许 `git add -A`**。
- ★ **提交范围(2026-09-26 使用者裁定)**:**代码提交照做**;**`docs/**` 一律先不提交**。
  - 哪些算"代码":`.py`、`.json`(含 `l0_columns.json` —— 它是**代码读、测试验**的数据文件,不是文档)、`tests/`。
  - 哪些算"文档"、**本里程碑不提交**:`docs/superpowers/specs/**`、`docs/superpowers/plans/**`、
    `docs/superpowers/sdd/**`、`docs/下一步.md`。
  - **Task 10 Step 6 那个提交整条跳过**(它全是文档)。其余任务若提交里混了文档,**只 add 代码部分**。
- 本里程碑**不改任何既有列的语义**、**不解封** `evidence_gate.QUARANTINE` 的 35 条、不碰 L1。
- **与上游 `2026-09-21` 设计冲突时,以上游为准**(spec §10 风险 5):本计划只负责把"没写的写下来",不改上游的任何裁定。
- **B 档行必须写 `basis`**(为什么这么定)—— spec §10 风险 1。定不下来的一律**降为 C**,不硬编。

## Review Focus

规格说明"该做什么",但对下面这五类输入是沉默的 —— 沉默不等于允许它坏。每一条都已在本计划里指定了归属任务的那个测试。

1. **"没测到"被写成 0** —— 新列在一帧没有数据时,如果落了 `0.0` 而不是空串,报告层会把它当"测到了 0"(本项目 `jitter=0 → 显示 100% 稳定`、`is_valid=True 的零值行` 都是这个形态)。**归属:Task 5/6/7 各自的空值测试。**
2. **`hand_visible_ratio` 在"兜底分槽"那一帧** —— 手势端点在模型没给 handedness 时,会把一只**来路不明**的手塞进空槽并把 `handedness_info[槽] = None`(`gesture_analysis/api/app.py:342-349`)。此时"槽非空"只说明**有手**,不说明**那是左手**。把它算作"左手可见"是错的。**归属:Task 6。**
3. **全静音段的 `voiced_prob` 均值** —— `pyin` 对全静音返回全 `False` 的 `voiced_flag` 与全 0 的概率。此时均值 `0.0` 是**真值**(确实一个浊音帧都没有),但 `f0` 全 `nan` ⟹ `pitch_mean` 该是空。两者必须分别处理,不能一起写成 0。**归属:Task 5。**
4. **老日志文件的列序** —— 新列插在中间会让历史 CSV 的列与表头错位,而报告层按列名读、错位后**静默读到别的列的值**。**归属:Task 5/6/7 的"末列位置"断言。**
5. **`legacy_allowlist` 变成万能垃圾桶** —— 它是给"还没进表的既有列"用的豁免名单。一旦允许无理由地往里加,双向钉子就废了。**归属:Task 1(每条必须带 `why` 与 `planned`,且钉子断言这一点)。**

---

### Task 1: `l0_columns.json` + 加载器 + 双向覆盖钉子(机制先立起来)

**Files:**
- Create: `l0_columns.json`
- Create: `l0_columns.py`
- Test: `tests/test_l0_column_table.py`

**Interfaces:**
- Consumes: 三份 logger 的列契约 —— `face_expression.utils.logger.DataLogger(log_type='video', session_id=...)`、`gesture_analysis.utils.logger.GestureLogger(session_id=...)`、`voice_interaction.utils.logger.VoiceLogger(log_type='interview', session_id=...)`;三者都有 `self.fieldnames`(list[str])。
  ⚠️ **落盘文件属性名不一致**(2026-09-26 实测):`DataLogger` 与 `GestureLogger` 叫 **`log_file`**,
  **只有** `VoiceLogger` 叫 `csv_file`。Task 5/6/7 写测试时别照抄错的那个。
- Produces:
  - `l0_columns.load() -> dict`(**深拷贝**,照 `connective_density.load_markers` 的先例)
  - `l0_columns.columns() -> list[dict]`
  - `l0_columns.schema_errors(doc: dict) -> list[str]`(空列表 = 合格)
  - JSON 的顶层键:`_schema` / `count_reconciliation` / `legacy_allowlist` / `columns`

- [ ] **Step 1: 写失败测试(两个方向各一条 + schema)**

```python
# tests/test_l0_column_table.py
"""M3.0 的双向覆盖钉子 —— 日志产出的列 <-> l0_columns.json 的行。

方向 1(产出 ⊆ 表):日志里出现的每一列,要么在表里,要么在 legacy_allowlist 里。
方向 2(表 ⊆ 产出):表里标 status=implemented 的每一列,必须真的在日志里。
两条都是**双向**的:任一边多一列、少一列都要红。
"""
import csv
import importlib

import pytest

l0 = importlib.import_module("l0_columns")


def _live_columns():
    """三份日志现在真的会写出的列名(构造 logger 即可,不写行)。"""
    from face_expression.utils.logger import DataLogger
    from gesture_analysis.utils.logger import GestureLogger
    from voice_interaction.utils.logger import VoiceLogger
    return {
        "face": list(DataLogger(log_type="video", session_id="x").fieldnames),
        "gesture": list(GestureLogger(session_id="x").fieldnames),
        "voice": list(VoiceLogger(log_type="interview", session_id="x").fieldnames),
    }


def test_every_live_column_is_either_in_the_table_or_allowlisted():
    """方向 1。红法:往任一 logger 的 fieldnames 里加一列而不进表/白名单。"""
    doc = l0.load()
    tabled = {(c["column"], c["modality"]) for c in doc["columns"]}
    allowed = {(a["column"], a["modality"]) for a in doc["legacy_allowlist"]}
    unaccounted = []
    for modality, cols in _live_columns().items():
        for col in cols:
            if (col, modality) not in tabled and (col, modality) not in allowed:
                unaccounted.append(f"{modality}.{col}")
    assert not unaccounted, (
        "这些列既不在表里、也不在 legacy_allowlist 里:\n  "
        + "\n  ".join(unaccounted)
        + "\n—— 要么给它们写一行定义,要么进白名单并写明为什么、归哪个里程碑。")


def test_every_implemented_row_really_is_produced():
    """方向 2。红法:把某行标成 implemented 而它其实不在 fieldnames 里。"""
    live = _live_columns()
    bad = [f"{c['modality']}.{c['column']}"
           for c in l0.columns() if c["status"] == "implemented"
           and c["column"] not in live.get(c["modality"], [])]
    assert not bad, f"表里标了已实现、日志里却没有:{bad}"


def test_allowlist_entries_carry_a_reason_and_a_target():
    """白名单不许变成万能垃圾桶:每条都要有 why 与 planned。"""
    for a in l0.load()["legacy_allowlist"]:
        assert a.get("why", "").strip(), f"{a.get('column')} 缺 why"
        assert a.get("planned", "").strip(), f"{a.get('column')} 缺 planned"


def test_schema_errors_is_empty_on_the_shipped_file():
    assert l0.schema_errors(l0.load()) == []


def test_load_hands_back_a_copy():
    """★ 红法:把 `load()` 改成 `return _CACHE`(去掉 deepcopy)⟹ 本测试必须红。

    ⚠️ 探针**不许用 `columns`**:出厂表里它是空列表(`"columns": []`),
    `a["columns"].clear()` 在两种实现下都留空 ⟹ **零区分力**
    (2026-09-26 实测踩过:这条测试红的原因不是它声称守护的那件事)。
    用 `_schema.version` —— 出厂表里就有值,与填表进度无关。
    """
    a = l0.load()
    a["_schema"]["version"] = "TAMPERED"
    assert l0.load()["_schema"]["version"] != "TAMPERED", "load() 交出了本体,调用方能改坏它"
```

- [ ] **Step 2: 跑测试,确认它红(模块还不存在)**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_l0_column_table.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'l0_columns'`

- [ ] **Step 3: 写加载器 `l0_columns.py`**

```python
# l0_columns.py
"""M3.0 的 L0 列清单(spec: docs/superpowers/specs/2026-09-26-l0-column-table-design.md)。

与 `media_retention.py` / `session_meta.py` 同层、同在仓库根:它描述的是
**三服务 + 报告层共用的横切事实**,不属于任何一个模态包。

**为什么是数据文件而不是 markdown 表格**:markdown 表格没有约束力。本项目栽过
「spec 说 ≥2 段、代码门槛是 5」这种互相矛盾(spec §3.1),靠 grep 才抓到。
有了 `tests/test_l0_column_table.py` 的双向钉子,这类矛盾**不可能静默存在**。
"""
import copy
import json
from pathlib import Path

TABLE_PATH = Path(__file__).resolve().parent / "l0_columns.json"

_REQUIRED_FIELDS = ("column", "modality", "definition", "source", "normalization",
                    "unit", "maturity", "status", "acceptance")
# B 档 = "缺的只是一个现在就能定的参数"。既然是**我们**定的,就必须写下依据 ——
# 否则它和"编一个值"没有区别(spec §10 风险 1)。
_BASIS_REQUIRED_FOR = ("B",)
MATURITIES = ("A", "B", "C")
STATUSES = ("implemented", "pending", "blocked")
MODALITIES = ("face", "gesture", "voice", "text", "covariate")

_CACHE: dict | None = None


def load() -> dict:
    """读出整张表。**交深拷贝** —— 调用方改坏它不该影响下一次读。"""
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    return copy.deepcopy(_CACHE)


def columns() -> list[dict]:
    return load()["columns"]


def schema_errors(doc: dict) -> list[str]:
    """结构错误清单。空列表 = 合格。**只判结构,不判定义对不对** ——
    定义对不对由 Task 2/3/4 的人工审与后续里程碑负责。"""
    errs: list[str] = []
    for row in doc.get("columns", []):
        name = row.get("column", "<无名>")
        for field in _REQUIRED_FIELDS:
            if not str(row.get(field, "")).strip():
                errs.append(f"{name}: 字段 {field} 为空")
        if row.get("maturity") not in MATURITIES:
            errs.append(f"{name}: maturity={row.get('maturity')!r} 不在 {MATURITIES}")
        if row.get("status") not in STATUSES:
            errs.append(f"{name}: status={row.get('status')!r} 不在 {STATUSES}")
        if row.get("modality") not in MODALITIES:
            errs.append(f"{name}: modality={row.get('modality')!r} 不在 {MODALITIES}")
        if row.get("maturity") == "C" and not str(row.get("l0_output", "")).strip():
            errs.append(f"{name}: maturity=C 必须写清 l0_output(L0 到底吐什么)")
        if row.get("maturity") in _BASIS_REQUIRED_FOR and not str(row.get("basis", "")).strip():
            errs.append(f"{name}: maturity={row['maturity']} 必须写 basis(这个值是我们定的,凭据在哪)")
        # ★ 预检裁定 F2(2026-09-26):标了 implemented 的行,modality 必须是**产出它的服务**。
        #   理由:钉子方向 2 拿 (column, modality) 去比 `_live_columns()` 的 key,
        #   而那个 dict 只有 face/gesture/voice 三个键 —— 填 `covariate` 的行永远匹配不上,
        #   一旦标 implemented 就会**假红**。"协变量用途"是用途,写在 definition 里。
        if row.get("status") == "implemented" and row.get("modality") not in ("face", "gesture", "voice"):
            errs.append(f"{name}: status=implemented 但 modality={row.get('modality')!r} "
                        f"—— 已实现的列必须标产出它的服务(face/gesture/voice)")
    if not doc.get("count_reconciliation"):
        errs.append("缺 count_reconciliation —— 「54」这个数必须登记来源与差异")
    return errs
```

- [ ] **Step 4: 写 `l0_columns.json` 的首版(骨架 + 白名单 + 已知行)**

`legacy_allowlist` 的 `why` 一律填**这一列当前真实的拦截原因**,`planned` 填归属里程碑。
`columns` 先只放**已有结论的行**(新列见 Task 5/6/7/8;逐列定义见 Task 2/3/4)。

```json
{
  "_schema": {
    "version": "1.0.0",
    "_provisional": true,
    "basis": "M3.0 spec(2026-09-26)。列定义来自 2026-09-21 特征重构设计 §4 的处置表 + 2026-09-26 的逐行核实。",
    "fields": ["column", "modality", "definition", "source", "normalization",
               "unit", "maturity", "l0_output", "status", "quarantine_ref", "acceptance"]
  },
  "count_reconciliation": {
    "doc_estimate": {"face": 22, "gesture": 13, "voice": 16, "text": 3},
    "actual": {"face": 0, "gesture": 0, "voice": 0, "text": 0},
    "deltas": [],
    "note": "actual 由 Task 2/3/4 逐列定义后填写;deltas 必须逐条写明差异原因(spec §4.3)。"
  },
  "legacy_allowlist": [
    {"column": "focus_score", "modality": "face",
     "why": "3 个硬编码取值,99.7% 恒 0.3(QUARANTINE 已封停)",
     "planned": "M3.2"},
    {"column": "symmetry_score", "modality": "face",
     "why": "未除人脸尺度,近常量,取景代理(QUARANTINE 已封停)",
     "planned": "M3.2"}
  ],
  "columns": []
}
```

- [ ] **Step 5: 跑测试。前三条应过,后两条视表内容**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_l0_column_table.py -v`
Expected: `test_load_hands_back_a_copy` / `test_allowlist_entries_carry_a_reason_and_a_target` PASS;
`test_schema_errors_is_empty_on_the_shipped_file` PASS;方向 1 那条**应为 FAIL** ——
它会把三份日志的全部列列出来当"未登记"。**这是对的**:那正是 Task 2/3/4 要填的东西。
把 FAIL 的输出**存下来当待办清单**(存进 `docs/superpowers/sdd/2026-09-26-m3-0-l0-column-table/progress.md`)。

- [ ] **Step 6: 反向复现(强制)**

把 `face_expression/utils/logger.py:43-59` 的 `fieldnames` 列表里**加一个假列** `"__probe__"`,
跑 `test_every_live_column_is_either_in_the_table_or_allowlisted` → **必须红**。
再删掉它,确认回绿。**清 `__pycache__` 后再跑一次**(`docs/下一步.md` §4.3)。

- [ ] **Step 7: Commit**

```bash
git add l0_columns.json l0_columns.py tests/test_l0_column_table.py
git commit -m "feat(m3.0): L0 列清单的载体与双向覆盖钉子"
```

---

### Task 2: 填表 —— 面部

**Files:**
- Modify: `l0_columns.json`(`columns` 追加 `modality: "face"` 的行;`legacy_allowlist` 补齐面部落选列)
- Modify: `l0_columns.json` 的 `count_reconciliation.actual.face` 与 `deltas`
- Test: `tests/test_l0_column_table.py`(复用,不新增文件)

**Interfaces:**
- Consumes: Task 1 的 `l0_columns.load()` / `schema_errors()`;Task 1 的钉子
- Produces:`columns` 里全部 `face` 行(供 Task 9 的收尾与后续 M3.2 使用)

**原料**:`docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md:159-211`(§4.1 两表,含「L0 形态」栏)+ `:212-214`(删 `_trend`/`_volatility`/`_change_rate` 84 列那行)。

- [ ] **Step 1: 把 §4.1 的每一行落成表行**

规则(照 spec §4.2/§4.3,不许现编):

- `maturity: A` 需**同时**满足:① 定义含公式或精确 landmark 下标 ② 含归一化分母 ③ **不需要任何标定集常量**。
- 只缺一个"现在就能定的参数" ⟹ `B`,并在 `definition` 里写下**你定的那个值**与 `basis` 附注。
- 定义里有一个量只能来自标定集(阈值/分位映射范围) ⟹ `C`,**且 `l0_output` 必须写清 L0 到底吐什么原始量**。
- 灰区一律判 **C**(spec §4.2)。

**已核实的定档(照抄,别推翻)**:

| 列 | §4.1 原文 | maturity |
|---|---|---|
| `au1_inner_brow_raise` | 「是否减视频内中位数**由标定定**」 | `C`,`l0_output`:吐未中心化的原值 |
| `au9_nose_wrinkle` | 「改用 129/358」 | `B`(有下标、**无归一化分母**) |
| `au26_jaw_drop` | 「÷ face_height,再用**标定集的 p1/p99** 线性映射」 | `C`,`l0_output`:吐 `(下巴−上唇竖直距) ÷ face_height` |
| `au23_lip_compression` | 「唇红厚度(上唇上缘→唇缝)」 | `B`(无下标、无分母) |
| `au4_frown` | 「双眉内侧点距(107/336 **或** 52/55)÷ 眼距」 | `B`(两套下标未定) |
| `head_yaw` / `head_pitch` | 「solvePnP 真 3D」 | `B` 或 `blocked` —— **见 Task 8** |
| `head_roll` | 「新增,solvePnP 顺带」 | **见 Task 8** |

删除类的行(`au2`/`au7`/`au14`/`au20`/`blink_rate_per_min`/`eye_closed_sec`/`gaze_direction_y`/`gaze_deviation`)**不进 `columns`**,进 `legacy_allowlist`,且 `why` 抄 §4.1 的「依据」栏原文、`planned` 写 `M3.2 删列`。

- [ ] **Step 2: 逐行核原文,不许照抄我上面的表**

对上表里**没有**列出的行,去 §4.1 读原文再定档。⚠️ 已知陷阱:`au9` 的 `129/358` 在
「L0 形态」栏,而 `234/455` 在「依据」栏 —— 别引错栏(2026-09-26 自审时就差点引错)。

- [ ] **Step 3: 填 `count_reconciliation`**

★ **口径先定死(预检裁定 C5)**:`actual` 数的是 **spec 口径的 L0 目标测量列**(≈54 那个口径)。
**不数** `session_id` / `timestamp` / `timestamp_iso` / `is_valid` 这类非测量列;
**不数** 52 个 `bs_*` —— 那批是 2026-09-26 新增的 blendshape 近似,2026-09-21 的设计表**早于它**,
所以它们进 `legacy_allowlist`(`why`:新增于 2026-09-26、设计表未覆盖;`planned: M3.2`)。
⚠️ 钉子方向 1 覆盖的 **187 条**是**日志列契约全量**,与这里数的"测量列"**不是一个口径** ——
不要拿 187 去对 `actual`。

`actual.face` = `len([c for c in columns if c["modality"] == "face"])`。
`deltas` 追加一条:文档估 **22**、实际 **N**、原因(例如「iris 一行拆 4 列」「`is_valid`/`session_id`/`timestamp` 三个非测量列不计入」——这两个都是 spec §1 点名的**未声明决定**,必须写出来)。

- [ ] **Step 4: 跑测试**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_l0_column_table.py -v`
Expected: `test_schema_errors_is_empty_on_the_shipped_file` PASS;方向 1 的 FAIL 列表**面部部分应消失**。

- [ ] **Step 5: 反向复现**

挑一行,把 `definition` 改成空串 → `test_schema_errors_is_empty_on_the_shipped_file` **必须红**;恢复。

- [ ] **Step 6: Commit**

```bash
git add l0_columns.json
git commit -m "feat(m3.0): L0 表填面部逐列定义 + 计数差异登记"
```

---

### Task 3: 填表 —— 手势(含「手部 10 列」)

**Files:**
- Modify: `l0_columns.json`(`modality: "gesture"` 的行 + 白名单 + `count_reconciliation`)

**原料**:`2026-09-21-...-design.md:215-227`(§4.2,⚠️ **该表没有「L0 形态」栏**,要自己定)+ 现行 67 列的实际列名(`gesture_analysis/utils/logger.py:57-161`)。

**难点(必须先解决,不许绕过)**:§4.2 只写「手部 10 列 | 重构 | …」,**没说重构成什么**。
这 10 列是 `{left,right}_hand_{score,jitter,fist_status,spread}` + 2 个 model_label(_conf)。定义它们时:

- 照 spec §4.2 末尾「新增」栏给的三个方向:`hand_visible_ratio`(左/右)、`shoulder_width`、`*_per_sec`;
- 每一列都要回答 spec §4.5 的两条贯穿规矩:① 受取景/设备影响的,要么按解剖尺度归一、要么**显式输出为协变量**;② 每一列都要能在标定里被证明有用,证明不了就删。

- [ ] **Step 1: 定 `fist_status` 的处置**

§4.2 依据栏写「`fist_threshold=0.08` → 100% 判握拳」。在表里写下**你定的阈值来源**:
若来自物理下限或自采数据 ⟹ `B`(写下值与依据);若只能来自标定集 ⟹ `C`。
**不许把 `0.08` 这个无出处的常数原样搬进表**(那正是被判"重构"的原因)。

- [ ] **Step 2: 手部 10 列逐列落行**

每行都要有 `definition`(算法或公式)、`normalization`(分母是什么;若为 `none`,写理由)、`unit`。

- [ ] **Step 2b: 手指角度那 10 列单独登记(spec §10 风险 4)**

`{left,right}_hand_{thumb,index,middle,ring,pinky}_angle` 是 2026-09-26 刚接进活路径的
(`下一步.md` §8.1),**在旧列处置表里没有对应行** ⟹ 它们既不在"保留"也不在"删除"里。
**单独登记,不硬塞进 13 列的估数**;`count_reconciliation.deltas` 里记一条说明这批是新接的。

- [ ] **Step 3: 手势的保留项落行**

§4.2 的保留项:8 个 `*_jitter`(合并为 1 列?) + 4 个 `*_is_valid`(合并为 1)+ `shoulder_shrug_level`。
**合并与否是一个决定,不是事实** —— 写进 `definition` 或 `count_reconciliation.deltas`,并说明依据。

- [ ] **Step 4: 填 `count_reconciliation.actual.gesture` 与 `deltas`**

文档估 **13**。写清实际数,以及为什么推不出 13(spec §1 已证:§4.2 无「L0 形态」栏)。

- [ ] **Step 5: 跑测试 + 反向复现 + Commit**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_l0_column_table.py -v`

```bash
git add l0_columns.json
git commit -m "feat(m3.0): L0 表填手势逐列定义(含手部 10 列)"
```

---

### Task 4: 填表 —— 语音 / 文本 / 协变量 + 计数收口

**Files:**
- Modify: `l0_columns.json`(其余 `modality` + `count_reconciliation` 收口)

**原料**:§4.3(`:233-239`)、§4.4 文本层(`:250-263`)、协变量(`:133`、`:227`)、现行 31 列(`voice_interaction/utils/logger.py:48-92`)。

- [ ] **Step 1: 语音行**

§4.3 只列 10 个旧列,扣掉 `speech_ratio`(删)、`duration_sec`(移出)后**只剩 8 个**。
文档说要 16 ⟹ 差额 8 列**由谁补、补什么**必须写清。已知候选:`voiced_prob`(Task 5)、
`micro_exp` 不属此模态、`n_chars`/`chars_per_sec`/`speech_duration_sec`/`reaction_time`
等 2026-09-26 新接的一阶量(见 `下一步.md` §8.1)。**逐条落行,不要凑数**。

- [ ] **Step 2: 文本层 3 个基础列**

`:250-254` 的 `transcript_raw` / `transcript_segments` / `asr_confidence` + `asr_model_version`。
**派生指标(`:256-263`)不落 `columns`** —— `:263`/`:370` 明写"标注到位前不实现"。进白名单,`planned: "M4 或更后"`。

- [ ] **Step 3: 协变量 3 个**

`face_scale` / `shoulder_width` / `hand_visible_*`。`modality` 填 `covariate`。
注意 `shoulder_width` 同时是 Task 7 的实现对象 —— **本任务只定义,Task 7 实现**。

- [ ] **Step 4: 收口 `count_reconciliation`**

四个模态的 `actual` 全部填满;`deltas` 至少四条(面/手/语/文本)。
**若实际总数不是 54**,在 `note` 里写明「文档的 54 是估数,实际为 N」并保留 `doc_estimate` 原值
—— **不许把估数悄悄改成实际数**(spec §4.3)。

- [ ] **Step 5: 跑测试,确认方向 1 全绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_l0_column_table.py -v`
Expected: 全部 PASS(方向 1 的 FAIL 列表应为空 —— 三份日志的每一列都有着落)。

- [ ] **Step 6: Commit**

```bash
git add l0_columns.json
git commit -m "feat(m3.0): L0 表填语音/文本/协变量 + 计数差异收口"
```

---

### Task 5: 新列 —— `voiced_prob`(语音)

**Files:**
- Modify: `voice_interaction/core/feature_extraction/prosody_extractor.py:45-48`(接住第 3 个返回值)
- Modify: `voice_interaction/utils/logger.py:48-92`(`fieldnames` **末尾**加列)
- Modify: `voice_interaction/api/app.py`(`_EXTRACTOR_TO_LOG_COLUMNS` 映射表,见 N1 账本 §6 —— 上游产出名 ≠ 下游列名是静默零值的根源)
- Test: `tests/test_voice_first_order_columns.py`

**Interfaces:**
- Consumes: `librosa.pyin` 返回 `(f0, voiced_flag, voiced_prob)`,现在 `:45` 把第 3 个用 `_` 丢弃
- Produces: 日志列 `voiced_prob_mean`(float 或 `""`)

**Review Focus 归属**:第 3 类(全静音段)与第 4 类(末列位置)。

- [ ] **Step 1: 写失败测试**

照 `tests/test_voice_first_order_columns.py:45-67` 的既有写法(它先断言列在 `fieldnames`,
再 `log_prosody` 落盘,再 `csv.DictReader` 读回)。

```python
def test_voiced_prob_reaches_the_log(tmp_path, monkeypatch):
    """★ 红法:去掉 fieldnames 里那一列 / 去掉 extractor 里接住第 3 个返回值的改动。"""
    monkeypatch.setattr(vl, "LOGS_DIR", str(tmp_path))
    log = vl.VoiceLogger(log_type="interview", session_id="20260926_120000_vvp1")
    assert "voiced_prob_mean" in log.fieldnames, "voiced_prob_mean 没进表头"
    # ⚠️ 必须**末列**:插中间会让老 CSV 的列序对不上
    assert log.fieldnames[-1] == "voiced_prob_mean", "新列必须在列序末尾"


def test_voiced_prob_is_empty_when_there_is_no_voiced_frame():
    """全静音:概率均值 0.0 是**真值**,但 f0 全 nan ⟹ pitch 该是空,不是 0。"""
    f = extractor_features(silence_audio)     # 见 Step 3 的实现
    assert f["voiced_prob_mean"] == 0.0
    assert f["pitch_mean"] is None or f["pitch_mean"] != 0
```

- [ ] **Step 2: 跑测试确认红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_voice_first_order_columns.py -k voiced_prob -v`
Expected: FAIL —— `voiced_prob_mean 没进表头`

- [ ] **Step 3: 实现**

`prosody_extractor.py`:`f0, voiced_flag, voiced_prob = librosa.pyin(...)`,
产出字典加 `"voiced_prob_mean"` = `float(np.mean(voiced_prob))`;**全静音不特殊处理**
(0.0 是真值),但**不要**让它影响 `pitch_mean`(后者已基于 `f0_voiced`,天然为空)。

`utils/logger.py`:`fieldnames` **末尾**加 `"voiced_prob_mean"`。
`api/app.py` 的 `_EXTRACTOR_TO_LOG_COLUMNS` 加一条映射(上游键 `voiced_prob_mean` → 列 `voiced_prob_mean`;
即便同名也要显式登记 —— N1 的教训是"少了映射表,只有那两列静默留 0")。

- [ ] **Step 4: 跑测试确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_voice_first_order_columns.py -v`
Expected: PASS

- [ ] **Step 5: 把表里那一行翻成 `implemented`**

`l0_columns.json` 里 `voiced_prob_mean` 的 `status` 改 `"implemented"`。
再跑 `tests/test_l0_column_table.py` —— **方向 2 会验它真的在产出里**。

- [ ] **Step 6: 端到端抽验(真素材)**

用 3 场正式素材之一(例如 `20260926_155559_caf0`)重跑 `experiments/replay_retained.py`,
确认新列在重放输出里**逐行有值或为空、且不为 0**。

- [ ] **Step 7: 反向复现 + Commit**

把 `fieldnames` 那一列删掉 → `test_voiced_prob_reaches_the_log` 必须红;恢复。清 `__pycache__` 复跑。

```bash
git add voice_interaction/core/feature_extraction/prosody_extractor.py \
        voice_interaction/utils/logger.py voice_interaction/api/app.py \
        tests/test_voice_first_order_columns.py l0_columns.json
git commit -m "feat(m3.0): 语音新列 voiced_prob —— pyin 的第 3 个返回值不再被丢"
```

---

### Task 6: 新列 —— `hand_visible_ratio_left` / `_right`(手势)

**Files:**
- Modify: `gesture_analysis/api/app.py`(在端点里按槽占用算出比值)
- Modify: `gesture_analysis/utils/logger.py:57-161`(末尾加 2 列;`_handedness_cells()` 见 `:189-203`)
- Test: `tests/test_gesture_task_outputs_wired.py`

**Interfaces:**
- Consumes: 槽占用的事实 —— `left_hand_score == ""` ⟺ 本帧左槽没有手(`_safe_get`,`utils/logger.py:341-355`;实测断言在 `tests/test_gesture_task_outputs_wired.py:230-235`)。`handedness_info[槽]`(可能为 `None`)来自 `api/app.py:328-350`。
- Produces: 日志列 `hand_visible_ratio_left` / `hand_visible_ratio_right`

**Review Focus 归属**:第 2 类(兜底分槽)与第 4 类(末列位置)。

⚠️ **机制已在 2026-09-26 核实,不要重新论证**:`GestureLogger` 只有**逐帧**写
(`utils/logger.py` 的 `log()` 一行一帧,没有"整场结束时补一行"的落点)。
所以**整场比值这一层现在无处可放** —— 那是 L1 的活。

**决定:落成逐帧列,名字叫 `hand_visible_left` / `hand_visible_right`(不是 `ratio`)。**
`l0_columns.json` 里登记这两个名字;`definition` 写"本帧该槽是否有手(1/0)",
`l0_output` 写"逐帧 0/1;整场比例是 L1 从这列派生"。
spec §4.2 的 `hand_visible_ratio` 是**设计意图**,不是列名契约 —— 这处改名要记进
`count_reconciliation.deltas`。

- [ ] **Step 1: 写失败测试(含兜底分槽那条)**

```python
def test_hand_visible_does_not_count_an_unattributed_hand_as_left(...):
    """★ 兜底分槽:槽非空但 handedness_info 为 None ⟹ 有手,但**不知道是哪只**。
    此时 left 那一格必须是空,不是 1。红法:把判据写成"槽非空即为真"。"""
```

- [ ] **Step 2: 跑测试确认红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_gesture_task_outputs_wired.py -k hand_visible -v`

- [ ] **Step 3: 实现 —— 复用 `_handedness_cells()` 的判据,不另起一套**

核心:**"有手"与"知道是哪只手"是两件事**。槽非空只说明有手(`_safe_get` 把 `None` 写成 `""`),
`handedness_info[槽]` 才是"哪只手"的依据。

```python
# gesture_analysis/utils/logger.py —— 加在 fieldnames 末尾
"hand_visible_left", "hand_visible_right",

# gesture_analysis/utils/logger.py —— 新方法,与 _handedness_cells() 同判据
def _hand_visible_cells(self, handedness_info: dict, hand_present: dict) -> list:
    """本帧该侧是否有**已署名**的手。

    ⚠️ 判据必须是 handedness_info 而不是"槽非空":api/app.py:342-349 的兜底支路会把一只
    **来路不明**的手塞进空槽并把 handedness_info[槽] 设为 None。那时"有手"是真的,
    "那是左手"是假的 —— 写 1 就是把不知道的事说成知道。
    """
    out = []
    for side in ("left", "right"):
        attributed = handedness_info.get(side) is not None
        out.append("1" if (attributed and hand_present.get(side)) else "")
    return out
```

⚠️ 两侧都**无依据**时写 `""`(空),**不写 `0`** —— `0` 的意思是"确定没有手",
与"不知道"是两回事(Review Focus 第 1 类)。

- [ ] **Step 4: 跑测试确认绿 / Step 5: 表里翻 `implemented` / Step 6: 反向复现**

- [ ] **Step 7: Commit**

```bash
git add gesture_analysis/api/app.py gesture_analysis/utils/logger.py \
        tests/test_gesture_task_outputs_wired.py l0_columns.json
git commit -m "feat(m3.0): 手势新列 hand_visible —— 兜底分槽那帧不许算作左手可见"
```

---

### Task 7: 新列 —— `shoulder_width`(手势,协变量)

**Files:**
- Modify: `gesture_analysis/core/feature_extraction/angles.py`(新增纯函数)
- Modify: `gesture_analysis/api/app.py:358` 附近(端点手里已有 `pose_landmarks`,归一化 33 点)
- Modify: `gesture_analysis/utils/logger.py`(末尾加列)
- Test: `tests/test_gesture_task_outputs_wired.py`

**Interfaces:**
- Consumes: `pose_landmarks`(`api/app.py:358` 的 `dets['pose'].detect_with_world(...)` 第 1 个返回值);
  **左肩索引 11、右肩索引 12**(映射表 `core/feature_extraction/angles.py:36-43` 的 `_POSE`;
  现有用法在 `core/analysis/shoulder_analyzer.py:73-74`)
- Produces: `shoulder_width`(归一化图像单位 —— **相机内参不可得,给不出米制**,见 Task 8)

**为什么它重要**:它是解封那批「取景代理」封停的钥匙之一(spec §4.5 规矩:受取景影响的量
要么按解剖尺度归一、要么显式输出为协变量)。

- [ ] **Step 1: 写失败测试(纯函数 + 缺关键点)**

```python
def test_shoulder_width_is_the_normalized_distance_between_lm11_and_lm12():
    lm = _pose_landmarks_with_shoulders((0.3, 0.4), (0.7, 0.4))
    assert angles.shoulder_width(lm) == pytest.approx(0.4, abs=1e-6)


def test_shoulder_width_is_none_when_a_shoulder_is_missing():
    """★ 缺关键点写空,不写 0 —— 0 会被下游当成"肩宽 0"。"""
    assert angles.shoulder_width(_pose_landmarks_missing_lm12()) is None
```

- [ ] **Step 2: 跑测试确认红**(`AttributeError: module has no attribute 'shoulder_width'`)
- [ ] **Step 3: 实现**

```python
# gesture_analysis/core/feature_extraction/angles.py —— 新增纯函数
_POSE_LEFT_SHOULDER = 11
_POSE_RIGHT_SHOULDER = 12


def shoulder_width(pose_landmarks) -> float | None:
    """双肩归一化图像坐标的欧氏距离。缺任一肩 ⟹ None(**不写 0**)。

    ⚠️ 单位是**归一化图像单位**,不是米 —— 转米需要相机内参,而本项目没有任何内参来源
    (2026-09-26 全仓核实,见 spec §10 风险 2)。所以这一列是**协变量**用途:
    报告层拿它除别的量来消掉取景/距离的影响(spec §4.5 规矩),
    **不许把它当"肩宽多少厘米"用**。
    """
    if pose_landmarks is None or len(pose_landmarks) <= _POSE_RIGHT_SHOULDER:
        return None
    a, b = pose_landmarks[_POSE_LEFT_SHOULDER], pose_landmarks[_POSE_RIGHT_SHOULDER]
    if a is None or b is None:
        return None
    return float(math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y)))
```

端点侧(`gesture_analysis/api/app.py`,与 `:378` 的 `angles_data.update(pose_angles(...))` 同一段):

```python
sw = shoulder_width(pose_landmarks)
angles_data["shoulder_width"] = sw      # None ⟹ logger 写空
```

`gesture_analysis/utils/logger.py`:`fieldnames` **末尾**加 `"shoulder_width"`,
行体取值走既有的 `_safe_get`(它已经把 `None` 处理成 `""`,别再写一套)。
- [ ] **Step 4: 跑测试确认绿 / Step 5: 表里翻 `implemented` / Step 6: 反向复现**
- [ ] **Step 7: 端到端抽验** —— 用真素材重放,确认新列有值(spec §8 验证策略第 3 条:实测取值域必须落在物理范围内;肩宽归一化后应在 (0, 1))
- [ ] **Step 8: Commit**

```bash
git add gesture_analysis/core/feature_extraction/angles.py gesture_analysis/api/app.py \
        gesture_analysis/utils/logger.py tests/test_gesture_task_outputs_wired.py l0_columns.json
git commit -m "feat(m3.0): 手势协变量 shoulder_width(lm11/lm12)"
```

---

### Task 8: 头姿 —— **登记为 blocked,不实现**

**Files:**
- Modify: `l0_columns.json`(`head_roll` 与 `head_yaw`/`head_pitch` 两行)

**为什么是登记而不是实现**:2026-09-26 全仓核实:

- `cv2.solvePnP`、`Rodrigues`:**代码里 0 命中**(只在文档文字里出现)
- 相机内参(焦距 / fx/fy/cx/cy / `camera_matrix` / `dist_coeff`)与 FOV:**没有任何来源**;
  `models/` 下只有 4 个 `.task` 模型文件,无标定 JSON/YAML

**没有内参就算不出真 3D 头姿。** spec §10 风险 2 明文规定:**不许编一个内参**。

- [ ] **Step 1: 写两行,`status: "blocked"`**

```json
{"column": "head_roll", "modality": "face",
 "definition": "solvePnP 真 3D 头姿的第三个欧拉角(spec §4.1:184 要求新增)",
 "source": "mediapipe face landmarker 的 478 点(需要 3D 对应模型点)",
 "normalization": "none", "unit": "度", "maturity": "C",
 "l0_output": "(未实现)真 3D 头姿需要相机内参,而本项目无任何内参来源",
 "status": "blocked",
 "quarantine_ref": "",
 "acceptance": "解封条件:取得相机内参(标定 / 从设备规格取得 / 用已知尺寸物标定)。在此之前本列不实现、不近似 —— 用 face_width 归一化的 2D 近似正是 §4.1 判『重构』的那个形态。"}
```

- [ ] **Step 2: 把 `blocked` 写进 `schema_errors` 的检查范围**

`l0_columns.py` 的 `schema_errors()` 里补一条:`status == "blocked"` 的行,
`acceptance` 必须包含解封条件(非空即可,但要在 `acceptance` 里写明"解封条件:")。

- [ ] **Step 3: 给钉子加一条测试**

```python
def test_blocked_rows_state_their_unblock_condition():
    """红法:把某条 blocked 行的 acceptance 清空。"""
    for c in l0.columns():
        if c["status"] == "blocked":
            assert "解封条件" in c["acceptance"], f"{c['column']} 没写解封条件"
```

- [ ] **Step 4: 跑测试 + 反向复现 + Commit**

```bash
git add l0_columns.json l0_columns.py tests/test_l0_column_table.py
git commit -m "feat(m3.0): 头姿 3D 登记为 blocked —— 无相机内参,不编"
```

---

### Task 9: 删手势死代码(4 文件 + `GesturePipeline`)

**Files:**
- Delete: `gesture_analysis/core/feature_extraction/{hand,arm,shoulder,upper_body}_feature_extractor.py`
- Delete: `gesture_analysis/pipeline/gesture_pipeline.py`
- Modify: `gesture_analysis/core/feature_extraction/__init__.py:7-16`
- Modify: `gesture_analysis/core/__init__.py:7-12,23-26`
- Modify: `gesture_analysis/__init__.py:24-27`
- Modify: `gesture_analysis/pipeline/__init__.py:7,10`

**已核实的事实(全仓,排除 `code_data_supplement/`)**:

- 4 个文件各只有 1 个类(`:13`)。
- 引用它们的活仓文件**只有 4 个**:上面列的那几个 `__init__.py` + `gesture_pipeline.py`。
- `gesture_analysis/__init__.py:24-27` 的 `__all__` 里那 4 个名字是**悬空名字**(该文件**没有**
  import 它们)—— `from gesture_analysis import *` 本来就会失败。删文件时**顺带修掉**。
- `gesture_pipeline.py` **没有任何 `.py` 生产代码 import 它**,**也没有任何测试碰它**。
- ⚠️ **import 链**:`gesture_analysis/__init__.py:13` → `core/__init__.py:7` →
  `core/feature_extraction/__init__.py:7` → `hand_feature_extractor.py`。**只删 4 个文件而不改
  那两个 `__init__.py` 会直接 `ModuleNotFoundError`。**

- [ ] **Step 1: 先写一条测试,证明"包还能 import 且导出表干净"**

```python
def test_gesture_package_imports_without_the_dead_extractors():
    """红法:删了文件但不改 __init__ 的导出表 ⟹ import 直接炸。"""
    import gesture_analysis
    import gesture_analysis.core.feature_extraction as fe
    for gone in ("HandFeatureExtractor", "ArmFeatureExtractor",
                 "ShoulderFeatureExtractor", "UpperBodyFeatureExtractor"):
        assert not hasattr(fe, gone), f"{gone} 还在导出表里"
        assert gone not in getattr(gesture_analysis, "__all__", []), \
            f"{gone} 还挂在 gesture_analysis.__all__ 上(悬空名字)"
```

- [ ] **Step 2: 跑测试确认红**(现在这 4 个名字都还在)
- [ ] **Step 3: 删文件 + 四处 `__init__` 改干净**
- [ ] **Step 4: 跑测试确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -x -q`
Expected: 全绿,**且总数不比基线 410 少**

- [ ] **Step 5: 确认无断链**

Run: `grep -rn "feature_extractor\|GesturePipeline" --include='*.py' . | grep -v code_data_supplement | grep -v __pycache__`
Expected: **0 命中**

- [ ] **Step 6: 反向复现 + Commit**

```bash
git add -A gesture_analysis/ tests/
git commit -m "refactor(gesture): 删 4 个死代码 extractor + GesturePipeline(不在 CSV 生产路径上)"
```

---

### Task 10: 收尾 —— 合并门 + 全量套件 + 账本

**Files:**
- 账本住在 SDD 工作区:`<repo>/.superpowers/sdd/2026-09-26-m3-0-l0-column-table/progress.md`
  (预检裁定 C4:计划早先写的 `docs/...` 那条路径是错的,以 `.superpowers/` 为准)
- 收尾时把它**拷一份**到 `docs/superpowers/sdd/2026-09-26-m3-0-l0-column-table/progress.md`
  (按裁定:文档不提交,等发话)
- Modify: `docs/下一步.md`(新增 §0.10,记录 M3.0 的结论;同样不提交)

- [ ] **Step 1: 复跑合并门(硬判据)**

```bash
cd ~/jingxin/experiments/duration_audit
PY=~/miniconda3/envs/jingxin/bin/python
$PY reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_probe_m30_final
$PY reaggregate_normalized.py --verify-legacy /tmp/legacy_probe_m30_final
```
Expected: `相对差 > 1e-4 的格子: 0 / 2835510`(2026-09-26 基线实测:最大绝对差 `1.886e-19`)

- [ ] **Step 2: 全量套件**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -q`
Expected: **≥ 410 passed**,0 failed

- [ ] **Step 3: 三场真素材端到端复跑**

对 `20260926_153202_2b11` / `153854_1592` / `155559_caf0` 各跑一次
`experiments/replay_retained.py` + `report_frontend.report_generator --session-id <sid>`,
确认:① 新列在重放输出里出现 ② 报告仍能生成 ③ 覆盖数**不低于**改动前(2/20 与 3/20)。
⚠️ 重抽等价性的口径见 spec §9:只比**两边共有**的列。

- [ ] **Step 4: 写账本**

`progress.md` 必须含:逐任务完成线、**每一处的反向复现记录**、
`count_reconciliation` 的最终数、以及**没做成的事**(例如 Task 8 的 blocked)。

- [ ] **Step 5: 更新 `docs/下一步.md`**

新增 §0.10:M3.0 交付了什么、表在哪、`actual` 总数是多少、M3.1–M3.5 的入口是什么。
**不改 §0.9 的历史记录**(账本里的历史不改,`docs/下一步.md` §4.6)。

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/sdd/2026-09-26-m3-0-l0-column-table/progress.md docs/下一步.md
git commit -m "docs(m3.0): 账本 + 下一步收口"
```

---

## 附:本计划核实过的事实(出处)

| 事实 | 出处 |
|---|---|
| face 日志 39 + 52 = 91 列 | `face_expression/utils/logger.py:43-59` |
| gesture 日志 67 列 | `gesture_analysis/utils/logger.py:57-161` |
| voice 日志 31 列 | `voice_interaction/utils/logger.py:48-92` |
| `pyin` 第 3 个返回值被 `_` 丢弃 | `voice_interaction/core/feature_extraction/prosody_extractor.py:45-48` |
| `solvePnP` / 相机内参 全仓 0 命中 | 2026-09-26 全仓 grep |
| 肩 landmark = 11 / 12 | `gesture_analysis/core/feature_extraction/angles.py:36-43` |
| 槽空 ⟺ 该手不在 | `gesture_analysis/utils/logger.py:341-355` + `tests/test_gesture_task_outputs_wired.py:230-235` |
| 兜底分槽会给 `handedness_info=None` | `gesture_analysis/api/app.py:342-349` |
| 4 个死文件各 1 个类 | 各自 `:13` |
| `GesturePipeline` 无活调用方、无测试 | 2026-09-26 全仓 grep |
| `pytest.ini` 只有 `testpaths = tests` | `pytest.ini` |
| conftest 已把 LOGS_DIR / 录音目录挪出仓库 | `tests/conftest.py` |
| 合并门基线 `0 / 2835510` | 2026-09-26 实测(最大绝对差 1.886e-19) |
