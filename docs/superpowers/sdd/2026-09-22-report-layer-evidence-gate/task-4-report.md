# Task 4 报告：删死分支与恢复 question_index

**状态:** DONE_WITH_CONCERNS
**提交:** `b5f2bba` fix: 保留 question_index 协变量,删除 fluency 死分支
**分支:** `fix/report-layer-evidence-gate`

---

## 1. 实现了什么

改动全部落在 `report_frontend/feature_engine.py` 的两个位置（按代码文本定位，未按 brief 的陈旧行号）：

### 1.1 恢复被静默丢弃的协变量 `question_index`

跳过规则原先把 `'index'` 与 `'id'`、`'unnamed'`、`'timestamp'` 并列，导致语音日志里真实存在的 `question_index` 在数值列扫描阶段被丢弃。改为两步判定：

- 模块级常量 `_SKIP_COLS = ('id', 'unnamed', 'timestamp')`（放在 `warnings.filterwarnings('ignore')` 之后）。
- 原第一道规则改用 `_SKIP_COLS`；新增第二道规则 `if 'index' in col.lower() and 'question' not in col.lower(): continue`，即只对含 `question` 的列名放行 `index`。

`_SKIP_COLS` 提到模块级而非留在循环体内，是相对 brief 片段的一处**有意偏离**：brief 片段把它写在 `if` 之前（循环体内会每列重建元组），语义完全等价，模块级更干净。

### 1.2 删除死分支

删除 `if 'speech_ratio' in col_lower and 'mean' in col_lower:` 整个分支（含其 `fluency_score_mean` / `fluency_proxy` 两行赋值），原处保留一条中文注释说明删除原因与 spec 出处，防止后续被重新引入。

---

## 2. 测试与结果

新增 2 个用例，追加在 `tests/test_report_layer.py` 末尾，原有 8 个用例一字未动。

- 聚焦运行：`~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k "question_index or dead_fluency" -v` → 2 passed
- 全量运行：`~/miniconda3/envs/jingxin/bin/python -m pytest` → **35 passed**（33 + 2，与预期一致），无 warnings summary，输出干净。

回归覆盖：`tests/test_evidence_gate.py` 24 passed、`tests/test_report_layer.py` 10 passed、`tests/test_smoke.py` 1 passed。

### 真实生产数据验证（超出单测的额外证据）

用真实日志 `data/logs/interview/research_emotion_log_20260224_233935.csv`（8 行）实测，其表头为：

```
unix_timestamp,timestamp,pitch_mean,pitch_variation,pitch_trend,pitch_direction,energy_mean,energy_variation,
speech_ratio,duration_sec,pause_duration_mean,pause_duration_max,pause_frequency,emotion,feedback,question_index,is_valid
```

- `question_index kept: ['research_question_index_mean', '..._std', '..._min', '..._max', '..._sum', '..._sample_size']` —— 修复前为空，修复后 6 个统计量。
- `fluency keys: []` —— 死分支删除后不再产出。
- `research_unix*` / `research_timestamp*` 仍被正常丢弃（`dropped index/timestamp cols intact: True`）。

这份表头同时实证了 brief 的两个前提：`question_index` 确实是真实列；`speech_ratio` 确实是真实列名且不含 `mean`，故分支确实是死代码。

---

## 3. TDD 证据

### RED

命令：`~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k "question_index or dead_fluency" -v`

**用 brief 原文的两个用例（未加强前）运行，结果与 brief 的 "Expected: FAIL" 不符：**

```
tests/test_report_layer.py::test_question_index_is_not_dropped FAILED    [ 50%]
tests/test_report_layer.py::test_dead_fluency_branch_removed PASSED      [100%]

E       AssertionError: question_index 仍被丢弃
E       assert 'question_index' in 'research_pitch_mean_mean research_pitch_mean_std ... _n_rows'

1 failed, 1 passed, 8 deselected in 0.26s
```

- `test_question_index_is_not_dropped` 失败，且失败原因正确：`voice_research` 词典里只有 `research_pitch_mean_*` 与 `_n_rows`，`question_index` 完全缺席——正是 `'index'` 跳过规则吞掉的。
- `test_dead_fluency_branch_removed` **通过**了。原因：该分支要求列名同时含 `speech_ratio` 与 `mean`，而 brief 用的列名就是 `speech_ratio`，永远不含 `mean`，所以这个断言**在改动前后都成立**——它是一条空断言，无法发现死分支是否存在，提供不了任何保护。

实测该分支的真实触发条件（改动前）：

```
'speech_ratio'             -> fluency keys: []
'speech_ratio_mean'        -> fluency keys: ['research_fluency_score_mean', 'research_fluency_proxy']
'research_speech_ratio_mean' -> fluency keys: ['research_fluency_score_mean', 'research_fluency_proxy']
```

**采取的处置**：保留 brief 用例的原始列名与断言，额外把 `speech_ratio_mean`（唯一能触发该分支的列名形态）并入同一用例并循环断言，使其真正可失败。这是相对 brief 的第二处**有意偏离**，理由如上：不可能失败的测试等于没有测试。偏离理由已写进用例 docstring。

加强后的 RED（改动前）：

```
tests/test_report_layer.py::test_question_index_is_not_dropped FAILED    [ 50%]
tests/test_report_layer.py::test_dead_fluency_branch_removed FAILED      [100%]

E   'fluency_score' is contained here:
E     research_speech_ratio_mean_mean ... research_fluency_score_mean research_fluency_proxy _n_rows

2 failed, 8 deselected in 0.29s
```

两个用例均失败，且失败原因各自正确：前者是协变量被吞，后者是死分支确实在产出 `fluency_score` / `fluency_proxy`。

### GREEN

命令：同上（聚焦），随后全量 `~/miniconda3/envs/jingxin/bin/python -m pytest`

```
tests/test_report_layer.py::test_question_index_is_not_dropped PASSED    [ 50%]
tests/test_report_layer.py::test_dead_fluency_branch_removed PASSED      [100%]

2 passed, 8 deselected in 0.22s
```

```
collected 35 items
tests/test_evidence_gate.py ........................                     [ 68%]
tests/test_report_layer.py ..........                                    [ 97%]
tests/test_smoke.py .                                                    [100%]

35 passed in 0.22s
```

---

## 4. 变更文件

| 文件 | 变更 |
| --- | --- |
| `report_frontend/feature_engine.py` | +8 / -8：跳过规则两段式 + `_SKIP_COLS` 常量；删除 fluency 死分支，留注释 |
| `tests/test_report_layer.py` | +31：追加 2 个用例（原有 8 个未动） |

提交只 `git add` 了这两个显式路径。`git add` 时 `feature_engine.py` 出现 "CRLF will be replaced by LF" 警告，属 `.gitattributes` 预期行为，非错误。工作区仍余若干被跟踪的 `__pycache__/*.pyc` 与无关产物（仓库既有噪声），按约束未纳入提交。

---

## 5. 自审发现

**跳过规则逐列实测（新规则）：**

| 列名 | 结果 | 走的哪条规则 |
| --- | --- | --- |
| `question_index` | **保留** | 不含 id/unnamed/timestamp；含 index 但含 question → 第二道放行 |
| `index` | 丢弃 | 第二道（含 index 且不含 question） |
| `id` | 丢弃 | 第一道（`_SKIP_COLS`） |
| `unix_timestamp` | 丢弃 | 第一道（`timestamp`） |
| `pitch_mean` | **保留** | 两道均不命中；且 `pitch_variation` 别名映射仍照常产出 |

额外bonus 验证：`frame_index` 被丢弃（真实行号列不会因为放行 `question` 而泄漏进来），`question_id` 被丢弃（第一道 `id` 命中，行为与改动前一致）。放行面被严格限制在「含 question 的 index 列」，没有扩大。

**死分支确认已删净：** `grep -n "fluency" report_frontend/feature_engine.py` 仅命中第 308 行那条说明注释，无任何代码引用 `fluency_score` 或 `fluency_proxy`。

**既有测试未受扰动：** `test_report_layer.py` 现有 10 个用例（8 旧 + 2 新）全部通过；全量 35 passed。改动前 33 passed 是干净的基线，新规则未破坏任何既有用例。

**输出洁净度：** 全量运行无 warnings summary；`pytest.ini` 的 `testpaths = tests` 生效，裸 `pytest` 安全。

---

## 6. 关切（Concerns）

1. **brief 的第二个用例是空断言，我已加强，这是一处有意偏离。** brief 声称两个用例都会 FAIL，实测只有第一个会 FAIL。`test_dead_fluency_branch_removed` 用 `speech_ratio` 列名时，在改动前后都通过，无法发现死分支。我在同一用例内追加了 `speech_ratio_mean` 这个唯一能触发该分支的列名形态。若下游审阅者要求严格照抄 brief 原文，请指出——但那样这个用例提供不了任何回归保护。理由已写进 docstring。

2. **`_SKIP_COLS` 提到模块级，是第二处有意偏离**（brief 片段写在循环体内）。语义等价，仅为避免每列重建元组。若需要与 brief 逐字一致，可以下移回循环体内。

3. **`question_id` 仍被丢弃（既有行为，未改）。** 第一道 `id` 规则会命中 `question_id`。我核对过的真实日志表头里只有 `question_index`、没有 `question_id`，且 spec 第 391 行明确要求的也只有 `question_index`，故未扩大放行范围。但若后续确认存在 `question_id` 这类题目级标识协变量，它会和当初的 `question_index` 一样被静默吞掉——建议后续在协变量清单里一并核对。

4. **删除的 `fluency_proxy` 语义上与 spec 其他条目重叠。** spec 第 230 行把 `*.fluency_proxy` 列为永久封停、第 248 行要求「永久删」（与 `fluency_score` 同一自由度数两遍）。本任务只删掉了 feature_engine 里唯一的产出点；`evidence_gate` 的封停名单仍保留 `fluency_score` 键（`tests/test_report_layer.py::test_all_slot_level_quarantine_keys_are_pinned` 正断言其被封停），两者不冲突——封停名单是防御性的，产出点没了也不该撤。仅作提示，未改动。

---

## 7. 结论

两处修复均已落地并被测试与真实数据双重验证；全量 35 passed，输出干净，提交范围为两个显式路径。唯一需要上游知悉的是第 6 节第 1 条：brief 的第二个用例本身不可失败，我做了最小加强，请审阅者确认这一偏离。
