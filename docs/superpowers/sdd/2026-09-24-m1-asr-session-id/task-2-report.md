# Task 2 报告:连接词密度(纯函数 + 标记表)

分支 `feat/m1-asr-session-id`,commit **dbbe1c4** `feat(m1): 连接词密度(每百字)+ 版本化标记表`
审计者参考:brief = `.superpowers/sdd/2026-09-24-m1-asr-session-id/task-2-brief.md`

## 1. 交付物(逐文件)

### `voice_interaction/asr/connective_markers.json`(新增,7 行)
版本化数据文件,四个键齐全:

- `version`: **`"1.0.0"`** —— M1 首版标记表。
- `_provisional`: `true`。
- `basis`: `"汉语话语连接标记的封闭清单。M1 首版:可辩护即可,不是标定产物。改表必须 bump version 并在报告里注明(spec §9.5)。"` —— 照 brief 逐字。
- `markers`: brief 给的 17 个,逐字照抄,未增删未排序:
  `然后 所以 但是 因为 而且 如果 虽然 不过 因此 另外 其实 首先 其次 最后 总之 比如 例如`

实测(见 §3 证据 G):17 个、无重复、**任意两个标记之间没有子串包含关系** —— 所以 v1 不存在"一个词被两个标记各计一次"的重复计数;将来往表里加词时必须重新查这一条(加 `为` 会让 `因为` 双计)。

### `voice_interaction/asr/connective_density.py`(新增,43 行)
- `load_markers(path=None) -> dict`,`@lru_cache(maxsize=1)`,默认读同目录 `connective_markers.json`。
- `connective_density(text, markers=None, min_chars=None, counter=None) -> float | None`。
- 下限来自 `load_config()["min_chars_for_density"]["value"]`(**没有**硬编码 10);`min_chars` 仅作为调用方覆盖口。
- docstring 按 brief 保留时长排除的理由(spec D4),并补了三段:只数"是否出现"(不数次数)的口径、下限出处、None 与 0.0 的区别(spec D8)。
- 只 import `count_cjk_chars, load_config`,**未**引用已不存在的 `_config`(采纳 controller 的更正)。

### `tests/test_connective_density.py`(新增,8 个测试)
brief 的 6 个逐字照抄,另加 2 个(理由见 §4)。

## 2. RED / GREEN 证据

**Step 2 — RED**(实现前)

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_connective_density.py -q
E   ModuleNotFoundError: No module named 'voice_interaction.asr.connective_density'
ERROR tests/test_connective_density.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.11s
```
失败原因正确:模块尚不存在,而不是导入路径写错或别的测试连带炸。

**Step 4 — GREEN**(实现后)

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_connective_density.py -q
........                                                                 [100%]
8 passed in 0.05s
```

**全量回归**(确认没碰坏 Task 1 的线)

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -q
90 passed in 1.07s
```
(去掉本任务文件单跑旧线实测 `pytest tests/ --ignore=tests/test_connective_density.py -q` = **82 passed in 1.00s**;82 + 8 = 90,增量全部来自本任务,无旧测试被改动、跳过或标记 xfail。)

## 3. 逐测试的 kill-check(哪一处生产代码改动会让它变红)

不是推断:每个测试都用**临时改写生产文件再跑**的方式实测过一次,改完立刻从 `/tmp` 备份恢复,恢复后 `diff` 干净(`MODULE-CLEAN`)且 8 passed。标记表的 M6 变异也手动改回并复跑通过。

| 变异(临时改生产代码/数据) | 实测结果 |
|---|---|
| M1 `hits / chars * 100` → `hits / chars` | `test_counts_markers_per_hundred_chars`、`test_punctuation_and_space_do_not_count_as_chars` FAILED(2 failed, 6 passed) |
| M2 `chars = count_cjk_chars(...)` → `len(text or "")` | `test_punctuation_and_space_do_not_count_as_chars` FAILED(1 failed, 7 passed;该例 108 字 → 0.9259) |
| M3 下限写死 `else 10` | `test_floor_is_read_from_config_not_hardcoded` FAILED(1 failed, 7 passed) |
| M4 低于下限 `return None` → `return 0.0` | `test_below_min_chars_returns_none_not_zero`、`test_empty_text_returns_none`、`test_floor_is_read_from_config_not_hardcoded` FAILED(3 failed, 5 passed) |
| M5 `hits == 0` 时改返回 `None` | `test_no_marker_returns_zero_not_none` FAILED(1 failed, 7 passed) |
| M6 标记表删 `但是` 且删 `_provisional`/`basis` | `test_marker_table_has_version_and_list`、`test_marker_table_carries_provenance` FAILED(2 failed, 6 passed) |

对应关系:

- `test_counts_markers_per_hundred_chars` —— 分母/单位被改就红(M1):这是"每百字"这个定义的唯一守卫。**⚠️ 此说法在审查后被证实为错,见 §F:该 fixture 恰好 100 字,分母写死或整个删掉它都照样绿。**
- `test_punctuation_and_space_do_not_count_as_chars` —— 字数算法退化成 `len()` 就红(M2)。**这条是 brief 六个用例里唯一真正钉住 `count_cjk_chars` 的**:若用 `len`,该例变 0.9259。
- `test_below_min_chars_returns_none_not_zero` / `test_empty_text_returns_none` —— 只要"没测出值"被写成 0.0 就红(M4)。这正是 brief 的分辨率 #3。
- `test_no_marker_returns_zero_not_none` —— 反向守卫:把"命中 0"误当成"无值"就红(M5)。它和上一条合起来才把 None/0.0 的两侧都锁住。
- `test_marker_table_has_version_and_list` —— 表缺 version、或标记数掉到 10 以下、或丢掉 `然后`/`但是` 就红(M6)。
- `test_marker_table_carries_provenance` —— 表退化成裸清单(丢 `_provisional`/`basis`)就红(M6)。全局约束"阈值/常量必须带出处"的直接守卫。
- `test_floor_is_read_from_config_not_hardcoded` —— 代码里写死 10 就红(M3)。这是唯一能钉住"下限只能来自数据文件"的用例:brief 的六个用例全部显式传 `min_chars=10`,**把下限写成裸常量它们照样全绿**。

## 4. 与 brief 的偏离

1. **`_config` → `load_config`(controller 指定,已采纳)。** brief 的 import 行与 `cfg = _config()[...]` 均改为 `load_config`。Task 1 落地后 `_config` 已不存在,不改则 `ImportError`。
2. **多写 2 个测试**(其余 6 个逐字未动)。理由分别是:
   - `test_marker_table_carries_provenance`:brief Step 3 的 JSON 里明确要求 `_provisional`/`basis`,但 brief 的测试只查了 `version` 与 `markers` —— 即"数据文件必须带出处"这条全局约束当时无人守卫(M6 实测:删掉这两个键,brief 的 6 个用例全绿)。
   - `test_floor_is_read_from_config_not_hardcoded`:controller 明确要求"不要在代码里硬编码 10",而 brief 的 6 个用例全部传 `min_chars=10`(M3 实测:写死常量 6 个用例全绿)。
   两条都属于把已写明的约束补上守卫,不是新增功能;生产代码的对外行为未因此变化。
3. **docstring 补了三段**(只数出现不数次数 / 下限出处 / None 与 0.0 的区别)。brief 只要求写时长排除的理由,该段原文保留;补的三段全部描述代码的实际行为,且是 brief 的分辨率 #1/#3 要求落到文件里的东西。
4. **`100` 仍按 brief 内联在表达式里。** 它是"每百字"这个单位的定义,不是可调阈值;brief 的数据文件形态也指定为恰好四个键,故未把 `100` 挪进 JSON。这是有意不偏离,若审查认为该挪,请示下。

## 5. 自查发现

- **`lru_cache(maxsize=1)` 看像 bug,实测不是。** 缓存键包含全部实参,`maxsize=1` 只限制保留条数(导致重算),不会串值。实测:先 `load_markers('/tmp/.../m.json')` 得到 version 9,再 `load_markers()` 仍正确返回 `1.0.0`/17 条。
- **重复说同一连接词只计一次**,与 docstring 一致:实测 `connective_density("然后然后然后"+"字"*47)` = 1.8868(1 命中 / 53 字)。
- **差异仅 3 个文件、98 行,`git show --stat` 核实**;`asr_config.json` 未被修改(`git status --porcelain` 对该路径无输出),Task 1 的 `models` 块与 basis 完好。全部新文件 LF(`grep -c $'\r'` 全为 0),未新增依赖,未触碰 `voice_interaction/asr/` 与 `tests/` 之外。
- 变异实验后已恢复到提交状态并复跑通过,工作区无残留。

## 6. 关切(未自行修,交 controller 判断)

1. **`min_chars=0` + 空文本会 `ZeroDivisionError`。** 实测 `connective_density('', min_chars=0)` 抛 `ZeroDivisionError: division by zero`。根因是守卫写成 `chars < floor`:当下限为 0 时空文本不满足 `<`,随后 `hits / chars` 除以 0。**当前不可达**:生产路径下限来自配置(实测值 10),且无调用方传 0,所以 brief 的 6 个用例与本次新增用例都覆盖不到。我按"严格照 brief 的守卫"保留原样,未改成 `if not chars or chars < floor` —— 后者能让"空文本必返 None"这条不依赖下限而恒成立,代价是一行偏离 brief 的改动。若希望这条不变量无条件成立,说一声我加(需同时补一个 `min_chars=0` 的用例)。
2. **标记表是子串匹配的 v1。** 当前 17 词无互相包含(已实测),但这是表的性质而不是代码的保证。将来加词时若引入包含关系(如加 `为`),密度会虚高;建议后续把这条写进改表检查,或在 `basis` 里注明。
3. **密度只反映"出现了哪些连接标记",不反映用法对错**(误用、堆砌均算命中)。M1 只承诺"有诚实的度量",这条边界宜在报告层文案里说清,不在本任务范围。

---

# 追加:审查裁定折入(commit **b0b7ab6**)

controller 裁定:关切 1、2 折进本轮,关切 3(`100` 保持内联)明确不改。以下是这一轮的改动与证据。

## A. 改动内容

`voice_interaction/asr/connective_density.py`

```python
    chars = (counter or count_cjk_chars)(text or "")
    # `not chars` 单独写:空文本恒返 None,不依赖"下限恰好大于 0",也免去除零。
    if not chars or chars < floor:
        return None
```
(原为 `if chars < floor:`)

docstring 补两处,均按裁定要求:
- 说明匹配是子串匹配 → 标记表内部不能有包含关系,该不变量由表守卫测试压着,改表时会红;
- 说明式中的 `100` 是"每百字"这个**单位**,不是可调阈值,故不写进数据文件。

`tests/test_connective_density.py` 新增 2 个测试(19 行):`test_empty_text_returns_none_even_with_zero_floor`、`test_no_marker_is_a_substring_of_another`。

**`connective_markers.json` 本轮未改**(`git diff` 对该路径无输出,17 词/4 键原样)。标记表的不变量是"表要满足的性质",不是"表当前有毛病";这一轮改的是守卫,不是表。

## B. 新增测试 1:`test_empty_text_returns_none_even_with_zero_floor`

**RED**(先写测试,未改生产代码)

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_connective_density.py -q
        chars = (counter or count_cjk_chars)(text or "")
        if chars < floor:
            return None
        table = markers if markers is not None else load_markers()["markers"]
        hits = sum(1 for m in table if m in (text or ""))
>       return round(hits / chars * 100, 4)
                     ^^^^^^^^^^^^
E       ZeroDivisionError: division by zero
voice_interaction/asr/connective_density.py:43: ZeroDivisionError
FAILED tests/test_connective_density.py::test_empty_text_returns_none_even_with_zero_floor
1 failed, 9 passed in 0.07s
```
失败理由正确:`chars < floor` 在 `floor == 0` 时对空文本判 False,放行到 `hits / chars` 才炸;不是断言写错、也不是被别的测试连带。

**GREEN**(改守卫后)

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_connective_density.py -q
..........                                                               [100%]
10 passed in 0.05s
```

**哪一处生产改动会让它红**:把 `not chars or` 去掉(回到 `chars < floor`)即 `ZeroDivisionError` 红 —— 即它唯一守卫的就是"空文本恒返 None 不依赖下限"这条契约。反向的 `return 0.0` 变异仍由既有的 `test_empty_text_returns_none` / `test_below_min_chars_returns_none_not_zero` 压着。

## C. 新增测试 2:`test_no_marker_is_a_substring_of_another`

这条**故意不靠改生产代码变红**,它守卫的是标记表这份数据。因此 RED 证据用一次**表变异**给出(改坏表 → 红 → 恢复):

```
== table mutated (added 为, a substring of 因为) ==
FAILED tests/test_connective_density.py::test_no_marker_is_a_substring_of_another
1 failed, 9 passed in 0.06s
  silent inflation demo: 因为 + 48 字 -> 4.0 (应为 2.0)
== restore table ==
..........                                                               [100%]
10 passed in 0.05s
```

第二行是这条守卫真正要拦的东西:表里加入 `为` 之后,`因为` 一句话同时命中 `因为` 与 `为` 两个标记,密度从 2.0 静默翻到 4.0 —— 生产代码一行没动、任何其他测试都不会红。恢复后 `git diff` 对标记表无输出,确认未把变异留在提交里。

**哪一处改动会让它红**:任何往 `markers` 里加入"已是另一标记子串"的词(双向检查,故 `因为`/`为` 这类任一方向的包含关系都能抓到)。

## D. 回归

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -q
92 passed in 0.96s
```
(本轮前 90;新增 2 个即全部增量。旧 82 条与上轮 8 条均未改动、未跳过。)

## E. 本轮自查

- 守卫改成 `not chars or chars < floor` 后,上轮的 M4/M5 kill-check **实测重跑过,结论不变且更强**:M4(`return None` → `return 0.0`)由 3 failed 变为 4 failed —— 新增的零下限用例也在守"不能写 0.0"这一侧;M5(`hits == 0` → `None`)仍恰好 1 failed(`test_no_marker_returns_zero_not_none`)。恢复后 `diff` 干净(`MODULE-CLEAN`),全量 92 passed。
- 变异实验后标记表已按备份恢复,`git diff` 无输出;提交只含 2 个文件(`--stat` 核实),`asr_config.json` 未动(`git status --porcelain` 无输出),两文件 LF(`grep -c $'\r'` 为 0)。
- 未 amend、未 rebase:早先的 `dbbe1c4` 原样保留,本轮为独立提交 `b0b7ab6`。
- 残留关切仅剩原第 3 条(命中不辨用法对错),按裁定不在本任务范围。

---

# 追加:任务审查 Needs fixes 的修复(commit **bf1758a**)

审查结论:0 Critical、1 Important、6 Minor。要求修 **1 Important + 3 Minor**,四条**全在同一个测试文件**,同一个病根:**这个指标的定义本身没被钉住**。四条修法都是加断言,**本轮生产代码一行未改**(`git status --porcelain -- voice_interaction/` 无输出,提交只含 test 文件)。

## F.0 先独立复核审查的核心断言(修之前)

审查说"把 `:49` 换成 `round(hits, 4)`(即把每百字归一化整个删掉)10 条测试全绿"。我没有直接采信,先自己复现:

```
== 'hits / chars * 100' -> 'hits' =
49:    return round(hits, 4)
..........                                                               [100%]
10 passed in 0.05s
== 'hits / chars * 100' -> 'hits / 100 * 100' =
49:    return round(hits / 100 * 100, 4)
..........                                                               [100%]
10 passed in 0.05s
RESTORED-CLEAN
```

**审查完全正确,我 §3 的"唯一守卫"是错的。** 根因:那条 fixture 恰好 100 字,`hits / chars * 100` 在该输入上**退化成 `hits`**;其余有命中的 fixture 也全是 100 字。所以它只覆盖了单位因子(`× 100`),**分母 `chars` 从未被任何测试守过** —— 而分母正是这个指标能跨句长比较的原因,也是 Task 4 要写进真实语音日志的数(真实回答永远不会正好 100 字)。

## F.1 Important:钉住分母(新增 1 条)

新增 `test_denominator_is_char_count_not_hit_count`:fixture 刻意**不是** 100 字 —— `"然后" + "字"*48`(50 字、命中 1,当前实现返回 2.0,已核实)。

**RED 证据(用审查指出的那两种变异给,红的理由是断言不等、不是异常)**

```
== N1: return round(hits, 4)  [删掉每百字归一化] ==
FAILED test_denominator_is_char_count_not_hit_count
FAILED test_chars_equal_to_floor_still_yields_a_value
FAILED test_repeated_marker_is_counted_once
3 failed, 10 passed in 0.08s
== N1b: return round(hits / 100 * 100, 4)  [分母写死常数] ==
FAILED test_denominator_is_char_count_not_hit_count
FAILED test_chars_equal_to_floor_still_yields_a_value
FAILED test_repeated_marker_is_counted_once
3 failed, 10 passed in 0.07s
```

修复前这两种变异红 **0** 条,修复后红 **3** 条(GREEN 为 13 passed)。这三条新用例的长度都不是 100,所以都能咬住分母;其中 `test_denominator_is_char_count_not_hit_count` 是**专门**为这条命名的守卫。断言的实际报错形如 `assert 1.0 == 2.0`,即断言不等,不是异常 —— 与审查对"红法"的预判一致。

## F.2 Minor 1:下限边界 `chars == floor` 要出值(新增 1 条)

spec §6.5 措辞是 `n_chars < min_chars_for_density`,**相等要出值**;此前没有任何 fixture 正好卡在下限上,故 `chars <= floor` 也全绿。

新增 `test_chars_equal_to_floor_still_yields_a_value`:`"然后" + "字"*8` = **正好 10 字**,命中 1 → 10.0。

```
== N2: chars < floor -> chars <= floor ==
FAILED test_chars_equal_to_floor_still_yields_a_value
1 failed, 12 passed in 0.06s
```
恰好 1 条红(其余用例都不在下限上,故不受影响),证明它唯一钉的就是这个边界。

## F.3 Minor 2:重复标记只计一次(新增 1 条)

docstring 写明 presence 而非 occurrence,但此前只有 §5 的手工验证(`1.8868`),没有 fixture 重复标记,故改成 `sum(text.count(m) for m in table)` 仍全绿。

新增 `test_repeated_marker_is_counted_once`:`"然后"*3 + "字"*44`(6 + 44 = 50 字,命中 1 → 2.0)。

```
== N3: presence -> occurrence  sum(text.count(m) for m in table) ==
FAILED test_repeated_marker_is_counted_once
1 failed, 12 passed in 0.06s
```
(occurrence 口径下该例得 6.0,红在断言不等。)长度算式已写进 docstring,免得再犯 F.4 那个错。

## F.4 自查抓到的一次自作自受(记录在案)

这条用例我第一版写的是 `"然后然后然后" + "字"*47` 并断言 `== 2.0`,**当场红了**:`assert 1.8868 == 2.0`。原因是我把 fixture 长度算错了 —— `"然后"*3` 是 **6** 字不是 3 字,6 + 47 = 53 字,`1/53*100 = 1.8868`。**实现是对的,错的是我的期望值。** 已改成把长度算清的写法。

顺带暴露一个连锁:那次红还带红了 `tests/test_assert_coverage.py::test_no_assert_is_dead`,因为它是"内层套件必须跑绿"的元测试(`inner_exit_code == 0`),任何一条测试红了它都会红。**这是连锁,不是第二个缺陷** —— 修掉 fixture 后两者同时转绿。记在这里是因为它说明本仓的"每条 assert 至少被执行一次"门是活的。

## F.5 Minor 3:标记表字面重复(改 1 条既有表测试)

`hits = sum(1 for m in table if m in text)` 会把表里**字面重复**的标记计两次;而 `test_no_marker_is_a_substring_of_another` 用 `a != b` 过滤,**按构造**排除了"完全相同的串"这一对 —— 同一个 M1-11 静默膨胀,换了个轴照样没人拦。

按裁定在同一条表测试里加 `assert len(ms) == len(set(ms))`。

```
== N4: 表里加一个字面重复的标记 =="
(重复 `然后`,它出现在 fixture 里)
FAILED test_counts_markers_per_hundred_chars
FAILED test_punctuation_and_space_do_not_count_as_chars
FAILED test_no_marker_is_a_substring_of_another
FAILED test_denominator_is_char_count_not_hit_count
FAILED test_chars_equal_to_floor_still_yields_a_value
FAILED test_repeated_marker_is_counted_once
6 failed, 7 passed in 0.08s
  silent inflation: 然后 + 48 字 -> 4.0 (应 2.0)
```

但上面这条还不够狠 —— 它之所以大面积红,是因为 `然后` 恰好出现在 fixture 里,是**数值断言顺带**发现的。真正证明去重断言**不可省**的是下面这条:重复一个**不出现在任何 fixture** 的标记(`例如`):

```
== N4b: 表里重复 `例如`(无任何 fixture 用到它) ==
FAILED test_no_marker_is_a_substring_of_another
1 failed, 12 passed in 0.06s
```

**12 条数值断言全绿,只有去重断言抓住它。** 即:没有这一行,这种坏表是完全静默的 —— 去重断言是承重的,不是装饰。

## F.6 回归与状态

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_connective_density.py -q
.............                                                            [100%]
13 passed in 0.06s
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -q
.......................                                                  [100%]
95 passed in 1.02s
```
(上轮 92 → 95,新增 3 条即全部增量;brief 原 6 条与上轮 4 条**逐字未动**,无跳过、无 xfail。)

### ⚠️ 全量数字的时效性说明(交给 triage 时请注意)

上面 `95 passed` 是我改完当即测的。**此后工作区里出现了两个不属于本任务的未跟踪测试文件** —— `tests/test_session_id_contract.py`、`tests/test_session_logging.py`(Task 3 的在制品,`git ls-files` 为空;我未创建也未改动它们)。它们一进来,全量跑就不再绿:

```
== 失败按文件归属 ==
  2 FAILED tests/test_session_id_contract.py
 16 FAILED tests/test_session_logging.py
  1 FAILED tests/test_assert_coverage.py
```
两次跑之间 Task 3 的失败条数还在变(10 → 19),因为该文件正被并发编辑 —— 全量数字目前在**移动中**。

我逐条归因过,**没有一条与本次改动有关**:
- 前两个文件的失败是 Task 3 自己的在制品;
- `test_assert_coverage.py::test_no_assert_is_dead` 是**连锁**:该测试的 `suite_paths()` 会 `rglob("tests/test_*.py")` 吃掉 Task 3 那两个文件,只要内层套件不绿它就红(`inner_exit_code == 0` 是它的前置条件)。只读验证(把 T3 两个文件排除后直接调 `check_assert_coverage`):

```
inner_exit_code with T3 files excluded: 0
n_tests: 93 | unexecuted: {} | never_called: []
```

`inner_exit_code == 0` 且 **`unexecuted = {}` / `never_called = []`** —— 即本仓"每条 assert 至少被执行过一次"这道门确认:**包含我本轮新增的全部断言在内,套件里没有一条死 assert**。93 = 95 减去 `test_assert_coverage.py` 自身那 2 条(它按设计排除自己)。

我没有去动 Task 3 的文件(越界),也没有为了让全量变绿而调整任何测试选择。

## F.7 其他状态

- 六次变异(N1/N1b/N2/N3/N4/N4b)全部实测、逐个恢复;生产文件与标记表 `diff` 均干净(`RESTORED-CLEAN`),提交只含 `tests/test_connective_density.py`(+29 行)。
- `asr_config.json` 未动(`git status --porcelain` 无输出);test 文件 LF(`grep -c $'\r'` = 0)。
- 未 amend、未 rebase:`dbbe1c4`、`b0b7ab6` 原样,本轮为独立提交 `bf1758a`。

## F.8 记档不修(按裁定,留给最终 triage;我没有顺手动)

1. `load_markers()` 的 `@lru_cache` **把可变对象直接发出去**(调用方 `append`/`sort` 会污染本进程后续所有调用),且带 `path` 调用会把默认项挤出缓存(`maxsize=1`)—— 裁定列为首要 deferred。
2. `test_floor_is_read_from_config_not_hardcoded` 是**单边**的(只断言 `is None`,写死 101 也能过)。
3. spec §10 的"全标记"用例缺口。


