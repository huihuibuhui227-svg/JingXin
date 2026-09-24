# 后续两件小事 —— 完工报告

- 日期:2026-09-24
- 范围:`tests/test_assert_coverage.py`(新增)、`docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md`(§10 一条)
- 解释器:`/home/huihuibuhui/miniconda3/envs/jingxin/bin/python`(3.11.15,pytest 9.1.1);未装新包;两文件均 LF
- 状态:**DONE_WITH_CONCERNS** —— 两件活都做完且套件全绿,但 Piece 1 的提交被一次并发提交干扰(见 §3)

---

## 1. Piece 1:断言覆盖率检查固化为常驻测试

### 1.1 文件与结构

`tests/test_assert_coverage.py`(340 行)。三条数据链:

| 环节 | 做什么 |
|---|---|
| ast 侧 `assert_map_from_source(source)` | 解析测试模块源码,得 `限定名 → assert 行号`;限定名与 pytest nodeid 的中间段同一坐标系(类方法 `Class::method`,模块级 `func`)。嵌套函数里的 assert 一并计入 |
| 追踪侧 `_TracePlugin` | `pytest_runtest_call`(hookwrapper)包住 **call 阶段**,`sys.settrace` 记录该阶段执行过的、位于被扫描模块内的 `(文件, 行号)`;按 nodeid 分桶 |
| 检查器 `check_assert_coverage(paths)` | 本进程内 `pytest.main(...)`,对差得 `unexecuted`:`{nodeid: [从未执行的 assert 行号]}` |

只有 call 阶段被追踪(setup/teardown 不记);追踪只对"被扫描模块"的帧开行事件(其他帧返回 `None` 不产生行事件,故 `report_frontend/` 的逐行开销为零,而这些帧调用的测试辅助函数仍照记)。

**不递归**:`suite_paths()` = `tests/test_*.py` 去掉本文件,内层 pytest 只跑这些路径。

### 1.2 防「假绿」的四道闸(除主结论外)

主结论(`unexecuted == {}`)单靠自己是可被绕过的 —— 收集一坏就退化成"0 个问题"。故并列四道闸,任一非空即失败:

1. **地板**:`n_tests >= 40`、`n_modules >= 3`、`n_assert_lines > 0`;
2. `unmapped`:`nodeid` 在 ast 表里找不到对应测试函数 → 该测试的 assert 根本没进对比表;
3. `never_called`:被收集却没进 call 阶段(跳过 / 前置失败)→ 它的 assert 一行都没被核实;
4. `never_collected`:源码里是 `test*` 却压根没被 pytest 收集(改名、放在非 `Test*` 类里、带参数)。

外加内层返回码闸:内层套件不绿时,行执行记录不可信,直接报"先修内层"而不是给覆盖率结论。

> 第 2 道闸不是设计出来的,是**被自己的自测抓出来的**:见 §1.5。

### 1.3 RED 证据(在最终修订上复跑)

注入方式:往真套件 `tests/test_smoke.py` 末尾追加一个**会通过**的测试,内含一条永远到不了的 assert —

```python
def test_injected_vacuous_for_red_check():
    """RED 用注入:测试通过,但有一条 assert 永远到不了。"""
    value = 1
    assert value == 1
    if value == 1:
        return
    assert value == 2          # ← 第 16 行,永远到不了
```

`pytest tests/test_assert_coverage.py::test_no_assert_is_dead -q` 输出(节选):

```
E       AssertionError: 以下测试里有从未执行过的 assert(声明了却没跑到,等于没测):
E           tests/test_smoke.py::test_injected_vacuous_for_red_check  行 [16]
E       assert {'tests/test_..._check': [16]} == {}
E         Left contains 1 more item:
E         {'tests/test_smoke.py::test_injected_vacuous_for_red_check': [16]}
--------------------------------- Captured stdout call ------
....................................................                     [100%]
52 passed in 0.54s
=========================== short test summary info ============================
FAILED tests/test_assert_coverage.py::test_no_assert_is_dead - AssertionError...
1 failed in 0.64s
```

两条要点:行号 **16** 精确指向那条死 assert;内层 52 个测试**全绿** —— 死断言不等于失败断言,这正是检查器存在的理由。

### 1.4 复原证明

| | sha256 `tests/test_smoke.py` |
|---|---|
| 注入前 | `02a9c5de557832562ef38ebfacd1d1d0ef4b870ed24226b2754942c19bd8dc91` |
| 复原后 | `02a9c5de557832562ef38ebfacd1d1d0ef4b870ed24226b2754942c19bd8dc91` |

`diff` 无输出;`git status --short tests/` 空 —— 逐字节一致。

### 1.5 自证能红:合成模块单测(单元级,不起子进程)

`test_checker_flags_dead_asserts_and_clears_sound_ones(tmp_path)` 把合成模块写进 `tmp_path`,**直接调 `check_assert_coverage([module])`**。合成模块 5 个测试:

| 测试 | 期望 |
|---|---|
| `test_all_asserts_run`(模块级,2 条) | 不点名 |
| `test_assert_after_early_return`(提前 return) | 点名,且行号精确 |
| `test_assert_in_never_entered_loop`(空循环) | 点名,且行号精确 |
| `TestSyntheticClass::test_method_all_asserts_run` | 不点名 |
| `TestSyntheticClass::test_method_with_dead_assert` | 点名,且行号精确 |

断言内容:`n_tests == 5`、`n_modules == 1`、`unmapped == []`、内层 rc == 0(合成模块本身是绿的)、点名的**恰好**是那三个、行号 == `_line_of(_VACUOUS, ...)` 逐条算出的行号、跑到的两个不得被点名(否则是"见谁都喊"而不是"抓到真死断言")。

**这道自测当场抓出一个真 bug**:最初 `check_assert_coverage` 从 nodeid 的路径段取模块路径(`os.path.abspath(parts[0])`),而 nodeid 的路径段是相对 rootdir 的。CWD 恰好是仓库根时真套件能对上(所以主检查看起来是绿的),`tmp_path` 里的模块**静默对不上** —— 报告退化成 `n_tests=3, n_modules=0, unexecuted={}`,假绿。已改为用 `item.path` 取绝对路径,并新增 `unmapped` 闸:对不上的 nodeid 一律报出来,不再静默 `continue`。

另外两个 hook 层面的坑,都已就地记入代码注释:
- pytest 9 下**裸生成器** hookimpl 不会被当成 hook 包装器,`pytest_runtest_call` 静默不触发(追踪不到任何行却报"0 个问题")→ 必须显式 `@pytest.hookimpl(hookwrapper=True)`;
- 模块路径要在 `pytest_collection_modifyitems` 里就记下:被跳过的测试不进 call 阶段,否则会在 `never_collected` 里被误报成"没被收集"。

### 1.6 GREEN 与数字

```
$ python -m pytest tests/ -q
.....................................................   [100%]
53 passed in 0.67s            # 51 + 本文件 2 条;基线 0.44s
$ python -m pytest tests/ -W error -q      # 清 __pycache__ 后
53 passed in 0.73s            # 零警告
```

实跑报告(真套件,排除本文件):

| 字段 | 值 | 地板 |
|---|---|---|
| `n_tests` | **51** | ≥ 40 |
| `n_modules` | **3** | ≥ 3 |
| `n_assert_lines` | **112** | > 0 |
| `unexecuted` | `{}` | 必须空 |
| `unmapped` / `never_called` / `never_collected` | `[]` / `[]` / `[]` | 必须空 |
| `inner_exit_code` | 0 | 必须 0 |

即:**112 条 assert 全部被执行过**,每一条都进了对比表。

### 1.7 盲区(已写进模块 docstring)

1. **只能抓「assert 从未执行」,抓不到「缺断言」** —— 第九次的失败模式:测试跑了、断言了、也通过了,但断言的性质比 docstring 声称的弱。每行 assert 都执行了,本检查给不出任何信号。**这是本检查最重要的盲区,docstring 里写明了"不要过度信任它"。**
2. 断言在执行的行上断言了**错的对象**(维度写错)—— 行执行了,语义错了。
3. **恒真断言**(`assert x == x`)。
4. **扫描范围只有 `tests/test_*.py`** —— 往 `tests/` 之外加测试,本检查静默。
5. **非 `test*` 辅助函数里的 assert 不在表内**(实测当前 `_rendered_html` / `_scan_targets` / `_string_literals` 一条 assert 都没有,今天无影响)。

另有两条**只在报告里**(没进 docstring,免得继续加码):

6. **一条 assert 都没有的测试**静默通过(实测当前 51 个测试函数无一如此)。
7. **fixture / `conftest.py` 里的 assert**:setup 阶段不追踪,且 `conftest.py` 不在扫描范围内。

---

## 2. Piece 2:设计文档 §10 补一条

### 2.1 实际写入的文字(§10 新增第 7 条,共 +1 行,其余一字未动)

> 7. **许可与伦理约束是硬约束,不是可以延后的事项。** RecruitView 是 CC BY-NC 4.0,其访问条款明文禁止将数据集**及在其上训练的模型**用于真实招聘、雇佣筛选或**心理画像**中的自动化决策(增补 §1.1)。**推论:任何在该数据集上拟合出来的参数都是数据集的衍生物,不得出现在任何上线路径里** —— 这不只包括 L2 标定系数(§6.3 已按此处理:阶段 1 只作离线研究结论),也包括**取自该数据集人群分布的 L1 阈值与归一化参数**:§4.1 的 `au26` 用"标定集 p1/p99"线性映射、§5.1 的 `au12` p90 取"常模人群"阈值,按同一读法都是衍生参数。上线路径需要这类参数时,其来源只能是**物理/定义性边界,或自采数据**;`report_frontend/evidence_thresholds.json` 是这条原则的样板(逐条登记每个因子的 `basis` 与 `basis_kind`,且 `_note` 声明依据是物理下限而非标定值),新参数照它写。并列的还有两条前置:采集自评标注(§6.2)以《科技伦理审查办法(试行)》的伦理审查为**强制前置**(增补 §1.3);MIT 与 First Impressions(ChaLearn)的许可条款**尚未核实**(增补 §1.1),核实之前这两条线同样不得进上线路径。

写法按 §10 其余条目的体例:编号项 + 粗体主句 + 散文;**不是任务清单**(无勾选项、无行动项、无"待办"字样)。引用增补的写法与 §6.3 里既有的"详见增补 §1.1"一致。

### 2.2 逐条来源

| 断言 | 来源 |
|---|---|
| CC BY-NC 4.0 + 访问条款禁止"数据集**或在其上训练的模型**"用于真实招聘/雇佣筛选/心理画像的自动化决策 | 增补 **§1.1** 第 1–3 条(许可三处互印证的逐字引用) |
| "数据集标定 → 先上线"这一组合已被移除;L2 系数只能作离线研究结论 | 增补 **§1.1**(加粗结论 + 两条合法去向 (a)/(b));设计文档 §6.3 已有同样处置,故此处写"已按此处理" |
| L2 标定系数是数据集衍生物 | 同上 |
| **L1 阈值与归一化参数同为衍生物**:`au26` p1/p99、`au12` p90 | 推论来自增补 §1.1 的许可措辞("models trained on it" ⇒ 任何拟合参数都是衍生物);两个具体参数的出处是设计文档自身:**§4.1**(`au26_jaw_drop`"用标定集的 p1/p99 线性映射")、**§5.1**(`au12` 的 p90"取自常模人群")。这两个参数在增补里没有点名,是本次按 §1.1 的读法**新纳入**的 |
| 上线路径需要此类参数时,来源只能是物理/定义性边界或自采数据 | 同上推论(增补 §1.1 的"不得进入任何已上线功能") |
| `evidence_thresholds.json` 是样板 | 该文件自身:第 4 行 `_note`"临时阈值。依据=物理下限,非标定值";`basis_kind` 取值 `definitional` / `physical` / `legacy_arbitrary` 逐条登记(实为 5 条 definitional、8 条 legacy_arbitrary,后者明写"无物理或定义依据"),无一条来自数据集标定 |
| 自评标注以《科技伦理审查办法(试行)》的伦理审查为强制前置 | 增补 **§1.3**(第 2 条第(一)项,2023-12-01 施行) |
| MIT / First Impressions 许可尚未核实 | 增补 **§1.1** 末段"另需独立核实(审查建议,尚未做)" |

---

## 3. 文件、提交与干扰

### 3.1 文件

| 文件 | 动作 | 最终 blob |
|---|---|---|
| `tests/test_assert_coverage.py` | 新增(340 行) | `683d7fcf80a4511f5fd0bd6c01bcbe8c5bf095be4b129c61d988e3a149471a63` |
| `docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md` | +1 行 | 见 `df0e72f` |

未触碰 `templates/`、`report_frontend/`、其他代码;工作区这两个文件均 clean。

### 3.2 提交

| Piece | SHA | 内容 |
|---|---|---|
| 2 | **`df0e72f`** `docs: 设计文档 §10 补入许可/伦理约束(数据集衍生物不得进上线路径)` | 恰好 1 文件 +1 行(用 `git commit --only <path>`,只带这一条路径) |
| 1 | **`60ed4ce`** `test: 断言覆盖率检查补记第五处盲区,并修掉 docstring 里的非法转义` | 只带 `tests/test_assert_coverage.py`,但**只含最后两处修订**(见 §3.3) |
| (背景) | `4479f9c` `exp: 时长审计与 C/C+ 白名单实验入库` | **不是我的提交**:检查器主体(含自证能红的合成模块)被它一并带走 |

全部提交均用 `git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com"`;未 amend、未 rebase、未 `git add -A`。

### 3.3 ⚠️ 并发提交干扰:Piece 1 没有得到独立提交

**发生了什么。** 我工作期间有另一个进程在同一仓库连续提交(与父任务给的同一身份):

```
e28d949 11:59:18  chore: 忽略实验产物目录,并清掉修复前生成的旧报告
3d78b42 11:59:25  docs: 把 ① 的设计、计划、审查与账本纳入版本控制
4479f9c 11:59:50  exp: 时长审计与 C/C+ 白名单实验入库(此前整个 experiments/ 都未跟踪)
9421942 ...       chore: 忽略规则改为可匹配任意深度        ← 我的两个提交之间又插进来一条
```

我开始时 HEAD 是 `1a84266`,现在 HEAD 是 `60ed4ce`,中间夹着 4 条不属于本任务的提交。

**后果。** 我执行 `git add tests/test_assert_coverage.py` 与提交之间,那个进程的提交(应为 `git add` 全量/宽范围)把我已进索引的文件一并带走 —— 于是 `tests/test_assert_coverage.py` 的**主体进入了 `4479f9c`**,我自己的 `git commit` 报"提交为空"。

**已核验**(`git show HEAD:tests/test_assert_coverage.py | sha256sum`):`4479f9c` 里的内容与我最终交付的修订**当时逐字节相同**(`332b6a85…`);其后我又补了第五处盲区并修了 docstring 转义,该终稿落在 `60ed4ce`(blob `683d7fcf…`,与工作区一致)。**没有任何工作丢失**,三个 SHA 均为 HEAD 的祖先,历史是线性追加的。

**我没有做的事。** 把文件从 `4479f9c` 里摘出来需要改写那条提交(`reset`/`rebase`/`filter-repo`),约束明确禁止 amend/rebase,且那是**别人的**提交,改写会波及对方尚未报告的引用 —— 故不动。**建议由控制器决定**是否需要事后拆分;若拆,`4479f9c` 里 `tests/test_assert_coverage.py` 那一段即 Piece 1 主体,`60ed4ce` 是其终稿增量。

**对约束的偏离说明。** "两个提交,一件一个"在 Piece 1 上未能干净达成(原因如上,非本人操作);`60ed4ce` 是 Piece 1 的一个真实、隔离的提交,但 diff 只有最后两处修订。若控制器认为必须有一条"整文件"的提交,唯一办法是改写 `4479f9c`,需要你放行。

---

## 4. 自审发现(除 §1.5 那个真 bug 外)

1. **`\*` 非法转义**:docstring 第 5 条初稿写成 `test\*`,非 raw docstring 里是非法转义,抛 `DeprecationWarning`。**只在模块重编译时复现**(清 `__pycache__` 才看到),头几轮跑套件时是干净的 —— 差点漏过。已改为无反斜杠写法;`-W error` 下套件零警告。
2. **自测 docstring 与断言不匹配**(本检查要抓的缺陷类型,自己差点犯):初稿写"三类测试各一个",实际合成模块有 5 个测试;已改成精确的"2 个全跑到 + 3 个带死 assert",并按文件清单逐条对上。
3. **`never_collected` 重复计数**:模块路径最初只在 call 阶段记录,导致被跳过的测试同时出现在 `never_called` 与 `never_collected` 两处。已把路径记录提前到 `pytest_collection_modifyitems`;用 `/tmp` 探针(跳过测试 + 非 `Test*` 类里的 `test_*` 方法)验证两者各自只报该报的。
4. **删除了一个未被使用的 `extra_args` 参数**(gold-plating 的反面:不留无人使用的扩展点)。
5. 三道新闸**都用探针实测过能红**(不是"写了就算"):跳过 → `never_called`;非 `Test*` 类里的 `test_*` 方法 → `never_collected`;模块路径映射错 → 当初就是它把 `n_modules` 打到 0 被自测抓住。

---

## 5. 遗留与建议

1. **Piece 1 的提交形态**(§3.3)—— 需控制器裁决是否改写 `4479f9c`。
2. **并发写入风险**:本任务期间 HEAD 移动 4 次,我的文件被卷走一次;若后续还要在此仓库并行提交,建议先约定"谁在提交"(或用独立 worktree),否则 `--only` 也只能保护我这一侧。
3. **可选的后续加固**(本次未做,避免超范围):
   - 把非 `test*` 辅助函数里的 assert 纳入对比表(按"全程至少执行一次"判定)—— 今天无影响,故未做;
   - 对"一条 assert 都没有的测试"设闸(今天也没有这种测试);
   - 在扫描范围上支持 `tests/` 之外的测试目录。
4. **本检查抓不到的三类**(§1.7):缺断言(第九次的真实失败模式)、断言对象错、恒真断言 —— 这三类只能靠人对 docstring 与断言的语义对应,不要因为本检查转绿就认为报告层的断言质量已被保证。
