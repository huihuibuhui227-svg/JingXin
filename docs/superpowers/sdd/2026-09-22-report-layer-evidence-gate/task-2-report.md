# Task 2 报告:证据门模块

- 状态:**DONE**(第一轮报 NEEDS_CONTEXT → 裁决后 fix → 审查 2 条 Important 修复完毕;25 项全绿)
- 提交:`08e31fe` → 审查修复 `fa44d99` → `e249450`(说明见第 8、9 节)
- 分支:`fix/report-layer-evidence-gate`(已确认,未新建)
- BASE:`8d156f4`
- 解释器:`~/miniconda3/envs/jingxin/bin/python`

**阅读顺序:** 第 1–7 节是第一轮(裁决前)的原文,保留作为审计记录 —— 它记录了 3 处矛盾的
发现过程与实测证据;第 8 节是裁决后的 fix 轮记录(改了什么、覆盖的测试、命令、输出)。
第 4 节当时提的两个候选补丁 C1/C2 已被裁决取代(控制器选 C1 并给出了理由)。

---

## 第一轮记录(裁决前)

---

## 1. 已实现/已写出的文件(逐字照抄 brief)

| 文件 | 状态 | 内容 |
|---|---|---|
| `report_frontend/evidence_thresholds.json` | 已创建,与 brief Step 1 逐字一致 | 8 条临时阈值 + `_version` / `_provisional` |
| `tests/test_evidence_gate.py` | 已创建,与 brief Step 2 逐字一致 | 20 条断言 |
| `report_frontend/evidence_gate.py` | 已创建,与 brief Step 4 逐字一致 | 四关 + 封停名单 + 置信度三值 |
| `pytest.ini` | 已创建(Ruling 9,scope 增量) | `[pytest]` / `testpaths = tests` |

四个文件均由本任务创建。第一轮结束时它们都未跟踪、未提交;第 8 节的 fix 轮把四个路径
一次提交为 `08e31fe`。行尾为 LF(`.gitattributes` 生效)。

---

## 2. TDD 证据

### RED(实现之前)

命令:

```
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_evidence_gate.py -v
```

关键输出:

```
rootdir: /home/huihuibuhui/jingxin
configfile: pytest.ini
plugins: anyio-4.14.2
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
_________________ ERROR collecting tests/test_evidence_gate.py _________________
ImportError while importing test module '/home/huihuibuhui/jingxin/tests/test_evidence_gate.py'.
...
tests/test_evidence_gate.py:4: in <module>
    from report_frontend.evidence_gate import (
E   ModuleNotFoundError: No module named 'report_frontend.evidence_gate'
=========================== short test summary info ============================
ERROR tests/test_evidence_gate.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.34s ===============================
```

为什么这个失败是预期的、且是正确的失败:测试文件先写、模块尚不存在,因此收集阶段必然
`ModuleNotFoundError`。brief Step 3 规定的正是这一条错误,**不是**别的错误(不是 import
`report_frontend` 失败、不是 plotly 缺失、不是语法错误)—— 说明解释器、`tests/conftest.py`
的 sys.path 注入、`pytest.ini` 三者都已正确就位,失败点恰好在被实现的模块上。

### GREEN(实现之后)

命令:

```
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_evidence_gate.py -v
```

结果:**17 passed / 3 failed**。三条失败的完整输出见下一节。**未达到 brief Step 5 要求的"全部 PASS"。**

### 全仓裸 pytest(Ruling 9 验证)

命令:`~/miniconda3/envs/jingxin/bin/python -m pytest`(仓库根)

结果:`3 failed, 18 passed in 0.24s` —— `pytest.ini` 生效,bare pytest 只收集
`tests/`(20 + 1 smoke = 21 项),**4 个既有 scratch 脚本的收集错误已消除**,无任何
collection error、无 warnings summary(输出干净)。

---

## 3. 阻断原因:brief 与自身矛盾的 3 处(brief Step 4 的实现无法让 brief Step 2 的测试全过)

按派发指令"如果 brief 给的测试与 brief 给的实现互相矛盾,停下来报告,不要静默选一个",
以下三条我**没有**自行裁决。三处均已实测复现,失败信息本身即为证据。

### 矛盾 1 —— `TestG3EnoughSamples::test_at_threshold_passes`

```python
assert gate("blink_rate", 0.5, 20, values=[1.0, 2.0]).ok is True
```

实测:

```
E  AssertionError: assert False is True
E   +  where False = Check(ok=False, gate_name='G4', reason='已封停:恒 0(值写进深拷贝)(解封:M2 修序列化)')
```

原因:brief 的 `QUARANTINE` 里含 `"blink_rate"`,`is_quarantined("blink_rate")` 命中
G4。G3 本身是通过的(n=20 ≥ blink 阈值 20),拦下它的是 G4。

**spec 站在实现这边**:spec §5.2 表列 `blink_rate_per_min` / `eye_closed_sec` 封停,
§5.3 表把 `blink_rate` 槽标为"恒 0 → 等 M2"。brief 用子串键 `"blink_rate"` 一次覆盖
`blink_rate_per_min` / `blink_rate_mean`,是刻意且有 spec 依据的。**缺陷在测试选的键名。**

附带事实:阈值注册表里唯一产出 20 的键就是 `"blink"`,而任何含 `blink` 的键都含
`blink_rate` → **这个 20 的边界无法用任何未被封停的键来测**。故修复方向只能是换一个
已登记阈值(推荐 `pitch` = 10,与文件里 G1/G2 用例统一用 `pitch_median`)。

### 矛盾 2 —— `TestG4NotProxy::test_clean_column_passes`

```python
assert gate("interview_pause_duration_mean", 0.8, 500, values=[0.7, 0.9]).ok is True
```

实测:

```
E  AssertionError: assert False is True
E   +  where False = Check(ok=False, gate_name='G4', reason='已封停:尾部静默被丢弃(解封:M3 修停顿检测)')
```

原因:`"pause_duration"` 是 `"interview_pause_duration_mean"` 的子串。

**spec 站在实现这边**:spec §5.2 明写封停模式是 `*.pause_duration_*` —— 两侧带通配符,
`interview_pause_duration_mean` 精确落在这个模式里;封停理由(尾部静默被丢弃)也正是
这个列自己的 bug。**缺陷在测试把一个被封停列当成了"干净列"。**

### 矛盾 3 —— `TestG4NotProxy::test_permanent_quarantine_has_no_unblock`

```python
q = is_quarantined("overall_score")
assert q is not None and q.permanent is True
```

实测:

```
E  AssertionError: assert (Quarantine(reason='属性不存在,getattr 走默认值', unblock='永不', permanent=False) is not None and False is True)
```

原因:`Quarantine` 的 `permanent` 字段默认 `False`,而 brief 的 `QUARANTINE` 字典
**28 条里没有任何一条传过 `permanent=True`**。实测计数:字典中 `unblock == "永不"`
的有 **8 条**,`permanent is True` 的有 **0 条**。

**spec 站在测试这边**:spec §5.2 表把 `shoulder_is_calibrated` / `upper_body_head_tilt` /
`overall_score` / `emotion_state` / `emotion_*` / `dominant_emotion` /
`micro_exp_onset_frame` / `*.fluency_proxy` 明确标为"**永久封停**",而 brief 只在
`unblock` 里写了字符串 `"永不"`、忘了置位。**缺陷在实现漏了 `permanent` 的赋值。**

这一处与前两处方向相反 —— 所以**不存在"只改测试"或"只改实现"一个方向的修法能让 20 条全过**;
两个文件都得动。这正是我不自行裁决的原因。

---

## 4. 建议的裁决与最小补丁(等控制器裁定后再动手)

我的建议:**实现方向是对的(前 2 条改测试、第 3 条改实现)**,因为前两条若反过来
(把 `blink_rate`、`pause_duration` 从封停名单里摘掉)会直接违反 spec §5.2/§5.3,
而这两列正是本次工作要拦下的坏列(`pause_duration` 有尾部静默 bug)。

若控制器同意,改动量为 3 处、共约 5 行:

**补丁 A —— `tests/test_evidence_gate.py`,`TestG3EnoughSamples` 两个用例的键**
(把 `blink_rate`/20 换成 `pitch_median`/10,保持"边界前失败、边界上通过"的原意):

```python
    def test_below_threshold_fails(self):
        c = gate("pitch_median", 0.5, 9, values=[1.0, 2.0])
        assert c.ok is False
        assert c.gate_name == "G3"

    def test_at_threshold_passes(self):
        assert gate("pitch_median", 0.5, 10, values=[1.0, 2.0]).ok is True
```

已实测:`pitch_median` 未被封停、阈值 10;n=9 → G3 失败;n=10 → 通过。注意 `TestGateOrder::test_first_failure_wins`
用的 `blink_rate` 键**不需要动**(它只断言 `gate_name == "G1"`,G1 先于 G4 返回,当前已通过)。

**补丁 B —— `tests/test_evidence_gate.py`,`test_clean_column_passes` 换一个真干净列**
(已实测 `interview_duration_mean` 未被封停、走未登记默认阈值 10、n=500 通过):

```python
    def test_clean_column_passes(self):
        assert gate("interview_duration_mean", 0.8, 500,
                    values=[0.7, 0.9]).ok is True
```

**补丁 C —— `report_frontend/evidence_gate.py`,让 `permanent` 有序**(二选一)

- C1(显式):给那 8 条 `unblock="永不"` 的条目加 `permanent=True`,字典字面量其余不变。
- C2(派生,更贴合测试名 `permanent_quarantine_has_no_unblock`):frozen dataclass 加

  ```python
  def __post_init__(self) -> None:
      if self.unblock == "永不":
          object.__setattr__(self, "permanent", True)
  ```

  好处:8 条字典字面量保持逐字不动,且"永久 ⟺ 无解封里程碑"成为不变式,而不是 8 个
  手写的布尔值(手写就会漏,正如现在)。

我倾向 **C2**。下游无风险:`permanent` 字段在整个计划里**只被这条测试读过**(已 grep
全部 8 个 task 的 brief:`is_quarantined` / `QUARANTINE` / `permanent` / `unblock` 的
全部命中只有 Task 2 的接口声明、实现、和这条测试;Task 3 只读 `chk.reason`)。

---

## 5. 自检结果

派发时点名的三项自检,均**实测**通过(用 `python -c` 直接探针,不只读代码):

1. **`is_quarantined` 最长匹配** —— 是的,`max(matched, key=len)` 行为正确:
   - `"dominant_emotion"` → 命中 16 字符的 `dominant_emotion`(reason `argmax,非置信度`),
     **不是** 7 字符的 `emotion_`(reason `7 个分量结构性恒 0`)。两条都是子串,长的胜出 ✅
   - `"tension_sources_au4"` → 命中 `tension_sources_` ✅
   - `"face_focus_score_mean"` → 命中 `focus_score` ✅
2. **`gate` 先失败先返回、顺序 G1→G2→G3→G4** —— 是的:
   - `gate("blink_rate", None, 0)` → `G1`(值缺失先于样本不足)✅
   - `gate("blink_rate", 1.0, 0, values=[1.0])` → `G2`(常量先于样本不足)✅
   - `gate("blink_rate", 1.0, 0, values=[1.0, 2.0])` → `G3` ✅
   - `gate("face_focus_score_mean", None, 0)` → `G1`(多重失败时取最早一关)✅
3. **`confidence_from` 永不返 `"高"`** —— 是的,穷举 `n_ok ∈ [0,60) × n_slots ∈ [0,60)`
   共 3600 组,**返回 `"高"` 的组数 = 0**;边界 `(1,2)→中`、`(1,3)→低`、`(2,4)→中`、
   `(0,0)→无` 均符合 spec §5.1 的三值定义 ✅
4. **frozen dataclass** —— `Check(True).ok = False` 抛 `FrozenInstanceError` ✅
5. **测试输出干净** —— 无 warnings summary、无 stray output;`pytest.ini` 生效后 bare
   pytest 无收集错误 ✅

### 自检额外发现(不在 brief 的自检清单里,但与下游有关)

**发现 A(实测,真缺陷,当前无测试覆盖):`evidence_thresholds.json` 里的 `"pause": 2` 是死条目。**

`_threshold_for` 用"按注册表顺序的第一个子串命中",而 `"au"` 排在 `"pause"` 前面;
`"pause"` 本身含子串 `"au"`,所以**任何含 `pause` 的键都会先命中 `"au"` 拿到 30**:

```
pause_duration_mean            registered=  2  actual= 30  DEAD
interview_pause_segment_count  registered=  2  actual= 30  DEAD
```

后果:停顿类指标的 G3 门槛实际是 30 而非登记的 2 —— 方向是**更严**(偏保守,不会放过坏列),
所以不造成假阳性;但它与 spec §5.1"阈值从配置读、禁止硬编码"的意图相悖,且是静默的
(无测试覆盖 `pause` 阈值)。顺带暴露一个设计不一致:`is_quarantined` 用**最长匹配**,
`_threshold_for` 用**注册表序首个匹配** —— 同一个模块里两套匹配规则。**我没有改**(属 brief 之外),
请控制器决定是否并入本次修复。

**发现 B(文档不一致,零运行时风险):brief 接口段与实现段的 `Check` 字段顺序不一致。**

- 接口段(L19):字段 `ok: bool`, `reason: str`, `gate_name: str`
- 实现段(L187-191):`Check(ok, gate_name, reason)` —— 且全部按位置构造 `Check(False, "G1", "无值")`

下游(Task 3 第 656 行)只用属性访问 `chk.reason`,**不按位置构造 `Check`**,故无运行时风险;
但若将来有人按接口段的顺序写 `Check(False, reason, name)`,字段会静默对调。建议以实现为准、
把接口段改成 `ok, gate_name, reason`。

---

## 6. 第一轮为什么不提交(裁决前;第 8 节已提交)

- brief Step 6 与派发指令要求"测试通过后提交"。当前 **3 条 brief 自带断言失败**,提交等于把
  红灯固化进历史,并让后续 review 面对一个"计划与实现互相矛盾"的提交。
- 三个文件是**互相依赖的一个单元**(实现 + 它的测试 + 测试的输入配置),只提交其中一部分
  (例如只提交 `pytest.ini` 或无测试的实现)会制造半成品状态,不如整单元等裁决。
- 因此:**零提交**,四个新文件以"逐字照抄 brief"的形态留在工作区。控制器裁定后,下一轮
  只需应用第 4 节的 3 处补丁、复跑、一次提交即可(补丁已实测验证过,预计一轮完成)。

工作区状态(已核验):

```
?? pytest.ini
?? report_frontend/evidence_gate.py
?? report_frontend/evidence_thresholds.json
?? tests/test_evidence_gate.py
```

`git diff --cached` 为空 —— 无任何 staged 内容,分支 HEAD 仍是 `8d156f4`,未受改动。

---

## 7. 第一轮结论

模块主体是好的:四关顺序、最长匹配、frozen dataclass、置信度永不返"高"、阈值启动校验
全部按 spec 工作,20 条断言里 17 条通过。卡住的是 **brief 自身 3 处测试/实现矛盾**,其中
1 处(矛盾 3)是**实现的真实遗漏**(`permanent` 从未置位,而 spec §5.2 有 8 条"永久封停"),
另 2 处是**测试选键失误**(选了被封停的列)。我不自行裁决方向,请控制器按第 4 节裁定;
补丁已逐条实测,预计一轮可 GREEN 并提交。

---
---

# 第 8 节:fix 轮记录(控制器裁决后)

## 8.1 裁决要点(控制器,已逐条核过我的三条矛盾)

| # | 我的上报 | 裁决 | 落法 |
|---|---|---|---|
| 矛盾 1 | `test_at_threshold_passes` 用被封停的 `blink_rate` 测 G3 边界 | **测试错,封停名单对**(spec §5.2:blink_rate 封停到 M2) | 测试改用 `pitch_median`(阈值 10,未被封停) |
| 矛盾 2 | `test_clean_column_passes` 把含 `pause_duration` 子串的列当干净列 | **测试错** | 测试改用 `interview_duration_mean` |
| 矛盾 3 | 8 条 `unblock="永不"` 的条目一条都没设 `permanent=True` | **实现真漏,测试对** | 显式 `permanent=True`;**不用 `__post_init__` 从字符串反推** |
| 自检发现 A | `"au"` 是 `"pause"` 的子串且排得更前 → `"pause": 2` 是死条目 | **确认为真 bug,采纳我的方向** | `_threshold_for` 改为最长匹配 + 新增 `test_longest_match_wins` 钉住 |
| 自检发现 B | `Check` 字段顺序:brief 接口段与实现段不一致 | **实现赢** | 计划的接口段已改写;代码不动 |

矛盾 3 裁决理由(控制器原话,值得记下):`unblock` 是给人看的显示文本,"永不" 是显示层的
字面量,用一个显示字符串去驱动布尔逻辑是隐式耦合,以后有人改措辞就会静默改掉行为。
—— 我当时倾向 `__post_init__` 派生(C2),这个理由推翻了我的偏好,我认同:字符串是显示层,
不是逻辑层。

## 8.2 改了什么(共 4 处,两个文件)

`tests/test_evidence_gate.py`(3 处)

1. `TestG3EnoughSamples` 加 3 行警示注释 + 两个用例的键 `blink_rate`/5 与 `/20` 改为
   `pitch_median`/9 与 `/10`(边界原意不变:阈值下失败、阈值上通过)。
2. `test_clean_column_passes` 改用 `interview_duration_mean` + 1 行注释说明为何不能用
   `interview_pause_duration_mean`。
3. `TestThresholds` 新增 `test_longest_match_wins`(钉住阈值的最长匹配)。

`report_frontend/evidence_gate.py`(2 处)

4. 8 条永久封停条目加 `permanent=True` + 2 行注释("unblock 是显示文本,逻辑判断一律用
   permanent 字段")。
5. `_threshold_for` 从"注册表序首个匹配"改为**最长匹配**(与 `is_quarantined` 同一套规则),
   加 7 行 docstring 说明为何不能用首个匹配。

`evidence_thresholds.json` 与 `pytest.ini` 本轮**未改**;JSON 内容与第一轮逐字一致。

两个文件改完后均已用 `diff` 与**重新生成的 brief**(392 行)的对应代码块逐字节比对:
`IDENTICAL`(测试块 `sed -n '48,167p'`,实现块 `sed -n '178,376p'`)。

## 8.3 覆盖的测试(全部有关键断言,不只是"跑过了")

| 裁决 | 钉住它的测试 | 断言 |
|---|---|---|
| 矛盾 1 | `TestG3EnoughSamples::test_below_threshold_fails` / `::test_at_threshold_passes` | n=9 时 `gate_name == "G3"` 且 `ok is False`;n=10 时 `ok is True`(阈值 10 的边界两侧) |
| 矛盾 2 | `TestG4NotProxy::test_clean_column_passes` | `interview_duration_mean` 四关全过 `ok is True` |
| 矛盾 3 | `TestG4NotProxy::test_permanent_quarantine_has_no_unblock` | `is_quarantined("overall_score").permanent is True` |
| 发现 A | `TestThresholds::test_longest_match_wins` | `_threshold_for("pause_duration_mean") == 2` 且 `_threshold_for("au12_smile_mean") == 30` |

另:第一轮已通过的 17 条断言本轮**未改动**,仍然通过(未为了让新代码过关而放宽任何断言)。

## 8.4 RED —— fix 前的失败(brief 已改、实现还是旧版)

命令:

```
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_evidence_gate.py -v
```

输出(尾部):

```
tests/test_evidence_gate.py:74: AssertionError
    def test_permanent_quarantine_has_no_unblock(self):
        q = is_quarantined("overall_score")
>       assert q is not None and q.permanent is True
E       AssertionError: assert (Quarantine(reason='属性不存在,getattr 走默认值', unblock='永不', permanent=False) is not None and False is True)

tests/test_evidence_gate.py:112: AssertionError
    def test_longest_match_wins(self):
>       assert _threshold_for("pause_duration_mean") == 2
E       AssertionError: assert 30 == 2
E        +  where 30 = <function _threshold_for at ...>('pause_duration_mean')

=========================== short test summary info ============================
FAILED tests/test_evidence_gate.py::TestG4NotProxy::test_permanent_quarantine_has_no_unblock
FAILED tests/test_evidence_gate.py::TestThresholds::test_longest_match_wins
========================= 2 failed, 19 passed in 0.29s =========================
```

为什么这 2 条失败是预期的、且正好是应有的 2 条:改的是测试、没改实现,所以**只有**
那两条指认实现缺陷的断言会红 —— `permanent` 从未置位、`pause_*` 拿到 au 的阈值 30。
两条**测试侧**修正(矛盾 1、2)此时已经转绿,符合"测试侧的错改了测试就好了"的预期。
若这里出现第 3 条失败,说明我改测试时碰坏了别的东西。

## 8.5 GREEN —— fix 后全绿

命令:

```
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_evidence_gate.py -v
```

输出(尾部):

```
collecting ... collected 21 items

tests/test_evidence_gate.py::TestG1HasValue::test_none_fails PASSED      [  4%]
...
tests/test_evidence_gate.py::TestG4NotProxy::test_permanent_quarantine_has_no_unblock PASSED [ 61%]
...
tests/test_evidence_gate.py::TestThresholds::test_longest_match_wins PASSED [ 95%]
tests/test_evidence_gate.py::TestGateOrder::test_first_failure_wins PASSED [100%]

============================== 21 passed in 0.31s ==============================
```

全仓裸 pytest(Ruling 9):

```
~/miniconda3/envs/jingxin/bin/python -m pytest
→ 22 passed in 0.24s        (21 条本任务 + 1 条 smoke)
```

无 collection error、无 warnings summary、无 stray output。21 项 = 第一轮 20 条断言 + 新增的
`test_longest_match_wins`。

## 8.6 改完后的复验探针(超出测试范围,直接查实现行为)

```
permanent count  : 8 (expect 8)          ← 恰好是 8 条 unblock=="永不"
unblock 永不      : 8
overall_score    : True                   ← 矛盾 3 已修
blink_rate perm  : False                  ← 正确:等 M2,不是永久封停
thresholds       : {'pause_duration_mean': 2, 'au12_smile_mean': 30,
                    'interview_pause_segment_count': 2, 'pitch_median': 10,
                    'speech_ratio': 2, 'blink_rate': 20, 'gaze_x': 30,
                    'head_pose_yaw': 30, 'energy_mean': 10}
default          : 10 (expect 10)         ← 未登记指标仍走保守默认
longest quarantine: argmax,非置信度        ← is_quarantined 最长匹配仍然正确
gate order        : G1 G2 G3              ← 关卡顺序仍然正确
confidence 高 cnt : 0                     ← 3600 组穷举仍无一返"高"
```

`pause_duration_mean` 从 30 变 2、`interview_pause_segment_count` 从 30 变 2 —— 即发现 A
的静默缺陷已消除;`au12_smile_mean` 仍然是 30(未被误伤);`blink_rate` 仍然 20(未被误伤);
第一轮的 3 项自检(最长匹配 / 关卡顺序 / 置信度永不返"高")在改动后**重新实测**仍然成立。

## 8.7 提交

命令(brief Step 6 的路径 + Ruling 9 的 `pytest.ini`;身份按 Ruling 10 用 `-c` 带参,未写任何配置文件):

```
git add report_frontend/evidence_gate.py report_frontend/evidence_thresholds.json \
        tests/test_evidence_gate.py pytest.ini
git -c user.name='huihuibuhui227' -c user.email='huihuibuhui227@gmail.com' \
    commit -m "feat: 新增证据门模块(四关 + 封停名单 + 置信度三值)" -m "..."
```

结果:

```
4 files changed, 337 insertions(+)
 create mode 100644 pytest.ini
 create mode 100644 report_frontend/evidence_gate.py
 create mode 100644 report_frontend/evidence_thresholds.json
 create mode 100644 tests/test_evidence_gate.py

08e31fe feat: 新增证据门模块(四关 + 封停名单 + 置信度三值)
8d156f4 feat: 三个模块 API 服务与报告层实时模式
```

- **恰好 4 个显式路径**,`git diff --cached --name-only` 逐个核对过,无 `-A`、无多余文件。
- 提交体记录了:四关语义、阈值可识别为临时、最长匹配的理由、`permanent` 为何显式置位、
  以及 `pytest.ini` 属 Ruling 9 的附带改动。
- 提交前后均确认四个文件为 **LF**(`.gitattributes` 生效)。
- 该提交**未被 amend / rebase**,是分支上的新提交。

工作区残留(均**非**本任务产出,未触碰):`report_frontend/__pycache__/*.pyc`(仓库既有
的 .pyc 跟踪问题,账本已记为 deferred)、`tests/__pycache__/`、`.pytest_cache/`、
以及 `report_frontend/data/output/*.html` —— 后者 mtime 为 2026-03-14 / 2026-04-22,
早于本会话(2026-09-22),是使用者自有产物。

## 8.8 第一轮遗留的两条(本轮均已闭环)

- **发现 A** → 已修(`_threshold_for` 最长匹配)+ 测试钉住。**不再是遗留。**
- **发现 B** → 无需改代码(brief 接口段已改),我先前建议的"以实现为准"已被采纳。**闭环。**

无新增 concerns。

---
---

# 第 9 节:第二轮审查修复(finding 1 / 2)

审查结论:**Spec ✅ 合规 / Task quality Approved,0 Critical,2 Important**。两条均本轮修完,
提交为**两个独立 fix 提交**(未 amend `08e31fe`)。

## 9.1 改了什么

### Finding 1(Important,plan-mandated)—— 默认阈值是裸常量

`_threshold_for` 的 `return 10  # 未登记指标的保守默认` 管辖所有未登记指标,却不在版本化
/临时标记体系内,M5 替换 JSON 时会残留下来静默管辖新指标。违反 spec §5.1 与计划的
Global Constraints("禁止写成代码里的裸常量")。**这是 brief 自身违反 spec,spec 是权威。**

| 改动 | 内容 |
|---|---|
| `evidence_thresholds.json` | 新增 `"_default_n_valid": 10`(放在 `_units` 之后) |
| `load_thresholds` | 守卫加严:`data.get("_provisional") is not True or not data.get("_version")`(顺带修 Minor:原守卫只查键存在,把 `_provisional` 改成 `false` 也能过) + 新增 `_default_n_valid` 存在性守卫 |
| `_threshold_for` | 改为 `data = load_thresholds()` 后 `return int(data["_default_n_valid"])` |
| 测试 | `TestThresholds` 新增 `test_default_threshold_lives_in_json` |

### Finding 2(Important,会漏到 T3/T6)—— 内部文案会渲进报告

`check_g4_not_proxy` 把面向维护者的封停理由写进 `Check.reason`;Task 3 的
`dim_gaps.append(f"{human_name}: {chk.reason}")` 会把它带进 Task 6 渲染的"证据缺口"一节,
用户会看到"已封停:可由同 block jitter 精确重构(解封:M3)"。spec §5.6 对 quarantine 文本的
豁免只覆盖"留在代码里不进输出"。

| 改动 | 内容 |
|---|---|
| `evidence_gate.py` | 新增 `_USER_MESSAGES`(G1~G4 各一句中性文案)与 `user_message(check)`;未知 `gate_name` 回退"证据不足" |
| `Check.reason` | **保持原样不动**(维护者用,Task 3 之后可进日志)—— 本次是**加**一层用户文案,不是替换 |
| 测试 | 新增 `TestUserMessage`,两条:一条扫泄露词(`封停`/`jitter`/`M3`/`精确重构`),一条钉四关文案(第 2 条用 `std=0.0` 走 `check_g2_from_std`,不是 `values=`) |

## 9.2 覆盖的测试(3 条新断言,均有实质内容)

| Finding | 测试 | 断言 |
|---|---|---|
| 1 | `TestThresholds::test_default_threshold_lives_in_json` | JSON 里有 `_default_n_valid`,且 `_threshold_for("some_unregistered_metric")` 等于它 |
| 2 | `TestUserMessage::test_user_message_hides_maintainer_text` | G4 缺口文案非空,且不含 `封停`/`jitter`/`M3`/`精确重构` |
| 2 | `TestUserMessage::test_user_message_per_gate` | 四关各返回规定文案(含 G2 走 `std=0.0` 路径) |

原有 22 条断言**一条未改**,仍全过(未为让新代码过关而放宽任何断言)。

## 9.3 RED —— 新测试对旧实现

命令:`~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_evidence_gate.py`

```
    def test_default_threshold_lives_in_json(self):
        data = load_thresholds()
>       assert "_default_n_valid" in data
E       AssertionError: assert '_default_n_valid' in {'_version': '0.1.0-provisional', '_provisional': True, ...}

=========================== short test summary info ============================
FAILED tests/test_evidence_gate.py::TestUserMessage::test_user_message_hides_maintainer_text
FAILED tests/test_evidence_gate.py::TestUserMessage::test_user_message_per_gate
FAILED tests/test_evidence_gate.py::TestThresholds::test_default_threshold_lives_in_json
========================= 3 failed, 21 passed in 0.31s =========================
```

为什么正好这 3 条:前两条 `ImportError: cannot import name 'user_message'`(函数尚不存在),
第三条断言 JSON 缺键。没有第 4 条失败 → 改动没有波及既有行为。

## 9.4 GREEN

```
$ ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_evidence_gate.py -v
tests/test_evidence_gate.py::TestUserMessage::test_user_message_hides_maintainer_text PASSED [ 79%]
tests/test_evidence_gate.py::TestUserMessage::test_user_message_per_gate PASSED [ 83%]
tests/test_evidence_gate.py::TestThresholds::test_default_threshold_lives_in_json PASSED [ 91%]
tests/test_evidence_gate.py::TestThresholds::test_longest_match_wins PASSED [ 95%]
tests/test_evidence_gate.py::TestGateOrder::test_first_failure_wins PASSED [100%]
============================== 24 passed in 0.26s ==============================

$ ~/miniconda3/envs/jingxin/bin/python -m pytest          # 全仓裸 pytest
============================== 25 passed in 0.26s ==============================
```

25 项 = 原 22 + 新增 3,与预期一致。无 collection error、无 warnings、无 stray output。

## 9.5 两个 fix 提交(分开提交,均自带绿测试)

按"多个逻辑单元分开提交"拆分,**按 finding 切**,先保存最终态到 `/tmp/t2final/`、
`git checkout --` 三个文件回到 `08e31fe`、只应用 Finding 1 → 跑绿 → 提交 →
再应用 Finding 2 → 跑绿 → 提交。这样**每个提交自身都是绿的**,不是把两个 findings 混在一起。

```
e249450 fix: 新增 user_message,报告层不得直接渲染 Check.reason     (2 files, +42)
fa44d99 fix: 未登记指标的默认阈值移入 JSON 并纳入临时标记           (3 files, +20 −4)
08e31fe feat: 新增证据门模块(四关 + 封停名单 + 置信度三值)
```

- 拆完后的最终态与拆分前逐字节相同(`diff /tmp/t2final/*` = 无差异),且三个文件与
  重新生成的 brief(451 行)对应代码块 `diff` 逐字节一致(测试块 `sed -n '50,200p'`,
  实现块 `sed -n '211,435p'`,JSON 块 `sed -n '28,44p'`)。
- **未 amend `08e31fe`**,两个都是新提交。
- 三个文件均为 LF;这三个路径的工作区为 clean(无残留改动)。

## 9.6 复验探针(提交后,直接在已提交状态上跑)

```
user_message G1..G4 : ['未采集到对应数据', '本次会话内无变化', '有效样本不足', '该指标本轮停用']
fallback (未知关)    : 证据不足
reason 保持原样      : 已封停:3 个硬编码取值,99.7% 恒 0.3(解封:M3)   ← 维护者文案未被破坏
默认值随 JSON 变     : JSON 里改成 7 → _threshold_for("unregistered_xyz") == 7   ← 证明不是硬编码
guard _provisional=false : ValueError ok
guard 缺 _default_n_valid: ValueError ok
permanent 计数: 8 | pause_duration_mean: 2 | au12_smile_mean: 30
quarantine 最长匹配: argmax,非置信度 | 关卡顺序: G1 G2 G3 | confidence 返"高"次数: 0
```

"默认值随 JSON 变"这条是关键的**反作弊**验证:只在 JSON 里改数字(不动代码)就能改变
行为,才证明默认值真的来自配置而非裸常量 —— 只断言"键存在"是测不出这点的。

## 9.7 审查者另报的 6 条 Minor(控制器判:不进本轮,记录在案)

G2 无 std 时 fail-open(brief 有意为之,已有测试钉住;**Task 3 派发时需点名要求必传
`_std`**)、`test_first_failure_wins` 断言可更严、未用 import 噪音、`_threshold_for` 每次
读盘、pause 族 G3 阈值由被遮蔽的 30 变为登记值 2(**修 bug 的预期结果,Task 3 需知悉**)。

最后一条与第 8 节的发现 A 是同一件事的两面:第 8 节把它作为"死条目"上报,本轮前的裁决
只修了匹配规则使登记值 2 生效;副作用是 **pause 族指标在 Task 3 里会从"几乎必过 G3"
变为"按 n_valid ≥ 2 判定"** —— 这是正确行为,但属于行为变化,已在 9.8 记录给 Task 3。

## 9.8 交接给 Task 3 的两条(不是 concern,是必需的上下文)

1. **`_std` 必传**:G2 在 `std=None` 时 fail-open(brief 有意为之)。Task 3 若只在有序列时
   才判定常量性,恒定的坏列会漏过 G2 —— 派发 Task 3 时应点名要求传 `_std`。
2. **报告层只准用 `user_message(check)`**,不得渲染 `check.reason`(内部文案会外泄)。
   `Check.reason` 仍保留给日志。
3. **pause 族 G3 阈值现在是 2**(此前被 `au` 遮蔽为 30)—— 行为变化,符合 spec 登记值。

无新增 concerns。
