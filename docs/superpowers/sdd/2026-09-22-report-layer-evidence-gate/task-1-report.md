# Task 1 报告:测试脚手架

**Status:** DONE_WITH_CONCERNS

## 实现内容

按 brief 逐字创建两个文件(未改动任何其他文件):

- `tests/conftest.py` — 把项目根 `/home/huihuibuhui/jingxin` 插入 `sys.path`
- `tests/test_smoke.py` — 冒烟测试 `test_package_imports`

**跳过 brief 的 Step 1**:controller 已预先创建分支 `fix/report-layer-evidence-gate`。
提交前用 `git rev-parse --abbrev-ref HEAD` 确认输出为 `fix/report-layer-evidence-gate`。

未创建 `tests/__init__.py`(pytest 在 rootdir + conftest 模式下不需要,且 brief 未要求)。

## 测试与结果

| 命令 | 结果 |
|---|---|
| `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_smoke.py -v` | **1 passed in 0.59s** |
| `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v` | **1 passed in 0.22s** |

输出干净:无 warning、无 stray output,`rootdir: /home/huihuibuhui/jingxin`。

### 额外验证(证明测试不是空转)

brief 的`assert X is not None`在 import 成功后不可能失败,真正的断言是那行 import 本身。
所以我做了两个对照,验证该测试确实"有牙齿":

1. **换 cwd**:`cd /tmp && pytest /home/.../tests/test_smoke.py` → **PASSED**。
   证明是 conftest 在起作用,而非 cwd 恰好是项目根。
2. **换解释器(负对照)**:`~/huihui/bin/python -c "import report_frontend"`
   → `ModuleNotFoundError: No module named 'plotly'`
   (栈:`report_frontend/visualizer.py:3: import plotly.graph_objects as go`)。
   而 jingxin 解释器 → `OK: 1.0.0`。
   即:若用错解释器,该测试会以 brief 预告的那个错误失败。

注:`~/huihui/bin/python` 未装 pytest,故负对照改用直接 import 而非 pytest 运行。

## TDD Evidence

**本任务没有 failing-test-first 环节。** brief 的 Step 2–4 是"先创建、再验证"的
create-then-verify 流程,不存在 RED 阶段,故不编造 RED/GREEN 输出。
上述两个对照实验是本任务能提供的最接近的"测试有效性"证据。

## 文件变更

Commit `0362b98` — `test: 为报告层引入 pytest 脚手架`

```
A  tests/conftest.py   (+7)
A  tests/test_smoke.py (+7)
```

`git show --name-only HEAD` 确认恰好这两个文件;按约束只用显式路径 `git add`
(未用 `git add -A/.`),已在提交前用 `git diff --cached --name-status` 核对只有 2 条 `A`。

## 自查发现

1. **diff 与 brief 逐字一致**(已 `git show HEAD` 通读)。
2. **测试输出无 warning**。
3. **未越界**:`report_frontend/`、`templates/` 均未被我修改(见"遗留问题"第 2 条)。
4. 已清理我自己造成的 collateral(见"遗留问题"第 2 条)。

## 遗留问题 / Concerns

1. **[需 controller 决策] 裸跑 `pytest` 在仓库根会收集失败。**
   根目录存在 4 个**未被 git 跟踪的**临时脚本,名字以 `test_` 开头,被 pytest 误当测试收集,
   导入期即报错(其中一个还硬编码了 Windows 路径 `D:/jingxin/...`):

   ```
   ERROR experiments/test_gpt56.py
   ERROR experiments/test_codexapis.py
   ERROR code_data_supplement/src/test_gpt56.py
   ERROR code_data_supplement/src/test_codexapis.py
   !!! Interrupted: 4 errors during collection !!!
   ```

   这是**先于本任务存在的**状态(4 个文件均 `git ls-files` 为空,即未跟踪,与本 diff 无关),
   本任务未引入、也未修复。但**后续任务/CI 若运行裸 `pytest` 会立刻撞上**。
   修法需在根加 `pytest.ini`(`testpaths = tests`)或改名这 4 个脚本 —— 两者都**超出
   Task 1 的 brief 规格,也超出"仅可改 `report_frontend/` 与 `templates/`"的 scope**,
   因此我**没有**动手,交 controller 决定。

2. **工作树在我开始前就是脏的**,与我的提交无关。`report_frontend/` 与 `templates/`
   有大量已跟踪文件显示为 modified。我核对了 mtime:`report_frontend/__init__.py` =
   2026-03-14、`data_loader.py` = 2026-05-25、`templates/dashboard.html` = 2026-03-14,
   **远早于本次会话**(我的文件 = 今天 21:41)。reflog 显示 `HEAD@{1}: checkout, moving
   from main to fix/report-layer-evidence-gate`,即脏状态是随分支切换带过来的。
   diff 形态为 182 insertions / 182 deletions ≈ 每行都变,典型是 **CRLF/LF 行尾差异**。
   注:我上下文里的 `gitStatus` 快照声称 "(clean)",**该快照是过期/不准的**。

3. **我的负对照命令留下过 collateral,已清理。** `~/huihui/bin/python -c "import
   report_frontend"` 在失败前仍对先前模块做了字节编译,生成了 5 个
   `report_frontend/__pycache__/*.cpython-312.pyc`(此前不存在)。**已全部删除**,
   现 `find -name "*.cpython-312.pyc"` 为空。
   剩余影响:运行 pytest 会重新生成 `report_frontend/__pycache__/*.cpython-311.pyc`
   (这些 pyc 在本仓库**是被跟踪的**)。这是"跑测试"这一强制验证步骤的必然结果,
   任何 reviewer 跑测试都会产生同样 churn,故保留未动。

4. **`.pytest_cache/` 未被 `.gitignore` 覆盖**(`git check-ignore` 返回 NOT ignored)。
   本任务因只用显式 `git add` 而未被提交,但若日后有人用 `git add -A` 会误入。
   `.gitignore` 属配置,超出 scope,未改。

5. **本仓库未配置 git identity**,`git commit` 直接报
   `fatal: 不允许空的姓名`。我用一次性 `git -c user.name="huihuibuhui227"
   -c user.email="huihuibuhui227@gmail.com"`(取自本仓库历史既有 author)**完成提交,
   未写入任何 global/local 配置文件**。后续任务提交时会遇到同样问题,需同样处理。

6. brief 的两行 `assert ... is not None` 在 import 成功后是恒真的,信息量主要来自 import
   语句本身。因 brief 要求逐字照抄,故保留原样,未"加强"。
