# M2 复审三项修复(17 / 18 / 19)实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ① 本场没有任何日志时报告**照样出**,并在头里说清是哪一场、缺了什么;② 前端两条读路径带上 `session_id`,且结构化 JSON 里**自己就能看出这是哪一场**;③ 科研评估链路**自己铸号**,不再把科研回答写进上一场面试的目录。

**Architecture:** 三处都落在既有的「静默失败」族上,不引入新机制。17 = 取消 `generate_report` 的 `raise`,把「目标会话」作为加载器的公开属性交给渲染层(实测:空数据下特征引擎给 0 个指标、映射器给 `覆盖 0/20`、图表照常生成,**这条链本身是空的能跑的**)。18 = 复用既有的 `withSession` 拼参,并让 `/api/report/structured` 回传 `session_id` + `selected_sessions`。19 = 让 `/research/start` 与 `/interview/start` 对称铸号(使用者 2026-09-25 裁定:方案 A)。

**Tech Stack:** Python 3.11(.venv 用 miniconda `jingxin`) / Flask(总控 `app.py`) / FastAPI(voice `:8001`) / pytest / React + Vite + TS(前端独立仓库 `~/JingXin-frontend`)

**Spec:**
- `docs/下一步.md` §3 第 17–19 条(本轮的三条要求)
- `docs/superpowers/specs/2026-09-24-m2-read-side-session-alignment-design.md` **§6 错误处理**(三行契约:一个模态都没找到 → 仍生成、逐条列缺什么、不抛;全部模态都缺 → 生成并显式说明本场没有任何日志、不回退拼别的场次;有帧落 `NONE` → 报告头要点出 `NONE` 桶存在)
- 复审原始报告 `docs/superpowers/sdd/2026-09-24-m2-read-side-session-alignment/final-review-report.md`(I2 / I3 / I4 及其修法建议)

## Global Constraints

- 解释器**固定** `~/miniconda3/envs/jingxin/bin/python`(`~/huihui/bin/python` 缺 websockets,不可用)。
- 任何改动都要能过**合并门**:
  ```bash
  cd ~/jingxin/experiments/duration_audit
  PY=~/miniconda3/envs/jingxin/bin/python
  $PY reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_probe
  $PY reaggregate_normalized.py --verify-legacy /tmp/legacy_probe   # 期望 0 / 2835510
  ```
- 全量 `pytest` 必须绿(本轮开始前是 **204 项**)。
- 报告层**不渲染分数与评级**(spec §5.4 / §5.6)。本次改动**不得**让 `total_score` / 档位标签回到任何**报告产物**里。
- 每个新测试都要能说出「**哪个生产改动会让它变红**」;写完做**反向复现**(撤掉生产改动 → 确认红得对),再做恢复。变异验证前清 `__pycache__`(同秒内"写入→跑→还原"会跑变异字节码)。
- 前端仓库 `~/JingXin-frontend` 工作树里有**一批使用者先前未提交的改动** —— 提交时**只 `git add` 本计划点名的文件**,绝不 `git add -A`。
- 阈值/常量一律进数据文件(带 `_provisional` + 依据)。本轮**不新增**任何阈值常量。

## Review Focus

以下五类输入,spec 暗示必须处理但没有测试覆盖,是最可能咬到真实使用者的地方(每条都在对应任务里配了钉子测试):

1. **本场一个模态都没有**(连 `NONE` 桶都没有)→ 仍要出一份报告,并写明「本场没有任何日志」;不许回退去拼别的场次。
2. **只有 `NONE` 桶**(前端还没拿到铸号就发帧)→ 报告头必须点出「另有 NONE 桶 N 行」。
3. **显式指了一个不存在的 `session_id`** → 四个模态逐条列「缺失」,不抛、不 500、不返回空。
4. **`/api/report/structured` 在空数据下** → 返回 `status=success` + `coverage 0/20` + `session_id`,**不是** `{"status":"error","message":"未找到评估日志数据"}`。
5. **面板子进程「没产出但 exit 0」** → `/api/task/<id>` 必须报 `error`,不再报「任务完成！」。

## 本轮明确不做(记下,防顺手扩)

- **前端 `ReportPage` 仍在渲染 `total_score` 与档位标签**(`~/JingXin-frontend/src/pages/ReportPage.tsx:92-95`:`report.total_score.toFixed(1)` + `getLevelLabel(report.total_score)`)—— 那是**把未标定标尺上的复合点分当对候选人的评定**,与报告层 §5.4/§5.6 已经停止渲染的东西直接冲突(HTML 报告里已经不印了,前端还在印)。**不在 17–19 范围内**(它是前端呈现层的独立决定,与 §3 第 21–23 条同族),但**必须记下并交给使用者裁定**。
- `~/shared/m2_acceptance.sh` 的两个卡死 bug(§3 第 24 条)、`socket.io-client` 死依赖(第 23 条)、报告入口收口(第 22 条)、OpenAPI → TS 类型(第 21 条)。
- `voice` 服务进程级单例(第 4 条)、409 归属契约(第 5 条)、`get_live_data` 丢指标(第 6 条)、`silence_energy_floor` 实测(第 9 条)、VIDEO 等价性门(第 15 条)、M3 阈值重登记(第 16 条)。

---

### Task 1: 报告层 —— 本场没有任何日志也出报告(第 17 条 / 复审 I2)

**Files:**
- Modify: `report_frontend/data_loader.py`(`LogDataLoader.__init__` 尾部加属性;`get_fused_latest_data` 的前 20 行)
- Modify: `report_frontend/report_generator.py`(`sources_disclosure` 签名;`generate_report` 去 `raise`;`generate_report_live` 传 `target`;`_build_html_report` 收 `target`;文件末尾的 `__main__` 换成 `main()`)
- Test: `tests/test_report_empty_session.py`(新建)

**Interfaces:**
- Consumes: 无(本任务是最底层的一环)
- Produces:
  - `LogDataLoader.target_session: Optional[str]` —— **每次** `get_fused_latest_data()` 都重设,`None` 表示"本场没有任何会话"
  - `sources_disclosure(sources: Dict[str, Dict], target: Optional[str] = None) -> str`
  - `report_generator.main(argv: Optional[list[str]] = None) -> int` —— 有报告落盘返回 `0`,否则 `1`
  - `ReportGenerator._build_html_report(..., sources=None, target=None)`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_report_empty_session.py`:

```python
# tests/test_report_empty_session.py
"""第 17 条(复审 I2):**本场没有任何日志时,报告照样要出**。

spec §6 的三行契约:
    目标 id 一个模态都没找到 → 报告仍生成,头里逐条列出缺什么;不抛
    目标 id 全部模态都缺     → 生成报告并显式说明本场没有任何日志;不回退去拼别的场次
    前端在拿到铸号前发帧     → 报告头要点出 NONE 桶存在

失效现场(2026-09-24 复审实测):`report_frontend/report_generator.py:121` 的
`raise ValueError("无面部数据")` 被同一个 try 的 `except` 吞成 `return ""` ——
**一份报告都没有**;而 `app.py:53` 只看子进程 returncode,于是面板报「任务完成！」。
"""

import csv
from pathlib import Path

import pytest

import report_frontend.report_generator as report_generator
from report_frontend.data_loader import LogDataLoader
from report_frontend.report_generator import ReportGenerator, main, sources_disclosure

_SID = "20260924_230914_262f"


def _write(log_dir: Path, filename: str, rows: list) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    fields = ["session_id", "timestamp", "connective_density", "connective_density_std", "n_rows"]
    p = log_dir / filename
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _none_row() -> dict:
    return {"session_id": "NONE", "timestamp": "2026-09-24T23:00:35",
            "connective_density": None, "connective_density_std": None, "n_rows": 1}


def _session_row(sid: str) -> dict:
    return {"session_id": sid, "timestamp": "2026-09-24T23:09:14",
            "connective_density": 2.5, "connective_density_std": 0.1, "n_rows": 1}


@pytest.fixture
def render(tmp_path, monkeypatch):
    """真加载器 + 真渲染,只把日志目录与"打开浏览器"换掉。

    **不** import 总控 `app.py`(F2 的端点由 Task 2 的测试压),这里只压报告这条链。
    """
    def _run(session_id=None, files=()) -> str:
        log_dir = tmp_path / "logs"
        for name, rows in files:
            _write(log_dir, name, rows)
        monkeypatch.setattr(report_generator, "LogDataLoader",
                            lambda *a, **k: LogDataLoader(str(log_dir)))
        monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)

        path = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(session_id)
        assert path, "报告没生成 —— 本测试的前提不成立"
        return Path(path).read_text(encoding="utf-8")

    return _run


def test_report_is_still_generated_when_only_the_none_bucket_exists(render):
    """★ 只有 NONE 桶(前端还没拿到铸号就发帧)→ 报告仍出,且点出 NONE 桶行数。

    红法:恢复 `if not data or 'face' not in data: raise ValueError("无面部数据")`
    → `generate_report` 返回 `''` → 上面的 `assert path` 先红。
    """
    html = render(files=[("face_au_log_NONE.csv", [_none_row()])])

    assert "本场没有任何日志" in html, "只有 NONE 桶时没说清"本场根本没有日志""
    assert "NONE 桶 1 行" in html, "存在 NONE 桶却没在报告头点出来(spec §6 行 3)"


def test_report_is_still_generated_when_there_are_no_logs_at_all(render):
    """★ 连 NONE 桶都没有 → 仍要出一份报告,并写明本场没有任何日志(Review Focus 第 1 条)。

    红法:同上的 `raise`;另一条红法是让 `sources_disclosure` 在 items 为空时走早退
    (旧的两处 `return "本报告没有装配任何模态日志。"`)→ 句子里没有「本场没有任何日志」。
    """
    html = render(files=[])

    assert "本场没有任何日志" in html


def test_report_lists_every_missing_modality_when_the_named_session_has_no_logs(render):
    """★ 显式指了一个没有日志的 id → 报告仍生成,四个模态逐条列「缺失」。

    红法:同上的 `raise`。
    """
    html = render(session_id=_SID, files=[("face_au_log_NONE.csv", [_none_row()])])

    assert _SID in html, "报告没点名它描述的是哪个 session_id"
    assert html.count("缺失（本场没有这个模态的日志）") == 4, (
        f"没有逐条列出四个模态缺什么,实际命中 {html.count('缺失（本场没有这个模态的日志）')} 次")


def test_a_loaded_session_still_renders_normally(render):
    """反向的那一侧:有数据的会话照旧出报告(改动不许把正常路径弄坏)。"""
    html = render(session_id=_SID,
                  files=[(f"face_au_log_{_SID}.csv", [_session_row(_SID)])])

    assert _SID in html and "已读入" in html
    assert "本场没有任何日志" not in html


def test_disclosure_without_a_target_says_there_is_no_session():
    """`target` 不给且没有任何模态条目 → 必须说「本场没有任何日志」,不许编一个 id 出来。

    红法:退回旧的两处早退(`if not items: return "本报告没有装配任何模态日志。"`)
    → 句子里没有「本场没有任何日志」→ 红。
    """
    html = sources_disclosure(
        {"none_bucket": {"session_id": "NONE", "status": "present", "rows": 3}})

    assert "本场没有任何日志" in html
    assert "NONE 桶 3 行" in html


def test_cli_exits_nonzero_when_no_report_was_written(monkeypatch):
    """★ 复审 I2 的额外症状:面板只看子进程 returncode ——
    "什么都没生成但 exit 0" 会被显示成「任务完成！」(`app.py:53`)。

    红法:退回旧的 `__main__`(不设退出码)→ `main` 不存在 → ImportError 红;
    若只把 `main` 写成恒返回 0,则本条断言红。
    """
    monkeypatch.setattr(ReportGenerator, "generate_report", lambda self, sid=None: "")

    assert main([]) == 1


def test_cli_exits_zero_and_passes_the_session_id_through(monkeypatch):
    """`--session-id` 要真的传下去,产出了报告才返回 0。"""
    seen = []
    monkeypatch.setattr(ReportGenerator, "generate_report",
                        lambda self, sid=None: seen.append(sid) or "/tmp/x.html")

    assert main(["--session-id", _SID]) == 0
    assert seen == [_SID]
```

- [ ] **Step 2: 跑测试,确认红**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_empty_session.py -v
```
Expected: `ImportError: cannot import name 'main'`(收集阶段就红);把 `main` 暂时从 import 里去掉后,前三条应以 `报告没生成 —— 本测试的前提不成立` 失败。

- [ ] **Step 3: 改 `report_frontend/data_loader.py`**

3a. `__init__` 里 `self.selected_sessions: Dict[str, str] = {}` 那段之后追加:

```python
        # 本报告以哪个 session_id 为准(M2.1 / 第 17 条)。`selected_sessions` 的各模态条目里
        # 也各带一份,但**只有 NONE 桶、或一场都没有**时那是空的 —— 而那正是最需要把
        # "本场是哪一场"说出口的情形,所以目标 id 单独作为一个公开属性交出去。
        self.target_session: Optional[str] = None
```

3b. `get_fused_latest_data` 开头改成(把 `none_bucket` 的统计**提到早退之前**):

```python
        target = self.resolve_target_session(session_id)
        self.selected_sessions = {}
        self.target_session = target
        print(f"   目标会话:{target or '(无 —— 本场没有任何日志)'}")

        # ⚠️ NONE 桶要在"有没有目标会话"**之前**填:只有 NONE 桶时 target 是 None,
        # 按原来的顺序(先 return 再数)这份报告连"另有 N 行没归入本场"都说不出来
        # —— 而那正是 spec §6 行 3 要求说出口的事。
        none_rows = self._none_bucket_rows()
        if none_rows:
            self.selected_sessions["none_bucket"] = {
                "session_id": self.NONE_SESSION, "status": "present", "rows": none_rows}
            print(f"   ℹ️  另有 NONE 桶 {none_rows} 行(未归入任何会话,不进聚合)")

        if target is None:
            print("❌ 没有可用于本报告的会话(只有 NONE 桶,或没有任何符合命名规范的日志)。")
            return {}
        print("-" * 70)
```

(删掉原 `print(f"   目标会话:{target}")`、原 `if target is None` 之后的 `none_rows` 那一段,以及它下面那句 `print("-" * 70)` —— 位置已经前移。)

- [ ] **Step 4: 改 `report_frontend/report_generator.py`**

4a. `sources_disclosure` 换成:

```python
def sources_disclosure(sources: Dict[str, Dict], target: Optional[str] = None) -> str:
    """披露「本报告描述的是哪一场、每个模态进来了没有」(M2 spec §5.2)。

    输入是 `LogDataLoader.selected_sessions`,值是三态之一:
    `loaded` / `unreadable`(文件在但读不出)/ `missing`(本场没有);
    另有一个特殊键 `none_bucket`(没带 session_id 的行,不进聚合但要说一声)。

    `target` 是"本报告以哪一场为准",由调用方给(`LogDataLoader.target_session` /
    实时路径的入参)。**为什么不从 items 里推第一个**:只有 NONE 桶时 items 是空的,
    推不出来 —— 而那正是最需要把话说清楚的情形。没给且 items 非空时退回"取第一个"
    (兼容直接调用本函数的调用方与测试)。
    """
    items = {k: v for k, v in sources.items() if k != "none_bucket"}
    none_bucket = sources.get("none_bucket")
    if target is None and items:
        target = next(iter(items.values()))["session_id"]

    lines = (["本场会话：<strong>（无 —— 本场没有任何日志）</strong>"] if target is None
             else [f"本场会话：<strong>{target}</strong>"])
    if not items:
        lines.append("本报告没有装配任何模态日志。")

    for key, info in sorted(items.items()):
        label = _MODALITY_LABELS.get(key, key)
        if info["status"] == "loaded":
            lines.append(f"{label} · 已读入（{info['rows']} 行）")
        elif info["status"] == "unreadable":
            lines.append(f'<strong style="color:#A23B72;">{label} · 未读到数据</strong>'
                         f'（文件在，但读不出来：空文件或损坏）')
        else:
            lines.append(f"{label} · 缺失（本场没有这个模态的日志）")

    if none_bucket:
        lines.append(f'<em>另有 NONE 桶 {none_bucket["rows"]} 行 —— '
                     f'那些请求没带 session_id，不属于本场，未参与计算。</em>')

    return "<br>".join(lines)
```

4b. `generate_report` 的 try 体开头改成:

```python
            loader = LogDataLoader()
            data = loader.get_fused_latest_data(session_id)
            if not data:
                # 本场一个模态都没读到:**照样出报告**(spec §6 行 1/2)——
                # 头里逐条列出缺什么,而不是让面板显示"什么都没发生"。
                # 实测:空 data 下特征引擎给 0 个指标、映射器给 覆盖 0/20、五维全 None、
                # 图表照常生成 —— 下面这条链本身是空的能跑的。
                print("   ⚠️  本场没有任何可用日志,仍生成一份只含缺口的报告")

            engine = PsychologicalFeatureEngine(data)
            features = engine.extract_all_features()

            mapper = ResearchCapabilityMapper()
            result = mapper.map_features_to_scores(features)

            viz = ReportVisualizer(output_dir=self.output_dir)
            chart_paths = viz.generate_all_charts(result, df_face=data.get('face'))
            static_images = self._scan_static_images()

            html_content = self._build_html_report(result, chart_paths, features, data,
                                                   static_images,
                                                   sources=loader.selected_sessions,
                                                   target=loader.target_session)
```

(即:删掉 `if not data or 'face' not in data: raise ValueError("无面部数据")`;`data['face']` → `data.get('face')`;多传一个 `target`。)

4c. `generate_report_live` 里那一处调用补上 target:

```python
            html_content = self._build_html_report(result, chart_paths, features, data,
                                                   static_images,
                                                   sources=live_sources(session_id, data),
                                                   target=session_id)
```

4d. `_build_html_report` 签名与头部渲染:

```python
    def _build_html_report(self, result: Dict[str, Any], chart_paths: Dict[str, Any],
                           features: Dict[str, Any], data: Dict[str, pd.DataFrame],
                           static_images: List[str],
                           sources: Optional[Dict[str, str]] = None,
                           target: Optional[str] = None) -> str:
```
头部那行 `{sources_disclosure(sources or {})}` → `{sources_disclosure(sources or {}, target)}`

4e. 文件末尾的 `__main__` 换成:

```python
def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口:`--session-id` 定要描述的那一场。

    **没产出报告就返回非 0**(第 17 条 / 复审 I2 的额外症状):总控面板
    `app.py:run_script` 只看子进程的 returncode —— "什么都没生成但 exit 0"
    会被显示成「任务完成!」。
    """
    import argparse

    ap = argparse.ArgumentParser(description="生成行为观测报告")
    ap.add_argument("--session-id", default=None,
                    help="要描述的那场会话 id;缺省取最新一场")
    args = ap.parse_args(argv)
    return 0 if ReportGenerator().generate_report(args.session_id) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 跑测试,确认绿**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_empty_session.py -v
```
Expected: 6 passed

- [ ] **Step 6: 跑报告侧既有测试(不许回归)**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_sources.py tests/test_read_side_session_selection.py tests/test_log_file_selection.py tests/test_report_layer.py -v
```
Expected: 全绿(`sources_disclosure` 的 `target` 是带缺省的追加参数,既有的一参调用不受影响)

- [ ] **Step 7: 反向复现**

1. 把 `generate_report` 里那三行 `raise` 加回去(`if not data or 'face' not in data: raise ValueError("无面部数据")`)→ `pytest tests/test_report_empty_session.py -v` → 期望前两条测试**红**(且红在 `assert path` 那句),第三条仍绿。
2. 把 `main` 的返回改成恒 `0` → `test_cli_exits_nonzero_when_no_report_was_written` 红。
3. 恢复两处改动,再跑一次确认绿。**清 `__pycache__`**:
   ```bash
   find ~/jingxin/report_frontend -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
   ```

- [ ] **Step 8: 提交**

```bash
cd ~/jingxin
git add report_frontend/data_loader.py report_frontend/report_generator.py tests/test_report_empty_session.py
git commit -m "fix(report): 本场没有任何日志也出报告,并把「哪一场」说出口(复审 I2 / 第 17 条)"
```

---

### Task 2: 前端读路径带 id + 结构化 JSON 自报场次(第 18 条 / 复审 I3)

**Files:**
- Modify: `app.py:217-246`(`/api/report/structured`)
- Modify: `~/JingXin-frontend/src/services/api.ts:33-35`、`:191-206`
- Modify: `~/JingXin-frontend/src/pages/ReportPage.tsx`(整文件,加披露卡)
- Test: `tests/test_structured_report_endpoint.py`(新建,Flask `app.test_client()`)

**Interfaces:**
- Consumes: Task 1 的 `LogDataLoader.target_session`、`loader.selected_sessions`
- Produces:
  - `GET /api/report/structured?session_id=<id>` → `{"status": "success", "session_id": str|None, "sources": {...}, "result": {...}}`(**空数据也是 `success`**)
  - 前端 `dashboardApi.getStructuredReport(sessionId?: string | null)`、`dashboardApi.runModule(module, sessionId?: string | null)`
  - 前端 `withSession(url, override: string | null = sessionId)`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_structured_report_endpoint.py`:

```python
# tests/test_structured_report_endpoint.py
"""第 18 条(复审 I3):`/api/report/structured` 必须说得出"这是哪一场",空数据也要照常回答。

失效现场(2026-09-24 复审实测):
    GET /api/report/structured                          → {"status":"error","message":"未找到评估日志数据"}
    GET /api/report/structured?session_id=<有数据那场>  → {"status":"success", ...}
第一条会打中的是**最新的一场**(可能是一场刚 /interview/start、还没有任何数据的会话),
于是前端报告页显示「暂无报告数据」,而盘上明明有一场三模态齐全的会话。
并且成功那一支的 JSON 里**没有** session_id、没有任何"这是哪一场"的字样
—— D2 的"写明是哪一场"此前只落在 HTML 报告里。
"""

import csv
from pathlib import Path

import pytest

from report_frontend.data_loader import LogDataLoader
import app as panel

_SID = "20260924_230914_262f"


def _write(log_dir: Path, filename: str, rows: list) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    fields = ["session_id", "timestamp", "connective_density", "connective_density_std", "n_rows"]
    p = log_dir / filename
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _row(sid: str, density=2.5) -> dict:
    return {"session_id": sid, "timestamp": "2026-09-24T23:09:14",
            "connective_density": density, "connective_density_std": 0.1, "n_rows": 1}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """把端点里 `LogDataLoader()` 的默认目录换到 tmp —— 端点用的是无参构造。"""
    def _make(files=()):
        log_dir = tmp_path / "logs"
        for name, rows in files:
            _write(log_dir, name, rows)
        # 端点是在函数体里 `from report_frontend.data_loader import LogDataLoader`,
        # 每次调用都重查模块属性 —— 所以补丁要打在**那个模块**上,而不是总控 `app` 上。
        import report_frontend.data_loader as dl
        monkeypatch.setattr(dl, "LogDataLoader", lambda *a, **k: LogDataLoader(str(log_dir)))
        return panel.app.test_client()

    return _make


def test_structured_report_names_the_session_it_describes(client):
    """★ 成功那一支必须带上 session_id 与三态来源。"""
    body = client([(f"face_au_log_{_SID}.csv", [_row(_SID)])]) \
        .get("/api/report/structured").get_json()

    assert body["status"] == "success", body
    assert body["session_id"] == _SID, f"JSON 里看不出这是哪一场:{body.keys()}"
    assert body["sources"]["face"]["status"] == "loaded", body["sources"]
    assert body["sources"]["gesture"]["status"] == "missing", body["sources"]


def test_structured_report_still_answers_when_the_target_has_no_logs(client):
    """★ 空数据不许变成 error —— spec §6 行 2:生成报告并显式说明本场没有任何日志。

    红法:退回 `if not data: return jsonify({"status":"error",...})`。
    """
    c = client([("face_au_log_NONE.csv", [_row("NONE", None)])])
    body = c.get(f"/api/report/structured?session_id={_SID}").get_json()

    assert body["status"] == "success", body
    assert body["session_id"] == _SID, "空数据时反而说不出描述的是哪一场"
    assert body["result"]["coverage"]["n_passed"] == 0, body["result"]["coverage"]
    assert all(v["status"] == "missing"
               for k, v in body["sources"].items() if k != "none_bucket"), body["sources"]


def test_structured_report_with_only_the_none_bucket_says_so(client):
    """只有 NONE 桶 → session_id 为 None,但 NONE 桶行数要在 sources 里报出来。"""
    c = client([("face_au_log_NONE.csv", [_row("NONE", None)])])
    body = c.get("/api/report/structured").get_json()

    assert body["status"] == "success", body
    assert body["session_id"] is None, body["session_id"]
    assert body["sources"]["none_bucket"]["rows"] == 1, body["sources"]
```

- [ ] **Step 2: 跑测试,确认红**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_structured_report_endpoint.py -v
```
Expected: 三条全红 —— 第 1 条 `KeyError: 'session_id'`;第 2、3 条 `status == "error"`。

- [ ] **Step 3: 改 `app.py` 的端点**

```python
@app.route('/api/report/structured')
def get_structured_report():
    """
    运行 report_generator 流水线，返回结构化评估 JSON。
    可选参数: session_id(要描述哪一场;缺省取最新一场)、type=interview|research

    M2.1(第 18 条):①**本场没有任何日志时也照常回答**(spec §6 行 1/2 要求"报告仍生成"),
    不再回 `{"status":"error","message":"未找到评估日志数据"}` —— 那种回答会让前端显示
    「暂无报告数据」,而盘上明明有别的会话;②回传 `session_id` 与 `sources`,让读的人
    **从 JSON 里就能确认这是哪一场、哪个模态没进来**(此前"写明是哪一场"只落在 HTML 报告里)。
    """
    try:
        from report_frontend.data_loader import LogDataLoader
        from report_frontend.feature_engine import PsychologicalFeatureEngine
        from report_frontend.research_mapper import ResearchCapabilityMapper

        session_id = request.args.get("session_id") or None
        loader = LogDataLoader()
        data = loader.get_fused_latest_data(session_id)

        # 空 data 也把链走完:实测 features={} → coverage 0/20、五维全 None、图表照常生成
        engine = PsychologicalFeatureEngine(data)
        features = engine.extract_all_features()

        mapper = ResearchCapabilityMapper()
        result = mapper.map_features_to_scores(features)

        return jsonify({
            "status": "success",
            "session_id": loader.target_session,
            "sources": loader.selected_sessions,
            "result": result,
        })

    except Exception as e:
        logger.exception("获取结构化评估报告失败")
        return jsonify({"status": "error", "message": str(e)})
```

- [ ] **Step 4: 跑测试,确认绿**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_structured_report_endpoint.py -v
```
Expected: 3 passed

- [ ] **Step 5: 改前端 `src/services/api.ts`**

5a. `withSession`(第 33-35 行)换成:

```ts
/** 给 URL 追一个 session_id 参数;**还没开始会话时不追**(而不是带上一个编出来的 id)。
 *  `override` 用于"要读的/要跑的不是当前这场"的调用(报告页可以指定另一场);
 *  缺省(传 `undefined`)就是当前这场。 */
const withSession = (url: string, override: string | null = sessionId): string =>
  override ? `${url}${url.includes('?') ? '&' : '?'}session_id=${override}` : url;
```

5b. `dashboardApi` 换成:

```ts
export const dashboardApi = {
  // M2.1(第 18 条):写路径也要带 id —— 不带的话后端取"最新一场",而那可能是一场刚
  // `/interview/start` 出来、还没有任何数据的会话(实测报告页因此显示「暂无报告数据」)。
  runModule: async (module: 'face' | 'gesture' | 'voice' | 'report', sessionId?: string | null) => {
    const response = await axios.post(withSession(`${API_BASE_URL}/api/run/${module}`, sessionId));
    return response.data;
  },

  getFiles: async (folderName: string) => {
    const response = await axios.get(`${API_BASE_URL}/api/files/${folderName}`);
    return response.data;
  },

  // M2.1:读路径带 id;`sessionId` 缺省 = 当前这场,都没传才让后端取最新一场
  getStructuredReport: async (sessionId?: string | null) => {
    const response = await axios.get(withSession(`${API_BASE_URL}/api/report/structured`, sessionId));
    return response.data;
  },
};
```

5c. `start`(第 152-189 行的 `research` 块)本任务不动 —— 它在 Task 3。

- [ ] **Step 6: 改前端 `src/pages/ReportPage.tsx`,让读者看得见"这是哪一场"**

在文件顶部(`const { Content } = Layout;` 之下)加标签表:

```tsx
// 后端 `/api/report/structured` 的 `sources` 是三态:loaded / unreadable / missing
const MODALITY_LABELS: Record<string, string> = {
  face: '面部', gesture: '手势', voice_interview: '语音（面试）', voice_research: '语音（科研）',
};
const STATUS_LABELS: Record<string, string> = {
  loaded: '已读入', unreadable: '未读到数据', missing: '缺失',
};
```

`useEffect` 里那段取报告换成:

```tsx
    setLoading(true);
    // M2.1(第 18 条):读报告要指明**哪一场** —— 路由给了就用路由的,否则用当前会话;
    // 都没有才让后端取最新一场(报告头会写明是它)。
    dashboardApi.getStructuredReport(id)
      .then(data => {
        if (data.status === 'success' && data.result) {
          setReport(data.result);
          setDescribedSession(data.session_id ?? null);
          setSources(data.sources ?? null);
        } else {
          setError(data.message || '无法获取报告数据');
        }
      })
```

对应的 state:

```tsx
  const [describedSession, setDescribedSession] = useState<string | null>(null);
  const [sources, setSources] = useState<Record<string, any> | null>(null);
```

在 `report` 非空的那一段、`<Card>` 与 `<ReportViewer>` 之间插入披露卡:

```tsx
        {/* M2.1(第 18 条):D2 的"写明是哪一场"此前只落在 HTML 报告里,前端这条路上没有 */}
        <Card size="small" style={{ marginBottom: '24px' }}>
          <div style={{ color: '#666' }}>
            <strong>本场会话：</strong>
            <code>{describedSession ?? '（无 —— 本场没有任何日志）'}</code>
          </div>
          {sources && (
            <ul style={{ margin: '8px 0 0', paddingLeft: '20px', color: '#666', fontSize: '13px' }}>
              {Object.entries(sources)
                .filter(([k]) => k !== 'none_bucket')
                .map(([k, v]: [string, any]) => (
                  <li key={k}>
                    {MODALITY_LABELS[k] ?? k} · {STATUS_LABELS[v.status] ?? v.status}
                    {v.status === 'loaded' ? `（${v.rows} 行）` : ''}
                  </li>
                ))}
              {sources.none_bucket && (
                <li>
                  <em>另有 NONE 桶 {sources.none_bucket.rows} 行 —— 那些请求没带 session_id，不属于本场</em>
                </li>
              )}
            </ul>
          )}
        </Card>
```

- [ ] **Step 7: 前端编译门**

```bash
cd ~/JingXin-frontend && npm run build
```
Expected: 构建成功(既有 `GazeHeatmap` chunk 体积警告可忽略,它是既有问题)

- [ ] **Step 8: 反向复现**

1. 把端点的 `"session_id": loader.target_session` 删掉 → `test_structured_report_names_the_session_it_describes` 红(`KeyError`)。
2. 把空数据那条早退分支加回来(`if not data: return jsonify({"status":"error",...})`)→ 第 2、3 条红。
3. 恢复,重跑确认绿。

- [ ] **Step 9: 提交**

```bash
cd ~/jingxin
git add app.py tests/test_structured_report_endpoint.py
git commit -m "fix(panel): 结构化报告自报场次,空数据不再回 error(复审 I3 / 第 18 条)"

cd ~/JingXin-frontend
# ⚠️ 只加本任务点名的两个文件 —— 工作树里还有一批使用者先前未提交的改动
git add src/services/api.ts src/pages/ReportPage.tsx
git commit -m "fix(report): 读路径带 session_id,并把「本场是哪一场」显示出来(第 18 条)"
```

---

### Task 3: 科研评估链路自己铸号(第 19 条 / 复审 I4)

**决策(使用者 2026-09-25 裁定,方案 A):** `/research/start` **铸新号**,与 `/interview/start` 对称。理由:①D1「`session_id` 契约以服务端铸号为唯一来源」;②科研回答原句不再写进面试会话目录,消灭"张冠李戴";③顺带解决 M1 账本里的「科研回答与面试回答混进同一个 `transcript.json` 且无法区分」—— 科研从此有自己的目录。

**Files:**
- Modify: `voice_interaction/api/app.py:460-472`(`/research/start`)
- Modify: `~/JingXin-frontend/src/services/api.ts:152-156`(`research.start()`)
- Test: `tests/test_research_session_minting.py`(新建)
- 新建(仓库外): `~/shared/m21_acceptance.sh`(端到端验收,承 `t7_acceptance.sh` / `m2_acceptance.sh` 的做法)

**Interfaces:**
- Consumes: `voice_interaction.asr.session.new_session_id()`、`voice_interaction.asr.transcript_store.ensure_manifest(sid, asr_meta)`
- Produces: `POST /research/start` → `{"status": "started", "session_id": "<YYYYMMDD_HHMMSS_xxxx>", "question": "..."}`;前端 `voiceApi.research.start()` 把它存进模块级 `sessionId`

**⚠️ 与 Ruling M1-14 的关系(请在审阅时确认):** M1 Task 4 的裁定是「测试里**不许** import `voice_interaction.api.app`」,理由是"一 import 就构造 TTS 引擎与两条评估管线(慢,而且会因为与本次改动无关的原因失败)"。**本计划复核了这条前提,它在本环境已不成立**:实测 `import` 耗时 **0.4 s** 且成功(TTS 后台线程正常启动)。因此本任务用 FastAPI `TestClient` 给 HTTP 层补上自动化覆盖 —— 复审报告反复点名的「全仓没有任何测试碰这条路径」正是这个盲区。**如你判断仍应遵守 M1-14,本任务的 Step 1/4 改为只保留验收脚本(Step 6),其余不变。**

- [ ] **Step 1: 写失败测试**

新建 `tests/test_research_session_minting.py`:

```python
# tests/test_research_session_minting.py
"""第 19 条(复审 I4):科研评估**自成一场**,不许沿用上一场面试的号。

失效现场(2026-09-24 复审,按代码路径判定;2026-09-25 本机**当场复现**了其中一条):
    A 新页面直接做科研 → 前端没有号 → `withSession` 不加参数 → 落 `NONE`;
    B 先面试、再科研 → 前端还留着**上一场面试的铸号** → 科研回答
      (`/research/answer_audio` → `transcript_store.append_utterance(sid, utt)`)
      被写进**面试会话**的目录 `~/shared/jingxin_recordings/<面试号>/transcript.json`。

    实测(2026-09-25 12:35):`POST /research/start` 的返回体是
    `{'status': 'started', 'question': '请描述一个你深入研究过的技术…'}` —— **没有 session_id**。

修法(使用者 2026-09-25 裁定,方案 A):`/research/start` 与 `/interview/start` 对称铸号。
"""

import importlib
import re

import pytest
from fastapi.testclient import TestClient

# ⚠️ 两个名字不要搞混(2026-09-25 实测):
#   * `from voice_interaction.api import app …` → 拿到的是 **FastAPI 实例**
#     (包 `__init__.py` 里 `from .app import app` 把子模块这个属性**遮蔽**了);
#   * `importlib.import_module("voice_interaction.api.app")` → 拿到**模块**,
#     端点里的 `tts_engine` 等全局在它上面。
from voice_interaction.api import app as voice_api

_voice_app_module = importlib.import_module("voice_interaction.api.app")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """把仓库外的录制根目录挪到 tmp,并把 TTS 换掉(本任务不测发声)。

    `JINGXIN_RECORDINGS_DIR` 是 `transcript_store._root()` 认的环境变量。
    """
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "rec"))
    monkeypatch.setattr(_voice_app_module.tts_engine, "speak", lambda *a, **k: None)
    # logger 的文件名带 session_id,不隔离的话会在仓库 data/logs 里留下测试产物
    import voice_interaction.utils.logger as voice_logger_module
    monkeypatch.setattr(voice_logger_module, "LOGS_DIR", str(tmp_path / "logs"))
    return TestClient(voice_api)


def test_research_start_mints_a_session_id(client):
    """★ 科研入口必须铸号(与 /interview/start 对称)。

    红法:退回 `return {"status": "started", "question": first_question}`
    → `session_id` 缺失 → 红。
    """
    body = client.post("/research/start").json()

    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{4}", body.get("session_id", "")), (
        f"科研评估没有铸号 —— 返回体是 {body}")


def test_research_start_writes_the_session_manifest(client, tmp_path):
    """铸了号就要建会话清单 —— 否则 `append_utterance` 会在另一处另建一个残缺目录。

    红法:只 `return` 铸号、不 `ensure_manifest` → 清单不存在 → 红。
    """
    sid = client.post("/research/start").json()["session_id"]

    assert (tmp_path / "rec" / sid / "session.json").exists(), "铸了号却没建会话清单"


def test_research_and_interview_get_different_session_ids(client):
    """★ 同一页面连做两场 → 两个号必须不同。

    相同就是"科研写进面试目录"那条老路(复审 I4 失效场景 B)。
    """
    interview_sid = client.post("/interview/start").json()["session_id"]
    research_sid = client.post("/research/start").json()["session_id"]

    assert interview_sid != research_sid, (
        f"科研与面试拿到了同一个号 {research_sid} —— 科研回答会被写进面试会话目录")
```

- [ ] **Step 2: 跑测试,确认红**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_research_session_minting.py -v
```
Expected: 第 1 条 `assert re.fullmatch(...)` 失败(返回体是 `{'status': 'started', 'question': …}`,没有 `session_id` —— 已实测);第 2、3 条 `KeyError: 'session_id'`。

- [ ] **Step 3: 改 `voice_interaction/api/app.py` 的 `/research/start`**

```python
@app.post("/research/start")
async def start_research_assessment():
    try:
        research_assessment.reset()
        first_question = research_assessment.get_next_question()
        if not first_question:
            raise HTTPException(status_code=500, detail="无法获取问题")

        # 科研评估**自成一场**(M2.1 / 第 19 条):与 /interview/start 对称铸号。
        # 不铸的话科研回答只能带客户端自己的 id —— 新页面直接做科研会落 NONE,
        # 而"先面试、再科研"会顺延上一场的号,把科研回答的原句写进**面试会话**的录制目录。
        sid = session_mod.new_session_id()
        transcript_store.ensure_manifest(sid, _asr_meta())

        tts_engine.speak(first_question)
        return {"status": "started", "session_id": sid, "question": first_question}
    except HTTPException:
        raise          # 别再包一层:否则 500 会变成"启动科研评估失败: 500: 无法获取问题"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动科研评估失败: {str(e)}")
```

- [ ] **Step 4: 跑测试,确认绿**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_research_session_minting.py -v
```
Expected: 3 passed

- [ ] **Step 5: 改前端 `src/services/api.ts` 的 `research.start()`**

```ts
    start: async () => {
      const response = await axios.post(`${VOICE_API_URL}/research/start`);
      // M2.1(第 19 条):科研评估**自成一场** —— 服务端也铸号了,这里必须存下来。
      // 不存的话会走两条坏路:新页面直接做科研 → 落 NONE 桶;先面试再科研 →
      // 顺延上一场面试的号,科研回答的原句被写进面试会话的录制目录。
      setSessionId(response.data?.session_id ?? null);
      return response.data;
    },
```

- [ ] **Step 6: 端到端验收(真服务 + 真 ASR;这是 HTTP 层唯一的强证据)**

新建 `~/shared/m21_acceptance.sh`:

```bash
#!/usr/bin/env bash
# M2.1 验收(第 19 条):科研评估自成一场,不再把科研回答写进面试会话目录。
#
# 只依赖语音服务(:8001)与局域网 FunASR —— 铸号、转写、录制目录三件事都在它手里。
# 语音样本用 ~/asr-test/zijijieshao.wav(真录音,16 kHz / 单声道 / 16 bit)截 12 秒。
set -u
V=http://127.0.0.1:8001
REC=${JINGXIN_RECORDINGS_DIR:-$HOME/shared/jingxin_recordings}
PY=~/miniconda3/envs/jingxin/bin/python
bad() { echo "❌ $1"; FAIL=1; }
ok()  { echo "✅ $1"; }
FAIL=0

sid_of() { $PY -c 'import sys,json;print(json.load(sys.stdin).get("session_id",""))'; }

ISID=$(curl -s -X POST "$V/interview/start" | sid_of)
[ -n "$ISID" ] || { echo "语音服务没起(:8001)?先跑 ~/shared/start_all.sh"; exit 1; }
RSID=$(curl -s -X POST "$V/research/start" | sid_of)
echo "面试会话 = $ISID"
echo "科研会话 = $RSID"
[ -n "$RSID" ] || bad "科研入口没铸号(第 19 条未修)"
[ "$ISID" != "$RSID" ] || bad "科研与面试拿到同一个号"

ffmpeg -y -loglevel error -ss 5 -t 12 -i ~/asr-test/zijijieshao.wav \
       -ar 16000 -ac 1 -c:a pcm_s16le /tmp/m21_ans.wav || bad "取音频样本失败"

echo "--- ① 用**科研自己的号**提交语音回答(修好之后前端的行为)---"
curl -s -X POST "$V/research/answer_audio?session_id=$RSID" \
     -F "audio=@/tmp/m21_ans.wav" | head -c 200; echo
[ -f "$REC/$RSID/transcript.json" ] && ok "科研回答落在科研号的目录" || bad "科研回答没落到 $RSID"
[ -f "$REC/$ISID/transcript.json" ] && bad "面试目录被写入了 —— 张冠李戴仍在" || ok "面试目录没被写入"

echo "--- ② 反向复现:把**上一场面试的号**喂给科研回答(修之前前端的行为)---"
curl -s -X POST "$V/research/answer_audio?session_id=$ISID" \
     -F "audio=@/tmp/m21_ans.wav" | head -c 200; echo
if [ -f "$REC/$ISID/transcript.json" ]; then
  ok "复现成功:带旧号时科研回答确实会写进面试目录(这就是修掉的那条路)"
else
  bad "复现不出来 —— 说明这条测试没压住那个失效模式,别当成已验"
fi

[ "$FAIL" = 0 ] && echo "M2.1 验收通过" || echo "M2.1 验收失败"
exit $FAIL
```

跑法:

```bash
~/miniconda3/envs/jingxin/bin/python -m voice_interaction.api.app &   # :8001(或 ~/shared/start_all.sh 起全套)
bash ~/shared/m21_acceptance.sh
```
Expected: ① 科研回答落在 `$REC/$RSID/`,`$REC/$ISID/` 不存在;② 反向复现成功(带旧号时确实污染面试目录)。**把两次输出原样记进账本**。

- [ ] **Step 7: 前端编译门 + 提交**

```bash
cd ~/JingXin-frontend && npm run build
git add src/services/api.ts
git commit -m "fix(research): 科研评估自成一场,存下服务端铸的号(第 19 条)"

cd ~/jingxin
git add voice_interaction/api/app.py tests/test_research_session_minting.py
git commit -m "fix(voice): /research/start 铸号并建会话清单(复审 I4 / 第 19 条)"
```

---

### Task 4: 收尾 —— 账本、文档口径、全量门

**Files:**
- Create: `docs/superpowers/sdd/2026-09-25-m2-review-fixes/progress.md`(账本;**建在 git 跟踪的目录里** —— 上一轮复审报告落在 gitignore 的目录里差点丢)
- Modify: `docs/下一步.md`(§0 现状、§3 第 17/18/19 条标完成、§5 索引加本计划与账本)
- Modify: `~/shared/m21_acceptance.sh` 的运行结果记入账本

- [ ] **Step 1: 账本**

`progress.md` 至少记:每个任务的**裁决**(做/不做/偏离)、**反向复现的原始输出**、验收脚本两次运行的原始输出、`deferred minor` 段(本轮又发现什么没修的)。

- [ ] **Step 2: 文档口径(本项目的第 6 条教训:文档要一起改,但账本里的历史记录不改)**

`docs/下一步.md`:
- §0 一句话现状 → 改成"M2 复审的 17/18/19 已清,下一步 = M3"
- §3 表格第 17/18/19 条的「性质」列改成「✅ 已完成(M2.1)」,并各附一句落地位置
- §5 关键文件索引加两行:本计划、本账本
- §3 第 20 条(RealtimeAnalysis 裁决)与第 21–24 条保持原样(本轮未动)

- [ ] **Step 3: 全量门**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest -q          # 期望 204 + 12 = 216 项全过
cd ~/jingxin/experiments/duration_audit
PY=~/miniconda3/envs/jingxin/bin/python
$PY reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_probe
$PY reaggregate_normalized.py --verify-legacy /tmp/legacy_probe             # 期望 0 / 2835510
```

- [ ] **Step 4: 提交**

```bash
cd ~/jingxin
git add docs/superpowers/sdd/2026-09-25-m2-review-fixes/progress.md docs/下一步.md
git commit -m "docs(m2.1): 17/18/19 的账本与下一步口径"
```

---

## 任务之间的依赖

```
Task 1(报告层)───→ Task 2(端点用 loader.target_session)
Task 3(科研铸号)── 独立,可并行
Task 4 ────────── 必须最后(要跑全量门 + 写账本)
```

Task 1 与 Task 3 之间**没有共享文件**,可以并行;Task 2 依赖 Task 1 新加的 `target_session` 属性。
