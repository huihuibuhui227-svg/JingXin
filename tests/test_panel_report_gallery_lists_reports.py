# tests/test_panel_report_gallery_lists_reports.py
r"""总控台「生成行为观测报告」那张卡片里,`iframe` 里装的**不是报告**。

失效现场(2026-10-26,为给使用者演示报告产出而逐个打开出口时发现):

`data/output/` 根目录下混着三类产物 —— `Research_Assessment_Report_*.html`(14 份)、
`radar_chart_*.html`(15 份,单份 4.8 MB 的 plotly 图)、
`evidence_*.html`(55 份,旧引擎的逐维证据图,产出方已随评分层摘除)。

`/api/files/reports`(面板对根目录的别名)**原样返回全部 84 个**,而
`templates/dashboard.html` 的 `loadFiles('report')` 取 `files.slice(0, 6)`。
按名字倒序排,`radar_chart_*` 的小写 `r` 排在大写 `R` 前面 ⟹ 那 6 个 iframe
里**一个报告都没有**,全是雷达图。

同一个事实(「哪些文件是报告」)此前被三处消费,而三处各写各的:
  · `Home.tsx` / `ReportsList.tsx` —— 各自在前端 `filter(/Assessment_Report.*\.html$/)`;
  · `templates/dashboard.html` —— **没有过滤**。
⟹ 第三处漏了,而它恰好是唯一直接给人看的那一处。

修法:把「什么是报告文件」收到后端一处(`app.REPORT_FILENAME_RE`),
`/api/files/reports` 只回报告。前端那两处过滤留着无妨(幂等),但事实只有一份。

红法:把 `get_files` 里那段 `if folder_name == REPORT_FOLDER` 的过滤去掉。
"""

import re
from pathlib import Path

import app as panel


def _write(path: Path, text: str = "<html></html>") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "OUTPUT_DIR", str(tmp_path))
    panel.app.config.update(TESTING=True)
    return panel.app.test_client()


def test_report_folder_lists_only_report_files(tmp_path, monkeypatch):
    """根目录混着雷达图与证据图,报告别名只能回报告 —— 否则卡片里装的是雷达图。"""
    _write(tmp_path / "Research_Assessment_Report_20261025_235725.html")
    _write(tmp_path / "radar_chart_1792943846.html")
    _write(tmp_path / "evidence_logical_thinking_1790609120.html")

    files = _client(tmp_path, monkeypatch).get("/api/files/reports").get_json()

    names = [f["name"] for f in files]
    assert names == ["Research_Assessment_Report_20261025_235725.html"], (
        f"报告别名回了非报告文件:{names}")


def test_the_six_the_panel_previews_are_all_reports(tmp_path, monkeypatch):
    """卡片取前 6 个 —— 那就必须是 6 份报告。这一条直接照抄 `slice(0, 6)`。"""
    for i in range(8):
        _write(tmp_path / f"Research_Assessment_Report_2026102{i}_120000.html")
    for i in range(10):
        _write(tmp_path / f"radar_chart_17929438{i:02d}.html")

    files = _client(tmp_path, monkeypatch).get("/api/files/reports").get_json()

    top6 = [f["name"] for f in files[:6]]
    assert len(top6) == 6, f"报告不足 6 份,面板那张卡片会空一半:{top6}"
    assert all("Assessment_Report" in n for n in top6), (
        f"面板卡片里装进了非报告文件:{top6}")


def test_other_folders_are_not_filtered(tmp_path, monkeypatch):
    """另一侧:别的模态目录照旧回全部(它们里面没有「报告」这个概念)。"""
    _write(tmp_path / "face_expression" / "emotion_plot_1790609120.html")
    _write(tmp_path / "face_expression" / "anything.png")

    files = _client(tmp_path, monkeypatch).get("/api/files/face_expression").get_json()

    assert {f["name"] for f in files} == {
        "emotion_plot_1790609120.html", "anything.png"}, files


def test_the_report_name_pattern_is_defined_once():
    """「什么是报告文件」只能有一处定义 —— 三处各写一份,就会有一处写漏(本次就是)。"""
    assert isinstance(panel.REPORT_FILENAME_RE, re.Pattern), (
        "`app.REPORT_FILENAME_RE` 不存在 —— 报告文件名判据又散回各处了")
    assert panel.REPORT_FILENAME_RE.search("Research_Assessment_Report_20261025_235725.html")
    assert not panel.REPORT_FILENAME_RE.search("radar_chart_1792943846.html")


# ── 面板那张卡片本身 ────────────────────────────────────────────────────────
#
# ⚠️ 上面四条测的是**接口回什么**。而卡片**根本不请求**接口,是另一件事:
#
#     const folderMap = { ..., 'report': '' };
#     const folder = folderMap[module];
#     if (!folder) return;          // ← 空字符串是 falsy,直接返回
#
# ⟹ 「生成行为观测报告」那张卡片的 gallery 永远停在 `<p>加载中...</p>`,
#    一次 `fetch` 都没发过。这是 2026-10-26 打开面板核对时发现的 ——
#    此前以为它只是"装错了文件",实际是**什么都没装**。
#
# 为什么测试要读模板源码:这是内联 `<script>`,没有模块可 import。
# 断言锚在**具体表达式**上,不查关键词 —— 本仓踩过子串匹配让改名也过的坑。

_DASHBOARD = Path(panel.__file__).resolve().parent / "templates" / "dashboard.html"


def _dashboard_js() -> str:
    """只取 `<script>` 里的 JS —— 注释里会提到这些标识符,不能当成代码。"""
    text = _DASHBOARD.read_text(encoding="utf-8")
    blocks = re.findall(r"<script>([\s\S]*?)</script>", text)
    assert blocks, "dashboard.html 里找不到 <script> —— 本测试的取样方式已失效"
    js = "\n".join(blocks)
    # 去行注释与块注释:注释里写着 `'report': ''` 会被当成渲染。
    js = re.sub(r"/\*[\s\S]*?\*/", "", js)
    return "\n".join(re.sub(r"//.*$", "", line) for line in js.splitlines())


def test_report_card_has_a_real_folder_to_request():
    """`'report'` 必须映射到一个**真文件夹名**,不能是空字符串。

    空字符串让 `if (!folder) return` 提前返回,那张卡片一次请求都不发。
    """
    js = _dashboard_js()
    m = re.search(r"'report'\s*:\s*'([^']*)'", js)
    assert m, "dashboard.html 的 folderMap 里找不到 'report' 这一项"
    assert m.group(1), (
        "`'report'` 映射到空字符串 —— `if (!folder) return` 会直接返回,"
        "那张卡片永远停在「加载中...」,一次请求都不发")
    assert m.group(1) == panel.REPORT_FOLDER, (
        f"`'report'` 映射到 {m.group(1)!r},而后端认的别名是 {panel.REPORT_FOLDER!r}")


def test_report_card_would_not_be_swallowed_by_the_falsy_guard():
    """另一侧:守卫本身别写成把空字符串当成合法值 —— 那会让上面的断言变成唯一防线。"""
    js = _dashboard_js()
    assert re.search(r"if\s*\(\s*!\s*folder\s*\)\s*return", js), (
        "`if (!folder) return` 这条守卫不见了 —— 本测试的前提变了,先重新核对")
    # 取值必须经过 folderMap,不能把 module 名直接当文件夹名用
    # (面板的 `report` ≠ 后端的 `reports`,两者不同名)。
    assert re.search(r"folderMap\s*\[\s*module\s*\]", js), (
        "`folderMap[module]` 不见了 —— 若改成直接用 module 名,`report` 会 403")
