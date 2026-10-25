"""HTML 报告与前端报告**必须出自同一个引擎** —— 否则同一场会话会给出两个数。

## 为什么

2026-10-18 起,`/api/report/structured`(前端报告页读的那个端点)默认走 **L1**
(`l1_report.build_l1_report`),旧引擎 `research_mapper` 只留在 `?engine=legacy` 下
作对照。但 **HTML 报告那条路没跟着换** —— `report_generator.generate_report()`
仍然跑 `PsychologicalFeatureEngine` → `ResearchCapabilityMapper`。

2026-10-25 在服务器上用**同一场真素材**(`20260927_212143_95a8`)实测:

    HTML 报告(旧引擎)        覆盖 11 / 14
    /api/report/structured    覆盖  7 / 14

而这两份东西**在界面上是并排的**:前端「历史报告」列表点开的是 HTML,
同一场会话的「报告页」读的是 structured 端点。两个数都写着「本次观测覆盖」,
谁也看不出它们量的是两套口径 —— 差的那 4 个槽正是旧引擎把「手掌张开度 /
肩部放松度 / 眼神接触」算成"填上了",而 L1 判定**当前管线根本不测**的那几个
(见 `docs/下一步.md` 第 29 条)。

⟹ 这不是"两个引擎并存待对照",是**同一个界面里两个互相矛盾的数**。
对照该在数据层做(`?engine=legacy`),不该让**交付物**分叉。

## 这个文件守什么

1. `generate_report()` 走的是 L1 —— 判据落在**它真的调用了** `build_l1_report`
   (本仓记过六次的「验证跑错了对象」:只断言"载荷形状对"证明不了生产路径用了它)。
2. 它**不再**构造旧引擎(`PsychologicalFeatureEngine` / `ResearchCapabilityMapper`)。
3. 产出的 HTML 里覆盖数与 `build_l1_report` 给的**逐字一致**。
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

import report_frontend.report_generator as report_generator
from report_frontend.data_loader import LogDataLoader
from report_frontend.l1_report import build_l1_report
from report_frontend.report_generator import ReportGenerator

_SID = "20260927_212143_95a8"

#: 一份"这场真的测到东西"的最小载荷 —— 直接喂给 `build_l1_report`。
#: 形状取自真实日志的列名(见 `report_frontend/l1_session.py` 的读法)。
_FACE = pd.DataFrame([{
    "session_id": _SID, "timestamp": "2026-09-27T21:21:43",
    "au12_smile": 0.31, "au4_frown": 0.12, "au1_inner_brow": 0.08,
    "gaze_direction_x": 0.02, "gaze_direction_x_std": 0.09,
    "blink_rate_per_min": 17.0, "frame_count": 1,
}])

_VOICE = pd.DataFrame([
    {"session_id": _SID, "timestamp": f"2026-09-27T21:2{i}:00",
     "connective_density": 0.4 * (i + 1), "connective_density_std": 0.0,
     "answer_length": 40 + i, "n_rows": 1}
    for i in range(4)
])


def _write_logs(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    _FACE.to_csv(log_dir / f"face_au_log_{_SID}.csv", index=False)
    _VOICE.to_csv(log_dir / f"interview_emotion_log_{_SID}.csv", index=False)


def _cover(html: str) -> str:
    m = re.search(r'cover-number">([^<]+)<', html)
    assert m, "报告里没有覆盖数 —— 渲染层被改形了?"
    return m.group(1).strip()


@pytest.fixture
def rendered(tmp_path, monkeypatch):
    """真加载器 + 真渲染,只换日志目录与"打开浏览器"。"""
    log_dir = tmp_path / "logs"
    _write_logs(log_dir)
    monkeypatch.setattr(report_generator, "LogDataLoader",
                        lambda *a, **k: LogDataLoader(str(log_dir)))
    monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)
    p = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(_SID)
    assert p, "报告没生成 —— 本测试的前提不成立"
    return Path(p).read_text(encoding="utf-8"), LogDataLoader(str(log_dir))


def test_html_report_renders_the_coverage_that_l1_computed(rendered, monkeypatch):
    """★ 红法:把 `generate_report` 换回 `ResearchCapabilityMapper` ⟹ 这里红。

    ⚠️ 第一版是"HTML 里的数 == `build_l1_report` 的返回值"。**那条是空断言**:
    这份最小 fixture 上两个引擎都给 **0 / 14**(实测),换不换引擎都过。
    改成**替换 L1 的返回值**成一个可识别的数 —— 渲染层若还在读旧引擎,这个数
    就出不来。
    """
    html, loader = rendered
    real = report_generator.build_l1_report

    def fake(session_id, logs=None):
        payload = real(session_id, logs=logs)
        # 9 / 14:与真实载荷一定不同(这份 fixture 下真值是 0/14),
        # 但形状合法,渲染层不需要为它开特例。
        payload["coverage"] = {**payload["coverage"], "n_passed": 9, "n_slots": 14}
        return payload

    monkeypatch.setattr(report_generator, "build_l1_report", fake)
    log_dir = Path(loader.log_dir)
    report_generator.LogDataLoader = lambda *a, **k: LogDataLoader(str(log_dir))
    out = Path(log_dir).parent / "out2"
    p = ReportGenerator(output_dir=str(out)).generate_report(_SID)
    assert p, "报告没生成"
    assert _cover(Path(p).read_text(encoding="utf-8")) == "9 / 14", (
        "HTML 里的覆盖数不是 L1 算出来的那个 —— 渲染层还在读旧引擎")


def test_generate_report_really_calls_build_l1_report(rendered, monkeypatch):
    """★ 红法:让 `generate_report` 自己拼一个同形状的载荷 ⟹ 这里红。

    只断言"HTML 里的数与 L1 相符"是不够的:渲染层可以拿别处算出来的数。
    这条盯的是**生产路径真的走了那个函数**。
    """
    calls = []
    real = report_generator.build_l1_report

    def spy(session_id, logs=None):
        calls.append((session_id, logs is not None))
        return real(session_id, logs=logs)

    monkeypatch.setattr(report_generator, "build_l1_report", spy)
    log_dir = Path(rendered[1].log_dir)
    report_generator.LogDataLoader = lambda *a, **k: LogDataLoader(str(log_dir))
    assert ReportGenerator(output_dir=str(log_dir.parent / "out2")).generate_report(_SID)
    assert calls == [(_SID, True)], f"没按预期调用 build_l1_report:{calls}"


def test_html_report_does_not_build_the_legacy_engine(rendered):
    """★ 红法:在 `generate_report` 里加回 `PsychologicalFeatureEngine(data)` ⟹ 这里红。

    旧引擎只在 `?engine=legacy` 下对照用,不该再参与**交付物**的生成。

    ⚠️ 只看**会执行的行**:函数里那段注释正好在解释"为什么换掉旧引擎",
    注释里点它的名字是应该的(而且必须),不能因此判红。
    """
    import inspect
    src = inspect.getsource(ReportGenerator.generate_report)
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    for name in ("PsychologicalFeatureEngine", "ResearchCapabilityMapper"):
        assert name not in code, (
            f"`generate_report` 仍在构造旧引擎 `{name}` —— "
            "HTML 报告会与前端报告页给出不同的覆盖数")
