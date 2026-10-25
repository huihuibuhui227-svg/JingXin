"""交付物里的数字与理由要**给人读**,而审计细节要留在载荷里。

## 两个毛病(2026-10-25 实测,同一份 HTML 报告)

### ① 原始值印满 17 位

    | 指标         | 原始值               | 有效样本量 |
    | 连接词密度   | 3.4175334323922733   | 673        |

那是一个 `float64` 的全部有效位。读者要的是"每百字大约 3.4 个标记",
不是双精度浮点的内存映像。前端 `L1IndicatorTable` 早有一条按**量级**选小数位的
规则(`fmtValue`):≥1000 取整、≥1 三位、≥0.001 四位、更小用科学计数 ——
HTML 报告是同一个数的另一个出口,不该有两套精度。

⚠️ **只改渲染,不改载荷**。`raw_value` 在 `/api/report/structured` 里必须是
原始 float(前端要拿它排序、比较);把载荷四舍五入就是**丢测量结果**。

### ② 删槽理由印的是开发者的审计串

    语音流畅度:本层不测这一条 —— 零额外信息:VAD 流畅度的三个分量里,发声占比
    =voiced_ratio、平均连续发声长度恒等于 60/pause_rate_per_min、发音速率
    =speech_rate;不流畅事件在转写文本里(self_repair_rate 已在数)⟹ 写它只会造出
    第二个流畅度量

那段话是**给核对口径的人**看的(它点了四个内部指标名),而且它必须保持完整 ——
`tests/test_m4_l1_mapper.py::test_dropped_slot_reasons_say_construct_mismatch_not_a_coverage_excuse`
要求理由里点名真实 L1 指标。⟹ 不能删,只能**分层**:

* `SLOT_MAP` 的 `reason` = 审计细节,留在载荷的 `l1.indicators` / 缺口元数据里;
* 报告里印的是**一句话的人话结论**(`DROP_REASON`),不含内部指标名。

## 这个文件守什么

1. 报告里不出现 8 位以上的小数。
2. 报告里不出现内部 L1 指标名(那是审计链的东西)。
3. 载荷里的 `raw_value` **仍是原始 float** —— 精度只在渲染处收,不在数据处收。
"""
from __future__ import annotations

import re

import pytest

import report_frontend.report_generator as report_generator
from report_frontend import l1_face, l1_gesture, l1_mapper, l1_text, l1_voice
from report_frontend.data_loader import LogDataLoader
from report_frontend.l1_report import build_l1_report
from report_frontend.report_generator import ReportGenerator

_SID = "20260924_220729_07d7"


def _all_l1_names() -> set[str]:
    """全部 L1 指标名 —— 这些是**审计链**的词汇,报告正文里不该出现。"""
    return (set(l1_face.indicator_names()) | set(l1_gesture.ORDER)
            | set(l1_voice.ORDER) | set(l1_text.ORDER))


def _text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


@pytest.fixture
def recordings(tmp_path, monkeypatch):
    """一份**真有转写**的录制目录。

    `discourse_marker_rate` 走 T 族,只依赖转写文本,不吃三个模态的日志 ⟹
    这是让「连接词密度」槽**有值**的最省路径。没有它,报告里就没有表格,
    精度那条会退化成空转。

    ⚠️ `tests/conftest.py` 已把 `JINGXIN_RECORDINGS_DIR` 兜到临时目录,
    所以凡是要读到这份转写的测试都**必须**用这个 fixture —— 直接调
    `build_l1_report()` 会读到 conftest 的空目录,断言就空了。
    """
    root = tmp_path / "recordings"
    rec = root / _SID
    rec.mkdir(parents=True)
    (rec / "transcript.json").write_text(
        '{"session_id": "%s", "merged": {"text": "呃，我叫王明，然后然后我读计算机。",'
        ' "n_chars": 16, "n_segments": 1, "vad_split": false}}' % _SID,
        encoding="utf-8")
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(root))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))
    return root


@pytest.fixture
def report_html(tmp_path, recordings, monkeypatch):
    """一份**真有值**的报告。"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(report_generator, "LogDataLoader",
                        lambda *a, **k: LogDataLoader(str(log_dir)))
    monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)
    p = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(_SID)
    assert p, "报告没生成 —— 本测试的前提不成立"
    html = open(p, encoding="utf-8").read()
    assert "连接词密度" in html, "没有过门指标 —— 精度那条会退化成空转"
    return html


def test_raw_values_are_rounded_for_the_reader(report_html):
    """★ 红法:把 `_fmt` 去掉、直接印 `{e['raw_value']}` ⟹ 这里红。

    实测产出过 `3.4175334323922733` —— 那是 float64 的内存映像,不是读数。
    """
    long = re.findall(r"\d+\.\d{8,}", _text_of(report_html))
    assert not long, f"报告里印着 8 位以上的小数:{sorted(set(long))[:5]}"


def test_the_payload_keeps_full_precision(recordings):
    """★ 红法:在 `_evidence_item` 里 `round(slot["value"], 3)` ⟹ 这里红。

    精度只能在**渲染**处收。载荷里的 `raw_value` 是测量结果本身,
    前端要拿它排序/比较;在这里四舍五入就是丢数据。
    """
    rec = build_l1_report(_SID)
    vals = [e["raw_value"] for d in rec["dimensions"].values()
            for e in d["evidence_chain"]]
    assert vals, "载荷里一条过门指标都没有 —— 本测试退化为空断言"
    # 至少有一个不是"整齐"的短小数 —— 那才证明没有被渲染层的规则污染。
    assert any(len(repr(float(v)).split(".")[-1]) > 3 for v in vals), (
        f"载荷里的值都被抹平了:{vals}")


def test_dropped_slot_reasons_in_the_report_are_plain_chinese(report_html):
    """★ 红法:把报告里的删槽理由换回 `SLOT_MAP` 的 `reason` ⟹ 这里红。

    那段 reason 点了 `voiced_ratio` / `self_repair_rate` 等内部指标名,
    是核对口径用的审计细节。读者要的是「这一条本层不测,别再等它」。
    """
    body = _text_of(report_html)
    leaked = sorted(n for n in _all_l1_names() if n in body)
    assert not leaked, (
        "报告正文里露出了内部指标名(它们属于审计链,不属于交付物):\n  "
        + "\n  ".join(leaked))


def test_every_dropped_slot_has_a_plain_reason():
    """★ 红法:从 `DROP_REASON` 里删一条 ⟹ 这里红。

    地板:当前有 6 个槽在 L1 路径上被删(`SLOT_MAP` 里 `l1 is None`)。
    没有这句,槽表被清空时下面的循环一次都不进,本测试静默全绿。
    """
    dropped = [s for s, (l1, _) in l1_mapper.SLOT_MAP.items() if l1 is None]
    assert len(dropped) == 6, f"删槽数变了:{dropped}"
    missing = sorted(set(dropped) - set(l1_mapper.DROP_REASON))
    assert not missing, f"这些槽没有给人看的删槽理由:{missing}"


def test_plain_drop_reasons_do_not_name_internal_indicators():
    """★ 红法:把 `DROP_REASON` 的值换成 `SLOT_MAP` 的 reason ⟹ 这里红。

    这是上一条的**内容**判据:表格里有条目 ≠ 条目能给人读。
    """
    known = _all_l1_names()
    bad = {s: r for s, r in l1_mapper.DROP_REASON.items()
           if any(n in r for n in known) or "**" in r or "`" in r}
    assert not bad, "删槽理由里还带着内部指标名或注释体记号:\n  " + "\n  ".join(
        f"{s}: {r}" for s, r in bad.items())


def test_the_audit_reason_is_not_lost():
    """★ 红法:把 `SLOT_MAP` 的 reason 也改成短句 ⟹ 这里红。

    分层的意思是**两处各有各的话**,不是把审计细节删掉。核对口径的人
    仍然要能查到"当初为什么判定它不是同一个构念"。
    """
    long_reasons = {s: r for s, (l1, r) in l1_mapper.SLOT_MAP.items()
                    if l1 is None and len(r) > 20}
    assert len(long_reasons) >= 5, (
        f"`SLOT_MAP` 的删槽理由被削短了,审计细节丢了:{long_reasons}")


# ---------------------------------------------------------------------------
# ③ 比较陷阱那段:2026-10-25 晚才发现**同一类泄漏**还在另一处
# ---------------------------------------------------------------------------
# `l1.comparability_notes` 的每一条都是 `旧槽键 → L1指标名：理由`。
# 前端 `L1IndicatorTable` 与 HTML 报告**两边都原样印了**,于是交付物里出现
# `connective_density → discourse_marker_rate` 这样的串 —— 6 个内部指标名。
#
# 这与 ② 是同一个毛病:审计标识符(槽键、L1 指标名)属于核对口径的人,
# 读者要的只是「旧的那份数的是 A,新的数的是 B,两个数不能相减」。


def test_comparability_notes_say_it_in_plain_chinese():
    """★ 红法:删掉 `comparability_notes_plain()` 或让它退回审计版 ⟹ 这里红。

    地板:当前有 6 条比较陷阱(见 `NOT_COMPARABLE`)。地板缺了,下面循环
    一次都不进,本测试静默全绿。
    """
    notes = l1_mapper.comparability_notes_plain()
    assert len(notes) == len(l1_mapper.NOT_COMPARABLE), (
        f"给人读的比较陷阱条数不对:{notes}")
    known = _all_l1_names()
    bad = [n for n in notes if any(k in n for k in known)]
    assert not bad, "给人读的比较陷阱里还带着内部指标名:\n  " + "\n  ".join(bad)
    bad_marks = [n for n in notes if "**" in n or "`" in n or "→" in n]
    assert not bad_marks, "还带着注释体记号或箭头:\n  " + "\n  ".join(bad_marks)


def test_the_plain_comparability_notes_still_say_what_changed():
    """★ 红法:把 `PLAIN_WHAT` 某条写成「口径变了」这类空话 ⟹ 这里红。

    这是内容判据 —— 「不可比」三个字对读者没有用,他要知道**两边各数的是什么**。
    与 `test_comparability_notes_say_what_changed_not_just_that_it_changed`
    对审计版的要求一致,只是换成人话。
    """
    notes = l1_mapper.comparability_notes_plain()
    for n in notes:
        assert "旧" in n and "新" in n, f"没说清旧新各数什么:{n}"
        assert len(n) >= 16, f"太短,读者看不出差别:{n}"


def test_the_plain_comparability_notes_cover_every_slot():
    """★ 红法:让 `PLAIN_WHAT` 漏一个槽、退回审计原话 ⟹ 这里红。

    漏登记时**宁可露出审计原话也不许吞掉那一条**(本仓的固定取向),
    但那份露出必须被测试发现,否则第 7 个槽会悄悄带着内部指标名进交付物。
    """
    missing = sorted(set(l1_mapper.NOT_COMPARABLE) - set(l1_mapper.PLAIN_WHAT))
    assert not missing, f"这些槽没有给人读的比较说明:{missing}"


def test_the_audit_comparability_notes_are_not_lost():
    """★ 红法:把 `comparability_notes()` 改成返回人话版 ⟹ 这里红。

    分层不是删掉审计细节 —— 核对口径的人仍要能查到「哪个旧槽键对到了哪条 L1 指标」。
    """
    notes = l1_mapper.comparability_notes()
    assert all("→" in n for n in notes), notes


def test_the_payload_carries_both_versions_of_the_comparability_notes():
    """★ 红法:让载荷只发审计版(或只发人话版)⟹ 这里红。

    载荷是**两个出口的共同来源**:前端报告页与 HTML 交付物都从它读。
    只发审计版 ⟹ 两个出口都漏内部指标名;只发人话版 ⟹ 核对口径的人查不到映射。
    """
    rec = build_l1_report(_SID, logs={})
    l1 = rec["l1"]
    assert l1["comparability_notes"] == l1_mapper.comparability_notes(), (
        "载荷里的审计版被换掉了 —— 核对口径的人要能查到槽键→指标名")
    assert l1["comparability_notes_plain"] == l1_mapper.comparability_notes_plain(), (
        "载荷里没有给人读的比较说明 —— 两个出口会各自去编一份")


def test_the_report_prints_the_plain_comparability_notes(recordings):
    """★ 红法:把 HTML 报告改回印 `comparability_notes` ⟹ 这里红。

    与 `test_dropped_slot_reasons_in_the_report_are_plain_chinese` 同一个理由:
    交付物正文里不该出现 `connective_density → discourse_marker_rate` 这种串。
    """
    html = ReportGenerator(output_dir=str(recordings)).generate_report(_SID)
    assert html, "报告没生成 —— 本测试的前提不成立"
    body = _text_of(open(html, encoding="utf-8").read())
    assert "不可相减" in body, "比较陷阱那段没进报告 —— 本测试退化为空断言"
    for line in l1_mapper.comparability_notes_plain():
        assert line[:14] in body, f"给人读的比较说明没进报告:{line}"
