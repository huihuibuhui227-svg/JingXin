# tests/test_jx_boundary_feasibility.py
"""**离线标边界的可行性核算**(裁定②)的数字必须可复算。

## 钉的是什么

1. **覆盖率按毫秒算。** `declared_ts` 是毫秒(`session_clock.py`),按秒读会把
   18 分钟的场算成 17843 分钟,于是每一场都显示"覆盖 0%" —— 实测踩过一次。
2. **三分类,不是两分类。** `cover is None` 有两种来路:
   没有转写 / 有转写但没有观测账本(实测 `孙偲齐`/`吴思诚` 两场)。
   把它们并进"截断"会把 2 场**误报成音频损坏**。
3. **"可标边界上界"只数覆盖完整的场次。** 截断场里标了也只能覆盖音频那一段,
   算进去会把工作量报大。

## 不依赖 `data/`

全部合成:`recordings` 与 `transcripts` 都是 `tmp_path`。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.jx_boundary_feasibility import FULL_COVER, _log_span_sec, survey

SID = "20260929_180047_6fee"
FORMAL = f"20260929_180047-2-廖怡涵-261210823-金融8班__{SID}"


def _session(root: Path, name: str = FORMAL, span_ms: int | None = 600_000) -> Path:
    """铺一场:目录名要过 `FORMAL_PATTERN`;`span_ms=None` ⟹ 不写账本。"""
    d = root / name
    d.mkdir(parents=True)
    if span_ms is not None:
        with (d / "retention.face.jsonl").open("w", encoding="utf-8") as f:
            # 三个点就够算跨度;`declared_ts` 单位是**毫秒**
            for i, ts in enumerate((0, span_ms // 2, span_ms), 1):
                f.write(json.dumps({"kind": "frame", "seq": i, "declared_ts": ts}) + "\n")
    return d


def _transcript(root: Path, sid: str = SID, audio_sec: float = 600.0, nseg: int = 20) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{sid}.json").write_text(json.dumps({
        "session_id": sid,
        "audio_seconds": audio_sec,
        "segments": [{"index": i, "text": "x"} for i in range(nseg)],
    }), encoding="utf-8")


def test_declared_ts_is_milliseconds(tmp_path):
    """★ 跨度按毫秒算。按秒算的话 600 秒会变成 600000 秒(166 小时)。

    红法:把 `_log_span_sec` 里的 `/ 1000.0` 删掉 ⟹ 本条红。
    """
    d = _session(tmp_path, span_ms=600_000)
    assert _log_span_sec(d) == pytest.approx(600.0)


def test_full_coverage_session_is_counted_with_its_segments(tmp_path):
    """覆盖 ≥90% ⟹ 进"可标",段数计入上界。"""
    rec, tr = tmp_path / "rec", tmp_path / "tr"
    _session(rec, span_ms=600_000)
    _transcript(tr, audio_sec=600.0, nseg=20)

    r = survey(rec, tr)
    t = r["totals"]
    assert (t["n_formal"], t["n_full_cover"], t["n_truncated"]) == (1, 1, 0)
    assert t["segments_full_cover"] == 20
    assert r["sessions"][0]["cover"] == pytest.approx(1.0)


def test_truncated_session_is_not_counted_toward_the_labeling_budget(tmp_path):
    """★ 截断场(音频只有观测的 6%)不进上界 —— 否则工作量报大。

    红法:把 `full` 的筛选条件从 `>= FULL_COVER` 改成 `is not None` ⟹ 本条红。
    """
    rec, tr = tmp_path / "rec", tmp_path / "tr"
    _session(rec, span_ms=600_000)
    _transcript(tr, audio_sec=36.0, nseg=5)   # 6%

    t = survey(rec, tr)["totals"]
    assert t["n_full_cover"] == 0
    assert t["n_truncated"] == 1
    assert t["segments_full_cover"] == 0


def test_missing_ledger_is_unknown_not_truncated(tmp_path):
    """★ 有转写、没有观测账本 ⟹ "无法核算",**不是**"覆盖 0%/截断"。

    实测 `20260929_185929-2-孙偲齐` 与 `20260929_191038-2-吴思诚` 两场就是这样:
    面部账本缺失,而音频在。并进"截断"会误报成音频损坏。

    红法:把 `unknown` 那条分类删掉(让 `cover is None` 落进 `trunc`)⟹ 本条红。
    """
    rec, tr = tmp_path / "rec", tmp_path / "tr"
    _session(rec, span_ms=None)
    _transcript(tr, audio_sec=540.0, nseg=22)

    r = survey(rec, tr)
    t = r["totals"]
    assert t["n_unknown_span"] == 1
    assert t["n_truncated"] == 0
    assert t["n_full_cover"] == 0
    assert t["unknown_span"][0]["audio_min"] == pytest.approx(9.0)


def test_missing_transcript_is_reported_separately(tmp_path):
    """没有转写 ⟹ 单列一类,不与"截断"或"无法核算"混。"""
    rec = tmp_path / "rec"
    _session(rec, span_ms=600_000)
    t = survey(rec, tmp_path / "tr")["totals"]
    assert t["n_no_transcript"] == 1
    assert t["n_truncated"] == 0
    assert t["n_unknown_span"] == 0


def test_non_formal_directories_are_skipped(tmp_path):
    """`_试跑` / `NONE` / 匿名测试场不是正式素材 —— 与 `corpus.audit_corpus` 同一套排除规则。"""
    rec = tmp_path / "rec"
    _session(rec, span_ms=600_000)
    for bad in ("_试跑", "NONE", "20260925_203826_2449"):
        (rec / bad).mkdir()
    assert survey(rec, tmp_path / "tr")["totals"]["n_formal"] == 1


def test_missing_recordings_root_is_an_error_not_an_empty_survey(tmp_path):
    """素材根不存在 ⟹ 报错。**不能**返回"0 场"——那会被读成"都标完了"。"""
    r = survey(tmp_path / "nope", tmp_path / "tr")
    assert "error" in r
