# tests/test_media_truncation_hookup.py
"""补强 A2:媒体完整性检测**接进会话收尾对账**。

本文件钉的是"接线",不是判据 —— 判据与量具的 47 条单测在
`tests/test_media_truncation.py`,这里一条都不重测。被钉住的是:

  1. (i)   收尾对账里被调用**一次且只一次**(重复对账不重复累积);
  2. (ii)  截断 / 超长的场次被标出来,并映射到**既有**缺口码(`no_samples` /
           `time_unusable`)—— 且**没有第 10 类**;
  3. (iii) 正常场次不被误标(含**真实语料**里的 `ok` 场次做正例);
  4. (iv)  无媒体 / 无台账 / 坏文件 / 清单坏掉 —— 一律"记录缺口",**不抛**;
  5. (v)   幂等:连跑两次结果一致,且第二次**不重写** `session.json`。

为什么"挂收尾、不挂上传"值得单列一条:上传那一刻 `questions.jsonl` 还没报完,
跨度比真实跨度短,一场正常录制会被判 `overlong`。见
`test_upload_style_early_check_would_misjudge_a_short_ledger` —— 它把那个误判
**演示**出来(不是把错误行为写进生产代码)。
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest

import media_retention
import media_truncation as mt
import media_truncation_probe as mp

session_meta = importlib.import_module("session_meta")

SID = "20260925_203826_2449"
T0 = time.time()          # `upsert_question` 有墙钟量程闸,时间戳必须是"现在"附近


# ── 夹具 ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """与 `tests/test_session_closeout.py` 同一套隔离:录制根挪到 tmp_path。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    session_meta._reset_for_tests()
    return tmp_path


def _manifest(root: Path, sid: str = SID) -> Path:
    """造一份**与真实场次同形**的 `session.json`(会话级元信息的落点)。"""
    d = root / sid
    d.mkdir(parents=True, exist_ok=True)
    p = d / "session.json"
    p.write_text(json.dumps({"session_id": sid, "started_at": "2026-09-25T20:38:26",
                             "asr": {}, "logs": {}}, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def _stub_probe(monkeypatch, span_sec, media_sec, *, calls: list | None = None,
                camera: bool = True):
    """把**量具**打桩,判定仍走真的 `mt.assess`。

    只桩"量到多少秒"这一件事:这样缺口码映射仍是生产那条路径(不是造出来的),
    而测试不必依赖盘上的真素材 / 本机有没有 ffprobe。
    """

    def probe(session_dir, *, track="auto", tol=mt.DEFAULT_TOL, ceiling=mt.DEFAULT_CEILING):
        if calls is not None:
            calls.append(str(session_dir))
        rows = [{"qid": "q0", "index": 0,
                 "ask_start": 1000.0, "ask_end": 1000.0 + span_sec}]
        media = {"camera.webm": media_sec} if media_sec is not None else {}
        res = mt.assess(rows, media, tol=tol, ceiling=ceiling)
        ev = dict(res.get("evidence") or {})
        ev.update({
            "session_dir": str(session_dir), "track": track,
            "ffprobe": "/usr/bin/ffprobe", "has_camera": camera,
            "per_file": ({"camera.webm": {"tracks": {"v:0": media_sec, "a:0": media_sec},
                                          "av_ratio": 1.0}} if media else {}),
        })
        out = dict(res)
        out["session"] = Path(session_dir).name
        out["evidence"] = ev
        return out

    monkeypatch.setattr(mp, "probe_session", probe)
    return probe


# ── (i) 收尾时被调用一次且只一次 ─────────────────────────────────────────

def test_closeout_calls_the_probe_exactly_once(_isolated, monkeypatch):
    """一次收尾 = **一次**检测。多调一次就多开一次 ffprobe(52 场就是 52 次浪费),
    少调一次就是"没判"却报绿。"""
    calls: list[str] = []
    _stub_probe(monkeypatch, 6.0, 5.9, calls=calls)
    _manifest(_isolated)

    got = session_meta.check_session(SID)
    assert len(calls) == 1, calls
    assert calls[0].endswith(SID), f"量的不是这一场的目录:{calls[0]}"
    assert got["media_integrity"]["session_id"] == SID

    # 判词那一步**不许**再量一次(它只读已经算好的结果)
    session_meta.closeout_verdict(got)
    assert len(calls) == 1, "closeout_verdict 又量了一次"


def test_check_session_still_reports_every_old_field(_isolated, monkeypatch):
    """只增不改:既有字段一个都不许少(下游按 `got["media"]` 这类键取数)。"""
    _stub_probe(monkeypatch, 6.0, 5.9)
    _manifest(_isolated)
    got = session_meta.check_session(SID)
    for k in ("session_id", "media", "video", "video_bytes", "degraded",
              "missing_meta", "reported_questions", "missing_questions",
              "unreadable_lines", "media_integrity"):
        assert k in got, f"少了既有字段 {k}"


# ── (ii) 截断/超长 → 既有缺口码,且没有第 10 类 ──────────────────────────

def test_truncated_session_is_flagged_and_mapped_to_no_samples(_isolated, monkeypatch):
    """`20260925_225936_a971` 的读数(跨度 6.24s / 媒体 0.924s,覆盖率 14.8%)。

    归因必须是既有的 `no_samples`:**不是**新增一类"媒体截断"。
    """
    _stub_probe(monkeypatch, 6.24, 0.924)
    p = _manifest(_isolated)

    got = session_meta.check_session(SID, expected_questions=1)
    mi = got["media_integrity"]
    assert mi["verdict"] == "truncated"
    assert mi["gap"] == "no_samples", mi
    assert mi["gap"] in mt.GAP_FOR_VERDICT.values()
    assert mi["judgeable"] is True
    assert mi["span_sec"] == pytest.approx(6.24, abs=0.01)
    assert mi["media_sec"] == pytest.approx(0.924, abs=0.001)

    # 判词要进收尾结论 —— 否则"发现了"等于没发现
    ok, reasons = session_meta.closeout_verdict(got, expected_questions=1)
    assert ok is False
    assert any("no_samples" in r for r in reasons), reasons

    # 并落进会话级元信息(与既有字段并列)
    saved = json.loads(p.read_text(encoding="utf-8"))
    assert saved["media_integrity"] == mi
    assert saved["session_id"] == SID and saved["logs"] == {}   # 别人的键一个没动


def test_overlong_session_is_mapped_to_time_unusable(_isolated, monkeypatch):
    """反向:录像 284s / 台账 6.2s(`20260928_165520_90e7`)⟹ 时间轴不可重建。

    这一类**不是**"录少了",补救动作是补台账 —— 所以码必须不同。
    """
    _stub_probe(monkeypatch, 6.227, 284.369)
    _manifest(_isolated)
    got = session_meta.check_session(SID)
    mi = got["media_integrity"]
    assert mi["verdict"] == "overlong"
    assert mi["gap"] == "time_unusable", mi
    ok, reasons = session_meta.closeout_verdict(got)
    assert ok is False and any("time_unusable" in r for r in reasons), reasons


def test_no_tenth_gap_code_can_be_persisted(_isolated, monkeypatch):
    """★ 探针就算给出一个**不认识**的码,也绝不许落进会话元信息。

    这是"9 类缺口"这条对外口径在**写盘处**的钉子:改坏 `probe_session` 不会
    多出一个第 10 类,只会把事实写进 `gap_detail`。
    """
    def probe(session_dir, **_kw):
        return {"session": Path(session_dir).name, "verdict": "truncated",
                "span_sec": 10.0, "media_sec": 1.0, "ratio": 0.1,
                "gap": "media_broken",          # ← 不存在的码
                "gap_detail": "我编的", "judgeable": True,
                "evidence": {"media_sec_by_file": {"camera.webm": 1.0},
                             "tol": 0.9, "ceiling": 1.5}}
    monkeypatch.setattr(mp, "probe_session", probe)
    _manifest(_isolated)

    mi = session_meta.check_session(SID)["media_integrity"]
    assert mi["gap"] is None, "不认识的码被写进去了"
    assert "media_broken" in mi["gap_detail"], "码没采用,但要让人看见它被拒了"
    assert mi["verdict"] == "truncated", "事实(verdict)照报,不因码不认识就吞掉"


def test_only_the_existing_nine_codes_are_reachable(_isolated, monkeypatch):
    """值与码都在**既有**集合里(能取到 `l1_indicators.KINDS` 就得对得上)。"""
    produced = set()
    for span, media in ((6.24, 0.924), (6.227, 284.369), (10.0, 10.0)):
        _stub_probe(monkeypatch, span, media)
        produced.add(session_meta.media_integrity(SID)["gap"])
    assert produced == {None, "no_samples", "time_unusable"}

    try:
        from report_frontend import l1_indicators as li
    except Exception:  # noqa: BLE001 - 仓库不在就跳过
        pytest.skip("拿不到 report_frontend.l1_indicators")
    assert len(li.KINDS) == 9, f"系统口径变了:{li.KINDS}"
    assert produced - {None} <= set(li.KINDS)


def test_session_without_camera_is_not_attributed_as_truncation(_isolated, monkeypatch):
    """★ 没有 `camera.webm` 的场次(音频-only / 只有分题音频块)**不归因缺口码**。

    实测 `..._afb2`(跨度 313.8s、最长音频单块 23.3s)、`..._0d89`、`..._302f` 三场
    都落在这一类:`media_sec` 是**最长单块**,不是整场连续录制,比出来的 `truncated`
    是**设计如此**(`probe_session` 的 `probe_note` 原话:"别当截断")。把它归因成
    `no_samples` 就是一条假 warn —— 假 warn 会淹掉真的录坏。
    """
    _stub_probe(monkeypatch, 313.787, 23.339, camera=False)
    _manifest(_isolated)

    mi = session_meta.check_session(SID)["media_integrity"]
    assert mi["verdict"] == "truncated", "事实要照报"
    assert mi["gap"] is None, "音频-only 口径不许给缺口码"
    assert mi["gap_skipped_reason"] == "no_camera"
    assert "不归因缺口码" in mi["gap_detail"]
    assert "no_samples" in mi["gap_detail"], "被拒的那个码要写清楚(可复核)"
    ok, reasons = session_meta.closeout_verdict({"reported_questions": [0], "video": True,
                                                 "media_integrity": mi})
    assert ok is True, f"不因'没有连续录制'变黄:{reasons}"


# ── (iii) 正常场次不被误标 ───────────────────────────────────────────────

def test_ok_session_is_not_flagged(_isolated, monkeypatch):
    _stub_probe(monkeypatch, 520.716, 550.514)      # 1592 的文件级读数
    _manifest(_isolated)
    mi = session_meta.check_session(SID)["media_integrity"]
    assert mi["verdict"] == "ok" and mi["gap"] is None and mi["judgeable"] is True
    ok, reasons = session_meta.closeout_verdict({"reported_questions": [0], "video": True,
                                                 "media_integrity": mi})
    assert ok is True, reasons


def test_judgeable_false_never_turns_a_green_closeout_yellow():
    """`judgeable=False`(没台账 / 没媒体 / ffprobe 不在)**不加理由**。

    否则每台没装 ffprobe 的机器上,所有收尾都会从绿变黄 —— 那又是一条假 warn。
    """
    clean = {"session_id": SID, "media": {"face": 1}, "video": True, "video_bytes": 10,
             "degraded": [], "missing_meta": [], "reported_questions": [0],
             "missing_questions": [], "unreadable_lines": 0,
             "media_integrity": {"verdict": "ok", "gap": None, "judgeable": False,
                                 "gap_detail": "一份媒体都没量到"}}
    ok, reasons = session_meta.closeout_verdict(clean)
    assert ok is True, reasons


_REAL_ROOT = Path.home() / "JingXin" / "recordings"
REAL_OK = "20260926_153854_1592"           # 跨度 520.716s / 视轨 550.514s → ok
REAL_TRUNCATED = "20260925_225936_a971"    # 跨度 6.24s / 录像 0.924s → truncated
REAL_AUDIO_ONLY = "20260927_195528_afb2"   # 无 camera.webm,只有分题音频块


def _real_dir(sid: str) -> Path | None:
    """真实语料里这一场的目录(存过标注的场次叫 `<标签>__<sid>`)。"""
    if not _REAL_ROOT.is_dir():
        return None
    d = _REAL_ROOT / sid
    if d.is_dir():
        return d
    for cand in sorted(_REAL_ROOT.glob(f"*__{sid}")):
        if cand.is_dir():
            return cand
    return None


@pytest.mark.skipif(_real_dir(REAL_OK) is None, reason="本机没有真实语料")
def test_real_ok_session_is_a_true_negative(monkeypatch):
    """★ 正例用**真实会话**(1592:8 道题,跨度 520.7s,视轨 550.5s)。

    `record_integrity=False` —— 复核/演示不许改动真实素材目录里的任何字节。
    """
    d = _real_dir(REAL_OK)
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(_REAL_ROOT))
    manifest = d / "session.json"
    before = manifest.read_bytes() if manifest.is_file() else None

    mi = session_meta.check_session(REAL_OK, record_integrity=False)["media_integrity"]
    assert mi["verdict"] == "ok", mi["gap_detail"]
    assert mi["gap"] is None and mi["judgeable"] is True
    assert mi["span_sec"] == pytest.approx(520.716, abs=1.0)
    assert mi["media_sec"] == pytest.approx(550.514, abs=1.0)
    assert mi["ratio"] == pytest.approx(1.057, abs=0.02)
    # 音视不同长(音轨 57.0s)的取证留着:它解释"为什么不能写录像被截断"
    assert mi["tracks"]["camera.webm"]["av_ratio"] == pytest.approx(0.104, abs=0.01)

    after = manifest.read_bytes() if manifest.is_file() else None
    assert before == after, "只读用法不许写真实会话目录"


@pytest.mark.skipif(_real_dir(REAL_TRUNCATED) is None, reason="本机没有真实语料")
def test_real_truncated_session_is_flagged_without_writing(monkeypatch):
    """真实截断场次(a971):0.924s 的录像对 6.24s 的跨度 ⟹ `no_samples`。"""
    d = _real_dir(REAL_TRUNCATED)
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(_REAL_ROOT))
    manifest = d / "session.json"
    before = manifest.read_bytes()

    mi = session_meta.check_session(REAL_TRUNCATED, record_integrity=False)["media_integrity"]
    assert mi["verdict"] == "truncated" and mi["gap"] == "no_samples"
    assert mi["media_sec"] == pytest.approx(0.924, abs=0.01)
    assert manifest.read_bytes() == before


@pytest.mark.skipif(_real_dir(REAL_AUDIO_ONLY) is None, reason="本机没有真实语料")
def test_real_audio_only_session_is_not_attributed(monkeypatch):
    """真实音频-only 场次(afb2):探针判 `truncated`,但**不给缺口码**。"""
    d = _real_dir(REAL_AUDIO_ONLY)
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(_REAL_ROOT))
    before = (d / "session.json").read_bytes()

    mi = session_meta.check_session(REAL_AUDIO_ONLY, record_integrity=False)["media_integrity"]
    assert mi["has_camera"] is False
    assert mi["verdict"] == "truncated" and mi["gap"] is None
    assert mi["gap_skipped_reason"] == "no_camera"
    assert (d / "session.json").read_bytes() == before


# ── (iv) 无媒体 / 无台账 / 坏文件:记录缺口,不抛 ────────────────────────

def test_missing_questions_jsonl_is_no_ledger_not_an_exception(_isolated):
    """没有台账 ⟹ `no_ledger`:**不给缺口码**(算不出跨度 ≠ 媒体坏了)。

    ⚠️ 这条**不打桩**(走真的 `probe_session`)—— 并且**不能用**
    `monkeypatch.undo()` 来撤桩:那会把夹具设的 `JINGXIN_RECORDINGS_DIR` 一起撤掉,
    于是真的写进使用者的录制目录(`tests/test_session_closeout.py` 记着同一个坑)。
    """
    p = _manifest(_isolated)

    got = session_meta.check_session(SID)    # 不抛
    mi = got["media_integrity"]
    assert mi["verdict"] == "no_ledger"
    assert mi["gap"] is None and mi["judgeable"] is False
    assert "questions.jsonl" in mi["gap_detail"]
    assert json.loads(p.read_text(encoding="utf-8"))["media_integrity"] == mi


def test_session_without_any_media_is_recorded_not_raised(_isolated):
    """有台账、没有任何媒体文件:不是"截断"(没有秒数可比),也不是通过。"""
    p = _manifest(_isolated)
    session_meta.upsert_question(SID, qid="Q0", index=0, ask_start=T0, ask_end=T0 + 6.0)

    mi = session_meta.check_session(SID)["media_integrity"]
    assert mi["verdict"] == "ok" and mi["gap"] is None
    assert mi["judgeable"] is False and mi["media_sec"] is None
    assert "没有任何 media 文件" in (mi["probe_note"] or "")
    assert json.loads(p.read_text(encoding="utf-8"))["media_integrity"]["judgeable"] is False


def test_broken_media_file_is_recorded_not_raised(_isolated):
    """**坏文件**(不是 webm 的一堆字节):量不到时长 → `judgeable=False`,不抛。

    注意**不许**判成 `truncated`:量不到 ≠ 录少了(判词比事实更狠)。
    """
    d = _manifest(_isolated).parent
    (d / "media").mkdir()
    (d / "media" / "camera.webm").write_bytes(b"not a webm at all")
    session_meta.upsert_question(SID, qid="Q0", index=0, ask_start=T0, ask_end=T0 + 6.0)

    mi = session_meta.check_session(SID)["media_integrity"]    # 不抛
    assert mi["judgeable"] is False or mi["verdict"] == "ok"
    assert mi["gap"] is None, mi


def test_probe_crash_degrades_to_probe_error(_isolated, monkeypatch):
    """探针自己抛 ⟹ `probe_error` + 一句"不是录坏了";对账其余字段照旧在。"""
    def boom(session_dir, **_kw):
        raise RuntimeError("ffprobe 挂了")
    monkeypatch.setattr(mp, "probe_session", boom)
    _manifest(_isolated)

    got = session_meta.check_session(SID)      # 不抛
    assert got["media_integrity"]["verdict"] == "probe_error"
    assert got["media_integrity"]["gap"] is None
    assert "不是录坏了" in got["media_integrity"]["gap_detail"]
    assert got["media"] == {}


def test_broken_session_json_is_left_alone(_isolated, monkeypatch):
    """`session.json` 坏掉 ⟹ **不改写**它(唯一一份清单不能被我们毁了),也不抛。"""
    _stub_probe(monkeypatch, 6.0, 5.9)
    d = _isolated / SID
    d.mkdir()
    (d / "session.json").write_text("{半截", encoding="utf-8")
    assert session_meta.record_media_integrity(SID) is False
    assert (d / "session.json").read_text(encoding="utf-8") == "{半截"
    assert session_meta.check_session(SID)["media_integrity"]["verdict"] == "ok"


def test_missing_session_json_is_not_created(_isolated, monkeypatch):
    """清单不存在 ⟹ **不新建**(它的形状不归本模块管),只报"没写成"。"""
    _stub_probe(monkeypatch, 6.0, 5.9)
    d = _isolated / SID
    d.mkdir()
    assert session_meta.record_media_integrity(SID) is False
    assert not (d / "session.json").exists()
    # 但结果照样回到对账里 —— 没写成不等于没量
    assert session_meta.check_session(SID)["media_integrity"]["gap"] is None


# ── (v) 幂等 ────────────────────────────────────────────────────────────

def test_repeated_closeout_is_idempotent(_isolated, monkeypatch):
    """★ 连跑两次:结果一致,且第二次**一个字节都不写**(mtime 也不许翻新)。

    为什么在意 mtime:"这一场什么时候变的"是事后排查的唯一线索;每次对账都
    `os.replace` 一遍会让这条线索永远指向"刚才"。
    """
    _stub_probe(monkeypatch, 6.24, 0.924)
    p = _manifest(_isolated)

    first = session_meta.check_session(SID)
    block1 = p.read_bytes()
    mtime1 = p.stat().st_mtime_ns

    second = session_meta.check_session(SID)
    assert first == second, "两次对账的结果不一致"
    assert p.read_bytes() == block1, "内容一致却又写了一遍"
    assert p.stat().st_mtime_ns == mtime1, "内容一致却把 mtime 翻了新"


def test_rewrite_happens_only_when_the_measurement_changes(_isolated, monkeypatch):
    """媒体变了 ⟹ 该更新。否则"上次判 ok、这次录短了"会一直显示旧结论。"""
    _stub_probe(monkeypatch, 6.24, 5.9)          # 先判 ok
    p = _manifest(_isolated)
    session_meta.check_session(SID)
    assert json.loads(p.read_text(encoding="utf-8"))["media_integrity"]["verdict"] == "ok"

    _stub_probe(monkeypatch, 6.24, 0.924)        # 再量:媒体短了
    mi = session_meta.check_session(SID)["media_integrity"]
    assert mi["verdict"] == "truncated" and mi["gap"] == "no_samples"
    assert json.loads(p.read_text(encoding="utf-8"))["media_integrity"] == mi


def test_upload_style_early_check_would_misjudge_a_short_ledger(_isolated, monkeypatch):
    """★ 演示"为什么不能挂在上传那一刻"。

    录制进行到一半时台账里只有**第 1 道题**(跨度 5.4s),录像已经 83s ——
    `assess` 当场判 `overlong` / `time_unusable`。同一场录完、题目报齐之后
    (跨度 300s)就是 `ok`。所以判定只能在收尾做。
    """
    _stub_probe(monkeypatch, 5.408, 83.292)      # 半场的台账(4579 那个形态)
    early = session_meta.media_integrity(SID)
    assert early["verdict"] == "overlong" and early["gap"] == "time_unusable"

    _stub_probe(monkeypatch, 300.0, 310.0)       # 录完、题目报齐(跨度 300s / 录像 310s)
    assert session_meta.media_integrity(SID)["gap"] is None


# ── 块形状:文档与代码不许各说各话 ───────────────────────────────────────

def test_block_schema_is_the_documented_one(_isolated, monkeypatch):
    _stub_probe(monkeypatch, 6.24, 0.924)
    _manifest(_isolated)
    mi = session_meta.check_session(SID)["media_integrity"]
    assert set(mi) == set(session_meta._empty_integrity_block(SID))
    assert mi["measured_by"] == "media_truncation_probe.probe_session"
    assert mi["attributed_by"] == "media_truncation.GAP_FOR_VERDICT"
    assert mi["track"] == "auto", "量具轨道必须是文件级(视轨优先),不是音轨"
    assert isinstance(mi["gap_detail"], str) and mi["gap_detail"]


def test_integrity_line_never_prints_a_bare_code(_isolated, monkeypatch):
    _stub_probe(monkeypatch, 6.24, 0.924)
    _manifest(_isolated)
    line = session_meta.integrity_line(session_meta.check_session(SID)["media_integrity"])
    assert "no_samples" in line and "覆盖率" in line, line
    ok_line = session_meta.integrity_line({"verdict": "ok", "gap": None,
                                           "judgeable": True, "gap_detail": "覆盖 100%"})
    assert ok_line.startswith("ok")
