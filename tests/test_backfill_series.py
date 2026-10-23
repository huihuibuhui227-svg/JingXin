# tests/test_backfill_series.py
"""`tools/jx_backfill_series.py` 的钉子 —— 它补的是**正式场 V 族唯一的数据源**。

## 为什么这个工具值得钉

正式场 24/24 语音日志 0 行 ⟹ V 族 12 条里那 8 条序列指标**全靠**
`data/l0_series/<sid>.npz`。2026-10-24 实测:37 场有 `camera.webm`,只有 7 场有缓存。
也就是说这个工具一跑,影响面是 **31 场的 V 族**;跑错一次,报告上会同时错 31 场。

## 钉的三件事

1. **目录名 → 真 sid**。目录可能带标注前缀(`…-2-候选甲-000000000-投资学__<sid>`),
   而缓存文件名、报告读缓存用的都是**真 sid**。算错 ⟹ 缓存写进去了、
   报告却按真 sid 找不到它 ⟹ 补过了还是 `not_extracted`,而且**不报错**。
2. **判成 `stale` 必须真的重算**。`voice_series.extract_session(force=False)`
   见到缓存就**直接返回它** —— 判"源变了"却传 `force=False`,跑完会报
   "backfilled"而一个字节都没重算。这正是本仓反复栽的「验证跑错了对象」。
3. **没有音频的场次不许写缓存**,也不许被算成"补成功"—— 那些场次在报告里
   该保留 `modality_missing`,把它们的计数混进来会让"补了几场"这个数虚高。

红法(都实测过):
* 把 `session_id_of` 改成恒返回 `sess_dir.name` ⟹ 第 1 条红;
* 把 `main` 里 `force=args.force or reason in ("stale", "forced")` 里的
  `reason in (...)` 删掉 ⟹ 第 2 条红;
* 把 `pending_reason` 的 `no_audio` 那一支删掉(让它落到 `missing`)⟹ 第 3 条红。
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

backfill = importlib.import_module("jx_backfill_series")


def _session_dir(root: Path, sid: str, *, with_camera: bool,
                 label: dict | None = None) -> Path:
    """造一场会话目录。目录名**故意带标注前缀** —— 真 sid 只在 `__` 之后。"""
    d = root / f"20260101_000000-1-候选甲-000000000-系__{sid}"
    (d / "media").mkdir(parents=True)
    if with_camera:
        (d / "media" / "camera.webm").write_bytes(b"not-a-real-webm")
    if label is not None:
        (d / "label.json").write_text(json.dumps(label, ensure_ascii=False),
                                      encoding="utf-8")
    return d


def _fake_extract(monkeypatch, calls: list):
    """替掉 `extract_session`,记下每次调用传的 `force`。"""
    from report_frontend import voice_series

    def fake(sid: str, force: bool = False):
        calls.append({"sid": sid, "force": force})
        return [voice_series.VoiceSeries(
            session_id=sid, index=None, frame_sec=0.1,
            f0=np.full(10, 120.0), voiced_prob=np.ones(10),
            rms=np.full(10, 0.05), source="camera.webm")]

    monkeypatch.setattr(voice_series, "extract_session", fake)


# ── 1. 目录名 → 真 sid ──────────────────────────────────────────────────
def test_session_id_comes_after_the_double_underscore(tmp_path):
    """带标注前缀的目录名里,**真 sid 是 `__` 之后那段**。

    红法:`session_id_of` 恒返回 `sess_dir.name` ⟹ 本条红。
    """
    d = _session_dir(tmp_path, "20260101_000000_ab12", with_camera=True)
    assert backfill.session_id_of(d) == "20260101_000000_ab12"


def test_label_json_wins_over_the_directory_name(tmp_path):
    """`label.json` 写着权威 session_id ⟹ 用它,不按 `__` 猜。

    (目录被改过名、但标注没改的场景;`jx_check_session._session_id` 同一优先级。)
    """
    d = _session_dir(tmp_path, "20260101_000000_ab12", with_camera=True,
                     label={"session_id": "20260101_000000_zz99"})
    assert backfill.session_id_of(d) == "20260101_000000_zz99"


def test_a_broken_label_falls_back_instead_of_dropping_the_session(tmp_path):
    """`label.json` 是坏 JSON ⟹ 退回目录名规则,**不许**让整场消失。"""
    d = _session_dir(tmp_path, "20260101_000000_ab12", with_camera=True)
    (d / "label.json").write_text("{ not json", encoding="utf-8")
    assert backfill.session_id_of(d) == "20260101_000000_ab12"


# ── 2. 该不该补:四种理由分开 ───────────────────────────────────────────
def test_no_audio_is_its_own_reason_not_missing(tmp_path, monkeypatch):
    """盘上没有音频 ⟹ `no_audio`,**不是** `missing`。

    这个码在报告里要保留成 `modality_missing`;把它算进"该补"会让
    "待补 N 场"这个数虚高,跑完还会多出 N 条 `failed`。

    红法:删掉 `pending_reason` 的 `no_audio` 那一支 ⟹ 本条红。
    """
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    sid = "20260101_000000_cc34"
    _session_dir(tmp_path, sid, with_camera=False)

    assert backfill.pending_reason(sid, {}) == "no_audio"


def test_missing_cache_with_audio_on_disk(tmp_path, monkeypatch):
    """有 `camera.webm`、没有缓存 ⟹ `missing`(本工具的主要目标)。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    sid = "20260101_000000_dd56"
    _session_dir(tmp_path, sid, with_camera=True)

    assert backfill.pending_reason(sid, {}) == "missing"


def test_a_cache_with_no_manifest_entry_is_rebuilt(tmp_path, monkeypatch):
    """有缓存、清单里没记录 ⟹ `unrecorded`(不知道它怎么来的,保守重建)。"""
    from report_frontend import voice_series

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    sid = "20260101_000000_ee78"
    _session_dir(tmp_path, sid, with_camera=True)
    voice_series.save_series([voice_series.VoiceSeries(
        session_id=sid, index=None, frame_sec=0.1, f0=np.full(5, 120.0),
        voiced_prob=np.ones(5), rms=np.full(5, 0.05), source="camera.webm")])

    assert backfill.pending_reason(sid, {}) == "unrecorded"


def test_a_cache_whose_source_changed_is_stale(tmp_path, monkeypatch):
    """源文件指纹与清单不一致 ⟹ `stale`。"""
    from report_frontend import voice_series

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    sid = "20260101_000000_ff90"
    _session_dir(tmp_path, sid, with_camera=True)
    voice_series.save_series([voice_series.VoiceSeries(
        session_id=sid, index=None, frame_sec=0.1, f0=np.full(5, 120.0),
        voiced_prob=np.ones(5), rms=np.full(5, 0.05), source="camera.webm")])

    old = {"source_sig": ["whole", [[None, "x", 1, 1]]]}
    assert backfill.pending_reason(sid, {sid: old}) == "stale"


def test_a_fresh_cache_is_left_alone(tmp_path, monkeypatch):
    """反方向对照:缓存与源都没变 ⟹ `None`(不重算)。

    没有这一条,上面几条都可能只是"永远返回需要补"。
    """
    from report_frontend import voice_series

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    sid = "20260101_000000_1199"
    _session_dir(tmp_path, sid, with_camera=True)
    voice_series.save_series([voice_series.VoiceSeries(
        session_id=sid, index=None, frame_sec=0.1, f0=np.full(5, 120.0),
        voiced_prob=np.ones(5), rms=np.full(5, 0.05), source="camera.webm")])

    fresh = {sid: {"source_sig": backfill.source_sig(sid)}}
    assert backfill.pending_reason(sid, fresh) is None


# ── 3. 判成 stale 必须真的重算 ──────────────────────────────────────────
def test_stale_entries_are_extracted_with_force(tmp_path, monkeypatch):
    """★ `stale` ⟹ `extract_session` 必须收到 `force=True`。

    `force=False` 时它见到缓存就**直接返回**,一个字节都不重算 ——
    而工具会照样打印"backfilled"。这是「验证跑错了对象」的标准形态:
    跑的那条路径根本没执行到要证的那段代码。

    红法:把 `main` 里的 `reason in ("stale", "forced")` 删掉 ⟹ 本条红。
    """
    from report_frontend import voice_series

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    monkeypatch.setattr(backfill, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(backfill, "MANIFEST", tmp_path / "logs" / "manifest.json")
    sid = "20260101_000000_2244"
    _session_dir(tmp_path, sid, with_camera=True)
    voice_series.save_series([voice_series.VoiceSeries(
        session_id=sid, index=None, frame_sec=0.1, f0=np.full(5, 120.0),
        voiced_prob=np.ones(5), rms=np.full(5, 0.05), source="camera.webm")])
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "manifest.json").write_text(json.dumps(
        {sid: {"source_sig": ["whole", [[None, "old", 1, 1]]]}}), encoding="utf-8")

    calls: list = []
    _fake_extract(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv",
                        ["jx_backfill_series.py", "--sessions-dir", str(tmp_path),
                         "--pending"])
    backfill.main()

    assert calls, "判成 stale 的场次必须真的去提取"
    assert all(c["force"] is True for c in calls), \
        f"stale 却传了 force=False ⟹ extract_session 会直接返回旧缓存:{calls}"


def test_missing_entries_do_not_need_force(tmp_path, monkeypatch):
    """反方向对照:缓存根本不存在 ⟹ `force` 无所谓(没有旧缓存可返回)。

    这条**不**断言 force 是 False —— 那会把实现细节钉死。它断言的是
    "`missing` 的场次确实被提取了",用来防上面那条退化成"所有场次都提取"。
    """
    from report_frontend import voice_series

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    monkeypatch.setattr(backfill, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(backfill, "MANIFEST", tmp_path / "logs" / "manifest.json")
    sid = "20260101_000000_3355"
    _session_dir(tmp_path, sid, with_camera=True)

    calls: list = []
    _fake_extract(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv",
                        ["jx_backfill_series.py", "--sessions-dir", str(tmp_path),
                         "--all"])
    backfill.main()

    assert [c["sid"] for c in calls] == [sid]


# ── 4. 没有音频 / 提取不出帧:不许算成成功 ───────────────────────────────
def test_a_session_without_audio_is_skipped_not_failed(tmp_path, monkeypatch):
    """没有音频 ⟹ `skip`,**不写缓存**、**不算补成功**。

    `--all` 会把整个录制根扫一遍;没有音频的场次必须在这里被挡掉,
    否则报告里的 `modality_missing` 会被"补过一次"这件事搅浑。
    """
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    sid = "20260101_000000_4466"
    _session_dir(tmp_path, sid, with_camera=False)

    r = backfill.backfill_one(sid)
    assert r["status"] == "skip"
    assert "modality_missing" in r["why"]


def test_empty_extraction_is_failed_not_backfilled(tmp_path, monkeypatch):
    """提取跑完但一帧都没有 ⟹ `failed`,**不落空缓存**。

    `save_series` 会拒绝空序列,而"看起来有缓存、实际零帧"比没有缓存更坏:
    报告那边会从 `not_extracted` 变成"有序列但一条都算不出",更难诊断。
    """
    from report_frontend import voice_series

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    sid = "20260101_000000_5577"
    _session_dir(tmp_path, sid, with_camera=True)
    monkeypatch.setattr(voice_series, "extract_session",
                        lambda sid, force=False: [voice_series.VoiceSeries(
                            session_id=sid, index=None, frame_sec=0.0,
                            f0=np.array([]), voiced_prob=np.array([]),
                            rms=np.array([]), source="camera.webm")])

    r = backfill.backfill_one(sid)
    assert r["status"] == "failed"
    assert voice_series.load_series(sid) is None, "空序列不许落盘"


def test_unrecorded_entries_are_extracted_with_force(tmp_path, monkeypatch):
    """★ `unrecorded`(有缓存、清单无记录)也必须 `force=True`。

    ## 这条是怎么来的(2026-10-24,服务器实跑)

    第一次全量回填,12 场 `unrecorded` 全部报「✓ backfilled」而耗时 **0.0 秒** ——
    因为 `extract_session(force=False)` 见到缓存就直接返回它,一个字节都没重算。
    `force` 只在 `("stale", "forced")` 里传,漏了 `unrecorded`。

    这正是本仓反复栽的那种「验证跑错了对象」:跑的那条路径没执行到要证的那段代码,
    而输出照样打勾。**光断言 status 抓不住它** —— status 就是 `backfilled`。
    所以这里断言的是 `force` 参数本身。

    红法:把 `main` 里的 `"unrecorded"` 从那个 tuple 里删掉 ⟹ 本条红。
    """
    from report_frontend import voice_series

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    monkeypatch.setattr(backfill, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(backfill, "MANIFEST", tmp_path / "logs" / "manifest.json")
    sid = "20260101_000000_7799"
    _session_dir(tmp_path, sid, with_camera=True)
    voice_series.save_series([voice_series.VoiceSeries(
        session_id=sid, index=None, frame_sec=0.1, f0=np.full(5, 120.0),
        voiced_prob=np.ones(5), rms=np.full(5, 0.05), source="camera.webm")])

    calls: list = []
    _fake_extract(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv",
                        ["jx_backfill_series.py", "--sessions-dir", str(tmp_path),
                         "--all"])
    backfill.main()

    assert calls, "unrecorded 的场次必须真的去提取"
    assert all(c["force"] is True for c in calls), \
        f"有缓存却没 force ⟹ extract_session 直接返回旧缓存,0 秒'补成功':{calls}"


def test_a_cached_session_without_force_is_not_reported_as_backfilled(
        tmp_path, monkeypatch):
    """★ 守卫:`backfill_one` 自己有缓存、又没 force ⟹ 报 `reused`,**不是** `backfilled`。

    这一层不靠调用方传对参数 —— 就算将来有人新加一条调用路径忘了 force,
    也不会出现"0 秒补成功"这种假计数。

    红法:删掉 `backfill_one` 里 `load_series(...) is not None and not force` 那一支 ⟹ 本条红。
    """
    from report_frontend import voice_series

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    sid = "20260101_000000_8811"
    _session_dir(tmp_path, sid, with_camera=True)
    voice_series.save_series([voice_series.VoiceSeries(
        session_id=sid, index=None, frame_sec=0.1, f0=np.full(5, 120.0),
        voiced_prob=np.ones(5), rms=np.full(5, 0.05), source="camera.webm")])

    r = backfill.backfill_one(sid, force=False)
    assert r["status"] == "reused", r
    assert "没有重算" in r["why"]
    # ★ 指纹照样要记:没有它,这些场次每轮都被判成 `unrecorded` ⟹ 每次例行回填
    #   都把 22 场逐题缓存重算一遍(实测单场 ~34 秒,一场 8 段就是几分钟)。
    assert r["source_sig"], "reused 也必须带源指纹,否则 --pending 永远判 unrecorded"


def test_backfilled_result_records_a_source_fingerprint(tmp_path, monkeypatch):
    """补成功必须记源指纹 —— 没有它,下次 `--pending` 只会判成 `unrecorded`,
    每次例行回填都把 31 场全部重算一遍(实测单场 ~34 秒)。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    sid = "20260101_000000_6688"
    _session_dir(tmp_path, sid, with_camera=True)

    calls: list = []
    _fake_extract(monkeypatch, calls)
    r = backfill.backfill_one(sid)

    assert r["status"] == "backfilled"
    assert r["source_sig"] and r["source_sig"][0] == "whole"
    assert r["source_sig"][1][0][2] is not None, "指纹里必须有 mtime"
