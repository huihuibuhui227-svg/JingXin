"""`media_truncation`(判定)+ `media_truncation_probe`(探针)的单测。

**不依赖盘上任何真实素材** —— 真实场次的数字全部硬编码(见 `REAL_CASES`),
所以这份测试在没装 ffmpeg/ffprobe、没有 `~/JingXin/recordings` 的机器上照样跑。

钉住的事:
  1. 三态判定与边界(`dur == tol*span` 是 `ok`,`dur == ceiling*span` 也是 `ok`);
  2. `overlong` 必须**先于** `ok` 判(超长场次的 dur 也满足 `>= tol*span`);
  3. "量不到"(无题目时间戳 / 跨度 ≤ 0 / 时长 None)⟹ `judgeable=False`,
     **不伪装成通过**,也不伪装成坏;
  4. 缺口码只映射到既有 9 类,**不许出现第 10 类**;
  5. 6 个真实场次的判定(含 1592 的两种读数:文件级 `ok` / 音轨级 `truncated`);
  6. 探针:缺 `questions.jsonl` → `no_ledger`;没有 ffprobe → 优雅降级为 None。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
for _c in (_HERE.parent / "patch", _HERE.parent, _HERE):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import media_truncation as mt  # noqa: E402
import media_truncation_probe as mp  # noqa: E402


# ── 工具 ────────────────────────────────────────────────────────────────

def qs(*pairs: tuple[float, float]) -> list[dict]:
    """`(ask_start, ask_end)` 对 → `questions.jsonl` 的行(字段名与真实文件一致)。"""
    return [{"qid": f"q{i}", "index": i, "ask_start": s, "ask_end": e,
             "source": "/session/question"}
            for i, (s, e) in enumerate(pairs)]


#: 6 个真实场次。数字是 2026-11-21 实测(`ffprobe` 的 packet pts 跨度),
#: 题目时间戳逐行抄自各自的 `questions.jsonl`。
#:
#: `media` 是**文件级**时长(`camera.webm` 取视轨,`assess` 的 `media_sec` 即最大值)。
REAL_CASES = {
    # 1592:8 道题,跨度 520.716s。⚠️ 视轨 550.514s、音轨只有 57.00s ——
    # 任务书里那句「整场录像只有 57.06 秒」说的是**音轨**,见
    # test_real_1592_audio_track_reading_is_truncated。
    "20260926_153854_1592": {
        "rows": [(1790408336.005, 1790408342.248),
                 (1790408572.850, 1790408576.270),
                 (1790408614.080, 1790408618.051),
                 (1790408676.205, 1790408680.481),
                 (1790408721.255, 1790408725.342),
                 (1790408762.606, 1790408767.511),
                 (1790408806.809, 1790408810.491),
                 (1790408853.616, 1790408856.721)],
        "media": {"camera.webm": 550.514},
        "span": 520.716, "media_sec": 550.514, "ratio": 1.057,
        "verdict": "ok", "gap": None,
    },
    # 0.92s 的录像对 6.24s 的跨度 —— 覆盖率 14.8%
    "20260925_225936_a971": {
        "rows": [(1790348376.454, 1790348382.694)],
        "media": {"camera.webm": 0.924},
        "span": 6.240, "media_sec": 0.924, "ratio": 0.148,
        "verdict": "truncated", "gap": "no_samples",
    },
    # 3.43s / 5.41s = 63%
    "20260928_230839_87c9": {
        "rows": [(1790608119.825, 1790608125.239)],
        "media": {"camera.webm": 3.427},
        "span": 5.414, "media_sec": 3.427, "ratio": 0.633,
        "verdict": "truncated", "gap": "no_samples",
    },
    # 反向:录像 284s,台账只有 1 道题(跨度 6.2s)
    "20260928_165520_90e7": {
        "rows": [(1790585724.180, 1790585730.407)],
        "media": {"camera.webm": 284.369,
                  "audio/0001.webm": 10.080,
                  "audio/0001_converted.wav": 10.112},
        "span": 6.227, "media_sec": 284.369, "ratio": 45.667,
        "verdict": "overlong", "gap": "time_unusable",
    },
    "20260927_191325_253a": {
        "rows": [(1790507607.036, 1790507613.266)],
        "media": {"camera.webm": 189.550,
                  "audio/0001.webm": 4.979,
                  "audio/0001_converted.wav": 4.992},
        "span": 6.230, "media_sec": 189.550, "ratio": 30.425,
        "verdict": "overlong", "gap": "time_unusable",
    },
    "20260927_211150_4579": {
        "rows": [(1790514711.403, 1790514716.811)],
        "media": {"camera.webm": 83.292},
        "span": 5.408, "media_sec": 83.292, "ratio": 15.402,
        "verdict": "overlong", "gap": "time_unusable",
    },
}


# ── 基本三态 ────────────────────────────────────────────────────────────

def test_ok_when_media_covers_the_span():
    r = mt.assess(qs((100.0, 110.0), (200.0, 220.0)), {"camera.webm": 120.0})
    assert r["verdict"] == "ok" and r["gap"] is None
    assert r["span_sec"] == 120.0 and r["media_sec"] == 120.0
    assert r["ratio"] == 1.0 and r["judgeable"] is True
    assert "覆盖" in r["gap_detail"]


def test_truncated_when_media_is_short_maps_to_no_samples():
    r = mt.assess(qs((100.0, 110.0), (300.0, 320.0)), {"camera.webm": 60.0})
    assert r["verdict"] == "truncated"
    assert r["gap"] == "no_samples", "截断=样本不完整,必须映射到既有码"
    assert r["judgeable"] is True
    assert r["ratio"] == pytest.approx(0.273, abs=0.002)
    # 详情要说清要多少秒、现在只有多少秒,而不是只丢一个码
    assert "220" in r["gap_detail"] and "60" in r["gap_detail"]


def test_overlong_when_media_is_far_longer_maps_to_time_unusable():
    r = mt.assess(qs((100.0, 106.0)), {"camera.webm": 284.0})
    assert r["verdict"] == "overlong"
    assert r["gap"] == "time_unusable", "台账与时间轴对不上 ⟹ time_unusable"
    assert r["judgeable"] is True
    assert "台账缺题" in r["gap_detail"]


def test_overlong_is_checked_before_ok():
    """`dur > ceiling*span` 的场次**也**满足 `dur >= tol*span` —— 先判 ok 会漏。

    这条不是假想的:90e7 / 253a / 4579 三场都是"录像远长于跨度",
    只要分支顺序反了,它们会全部读成 `ok`。
    """
    r = mt.assess(qs((100.0, 106.0)), {"camera.webm": 100.0})
    assert r["media_sec"] >= mt.DEFAULT_TOL * r["span_sec"]  # 同时满足 ok 的条件
    assert r["verdict"] == "overlong", "超长必须先判,不能被 ok 抢走"


def test_output_contract():
    """字段名与类型是调用方契约(`capture_health` 那一套也是这么钉的)。"""
    r = mt.assess(qs((0.0, 10.0)), {"a.webm": 9.5})
    for k in ("verdict", "span_sec", "media_sec", "ratio", "gap", "gap_detail",
              "evidence", "judgeable"):
        assert k in r, f"缺字段 {k}"
    assert r["verdict"] in mt.VERDICTS
    assert isinstance(r["evidence"], dict)
    assert isinstance(r["gap_detail"], str) and r["gap_detail"]
    assert r["evidence"]["tol"] == mt.DEFAULT_TOL
    assert r["evidence"]["media_sec_by_file"] == {"a.webm": 9.5}


# ── 跨度的算法 ──────────────────────────────────────────────────────────

def test_span_is_max_end_minus_min_start():
    """两列**分别**取极值 —— 上游偶有 `ask_end` 缺行的题。"""
    rows = qs((500.0, 510.0), (600.0, 640.0), (700.0, 720.0))
    rows.append({"qid": "broken", "ask_start": 800.0})  # 只有 ask_start
    r = mt.assess(rows, {"c": 720.0})
    assert r["span_sec"] == 220.0, "max(ask_end)=720 − min(ask_start)=500(800 那条只有 start)"
    assert r["evidence"]["n_questions"] == 4
    assert r["evidence"]["n_ask_end"] == 3


def test_non_numeric_timestamps_are_ignored_not_treated_as_zero():
    """`null` 当 0 会造出一个 1790408856 秒的假跨度 —— 那比漏报更糟。"""
    rows = qs((1000.0, 1010.0)) + [{"ask_start": None, "ask_end": None},
                                   {"ask_start": "abc", "ask_end": ""},
                                   {"ask_start": True, "ask_end": False}]
    r = mt.assess(rows, {"c": 10.0})
    assert r["span_sec"] == 10.0 and r["verdict"] == "ok"


def test_string_timestamps_are_accepted():
    r = mt.assess([{"ask_start": "1000.5", "ask_end": "1010.5"}], {"c": 10.0})
    assert r["span_sec"] == 10.0 and r["verdict"] == "ok"


# ── "量不到"的三种输入:不伪装通过,也不伪装坏 ──────────────────────────

def test_no_ask_timestamps_is_not_judgeable():
    r = mt.assess([], {"camera.webm": 100.0})
    assert r["judgeable"] is False
    assert r["verdict"] == "ok" and r["gap"] is None and r["ratio"] is None
    assert "不是通过" in r["gap_detail"]


def test_zero_span_is_not_judgeable():
    """一题、或所有题时间戳相同 ⟹ 跨度 0。**不判**超长(`dur > 0` 恒真)。"""
    one = mt.assess(qs((100.0, 100.0)), {"camera.webm": 600.0})
    assert one["span_sec"] == 0.0
    assert one["verdict"] == "ok" and one["judgeable"] is False
    assert one["ratio"] is None and one["gap"] is None
    assert "跨度 ≤ 0" in one["gap_detail"]


def test_negative_span_is_not_judgeable():
    """`ask_end < ask_start`(时钟回拨过的场次)也不许被读成截断。"""
    r = mt.assess(qs((200.0, 100.0)), {"camera.webm": 10.0})
    assert r["span_sec"] < 0
    assert r["verdict"] == "ok" and r["judgeable"] is False


def test_all_media_durations_none_is_not_judgeable():
    r = mt.assess(qs((100.0, 200.0)), {"camera.webm": None, "audio/1.webm": None})
    assert r["media_sec"] is None and r["ratio"] is None
    assert r["judgeable"] is False and r["verdict"] == "ok" and r["gap"] is None
    assert "一份媒体都没量到" in r["gap_detail"]


def test_none_entries_do_not_poison_the_max():
    """一份量不到**不许**被当 0 —— 那会把正常的录像判成截断。"""
    r = mt.assess(qs((100.0, 200.0)), {"camera.webm": 95.0, "audio/1.webm": None})
    assert r["media_sec"] == 95.0 and r["verdict"] == "ok"
    assert r["evidence"]["media_sec_by_file"]["audio/1.webm"] is None
    assert r["evidence"]["longest_media"] == "camera.webm"


def test_nan_and_negative_are_ignored():
    r = mt.assess(qs((100.0, 200.0)),
                  {"bad": float("nan"), "bad2": -5.0, "camera.webm": 95.0})
    assert r["media_sec"] == 95.0
    assert r["evidence"]["media_sec_by_file"]["bad"] is None


def test_empty_media_dict_is_not_judgeable():
    r = mt.assess(qs((100.0, 200.0)), {})
    assert r["judgeable"] is False and r["media_sec"] is None


# ── tol / ceiling 边界 ──────────────────────────────────────────────────

@pytest.mark.parametrize("dur, expect", [
    (9.0, "ok"),          # dur == tol*span ⟹ 通过(含等号)
    (8.9999, "truncated"),
    (5.0, "truncated"),
    (10.0, "ok"),
    (15.0, "ok"),         # dur == ceiling*span ⟹ 通过(ceiling 是开区间上界)
    (15.0001, "overlong"),
])
def test_tol_ceiling_boundaries(dur, expect):
    r = mt.assess(qs((0.0, 10.0)), {"c": dur})  # span = 10
    assert r["verdict"] == expect, f"dur={dur}"


def test_custom_tol_and_ceiling():
    rows, media = qs((0.0, 100.0)), {"c": 70.0}
    assert mt.assess(rows, media)["verdict"] == "truncated"          # 0.9 默认
    assert mt.assess(rows, media, tol=0.7)["verdict"] == "ok"         # 放宽到 0.7
    assert mt.assess(rows, media, tol=0.8, ceiling=0.6)["verdict"] == "overlong"
    assert mt.assess(rows, media, ceiling=1.0)["evidence"]["ceiling"] == 1.0


def test_rejects_bad_tol():
    with pytest.raises(ValueError):
        mt.assess(qs((0.0, 10.0)), {"c": 9.0}, tol=-1)
    with pytest.raises(ValueError):
        mt.assess(qs((0.0, 10.0)), {"c": 9.0}, ceiling=float("nan"))


def test_ratio_is_media_over_span_and_the_task_number_is_its_reciprocal():
    """任务书里的 `9.13` 是 `span/media`;`ratio` 存的是 `media/span`。

    两个数都留在结果里,免得读者拿 9.13 去比 `ratio` 然后以为判据反了。
    """
    r = mt.assess(qs((0.0, 520.716)), {"camera.webm": 57.0})
    assert r["ratio"] == pytest.approx(0.109, abs=0.001)
    assert r["evidence"]["span_over_media"] == pytest.approx(9.135, abs=0.01)


# ── 缺口码:只用既有 9 类 ────────────────────────────────────────────────

def test_gap_codes_are_the_existing_nine():
    """★ 本模块**不许**新增第 10 类缺口码。

    真去 `report_frontend.l1_indicators` 取一遍 `KINDS`(只读 import);
    取不到就跳过 —— 但取到了就必须对得上,否则对外口径就裂了。
    """
    try:
        sys.path.insert(0, str(Path.home() / "JingXin/jingxin"))
        from report_frontend import l1_indicators as li
    except Exception:  # noqa: BLE001 - 仓库不在就跳过,不算失败
        pytest.skip("拿不到 report_frontend.l1_indicators(仓库不在本机)")
    assert len(li.KINDS) == 9, f"系统口径变了:{li.KINDS}"
    assert mt.GAP_NO_SAMPLES in li.KINDS
    assert mt.GAP_TIME_UNUSABLE in li.KINDS
    assert set(mt.GAP_FOR_VERDICT.values()) - {None} <= set(li.KINDS)


def test_every_verdict_maps_into_the_nine():
    assert set(mt.GAP_FOR_VERDICT) == set(mt.VERDICTS)
    produced = set()
    for dur in (0.1, 5.0, 9.0, 15.0, 500.0):
        produced.add(mt.assess(qs((0.0, 10.0)), {"c": dur})["gap"])
    assert produced == {None, "no_samples", "time_unusable"}


# ── 6 个真实场次(硬编码,离线可跑) ───────────────────────────────────────

@pytest.mark.parametrize("sid", sorted(REAL_CASES))
def test_real_cases(sid):
    case = REAL_CASES[sid]
    r = mt.assess(qs(*case["rows"]), case["media"])
    assert r["verdict"] == case["verdict"], f"{sid}: {r['gap_detail']}"
    assert r["gap"] == case["gap"]
    assert r["span_sec"] == pytest.approx(case["span"], abs=0.002)
    assert r["media_sec"] == pytest.approx(case["media_sec"], abs=0.002)
    assert r["ratio"] == pytest.approx(case["ratio"], abs=0.01)
    assert r["judgeable"] is True


def test_real_1592_audio_track_reading_is_truncated():
    """1592 的**音轨**只有 57.00s,而跨度 520.716s ⟹ 按音轨算就是截断(9.13 倍)。

    任务书里「整场录像只有 57.06 秒」这个数是音轨,不是录像:同一个
    `camera.webm` 的视轨实测 550.514s(见 `test_real_cases`),所以**文件级**
    不判截断。两条都留在这里,免得其中一个读数被当成全部事实。
    """
    case = REAL_CASES["20260926_153854_1592"]
    r = mt.assess(qs(*case["rows"]), {"camera.webm": 57.0})  # 音轨读数
    assert r["verdict"] == "truncated" and r["gap"] == "no_samples"
    assert r["evidence"]["span_over_media"] == pytest.approx(9.13, abs=0.02)


def test_real_cases_are_not_judged_by_one_rule_only():
    """6 场里三态都得出现 —— 全判同一类的模块等于没判。"""
    verdicts = {mt.assess(qs(*c["rows"]), c["media"])["verdict"]
                for c in REAL_CASES.values()}
    assert verdicts == set(mt.VERDICTS)


# ── 探针:读盘与降级 ────────────────────────────────────────────────────

def make_session(tmp_path: Path, rows: list[dict] | None,
                 camera: bytes | None = b"x") -> Path:
    d = tmp_path / "20260101_000000_test"
    d.mkdir(parents=True)
    if rows is not None:
        (d / "questions.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8")
    if camera is not None:
        (d / "media").mkdir()
        (d / "media" / "camera.webm").write_bytes(camera)
    return d


def test_probe_missing_questions_jsonl_is_no_ledger(tmp_path):
    d = make_session(tmp_path, rows=None)
    r = mp.probe_session(d)
    assert r["verdict"] == "no_ledger"
    assert r["gap"] is None and r["judgeable"] is False
    assert "questions.jsonl" in r["gap_detail"]


def test_probe_empty_questions_jsonl_is_no_ledger(tmp_path):
    d = make_session(tmp_path, rows=[])
    r = mp.probe_session(d)
    assert r["verdict"] == "no_ledger"


def test_probe_degrades_when_ffprobe_is_missing(tmp_path, monkeypatch):
    """★ 没有 ffprobe:时长全 None、不伪造通过、也不说"文件坏了"。"""
    monkeypatch.setattr(mp, "_ffprobe", lambda: None)
    d = make_session(tmp_path, rows=qs((100.0, 110.0)))
    r = mp.probe_session(d)
    assert r["verdict"] == "ok" and r["judgeable"] is False
    assert r["media_sec"] is None
    assert r["evidence"]["ffprobe"] is None
    assert "ffprobe 不在 PATH" in r["evidence"]["probe_note"]


def test_probe_uses_the_assess_result(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(mp, "_span_sec",
                        lambda fp, path, st: 12.0 if st == "v:0" else None)
    d = make_session(tmp_path, rows=qs((100.0, 130.0)))
    r = mp.probe_session(d)
    assert r["span_sec"] == 30.0 and r["media_sec"] == 12.0
    assert r["verdict"] == "truncated" and r["gap"] == "no_samples"
    assert r["evidence"]["per_file"]["camera.webm"]["tracks"]["v:0"] == 12.0


def test_probe_auto_track_prefers_video_then_falls_back_to_audio(tmp_path, monkeypatch):
    """视轨有就用视轨;没有视轨(音频-only 场次)就退音轨,**不许**记成 0。"""
    monkeypatch.setattr(mp, "_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(mp, "_span_sec",
                        lambda fp, path, st: 550.514 if st == "v:0" else 57.0)
    d = make_session(tmp_path, rows=qs((100.0, 620.716)))
    r = mp.probe_session(d)
    assert r["media_sec"] == 550.514 and r["verdict"] == "ok"
    assert r["evidence"]["per_file"]["camera.webm"]["av_ratio"] == \
        pytest.approx(0.104, abs=0.001)

    monkeypatch.setattr(mp, "_span_sec",
                        lambda fp, path, st: None if st == "v:0" else 57.0)
    r2 = mp.probe_session(d)
    assert r2["media_sec"] == 57.0, "没有视轨 ⟹ 退音轨,而不是判 0 秒"


def test_probe_audio_track_mode_reproduces_the_9_13_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(mp, "_span_sec",
                        lambda fp, path, st: 550.514 if st == "v:0" else 57.0)
    d = make_session(tmp_path, rows=qs((100.0, 620.716)))
    assert mp.probe_session(d, track="auto")["verdict"] == "ok"
    r = mp.probe_session(d, track="audio")
    assert r["verdict"] == "truncated"
    assert r["evidence"]["span_over_media"] == pytest.approx(9.135, abs=0.02)
    rv = mp.probe_session(d, track="video")
    assert rv["media_sec"] == 550.514


def test_probe_rejects_unknown_track(tmp_path):
    d = make_session(tmp_path, rows=qs((100.0, 110.0)))
    with pytest.raises(ValueError):
        mp.probe_session(d, track="both")


def test_probe_skips_bad_lines_without_losing_the_session(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(mp, "_span_sec", lambda fp, path, st: 4.6)
    d = make_session(tmp_path, rows=None)
    (d / "questions.jsonl").write_text(
        '{"ask_start": 100.0, "ask_end": 105.0}\n'
        "这不是 JSON\n"
        "\n"
        '[1,2,3]\n'                       # 不是 dict → 跳过
        '{"ask_start": null, "ask_end": null}\n',
        encoding="utf-8")
    r = mp.probe_session(d)
    assert r["span_sec"] == 5.0 and r["verdict"] == "ok"
    assert r["evidence"]["n_questions"] == 2


def test_probe_without_any_media_file(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_ffprobe", lambda: "/usr/bin/ffprobe")
    d = make_session(tmp_path, rows=qs((100.0, 110.0)), camera=None)
    r = mp.probe_session(d)
    assert r["judgeable"] is False and r["media_sec"] is None
    assert "没有任何 media 文件" in r["evidence"]["probe_note"]


def test_probe_flags_a_session_without_camera(tmp_path, monkeypatch):
    """没有 `camera.webm`(音频-only / 只有分题音频块)必须标出来。

    这种场次里 `media_sec` 是**最长单块**,不是整场连续录制。实测
    `..._0d89`(跨度 288.9s)、`..._302f`(125.4s) 就是这种 —— 它们真的没有整段
    录像,但那是设计如此,不是"录像被截断"。不标这一句就是一条假 warn。
    """
    monkeypatch.setattr(mp, "_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(mp, "_span_sec", lambda fp, path, st: 16.128)
    d = make_session(tmp_path, rows=qs((100.0, 388.9)), camera=None)
    (d / "media" / "audio").mkdir(parents=True)
    (d / "media" / "audio" / "0001.webm").write_bytes(b"x")
    r = mp.probe_session(d)
    assert r["evidence"]["has_camera"] is False
    assert "最长**单块**" in r["evidence"]["probe_note"]
    assert r["verdict"] == "truncated"  # 判定不变,只是解读要带上那句话


def test_probe_does_not_write_anything(tmp_path, monkeypatch):
    """★ 探针是只读的:跑完目录里的文件一个都不许多、一个字节都不许变。"""
    monkeypatch.setattr(mp, "_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(mp, "_span_sec", lambda fp, path, st: 447.0)
    d = make_session(tmp_path, rows=qs((100.0, 600.0)))
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
              for p in d.rglob("*") if p.is_file()}
    mp.probe_session(d)
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
             for p in d.rglob("*") if p.is_file()}
    assert before == after


def test_cli_prints_one_json_line_per_session(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mp, "_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(mp, "_span_sec", lambda fp, path, st: 5.0)
    d1 = make_session(tmp_path, rows=qs((100.0, 110.0)))
    d2 = tmp_path / "20260101_000001_noledger"
    d2.mkdir()
    rc = mp.main([str(d1), str(d2)])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert rc == 0 and len(lines) == 2
    a, b = (json.loads(x) for x in lines)
    assert a["verdict"] == "truncated" and a["session"] == d1.name
    assert b["verdict"] == "no_ledger"


def test_cli_reports_missing_dir_without_crashing(tmp_path, capsys):
    rc = mp.main([str(tmp_path / "nope")])
    out = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
    assert rc == 2 and out[0]["verdict"] == "no_session"
