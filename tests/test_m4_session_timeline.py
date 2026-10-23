# tests/test_m4_session_timeline.py
"""**M4.0 统一会话时间轴**的守卫。

## 这条测试钉的是什么(以及为什么值得钉)

2026-10-18 建 M4 时,对着真账本核出一个**此前没登记的缺陷**:

    账本里的 `received_at_wall` 取自 `time.time()`,它**会倒退**;
    `declared_ts` 取自 `SessionClock`(`time.monotonic`),物理上不可能倒退。

全语料 34 场有 face 账本的场次里,3 场墙钟倒退(09-26 那三场,累计 -14 s),
1 场墙钟跑快一倍(09-28 `1320`:墙钟 2048 s vs 单调 1070 s),其余 30 场无碍。

⟹ 时间轴的主时基必须是 `declared_ts`。墙钟只能用来"求相位",而且**必须先体检**。

本文件钉四件事:
1. 墙钟倒退的场次 ⟹ 状态是 `WALL_UNRELIABLE`,且**仍然给出可用时间轴**(用 declared);
2. 干净的场次 ⟹ `OK`,且时间轴与 declared 逐帧一致;
3. 账本与日志行数对不上 ⟹ `UNRECONSTRUCTIBLE`,**不按序硬配**;
4. 没有提问台账 ⟹ `NO_LEDGER`,整场不建轴(戊-15 那 26 场自然对话就是这一档)。

## 不依赖 `data/`

全部用合成账本(临时目录 + `JINGXIN_RECORDINGS_DIR`),不读任何真素材。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import media_retention
import session_meta
from report_frontend import session_timeline as st

SID = "20261018_120000_ab12"
ORIGIN = 1_790_400_000.0            # 会话原点(墙钟)


# ── 夹具 ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def rec_dir(tmp_path, monkeypatch):
    """把录制根指到临时目录 —— `media_retention.root()` 每次调用都读环境变量。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    return tmp_path


def _write_ledger(root: Path, writer: str, rows: list[dict]) -> None:
    d = root / SID
    d.mkdir(parents=True, exist_ok=True)
    (d / f"retention.{writer}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _write_questions(root: Path, starts: list[float]) -> None:
    d = root / SID
    d.mkdir(parents=True, exist_ok=True)
    (d / "questions.jsonl").write_text(
        "\n".join(json.dumps({
            "qid": f"q{i}", "index": i,
            "ask_start": s, "ask_end": s + 3.0,
            "source": "/session/question", "session_id": SID,
        }) for i, s in enumerate(starts)) + "\n", encoding="utf-8")


def _frame_rows(declared_sec: np.ndarray, wall_sec: np.ndarray, modality: str = "face"):
    return [{
        "kind": "frame", "modality": modality, "seq": i + 1,
        "file": f"media/{modality}/{i + 1:06d}.jpg", "bytes": 10, "sha256": "x",
        "received_at_wall": float(wall_sec[i]),
        "declared_ts": int(round(float(declared_sec[i]) * 1000)),
        "source_endpoint": "/analyze", "session_id": SID,
    } for i in range(len(declared_sec))]


def _log_df(n: int) -> pd.DataFrame:
    """最简日志:只要一行一帧、带 timestamp 列即可(时间列本身不参与换算)。"""
    return pd.DataFrame({"timestamp": np.arange(n, dtype=float)})


# ── 1. 墙钟倒退 ⟹ WALL_UNRELIABLE,但模态内仍可用 ──────────────────────────

def test_backward_wall_clock_is_flagged_but_declared_still_used(rec_dir):
    """★ 核心守卫:墙钟倒退 17 次、每 30 帧一次,时间轴必须改用 declared。

    红法:`align_frame_modality` 改回用 `received_at_wall` 做主时基 ⟹ 时间轴出现
    负增量,`np.all(np.diff(t) >= 0)` 那条断言直接红。
    """
    _write_questions(rec_dir, [ORIGIN])
    n = 96
    declared = np.arange(n) * 1.07                      # 单调,~1.07 s/帧
    wall = ORIGIN + declared.copy()
    # 每 30 帧回拨一次,幅度必须 **大于一帧间隔** 才真的构成"倒退":
    # 实测 09-26 那三场就是"墙钟净倒退 ~0.8 s / declared 照常 +1.07 s"。
    for k in (30, 60, 90):
        wall[k:] -= 2.0 * (k // 30)
    _write_ledger(rec_dir, "face", _frame_rows(declared, wall))

    tb = st.align_frame_modality(SID, "face", _log_df(n), ORIGIN)

    assert tb.status is st.TimebaseStatus.WALL_UNRELIABLE, tb.status
    assert tb.wall_back_steps == 3, tb.wall_back_steps
    assert tb.usable, "墙钟坏掉不等于这一列不能用 —— 模态内用 declared"
    assert tb.gap and "墙钟" in tb.gap

    t = tb.t_sec.to_numpy()
    assert np.all(np.diff(t) > 0), "declared 单调 ⟹ 时间轴必须也单调(墙钟倒退不能传染)"
    # 模态内相邻间隔就是 declared 的间隔,与墙钟无关
    assert np.allclose(np.diff(t), 1.07, atol=1e-6)


# ── 2. 干净的场次 ⟹ OK,且与 declared 逐帧一致 ────────────────────────────

def test_clean_session_is_ok_and_matches_declared(rec_dir):
    """红法:把 `WALL_RATE_TOLERANCE` 调到 0 ⟹ 采样抖动被误判成不可信(`OK` 变 `WALL_UNRELIABLE`)。"""
    _write_questions(rec_dir, [ORIGIN])
    n = 40
    declared = np.arange(n) * 1.0
    # 极小的采样抖动:1e-3 秒(远小于 5% 容差)
    wall = ORIGIN + declared + np.linspace(0, 1e-3, n)
    _write_ledger(rec_dir, "face", _frame_rows(declared, wall))

    tb = st.align_frame_modality(SID, "face", _log_df(n), ORIGIN)

    assert tb.status is st.TimebaseStatus.OK, (tb.status, tb.gap)
    assert tb.wall_back_steps == 0
    t = tb.t_sec.to_numpy()
    assert np.allclose(t, declared, atol=1e-3)


# ── 3. 行数对不上 ⟹ 不硬配 ────────────────────────────────────────────────

def test_row_count_mismatch_is_unreconstructible_not_zipped(rec_dir):
    """★ 账本 100 帧、日志 90 行 ⟹ 必须拒绝,而不是按序切前 90 条。

    红法:把 `len(ledger) != n` 那个判断去掉、改成 `min(len, n)` ⟹ 本条红。
    """
    _write_questions(rec_dir, [ORIGIN])
    n = 100
    declared = np.arange(n) * 1.0
    _write_ledger(rec_dir, "face", _frame_rows(declared, ORIGIN + declared))

    tb = st.align_frame_modality(SID, "face", _log_df(90), ORIGIN)

    assert tb.status is st.TimebaseStatus.UNRECONSTRUCTIBLE, tb.status
    assert not tb.usable
    assert tb.t_sec.isna().all(), "拒绝重建时不许留下一半像样的数"
    assert tb.gap and "对不上" in tb.gap


# ── 4. 没有提问台账 ⟹ 不建轴 ──────────────────────────────────────────────

def test_no_question_ledger_means_no_timeline(rec_dir):
    """戊-15 那 26 场自然对话的形态:没有题库台账,原点无从定义。

    红法:把 `session_origin_wall` 的"没有台账"分支改成回退到首帧墙钟 ⟹ 本条红
    (那等于把相机开机时刻冒充成"会话开始",而它俩先后不定)。
    """
    rec_dir.joinpath(SID).mkdir(parents=True, exist_ok=True)
    declared = np.arange(10) * 1.0
    _write_ledger(rec_dir, "face", _frame_rows(declared, ORIGIN + declared))

    origin, gap = st.session_origin_wall(SID)
    assert origin is None and gap and "提问台账" in gap

    tl = st.build_timeline(SID, logs={"face": _log_df(10)})
    assert tl.status is st.TimebaseStatus.NO_LEDGER, tl.status
    assert not tl.usable


# ── 5. 会话级汇总:一个模态墙钟坏 ⟹ 整场跨模态相位带误差 ──────────────────

def test_session_level_status_propagates_wall_unreliable(rec_dir):
    """语音没有可交叉验证的第二根时钟 ⟹ 墙钟坏不坏必须由别的模态汇总上来。

    红法:把 `build_timeline` 末尾那段汇总去掉 ⟹ 会话状态停在 `OK`,本条红。
    """
    _write_questions(rec_dir, [ORIGIN])
    n = 60
    declared = np.arange(n) * 1.0
    wall = ORIGIN + declared.copy()
    wall[30:] -= 2.5                                    # 一次倒退(幅度大于一帧间隔)
    _write_ledger(rec_dir, "face", _frame_rows(declared, wall))

    voice = pd.DataFrame({"unix_timestamp": ORIGIN + np.arange(8) * 5.0})
    tl = st.build_timeline(SID, logs={"face": _log_df(n), "voice": voice})

    assert tl.status is st.TimebaseStatus.WALL_UNRELIABLE, tl.status
    assert tl.gap and "墙钟" in tl.gap
    # 语音自身没有可核对的时钟,单看它总是 OK —— 这正是要汇总的理由
    assert tl.modalities["voice"].status is st.TimebaseStatus.OK


# ── 6. 碎行不崩 ───────────────────────────────────────────────────────────

def test_torn_ledger_line_is_skipped_not_fatal(rec_dir):
    """账本是 O_APPEND 单次写,但进程被杀仍可能留半行(spec §6)。读侧必须跳过硬扛。"""
    _write_questions(rec_dir, [ORIGIN])
    declared = np.arange(20) * 1.0
    d = rec_dir / SID
    d.mkdir(parents=True, exist_ok=True)
    rows = _frame_rows(declared, ORIGIN + declared)
    body = "\n".join(json.dumps(r) for r in rows[:10])
    body += '\n{"kind": "frame", "seq": 11, "declared_ts":'        # 半行(seq 11 丢失)
    body += "\n" + "\n".join(json.dumps(r) for r in rows[11:])
    (d / "retention.face.jsonl").write_text(body + "\n", encoding="utf-8")

    tb = st.align_frame_modality(SID, "face", _log_df(20), ORIGIN)
    # 半行被跳过 ⟹ 账本只剩 19 行,与日志 20 行对不上 ⟹ 拒绝重建(不静默)
    assert tb.status is st.TimebaseStatus.UNRECONSTRUCTIBLE
    assert tb.t_sec.isna().all()


# ── 7. 没有模态时不许假装"一切正常" ──────────────────────────────────────

def test_timeline_with_no_modalities_says_so(rec_dir):
    """原点成立但一个模态都没上来 ⟹ `gap` 必须说清楚。

    红法:把 `build_timeline` 末尾那段 `if not timeline.modalities` 删掉 ⟹ 本条红
    (状态是 ok、gap 是 None,读的人会以为这一场有数据)。
    """
    _write_questions(rec_dir, [ORIGIN])
    tl = st.build_timeline(SID, logs={})
    assert tl.status is st.TimebaseStatus.OK, "轴本身成立"
    assert tl.gap and "没有任何模态" in tl.gap


# ── 8. 显式开启时,没有台账也能建轴(正式场)────────────────────────────

def test_frame_origin_fallback_only_when_explicitly_allowed(rec_dir):
    """★ 默认**不**回退;调用方显式要求时才用首个留存帧的墙钟当原点。

    ## 为什么要有这个开关(2026-10-18 实测)

    正式场(24 场实名场)**全都没有 `questions.jsonl`**。按默认口径,原点恒为空
    ⟹ `compute_session_l1` 在开头就 return,**整场一条指标都不出** ——
    哪怕面部覆盖 98%、手势 100%、整场音频都在。实测 `c9f0` 就是这样从 0 条变 18 条的。

    但上面 `test_no_question_ledger_means_no_timeline` 记录的那条裁定
    (「首帧墙钟不等于会话开始」)仍然成立,不能因为多了个能力就推翻它。
    ⟹ 折中:**默认不变,调用方显式开启**,并在 `gap` 里写明"会话级粒度"。

    红法:把 `allow_frame_origin` 的默认值改成 True ⟹ 上面那条测试红;
    把这一支整个删掉 ⟹ 本条红。
    """
    rec_dir.joinpath(SID).mkdir(parents=True, exist_ok=True)
    declared = np.arange(10) * 1.0
    _write_ledger(rec_dir, "face", _frame_rows(declared, ORIGIN + declared))

    origin, gap = st.session_origin_wall(SID, allow_frame_origin=True)
    assert origin == pytest.approx(ORIGIN), origin
    assert gap and "提问台账" in gap and "会话级" in gap

    tl = st.build_timeline(SID, logs={"face": _log_df(10)}, allow_frame_origin=True)
    assert tl.usable, tl.status
    assert "face" in tl.modalities


def test_span_sec_measures_the_covered_range_not_the_frame_count():
    """★ `span_sec` 必须覆盖**整段范围**,与 `n_frames` 是两个量。

    ## 为什么必须有这个字段

    判断"两族看的是不是同一段时间"**不能比帧数**:
    候选B那场 1 Hz 的 6968 帧跨 7427 秒,而语音 31.25 Hz 的 13585 帧只跨 434.7 秒。
    只看 `n_frames`(6968 vs 13585)甚至会觉得语音"看得更多",实际语音只覆盖最后 7 分钟。
    比 `span_sec` 才能一眼看出 7427 vs 434.7。

    红法:删掉 `Timebase.__post_init__` ⟹ `span_sec` 恒为 None ⟹ 本条红。
    """
    from report_frontend.session_timeline import Timebase, TimebaseStatus

    # 100 帧、1 Hz,中间有一处 60 秒空档 ⟹ 覆盖 159 秒,而不是 99 秒。
    t = list(range(50)) + [i + 50 + 60 for i in range(50)]
    tb = Timebase("face", TimebaseStatus.OK, pd.Series(t, dtype=float),
                  n_frames=len(t))
    assert tb.n_frames == 100
    assert tb.span_sec == pytest.approx(159.0)

    # 显式给值时不覆盖。
    tb2 = Timebase("voice", TimebaseStatus.OK, pd.Series([0.0, 1.0]),
                   n_frames=2, span_sec=7.5)
    assert tb2.span_sec == 7.5
