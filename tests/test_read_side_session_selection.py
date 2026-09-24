# tests/test_read_side_session_selection.py
"""M2 读侧按 `session_id` 对齐。spec §5.1 / §5.2 / §5.4。

**地基(2026-09-24 真实前端使用实测,不是推断)**:加载器按「每模态文件名时间戳取最新」
把**三场不同的会话**拼在一起 —— 面/手来自一场、语音来自另一场(且是空文件)、
研究语音来自下午那场;空的那份把有数据的压掉 → 报告恒为「本次观测覆盖 0 / 20」。
"""

import csv
from pathlib import Path

from report_frontend.data_loader import LogDataLoader
from report_frontend.report_generator import sources_disclosure

_OLD = "20260924_132055"            # 时间戳**较小**
_NEW = "20260924_221811_9212"       # 时间戳**较大**


def _write(log_dir: Path, filename: str, rows: list) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / filename
    fields = ["session_id", "timestamp", "connective_density", "connective_density_std", "n_rows"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _voice_row(sid, density):
    return {"session_id": sid, "timestamp": "2026-09-24T13:20:55",
            "connective_density": density, "connective_density_std": 0.5, "n_rows": 1}


def _scene(tmp_path: Path, old_rows=2, new_rows=1) -> Path:
    """两场会话;刻意让**新的那场语音更少**,这样"按时间戳取最新"会取错。"""
    _write(tmp_path, f"interview_emotion_log_{_OLD}.csv",
           [_voice_row(_OLD, 3.0)] * old_rows)
    _write(tmp_path, f"interview_emotion_log_{_NEW}.csv",
           [_voice_row(_NEW, 1.0)] * new_rows)
    return tmp_path


# ── 1. 按 id 选,不按时间戳 ──────────────────────────────────────────────

def test_selection_is_by_session_id_not_by_newest_timestamp(tmp_path):
    """★ spec §5.1:目标 id 定了之后,每个模态只找**该 id** 的那份文件。

    场景刻意构造成"按时间戳取最新"会取错:目标 `_OLD`(时间戳较小)有 **2 行**语音,
    `_NEW`(时间戳较大)只有 **1 行**。断言指定 `_OLD` 时读进来的是 **2 行**那份。

    红法:把选取退回 `max(files, key=lambda x: x['timestamp_val'])` → 取到 `_NEW`,
    得到 1 行 → 红。
    """
    _scene(tmp_path)
    loader = LogDataLoader(str(tmp_path))
    data = loader.get_fused_latest_data(session_id=_OLD)

    assert "voice_interview" in data, "本测试的前提不成立:语音日志没被装载"
    assert len(data["voice_interview"]) == 2, (
        f"读到的不是目标会话那份 —— 行数 {len(data['voice_interview'])},期望 2"
        f"(说明按时间戳取了最新,而不是按 session_id)")


def test_selected_sessions_reports_the_status_of_every_modality(tmp_path):
    """★ spec §5.1 三态 + §5.2:选了但**读不出来**的模态不许静默消失。

    I4 的实现只记"真正读出来的"模态 —— 后果是语音整个从披露里消失,而报告还说
    「以上模态来自同一场会话」(spec §3.3 实测)。这里要求三态齐全。

    红法:退回"只记读到的" → `voice_interview` 不在 `selected_sessions` 里 → 红。
    """
    _scene(tmp_path)
    # 把本场的语音文件清空(只剩表头)—— 复刻真实故障里那份 292 字节的空日志
    (tmp_path / f"interview_emotion_log_{_NEW}.csv").write_text("session_id,\n", encoding="utf-8")
    # 再加一份本场的 face,让"部分模态读不到"这个形态成立
    _write(tmp_path, f"face_au_log_{_NEW}.csv",
           [{"session_id": _NEW, "timestamp": "2026-09-24T22:18:11",
             "connective_density": None, "connective_density_std": None, "n_rows": None}])

    loader = LogDataLoader(str(tmp_path))
    data = loader.get_fused_latest_data(session_id=_NEW)

    assert "voice_interview" not in data, "前提不成立:空文件居然加载进去了"
    st = loader.selected_sessions
    assert st.get("face", {}).get("status") == "loaded", st
    assert st.get("voice_interview", {}).get("status") == "unreadable", (
        f"读了但读不出来的模态必须留下痕迹,实际:{st}")
    assert st["voice_interview"]["session_id"] == _NEW, st


def test_disclosure_lists_unreadable_modalities_instead_of_omitting_them(tmp_path):
    """spec §5.2:披露要写「未读到数据」,而不是把该模态整行删掉。

    红法:退回 I4 的"只列 loaded" → 披露里没有语音那一项 → 红。
    """
    html = sources_disclosure({
        "face": {"session_id": _NEW, "status": "loaded", "rows": 1},
        "voice_interview": {"session_id": _NEW, "status": "unreadable", "rows": 0},
        "gesture": {"session_id": _NEW, "status": "missing", "rows": 0},
    })

    assert "未读到数据" in html, f"读不出来的模态没被点名:{html}"
    assert "缺失" in html, f"缺失的模态没被点名:{html}"
    assert _NEW in html, html


def test_disclosure_no_longer_claims_same_session(tmp_path):
    """spec §5.2:按 id 选之后**不存在"不同场"这个状态**,那句同场/不同场的话随之取消
    —— 真正要报的是"哪几个模态没进来、为什么"。

    红法:保留 I4 的"同一场会话"/"不同的会话"两态措辞 → 红。
    """
    html = sources_disclosure({"face": {"session_id": _NEW, "status": "loaded", "rows": 1}})

    assert "同一场会话" not in html and "不同的会话" not in html, f"旧的同场/不同场措辞还在:{html}"
    assert _NEW in html, "仍然要点名这是哪一场"


# ── 2. 缺省:取"最新一场",并写明是谁 ──────────────────────────────────

def test_default_target_is_the_newest_session_and_is_named(tmp_path):
    """spec §5.1/§5.2:不给 id 就取文件名时间戳最大的那一场,**并写明它是谁**。

    红法:缺省时返回 None / 返回 "NONE" / 报告头不写是哪一场。
    """
    _scene(tmp_path)
    loader = LogDataLoader(str(tmp_path))

    assert loader.resolve_target_session(None) == _NEW, "缺省没有取到最新那一场"
    assert loader.resolve_target_session(_OLD) == _OLD, "显式给了 id 却没照用"


def test_none_bucket_is_never_treated_as_a_session(tmp_path):
    """`NONE` 是"没有 id"的占位,不是一场会话。它必须能被**指出来**,但不能被当成目标。

    红法:把 NONE 也当合法 session_id → 只有 NONE 时会返回 "NONE" 而不是 None。
    """
    _write(tmp_path, f"interview_emotion_log_NONE_20260924_221800.csv", [_voice_row("NONE", 1.0)])
    loader = LogDataLoader(str(tmp_path))

    assert loader.resolve_target_session(None) is None, "只有 NONE 桶时不该当成一场会话"


def test_none_bucket_counts_the_form_without_a_trailing_segment(tmp_path):
    """★ 复审 I1:face/gesture 的 NONE 文件**没有尾段**(`face_au_log_NONE.csv`),
    而 glob 写的是 `*_log_NONE_*.csv` —— 实测盘上 224 行、数出来 **0**,
    于是"`NONE` 必须出现在披露里"这条在真实数据上完全不成立,
    而账本里那条 `RealtimeAnalysis` 裁决的论据正是依赖它。

    红法:glob 退回 `*_log_NONE_*.csv` → 只数到 1 行(带尾段那份)。
    """
    _scene(tmp_path)
    _write(tmp_path, "face_au_log_NONE.csv", [_voice_row("NONE", 1.0)])
    _write(tmp_path, "interview_emotion_log_NONE_20260924_221800.csv", [_voice_row("NONE", 2.0)])

    loader = LogDataLoader(str(tmp_path))
    loader.get_fused_latest_data(session_id=_NEW)

    assert loader.selected_sessions.get("none_bucket", {}).get("rows") == 2, (
        f"NONE 桶只数到 {loader.selected_sessions.get('none_bucket')} —— "
        f"无尾段那种形态被 glob 漏了")


def test_live_sources_uses_the_same_three_state_shape():
    """★ 复审 C1:`generate_report_live` 那条路还在传**旧形状**(`Dict[str, str]`),
    于是 `sources_disclosure` 里的 `info["status"]` 抛
    `TypeError: string indices must be integers` —— 而面板报 `success`。
    又一个静默失败,而且**全仓没有任何测试碰这条路径**。

    红法:让 `live_sources` 返回 `{k: session_id}`(旧形状)。
    """
    import pandas as pd

    from report_frontend.report_generator import live_sources

    data = {"face": pd.DataFrame({"a": [1, 2]}), "gesture": pd.DataFrame({"a": [1]})}
    sources = live_sources("20260924_221811_9212", data)

    assert sources["face"] == {"session_id": "20260924_221811_9212",
                               "status": "loaded", "rows": 2}, sources
    sources_disclosure(sources)      # 形状不对的话这里会 TypeError


def test_none_bucket_is_disclosed_when_the_target_session_is_loaded(tmp_path):
    """`NONE` 桶里的行**不进聚合**(既有不变量),但存在时必须让读者知道有帧没归入本场。

    红法:披露里完全不提 NONE。
    """
    _scene(tmp_path)
    _write(tmp_path, f"interview_emotion_log_NONE_20260924_221800.csv", [_voice_row("NONE", 9.9)])
    loader = LogDataLoader(str(tmp_path))
    loader.get_fused_latest_data(session_id=_NEW)

    html = sources_disclosure(loader.selected_sessions)
    assert "NONE" in html, f"存在 NONE 桶却没在披露里点出来:{html}"
