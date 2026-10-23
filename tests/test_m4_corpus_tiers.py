# tests/test_m4_corpus_tiers.py
"""**有效素材分级**的守卫。

## 钉的是什么

1. 档位是**纯函数**判出来的(`classify`),只看三个事实:是否正式场、脸覆盖、臂覆盖;
2. **"正式场"的判据是目录名里的实名+学号**,不是"在 recordings 目录里";
3. **没有面部日志 ≠ 面部 0%**:前者判不出(`?`),后者是确定的事实(`D` 或 `B`);
4. 覆盖率按**真正带数据的行**算,不是按行数 —— 实测面部日志 25.5% 的行整行为空。

## 为什么这条分级重要

实测 41 个录制目录里混着 17 场无名测试场、4 场正式但不可用、13 场全面可用。
拿 `3d8f`(脸 0% / 臂 0%)去出报告,16 条 F 指标全是 `n_valid=0`,
而报告只会说"面部已提供"。分级就是**在出报告之前**把这件事挡住的机制。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from report_frontend import corpus


# ── 1. 纯函数:档位只由事实决定 ─────────────────────────────────────────

@pytest.mark.parametrize("face,arm,expect", [
    (95.0, 90.0, "A"),    # 两样都达标
    (95.0, 50.0, "B"),    # 只有脸
    (50.0, 90.0, "C"),    # 只有臂
    (50.0, 50.0, "D"),    # 都不达标
    (80.0, 70.0, "A"),    # ★ 边界:恰好等于阈值 ⟹ 达标(>= 不是 >)
    (79.9, 70.0, "C"),    # ★ 边界:脸差 0.1 掉出 A;臂仍达标 ⟹ 落到"仅手势"
    (79.9, 69.9, "D"),    # ★ 边界:两样都差一点 ⟹ 全不达标
])
def test_classify_uses_only_the_three_facts(face, arm, expect):
    """红法:把 `>=` 改成 `>`,边界两例立刻红 —— 阈值"含不含等号"必须钉死。"""
    assert corpus.classify(True, face, arm) == expect


def test_test_session_is_tier_t_not_a_failed_formal_one():
    """★ 无名测试场是 `T`,**不是** `D`。

    两者含义完全不同:`T` 是"没按协议采,别拿去出报告";`D` 是"按协议采了,
    但这一场没测到"。混成一样会让人以为测试场也该进语料。
    """
    assert corpus.classify(False, 100.0, 100.0) == "T"
    assert corpus.classify(True, 0.0, 0.0) == "D"


def test_missing_face_log_is_unknown_not_zero():
    """★ 没有面部日志 ⟹ `?`(判不出),**不是** `D`(脸 0%)。

    红法:把 `if face is None: return "?"` 删掉 ⟹ 会落到 `D`,本条红。
    这个区分很实在:前者要去查"日志为什么没写",后者要去查"为什么没检出脸"。
    """
    assert corpus.classify(True, None, 90.0) == "?"


# ── 2. 正式场的判据 ─────────────────────────────────────────────────────

def test_formal_pattern_matches_real_names_and_rejects_hash_dirs():
    """实测 24 场正式场全部命中,15 场测试场全部不命中。"""
    ok = "20260929_200333-2-候选丙-000000000-投资学__20260929_200333_2e84"
    m = corpus.FORMAL_PATTERN.search(ok)
    assert m and m.group(1) == "候选丙" and m.group(2) == "000000000"

    for bad in ("20260927_212143_95a8",
                "20260928_210726-1-2-3-4__20260928_210726_1d00",
                "20260928_230839-1-2-3-5__20260928_230839_87c9"):
        assert not corpus.FORMAL_PATTERN.search(bad), bad


# ── 3. 覆盖率按"带数据的行"算 ───────────────────────────────────────────

def test_coverage_counts_rows_with_data_not_rows(tmp_path):
    """★ 4 行里只有 2 行有 AU 值 ⟹ 50%,**不是** 100%。

    红法:把 `notna().any(axis=1)` 换成 `len(df)` ⟹ 得 100%,本条红。
    实测背景:面部日志 21861 行里 25.5% 整行为空,按行数算会把空日志判成有数据。
    """
    p = tmp_path / "face.csv"
    pd.DataFrame({"au12_smile": [0.1, np.nan, 0.3, np.nan],
                  "avg_ear": [0.3, np.nan, 0.3, np.nan]}).to_csv(p, index=False)
    assert corpus._coverage(p, corpus.FACE_COLS) == pytest.approx(50.0)


def test_coverage_of_missing_file_is_none_and_empty_file_is_zero(tmp_path):
    """不存在 ⟹ `None`(判不出);存在但 0 行 ⟹ `0.0`(确定的事实)。"""
    assert corpus._coverage(tmp_path / "nope.csv", corpus.FACE_COLS) is None
    p = tmp_path / "empty.csv"
    pd.DataFrame({"au12_smile": []}).to_csv(p, index=False)
    assert corpus._coverage(p, corpus.FACE_COLS) == 0.0


# ── 4. 端到端:一个假录制目录 ───────────────────────────────────────────

def test_audit_session_end_to_end(tmp_path, monkeypatch):
    """造一个正式场目录:脸有数据、臂没有、语音日志 0 行、整场录音在。

    期望:`B`(只有脸),并给出两条**可执行**的说明 —— 语音可从整场录音重算、
    没有题库只能出会话级。
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    pd.DataFrame({"au12_smile": [0.2] * 5, "avg_ear": [0.3] * 5,
                  "head_pitch": [0.0] * 5}).to_csv(
        logs / "face_au_log_x.csv", index=False)
    pd.DataFrame({"left_elbow_angle": [np.nan] * 5,
                  "right_elbow_angle": [np.nan] * 5}).to_csv(
        logs / "gesture_emotion_log_x.csv", index=False)
    pd.DataFrame({"pitch_mean": []}).to_csv(
        logs / "interview_emotion_log_x.csv", index=False)

    sd = tmp_path / "20260929_200333-2-候选丙-000000000-投资学__x"
    (sd / "media").mkdir(parents=True)
    (sd / "media" / "camera.webm").write_bytes(b"fake")

    monkeypatch.setattr(corpus, "_logs_dir", lambda: logs)
    a = corpus.audit_session(sd, manifest={})

    assert a.session_id == "x"
    assert a.formal is True and a.name == "候选丙"
    assert a.tier == "B"
    assert a.voice_rows == 0 and a.has_whole_audio is True
    joined = ";".join(a.notes)
    assert "camera.webm" in joined and "会话级" in joined


def test_media_reuploaded_in_segments_is_not_whole_audio(tmp_path, monkeypatch):
    """★ 多次整文件重传 ⟹ `has_whole_audio` 必须是 False,并给出说明。

    ## 为什么

    `has_whole_audio` 原来只判 `(session_dir/"media"/"camera.webm").exists()`。
    实测 13 场 A 档里**候选己一场是 9 段录制拼接**:`retention.camera.jsonl` 有 9 条
    记录全指向同一个 `media/camera.webm`、`seq` 都是 1,但 9 个不同的大小与 sha256
    ⟹ 每段结束就整文件重传并覆盖,盘上只剩最后一段(实测 434.7 秒,而整场 6959 秒)。
    按"文件在不在"判,这一场会被标成"整场录音在 ⟹ V 族可从 camera.webm 重算" ——
    而实际上重算出来的只覆盖最后 7 分钟。这是**一句会误导下游的断言**。

    红法:把 `_camera_uploads(session_dir) <= 1` 去掉(退回只判文件存在)⟹ 本条红。
    """
    import json
    logs = tmp_path / "logs"
    logs.mkdir()
    pd.DataFrame({"au12_smile": [0.2] * 5, "avg_ear": [0.3] * 5,
                  "head_pitch": [0.0] * 5}).to_csv(logs / "face_au_log_y.csv", index=False)
    pd.DataFrame({"left_elbow_angle": [1.0] * 5,
                  "right_elbow_angle": [1.0] * 5}).to_csv(
        logs / "gesture_emotion_log_y.csv", index=False)
    pd.DataFrame({"pitch_mean": []}).to_csv(logs / "interview_emotion_log_y.csv", index=False)

    sd = tmp_path / "20260928_180131-10-候选己-000000000-金融学院__y"
    (sd / "media").mkdir(parents=True)
    (sd / "media" / "camera.webm").write_bytes(b"fake")
    recs = [{"kind": "video", "modality": "camera", "seq": 1,
             "file": "media/camera.webm", "bytes": b, "sha256": f"{b}",
             "received_at_wall": 1.0 + i, "declared_ts": None,
             "source_endpoint": "/session/media", "session_id": "y"}
            for i, b in enumerate((250, 293, 136))]
    (sd / "retention.camera.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    monkeypatch.setattr(corpus, "_logs_dir", lambda: logs)
    a = corpus.audit_session(sd, manifest={})

    assert a.has_whole_audio is False
    joined = ";".join(a.notes)
    assert "3 次整文件重传覆盖" in joined
    assert "只剩最后一段" in joined
    # 那句会误导下游的话**不能**出现。
    assert "整场录音在" not in joined
