# tests/test_log_file_selection.py
"""报告侧按文件名挑日志:两种命名形态都要认,无 id 的 NONE 桶必须继续不可见。

背景(T6 审查 Important 1):`data_loader` 的选取正则只认「以纯时间戳结尾」的旧形态
`..._log_YYYYMMDD_HHMMSS.csv`。而 T3 起,M1 给三个模块的文件名都带上了 `session_id`
(`voice_interaction/asr/session.py:new_session_id` = `YYYYMMDD_HHMMSS_<4 位十六进制>`):

    face_au_log_<session_id>.csv        (face_expression/api/app.py)
    gesture_emotion_log_<session_id>.csv(gesture_analysis/api/app.py)
    interview_emotion_log_<session_id>.csv(voice_interaction/utils/logger.py)

即**没有任何产出方**会再写出旧形态的名字 —— 报告路径因此加载不到任何文件。
"""

import csv
from pathlib import Path

import pytest

from report_frontend.data_loader import LogDataLoader

# 每份日志带一列 marker,用来直接断定「加载器选中的是哪一份文件」
_FIELDS = ["marker", "value"]


def _write_csv(path: Path, marker: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerow({"marker": marker, "value": 1.0})


def _markers(directory: Path) -> dict:
    """{模态: 被选中的那份文件的 marker} —— 直接回答「选了哪个文件」。"""
    data = LogDataLoader(str(directory)).get_fused_latest_data()
    return {key: df["marker"].iloc[0] for key, df in data.items()}


def test_m1_session_named_logs_are_loadable(tmp_path):
    """审查者的复现反做:目录里**只有** M1 命名日志时,三个模态都要被取到。

    红在:修复前选取正则要求 `..._log_<8 digits>_<6 digits>.csv`,M1 形态
    (`..._log_<YYYYMMDD>_<HHMMSS>_<4 hex>.csv`)尾部多一段十六进制 → 不匹配
    → `get_fused_latest_data()` 返回 `{}`,本测试第一句就红(实测:MODALITIES: [])。

    ⚠️ M2 起 fixture 改成**三个模态同一个 session_id**:一场会话只有一个 id(这正是 M1
    铸号的意义),而 M2 的选取是"先定目标 id、再按 id 取"。老 fixture 里三个模态各用
    一个 id,那是"跨场次拼接"的形态 —— 现在按 id 选时只会取到其中一场。
    """
    _write_csv(tmp_path / "face_au_log_20260924_120000_a1b2.csv", "m1_face")
    _write_csv(tmp_path / "gesture_emotion_log_20260924_120000_a1b2.csv", "m1_gesture")
    _write_csv(tmp_path / "interview_emotion_log_20260924_120000_a1b2.csv", "m1_interview")

    markers = _markers(tmp_path)

    assert markers == {"face": "m1_face", "gesture": "m1_gesture",
                       "voice_interview": "m1_interview"}, (
        f"M1 命名的日志没被加载(修前这里是不含任何键的 {{}}):{markers}"
    )


def test_selection_accepts_legacy_and_m1_forms_but_never_none(tmp_path):
    """旧形态要认、NONE 桶永远不参与。

    红在:① 修复前 M1 形态与 NONE 形态**都**不被识别,face 会选到旧形态那份
    (或 research 根本没有候选);② 若有人把正则放宽成「时间戳可选后缀」,名字更"新"的
    `gesture_emotion_log_NONE_20260924_235959.csv` 会赢过真日志 → gesture 的 marker 变
    `none_gesture`,NONE 桶重新可见并跨天混行。

    M2 之后本测试的 fixture 改成**一场旧形态会话**(三个模态同一个 id):选取是"先定目标
    `session_id`、再按 id 取",所以"同一模态两种形态取更新那份"这种比较不再成立 ——
    session_id 就是从文件名推出来的,两种形态必然是两个不同的 id、两场不同的会话。
    """
    # 一场**旧形态**(纯时间戳)会话的三份日志 —— 旧形态必须仍被识别,否则历史日志读不到
    # 一场**旧形态**(纯时间戳)会话的三份日志 —— 旧形态必须仍被识别,否则历史日志读不到
    _write_csv(tmp_path / "face_au_log_20260923_090000.csv", "legacy_face")
    _write_csv(tmp_path / "gesture_emotion_log_20260923_090000.csv", "legacy_gesture")
    _write_csv(tmp_path / "research_emotion_log_20260923_090000.csv", "legacy_research")
    # 两种 NONE 形状的名字都比上面"新" —— 它们必须**不进候选**(既不进聚合,也不当目标)
    _write_csv(tmp_path / "gesture_emotion_log_NONE_20260924_235959.csv", "none_gesture")
    _write_csv(tmp_path / "face_au_log_NONE.csv", "none_face")
    # 非日志文件:没有模态前缀 / 模态名不在名单里
    _write_csv(tmp_path / "notes.csv", "not_a_log")
    _write_csv(tmp_path / "random_thing_20260924_120000.csv", "wrong_modality")

    markers = _markers(tmp_path)

    assert set(markers) == {"face", "gesture", "voice_research"}, (
        f"三模态没齐全(interview 不在本目录;research 模态在加载器里的键是 voice_research):{markers}"
    )
    assert markers["face"] == "legacy_face", (
        f"旧形态必须仍被接受(否则历史日志再也读不到):{markers}"
    )
    assert markers["voice_research"] == "legacy_research", (
        f"旧形态必须仍被接受(否则历史日志再也读不到):{markers}"
    )
    assert markers["gesture"] == "legacy_gesture", (
        f"NONE 桶重新可见了(名字更新的 NONE 文件赢过了真日志)—— "
        f"它跨天增长,一次聚合会把不同天的行混在一起:{markers}"
    )
    assert "none_gesture" not in markers.values() and "none_face" not in markers.values()
    assert "not_a_log" not in markers.values() and "wrong_modality" not in markers.values()


def test_reported_session_is_the_filename_session(tmp_path):
    """加载器自报的会话必须是文件名里那一段(含随机段),且不含任何 NONE。

    红在:修复前 M1 形态不被识别 → 概览里什么都报不出来(「无数据」);
    另一个反向:若把 NONE 形态也放进来,概览会把跨天的 NONE 会话当成一场会话报出来。
    """
    _write_csv(tmp_path / "face_au_log_20260924_120000_a1b2.csv", "m1_face")
    _write_csv(tmp_path / "face_au_log_NONE_20260923_010101.csv", "none_face")

    summary = LogDataLoader(str(tmp_path)).get_available_modalities_summary()

    assert "20260924_120000_a1b2" in summary, f"自报的会话不是文件名里的会话:{summary}"
    assert "NONE" not in summary, f"NONE 会话不该出现在报告侧的视野里:{summary}"


@pytest.mark.parametrize("name", [
    "face_au_log_NONE.csv",                 # 单文件 NONE(face/gesture 的无 id 形态)
    "gesture_emotion_log_NONE.csv",
    "interview_emotion_log_NONE_20260924_133357.csv",  # 尾部恰好也是时间戳
    "research_emotion_log_NONE.csv",
])
def test_none_shaped_filenames_are_rejected_one_by_one(tmp_path, name):
    """逐个钉 NONE 形态:放宽正则时最容易顺手放进来的是这一类。

    目录里只有这一份文件,故正确的行为是**一个模态都加载不出来**(而不是选中它)。
    """
    _write_csv(tmp_path / name, "none_marker")
    assert _markers(tmp_path) == {}, f"{name} 被当成日志选中了"
