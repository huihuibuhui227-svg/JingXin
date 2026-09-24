"""三个 logger 都写 `session_id`:CSV 首列 + 带会话 id 的文件名。

背景:三个模块原本只按墙上时间命名日志,报告层因此把「5 月的脸 + 2 月的语音」
拼成过一份报告。文件名与首列都带上 session 之后,同一会话的三种日志一眼可归堆,
跨会话的错配在拼接时就能被发现。
"""

import csv
from pathlib import Path

# 以模块对象导入:三个 logger 的模块级 LOGS_DIR 在构造时才被读取,
# 测试里改指向 tmp_path,免得往仓库的 data/logs/ 下写真实日志文件。
import face_expression.utils.logger as face_logger_module
import gesture_analysis.utils.logger as gesture_logger_module
from face_expression.utils.logger import DataLogger
from gesture_analysis.utils.logger import GestureLogger
from voice_interaction.utils.logger import VoiceLogger

SID = "20260924_153012_9f3c"


def _rows(path) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------- voice ----------

def test_voice_logger_filename_and_first_column(tmp_path):
    lg = VoiceLogger(log_type="interview", log_dir=str(tmp_path), session_id=SID)
    assert lg.csv_file.name == f"interview_emotion_log_{SID}.csv"
    assert lg.fieldnames[0] == "session_id"
    lg.log_prosody({"pitch_mean": 1.0}, question_index=1, emotion="neutral",
                   feedback="", connective_density=3.5, connective_density_std=0.4, n_rows=4)
    rows = _rows(lg.csv_file)
    assert rows[0]["session_id"] == SID
    assert rows[0]["connective_density"] == "3.5"


def test_voice_logger_without_id_writes_none(tmp_path):
    lg = VoiceLogger(log_type="interview", log_dir=str(tmp_path))
    assert lg.csv_file.name.startswith("interview_emotion_log_NONE_")
    assert lg.fieldnames[0] == "session_id"


def test_voice_logger_research_prefix_and_json_stem_follow_the_session(tmp_path):
    lg = VoiceLogger(log_type="research", log_dir=str(tmp_path), session_id=SID)
    assert lg.csv_file.name == f"research_emotion_log_{SID}.csv"
    assert lg.json_file.stem == lg.csv_file.stem


def test_voice_logger_density_columns_default_to_empty_not_zero(tmp_path):
    """三列默认 None:Task 2 的 None 语义是"过短不出值,不写 0"(见 Ruling)。

    这里断言的是 `log_prosody` 没收到它们时**不编造 0.0** —— 写成
    `or 0.0` 会让日志把"没算"伪装成"算出来是 0",报告侧看不出差别。
    """
    lg = VoiceLogger(log_type="interview", log_dir=str(tmp_path), session_id=SID)
    lg.log_prosody({"pitch_mean": 1.0}, question_index=1, emotion="neutral", feedback="")
    rows = _rows(lg.csv_file)
    assert rows[0]["connective_density"] == ""
    assert rows[0]["connective_density_std"] == ""
    assert rows[0]["n_rows"] == ""


# ---------- face ----------

def test_face_logger_writes_session_id_in_first_column(tmp_path, monkeypatch):
    monkeypatch.setattr(face_logger_module, "LOGS_DIR", str(tmp_path))
    lg = DataLogger(log_type="video", session_id=SID)
    assert Path(lg.log_file).name == f"face_au_log_{SID}.csv"
    assert lg.fieldnames[0] == "session_id"
    assert lg.log({"focus_score": 0.3})
    assert _rows(lg.log_file)[0]["session_id"] == SID


def test_face_logger_without_id_and_caller_path_override(tmp_path, monkeypatch):
    """`face_expression/api/app.py` 构造后会把 `log_file` 覆盖成自己算的路径
    (Task 5 换成带 session 的路径),覆盖必须照旧生效,且覆盖后的文件也要有表头。
    """
    monkeypatch.setattr(face_logger_module, "LOGS_DIR", str(tmp_path))
    lg = DataLogger(log_type="video")
    assert Path(lg.log_file).name.startswith("face_au_log_NONE_")

    override = tmp_path / f"face_au_log_{SID}.csv"
    lg.log_file = str(override)
    assert lg.log({"focus_score": 0.3})
    row = _rows(override)[0]
    assert row["session_id"] == "NONE"
    assert row["focus_score"] == "0.3"


# ---------- gesture ----------

def test_gesture_logger_writes_session_id_in_first_column(tmp_path, monkeypatch):
    monkeypatch.setattr(gesture_logger_module, "LOGS_DIR", str(tmp_path))
    lg = GestureLogger(session_id=SID)
    assert lg.log_file.name == f"gesture_emotion_log_{SID}.csv"
    assert lg.fieldnames[0] == "session_id"
    assert lg.log(None, None, None)
    assert _rows(lg.log_file)[0]["session_id"] == SID


def test_gesture_logger_without_id_and_explicit_path_still_wins(tmp_path, monkeypatch):
    """`gesture_analysis/api/app.py` 传的是 `log_file_path=`,它必须继续优先于
    session_id 拼名,且 `log_file` 仍是 Path(face 是 str,两边各自保持现状)。
    """
    monkeypatch.setattr(gesture_logger_module, "LOGS_DIR", str(tmp_path))
    lg = GestureLogger()
    assert lg.log_file.name.startswith("gesture_emotion_log_NONE_")

    explicit = tmp_path / "explicit_gesture_log.csv"
    lg2 = GestureLogger(log_file_path=str(explicit), session_id=SID)
    assert isinstance(lg2.log_file, Path)
    assert lg2.log_file == explicit
    assert lg2.fieldnames[0] == "session_id"
    assert lg2.log(None, None, None)
    assert _rows(explicit)[0]["session_id"] == SID
