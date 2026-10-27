"""数据导出:CSV 清单 / 转写导出 / 图表 PNG。

这些接口不参与报告口径，但**导出的内容会被拿去做分析** —— 所以至少压住：
* 清单只说盘上事实（存在/行数/大小）；
* 转写里带得上题干与标注（否则一堆 sid 没法对号）；
* 图能在缺列/空场时给出"没数据"，而不是画一张空图。
"""
import json

import pytest

import exports

SID = "20261026_002930_cb79"

FACE = (
    "session_id,timestamp,au12_smile,au4_frown,gaze_direction_x,gaze_direction_y\n"
    + "".join(f"{SID},{i},0.1,0.2,0.01,-0.02\n" for i in range(4))
)
GESTURE = (
    "session_id,timestamp,left_hand_jitter,right_hand_jitter,left_hand_spread,right_hand_spread\n"
    + "".join(f"{SID},{1792945700 + i},0.01,0.02,2.0,1.5\n" for i in range(3))
)
VOICE = (
    "session_id,timestamp,pitch_mean,energy_mean,duration_sec,connective_density\n"
    + "".join(f"{SID},2026-10-26T00:31:0{i},100,0.05,8.0,0.0\n" for i in range(3))
)


@pytest.fixture
def session(tmp_path, monkeypatch):
    """一场有面板/语音日志、有转写与标注的假会话。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / f"face_au_log_{SID}.csv").write_text(FACE, encoding="utf-8")
    (logs / f"gesture_emotion_log_{SID}.csv").write_text(GESTURE, encoding="utf-8")
    (logs / f"research_emotion_log_{SID}.csv").write_text(VOICE, encoding="utf-8")
    monkeypatch.setattr(exports, "_logs_dir", lambda: logs)

    sess = tmp_path / f"1026-候选Y__{SID}"
    sess.mkdir()
    (sess / "label.json").write_text(json.dumps({
        "session_id": SID, "name": "候选Y", "student_id": "241650137",
        "department": "计算机与人工智能学院", "serial": "1026",
    }, ensure_ascii=False), encoding="utf-8")
    (sess / "questions.jsonl").write_text(
        json.dumps({"index": 0, "qid": "请描述一个你研究过的问题？"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (sess / "transcript.json").write_text(json.dumps({
        "session_id": SID,
        "recorded_at": "2026-10-26T00:31:02+08:00",
        "merged": {"text": "回答一", "n_chars": 3},
        "segments": [{"index": 0, "text": "回答一"}],
    }, ensure_ascii=False), encoding="utf-8")
    return SID


def test_files_lists_what_is_on_disk(session):
    d = exports.list_payload(session)
    by_key = {x["key"]: x for x in d["logs"]}
    assert by_key["face"]["exists"] and by_key["face"]["rows"] == 4
    assert by_key["gesture"]["exists"]
    assert by_key["voice_research"]["exists"]
    assert not by_key["voice_interview"]["exists"]      # 面试那份本来就没有
    assert d["transcript"]["exists"]
    assert d["transcript"]["n_segments"] == 1
    assert d["questions"] == 1


def test_transcript_txt_carries_subject_and_question(session):
    txt = exports.transcript_txt(session)
    assert "候选Y" in txt and "241650137" in txt
    assert "请描述一个你研究过的问题？" in txt
    assert "回答一" in txt


def test_transcript_csv_has_subject_column(session):
    text = exports.transcript_csv(session)
    first = text.splitlines()[0]
    assert first.startswith("index,subject,question,answer,n_chars")
    assert "候选Y" in text.splitlines()[1]


def test_chart_png_is_a_real_png(session):
    png = exports.chart_png(session, "face")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 5000


def test_chart_without_data_says_so(tmp_path, monkeypatch):
    """没有这一场的日志 ⟹ 抛 NoData（路由转 404），不画空图。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(exports, "_logs_dir", lambda: logs)
    with pytest.raises(exports.NoData):
        exports.chart_png(SID, "face")
