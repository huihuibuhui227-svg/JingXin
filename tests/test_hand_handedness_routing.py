# tests/test_hand_handedness_routing.py
"""左右手按**模型给的 handedness** 分,不按检出顺序。

2026-09-26 之前是 `analyzer_key = 'left_hand' if hand_id == 0 else 'right_hand'`
—— `hand_id` 是"第几个被检出",换个姿势左右就互换,而日志里那两列看着像左右手,
报告里还叫「左臂姿态 / 右臂姿态」。模型本来就算得出 handedness,只是没人读。

⚠️ 模型按**镜像(自拍)输入**判左右(官方文档:handedness 是 "determined assuming the
input image is mirrored"),而本项目的帧是画布原样绘制的非镜像图 ⟹ **标签要翻过来**。
本文件把那个翻法钉住:模型说 "Right" ⟹ 那只手是人的**左手**。
"""
import asyncio
import importlib
import types

import pytest

import media_retention                                                # noqa: E402
gesture_app = importlib.import_module("gesture_analysis.api.app")     # noqa: E402


class _FakeUpload:
    def __init__(self, data: bytes):
        self._data, self.filename, self.content_type = data, "f.jpg", "image/jpeg"

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    async def form(self):
        raise RuntimeError("no multipart")


class _FakeImage:
    shape = (720, 1280, 3)


class _RecordingAnalyzer:
    """记下自己收到的是哪一组 landmark —— 本文件要验的正是"谁收到了哪只手"。"""
    def __init__(self):
        self.seen = []

    def update(self, landmarks):
        self.seen.append(landmarks)

    def get_results(self):
        # 端点要读 `['resilience_score']`(少这个键是 KeyError,不是我要的那条断言红)
        return {"resilience_score": 50.0}


class _FakeEmotionAnalyzer:
    def infer_emotion(self, *_args):
        return {"emotion_state": "neutral", "overall_score": 50.0, "emoji": "😐",
                "feedback": "", "used_features": []}


def _wired(monkeypatch, tmp_path, groups, handedness):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    monkeypatch.setattr(gesture_app, "cv2", types.SimpleNamespace(
        imdecode=lambda buf, flag: _FakeImage(), cvtColor=lambda i, c: i,
        IMREAD_COLOR=1, COLOR_BGR2RGB=1))
    monkeypatch.setattr(gesture_app, "np", types.SimpleNamespace(
        frombuffer=lambda b, t: b, uint8="u8"))
    analyzers = {k: _RecordingAnalyzer() for k in
                 ("left_hand", "right_hand", "shoulder", "left_arm", "right_arm",
                  "upper_body")}
    analyzers["world"] = {k: _RecordingAnalyzer() for k in
                          ("shoulder", "left_arm", "right_arm")}
    analyzers["emotion"] = _FakeEmotionAnalyzer()
    monkeypatch.setattr(gesture_app, "get_or_create_analyzers", lambda sid: analyzers)
    monkeypatch.setattr(gesture_app, "get_or_create_detectors", lambda sid: {
        "hands": types.SimpleNamespace(
            detect_with_handedness=lambda img, ts: (groups, handedness),
            detect=lambda img, ts: groups),
        "pose": types.SimpleNamespace(detect=lambda img, ts: [],
                                      detect_with_world=lambda img, ts: (None, None)),
    })
    loggers = {}
    monkeypatch.setattr(gesture_app, "session_loggers", loggers)
    return analyzers, loggers


def _post(payload: bytes, sid: str):
    return asyncio.run(gesture_app.analyze_image(
        request=_FakeRequest(), file=_FakeUpload(payload), session_id=sid))


def test_handedness_decides_left_right_not_detection_order(monkeypatch, tmp_path):
    """★ 第一只被检出的手是**右手**时,它必须进 `right_hand` 槽,而不是"左手"。

    红法:退回按顺序分(`hand_id == 0 → left_hand`)—— 本测试里第一只手的
    `right_hand` 槽会收到**另一组** landmark,断言立刻红。
    """
    first, second = [("第一只手",)], [("第二只手",)]
    analyzers, _ = _wired(monkeypatch, tmp_path, [first, second],
                          [("Left", 0.91), ("Right", 0.97)])
    _post(b"jpeg", "20260926_120000_hhhh")

    # 模型说 Left ⟹ 非镜像输入要翻 ⟹ 它是人的右手
    assert analyzers["right_hand"].seen == [first], analyzers["right_hand"].seen
    assert analyzers["left_hand"].seen == [second], analyzers["left_hand"].seen


def test_order_does_not_matter_swap_the_two_hands(monkeypatch, tmp_path):
    """对照臂:构造一组"顺序与 handedness 给出**相反**结论"的数据。

    ⚠️ 这条第一版写错过:那时数据恰好让两种分法得到同一个结果,于是**变异测试里
    它不红** —— 看着是一个对照臂,其实什么都没对照(§4.1「测试通过 ≠ 有约束力」)。
    现在:b 先被检出、模型说它 Left ⟹ 翻过来它是人的**右手**;
    按顺序分会说它是左手 ⟹ 两种分法结论相反,这条才真的在测。
    """
    a, b = [("A",)], [("B",)]
    analyzers, _ = _wired(monkeypatch, tmp_path, [b, a],
                          [("Left", 0.91), ("Right", 0.97)])
    _post(b"jpeg", "20260926_120000_iiii")
    assert analyzers["right_hand"].seen == [b], analyzers["right_hand"].seen
    assert analyzers["left_hand"].seen == [a], analyzers["left_hand"].seen


def test_no_handedness_falls_back_without_losing_the_hands(monkeypatch, tmp_path):
    """模型没给 handedness 时:两只手**照旧都收下**(不丢数据),只是左右无依据。

    红法:把兜底那支删掉 —— 没 handedness 的帧会一只都收不到,手部维度直接空。
    """
    analyzers, _ = _wired(monkeypatch, tmp_path, [[("只手",)], [("另一只",)]], [None, None])
    assert _post(b"jpeg", "20260926_120000_jjjj")["status"] == "success"
    assert analyzers["left_hand"].seen and analyzers["right_hand"].seen, \
        "没 handedness 就把手丢了 —— 那是静默丢数据"


def test_log_carries_the_model_label_and_confidence(tmp_path, monkeypatch):
    """★ 四列 provenance:`模型原始标签` + `置信度`,没依据时留**空**。

    留原始标签(而不是只留翻好的结论)是为了让"翻没翻对"**可审计** ——
    翻法对不对靠的是 mediapipe 那条镜像约定,记下原值才验得了。

    红法:把 `_handedness_cells` 换成写死 "Left"/0 —— 未知就变成了一个看着已知的值。
    """
    import csv

    from gesture_analysis.utils.logger import GestureLogger
    gl = importlib.import_module("gesture_analysis.utils.logger")
    monkeypatch.setattr(gl, "LOGS_DIR", str(tmp_path / "logs"))
    log = GestureLogger(session_id="20260926_120000_kkkk")

    assert {"left_hand_model_label", "left_hand_model_label_conf",
            "right_hand_model_label", "right_hand_model_label_conf"} <= set(log.fieldnames), \
        f"四列没进表头:{[c for c in log.fieldnames if 'model_label' in c]}"

    log.log(left_hand_result={}, right_hand_result={}, shoulder_result={},
            handedness_info={"left_hand": ("Right", 0.9712), "right_hand": None})
    rows = list(csv.DictReader(open(log.log_file, encoding="utf-8")))
    assert rows, "一行都没写"
    assert rows[-1]["left_hand_model_label"] == "Right", rows[-1]["left_hand_model_label"]
    assert float(rows[-1]["left_hand_model_label_conf"]) == 0.971
    assert rows[-1]["right_hand_model_label"] == "", \
        f"没依据的那一槽被写成了已知值:{rows[-1]['right_hand_model_label']!r}"
    assert rows[-1]["right_hand_model_label_conf"] == ""
