# tests/test_gesture_task_outputs_wired.py
"""手势那 24 个"结构性空/恒 0"的列,以及 world 米制那 8 列 —— 2026-09-26 接上。

实测(场 `20260926_124439_0f5b`,275 帧)之前的真相:
  · **18 列整列全空**:10 个手指角度 + 4 个姿态角度 + 4 个"屏幕角度" ——
    端点调 `logger.log()` 时压根没传 `angles_data`,算它的代码只活在 `examples/` 里;
  · **head_*/torso_* 六列恒 0**:`UpperBodyAnalyzer` 只在 `examples/` 里被构造过;
  · 没收到手的槽照样写 **50.0**(分析器默认值)—— "没测到"变成一个看着合法的数。

本文件钉四件事:角度定义(照 examples 移植)、四路接线、world 列、空槽留空。
"""
import asyncio
import csv
import importlib
import types

import pytest

import media_retention                                                # noqa: E402
gesture_app = importlib.import_module("gesture_analysis.api.app")     # noqa: E402
gl = importlib.import_module("gesture_analysis.utils.logger")         # noqa: E402
from gesture_analysis.core.feature_extraction.angles import (         # noqa: E402
    finger_angles, joint_angle, pose_angles)

SID = "20260926_120000_zzzz"


class _Pt:
    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _Upload:
    def __init__(self, data=b"jpeg"):
        self._data, self.filename, self.content_type = data, "f.jpg", "image/jpeg"

    async def read(self):
        return self._data


class _Req:
    async def form(self):
        raise RuntimeError("no multipart")


class _Img:
    shape = (720, 1280, 3)


class _Analyzer:
    """按类型交一份**已知**结果 —— 但**必须先被喂过**:没喂过就交空。

    ⚠️ 这一条是变异测试逼出来的:第一版替身的 `get_results()` 无条件交那份结果,
    于是"端点到底有没有喂它"**测不出来** —— 把 `upper_body.update(...)` 删掉,
    测试照样绿。替身得像真分析器一样:**没有输入就没有输出**。
    """
    def __init__(self, result):
        self.result = result
        self.updates = 0

    def update(self, _lm):
        self.updates += 1

    def get_results(self):
        return dict(self.result) if self.updates else {}


HAND = {"resilience_score": 77.0, "jitter": 0.011, "fist_status": 0,
        "spread": 0.2, "is_valid": True}
SHOULDER = {"shoulder_score": 61.0, "left_jitter": 0.02, "right_jitter": 0.03,
            "shrug_level": 0.1, "is_calibrated": True, "is_valid": True}
ARM = {"arm_score": 55.0, "wrist_jitter": 0.04, "elbow_jitter": 0.05,
       "arm_angle": 33.0, "arm_stability": 0.6, "is_valid": True}
UPPER = {"head_score": 44.0, "head_jitter": 0.06, "head_tilt": 0.07,
         "torso_score": 48.0, "torso_jitter": 0.08, "torso_stability": 0.5,
         "is_valid": True}


def _hand21():
    """21 点的手:每根手指的三个关节**共线**(指角 180°),好断言。

    ⚠️ 别拿重合点当替身:重合是退化输入,`finger_angles` 会(正确地)交空字典 ——
    第一版测试就是这么写错的,那条断言于是测的是"我没给像样的数据"。
    """
    pts = [_Pt(0.5, 0.5) for _ in range(21)]
    for n, (i, j, k) in enumerate(((2, 3, 4), (5, 6, 8), (9, 10, 12),
                                   (13, 14, 16), (17, 18, 20))):
        y = 0.10 + 0.05 * n
        pts[i] = _Pt(0.10, y)
        pts[j] = _Pt(0.12, y)
        pts[k] = _Pt(0.14, y)
    return pts


def _pose33():
    """33 个点,其中我们用到的那几个摆成**可预期的几何**:左肘 180°(伸直)。

    其余点给个可见度 0.0 的占位 —— 它们不该参与任何一条角度(见可见度门限)。
    """
    pts = [_Pt(0.5, 0.5, visibility=0.0) for _ in range(33)]
    def put(i, x, y, vis=1.0):
        pts[i] = _Pt(x, y, visibility=vis)
    put(11, 0.40, 0.50)     # left_shoulder
    put(13, 0.40, 0.70)     # left_elbow
    put(15, 0.40, 0.90)     # left_wrist  → 肩-肘-腕 共线 ⟹ 180°
    put(23, 0.38, 0.95)     # left_hip
    return pts


def _wired(monkeypatch, tmp_path, *, pose=None, world=None, hands=(), handedness=(), record=True):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    monkeypatch.setattr(gl, "LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(gesture_app, "cv2", types.SimpleNamespace(
        imdecode=lambda buf, flag: _Img(), cvtColor=lambda i, c: i,
        IMREAD_COLOR=1, COLOR_BGR2RGB=1))
    monkeypatch.setattr(gesture_app, "np", types.SimpleNamespace(
        frombuffer=lambda b, t: b, uint8="u8"))
    analyzers = {
        "left_hand": _Analyzer(HAND), "right_hand": _Analyzer(HAND),
        "shoulder": _Analyzer(SHOULDER),
        "left_arm": _Analyzer(ARM), "right_arm": _Analyzer(ARM),
        "upper_body": _Analyzer(UPPER),
        "world": {"shoulder": _Analyzer(SHOULDER), "left_arm": _Analyzer(ARM),
                  "right_arm": _Analyzer(ARM)},
        "emotion": type("E", (), {"infer_emotion": lambda *a, **k: {
            "emotion_state": "neutral", "overall_score": 50.0, "emoji": "😐",
            "feedback": "", "used_features": []}})(),
    }
    monkeypatch.setattr(gesture_app, "get_or_create_analyzers", lambda sid: analyzers)
    monkeypatch.setattr(gesture_app, "get_or_create_detectors", lambda sid: {
        "hands": types.SimpleNamespace(
            detect_with_handedness=lambda img, ts: (list(hands), list(handedness)),
            detect=lambda img, ts: list(hands)),
        "pose": types.SimpleNamespace(
            detect=lambda img, ts: pose, detect_with_world=lambda img, ts: (pose, world)),
    })
    loggers = {}
    if record:
        loggers[SID] = (gl.GestureLogger(session_id=SID), None)
    monkeypatch.setattr(gesture_app, "session_loggers", loggers)
    return loggers


def _row(loggers):
    path = loggers[SID][0].log_file
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert rows, "一行都没写"
    return rows[-1]


def _post():
    return asyncio.run(gesture_app.analyze_image(
        request=_Req(), file=_Upload(), session_id=SID))


# ── 角度定义(照 examples 移植)──────────────────────────────────────
def test_joint_angle_is_the_angle_at_the_middle_point():
    """红法:把顶点当成首点 —— 直角那个用例会得到 45° 而不是 90°。"""
    assert joint_angle(_Pt(0, 0), _Pt(1, 0), _Pt(1, 1)) == 90.0
    assert joint_angle(_Pt(0, 0), _Pt(1, 0), _Pt(2, 0)) == 180.0


def test_visibility_floor_keeps_low_confidence_points_out():
    """可见度 ≤0.6 的点不参与(与 examples 的 0.6 门限一致)⟹ 该键**缺席**,不是 0。"""
    pose = _pose33()
    assert "left_elbow_angle" in pose_angles(pose)
    pose[15] = _Pt(0.4, 0.9, visibility=0.5)          # 腕子看不见了
    assert "left_elbow_angle" not in pose_angles(pose), \
        "可见度不足仍然算出了角度 —— 那是拿看不见的点编一个数"


def test_finger_angles_need_a_full_hand():
    """21 点的手才有指角;点不够交**空字典**(不编 0)。"""
    assert finger_angles([_Pt(0.5, 0.5)] * 20) == {}
    full = [_Pt(0.5, 0.5)] * 21
    # 21 个重合点 → 全部退化 → 空(这与"点不够"是两回事,所以两种都钉住)
    assert finger_angles(full) == {}


# ── 四路接线 ────────────────────────────────────────────────────────
def test_pose_angles_reach_the_log_row(monkeypatch, tmp_path):
    """★ 那 18 列此前**从来没有产出方**。现在 `angles_data` 真被传进去了。

    红法:把 `angles_data=angles_data` 从 `logger.log(...)` 里去掉 —— 该列变空。
    """
    loggers = _wired(monkeypatch, tmp_path, pose=_pose33(),
                     hands=[_hand21()], handedness=[("Left", 0.9)])
    _post()
    row = _row(loggers)
    assert row["left_elbow_angle"] == "180.0", row["left_elbow_angle"]
    assert row["left_shoulder_angle"] != "", "肩关节角没写进去"
    # ⚠️ 断言的是**右手**的指角:模型说 "Left" ⟹ 翻转后这只手进的是右手槽
    #    (见 test_hand_handedness_routing)。写 left_ 会得到一个空值,而那不是缺陷。
    assert row["right_thumb_angle"] == "180.0", (
        f"手指角度没写进去或不对:{row['right_thumb_angle']!r}")
    assert row["left_thumb_angle"] == "", "左手槽这一帧没收到手,指角不该有值"


def test_upper_body_columns_are_no_longer_structurally_zero(monkeypatch, tmp_path):
    """★ `UpperBodyAnalyzer` 此前只在 examples 里被构造过 ⟹ head_*/torso_* 恒 0。

    红法:把 `analyzers['upper_body'].update(...)` 去掉 —— 这几列又回到全 0。
    """
    loggers = _wired(monkeypatch, tmp_path, pose=_pose33())
    _post()
    row = _row(loggers)
    assert row["head_jitter"] == "0.06", row["head_jitter"]
    assert row["torso_jitter"] == "0.08", row["torso_jitter"]


def test_world_columns_carry_the_metric_versions(monkeypatch, tmp_path):
    """★ world(米制 3D)那八列 —— 与取景脱钩的那一份,是那批"取景代理"封停的解药。"""
    loggers = _wired(monkeypatch, tmp_path, pose=_pose33(), world=_pose33())
    _post()
    row = _row(loggers)
    assert row["left_arm_angle_world"] == "33.0", row["left_arm_angle_world"]
    assert row["left_wrist_jitter_world"] == "0.04", row["left_wrist_jitter_world"]
    assert row["left_shoulder_jitter_world"] == "0.02", row["left_shoulder_jitter_world"]


def test_unfed_slot_is_empty_not_fifty(monkeypatch, tmp_path):
    """★ 本帧没收到手的槽 ⟹ **空格子**,不是 50.0(分析器默认值)。

    红法:把端点的 `_fresh(...)` 换回 `analyzers[key].get_results()` ——
    "没测到"立刻变回一个看着合法的 50。实测该场:只有一只手却两槽都写 50.0。
    """
    loggers = _wired(monkeypatch, tmp_path, pose=_pose33(),
                     hands=[_hand21()], handedness=[("Left", 0.9)])   # 只有一只手
    _post()
    row = _row(loggers)
    assert row["right_hand_model_label"] == "Left"        # 翻转后进了右手槽
    assert row["left_hand_score"] == "", f"没收到手的槽写了值:{row['left_hand_score']!r}"
    assert row["right_hand_score"] == "77.0"


def test_no_pose_means_empty_not_stale(monkeypatch, tmp_path):
    """没有姿态的帧:肩/臂/上半身那几列**空**,而不是写上一帧的旧值。

    红法:去掉 `_fresh` 的 fed 判断 —— 拿到的会是分析器**保留的上一帧结果**,
    于是"上一帧的度量"顶替"这一帧的度量",不留任何痕迹。
    """
    loggers = _wired(monkeypatch, tmp_path, pose=_pose33())
    _post()                                   # 第一帧有姿态
    first = _row(loggers)
    assert first["shoulder_score"] == "61.0"

    # 第二帧没有姿态(同一批分析器对象,状态还在)
    loggers2 = _wired(monkeypatch, tmp_path, pose=None)
    gesture_app.session_loggers[SID] = loggers2[SID]
    _post()
    rows = list(csv.DictReader(open(loggers2[SID][0].log_file, encoding="utf-8")))
    assert rows[-1]["shoulder_score"] == "", \
        f"没有姿态的帧写了旧值:{rows[-1]['shoulder_score']!r}"
