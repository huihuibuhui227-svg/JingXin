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
    finger_angles, joint_angle, pose_angles, shoulder_width)

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

    def update(self, _lm, **_kw):
        # `**_kw` = 一帧的 `timestamp_ms` / `shoulder_width`(M3 B3 起真分析器要它们)。
        # 本替身只用一个计数来回答"端点喂没喂过它",那两个量在这里不参与。
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


def _pose_with_shoulders(lm11, lm12):
    """33 点姿态,只把**左右肩**放到指定的归一化坐标(其余保持 `_pose33` 的占位)。

    `lm11` / `lm12` 传 `None` 表示**那一侧的肩缺了** —— mediapipe 的 33 点总是整份给,
    所以这只是替「模型这一帧没交出那一只肩」构造输入,不是在模拟常规帧。
    """
    pts = _pose33()
    if lm11 is not None:
        pts[11] = _Pt(*lm11)
    else:
        pts[11] = None
    if lm12 is not None:
        pts[12] = _Pt(*lm12)
    else:
        pts[12] = None
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


# ── 手势新列 hand_visible_*(2026-09-26 Task 6)────────────────────────
def test_hand_visible_is_one_when_an_attributed_hand_is_there(monkeypatch, tmp_path):
    """两只手、模型都给了 handedness ⟹ 两格都是 `1`。

    `("Left", …)` 翻过来是人的**右手**(模型按镜像输入判左右,见 test_hand_handedness_routing),
    所以第一只进 `right_hand` 槽、第二只进 `left_hand` 槽 —— 两槽都有**已署名**的手。

    红法:把端点里的 `hand_present=...` 去掉(或把 `_hand_visible_cells` 的
    `attributed` 判据反过来)⟹ 这一格变空。
    """
    loggers = _wired(monkeypatch, tmp_path, pose=_pose33(),
                     hands=[_hand21(), _hand21()],
                     handedness=[("Left", 0.9), ("Right", 0.95)])
    _post()
    row = _row(loggers)
    assert row["hand_visible_right"] == "1", row["hand_visible_right"]
    assert row["hand_visible_left"] == "1", row["hand_visible_left"]
    # 两列只能吐这两个值之一 —— `0` 不在字母表里(见 test_hand_visible_is_never_zero)
    for col in ("hand_visible_left", "hand_visible_right"):
        assert row[col] in ("1", ""), f"{col} 吐了字母表外的值:{row[col]!r}"


def test_hand_visible_does_not_count_an_unattributed_hand_as_left(monkeypatch, tmp_path):
    """★ 兜底分槽:槽非空但 `handedness_info` 为 `None` ⟹ 有手,但**不知道是哪只**。

    模型这一帧**没给** handedness ⟹ `api/app.py` 的兜底支路把这只手塞进空着的
    `left_hand` 槽,并把 `handedness_info['left_hand']` 置 `None`。此时:
      · 「这一槽**有手**」是真的 —— `left_hand_score` 有值(不是空);
      · 「那是**左手**」是**假**的 —— 没有依据。
    ⟹ `hand_visible_left` 必须是**空**,不是 `1`。

    红法:把判据写成「槽非空即为真」(丢掉 `attributed` 那一半)⟹ 本测试变红。
    """
    loggers = _wired(monkeypatch, tmp_path, pose=_pose33(),
                     hands=[_hand21()], handedness=[None])       # 一只手,无 handedness
    _post()
    row = _row(loggers)
    # 先证明这一帧**确实收到了手** —— 否则下面那条断言测的是"我没给手"
    assert row["left_hand_score"] != "", \
        f"这一帧左手槽没收到手,测试没有区分力:{row['left_hand_score']!r}"
    assert row["left_hand_model_label"] == "", "兜底那一槽不该有标签"
    assert row["hand_visible_left"] == "", \
        f"一只来路不明的手被写成了「左手可见」:{row['hand_visible_left']!r}"
    assert row["hand_visible_right"] == "", row["hand_visible_right"]


def test_hand_visible_is_empty_when_no_hand_is_seen(monkeypatch, tmp_path):
    """一帧没有手 ⟹ 两格都**空**。**不写 `0`** —— `0` 是「确定没有这只手」,与「没测到」两回事。"""
    loggers = _wired(monkeypatch, tmp_path, pose=_pose33())   # hands=() / handedness=()
    _post()
    row = _row(loggers)
    assert row["left_hand_score"] == "", "前提:这一帧没收到手"
    assert row["hand_visible_left"] == "", \
        f"没测到被写成了「确定没有」:{row['hand_visible_left']!r}"
    assert row["hand_visible_right"] == "", row["hand_visible_right"]


def test_hand_visible_is_never_zero(monkeypatch, tmp_path):
    """★ 字母表只有 `{1, 空}` —— **`0` 不是本列的合法值**(表里 `unit` 栏:布尔(1 / 空))。

    红法:把 `_hand_visible_cells` 的 `else ""` 改成 `else "0"`。
    """
    for hands, handedness in (((), ()), ([_hand21()], [None])):
        loggers = _wired(monkeypatch, tmp_path, pose=_pose33(),
                         hands=hands, handedness=handedness)
        _post()
        row = _row(loggers)
        for col in ("hand_visible_left", "hand_visible_right"):
            assert row[col] != "0", f"{col} 写了 0 —— 「没测到」被说成「测到了没有」"


def test_hand_visible_needs_the_hand_to_be_there_at_all(tmp_path, monkeypatch):
    """★ 「知道是哪只手」**且**「这只手在」—— 两个条件缺一不可,都缺就**空**。

    这一条走的是 `GestureLogger.log()` 的公开入参(端点那条路构造不出这个组合:
    `api/app.py` 只在**填槽的同时**写 `handedness_info`,所以「有署名、槽却空」在活路径
    不可达)。但表里 `definition` 白纸黑字写的是**两个条件的合取** —— 这条钉子守的就是
    那个合取,免得后来人把 `and present.get(slot)` 当成多余的一半删掉。

    红法:删掉 `and present.get(slot)` ⟹ 本测试第一格变 `1`。
    """
    monkeypatch.setattr(gl, "LOGS_DIR", str(tmp_path / "logs"))
    log = gl.GestureLogger(session_id="20260926_120000_vvvv")
    log.log(left_hand_result=None, right_hand_result=None, shoulder_result=None,
            handedness_info={"left_hand": ("Right", 0.9)},        # 有署名…
            hand_present={"left_hand": False, "right_hand": True})  # …但槽里没有手
    rows = list(csv.DictReader(open(log.log_file, encoding="utf-8")))
    assert rows, "一行都没写"
    assert rows[-1]["hand_visible_left"] == "", \
        f"手不在,却被写成可见:{rows[-1]['hand_visible_left']!r}"
    # 反向的一半:有署名 + 手在 ⟹ 1(否则上面那条可以靠"永远写空"骗过去)
    log.log(left_hand_result={"resilience_score": 1.0}, right_hand_result=None,
            shoulder_result=None,
            handedness_info={"left_hand": ("Right", 0.9)},
            hand_present={"left_hand": True, "right_hand": False})
    rows = list(csv.DictReader(open(log.log_file, encoding="utf-8")))
    assert rows[-1]["hand_visible_left"] == "1", rows[-1]["hand_visible_left"]


# ── 手势协变量 shoulder_width(2026-09-26 Task 7)──────────────────────
# ⚠️ 单位是**归一化图像单位**,不是米。转米要相机内参,而本项目**没有任何内参来源**
#    (2026-09-26 全仓 grep `solvePnP` / `camera_matrix` / `Rodrigues` / 焦距:0 命中;
#    `models/` 下只有 4 个 `.task`、无标定文件)—— Task 8 已据此把 3D 头姿登记为 blocked。
#    这一列是**协变量**用途:报告层拿它作**分母**消掉取景/距离的影响(spec §4.5 规矩)。
#    下面这几条测试**证不了单位**(单位是约束不是可测事实),能测的是几何与缺失行为;
#    单位那一半由 `tests/test_l0_column_table.py` 的钉子钉在**表里那行**上。
def test_shoulder_width_is_the_normalized_distance_between_lm11_and_lm12():
    """双肩归一化图像坐标的**欧氏**距离 —— 轴对齐与斜着各验一遍。

    红法:把 `shoulder_width` 改成只取 |Δx|(斜着那个用例会得 0.3,而不是 0.5)。
    """
    assert shoulder_width(_pose_with_shoulders((0.3, 0.4), (0.7, 0.4))) == pytest.approx(0.4, abs=1e-6)
    assert shoulder_width(_pose_with_shoulders((0.3, 0.4), (0.6, 0.8))) == pytest.approx(0.5, abs=1e-6)


def test_shoulder_width_is_none_when_a_shoulder_is_missing():
    """★ 缺任一肩 ⟹ `None`(**不写 0**)。

    为什么 0 不行:在归一化坐标里 `0` 的意思是「两肩重合」—— 一个**看着像测量值**的
    假数,下游拿它做分母会得到 inf/NaN,而不是"这一帧没有这个量"。

    四种「缺」都验:单缺右肩 / 单缺左肩 / 点数不够(只有 13 个点)/ 整份姿态为 `None`。
    红法:把任一条缺失分支改成 `return 0.0`。
    """
    assert shoulder_width(_pose_with_shoulders((0.3, 0.4), None)) is None
    assert shoulder_width(_pose_with_shoulders(None, (0.7, 0.4))) is None
    assert shoulder_width([_Pt(0.5, 0.5)] * 12) is None
    assert shoulder_width(None) is None


def test_shoulder_width_reaches_the_log_row(monkeypatch, tmp_path):
    """★ 端点手里的 `pose_landmarks` 真的变成了日志里那一格(**归一化图像单位**)。

    红法:把 `angles_data["shoulder_width"] = ...` 从端点里去掉 ⟹ 该格变空。
    """
    loggers = _wired(monkeypatch, tmp_path,
                     pose=_pose_with_shoulders((0.30, 0.40), (0.70, 0.40)))
    _post()
    row = _row(loggers)
    assert row["shoulder_width"] != "", "新列没写进日志(端点没把手里的 pose 用上)"
    assert float(row["shoulder_width"]) == pytest.approx(0.40, abs=1e-6), row["shoulder_width"]


def test_shoulder_width_is_empty_when_one_shoulder_is_missing_in_the_live_path(monkeypatch, tmp_path):
    """★ 缺一只肩 —— 走**端点 → CSV** 整条路,该格必须是**空**,不是 `0`。

    活路径上 mediapipe 的 33 点总是整份给(要么整份 `None`),所以「只缺一只肩」这条
    分支由替身探测器构造。它守的是表里 acceptance ② 那句话,也是**下游最怕的那一格**:
    `0` 会被当成「肩宽 = 0」。

    红法:把 `shoulder_width` 的 `if a is None or b is None: return None` 改成 `return 0.0`。
    """
    loggers = _wired(monkeypatch, tmp_path,
                     pose=_pose_with_shoulders((0.30, 0.40), None))    # 右肩缺
    _post()
    row = _row(loggers)
    assert row["shoulder_score"] != "", "前提:这一帧是有姿态的(否则测的是「没姿态」那条)"
    assert row["shoulder_width"] == "", f"缺一只肩却写了值:{row['shoulder_width']!r}"


def test_shoulder_width_is_empty_not_zero_when_there_is_no_pose(monkeypatch, tmp_path):
    """没有姿态的帧 ⟹ **空**,不是 `0`。

    红法:把端点那句写成 `... or 0.0`,或把 `shoulder_width` 的缺失分支改成 `return 0.0`。
    """
    loggers = _wired(monkeypatch, tmp_path, pose=None)
    _post()
    row = _row(loggers)
    assert row["shoulder_score"] == "", "前提:这一帧确实没有姿态"
    assert row["shoulder_width"] == "", f"没有姿态却写了值:{row['shoulder_width']!r}"


# ── 两个手部单元格帮手的**分歧机制**(2026-09-26 Task 7 顺手修的说明)──────
def test_the_two_hand_cell_helpers_diverge_on_two_different_mechanisms():
    """★ `logger.py` 里那段"两处细微不同"的说明,机制**是两个**,不是一个。

    原话是「差别只在**假值但非 None** 的条目上(如 `("", 0.9)`)」——
    **`("", 0.9)` 是非空元组 = 真值**,`_handedness_cells` 走的是**正常分支**,
    根本不会碰 `("", "")` 那一支;它的 label 格空是**另一个原因**。这条测试把两种机制分开钉住:

      · **甲** `()`(假值非 None):那个三元把**整个条目**换成 `("", "")`
        ⟹ label 与 conf **两格都空**;而 `_hand_visible_cells` 判 `is not None` ⟹ `1`。
      · **乙** `("", 0.9)`(真值):正常解包 ⟹ conf 格**有值** `0.9`,
        label 格空是因为**模型给的 label 就是空串**(不是三元的功劳);`_hand_visible_cells` 也写 `1`。

    红法(都是"把两套判据统一起来"这种自然改动):
      · 把 `_hand_visible_cells` 的 `known = info.get(slot) is not None` 改成
        `known = bool(info.get(slot))` ⟹ 甲的第三条断言变红;
      · 把 `_handedness_cells` 改成「label 空即视为没有依据」⟹ 乙的 conf 格从 `0.9` 变空,变红。
    """
    G = gl.GestureLogger

    # 甲
    cells = G._handedness_cells({"left_hand": ()})
    assert cells["left_hand_model_label"] == ""
    assert cells["left_hand_model_label_conf"] == "", \
        "假值条目被当成了'有置信度'的条目 —— 三元那一支没生效"
    assert G._hand_visible_cells({"left_hand": ()},
                                {"left_hand": True})["hand_visible_left"] == "1", \
        "判据若是真值判断,这一格会是空 —— 那说明两套判据被统一了"

    # 乙
    cells = G._handedness_cells({"left_hand": ("", 0.9)})
    assert cells["left_hand_model_label"] == ""
    assert cells["left_hand_model_label_conf"] == pytest.approx(0.9), \
        "非空元组(真值)该走**正常分支** —— conf 有值;空的是 label 元素本身"
    assert G._hand_visible_cells({"left_hand": ("", 0.9)},
                                {"left_hand": True})["hand_visible_left"] == "1"
