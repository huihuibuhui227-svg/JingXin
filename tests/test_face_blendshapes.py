# tests/test_face_blendshapes.py
"""面部的 blendshape 头:模型的这一半输出从没进过日志,2026-09-26 接上。

背景(实测):`models/mediapipe/face_landmarker.task` 输出 **52** 个 blendshape 分数
(ARKit 风格:`eyeBlinkLeft` / `jawOpen` / `browDownLeft` …),而
`FaceLandmarkerOptions` **没有开** `output_face_blendshapes` ⟹ 全仓 `grep blendshape`
零命中。日志里那 13 个 `au*` 列是 `au_calculator.py` 的**手写几何比率**
(实测 `au7_eye_squeeze` 229/229 帧恰好 = `1 - avg_ear` ⟹ 零额外信息)。

本文件钉三件事:
  ① 工厂真的开了那个开关(不开关,后面全白搭);
  ② 模型给的名字**按名字**落进字典,不是按下标猜;
  ③ 没检出脸时是**空**,不是 52 个 0 —— 编 0 会让"没测到"看起来像"测到了 0"。
"""
import numpy as np

from face_expression.models.blendshapes import (BLENDSHAPE_COLUMNS,
                                                BLENDSHAPE_NAMES, blend_column,
                                                unknown_names)
from face_expression.pipeline.detector import FaceDetector

_FRAME = np.zeros((48, 48, 3), dtype=np.uint8)
_HOLDER = type("H", (), {"x": 0.5, "y": 0.5})


class _Category:
    def __init__(self, name, score):
        self.category_name, self.score = name, score


class _FakeLandmarker:
    """假探测器:交 landmarks + blendshapes,形状与 tasks 的返回值一致。"""

    def __init__(self, *, faces=True, blendshapes=True, extra=()):
        self._faces, self._bs, self._extra = faces, blendshapes, extra
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        groups = [[_HOLDER(), _HOLDER()]] if self._faces else []
        bs = None
        if self._faces and self._bs:
            bs = [[_Category("eyeBlinkLeft", 0.42), _Category("jawOpen", 0.07),
                   _Category("browDownRight", 0.31)]
                  + [_Category(n, 0.5) for n in self._extra]]
        return type("R", (), {"face_landmarks": groups, "face_blendshapes": bs})()

    def close(self):
        self.closed = True


def _detector(**kw):
    lm = _FakeLandmarker(**kw)
    return FaceDetector("unused.task", factory=lambda *a, **o: lm), lm


def test_face_detector_factory_enables_the_blendshape_output(monkeypatch):
    """★ 根子上的一条:不开 `output_face_blendshapes`,模型就算不出来。

    红法:把那行去掉 —— `options` 里是 False,整条链路上 52 列全是空的。

    (`_default_factory` 内部才 import mediapipe —— 本文件因此仍能跑在没装
     mediapipe 的机器上,这里用 monkeypatch 换掉那个类方法。)
    """
    from pathlib import Path

    from mediapipe.tasks.python import vision

    captured = {}

    def fake_create(options, *args, **kwargs):
        # `create_from_options(options)` 是**位置传参** —— 收 kwargs 是收不到的
        captured["options"] = options
        return _FakeLandmarker()

    monkeypatch.setattr(vision.FaceLandmarker, "create_from_options",
                        staticmethod(fake_create))

    from face_expression.pipeline.detector import _default_factory
    _default_factory(Path("unused.task"), num_faces=1,
                     min_detection_confidence=0.8, min_tracking_confidence=0.8)

    opts = captured.get("options")
    assert opts is not None, "工厂没把 options 交给 create_from_options"
    assert getattr(opts, "output_face_blendshapes", None) is True, \
        f"blendshape 输出没开,模型这一半等于不存在:{opts!r}"


def test_categories_land_in_the_dict_by_name_not_by_index():
    """按**名字**取,不按下标 —— 下标会随模型版本漂移,而名字不会。"""
    d, _ = _detector()
    landmarks, bs = d.detect_with_blendshapes(_FRAME, 0)
    assert landmarks and len(landmarks) == 2, "landmark 那条路不该受影响"
    assert bs["eyeBlinkLeft"] == 0.42 and bs["jawOpen"] == 0.07, bs
    assert set(bs) == {"eyeBlinkLeft", "jawOpen", "browDownRight"}, bs


def test_detect_keeps_the_old_contract():
    """`detect()` 的契约逐字不变(`[(x, y)]` 或 None)—— test_detector_contract 钉着它。"""
    d, _ = _detector()
    assert d.detect(_FRAME, 0) == [(0.5, 0.5), (0.5, 0.5)]


def test_no_face_is_empty_not_fifty_two_zeros():
    """没检出脸 ⟹ 空字典,且序列化出来的行里**一个 bs_ 键都没有**。

    红法:把空字典换成 `{n: 0.0 for n in BLENDSHAPE_NAMES}` —— 于是"这一帧没人脸"
    看起来像"测到 52 个 0 分",而 0 是个合法分数。
    """
    from face_expression.models.results import AnalysisFrameResult

    d, _ = _detector(faces=False)
    landmarks, bs = d.detect_with_blendshapes(_FRAME, 0)
    assert landmarks is None and bs == {}, (landmarks, bs)

    au = __import__("face_expression.models.features", fromlist=["AUFeatures"]).AUFeatures()
    from face_expression.models.results import (EmotionResult, TensionResult)
    from face_expression.models.features import MicroExpressionResult, TemporalStats
    row = AnalysisFrameResult(
        session_id="s", timestamp=0.0, focus_score=0.0, au_features=au,
        temporal_stats=TemporalStats(data={}),
        micro_expressions=MicroExpressionResult(data={}),
        emotion_result=EmotionResult({}, "neutral", 0.0, [], ""),
        tension_result=TensionResult(0.0, "low", {}),
        blendshapes=bs).to_dict()
    assert not [k for k in row if k.startswith("bs_")], \
        f"没检出脸的帧里出现了 blendshape 列:{[k for k in row if k.startswith('bs_')][:5]}"


def test_unknown_model_names_are_reported_not_swallowed():
    """模型给的表外名字要能**被报出来**(换模型版本时先响,而不是某列永远空着)。"""
    assert unknown_names(["eyeBlinkLeft", "brandNewAU"]) == ["brandNewAU"]
    assert unknown_names(["eyeBlinkLeft"]) == []


def test_the_name_table_is_the_52_the_model_actually_gives():
    """表里必须是 52 个,且列名由 `blend_column` 统一生成(别处不许拼字符串)。"""
    assert len(BLENDSHAPE_NAMES) == 52, len(BLENDSHAPE_NAMES)
    assert len(set(BLENDSHAPE_NAMES)) == 52, "有重名"
    assert blend_column("jawOpen") == "bs_jawOpen"
    assert len(BLENDSHAPE_COLUMNS) == 52
    # 抽查几个"有依据的 AU 近似"确实在表里(它们是接这一半输出的全部理由)
    for must in ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "browDownLeft",
                 "cheekSquintLeft", "mouthSmileLeft", "noseSneerRight"):
        assert must in BLENDSHAPE_NAMES, must


def _row(**kw):
    from face_expression.models.features import (AUFeatures,
                                                 MicroExpressionResult,
                                                 TemporalStats)
    from face_expression.models.results import (AnalysisFrameResult,
                                                EmotionResult, TensionResult)
    return AnalysisFrameResult(
        session_id="s", timestamp=0.0, focus_score=0.0,
        au_features=AUFeatures(), temporal_stats=TemporalStats(data={}),
        micro_expressions=MicroExpressionResult(data={}),
        emotion_result=EmotionResult({}, "neutral", 0.0, [], ""),
        tension_result=TensionResult(0.0, "low", {}), **kw).to_dict()


def test_serialized_row_carries_the_scores_with_the_bs_prefix():
    """模型给的分数要**原样**出现在这一行里,列名带 bs_ 前缀(与手写比率的 au* 区分)。

    红法:去掉 `to_dict()` 里那段展开 —— 模型算了,日志里一个数都没有(这正是
    2026-09-26 之前的状态,只不过那时是连算都没算)。
    """
    row = _row(blendshapes={"eyeBlinkLeft": 0.42, "jawOpen": 0.0666})
    assert row.get("bs_eyeBlinkLeft") == 0.42, row.get("bs_eyeBlinkLeft")
    assert row.get("bs_jawOpen") == 0.067, row.get("bs_jawOpen")   # 与 au* 同精度(3 位)
    # 手写比率那些列一个都不能少(旧列不动)
    assert "au7_eye_squeeze" in row and "avg_ear" in row


def test_logger_writes_the_52_columns(tmp_path, monkeypatch):
    """★ 断链守卫:表里 52 个名字 ⟹ 日志文件里 52 个列头 ⟹ 值真的落到那张表里。

    红法:logger 那行 `+ list(BLENDSHAPE_COLUMNS)` 去掉 —— 模型给了数,CSV 里没有列,
    下次读日志的人只会看到"文件里没这个东西"。
    """
    import face_expression.utils.logger as lg
    monkeypatch.setattr(lg, "LOGS_DIR", str(tmp_path))
    log = lg.DataLogger("video", session_id="20260926_120000_bbbb")

    assert set(BLENDSHAPE_COLUMNS) <= set(log.fieldnames), "日志表里没有 blendshape 列"
    assert log.fieldnames[-1] == BLENDSHAPE_COLUMNS[-1], \
        "52 列该加在**末尾**:加在中间会让老文件的列序对不上"

    log.log(_row(blendshapes={"eyeBlinkLeft": 0.42, "jawOpen": 0.07}))
    import csv as _csv
    with open(log.get_log_path(), newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert rows, "一行都没写进去"
    assert float(rows[0]["bs_eyeBlinkLeft"]) == 0.42, rows[0].get("bs_eyeBlinkLeft")
    assert len([c for c in log.fieldnames if c.startswith("bs_")]) == 52
