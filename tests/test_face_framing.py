"""`face_expression/pipeline/framing.py` 的钉子。

这里的每条断言都对着一个**实测踩过的**东西，不是补覆盖率：

- `is_detected` 的判据：我用"特征字典非空"当判据时，把 `emotion='no_face'`
  算成了检出，得出过**假的 100%**。这条钉住正确判据。
- `apply_crop(None)` 必须**原样返回**：全画幅是既有行为，能检出的场次一个字节
  都不许动（实测裁切会让 190 列里 142 列漂移）。
- `probe_crop` 的取景选择：必须**试完整条阶梯取最好的**，而不是遇到第一个及格的
  就收手（姬欣怡那场第一级够格但整场只有 79%）。
- 平手时保留**更不激进**的那一级。
"""
from __future__ import annotations

import numpy as np
import pytest

from face_expression.pipeline.framing import (
    CROP_LADDER,
    apply_crop,
    is_detected,
    probe_crop,
)


# ──────────────────────────────────────────────────────────────
# is_detected：★ 唯一正确的检出判据
# ──────────────────────────────────────────────────────────────

def test_no_face_dict_is_not_detected():
    """`VideoPipeline` 未检出时给的是**非空**字典 —— 不能用"非空"当判据。

    生产改动会让它变红：把 `is_detected` 改成 `bool(feats)`。
    """
    assert is_detected({"emotion": "no_face", "timestamp": 0.0}) is False


@pytest.mark.parametrize("feats,expected", [
    ({"emotion": "happy"}, True),
    ({"emotion": "neutral", "au12_smile": 0.3}, True),
    ({}, False),
    (None, False),
    ({"timestamp": 1.0}, True),      # 没有 emotion 键时**不**当作没检出（保守放行）
])
def test_is_detected(feats, expected):
    assert is_detected(feats) is expected


def test_first_ladder_rung_must_be_full_frame():
    """阶梯第一级必须是 `None`：能检出就不许改取景。"""
    name, crop = CROP_LADDER[0]
    assert (name, crop) == ("full", None)


# ──────────────────────────────────────────────────────────────
# apply_crop
# ──────────────────────────────────────────────────────────────

def test_apply_crop_none_is_identity():
    """`None` 必须**原样返回同一个对象**，不是等值副本 —— 全画幅路径零开销、零行为变化。"""
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    assert apply_crop(img, None) is img


def test_apply_crop_center50_halves_each_side():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = apply_crop(img, (0.25, 0.0, 0.75, 0.75))
    # 宽 200×(0.75-0.25)=100；高 100×(0.75-0.0)=75
    assert out.shape == (75, 100, 3)


def test_apply_crop_actually_slices_content():
    """确认裁的是**正确的那块**，不是随便切一块。"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0:50, 30:70] = 255                      # 在中央偏上放一块白
    out = apply_crop(img, (0.25, 0.0, 0.75, 0.75))
    assert out.max() == 255                     # 白块落在裁切窗口内
    out2 = apply_crop(img, (0.80, 0.80, 1.0, 1.0))
    assert out2.max() == 0                      # 右下角没有白块


# ──────────────────────────────────────────────────────────────
# probe_crop
# ──────────────────────────────────────────────────────────────

class _FakePipeline:
    """假 pipeline：只有当输入**比原图小**（即被裁过）时才"检出"。

    用来验证取景阶梯的选择逻辑，不需要 mediapipe。
    """

    def __init__(self, detect_above_zoom: float):
        self.detect_above_zoom = detect_above_zoom

    def process_frame(self, img, ts):
        h, w = img.shape[:2]
        zoom = 100.0 / max(h, 1)                # 假图恒为 100×100 起
        emo = "happy" if zoom >= self.detect_above_zoom else "no_face"
        return (None, None, {"emotion": emo, "timestamp": ts})


def _fake_factory(detect_above_zoom: float):
    def factory(tag: str):
        return _FakePipeline(detect_above_zoom)
    return factory


def _frames(n: int = 6):
    return [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(n)]


def test_probe_returns_none_when_nothing_works():
    """整条阶梯都检不到 ⟹ 返回 `none`，调用方据此保持"没检出"，不要硬凑。"""
    name, crop, stat = probe_crop(
        _frames(), list(range(6)),
        pipeline_factory=_fake_factory(999.0),   # 永远检不到
    )
    assert (name, crop) == ("none", None)
    assert len(stat["tried"]) == len(CROP_LADDER)


def test_probe_prefers_full_frame_when_it_works():
    """全画幅就能检出 ⟹ 必须选 `full`，不许无故裁切。"""
    name, crop, _ = probe_crop(
        _frames(), list(range(6)),
        pipeline_factory=_fake_factory(1.0),     # 全画幅即可
    )
    assert name == "full" and crop is None


def test_probe_picks_best_rung_not_first_passable():
    """★ 要**试完**取最好，不是遇到第一个及格的就停。

    这里让 `center50` 勉强及格（3/6），`center30` 满分（6/6）。
    只看第一级及格的实现会返回 `center50`，正确实现返回 `center30`。
    """
    class _Graded(_FakePipeline):
        def process_frame(self, img, ts):
            h, w = img.shape[:2]
            zoom = 100.0 / max(h, 1)
            # zoom: full=1.0, center50≈1.33, center40≈1.67, center30≈2.0
            emo = "happy" if zoom >= 1.6 else ("happy" if ts % 2 == 0 else "no_face")
            return (None, None, {"emotion": emo, "timestamp": ts})

    name, crop, stat = probe_crop(
        _frames(6), list(range(6)),
        pipeline_factory=lambda tag: _Graded(0),
    )
    assert name == "center40", f"应选检出最多的那级，实得 {name}（{stat}）"


def test_probe_tie_keeps_less_aggressive_rung():
    """平手时保留**更不激进**的那一级（阶梯从宽到窄）。"""
    name, crop, _ = probe_crop(
        _frames(6), list(range(6)),
        pipeline_factory=_fake_factory(1.1),     # 只要裁过就满分 ⟹ 全平手
    )
    assert name == "center50", f"平手时该留更宽的取景，实得 {name}"


def test_probe_empty_input_is_not_an_error():
    name, crop, stat = probe_crop([], [], pipeline_factory=_fake_factory(1.0))
    assert (name, crop) == ("none", None)
    assert "没有可用的帧" in stat["reason"]
