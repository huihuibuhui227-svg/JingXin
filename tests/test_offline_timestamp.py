# tests/test_offline_timestamp.py
"""M2.5:离线路径的时间戳 = `帧序号 × FRAME_SKIP / src_fps`(spec §4 表)。"""

from experiments.extract_features import offline_timestamp_ms


def test_timestamp_is_frame_index_times_skip_over_src_fps():
    """30 fps 源、每 3 帧取 1 帧 → 每个提交帧相隔 100 ms。

    红法:传 `src_fps` 而不是 `src_fps / FRAME_SKIP`(等于沿用旧行为)——
    第 3 帧会给出 100 而不是 300。
    """
    assert offline_timestamp_ms(0, 3, 30.0) == 0
    assert offline_timestamp_ms(1, 3, 30.0) == 100
    assert offline_timestamp_ms(3, 3, 30.0) == 300


def test_zero_or_nan_src_fps_falls_back_instead_of_dividing_by_zero():
    """★ Review Focus ④:容器读不出帧率时不许除零、不许产 NaN。

    红法:去掉守卫 —— 第一个断言会 ZeroDivisionError,
    第二个会把 `nan` 一路写进 CSV。
    """
    assert offline_timestamp_ms(1, 3, 0.0) == 100        # 回退 30 fps
    assert offline_timestamp_ms(1, 3, float("nan")) == 100


def test_returns_int():
    """时间戳必须是 int(mediapipe 吃 int;float 会在某些帧率下退化成同一个值)。"""
    assert isinstance(offline_timestamp_ms(7, 3, 29.97), int)
