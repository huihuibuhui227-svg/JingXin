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
    best_below_bar,
    is_detected,
    probe_crop,
    spread_indices,
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


# ──────────────────────────────────────────────────────────────
# spread_indices：★ 探针采样必须铺满全场，不能只取前 N 帧
# ──────────────────────────────────────────────────────────────

def test_spread_is_not_just_the_first_n():
    """★★ 核心回归：均匀采样**不能**退化成"前 n 帧"。

    这条红了 ⟹ 又回到取前 12 帧，录制开始时人还没坐下的场次会被**误判成救不回来**。
    实测 `..._afb2`：前 12 帧全 0/12，均匀 12 帧 center40 = 8/12 —— 就是被这个冤枉的。
    """
    idx = spread_indices(298, 12)
    assert idx != list(range(12)), "又变成取前 12 帧了"
    assert len(idx) == 12


def test_spread_reaches_both_ends():
    """首尾都要取到：首帧反映"开场就坐定"，末帧反映"人后来才入座"，两头都不能丢。

    这条红了 ⟹ 人只在末段出现的场次（如 `..._1320`，检出散布在帧 14~1490）
    探针根本看不到。
    """
    idx = spread_indices(298, 12)
    assert idx[0] == 0, "没取到首帧"
    assert idx[-1] == 297, "没取到末帧"
    assert max(idx) == 297


def test_spread_is_deterministic_sorted_and_unique():
    """同一个输入必须给同一串下标（否则结果不可复现）。

    这条红了 ⟹ 探针结果每次都不同，闸门/结论无法复核。
    """
    assert spread_indices(1000, 12) == spread_indices(1000, 12)
    idx = spread_indices(1000, 12)
    assert idx == sorted(idx), "下标没排序"
    assert len(idx) == len(set(idx)), "下标有重复"


def test_spread_short_input_returns_everything():
    """帧数不足时全取，不造不存在的下标。

    这条红了 ⟹ 会拿越界下标去读盘（`..._1d00` 只有 25 帧、`..._87c9` 只有 3 帧）。
    """
    assert spread_indices(5, 12) == [0, 1, 2, 3, 4]
    assert spread_indices(3, 3) == [0, 1, 2]
    assert spread_indices(1, 12) == [0]


def test_spread_handles_degenerate_n():
    """`n<=0` 给空表，不能除零。

    这条红了 ⟹ `n=1` 时那个 `(n-1)` 分母会 ZeroDivisionError。
    """
    assert spread_indices(10, 0) == []
    assert spread_indices(10, 1) == [0]
    assert spread_indices(0, 5) == []


def test_spread_on_a_late_appearing_subject_would_see_the_tail():
    """★ 场景化：人只在**后半段**出现的场次，采样必须覆盖到后半段。

    模拟 `..._90e7` / `..._1320`——人后面才入座。若采样只覆盖前 1/3，
    这类场次永远探不到人。
    这条红了 ⟹ 后半段没人被探到，那类场次会被判"救不回来"。
    """
    total = 1491
    idx = spread_indices(total, 12)
    tail = [i for i in idx if i >= total * 2 / 3]
    assert len(tail) >= 3, f"后半段只采到 {len(tail)} 帧，覆盖不足: {idx}"


# ──────────────────────────────────────────────────────────────
# best_below_bar：绝对线没跨过时，谁值得让整场重放来裁决
# ──────────────────────────────────────────────────────────────

def _stat(**hits_by_crop):
    """造一个 probe_crop 风格的 stat。"""
    return {"probed": 12, "tried": [
        {"crop": k, "probed": 12, "hits": v, "need": 4} for k, v in hits_by_crop.items()
    ]}


def test_below_bar_never_forces_a_crop_when_everything_is_zero():
    """★★ 防"硬凑"：全阶梯都是 0 ⟹ 必须返回 None。

    这是**对照实验**那一侧：实测空房间(`..._35e1`)、全黑帧(`..._3d8f`)、
    脸被水瓶挡(`..._1d00`) 在"前 12 帧"和"均匀 12 帧"两种采样下**都是 0**。
    如果这条红了 ⟹ 那几场会被硬塞一个裁切、把"没测到"写成"测到了"。

    实测牙齿（反向复现得到的**确切**对应关系）：
    - 去掉**全部闸门**、无条件返回最佳非-full 级 ⟹ 这条**红**（连同下面两条一起红）；
    - 只把 `min_hits` 从 2 降到 1 ⟹ 这条**不红**（全 0 时 `0 >= 1` 仍为假），
      那种改动由 `test_below_bar_rejects_a_single_frame_fluke` 负责。
    写清楚是因为本仓栽过"声称有牙、实测不红"的坑
    （见 `test_face_live_framing.py` 里那条已更正的说明）。
    """
    assert best_below_bar(_stat(full=0, center50=0, center40=0, center30=0)) is None


def test_below_bar_lets_a_clearly_better_rung_through():
    """★ 明显强于全画幅的一级要放行 —— 12 帧探针只是代理，整场重放才是裁判。

    实测 `..._1320`：full 0/12、center50 3/12（相对它自己 6% 的基线是 4 倍）。
    这条红了 ⟹ 这种"明显更强但没跨 1/3 绝对线"的取景永远轮不到重放裁决。
    """
    picked = best_below_bar(_stat(full=0, center50=3, center40=1, center30=0))
    assert picked is not None
    name, crop = picked
    assert name == "center50"
    assert crop == (0.25, 0.00, 0.75, 0.75), "裁切框要与阶梯里的一致"


def test_below_bar_rejects_a_single_frame_fluke():
    """只中 1 帧不算数：不值得为一次偶然重放整场（重放代价是分钟级）。

    这条红了 ⟹ 会把单帧偶然当成证据。
    """
    assert best_below_bar(_stat(full=0, center50=1, center40=0, center30=0)) is None


def test_below_bar_requires_strictly_better_than_full():
    """★ 必须**严格强于**全画幅 —— 平手时没有理由改（改了只会让数据漂移）。

    这条红了 ⟹ 全画幅本来检出的场次会被无谓地换成裁切（190 列里 142 列会漂）。
    """
    assert best_below_bar(_stat(full=5, center50=5, center40=3, center30=1)) is None
    assert best_below_bar(_stat(full=5, center50=4, center40=6, center30=1)) is not None


def test_below_bar_picks_the_best_not_the_first():
    """取**命中最多**的那级，不是阶梯里第一个够格的。

    这条红了 ⟹ 会退化成"遇到第一个及格的收手"，实测姬欣怡那场就是这么吃亏的。
    """
    picked = best_below_bar(_stat(full=0, center50=2, center40=3, center30=5))
    assert picked is not None and picked[0] == "center30"


def test_below_bar_tolerates_missing_or_broken_stat():
    """统计缺失时给 None，不许抛。

    这条红了 ⟹ 探针异常会连带把整场处理打挂（取景是优化，不该让流程崩）。
    """
    assert best_below_bar({}) is None
    assert best_below_bar(None) is None
    assert best_below_bar({"tried": []}) is None
    assert best_below_bar({"tried": [{"crop": "full", "hits": 0}]}) is None
