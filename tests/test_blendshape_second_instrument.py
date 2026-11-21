# tests/test_blendshape_second_instrument.py
"""第二套仪器(`blendshape`)的单元测试。

用 `tmp_path` 造小日志,**不依赖真实数据** —— 真实数据会变(55 份日志的 bs 列
在 11 份里整列为空),测试不该跟着它漂。

钉住的是四件容易悄悄错的事:
1. 五个返回键一个不多一个不少(报告模板照这个形状读);
2. 上升沿定义与 `l1_face._rising_edges` 一致(首帧为真**不算**一次);
3. 缺列 / 全空 / 空文件 ⟹ `NaN` + 原因,**不抛异常、不拿 0 冒充**;
4. `bs__neutral` 不算"肌肉动作"(否则"没表情"会被算成"有表情");
5. 声明常量的出处必须写在返回里(`threshold_basis` 非空)。
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pandas as pd
import pytest

_HERE = Path(__file__).resolve().parent


def _candidates(name: str) -> list[Path]:
    out = []
    env = os.environ.get("JX_PATCH_DIR")
    if env:
        out.append(Path(env) / f"{name}.py")
    out.append(_HERE.parent / "patch" / f"{name}.py")
    out.append(_HERE / f"{name}.py")
    out.append(_HERE.parent / "report_frontend" / f"{name}.py")
    return out


def _load(name: str):
    for path in _candidates(name):
        if path.is_file():
            spec = importlib.util.spec_from_file_location(f"_jxp_{name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    tried = "\n".join(f"  - {p}" for p in _candidates(name))
    raise ImportError(f"找不到 {name}.py。试过:\n{tried}")


bs = _load("blendshape_second_instrument")

KEYS = {"construct", "blendshape_value", "n_frames", "threshold_used", "threshold_basis"}


def _write(tmp_path: Path, rows: list[dict], name: str = "face_au_log_test.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _log(n: int = 10, **overrides) -> list[dict]:
    """`n` 帧的合成日志:默认两端闭合、无表情、eyeBlink 全 0.1。"""
    rows = []
    for i in range(n):
        row = {
            "timestamp": float(i),                 # 1 Hz ⟹ 1 帧 = 1 秒
            "bs_mouthSmileLeft": 0.0, "bs_mouthSmileRight": 0.0,
            "bs_browDownLeft": 0.0, "bs_browDownRight": 0.0,
            "bs_eyeBlinkLeft": 0.1, "bs_eyeBlinkRight": 0.1,
            "bs_jawOpen": 0.0,
            "bs__neutral": 0.9,                    # 中性拉满:不该被当成肌肉激活
        }
        for key, seq in overrides.items():
            row[key] = seq[i]
        rows.append(row)
    return rows


# ── 形状 ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("construct", bs.CONSTRUCTS)
def test_result_has_exactly_the_five_agreed_keys(tmp_path, construct):
    path = _write(tmp_path, _log())
    got = bs.compute_second_instrument(path, construct)
    assert set(got) == KEYS
    assert got["construct"] == construct
    assert isinstance(got["blendshape_value"], float)
    assert isinstance(got["n_frames"], int)
    assert got["threshold_basis"], "阈值依据必须写出来 —— 空字符串等于没写"


@pytest.mark.parametrize("construct", bs.CONSTRUCTS)
def test_declared_thresholds_carry_a_basis(tmp_path, construct):
    """三个声明常量 + 一个分位水平,每个都要有依据文本。"""
    assert construct in bs.THRESHOLD_BASIS
    assert len(bs.THRESHOLD_BASIS[construct]) > 40


def test_unknown_construct_is_a_programming_error(tmp_path):
    """构念名打错是调用方的事,与数据无关 ⟹ 抛错,不静默返回 NaN。"""
    path = _write(tmp_path, _log())
    with pytest.raises(ValueError):
        bs.compute_second_instrument(path, "smile_ratio_v2")


def test_all_four_constructs_are_implemented():
    assert len(bs.CONSTRUCTS) >= 4
    for c in (bs.SMILE, bs.BROW, bs.BLINK, bs.ANY_AU):
        assert c in bs.CONSTRUCTS


# ── smile / brow:比率与双侧配对 ───────────────────────────────────────────────

def test_smile_ratio_counts_frames_above_threshold(tmp_path):
    # 第 2、3 帧双侧 0.6(>0.5)⟹ 2/10
    left = [0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    right = list(left)
    path = _write(tmp_path, _log(bs_mouthSmileLeft=left, bs_mouthSmileRight=right))
    got = bs.compute_second_instrument(path, bs.SMILE)
    assert got["blendshape_value"] == pytest.approx(0.2)
    assert got["n_frames"] == 10
    assert got["threshold_used"] == 0.5


def test_smile_uses_bilateral_mean_not_and_conjunction(tmp_path):
    """一侧 0.9、一侧 0.1 的帧,均值 0.5 —— 严格 `>` 不判激活。

    这条钉的是"别用 AND":六条 AU 取交集的旧构念实测恒为 0(见 l1_face 那段实测),
    双侧均值不会因为一侧低就整帧归零,也不会因为一侧高就整帧算数。
    """
    path = _write(tmp_path, _log(
        bs_mouthSmileLeft=[0.9] * 10, bs_mouthSmileRight=[0.1] * 10))
    assert bs.compute_second_instrument(path, bs.SMILE)["blendshape_value"] == 0.0
    path2 = _write(tmp_path, _log(
        bs_mouthSmileLeft=[0.9] * 10, bs_mouthSmileRight=[0.8] * 10),
        name="face_au_log_test2.csv")
    assert bs.compute_second_instrument(path2, bs.SMILE)["blendshape_value"] == 1.0


def test_frames_missing_one_side_are_excluded_from_both_sides(tmp_path):
    """配对:只有一侧有值的帧不进分母 —— 不在分母里混"半个观测"。"""
    rows = _log()
    rows[0]["bs_mouthSmileRight"] = float("nan")     # 第 0 帧缺右侧
    path = _write(tmp_path, rows)
    got = bs.compute_second_instrument(path, bs.SMILE)
    assert got["n_frames"] == 9


def test_brow_ratio_threshold_is_documented_as_low_resolution(tmp_path):
    path = _write(tmp_path, _log())
    got = bs.compute_second_instrument(path, bs.BROW)
    assert got["threshold_used"] == 0.5
    # 依据里必须写明"接近 0 是分辨率问题,不是此人从不皱眉"
    assert "分辨力" in got["threshold_basis"] or "分辨率" in got["threshold_basis"]


# ── blink:上升沿 ──────────────────────────────────────────────────────────────

def test_blink_counts_rising_edges_and_converts_to_per_minute(tmp_path):
    # 10 帧 1 Hz(跨度 9 秒)。闭眼在第 2–3 帧、第 6 帧 ⟹ 2 次上升沿
    seq = [0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
    path = _write(tmp_path, _log(bs_eyeBlinkLeft=seq, bs_eyeBlinkRight=seq))
    got = bs.compute_second_instrument(path, bs.BLINK)
    assert got["blendshape_value"] == pytest.approx(2 / (9 / 60))
    assert got["n_frames"] == 10


def test_first_frame_true_is_not_a_rising_edge(tmp_path):
    """首帧为真不算上升沿 —— 与 `l1_face._rising_edges` 同一约定。

    若把它算进去,两侧速率天生差 1 次,读的人会以为是仪器分歧。
    """
    seq = [0.9] + [0.1] * 9
    path = _write(tmp_path, _log(bs_eyeBlinkLeft=seq, bs_eyeBlinkRight=seq))
    got = bs.compute_second_instrument(path, bs.BLINK)
    assert got["blendshape_value"] == 0.0


def test_blink_without_timestamp_is_unavailable_not_zero(tmp_path):
    rows = _log()
    for r in rows:
        del r["timestamp"]
    path = _write(tmp_path, rows)
    got = bs.compute_second_instrument(path, bs.BLINK)
    assert math.isnan(got["blendshape_value"])
    assert got["threshold_basis"].startswith("算不出来")
    assert got["n_frames"] == 0


# ── 任一 AU 激活:逐列 p90 ────────────────────────────────────────────────────

def test_any_au_ratio_is_governed_by_the_quantile_level(tmp_path):
    """逐列 p90 是"本文件内"的分位 ⟹ 单列恰约 10% 的帧超阈,"任一"随列数上升。

    **这是分辨率自检量,不是行为量。** 测试把这个性质钉下来,免得下一个人
    把它当"表情丰富度"读进报告。真实 51 列日志上实测 0.80–0.96(见 `verify/`),
    这里只用合成数据证明"为什么"。
    """
    one = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    path = _write(tmp_path, _log(bs_jawOpen=one))
    got = bs.compute_second_instrument(path, bs.ANY_AU)
    assert got["threshold_used"] == bs.ANY_AU_QUANTILE
    assert got["blendshape_value"] == pytest.approx(0.1)
    assert "分辨率" in got["threshold_basis"]

    # 5 列各在**不同**帧上冲高 ⟹ 并集 5/10。列越多,"任一超阈"越接近 1。
    cols = {f"bs_col{k}": [0.9 if i == k else 0.1 for i in range(10)] for k in range(5)}
    path5 = _write(tmp_path, _log(**cols), name="face_au_log_five.csv")
    assert bs.compute_second_instrument(path5, bs.ANY_AU)["blendshape_value"] == pytest.approx(0.5)


def test_any_au_ignores_the_neutral_class(tmp_path):
    """只有 `bs__neutral` 在动 ⟹ 肌肉动作一个没激活,占比必须是 0。

    `bs__neutral` 是模型自带的"中性"类,不是肌肉动作(`blendshapes.py` 原话)。
    把它算进去等于"没有表情"也算激活。
    """
    neutral = [0.9] * 10
    path = _write(tmp_path, _log(bs__neutral=neutral))
    got = bs.compute_second_instrument(path, bs.ANY_AU)
    assert got["blendshape_value"] == 0.0


def test_any_au_constant_columns_never_activate(tmp_path):
    """常数列的 p90 就是该常数,严格 `>` 不会把"恒为同一值"判成激活(τ 退化)。"""
    path = _write(tmp_path, _log(bs_jawOpen=[0.5] * 10))
    assert bs.compute_second_instrument(path, bs.ANY_AU)["blendshape_value"] == 0.0


# ── 缺数据:报 NaN + 原因,不抛、不冒 0 ────────────────────────────────────────

@pytest.mark.parametrize("construct", bs.CONSTRUCTS)
def test_missing_blendshape_columns_are_unavailable(tmp_path, construct):
    rows = [{"timestamp": float(i), "au12_smile": 0.3} for i in range(5)]
    path = _write(tmp_path, rows)
    got = bs.compute_second_instrument(path, construct)
    assert math.isnan(got["blendshape_value"])
    assert got["n_frames"] == 0
    assert got["threshold_basis"].startswith("算不出来")


@pytest.mark.parametrize("construct", bs.CONSTRUCTS)
def test_all_blank_blendshape_columns_are_unavailable(tmp_path, construct):
    """列在、但整列为空(实测 55 份日志里 11 份是这样)⟹ 不可算。"""
    nan = [float("nan")] * 6
    path = _write(tmp_path, _log(6, bs_mouthSmileLeft=nan, bs_mouthSmileRight=nan,
                                 bs_browDownLeft=nan, bs_browDownRight=nan,
                                 bs_eyeBlinkLeft=nan, bs_eyeBlinkRight=nan,
                                 bs_jawOpen=nan, bs__neutral=nan))
    got = bs.compute_second_instrument(path, construct)
    assert math.isnan(got["blendshape_value"])
    assert got["n_frames"] == 0


def test_header_only_file_does_not_raise(tmp_path):
    """`data/logs/face_au_log_x.csv` 就是这种(0 行 / 有表头)。"""
    path = tmp_path / "face_au_log_empty.csv"
    path.write_text("timestamp,bs_mouthSmileLeft,bs_mouthSmileRight\n", encoding="utf-8")
    got = bs.compute_second_instrument(path, bs.SMILE)
    assert math.isnan(got["blendshape_value"])
    assert got["threshold_basis"].startswith("算不出来")


def test_truly_empty_file_does_not_raise(tmp_path):
    path = tmp_path / "face_au_log_blank.csv"
    path.write_text("", encoding="utf-8")
    got = bs.compute_second_instrument(path, bs.ANY_AU)
    assert math.isnan(got["blendshape_value"])


def test_missing_file_raises(tmp_path):
    """文件不存在是路径错,不是数据缺口 —— 这个要炸。"""
    with pytest.raises(FileNotFoundError):
        bs.compute_second_instrument(tmp_path / "nope.csv", bs.SMILE)


def test_compute_all_returns_every_construct(tmp_path):
    path = _write(tmp_path, _log())
    got = bs.compute_all(path)
    assert set(got) == set(bs.CONSTRUCTS)
    for construct, row in got.items():
        assert row["construct"] == construct
