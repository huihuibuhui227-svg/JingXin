# tests/test_gesture_reframe_criterion.py
"""手势 acceptance ② 的**替代判据** —— 判据函数本身的钉子(不碰模型、不碰媒体)。

为什么需要这条钉子(2026-09-27,M3.5):②原话是「该列不再随人离镜头远近变(平移/缩放
裁剪框该列不变)」。真像素实测(见 `experiments/gesture_reframe_probe.py` 的三条腿)
证明**绝对不变不可达**:重取景必然改构图与外侧像素,而模型对它们的响应是它自己的
固有噪声(实测:同一串插值、**构图完全不动**的 `blur` 腿就足以让那 6 个米制列的中位
偏差到 2–4%)。所以判据只能改成**相对**的:

    中位 |ln(本列在扰动下的逐帧比值)|  ≤  **同一扰动下控制臂的 p90**

控制臂 = 米制那 6 个 `*_jitter_world` 列(它们不除肩宽,本批对它们是中性的)——
"本列的残留不超过模型自己在同一扰动下的固有响应"。

⚠️ **本文件钉的是判据函数(阈值、比较方向、控制臂取哪个统计量),不是实测结论** ——
实测数字在三场正式素材的 `reframe_*.csv` 上,那是 `--mode analyze` 的活;这里用合成
序列把判据的**行为**钉死:方向反了、把 p90 换成中位数、把 `abs` 去掉,都会红。

红法(逐条,都是生产改动):
  ① `verdict()` 里 `dev <= bar` 改成 `dev >= bar` ⟹ 两个"该 within"的用例红;
  ② `quantiles(logs, qs=(0.90,))[0]` 改成中位数 ⟹ 第二个用例红(见下面对照组的说明);
  ③ 去掉 `abs(math.log(...))` ⟹ 比值小于 1 的那一列(它的 ln 是负的)在"该红"的用例上变绿。
"""
import importlib

probe = importlib.import_module("experiments.gesture_reframe_probe")


def _ratios(n, value):
    return [value] * n


# 控制臂:8 帧,比值 1.0/1.02/1.05/1.10/0.95/0.90/1.30/0.80
# ⟹ |ln| 的 p90 = 0.2231(1.25 那一档),中位数 ≈ 0.0513
_CONTROL = [1.0, 1.02, 1.05, 1.10, 0.95, 0.90, 1.30, 0.80]


def test_verdict_is_within_when_the_column_is_no_worse_than_its_own_control():
    """本列的中位偏差落在控制臂的 p90 之内 ⟹ `within`(三场实测里 116/130 格是这样)。"""
    assert probe.verdict(_ratios(50, 1.10), _CONTROL) == "within", (
        "一列只动 10%(|ln|=0.095)却判成超出 —— 判据比模型自己的固有响应还严")


def test_verdict_is_within_exactly_at_the_bar_and_exceeds_just_over_it():
    """★ 边界是**闭区间**:恰好等于控制臂 p90 算达标,刚过一点就算超出。

    这一条同时挡住"把 p90 换成中位数"的改写:控制臂的中位数 ≈ 0.051,而 p90 = 0.223 ——
    换成中位数时,下面这个 `within` 用例(0.20 < 0.223)会变成 `exceeds` ⟹ 红。
    """
    within = _ratios(50, 1.2231)          # |ln| = 0.2014 < 0.2231
    over = _ratios(50, 2.0)               # |ln| = 0.6931 > 0.2231
    assert probe.verdict(within, _CONTROL) == "within"
    assert probe.verdict(over, _CONTROL) == "exceeds", (
        "一列整体翻了一倍(比控制臂的 p90 大 3 倍)却判成达标 —— 判据失去区分力")


def test_verdict_looks_at_the_magnitude_not_the_sign():
    """★ 比值 < 1(列变小)与 > 1(列变大)同等对待 —— 少了 `abs` 就会把变小当成"没问题"。"""
    for r in (1.0 / 2.0, 2.0):
        assert probe.verdict(_ratios(50, r), _CONTROL) == "exceeds", (
            f"比值 {r} 的整体变化没被判出来(少了 abs?)")


def test_verdict_is_insufficient_without_enough_frames():
    """配对数不够 ⟹ `insufficient`(不是"默认达标"—— 那会把没测到说成合格)。"""
    assert probe.verdict([1.10, 1.10, 1.10], _CONTROL) == "insufficient"
    assert probe.verdict(_ratios(50, 1.10), [1.0, 1.0]) == "insufficient"
