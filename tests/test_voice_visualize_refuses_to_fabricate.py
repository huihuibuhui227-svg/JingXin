# tests/test_voice_visualize_refuses_to_fabricate.py
"""`voice_interaction/utils/visualize.py` 不许拿**随机假数据**顶替真日志。

## 查清的事实(2026-10-24 夜)

`visualize_voice_log()` 读进 CSV 之后有一句:

    if df.empty or len(df) <= 1:
        print("⚠️ 日志为空，生成示例数据...")
        df = generate_sample_data()

`generate_sample_data()` 是 `np.random.seed(42)` + `uniform` 造出来的 10 行,
**每张图都照样画、照样存盘、函数照样 `return True`**。于是一条"这一场什么都没采到"
的结论,会被渲染成一张看起来正常的多维分析图 —— 这正是本仓在杀的形态
(「没产出不许报成功」)。

更糟的是它的**触发面**:`len(df) <= 1` 意味着**只有一行**的真日志也算"空"。
真日志里单行是存在的(试跑场、只答了一题的场),那时它会把那一行**扔掉**,
换成 10 行随机数。

## 这个文件守什么

空/单行 ⟹ 明说"没有数据可画",`return False`,**一个文件都不写**。
"""
import csv
from pathlib import Path

import pytest

viz = pytest.importorskip("voice_interaction.utils.visualize")


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    d = tmp_path / "out"
    monkeypatch.setattr(viz, "OUTPUT_DIR", d)
    return d


def _write(path: Path, rows: int) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "timestamp", "pitch_mean", "pitch_variation",
                    "energy_mean", "speech_ratio", "duration_sec"])
        for i in range(rows):
            w.writerow(["20260927_212143_95a8", f"2026-09-27T21:2{i}:00",
                        180.0 + i, 20.0, 0.04, 0.8, 12.0])
    return path


def test_empty_log_is_refused_instead_of_fabricated(tmp_path, out_dir):
    """★ 红法:把 `visualize_voice_log` 里那句 `return False` 改回
    `df = generate_sample_data()`。

    断言是"输出目录里一个文件都没有",不是"返回值是 False" ——
    只看返回值的话,先造假再返回 True 也算"没通过",但先造假再返回 False
    同样会骗过一条只看返回值的断言,而盘上已经多了一张假图。
    """
    empty = _write(tmp_path / "empty.csv", rows=0)

    ok = viz.visualize_voice_log(empty, "interview")

    assert ok is False, "空日志被当成成功"
    assert not out_dir.exists() or not list(out_dir.iterdir()), (
        f"空日志却画出了图:{[p.name for p in out_dir.iterdir()] if out_dir.exists() else []}"
    )


def test_single_row_log_is_not_replaced_by_random_data(tmp_path, out_dir):
    """只有一行**是数据**,不是空 —— 原先 `len(df) <= 1` 会把这一行扔掉换成 10 行随机数。

    这条不要求它画图(单行画多维图没有意义),只要求它**不要造假**:
    要么如实拒绝,要么真用那一行。判据是"输出里没有随机造出来的量" ——
    随机造的那批 `pitch_mean` 落在 150~300、`question_index` 是 1~10,
    而真值是 180.0。这里用"拒绝"这条更保守的路。
    """
    one = _write(tmp_path / "one.csv", rows=1)

    ok = viz.visualize_voice_log(one, "interview")

    assert ok is False, "单行日志被当成空日志处理并造了假数据"
    assert not out_dir.exists() or not list(out_dir.iterdir())


def test_generate_sample_data_is_not_reachable_from_the_render_path():
    """结构钉子:渲染路径里不许再出现"造数据"这个动作。

    红法:把 `df = generate_sample_data()` 加回 `visualize_voice_log`。
    用源码扫描而不是行为断言,是因为它一旦回来,**行为断言只会在"恰好跑到空日志"
    时才红**;而这条无论哪次运行都会红。
    """
    import inspect
    src = inspect.getsource(viz.visualize_voice_log)
    assert "generate_sample_data" not in src, (
        "渲染路径又接回了随机示例数据 —— 那会把「没采到」渲染成一张看起来正常的图"
    )
