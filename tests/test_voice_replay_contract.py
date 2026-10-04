# tests/test_voice_replay_contract.py
"""重放腿**必须**写 `VoiceLogger` 的契约形状,产不出的列必须**留空**(甲-4)。

## 为什么需要

`experiments/replay_retained.py --modality voice` 与活路径是**两个调用方、同一张表**。
它们一旦漂开,后果不是"某个数算错了",而是**那份 CSV 喂给报告时语音系的槽成片掉光**
—— 而报告不会报错,只会说「未采集到对应数据」。这正是甲-4 记的那个形态。

所以本文件钉三件事:

1. **行的形状只有一处实现**:`VoiceLogger.build_row` 的键序**逐列等于** `fieldnames`
   (不是"集合相等" —— 顺序也算契约,`log_prosody` 是按 `fieldnames` 写 CSV 的);
2. **产不出的列有登记**:契约 32 列 = 重放会写的 + `NOT_WRITTEN_BY_REPLAY`,不重不漏;
3. **登记为产不出的列,在行里真的是空的**(`None`),不是 0、不是占位符
   —— 免得将来有人"顺手补个 0"把「没算出来」写成「算出来是 0」。

红法(实测):把 `build_row` 里任一列的键**改名**(如 `voiced_prob_mean` → `voiced_prob`)
⟹ 第 1 条红(`DictWriter` 那边会静默留空,所以这条必须钉在**形状**上);
把 `is_valid` 从 `NOT_WRITTEN_BY_REPLAY` 里删掉 ⟹ 第 2 条红。
"""
from __future__ import annotations

import pytest

from voice_interaction.utils.logger import VoiceLogger
from voice_interaction.utils.replay_contract import (
    NOT_WRITTEN_BY_REPLAY,
    arrival_wall_of,
    columns_written_by_replay,
    infer_question_index,
)


@pytest.fixture()
def vlog(tmp_path):
    """一个**只用来问形状**的 logger(表头写进 tmp,不碰 data/logs)。"""
    return VoiceLogger(log_type="interview", log_dir=str(tmp_path), session_id="20260927_212143_95a8")


def test_the_row_shape_is_the_contract(vlog):
    """★ 行的键序必须**逐列等于** `fieldnames` —— 顺序也算契约。

    为什么不能用 `set(...) == set(...)`:`log_prosody` 用
    `DictWriter(fieldnames=self.fieldnames)`,**键名对不上时那一列静默留空**,
    而集合相等对"顺序错了"和"名字改了"都不响。实测过:把 `voiced_prob_mean`
    改名成 `voiced_prob`,集合断言红、但真实后果是报告读不到这一列。
    """
    row = vlog.build_row({}, question_index=0, emotion="", feedback="")
    assert list(row) == vlog.fieldnames, (
        "行的键序与契约不符 —— 这两处必须由**同一个实现**产生(`build_row`)"
        f"\n  行里多出:{sorted(set(row) - set(vlog.fieldnames))}"
        f"\n  行里缺少:{sorted(set(vlog.fieldnames) - set(row))}")


def test_contract_is_partitioned_into_written_and_registered_not_written(vlog):
    """★ 契约 32 列 = 重放会写的 + 登记为写不出的,不重不漏。"""
    written = columns_written_by_replay(vlog.fieldnames)
    not_written = list(NOT_WRITTEN_BY_REPLAY)
    assert set(written) | set(not_written) == set(vlog.fieldnames), (
        f"未归类的列:{sorted(set(vlog.fieldnames) - set(written) - set(not_written))}"
        f"\n  登记了但不存在的列:{sorted(set(not_written) - set(vlog.fieldnames))}")
    assert not (set(written) & set(not_written)), (
        f"同一列既说会写又登记为写不出:{sorted(set(written) & set(not_written))}")
    for col, why in NOT_WRITTEN_BY_REPLAY.items():
        assert why.strip(), f"{col} 登记为写不出,但没写为什么 —— 登记没理由是免罪符"


def test_registered_not_written_columns_are_empty_not_zero(vlog):
    """★ 登记为写不出的列,在行里必须是**空**(`None`),不是 0 / 空串 / 占位符。

    红法(实测):把 `build_row` 里 `"n_chars": prosody_data.get("n_chars")` 改成
    `... , 0)` ⟹ 本条红。那正是"把没算出来写成算出来是 0"的形态。
    """
    row = vlog.build_row({}, question_index=0, emotion="", feedback="", is_valid=None)
    bad = {c: row[c] for c in NOT_WRITTEN_BY_REPLAY if row[c] is not None}
    assert not bad, f"登记为写不出的列却有值(补 0 会把缺值伪装成测量):{bad}"


def test_replay_does_not_copy_the_live_is_valid_placeholder(vlog):
    """`is_valid` 必须是空的 —— 重放不复制一个它没有做出的判断。

    活路径恒写 `True`(`log_prosody` 的默认值,调用方从未传别的值),封停表里
    「恒 1 且下游从未使用」记的就是它。重放若照抄那个 `True`,那份文件就会
    声称做出了一个它没做的有效性判断。
    """
    assert "is_valid" in NOT_WRITTEN_BY_REPLAY
    row = vlog.build_row({}, question_index=0, emotion="", feedback="", is_valid=None)
    assert row["is_valid"] is None


# ── 两条推断规则 ────────────────────────────────────────────────────────
_LEDGER = [
    {"kind": "raw", "seq": 3, "received_at_wall": 100.500},
    {"kind": "converted", "seq": 3, "received_at_wall": 100.700},
    {"kind": "raw", "seq": 5, "received_at_wall": 200.000},
]


def test_arrival_wall_uses_the_raw_record_not_the_converted_one():
    """★ 到达时刻取 **raw** 记录的 `received_at_wall`。

    实测定位过:`_record()` 给**每条**留存记录各采一次 `time.time()`,同一次上传的
    `raw` 与 `converted` 因此相差 **150~283 ms**;`reaction_time` 用的是 raw 那条
    (与活日志吻合 **±5 ms**)。取 converted 会系统性偏晚约 0.2 秒 —— 在一个
    1~65 秒量级的量上**看不出来**。
    """
    seg = {"kind": "converted", "seq": 3}
    assert arrival_wall_of(seg, _LEDGER) == 100.500, (
        "取到了 converted 那条(偏晚)或没取到 raw —— 这会让 reaction_time 系统性偏晚")


def test_arrival_wall_refuses_to_fall_back():
    """没有对应的 raw 记录 ⟹ `None`,**不许**退回 converted 的值。

    「宁缺勿偏」:退回就等于悄悄用了那个已知偏晚的时刻。返回 `None` 之后
    `reaction_time_features` 会整组不写(它收到 `None` 就交 `{}`),而不是写一个错的。
    """
    assert arrival_wall_of({"kind": "converted", "seq": 99}, _LEDGER) is None
    assert arrival_wall_of({"seq": None}, _LEDGER) is None


_QUESTIONS = [
    {"index": 0, "ask_end": 100.0},
    {"index": 1, "ask_end": 150.0},
    {"index": 2, "ask_end": 190.0},
]


def test_question_index_is_the_last_ask_end_before_arrival():
    """★ 归题规则:最后一个 `ask_end ≤ 到达时刻` 的题号。

    实测背书(场 `20260927_212143_95a8`,8 段回答):与活日志的 `question_index`
    **8/8 全对**。红法:改成"第一个 `ask_end ≤ 到达`"(取最早的那题)⟹ 本条红,
    而且真实数据上会从第 2 段起全错。
    """
    assert infer_question_index(100.0, _QUESTIONS) == 0
    assert infer_question_index(160.0, _QUESTIONS) == 1
    assert infer_question_index(195.0, _QUESTIONS) == 2


def test_question_index_refuses_to_guess():
    """早于所有 `ask_end`、或没有题台账 ⟹ `None`,不猜第 0 题。"""
    assert infer_question_index(99.0, _QUESTIONS) is None
    assert infer_question_index(160.0, []) is None
    assert infer_question_index(None, _QUESTIONS) is None
    # 题台账里 `ask_end` 缺失/不是数的条目不许把整条推断带崩
    assert infer_question_index(160.0, [{"index": 0}, {"index": 1, "ask_end": None}]) is None
