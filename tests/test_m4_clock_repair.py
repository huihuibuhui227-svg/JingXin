# tests/test_m4_clock_repair.py
"""**墙钟修复**的守卫。

## 钉的是什么

1. **回拨落在区间内 ⟹ 修正量等于回拨量**(不是"大概修一下");
2. **墙钟本来就单调 ⟹ 修正量恰好 0**,而且状态是 `clean`(可信的 0);
3. **没有账本 ⟹ 状态是 `no_ledger`,不是 `clean`** ——
   把"查不了"说成"没问题"是本仓最忌讳的一类错误;
4. **间隔超出账本覆盖范围 ⟹ `out_of_range`**,不外推着给一个像样的数;
5. `reaction_time` 的修正**不碰采集侧**:只并列给出,不改历史日志。
"""

from __future__ import annotations

import json

import pytest

from report_frontend import clock_repair as cr

SID = "20261018_140000_ab12"


@pytest.fixture()
def rec_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    return tmp_path


def _ledger(root, declared_s: list[float], wall_s: list[float], modality="face"):
    """写一份 frame 账本。`declared_s` 单调,`wall_s` 可以故意倒退。"""
    d = root / SID
    d.mkdir(parents=True, exist_ok=True)
    base = 1_790_400_000.0
    lines = []
    for i, (dec, wall) in enumerate(zip(declared_s, wall_s)):
        lines.append(json.dumps({
            "kind": "frame", "modality": modality, "seq": i + 1,
            "file": f"media/face/{i + 1:04d}.jpg", "bytes": 1, "sha256": "x",
            "received_at_wall": base + wall, "declared_ts": int(dec * 1000),
            "source_endpoint": "/face/frame", "session_id": SID,
        }))
    (d / f"retention.{modality}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return base


# ── 1. 回拨落在区间内 ⟹ 修正量 = 回拨量 ─────────────────────────────────

def test_backstep_inside_interval_is_added_back(rec_dir):
    """★ 回拨必须落在**区间之内**才会污染间隔 —— 这是本模块的核心语义。

    每 1 s 一帧。declared 20 处墙钟倒退 2 s(之后继续走)。
    区间起点在墙钟 5(回拨前)、终点在墙钟 33(回拨后)⟹ 墙钟只走了 28 s,
    而真实经过 30 s ⟹ 修正量必须正好 +2 s。

    ⚠️ 反例(踩过):把回拨放在区间**之前**时,两个端点被同样平移,差值本来就是对的,
    修正量应当是 0。我第一版测试就是这么写错的。

    红法:把 `elapsed` 改成直接返回入参 ⟹ 修正量变 0,本条红。
    """
    declared = [float(i) for i in range(0, 60)]
    wall = [float(i) for i in range(0, 60)]
    for i in range(20, 60):          # declared≥20 之后墙钟整体少 2 s(一次回拨)
        wall[i] -= 2.0
    base = _ledger(rec_dir, declared, wall)

    rmap = cr.build_repair_map(SID)
    assert rmap is not None
    assert rmap.n_back_steps == 1
    # ⚠️ 帧间隔 1 s,一次 2 s 的回拨只表现为 1 s 的**帧间**倒退 ⟹ 这是下界,不是调整量
    assert rmap.max_back_step == pytest.approx(1.0, abs=1e-6)

    # 起点 5 是回拨前的读数;终点取回拨后的 33(对应真实 35)
    iv = cr.repair_interval(rmap, base + 5.0, base + 33.0)
    assert iv.raw == pytest.approx(28.0), "墙钟只走了 28 s"
    assert iv.repaired == pytest.approx(30.0), "真实经过 30 s"
    assert iv.delta == pytest.approx(2.0)
    assert iv.status == cr.REPAIRED


# ── 2. 干净墙钟 ⟹ 恰好 0,且可信 ─────────────────────────────────────────

def test_clean_clock_repairs_to_exactly_zero(rec_dir):
    """红法:让 `status` 恒返回 `REPAIRED` ⟹ 本条红(那是把干净的说成有问题的)。"""
    declared = [float(i) for i in range(0, 60)]
    base = _ledger(rec_dir, declared, list(declared))

    rmap = cr.build_repair_map(SID)
    assert rmap.status == cr.CLEAN
    iv = cr.repair_interval(rmap, base + 5.0, base + 35.0)
    assert iv.delta == pytest.approx(0.0)
    assert iv.status == cr.CLEAN


# ── 3. 没有账本 ⟹ 不许报 clean ──────────────────────────────────────────

def test_no_ledger_is_not_clean(rec_dir):
    """★ 查不了 ≠ 没问题。红法:把 `rmap is None` 那一支改成返回 `CLEAN` ⟹ 本条红。"""
    rec_dir.joinpath(SID).mkdir(parents=True, exist_ok=True)

    assert cr.build_repair_map(SID) is None
    iv = cr.repair_interval(None, 1_790_400_010.0, 1_790_400_020.0)
    assert iv.status == cr.NO_LEDGER
    assert iv.status != cr.CLEAN
    assert iv.note, "必须说明'未经核对',不能只给一个数"
    assert iv.delta == 0.0


# ── 4. 超出覆盖范围 ⟹ out_of_range ───────────────────────────────────────

def test_interval_outside_ledger_coverage_is_flagged(rec_dir):
    """红法:删掉 `covers` 检查 ⟹ 外推给出一个像样的数,本条红。"""
    declared = [float(i) for i in range(0, 10)]
    base = _ledger(rec_dir, declared, list(declared))
    rmap = cr.build_repair_map(SID)

    iv = cr.repair_interval(rmap, base + 100.0, base + 200.0)
    assert iv.status == cr.OUT_OF_RANGE
    assert "超出" in iv.note


# ── 5. 逐题反应延迟:并列给出,不改采集侧 ────────────────────────────────

def test_reaction_times_reports_both_values_and_never_rewrites_the_log(rec_dir):
    """★ 采集侧算的 `reaction_time` 一个字节都不动 —— 只并列给修正值。

    回拨在 declared 51(-3 s)。题 0 的区间 [5,10] 完全在回拨之前 ⟹ 修正量 0;
    题 1 的区间 [20,52] 跨过回拨 ⟹ 墙钟只走了 32 s,真实 35 s ⟹ +3 s。
    两个端点都不在"被走两遍"的带里([48,50]),所以修正是**确定**的。

    红法:把 `raw_reaction_time` 改成返回修正值 ⟹ 本条红(历史与新日志就不可比了)。
    """
    declared = [float(i) for i in range(0, 120)]
    wall = [float(i) for i in range(0, 120)]
    for i in range(51, 120):
        wall[i] -= 3.0
    base = _ledger(rec_dir, declared, wall)

    questions = [{"index": 0, "qid": "题目0", "ask_start": base + 4.0, "ask_end": base + 5.0},
                 {"index": 1, "qid": "题目1", "ask_start": base + 19.0, "ask_end": base + 20.0}]
    voice = [{"index": 0, "answer_onset_wall": base + 10.0},
             {"index": 1, "answer_onset_wall": base + 52.0}]

    rows = cr.reaction_times(SID, questions, voice)
    assert len(rows) == 2

    q0, q1 = rows
    assert q0["raw_reaction_time"] == pytest.approx(5.0)
    assert q0["delta"] == pytest.approx(0.0), "回拨之前的题不该被改"
    assert q0["status"] == cr.REPAIRED      # 本场确实有回拨,只是没落在这道题上

    assert q1["raw_reaction_time"] == pytest.approx(32.0), "墙钟只走了 32 s"
    assert q1["repaired_reaction_time"] == pytest.approx(35.0), "真实经过 35 s"
    assert q1["delta"] == pytest.approx(3.0)
    assert q1["ambiguous"] is False

    # 原始值必须原样保留在 `raw_reaction_time` 里
    assert q1["raw_reaction_time"] == pytest.approx(32.0)


# ── 6. 端点落在"被走两遍"的带里 ⟹ 报区间,不报单点 ─────────────────────

def test_endpoint_in_the_revisited_band_yields_a_bound_not_a_single_number(rec_dir):
    """★ 回拨若落在区间内,起点读数**必然**落在被走两遍的带里。

    回拨发生在 declared 51(-3 s):墙钟从 50 退到 48,于是 48~50 这段被走两遍。
    取 `ask_end` 正好等于 50(带内)⟹ 它对应的真实时刻在 [50, 53] 之间,
    没有唯一解。此时**必须**给出不确定度,而不是挑一个数当答案。

    红法:把 `elapsed_range` 的高端锚点去掉(只留平台首端)⟹ `bound` 恒 0、
    `ambiguous` 恒 False,本条红。
    """
    declared = [float(i) for i in range(0, 120)]
    wall = [float(i) for i in range(0, 120)]
    for i in range(51, 120):
        wall[i] -= 3.0
    base = _ledger(rec_dir, declared, wall)

    questions = [{"index": 0, "qid": "题目0", "ask_start": base + 49.0, "ask_end": base + 50.0}]
    voice = [{"index": 0, "answer_onset_wall": base + 52.0}]

    row = cr.reaction_times(SID, questions, voice)[0]
    assert row["ambiguous"] is True
    assert "不确定度" in row["note"]
    # 真实答案落在 [2, 5] 之间(取决于 ask_end 落在第一遍还是第二遍)
    assert 2.0 - 1e-6 <= row["repaired_reaction_time"] <= 5.0 + 1e-6
    # 约定:歧义时取区间**中点**(误差对称),并把半宽作为不确定度报出。
    # 这条约定必须显式钉住,否则"取首端/取末端/取中点"三种都能过测试,
    # 而它们的值能差出整整一个回拨量。
    assert row["repaired_reaction_time"] == pytest.approx(3.5)
    assert "±1.50 s" in row["note"]
