# tests/test_l0_column_table.py
"""M3.0 的双向覆盖钉子 —— 日志产出的列 <-> l0_columns.json 的行。

方向 1(产出 ⊆ 表):日志里出现的每一列,要么在表里,要么在 legacy_allowlist 里。
方向 2(表 ⊆ 产出):表里标 status=implemented 的每一列,必须真的在日志里。
两条都是**双向**的:任一边多一列、少一列都要红。
"""
import importlib

l0 = importlib.import_module("l0_columns")


def _live_columns():
    """三份日志现在真的会写出的列名(构造 logger 即可,不写行)。"""
    from face_expression.utils.logger import DataLogger
    from gesture_analysis.utils.logger import GestureLogger
    from voice_interaction.utils.logger import VoiceLogger
    return {
        "face": list(DataLogger(log_type="video", session_id="x").fieldnames),
        "gesture": list(GestureLogger(session_id="x").fieldnames),
        "voice": list(VoiceLogger(log_type="interview", session_id="x").fieldnames),
    }


def test_every_live_column_is_either_in_the_table_or_allowlisted():
    """方向 1。红法:往任一 logger 的 fieldnames 里加一列而不进表/白名单。"""
    doc = l0.load()
    tabled = {(c["column"], c["modality"]) for c in doc["columns"]}
    allowed = {(a["column"], a["modality"]) for a in doc["legacy_allowlist"]}
    unaccounted = []
    for modality, cols in _live_columns().items():
        for col in cols:
            if (col, modality) not in tabled and (col, modality) not in allowed:
                unaccounted.append(f"{modality}.{col}")
    assert not unaccounted, (
        "这些列既不在表里、也不在 legacy_allowlist 里:\n  "
        + "\n  ".join(unaccounted)
        + "\n—— 要么给它们写一行定义,要么进白名单并写明为什么、归哪个里程碑。")


def test_every_implemented_row_really_is_produced():
    """方向 2。红法:把某行标成 implemented 而它其实不在 fieldnames 里。"""
    live = _live_columns()
    bad = [f"{c['modality']}.{c['column']}"
           for c in l0.columns() if c["status"] == "implemented"
           and c["column"] not in live.get(c["modality"], [])]
    assert not bad, f"表里标了已实现、日志里却没有:{bad}"


def test_live_contract_matches_the_real_log_contract():
    """★ 红法:往任一 logger 的 `fieldnames` 加一列(或删一列)而不更新 `live_contract` ⟹ 必须红。

    为什么需要这一条:三个模态的**活列数**此前只写在 `count_reconciliation.scope` 的**散文里**
    (「face 91 条 = …;gesture 67 条 = …;voice 31 条 = …」),而 `schema_errors()` 的计数断言查的是
    `actual` vs `columns` 的**行数**,**不查这句散文**。于是 2026-09-26 Task 5 往 `VoiceLogger.fieldnames`
    末尾加了 `voiced_prob_mean`(voice 活列 31 → 32)之后,那句散文**静默过期**,而它自己正写着
    「这四行不再靠手工同步」—— 一句话自称不再手工同步,而它自己就是手工同步的。

    处置(2026-09-26,使用者裁定):把那些数字从散文里抽成 `count_reconciliation.live_contract`
    这个**结构化字段**,由本测试对着三个 logger 的 `fieldnames` 实测;散文只引用字段名,不再内联数字。
    """
    doc = l0.load()
    live = _live_columns()
    claimed = doc["count_reconciliation"]["live_contract"]
    assert set(claimed) == set(live), (
        f"模态对不上:live_contract 声称 {sorted(claimed)},日志契约实测 {sorted(live)}")
    for modality, n in claimed.items():
        assert n == len(live[modality]), (
            f"{modality}: live_contract 声称日志契约有 {n} 列,实测 {len(live[modality])} 列 —— "
            f"加/删一列就要同步这个字段(红法:往该 logger 的 fieldnames 末尾加一列)")


def test_allowlist_entries_carry_a_reason_and_a_target():
    """白名单不许变成万能垃圾桶:每条都要有 why 与 planned。"""
    for a in l0.load()["legacy_allowlist"]:
        assert a.get("why", "").strip(), f"{a.get('column')} 缺 why"
        assert a.get("planned", "").strip(), f"{a.get('column')} 缺 planned"


def test_schema_errors_is_empty_on_the_shipped_file():
    assert l0.schema_errors(l0.load()) == []


def test_load_hands_back_a_copy():
    """★ 红法:把 `load()` 改成 `return _CACHE`(去掉 deepcopy)⟹ 本测试必须红。

    ⚠️ 探针**不许用 `columns`**:出厂表里它是空列表(`"columns": []`),
    `a["columns"].clear()` 在两种实现下都留空 ⟹ **零区分力**(2026-09-26 实测踩过)。
    用 `_schema.version` —— 出厂表里就有值,与填表进度无关。
    """
    a = l0.load()
    a["_schema"]["version"] = "TAMPERED"
    assert l0.load()["_schema"]["version"] != "TAMPERED", "load() 交出了本体,调用方能改坏它"
