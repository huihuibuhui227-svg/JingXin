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


def test_hand_visible_rows_match_the_logger_contract(tmp_path, monkeypatch):
    """★ 裁定 F3 的钉子:表里**登记的东西**与**实现**逐条对得上。

    为什么需要它:Task 3 登记 `hand_visible_left` / `hand_visible_right` 时这两行只是
    `pending` —— 而 `schema_errors()` 不校验 pending 行的语义、钉子方向 2 也**只查
    implemented 行**。⟹ 在翻状态之前,这两行**没有任何钉子**证明「表里写的」与
    「代码做的」是同一件事;翻状态之后,方向 2 也只证明**列名存在**,证不了
    `definition` 里那句「`1` ⟺ 有依据 **且** 手在」和「**永不写 `0`**」。

    本测试把表里**可机检**的部分全部钉住:
      ① 表里的列名 == 日志 `fieldnames` 里真有的列名(且在最末尾 —— acceptance ③);
      ② `status` 已翻 `implemented`;
      ③ `unit` 说的字母表 `{1, 空}`:实现只吐这两个值,**没有 `0`**;
      ④ `definition` 说的合取:只有「知道是哪只手」**且**「手在」才写 `1`。

    红法(逐条,都是生产改动):往 `fieldnames` 中间插一列(①)、把某行的 `status`
    改回 `pending`(②)、把 `_hand_visible_cells` 的 `else ""` 改成 `else "0"`(③)、
    或把它的 `known and present.get(slot)` 删掉一半(④)。
    """
    import gesture_analysis.utils.logger as glog

    monkeypatch.setattr(glog, "LOGS_DIR", str(tmp_path / "logs"))

    rows = {c["column"]: c for c in l0.columns()}
    names = ["hand_visible_left", "hand_visible_right"]
    for name in names:
        assert name in rows, f"表里没有 {name} 这一行"
        row = rows[name]
        assert row["modality"] == "gesture", row["modality"]
        assert row["status"] == "implemented", (
            f"{name} 的 status 还是 {row['status']!r} —— 实现已落地,表里没跟着翻")
        assert row["unit"] == "布尔(1 / 空)", row["unit"]

    fieldnames = list(glog.GestureLogger(session_id="x").fieldnames)
    for name in names:
        assert name in fieldnames, f"表里登记了 {name},日志的 fieldnames 里却没有"
    assert fieldnames[-len(names):] == names, (
        f"新列不在列序末尾(acceptance ③)—— 末尾实为 {fieldnames[-4:]}")

    # ④/③ —— 判据的合取关系与字母表(穷举 handedness_info × hand_present 的四种组合)
    alphabet = set()
    for entry in (None, ("Right", 0.9)):
        for present in (True, False):
            cells = glog.GestureLogger._hand_visible_cells(
                {"left_hand": entry, "right_hand": entry},
                {"left_hand": present, "right_hand": present})
            assert set(cells) == set(names), sorted(cells)
            alphabet |= set(cells.values())
            expected = "1" if (entry is not None and present) else ""
            assert cells["hand_visible_left"] == expected, (
                f"有依据={entry is not None}、手在={present} 时写的是 "
                f"{cells['hand_visible_left']!r},表里 definition 要求 {expected!r}")
    assert alphabet <= {"1", ""}, (
        f"表里 unit 写的是「布尔(1 / 空)」,实现却吐了 {sorted(alphabet)} "
        f"—— `0` 的意思是「确定没有这只手」,与「没测到」不是一回事")

