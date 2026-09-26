# tests/test_l0_column_table.py
"""M3.0 的双向覆盖钉子 —— 日志产出的列 <-> l0_columns.json 的行。

方向 1(产出 ⊆ 表):日志里出现的每一列,要么在表里,要么在 legacy_allowlist 里。
方向 2(表 ⊆ 产出):表里标 status=implemented 的每一列,必须真的在日志里。
两条都是**双向**的:任一边多一列、少一列都要红。
"""
import importlib
import math

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
      ① 表里的列名 == 日志 `fieldnames` 里真有的列名(且是**追加**上去的 —— acceptance ③);
      ② `status` 已翻 `implemented`;
      ③ `unit` 说的字母表 `{1, 空}`:实现只吐这两个值,**没有 `0`**;
      ④ `definition` 说的合取:只有「知道是哪只手」**且**「手在」才写 `1`;
      ⑤ ★ `definition` **散文本身**必须还写着那两件事(`handedness_info` 这个判据入口,与
         「不写 0」这条禁令)。复核 Minor 2:没有 ⑤ 的时候,④ 比的是**本测试自己抄的一遍规则** ——
         表里 `definition` 若被改成「槽非空即为真」,④ 照样绿(只有 `unit` 串字面比对,
         改措辞反而误红)。⑤ 把「表里那句话」也拉进断言。

    ★ **① 的形态在 2026-09-26 Task 7 改过(改的是断言,不是判据)**:
    原文是 `fieldnames[-2:] == names`(即"这两列是**全局最后两列**")。Task 7 按它自己的
    acceptance ③ 往末尾追加了 `shoulder_width` ⟹ 那条断言**当场变红**,而它红的**不是**
    要防的东西(追加是安全的;"插进中间"才危险)。**两条 acceptance 都写「加在列序末尾」,
    而末尾只有一个** —— 先实施的那条若按字面钉死"永远是最后一列",每一次后续加列都会误红。
    所以这里改成钉**「追加而非插入」**这个性质:`hand_visible_left` 必须**紧跟**
    `right_shoulder_jitter_world`(Task 6 加它时它前面那一列),两列相邻。
    ⟹ 它仍然守得住 acceptance ③ 的原意(往中间插一列会红),但**不会**被后来的追加误伤。
    表里那两行的 acceptance ③ 措辞也同步成"追加而非插入"(免成一句静默过期的话)。

    红法(逐条,都是生产改动):往 `fieldnames` 中间插一列(①)、把某行的 `status`
    改回 `pending`(②)、把 `_hand_visible_cells` 的 `else ""` 改成 `else "0"`(③)、
    把它的 `known and present.get(slot)` 删掉一半(④)、
    **把表里 `definition` 换成「槽非空即为真」**(⑤)。
    ⚠️ ⑤ 的边界(如实说):它钉的是**那两句话还在不在**,不是"散文与代码语义等价" ——
    若有人保留这两个词、却把语义反过来写,⑤ 抓不到(那种改写只能靠人读)。
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
        # ⑤ 判据入口与「不写 0」这两句必须在表里还写着(红法:改成「槽非空即为真」)
        assert "handedness_info" in row["definition"], (
            f"{name} 的 definition 不再点名判据入口 `handedness_info` —— "
            f"「槽非空即为真」正是这条要防的写法:{row['definition'][:80]!r}")
        assert "不写 0" in row["definition"], (
            f"{name} 的 definition 不再写「不写 0」—— 「没测到」与「测到了没有」"
            f"的区别在表里必须有:{row['definition'][:80]!r}")

    fieldnames = list(glog.GestureLogger(session_id="x").fieldnames)
    for name in names:
        assert name in fieldnames, f"表里登记了 {name},日志的 fieldnames 里却没有"
    # ★ 「追加而非插入」:两列相邻,且紧跟在 Task 6 之前的那一列后面
    #   (红法:把这两列挪到 `fieldnames` 中间任意位置)
    anchor = "right_shoulder_jitter_world"
    assert anchor in fieldnames, f"锚点列 {anchor} 不见了 —— 列序被大改过"
    k = fieldnames.index(anchor)
    assert fieldnames[k + 1:k + 1 + len(names)] == names, (
        f"这两列不是**追加**在列序上的(acceptance ③):{anchor} 之后实为 "
        f"{fieldnames[k + 1:k + 4]}")

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


def test_shoulder_width_row_matches_the_implementation():
    """★ Task 7 的钉子:`shoulder_width` 表行与实现逐条对得上(**含表里 definition 的散文**)。

    为什么需要它:方向 2 只证「列名存在」。这一行**三件事都错得起**,而那三件它一件都证不了:
      · **单位**写成米 —— 而本项目**没有任何相机内参来源**(2026-09-26 全仓 grep
        `solvePnP` / `camera_matrix` / `Rodrigues` / 焦距:**0 命中**;`models/` 下只有 4 个
        `.task`、无标定文件)⟹ 米制肩宽只能靠**编一个内参**(Task 8 已据此把 3D 头姿登记 blocked);
      · **下标**换掉:`dist(lm[11], lm[12])` 被改成别的关节;
      · **缺失**写成 `0` —— 归一化坐标里 `0` 是「两肩重合」,一个**像测量值**的假数。

    ★ ③④⑤ 三段**一开始就连着表里 `definition` 的散文**(Task 6 复核 Minor 2 的教训:
    只比「测试自己抄的一遍规则」时,表里那行被改成别的说法,钉子照样绿),
    而且断言**限定在「判据区」**(第一个 `⚠️` 之前那一段)—— 理由是实测出来的:
    用"整条 definition 里出现过这几个字"写时,把判据那句换成「米制/厘米」的变异**抓不到**
    (那几个字在后面的 ⚠️ 段里都还在)。见 `criterion = definition.split("⚠️")[0]` 那段注释。

    ⚠️ ③④⑤ 的边界(如实说):它们钉的是**判据区里那几件事还写着、且没被写成反面**
    (单位不是米、式子是 dist(lm[11], lm[12])、缺了写空不写 0),**不**等于"散文与代码语义等价" ——
    若有人保留这些词、却把语义绕成等价于另一回事(比如补一句"但实际按 3D 算"),这几条抓不到。

    红法(逐条,都是生产改动,**每一条都实测红过**):
      ① 把这一列插到 `fieldnames` 的**中间**(不是追加);
      ② `status` 翻回 `pending`;
      ③ 表里 `unit` 改成米,或把**判据区**改成「**米制**下的双肩距离(单位:厘米)」;
      ④ 表里判据区的式子改成 `dist(lm[13], lm[14])`;
      ⑤ 表里判据区不再写「不写 0」;
      ⑥ 实现改用别的下标(lm[13]/lm[14]);
      ⑦ 实现里缺肩 `return 0.0`;
      ⑧ 给实现加一个 `visibility <= 0.6` 的门限(表里写着没有)。

    ⚠️ ⑧ 的边界(如实说):它钉的是「表里写着没有门限、代码也不许有」。**加门限本身不是错**,
    错的是**偷偷加**(表里那句还写着"没有")。反过来,若有人把 definition 里那句删掉再加门限
    —— ⑤⑧ 里"可见度门限"那条子串断言会红,提醒两处要一起改。
    """
    import gesture_analysis.utils.logger as glog
    from gesture_analysis.core.feature_extraction import angles

    rows = {c["column"]: c for c in l0.columns()}
    name = "shoulder_width"
    assert name in rows, f"表里没有 {name} 这一行"
    row = rows[name]
    definition, acceptance = row["definition"], row["acceptance"]
    assert row["modality"] == "gesture", row["modality"]

    # ② status 已翻
    assert row["status"] == "implemented", (
        f"{name} 的 status 还是 {row['status']!r} —— 实现已落地,表里没跟着翻")

    # ★ 表里那一行的**判据区** = 第一个 `⚠️` 之前那一段(⚠️ 之后是说明与禁令)。
    #   断言**限定在这一区**里 —— 而不是"整条 definition 里某处出现过这几个字":
    #   后者是**有洞**的:2026-09-26 实测过一个 M3 变异(把判据那句改成
    #   「**米制**下的双肩距离(单位:厘米)」),整条 definition 里"归一化图像单位""不是米"
    #   这些字**都还在**(它们出现在后面的 ⚠️ 段里)⟹ 松散的 `in definition` 抓不到。
    #   判据区的措辞是可以被改的,但**判据本身**(单位、公式、缺失怎么办)不许与实现不符。
    criterion = definition.split("⚠️")[0]

    # ③ 单位:表里 `unit` 必须是归一化图像单位,且**判据区**不许出现米制单位
    assert row["unit"] == "归一化图像单位", row["unit"]
    assert "归一化图像单位" in criterion, (
        f"{name} 的**判据区**没写单位是归一化图像单位 —— 「不是米」这句警告还在它后面的"
        f"⚠️ 段里,但判据自己必须先说对:{criterion[:90]!r}")
    assert "厘米" not in criterion and "米制" not in criterion, (
        f"{name} 的判据区出现了米制单位 —— 本项目没有任何相机内参来源,给不出米制:"
        f"{criterion[:90]!r}")
    # 而「不是米」这条警告必须还在(它拦的是"把这一列当人体测量值用")
    assert "不是米" in definition, (
        f"{name} 的 definition 不再写「不是米」—— 这一列只差一步就会被当成人体测量值:"
        f"{definition[:80]!r}")

    # ④ 下标与公式:判据区里点的必须是 `dist(lm[11], lm[12])` 这个式子
    #    (不是分散的 `lm[11]`/`lm[12]` 两个子串 —— 那允许把式子换成别的词)
    assert "dist(lm[11], lm[12])" in criterion, (
        f"{name} 的判据区不再是 dist(lm[11], lm[12]):{criterion[:90]!r}")

    # ⑤ 缺失:判据区必须写着「不写 0」,acceptance 必须写着「空」
    assert "不写 0" in criterion, (
        f"{name} 的判据区不再写「不写 0」—— 「没测到」与「测到了 0」的区别在判据里必须有:"
        f"{criterion[:90]!r}")
    assert "空" in acceptance, (
        f"{name} 的 acceptance 不再写「缺 ⟹ 空」:{acceptance[:80]!r}")

    # ① 列名 == `fieldnames` 里真有的列名,且是**追加**在列序上的(表里 acceptance ③)
    #    ⚠️ 这里**不**断言"它是全局最后一列":后续任务还会往末尾追加别的列,
    #    "永远是最后一列"会误伤每一次追加(本任务自己就在 Task 6 那条上踩到了 ——
    #    见 `test_hand_visible_rows_match_the_logger_contract` 的说明)。
    #    钉的性质与那里一致:**追加而非插入** —— 紧跟本任务加它时的前一列 `hand_visible_right`。
    fieldnames = list(glog.GestureLogger(session_id="x").fieldnames)
    assert name in fieldnames, f"表里登记了 {name},日志的 fieldnames 里却没有"
    anchor = "hand_visible_right"
    assert anchor in fieldnames, f"锚点列 {anchor} 不见了 —— 列序被大改过"
    assert fieldnames[fieldnames.index(anchor) + 1] == name, (
        f"{name} 不是**追加**在 `{anchor}` 之后的(表里 acceptance ③):"
        f"{anchor} 之后实为 {fieldnames[fieldnames.index(anchor) + 1:fieldnames.index(anchor) + 3]}")
    assert "追加" in acceptance, (
        f"{name} 的 acceptance 不再写「追加」(acceptance ③ 的原意):{acceptance[:80]!r}")

    # ⑥ ★ 判据可执行:表里说的那两个下标 + 「欧氏距离」⟹ **测试自己**算期望值,再拿实现比。
    #    期望值不来自实现:它来自表里 definition 的 lm[11]/lm[12] 与 `dist(...)` 这个说法。
    #    下面那 33 点里,11/12 之外全是同一个占位(0.5, 0.5)⟹ 实现若改用别的下标,
    #    算出来的是 0.0 或别的数,与 expected 对不上。
    class _Pt:
        def __init__(self, x, y):
            self.x, self.y = x, y

    lm11, lm12 = (0.31, 0.42), (0.67, 0.66)
    pose = [_Pt(0.5, 0.5) for _ in range(33)]
    pose[11], pose[12] = _Pt(*lm11), _Pt(*lm12)
    expected = math.hypot(lm11[0] - lm12[0], lm11[1] - lm12[1])
    got = angles.shoulder_width(pose)
    assert got is not None and abs(got - expected) < 1e-9, (
        f"表里写的是 dist(lm[11], lm[12]) = {expected:.6f},实现算出来 {got!r}")

    # ⑦ 表里 acceptance ② 说的「缺肩写空」:缺任一肩 ⟹ `None`(**不是 `0`**)
    for missing in (11, 12):
        broken = list(pose)
        broken[missing] = None
        assert angles.shoulder_width(broken) is None, (
            f"缺 lm[{missing}] 时交的不是 None —— 归一化坐标里 `0` 是「两肩重合」,不是「没测到」")
    assert angles.shoulder_width([_Pt(0.5, 0.5)] * 12) is None, "点数不够时交的不是 None"

    # ⑧ ★ 判据**逐条**一致:表里 `definition` 现在明写「**没有**可见度门限」
    #    (与同模块 `pose_angles` 的 0.6 门限不同)⟹ 实现也不许有。
    #    ⚠️ 这条**不是**替「不加门限」背书 —— 它钉的是「表与代码一致」。要不要加门限是**口径**
    #    问题:加的话两边一起改(definition 里把那句去掉/改写 + 实现里加一句),这颗钉子会红一次
    #    提醒你两处都动了没有。
    #    红法:给 `shoulder_width` 加 `if min(_visibility(a), _visibility(b)) <= VISIBILITY_FLOOR: return None`。
    class _PtVis(_Pt):
        def __init__(self, x, y, visibility):
            super().__init__(x, y)
            self.visibility = visibility

    invisible = list(pose)
    invisible[11] = _PtVis(*lm11, visibility=0.0)
    invisible[12] = _PtVis(*lm12, visibility=0.0)
    assert "可见度门限" in definition, (
        f"{name} 的 definition 不再写「可见度门限」这件事 —— 表与代码的判据要能逐条对上:"
        f"{definition[:80]!r}")
    got2 = angles.shoulder_width(invisible)
    assert got2 is not None and abs(got2 - expected) < 1e-9, (
        f"两个 visibility=0.0 的肩就交不出距离了({got2!r})—— 而表里 definition 写着本列"
        f"**没有**可见度门限。要么改实现,要么把 definition 那句一起改掉")

