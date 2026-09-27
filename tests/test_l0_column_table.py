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


def test_blocked_rows_state_their_unblock_condition():
    """★ 红法:把某条 `blocked` 行的 `acceptance` 清空(或把「解封条件」四个字删掉)。

    为什么需要它:`blocked` 是**唯一一个「什么都不产出也算合格」的状态** ——
    `schema_errors()` 只判结构,钉子方向 2 又只查 `implemented` 行。于是不写解封条件的话,
    「卡在某个外部条件上(缺相机内参 / 缺置信度来源)」与「只是不打算做」在表上**长得一模一样**,
    读表的人无从知道它等的是什么。判据形状与 `asr_confidence` 那行一致(它写的是
    `**解封条件(缺什么、怎么拿到)**:`)。

    ★ 同一条规则也写在 `l0_columns.py::schema_errors()` 里 —— 两处都放:那一条让
    **写表的人**立刻看到错,这一条让**读表/复核的人**不依赖 schema_errors 的实现。

    ⚠️ 边界(如实说):它钉的是**那四个字还在不在**,不是「解封条件写得对不对」——
    写一句「等有空」也能过。要判内容对不对只能人读;这条断言的价值在于**删掉**那句
    或者**清空 `acceptance`** 会立刻红(后者同时被 `_REQUIRED_FIELDS` 的空值检查覆盖,
    但那条说的是「字段为空」、这条说的是「字段写了却没写解封条件」,两种漏法都要挡)。
    """
    blocked = [c for c in l0.columns() if c["status"] == "blocked"]
    assert blocked, (
        "表里一条 blocked 行都没有 —— 本测试会**假绿**(它靠 blocked 行存在才有意义);"
        "若 blocked 确实已清零,请连同这条断言一起删掉")
    for c in blocked:
        assert "解封条件" in c["acceptance"], (
            f"{c['column']} 是 blocked,但 acceptance 里没写「解封条件」—— "
            f"读表的人无从知道它等的是什么(缺什么、怎么拿到):{c['acceptance'][:80]!r}")


def test_count_prose_may_not_inline_the_numbers():
    """★ 红法(两条,都是往 `count_reconciliation.note` 里写回一个数字):
      ① 把逐模态的数内联回去(`"<模态名> <它的数>"` 这种形状);
      ② 把派生合计内联回去(`实际为 N` / `合计 N` 这种措辞)。
      实测:两条都红;把 `schema_errors()` 里那段散文检查删掉 ⟹ 两条都绿(即本测试红)。

    为什么需要它(2026-09-26 Task 8 复核 Important 1):`note` 里内联着逐模态的四个数与
    **一个派生合计**,而本任务把 face 那一项加一之后,那个合计**当场过期** ——
    **是这次提交自己引入的**,而且**机器一条都不查它**(`schema_errors()` 只逐模态比
    `actual` vs 行数)。这正是本任务与 Task 7 反复在收的「散文里的数静默过期」。
    处置照 Task 5 对 `live_contract` 的处置:**数字只说一次**(放在结构化字段 `actual` 里),
    散文只引用字段名 —— 并且**给这句规矩一颗钉子**(就是本测试)。

    ⚠️ 边界(如实说):它钉的是**这两种形状**不许出现,不是"散文里绝不许有数字" ——
    换一种写法(把数写成中文数字、或把语序倒过来)本测试抓不到。它拦的是**实际发生过的那一次**
    与最容易复发的形状;真正结构性的保证是"数只在 `actual` 里说一次"。
    """
    doc = l0.load()
    cr = doc["count_reconciliation"]
    total = sum(cr["actual"].values())
    modality, n = next(iter(cr["actual"].items()))
    probes = [
        (f"逐模态的数见上;{modality} {n} 行", "把逐模态的数内联回去了"),
        (f"四个模态的 actual 见上。实际为 {total}", "把派生合计内联回去了"),
    ]
    for text, why in probes:
        bad = l0.load()          # `load()` 交的是深拷贝 ⟹ 改坏它不会污染本体
        bad["count_reconciliation"]["note"] = text
        errs = l0.schema_errors(bad)
        assert any("内联" in e for e in errs), (
            f"{why},而 `schema_errors()` 没报 —— 这句散文就又能静默过期了:{errs}")
    # 出厂表自己必须是干净的(否则上面两条"红"没有对照)
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
      ⑦-3 删掉实现里那个 `except (TypeError, ValueError)`(坐标取不出来时不再交 `None`);
      ⑧ 给实现加一个 `visibility <= 0.6` 的门限(表里写着没有);
      ⑧-反向 把表里那句从「**没有可见度门限**」改成「**有可见度门限**」,或改成
             「**没有**可见度门限」→「**有**可见度门限」(两种加粗写法都要红),或
             **在它之前插一段正确的**「⚠️ 本列没有可见度门限(见下)」**再改反真正那句**;
      ⑨ 把表里那个「**10** 个画面坐标 jitter 列」的数改错(改回 16 / 改成别的);

    ⚠️ ⑧ 的边界(如实说):它钉的是「表里写着没有门限、代码也不许有」。**加门限本身不是错**,
    错的是**偷偷加**(表里那句还写着"没有")。反过来,若有人把 definition 里那句删掉再加门限
    —— ⑤⑧ 里"可见度门限"那条子串断言会红,提醒两处要一起改。

    ⚠️ ⑧-反向 的边界(如实说):它钉的是**判据那句**是否定式(= 含「可见度门限」的 ⚠️ 段里、
    破折号之前那一段),而**不**管破折号之后的对照句 —— 所以
    「本列没有可见度门限;同模块 `pose_angles` 那 5 个角度**有可见度门限**」这种**合法的对照句
    不会误红**。它是同一条判据的**极性**那一半,与 ⑤⑧ 合起来才是"表与代码逐条对上"。
    ⚠️ **每一段**含「可见度门限」的 ⚠️ 段都要满足(只看第一段会被"前面插一段正确的话"绕过,实测过)。

    ⚠️ ⑨ 的边界(如实说):它钉的是「散文里的数 == 表里数出来的数」与「每个 jitter 行都有个
    明确归属」,**不**保证"10 这个集合语义上就该是这 10 条" —— 若有人把某条改错、又顺手把散文
    改成新数,⑨ 会绿。那种错只能靠人读 `normalization` 那一栏。它拦的是**静默过期**那一类
    (T7 复核 Minor 4 的原始形态:散文写 16、表里实为 10,而没有任何断言数过它)。
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

    # ⑦-3 ★ 坐标取不出来 ⟹ 与「肩缺了」同等对待(`_vec` 抛的 `TypeError`/`ValueError`
    #    由 `shoulder_width` 接住 ⟹ `None`)。红法:**删掉实现里那个
    #    `except (TypeError, ValueError)`** ⟹ 这里当场抛出去,本测试红。
    #    ⚠️ 为什么补它(Task 7 复核 Minor 3):那个 except 此前**没有任何测试覆盖**,
    #    活路径也进不去(mediapipe 的 landmark 必带 x/y)。选「补用例」而不是「删掉」的理由:
    #    本列的判据是「**缺 ⟹ 空,不写 0**」,而 `x` 取不出来正是「缺」的一种形态 ——
    #    删掉 except 会让它从「交空」变成「整帧抛异常」,那是**改判据**而不是去掉死代码;
    #    而 `shoulder_width` 按设计就是纯函数、可单测(表里 `source` 那栏写着),
    #    补一条断言的成本低于改语义的风险。
    class _PtNoX:
        y = 0.5                      # 有 y、没有 x ⟹ float(None) 抛 TypeError

    class _PtStrX:
        x = "abc"                    # 坐标不是数 ⟹ float("abc") 抛 ValueError
        y = 0.5

    for cls, why in ((_PtNoX, "x 缺了"), (_PtStrX, "x 不是数")):
        broken = list(pose)
        broken[11] = cls()
        assert angles.shoulder_width(broken) is None, (
            f"lm[11] 的 {why}(=「量不出来」)时交的不是 None —— 判据里「缺 ⟹ 空」"
            f"对坐标取不出来的情形同样成立")

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
    # ⑧-反向 ★ **极性**:上一条只证「可见度门限」这几个字还在 —— 把表里那句从
    #    「**没有**可见度门限」改成「**有**可见度门限」,上一条**照样绿**(2026-09-26
    #    Task 7 复核 Minor 1 实测)。那正是"表里定义被改反、钉子不响"这一类。
    #    形态照同文件 ③(`assert "厘米" not in criterion and "米制" not in criterion`):
    #    正面那句必须还在,而且**肯定式不许出现**。
    #    ⚠️ 三个坑,都是实测踩出来的:
    #      ① 必须先去掉 markdown 强调符:`**有**可见度门限` 里的 `**` 会把
    #         `有可见度门限` 这个子串**切断**,直接 `"有可见度门限" not in definition`
    #         抓不到这一种改法;
    #      ② `没有可见度门限` **自己就包含** `有可见度门限` 这个子串 ⟹ 判定必须写成
    #         "把正确那句整段删掉之后,剩下的话里不许再出现肯定式";
    #      ③ 上面那条若拿**整条 definition** 去判,会对**合法的对照句**误红 ——
    #         「本列没有可见度门限;同模块 `pose_angles` 那 5 个角度**有可见度门限**」
    #         是一句真话,却会被判成"写反了"(2026-09-26 实测过这个形态)。
    #         ⟹ 判定**限定在判据那句本身**(= 含「可见度门限」那个 ⚠️ 段里、破折号
    #         之前那一段;破折号之后是对照与说明)。这与 ③ 的「判据区」是同一个手法。
    #    ⚠️ 第四坑(2026-09-26 Task 8 复核实测):只看**第一段**会有洞 —— 在真正那句
    #       **之前**插一段「⚠️ 本列没有可见度门限(见下) —— 重申一次。」并把真正那句改成
    #       「**有**可见度门限」⟹ 只看 `gate_sections[0]` 时**绿**。所以**每一段**都要满足。
    gate_sections = [s for s in definition.split("⚠️") if "可见度门限" in s]
    assert gate_sections, (
        f"{name} 的 definition 里「可见度门限」不再出现在任何 ⚠️ 说明段里 —— "
        f"判据与禁令的写法被大改过,请人工读一遍:{definition[:80]!r}")
    for _i, _sec in enumerate(gate_sections):
        gate_claim = _sec.replace("*", "").split("——")[0]
        assert "没有可见度门限" in gate_claim, (
            f"{name} 的 definition 第 {_i + 1} 段提到「可见度门限」,但它那句**不是否定式**"
            f"(或那句「没有可见度门限」被写反/删掉了):{gate_claim[:80]!r}")
        assert "有可见度门限" not in gate_claim.replace("没有可见度门限", ""), (
            f"{name} 的 definition 第 {_i + 1} 段把那句判据**写反了**"
            f"(出现了肯定式的「有可见度门限」)—— 而实现里没有门限:{gate_claim[:80]!r}")
    got2 = angles.shoulder_width(invisible)
    assert got2 is not None and abs(got2 - expected) < 1e-9, (
        f"两个 visibility=0.0 的肩就交不出距离了({got2!r})—— 而表里 definition 写着本列"
        f"**没有**可见度门限。要么改实现,要么把 definition 那句一起改掉")

    # ⑨ ★ 「它是几个 jitter 列的分母」这个**数**,拿**表自己**数出来,不靠散文。
    #    红法(都是生产改动):① 再往表里加/删一行画面坐标 jitter ⟹ 数变了而
    #    shoulder_width 那两处散文没改 ⟹ 红;② 把某个 `*_jitter_world` 行的
    #    `normalization` 改成「÷ 肩宽」⟹ 它被数进分母、而「不除肩宽」那批少一条 ⟹ 红;
    #    ③ 直接改散文里的数字(改错) ⟹ 红。
    #    为什么需要它(Task 7 复核 Minor 4 的同类):这里原先写的是「16 个 jitter 列」,
    #    而真正把肩宽写进 `normalization` 当分母的是 **10 行**(另 6 个 `*_jitter_world`
    #    明写「**不**除肩宽 —— 米制已是解剖尺度」)。**没有任何断言数过它** ⟹ 那个 16
    #    既核不到、也不会因为加了一行而红 —— 与 `live_contract` 那件事同形。
    #    ⚠️ 判 `normalization` 里有没有「不除肩宽」时**先去 markdown 强调符** ——
    #    表里写的是 `**不**除肩宽`,带星号时子串 `不除肩宽` **匹配不上**(2026-09-26 实测)。
    def _norm(c):
        return c["normalization"].replace("*", "")

    jitter_rows = [c for c in l0.columns()
                   if c["normalization"].strip()
                   and (c["column"].endswith("_jitter") or c["column"].endswith("_jitter_world"))]
    uses = [c for c in jitter_rows if "肩宽" in _norm(c) and "不除肩宽" not in _norm(c)]
    world = [c for c in jitter_rows if "不除肩宽" in _norm(c)]
    # 二分律:每个 jitter 行要么拿肩宽当分母、要么明写「不除肩宽」—— 没有第三种状态。
    # (红法:加一个 `normalization` 里两边都不提的新 jitter 行 ⟹ 「几个」这个数就失去含义)
    unclassified = [c["column"] for c in jitter_rows if c not in uses and c not in world]
    assert not unclassified, (
        f"这些 jitter 行的 `normalization` 既没拿肩宽当分母、也没写「除/不除肩宽」,"
        f"于是「有几个 jitter 列拿肩宽当分母」**数不出来**:{unclassified}")
    # ⚠️ 这里**不**钉「jitter 行恰好 16 条」:追加一行是常规操作,钉死条数会误伤每一次追加
    #    (Task 7 在 `fieldnames[-2:]` 上刚踩过这个坑,见另一个测试的说明)。
    #    要钉的性质是「**散文里的数** == **表里数出来的数**」。
    for field in ("definition", "acceptance"):
        assert f"{len(uses)} 个画面坐标 jitter 列" in row[field].replace("*", ""), (
            f"{name} 的 {field} 里那个「几个 jitter 列」的数与表里数出来的"
            f"({len(uses)} 条:{[c['column'] for c in uses]})对不上 —— "
            f"加/删一行 jitter 就要同步这句:{row[field][:80]!r}")


# ══════════════════════════════════════════════════════════════════════════════
# `count_reconciliation.deltas` 散文里的数(2026-09-26 整支复核 Important 2 + Minor 11)
# ══════════════════════════════════════════════════════════════════════════════

# gesture:被 §4.2 **处置表**逐项点到名的 28 条活列。
# ⚠️ **这 28 条是人工读 markdown 表点出来的** —— §4.2 没有机器可读形态,所以
#    「它们真的都被点到名」这件事**机器核不了**(要核只能人再读一遍 §4.2)。
#    本测试核得到的是另外三件:① 这 28 个名字**全部真的在活列里**;
#    ② 它们把 70 条活列**切成 28 + 42**;③ 那 42 条按下面的分族枚举**逐条对得上**。
#    ⟹ 名单本身不能悄悄漂(改一个名字,②或③当场红),这正是「数不再手抄」要的效果。
_NAMED_IN_SECTION_4_2 = frozenset("""
    left_hand_score right_hand_score left_hand_fist_status right_hand_fist_status
    left_hand_spread right_hand_spread left_hand_jitter right_hand_jitter
    shoulder_score head_score torso_score left_arm_score right_arm_score
    left_shoulder_jitter right_shoulder_jitter left_wrist_jitter right_wrist_jitter
    left_elbow_jitter right_elbow_jitter head_jitter torso_jitter
    left_arm_stability right_arm_stability torso_stability
    shrug_level is_calibrated head_tilt is_valid
""".split())

# gesture:§4.2 处置表**覆盖不到的** 42 条活列,按 `deltas[1].reasons[1]` 的散文分族逐条枚举。
# 这条枚举是那句散文的**机器可读副本** —— 散文说「42 = 3 + 4 + 10 + 8 + 2 + 8 + 4 + 3」,
# 这里就把每一族的名字写出来;名字与族对不上、或少了/多了一条,下面逐条比。
_UNCOVERED_FAMILIES = {
    "3 非测量列": ("session_id", "timestamp", "timestamp_iso"),
    "4 个 handedness 标签": ("left_hand_model_label", "left_hand_model_label_conf",
                            "right_hand_model_label", "right_hand_model_label_conf"),
    "10 个手指角度": tuple(f"{side}_{finger}_angle" for side in ("left", "right")
                        for finger in ("thumb", "index", "middle", "ring", "pinky")),
    "8 个姿态角度": ("left_shoulder_angle", "right_shoulder_angle", "torso_angle",
                  "head_tilt_angle", "head_pitch_angle", "shoulder_angle",
                  "left_elbow_angle", "right_elbow_angle"),
    "2 个 *_arm_angle": ("left_arm_angle", "right_arm_angle"),
    "8 个 world 列": ("left_wrist_jitter_world", "left_elbow_jitter_world",
                    "right_wrist_jitter_world", "right_elbow_jitter_world",
                    "left_arm_angle_world", "right_arm_angle_world",
                    "left_shoulder_jitter_world", "right_shoulder_jitter_world"),
    "4 个情绪块": ("overall_score", "emotion_state", "feedback", "used_features"),
    "3 个 2026-09-26 新增列": ("hand_visible_left", "hand_visible_right", "shoulder_width"),
}


def test_delta_reason_numbers_match_the_table():
    """★ `count_reconciliation.deltas` 那三段散文里的数,**从表里算出来再比散文**。

    为什么要它(2026-09-26 整支复核 Important 2 + Minor 11):`deltas[1].reasons[1]` 原先写
    「`live_contract.gesture` 那 **67** 条活列…剩下 **39** 条…其余 **23** 条进 columns」——
    **而 `live_contract.gesture` 早已是 70**:那句话自己点名引用了那个被实测的字段,却差 3。
    同族的还有 `deltas[0].reasons[4]` 的 16 / 6 与 voice `reasons[3]` 的 13。
    ⟹ 处置照本文件 ⑨ 的形态:**把数从表里算出来,再去比散文**;散文改了数、表改了行,两边都有一次机会红。

    红法(逐条,都是生产改动):
      · 把 `live_contract.gesture` 改掉(或往 `GestureLogger.fieldnames` 加/删一列而不动表)⟹ 70 那一组红;
      · 把 `deltas[1].reasons[1]` 里那个「70 / 42 / 9 / 7 / 26」改回 67 / 39 / 23 之类的旧数 ⟹ 红;
      · 把 `columns` 里某条 gesture 行搬进 `legacy_allowlist`(或不搬而行数对不上)⟹ 9/7/26 的二分红;
      · 往 `_UNCOVERED_FAMILIES` 覆盖的某族里加一列(例如再加一个手指角度)⟹ 42 这个数、
        以及「9 + 7 + 26 = 42」当场红;
      · 把 `deltas[0].reasons[4]` 的 16 / 6 或 voice `reasons[3]` 的 13 改错 ⟹ 各自红。

    ⚠️ **边界(如实说)**:`legacy_allowlist` 是按**模态**分组的,没有「判删 / 元数据」这个字段,
    所以 9 与 7 的切法是「这 42 条里,在 `_UNCOVERED_FAMILIES` 中被点名是元数据的那 3 + 4 条」
    —— 换句话说,**7 那一半靠上面那份族的枚举,不靠白名单自己声明**。白名单若给每条加一个
    `disposition` 字段,这条就该改成直接读它(那才是结构性的)。
    """
    doc = l0.load()
    cr = doc["count_reconciliation"]
    live = list(_live_columns()["gesture"])
    claimed_live = cr["live_contract"]["gesture"]

    # ── ① 70:活列契约的结构化字段 vs 真的 logger ────────────────────────────
    assert claimed_live == len(live), (
        f"live_contract.gesture={claimed_live},而 GestureLogger 实测 {len(live)} 列")

    # ── ② 70 必须 == 表里 gesture 的活列行 + 白名单条目 ──────────────────────
    gesture_cols = {c["column"] for c in doc["columns"] if c["modality"] == "gesture"}
    gesture_allow = {a["column"] for a in doc["legacy_allowlist"] if a["modality"] == "gesture"}
    assert set(live) == gesture_cols | gesture_allow, (
        f"gesture 活列(70)与「表内行 ∪ 白名单」对不上:"
        f"只在日志里={sorted(set(live) - gesture_cols - gesture_allow)},"
        f"只在表里={sorted((gesture_cols | gesture_allow) - set(live))}")
    assert not (gesture_cols & gesture_allow), (
        f"同一列同时进 `columns` 与 `legacy_allowlist`:{sorted(gesture_cols & gesture_allow)}")

    # ── ③ 28 + 42 = 70,而且那 42 条按族枚举逐条对得上 ───────────────────────
    named = _NAMED_IN_SECTION_4_2
    assert named <= set(live), (
        f"`_NAMED_IN_SECTION_4_2` 里有名字不在活列里(名单过期了):{sorted(named - set(live))}")
    uncovered = set(live) - named
    enumerated = [c for names in _UNCOVERED_FAMILIES.values() for c in names]
    assert len(enumerated) == len(set(enumerated)), (
        f"分族枚举里有重复名字——同一列被数进两族:{sorted({c for c in enumerated if enumerated.count(c) > 1})}")
    assert uncovered == set(enumerated), (
        f"「§4.2 覆盖不到的」那 {len(uncovered)} 条与分族枚举的 {len(set(enumerated))} 条对不上:"
        f"没被枚举的={sorted(uncovered - set(enumerated))},"
        f"枚举了却不在活列里的={sorted(set(enumerated) - uncovered)}")

    # ── ④ 9 / 7 / 26:那 42 条按「元数据进白名单 / 判删进白名单 / 进 columns」三分 ──
    meta = set(_UNCOVERED_FAMILIES["3 非测量列"]) | set(_UNCOVERED_FAMILIES["4 个 handedness 标签"])
    assert meta <= gesture_allow, (
        f"非测量列与 handedness 标签应当**进白名单**:漏的={sorted(meta - gesture_allow)}")
    n_meta = len(meta)                                     # 7
    n_allow_42 = len(uncovered & gesture_allow)            # 16 = 9 + 7
    n_del = n_allow_42 - n_meta                            # 9(白名单里去掉元数据那 7 条)
    n_cols_42 = len(uncovered & gesture_cols)              # 26
    assert n_meta + n_del + n_cols_42 == len(uncovered), (
        f"「{n_meta} 元数据 + {n_del} 判删 + {n_cols_42} 进 columns」= "
        f"{n_meta + n_del + n_cols_42},与那 {len(uncovered)} 条对不上")

    # ── ⑤ 散文必须写着**算出来的**那一组数(这一句才是钉子)────────────────────
    reason = cr["deltas"][1]["reasons"][1]
    computed = f"{claimed_live} / {len(uncovered)} / {n_del} / {n_meta} / {n_cols_42}"
    assert computed in reason, (
        f"`deltas[1].reasons[1]` 里写的数与表里数出来的对不上:表算出「{computed}」,"
        f"而那句散文里找不到这个串(它此前写的是 67 / 39 / 23)—— 散文:{reason[:120]!r}")
    assert f"只有 {len(named)} 条" in reason, (
        f"`deltas[1].reasons[1]` 里「被 §4.2 处置表点到名的只有 N 条」的 N 与"
        f"`_NAMED_IN_SECTION_4_2` 的 {len(named)} 条对不上")
    assert f"剩下 {len(uncovered)} 条" in reason, (
        f"`deltas[1].reasons[1]` 里「剩下 N 条」与表里数出来的 {len(uncovered)} 对不上")

    # ── ⑥ face:16 / 6(deltas[0].reasons[4])──────────────────────────────────
    face_live = _live_columns()["face"]
    face_allow = [a["column"] for a in doc["legacy_allowlist"] if a["modality"] == "face"]
    face_in = [c for c in face_allow if c in face_live]
    face_out = [c for c in face_allow if c not in face_live]
    n_bs = len([c for c in face_live if c.startswith("bs_")])
    non_measure = ("session_id", "timestamp")              # 散文点名的 2 条非测量列
    n_in = len(face_in) - n_bs - len(non_measure) - 1      # −1 = micro_exp_duration_frames
    face_reason = cr["deltas"][0]["reasons"][4]
    assert f"共 {n_in} 条" in face_reason, (
        f"face 那条散文里的「判删的活列共 N 条」与从白名单数出来的 {n_in} 条对不上"
        f"(白名单在日志契约里 {len(face_in)} 条 − {n_bs} 个 bs_* − {len(non_measure)} 个非测量列"
        f" − micro_exp_duration_frames 1 条);散文:{face_reason[:120]!r}")
    assert f"另有 {len(face_out)} 条" in face_reason, (
        f"face 那条散文里的「另有 N 条族级/非活列」与数出来的 {len(face_out)} 条对不上;"
        f"散文:{face_reason[:120]!r}")

    # ── ⑦ voice:13(deltas[2].reasons[3])─────────────────────────────────────
    voice_cols = {c["column"] for c in doc["columns"] if c["modality"] == "voice"}
    voice_allow = {a["column"] for a in doc["legacy_allowlist"] if a["modality"] == "voice"}
    n_voice = cr["live_contract"]["voice"] - len(voice_cols)
    assert n_voice == len(voice_allow), (
        f"voice 活列 {cr['live_contract']['voice']} − 表内 voice 行 {len(voice_cols)} "
        f"= {n_voice},而白名单里 voice 条目是 {len(voice_allow)} 条 —— 两者应当相等")
    voice_reason = cr["deltas"][2]["reasons"][3]
    assert f"{n_voice} 条活列进" in voice_reason, (
        f"voice 那条散文里的「N 条活列进 legacy_allowlist」与数出来的 {n_voice} 条对不上;"
        f"散文:{voice_reason[:120]!r}")


def test_quarantine_refs_point_at_real_keys():
    """★ 表里对 `evidence_gate` 的引用必须是**键名**,不能是行号 —— 键名不会过期。

    2026-09-27(B0)实测踩到:表里原有 **78 处** `evidence_gate.py:NN` 行号引用,当时
    **全部是准的**;B0 只往 `evidence_gate.py` 加了几行(一个 dataclass 字段 + 注释),
    78 处**集体失效** —— 而失效是**静默**的,读表的人会照着错行号去读错的代码。
    这正是纪律 2「散文里的数会静默过期」的教科书形态 ⟹ 全部换成键名引用
    (`evidence_gate.QUARANTINE['jitter']` / `evidence_gate.is_quarantined`),本钉子守住。

    红法:把任一处改成指向不存在的键(如 `QUARANTINE['jitter_world']`),或改回行号形式 ⟹ 立刻红。
    """
    import json
    import re

    from report_frontend.evidence_gate import QUARANTINE

    doc = l0.load()
    raw = json.dumps(doc, ensure_ascii=False)

    assert not re.search(r"evidence_gate\.py:\d+", raw), (
        "表里又出现了 evidence_gate.py 的**行号**引用 —— 行号会随任何一次编辑静默失效,"
        "请写成键名(evidence_gate.QUARANTINE['<键>'])或符号名(evidence_gate.is_quarantined)")

    refs = set(re.findall(r"QUARANTINE\['([^']+)'\]", raw))
    assert refs, "表里一处 QUARANTINE 引用都没有了?那封停依据就无处可查了"
    dangling = sorted(r for r in refs if r not in QUARANTINE)
    assert not dangling, (
        f"表里引用了封停名单里**不存在**的键:{dangling} —— "
        f"要么键改名了(改引用),要么键被删了(删引用)")
