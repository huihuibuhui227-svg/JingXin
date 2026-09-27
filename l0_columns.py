"""M3.0 的 L0 列清单(spec: docs/superpowers/specs/2026-09-26-l0-column-table-design.md)。

与 `media_retention.py` / `session_meta.py` 同层、同在仓库根:它描述的是
**三服务 + 报告层共用的横切事实**,不属于任何一个模态包。

**为什么是数据文件而不是 markdown 表格**:markdown 表格没有约束力。本项目栽过
「spec 说 ≥2 段、代码门槛是 5」这种互相矛盾(spec §3.1),靠 grep 才抓到。
有了 `tests/test_l0_column_table.py` 的双向钉子,这类矛盾**不可能静默存在**。
"""
import copy
import json
import re
from pathlib import Path

TABLE_PATH = Path(__file__).resolve().parent / "l0_columns.json"

_REQUIRED_FIELDS = ("column", "modality", "definition", "source", "normalization",
                    "unit", "maturity", "status", "acceptance")
# B 档 = "缺的只是一个现在就能定的参数"。既然是**我们**定的,就必须写下依据 ——
# 否则它和"编一个值"没有区别(spec §10 风险 1)。
_BASIS_REQUIRED_FOR = ("B",)
MATURITIES = ("A", "B", "C")
# ★ `l0_done`(2026-09-27 使用者裁定):**L0 侧已按本表落地并在三场正式素材上实测**,
#   但验收里还有子项只能来自标定集 / 后续里程碑 ⟹ 不标 `implemented`(那是"整条做完")。
#   与 `blocked` 的区别:`blocked` 是"什么都不产出也算合格"(L0 还没做),`l0_done` 是
#   "L0 做完且真在产出,只是还差一条兑现不了的判据"。目前 11 条 C 档全是这个状态。
#   与 `pending` 的区别:`pending` 读表的人会以为**一行代码都没做** —— 那对这批是假话。
STATUSES = ("implemented", "pending", "blocked", "l0_done")
MODALITIES = ("face", "gesture", "voice", "text", "covariate")

_CACHE: dict | None = None


def load() -> dict:
    """读出整张表。**交深拷贝** —— 调用方改坏它不该影响下一次读。"""
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    # 收窄给类型检查器看(`_CACHE` 的标注必须是 `dict | None` 才能表达"还没读")。
    # 上一行刚赋过值 ⟹ 这里恒真,不是业务断言。
    assert _CACHE is not None
    return copy.deepcopy(_CACHE)


def columns() -> list[dict]:
    return load()["columns"]


def schema_errors(doc: dict) -> list[str]:
    """结构错误清单。空列表 = 合格。**只判结构,不判定义对不对** ——
    定义对不对由 Task 2/3/4 的人工审与后续里程碑负责。"""
    errs: list[str] = []
    for row in doc.get("columns", []):
        name = row.get("column", "<无名>")
        for field in _REQUIRED_FIELDS:
            if not str(row.get(field, "")).strip():
                errs.append(f"{name}: 字段 {field} 为空")
        if row.get("maturity") not in MATURITIES:
            errs.append(f"{name}: maturity={row.get('maturity')!r} 不在 {MATURITIES}")
        if row.get("status") not in STATUSES:
            errs.append(f"{name}: status={row.get('status')!r} 不在 {STATUSES}")
        if row.get("modality") not in MODALITIES:
            errs.append(f"{name}: modality={row.get('modality')!r} 不在 {MODALITIES}")
        if row.get("maturity") == "C" and not str(row.get("l0_output", "")).strip():
            errs.append(f"{name}: maturity=C 必须写清 l0_output(L0 到底吐什么)")
        if row.get("maturity") in _BASIS_REQUIRED_FOR and not str(row.get("basis", "")).strip():
            errs.append(f"{name}: maturity={row['maturity']} 必须写 basis(这个值是我们定的,凭据在哪)")
        # ★ 预检裁定 F2(2026-09-26):标了 implemented 的行,modality 必须是**产出它的服务**。
        #   理由:钉子方向 2 拿 (column, modality) 去比 `_live_columns()` 的 key,
        #   而那个 dict 只有那三个键 —— 填 `covariate` 的行永远匹配不上,
        #   一旦标 implemented 就会**假红**。"协变量用途"是用途,写在 definition 里。
        # ★ 2026-09-27 使用者裁定:把 **`text` 加成第四种产出方**。
        #   原来这里只有 face/gesture/voice ⟹ text 三行**结构上永远不可能**标 implemented,
        #   哪怕它们的验收全部达成(实测:翻一下就各报一条错)。那是**表的表达力缺口**,
        #   不是"text 没做"(它真在产出:仓库外的 `transcript.json`)。
        #   配套:钉子方向 2 的 `_live_columns()` 补一个 `text` 键,**由 transcript 的真实
        #   产出方推出来**(不是抄表)—— 这样"产出方没了"照样会红。
        if row.get("status") == "implemented" and row.get("modality") not in ("face", "gesture", "voice", "text"):
            errs.append(f"{name}: status=implemented 但 modality={row.get('modality')!r} "
                        f"—— 已实现的列必须标产出它的服务(face/gesture/voice/text)")
        # ★ Task 8(2026-09-26):`blocked` 的行必须在 `acceptance` 里写下**解封条件**。
        #   理由:`blocked` 是唯一一个"什么都不产出也算合格"的状态 —— 不写解封条件,
        #   「卡在某个外部条件上」与「只是不打算做」在表上长得一模一样,读表的人无从
        #   知道它等的是什么。判据是那四个字在不在(同 `asr_confidence` 那行的写法)。
        if row.get("status") == "blocked" and "解封条件" not in str(row.get("acceptance", "")):
            errs.append(f"{name}: status=blocked 但 acceptance 里没写「解封条件」"
                        f"—— 卡住的行必须写清它等的是什么(缺什么、怎么拿到)")
        # ★ 同一条理由适用于 `l0_done`:它是"L0 做完、还差一条兑现不了的判据"。
        #   不写「解封条件」的话,它与"我做完 L0 就不管了"在表上长得一模一样。
        if row.get("status") == "l0_done" and "解封条件" not in str(row.get("acceptance", "")):
            errs.append(f"{name}: status=l0_done 但 acceptance 里没写「解封条件」"
                        f"—— 这一档的含义就是「L0 已落地、还差一条判据」,必须写清那条判据等的是什么")
    if not doc.get("count_reconciliation"):
        errs.append("缺 count_reconciliation —— 「54」这个数必须登记来源与差异")
    # ★ 计数收口(2026-09-26,预检裁定 C5 + Task 2 复核 §5 C10):`actual` 的每个模态数
    #   必须等于 `columns` 里该模态的行数。在此之前这两处全靠人记得同步 —— 加了一行忘了
    #   改 `actual` **不会红**,是这份数据里唯一没有钉子守着的部分。一条断言把它变成机器检查。
    #   口径:按**行**数,不按活列数(裁定 F2 已把协变量列按产出它的服务计入,见 count_reconciliation.scope)。
    #   ⚠️ **两个方向都要查**:只遍历 `actual` 里已有的键时,「把整个模态键删掉 / 清空 actual」
    #   不报错(2026-09-26 复核实测)—— 那正是"忘了同步"的另一种形态,所以遍历两边的并集。
    cr = doc.get("count_reconciliation") or {}
    actual = cr.get("actual") or {}
    per_modality: dict = {}
    for row in doc.get("columns", []):
        key = row.get("modality")
        per_modality[key] = per_modality.get(key, 0) + 1
    for modality in sorted(set(actual) | set(per_modality), key=str):
        if modality not in actual:
            errs.append(f"count_reconciliation.actual 缺模态 {modality!r} 的键,而 columns 里有 "
                        f"{per_modality[modality]} 行 —— 新模态要在这里登记")
            continue
        counted = per_modality.get(modality, 0)
        if counted != actual[modality]:
            errs.append(f"count_reconciliation.actual.{modality}={actual[modality]} 与 columns 里该模态的行数 "
                        f"{counted} 不等 —— 加/删一行就要同步这四个数")
    # ★ Task 8 复核 Important 1(2026-09-26):`count_reconciliation` 的**散文**里不许内联
    #   逐模态的数、也不许内联它们的合计 —— 那些数**已经**在 `actual` 里,再写一遍就是
    #   第二份要手工同步的副本。`note` 此前写着「face 22 / … ,实际为 85」,而 Task 8 把
    #   face 那一项加一之后,那个合计**当场过期**(21+41+19+4=85 是对的,22+…=86 才对)。
    #   处置照 Task 5 对 `live_contract` 的处置:数字**只说一次**(放在结构化字段里),
    #   散文只引用字段名。
    prose = {k: str(cr.get(k, "")) for k in ("note", "scope")}
    for modality, n in actual.items():
        for where, text in prose.items():
            if f"{modality} {n}" in text:
                errs.append(f"count_reconciliation.{where}: 内联了逐模态的数「{modality} {n}」"
                            f"—— 它已经在 actual 里,散文再写一遍就是第二份要手工同步的副本")
    for where, text in prose.items():
        if re.search(r"(实际|合计|总数)\s*[为是]?\s*\d+", text):
            errs.append(f"count_reconciliation.{where}: 内联了派生合计(「实际 / 合计 / 总数 … 数字」"
                        f"这种措辞)—— 合计 = sum(actual.values()) = {sum(actual.values())},"
                        f"不是一个要写在散文里的常数")
    return errs
