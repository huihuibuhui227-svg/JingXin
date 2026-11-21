# report_frontend/instrument_provenance.py
"""**仪器来源披露** —— 每条 L1 指标到底由哪一套"仪器"产出。

## 为什么需要这一层

本系统的面部测量**同时存在两套输出**,而且它们不是同一个东西:

* **`geometric`(几何代理)**:`au_calculator.py` 手写的关键点比率
  (`au4_frown = 眉内距 ÷ 眼距`、`au7_eye_squeeze = 1 − avg_ear` …),
  外加 `gaze_*` / `head_*` / `is_blink`;
* **`blendshape`(模型回归)**:`face_expression/models/blendshapes.py` 产出的
  52 个 FaceLandmarker 系数(`bs_mouthSmileLeft` …)。

离线实测(32 场 / 389 条,见 `07_…/两套面部仪器一致性_v2.md`)表明二者
**秩一致性因构念而异**:12 个构念里只有 1 个达 ρ ≥ 0.7。⟹ 两套数**不可混用**,
报告必须标明每个面部指标由哪一套产出。此前全仓 `grep -niE "blendshape|instrument|仪器"`
在 `l1_report.py` / `report_generator.py` / `l1_indicators.py` 里**零命中** ——
"仪器来源披露"只存在于文案里,代码里没有。

本模块把它变成可计算的东西:**只做披露,不改任何指标的值**。

## 三条规矩

1. **不改既有字段。** `annotate_indicators` 返回的是**副本**,原有键值一个不动
   (报告的既有消费方不会因为多了两个键而改变行为)。
2. **不猜。** 依赖列一条都认不出来 ⟹ 记 `unknown`,不按"看起来像面部"归到
   `geometric`。宁可让缺口露出来。
3. **多仪器就是多仪器。** `speech_rate = n_chars ÷ speech_duration_sec` 一头是
   转写(`text`)、一头是音频(`audio`)⟹ 它必须记成 `instruments=["audio","text"]`,
   不能挑一个写上去。

## 形状(谁读它)

* 单仪器:新增 `instrument: str`;
* 多仪器:新增 `instruments: list[str]`;
* 两种都用 `instruments_of()` 读 —— 报告模板与前端**只认这一个访问器**,
  免得两处各自判断"到底该读哪个键"。

⚠️ 一条指标记多仪器时,`describe()` 会在**每一个**涉及的仪器下各计 1 ——
页脚那句话的意思是"这条证据来自哪几路",不是"账能对上"。总数要看 `total` 之外的
`len(indicators)`,别拿页脚几项相加当总数(会重复)。
"""

from __future__ import annotations

from typing import Any, Iterable

# ── 仪器名 ────────────────────────────────────────────────────────────────────

INSTRUMENT_BLENDSHAPE = "blendshape"
INSTRUMENT_GEOMETRIC = "geometric"
INSTRUMENT_AUDIO = "audio"
INSTRUMENT_TEXT = "text"
#: 依赖列一条都认不出来时记这个。**不是**一种仪器,是一个披露缺口。
INSTRUMENT_UNKNOWN = "unknown"

#: 固定顺序 —— `describe()` 与多仪器列表都按这个序输出,页面上的顺序不随输入抖动。
INSTRUMENTS: tuple[str, ...] = (
    INSTRUMENT_BLENDSHAPE,
    INSTRUMENT_GEOMETRIC,
    INSTRUMENT_AUDIO,
    INSTRUMENT_TEXT,
    INSTRUMENT_UNKNOWN,
)

_ORDER = {name: i for i, name in enumerate(INSTRUMENTS)}


# ── 列 → 仪器 ─────────────────────────────────────────────────────────────────

#: 前缀 → 仪器。**最长前缀优先**(`bs_` 与 `b_` 同时命中时取长的)。
#:
#: 前四组是任务给定的最小集合;`iris_` / `eye_` / `face_` / `micro_exp_` /
#: `torso_` / `hand_` / `shoulder_` / `left_` / `right_` 是本仓**实际列名**的补充登记
#: (`left_wrist_jitter` 这类手势列也是同一路视觉管道出来的关键点几何量)。
#: ⚠️ 补充登记的风险如实写在这里:`left_` / `right_` 是宽前缀,
#: 将来若出现 `left_*` 的**非**关键点列,它会被错记成 `geometric` ——
#: 新增列时请改用精确名登记(`INSTRUMENT_OF_EXACT`),不要靠猜。
INSTRUMENT_OF_PREFIX: dict[str, str] = {
    # 模型回归出来的 52 个 FaceLandmarker 系数
    "bs_": INSTRUMENT_BLENDSHAPE,
    # 关键点几何代理:手写 AU 比率 / 视线 / 头姿 / 眨眼
    #
    # ⚠️ `au_` 是任务给的写法,但**实测列名是 `au4_frown` / `au12_smile`**(数字紧跟
    # `au`,没有下划线)⟹ 只写 `au_` 会让 F 族 16 条全部落进 `unknown`
    # (本仓 `l0_columns.json` 里真实列名如此)。两个前缀都登记,长的优先。
    "au_": INSTRUMENT_GEOMETRIC,
    "au": INSTRUMENT_GEOMETRIC,
    "gaze_": INSTRUMENT_GEOMETRIC,
    "head_": INSTRUMENT_GEOMETRIC,
    "blink": INSTRUMENT_GEOMETRIC,          # 题目给的是 `blink*`(无下划线)
    "iris_": INSTRUMENT_GEOMETRIC,
    "eye_": INSTRUMENT_GEOMETRIC,
    "face_": INSTRUMENT_GEOMETRIC,
    "symmetry": INSTRUMENT_GEOMETRIC,
    "micro_exp_": INSTRUMENT_GEOMETRIC,
    # 手势 / 姿态关键点(同一路视觉管道,仍是几何量)
    "torso_": INSTRUMENT_GEOMETRIC,
    "hand_": INSTRUMENT_GEOMETRIC,
    "shoulder_": INSTRUMENT_GEOMETRIC,
    "left_": INSTRUMENT_GEOMETRIC,
    "right_": INSTRUMENT_GEOMETRIC,
    # 音频:音高 / 能量 / VAD / 停顿 / 语音时长
    "pitch_": INSTRUMENT_AUDIO,
    "energy_": INSTRUMENT_AUDIO,
    "voiced_": INSTRUMENT_AUDIO,
    "pause_": INSTRUMENT_AUDIO,
    "speech_": INSTRUMENT_AUDIO,
    "duration_": INSTRUMENT_AUDIO,
    "reaction_": INSTRUMENT_AUDIO,
    # 文本:ASR 转写
    "transcript": INSTRUMENT_TEXT,
    "text": INSTRUMENT_TEXT,
    "token": INSTRUMENT_TEXT,
    "word": INSTRUMENT_TEXT,
}

#: 精确名登记 —— 名字**不以**任何前缀开头,或前缀会误伤的那些列。
#:
#: 为什么要单独一张表:任务给的 `pitch_*` 匹配不到裸列 `pitch`,
#: `blink*` 匹配不到 `is_blink`(它是后缀)。这两处若不登记,`speech_rate` /
#: `blink_rate_per_min` 这类核心指标会被记成 `unknown` —— 而它们是**知道**出处了,
#: 只是名字不规则。"知道却说不知道"和"不知道却说知道"一样是错的。
INSTRUMENT_OF_EXACT: dict[str, str] = {
    "is_blink": INSTRUMENT_GEOMETRIC,
    "avg_ear": INSTRUMENT_GEOMETRIC,
    "blink_rate_per_min": INSTRUMENT_GEOMETRIC,   # 日志里那一列是滑动 60 s 窗口量
    "eye_closed_sec": INSTRUMENT_GEOMETRIC,
    "symmetry_score": INSTRUMENT_GEOMETRIC,
    "tension_score": INSTRUMENT_GEOMETRIC,        # 面部派生(合成别名,见 l1_face 坑 2)
    "tension_level": INSTRUMENT_GEOMETRIC,
    "pitch": INSTRUMENT_AUDIO,
    "energy": INSTRUMENT_AUDIO,
    "voiced_flag": INSTRUMENT_AUDIO,
    "n_chars": INSTRUMENT_TEXT,                   # 来自转写字数,不是音频
    "transcript_text": INSTRUMENT_TEXT,
    "left_wrist_jitter": INSTRUMENT_GEOMETRIC,
    "right_wrist_jitter": INSTRUMENT_GEOMETRIC,
    "left_elbow_jitter": INSTRUMENT_GEOMETRIC,
    "right_elbow_jitter": INSTRUMENT_GEOMETRIC,
    "left_shoulder_jitter": INSTRUMENT_GEOMETRIC,
    "right_shoulder_jitter": INSTRUMENT_GEOMETRIC,
    "hand_visible_left": INSTRUMENT_GEOMETRIC,
    "hand_visible_right": INSTRUMENT_GEOMETRIC,
}

#: **不是仪器**的列:时钟 / 会话标识 / 台账序号。它们参与每条指标的
#: `required_l0`,若把它们报成"未登记"或"未知仪器",每条指标都会背一条假缺口。
NON_INSTRUMENT_COLUMNS: frozenset[str] = frozenset({
    "timestamp", "session_id", "index", "frame_index", "qid",
    "t_start", "t_end", "window", "granularity",
})


def is_non_instrument_column(column: str) -> bool:
    """时钟 / 标识类列 —— **不是仪器**,也不算披露缺口。"""
    return column in NON_INSTRUMENT_COLUMNS


def instrument_of_column(column: str) -> str | None:
    """单列 → 仪器名。认不出来返回 `None`(**不是** `"unknown"`)。

    查找顺序:精确名 → 最长前缀 → `None`。时钟/标识类列同样返回 `None`,
    用 `unmapped_columns()` 区分"缺口"与"本来就不是仪器"。
    """
    if not isinstance(column, str) or not column:
        return None
    if column in NON_INSTRUMENT_COLUMNS:
        return None
    if column in INSTRUMENT_OF_EXACT:
        return INSTRUMENT_OF_EXACT[column]
    hit: str | None = None
    best = -1
    for prefix, instrument in INSTRUMENT_OF_PREFIX.items():
        if column.startswith(prefix) and len(prefix) > best:
            hit, best = instrument, len(prefix)
    return hit


def unmapped_columns(columns: Iterable[str]) -> set[str]:
    """依赖列里**既不是仪器、也没登记**的那些 —— 披露缺口清单。

    报告侧可以拿它自查:"这一条指标的出处里还有一片没写清楚"。
    时钟/标识列不算缺口(见 `NON_INSTRUMENT_COLUMNS`)。
    """
    return {c for c in _as_column_list(columns)
            if instrument_of_column(c) is None and not is_non_instrument_column(c)}


def instrument_of_columns(columns: Iterable[str]) -> set[str]:
    """依赖列 → 仪器集合。

    只收集**认得出**的仪器;认不出的列不进集合(它们由 `unmapped_columns` 报)。
    空输入、`None`、全是不认识的列 ⟹ 空集 —— 由调用方决定记 `unknown`。
    同一列的重复出现只算一次(输入是集合语义)。
    """
    out: set[str] = set()
    for c in _as_column_list(columns):
        inst = instrument_of_column(c)
        if inst is not None:
            out.add(inst)
    return out


def _as_column_list(columns: Any) -> list[str]:
    """把 `required_l0` 可能的各种写法摊平成 `list[str]`。

    `required_l0` 实测有三种写法:`('au12_smile',)`、`['au12_smile']`、
    以及空窗口时整条指标不带依赖。把它当字符串迭代会**逐字符**拆开
    (`"au12_smile"` → 10 个单字母列名),那是静默错 —— 这里显式拦掉。
    """
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    out: list[str] = []
    for c in columns:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, (list, tuple)):
            # 有人把依赖写成 `(('au1', 'au6'),)` 那种分组形式
            out.extend(x for x in c if isinstance(x, str))
    return out


# ── 指标 → 仪器 ───────────────────────────────────────────────────────────────

def _as_dict(indicator: Any) -> dict:
    """接受 `dict` 或 `L1Indicator`(后者走它的 `to_dict()`)。

    报告链路上两种形态同时存在:`l1_quality.QualityReport.indicators` 是 `list[dict]`,
    而 `l1_session.L1Result.indicators` 是 `list[L1Indicator]`。要求调用方先各自
    转换,早晚会出现"两处转得不一样" — 所以这里一次收干净。
    """
    if isinstance(indicator, dict):
        return dict(indicator)
    to_dict = getattr(indicator, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    raise TypeError(
        f"annotate_indicators 只收 dict 或带 to_dict() 的指标对象,收到 {type(indicator)!r}")


def annotate_indicators(indicators: list) -> list[dict]:
    """给每条指标**新增** `instrument`(单值) 或 `instruments`(多值)。

    * **不改任何既有字段**:返回新字典,输入对象与输入的字典都不被修改;
    * 单仪器 ⟹ 只加 `instrument`;多仪器 ⟹ 只加 `instruments`(按 `INSTRUMENTS` 定序);
    * 一条依赖列都认不出 ⟹ `instrument = "unknown"`,并**不**猜一个最近的仪器。

    读的时候请统一用 `instruments_of()`,不要自己判断该读哪个键。
    """
    out: list[dict] = []
    for ind in indicators or []:
        row = _as_dict(ind)
        found = instrument_of_columns(row.get("required_l0"))
        if not found:
            row["instrument"] = INSTRUMENT_UNKNOWN
        elif len(found) == 1:
            row["instrument"] = next(iter(found))
        else:
            row["instruments"] = sorted(found, key=lambda n: _ORDER.get(n, len(_ORDER)))
        out.append(row)
    return out


def instruments_of(indicator: Any) -> list[str]:
    """统一的读法:已标注/未标注、单值/多值都返回 `list[str]`。

    未标注的指标**当场算**(不抛错)—— 报告页脚可以在任何阶段调用,
    不必先跑一遍 `annotate_indicators`。
    """
    row = _as_dict(indicator)
    if row.get("instruments"):
        vals = list(row["instruments"])
        return sorted({v for v in vals if isinstance(v, str)},
                      key=lambda n: _ORDER.get(n, len(_ORDER)))
    inst = row.get("instrument")
    if isinstance(inst, str) and inst:
        return [inst]
    found = instrument_of_columns(row.get("required_l0"))
    if not found:
        return [INSTRUMENT_UNKNOWN]
    return sorted(found, key=lambda n: _ORDER.get(n, len(_ORDER)))


def describe(indicators: list) -> dict:
    """各仪器的指标计数 —— 报告页脚那一行。

    返回**固定 5 个键**(含计数为 0 的仪器):`blendshape` 为 0 本身就是要披露的事实
    (说明这份报告的每一个面部指标都来自手写几何代理,没有一条来自模型系数)。
    键顺序固定,页面不随输入抖动。

    ⚠️ 多仪器指标在每个涉及的仪器下各计 1 ⟹ 各数相加可能大于 `len(indicators)`。
    页脚要写"指标 N 条",请另取 `len(indicators)`,不要拿本字典求和。
    """
    counts = {name: 0 for name in INSTRUMENTS}
    for ind in indicators or []:
        for inst in instruments_of(ind):
            counts[inst] = counts.get(inst, 0) + 1
    return counts


def format_footer(indicators: list) -> str:
    """页脚一句话。**渲染层不许自己拼这句**(两处拼法早晚不一致)。

    例:`共 25 条指标 · 仪器来源:模型系数 0 / 几何代理 19 / 音频 4 / 文本 2 / 未登记 0`
    (顺序固定 = `INSTRUMENTS`:`blendshape` 在前 —— 页面上"两套面部仪器"永远并排,
    读的人才不会把 `au12_smile` 和 `bs_mouthSmile` 当成一回事。)
    """
    counts = describe(indicators)
    label = {
        INSTRUMENT_GEOMETRIC: "几何代理",
        INSTRUMENT_BLENDSHAPE: "模型系数",
        INSTRUMENT_AUDIO: "音频",
        INSTRUMENT_TEXT: "文本",
        INSTRUMENT_UNKNOWN: "未登记",
    }
    parts = " / ".join(f"{label[k]} {counts[k]}" for k in INSTRUMENTS)
    return f"共 {len(indicators or [])} 条指标 · 仪器来源:{parts}"
