"""媒体完整性 —— 录到的秒数够不够覆盖这一场任务的时间跨度。

**纯函数,不读盘、不碰 ffprobe、无副作用。** 时长由调用方量好传进来
(`media_truncation_probe.py` 负责量,`capture_health._span_sec` 是量具)。
这样这张判定逻辑可以在离线机器上单测,也可以被后端在**不碰文件**的地方复用。

为什么需要它(戊-16):

`questions.jsonl` 里有每道题的 `ask_start` / `ask_end`(**epoch 绝对秒**),
两者一减就是这一场任务**应该**覆盖的时间跨度。盘上实测确实存在两类硬伤:

| 会话 | 题目跨度 | camera.webm | 现象 |
|---|---|---|---|
| `20260925_225936_a971` | 6.24 s | 0.92 s | 跨度 6.7 倍于录像 |
| `20260928_230839_87c9` | 5.41 s | 3.43 s | 只落到 63% |
| `20260928_165520_90e7` | 6.23 s | 284.37 s | 录像 45.7 倍于跨度 |
| `20260927_191325_253a` | 6.23 s | 189.55 s | 30.4 倍 |
| `20260927_211150_4579` | 5.41 s | 83.29 s | 15.4 倍 |

前两行是**媒体没录全**,后三行是**台账缺题**(录了 5 分钟,`questions.jsonl`
里只有 1 道题,跨度 6 秒)。两类都要报,但**判词不同**,补救动作也相反:
前者要重录或标注样本不完整,后者要补台账而不是补录像。

⚠️ **不新增第 10 类缺口码。** 本仓对外口径是 `report_frontend.l1_indicators.KINDS`
里的 9 类,这里只做映射:

| verdict | gap | 为什么是这一类 |
|---|---|---|
| `truncated` | `no_samples` | 时间窗里没有观测 —— 秒数就是不够,样本不完整 |
| `overlong` | `time_unusable` | 时间轴对不上(录了 284 秒,台账只有 6 秒)⟹ 按台账切窗必然错位 |
| `ok` | `None` | 没有缺口 |

具体是哪一类问题写在 `gap_detail` 的**人话**里,**不新增码**(自由文本没法统计,
见 `l1_indicators` 里 `KINDS` 上面那段注释)。

★ 一个实测纠正(2026-11-21,本次落地时量的):

任务书里写「`20260926_153854_1592` 题目跨度 520.7 s,camera.webm 仅 57.06 s」。
**57.06 秒是这个文件里的音轨,不是录像。** 同一个 `camera.webm` 实测:
视轨 `v:0` 跨度 **550.514 s**、音轨 `a:0` 跨度 **57.00 s**。所以

* 按**文件**算(`dur` 取所有媒体里的最大值,即本模块的 `assess`):550.5 / 520.7
  = **1.057 → `ok`**,这一场的录像**没有**短于跨度;
* 按**音轨**算:520.7 / 57.0 = **9.13** —— 这正好是任务书里那个 9.13,
  它说的是**音轨被截断**,由 `capture_health.probe` 的 `av_ratio`(0.104 ≪ 0.8)
  当场报出来,不是本模块的职责。

本模块的判据是「录像/音频的**文件级**时长 vs 任务跨度」,所以 1592 判 `ok`;
`media_truncation_probe.probe_session(track="audio")` 可以复现按音轨算的
`truncated`(见该文件的 `track` 参数)。**两个数都真,别只信一个。**

判据:

    overlong   dur > ceiling * span     (先判:它的 dur 也满足 >= tol*span)
    ok         dur >= tol * span
    truncated  dur <  tol * span

`tol=0.9` / `ceiling=1.5` 是**容忍带**,不是拟合出来的最优阈值:0.9 的意思是
「录到的秒数至少覆盖任务的九成」,1.5 的意思是「比任务长一半以上就不像同一件事」。
这两个数是运维口径,换语料不用重标;要调就显式传参,**不要**改默认值再声称验证过。
"""
from __future__ import annotations

import math
from typing import Any, Iterable

__all__ = [
    "assess", "VERDICT_OK", "VERDICT_TRUNCATED", "VERDICT_OVERLONG",
    "VERDICTS", "GAP_FOR_VERDICT", "DEFAULT_TOL", "DEFAULT_CEILING",
    "GAP_NO_SAMPLES", "GAP_TIME_UNUSABLE",
]

#: 三态。**只有这三个** —— 判定必须能被统计,不能是自由文本。
VERDICT_OK = "ok"
VERDICT_TRUNCATED = "truncated"
VERDICT_OVERLONG = "overlong"
VERDICTS = (VERDICT_OK, VERDICT_TRUNCATED, VERDICT_OVERLONG)

#: 映射到系统既有的缺口码(`report_frontend.l1_indicators.KINDS`,共 9 类)。
#: ⚠️ 这两个字面量是**抄自** `l1_indicators.py`,不是新增 —— 那边改了这里要跟着改,
#: 单测 `test_gap_codes_are_the_existing_nine` 会在两边对不上时报错。
GAP_NO_SAMPLES = "no_samples"      # 样本不完整:时间窗里一条观测都没有
GAP_TIME_UNUSABLE = "time_unusable"  # 时间轴不可重建:台账与媒体对不上

GAP_FOR_VERDICT: dict[str, str | None] = {
    VERDICT_OK: None,
    VERDICT_TRUNCATED: GAP_NO_SAMPLES,
    VERDICT_OVERLONG: GAP_TIME_UNUSABLE,
}

#: 录到秒数 >= 任务跨度的这个比例 ⟹ 不算截断。
DEFAULT_TOL = 0.9
#: 录到秒数 > 任务跨度的这个倍数 ⟹ 录像远长于任务(多为台账缺题)。
DEFAULT_CEILING = 1.5


def _num(x: Any) -> float | None:
    """把 JSON 里的时间戳转成 float。转不了 / 不是有限数 → None(**不许**当 0 用)。

    `questions.jsonl` 正常是 float,但上游换过写法,出现过字符串与 `null`;
    把 `null` 当 0 会让 `span` 变成一个巨大的假跨度 —— 那正是"判词比事实更狠"。
    """
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str):
        try:
            v = float(x.strip())
        except ValueError:
            return None
    else:
        return None
    return v if math.isfinite(v) else None


def _question_span(questions: Iterable[dict] | None) -> tuple[float | None, dict]:
    """`(跨度, 统计)`。跨度 = `max(ask_end) - min(ask_start)`,量不到返回 None。

    两列**分别**取极值(而不是逐题相减),因为上游偶有 `ask_end` 缺行的题;
    分别取极值能把这类缺行容忍掉,而逐题相减会把整场判成"没有跨度"。
    """
    starts: list[float] = []
    ends: list[float] = []
    n_rows = 0
    for q in questions or ():
        if not isinstance(q, dict):
            continue
        n_rows += 1
        s = _num(q.get("ask_start"))
        e = _num(q.get("ask_end"))
        if s is not None:
            starts.append(s)
        if e is not None:
            ends.append(e)
    stats = {
        "n_questions": n_rows,
        "n_ask_start": len(starts),
        "n_ask_end": len(ends),
        "first_ask_start": round(min(starts), 3) if starts else None,
        "last_ask_end": round(max(ends), 3) if ends else None,
    }
    if not starts or not ends:
        return None, stats
    return max(ends) - min(starts), stats


def _durations(media_durations: dict[str, Any] | None) -> tuple[float | None, dict[str, float | None]]:
    """`(最长媒体秒数, 规整后的明细)`。

    `None` / `NaN` / 负数一律当"这一份没量到"并记进明细(排查时要知道是哪一份),
    **不参与取最大值** —— 把量不到的当 0 会让一场正常的录像被判成截断。
    """
    clean: dict[str, float | None] = {}
    best: float | None = None
    best_key: str | None = None
    for name, raw in (media_durations or {}).items():
        v = _num(raw)
        if v is None or v < 0:
            clean[str(name)] = None
            continue
        clean[str(name)] = round(v, 3)
        if best is None or v > best:
            best, best_key = v, str(name)
    return best, clean


def assess(
    questions: list[dict],
    media_durations: dict[str, float],
    *,
    tol: float = DEFAULT_TOL,
    ceiling: float = DEFAULT_CEILING,
) -> dict:
    """这一场的媒体够不够覆盖任务跨度。→ **只含事实与判定**的 dict,**永不抛异常**。

    参数:
      questions        `questions.jsonl` 的行(含 `ask_start`/`ask_end`,epoch 绝对秒)
      media_durations  `{文件名: 解码时长秒}` —— 由调用方量好传入(见
                       `media_truncation_probe.probe_session`)。值为 `None` 表示
                       这一份**没量到**,不参与比较;全部为 `None` ⟹ 无从判定。
      tol              `dur >= tol * span` 即 `ok`(默认 0.9)
      ceiling          `dur > ceiling * span` 即 `overlong`(默认 1.5)

    返回:
      verdict     `"ok" | "truncated" | "overlong"`
      span_sec    题目跨度(秒);量不到 / 不为正 → `None`
      media_sec   最长媒体时长(秒);一份都没量到 → `None`
      ratio       `media_sec / span_sec`(>= tol 通过;< tol 截断;> ceiling 超长)。
                  **不是** `span/media` —— 任务书里那个 `9.13` 是 `1/ratio`,见
                  `evidence["span_over_media"]`。无从比较 → `None`
      gap         `"no_samples" | "time_unusable" | None`,**只用既有 9 类**
      gap_detail  人话,说清是哪一类问题(不新增码)
      evidence    取证:阈值、参与比较的媒体明细、哪一份最长、题目行数 …
      judgeable   `ratio` 是否真的可比较。`False` 时 `verdict` 是 `"ok"` 但
                  **不等于「检查通过」**,只是「没得比」—— 别把两者读成一样。

    ⚠️ 三态里没有"unknown",而"没有题目时间戳 / 跨度 ≤ 0 / 一份媒体都没量到"
    这三件事必须能表达出来。做法:`verdict="ok"` 且 `judgeable=False`、
    `gap=None`、`gap_detail` 写明原因。**不新增第 10 类缺口码**,也不把"量不到"
    伪装成"坏"或"好"。
    """
    tol_f = _num(tol)
    ceiling_f = _num(ceiling)
    if tol_f is None or ceiling_f is None or tol_f < 0 or ceiling_f < 0:
        raise ValueError(f"tol/ceiling 必须是非负数:{tol!r} / {ceiling!r}")

    span, q_stats = _question_span(questions)
    dur, media_clean = _durations(media_durations)
    longest = None
    if dur is not None and media_clean:
        # 最大值对应的那一份(并列时取名字序第一,保证可复现)
        cands = sorted((v, k) for k, v in media_clean.items() if v is not None)
        longest = cands[-1][1] if cands else None

    evidence: dict = {
        "tol": tol_f,
        "ceiling": ceiling_f,
        "thresholds_sec": None,
        "media_sec_by_file": media_clean,
        "longest_media": longest,
        "n_media_files": len(media_clean),
        "n_media_measured": sum(1 for v in media_clean.values() if v is not None),
        "span_over_media": None,
        **q_stats,
    }

    # ── 不可比较的三种输入 ────────────────────────────────────────────────
    if span is None:
        return {
            "verdict": VERDICT_OK, "span_sec": None, "media_sec": _round(dur),
            "ratio": None, "gap": None, "judgeable": False,
            "gap_detail": "没有可用的 ask_start/ask_end —— 算不出任务跨度,"
                          "既不判截断也不判超长(不是通过:根本没比)",
            "evidence": evidence,
        }
    span_r = _round(span)
    evidence["thresholds_sec"] = {"truncated_below": round(tol_f * span, 3),
                                  "overlong_above": round(ceiling_f * span, 3)}
    if span <= 0:
        return {
            "verdict": VERDICT_OK, "span_sec": span_r, "media_sec": _round(dur),
            "ratio": None, "gap": None, "judgeable": False,
            "gap_detail": f"题目跨度 ≤ 0({span_r}s:只有一题,或时间戳相同)—— "
                          f"没有「应有跨度」可比,不判截断也不判超长(不是通过)",
            "evidence": evidence,
        }
    if dur is None:
        return {
            "verdict": VERDICT_OK, "span_sec": span_r, "media_sec": None,
            "ratio": None, "gap": None, "judgeable": False,
            "gap_detail": "一份媒体都没量到时长(ffprobe 不在 / 文件读不懂 / 没有录像)"
                          " —— 无从比较,不是通过",
            "evidence": evidence,
        }

    ratio = dur / span
    evidence["span_over_media"] = _round(span / dur, 3) if dur else None
    base = {
        "span_sec": span_r,
        "media_sec": _round(dur),
        "ratio": _round(ratio, 3),
        "judgeable": True,
        "evidence": evidence,
    }

    # ── 三态判定。**overlong 必须先判**:超长场次的 dur 也满足 `>= tol*span`,
    #    先判 ok 会把 90e7(284s / 6.2s)读成"通过"。
    if dur > ceiling_f * span:
        gap_detail = (
            f"录像 {_round(dur)}s 是题目跨度 {span_r}s 的 {_round(ratio, 1)} 倍"
            f"(上限 {ceiling_f}×)—— 媒体远长于任务,通常是**台账缺题**:"
            f"{q_stats['n_questions']} 道题的 ask_start/ask_end 覆盖不了那么长的时间轴,"
            f"按台账切窗会与画面对不上(时间轴不可重建)")
        return {**base, "verdict": VERDICT_OVERLONG,
                "gap": GAP_TIME_UNUSABLE, "gap_detail": gap_detail}
    if dur >= tol_f * span:
        return {**base, "verdict": VERDICT_OK, "gap": None,
                "gap_detail": f"媒体 {_round(dur)}s 覆盖了题目跨度 {span_r}s 的 "
                              f"{_round(ratio * 100, 1)}%(门槛 {tol_f:.0%})"}
    gap_detail = (
        f"媒体只有 {_round(dur)}s,而题目跨度 {span_r}s —— 覆盖率 "
        f"{_round(ratio * 100, 1)}%,低于门槛 {tol_f:.0%}"
        f"({span_r}s 需要至少 {_round(tol_f * span)}s)"
        + (f";最长的是 {longest}" if longest else "")
        + ";整段末尾的样本根本没落下")
    return {**base, "verdict": VERDICT_TRUNCATED,
            "gap": GAP_NO_SAMPLES, "gap_detail": gap_detail}


def _round(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)
