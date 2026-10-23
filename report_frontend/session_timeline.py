# report_frontend/session_timeline.py
"""**M4.0 统一会话时间轴** —— 把三个模态的时间列换算到会话原点。

## 结论先行(2026-10-18 实测,改过两次判断)

**账本里的 `received_at_wall` 不可信;`declared_ts` 可信。**

`received_at_wall` 取自 `time.time()`(`media_retention._record`),会被系统时钟
调整/VM 时钟漂移**回拨**;`declared_ts` 取自 `session_clock.SessionClock`,
内部是 `time.monotonic()`,**物理上不可能倒退**。

实测证据(全语料 34 场有 face 账本的场次):

```
场次                       帧数   墙钟倒退次数   declared跨度  wall跨度   差
20260925_230045_b901        50        0           47.0      48.0     -0.9
20260926_153202_2b11       300        6          316.7     299.9    +16.8
20260926_153854_1592       544       17          582.0     549.7    +32.2
20260926_155559_caf0       574       18          606.4     573.7    +32.7
20260928_172642_1320      1491        0         1070.6    2048.1   -977.5
20260929 起的 20 场            —         0           —          —       0.0
```

三种形态,都指向同一个结论:

* **09-26 那三场**:墙钟每隔约 30 帧回拨 0.7–0.9 s,累计 -14.1 s(17 次)。
  而 declared 逐帧稳定 +1.07 s、无一次倒退 ⟹ **declared 才是真实经过时间**。
* **09-28 那场 `1320`**:墙钟跑得比单调时钟**快一倍**(2048 vs 1070),同样不可信。
* **09-29 之后**:两列逐帧相同,没有这个问题。

⟹ 本模块的**主时基用 `declared_ts`**;`received_at_wall` 只在"跨模态求相位"
时用,而且**必须先检查它单调**。不单调 ⟹ 明说,并把误差上界一起报出来。

## 三档状态

* `OK`                   —— 两列一致,或墙钟单调且与 declared 同向;
* `WALL_UNRELIABLE`      —— 墙钟不单调 / 与 declared 速率差太大。**模态内可用**
                            (用 declared),**跨模态相位带误差上界**;
* `UNRECONSTRUCTIBLE`    —— 账本缺帧、行数对不上、锚是 None ⟹ 整列不换算。

## 不做什么

不补插值、不重采样、不改日志、不做跨模态融合(spec §8)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import media_retention
import session_meta

#: 留存账本里"逐帧模态"的写入者。`audio` / `camera` 是整段留存的,不在此列。
FRAME_WRITERS = ("face", "gesture")

#: 判定墙钟"跑得不对"的阈值:两列总跨度相差超过这个比例就算不可信。
#: 09-26 那三场是 5.9%/5.7%,09-28 `1320` 是 91% ⟹ 5% 能分开,又不至于把
#: 正常的采样抖动(09-25 `b901` 的 2%)误判。
WALL_RATE_TOLERANCE = 0.05


class TimebaseStatus(str, Enum):
    """一条时间轴可不可信。`str` 混入是为了直接进 JSON。"""

    OK = "ok"
    WALL_UNRELIABLE = "wall_unreliable"
    UNRECONSTRUCTIBLE = "unreconstructible"
    NO_LEDGER = "no_question_ledger"


@dataclass
class Timebase:
    """一个模态换算出来的结果。`t_sec` 与 `df` 的 index **一一对齐**。"""

    modality: str
    status: TimebaseStatus
    t_sec: pd.Series
    n_frames: int = 0
    wall_back_steps: int = 0           # 墙钟倒退次数(0 才可信)
    drift_sec: float | None = None     # declared 与 wall 总跨度之差
    gap: str | None = None
    #: 这条时间轴**覆盖到的时间范围**(秒)= `t_sec.max() - t_sec.min()`,含重启空档。
    #:
    #: ⚠️ 与 `n_frames` 是两个量,不能互相替代:1 Hz 的 6968 帧可以跨 7427 秒
    #: (中间有 8 处重启空档),而 31.25 Hz 的 13585 帧只跨 434.7 秒。
    #: 判断"两族看的是不是同一段时间"要比 `span_sec` —— 候选B那场就是
    #: 面部/手势 7427 秒、语音 434.7 秒(视频只剩最后一段),比帧数根本看不出来。
    span_sec: float | None = None

    def __post_init__(self) -> None:
        if self.span_sec is None and self.t_sec is not None and len(self.t_sec):
            v = pd.to_numeric(self.t_sec, errors="coerce").dropna()
            self.span_sec = float(v.max() - v.min()) if len(v) >= 2 else 0.0

    @property
    def usable(self) -> bool:
        """这一列能不能拿去做**模态内**的时间窗指标。

        `WALL_UNRELIABLE` 也算可用 —— 模态内用 `declared_ts`,它没坏。
        只有 `UNRECONSTRUCTIBLE` / `NO_LEDGER` 不可用。
        """
        return self.status in (TimebaseStatus.OK, TimebaseStatus.WALL_UNRELIABLE)


@dataclass
class SessionTimeline:
    session_id: str
    status: TimebaseStatus
    origin_wall: float | None = None
    modalities: dict[str, Timebase] = field(default_factory=dict)
    gap: str | None = None

    @property
    def usable(self) -> bool:
        return self.status in (TimebaseStatus.OK, TimebaseStatus.WALL_UNRELIABLE)

    def t_sec(self, modality: str) -> pd.Series | None:
        tb = self.modalities.get(modality)
        return None if tb is None or not tb.usable else tb.t_sec


# ── 读写盘 ────────────────────────────────────────────────────────────────────

def _read_ledger(session_id: str, writer: str) -> list[dict]:
    """读账本。`media_retention.root()` 每次调用都读环境变量 ⟹ 测试可换目录。"""
    path = media_retention.resolve_recording_dir(session_id) / (
        f"{media_retention.LEDGER_PREFIX}.{writer}{media_retention.LEDGER_SUFFIX}")
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # 碎行/半行:与 `experiments/replay_retained.py` 同一处置 —— 跳过,不崩。
            continue
    return out


def frames_ledger(session_id: str, modality: str) -> pd.DataFrame:
    """该模态的 frame 账行,按 `seq` 排。缺 `declared_ts` / `received_at_wall` 的行剔除。"""
    rows = [r for r in _read_ledger(session_id, modality)
            if r.get("kind") == "frame" and r.get("modality") == modality]
    if not rows:
        return pd.DataFrame(columns=["seq", "declared_ts", "received_at_wall"])
    df = pd.DataFrame(rows)[["seq", "declared_ts", "received_at_wall"]].copy()
    df = df[df["declared_ts"].notna() & df["received_at_wall"].notna()]
    return df.sort_values("seq").reset_index(drop=True)


# ── 原点 ──────────────────────────────────────────────────────────────────────

def session_origin_wall(session_id: str,
                        allow_frame_origin: bool = False) -> tuple[float | None, str | None]:
    """会话原点 = 最早一次 `ask_start`(墙钟)。返回 `(origin, gap)`。

    它是**真实事件**(第一次提问),两个服务都亲眼见过,不依赖客户端时钟。
    但它是墙钟量 —— 如果这场的墙钟不可信,原点也会带着同样的偏差,
    所以跨模态相位只能算到"误差上界"这个精度,见 `Timebase.drift_sec`。
    """
    try:
        questions = session_meta.read_questions(session_id)
    except Exception as exc:
        return None, f"提问台账读不出:{exc}"
    if not questions:
        why = "本场没有提问台账(戊-15:自然对话的粒度待裁定)"
        if allow_frame_origin:
            return _origin_from_first_frame(session_id, why)
        return None, why
    starts = [float(q["ask_start"]) for q in questions if q.get("ask_start") is not None]
    if not starts:
        why = "提问台账里没有 ask_start"
        if allow_frame_origin:
            return _origin_from_first_frame(session_id, why)
        return None, why
    return min(starts), None


def _origin_from_first_frame(session_id: str, why: str) -> tuple[float | None, str | None]:
    """**没有提问台账时的原点** = 最早一个留存帧的 `received_at_wall`。

    ## 为什么必须有这一支(2026-10-18 实测)

    正式场(24 场实名场)**全都没有 `questions.jsonl`**,于是原点恒为空,
    `compute_session_l1` 在开头就 `return`,**整场一条指标都不出** ——
    哪怕面部覆盖 98%、手势 100%、整场音频都在。

    实测 `20260928_221315_c9f0`:`retention.face.jsonl` **163/163 行都有 `declared_ts`**,
    时间轴完全建得起来。挡住它的不是数据,是"原点只能来自第一次提问"这个假设。

    ## 为什么这样切不违反戊-15

    戊-15 待裁定的是**窗口粒度**(自然对话怎么切成"回答"),不是**原点**。
    原点只是"t=0 在哪",取第一个留存帧不需要任何判断;窗口粒度仍然
    **不决定** —— 本函数只在 `gap` 里如实写明"只能出会话级指标",
    `build_windows` 拿到没有台账的场次也只给**一个 `session` 窗口**。

    ## 精度

    `received_at_wall` 是 `time.time()`,本场可能不单调(见丙-21)。但它在这里
    **只当相位参考**:主时基仍是账本的 `declared_ts`,`drift_sec` 照旧计算并上报。
    所以这一支不引入新的误差来源,只是把"没有原点"降级成"原点带墙钟偏差"。
    """
    for modality in FRAME_WRITERS:
        try:
            df = frames_ledger(session_id, modality)
        except Exception:
            continue
        if len(df):
            return float(df["received_at_wall"].iloc[0]), (
                f"{why};原点改取首个留存帧的墙钟(会话级粒度,逐题窗口仍待戊-15裁定)")
    return None, why


def audio_timebase(series) -> Timebase | None:
    """从**帧序列**造一条 `voice` 时间轴。

    ## 为什么需要(2026-10-18 实测)

    正式场 24/24 场的语音**日志**是 0 行,但整场录音 `camera.webm` 在。
    `voice_series` 从音频重算出逐帧序列后,V 族那 8 条序列指标就有值了 ——
    可时间轴上**没有** `voice` 这个模态,于是质量报告的硬不变量会(正确地)
    拦下"`voiced_ratio` 有值但 voice 模态不存在"。

    这不是不变量错了,是**前提错了**:voice 模态**在**,只不过它的来源是
    原始音频而不是日志。

    ## 口径

    `t_sec = arange(n) × frame_sec`,以音频起点为 0。整场录音的起点
    就是会话起点(录制随会话开始),所以这条轴与其它模态**同相位**;
    但它**没有墙钟可比对** ⟹ `drift_sec=None`、`wall_back_steps=0`,
    并在 `gap` 里写明"来源是原始音频,未与墙钟核对"。
    """
    if not series:
        return None
    import numpy as _np

    total = sum(int(len(s.f0)) for s in series)
    if total == 0:
        return None
    frame = series[0].frame_sec or 0.0
    if frame <= 0:
        return None
    return Timebase(modality="voice", status=TimebaseStatus.OK,
                    t_sec=pd.Series(_np.arange(total, dtype=float) * frame),
                    n_frames=total, wall_back_steps=0, drift_sec=None,
                    gap="时间轴由原始音频的帧序列重建(未与墙钟核对)")


# ── 单模态换算 ────────────────────────────────────────────────────────────────

def _time_column(df: pd.DataFrame, modality: str) -> str | None:
    """日志里代表时间的那一列。按模态给出候选,按顺序取第一个存在的。"""
    candidates = {
        "face": ("timestamp",),
        "gesture": ("timestamp", "timestamp_iso"),
        "voice": ("unix_timestamp", "timestamp"),
        "voice_interview": ("unix_timestamp", "timestamp"),
    }.get(modality, ("timestamp",))
    for name in candidates:
        if name in df.columns:
            return name
    return None


def align_frame_modality(session_id: str, modality: str, df: pd.DataFrame,
                         origin_wall: float) -> Timebase:
    """把一个逐帧模态的 `df` 对齐到会话轴。

    **主时基是账本的 `declared_ts`**(monotonic 派生),`received_at_wall` 只做两件事:
    ① 判它自己可不可信;② 给"本模态的 0 点距会话原点多少秒"提供一个相位。

    日志里的时间列只用来**核对**,不直接用 —— 因为它要么是 declared 的秒版(face),
    要么是墙钟(gesture),两者混在一起会让人以为"减个常数就行"。
    """
    n = len(df)
    if n == 0:
        return Timebase(modality, TimebaseStatus.UNRECONSTRUCTIBLE,
                        pd.Series(dtype=float), gap="日志为空")

    ledger = frames_ledger(session_id, modality)
    if ledger.empty:
        return Timebase(modality, TimebaseStatus.UNRECONSTRUCTIBLE,
                        pd.Series([np.nan] * n, index=df.index), n_frames=n,
                        gap=f"账本里没有 {modality} 的 frame 行 —— 没有可挂靠的 declared_ts")

    if len(ledger) != n:
        # 不硬配:按序硬配会把相邻两帧的时间戳串位,而且是静默的。
        return Timebase(modality, TimebaseStatus.UNRECONSTRUCTIBLE,
                        pd.Series([np.nan] * n, index=df.index),
                        n_frames=n, gap=f"日志 {n} 行 vs 账本 {len(ledger)} 帧 —— 行数对不上")

    declared = ledger["declared_ts"].to_numpy(dtype=float) / 1000.0   # 秒,会话相对
    wall = ledger["received_at_wall"].to_numpy(dtype=float)           # 纪元秒

    # ① 墙钟体检:倒退次数 + 与 declared 的速率差。
    wall_back = int((np.diff(wall) < 0).sum())
    d_declared = declared[-1] - declared[0]
    d_wall = wall[-1] - wall[0]
    rate_off = abs(d_wall - d_declared) / d_declared if d_declared > 0 else 0.0
    wall_bad = wall_back > 0 or rate_off > WALL_RATE_TOLERANCE

    # ② 相位:本模态 declared=0 那一刻的墙钟,距会话原点多少秒。
    #    墙钟可信时它就是真相位;墙钟不可信时它带着误差,误差上界 = 墙钟累计倒退量。
    phase = float(wall[0]) - declared[0] - origin_wall
    if wall_bad:
        back_total = float(-np.diff(wall)[np.diff(wall) < 0].sum())
        gap_note = (f"{modality} 的墙钟不可信:{wall_back} 次倒退(累计 {back_total:.1f} s),"
                    f"且总跨度与 declared 差 {d_wall - d_declared:+.1f} s;"
                    f"时间轴改用 declared_ts,跨模态相位误差上界 ≈{back_total:.1f} s")
        status = TimebaseStatus.WALL_UNRELIABLE
    else:
        gap_note = None
        status = TimebaseStatus.OK

    t = pd.Series(declared + phase, index=df.index)
    return Timebase(modality, status, t, n_frames=n,
                    wall_back_steps=wall_back,
                    drift_sec=round(d_wall - d_declared, 3),
                    gap=gap_note)


def align_voice(df: pd.DataFrame, session_id: str, origin_wall: float) -> Timebase:
    """语音:`unix_timestamp` 是**墙钟**,减原点即可。

    语音没有逐帧留存账本(音频是整段存的,账本里 `declared_ts` 恒为 `None`),
    所以这里**没有第二根时钟可以交叉验证**。它的墙钟可不可信,只能靠调用方
    用别的模态的体检结果来判(见 `build_timeline` 的汇总)。
    """
    n = len(df)
    if n == 0:
        return Timebase("voice", TimebaseStatus.UNRECONSTRUCTIBLE,
                        pd.Series(dtype=float), gap="日志为空")
    col = _time_column(df, "voice")
    if col is None:
        return Timebase("voice", TimebaseStatus.UNRECONSTRUCTIBLE,
                        pd.Series([np.nan] * n, index=df.index),
                        gap="语音日志里没有 unix_timestamp / timestamp 列")
    wall = pd.to_numeric(df[col], errors="coerce")
    if wall.isna().any():
        return Timebase("voice", TimebaseStatus.UNRECONSTRUCTIBLE,
                        pd.Series([np.nan] * n, index=df.index),
                        gap=f"语音日志的 {col} 列里有空值")
    return Timebase("voice", TimebaseStatus.OK,
                    pd.Series(wall.to_numpy(dtype=float) - origin_wall, index=df.index),
                    n_frames=n)


# ── 组装 ──────────────────────────────────────────────────────────────────────

def build_timeline(session_id: str,
                   logs: dict[str, pd.DataFrame] | None = None,
                   reader: Callable[[str], pd.DataFrame] | None = None,
                   allow_frame_origin: bool = False) -> SessionTimeline:
    """把一场会话所有能拿到的时间轴建出来。

    `logs`:"模态 → 日志 DataFrame"(通常来自 `LogDataLoader`);
    `reader`:兜底的读取函数。两者都不给 ⟹ 只建原点,不建模态。
    `allow_frame_origin`:**显式**允许在没有提问台账时用首个留存帧的墙钟当原点。
        默认 **False** —— 戊-15 那批自然对话的裁定是"没有台账就不建轴",
        那条决定不因为这里多了一个能力就被推翻;需要会话级指标的调用方
        (正式场:`compute_session_l1`)自己显式开启。
    """
    origin, origin_gap = session_origin_wall(session_id,
                                             allow_frame_origin=allow_frame_origin)
    if origin is None:
        return SessionTimeline(session_id, TimebaseStatus.NO_LEDGER, gap=origin_gap)

    logs = dict(logs or {})
    if reader is not None:
        for m in FRAME_WRITERS + ("voice",):
            if m not in logs:
                try:
                    got = reader(m)
                except Exception:
                    got = None
                if isinstance(got, pd.DataFrame):
                    logs[m] = got

    timeline = SessionTimeline(session_id, TimebaseStatus.OK, origin_wall=origin)
    for modality, df in logs.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        # 键一律用 **Timebase 自报的 modality**,不用调用方传进来的键 ——
        # `LogDataLoader` 把语音叫 `voice_interview` / `voice_research`,而时间轴
        # 只认 `voice`。两套键并存会让质量报告说"voice 未提供"(其实提供了),
        # 这类"名字不一致导致的假缺失"正是本仓反复踩过的坑。
        if modality in FRAME_WRITERS:
            tb = align_frame_modality(session_id, modality, df, origin)
        elif modality in ("voice", "voice_interview", "voice_research"):
            tb = align_voice(df, session_id, origin)
        else:
            continue
        timeline.modalities[tb.modality] = tb

    if not timeline.modalities:
        # 原点找到了,但一个模态都没映射上来。状态仍是 OK(轴本身成立),
        # 但**必须说出来** —— 否则 `status == "ok"` 会让人以为这一场有可用数据。
        timeline.gap = "时间轴成立,但没有任何模态日志被提供"

    # 汇总:只要有一个模态的墙钟坏了,整场的"跨模态相位"就带误差 ——
    # 这是**会话级**的结论,不能只写在单个模态上(语音自己没有可核对的时钟)。
    if any(tb.status is TimebaseStatus.WALL_UNRELIABLE
           for tb in timeline.modalities.values()):
        timeline.status = TimebaseStatus.WALL_UNRELIABLE
        worst = max((tb.drift_sec or 0) for tb in timeline.modalities.values())
        timeline.gap = (f"本场墙钟至少在一个模态上不可信(最大偏差 {worst:+.1f} s);"
                        f"模态内指标用 declared_ts,跨模态相位带同等量级的不确定度")
    return timeline
