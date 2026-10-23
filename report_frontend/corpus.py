# report_frontend/corpus.py
"""**有效素材分级** —— 一场会话到底能不能用来出报告。

## 为什么需要它(2026-10-18 实测背景)

44 个录制目录里混着**三种完全不同的东西**:

```
实名正式场(24):`20260929_200333-2-候选丙-000000000-投资学__20260929_200333_2e84`
                有人名 + 学号 + 学院 ⟹ 按协议采集
无名测试场(15):`20260927_212143_95a8` —— 只有时间戳 + 哈希 ⟹ 调试/试机
作废区(2):     `_回收站` / `_已作废`
```

而且**正式场之间也差得很远**。实测(逐场核过日志):

```
              脸覆盖    臂覆盖   语音日志
候选戊 f98d     98%      100%      0 行
候选丁 c9f0    100%      100%      0 行
候选甲 3d8f      0%        0%      0 行
候选庚   1320     27%       33%      0 行
```

⟹ **"在服务器上"不等于"能用"**。拿 3d8f 出报告,16 条 F 指标全是 `n_valid=0`,
读者却只会看到"面部已提供"。分级的目的就是**在出报告之前**把这件事说清楚。

## 分级的判据(每条都写死,可复核)

| 档 | 判据 |
|---|---|
| `A` 全面可用 | 正式场 · 脸 ≥ 80% · 臂 ≥ 70% |
| `B` 仅面部可用 | 正式场 · 脸 ≥ 80% · 臂 < 70% |
| `C` 仅手势可用 | 正式场 · 脸 < 80% · 臂 ≥ 70% |
| `D` 不可用 | 正式场 · 两者都不达标 |
| `T` 测试场 | 目录名里没有人名+学号 |
| `?` 无法判定 | 日志缺失,判不出来 —— **不猜** |

阈值 80% / 70% 的来历:不是标定出来的,是**"低于这个数,窗口内样本不足以支撑
会话级统计"**的操作性判据。它们住在 `TIER_THRESHOLDS` 里,改它要重跑全语料分级。

## 它**不**做的事

* 不判断素材的**质量**(模糊、背光、说话听不清)—— 那要人工看,本模块只判**可测性**;
* 不替报告层决定"要不要出报告"—— 它只**给出事实与档位**,决定权在调用方;
* 不读 `data/` 以外的任何东西,也不写任何东西(**只读**)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import media_retention

#: 目录名里"实名正式场"的判据:`-序号-姓名-学号-学院`。
#: 姓名限定 2–4 个汉字,学号 9–10 位数字 —— 实测 24 场全部命中,15 场测试场全部不命中。
FORMAL_PATTERN = re.compile(r"-\d+-([\u4e00-\u9fa5]{2,4})-(\d{9,10})-([\u4e00-\u9fa5]+)")

#: 分级阈值。见模块 docstring「分级的判据」。
TIER_THRESHOLDS = {"face": 80.0, "arm": 70.0}

#: **已裁定放弃**的场次 —— 不再进任何报告、不再重跑、不再尝试救。
#:
#: 判据不是"数据差"而是"救不回来":取景阶梯 `full→center50→center40→center30`
#: 逐级试过之后仍然测不到,再试也是同一批像素。
ABANDONED: dict[str, str] = {
    "20260928_172642_1320": (
        "候选庚:脸 27% / 臂 33%。取景阶梯试到 center40 仍然大半窗口无观测 ⟹ "
        "会话级统计的样本不足以支撑任何一条指标。**2026-10-18 使用者裁定放弃**,"
        "不重录、不进报告;保留在语料里作为「取景不足」的样本。"),
}

#: 判定"这一行有没有数据"的列(与 `l1_quality.COVERAGE_COLUMNS` 同一套判据)。
FACE_COLS = ("au12_smile", "avg_ear", "head_pitch")
ARM_COLS = ("left_elbow_angle", "right_elbow_angle")


@dataclass
class SessionAudit:
    """一场会话的可用性事实。**每个字段都是量出来的,不是估的。**"""

    session_id: str
    formal: bool = False
    name: str = ""
    crop: str | None = None
    face_coverage: float | None = None      # 0–100;None = 没有面部日志
    arm_coverage: float | None = None       # 0–100;None = 没有手势日志
    voice_rows: int = 0
    #: 盘上的 `media/camera.webm` 是否覆盖**整场**。
    #:
    #: ⚠️ 不是"文件在不在"。客户端每段录完就**整文件重传并覆盖**同一个路径
    #: (见 `retention.camera.jsonl`:多条记录、同 file、同 seq、不同 sha256),
    #: 所以盘上只剩**最后一段**。实测 13 场 A 档里候选己一场就是 9 段拼接,
    #: 而 `camera.webm` 只有 434.7 秒(整场 6959 秒)⟹ 转写只覆盖最后 7 分钟。
    has_whole_audio: bool = False
    n_questions: int = 0
    tier: str = "?"
    abandoned: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _camera_uploads(session_dir: Path) -> int:
    """`retention.camera.jsonl` 里的上传记录数;读不到返回 0。

    每段录完客户端都会把**整个** `media/camera.webm` 重传一次(同 file、同 seq=1、
    不同 sha256),所以记录数 > 1 就意味着盘上那份被后续上传覆盖过,只剩最后一段。
    """
    f = session_dir / "retention.camera.jsonl"
    if not f.exists():
        return 0
    try:
        return sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def classify(formal: bool, face: float | None, arm: float | None) -> str:
    """纯函数:**只按事实定档**,不读文件、不看时间。便于单独测。"""
    if not formal:
        return "T"
    if face is None:
        return "?"                      # 没有面部日志 ⟹ 判不出,不猜
    face_ok = face >= TIER_THRESHOLDS["face"]
    arm_ok = arm is not None and arm >= TIER_THRESHOLDS["arm"]
    if face_ok and arm_ok:
        return "A"
    if face_ok:
        return "B"
    if arm_ok:
        return "C"
    return "D"


# ── 从磁盘读事实 ──────────────────────────────────────────────────────────────

def _coverage(csv_path: Path, cols: tuple[str, ...]) -> float | None:
    """日志里**真正带数据**的行占比(0–100)。文件不存在返回 `None`。

    ⚠️ 判据是"这几列**任一**非空",不是"行数 > 0" —— 实测面部日志
    21861 行里 25.5% 整行为空,按行数算会把空日志判成"有数据"。
    """
    if not csv_path.exists():
        return None
    import pandas as pd

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None
    if len(df) == 0:
        return 0.0
    present = [c for c in cols if c in df.columns]
    if not present:
        return 0.0
    return float(df[present].notna().any(axis=1).mean() * 100.0)


def _logs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "logs"


def _manifest() -> dict:
    """离线回填清单(`data/logs/.face_backfill_manifest.json`),取景决定的**唯一**来源。"""
    p = _logs_dir() / ".face_backfill_manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def audit_session(session_dir: Path, manifest: dict | None = None) -> SessionAudit:
    """一个录制目录 → 一份事实。"""
    name = session_dir.name
    sid = name.split("__")[-1]
    m = FORMAL_PATTERN.search(name)
    man = (manifest if manifest is not None else _manifest()).get(sid) or {}

    logs = _logs_dir()
    face = _coverage(logs / f"face_au_log_{sid}.csv", FACE_COLS)
    arm = _coverage(logs / f"gesture_emotion_log_{sid}.csv", ARM_COLS)

    voice_csv = logs / f"interview_emotion_log_{sid}.csv"
    voice_rows = 0
    if voice_csv.exists():
        try:
            import pandas as pd
            voice_rows = len(pd.read_csv(voice_csv, low_memory=False))
        except Exception:
            voice_rows = 0

    q = session_dir / "questions.jsonl"
    nq = 0
    if q.exists():
        nq = sum(1 for line in q.read_text(encoding="utf-8", errors="replace").splitlines()
                 if line.strip())

    a = SessionAudit(
        session_id=sid,
        formal=bool(m),
        name=m.group(1) if m else "",
        crop=man.get("crop"),
        face_coverage=face,
        arm_coverage=arm,
        voice_rows=voice_rows,
        has_whole_audio=((session_dir / "media" / "camera.webm").exists()
                         and _camera_uploads(session_dir) <= 1),
        n_questions=nq,
    )
    a.tier = classify(a.formal, face, arm)
    if sid in ABANDONED:
        a.tier = "X"                       # 已裁定放弃 —— 与"测不到"(D)不是一回事
        a.abandoned = True
        a.notes.append(ABANDONED[sid])
    # 三条"事实之外还要说一句"的情况 —— 都是实测踩过的坑。
    _up = _camera_uploads(session_dir)
    if _up > 1:
        a.notes.append(
            f"media/camera.webm 被 {_up} 次整文件重传覆盖,盘上只剩最后一段"
            " ⟹ 转写与 V/T 两族只覆盖最后一段,而 F/G 覆盖全部段")
    elif voice_rows == 0 and a.has_whole_audio:
        a.notes.append("语音日志 0 行,但整场录音在 ⟹ V 族可从 camera.webm 重算")
    if a.formal and nq == 0:
        a.notes.append("没有 questions.jsonl ⟹ 只能出会话级指标,没有逐题窗口")
    if a.face_coverage == 0.0:
        a.notes.append("面部日志每一行都是空的(没检出脸)")
    return a


def audit_corpus(root: Path | None = None) -> list[SessionAudit]:
    """全部录制目录 → 一份清单。跳过 `_回收站` / `_已作废` / `NONE`。"""
    base = root or Path(media_retention.root())
    man = _manifest()
    out: list[SessionAudit] = []
    if not base.exists():
        return out
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or d.name == "NONE":
            continue
        out.append(audit_session(d, man))
    return out


def select(tier: str, audits: list[SessionAudit] | None = None) -> list[SessionAudit]:
    """按档取场次。`tier="A"` 就是"全面可用的那批"。"""
    return [a for a in (audits if audits is not None else audit_corpus()) if a.tier == tier]
