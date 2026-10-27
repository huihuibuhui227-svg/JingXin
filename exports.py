"""数据导出与图表 —— 把「采集下来的原始表」直接交给分析的人。

报告层只印过了证据门的指标；而复核、换算法、自己画图，需要的是**三份原始 CSV**
和**语音转写文本**。这些一直都在盘上（`data/logs/*.csv`、会话目录的
`transcript.json`），只是此前没有出口。这里补上：

* `GET /api/export/<sid>/files`        这一场有哪些可导出的东西（大小/行数/段数）
* `GET /api/export/<sid>/csv/<key>`    下载某一份 CSV（face / gesture / voice_*）
* `GET /api/export/<sid>/transcript`   转写导出：`txt`（人读）/ `json`（原始）/ `csv`（题干+回答）
* `GET /api/export/<sid>/chart`        按模块画 PNG（matplotlib，不依赖前端/CDN）

⚠️ 图表只画**盘上真实存在的列**，缺数据就如实说"没有可画的列"，不补零、不插值。
"""
from __future__ import annotations

import csv
import io
import json
import math
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Blueprint, Response, jsonify, request, send_file

import media_retention

bp = Blueprint("data_export", __name__)

#: CSV 键 → 中文名。**键是接口契约**，前端按它拼下载地址。
LOG_KEYS: list[tuple[str, str]] = [
    ("face", "面部表情（AU / 视线）"),
    ("gesture", "手势与体态"),
    ("voice_interview", "语音（面试）"),
    ("voice_research", "语音（科研）"),
]

#: 画图只认这三个模块；voice 自动挑"有数据行"的那份 CSV。
CHART_MODULES = {"face": "面部表情", "gesture": "手势与体态", "voice": "语音"}

_MPL_LOCK = threading.Lock()


class NoData(Exception):
    """这一场没有可画的数据（不是 500：请求没问题，是没东西可画）。"""


def _repo() -> Path:
    return Path(__file__).resolve().parent


def _logs_dir() -> Path:
    return _repo() / "data" / "logs"


def _log_path(sid: str, key: str) -> Path | None:
    table = {
        "face": _logs_dir() / f"face_au_log_{sid}.csv",
        "gesture": _logs_dir() / f"gesture_emotion_log_{sid}.csv",
        "voice_interview": _logs_dir() / f"interview_emotion_log_{sid}.csv",
        "voice_research": _logs_dir() / f"research_emotion_log_{sid}.csv",
    }
    return table.get(key)


def _rows(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _session_dir(sid: str) -> Path:
    return media_retention.resolve_recording_dir(sid)


def _read_transcript(sid: str) -> dict | None:
    p = _session_dir(sid) / "transcript.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_questions(sid: str) -> list[dict]:
    p = _session_dir(sid) / "questions.jsonl"
    out: list[dict] = []
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    out.sort(key=lambda r: r.get("index", 0))
    return out


def list_payload(sid: str) -> dict:
    """这一场可导出的东西。**只陈述盘上事实**，不猜。"""
    dir_ = _session_dir(sid)
    logs = []
    for key, label in LOG_KEYS:
        p = _log_path(sid, key)
        exists = bool(p and p.is_file())
        logs.append({
            "key": key,
            "label": label,
            "file": p.name if p else "",
            "exists": exists,
            "bytes": p.stat().st_size if exists else 0,
            "rows": _rows(p) if exists else 0,
        })
    tr = _read_transcript(sid)
    transcript = {
        "exists": tr is not None,
        "n_segments": len(tr.get("segments") or []) if tr else 0,
        "n_chars": (tr.get("merged") or {}).get("n_chars") if tr else None,
        "recorded_at": tr.get("recorded_at") if tr else None,
        "preview": (((tr.get("merged") or {}).get("text") or "")[:240]) if tr else "",
        "bytes": (dir_ / "transcript.json").stat().st_size
        if (dir_ / "transcript.json").is_file() else 0,
    }
    questions = _read_questions(sid)
    # 这一场是谁的（与报告头同一份 `label.json`）。导出页要按人对号，只有 sid 不够。
    try:
        from session_meta import read_label
        rec = read_label(sid) or {}
    except Exception:
        rec = {}
    subject = {k: rec[k] for k in ("name", "student_id", "department", "serial")
               if rec.get(k)}
    return {
        "session_id": sid,
        "subject": subject,
        "dir_name": dir_.name,
        "logs": logs,
        "transcript": transcript,
        "questions": len(questions),
        "charts": sorted(CHART_MODULES),
    }


# ── CSV 下载 ───────────────────────────────────────────────────────────────

def csv_file(sid: str, key: str) -> tuple[Path, str]:
    p = _log_path(sid, key)
    if p is None:
        raise KeyError(f"未知的数据表 {key!r}")
    if not p.is_file():
        raise FileNotFoundError(f"这一场没有 {key} 的数据表")
    # 下载名用 ASCII 键(`<sid>-face.csv`):中文名在 `Content-Disposition` 的
    # ASCII 回退里会被剥成空格/括号(实测 `-(AU _ ).csv`),旧浏览器会存成一团。
    return p, f"{sid}-{key}.csv"


# ── 语音转写 ───────────────────────────────────────────────────────────────

def transcript_payload(sid: str) -> dict:
    tr = _read_transcript(sid)
    if tr is None:
        raise FileNotFoundError("这一场没有语音转写（transcript.json 不存在）")
    return tr


def _qa_rows(sid: str) -> list[tuple[int, str, str]]:
    """(题号, 题干, 回答)。题干来自 `questions.jsonl`；没有台账就只有回答。"""
    tr = _read_transcript(sid) or {}
    qs = {int(r.get("index", i)): str(r.get("qid") or "")
          for i, r in enumerate(_read_questions(sid))}
    rows = []
    for i, seg in enumerate(tr.get("segments") or []):
        seg_i = int(seg.get("index", i))
        rows.append((seg_i, qs.get(seg_i, ""), str(seg.get("text") or "")))
    return rows


def _label_line(sid: str) -> str:
    """这一场是谁的。读不到标注就如实写"未标注"，不空着。"""
    try:
        from session_meta import read_label
        rec = read_label(sid) or {}
    except Exception:
        return "被测者：（标注读取失败 —— 检查会话目录 label.json）"
    fields = [("姓名", rec.get("name")), ("学号", rec.get("student_id")),
              ("院系", rec.get("department")), ("场次", rec.get("serial"))]
    got = [f"{k}：{str(v).strip()}" for k, v in fields if v and str(v).strip()]
    return "被测者：" + ("　".join(got) if got else "（未标注）")


def transcript_txt(sid: str) -> str:
    tr = _read_transcript(sid)
    if tr is None:
        raise FileNotFoundError("这一场没有语音转写（transcript.json 不存在）")
    merged = tr.get("merged") or {}
    rows = _qa_rows(sid)
    head = [
        "JingXin 语音转写导出",
        f"会话：{sid}",
        _label_line(sid),
        f"录制时间：{tr.get('recorded_at') or '—'}",
        f"转写段落：{len(rows)}　字数：{merged.get('n_chars') or '—'}",
        "说明：FunASR 自动转写，未经人工校对；标点/数字可能有误，引用前请对照原始录音。",
        "",
    ]
    body = []
    for idx, q, a in rows:
        body.append("─" * 46)
        body.append(f"第 {idx + 1} 题" + (f"：{q}" if q else "（本场没有提问台账）"))
        body.append(f"答：{a}" if a else "答：（空）")
        body.append("")
    return "\n".join(head + body)


def transcript_csv(sid: str) -> str:
    if _read_transcript(sid) is None:
        raise FileNotFoundError("这一场没有语音转写（transcript.json 不存在）")
    buf = io.StringIO()
    w = csv.writer(buf)
    try:
        from session_meta import read_label
        subject = str((read_label(sid) or {}).get("name") or "")
    except Exception:
        subject = ""
    w.writerow(["index", "subject", "question", "answer", "n_chars"])
    for idx, q, a in _qa_rows(sid):
        w.writerow([idx + 1, subject, q, a, len(a)])
    return buf.getvalue()


# ── 图表（matplotlib → PNG）──────────────────────────────────────────────

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    families = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei", "SimHei"):
        if cand in families:
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


_COLORS = ["#2a78d6", "#eb6834", "#2e9e5b", "#8a5a10"]


def _panel(ax, title: str, xlabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color="#333333")
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)          # 网格画在柱子/折线**后面**,不切碎图形
    ax.tick_params(labelsize=8, colors="#666666")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color="#666666")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _pick_voice_csv(sid: str) -> Path | None:
    """语音画图优先选**有数据行**的那份（面试/科研两个文件可能同时存在）。"""
    for key in ("voice_research", "voice_interview"):
        p = _log_path(sid, key)
        if p and p.is_file() and _rows(p) > 0:
            return p
    for key in ("voice_research", "voice_interview"):
        p = _log_path(sid, key)
        if p and p.is_file():
            return p
    return None


def _time_axis(df: pd.DataFrame) -> pd.Series:
    """时间轴（秒，从 0 开始）。时间列缺失就退回采样序号 —— 不假装有时间。"""
    t = _num(df, "timestamp")
    if not t.notna().any():
        return pd.Series(range(len(df)), dtype=float)
    t = t.ffill().bfill()
    return t - t.min()


def _no_data(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            color="#999999", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _donut(ax, valid: int, total: int,
           title: str, ok: str = "有数据", miss: str = "未检出") -> None:
    """覆盖率环形图 —— 比"一条线画到一半断掉"直观：先让人知道**数据够不够看**。"""
    valid = max(int(valid), 0)
    total = max(int(total), valid)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color="#333333")
    if total == 0:
        _no_data(ax, "没有样本")
        return
    gap = total - valid
    ax.pie([valid, gap] if gap else [valid, 0],
           colors=["#2a78d6", "#e6e6e6"], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.34, edgecolor="white"))
    ax.text(0, 0.04, f"{valid / total * 100:.0f}%", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#222222")
    ax.text(0, -0.16, "有数据", ha="center", va="center", fontsize=8, color="#666666")
    ax.legend([f"{ok} {valid} 帧", f"{miss} {gap} 帧"], loc="lower center",
              bbox_to_anchor=(0.5, -0.16), fontsize=8, frameon=False, ncol=2)
    ax.set_aspect("equal")


def _hbar(ax, labels: list[str], values: list[float], title: str,
          fmt: str = "{:.2f}") -> None:
    """横条 + 数值标签 —— 一眼看出"哪一项高、高多少"，不用去读坐标轴。"""
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color="#333333")
    if not values:
        _no_data(ax, "没有可统计的数据")
        return
    y = list(range(len(labels)))
    ax.barh(y, values, color=_COLORS[:len(labels)], height=0.55)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color="#e6e6e6", linewidth=0.7)
    for i, v in enumerate(values):
        ax.text(v, i, f" {fmt.format(v)}", va="center", fontsize=8.5, color="#333333")
    top = max(values) if values else 0
    ax.set_xlim(0, top * 1.3 if top > 0 else 1)
    ax.tick_params(labelsize=8, colors="#666666")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _bars(ax, values: list[float], title: str,
          fmt: str = "{:.2f}", color: str = "#2a78d6",
          xlabel: str = "第几题") -> None:
    """竖向柱状图 + 数值标签（语音按题看最直观）。"""
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color="#333333")
    xs = list(range(1, len(values) + 1))
    finite = [v for v in values if pd.notna(v)]
    ax.bar(xs, [v if pd.notna(v) else 0 for v in values], color=color, width=0.62)
    ax.set_xticks(xs)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    for x, v in zip(xs, values):
        if pd.notna(v):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=8,
                    color="#333333")
    top = max(finite) if finite else 0
    ax.set_ylim(0, top * 1.22 if top > 0 else 1)
    ax.tick_params(labelsize=8, colors="#666666")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5, color="#666666")


def _binned(t: pd.Series, series: dict[str, pd.Series],
            bin_sec: float = 10.0) -> dict[str, tuple[list[float], list[float]]]:
    """按 `bin_sec` 分箱取均值。**只在箱内有数据时出点** —— 不插值、不补零。"""
    idx = (t // bin_sec).astype("Int64")
    out: dict[str, tuple[list[float], list[float]]] = {}
    for label, s in series.items():
        if not s.notna().any():
            continue
        m = s.groupby(idx).mean().dropna()
        if len(m):
            out[label] = ([float(i) * bin_sec for i in m.index], [float(v) for v in m.values])
    return out


_AU_LABELS = {
    1: "AU1 抬眉", 2: "AU2 眉头上扬", 4: "AU4 皱眉", 6: "AU6 脸颊上提",
    7: "AU7 眼睑收紧", 9: "AU9 鼻皱", 10: "AU10 上唇抬", 12: "AU12 微笑",
    14: "AU14 酒窝", 15: "AU15 嘴角下压", 20: "AU20 唇拉伸", 23: "AU23 唇收紧",
    25: "AU25 张口", 26: "AU26 下颌下落",
}

#: 面部"其他指标"（表格式展示）。单位不同的量**不并进同一张条形图** —— 混着比长短
#: 是误导；表格里每行带指标名与自己的小数位，读起来才准确。
_FACE_OTHER = [
    ("gaze_direction_x", "视线 X（归一化）", 3),
    ("gaze_direction_y", "视线 Y（归一化）", 3),
    ("gaze_deviation", "视线偏离度", 3),
    ("head_yaw", "头部偏航", 3),
    ("head_pitch", "头部俯仰", 3),
    ("avg_ear", "平均眼开度 EAR", 3),
    ("blink_rate_per_min", "眨眼率（次/分）", 2),
    ("eye_closed_sec", "闭眼时长（秒）", 2),
    ("symmetry_score", "面部对称度", 3),
    ("focus_score", "专注度", 2),
]

#: 手势的稳定性指标（抖动类，单位一致，可同表比较）。
_GESTURE_JITTER = [
    ("left_hand_jitter", "左手抖动", 3),
    ("right_hand_jitter", "右手抖动", 3),
    ("left_wrist_jitter", "左腕抖动", 3),
    ("right_wrist_jitter", "右腕抖动", 3),
    ("left_elbow_jitter", "左肘抖动", 3),
    ("right_elbow_jitter", "右肘抖动", 3),
    ("left_shoulder_jitter", "左肩抖动", 3),
    ("right_shoulder_jitter", "右肩抖动", 3),
    ("head_jitter", "头部抖动", 3),
    ("torso_jitter", "躯干抖动", 3),
]

#: 关节角度（度）——与手指角度分开画：一个是"手指弯曲"，一个是"肢体姿态"。
_GESTURE_JOINTS = [
    ("left_arm_angle", "左臂"),
    ("right_arm_angle", "右臂"),
    ("left_elbow_angle", "左肘"),
    ("right_elbow_angle", "右肘"),
    ("head_tilt_angle", "头倾斜"),
    ("head_pitch_angle", "头俯仰"),
]

_FINGERS = [("thumb", "拇指"), ("index", "食指"), ("middle", "中指"),
            ("ring", "无名指"), ("pinky", "小指")]


def _table(ax, rows: list[tuple[str, float, int]], title: str,
           headers: tuple[str, str] = ("指标", "均值")) -> None:
    """指标表格。**比条形图更适合单位不同的量**：每行一个指标 + 自己的小数位。"""
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color="#333333")
    ax.axis("off")
    if not rows:
        _no_data(ax, "没有可统计的指标")
        return
    cells = [[name, f"{value:.{nd}f}"] for name, value, nd in rows]
    tbl = ax.table(cellText=cells, colLabels=list(headers), loc="upper center",
                   cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.12)
    tbl.auto_set_column_width([0, 1])
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#e6e6e6")
        if r == 0:
            cell.set_text_props(weight="bold", color="#333333")


def _au_means(df: pd.DataFrame) -> list[tuple[str, float]]:
    """**所有**有数据的 AU 列 → [(标签, 均值)]，按 AU 编号排序。

    2026-10-26 使用者问「有效 AU 只有三个吗」——不是:三份日志里 AU 列有 14 个，
    老图只挑了 3 个画。这里按列名动态取，不写死。
    """
    import re as _re
    found: list[tuple[int, str, float]] = []
    for col in df.columns:
        m = _re.match(r"^au(\d+)_", col)
        if not m:
            continue
        y = pd.to_numeric(df[col], errors="coerce")
        if y.notna().any():
            n = int(m.group(1))
            found.append((n, f"{_AU_LABELS.get(n, f'AU{n}')}", float(y.mean())))
    found.sort(key=lambda x: x[0])
    return [(label, value) for _, label, value in found]


def _grouped_bars(ax, groups: list[str], left: list[float], right: list[float],
                  title: str, fmt: str = "{:.0f}", unit: str = "°") -> None:
    """左右对照的分组柱状图（手指/关节角度用它，一眼看出两侧差多少）。"""
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color="#333333")
    x = list(range(len(groups)))
    ax.bar([i - 0.19 for i in x], left, width=0.36, label="左手", color=_COLORS[0])
    ax.bar([i + 0.19 for i in x], right, width=0.36, label="右手", color=_COLORS[1])
    ax.set_xticks(x, groups, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
    finite = [v for v in list(left) + list(right) if pd.notna(v)]
    if finite:
        top = max(finite)
        ax.set_ylim(0, top * 1.2 if top > 0 else 1)
    for i, v in zip(x, left):
        if pd.notna(v):
            ax.text(i - 0.19, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=7.5, color="#333333")
    for i, v in zip(x, right):
        if pd.notna(v):
            ax.text(i + 0.19, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=7.5, color="#333333")
    ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper right")
    ax.tick_params(labelsize=8, colors="#666666")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _chart_face(sid: str) -> bytes:
    p = _log_path(sid, "face")
    if not p or not p.is_file():
        raise NoData("这一场没有面部数据表")
    df = pd.read_csv(p, low_memory=False)
    t = _time_axis(df)

    series = {label: _num(df, col) for col, label in
              (("au12_smile", "微笑 AU12"), ("au4_frown", "皱眉 AU4"),
               ("au1_inner_brow_raise", "抬眉 AU1"))}
    got = {k: v for k, v in series.items() if v.notna().any()}
    total = len(df)
    au_cols = [c for c in df.columns if c.startswith("au")]
    valid = int(pd.concat([_num(df, c) for c in au_cols], axis=1)
                .notna().any(axis=1).sum()) if au_cols else 0
    au_rows = _au_means(df)
    other_rows = []
    for col, label, nd in _FACE_OTHER:
        y = _num(df, col)
        if y.notna().any():
            other_rows.append((label, float(y.mean()), nd))

    plt = _plt()
    fig = plt.figure(figsize=(10, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 2.0, 1.0],
                          width_ratios=[1.0, 1.35])
    _donut(fig.add_subplot(gs[0, 0]), valid, total, "AU 帧覆盖率",
           ok="有效", miss="未检出")
    _table(fig.add_subplot(gs[0, 1]), other_rows, "其他指标平均值（非 AU）")

    ax_au = fig.add_subplot(gs[1, :])
    if au_rows:
        labels = [n for n, _ in au_rows]
        values = [v for _, v in au_rows]
        # 14 根条用同一种颜色：这是同一类量（AU 强度），不靠颜色分类。
        ax_au.set_title("各面部动作单元（AU）平均强度 —— 共 "
                        f"{len(au_rows)} 项", loc="left", fontsize=10.5,
                        fontweight="bold", color="#333333")
        y = list(range(len(labels)))
        ax_au.barh(y, values, color="#2a78d6", height=0.62)
        ax_au.set_yticks(y, labels, fontsize=8.5)
        ax_au.invert_yaxis()
        ax_au.set_axisbelow(True)
        ax_au.grid(True, axis="x", color="#e6e6e6", linewidth=0.7)
        for i, v in enumerate(values):
            ax_au.text(v, i, f" {v:.3f}", va="center", fontsize=8, color="#333333")
        ax_au.set_xlim(0, max(values) * 1.28 if values else 1)
        ax_au.tick_params(labelsize=8, colors="#666666")
        for side in ("top", "right"):
            ax_au.spines[side].set_visible(False)
    else:
        _no_data(ax_au, "本场没有 AU 数据")

    ax_t = fig.add_subplot(gs[2, :])
    binned = _binned(t, got)
    if binned:
        for i, (label, (xs, ys)) in enumerate(binned.items()):
            ax_t.plot(xs, ys, label=label, linewidth=1.6, marker="o", markersize=3,
                      color=_COLORS[i % 4])
        ax_t.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper right")
        _panel(ax_t, "每 10 秒平均强度（趋势）", xlabel="时间（秒）")
    else:
        _no_data(ax_t, "没有可画的面部时间序列")
        ax_t.set_title("每 10 秒平均强度（趋势）", loc="left",
                       fontsize=10.5, fontweight="bold", color="#333333")

    fig.suptitle(f"{sid} · 面部表情　AU 列 {len(au_cols)} 项 · 有效帧 {valid} / {total}",
                 fontsize=12.5, fontweight="bold", color="#222222")
    return _save(fig, plt)


def _chart_gesture(sid: str) -> bytes:
    p = _log_path(sid, "gesture")
    if not p or not p.is_file():
        raise NoData("这一场没有手势数据表")
    df = pd.read_csv(p, low_memory=False)

    jitter_rows = []
    for col, label, nd in _GESTURE_JITTER:
        y = _num(df, col)
        if y.notna().any():
            jitter_rows.append((label, float(y.mean()), nd))

    total = len(df)
    frame_has_hand = pd.Series(False, index=df.index)
    for col in ("left_hand_jitter", "right_hand_jitter",
                "left_hand_spread", "right_hand_spread",
                "left_thumb_angle", "right_thumb_angle"):
        frame_has_hand |= _num(df, col).notna()
    valid = int(frame_has_hand.sum())

    plt = _plt()
    fig = plt.figure(figsize=(10, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.55, 1.35],
                          width_ratios=[1.0, 1.35])
    _donut(fig.add_subplot(gs[0, 0]), valid, total, "手部检出帧覆盖率",
           ok="检出", miss="未检出")
    _table(fig.add_subplot(gs[0, 1]), jitter_rows, "稳定性指标平均值（抖动，越低越稳）")

    ax_f = fig.add_subplot(gs[1, :])
    left_vals, right_vals = [], []
    for key, _name in _FINGERS:
        lv = _num(df, f"left_{key}_angle")
        rv = _num(df, f"right_{key}_angle")
        left_vals.append(float(lv.mean()) if lv.notna().any() else float("nan"))
        right_vals.append(float(rv.mean()) if rv.notna().any() else float("nan"))
    if any(pd.notna(v) for v in left_vals + right_vals):
        _grouped_bars(ax_f, [n for _k, n in _FINGERS], left_vals, right_vals,
                      "手指弯曲角度平均值（左手 / 右手）")
    else:
        _no_data(ax_f, "本场没有手指角度数据")
        ax_f.set_title("手指弯曲角度平均值", loc="left", fontsize=10.5,
                       fontweight="bold", color="#333333")

    ax_j = fig.add_subplot(gs[2, :])
    joint_labels, joint_left, joint_right = [], [], []
    for col, label in _GESTURE_JOINTS:
        y = _num(df, col)
        if not y.notna().any():
            continue
        joint_labels.append(label)
        joint_left.append(float(y.mean()))
        joint_right.append(float("nan"))
    # 关节角度同一根柱子只有一个值（不是左右成对）—— 这里用单系列柱，
    # 左右臂/左肘右肘各占一格，别硬凑成"左右对照"。
    if joint_labels:
        ax_j.set_title("身体姿态角度平均值（度）", loc="left", fontsize=10.5,
                       fontweight="bold", color="#333333")
        xs = list(range(len(joint_labels)))
        vals = joint_left
        # 同一类量（角度）用同一种颜色 —— 不靠颜色再分一次类。
        ax_j.bar(xs, vals, color="#2a78d6", width=0.6)
        ax_j.set_xticks(xs, joint_labels, fontsize=8.5)
        ax_j.set_axisbelow(True)
        ax_j.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
        for i, v in enumerate(vals):
            ax_j.text(i, v, f"{v:.0f}°", ha="center", va="bottom", fontsize=8,
                      color="#333333")
        ax_j.set_ylim(0, max(vals) * 1.2 if vals else 1)
        ax_j.tick_params(labelsize=8, colors="#666666")
        for side in ("top", "right"):
            ax_j.spines[side].set_visible(False)
    else:
        _no_data(ax_j, "本场没有关节角度数据")
        ax_j.set_title("身体姿态角度平均值", loc="left", fontsize=10.5,
                       fontweight="bold", color="#333333")

    fig.suptitle(f"{sid} · 手势与体态　CSV 列 {len(df.columns)} 项 · 检出 {valid} / {total} 帧",
                 fontsize=12.5, fontweight="bold", color="#222222")
    return _save(fig, plt)


def _chart_voice(sid: str) -> bytes:
    p = _pick_voice_csv(sid)
    if p is None:
        raise NoData("这一场没有语音数据表")
    df = pd.read_csv(p, low_memory=False)
    if df.empty:
        raise NoData("这一场的语音数据表里没有数据行")
    n = len(df)
    durations = _num(df, "duration_sec")
    total_sec = float(durations.sum()) if durations.notna().any() else 0.0

    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), constrained_layout=True)
    panels = [
        (axes[0][0], "pitch_mean", "基频（Hz）", "{:.1f}", _COLORS[0]),
        (axes[0][1], "energy_mean", "能量均值", "{:.3f}", _COLORS[1]),
        (axes[0][2], "chars_per_sec", "语速（字/秒）", "{:.1f}", _COLORS[2]),
        (axes[1][0], "duration_sec", "每段回答时长（秒）", "{:.1f}", _COLORS[3]),
        (axes[1][1], "pitch_variation", "音高变化（Hz）", "{:.1f}", _COLORS[0]),
        (axes[1][2], "pause_frequency", "停顿频率（次/分）", "{:.1f}", _COLORS[1]),
    ]
    for ax, col, title, fmt, color in panels:
        y = _num(df, col)
        if y.notna().any():
            _bars(ax, [float(v) if pd.notna(v) else float("nan") for v in y],
                  title, fmt=fmt, color=color)
        else:
            _no_data(ax, "没有这一列数据")
            ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold",
                         color="#333333")

    fig.suptitle(f"{sid} · 语音　{n} 段回答 · 总时长 {total_sec:.0f} 秒"
                 f"（{p.name}）",
                 fontsize=12.5, fontweight="bold", color="#222222")
    return _save(fig, plt)


def _save(fig, plt) -> bytes:
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="white")
    finally:
        plt.close(fig)
    return buf.getvalue()


_CHARTS = {"face": _chart_face, "gesture": _chart_gesture, "voice": _chart_voice}




def chart_png(sid: str, module: str) -> bytes:
    fn = _CHARTS.get(module)
    if fn is None:
        raise KeyError(f"未知模块 {module!r}（只允许 {', '.join(CHART_MODULES)}）")
    with _MPL_LOCK:          # matplotlib 不是线程安全的；面板是多线程的
        return fn(sid)


# ── 路由 ───────────────────────────────────────────────────────────────────

def _sid_or_400(sid: str) -> str:
    return media_retention.validate_session_id(sid)


@bp.get("/api/export/sessions")
def export_sessions():
    """会话列表 + 个人信息 —— 导出页要按人对号，不能只有一串会话号。

    直接复用素材浏览页的扫描（`_iter_sessions` / `_record`）：目录名、标注、录像
    与帧数只在这一处算，免得两页对同一场给出两个数。
    """
    import recordings_browser as rb

    rows = []
    for sid, d in rb._iter_sessions():
        try:
            rec = rb._record(sid, d, admin=True, report_counts={})
        except Exception:
            continue
        label = rec.get("label") or {}
        rows.append({
            "sid": sid,
            "name": label.get("name"),
            "student_id": label.get("student_id"),
            "department": label.get("department"),
            "serial": label.get("serial"),
            "modified": rec["modified"],
            "has_video": rec["has_video"],
            "video_bytes": rec["video_bytes"],
            "frames": rec["frames"],
        })
    rows.sort(key=lambda r: r["modified"], reverse=True)
    return jsonify({"sessions": rows})


@bp.get("/api/export/<sid>/files")
def export_files(sid: str):
    try:
        return jsonify(list_payload(_sid_or_400(sid)))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@bp.get("/api/export/<sid>/csv/<key>")
def export_csv(sid: str, key: str):
    try:
        path, download_name = csv_file(_sid_or_400(sid), key)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    return send_file(path, as_attachment=True, download_name=download_name,
                     mimetype="text/csv")


@bp.get("/api/export/<sid>/transcript")
def export_transcript(sid: str):
    fmt = (request.args.get("format") or "txt").strip().lower()
    try:
        sid = _sid_or_400(sid)
        if fmt == "json":
            return Response(json.dumps(transcript_payload(sid), ensure_ascii=False,
                                       indent=2),
                            mimetype="application/json",
                            headers={"Content-Disposition":
                                     f'attachment; filename="{sid}-transcript.json"'})
        if fmt == "csv":
            return Response(transcript_csv(sid), mimetype="text/csv",
                            headers={"Content-Disposition":
                                     f'attachment; filename="{sid}-qa.csv"'})
        return Response(transcript_txt(sid), mimetype="text/plain; charset=utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{sid}-transcript.txt"'})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404


@bp.get("/api/export/<sid>/chart")
def export_chart(sid: str):
    module = (request.args.get("module") or "").strip().lower()
    try:
        sid = _sid_or_400(sid)
        png = chart_png(sid, module)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except NoData as exc:
        return jsonify({"error": str(exc)}), 404
    return Response(png, mimetype="image/png",
                    headers={"Cache-Control": "no-store"})
