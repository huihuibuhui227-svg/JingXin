#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单场 5 分钟一键演示 —— 从"盘上这一场"走到"能投屏的报告 HTML"。

    python demo_run.py --session 20260927_212143_95a8
    python demo_run.py --session ~/JingXin/recordings/20260927_212143_95a8 --open
    python demo_run.py --session 20261026-1026-王国梁-...__20261026_002930_cb79 --open

## 为什么需要它

答辩现场只有 5 分钟:要证明的不是"这套代码能跑",而是
**同一份素材 → L0 列 → L1 指标 → 报告文件** 这条链一次走完、且数字可核对。
此前这条链分散在四个入口(`report_frontend.l1_session` 是库、
`python -m report_frontend.report_generator` 会**无条件弹浏览器**、
`/api/run/report` 要起 flask、`tools/jx_*` 是批处理),
现场没有一个能"一条命令、说清每一步、出错不崩栈"的入口。

## 它不做什么

* **不新造口径**:L1 用 `build_l1_report`(与 `/api/report/structured`、
  `report_generator` **同一个引擎**),数字全部从载荷里读,不在这里重算;
* **不改素材**:只写 `data/output/` 下的报告与 `data/l0_series/` 的派生缓存
  (`ensure_series` 自己的边界写在它的 docstring 里);
* **不假装成功**:任一阶段失败都打印**具体原因**,并返回非 0 —— 退出码见下。

## 退出码

| 码 | 含义 |
|---|---|
| 0 | 报告已落盘(即使 `--open` 时浏览器没弹起来,也只降级为警告) |
| 2 | ① 依赖/模型不满足 —— 后面的阶段做不了 |
| 3 | 会话定位失败(sid 不合法,或录制目录里没有这一场) |
| 4 | L1 计算失败(报告与数字都依赖它) |
| 5 | 报告生成/写盘失败(但前面算出来的数字已经打印了) |

`--verbose` 打完整 traceback(默认只给一句人话 + 异常类型)。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

#: 仓库根(本文件所在目录)。所有相对路径都从这里出发 —— 从别的 cwd 调也不跑偏。
REPO = Path(__file__).resolve().parent

EXIT_OK = 0
EXIT_DEPS = 2
EXIT_SESSION = 3
EXIT_L1 = 4
EXIT_REPORT = 5

#: 硬依赖:没有它们连一行 L1 都算不出来。
REQUIRED_MODULES: list[tuple[str, str]] = [
    ("pandas", "读日志 CSV、算 L1 指标"),
    ("numpy", "L0 序列缓存 / 指标计算"),
]

#: 软依赖:只在特定分支上用到,缺了本次演示照样能跑。
OPTIONAL_MODULES: list[tuple[str, str]] = [
    ("mediapipe", "仅当 data/l0_series/<sid>.npz 缺失、需要从录音重算序列时用到"),
    ("flask", "只有起 Web 服务时才需要;本脚本不用"),
]

#: 模型文件。三个模态的提取都要它,但**报告路径不重新提取特征** ——
#: 日志已在盘上,所以它们缺失只降级为警告(且只在需要重算序列时才致命)。
MODEL_FILES: list[tuple[str, str]] = [
    ("models/mediapipe/face_landmarker.task", "面部 478 点"),
    ("models/mediapipe/hand_landmarker.task", "手部 21 点"),
    ("models/mediapipe/pose_landmarker_lite.task", "体态(轻)"),
]

W = 74  # 分隔线宽度


def hr(title: str = "") -> None:
    if title:
        print(f"\n{'─' * 8} {title} {'─' * max(0, W - 10 - len(title))}")
    else:
        print("─" * W)


def say(msg: str = "") -> None:
    print(msg)


def warn(msg: str) -> None:
    print(f"   ⚠️  {msg}")


def bad(msg: str) -> None:
    print(f"   ❌ {msg}")


def good(msg: str) -> None:
    print(f"   ✅ {msg}")


class Stage:
    """一个阶段的计时器 + 失败时的统一出口(不崩栈)。"""

    def __init__(self, no: str, title: str, verbose: bool = False) -> None:
        self.no, self.title, self.verbose = no, title, verbose
        self.t0 = 0.0

    def __enter__(self) -> "Stage":
        hr(f"{self.no} {self.title}")
        self.t0 = time.time()
        return self

    def __exit__(self, et, ev, tb) -> bool:
        # 只兜 Exception:KeyboardInterrupt / SystemExit 必须原样往上走,
        # 否则现场按 Ctrl-C 会被这里吃掉。
        if et is not None and issubclass(et, Exception):
            bad(f"{self.title} 阶段异常:{et.__name__}: {ev}")
            if self.verbose:
                traceback.print_exception(et, ev, tb)
            else:
                say("      (加 --verbose 看完整 traceback;本阶段后的步骤按下面的报错决定是否继续)")
            self.error = ev  # type: ignore[attr-defined]
            return True
        print(f"   ⏱  {time.time() - self.t0:.2f}s")
        return False

    error: Any = None


# ── ① 依赖与模型 ───────────────────────────────────────────────────────────

def stage_deps(out_dir: Path | None = None) -> tuple[bool, list[str], list[str]]:
    """返回 (能不能继续, 挡路的硬问题, 只降级为警告的软问题)。

    两者**分开**:硬问题是"现在算不下去"(缺 pandas),软问题是"这次用不上但要知道"
    (缺 mediapipe 且序列缓存已在盘上)。混在一起打印会让人分不清哪个才是红的那条。
    """
    problems: list[str] = []
    notes: list[str] = []

    py = sys.version_info
    say(f"   Python {py.major}.{py.minor}.{py.micro}  ({sys.executable})")
    if (py.major, py.minor) < (3, 9):
        problems.append(f"Python 版本过低({py.major}.{py.minor});本仓要求 ≥3.9")

    say("   依赖:")
    for mod, why in REQUIRED_MODULES:
        ok = importlib.util.find_spec(mod) is not None
        print(f"     {'✅' if ok else '❌'} {mod:<12} {why}")
        if not ok:
            problems.append(f"缺少硬依赖 {mod}({why})⟹ 装法:pip install -r requirements.txt")

    for mod, why in OPTIONAL_MODULES:
        ok = importlib.util.find_spec(mod) is not None
        print(f"     {'✅' if ok else '⚠️ '} {mod:<12} {why}")
        if not ok:
            notes.append(f"{mod} 未安装 —— {why}")

    say("   模型:")
    for rel, what in MODEL_FILES:
        p = REPO / rel
        size = f"{p.stat().st_size / 1e6:.1f} MB" if p.is_file() else "—"
        print(f"     {'✅' if p.is_file() else '⚠️ '} {rel:<45} {what} {size}")
        if not p.is_file():
            notes.append(f"模型缺失:{rel}({what})")

    # 目录:日志要能读,输出要能写 —— 这两件事失败原因不同,分开说。
    # 探的是**这次真正要写的那个**目录(可能是 --out-dir 指的别处),不是默认那个。
    logs = REPO / "data" / "logs"
    out = out_dir or (REPO / "data" / "output")
    print(f"   日志目录 {logs} → {'存在' if logs.is_dir() else '不存在'}")
    if not logs.is_dir():
        problems.append(f"日志目录不存在:{logs}(L1 只能读到 0 行)")
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".demo_run_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        print(f"   输出目录 {out} → 可写")
    except OSError as exc:
        problems.append(f"输出目录不可写:{out}({exc})")

    return (not problems), problems, notes


# ── ① 会话定位 ─────────────────────────────────────────────────────────────

def resolve_session(arg: str) -> tuple[str, Path | None]:
    """`<sid>` 或 `<录制目录>` → (sid, 目录 or None)。不合法就抛 `ValueError`。"""
    import media_retention
    from session_purge import SID_PAT

    raw = arg.strip().rstrip(os.sep)
    cand_dir: Path | None = None

    p = Path(raw).expanduser()
    if p.is_dir():
        cand_dir = p
        # 目录名两种形态:`<sid>`(旧)与 `<序号>-<姓名>-…__<sid>`(2026-09-27 起)。
        name = p.name
        raw = name.split("__")[-1] if "__" in name else name

    if not SID_PAT.fullmatch(raw):
        raise ValueError(
            f"不是合法的 session_id:{raw!r}(形如 20260927_212143_95a8;"
            f"也可以直接给录制目录)")

    sid = raw
    d = cand_dir if cand_dir is not None else media_retention.resolve_recording_dir(sid)
    d = d if d is not None and Path(d).is_dir() else None
    return sid, d


def stage_session(sid: str, d: Path | None) -> None:
    say(f"   会话 id:{sid}")
    if d is None:
        try:
            import media_retention
            root = media_retention.root()
        except Exception:
            root = "录制根"
        warn(f"录制目录里没有这一场({root} 下按 `__{sid}` 后缀找不到)")
        warn("   ⟹ 只影响'素材/转写'那两个数据源;日志在盘上的部分照常算。")
        return
    say(f"   录制目录:{d}")
    media = d / "media"
    if media.is_dir():
        vids = sorted(x.name for x in media.glob("*") if x.is_file())
        say(f"   素材:{', '.join(vids + ['preview/']) if vids else 'media/ 下没有文件'}")
    else:
        warn("目录里没有 media/ —— 录像/预览不可用,但日志与转写照常")
    tr = d / "transcript.json"
    say(f"   转写:{'有 transcript.json' if tr.is_file() else '无 transcript.json(T 族 3 条会不可出)'}")


# ── ② L0 ───────────────────────────────────────────────────────────────────

def stage_l0(sid: str, logs: dict[str, Any]) -> None:
    """L0 = 列清单落位 + 逐帧序列缓存。**只陈述盘上事实**,不补数(所以只打印)。"""
    import l0_columns

    doc = l0_columns.load()
    cols = l0_columns.columns()
    errs = l0_columns.schema_errors(doc)
    from collections import Counter

    by_mod = Counter(c["modality"] for c in cols)
    by_status = Counter(c["status"] for c in cols)
    say(f"   L0 列清单:{len(cols)} 列  (schema 错误 {len(errs)} 条)")
    say(f"     按模态 {dict(by_mod)}")
    say(f"     按状态 {dict(by_status)}")
    if errs:
        for e in errs[:5]:
            bad(f"列清单 schema:{e}")
        warn(f"列清单有 {len(errs)} 条不合规 —— 不影响本场出数,但要写进材料风险项")

    # 本场日志里,表上登记过的列真到了几条 —— 这是"L0 落位"最直接的证据。
    hit_total = declared_total = 0
    for mod, df in logs.items():
        key = {"voice_interview": "voice", "voice_research": "voice"}.get(mod, mod)
        declared = {c["column"] for c in cols if c["modality"] == key}
        if not declared:
            continue
        have = {str(c) for c in df.columns}
        hit = declared & have
        declared_total += len(declared)
        hit_total += len(hit)
        say(f"     日志[{mod}] {len(df)} 行 × {len(df.columns)} 列 → "
            f"表上登记 {len(declared)} 列命中 {len(hit)}")
        miss = sorted(declared - have)
        if miss:
            say(f"       未出现:{', '.join(miss[:6])}{' …' if len(miss) > 6 else ''}")
    if declared_total:
        say(f"   L0 落位合计:{hit_total}/{declared_total} 列在日志里出现")
    else:
        say("   本场一条日志都没有 ⟹ 没有可核对的列(报告会照实报缺口)")

    # 逐帧序列缓存(V 族 8 条只吃它)
    try:
        from report_frontend import voice_series
        cache = voice_series.series_dir() / f"{sid}.npz"
        if cache.is_file():
            series = voice_series.load_series(sid)
            n = len(series or [])
            say(f"   L0 序列缓存:{cache.name} 存在({cache.stat().st_size / 1024:.0f} KB,"
                f"{n} 段)")
        else:
            warn(f"L0 序列缓存缺失({cache})⟹ L1 阶段会拿 media/camera.webm 现提一次"
                 f"(约 30 秒);录音不在或提不出来时,V 族 8 条照实报缺")
    except Exception as exc:  # 序列缓存是派生物,读不出来不该拖垮演示
        warn(f"读 L0 序列缓存失败:{exc.__class__.__name__}: {exc}")


# ── ② L1 ───────────────────────────────────────────────────────────────────

def stage_load(sid: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """按 id 装配日志。返回 (实际描述的会话, 日志 dict, 模态来源三态)。

    三态(`loaded` / `unreadable` / `missing`)**由加载器给** —— 报告头据此披露
    "哪几个模态没进来",不在这里重推。
    """
    from report_frontend.data_loader import LogDataLoader

    loader = LogDataLoader()
    logs = loader.get_fused_latest_data(sid) or {}
    say(f"   日志装配:{ {k: f'{len(v)} 行' for k, v in logs.items()} or '（一条都没读到）'}")
    say(f"   实际描述的会话:{loader.target_session}(报告头按它写,不冒充请求的 id)")
    if loader.target_session != sid:
        warn(f"加载器最终锁定的是 {loader.target_session},不是请求的 {sid}")
    return loader.target_session, logs, dict(loader.selected_sessions or {})


def stage_l1(target: str, logs: dict[str, Any]) -> dict[str, Any]:
    """L1 载荷 —— 与 `/api/report/structured`、`report_generator` 同一个引擎。"""
    from report_frontend.l1_report import build_l1_report

    return build_l1_report(target, logs=logs)


# ── ③ 报告 ─────────────────────────────────────────────────────────────────

def stage_report(sid: str, logs: dict[str, Any], payload: dict[str, Any],
                 sources: dict[str, Any], out_dir: Path | None) -> Path | None:
    """把载荷渲染成交付版 HTML,落到 data/output/(与面板产出的命名规则一致)。"""
    import datetime as _dt

    from report_frontend.report_generator import (
        ReportGenerator, _subject_slug, _unique_report_path,
    )

    gen = ReportGenerator(output_dir=str(out_dir or (REPO / "data" / "output")))
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"Research_Assessment_Report_{ts}{_subject_slug(sid)}"
    path = Path(_unique_report_path(gen.output_dir, stem, sid))
    say(f"   渲染引擎:{payload.get('engine')}(与 /api/report/structured 同一份)")
    # ⚠️ 这里不走 `ReportGenerator.generate_report()`:它末尾**无条件** `webbrowser.open`
    #    (面板要的就是那个行为),而本脚本的浏览器开关由 `--open` 决定。
    #    也顺带避开它内部再算一遍 L1 —— 载荷已经在手上了。
    html = gen._build_html_report(payload, {}, {}, logs,
                                  gen._scan_static_images(),
                                  sources=sources, target=sid)
    path.write_text(html, encoding="utf-8")
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"写盘后文件不存在或为空:{path}")
    good(f"报告:{path}  ({path.stat().st_size / 1024:.0f} KB)")
    return path


# ── ④ 关键数字 ─────────────────────────────────────────────────────────────

def stage_numbers(payload: dict[str, Any]) -> None:
    l1 = payload.get("l1") or {}
    cov = payload.get("coverage") or {}
    by_conf = l1.get("by_confidence") or {}
    by_gap = l1.get("by_gap") or {}

    n_total = l1.get("n_total")
    n_meas = l1.get("n_measured")
    say(f"   指标条数      {n_total} 条(会话级 L1,口径:所有回答窗内帧的并集)")
    say(f"   出值条数      {n_meas} 条"
        f"{f'，未出值 {n_total - n_meas} 条' if isinstance(n_total, int) and isinstance(n_meas, int) else ''}")
    say(f"   置信度分布    高 {by_conf.get('高', 0)} · 中 {by_conf.get('中', 0)} · "
        f"低 {by_conf.get('低', 0)} · 无 {by_conf.get('无', 0)}")
    nz = {k: v for k, v in by_gap.items() if v}
    say(f"   缺口分布      {nz or '（无缺口）'}")
    # ★ 2026-11-21:这一行是答辩现场投屏的那行,原来印「过证据门 8/14 槽」。
    #   14 里有 6 个槽永远填不上(3 个这套采集不测 + 3 个整维没有对应指标),
    #   8/14 会被读成"只覆盖了 57%",而可测量的那 8 个槽是 8/8。
    #   数字与措辞都由载荷给(`coverage.summary_line`)—— 不在这里另拼一句,
    #   否则命令行与报告 HTML 迟早说的不是同一件事。
    line = cov.get("summary_line")
    if line:
        say(f"   覆盖          {line}")
        # 通过率那一栏:分母是**可测量槽**(`n_measurable`),为 0 时说"不适用" ——
        # 印「0/0」会被读成一个数。
        n_meas = cov.get("n_measurable")
        rate = (f"通过率 {cov.get('n_passed')}/{n_meas}(分母只含第一段)"
                if isinstance(n_meas, int) and n_meas else
                "通过率不适用(可测量 0 槽)")
        say(f"                 {rate},置信度上限 {cov.get('confidence_cap')}")
    else:
        # 旧载荷没有这一句:照它自己带的数印,不在这里补算(补算=第二份口径)。
        say(f"   覆盖          这份载荷没带分段覆盖数,只有通过槽数 "
            f"{cov.get('n_passed')}/{cov.get('n_slots')} 槽,"
            f"置信度上限 {cov.get('confidence_cap')}")

    msum = l1.get("modality_summary") or {}
    if msum:
        say("   逐模态产出:")
        for mod, s in msum.items():
            say(f"     {mod:<8} {s.get('n_measured')}/{s.get('n_total')} 条"
                + (f"  ({s.get('source_note')})" if s.get("source_note") else ""))

    covd = l1.get("modality_coverage") or {}
    if covd:
        say("   逐模态日志可用率:")
        for mod, c in covd.items():
            ratio = c.get("usable_ratio")
            ratio_txt = f"{float(ratio) * 100:.1f}%" if isinstance(ratio, (int, float)) else "—"
            say(f"     {mod:<8} {c.get('rows')} 行,可用 {c.get('usable_rows')} 行({ratio_txt})")

    notes = l1.get("gaps") or []
    if notes:
        say("   本场自报的时间轴/数据缺口:")
        for g in notes:
            say(f"     · {g}")


# ── main ───────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="demo_run.py",
        description="单场 5 分钟一键演示:依赖检查 → L0/L1 → 报告 → 关键数字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python demo_run.py --session 20260927_212143_95a8\n"
               "  python demo_run.py --session ~/JingXin/recordings/20260927_212143_95a8 --open\n")
    ap.add_argument("--session", required=True, metavar="<session_id_or_dir>",
                    help="会话 id(20260927_212143_95a8)或该场的录制目录")
    ap.add_argument("--open", action="store_true", help="生成后用 webbrowser 打开报告")
    ap.add_argument("--out-dir", default=None, metavar="DIR",
                    help="报告落点,默认 <仓库>/data/output")
    ap.add_argument("--verbose", action="store_true", help="失败时打完整 traceback")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else None
    t_all = time.time()
    say("=" * W)
    say("  JingXin 单场一键演示 · 依赖 → L0 → L1 → 报告")
    say(f"  会话:{args.session}")
    say(f"  仓库:{REPO}")
    say("=" * W)

    # ① 依赖
    blocked = False
    with Stage("①", "依赖与模型可用性", args.verbose):
        ok, problems, notes = stage_deps(out_dir)
        for m in problems:
            bad(m)
        for m in notes:
            warn(m)
        if ok:
            good("依赖与模型齐全" if not notes else "硬依赖齐全(上面几条只影响特定分支)")
        else:
            bad("依赖不满足 ⟹ 后面所有阶段都会因此失败,不硬跑")
        blocked = not ok
    if blocked:
        say(f"\n结论:未开始计算,退出码 {EXIT_DEPS}(装依赖后重跑同一命令即可)")
        return EXIT_DEPS

    # ① 会话定位(同属"准备",失败就没法往下走)
    sid = ""
    sdir: Path | None = None
    with Stage("①", "会话定位", args.verbose):
        try:
            sid, sdir = resolve_session(args.session)
            stage_session(sid, sdir)
            good("会话已定位")
        except Exception as exc:
            bad(f"会话定位失败:{exc}")
            sid = ""
    if not sid:
        say(f"\n结论:没有任何可算的会话,退出码 {EXIT_SESSION}")
        return EXIT_SESSION

    # ② 日志装配 → L0 → L1(任务书里的"计算 L0/L1")
    target, logs, sources = sid, {}, {}
    payload: dict[str, Any] | None = None
    with Stage("②", "装配日志(按 session_id 取,不跨场拼接)", args.verbose):
        try:
            target, logs, sources = stage_load(sid)
            if not logs:
                warn("一条日志都没读到 —— L1 仍会跑(38 条全部报缺),报告照样出,只是没有数")
        except Exception as exc:
            bad(f"日志装配失败:{exc}")
            if not args.verbose:
                say("     常见原因:data/logs 不存在 / CSV 被占用 / pandas 版本不符")
    # L0 在 L1 之前:它查的是"日志里的列有没有按 L0 表落位",L1 算的是值。
    # 空日志是合法情形(报告只讲缺口)—— 那时 L0 仍会把列清单自检打完。
    hr("② L0(列清单落位 + 逐帧序列缓存)")
    try:
        stage_l0(sid, logs)
    except Exception as exc:
        warn(f"L0 检查失败(不影响 L1):{exc.__class__.__name__}: {exc}")

    hr("② L1(会话级 38 条指标)")
    t_l1 = time.time()
    try:
        payload = stage_l1(target, logs)
        good(f"L1 算完:{len(payload.get('l1', {}).get('indicators') or [])} 条  "
             f"({time.time() - t_l1:.2f}s)")
    except Exception as exc:
        bad(f"L1 计算失败:{exc.__class__.__name__}: {exc}")
        if args.verbose:
            traceback.print_exc()
        else:
            say("     常见原因:日志 CSV 列名变了 / 时间轴建不起来 / pandas 版本不符"
                "(加 --verbose 看 traceback)")
    if payload is None:
        say(f"\n结论:算不出指标,报告也就无从生成,退出码 {EXIT_L1}")
        return EXIT_L1

    # ③ 报告
    report_path: Path | None = None
    with Stage("③", f"生成报告({out_dir or (REPO / 'data' / 'output')})",
               args.verbose):
        try:
            report_path = stage_report(sid, logs, payload, sources, out_dir)
        except Exception as exc:
            bad(f"报告生成失败:{exc}")
            if not args.verbose:
                say("     L1 数字已经在上一步算出来了,可以先用它们讲;报告故障见上方原因")
    if report_path is None:
        stage_numbers(payload)
        say(f"\n结论:指标可用、报告不可用,退出码 {EXIT_REPORT}")
        return EXIT_REPORT

    with Stage("④", "关键数字", args.verbose):
        try:
            stage_numbers(payload)
        except Exception as exc:
            warn(f"打印关键数字时出错:{exc}(报告已生成,不受影响)")

    # ⑤ 打开
    if args.open:
        with Stage("⑤", "打开报告", args.verbose):
            try:
                import webbrowser
                url = "file://" + os.path.realpath(report_path)
                opened = webbrowser.open(url)
                if opened:
                    good(f"已交给浏览器:{url}")
                else:
                    warn(f"webbrowser 说没打开(无图形界面时常见)。手动打开:{report_path}")
            except Exception as exc:
                warn(f"打开浏览器失败:{exc.__class__.__name__}: {exc}。手动打开:{report_path}")
    else:
        hr("⑤ 打开报告")
        say(f"   未加 --open ⟹ 不弹浏览器(报告已在盘上):{report_path}")
        say(f"   手动看:file://{os.path.realpath(report_path)}")

    say("\n" + "=" * W)
    good(f"演示完成:总耗时 {time.time() - t_all:.1f}s")
    say(f"   报告:{report_path}")
    say("=" * W)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
