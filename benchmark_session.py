#!/usr/bin/env python3
"""端到端性能基准 —— 单场会话的分阶段耗时与峰值内存。

## 它回答哪个问题

说明书第二章要两个硬指标:**单场端到端耗时** 与 **峰值内存**。本仓此前**没有任何**
这两项的实测数字(各模块 docstring 里散落着单点观测:序列提取 34 秒、缓存命中
"直接返回"、`/api/report/structured` 无耗时记录),因此本章只能空着或写"待测"。

本脚本把**产品真实走的那条路**逐段计时并落成 JSON:

    LogDataLoader.get_fused_latest_data        读盘 + 归一化        → l0_load_ms
    PsychologicalFeatureEngine.extract_all_features   L0 特征向量    → l0_features_ms
    l1_session.compute_session_l1              L1 指标(会话级)      → l1_ms
    l1_report.build_l1_report                  L1 → 报告载荷        → report_payload_ms
    ReportGenerator._build_html_report         HTML 渲染 + 落盘     → report_html_ms

`app.py:285-322`(`/api/report/structured`)走的是「读盘 → l1_report」;
`report_generator.generate_report`(HTML 交付物)走的是「读盘 → l1_report → HTML」。
两段都在这里,所以本脚本覆盖 **API 路径与交付物路径**。

## ⚠️ 本脚本**不**测量的东西(别把数字读大)

* **不跑 ASR**(`tools/jx_transcribe_recordings.py` 的推理):那是 26 场批量转写,
  单场要看音频长度,与"报告端到端"不是同一件事。转写元信息见另一份补丁。
* **不跑 Face/Gesture/Voice 推理服务**:本脚本读它们**已经落盘的 CSV**。
  真"端到端"若从摄像头算起,要加上这三条推理链 —— 见最终答复"其他可测点"。
* **不生成逐题音频的 f0/rms 序列**:`voice_series.ensure_series` 命中缓存时直接返回,
  未命中会**现提一次(单场约 34 秒)**。这一项会以 `series_cache_hit` 字段如实标注,
  否则同一场两次跑会差 34 秒而读者以为是抖动。

## 只读仓库下怎么跑

仓库对我只读。三条措施,跑完会自检并写进 `repo_writes` 字段(应当为空):

1. `sys.dont_write_bytecode = True` —— 不往仓库里写 `__pycache__`;
2. `logging_config.LOG_DIR` 被重定向到 workspace,且**从不调用** `setup_logging()`
   (那是 `RotatingFileHandler` 指向 `data/logs/jingxin.log` 的唯一入口,只读时会抛
   `Unable to configure handler 'file'`);
3. `ReportGenerator(output_dir=…)` 指到 workspace —— HTML 落在本目录,不落 `data/output`。

`data/logs` / `data/transcripts` / `data/l0_series` / `data/output` 的 mtime 快照在
跑前跑后各取一次,差异进 `repo_writes`。

## 用法

    PY=~/anaconda3/envs/jingxin/bin/python
    $PY benchmark_session.py --session 20260927_212143_95a8 --repeat 2
    $PY benchmark_session.py --sessions a,b,c --repeat 2 --out verify/benchmark_output.json
    $PY benchmark_session.py --session <sid> --repeat 1 --isolate   # 每轮独立进程,拿冷启动峰值 RSS

## 计时与内存**分开测**(2026-11-21 实测踩到)

`tracemalloc` 开着的时候,分配密集的阶段会被显著放大 —— 同一场 `l0_load_ms`
实测 **167 ms → 552 ms(3.3 倍)**。所以本脚本默认:

* **计时轮**:不开 `tracemalloc`;
* **内存轮**:每个会话额外跑一轮、只开 `tracemalloc` 取 `peak_mb_tracemalloc`,
  该轮的耗时**不**计入统计(它的数放在 `memory_pass` 里,并注明不可当性能数字)。

要复现"开着 trace 计时"的老口径加 `--trace-inline`。

## 内存口径(两个数,别混用)

* `peak_mb_tracemalloc` —— `tracemalloc` 峰值。**只统计 Python 分配器**;numpy/pandas
  的大块走 `malloc`,**不计入** ⟹ 这个数系统性偏低,用于看"Python 对象层"的开销。
* `peak_rss_mb` —— `resource.getrusage(RUSAGE_SELF).ru_maxrss`(Linux 单位 KB)。
  含 C 扩展的 native 内存,**这才是"峰值内存"该引的数**。但它是**进程生命周期内的
  单调最大值** ⟹ `--repeat N` 时第 2 轮起必然 ≥ 第 1 轮,不是独立样本。
  要独立样本用 `--isolate`(每轮 fork 一个新进程)。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import resource
import subprocess
import sys
import time
import tracemalloc
from contextlib import redirect_stdout
from pathlib import Path

# 只读仓库:第一行就把字节码写盘关掉(必须在 import 仓库模块之前)。
sys.dont_write_bytecode = True

DEFAULT_REPO = Path("/home/zgy/JingXin/jingxin")
#: 每轮之间判"仓库有没有被写"的目录(相对仓库根)。
WATCHED = ("data/logs", "data/transcripts", "data/l0_series", "data/output")


# ─────────────────────────── 基础设施 ───────────────────────────

def _rss_mb() -> float:
    """本进程当前常驻内存(MB)。非峰值,给"跑完有没有涨"做参考。"""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024.0, 2)
    except OSError:
        pass
    return 0.0


def _peak_rss_mb() -> float:
    """进程生命周期峰值 RSS(MB)。Linux 上 `ru_maxrss` 是 KB。"""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)


def _snapshot(roots: list[Path]) -> dict[str, tuple[int, int]]:
    """目录 → (文件数, 最大 mtime)。只取两元组,不记录内容(不读大文件)。"""
    snap: dict[str, tuple[int, int]] = {}
    for r in roots:
        if not r.exists():
            snap[str(r)] = (-1, -1)
            continue
        n, newest = 0, 0
        try:
            for p in r.rglob("*"):
                if p.is_file():
                    n += 1
                    newest = max(newest, int(p.stat().st_mtime_ns))
        except OSError:
            newest = -1
        snap[str(r)] = (n, newest)
    return snap


def _diff(before: dict, after: dict) -> list[str]:
    return [f"{k} {_fmt(v)} → {_fmt(after.get(k, ('?', '?')))}"
            for k, v in before.items() if after.get(k) != v]


def _fmt(t) -> str:
    return f"(n={t[0]}, mtime={t[1]})" if t else "(?)"


def _redirect_side_effects(repo: Path, workdir: Path) -> dict:
    """把「会往只读仓库写」的两处副作用改道,返回改道记录(进 JSON)。"""
    info: dict[str, str] = {}

    # ① 日志:LOG_DIR 指向 workspace。
    #    `logging_config` 在模块顶层就 `LOG_DIR.mkdir(parents=True, exist_ok=True)` ——
    #    只读盘上它是**存在即免写**,所以 import 本身是安全的;真正会炸的是
    #    `setup_logging()` 装 `RotatingFileHandler`。本脚本不调用它,这里只兜住
    #    "被某条 import 链间接调用"的情况。
    log_dir = workdir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        import logging_config  # noqa: PLC0415 - 必须等 sys.path 生效后

        logging_config.LOG_DIR = log_dir
        info["logging_config.LOG_DIR"] = str(log_dir)
    except Exception as e:  # noqa: BLE001 - 改道失败不是致命错误,但要写出来
        info["logging_config"] = f"未导入({type(e).__name__}: {e})"
    info["setup_logging_called"] = "否(本脚本从不调用)"

    # ② 系列缓存:默认读仓库里已生成的 `data/l0_series`(只读),不改道;
    #    若调用方要用别的缓存目录,靠 `JINGXIN_SERIES_DIR`(模块自己的开关)。
    if os.environ.get("JINGXIN_SERIES_DIR"):
        info["JINGXIN_SERIES_DIR"] = os.environ["JINGXIN_SERIES_DIR"]
    return info


# ─────────────────────────── 一轮测量 ───────────────────────────

def _one_round(repo: Path, session_id: str, html_dir: Path,
               keep_html: bool, verbose: bool, trace: bool = False) -> dict:
    """跑一遍完整链路,返回这一轮的记录。指标/报告复用同一次 L1 计算。

    `trace=False`(**计时用**):不开 `tracemalloc`。开着它计时会把分配密集的阶段
    显著放大 —— 实测同一场 `l0_load_ms` 从 167 ms 涨到 552 ms(3.3 倍),因为
    `_none_bucket_rows()` 每轮都要用 pandas 读一遍 NONE 桶的 832 行。
    `trace=True`(**专门的内存那一轮**):只为拿 `peak_mb_tracemalloc`。
    """
    import report_frontend.l1_report as l1_report
    from report_frontend.data_loader import LogDataLoader
    from report_frontend.feature_engine import PsychologicalFeatureEngine
    from report_frontend.l1_session import compute_session_l1
    from report_frontend.report_generator import ReportGenerator

    stages: dict[str, float] = {}
    buf = io.StringIO()
    sink = sys.stdout if verbose else buf

    if trace:
        tracemalloc.start()
        tracemalloc.reset_peak()
    rss_before = _rss_mb()
    t_all0 = time.perf_counter()

    with redirect_stdout(sink):
        # ── L0-a:读盘 + 归一化(`app.py:293` 的第一段)───────────────
        t0 = time.perf_counter()
        loader = LogDataLoader()                       # log_dir 由模块自算 → 仓库 data/logs
        logs = loader.get_fused_latest_data(session_id)
        stages["l0_load_ms"] = (time.perf_counter() - t0) * 1000

        # ── L0-b:L0 特征向量(旧链 `feature_engine`,报告载荷已不用,留作对照)──
        t0 = time.perf_counter()
        features = PsychologicalFeatureEngine(logs).extract_all_features()
        stages["l0_features_ms"] = (time.perf_counter() - t0) * 1000

        # ── L1:会话级指标(唯一入口 `l1_session.compute_session_l1`)───
        t0 = time.perf_counter()
        result = compute_session_l1(session_id, logs=logs)
        stages["l1_ms"] = (time.perf_counter() - t0) * 1000

        # ── 报告载荷(`l1_report.build_l1_report`)────────────────────
        #
        # ⚠️ 它内部**自己**再调一次 `compute_session_l1`(l1_report.py:155)。
        #    直接计时就等于把 L1 算两遍,`report_payload_ms` 会虚高一整段 L1。
        #    ⟹ 把模块里的 `compute_session_l1` 临时换成"直接返回上面那份",
        #    这样计到的是**真的报告成形代码**(映射 + 质量报告 + 人话化),
        #    且没有重复计算。这是本脚本唯一一处 monkeypatch,跑完立刻还原。
        orig = l1_report.compute_session_l1
        l1_report.compute_session_l1 = lambda _sid, logs=None: result
        try:
            t0 = time.perf_counter()
            payload = l1_report.build_l1_report(session_id, logs=logs)
            stages["report_payload_ms"] = (time.perf_counter() - t0) * 1000
        finally:
            l1_report.compute_session_l1 = orig

        # ── 报告交付物:HTML 渲染 + 落盘(`report_generator.generate_report` 的后半)──
        #
        # 不走 `generate_report()`:它自己 new 一个 loader(再读一遍盘)、再把载荷算一遍,
        # 还会 `webbrowser.open`。这里只调它渲染那一步,输入与它逐字相同。
        gen = ReportGenerator(output_dir=str(html_dir))
        t0 = time.perf_counter()
        static_images = gen._scan_static_images()
        html = gen._build_html_report(payload, {}, {}, logs, static_images,
                                      sources=loader.selected_sessions,
                                      target=loader.target_session)
        out_html = html_dir / f"{session_id}.html"
        out_html.write_text(html, encoding="utf-8")
        stages["report_html_ms"] = (time.perf_counter() - t0) * 1000

    t_all = (time.perf_counter() - t_all0) * 1000
    peak = None
    if trace:
        peak, _cur = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    if not keep_html:
        try:
            out_html.unlink()
        except OSError:
            pass

    # 序列缓存命中与否 —— 未命中时 l1_ms 里含一次"现提"(单场约 34 秒),
    # 不标出来就会被读成抖动。
    try:
        from report_frontend import voice_series
        series_hit = voice_series.load_series(session_id) is not None
    except Exception:  # noqa: BLE001
        series_hit = None

    tail = "" if verbose else "\n".join(buf.getvalue().strip().splitlines()[-6:])
    return {
        "stage_times_ms": {k: round(v, 2) for k, v in stages.items()},
        "stage_sum_ms": round(sum(stages.values()), 2),
        # 产品**默认**走的那条路不含 `l0_features`(旧链的 14 槽特征向量,只在
        # `?engine=legacy` 时才跑)。把这两个数分开给,免得读者把"全链"当"交付物耗时"。
        "stage_sum_excl_legacy_ms": round(
            sum(v for k, v in stages.items() if k != "l0_features_ms"), 2),
        "total_ms": round(t_all, 2),
        "peak_mb_tracemalloc": None if peak is None else round(peak / 1024 / 1024, 2),
        "peak_rss_mb": _peak_rss_mb(),
        "vmrss_mb_before": rss_before,
        "vmrss_mb_after": _rss_mb(),
        "n_indicators": len(result.indicators),
        "n_with_value": result.n_measured,
        "n_modalities": len(logs),
        "modalities": sorted(logs),
        "n_payload_slots_passed": payload["coverage"]["n_passed"],
        # `extract_all_features()` 返回 {模态: {特征名: 值}} —— 规模按**叶子**数,
        # 不是顶层模态数(3),否则会被读成"L0 只出了 3 个特征"。
        "n_features_l0": sum(len(v) for v in features.values()),
        "n_features_l0_modalities": len(features),
        "html_bytes": len(html.encode("utf-8")),
        "series_cache_hit": series_hit,
        "engines": {"payload": payload.get("engine"),
                    "payload_version": payload.get("model_metadata", {}).get("version")},
        "loader_log_tail": tail.splitlines(),
    }


# ─────────────────────────── 隔离模式 ───────────────────────────

def _isolated(session_id: str, argv0: str, workdir: Path, extra: list[str]) -> dict:
    """把一个进程跑一轮 —— 拿**独立**的峰值 RSS(ru_maxrss 是单调的,同进程会串味)。"""
    cmd = [sys.executable, argv0, "--session", session_id, "--repeat", "1",
           "--warmup", "0", "--workdir", str(workdir), "--_child"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": f"child exit {r.returncode}", "stderr": r.stderr[-2000:]}
    try:
        got = json.loads(r.stdout)
    except ValueError:
        return {"error": "child stdout 不是 JSON", "stdout": r.stdout[-2000:]}
    rec = (got.get("records") or [{}])[0]
    # 冷启动基线:子进程在跑链路**之前**已经 import 了 pandas/numpy/html 模板,
    # 那部分 RSS 不该算到"这一场"头上,但也确实占着内存 —— 两个数都给。
    rec["cold_peak_rss_mb"] = rec.get("peak_rss_mb")
    rec["baseline_rss_mb"] = got.get("baseline_rss_mb")
    # 子进程里那一轮专用内存 pass 的峰值也带上来(冷进程口径)
    mp = (got.get("memory_pass") or {}).get(session_id) or {}
    if mp.get("peak_mb_tracemalloc") is not None:
        rec["peak_mb_tracemalloc"] = mp["peak_mb_tracemalloc"]
        rec["peak_mb_tracemalloc_source"] = "dedicated_traced_pass"
    return rec


# ─────────────────────────── CLI ───────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="单场会话端到端耗时 + 峰值内存基准",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--session", help="会话 id")
    ap.add_argument("--sessions", help="逗号分隔的多个会话 id(与 --session 二选一)")
    ap.add_argument("--repeat", type=int, default=2, help="测量轮数(默认 2)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="预热轮数(不计入结果;默认 1 —— 首轮的 import/模板缓存开销"
                         "不属于稳态成本)")
    ap.add_argument("--repo", default=str(os.environ.get("JINGXIN_REPO", DEFAULT_REPO)),
                    help=f"仓库根(默认 {DEFAULT_REPO})")
    ap.add_argument("--workdir", default=None,
                    help="副作用落点(日志/HTML)。默认 = 本脚本同级 ../.bench_work")
    ap.add_argument("--out", default=None, help="把结果 JSON 写到这个文件")
    ap.add_argument("--isolate", action="store_true",
                    help="每轮独立子进程(拿互不串味的峰值 RSS;慢一点)")
    ap.add_argument("--keep-html", action="store_true", help="保留生成的 HTML(默认删)")
    ap.add_argument("--trace-inline", action="store_true",
                    help="计时时也开 tracemalloc(会把分配密集阶段放大 2~3 倍,仅供对照)")
    ap.add_argument("--verbose", action="store_true", help="不吞被测链路自己的 print")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    if not (repo / "report_frontend" / "l1_session.py").is_file():
        print(f"✗ 不像是仓库根:{repo}", file=sys.stderr)
        return 2
    sids = [s.strip() for s in (args.sessions or "").split(",") if s.strip()]
    if args.session:
        sids.append(args.session)
    if not sids:
        print("✗ 需要 --session 或 --sessions", file=sys.stderr)
        return 2

    workdir = Path(args.workdir).expanduser() if args.workdir else \
        Path(__file__).resolve().parent.parent / ".bench_work"
    (workdir / "html").mkdir(parents=True, exist_ok=True)

    # sys.path 先就位,**然后**才 import 仓库模块(改道要赶在它们之前)。
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    side_effects = _redirect_side_effects(repo, workdir)

    # 基线 RSS:import 完大件、但还没跑任何一场。隔离模式靠它把"库占地"与"这一场占地"分开。
    import pandas  # noqa: F401,PLC0415 - 顺序有意:基线要在 import 之后取
    baseline_rss = _peak_rss_mb()

    watched = [repo / w for w in WATCHED]
    snap_before = _snapshot(watched)

    out: dict = {
        "tool": "benchmark_session",
        "repo": str(repo),
        "workdir": str(workdir),
        "baseline_rss_mb": baseline_rss,
        "side_effects": side_effects,
        "stages_doc": {
            "l0_load_ms": "LogDataLoader.get_fused_latest_data —— 读 CSV + 归一化",
            "l0_features_ms": "PsychologicalFeatureEngine.extract_all_features —— L0 特征向量(旧链,载荷已不用)",
            "l1_ms": "l1_session.compute_session_l1 —— L1 会话级指标",
            "report_payload_ms": "l1_report.build_l1_report —— L1→报告载荷(L1 结果复用,不重算)",
            "report_html_ms": "ReportGenerator._build_html_report —— HTML 渲染 + 落盘",
        },
        "records": [],
        #: 每个会话额外跑一轮"开 tracemalloc"的,只为拿 Python 层峰值。
        #: 它**慢**(见 `_one_round` 的说明),所以不参与计时统计,数字单独放这里。
        "memory_pass": {},
    }

    for sid in sids:
        if args.isolate:
            rec = _isolated(sid, str(Path(__file__).resolve()), workdir,
                            [] if not args.keep_html else ["--keep-html"])
            rec["session"] = sid
            out["records"].append(rec)
            continue
        for w in range(max(0, args.warmup)):
            _one_round(repo, sid, workdir / "html", args.keep_html, args.verbose)
        for i in range(1, max(1, args.repeat) + 1):
            rec = _one_round(repo, sid, workdir / "html", args.keep_html, args.verbose)
            rec["session"] = sid
            rec["repeat"] = i
            out["records"].append(rec)
            print(f"  [{sid}] 第 {i} 轮  total={rec['total_ms']:.0f} ms  "
                  f"L1 出值 {rec['n_with_value']}/{rec['n_indicators']}  "
                  f"RSS 峰值={rec['peak_rss_mb']:.1f} MB", file=sys.stderr)

        # 内存那一轮(不计时)。`peak_mb_tracemalloc` 每个会话只测一次:
        # 同进程同输入,跑两遍拿到的是同一个峰值,多测只是多花时间。
        if not args.trace_inline:
            mp = _one_round(repo, sid, workdir / "html", args.keep_html,
                            args.verbose, trace=True)
            out["memory_pass"][sid] = {
                "peak_mb_tracemalloc": mp["peak_mb_tracemalloc"],
                "total_ms_under_tracemalloc": mp["total_ms"],
                "stage_times_ms_under_tracemalloc": mp["stage_times_ms"],
                "note": "开着 tracemalloc 跑的一轮,只用来取 Python 层峰值;"
                        "它的耗时**不可**当作性能数字(分配密集阶段被放大)",
            }
            for rec in out["records"]:
                if rec["session"] == sid and rec.get("peak_mb_tracemalloc") is None:
                    rec["peak_mb_tracemalloc"] = mp["peak_mb_tracemalloc"]
                    rec["peak_mb_tracemalloc_source"] = "dedicated_traced_pass"

    snap_after = _snapshot(watched)
    out["repo_writes"] = _diff(snap_before, snap_after)
    out["repo_readonly_ok"] = not out["repo_writes"]

    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        p = Path(args.out).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"\n已写出 {p}({len(out['records'])} 条记录)", file=sys.stderr)
    if args._child or not args.out:
        print(text)
    if not out["repo_readonly_ok"]:
        print(f"\n⚠️ 仓库被写了: {out['repo_writes']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
