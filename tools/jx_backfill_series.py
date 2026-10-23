"""把**缺失的语音帧序列缓存**补出来(2026-10-24,`not_extracted` 的下游)。

## 为什么要补

`data/l0_series/<sid>.npz` 是 `voice_series` 的派生缓存(整场录音逐帧 f0/rms)。
V 族 12 条里有 **8 条只吃这个序列**,而正式场 24/24 的语音日志是 0 行 ——
也就是说正式语料的 V 族**全靠这份缓存**。

2026-10-24 实测(服务器,`JINGXIN_RECORDINGS_DIR=/home/zgy/JingXin/recordings`):

```
44 个录制目录 · 37 场有 media/camera.webm · 只有 7 场有序列缓存
⟹ 31 场「素材在盘上、序列没生成」
```

这 31 场此前在报告里全被标成 `modality_missing`,前端提示「补救:**查采集**」——
而采集没坏。补一次提取(实测单场 ~34 秒)V 族就能从 0/12 变成 10/12。

## 它做什么

对每一场:调 `voice_series.extract_session(sid)` —— **就是报告读侧缺的那一步**,
不另写一份提取逻辑。有逐题音频就逐题,否则退回整场 `camera.webm`
(优先级在 `voice_series.audio_source` 里,本工具不复制)。

## 它**不**做什么(边界)

* **不重提取已有缓存**。`load_series` 读得出就跳过 —— 缓存是派生品,
  重建一次要几十秒,而"有没有"才是本工具要回答的问题。要重建用 `--force`。
* **不碰没有音频的场次**。`audio_source` 返回 `none` 的场次直接跳过并如实计数 ——
  那些场次是真的 `modality_missing`,报告里该保留那个码。
* **不改任何日志 / CSV**。只写 `data/l0_series/` 与一份清单。
* **不在报告路径上跑**。`l1_voice._source_on_disk` 只问"素材在不在",
  提取永远是这个离线工具的事 —— 报告是读侧,不许有副作用。

## 溯源

每场的结果记进 `data/logs/.series_backfill_manifest.json`:哪一场、音频来源
(`answer` 逐题 / `whole` 整场)、几段、帧数、耗时、以及**源文件的指纹**
(`路径 + mtime + 字节数`)。指纹让 `--pending` 只处理"源变过"的场次 ——
新录完一场跑一次几秒到几分钟,不必全量重扫。

## 用法

    PY=~/anaconda3/envs/jingxin/bin/python
    $PY tools/jx_backfill_series.py --list            # 只列出哪几场该补
    $PY tools/jx_backfill_series.py --dry-run         # 同上,但连"能不能取到音频"也探一遍
    $PY tools/jx_backfill_series.py --all             # 补全部缺缓存的场次
    $PY tools/jx_backfill_series.py --session <sid>   # 补一场(建议先试跑)
    $PY tools/jx_backfill_series.py --pending         # 例行回填:只处理源变过的

⚠️ 录制根由 `JINGXIN_RECORDINGS_DIR` 决定(与 `media_retention.root()` 同一个开关)。
本工具把 `--sessions-dir` 与它**绑成一处**,免得"扫的目录"与"读素材的目录"不是同一个
—— `jx_backfill_voice.py` 的 docstring 记着那次踩坑的后果。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LOG_DIR = REPO / "data" / "logs"
MANIFEST = LOG_DIR / ".series_backfill_manifest.json"


def session_id_of(sess_dir: Path) -> str:
    """目录名 → **真 session_id**。

    目录名可能带标注前缀(`20260929_195403-2-候选甲-000000000-投资学__20260929_195403_040b`),
    真 sid 是 `__` 之后那段。优先读 `label.json`(它写着权威值),再退回按 `__` 切。

    判据与 `tools/jx_check_session.py::_session_id` **同一套** —— 两边算错的方式也一样:
    拿目录名当 sid 会拼出 `data/l0_series/<带人名的长名>.npz`,缓存写进去了但报告
    按真 sid 找不到它,于是"补过了却还是 not_extracted"。
    """
    lp = sess_dir / "label.json"
    if lp.is_file():
        try:
            sid = json.loads(lp.read_text(encoding="utf-8")).get("session_id")
            if sid:
                return str(sid)
        except Exception:  # noqa: BLE001 - 标注坏了不该让整场消失,退回目录名规则
            pass
    return sess_dir.name.split("__")[-1] if "__" in sess_dir.name else sess_dir.name


def source_sig(session_id: str) -> list | None:
    """本场音频的**指纹** —— 用来判断"自上次补过之后源有没有变"。

    记 `[kind, [每个源文件的名字, mtime, 字节数]]`。只 stat,不读内容:
    31 场全量提取要十几分钟,而例行回填只该问"新录的/还在追加的有没有"。
    """
    from report_frontend import voice_series

    kind, items = voice_series.audio_source(session_id)
    if kind == "none":
        return None
    sig: list = [kind, []]
    for index, path in items:
        try:
            st = Path(path).stat()
        except OSError:
            sig[1].append([index, str(path), None, None])
            continue
        sig[1].append([index, str(path), int(st.st_mtime), int(st.st_size)])
    return sig


def pending_reason(session_id: str, manifest: dict) -> str | None:
    """该不该补这一场。返回 `None` = 不用补,否则返回**理由**(人话)。

    四种理由,判据分开写清楚 —— 混成一句"需要补"就没法统计到底在补什么:

    * `no_audio`    —— 盘上没有可用音频(真的 `modality_missing`,报告该保留那个码);
    * `missing`     —— 有音频、没有缓存(本工具的主要目标);
    * `unrecorded`  —— 有缓存,但清单里没有它的记录 ⟹ 不知道这份缓存是怎么来的,
                       保守地重建一次(与 `jx_backfill_face.py` 的 `--pending` 同一取舍);
    * `stale`       —— 有缓存、清单有记录,但源文件变过(整场录音被重新上传/覆盖)。

    ⚠️ 后两种都要**真的重算**(`force=True`):`extract_session(force=False)` 见到缓存
    就直接返回它,不 force 的话"补好了"这句话里一个字节都没变。
    """
    from report_frontend import voice_series

    kind, items = voice_series.audio_source(session_id)
    if kind == "none" or not items:
        return "no_audio"
    cached = voice_series.load_series(session_id)
    if cached is None:
        return "missing"
    prev = (manifest.get(session_id) or {}).get("source_sig")
    if prev is None:
        return "unrecorded"
    if prev != source_sig(session_id):
        return "stale"
    return None


def backfill_one(session_id: str, *, dry_run: bool = False,
                 force: bool = False) -> dict:
    """补一场。返回结果字典(写进清单)。**异常不外抛** —— 由调用方记成 failed。

    ⚠️ `force` 必须由调用方按 `pending_reason` 的结论传:判成 `stale`(源变过)时
    要传 `True`。`extract_session(force=False)` 见到缓存就**直接返回它**,
    于是"源变了、判成该补、跑完报 backfilled"这句里一个字节都没重算 ——
    这正是本仓反复栽的那种「验证跑错了对象」。调用方见 `main`。
    """
    from report_frontend import voice_series

    t0 = time.time()
    kind, items = voice_series.audio_source(session_id)
    out = {"session": session_id, "audio_kind": kind, "n_sources": len(items)}
    if kind == "none" or not items:
        out.update(status="skip", why="盘上没有可用音频(报告里该保留 modality_missing)")
        return out
    if dry_run:
        out.update(status="dry-run",
                   why=f"{kind}:{len(items)} 段 → 会写 {voice_series.series_dir()}/"
                       f"{session_id}.npz")
        return out
    # ★ 守卫:**有缓存、又不 force ⟹ 不许报 backfilled**。
    #
    # 2026-10-24 实测踩到:`unrecorded` 那 12 场跑完耗时 **0.0 秒** —— 因为
    # `extract_session(force=False)` 见到缓存就直接返回它,一个字节都没重算,
    # 而工具照打「✓ backfilled」。这正是本仓反复栽的那种
    # 「验证跑错了对象」:跑的那条路径没执行到要证的那段代码。
    # 所以这里不靠调用方传对参数,而是**当场判**:没重算就如实说没重算。
    if voice_series.load_series(session_id) is not None and not force:
        # 指纹照样记:这份缓存"我们看过、源没变",下次 `--pending` 就不必再问一遍。
        out.update(status="reused",
                   source_sig=source_sig(session_id),
                   why="已有缓存且未 force ⟹ extract_session 会直接返回旧缓存,"
                       "这一场**没有重算**(计数不许把它算成 backfilled)")
        return out
    series = voice_series.extract_session(session_id, force=force)
    usable = [s for s in series if getattr(s, "f0", None) is not None and s.f0.size > 0]
    if not usable:
        # 提取跑完了但一帧都没出来 ⟹ 素材本身解不出音频。**不写缓存**
        # (`save_series` 会拒绝空序列),如实记 failed —— 报告那边会继续报
        # `not_extracted`,至少不会变成一个"看起来有序列"的空缓存。
        out.update(status="failed", why="提取跑完但没有任何可用帧(音频解不出来?)")
        return out
    cache = voice_series.series_dir() / f"{session_id}.npz"
    out.update(status="backfilled",
               n_series=len(series),
               frames=int(sum(len(s.f0) for s in series)),
               voiced_sec=round(float(sum(s.voiced_sec for s in series)), 1),
               # 记相对路径:清单会进 git,绝对路径在另一台机器上无意义。
               cache=(str(cache.relative_to(REPO)) if cache.is_relative_to(REPO)
                      else str(cache)),
               source_sig=source_sig(session_id),
               seconds=round(time.time() - t0, 1))
    return out


def _sessions(root: Path) -> list[Path]:
    """录制根下的会话目录。跳过 `_` 开头的(`_回收站` / `_已作废`)。"""
    return sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))


def main() -> int:
    import media_retention

    ap = argparse.ArgumentParser(description="补齐缺失的语音帧序列缓存(data/l0_series)")
    ap.add_argument("--sessions-dir", default=None,
                    help="录制根。默认 = media_retention.root()(即 JINGXIN_RECORDINGS_DIR);"
                         "显式给了就写进环境变量,保证扫的目录与读素材的目录是同一个")
    ap.add_argument("--session", default=None, help="只补这一场(sid)")
    ap.add_argument("--all", action="store_true", help="补全部缺缓存的场次")
    ap.add_argument("--pending", action="store_true",
                    help="例行回填:只处理源指纹与清单不一致的场次")
    ap.add_argument("--list", action="store_true", help="只列出哪些场次该补")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 场(试跑用)")
    ap.add_argument("--dry-run", action="store_true", help="只探测,不写缓存")
    ap.add_argument("--force", action="store_true", help="已有缓存也重建")
    args = ap.parse_args()

    if args.sessions_dir:
        os.environ["JINGXIN_RECORDINGS_DIR"] = args.sessions_dir
    root = media_retention.root()
    if not root.is_dir():
        print(f"✗ 录制根不存在:{root}", file=sys.stderr)
        return 2

    manifest = (json.loads(MANIFEST.read_text(encoding="utf-8"))
                if MANIFEST.is_file() else {})

    dirs = _sessions(root)
    if args.session:
        key = args.session.rstrip("/")
        dirs = [d for d in dirs if session_id_of(d) == key or d.name.endswith(key)]
        if not dirs:
            print(f"✗ 找不到场次:{key}", file=sys.stderr)
            return 2

    # 逐场定"要不要补"。`--list` 与 `--dry-run` 走同一段判据 ——
    # 两份实现会分叉,而"列出来的与真跑的不是同一批"正是最难发现的那种错。
    todo: list[tuple[Path, str, str]] = []      # (目录, sid, 理由)
    n_no_audio = 0
    for d in dirs:
        sid = session_id_of(d)
        reason = "forced" if args.force else pending_reason(sid, manifest)
        if reason == "no_audio":
            n_no_audio += 1
            continue
        if reason is None and not args.force:
            continue
        todo.append((d, sid, reason or "forced"))

    print(f"录制根 {root}")
    print(f"扫到 {len(dirs)} 个会话目录;没有可用音频 {n_no_audio} 场(跳过,报告该保留 modality_missing)")
    print(f"待补 {len(todo)} 场")
    if not todo:
        print("没有需要处理的场次。")
        return 0

    if args.list:
        # ⚠️ **只打 sid,不打目录名**。正式场的目录名是 `-序号-姓名-学号-学院__<sid>`,
        #    把整行打到终端/日志里就是把这些人的姓名与学号搬到另一个地方去 ——
        #    而这一列对使用者没有任何用处(sid 唯一定位一场,缓存文件名用的也是它)。
        #    两个公开仓都踩过这件事,不再重复。
        for _d, sid, reason in todo:
            print(f"  {reason:10s} {sid}")
        return 0

    if args.limit:
        todo = todo[: args.limit]
    print()

    results: list[dict] = []
    for i, (_d, sid, reason) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {sid}  ({reason})", flush=True)
        t0 = time.time()
        try:
            # ★ 判成 `stale` / `unrecorded` 时**必须** `force=True` —— 否则
            #   `extract_session` 见到缓存就直接返回它,跑完报"补好了"而实际
            #   一个字节都没重算。2026-10-24 实测:漏掉 `unrecorded` 这一项,
            #   12 场全部 0.0 秒"补成功"。
            r: dict = backfill_one(sid, dry_run=args.dry_run,
                                   force=args.force
                                   or reason in ("stale", "unrecorded", "forced"))
        except Exception as e:  # noqa: BLE001 - 一场炸掉不许带走其余 30 场
            r = {"session": sid, "status": "failed",
                 "why": f"{type(e).__name__}: {e}"}
        r.setdefault("reason", reason)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        tag = {"backfilled": "✓", "reused": "·", "skip": "·", "dry-run": "?",
               "failed": "✗"}[r["status"]]
        detail = (f"{r.get('n_series')} 段 / {r.get('frames')} 帧 / "
                  f"浊音 {r.get('voiced_sec')} 秒" if r.get("n_series")
                  else (r.get("why") or ""))
        print(f"    {tag} {r['status']}: {detail}  ({r['seconds']}s)")

    if not args.dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        for r in results:
            # `reused` 也记 —— 它带着源指纹,正是下次 `--pending` 要比对的东西。
            # 不记的话这些场次每轮都会被判成 `unrecorded` ⟹ 每次例行回填都重算一遍。
            if r["status"] in ("backfilled", "reused", "failed"):
                manifest[r["session"]] = r
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                       sort_keys=True), encoding="utf-8")
        print(f"\n清单已写入 {MANIFEST}")

    n_ok = sum(1 for r in results if r["status"] == "backfilled")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'─'*60}\n补成功 {n_ok}   失败 {n_fail}   其余跳过 {len(results) - n_ok - n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
