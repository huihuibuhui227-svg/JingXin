"""拿留存帧重跑面部提取，给"检出率 0%"的场次补数据。

## 为什么需要

44 场里有 **23 场面部检出率是 0.0%**（判据见 `tools/jx_check_session.py`）。
根因**不是**代码坏、也不是画面差，而是 mediapipe 正面人脸检测器**对"脸占画面的比例"
敏感** —— 受试者坐得远、脸占比小时就检不到。实测：未裁切 0/60，**中央裁切放大后 60/60**。

详见 `face_expression/pipeline/framing.py` 与 `docs/素材体检_2026-10-03.md` §2。

## 它怎么做（三条自我约束）

1. **只在全画幅失败时才裁。** 全画幅能用的场次**一个字节都不碰** ——
   实测裁切会让 190 列里的 142 列漂移（blendshape 对分辨率敏感、
   `*_trend` 是滚动窗口），对已正常的场次裁切是净损害。
2. **取景按场次定一次、整场固定。** 逐帧变取景会污染 `*_trend`，比任何单一取景都坏。
3. **判据只有 `emotion != 'no_face'`**（`framing.is_detected`）。
   用"特征字典非空"会把 `no_face` 算成检出 —— 我实测踩过，得出过假的 100%。

## 它**不**做什么

- **不改 CSV 契约**。产出仍是 `DataLogger` 写的 92 列同构文件；裁切模式不写进 CSV，
  记在 `data/logs/.face_backfill_manifest.json`（本仓要求每列在 `l0_columns.json`
  登记，而 `tests/test_face_blendshapes.py` 还钉着 `bs_` 必须 52 列，所以不加列）。
- **不重编码视频**。喂进去的就是留存的那份 JPEG 字节解出来的图。

## 用法

    PY=~/anaconda3/envs/jingxin/bin/python
    $PY tools/jx_backfill_face.py --list                # 只列出哪些场次该补
    $PY tools/jx_backfill_face.py --session <sid>       # 补一场（建议先试跑）
    $PY tools/jx_backfill_face.py --all                 # 补全部低检出场次
    $PY tools/jx_backfill_face.py --all --dry-run       # 只探测定取景，不写文件

原文件会改名成 `<原名>.predrop.bak`，**不删**。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RECORDINGS = Path(os.environ.get("JX_RECORDINGS") or (Path.home() / "JingXin" / "recordings"))
LOG_DIR = REPO / "data" / "logs"
MANIFEST = LOG_DIR / ".face_backfill_manifest.json"

#: ★ 只补"基本全空"的场次（检出率 ≤ 此值）。
#:
#: 为什么不补"半有半无"的（例如 61.7%）:裁切会让**已经正常的那部分**特征漂移
#: （实测 190 列里 142 列会变），那是**净损害**。而全空的场次**本来就没有数据**,
#: 任何值都好过没有。所以门槛卡在"几乎没有",不是卡在"不达标"。
BACKFILL_IF_RATE_AT_MOST = 0.05

#: 达到这个检出率就算正常，不用管（与 `tools/jx_check_session.py` 的 §8.4 一致）。
HEALTHY_AT_OR_ABOVE = 0.80

#: 补完之后至少要到这个检出率才算"补成了"。
#: ⚠️ **不能用 80%** —— 实测姬欣怡那场 `center50` 重放后是 **79%**（原来是 0.9%），
#: 那是从"什么都没有"到"基本可用"的巨大改善，卡在 80% 会把它当失败丢掉。
ACCEPT_MIN_RATE = 0.50

#: 且必须比原来至少高出这么多，才算"补有意义"（防止拿一个平庸结果换掉旧文件）。
ACCEPT_MIN_GAIN = 0.25

#: 算 CSV 检出率时要跳过的非特征列。
_CSV_SKIP = {"session_id", "timestamp", "timestamp_iso"}


def detection_rate(csv_path: Path) -> float | None:
    """CSV 的检出率。

    ⚠️ **这里的判据与 `framing.is_detected()` 不同，是有意的 —— 两个层级的"没检出"
    长得不一样：**

    - **pipeline 返回的特征字典**：没检出时给的是非空字典
      `{'emotion': 'no_face', 'timestamp': …}` ⟹ 必须判 `emotion != 'no_face'`
      （`framing.is_detected`）。
    - **落盘 CSV**：`emotion` **根本不是列**（logger 的 `fieldnames` 里是
      `dominant_emotion`，而 `emotion` 不在其中，不会被拷贝），没检出时那一行是
      **空串** ⟹ 判"除 id/时间戳外还有非空值"。

    实测踩过：我第一版在这函数里查 `emotion` 列，结果**每一场都算成 0%**
    （含本该 73% 的场次）—— 因为那列不存在，`r.get('emotion')` 恒为 `None`。
    判据要跟着**数据实际长什么样**走，不能想当然。
    """
    if not csv_path.is_file():
        return None
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0
    ok = 0
    for r in rows:
        vals = {k: (v or "").strip() for k, v in r.items() if k not in _CSV_SKIP}
        # 防御：现有 215 份 CSV 里 `no_face` **一次都没出现过**（"无脸"的表示法是整行留空），
        # 所以下面这一跳在当前素材上是 no-op。但哪天有人开始写 `no_face` 哨兵值，
        # "有非空值"这个判据就会把无脸行算成检出 ⟹ 检出率**静默虚高**。
        # 判据与 `face_expression.pipeline.framing.is_detected` 对齐：no_face 不算检出。
        if vals.get("dominant_emotion") == "no_face":
            continue
        if any(v not in ("", "None") for v in vals.values()):
            ok += 1
    return ok / len(rows)


def ledger_sig(sess_dir: Path) -> list | None:
    """台账的"指纹"（`mtime` + 字节数）—— 用来判断这场**自上次处理后有没有变**。

    例行回填（`--pending`）靠它把"全量重跑"降成"只处理新录的 / 还在追加的"：
    44 场全量重跑约半小时，每晚跑一次不现实；比指纹只 stat 两个数，几毫秒。
    台账一变（新录一场、或某场还在追加帧）指纹就变 ⟹ 会被重新处理。
    """
    led = sess_dir / "retention.face.jsonl"
    if not led.is_file():
        return None
    st = led.stat()
    return [int(st.st_mtime), int(st.st_size)]


def backup_path(out_csv: Path) -> Path:
    """给一份要顶掉的 CSV 选备份名 —— **首个备份永不覆盖**。

    此前是直接 `shutil.move(out_csv, out_csv + ".predrop.bak")`，无条件盖掉同名备份。
    而 `--uniform` 会对**已经补过一次**的场次再跑一遍，于是 `.predrop.bak` 里那份
    "回填前的原始 CSV"（检出 0.1% 的那份，是"0.1% → 93%"这个结论**唯一的物证**）
    会被回填后的结果顶掉。2026-10-03 复查时发现。

    现在往后找第一个空位：`.csv.predrop.bak` 永远是最初那份，再跑落 `.csv.predrop.1.bak`，
    再跑落 `.2.bak`…… 谁也不丢。
    """
    bak = out_csv.with_suffix(".csv.predrop.bak")
    if not bak.exists():
        return bak
    i = 1
    while True:
        alt = out_csv.with_suffix(f".csv.predrop.{i}.bak")
        if not alt.exists():
            return alt
        i += 1


def ledger_frames(sess_dir: Path) -> list[tuple[Path, int]]:
    """从 `retention.face.jsonl` 读 `(帧路径, declared_ts)`。

    ★ 必须用账本的 `declared_ts` 当时间戳：服务端把**同一个** `timestamp_ms`
    既喂给 `process_frame` 又写进账本（见 `face_expression/api/app.py`），
    所以只有它能让重放的时间基与当场一致。用墙钟会得出假结论（本仓纪律 #3）。
    """
    led = sess_dir / "retention.face.jsonl"
    if not led.is_file():
        return []
    out = []
    for line in led.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                      # 碎行跳过（账本可能被中断过）
        if rec.get("kind") != "frame":
            continue
        rel, ts = rec.get("file"), rec.get("declared_ts")
        if not rel or ts is None:
            continue
        p = sess_dir / rel
        if p.is_file():
            out.append((p, int(ts)))
    return out


def backfill_one(sess_dir: Path, *, dry_run: bool = False, force: bool = False,
                 uniform: bool = False) -> dict:
    """处理一场。返回结果字典（含 `status`）。

    `uniform=True` 时走「统一协议重跑」：**不论现在检出率多少都重提取**，
    好让全部场次由**同一套程序**产出（协议 = "每场探测定取景"，不是说都用同一个裁切框）。
    接受条件是「不比原来差」，不是「必须跨过某条线」。

    > 前提已验证：拿今天的代码重提取一场全画幅正常的场次，与当场活跑的 CSV
    > **逐格相等**（84 列 × 38 帧，0 处差异）⟹ 重跑是**保真**的，不丢当场记录的值。
    """
    import cv2

    from face_expression.pipeline.framing import apply_crop, is_detected, probe_crop, PROBE_FRAMES
    from face_expression.pipeline.video_pipeline import VideoPipeline
    from face_expression.utils.logger import DataLogger

    sid = sess_dir.name.split("__")[-1]
    out_csv = LOG_DIR / f"face_au_log_{sid}.csv"

    before = detection_rate(out_csv)
    if before is None:
        return {"session": sid, "status": "skip", "why": "没有既有 face 日志"}
    if not uniform:
        if before >= HEALTHY_AT_OR_ABOVE and not force:
            return {"session": sid, "status": "ok", "detection_before": round(before, 4),
                    "why": f"检出率 {before:.0%} ≥ {HEALTHY_AT_OR_ABOVE:.0%}，全画幅可用，不碰"}
        if before > BACKFILL_IF_RATE_AT_MOST and not force:
            # "半有半无"：裁切会改掉已经正常的那部分 ⟹ 补录模式下只报告，不自动动。
            return {"session": sid, "status": "partial", "detection_before": round(before, 4),
                    "why": f"检出率 {before:.0%} 属于「半有半无」——裁切会改掉已正常的那部分，"
                           f"不自动补（要动加 --force，或用 --uniform 统一重跑）"}

    frames = ledger_frames(sess_dir)
    if not frames:
        return {"session": sid, "status": "skip", "why": "账本里没有可用的留存帧"}

    # ── 1. 探测取景（只用前 PROBE_FRAMES 帧，内存安全）
    probe_imgs, probe_ts = [], []
    for p, ts in frames[:PROBE_FRAMES]:
        im = cv2.imread(str(p))
        if im is not None:
            probe_imgs.append(im)
            probe_ts.append(ts)
    if not probe_imgs:
        return {"session": sid, "status": "skip", "why": "留存帧解不出来"}

    name, crop, stat = probe_crop(probe_imgs, probe_ts)
    del probe_imgs
    if name == "none":
        # 探测全失败有**两种**完全不同的原因，报错必须分开说 —— 否则会让人去查
        # 朝向/遮挡，而真实原因可能是"帧太少"或"本来就达标、根本不用管"。
        # 2026-10-03 实测踩到：`20260928_230839_87c9` 台账只有 3 帧、检出率却是
        # **100%**（早已达标），却被报成"整条取景阶梯都检不到"，白读成一场失败。
        n_probe = len(probe_ts)
        if before >= HEALTHY_AT_OR_ABOVE:
            return {"session": sid, "status": "ok", "detection_before": round(before, 4),
                    "why": f"本来就达标（检出 {before:.0%}），未改动；"
                           f"探针在 {n_probe} 帧里一帧都没检出，样本太少，不作为失败计",
                    "probe": stat}
        if n_probe < PROBE_FRAMES:
            return {"session": sid, "status": "failed", "detection_before": round(before, 4),
                    "why": f"整条取景阶梯都检不到，但只有 {n_probe} 帧可探"
                           f"（不足 {PROBE_FRAMES} 帧）—— 样本太少，未必是真检不到，"
                           f"保持原样（不要硬凑）", "probe": stat}
        return {"session": sid, "status": "failed", "detection_before": round(before, 4),
                "why": f"整条取景阶梯都检不到（{n_probe} 帧探针、检出 {before:.0%}），"
                       f"保持原样（不要硬凑）", "probe": stat}
    if crop is None and not (force or uniform):
        return {"session": sid, "status": "ok", "detection_before": round(before, 4),
                "why": "全画幅探针就能检出 —— 说明该场日志与留存帧不一致，先查这个再说",
                "probe": stat}

    if dry_run:
        return {"session": sid, "status": "dry-run", "detection_before": round(before, 4),
                "crop": name, "probe": stat}

    # ── 2. 重放：新 pipeline + 新 logger，逐帧喂 declared_ts
    tmp_csv = out_csv.with_suffix(".csv.rebuilding")
    if tmp_csv.exists():
        tmp_csv.unlink()
    lg = DataLogger(log_type="video", session_id=sid)
    lg.log_file = str(tmp_csv)            # 显式改路径（app.py 也这么做）

    pipe = VideoPipeline(session_id=sid)
    written = detected = 0
    for p, ts in frames:
        im = cv2.imread(str(p))
        if im is None:
            continue
        img = apply_crop(im, crop)
        try:
            feats = pipe.process_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ts)[2] or {}
        except Exception as e:
            print(f"    ✗ 第 {written} 帧抛异常: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        lg.log(feats)
        written += 1
        if is_detected(feats):
            detected += 1

    after = detection_rate(tmp_csv)
    if uniform:
        # 统一重跑：只要**不比原来差**就接受。探针取的是检出最多的那级，
        # 所以正常情况下 after ≥ before（全画幅那批会原样重现 ⟹ 逐格相等）。
        good = after is not None and after >= before - 0.02
    else:
        good = (after is not None and after >= ACCEPT_MIN_RATE
                and after >= before + ACCEPT_MIN_GAIN)
    if not good:
        tmp_csv.unlink(missing_ok=True)
        return {"session": sid, "status": "failed", "detection_before": round(before, 4),
                "detection_after": None if after is None else round(after, 4),
                "why": (f"重放后 {after if after is None else f'{after:.0%}'} 比原来"
                        f"（{before:.0%}）还差，旧文件保持原样" if uniform else
                        f"重放后 {after if after is None else f'{after:.0%}'} 不满足接受条件"
                        f"（需 ≥{ACCEPT_MIN_RATE:.0%} 且比原值高 ≥{ACCEPT_MIN_GAIN:.0%}），"
                        f"旧文件保持原样"), "crop": name}

    # ── 3. 备份旧文件再换上去（选名规则见 `backup_path`）
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    bak = backup_path(out_csv)
    if out_csv.exists():
        shutil.move(str(out_csv), str(bak))
    shutil.move(str(tmp_csv), str(out_csv))

    return {"session": sid, "status": "rebuilt" if uniform else "backfilled",
            "detection_before": round(before, 4), "detection_after": round(after, 4),
            "crop": name, "crop_box": crop, "frames": written,
            "detected": detected, "backup": str(bak), "probe": stat}


def main() -> int:
    ap = argparse.ArgumentParser(description="拿留存帧重跑面部提取，补 0% 的场次")
    ap.add_argument("--sessions-dir", default=str(DEFAULT_RECORDINGS))
    ap.add_argument("--session", help="只补这一场（session_id 或目录名）")
    ap.add_argument("--all", action="store_true", help="补全部低检出场次")
    ap.add_argument("--list", action="store_true", help="只列出哪些场次该补")
    ap.add_argument("--dry-run", action="store_true", help="只探测定取景，不写文件")
    ap.add_argument("--uniform", action="store_true",
                    help="统一协议重跑：不论现有检出率多少都重提取，让全部场次由同一套"
                         "程序产出（协议 = 每场探测定取景）。前提已验证：重跑是保真的")
    ap.add_argument("--pending", action="store_true",
                    help="只处理**台账变过**的场次（例行回填用）。新录完一场跑这个："
                         "几秒到几分钟，不必全量重跑。指纹记在清单的 ledger_sig 里")
    ap.add_argument("--force", action="store_true",
                    help="连「半有半无」与已达标的场次也重跑（⚠️ 会改变既有数据，慎用）")
    args = ap.parse_args()

    root = Path(args.sessions_dir).expanduser()
    if not root.is_dir():
        print(f"✗ 素材目录不存在: {root}", file=sys.stderr)
        return 2

    if args.session:
        key = args.session.rstrip("/")
        cands = [d for d in root.iterdir() if d.is_dir() and (d.name == key or d.name.endswith(key))]
        if not cands:
            print(f"✗ 找不到场次: {key}", file=sys.stderr)
            return 2
        sessions = cands[:1]
    else:
        sessions = sorted(d for d in root.iterdir()
                          if d.is_dir() and (d / "retention.face.jsonl").is_file())

    if args.list:
        print(f"{'场次':<56} {'检出率':>8}  判定")
        print("-" * 78)
        todo = 0
        for d in sessions:
            sid = d.name.split("__")[-1]
            r = detection_rate(LOG_DIR / f"face_au_log_{sid}.csv")
            if r is None:
                print(f"{d.name[:54]:<56} {'—':>8}  无日志"); continue
            if r <= BACKFILL_IF_RATE_AT_MOST:
                tag = "★ 可补（基本全空）"; todo += 1
            elif r < HEALTHY_AT_OR_ABOVE:
                tag = "⚠ 半有半无，不自动补"
            else:
                tag = "ok"
            print(f"{d.name[:54]:<56} {r:>7.1%}  {tag}")
        print(f"\n共 {len(sessions)} 场：{todo} 场基本全空（可补）；"
              f"正常门槛 {HEALTHY_AT_OR_ABOVE:.0%}")
        return 0

    if args.uniform:
        args.all = True                      # uniform 天然是全量
    if args.pending:
        # ── 例行回填：只处理台账指纹与清单里记的不一致的场次
        # 指纹缺失（清单是这版工具之前写的）一律当作"变过"⟹ 保守地重跑一次，
        # 跑完指纹就补上了，之后每次都是几毫秒的比对。
        args.all = True
        man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
        before = len(sessions)
        sessions = [d for d in sessions
                    if man.get(d.name.split("__")[-1], {}).get("ledger_sig") != ledger_sig(d)]
        print(f"例行回填：{before} 场里 {len(sessions)} 场自上次处理后变过，"
              f"其余指纹一致已跳过\n")
        if not sessions:
            print("没有需要处理的场次。")
            return 0
    if not (args.all or args.session):
        ap.print_help()
        return 2

    # 台账指纹：写清单时记下，供下次 `--pending` 比对
    sig_by_sid = {d.name.split("__")[-1]: ledger_sig(d) for d in sessions}

    print(f"处理 {len(sessions)} 场 → 日志目录 {LOG_DIR}\n")
    results = []
    for i, d in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}] {d.name[:52]}", flush=True)
        t0 = time.time()
        try:
            r = backfill_one(d, dry_run=args.dry_run, force=args.force,
                             uniform=args.uniform or args.pending)
        except Exception as e:
            r = {"session": d.name, "status": "error", "why": f"{type(e).__name__}: {e}"}
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        tag = {"ok": "·", "skip": "·", "partial": "⚠", "dry-run": "?", "backfilled": "✓",
               "rebuilt": "✓", "failed": "✗", "error": "✗"}.get(r["status"], "?")
        print(f"    {tag} {r['status']}: {r.get('why') or ''}"
              f"{('  取景=' + r['crop']) if r.get('crop') else ''}"
              f"{('  ' + format(r['detection_before'], '.0%') + '→' + format(r['detection_after'], '.0%')) if r.get('detection_after') is not None else ''}"
              f"  ({r['seconds']}s)")

    # ── 记清单（不改 CSV 契约，但可审计）
    if not args.dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        old = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
        for r in results:
            if r["status"] in ("backfilled", "rebuilt", "failed"):
                old[r["session"]] = {k: v for k, v in r.items() if k != "probe"}
                # 记台账指纹 ⟹ 下次 `--pending` 只挑变过的（见 ledger_sig）
                sig = sig_by_sid.get(r["session"])
                if sig is not None:
                    old[r["session"]]["ledger_sig"] = sig
        MANIFEST.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n清单已写入 {MANIFEST}")

    n_ok = sum(1 for r in results if r["status"] in ("backfilled", "rebuilt"))
    n_fail = sum(1 for r in results if r["status"] in ("failed", "error"))
    n_partial = sum(1 for r in results if r["status"] == "partial")
    print(f"\n{'─'*60}\n补成功 {n_ok}   失败 {n_fail}   半有半无(未动) {n_partial}   "
          f"其余跳过/已达标 {len(results)-n_ok-n_fail-n_partial}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
