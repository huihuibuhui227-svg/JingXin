"""把留存下来的帧按**当时的时间戳**重放一遍 —— M2.6 的验收工具。

为什么需要一个专门的脚本(spec §7.1):留存的意义是"事后能重算"。要证明这一点,
就得拿盘上的帧重新跑一遍提取器,看结果是不是与**当场活跑**写下的 CSV 逐格相等。
相等 ⟹ 存的确实是我抽过的那些东西;不等 ⟹ 存错了(重编码?顺序错?时间戳造错了?)。

用法(WSL 侧):
    PY=~/miniconda3/envs/jingxin/bin/python
    $PY experiments/replay_retained.py --session-id 20260925_120000_aaaa \
        --out ~/shared/jingxin_recordings/20260925_120000_aaaa/replay_face.csv
然后把 replay_face.csv 与 data/logs/face_au_log_<sid>.csv 逐格比。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from media_retention import LEDGER_PREFIX, LEDGER_SUFFIX, recording_dir   # noqa: E402


def _ledger_books(session_dir: Path) -> list[Path]:
    """本目录下的全部账本文件(每个写入者一个;也认旧的单文件形态)。"""
    return sorted(set(session_dir.glob(f"{LEDGER_PREFIX}.*{LEDGER_SUFFIX}"))
                  | set(session_dir.glob(f"{LEDGER_PREFIX}{LEDGER_SUFFIX}")))


def load_frames(session_dir: Path | str, modality: str = "face"
                ) -> tuple[list[tuple[Path, int]], dict]:
    """读出本会话该模态的帧,按序号排好,带上**账本里记的当时时间戳**。

    返回 `(帧列表, 缺口报告)`。**缺口报告不是装饰**,它是这个工具最容易骗人的地方:

      以**盘上的文件**为准,账本只用来取 `declared_ts`。进程被杀在"写了文件、还没写账"
      之间时,账本会缺行 —— 那些帧**盘上还在,但没有当时的时间戳**。§7.1 要求重抽必须
      喂当时那个值、**不许另编一个**(编了就不是"逐格相等"了),所以它们的正确处置是
      **被报出来**,而不是静默跳过。

    实测代价(2026-09-25 使用者第一场真会话):账本被跨进程追加写碎了 25%,187 个 face
    帧只有 105 帧还有可用时间戳 —— 而当时的写法只打印"重放 105 帧",看的人根本不知道
    少了 82 帧、更不知道原因在账本。

    返回的 report 含:`frames_on_disk` / `frames_with_ts` / `missing_ts` / `torn_lines`。
    """
    session_dir = Path(session_dir)

    # ① 以盘为准:帧文件本身才是"有没有素材"的判据
    on_disk = sorted((session_dir / "media" / modality).glob("*.jpg"))
    by_seq: dict[int, Path] = {}
    for p in on_disk:
        stem = p.stem
        if stem.isdigit():
            by_seq[int(stem)] = p

    # ② 账本只用来取 declared_ts(以及后续扩展字段)
    ts_by_file: dict[str, int] = {}
    torn = 0
    for book in _ledger_books(session_dir):
        for line in book.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                torn += 1                     # 碎行/半行 → 计数,不崩
                continue
            if rec.get("kind") != "frame" or rec.get("modality") != modality:
                continue
            if rec.get("declared_ts") is None:
                continue
            rel = str(rec.get("file") or "")
            if rel:
                ts_by_file[rel] = int(rec["declared_ts"])

    frames: list[tuple[Path, int]] = []
    missing: list[Path] = []
    for seq in sorted(by_seq):
        p = by_seq[seq]
        rel = f"media/{modality}/{p.name}"
        if rel in ts_by_file:
            frames.append((p, ts_by_file[rel]))
        else:
            missing.append(p)

    report = {
        "frames_on_disk": len(by_seq),
        "frames_with_ts": len(frames),
        "missing_ts": [p.name for p in missing],
        "torn_lines": torn,
    }
    return frames, report


def replay_face(session_dir: Path, out_csv: Path) -> int:
    """按留存帧重跑 VideoPipeline,把每帧的序列化结果写成 CSV。返回帧数。"""
    import cv2
    from face_expression.pipeline.video_pipeline import VideoPipeline

    frames, report = load_frames(session_dir, "face")
    print(f"[素材] 盘上 {report['frames_on_disk']} 帧,其中 {report['frames_with_ts']} 帧有时间戳;"
          f"账本碎行 {report['torn_lines']} 行")
    if report["missing_ts"]:
        # 不静默:这些帧的图还在,但当时的时间戳丢了 —— §7.1 不允许另编一个
        print(f"⚠️  {len(report['missing_ts'])} 帧**没有可用时间戳**,本次不重放;"
              f"前几个:{report['missing_ts'][:5]}", file=sys.stderr)
    pipeline = VideoPipeline(session_id=session_dir.name)
    rows = []
    for path, ts in frames:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[跳过] 读不出:{path}", file=sys.stderr)
            continue
        _obj, _mesh, features = pipeline.process_frame(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ts)
        if features:
            rows.append(features)
    pipeline.close()
    if rows:
        keys = sorted({k for r in rows for k in r})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="重放留存帧(M2.6 验收)")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--root", default=None, help="默认取 JINGXIN_RECORDINGS_DIR")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    d = Path(args.root) / args.session_id if args.root else recording_dir(args.session_id)
    n = replay_face(d, Path(args.out))
    print(f"[完成] 重放 {n} 帧 -> {args.out}")
    print("下一步:与 data/logs/face_au_log_<sid>.csv 逐格比对(spec §7.1)")
    # 退出码:只有**盘上每一帧都重放了**才算通过 —— 有缺口就是 2,
    # 免得调用方把"重放了 105/187"当成成功。
    _frames, report = load_frames(d, "face")
    if report["missing_ts"] or report["torn_lines"]:
        print(f"[不完整] 缺口 {len(report['missing_ts'])} 帧 / 碎行 {report['torn_lines']} 行",
              file=sys.stderr)
        return 2
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
