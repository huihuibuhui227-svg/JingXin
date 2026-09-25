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

from media_retention import RETENTION_FILENAME, recording_dir   # noqa: E402


def load_frames(session_dir: Path | str, modality: str = "face") -> list[tuple[Path, int]]:
    """读出本会话该模态的帧,按序号排好,带上**账本里记的当时时间戳**。

    以**盘上的文件**为准,不以账本为准:进程被杀在"写了文件、还没写账"之间时,
    账本会缺行;反过来若账本有行而文件不在,那一行直接跳过而不是崩。
    账本最后一行还可能是**半行**(被杀在写行中间)—— 也要跳过,不要崩。
    """
    session_dir = Path(session_dir)
    book = session_dir / RETENTION_FILENAME
    out: list[tuple[Path, int, int]] = []
    if book.exists():
        for line in book.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                      # 半行(被杀在半途)→ 跳过,不崩
            if rec.get("kind") != "frame" or rec.get("modality") != modality:
                continue
            p = session_dir / rec["file"]
            if not p.exists():
                continue
            out.append((p, int(rec["seq"]), int(rec.get("declared_ts") or 0)))
    out.sort(key=lambda t: t[1])
    return [(p, ts) for p, _seq, ts in out]


def replay_face(session_dir: Path, out_csv: Path) -> int:
    """按留存帧重跑 VideoPipeline,把每帧的序列化结果写成 CSV。返回帧数。"""
    import cv2
    from face_expression.pipeline.video_pipeline import VideoPipeline

    frames = load_frames(session_dir, "face")
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
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
