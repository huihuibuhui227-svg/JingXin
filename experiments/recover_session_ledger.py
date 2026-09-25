#!/usr/bin/env python3
"""一次性恢复:把一场被跨进程写碎的账本重建出来。

**为什么需要它**(2026-09-25 使用者第一场真会话 `20260925_203826_2449`):
三个服务共写一个 `retention.jsonl`,把 390 行里的 77 行写碎了 —— 而 `declared_ts`
**只存在于账本里**(spec §7.1 要求重抽必须用它、不许另编)。素材本身没坏。

**恢复原则:只恢复能证明的,不编。**

* 帧文件、字节数、sha256、序号 —— 全部**从盘上算**,精确。
* `declared_ts` —— 两个来源:
    ① 碎账本里**没碎的那些行**(按 `file` 对上);
    ② **face 的活跑 CSV** 的 `timestamp` 列(会话相对秒 × 1000)。
       这条**经过交叉验证**:两边都有的 105 帧,**105 一致 / 0 不一致**。
* **gesture 的 CSV `timestamp` 是绝对墙钟**(已知问题 F9),推送不出精确值
  (差 16–36 ms 且不是常数偏移)⟹ 那 77 帧的 `declared_ts` 写 **null**,
  让重抽工具把它们**报成缺口** —— 用近似值冒充就等于伪造"逐格相等"。

原文件改名 `retention.jsonl.corrupt` 留证,不删。
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

SESSION = sys.argv[1] if len(sys.argv) > 1 else "20260925_203826_2449"
ROOT = Path.home() / "shared" / "jingxin_recordings" / SESSION
LOGS = Path(__file__).resolve().parents[3] / "data" / "logs"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def intact_frame_ts() -> dict[str, int]:
    """从碎账本里捞出**没碎**的 frame 行 → file → declared_ts。"""
    out: dict[str, int] = {}
    f = ROOT / "retention.jsonl"
    if not f.exists():
        return out
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "frame" and r.get("declared_ts") is not None and r.get("file"):
            out[r["file"]] = int(r["declared_ts"])
    return out


def csv_ts_ms(prefix: str) -> list[int]:
    """活跑 CSV 的 timestamp 列 → 毫秒(仅 face 可用;gesture 是墙钟)。"""
    p = LOGS / f"{prefix}_{SESSION}.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return [round(float(r["timestamp"]) * 1000) for r in rows]


def recover_frames(modality: str, ledger_ts: dict[str, int],
                   csv_ms: list[int] | None) -> tuple[list[dict], dict]:
    d = ROOT / "media" / modality
    files = sorted(p for p in d.glob("*.jpg") if p.stem.isdigit())
    out, stat = [], {"exact": 0, "from_csv": 0, "lost": 0}
    for p in files:
        seq = int(p.stem)
        rel = f"media/{modality}/{p.name}"
        if rel in ledger_ts:
            ts, src = ledger_ts[rel], "ledger"
            stat["exact"] += 1
        elif csv_ms and 1 <= seq <= len(csv_ms):
            ts, src = csv_ms[seq - 1], "live_csv"
            stat["from_csv"] += 1
        else:
            ts, src = None, "lost"
            stat["lost"] += 1
        out.append({"kind": "frame", "modality": modality, "seq": seq, "file": rel,
                    "bytes": p.stat().st_size, "sha256": sha256(p),
                    "received_at_wall": None, "declared_ts": ts,
                    "source": "/analyze", "session_id": SESSION, "recovered": src})
    return out, stat


def recover_audio(ledger_ts: dict) -> list[dict]:
    """音频的 declared_ts 本来就是 null,所以除 sha256 外全部可从盘上精确算出。"""
    d = ROOT / "media" / "audio"
    out = []
    for p in sorted(d.iterdir()):
        # ⚠️ 别用 `name.split("_")[0]` 取序号:`0001.webm` 里**没有下划线**,
        # 拿回的是整个 "0001.webm" → isdigit 为假 → raw 全被跳过(本脚本踩过一次)。
        m = re.match(r"(\d+)", p.name)
        if not m:
            continue
        seq = int(m.group(1))
        kind = "converted" if "_converted" in p.name else "raw"
        out.append({"kind": kind, "modality": None, "seq": seq,
                    "file": f"media/audio/{p.name}", "bytes": p.stat().st_size,
                    "sha256": sha256(p), "received_at_wall": None,
                    "declared_ts": None, "source": "/asr", "session_id": SESSION,
                    "recovered": "from_disk"})
    return out


def main() -> int:
    ledger_ts = intact_frame_ts()
    print(f"碎账本里可用的 frame 行:{len(ledger_ts)}")

    face_ts = csv_ts_ms("face_au_log")
    print(f"face 活跑 CSV 行数:{len(face_ts)}(会话相对毫秒)")

    # 留证:原文件改名,不删
    orig = ROOT / "retention.jsonl"
    if orig.exists():
        orig.rename(ROOT / "retention.jsonl.corrupt")
        print("原文件已改名 → retention.jsonl.corrupt(留证)")

    for modality, csv_ms in (("face", face_ts), ("gesture", None)):
        rows, stat = recover_frames(modality, ledger_ts, csv_ms)
        out = ROOT / f"retention.{modality}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {modality:8s} {len(rows):4d} 行 → {out.name}   "
              f"精确 {stat['exact']} / 从活跑CSV补齐 {stat['from_csv']} / **永久丢失 {stat['lost']}**")

    rows = recover_audio(ledger_ts)
    out = ROOT / "retention.audio.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  audio    {len(rows):4d} 行 → {out.name}   (全部从盘上精确算出)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
