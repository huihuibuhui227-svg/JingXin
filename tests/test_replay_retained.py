# tests/test_replay_retained.py
"""M2.6 验收工具:把留存的帧按**当时的时间戳**排好序交给重抽。"""
import json
from pathlib import Path

import media_retention
from experiments.replay_retained import load_frames


def _mk(root: Path, sid: str, entries):
    d = root / sid
    (d / "media" / "face").mkdir(parents=True)
    lines = []
    for seq, (name, ts) in enumerate(entries, start=1):
        (d / "media" / "face" / name).write_bytes(b"x")
        lines.append(json.dumps({"kind": "frame", "modality": "face", "seq": seq,
                                 "file": f"media/face/{name}", "declared_ts": ts,
                                 "bytes": 1, "sha256": "", "received_at_wall": 0.0,
                                 "source": "/analyze", "session_id": sid}))
    (d / media_retention.RETENTION_FILENAME).write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
    return d


def test_frames_come_back_in_time_order_with_their_own_timestamps(tmp_path):
    """回放要按**账本里的 declared_ts**,不是"文件的修改时间"、也不是重新计时。

    红法:改成用 index×200ms 重新造时间戳 —— 重抽出来的值不再等于当场算的值,
    而"逐格相等"正是留存的验收判据(spec §7.1)。
    """
    d = _mk(tmp_path, "s1", [("000001.jpg", 0), ("000002.jpg", 1200),
                             ("000010.jpg", 9000)])
    got = load_frames(d)
    assert [p.name for p, _ in got] == ["000001.jpg", "000002.jpg", "000010.jpg"]
    assert [ts for _, ts in got] == [0, 1200, 9000]


def test_frames_are_read_from_disk_not_from_the_ledger(tmp_path):
    """以**盘上文件**为准:账本里有一行但文件不在(进程被杀在两步之间)→ 跳过它。

    红法:改成遍历账本 —— 于是遇到坏行就崩在"读一个不存在的文件"上,
    而那正是"服务被中断"必然产生的形态。
    """
    d = _mk(tmp_path, "s1", [("000001.jpg", 0), ("000002.jpg", 100)])
    (d / "media" / "face" / "000002.jpg").unlink()
    got = load_frames(d)
    assert [p.name for p, _ in got] == ["000001.jpg"]


def test_half_written_ledger_line_is_skipped_not_fatal(tmp_path):
    """账本最后一行可能是半行(被杀在写行中间)—— 跳过它,不要崩。

    红法:去掉 json.JSONDecodeError 的 except。
    """
    d = _mk(tmp_path, "s1", [("000001.jpg", 0)])
    with (d / media_retention.RETENTION_FILENAME).open("a", encoding="utf-8") as f:
        f.write('{"kind": "frame", "modality": "fa')     # 半行
    got = load_frames(d)
    assert [p.name for p, _ in got] == ["000001.jpg"]


def test_other_modalities_are_not_mixed_in(tmp_path):
    """只要 face 的帧 —— 手势那些不属于这条重抽线。"""
    d = _mk(tmp_path, "s1", [("000001.jpg", 0)])
    with (d / media_retention.RETENTION_FILENAME).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "frame", "modality": "gesture", "seq": 1,
                            "file": "media/gesture/000001.jpg", "declared_ts": 5,
                            "bytes": 1, "sha256": "", "received_at_wall": 0.0,
                            "source": "/analyze", "session_id": "s1"}) + "\n")
    (d / "media" / "gesture").mkdir()
    (d / "media" / "gesture" / "000001.jpg").write_bytes(b"x")
    got = load_frames(d, "face")
    assert [p.name for p, _ in got] == ["000001.jpg"]
