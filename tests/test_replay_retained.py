# tests/test_replay_retained.py
"""M2.6 验收工具:把留存的帧按**当时的时间戳**排好序交给重抽。

这一层被独立审查挑出过一个**方向性错误**(Important 3):原实现遍历**账本**建列表,
与它自己的 docstring 和"以盘为准"的要求相反 —— 于是账本缺行的那些帧(盘上明明还在)
被静默丢掉,工具还打印"完成"。真会话上那是 187 帧里少 82 帧。
现在改成:**盘上文件建列表,账本只用来取时间戳,取不到的单独报数**。
"""
import json
from pathlib import Path

import media_retention
from experiments.replay_retained import load_frames


def _mk(root: Path, sid: str, entries, ledger=True):
    """建一场:帧文件写在盘上,账本(可选)单独写。"""
    d = root / sid
    (d / "media" / "face").mkdir(parents=True)
    lines = []
    for seq, (name, ts) in enumerate(entries, start=1):
        (d / "media" / "face" / name).write_bytes(b"x")
        lines.append(json.dumps({"kind": "frame", "modality": "face", "seq": seq,
                                 "file": f"media/face/{name}", "declared_ts": ts,
                                 "bytes": 1, "sha256": "", "received_at_wall": 0.0,
                                 "source": "/analyze", "session_id": sid}))
    if ledger:
        (d / f"retention.face.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_frames_come_back_in_time_order_with_their_own_timestamps(tmp_path):
    """回放要按**账本里的 declared_ts**,不是"文件的修改时间"、也不是重新计时。

    红法:改成用 index×200ms 重新造时间戳 —— 重抽出来的值不再等于当场算的值,
    而"逐格相等"正是留存的验收判据(spec §7.1)。
    """
    d = _mk(tmp_path, "s1", [("000001.jpg", 0), ("000002.jpg", 1200),
                             ("000010.jpg", 9000)])
    got, report = load_frames(d)
    assert [p.name for p, _ in got] == ["000001.jpg", "000002.jpg", "000010.jpg"]
    assert [ts for _, ts in got] == [0, 1200, 9000]
    assert report["frames_on_disk"] == 3 and report["missing_ts"] == []


def test_frame_on_disk_without_a_ledger_line_is_REPORTED_not_silently_dropped(tmp_path):
    """盘上有帧、账本里没有它的行 → 那一帧**必须出现在缺口报告里**。

    这是 Important 3 的修法(原实现把它静默跳过,于是"重放 105 帧"看起来像成功):
    进程被杀在"写了文件、还没写账"之间,以及账本被写碎时,都是这个形态。

    红法:改回"遍历账本建列表"—— 该帧直接从结果里消失,报告也不会提它。
    """
    d = _mk(tmp_path, "s1", [("000001.jpg", 0)])
    (d / "media" / "face" / "000002.jpg").write_bytes(b"x")      # 盘上有,账本无
    got, report = load_frames(d)
    assert [p.name for p, _ in got] == ["000001.jpg"]
    assert report["frames_on_disk"] == 2
    assert report["missing_ts"] == ["000002.jpg"]


def test_all_frames_on_disk_but_no_ledger_at_all_reports_everything(tmp_path):
    """账本整个不在(进程被杀在第一帧之前 / 被删)—— 帧全在盘上,全部报为缺口。"""
    d = _mk(tmp_path, "s1", [("000001.jpg", 0), ("000002.jpg", 100)], ledger=False)
    got, report = load_frames(d)
    assert got == []
    assert report["frames_on_disk"] == 2 and len(report["missing_ts"]) == 2


def test_torn_ledger_lines_are_counted_not_fatal(tmp_path):
    """碎行/半行要**计数**并继续,不能崩 —— 真会话上实测碎过 25%。

    红法:去掉 json.JSONDecodeError 的 except。
    """
    d = _mk(tmp_path, "s1", [("000001.jpg", 0)])
    with (d / "retention.face.jsonl").open("a", encoding="utf-8") as f:
        f.write('449"}\n')                        # 真会话里那种碎片
        f.write('{"kind": "frame", "modality": "fa')   # 半行
    got, report = load_frames(d)
    assert [p.name for p, _ in got] == ["000001.jpg"]
    assert report["torn_lines"] == 2


def test_other_modalities_are_not_mixed_in(tmp_path):
    """只要 face 的帧 —— 手势那些不属于这条重抽线。"""
    d = _mk(tmp_path, "s1", [("000001.jpg", 0)])
    (d / "media" / "gesture").mkdir()
    (d / "media" / "gesture" / "000001.jpg").write_bytes(b"x")
    with (d / "retention.gesture.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "frame", "modality": "gesture", "seq": 1,
                            "file": "media/gesture/000001.jpg", "declared_ts": 5,
                            "bytes": 1, "sha256": "", "received_at_wall": 0.0,
                            "source": "/analyze", "session_id": "s1"}) + "\n")
    got, report = load_frames(d, "face")
    assert [p.name for p, _ in got] == ["000001.jpg"]
    assert report["frames_on_disk"] == 1


def test_ledger_files_are_read_across_writers(tmp_path):
    """账本按写入者分了多个文件 —— 读侧要把它们合起来看。"""
    d = _mk(tmp_path, "s1", [("000001.jpg", 0)])
    with (d / "retention.face.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "frame", "modality": "face", "seq": 2,
                            "file": "media/face/000002.jpg", "declared_ts": 700,
                            "bytes": 1, "sha256": "", "received_at_wall": 0.0,
                            "source": "/analyze", "session_id": "s1"}) + "\n")
    (d / "media" / "face" / "000002.jpg").write_bytes(b"x")
    got, report = load_frames(d)
    assert [ts for _, ts in got] == [0, 700] and report["missing_ts"] == []
