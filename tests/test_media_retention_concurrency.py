# tests/test_media_retention_concurrency.py
"""M2.6:三个服务是**三个独立进程**,却都要往同一个会话的账本里追加行。

**这个文件是被一个真 bug 逼出来的**(2026-09-25,使用者第一场真会话):

    那场 face 187 帧 + gesture 187 帧 + 音频 16 段,账本应有 390 行,
    实测 308 行,其中 **77 行是碎的** —— 坏行的开头是 `449"}`、`_2449"}` 这类
    片段,即"一行的字节被另一个进程的行从中间插了进来"。

根因:进程内的 `threading.Lock` **跨不了进程**;而 `open(p, "a")` 是**带缓冲**的,
flush 时会把一次逻辑写拆成多次系统调用。两者一叠加,行就被劈开了。

修法两道防线(**这个文件钉的就是它们**):
  ① 每个写入者一个文件 `retention.<writer>.jsonl` —— 从根上让两个进程不碰同一个文件,
     不依赖任何文件系统的锁语义(`~/shared` 是 9p/drvfs,flock 支持不可靠);
  ② 单次 `os.write` 的 O_APPEND —— 行小于 4096 字节时一次写是原子的。

**反向复现**:把 `ledger_path` 改回"所有写入者共用一个文件名"→ 本文件必红。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import media_retention

_DRIVER = """
import sys
sys.path.insert(0, {root!r})
import media_retention
media_retention._reset_for_tests()
sid, modality, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
for i in range(n):
    if modality == "audio":
        media_retention.retain_audio(sid, "raw", b"RIFF" + b"\\x00" * 60)
    else:
        media_retention.retain_frame(sid, modality, b"x" * 200, declared_ts=i)
"""


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


def test_each_writer_gets_its_own_ledger_file(_isolated_root):
    """三个写入者各写各的文件 —— 这是"两个进程不碰同一个文件"的直接体现。

    红法:让 ledger_path 对所有 writer 返回同一个名字 → 立刻红。
    """
    media_retention.retain_frame("s1", "face", b"f")
    media_retention.retain_frame("s1", "gesture", b"g")
    media_retention.retain_audio("s1", "raw", b"RIFF" + b"\x00" * 60)

    names = sorted(p.name for p in media_retention.ledger_files("s1"))
    assert names == ["retention.audio.jsonl", "retention.face.jsonl",
                     "retention.gesture.jsonl"]
    # 各文件里只该有自己那一类的行
    def rows(w):
        return [json.loads(l) for l in
                media_retention.ledger_path("s1", w).read_text(encoding="utf-8").splitlines()
                if l.strip()]
    assert [r["modality"] for r in rows("face")] == ["face"]
    assert [r["modality"] for r in rows("gesture")] == ["gesture"]
    assert [r["kind"] for r in rows("audio")] == ["raw"]


def test_three_processes_never_tear_a_line(_isolated_root):
    """**三个真进程**并发写 → 每个账本文件的每一行都必须能解析,行数一件不少。

    这就是那个真 bug 的复现:修之前三个进程共用一个 `retention.jsonl` 且用带缓冲的
    写法,实测 308 行里 77 行碎掉。修之后每个写入者一个文件 + 单次 os.write。

    红法:把 ledger_path 改回共用一个文件名 → 本测试红(行被劈开)。
    """
    n = 120
    procs = []
    for modality in ("face", "gesture", "audio"):
        procs.append(subprocess.Popen(
            [sys.executable, "-c", _DRIVER.format(root=str(Path(__file__).resolve().parents[1])),
             "s1", modality, str(n)],
            env={**os.environ, "JINGXIN_RECORDINGS_DIR": str(_isolated_root)},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    for p in procs:
        _out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode()[-800:]

    ledger_dir = _isolated_root / "s1"
    files = sorted(ledger_dir.glob("retention.*.jsonl"))
    assert len(files) == 3, [f.name for f in files]

    total = 0
    for f in files:
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, l in enumerate(lines, 1):
            try:
                json.loads(l)
            except json.JSONDecodeError as e:      # 碎的
                pytest.fail(f"{f.name} 第 {i} 行被劈开了:{e};开头 {l[:40]!r}")
        assert len(lines) == n, f"{f.name} 只有 {len(lines)} 行,应为 {n}"
        total += len(lines)
    assert total == 3 * n


def test_media_files_are_not_torn_either(_isolated_root):
    """素材文件本身也不能被劈开 —— 不同模态写**不同目录**,从根上没有争用。

    红法:把帧和音频都写进同一个目录且用同一个序号空间。
    """
    media_retention.retain_frame("s1", "face", b"F" * 500)
    media_retention.retain_frame("s1", "gesture", b"G" * 500)
    assert (_isolated_root / "s1" / "media" / "face" / "000001.jpg").read_bytes() == b"F" * 500
    assert (_isolated_root / "s1" / "media" / "gesture" / "000001.jpg").read_bytes() == b"G" * 500
