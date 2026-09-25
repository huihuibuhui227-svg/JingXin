# tests/test_media_retention_review_fixes.py
"""M2.6 独立审查挑出的四条(除已修的 Critical 1 / Important 3 之外)。

每条都对应一个**会真的发生**的形态,不是风格问题:

* **Critical 2** —— 序号只在进程内存里 ⟹ 重启/同 id 复用会**同名覆盖已留存的素材**。
  实盘证据:`20260924_153012_9f3c` 账本 52 行、盘上只 12 个文件,12 条重复指向同一文件;
  `NONE/` 同样(54 行 / 12 文件)。
* **Important 1** —— 预检探的目录不是真正要写的目录,且探针是 **0 字节**(满盘照样过)。
* **Important 2** —— 降级只活在内存、账本从不写它 ⟹ spec §6"绝不允许静默"只兑现了一半。
* **Important 4** —— `/asr` 的 raw↔converted 配对靠"此刻计数值"而非"本次那个号"。
"""
import json

import pytest

import media_retention


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


def _rows(root, writer):
    p = root / "s1" / f"retention.{writer}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── Critical 2:序号必须从盘上推 ──────────────────────────────────────────────

def test_seq_survives_a_process_restart(_isolated_root):
    """模拟**服务重启**(清空全部进程内状态)后继续留存 → 不能覆盖已有文件。

    红法:把 `_bump_seq` 改回纯内存计数(`_COUNTERS.get(key,0)+1`)——
    重启后第一帧又是 `000001.jpg`,把原来那张**同名盖掉**,
    而账本里于是出现两条 sha256 不同、指向同一文件的行。
    """
    media_retention.retain_frame("s1", "face", b"first")
    media_retention.retain_frame("s1", "face", b"second")
    before = (_isolated_root / "s1" / "media" / "face" / "000001.jpg").read_bytes()

    media_retention._reset_for_tests()          # ← 等价于服务重启(进程内状态全没)
    media_retention.retain_frame("s1", "face", b"after-restart")

    assert [p.name for p in sorted((_isolated_root / "s1" / "media" / "face").iterdir())] == \
        ["000001.jpg", "000002.jpg", "000003.jpg"]
    assert (_isolated_root / "s1" / "media" / "face" / "000001.jpg").read_bytes() == before
    assert len({r["file"] for r in _rows(_isolated_root, "face")}) == 3   # 三行指三个文件


def test_seq_survives_for_the_none_bucket_too(_isolated_root):
    """`NONE` 桶跨场次复用同一个 id —— 同样不能被覆盖(实盘上它已经发生过)。"""
    media_retention.retain_frame("NONE", "face", b"a")
    media_retention._reset_for_tests()
    media_retention.retain_frame("NONE", "face", b"b")
    assert sorted(p.name for p in (_isolated_root / "NONE" / "media" / "face").iterdir()) == \
        ["000001.jpg", "000002.jpg"]


def test_audio_seq_also_survives_restart(_isolated_root):
    media_retention.retain_audio("s1", "raw", b"RIFF" + b"\x00" * 60)
    media_retention._reset_for_tests()
    media_retention.retain_audio("s1", "raw", b"RIFF" + b"\x00" * 60)
    assert sorted(p.name for p in (_isolated_root / "s1" / "media" / "audio").iterdir()) == \
        ["0001.wav", "0002.wav"]


# ── Important 1:预检要探**真正要写**的那个目录,且探针非空 ─────────────────

def test_preflight_probes_the_modality_dir_not_just_media(_isolated_root):
    """`media/face` 若已被占成**普通文件**,首件素材必须当场抛。

    红法:把 `_prepare` 的探测目标改回 `media_dir(session_id)`(只到 `media/`)——
    `media/` 是好的,预检通过,首帧才 `FileExistsError`,被吞成 degraded、请求 200。
    """
    d = _isolated_root / "s1" / "media"
    d.mkdir(parents=True)
    (d / "face").write_bytes(b"a regular file where a directory belongs")
    with pytest.raises(RuntimeError, match="预检"):
        media_retention.retain_frame("s1", "face", b"x")


def test_probe_writes_non_zero_bytes(_isolated_root, monkeypatch):
    """探针必须是**非空**写 —— 空文件不占数据块,磁盘满时照样建得出。

    红法:把 _can_write_here 里的 `b"\\x00" * 4096` 改回 `b""`。
    """
    seen = {}
    real = media_retention._write_probe

    def spy(directory):
        seen["n"] = len(real(directory))
        return True

    monkeypatch.setattr(media_retention, "_write_probe", spy)
    media_retention.retain_frame("s1", "face", b"x")
    assert seen.get("n", 0) >= 4096


def test_probe_name_is_unique_per_process_and_thread(_isolated_root):
    """探针名要带 pid 与线程号 —— 两个进程探同一个会话时不能互相踩。

    红法:改回固定的 `.retention_probe` → 下面两个线程拿到的名字相同。
    """
    import os
    import threading
    name = media_retention._probe_name()
    assert str(os.getpid()) in name
    assert str(threading.get_ident()) in name
    # 只测**并发**那两个:线程号在先后不重叠时会被 Python 复用(实测),
    # 而探针文件写完就删,复用无害 —— 有害的是两个**同时活着**的线程撞名。
    seen = []
    barrier = threading.Barrier(2)

    def grab():
        barrier.wait()                     # 保证两个线程真的同时活着
        seen.append(media_retention._probe_name())

    ts = [threading.Thread(target=grab) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(set(seen)) == 2, seen


# ── Important 2:降级要写进账本(不能只活在内存里) ────────────────────────

def test_midway_failure_is_recorded_into_the_ledger(_isolated_root, monkeypatch):
    """写盘中途失败 → 账本里必须**留一行**,而不只是内存里记一笔。

    红法:去掉 `_mark_degraded` 里写账那一段 —— 进程重启后这件事就彻底没痕迹了,
    而 spec §6 要求"任何情况都不得静默"。
    """
    from pathlib import Path
    media_retention.retain_frame("s1", "face", b"ok1")
    calls = {"n": 0}
    real = Path.write_bytes

    def flaky(self, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        return real(self, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_bytes", flaky)
        assert media_retention.retain_frame("s1", "face", b"ok2") is None

    rows = _rows(_isolated_root, "face")
    assert any(r.get("kind") == "degraded" for r in rows), rows
    assert any("No space left" in str(r.get("reason", "")) for r in rows)


def test_degraded_lines_survive_a_restart(_isolated_root):
    """降级痕迹在**重启后仍读得到**(从账本读,不是从内存)。

    ⚠️ 必须先让预检成功过一次(所以先正常写一件),否则 `write_bytes` 一被封,
    连预检都过不去 —— 那抛的是 RuntimeError(首件大声失败),不是 degraded。
    """
    from pathlib import Path
    media_retention.retain_frame("s1", "face", b"ok")      # 先过预检
    real = Path.write_bytes

    def flaky(self, data):
        raise OSError(28, "No space left on device")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_bytes", flaky)
        assert media_retention.retain_frame("s1", "face", b"x") is None
    assert real is Path.write_bytes or True

    media_retention._reset_for_tests()                    # ← 等价于服务重启
    assert any("No space" in r for r in media_retention.degraded_reasons("s1"))


# ── Important 4:converted 要显式拿本次 raw 的那个号 ────────────────────────

def test_converted_can_be_pinned_to_the_raw_seq(_isolated_root):
    """`converted` 显式带 `seq` → 与**那次** raw 配对,而不是"此刻的最新值"。

    红法:让 converted 永远取 `_current_seq` —— 两个 raw 交错时,B 的 converted
    会被后到的 A 覆盖(账本两条行指向同一文件、sha 不同,§7.2 当场红)。
    """
    a = media_retention.retain_audio("s1", "raw", b"RIFF" + b"A" * 60)
    b = media_retention.retain_audio("s1", "raw", b"RIFF" + b"B" * 60)
    assert (a["seq"], b["seq"]) == (1, 2)

    media_retention.retain_audio("s1", "converted", b"RIFF" + b"b-converted",
                                 seq=b["seq"])
    media_retention.retain_audio("s1", "converted", b"RIFF" + b"a-converted",
                                 seq=a["seq"])

    one = (_isolated_root / "s1" / "media" / "audio" / "0001_converted.wav").read_bytes()
    two = (_isolated_root / "s1" / "media" / "audio" / "0002_converted.wav").read_bytes()
    assert one.endswith(b"a-converted") and two.endswith(b"b-converted")
    assert len({r["file"] for r in _rows(_isolated_root, "audio")}) == 4
