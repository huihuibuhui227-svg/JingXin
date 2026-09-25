# tests/test_media_retention.py
"""M2.6 服务端留存:路径/守卫/开关/预检这一层。

这一层是"素材会不会被静默丢掉"的唯一防线,所以每条测试都要能说出
**撤掉哪个生产改动会让它变红**。
"""
import hashlib
import json
import threading
from pathlib import Path

import pytest

import media_retention


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    """每个测试都落在自己的临时目录里 —— 绝不碰真的 ~/shared。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


def test_default_root_is_the_shared_volume():
    """默认落点必须是 ~/shared/jingxin_recordings(→ /mnt/d)。

    红法:把 DEFAULT_ROOT 改成仓库内的路径 —— 那会让原始媒体进 git,
    也会撑 WSL 的 ext4.vhdx(下一步 §4.7)。
    """
    assert media_retention.DEFAULT_ROOT == Path.home() / "shared" / "jingxin_recordings"


def test_enabled_by_default_and_can_be_switched_off(monkeypatch):
    """默认开 —— "忘了开"正是丢素材的那条路径(spec §6)。"""
    assert media_retention.enabled() is True
    for off in ("0", "false", "FALSE", "no", "off", " off "):
        monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", off)
        assert media_retention.enabled() is False, off


def test_path_traversal_is_rejected():
    """会话 id 是客户端可控的,而它会被拼进路径 —— 必须拦在拼路径之前。

    红法:去掉 validate_session_id 里的正则。`../../jingxin/x` 就能在录制根
    **之外**建目录并写入原始媒体。
    """
    for bad in ("../x", "a/b", "..", "", "  ", "a" * 129, "a.b"):
        with pytest.raises(ValueError):
            media_retention.validate_session_id(bad)
    assert media_retention.validate_session_id("20260925_124346_6e4a") == "20260925_124346_6e4a"
    assert media_retention.validate_session_id("NONE") == "NONE"


def test_recording_dir_honors_the_env_override(_isolated_root):
    d = media_retention.recording_dir("s1")
    assert d == _isolated_root / "s1"
    assert d.is_dir()


def test_preflight_is_loud_when_the_root_is_not_writable(tmp_path, monkeypatch):
    """首件素材预检不过 → **抛**,不是 warning。

    红法:把 _prepare 里的 raise 改成 return False —— 于是磁盘坏了也一路静默,
    跑完一场才发现一张图都没存(而这一场是人重跑不回来的)。
    """
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)                       # 只读目录
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(blocked / "sub"))
    media_retention._reset_for_tests()
    if media_retention._can_write_here(blocked):   # root 用户跑测试时 chmod 拦不住
        pytest.skip("本环境以 root 运行,chmod 拦不住写")
    with pytest.raises(RuntimeError, match="预检"):
        media_retention.retain_frame("s1", "face", b"\xff\xd8jpeg", declared_ts=0)


def test_preflight_failure_is_only_loud_once(tmp_path, monkeypatch):
    """首件大声报过之后,后续件不再抛 —— 中断也换不回已丢的字节,还白耗人的时间。

    但要**留痕**(见 Task 4)。红法:把 _LOUD_DONE 判断去掉 → 每一帧都 500,
    整场用不了。
    """
    blocked = tmp_path / "blocked2"
    blocked.mkdir()
    blocked.chmod(0o500)
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(blocked / "sub"))
    media_retention._reset_for_tests()
    if media_retention._can_write_here(blocked):
        pytest.skip("本环境以 root 运行,chmod 拦不住写")
    with pytest.raises(RuntimeError):
        media_retention.retain_frame("s1", "face", b"x", declared_ts=0)
    assert media_retention.retain_frame("s1", "face", b"x", declared_ts=1) is None
    assert media_retention.degraded_reasons("s1")


# ── Task 2:存帧 ────────────────────────────────────────────────────────────

def _jsonl(root: Path, sid: str = "s1") -> list[dict]:
    p = root / sid / media_retention.RETENTION_FILENAME
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_frame_bytes_land_verbatim(_isolated_root):
    """盘上的字节必须与收到的**逐字节相同**,账本里的 sha256 要能证明这一点。

    红法:把 `path.write_bytes(data)` 换成一进一出的 `cv2.imencode/imdecode`
    —— 看起来"还存了一张图",但重抽出来的值不再等于当场算的值,而这是留存存在的
    全部理由(spec §7.1)。
    """
    payload = b"\xff\xd8\xff\xe0not-a-real-jpeg-but-bytes-are-bytes"
    rec = media_retention.retain_frame("s1", "face", payload, declared_ts=1500,
                                       source="/analyze")
    assert rec is not None
    f = _isolated_root / "s1" / "media" / "face" / "000001.jpg"
    assert f.read_bytes() == payload
    assert rec["sha256"] == hashlib.sha256(payload).hexdigest()
    assert rec["bytes"] == len(payload)
    assert rec["kind"] == "frame" and rec["modality"] == "face" and rec["seq"] == 1
    assert rec["declared_ts"] == 1500 and rec["source"] == "/analyze"
    assert rec["session_id"] == "s1"
    assert rec["file"] == "media/face/000001.jpg"
    assert isinstance(rec["received_at_wall"], float)


def test_sequence_is_zero_padded_so_name_order_is_time_order(_isolated_root):
    """`000001` 而不是 `1` —— 字典序即时间序,重抽脚本才能直接按文件名排。

    红法:去掉 `:06d` 里的补零。第 10 帧会排到第 2 帧前面。
    """
    for i in range(12):
        media_retention.retain_frame("s1", "face", b"x", declared_ts=i)
    names = sorted(p.name for p in (_isolated_root / "s1" / "media" / "face").iterdir())
    assert names[0] == "000001.jpg" and names[1] == "000002.jpg"
    assert names[9] == "000010.jpg"
    assert [json.loads(l)["seq"] for l in
            (_isolated_root / "s1" / media_retention.RETENTION_FILENAME)
            .read_text(encoding="utf-8").splitlines()] == list(range(1, 13))


def test_two_modalities_count_separately(_isolated_root):
    """face 与 gesture 各自从 1 开始 —— 它们是两条独立的字节流。"""
    a = media_retention.retain_frame("s1", "face", b"f")
    b = media_retention.retain_frame("s1", "gesture", b"g")
    assert a["seq"] == 1 and b["seq"] == 1
    assert (_isolated_root / "s1" / "media" / "gesture" / "000001.jpg").exists()


def test_retention_off_writes_nothing(_isolated_root, monkeypatch):
    """关掉时**一个文件都不写**,且不报错(它是旁路,关了就该与今天一样)。"""
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    assert media_retention.retain_frame("s1", "face", b"x") is None
    assert not (_isolated_root / "s1" / "media").exists()
    assert media_retention.degraded_reasons("s1") == []


def test_none_bucket_is_retained_like_any_other_id(_isolated_root):
    """`NONE` 是既有设计里的一个正常会话(transcript_store 也给它建目录)。

    ⚠️ 代价要写在这里:不同场次、不同人的素材会**混进同一个 `NONE/` 目录**
    (与下一步 §3 第 11 条的既有问题同源)。真要区分只能靠 `received_at_wall`。
    """
    rec = media_retention.retain_frame("NONE", "face", b"x")
    assert rec is not None and rec["session_id"] == "NONE"


def test_concurrent_frames_do_not_collide(_isolated_root):
    """同一会话并发 40 帧 → 40 个文件、40 行账、序号无重复。

    红法:去掉 _session_lock 的 with —— "取序号→写文件→写账"三步会交错,
    出现同名文件互相覆盖(盘上文件数 < 账本行数)。
    """
    def burst():
        for _ in range(10):
            media_retention.retain_frame("s1", "face", b"x")
    ts = [threading.Thread(target=burst) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    files = list((_isolated_root / "s1" / "media" / "face").iterdir())
    assert len(files) == 40
    seqs = [json.loads(l)["seq"] for l in
            (_isolated_root / "s1" / media_retention.RETENTION_FILENAME)
            .read_text(encoding="utf-8").splitlines()]
    assert sorted(seqs) == list(range(1, 41))


# ── Task 3:存音频 ──────────────────────────────────────────────────────────

def test_raw_and_converted_share_one_sequence_number(_isolated_root):
    """一段回答的原始容器与转换后的 WAV **共用同一个序号**。

    为什么:它们说的是同一段话,文件名要能对上(`0001.webm` ↔ `0001_converted.wav`)。
    红法:让 converted 也走 _bump_seq —— 变成 0001.webm 与 0002_converted.wav,
    两件互不相干的编号,重抽时对不上是哪段回答。
    """
    webm = b"\x1a\x45\xdf\xa3fake-webm"
    wav = b"RIFF" + b"\x00" * 60
    r1 = media_retention.retain_audio("s1", "raw", webm, source="/asr")
    r2 = media_retention.retain_audio("s1", "converted", wav, source="/asr")
    assert r1["seq"] == 1 and r2["seq"] == 1
    assert r1["file"] == "media/audio/0001.webm"
    assert r2["file"] == "media/audio/0001_converted.wav"
    assert (_isolated_root / "s1" / r1["file"]).read_bytes() == webm
    assert (_isolated_root / "s1" / r2["file"]).read_bytes() == wav


def test_second_answer_gets_sequence_two(_isolated_root):
    media_retention.retain_audio("s1", "raw", b"RIFF" + b"\x00" * 60)
    r = media_retention.retain_audio("s1", "raw", b"\x1a\x45\xdf\xa3x")
    assert r["seq"] == 2 and r["file"] == "media/audio/0002.webm"


def test_extension_is_sniffed_from_content_not_assumed(_isolated_root):
    """扩展名由**内容**决定 —— 调用方不必知道自己在送什么容器。

    红法:写死 `.webm`。`/interview/answer_audio` 送的是 WAV,会被存成
    `0001.webm`,重抽时按扩展名选的解码器全错。
    """
    assert media_retention.sniff_audio_ext(b"RIFF....WAVEfmt ") == "wav"
    assert media_retention.sniff_audio_ext(b"\x1a\x45\xdf\xa3\x01\x00") == "webm"
    assert media_retention.sniff_audio_ext(b"\x00\x01\x02") == "bin"
    r = media_retention.retain_audio("s1", "raw", b"RIFF" + b"\x00" * 60)
    assert r["file"].endswith(".wav")


def test_audio_off_writes_nothing(_isolated_root, monkeypatch):
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    assert media_retention.retain_audio("s1", "raw", b"RIFF" + b"\x00" * 60) is None


# ── Task 4:中途失败留痕 + 守卫副本的契约 ────────────────────────────────────

def test_midway_write_failure_is_recorded_and_does_not_raise(_isolated_root):
    """写盘中途失败 → **不抛**(不阻断分析)、但必须留痕。

    红法:把 except 里的 _mark_degraded 去掉、只 return None —— 于是这一场静默地
    少了一批素材,而报告里什么都看不出来。

    ⚠️ 用 `MonkeyPatch.context()` 而不是 `monkeypatch.undo()`:后者会把 autouse fixture
    设的 `JINGXIN_RECORDINGS_DIR` 一起撤掉,于是下面那句真的写进 `~/shared`。
    """
    media_retention.retain_frame("s1", "face", b"ok1")     # 先过预检
    calls = {"n": 0}
    real = Path.write_bytes

    def flaky(self, data):
        calls["n"] += 1
        if calls["n"] == 1:                # 预检已过,这里的第 1 次就是本帧的写
            raise OSError(28, "No space left on device")
        return real(self, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_bytes", flaky)
        assert media_retention.retain_frame("s1", "face", b"ok2") is None   # 不抛
    assert any("No space left" in r for r in media_retention.degraded_reasons("s1"))
    # 留痕之后不瘫痪:后续件照常能写
    assert media_retention.retain_frame("s1", "face", b"ok3") is not None


def test_degraded_reasons_is_empty_when_all_is_well(_isolated_root):
    media_retention.retain_frame("s1", "face", b"x")
    assert media_retention.degraded_reasons("s1") == []
    assert media_retention.degraded_reasons("never-seen") == []


def test_guard_regex_matches_transcript_store():
    """本模块的守卫与 `transcript_store` 必须是**同一个口径**。

    红法:只改一份(本模块放宽一位长度,或允许点号)—— 同一个客户端请求在
    "原句落哪"与"像素落哪"两处得到不同判定。
    与三份 `_resolve_session_id` 的源码级契约测试同一手法。
    """
    from voice_interaction.asr import transcript_store
    assert media_retention.SESSION_ID_PAT.pattern == transcript_store.SESSION_ID_PAT.pattern
