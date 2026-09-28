# tests/test_media_retention_video.py
"""前端 `MediaRecorder` 录的原生音视频必须原样落成 `media/camera.webm`。

它和帧/音频的**两处不同**(所以单开一个文件):
  ① 文件落在 `media/` **下面**,不是 `media/camera/` —— 不许建出空目录;
  ② 账本写入者是 `camera`(既有三个是 face/gesture/audio)。
"""
import importlib
from pathlib import Path

import pytest

media_retention = importlib.import_module("media_retention")

WEBM = b"\x1a\x45\xdf\xa3" + b"native-camera-payload" * 40      # EBML 头


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


def _write(pcm=WEBM, sid="20260925_203826_2449"):
    return media_retention.retain_uploaded_video(sid, pcm, source="/session/x/media")


def test_video_bytes_land_verbatim(_isolated):
    """存的是**同一份 bytes**:盘上逐字节等于收到的。"""
    rec = _write()
    p = _isolated / "20260925_203826_2449" / "media" / "camera.webm"
    assert p.read_bytes() == WEBM
    assert rec["bytes"] == len(WEBM)
    assert rec["kind"] == "video"
    assert rec["file"] == "media/camera.webm"
    assert rec["sha256"] == media_retention.hashlib.sha256(WEBM).hexdigest()


def test_no_empty_camera_directory_is_created(_isolated):
    """`media/camera/` **不许**出现 —— 它只会在人工核对时制造困惑。

    红法:把 `retain_uploaded_video` 里的 `_prepare(sid, "camera", probe_parts=())`
    改回 `_prepare(sid, "camera")`。
    """
    _write()
    assert not (_isolated / "20260925_203826_2449" / "media" / "camera").exists()


def test_video_ledger_is_its_own_writer(_isolated):
    """账本写给 `retention.camera.jsonl`,与另外三个写入者分开。"""
    _write()
    names = sorted(p.name for p in media_retention.ledger_files("20260925_203826_2449"))
    assert names == ["retention.camera.jsonl"], names
    assert "camera" in media_retention.LEDGER_WRITERS


def test_second_upload_is_refused_and_the_first_one_survives(_isolated):
    """★ 第二份 `camera.webm` **被拒绝**,而第一份**逐字节没动**。

    **这条测试在 2026-09-28 反过来了** —— 它原来钉的是"第二次上传覆盖文件、
    账本留两行 sha 各不同 ⟹ 有据可查"。那个设计的漏洞当天在实盘上炸了:
    使用者**在一个页面里连录 9 个学生**,9 次上传挤进同一场,前 8 份原生录像被
    原地覆盖,丢了 2.1 GB。**账本那两行 sha 证明不了任何字节还在** —— 它只能证明
    "曾经有过一份",而原始素材不可再生的意思正是"没有第二次机会"。

    现在的规矩:一个会话只许有一份原生录像。要再录就开新的一场(前端每次
    「开始录制」铸新号)。

    红法:把 `retain_uploaded_video` 里那个 `if target.exists(): raise` 去掉
    → 本测试红在 `pytest.raises` 没等到异常,而且第一份的字节会被第二份盖掉。
    """
    first = _write(WEBM)
    p = _isolated / "20260925_203826_2449" / "media" / "camera.webm"
    before = p.read_bytes()

    with pytest.raises(media_retention.CameraAlreadyRetained):
        _write(WEBM + b"-second-half")

    assert p.read_bytes() == before == WEBM, "第一份必须逐字节没动"
    lines = (media_retention.ledger_path("20260925_203826_2449", "camera")
             ).read_text(encoding="utf-8").splitlines()
    assert len([l for l in lines if l.strip()]) == 1, "被拒的那次不该进账本"
    assert first["bytes"] == len(WEBM)


def test_the_refusal_is_not_swallowed_as_a_degraded_write(_isolated):
    """⚠️ 拒绝必须**抛出去**,不能被下面那个 `except Exception` 吞成 degraded。

    这是本改动最容易写错的一处:守卫若放在 `try` **里面**,就会被当成"写失败",
    于是"拒绝覆盖"退化成一条静默降级 —— 正是这个守卫要消灭的形态。

    红法:把 `if target.exists(): raise ...` 挪进 `try` 块 → 本测试红在
    没等到 `CameraAlreadyRetained`,而且 `degraded_reasons()` 里会多出一条。
    """
    _write()
    with pytest.raises(media_retention.CameraAlreadyRetained):
        _write(WEBM + b"-x")
    assert not any("拒绝覆盖" in r or "camera.webm" in r
                   for r in media_retention.degraded_reasons("20260925_203826_2449"))


def test_same_bytes_uploaded_twice_is_still_refused(_isolated):
    """**逐字节相同**也要拒 —— 判据是"这个落点有主了",不是"内容变了没变"。

    红法:把守卫的判据从 `target.exists()` 改成"内容不同才拒" → 本测试红。
    """
    _write()
    with pytest.raises(media_retention.CameraAlreadyRetained):
        _write(WEBM)


def test_write_failure_is_recorded_and_does_not_raise(_isolated):
    """中途写失败:不抛、记 degraded(与帧/音频同一口径)。

    ⚠️ 三处讲究:
      ① **先成功写一件把预检过掉**。预检 `_write_probe` 自己也调 `Path.write_bytes`
         —— 直接让 `write_bytes` 全抛的话,失败落在**探针**上,而首件预检失败是
         **要大声抛 `RuntimeError`** 的(`_loud_or_silent`)。那样这个测试就在验
         另一件事,而且会以 RuntimeError 而非断言失败的形式红掉。
      ② **然后把那份文件删掉**。2026-09-28 起"本场已有 camera.webm"会被**拒绝**
         (见上),不删的话第二次写会被守卫拦下,测的就成了守卫而不是写失败。
         删掉 = 模拟"这一场还没有录像"这个合法状态。
      ③ 用 `pytest.MonkeyPatch.context()` 而**不是** `monkeypatch.undo()` ——
         后者会把 autouse fixture 设的 `JINGXIN_RECORDINGS_DIR` 一起撤掉,
         于是后面那句**真的写进 `~/shared`**。这个事故发生过(见那个文件的注释)。
    """
    assert _write() is not None                      # 先过预检
    (_isolated / "20260925_203826_2449" / "media" / "camera.webm").unlink()   # ②
    calls = {"n": 0}
    real = Path.write_bytes

    def flaky(self, data):
        calls["n"] += 1
        if calls["n"] == 1:                          # 预检已过 ⟹ 这就是本次的写
            raise OSError(28, "No space left on device")
        return real(self, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_bytes", flaky)
        assert _write() is None                      # 不抛
    assert any("No space left" in r
               for r in media_retention.degraded_reasons("20260925_203826_2449"))


def test_retention_off_writes_nothing(_isolated, monkeypatch):
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    assert _write() is None
    assert not (_isolated / "20260925_203826_2449").exists()


def test_path_traversal_is_rejected(_isolated):
    with pytest.raises(ValueError):
        media_retention.retain_uploaded_video("../escape", WEBM)
