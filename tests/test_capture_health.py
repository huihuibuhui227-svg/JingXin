"""`capture_health` —— 录完当场体检。

为什么这个模块值得单独测:它的**全部价值**在于「录坏的那一场,当场就知道」,
而它同时又必须**绝不阻塞留存**。这两件事在测试里是两套相反的断言:

  · 判据要**真的判得出**坏文件(否则它只是一段看着很专业的日志);
  · 探针出任何事都**不许**让上传失败(否则一次坏探针会丢一份不可再生的录像)。

所以下面既有「真造一个只有音轨的 webm ⟹ 判 no video」这种硬断言,
也有「ffprobe 不存在 / 文件不存在 / 清单写不进去 ⟹ 不抛异常」这种软断言。
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import capture_health as ch

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="没有 ffmpeg/ffprobe")


def _webm(tmp_path: Path, *, video: bool, audio: bool, sec: int = 2,
          dark: bool = False) -> Path:
    """真造一份 webm。`video=False` ⟹ 只有音轨(就是盘上那 2 场的形态)。"""
    out = tmp_path / "camera.webm"
    args = ["ffmpeg", "-v", "error", "-y"]
    if video:
        src = "color=c=black:s=64x64" if dark else "testsrc=size=64x64:rate=10"
        args += ["-f", "lavfi", "-i", f"{src}:duration={sec}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={sec}"]
    if video:
        args += ["-c:v", "libvpx", "-b:v", "100k"]
    if audio:
        args += ["-c:a", "libopus"]
    args.append(str(out))
    r = subprocess.run(args, capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not out.exists():
        pytest.skip(f"ffmpeg 造不出样本:{r.stderr[:200]}")
    return out


# ── 判据:真的判得出坏文件 ──────────────────────────────────────────────

@needs_ffmpeg
def test_a_good_recording_passes(tmp_path):
    r = ch.probe(_webm(tmp_path, video=True, audio=True))
    assert r["status"] == "ok", r
    assert r["has_video"] and r["has_audio"]
    assert r["video_sec"] and r["audio_sec"]
    assert 0.8 <= r["av_ratio"] <= 1.2
    assert r["findings"] == []


@needs_ffmpeg
def test_audio_only_recording_is_flagged_as_missing_video(tmp_path):
    """★ 盘上那 2 场的形态:文件在、音频完整、**就是没有视频轨**。

    旧文案把它说成「没走到结束页」会让人白重录一场 —— 这里的断言同时守住
    「判得出」与「说对了是哪种坏」。
    """
    r = ch.probe(_webm(tmp_path, video=False, audio=True))
    assert r["status"] == "warn"
    assert r["has_video"] is False
    assert r["has_audio"] is True
    assert r["audio_sec"] and r["audio_sec"] > 1
    blob = " ".join(r["findings"])
    assert "没有视频轨" in blob
    assert "音频可能完整" in blob, "要告诉人别整场重录 —— 音频是好的"
    assert "没走到结束页" not in blob, "缺轨 ≠ 没录到,别混成一句"


@needs_ffmpeg
def test_video_only_recording_is_flagged_as_missing_audio(tmp_path):
    r = ch.probe(_webm(tmp_path, video=True, audio=False))
    assert r["status"] == "warn"
    assert r["has_audio"] is False
    assert "没有音轨" in " ".join(r["findings"])


@needs_ffmpeg
def test_an_all_black_recording_is_flagged(tmp_path):
    """全黑 = 摄像头被挡/断电。亮度是**物理下限**,不是拟合出来的阈值。"""
    r = ch.probe(_webm(tmp_path, video=True, audio=True, dark=True))
    assert r["status"] == "warn"
    assert r["mean_luma"] is not None and r["mean_luma"] < ch.MIN_MEAN_LUMA
    assert "全黑" in " ".join(r["findings"])


# ── 探针坏了不许装成「录坏了」,也不许抛 ────────────────────────────────

def test_missing_file_says_so_and_does_not_raise(tmp_path):
    r = ch.probe(tmp_path / "nope.webm")
    assert r["status"] == "unknown"
    assert r["has_video"] is None, "没量成 ≠ 没有视频轨"
    assert "文件不存在" in " ".join(r["findings"])


def test_without_ffprobe_it_degrades_to_unknown(tmp_path, monkeypatch):
    """没装 ffprobe ⟹ unknown,而**不是** warn。

    这条分得很细:warn 会让人以为素材坏了;真实情况是**没做体检**。
    """
    monkeypatch.setattr(ch.shutil, "which", lambda name: None)
    r = ch.probe(tmp_path / "whatever.webm")
    assert r["status"] == "unknown"
    assert r["has_video"] is None
    assert "没做体检" in " ".join(r["findings"])


def test_a_file_ffprobe_cannot_read_is_unknown_not_warn(tmp_path):
    """ffprobe 读不懂的字节 ⟹ unknown,**不是** warn。

    这个区分很贵:"warn" 会让人去重录一场,而真实情况是**没量成**。
    ⚠️ 别写成 `if 有ffprobe: … else: …` 两个分支各断言一次 —— 那会让其中一条
    assert **永远执行不到**,而 `test_assert_coverage` 会当场把它揪出来
    (2026-10-24 实测,正是这一条先红)。没装 ffprobe 的机器由上面的
    `test_without_ffprobe_it_degrades_to_unknown` 覆盖,不必在这里分叉。
    """
    junk = tmp_path / "junk.webm"
    junk.write_bytes(b"not a video at all")
    r = ch.probe(junk)
    assert r["status"] == "unknown", r
    assert r["has_video"] is None
    assert "读不懂" in " ".join(r["findings"])


# ── 写 session.json:能写就写,写不进去也不许抛 ──────────────────────────

def test_record_writes_into_session_json(tmp_path):
    p = tmp_path / "session.json"
    p.write_text(json.dumps({"session_id": "S", "logs": {"face": {}}}), encoding="utf-8")
    assert ch.record(p, {"status": "warn", "findings": ["x"]}) is True
    got = json.loads(p.read_text(encoding="utf-8"))
    assert got["capture_health"]["status"] == "warn"
    assert got["logs"] == {"face": {}}, "体检不许把清单里别的东西弄丢"


def test_record_returns_false_when_the_manifest_is_missing(tmp_path):
    """清单还没写(会话没走 /start)⟹ 不写、不抛、返回 False。"""
    assert ch.record(tmp_path / "nope.json", {"status": "ok"}) is False
    assert not (tmp_path / "nope.json").exists()


def test_record_leaves_no_temp_files_behind(tmp_path):
    p = tmp_path / "session.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    ch.record(p, {"status": "ok"})
    assert sorted(x.name for x in tmp_path.iterdir()) == ["session.json"]


def test_summarize_never_crashes_on_a_partial_result():
    """面板/日志会拿任意一个 dict 调它 —— 缺键不许 KeyError。"""
    assert ch.summarize({})
    assert ch.summarize({"status": "ok"})
