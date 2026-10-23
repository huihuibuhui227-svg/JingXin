"""`tools/jx_check_session.py` 的「原生视频」一条:两种失败必须分开报。

为什么值得单独测:这条检查对两种失败都判红,但**补救动作完全不同** ——

  · `camera.webm` 不存在   ⟹ 录制没走完,整场重录;
  · 文件在但缺视频轨       ⟹ 音频往往是**完整的**(实测 178/185 秒),
                              只有视觉不可用 —— 说成"没录到"会让人白重录一场。

旧实现对两者一律输出「没走到结束页」。这正是本仓反复栽的形态:判词比事实更狠,
而读的人照着判词去补救,补错了对象。
"""
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "jx_check_session", ROOT / "tools/jx_check_session.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load()


def _facts(verdict: str) -> dict:
    return {"camera_verdict": verdict, "n_questions": 8}


def _rows_for_camera(ok: bool, name: str = "原生视频"):
    return [cs.Row(name, ok, "x")]


def test_missing_file_says_the_recording_did_not_finish():
    v = cs.verdict(_rows_for_camera(False), _facts("no_file"))
    assert "没走到结束页" in v
    assert "只有音轨" not in v


def test_missing_video_track_says_audio_is_usable():
    """★ 这条是本次修的东西:缺视频轨**不等于**没录到。"""
    v = cs.verdict(_rows_for_camera(False), _facts("no_video_track"))
    assert "只有音轨没有视频轨" in v
    assert "音频可用" in v
    # 不许再把这种情况归成"没走到结束页"
    assert "没走到结束页" not in v


def test_missing_audio_track_is_reported_as_its_own_thing():
    v = cs.verdict(_rows_for_camera(False), _facts("no_audio_track"))
    assert "只有视频轨没有音轨" in v
    assert "转写" in v


def test_verdict_without_a_camera_verdict_fact_does_not_crash():
    """`facts` 里没有这一项时(老调用点)不许 KeyError。"""
    v = cs.verdict(_rows_for_camera(False), {"n_questions": 8})
    assert v


def test_a_passing_camera_row_is_not_reported_as_a_hard_failure():
    v = cs.verdict(_rows_for_camera(True), _facts("ok"))
    assert "不够格" not in v


@pytest.mark.skipif(subprocess.run(["which", "ffprobe"], capture_output=True).returncode != 0,
                    reason="没有 ffprobe")
def test_a_real_audio_only_file_is_recognised_as_missing_video(tmp_path):
    """用真文件验一次:只造音轨 ⟹ 探到的 codec 列表里没有视频编码。"""
    f = tmp_path / "a.webm"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:a", "libopus", "-y", str(f)],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not f.exists():
        pytest.skip(f"ffmpeg 造不出样本:{r.stderr[:200]}")
    codes = cs._ffprobe_codecs(f)
    assert "opus" in codes
    assert not any(c in codes for c in ("vp8", "vp9", "h264")), codes
