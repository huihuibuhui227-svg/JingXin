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


def test_unmeasurable_says_cannot_judge_not_disqualified():
    """★ 量不出来 **≠** 录坏了。

    `capture_health.probe` 在没装 ffprobe / 文件读不懂时返回 `unknown`。
    若把它归进"不够格",人会去重录一场可能完全正常的素材 —— 而真实情况是
    量具不在。判词必须是「判不了」,而且**不许**出现「不够格」。
    """
    v = cs.verdict(_rows_for_camera(False), _facts("unmeasurable"))
    assert "判不了" in v
    assert "不够格" not in v
    assert "不等于录坏了" in v


def test_a_passing_camera_row_is_not_reported_as_a_hard_failure():
    v = cs.verdict(_rows_for_camera(True), _facts("ok"))
    assert "不够格" not in v


@pytest.mark.skipif(subprocess.run(["which", "ffprobe"], capture_output=True).returncode != 0,
                    reason="没有 ffprobe")
def test_a_real_audio_only_file_is_recognised_as_missing_video(tmp_path):
    """用真文件验一次:只造音轨 ⟹ 判成 `no_video_track`,并说音频可用。

    ⚠️ 这条走的是**判据那一层**(`_camera_verdict`),不是自己再调一次 ffprobe ——
    本文件此前直接调 `_ffprobe_codecs`,于是"工具判成哪一档"这件事**没被验过**,
    只验了 ffprobe 会说话。2026-10-24 把量具收进 `capture_health` 后一并改掉。
    """
    f = tmp_path / "a.webm"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:a", "libopus", "-y", str(f)],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not f.exists():
        pytest.skip(f"ffmpeg 造不出样本:{r.stderr[:200]}")
    verdict, detail = cs._camera_verdict(f)
    assert verdict == "no_video_track", (verdict, detail)
    assert "音频可用" in detail


@pytest.mark.skipif(subprocess.run(["which", "ffprobe"], capture_output=True).returncode != 0,
                    reason="没有 ffprobe")
def test_a_real_a_v_file_is_judged_ok(tmp_path):
    """反方向对照:音视频都在 ⟹ `ok`。没有这一条,上面那条可能只是"恒判坏"。"""
    f = tmp_path / "av.webm"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libvpx", "-b:v", "100k", "-c:a", "libopus", "-y", str(f)],
        capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not f.exists():
        pytest.skip(f"ffmpeg 造不出样本:{r.stderr[:200]}")
    verdict, detail = cs._camera_verdict(f)
    assert verdict == "ok", (verdict, detail)


# ── 量具只有一处 ──────────────────────────────────────────────────────────

def test_the_tool_does_not_carry_its_own_ffprobe_implementation():
    """★ 同一个问题不许有两份实现。

    这里曾有一份自己的 `_ffprobe_codecs`(判「`vp8` 与 `opus` 都在吗」),而采集端
    `capture_health` 也要判同一件事。两份实现**必然分叉** —— 换编码器时一处认、
    一处不认,而"看着都对"最难查。本仓为此栽过太多次,所以现在量具只留
    `capture_health.probe` 一份,本工具只负责把事实翻成判词。

    红法:把 `_camera_verdict` 改回自己调 ffprobe ⟹ 本条红。
    """
    import inspect

    # ⚠️ 只查 `_camera_verdict` 这一个函数体里的 `subprocess`,**不是**整个文件的字符串
    #    —— 本文件的注释里必须能写「ffprobe」这三个字(要解释为什么不再自己调它)。
    #    第一版写成 `"ffprobe" not in 全文`,于是注释里提一句就红,那是在测注释。
    body = inspect.getsource(cs._camera_verdict)
    assert "subprocess" not in body, "本工具不该再自己调 ffprobe —— 量具在 capture_health"
    assert "capture_health" in body
    assert not hasattr(cs, "_ffprobe_codecs"), "旧的自带量具必须删掉,不许留着并存"


def test_camera_verdict_goes_through_capture_health(tmp_path, monkeypatch):
    """`_camera_verdict` 必须**真的**用 `capture_health.probe` 的结果,不是自己再量。

    打桩成"只有视频没有音频" ⟹ 判词必须是 `no_audio_track`。若它还自己调
    ffprobe,桩打不上,判词会是 `unmeasurable` 或 `ok`,当场红。
    """
    import capture_health

    f = tmp_path / "x.webm"
    f.write_bytes(b"whatever")
    monkeypatch.setattr(capture_health, "probe", lambda p: {
        "status": "warn", "has_video": True, "has_audio": False,
        "codecs": ["vp8"], "findings": ["没有音轨"]})
    verdict, detail = cs._camera_verdict(f)
    assert verdict == "no_audio_track", (verdict, detail)
    assert "转写" in detail


def test_camera_verdict_passes_through_unknown_as_unmeasurable(tmp_path, monkeypatch):
    """`probe` 说 `unknown` ⟹ 本工具必须说「量不出来」,不许猜成缺轨。"""
    import capture_health

    f = tmp_path / "y.webm"
    f.write_bytes(b"whatever")
    monkeypatch.setattr(capture_health, "probe", lambda p: {
        "status": "unknown", "has_video": None, "has_audio": None,
        "codecs": [], "findings": ["没装 ffprobe"]})
    verdict, detail = cs._camera_verdict(f)
    assert verdict == "unmeasurable", (verdict, detail)
    assert "没装 ffprobe" in detail
