# tests/test_capture_health_audio_only.py
"""只采声音的场次不许被判成「录坏了」。

## 为什么这个文件存在

`capture_health.probe()` 只看文件里的流:没有视频轨 ⟹ `findings` 追加
「**文件里没有视频轨** —— 视觉指标与面部/手势全不可出」⟹ `status = warn`。

而 `consent == "audio_only"`(受试者只同意音频)的场次,前端**根本不打开摄像头**
(`useCamera.startCamera` 的 audioOnly 分支取的是 `{audio: …, video: false}`)。
所以那一场**按设计**就没有视频轨 —— 它不是"录坏了",是"本来就只该有声音"。

不区分的话,每一场 audio-only 的 session.json 都会带一条假的 warn,而这条 warn
会顺着 `tools/jx_check_session.py` 的判词报给使用者 ⟹ 真实的录坏被淹掉。

## 这个文件守什么

1. `probe()` 认 `audio_only=True`:不再报「没有视频轨」那条 finding,status 也不因它变 warn;
2. **仍然报**它该报的:没有音轨、音视频不同长(后者对 audio-only 不适用)、全黑;
3. 不传这个参数时行为**一个字都不变** —— 老调用方不受影响;
4. 端到端:`/session/{sid}/retain_video` 体检时**真的读到了** session.json 里的 consent。

⚠️ 第 4 条是本仓记过多次的形态:单独测 `probe(audio_only=True)` 只证明函数本身,
证明不了**生产那条路真的传了这个参数**。
"""
from __future__ import annotations

import importlib
import json

import pytest

import capture_health as ch


def _streams(video=True, audio=True):
    out = []
    if video:
        out.append({"type": "video", "codec": "vp8"})
    if audio:
        out.append({"type": "audio", "codec": "opus"})
    return out


@pytest.fixture
def fake_probe(monkeypatch, tmp_path):
    """把 ffprobe 那两处换成假的 —— `probe()` 的其余逻辑原样跑。"""
    cam = tmp_path / "camera.webm"
    cam.write_bytes(b"\x1aE\xdf\xa3" + b"x" * 100)

    def setup(*, video=True, audio=True, video_sec=10.0, audio_sec=10.0, luma=120.0):
        monkeypatch.setattr(ch, "_ffprobe", lambda: "ffprobe")
        monkeypatch.setattr(ch, "_streams", lambda f, p: _streams(video, audio))

        def span(f, p, spec):
            return video_sec if spec.startswith("v") else audio_sec

        monkeypatch.setattr(ch, "_span_sec", span)
        monkeypatch.setattr(ch, "_mean_luma", lambda p: luma)
        return cam

    return setup


def test_audio_only_is_not_a_warning(fake_probe):
    """★ 红法:把 `audio_only` 那个分支去掉 ⟹ 本条红。

    只采声音的场次**按设计**没有视频轨。报 warn 就是拿"设计如此"当缺陷,
    而这条假 warn 会淹没真的录坏。
    """
    cam = fake_probe(video=False, audio=True, audio_sec=10.0, video_sec=None)

    r = ch.probe(cam, audio_only=True)

    assert r["status"] == ch.STATUS_OK, f"只采声音被判成 {r['status']}:{r['findings']}"
    assert not any("没有视频轨" in f for f in r["findings"]), r["findings"]
    assert r["has_video"] is False, "事实还是要如实记:没有视频轨"


def test_without_the_flag_the_old_behaviour_is_unchanged(fake_probe):
    """★ 红法:把 `audio_only` 的默认值改成 True ⟹ 本条红。

    老调用方(`tools/jx_check_session.py`)不传这个参数。默认值一变,
    它们会把"真的没录到画面"也放过。
    """
    cam = fake_probe(video=False, audio=True, audio_sec=10.0, video_sec=None)

    r = ch.probe(cam)

    assert r["status"] == ch.STATUS_WARN
    assert any("没有视频轨" in f for f in r["findings"]), r["findings"]


def test_audio_only_still_reports_a_missing_audio_track(fake_probe):
    """只采声音时,「没有音轨」是**真的**缺陷 —— 这一场什么都没录到。

    红法:把 audio_only 写成"一律 status=ok、findings 清空" ⟹ 本条红。

    ⚠️ 不断言具体是 `warn` 还是 `unknown`:一条流都没有时 `probe` 走的是
    「ffprobe 读不出流」那条**早退**,状态是 `unknown`(既有行为,不是本次改动)。
    本测试要守的是**它没被判成 ok**。
    """
    cam = fake_probe(video=False, audio=False)

    r = ch.probe(cam, audio_only=True)

    assert r["status"] != ch.STATUS_OK, f"一条轨都没有却判 ok:{r}"
    assert r["findings"], "一条轨都没有却一条 finding 都没有"


def test_audio_only_does_not_report_av_ratio_mismatch(fake_probe):
    """audio-only 下没有视频时长可比 —— 不许冒出「音视频不同长」。

    红法:把 av_ratio 那段的 `video_sec and audio_sec` 改成只看 `audio_sec` ⟹ 本条红。
    """
    cam = fake_probe(video=False, audio=True, audio_sec=10.0, video_sec=None)

    r = ch.probe(cam, audio_only=True)

    assert r["av_ratio"] is None
    assert not any("时长对不上" in f for f in r["findings"]), r["findings"]


# ── 端到端:生产那条路真的把 consent 读出来了 ───────────────────────────────

def test_retain_video_passes_the_consent_from_session_json(tmp_path, monkeypatch):
    """★ 红法:把 `_session_is_audio_only(sid)` 那一行删掉 ⟹ 本条红。

    只测 `probe(audio_only=True)` 证明不了**生产那条路真的传了这个参数** ——
    这正是本仓反复栽的「验证跑错了对象」。
    """
    import media_retention as mr

    root = tmp_path / "recordings"
    root.mkdir()
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(root))
    mr._reset_for_tests()

    sid = "20260928_212130_5b10"
    d = root / f"18-赵六-2021004-物理学院__{sid}"
    d.mkdir()
    (d / "session.json").write_text(
        json.dumps({"session_id": sid, "consent": "audio_only"}), encoding="utf-8")

    app_mod = importlib.import_module("voice_interaction.api.app")
    got = app_mod._session_is_audio_only(sid)

    assert got is True, (
        "体检那条路没从 session.json 读出 consent —— 只采声音的场次会被误报成「录坏了」")


# ── 端到端(走真端点):参数**真的**传进 probe 了吗 ───────────────────────────

class _FakeUpload:
    """替身 UploadFile —— 端点分块读,必须带游标,否则读循环拿不到 EOF。"""

    def __init__(self, data, name="camera.webm"):
        self._data, self.filename, self.content_type = data, name, "video/webm"
        self._pos = 0

    async def read(self, n=-1):
        if n is None or n < 0:
            n = len(self._data) - self._pos
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk


def _post_media(app_mod, sid, payload):
    import asyncio
    return asyncio.run(app_mod.submit_session_media(
        session_id=sid, file=_FakeUpload(payload)))


def test_the_endpoint_really_passes_audio_only_to_probe(tmp_path, monkeypatch):
    """★ 红法:把调用点的 `audio_only=_session_is_audio_only(sid)` 删掉 ⟹ 本条红。

    上一条只证明 `_session_is_audio_only` 这个**函数**会读 consent;这一条证明
    **端点那条路真的把它传给了 probe**。两者缺一,修法就只是摆设 ——
    本仓记过多次的「验证跑错了对象」正是只测了前者。
    """
    import media_retention as mr

    root = tmp_path / "recordings"
    root.mkdir()
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(root))
    monkeypatch.setenv("JINGXIN_CAPTURE_PROBE", "1")
    mr._reset_for_tests()

    app_mod = importlib.import_module("voice_interaction.api.app")

    seen = []
    monkeypatch.setattr(ch, "probe",
                        lambda path, **kw: seen.append(kw) or {"status": "ok"})
    monkeypatch.setattr(ch, "record", lambda p, h: True)

    # ① consent=audio_only 的场次 ⟹ probe 必须收到 audio_only=True
    sid = "20260928_212130_5b10"
    d = root / sid
    (d / "media").mkdir(parents=True)
    (d / "session.json").write_text(
        json.dumps({"session_id": sid, "consent": "audio_only"}), encoding="utf-8")
    _post_media(app_mod, sid, b"\x1a\x45\xdf\xa3" + b"x" * 64)

    assert seen, "端点根本没跑体检(JINGXIN_CAPTURE_PROBE 没生效?)"
    assert seen[-1].get("audio_only") is True, (
        f"端点没把 audio_only 传给 probe,实收 {seen[-1]!r} —— "
        "只采声音的场次会照旧被记一条假的「没有视频轨」")

    # ② 对照组:全同意的场次必须传 False,否则「真的没录到画面」会被放过
    sid2 = "20260928_212131_5b10"
    d2 = root / sid2
    (d2 / "media").mkdir(parents=True)
    (d2 / "session.json").write_text(
        json.dumps({"session_id": sid2, "consent": "full"}), encoding="utf-8")
    _post_media(app_mod, sid2, b"\x1a\x45\xdf\xa3" + b"y" * 64)

    assert seen[-1].get("audio_only") is False, (
        f"全同意的场次却传了 {seen[-1]!r} —— 真的没录到画面会被放过")


def test_session_is_audio_only_is_false_for_full_consent(tmp_path, monkeypatch):
    """对照组:全同意(或没征询)的场次不许被当成 audio-only。

    红法:把 `_session_is_audio_only` 写成 `return True` 或读不到就 True ⟹ 本条红。
    那样**真的没录到画面**的场次会被放过。
    """
    import media_retention as mr

    root = tmp_path / "recordings"
    root.mkdir()
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(root))
    mr._reset_for_tests()

    app_mod = importlib.import_module("voice_interaction.api.app")

    for consent, expect in (("full", False), (None, False), ("audio_only", True)):
        sid = f"20260928_21213{consent and 1 or 0}_5b10"
        d = root / f"18-赵六-2021004-物理学院__{sid}"
        d.mkdir(exist_ok=True)
        payload = {"session_id": sid}
        if consent is not None:
            payload["consent"] = consent
        (d / "session.json").write_text(json.dumps(payload), encoding="utf-8")
        assert app_mod._session_is_audio_only(sid) is expect, f"consent={consent!r} 判错了"


def test_session_is_audio_only_is_false_when_there_is_no_session_json(tmp_path, monkeypatch):
    """读不到 session.json ⟹ **不许猜成 audio-only**。

    猜成 audio-only 会把「真的没录到画面」放过 —— 那是这个修法最坏的方向。
    红法:把 `except` 分支的 `return False` 改成 `True` ⟹ 本条红。
    """
    import media_retention as mr

    root = tmp_path / "recordings"
    root.mkdir()
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(root))
    mr._reset_for_tests()

    app_mod = importlib.import_module("voice_interaction.api.app")
    assert app_mod._session_is_audio_only("20260928_212130_5b10") is False
