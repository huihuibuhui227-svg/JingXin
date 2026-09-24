import json

import pytest

from voice_interaction.asr import funasr_engine
from voice_interaction.asr.funasr_engine import FunASREngine


class _Res:
    def __init__(self, text, final=True, raw=None, segments=None):
        self.text, self.is_final, self.raw = text, final, raw or {}
        self.segments = segments or []


class FakeClient:
    """记录调用并返回预设结果;不碰网络。"""
    def __init__(self, result=None, exc=None, delay=None):
        self.result, self.exc, self.delay, self.calls = result, exc, delay, []

    def recognize_pcm(self, pcm, **kw):
        self.calls.append((len(pcm), kw))
        if self.exc:
            raise self.exc
        return self.result


def test_uses_raw_timestamp_not_attribute():
    """时间戳在 raw 里(实测),不在 ASRResult 属性上 —— 取错就全空。"""
    raw = {"timestamp": [[210, 450], [450, 690]], "punc_array": [1, 2]}
    eng = FunASREngine(client=FakeClient(_Res("欢迎", raw=raw)))
    utt = eng.transcribe_pcm(b"\x00" * 3200)
    assert utt.segments[0]["timestamps_ms"] == [[210, 450], [450, 690]]
    assert utt.segments[0]["punc_array"] == [1, 2]


def test_multi_segment_merge_is_kept():
    """VAD 多段必须全留(回归 _merge_finals 的存在意义)。"""
    segs = [_Res("前半段"), _Res("后半段")]
    main = _Res("前半段后半段", segments=segs, raw={"timestamp": [[0, 100], [200, 300]]})
    eng = FunASREngine(client=FakeClient(main))
    utt = eng.transcribe_pcm(b"\x00" * 3200)
    assert utt.n_segments == 2
    assert utt.vad_split is True
    assert [s["text"] for s in utt.segments] == ["前半段", "后半段"]


def test_single_segment_is_not_marked_split():
    eng = FunASREngine(client=FakeClient(_Res("欢迎", raw={"timestamp": [[210, 450]]})))
    utt = eng.transcribe_pcm(b"\x00" * 3200)
    assert utt.n_segments == 1 and utt.vad_split is False


def test_engine_error_propagates_not_swallowed():
    eng = FunASREngine(client=FakeClient(exc=RuntimeError("服务不可达")))
    with pytest.raises(RuntimeError, match="服务不可达"):
        eng.transcribe_pcm(b"\x00" * 3200)


def test_empty_text_is_returned_as_empty_not_zero():
    eng = FunASREngine(client=FakeClient(_Res("")))
    utt = eng.transcribe_pcm(b"\x00" * 3200)
    assert utt.text == "" and utt.n_chars == 0


def test_configured_host_port_timeout_reach_the_client():
    """构造函数里的 port/timeout 必须真传给客户端(vendored 客户端接受 host/port/timeout):
    只传 host 的话这两个参数就是死参,超时也只是摆设。"""
    fc = FakeClient(_Res("欢迎", raw={"timestamp": [[0, 10]]}))
    eng = FunASREngine(client=fc, host="10.0.0.9", port=12345, timeout_s=7.5)
    eng.transcribe_pcm(b"\x00" * 3200)
    _, kw = fc.calls[0]
    assert kw["host"] == "10.0.0.9"
    assert kw["port"] == 12345
    assert kw["timeout"] == 7.5


def test_defaults_are_read_from_asr_config_json(tmp_path, monkeypatch):
    """地址/端口/超时只能来自 asr_config.json,不能是代码里的裸常量。"""
    p = tmp_path / "asr_config.json"
    p.write_text(json.dumps({"funasr_host": "10.1.2.3", "funasr_port": 4321,
                             "timeout_s": 1.5}), encoding="utf-8")
    monkeypatch.setattr(funasr_engine, "_CONFIG_PATH", p)
    eng = FunASREngine(client=FakeClient(_Res("")))
    assert (eng.host, eng.port, eng.timeout_s) == ("10.1.2.3", 4321, 1.5)


def test_against_vendored_asrresult_and_merge_finals_no_network():
    """用 vendored 客户端自己的 ASRResult/_merge_finals 造结果(不走网络),
    钉住真实数据形状:合并后每个子段的 raw['timestamp'] 仍是【段内相对】值,
    而拼接链只在合并结果自己的 raw 里 —— 读错层就全错位。"""
    from voice_interaction.asr import funasr_client as fc

    a = fc.ASRResult(text="前半段", is_final=True, mode="2pass-offline",
                     raw={"timestamp": [[0, 100], [100, 200]], "punc_array": [1]})
    b = fc.ASRResult(text="后半段", is_final=True, mode="2pass-offline",
                     raw={"timestamp": [[0, 50]], "punc_array": [2]})
    res = fc._merge_finals([a, b])
    assert res.raw["timestamp"] == [[0, 100], [100, 200], [200, 250]]   # 拼接链,在合并结果 raw 上

    eng = FunASREngine(client=FakeClient(res))
    utt = eng.transcribe_pcm(b"\x00" * 3200)
    assert utt.text == "前半段后半段" and utt.n_chars == 6
    assert utt.n_segments == 2 and utt.vad_split is True
    assert utt.segments[0]["timestamps_ms"] == [[0, 100], [100, 200]]
    assert utt.segments[1]["timestamps_ms"] == [[0, 50]]                # 段内相对,未被累加
    assert [s["ts_origin"] for s in utt.segments] == ["segment_relative"] * 2


def test_count_cjk_chars_ignores_punctuation_and_ascii():
    """字数只算中文字符(去标点/空白/拉丁),否则密度会被标点稀释(spec §6.5)。"""
    assert funasr_engine.count_cjk_chars("欢迎,你好!abc 123") == 4
    assert funasr_engine.count_cjk_chars("") == 0
