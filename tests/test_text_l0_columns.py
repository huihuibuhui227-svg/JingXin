# tests/test_text_l0_columns.py
"""text 三条(`transcript_raw` / `transcript_segments` / `asr_model_version`)的行为核实。

这三行是 C/B 档里**只要求核实现有行为**的那一类(它们的 `acceptance` 逐条列在
下面的测试里)。落点在仓库外,所以本文件一律把 `JINGXIN_RECORDINGS_DIR` 指到
`tmp_path` —— 测的是**产出方**(`voice_interaction/asr/transcript_store.py` 与
`funasr_engine.py`),不是使用者真实的那份数据。

三条行**都不会**因此翻 `implemented`:它们的 `basis` 已写明,`implemented` 的钉子
(方向 2)只对着 face/gesture/voice 三个 logger 的产出比,text 行匹配不上
(`l0_columns.json` 的 `schema_errors` 也会直接报错,见
`tests/test_l0_column_table.py`)⟹ 口径上它们停在 `pending`。
"""
import asyncio
import importlib
import json
import types
from pathlib import Path

import pytest

store = importlib.import_module("voice_interaction.asr.transcript_store")
engine_mod = importlib.import_module("voice_interaction.asr.funasr_engine")
client_mod = importlib.import_module("voice_interaction.asr.funasr_client")

REPO_ROOT = Path(__file__).resolve().parents[1]
SID = "20260926_120000_txt1"

# 段形状的**契约**(spec §6.4):只许有这 6 个键,引擎以后加字段也不许漏进来。
SEGMENT_KEYS = ["index", "text", "n_chars", "timestamps_ms", "punc_array", "ts_origin"]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    return tmp_path


def _utt(texts, ts_step=1000):
    """造一个"VAD 切了 N 段"的识别结果(形状与引擎产出的一致)。"""
    segs = []
    for i, t in enumerate(texts):
        segs.append({"index": i, "text": t,
                     "n_chars": engine_mod.count_cjk_chars(t),
                     "timestamps_ms": [[i * ts_step, (i + 1) * ts_step]],
                     "punc_array": [1], "ts_origin": "segment_relative"})
    return engine_mod.AsrUtterance(
        text="".join(texts), n_chars=engine_mod.count_cjk_chars("".join(texts)),
        n_segments=len(segs), vad_split=len(segs) > 1, segments=segs)


# =====================================================================
# transcript_raw ①:落点在仓库外(不进 data/)
# transcript_segments ①②:6 个键、ts_origin=段内相对
# =====================================================================
def test_transcript_lands_outside_the_repo_with_the_documented_shape(tmp_path):
    """一次识别落盘后:文件在**仓库外**、顶层 5 键、每段**恰好** 6 个契约键。

    红法:① 把 `DEFAULT_ROOT` 改成仓库内的 `data/`;② 让 `_segment_records` 原样
    透传引擎的 seg(引擎多一个字段就漏进契约)。
    """
    store.append_utterance(SID, _utt(["我 叫 王 明", "今 年 二 十 岁"]))
    p = tmp_path / SID / "transcript.json"

    assert p.exists(), f"没落到 {p}"
    assert REPO_ROOT not in p.resolve().parents, (
        f"转写落进了仓库({p})—— spec D2 要求原句只在仓库外")
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert list(doc) == ["session_id", "recorded_at", "asr", "merged", "segments"], list(doc)
    assert all(list(s) == SEGMENT_KEYS for s in doc["segments"]), doc["segments"]
    assert {s["ts_origin"] for s in doc["segments"]} == {"segment_relative"}, doc["segments"]


# =====================================================================
# transcript_raw ②:与 transcript_segments 的拼接**逐字**一致(_merge 的口径)
# =====================================================================
def test_merged_text_is_the_verbatim_join_of_the_segments(tmp_path):
    """`merged` 是**全段重算**:文本逐字相接、字数求和、段数相等。

    ⚠️ 这条**不重写** `_merge` 的公式:它读的是**写下去的那份文件**,验的是文件
    自己内部的三个数互相自洽(段数组是唯一的真相源,`merged` 只是它的一个视图)。
    红法:把 `_merge` 的 `"".join` 改成 `max(...)`(只留最长的一段)—— 逐字相等立刻红。
    """
    store.append_utterance(SID, _utt(["我 叫 王 明", "今 年 二 十 岁", "谢 谢"]))
    doc = json.loads((tmp_path / SID / "transcript.json").read_text(encoding="utf-8"))

    segs = doc["segments"]
    assert doc["merged"]["text"] == "".join(s["text"] for s in segs)
    assert doc["merged"]["n_chars"] == sum(s["n_chars"] for s in segs)
    assert doc["merged"]["n_segments"] == len(segs)
    # 三段的文字都必须真的在里面(拼接过短/只留一段都会红)
    for s in segs:
        assert s["text"] in doc["merged"]["text"], s["text"]
    assert doc["merged"]["vad_split"] is True


# =====================================================================
# transcript_segments ③:index 跨次追加连续(同一会话的第二次回答接着编号)
# =====================================================================
def test_index_continues_across_appends_without_clearing_earlier_segments(tmp_path):
    """同一会话第二次识别 ⟹ 段**追加**、`index` 接着编号、既有段一个不动。

    红法:把 `segments.extend(...)` 改回 `segments = _segment_records(utt, 0)`
    —— 第二次回答会把第一次整个盖掉,而总段数只是"少了几段",不报错。
    """
    store.append_utterance(SID, _utt(["第 一 次", "回 答 一"]))
    first = json.loads((tmp_path / SID / "transcript.json").read_text(encoding="utf-8"))
    store.append_utterance(SID, _utt(["第 二 次", "回 答 二", "结 尾"]))

    doc = json.loads((tmp_path / SID / "transcript.json").read_text(encoding="utf-8"))
    assert [s["index"] for s in doc["segments"]] == [0, 1, 2, 3, 4], doc["segments"]
    assert doc["segments"][:2] == first["segments"], "第二次回答把第一次的段改了/盖了"
    assert doc["recorded_at"] == first["recorded_at"], "recorded_at 只由**第一次**写入决定"
    assert doc["merged"]["text"] == "".join(s["text"] for s in doc["segments"])


# =====================================================================
# transcript_raw ③:VAD 把长音频切成多段时,拼接完整、不丢前半段
# =====================================================================
class _FakeWS:
    """一个只回放脚本消息的假 websocket —— 用来驱动**真的** `arecognize_pcm`。

    为什么非要在这一层造假:那个坑(「早期版本是 `final = r` 覆盖赋值,只会留下最后一段」)
    就在 `arecognize_pcm` 的**收结果循环**里,不在任何别的函数里。绕过它去测
    `_merge_finals` 或 `_to_utterance`,等于"验证跑错了对象"。
    """

    def __init__(self, messages):
        self._msgs = list(messages)
        self.sent: list = []

    async def send(self, m):
        self.sent.append(m)

    async def recv(self):
        if self._msgs:
            await asyncio.sleep(0)          # 让 `wait_for(..., 0.001)` 拿到它
            return self._msgs.pop(0)
        await asyncio.sleep(3600)           # 没消息了 ⟹ 由 wait_for 超时收场

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _vad_split_messages(parts, ts_step=1000):
    """服务端按 VAD 段发来的多条 2pass-offline 结果(最后一条带 `is_end`)。"""
    msgs = []
    for i, t in enumerate(parts):
        msgs.append(json.dumps({
            "mode": "2pass-offline", "is_final": True, "text": t,
            "timestamp": [[i * ts_step, (i + 1) * ts_step]],
            "punc_array": [1], "segment_count": len(parts),
            "is_end": i == len(parts) - 1,
        }))
    return msgs


def test_vad_split_long_audio_keeps_the_first_half(monkeypatch):
    """长音频被 VAD 切成多段 ⟹ **每一段都要在**,尤其第一段(前面那些秒数)。

    已知坑(funasr_client 的 docstring 自己记着):早期版本是 `final = r` 覆盖赋值,
    前 30 多秒的文本被静默丢掉,表现为"前面没识别进去"。
    红法:**只留最后一条 final**(把收结果那两处 `finals.append(r)` 换成 `finals = [r]`)
    ⟹ 下面三句同时红,而且红在"第一段不见了"上。
    """
    parts = ["前 半 段 的 话", "中 间 的 话", "后 半 段 的 话"]
    fake = _FakeWS(_vad_split_messages(parts))
    monkeypatch.setattr(client_mod, "websockets",
                        types.SimpleNamespace(connect=lambda *a, **k: fake))

    # `timeout` 是客户端的**收尾等待**上限:脚本消息放完就没人回了,给 0.05 s 收场
    # (默认 60 s —— 那是给真服务端留的余量,测试里等它没意义)。
    r = client_mod.recognize_pcm(b"\x00\x00" * 960 * 3, host="127.0.0.1", port=1,
                                 timeout=0.05)

    assert r.text == "".join(parts), f"拼接不完整:{r.text!r}"
    assert r.text.startswith(parts[0]), f"前半段丢了:{r.text!r}"
    assert len(r.segments) == len(parts), f"段数不对:{len(r.segments)}"


def test_vad_split_utterance_reaches_the_file_with_every_segment(tmp_path, monkeypatch):
    """同一条链走到落盘:`FunASREngine` → `transcript_store` → 仓库外那份 json。

    端到端地再问一次"第一段在不在"—— 只看中间某一层绿,证明不了这条链整体没丢东西。
    """
    parts = ["前 半 段 的 话", "后 半 段 的 话"]
    fake = _FakeWS(_vad_split_messages(parts))
    monkeypatch.setattr(client_mod, "websockets",
                        types.SimpleNamespace(connect=lambda *a, **k: fake))

    utt = engine_mod.FunASREngine(timeout_s=0.05).transcribe_pcm(b"\x00\x00" * 960 * 2)
    store.append_utterance(SID, utt)
    doc = json.loads((tmp_path / SID / "transcript.json").read_text(encoding="utf-8"))

    assert doc["merged"]["vad_split"] is True
    assert doc["merged"]["n_segments"] == len(parts)
    assert doc["segments"][0]["text"] == parts[0]
    assert doc["merged"]["text"].startswith(parts[0]), doc["merged"]["text"]
    # 时间戳是**段内相对**(ts_origin 声明了),第二段的起点接在第一段末尾之后
    assert doc["segments"][1]["timestamps_ms"][0][0] >= doc["segments"][0]["timestamps_ms"][-1][1]


# =====================================================================
# asr_model_version ①③:归因对 + 模型换版时本列跟着变、不需要改代码
# =====================================================================
def test_model_block_is_read_from_config_at_call_time(tmp_path, monkeypatch):
    """模型标识是**现取**的 —— 改配置就变,代码一行不动。

    红法:把 `models` 写死在 `_asr_meta()` 里(不再 `dict(cfg.get("models"))`)⟹
    改了配置本列纹丝不动,这条红。
    """
    cfg_path = tmp_path / "asr_config.json"
    cfg = json.loads((Path(engine_mod.__file__).with_name("asr_config.json"))
                     .read_text(encoding="utf-8"))
    cfg["models"] = dict(cfg["models"], asr_offline="test/other-model-v9")
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(engine_mod, "_CONFIG_PATH", cfg_path)

    store.append_utterance(SID, _utt(["一 句 话"]))
    doc = json.loads((tmp_path / SID / "transcript.json").read_text(encoding="utf-8"))
    assert doc["asr"]["models"]["asr_offline"] == "test/other-model-v9", doc["asr"]

    # 落点与 `transcript_raw` 同:同一个 `models` 块也进会话清单
    store.ensure_manifest(SID, store._asr_meta())
    sess = json.loads((tmp_path / SID / "session.json").read_text(encoding="utf-8"))
    assert sess["asr"]["models"] == doc["asr"]["models"], (sess["asr"], doc["asr"])


def test_model_version_and_confidence_form_the_attribution_pair(tmp_path):
    """① 归因对:一次识别结果要能追到「哪个模型 + 置信度多少」—— 四个键一起落盘。

    置信度在本部署拿不到,所以它的形状是「显式的 null + 来源说明」,**不是省略**
    (spec §9.1:缺失也要写出来,否则读的人分不清"没有"与"没记")。
    """
    store.append_utterance(SID, _utt(["一 句 话"]))
    asr = json.loads((tmp_path / SID / "transcript.json").read_text(encoding="utf-8"))["asr"]

    assert set(asr["models"]) == {"asr_online", "asr_offline", "vad", "punc"}, asr["models"]
    assert "asr_confidence" in asr and asr["asr_confidence"] is None, asr
    assert asr["asr_confidence_source"] == "unavailable", asr
    assert asr["engine"] == "funasr" and asr["endpoint"].startswith("ws://"), asr


def test_the_two_asr_meta_producers_agree(tmp_path, monkeypatch):
    """端点的 `_asr_meta` 与 store 的 `_asr_meta` 必须是**同一个形状同一个值**。

    红法:只改一处(例如给端点那份换一个 endpoint 格式)⟹ `transcript.json` 与
    `session.json` 对同一个会话给出两份 provenance,而没有任何东西会报错。
    """
    import importlib as _il

    app = _il.import_module("voice_interaction.api.app")
    a = app._asr_meta()
    b = store._asr_meta()
    b.pop("session_id", None)
    assert a == b, (a, b)
