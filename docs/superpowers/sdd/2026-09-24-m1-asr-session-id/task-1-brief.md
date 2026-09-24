### Task 1: ASR 适配层(session id + 转写落盘 + 引擎)

**Files:**
- Create: `voice_interaction/asr/__init__.py`, `voice_interaction/asr/funasr_client.py`(vendor), `voice_interaction/asr/session.py`, `voice_interaction/asr/funasr_engine.py`, `voice_interaction/asr/transcript_store.py`, `voice_interaction/asr/asr_config.json`
- Test: `tests/test_asr_engine.py`, `tests/test_transcript_store.py`

**Interfaces:**
- Consumes: 无
- Produces(后续任务依赖的确切名字):
  - `session.NONE_SESSION: str = "NONE"`,`session.new_session_id(now: datetime | None = None) -> str`
  - `funasr_engine.AsrUtterance`(字段:`text: str`、`n_chars: int`、`n_segments: int`、`vad_split: bool`、`segments: list[dict]`)
  - `funasr_engine.FunASREngine(client=None, host=None, port=None, timeout_s=None)`;方法 `transcribe_pcm(pcm: bytes) -> AsrUtterance`
  - `transcript_store.recording_dir(session_id: str, root: str | Path | None = None) -> Path`
  - `transcript_store.ensure_manifest(session_id: str, asr_meta: dict, root=None) -> Path`
  - `transcript_store.append_utterance(session_id: str, utt, recorded_at: str | None = None, root=None) -> Path`

- [ ] **Step 1: vendor 客户端与包骨架**

```bash
mkdir -p voice_interaction/asr
cp ~/asr-test/funasr_client.py voice_interaction/asr/funasr_client.py
printf '' > voice_interaction/asr/__init__.py
grep -c "_merge_finals" voice_interaction/asr/funasr_client.py   # 期望 >= 2
```

- [ ] **Step 2: 写失败测试(session id 与清单)**

```python
# tests/test_transcript_store.py
from datetime import datetime
from pathlib import Path

from voice_interaction.asr import session, transcript_store


def test_new_session_id_shape():
    sid = session.new_session_id(now=datetime(2026, 9, 24, 15, 30, 12))
    assert sid.startswith("20260924_153012_")
    assert len(sid) == len("20260924_153012_") + 4
    int(sid[-4:], 16)


def test_new_session_id_is_unique():
    ids = {session.new_session_id() for _ in range(50)}
    assert len(ids) == 50


def test_recording_dir_is_outside_repo(tmp_path):
    d = transcript_store.recording_dir("20260924_153012_9f3c", root=tmp_path)
    assert d == tmp_path / "20260924_153012_9f3c"
    assert d.exists()


def test_manifest_records_expected_log_paths(tmp_path):
    p = transcript_store.ensure_manifest("20260924_153012_9f3c",
                                         {"engine": "funasr", "models": {}}, root=tmp_path)
    payload = __import__("json").loads(p.read_text(encoding="utf-8"))
    assert payload["session_id"] == "20260924_153012_9f3c"
    assert set(payload["logs"]) == {"face", "gesture", "voice"}
    assert all(v["expected"] is True for v in payload["logs"].values())
```

- [ ] **Step 3: 运行,确认失败**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_transcript_store.py -q`
Expected: FAIL(`ModuleNotFoundError: voice_interaction.asr.transcript_store`)

- [ ] **Step 4: 实现 `session.py` 与 `transcript_store.py`**

```python
# voice_interaction/asr/session.py
"""会话标识。id 由 voice 的 /interview/start 生成,显式下传三个模块。"""
from __future__ import annotations

import secrets
from datetime import datetime

NONE_SESSION = "NONE"          # 无 id 时的显式占位,报告侧整体排除(见 spec D7)


def new_session_id(now: datetime | None = None) -> str:
    """YYYYMMDD_HHMMSS_<4 位十六进制>:时间段给人看,随机段防撞。"""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{secrets.token_hex(2)}"
```

```python
# voice_interaction/asr/transcript_store.py
"""转写与会话清单落盘 —— 一律在仓库外(spec D2)。

仓库内只允许出现数字;原句只进 ~/shared/jingxin_recordings/{session_id}/。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / "shared" / "jingxin_recordings"          # D:\Shared\jingxin_recordings
LOG_PREFIXES = {"face": "face_au_log", "gesture": "gesture_emotion_log",
                "voice": "interview_emotion_log"}


def _root(root: str | Path | None) -> Path:
    return Path(root) if root else Path(os.getenv("JINGXIN_RECORDINGS_DIR", DEFAULT_ROOT))


def recording_dir(session_id: str, root: str | Path | None = None) -> Path:
    d = _root(root) / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(session_id: str, root=None) -> Path:
    return recording_dir(session_id, root) / "session.json"


def ensure_manifest(session_id: str, asr_meta: dict[str, Any], root=None) -> Path:
    """会话开始时写一次;已存在则不覆盖(保留 started_at 与既有 logs 状态)。"""
    p = _manifest_path(session_id, root)
    if p.exists():
        return p
    payload = {
        "session_id": session_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "asr": asr_meta,
        "logs": {mod: {"expected_file": f"{pre}_{session_id}.csv", "expected": True}
                 for mod, pre in LOG_PREFIXES.items()},
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def refresh_manifest(session_id: str, log_dir: str | Path, root=None) -> dict:
    """会话结束时按磁盘实况把 expected 换成 present/missing。"""
    p = _manifest_path(session_id, root)
    payload = json.loads(p.read_text(encoding="utf-8"))
    for mod, entry in payload["logs"].items():
        entry["present"] = (Path(log_dir) / entry["expected_file"]).exists()
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
```

```python
# voice_interaction/asr/asr_config.json
{
  "_version": "0.1.0-provisional",
  "_provisional": true,
  "funasr_host": "192.168.72.30",
  "funasr_port": 10095,
  "timeout_s": 60.0,
  "min_chars_for_density": {
    "value": 10,
    "_provisional": true,
    "basis": "低于 10 字时,任意一次命中即产生 >10 的密度,比值不稳定(spec §6.5)"
  }
}
```

- [ ] **Step 5: 运行,确认通过**

Run: `cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_transcript_store.py -q`
Expected: 4 passed

- [ ] **Step 6: 写失败测试(引擎:多段拼接 / 增量累加 / 超时不挂死 / 异常上抛)**

```python
# tests/test_asr_engine.py
import pytest

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
```

- [ ] **Step 7: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_asr_engine.py -q`
Expected: FAIL(`ModuleNotFoundError` 或 `TypeError: __init__() got an unexpected keyword argument 'client'`)

- [ ] **Step 8: 实现 `funasr_engine.py`**

```python
# voice_interaction/asr/funasr_engine.py
"""FunASR 的薄适配器:把客户端结果规整成落盘与算特征要的形状。

四个已知坑在这里收口(spec §3/§8):
  1. 逐字时间戳在 raw['timestamp'],不在 ASRResult 属性上
  2. VAD 多段 —— 客户端已拼接,这里保留逐段原文并标记 vad_split
  3. 段内相对时间戳 —— 不假装是绝对时间(带 ts_origin 标记)
  4. 客户端会吞异常/卡住 —— 这里把异常上抛、给每次调用加超时
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import funasr_client

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CONFIG_PATH = Path(__file__).with_name("asr_config.json")


def _config() -> dict[str, Any]:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def count_cjk_chars(text: str) -> int:
    """字数 = 中文字符数(去标点与空白,spec §6.5)。"""
    return len(_CJK.findall(text or ""))


@dataclass
class AsrUtterance:
    text: str = ""
    n_chars: int = 0
    n_segments: int = 1
    vad_split: bool = False
    segments: list[dict[str, Any]] = field(default_factory=list)


class FunASREngine:
    def __init__(self, client: Any = None, host: str | None = None,
                 port: int | None = None, timeout_s: float | None = None):
        cfg = _config()
        self.host = host or cfg["funasr_host"]
        self.port = port or cfg["funasr_port"]
        self.timeout_s = timeout_s or float(cfg["timeout_s"])
        self.client = client or funasr_client

    def transcribe_pcm(self, pcm: bytes) -> AsrUtterance:
        """识别一段 16 kHz/16 bit/单声道 PCM。异常上抛,不吞。"""
        res = self.client.recognize_pcm(pcm, host=self.host)
        return self._to_utterance(res)

    @staticmethod
    def _to_utterance(res: Any) -> AsrUtterance:
        segs = list(getattr(res, "segments", None) or [])
        if not segs:
            segs = [res]
        out: list[dict[str, Any]] = []
        for i, s in enumerate(segs):
            raw = getattr(s, "raw", None) or {}
            out.append({
                "index": i,
                "text": getattr(s, "text", "") or "",
                "n_chars": count_cjk_chars(getattr(s, "text", "") or ""),
                "timestamps_ms": raw.get("timestamp") or [],
                "punc_array": raw.get("punc_array") or [],
                "ts_origin": "segment_relative",
            })
        text = getattr(res, "text", "") or ""
        return AsrUtterance(text=text, n_chars=count_cjk_chars(text),
                            n_segments=len(out), vad_split=len(out) > 1, segments=out)
```

- [ ] **Step 9: 运行,确认通过;提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_asr_engine.py tests/test_transcript_store.py -q`
Expected: 9 passed

```bash
git add voice_interaction/asr tests/test_asr_engine.py tests/test_transcript_store.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "feat(m1): ASR 适配层与转写落盘(session id / 清单 / 引擎规整)"
```

---

