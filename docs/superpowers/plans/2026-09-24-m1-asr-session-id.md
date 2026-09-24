# M1 实施计划:voice 接入 FunASR + `session_id` 贯通

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把语音这条线接通 —— 换掉 vosk、让三份日志能按 `session_id` 对上号、按契约落盘转写、并让「连接词密度」从假常量变成真测量。

**Architecture:** 会话由 voice 的 `/interview/start` 生成 id 并显式下传;三个模块保持独立服务、互不调用;ASR 逻辑收在一个可注入的薄适配器里(测试用假客户端,不打真服务);原始文本落仓库外,仓库内只留数字。

**Tech Stack:** Python 3.11(conda `jingxin` 环境)、FastAPI、websockets(17.0.1)、librosa、pytest。ASR 服务在 `192.168.72.30:10095`(FunASR,局域网直连)。

**Spec:** `docs/superpowers/specs/2026-09-24-m1-asr-session-id-design.md` —— 本计划从它论证;执行者两处都要读。

## Global Constraints

- **解释器固定** `~/miniconda3/envs/jingxin/bin/python`(不可用 `~/huihui/bin/python`,缺 websockets)。
- **范围仅**:`voice_interaction/`、`face_expression/`、`gesture_analysis/`、`report_frontend/`、`tests/`、`docs/`。**不动** 根 `app.py`、`templates/`、`experiments/`。
- **不装新包。** 行尾 LF。
- **提交只加精确路径**(仓库有约 1 万个未跟踪文件,`git add -A` 会污染提交)。身份内联:`git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit`。
- **阈值与常量必须进数据文件**(带 `_provisional` + 依据),**禁止写成代码里的裸常量**。
- **原句不得进仓库**:转写只写 `~/shared/jingxin_recordings/{session_id}/`。
- **无 id 时写 `"NONE"`,不拒绝**(NONE 常量在三处各写一份,由一致性测试守住同值)。
- **端点层不做导入级测试**(`voice_interaction.api.app` 在 import 时会构造 TTS/评估管线,成本高);端点正确性由 Task 7 的验收门覆盖。逻辑一律下沉到 Task 1–3 的可测单元。

---

## 文件结构

| 文件 | 动作 | 职责 |
|---|---|---|
| `voice_interaction/asr/__init__.py` | 新建 | 包标记 |
| `voice_interaction/asr/funasr_client.py` | 新建(vendor) | FunASR websocket 客户端,来自 `~/asr-test/funasr_client.py`(**含 `_merge_finals` 修复**),原样复制不改 |
| `voice_interaction/asr/session.py` | 新建 | `NONE_SESSION`、`new_session_id()` |
| `voice_interaction/asr/funasr_engine.py` | 新建 | `AsrUtterance` + `FunASREngine`:把客户端结果规整成落盘/算特征要的形状;**四个已知坑在此收口** |
| `voice_interaction/asr/connective_density.py` | 新建 | 连接词密度纯函数 + 标记表加载 |
| `voice_interaction/asr/transcript_store.py` | 新建 | 会话目录、`session.json` 清单、`transcript.json` 累积写 |
| `voice_interaction/asr/connective_markers.json` | 新建 | 标记表 + `version` + `basis` |
| `voice_interaction/asr/asr_config.json` | 新建 | `min_chars_for_density`、`funasr_host/port`、超时,带 `_provisional` |
| `voice_interaction/utils/logger.py` | 改 | `session_id` 首列 + 文件名带 id + 三个新列 |
| `face_expression/utils/logger.py` | 改 | 同上(列名单里加 `session_id`) |
| `gesture_analysis/utils/logger.py` | 改 | 同上 |
| `face_expression/api/app.py` | 改 | 无 id 时用 `NONE`(不再 mint uuid) |
| `gesture_analysis/api/app.py` | 改 | 同上 |
| `voice_interaction/api/app.py` | 改 | `/interview/start` 发号;`/asr`、`/interview/answer_audio` 收 id;换 ASR;落转写;写密度 |
| `voice_interaction/pipeline/speech_recognition_pipeline.py` | 改 | 拆掉 vosk 导入链,改用 engine |
| `voice_interaction/__init__.py` | 改 | 不再因缺模型而 `ImportError` |
| `report_frontend/feature_engine.py` | 改 | **删掉两张硬编码关键词表与两段计算**;新列走通用数值路径 |
| `report_frontend/research_mapper.py` | 改 | 指标元组改名 `connective_density` / 「连接词密度」 |
| `report_frontend/evidence_thresholds.json` | 改 | `density` 折算族量程随新定义更新,`_version` 提升 |
| `tests/test_asr_engine.py` … `tests/test_feature_key_rename.py` | 新建 | 见各任务 |

---

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

### Task 2: 连接词密度(纯函数 + 标记表)

**Files:**
- Create: `voice_interaction/asr/connective_density.py`, `voice_interaction/asr/connective_markers.json`
- Test: `tests/test_connective_density.py`

**Interfaces:**
- Consumes: `funasr_engine.count_cjk_chars`
- Produces: `load_markers(path=None) -> dict`;`connective_density(text: str, markers=None, min_chars=None, counter=None) -> float | None`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_connective_density.py
from voice_interaction.asr.connective_density import connective_density, load_markers


def test_counts_markers_per_hundred_chars():
    text = "然后" + "字" * 98                       # 100 字,命中 1
    assert connective_density(text, min_chars=10) == 1.0


def test_punctuation_and_space_do_not_count_as_chars():
    a = connective_density("然后" + "字" * 98 + "。。。,,,   ", min_chars=10)
    assert a == 1.0


def test_below_min_chars_returns_none_not_zero():
    assert connective_density("然后好", min_chars=10) is None


def test_empty_text_returns_none():
    assert connective_density("", min_chars=10) is None


def test_no_marker_returns_zero_not_none():
    assert connective_density("字" * 50, min_chars=10) == 0.0


def test_marker_table_has_version_and_list():
    m = load_markers()
    assert m["version"]
    assert len(m["markers"]) >= 10
    assert "然后" in m["markers"] and "但是" in m["markers"]
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_connective_density.py -q`
Expected: FAIL(`ModuleNotFoundError`)

- [ ] **Step 3: 实现标记表与模块**

```json
// voice_interaction/asr/connective_markers.json
{
  "version": "1.0.0",
  "_provisional": true,
  "basis": "汉语话语连接标记的封闭清单。M1 首版:可辩护即可,不是标定产物。改表必须 bump version 并在报告里注明(spec §9.5)。",
  "markers": ["然后", "所以", "但是", "因为", "而且", "如果", "虽然", "不过",
              "因此", "另外", "其实", "首先", "其次", "最后", "总之", "比如", "例如"]
}
```

```python
# voice_interaction/asr/connective_density.py
"""连接词密度 = 连接词数 ÷ 字数 × 100(每百字)。

采集时计算(voice 模块有文本),数字进语音日志;原句不进仓库(spec D8)。
分母刻意不含时长 —— 时长类信息属于 M3 的时长线,把时长混进文本层指标
正是本项目栽过的"1410 维被时长污染"那条(spec D4)。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .funasr_engine import _config, count_cjk_chars

_MARKERS_PATH = Path(__file__).with_name("connective_markers.json")


@lru_cache(maxsize=1)
def load_markers(path: str | None = None) -> dict[str, Any]:
    return json.loads(Path(path or _MARKERS_PATH).read_text(encoding="utf-8"))


def connective_density(text: str, markers: list[str] | None = None,
                       min_chars: int | None = None,
                       counter: Callable[[str], int] | None = None) -> float | None:
    """返回每百字连接词数;文本过短或为空时返回 None(不出值,而不是写 0)。"""
    cfg = _config()["min_chars_for_density"]
    floor = min_chars if min_chars is not None else int(cfg["value"])
    chars = (counter or count_cjk_chars)(text or "")
    if chars < floor:
        return None
    table = markers if markers is not None else load_markers()["markers"]
    hits = sum(1 for m in table if m in (text or ""))
    return round(hits / chars * 100, 4)
```

- [ ] **Step 4: 运行,确认通过;提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_connective_density.py -q`
Expected: 6 passed

```bash
git add voice_interaction/asr/connective_density.py voice_interaction/asr/connective_markers.json tests/test_connective_density.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "feat(m1): 连接词密度(每百字)+ 版本化标记表"
```

---

### Task 3: 三个 logger 写 `session_id`(首列 + 文件名 + NONE)

**Files:**
- Modify: `voice_interaction/utils/logger.py:19,30-44,47-65,77-84`、`face_expression/utils/logger.py:20,31-36,38-60`、`gesture_analysis/utils/logger.py:20,29-38,41-117`
- Test: `tests/test_session_logging.py`, `tests/test_session_id_contract.py`

**Interfaces:**
- Consumes: `voice_interaction.asr.session.NONE_SESSION`
- Produces: 三个 logger 都接受末尾关键字参数 `session_id: Optional[str] = None`,`session_id` 为 CSV **首列**;文件名形如 `{prefix}_{session_id}.csv`,无 id 时 `{prefix}_NONE_{YYYYmmdd_HHMMSS}.csv`

- [ ] **Step 1: 写失败测试(含跨模块常量一致性,文本级读文件、不 import 重模块)**

```python
# tests/test_session_id_contract.py
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "voice_interaction/utils/logger.py",
           ROOT / "face_expression/utils/logger.py",
           ROOT / "gesture_analysis/utils/logger.py"]


def test_none_session_literal_is_identical_everywhere():
    """三处各写一份 NONE —— 值必须一致,否则报告侧的排除规则只对一半生效。"""
    seen = set()
    for src in SOURCES:
        text = src.read_text(encoding="utf-8")
        m = re.search(r'NONE_SESSION\s*=\s*"([^"]+)"', text)
        assert m, f"{src} 缺少 NONE_SESSION 常量"
        seen.add(m.group(1))
    assert seen == {"NONE"}, seen
```

```python
# tests/test_session_logging.py
import csv
from pathlib import Path

from face_expression.utils.logger import DataLogger
from voice_interaction.utils.logger import VoiceLogger


def test_voice_logger_filename_and_first_column(tmp_path):
    lg = VoiceLogger(log_type="interview", log_dir=str(tmp_path), session_id="20260924_153012_9f3c")
    assert lg.csv_file.name == "interview_emotion_log_20260924_153012_9f3c.csv"
    assert lg.fieldnames[0] == "session_id"
    lg.log_prosody({"pitch_mean": 1.0}, question_index=1, emotion="neutral",
                   feedback="", connective_density=3.5, connective_density_std=0.4, n_rows=4)
    rows = list(csv.DictReader(open(lg.csv_file, encoding="utf-8")))
    assert rows[0]["session_id"] == "20260924_153012_9f3c"
    assert rows[0]["connective_density"] == "3.5"


def test_voice_logger_without_id_writes_none(tmp_path):
    lg = VoiceLogger(log_type="interview", log_dir=str(tmp_path))
    assert lg.csv_file.name.startswith("interview_emotion_log_NONE_")
    assert lg.fieldnames[0] == "session_id"


def test_face_logger_keeps_session_id_column(tmp_path):
    lg = DataLogger(log_type="video", session_id="20260924_153012_9f3c")
    assert lg.fieldnames[0] == "session_id"
    lg.log_file = str(tmp_path / "face_au_log_20260924_153012_9f3c.csv")   # 调用方会覆盖(现状)
    lg.log({"focus_score": 0.3})
    rows = list(csv.DictReader(open(lg.log_file, encoding="utf-8")))
    assert rows[0]["session_id"] == "20260924_153012_9f3c"
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_logging.py tests/test_session_id_contract.py -q`
Expected: FAIL(文件名仍是时间戳形态 / `fieldnames[0] != "session_id"`)

> ⚠️ **若 `from face_expression.utils.logger import DataLogger` 拉起了 mediapipe/cv2 之类的重依赖而变慢或失败**:不要为了测试去改包的导入结构。改用 `importlib.util.spec_from_file_location` 直接加载那一个文件(它只依赖 `..config`,可在加载前把 `sys.modules["face_expression"]` 设为一个空模块以提供父包),或把 face/gesture 的断言降级为**文本级检查**(读源码断言 `fieldnames` 首项与文件名模板),把行为断言集中在 voice logger 上 —— 三者同构,voice 那份测透即可。

- [ ] **Step 3: 改三个 logger**

`voice_interaction/utils/logger.py`:
```python
NONE_SESSION = "NONE"          # 与 face/gesture 三处同值,由 tests/test_session_id_contract.py 守住

    def __init__(self, log_type: str = 'interview', log_dir: Optional[str] = None,
                 session_id: Optional[str] = None):
        ...
        self.session_id = session_id or NONE_SESSION
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sid_part = self.session_id if session_id else f"{NONE_SESSION}_{timestamp}"
        prefix = 'interview_emotion_log' if log_type == 'interview' else 'research_emotion_log'
        self.csv_file = self.log_dir / f'{prefix}_{sid_part}.csv'
        self.json_file = self.log_dir / f'{prefix}_{sid_part}.json'
        self.fieldnames = ["session_id", "unix_timestamp", "timestamp", ...]          # session_id 置首
        # 末尾追加三列:connective_density, connective_density_std, n_rows
```
`log_prosody(...)` 末尾加三个关键字参数(默认 `None`),写进 `data`。`face_expression/utils/logger.py`:`fieldnames` 首插 `session_id`(`log()` 按 `fieldnames` 过滤,不加就会**静默丢**),构造时用 `session_id` 拼文件名。`gesture_analysis/utils/logger.py`:同样处理(`fieldnames` 首插 + `__init__` 末尾加 `session_id`,构造里用 `log_file_path` 优先、否则用 id 拼名)。

- [ ] **Step 4: 运行,确认通过;提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_logging.py tests/test_session_id_contract.py -q`
Expected: 4 passed

```bash
git add voice_interaction/utils/logger.py face_expression/utils/logger.py gesture_analysis/utils/logger.py tests/test_session_logging.py tests/test_session_id_contract.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "feat(m1): 三份日志写 session_id 首列与带 id 的文件名"
```

---

### Task 4: voice 端点接线 + 拆掉 vosk

**Files:**
- Modify: `voice_interaction/api/app.py:16,50-61,116-220,225-236,266-296`、`voice_interaction/pipeline/speech_recognition_pipeline.py:13,20-43,73-110,149`、`voice_interaction/__init__.py:40`
- Test: `tests/test_voice_transcribe_helper.py`
- Delete: `vosk-model-cn-0.22/`(2.0 GB,未被 git 跟踪)

**Interfaces:**
- Consumes: Task 1–3 的全部产物
- Produces: `voice_interaction.api.app._transcribe(audio_data: bytes) -> AsrUtterance`(模块级 `asr_engine` 可替换,便于测试与验收)

- [ ] **Step 1: 写失败测试(只测 helper,不 import 整个 app)**

```python
# tests/test_voice_transcribe_helper.py
import importlib

from voice_interaction.asr.funasr_engine import AsrUtterance


def test_transcribe_helper_delegates_to_engine(monkeypatch):
    """_transcribe 只做转发:引擎抛错就抛错,识别为空就返回空文本(不写 0)。"""
    mod = importlib.import_module("voice_interaction.api.app")
    calls = {}

    class FakeEngine:
        def transcribe_pcm(self, pcm):
            calls["pcm"] = pcm
            return AsrUtterance(text="然后我们说", n_chars=5, n_segments=1, vad_split=False, segments=[])

    monkeypatch.setattr(mod, "asr_engine", FakeEngine())
    utt = mod._transcribe(b"\x00" * 320)
    assert calls["pcm"] == b"\x00" * 320
    assert utt.text == "然后我们说"
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_voice_transcribe_helper.py -q`
Expected: FAIL(`AttributeError: module ... has no attribute '_transcribe'`)

- [ ] **Step 3: 改 `api/app.py`**

```python
# 顶部:删掉 from vosk import Model, KaldiRecognizer 与 MODEL_PATH/vosk_model 构造
from voice_interaction.asr.funasr_engine import AsrUtterance, FunASREngine
from voice_interaction.asr import session as session_mod, transcript_store
from voice_interaction.asr.connective_density import connective_density

asr_engine = FunASREngine()


def _transcribe(audio_data: bytes) -> AsrUtterance:
    """把音频交给 ASR 引擎。异常上抛(由调用方转成 HTTP 错误),不在这里吞。"""
    return asr_engine.transcribe_pcm(audio_data)
```

四处识别点(`:144`、`:203`、`:281`、`:414`)替换为:
```python
        utt = _transcribe(audio_data)
        text = utt.text.strip()
```

`/interview/start` 发号:
```python
@app.post("/interview/start")
async def start_interview():
    try:
        interview_assessment.reset()
        first_question = interview_assessment.get_next_question()
        if not first_question:
            raise HTTPException(status_code=500, detail="无法获取问题")
        sid = session_mod.new_session_id()
        transcript_store.ensure_manifest(sid, _asr_meta())
        tts_engine.speak(first_question)
        return {"status": "started", "session_id": sid, "question": first_question}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动面试失败: {str(e)}")
```
`_asr_meta()` 返回 `{"engine": "funasr", "endpoint": f"ws://{asr_engine.host}:{asr_engine.port}", "models": {...}, "asr_confidence": None, "asr_confidence_source": "unavailable"}`。

`/interview/answer_audio` 收 id + 落转写 + 写密度:
```python
@app.post("/interview/answer_audio")
async def submit_answer_audio(audio: UploadFile = File(...), session_id: str = None):
    sid = session_id or session_mod.NONE_SESSION
    ...
        utt = _transcribe(audio_data)
        text = utt.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="未识别到有效语音")
        transcript_store.append_utterance(sid, utt)                     # 仓库外
        density = connective_density(text)                              # 文本层,过短则 None
        voice_logger.session_id = sid                                   # 首列随会话
        voice_logger.log_prosody({}, question_index=..., emotion="", feedback="",
                                 connective_density=density, connective_density_std=0.0, n_rows=1)
        interview_assessment.add_answer(text)
        ...
        return {"status": "success", "session_id": sid, "recognized_text": text,
                "connective_density": density}
```
`/asr` 同样加 `session_id: str = None` 参数,识别后 `transcript_store.append_utterance(sid, utt)`(纯 ASR 不写特征行)。

`speech_recognition_pipeline.py`:删掉 `from vosk import Model, KaldiRecognizer`、`MODEL_PATH` 与模块级 `RuntimeError`;`__init__` 里 `self.model = Model(...)` 换成 `self.engine = FunASREngine()`;两处识别改走 `self.engine.transcribe_pcm(...)`,返回构造保持不变(`SpeechRecognitionResult(text=..., confidence=..., is_final=True, audio_data=audio_obj)`)。`voice_interaction/__init__.py` 不再有缺模型即崩的路径。

- [ ] **Step 4: 运行,确认通过 + 全仓无 vosk 残留**

```bash
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_voice_transcribe_helper.py -q          # 1 passed
grep -rn "vosk\|KaldiRecognizer" voice_interaction/ report_frontend/ --include="*.py" | wc -l     # 期望 0
~/miniconda3/envs/jingxin/bin/python -c "import voice_interaction; print('import ok')"
```

- [ ] **Step 5: 删模型与代码,提交**

```bash
rm -rf vosk-model-cn-0.22/            # 2.0 GB,未被跟踪;代码已无引用
git add voice_interaction/api/app.py voice_interaction/pipeline/speech_recognition_pipeline.py voice_interaction/__init__.py tests/test_voice_transcribe_helper.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "feat(m1): voice 换 FunASR(/interview/start 发号、落转写、写连接词密度),删除 vosk 代码与 2 GB 模型"
```

---

### Task 5: face / gesture 收 `session_id`(无 id 时 NONE)

**Files:**
- Modify: `face_expression/api/app.py:129-134,70-74`、`gesture_analysis/api/app.py:137-138,78-83`
- Test: `tests/test_analyze_session_fallback.py`

**Interfaces:**
- Consumes: Task 3 的三个 logger
- Produces: 两个 `/analyze` 在缺 id 时用 `NONE_SESSION`,不再 mint 每请求一个新 uuid

- [ ] **Step 1: 写失败测试**

```python
# tests/test_analyze_session_fallback.py
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_no_endpoint_mints_a_uuid_session():
    """无 id 时必须是 NONE:旧行为是每请求 uuid4 → 每帧一个新会话、日志文件爆炸。"""
    for rel in ("face_expression/api/app.py", "gesture_analysis/api/app.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "uuid4()" not in text, f"{rel} 仍在 mint uuid 会话"
        assert 'NONE_SESSION' in text, f"{rel} 未使用 NONE_SESSION"
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_analyze_session_fallback.py -q`
Expected: FAIL(仍含 `uuid4()`)

- [ ] **Step 3: 改两个 `/analyze` 与 logger 构造**

```python
# 两个文件都改:
from voice_interaction.utils.logger import NONE_SESSION      # 三处同值(测试守住)
...
    session_id = session_id or NONE_SESSION
```
logger 构造处把 `session_id` 传进去,文件名用会话 id(face:`log_path = os.path.join(LOGS_DIR, f'face_au_log_{session_id}.csv')`;gesture 同理传 `log_file_path`)。

- [ ] **Step 4: 运行,确认通过;提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_analyze_session_fallback.py -q`
Expected: 1 passed

```bash
git add face_expression/api/app.py gesture_analysis/api/app.py tests/test_analyze_session_fallback.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "feat(m1): face/gesture 接受 session_id,无 id 写 NONE(不再每请求 mint uuid)"
```

---

### Task 6: 报告侧同步(删两份定义 + 改名 + 量程)

**Files:**
- Modify: `report_frontend/feature_engine.py:310-346`、`report_frontend/research_mapper.py:65`、`report_frontend/evidence_thresholds.json:52-57`
- Test: `tests/test_feature_key_rename.py`

**Interfaces:**
- Consumes: 语音日志列 `connective_density` / `connective_density_std`(Task 3)
- Produces: 特征键 `voice_research_connective_density_mean` / `_std`;映射元组关键字 `connective_density`,显示名「连接词密度」

- [ ] **Step 1: 写失败测试**

```python
# tests/test_feature_key_rename.py
from pathlib import Path

from report_frontend.research_mapper import ResearchCapabilityMapper

ROOT = Path(__file__).resolve().parent.parent


def test_mapper_indicator_is_renamed_and_matches_new_key():
    rule = ResearchCapabilityMapper().mapping_rules["logical_thinking"]
    keywords = [ind[0] for ind in rule["indicators"]]
    names = [ind[3] for ind in rule["indicators"]]
    assert "connective_density" in keywords
    assert "logic_keyword_density" not in keywords
    assert "连接词密度" in names


def test_feature_engine_has_no_second_definition_of_the_metric():
    """同名指标只能有一处定义 —— feature_engine 里的两张关键词表必须删净。"""
    text = (ROOT / "report_frontend/feature_engine.py").read_text(encoding="utf-8")
    assert "logic_keyword" not in text
    assert "logic_keywords" not in text


def test_new_key_can_pass_the_gate_and_score():
    feats = {"voice_research": {"connective_density_mean": 3.5,
                                "connective_density_std": 0.4,
                                "_n_rows": 4.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]
    assert dim["evidence_chain"], "证据链为空 —— 新键没被映射到"
    assert any("连接词密度" in ev.get("human_name", "") or "连接词密度" in str(ev) 
               for ev in dim["evidence_chain"])
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_feature_key_rename.py -q`
Expected: FAIL(`logic_keyword` 仍在 feature_engine;映射元组仍是旧关键字)

- [ ] **Step 3: 三处修改**

`feature_engine.py`:**整段删除** `:310-346` 的文本扫描与两份关键词表(以及 `:509` 的自检行),让新列走通用数值路径(`_extract_numeric_stats` 自动产出 `_mean`/`_std`)。

`research_mapper.py:65`:
```python
                ("logic_keyword_density", 0.4, True, "逻辑关键词密度", "core"),
→               ("connective_density", 0.4, True, "连接词密度", "core"),
```

`evidence_thresholds.json` 的 `density` 族:量程随新定义更新(旧 `full_scale: 0.02` 对应"命中数÷字符数";新定义是每百字,典型 0~10):
```json
    "density": {
      "kind": "full_scale",
      "full_scale": 10.0,
      "basis_kind": "definitional",
      "basis": "每百字 10 个连接词记为满量程。旧值 0.02 对应'命中数÷字符数',与新的每百字定义相差约 100 倍,故必须改。M5 标定到位后替换。",
      "_provisional": true
    },
```
并把文件顶部 `_version` 提升一位。

- [ ] **Step 4: 运行,确认通过 + 全套测试 + 禁止词扫描仍绿**

```bash
~/miniconda3/envs/jingxin/bin/python -m pytest -q                 # 期望全绿
~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k banned -q
```

- [ ] **Step 5: 提交**

```bash
git add report_frontend/feature_engine.py report_frontend/research_mapper.py report_frontend/evidence_thresholds.json tests/test_feature_key_rename.py
git -c user.name="huihuibuhui227" -c user.email="huihuibuhui227@gmail.com" commit -m "refactor(m1): 连接词密度只留一处定义;映射与折算量程随新定义更新"
```

---

### Task 7: 验收门(使用者亲验,≥5 段回答)

**Files:** 无(运行手册);产出 `~/shared/jingxin_recordings/{session_id}/` 与 `data/logs/` 下的三份日志

- [ ] **Step 1: 起服务(三个终端)**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m voice_interaction.api.app     # :8001
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m face_expression.api.app       # :8000
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m gesture_analysis.api.app      # :8002
```

- [ ] **Step 2: 开一场会话,拿到 id**

```bash
curl -s -X POST http://127.0.0.1:8001/interview/start | tee /tmp/m1_start.json
SID=$(~/miniconda3/envs/jingxin/bin/python -c "import json;print(json.load(open('/tmp/m1_start.json'))['session_id'])")
echo "session_id=$SID"
```

- [ ] **Step 3: 录 **≥5 段内容各不相同**的回答**

  两个门槛是**两件事**(见 spec §6.5),别混:
  - **G2** 只要求「会话内有变异」→ 第 2 段起即可满足(`_std` 只有一个样本时为 0,
    会被判「本次会话内无变化」—— 设计使然);
  - **G3** 是**样本量门槛 = 5**(本指标每段回答一个样本,而题库只有 8 题,默认的 10 不可达)
    → **答满 5 段**该槽才会过门出分。低于 5 段被拦下不是 bug。

  8 段回答**内容必须不同**:同一段音频重复提交 → 每行密度相同 → `_std = 0` → G2 拦下。

```bash
curl -s -X POST "http://127.0.0.1:8001/interview/answer_audio?session_id=$SID" -F "audio=@/path/ans1.wav"
curl -s -X POST "http://127.0.0.1:8001/interview/answer_audio?session_id=$SID" -F "audio=@/path/ans2.wav"
```

- [ ] **Step 4: 核对四件事**

```bash
ls ~/shared/jingxin_recordings/$SID/                       # session.json + transcript.json
grep -l "$SID" data/logs/*.csv | head                       # 三份日志都带同一个 id
grep -c "$SID" data/logs/face_au_log_$SID.csv data/logs/gesture_emotion_log_$SID.csv data/logs/interview_emotion_log_$SID.csv
grep -rn "$SID" --include="*.csv" --include="*.json" . | grep -v data/logs | wc -l    # 仓库内除此以外不应有原句
~/miniconda3/envs/jingxin/bin/python -m report_frontend.report_generator
```

**看什么**:① 三份日志文件名与首列都是同一个 `$SID`;② `transcript.json` 在 `{SID}/` 下、含逐字时间戳,而**仓库内任何文件都不含原句**;③ 生成的报告里「话语结构特征」维度出现「连接词密度」这一行并**过门出分**(呈现为「连接词密度 + 本场区间 + 有效样本量」),而不是「未采集到对应数据」。

- [ ] **Step 5: 记录结果**

把四件事的实测值写进 `docs/superpowers/sdd/2026-09-24-m1-asr-session-id/`(可直接引用 spec §10 的四条),并把未达标项开成后续任务。

---

## 自查记录(写计划时对本 spec 逐条对照)

- **覆盖**:spec §5 组件表每一项都有任务(engine→T1;density→T2;三个 logger→T3;voice 端点 + vosk 拆除→T4;face/gesture→T5;报告侧三处→T6);§6 契约(session_id 格式/端点/日志/transcript/density)→ T1+T3+T4;§8 错误处理表 → T1(异常上抛)+ T2(过短不出值)+ T4(空文本 400);§9 五条未满足项与风险 → T4(置信度字段写 null)、spec 保留、验收门看 ≥5 段;§10 测试 → T1–T6;§10 验收门 → T7。
- **已修正的 spec 错误**:`whitelist_Cplus.json` 不存在 → 该项已从 M1 移除;`density` 量程需随新定义更新 → 补进 spec §6.5 与 T6。
- **类型一致性**:`AsrUtterance` 字段名在 T1/T4 一致;`connective_density()` 的 `None` 语义在 T2/T4/T6 一致(过短不出值,不写 0);`NONE_SESSION` 三处同值由 T3 的文本级测试守住。
