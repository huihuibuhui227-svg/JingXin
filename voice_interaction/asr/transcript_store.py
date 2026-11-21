"""转写与会话清单落盘 —— 一律在仓库外(spec D2)。

仓库内只允许出现数字;原句只进 ~/shared/jingxin_recordings/{session_id}/。

## 本文件与上游的差异(补丁 `patch/transcript_store.py`,2026-11-21)

`transcript.json` 的顶层 `asr` 块补全为**可自证的 provenance**(只增键,不改既有键):

    confidence / confidence_source / confidence_reason   置信度三态(test §9.1)
    segments / chars                                     本文件当前的段数与字数

并修一个**潜在崩溃**:旧文件(没有 `asr` 块)再次追加时会 `KeyError: 'asr'`。
`_asr_meta()` 的形状**一个字都没动** —— 它与 `voice_interaction/api/app.py::_asr_meta`
有 `test_the_two_asr_meta_producers_agree` 的**逐字相等**钉子,动它会把清单也一起改。
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .session import NONE_SESSION
# 目录名规则(含"存过标注就叫 `<标签>__<sid>`")只允许有一处定义,见下面
# `recording_dir` 的说明。别名导入:本模块自己也有一个 `recording_dir`。
from media_retention import recording_dir as _shared_recording_dir
import media_retention as _media_retention

DEFAULT_ROOT = Path.home() / "shared" / "jingxin_recordings"          # D:\Shared\jingxin_recordings
TRANSCRIPT_FILENAME = "transcript.json"        # spec §6.4:一个会话一个文件,累积写
LOG_PREFIXES = {"face": "face_au_log", "gesture": "gesture_emotion_log",
                "voice": "interview_emotion_log"}

# 会话 id 直接当目录名用,而它是【客户端可控】的(query/表单参数),所以必须限量。
SESSION_ID_PAT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# ── 置信度三键的**唯一真源**(spec §9.1)────────────────────────────────────
#
# 为什么是常量而不是现场算:本部署 FunASR 的 raw 里**根本没有**逐字/逐段概率字段。
# 实测 raw 键只有 is_final / mode / punc_array / segment_count / spk_name /
# spk_score / text / timestamp / wav_name(见 `asr_config.json` 的 `models_basis`
# 与 spec §3)。`spk_score` 是**声纹验证**分数,拿它冒充识别置信度就是"编一个数"。
# 既然恒为同一组值,就只定义一次;两个键名(见 `_ASR_CONST_KEYS` 的说明)共用这一份,
# 由 `verify/verify_transcript_store.py` 断言它们不会被写成不一样的值。
ASR_CONFIDENCE: float | None = None
ASR_CONFIDENCE_SOURCE = "unavailable"
ASR_CONFIDENCE_REASON = (
    "服务端 raw 响应不含逐字概率字段"
    "(raw 键只有 is_final/mode/punc_array/segment_count/spk_name/spk_score/"
    "text/timestamp/wav_name;spk_score 是声纹验证分数,不是识别置信度)")

#: `asr` 块里"值恒为同一组"的键 → 当前值。`segments`/`chars` 不在其中:
#: 它们描述**这个文件当前**的规模,由调用方按刚算出来的 `merged` 给。
#:
#: 为什么两个名字都写(`confidence` 与 `asr_confidence`):
#:   台账(`l0_columns.json`)的列名是 `asr_confidence`/`asr_confidence_source`,
#:   说明书用的短名是 `confidence`/`confidence_source`。它们是**同一个事实**的两个
#:   名字,写进同一份文件是**有意**的 —— 值全部来自下面这三个常量(唯一真源),
#:   由 `verify/verify_transcript_store.py` 断言两组值恒等。
#:   要退役任一组,删掉对应键 + 同步改台账或说明书即可,不会有两套算法。
_ASR_CONST_KEYS: dict[str, Any] = {
    "confidence": ASR_CONFIDENCE,
    "confidence_source": ASR_CONFIDENCE_SOURCE,
    "confidence_reason": ASR_CONFIDENCE_REASON,
    "asr_confidence": ASR_CONFIDENCE,
    "asr_confidence_source": ASR_CONFIDENCE_SOURCE,
}
#: `asr` 块的**落盘顺序**(读者从上到下就看得懂:谁认的 → 认得准不准 → 有多少)。
_ASR_KEY_ORDER = ("engine", "endpoint", "models",
                  "confidence", "confidence_source", "confidence_reason",
                  "segments", "chars",
                  "asr_confidence", "asr_confidence_source")


def validate_session_id(session_id: str) -> str:
    """守卫:会话 id 只允许 `[A-Za-z0-9_-]{1,128}`(minted 形状与 `NONE` 都在内)。

    为什么必须在 store 这一层拦:端点把 `session_id` 当 query/表单参数收下(客户端可控),
    而这个值会被拼进 `recording_dir` 的路径再 `mkdir(parents=True)` ——
    `../../jingxin/x` 这类值能在录制根**之外**建目录并写入含原句的 transcript.json,
    把"原句绝不进仓库"(spec D2)这条不变量整个打穿。
    放在这里而不是端点里:任何调用方(含以后的 face/gesture)都自动受保护。
    公开出来是给端点用的:端点在跑识别之前先校验,好回 400 而不是等落盘时炸成 500。
    """
    if not isinstance(session_id, str) or not SESSION_ID_PAT.fullmatch(session_id):
        raise ValueError(
            f"非法 session_id: {session_id!r} —— 只允许字母/数字/下划线/连字符,1–128 位"
            f"(它是目录名,不接受路径分隔符与 '..')")
    return session_id


def normalize_session_id(raw: str | None) -> str:
    """把「客户端给的原始 id」规范化:**只有"参数没给"算没给**。

    与 `validate_session_id`(什么算非法)是**分工**,不是两层各判一次:
      * `None`(参数不存在 / 表单里没这个键)→ `NONE`,这是无会话客户端的正常路径;
      * 给了但内容是空的(`""` / `"  "`)→ **原样交出,由守卫判非法 → 400**。

    为什么空串不"顺手归 NONE":空串与"没给"在客户端那里是两件事 —— 后者是没接会话,
    前者是**参数拼错了**(例如 `?session_id=${sid}` 而 sid 为空)。静默归进 `NONE` 之后
    客户端拿到的是 200 和一份看着正常的响应,问题只在报告里以"数据对不上"的形式浮出来。
    三份副本(voice/face/gesture)由 tests/test_session_id_normalization.py 压着逐字相同。
    """
    return NONE_SESSION if raw is None else raw


# 每个 session_id 一把锁,保护 append_utterance 的读-改-写(见 _session_lock)。
# 放进程内:这个服务是单进程的 uvicorn;若以后多进程/多机部署,这里要换成文件锁。
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _session_lock(session_id: str) -> threading.Lock:
    """取(或建)本会话的写锁。

    `append_utterance` 是「读 json → 追加段 → 原子替换」:两个并发请求读到同一份旧
    segments 时,后写的会把先写的整个盖掉 —— 丢掉的那次回答在数字报告里看不出来
    (总段数只是少了一段,不报错),所以必须在写侧按会话串行。加锁粒度是会话:不同
    会话之间没有共享状态,不必互相等。

    `ensure_manifest` / `refresh_manifest` 不加锁:前者只在会话开始写一次且已存在即
    返回(幂等),后者只在会话结束写一次,都不在"同一会话并发写"的路径上。
    """
    with _LOCKS_GUARD:
        lock = _LOCKS.get(session_id)
        if lock is None:
            lock = _LOCKS[session_id] = threading.Lock()
        return lock


def _root(root: str | Path | None) -> Path:
    """录制根。`root=None`(生产那条路)时**委托给 `media_retention.root()`**。

    ⚠️ 2026-10-24 起不再自己算回退:此前这里也写 `os.getenv(..., DEFAULT_ROOT)`,
    于是"没设环境变量时读哪一份素材"有**两处**定义。服务器上 `DEFAULT_ROOT`
    (`~/shared/jingxin_recordings`)是个空壳、真素材在 `~/JingXin/recordings`,
    两处一旦不一致,报告与转写就会读到不同的盘 —— 而且是**静默**的。
    现在只留 `media_retention.root()` 一处(见那边的 docstring)。
    """
    if root:
        return Path(root)
    return _media_retention.root()


def recording_dir(session_id: str, root: str | Path | None = None) -> Path:
    """本场的录制目录。

    ⚠️ 目录**名**不是死的:存过标注的场次叫 `<标签>__<sid>`(`media_retention`
    的 `resolve_recording_dir`)。所以 `root=None`(生产那条路)时必须**委托给它**,
    不能在这里按 `root()/<sid>` 自己算 —— 自己算的话,改过名的场次会被劈成两个
    目录:`session.json` 写进旧的空目录、而帧写进新的,**两边都"成功"**。
    `root` 显式给了的调用方(测试/工具)按老样子走,不受目录名规则影响。
    """
    if root is None:
        return _shared_recording_dir(session_id)
    d = _root(root) / validate_session_id(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(session_id: str, root=None) -> Path:
    return recording_dir(session_id, root) / "session.json"


def _transcript_path(session_id: str, root=None) -> Path:
    return recording_dir(session_id, root) / TRANSCRIPT_FILENAME


def asr_provenance() -> dict[str, Any]:
    """`asr` 块的**共享那部分**(spec §6.4)—— 本模块里唯一的组装处。

    engine/endpoint/models 从 `asr_config.json` 现取(经公开访问器 `load_config`)——
    不在这里再存一份 host/port。
    `asr_confidence` 恒为 null 且 `asr_confidence_source` 恒为 "unavailable":
    该部署的 raw 里没有置信度字段(实测,spec §3),必须显式落盘而非省略(spec §9.1)。

    ⚠️ **这个形状不许改**(补丁也一样):`voice_interaction/api/app.py` 里另有一份
    同形同源的 `_asr_meta()`,钉子是 `tests/test_text_l0_columns.py::
    test_the_two_asr_meta_producers_agree`(逐字相等)。要加字段就加到
    `asr_block()` 里 —— 那才是"只给 transcript.json"的形状。
    """
    from .funasr_engine import load_config

    cfg = load_config()
    return {
        "engine": "funasr",
        "endpoint": f"ws://{cfg['funasr_host']}:{cfg['funasr_port']}",
        "models": dict(cfg.get("models") or {}),
        "asr_confidence": ASR_CONFIDENCE,
        "asr_confidence_source": ASR_CONFIDENCE_SOURCE,
    }


def _asr_meta() -> dict[str, Any]:
    """`session.json` 用的 provenance 块 —— 与 `asr_provenance()` 同一个值。

    保留这个名字是因为端点与既有测试都按私有名取它(见上一条 ⚠️)。
    """
    return asr_provenance()


def asr_block(existing: dict[str, Any] | None = None,
              n_segments: int | None = None,
              n_chars: int | None = None) -> dict[str, Any]:
    """`transcript.json` 顶层 `asr` 块 = provenance + 置信度三键 + 本文件计数。

    `existing`:上一次写下的 `asr` 块(旧文件可能缺键,也可能整个没有)。
      * 缺键 ⟹ 按 `_ASR_CONST_KEYS` 补齐 —— **只增不改**;
      * `engine`/`endpoint`/`models` ⟹ **原样保留**,那是"这一场是用什么认出来"的
        历史事实,后面换了模型也不许追改(spec §6.4 的 provenance 语义)。
    `n_segments`/`n_chars`:**未知写 `null`,不写 0** —— 0 段 0 字是"这一场没人说话",
      与"还没算"是两件事(本仓通则;`transcript_chars.n_chars` 同一条)。

    键序见 `_ASR_KEY_ORDER`;`existing` 里本版本不认识的键**原样带过**(不静默丢)。
    """
    base = dict(existing) if isinstance(existing, dict) else {}
    fresh: dict[str, Any] = {
        **asr_provenance(), **_ASR_CONST_KEYS,
        "segments": None if n_segments is None else int(n_segments),
        "chars": None if n_chars is None else int(n_chars),
    }
    # 旧块里可能留着**过期**的计数,不能让它盖掉刚算的那份
    for stale in ("segments", "chars"):
        base.pop(stale, None)
    out = {k: base.pop(k, fresh.get(k)) for k in _ASR_KEY_ORDER}
    out.update(base)                              # 未识别的键(如 backfilled)原样保留
    return out


def _segment_records(utt, base_index: int) -> list[dict[str, Any]]:
    """按 spec §6.4 的段形状落盘(只留契约里的 6 个键,引擎以后加字段也不会漏进契约)。"""
    segs = list(getattr(utt, "segments", None) or [])
    return [{
        "index": base_index + i,
        "text": seg.get("text", "") or "",
        "n_chars": seg.get("n_chars", 0),
        "timestamps_ms": seg.get("timestamps_ms") or [],
        "punc_array": seg.get("punc_array") or [],
        "ts_origin": "segment_relative",     # spec §6.4:段内相对,不假装绝对
    } for i, seg in enumerate(segs)]


def _merge(segments: list[dict[str, Any]], vad_split: bool) -> dict[str, Any]:
    """merged = 全段重算:文本拼接、字数求和、段数、是否裂过(spec §6.4)。"""
    return {
        "text": "".join(s["text"] for s in segments),
        "n_chars": sum(s["n_chars"] for s in segments),
        "n_segments": len(segments),
        "vad_split": vad_split,
    }


def append_utterance(session_id: str, utt, recorded_at: str | None = None,
                     root: str | Path | None = None) -> Path:
    """把一次识别结果累积进仓库外的 `transcript.json`(spec §6.4 / D2)。

    一个会话一个文件:本次的段**追加**到既有 segments 之后(`index` 连续、不清空既有段),
    `merged` 按全部段重算;`recorded_at` 只由**第一次写入**决定,后续追加不刷新;
    `asr` 块的 provenance 部分(engine/endpoint/models)也保留第一次写入的那份,
    但 `asr.segments`/`asr.chars` 随本次写入刷新 —— 它们描述的是**这个文件现在**
    有多少段多少字,必须与 `merged` 逐字对齐,冻结在第一次就自相矛盾了。
    任一次识别被 VAD 裂过,整场 `merged.vad_split` 即为 true(累积 OR)。

    ⚠️ 旧文件(补丁前写的、没有 `asr` 块)再追加时,原来那行 `existing["asr"]`
    会 `KeyError: 'asr'` —— 而那正是**本次补丁要服务的存量对象**(26 场批量转写
    落的是另一种扁平形状,但录制目录里的 `transcript.json` 也有一批更早的)。
    现在缺块就按 `asr_block()` 重建一份并如实标 `confidence_source=unavailable`。

    整段读-改-写在**本会话的锁**内完成(见 `_session_lock`):同一会话的并发调用
    (客户端重试 / 重复提交 / `/asr` 与 `/answer_audio` 撞车)必须串行,否则后写的
    会拿旧快照把先写的那次回答整个盖掉,而且丢得无声无息。
    """
    with _session_lock(session_id):
        p = _transcript_path(session_id, root)
        existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

        if existing is None:
            segments: list[dict[str, Any]] = []
            vad_split = bool(utt.vad_split)
            first_recorded_at = recorded_at or datetime.now().astimezone().isoformat(timespec="seconds")
            prev_asr = None
        else:
            segments = existing["segments"]
            vad_split = bool(existing["merged"]["vad_split"]) or bool(utt.vad_split)
            first_recorded_at = existing["recorded_at"]
            # `.get` 而不是 `[]`:补丁前的文件没有这个键(见 docstring 的 ⚠️)
            prev_asr = existing.get("asr")

        segments.extend(_segment_records(utt, len(segments)))
        merged = _merge(segments, vad_split)
        # 计数取自刚算好的 `merged`,不在别处再数一遍 —— 同一个事实只算一次
        asr_meta = asr_block(prev_asr, merged["n_segments"], merged["n_chars"])
        payload = {
            "session_id": session_id,
            "recorded_at": first_recorded_at,
            "asr": asr_meta,
            "merged": merged,
            "segments": segments,
        }
        # 先写临时文件再原子替换:read-modify-write 中途崩掉不会毁掉整场既有段
        tmp = p.parent / (p.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
        return p


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
