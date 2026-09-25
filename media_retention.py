"""原始媒体留存 —— 把三路服务收到的字节原样存到仓库外的共享盘。

为什么要有这个文件(M2.6 spec §3.2):
  盘上**一份原始媒体都没留**,而且**没有任何生产代码在写** —— 全仓无 `VideoWriter`、
  生产路径无 `cv2.imwrite`、出现的 `wave.open` 全是 `'rb'` 读入即弃。
  而管线的缺陷是**已知的**(§4 那张处置表就是缺陷清单)。没有原始媒体 → 那些缺陷被
  永久烘进数据,事后无法重抽。本模块做的就是"收下什么就存什么"。

与 `transcript_store.py` 的分工:那个模块负责原句(transcript.json),本模块负责像素与
声音。两者落点、守卫、会话锁语义**一致**,但**各持一份副本** —— face/gesture 不 import
voice 包(否则会把 TTS/ASR 整条 import 链拉起来,M1 Ruling M1-2)。副本由
`tests/test_media_retention.py` 的源码级契约测试压着,与三份 `_resolve_session_id` 同一手法。

与 `logging_config.py` / `session_clock.py` 同层放在仓库根:三个服务都以
`-m <模块>.api.app` 启动,仓库根在 sys.path 上,所以三边都 import 得到。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

DEFAULT_ROOT = Path.home() / "shared" / "jingxin_recordings"   # D:\Shared\jingxin_recordings
RETENTION_FILENAME = "retention.jsonl"
MEDIA_SUBDIR = "media"

# 会话 id 直接当目录名用,而它是【客户端可控】的(query/表单参数)—— 与
# transcript_store.SESSION_ID_PAT 同一守卫、同一理由(见那边的 docstring)。
SESSION_ID_PAT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_OFF_VALUES = {"0", "false", "no", "off"}


def root() -> Path:
    """录制根。`JINGXIN_RECORDINGS_DIR` 可覆盖(与 transcript_store 同一开关)。

    每次调用都读环境变量,不用模块级常量 —— 否则测试没法在不重载模块的前提下换目录。
    """
    return Path(os.getenv("JINGXIN_RECORDINGS_DIR", DEFAULT_ROOT))


def enabled() -> bool:
    """默认开。只有显式写 `0/false/no/off` 才关(spec §6)。"""
    return os.getenv("JINGXIN_RETAIN_MEDIA", "1").strip().lower() not in _OFF_VALUES


def validate_session_id(session_id: str) -> str:
    """守卫:只允许 `[A-Za-z0-9_-]{1,128}`(minted 形状与 `NONE` 都在内)。

    为什么必须在模块这一层拦:端点把 id 当 query/表单参数收下(客户端可控),而这个值
    会被拼进 `recording_dir` 的路径再 `mkdir(parents=True)` —— `../../jingxin/x` 这类值
    能在录制根**之外**建目录。放在这里而不是端点里:任何调用方都自动受保护。
    """
    if not isinstance(session_id, str) or not SESSION_ID_PAT.fullmatch(session_id):
        raise ValueError(
            f"非法 session_id: {session_id!r} —— 只允许字母/数字/下划线/连字符,1–128 位"
            f"(它是目录名,不接受路径分隔符与 '..')")
    return session_id


def validate_modality(modality: str) -> str:
    """模态名也是目录名的一部分 —— 与 session_id 同一类守卫。

    放在本步(而不是用它的 Task 2)是因为它是**守卫**,和 `validate_session_id` 是
    同一层的东西;Task 1 的预检路径也要先过它。
    """
    if modality not in ("face", "gesture"):
        raise ValueError(f"未知模态: {modality!r}(只允许 face / gesture)")
    return modality


def recording_dir(session_id: str) -> Path:
    d = root() / validate_session_id(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def media_dir(session_id: str, *parts: str) -> Path:
    """`<会话>/media/<parts…>`,不存在就建。

    为什么单独一个函数:帧与音频都写在 `media/` **下面**的子目录里,而
    `write_bytes` 不会替调用方建父目录 —— 少了它,第一帧就 FileNotFoundError
    (实测:这正是本模块第一次跑测试时的红法)。
    """
    d = recording_dir(session_id) / MEDIA_SUBDIR
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d


def _can_write_here(directory: Path) -> bool:
    """真写一个探针文件再删掉 —— 比 `os.access` 可信(WSL 的 9p 挂载会骗人)。"""
    try:
        probe = directory / ".retention_probe"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except Exception:
        return False


# ── 进程内状态 ──────────────────────────────────────────────────────────────
# 本服务是单进程 uvicorn;若以后多进程/多机部署,这里要换成文件锁
# (与 transcript_store._LOCKS 的注释同一句话、同一个前提)。
_LOCKS: dict[str, threading.Lock] = {}
_STATE_GUARD = threading.Lock()
_PREFLIGHTED: set[str] = set()
_LOUD_DONE: set[str] = set()
_DEGRADED: dict[str, list[str]] = {}
_COUNTERS: dict[tuple[str, str], int] = {}


def _reset_for_tests() -> None:
    """清空全部进程内状态。只在测试里用(每个测试都要一个干净的起点)。"""
    with _STATE_GUARD:
        _LOCKS.clear()
        _PREFLIGHTED.clear()
        _LOUD_DONE.clear()
        _DEGRADED.clear()
        _COUNTERS.clear()


def _session_lock(session_id: str) -> threading.Lock:
    """取(或建)本会话的写锁。

    为什么要有:同一会话的多帧可能并发到达,而"取序号 → 写文件 → 追加账本"三步必须
    串行,否则两帧会拿到同一个序号(后者盖掉前者),或者账本行交错。加锁粒度是会话:
    不同会话之间没有共享状态。
    """
    with _STATE_GUARD:
        lock = _LOCKS.get(session_id)
        if lock is None:
            lock = _LOCKS[session_id] = threading.Lock()
        return lock


def _prepare(session_id: str) -> bool:
    """本次能不能写。首件素材预检不过 → **抛**;之后只留痕、返回 False。

    spec §6 的两级:意思是"半路才发现等于已经丢了一半素材,而这一场是人重跑不回来
    的",所以第一件必须当场响;但中断也换不回已丢的字节、还白耗人的时间,所以只响一次。
    """
    if session_id in _PREFLIGHTED:
        return True
    try:
        # 探到**真正要写的那个目录**(media/),不是只探会话目录 ——
        # "会话目录建得出、里面的子目录建不出"这种情形也必须在首件就被拦住。
        d = media_dir(session_id)
    except Exception as exc:
        _mark_degraded(session_id, f"preflight: 建目录失败 {type(exc).__name__}: {exc}")
        return _loud_or_silent(session_id, f"建目录失败 {type(exc).__name__}: {exc}")
    if _can_write_here(d):
        _PREFLIGHTED.add(session_id)
        return True
    _mark_degraded(session_id, f"preflight: 落点不可写 {d}")
    return _loud_or_silent(session_id, f"落点不可写 {d}")


def _loud_or_silent(session_id: str, why: str) -> bool:
    with _STATE_GUARD:
        if session_id in _LOUD_DONE:
            return False
        _LOUD_DONE.add(session_id)
    raise RuntimeError(
        f"留存预检失败(本会话首件素材):{why} —— "
        f"修好落点再跑;这一场不重跑就永久没有素材了")


def _mark_degraded(session_id: str, why: str) -> None:
    with _STATE_GUARD:
        _DEGRADED.setdefault(session_id, []).append(why)


def degraded_reasons(session_id: str) -> list[str]:
    """本会话留存失败的**全部原因**(空列表 = 一切正常)。会话收尾时读。"""
    with _STATE_GUARD:
        return list(_DEGRADED.get(session_id, []))


def _bump_seq(session_id: str, kind: str) -> int:
    key = (session_id, kind)
    _COUNTERS[key] = _COUNTERS.get(key, 0) + 1
    return _COUNTERS[key]


def _current_seq(session_id: str, kind: str) -> int:
    return _COUNTERS.get((session_id, kind), 0)


def _append_jsonl(session_id: str, record: dict) -> None:
    """追加一行账。**追加**而不是读-改-写:进程被杀时已写下的行不会损坏。"""
    p = recording_dir(session_id) / RETENTION_FILENAME
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record(session_id: str, kind: str, modality: str | None, seq: int,
            rel_file: str, data: bytes, declared_ts: int | None,
            source: str) -> dict:
    return {
        "kind": kind,
        "modality": modality,
        "seq": seq,
        "file": rel_file,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "received_at_wall": time.time(),
        "declared_ts": declared_ts,
        "source": source,
        "session_id": session_id,
    }


def retain_frame(session_id: str, modality: str, data: bytes,
                 declared_ts: int | None = None, source: str = "") -> dict | None:
    """把一帧的**原始字节**存下来。返回写进账本的那一行;没存则 None。

    存的是服务端收到的**同一份 bytes**,不重新编码、不缩放(录制需求 §5)。
    """
    if not enabled():
        return None
    sid = validate_session_id(session_id)
    validate_modality(modality)
    with _session_lock(sid):
        if not _prepare(sid):
            return None
        seq = _bump_seq(sid, modality)
        try:
            fname = f"{seq:06d}.jpg"
            (media_dir(sid, modality) / fname).write_bytes(data)
            rel = f"{MEDIA_SUBDIR}/{modality}/{fname}"
            rec = _record(sid, "frame", modality, seq, rel, data, declared_ts, source)
            _append_jsonl(sid, rec)
            return rec
        except Exception as exc:
            _mark_degraded(sid, f"frame {modality}#{seq}: {type(exc).__name__}: {exc}")
            return None


def retain_audio(session_id: str, kind: str, data: bytes, source: str = "") -> dict | None:
    """把一段音频的**原始字节**存下来。`kind` ∈ {"raw", "converted"}。"""
    if not enabled():
        return None
    if kind not in ("raw", "converted"):
        raise ValueError(f"未知音频种类: {kind!r}(只允许 raw / converted)")
    sid = validate_session_id(session_id)
    with _session_lock(sid):
        if not _prepare(sid):
            return None
    raise NotImplementedError("Task 3 实现落盘部分")
