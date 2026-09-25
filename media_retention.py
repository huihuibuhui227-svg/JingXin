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
MEDIA_SUBDIR = "media"

# 账本文件名:`retention.<写入者>.jsonl`。**每个写入者一个文件** —— 理由见
# `_append_jsonl` 的 docstring(三个服务是三个进程,共用一个文件会把行劈开)。
LEDGER_PREFIX = "retention"
LEDGER_SUFFIX = ".jsonl"
LEDGER_WRITERS = ("face", "gesture", "audio")

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


PROBE_BYTES = b"\x00" * 4096          # 探针要**非空**:空文件不占数据块


def _probe_name() -> str:
    """探针文件名。**必须每进程/每线程唯一**。

    为什么(独立审查 Minor 4):face 与 gesture 是两个进程,却可能探同一个会话。
    固定名字会让"A 写、B 写、A 删、B 删"变成 B 的 unlink 抛 FileNotFoundError
    → `_can_write_here` 判 False → **B 的首件素材假报 500**(磁盘其实好好的)。
    """
    return f".retention_probe.{os.getpid()}.{threading.get_ident()}"


def _write_probe(directory: Path) -> bytes:
    """往目标目录真写一个**非空**探针文件,返回写下的字节;写完删掉。

    为什么非空(独立审查 Important 1):空文件不需要数据块,**磁盘满时照样建得出** ——
    于是预检通过、首帧真实的几十 KB 写入 ENOSPC,又被吞成 degraded、请求还回 200。
    用户看到的是"跑完一场,报告里数据对不上",而不是"当时就报错"。

    为什么是"真写"而不是 `os.access`/`shutil.disk_usage`:前者在 WSL 的 9p 挂载上会骗人;
    后者只是余量估算,证明不了这个目录真写得进。写一个 4 KiB 的块是直接证据。
    """
    probe = directory / _probe_name()
    probe.write_bytes(PROBE_BYTES)
    try:
        probe.unlink()
    except FileNotFoundError:              # 名字唯一,理论上不会;真撞了也不该让预检判失败
        pass
    return PROBE_BYTES


def _can_write_here(directory: Path) -> bool:
    try:
        _write_probe(directory)
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


def _prepare(session_id: str, writer: str) -> bool:
    """本次能不能写。首件素材预检不过 → **抛**;之后只留痕、返回 False。

    spec §6 的两级:意思是"半路才发现等于已经丢了一半素材,而这一场是人重跑不回来
    的",所以第一件必须当场响;但中断也换不回已丢的字节、还白耗人的时间,所以只响一次。

    ⚠️ 探的必须是**真正要写的那个目录** `media/<writer>/`(独立审查 Important 1):
    只探到 `media/` 会把边界下移一层 —— `media/face` 若被占成一个普通文件,
    预检照样过,首帧才 FileExistsError 并被吞成 degraded(实测已复现)。
    """
    key = f"{session_id}\x00{writer}"
    if key in _PREFLIGHTED:
        return True
    try:
        d = media_dir(session_id, writer)
    except Exception as exc:
        why = f"建目录失败 {type(exc).__name__}: {exc}"
        _mark_degraded(session_id, writer, f"preflight: {why}")
        return _loud_or_silent(session_id, why)
    if _can_write_here(d):
        _PREFLIGHTED.add(key)
        return True
    why = f"落点不可写 {d}"
    _mark_degraded(session_id, writer, f"preflight: {why}")
    return _loud_or_silent(session_id, why)


def _loud_or_silent(session_id: str, why: str) -> bool:
    with _STATE_GUARD:
        if session_id in _LOUD_DONE:
            return False
        _LOUD_DONE.add(session_id)
    raise RuntimeError(
        f"留存预检失败(本会话首件素材):{why} —— "
        f"修好落点再跑;这一场不重跑就永久没有素材了")


def _mark_degraded(session_id: str, writer: str, why: str) -> None:
    """记一笔降级:**内存里记一份 + 往账本追加一行**。

    为什么两处都要(独立审查 Important 2):只记内存的话,进程一重启就彻底没痕迹
    —— 而 spec §6 写的是"任何情况都不得静默"。账本那一行是**唯一能活过重启**的痕迹。

    写账这一步是**尽力而为**:磁盘坏掉时它自己也会失败,那时只能放弃(再抛就成了
    "因为记不下来所以整场崩",比静默还糟)。所以这里吞掉异常,但内存那份仍然有。
    """
    with _STATE_GUARD:
        _DEGRADED.setdefault(session_id, []).append(why)
    try:
        rec = {"kind": "degraded", "modality": writer, "seq": None, "file": None,
               "bytes": 0, "sha256": None, "received_at_wall": time.time(),
               "declared_ts": None, "source": "media_retention", "reason": why,
               "session_id": session_id}
        _append_jsonl(session_id, writer, rec)
    except Exception:
        pass


def degraded_reasons(session_id: str) -> list[str]:
    """本会话留存失败的**全部原因**(空列表 = 一切正常)。

    **合并两处**:进程内存里那份(本次运行)与账本里的 `degraded` 行(活过重启的那份)。
    """
    with _STATE_GUARD:
        out = list(_DEGRADED.get(session_id, []))
    try:
        for book in ledger_files(session_id):
            for line in book.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip() or '"degraded"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") == "degraded" and rec.get("reason"):
                    out.append(str(rec["reason"]))
    except Exception:
        pass
    return out


def _highest_seq_on_disk(session_id: str, kind: str) -> int:
    """盘上已有的最大序号(`000123.jpg` / `0003_converted.wav` → 123 / 3)。

    音频文件形如 `0001.webm` 与 `0001_converted.wav`,所以取 `_` 之前那一段。
    """
    d = recording_dir(session_id) / MEDIA_SUBDIR / kind
    if not d.is_dir():
        return 0
    hi = 0
    for p in d.iterdir():
        stem = p.name.split("_")[0].split(".")[0]
        if stem.isdigit():
            hi = max(hi, int(stem))
    return hi


def _bump_seq(session_id: str, kind: str) -> int:
    """下一个序号。**从盘上推**,不从进程内存推。

    为什么(独立审查 Critical 2,有实盘证据):序号若是纯内存状态,服务重启
    (`start_all.sh` 按端口幂等,重启是例行操作)或同一 id 复用(含 `NONE` 桶)时,
    计数器从 0 重来 → 首件又是 `000001.jpg` → **把已留存的素材同名覆盖掉**,而账本继续
    追加,于是同一个 `file` 出现两条 sha256 不同的行。
    实测:`~/shared/jingxin_recordings/20260924_153012_9f3c` 账本 **52 行、盘上只 12 个文件**,
    12 条重复指向同一文件;`NONE/` 同样(54 行 / 12 文件)。

    性能:只在**本进程第一次**用到这个 (会话, 类别) 时扫一次目录,之后走内存高水位 ——
    不是每帧都扫。`_session_lock` 已经把同会话的调用串起来了。
    """
    key = (session_id, kind)
    with _STATE_GUARD:
        seen = key in _COUNTERS
    hi = 0 if seen else _highest_seq_on_disk(session_id, kind)
    with _STATE_GUARD:
        nxt = max(_COUNTERS.get(key, 0), hi) + 1
        _COUNTERS[key] = nxt
        return nxt


def _current_seq(session_id: str, kind: str) -> int:
    return _COUNTERS.get((session_id, kind), 0)


def ledger_path(session_id: str, writer: str) -> Path:
    """某个写入者的账本文件。`writer` ∈ {"face", "gesture", "audio"}。"""
    if writer not in LEDGER_WRITERS:
        raise ValueError(f"未知写入者: {writer!r}(只允许 {LEDGER_WRITERS})")
    return recording_dir(session_id) / f"{LEDGER_PREFIX}.{writer}{LEDGER_SUFFIX}"


def ledger_files(session_id: str) -> list[Path]:
    """本会话**全部**账本文件(每个写入者一个)。读侧要把它们合起来看。"""
    d = recording_dir(session_id)
    return sorted(set(d.glob(f"{LEDGER_PREFIX}.*{LEDGER_SUFFIX}"))
                  | set(d.glob(f"{LEDGER_PREFIX}{LEDGER_SUFFIX}")))


def _append_jsonl(session_id: str, writer: str, record: dict) -> None:
    """追加一行账。**追加 + 单次 `os.write`**,进程被杀时已写下的行不会损坏。

    ⚠️ 为什么不能用 `with open(p, "a", encoding="utf-8") as f: f.write(...)`:
      ① **三个服务是三个独立进程**(face :8000 / gesture :8002 / voice :8001),
         而它们都要往同一个会话写账。进程内的 `threading.Lock` **跨不了进程**。
      ② 那种写法是**带缓冲**的,flush 时会把一次逻辑写拆成多次系统调用。
      两者一叠加,一行的字节会被另一个进程的行从中间插进来劈开。

    **实测代价**(2026-09-25,使用者第一场真会话):应予 390 行,实得 308 行,
    **其中 77 行是碎的** —— 坏行开头是 `449"}`、`_2449"}` 这类片段。
    素材本身没事(不同模态写不同目录),坏掉的是元数据。

    两道防线:
      ① **每个写入者一个文件**(`retention.<writer>.jsonl`)—— 从根上让两个进程不碰
         同一个文件;不依赖任何文件系统的锁语义(`~/shared` 是 9p/drvfs,flock 未必支持)。
      ② 单次 `os.write` 的 O_APPEND —— 行小于 4096 字节时一次写是原子的。
    """
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(ledger_path(session_id, writer),
                 os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


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
        if not _prepare(sid, modality):
            return None
        seq = _bump_seq(sid, modality)
        try:
            fname = f"{seq:06d}.jpg"
            (media_dir(sid, modality) / fname).write_bytes(data)
            rel = f"{MEDIA_SUBDIR}/{modality}/{fname}"
            rec = _record(sid, "frame", modality, seq, rel, data, declared_ts, source)
            _append_jsonl(sid, modality, rec)
            return rec
        except Exception as exc:
            _mark_degraded(sid, modality,
                           f"frame {modality}#{seq}: {type(exc).__name__}: {exc}")
            return None


def sniff_audio_ext(data: bytes) -> str:
    """按内容判容器。只认得出两种就够 —— 认不出就 `.bin`,不假装。"""
    if data[:4] == b"RIFF":
        return "wav"
    if data[:4] == b"\x1a\x45\xdf\xa3":       # EBML(webm/mkv)
        return "webm"
    return "bin"


def retain_audio(session_id: str, kind: str, data: bytes, source: str = "",
                 seq: int | None = None) -> dict | None:
    """把一段音频的**原始字节**存下来(kind: "raw" | "converted")。

    `converted` 是 `/asr` 经 ffmpeg 转出的 16k 单声道 WAV —— **那才是特征提取器真正
    读的 PCM**,所以它必须与原始上传一起留(spec §4 的"管线所见 + 原始上传都有据")。

    **`converted` 请显式传 `seq`**:把 `retain_audio("raw", …)` 返回的那一行的 `seq`
    传进来,配对就钉死在**本次请求那个号**上。不传则退回"此刻计数器的最新值" ——
    那在"两个 raw 交错、converted 晚到"时会错位(独立审查 Important 4):
    B 的 converted 会被后到的 A 覆盖,账本里两行指向同一文件、sha 不同。
    当前部署够不到(ffmpeg 是同步阻塞、且单 worker),但那是**没写下来的隐含前提**。
    """
    if not enabled():
        return None
    if kind not in ("raw", "converted"):
        raise ValueError(f"未知音频种类: {kind!r}(只允许 raw / converted)")
    sid = validate_session_id(session_id)
    with _session_lock(sid):
        if not _prepare(sid, "audio"):
            return None
        if kind == "converted":
            seq = seq or _current_seq(sid, "audio") or 1
            fname = f"{seq:04d}_converted.wav"
        else:
            seq = _bump_seq(sid, "audio")
            fname = f"{seq:04d}.{sniff_audio_ext(data)}"
        rel = f"{MEDIA_SUBDIR}/audio/{fname}"
        try:
            (media_dir(sid, "audio") / fname).write_bytes(data)
            rec = _record(sid, kind, None, seq, rel, data, None, source)
            _append_jsonl(sid, "audio", rec)
            return rec
        except Exception as exc:
            _mark_degraded(sid, "audio",
                           f"audio {kind}#{seq}: {type(exc).__name__}: {exc}")
            return None
