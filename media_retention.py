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
LEDGER_WRITERS = ("face", "gesture", "audio", "camera")

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


# ── 墓碑:被「不留存 / 不同意录制」作废掉的场次 ────────────────────────────
#
# 为什么需要:作废会**删掉**场次目录,而**后续任何一次写入都会把那个 sid 的目录
# 重新建出来**。2026-09-28 实测:删完之后又来了一次 `/session/<sid>/question`,
# 于是在录制根下留下一个只含 `questions.jsonl` 的裸目录 —— 而素材列表显示的就是
# 盘上的东西,于是"已经删掉的一场"又挂在那里。**"删了"与"看着删了"必须是一回事。**
#
# 印记是一个**空文件**。它不进素材列表(见 `recordings_browser._iter_sessions`),
# 但让所有**创建**路径拒绝再写(见 `assert_not_purged` 的两个调用点)。
TOMBSTONE_DIR_NAME = "_已作废"


def tombstone_path(session_id: str) -> Path:
    return root() / TOMBSTONE_DIR_NAME / validate_session_id(session_id)


def is_purged(session_id: str) -> bool:
    """这一场是不是被明确作废过。取值失败一律当**没作废**(读侧不该因它炸)。"""
    try:
        return tombstone_path(session_id).is_file()
    except (ValueError, OSError):
        return False


def mark_purged(session_id: str) -> None:
    p = tombstone_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")


def assert_not_purged(session_id: str) -> None:
    """**创建**素材前必须过这一关。两个调用点:`recording_dir` 与
    `session_meta._session_dir(create=True)` —— 那是全仓仅有的两条创建路径。"""
    if is_purged(session_id):
        raise ValueError(
            f"本场已被作废({session_id}):它的素材被明确删除过,不能再往里写。"
            f"要重录请铸一个新号(一次录制 = 一场)。")


def recording_dir(session_id: str) -> Path:
    assert_not_purged(session_id)
    d = resolve_recording_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── 目录名带本场标注:`<标签>__<sid>`(使用者 2026-09-27 裁定)──────────────────
#
# 为什么 sid 必须留在目录名**尾部**:`session_id` 同时是 `data/logs/` 里三份日志的
# 文件名,而报告侧的 `data_loader` 就按那个正则找日志。目录名可以改,但必须**还能从
# sid 推出来** —— 否则下面那些 `root()/<sid>` 的调用点(media_retention /
# transcript_store / session_meta / 重放工具)会集体失效,而失效的形态是
# **素材静默写不进去**,本仓最贵的那一类。

LABEL_DIR_SEP = "__"

# 目录名的字节上限。文件系统是 255 字节,而中文在 UTF-8 里 3 字节/字 ——
# 留出余量取 240。超了**拒**(见 `label_dir_name`),不悄悄不改名。
DIR_NAME_MAX_BYTES = 240

_UNSAFE_IN_PATH_COMPONENT = ("/", "\\", "\x00")


def _safe_path_component(value: str, what: str) -> str:
    """守卫:一段值要被拼进**目录名**,就得保证它自己不是路径。

    ⚠️ 与 `validate_session_id` 同一类风险、同一类守卫。标注是**客户端可控**的
    (走 HTTP 表单),而它现在会被拼成目录名 —— `../..` 之类的值能在录制根**之外**
    建目录。放它进来等于把刚焊好的那道门撬开一个口子。
    """
    if not isinstance(value, str):
        raise ValueError(f"{what} 必须是字符串,收到 {type(value).__name__}")
    v = value.strip()
    if not v:
        raise ValueError(f"{what} 是空的 —— 空的当不了目录名的一段")
    for ch in _UNSAFE_IN_PATH_COMPONENT:
        if ch in v:
            raise ValueError(f"{what} 里有路径分隔符或 NUL:{v!r} —— 它会被拼进目录名")
    if v.startswith("."):
        # 一并挡掉 `.` / `..` / `.hidden`
        raise ValueError(f"{what} 不能以点开头:{v!r}(`.`/`..`/隐藏名都不行)")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in v):
        raise ValueError(f"{what} 里有控制字符:{v!r}")
    return v


def label_dir_name(label: str, session_id: str) -> str:
    """`<标签>__<sid>`。标签本身过长/含路径分隔符时抛 `ValueError`(请求的毛病 → 400)。

    标签里出现 `__` 不影响解析:`resolve_recording_dir` 是按**后缀** `__<sid>` 认的,
    而 sid 唯一。
    """
    sid = validate_session_id(session_id)
    comp = _safe_path_component(label, "本场标注")
    name = f"{comp}{LABEL_DIR_SEP}{sid}"
    n = len(name.encode("utf-8"))
    if n > DIR_NAME_MAX_BYTES:
        raise ValueError(
            f"标注太长:与 session_id 拼成目录名后有 {n} 字节,超过上限 "
            f"{DIR_NAME_MAX_BYTES}(文件系统是 255;中文 3 字节/字)—— 请写短一点")
    return name


def resolve_recording_dir(session_id: str) -> Path:
    """定这一场的目录。**不建目录**(读侧不该有副作用)。

    三条路,按顺序:
    1. `root()/<sid>` 在 ⟹ 就用它。**历史素材(本轮之前那 56 场)照旧不动**,
       没标的场次也走这条。
    2. 否则找后缀 `__<sid>` 的目录 —— 那是存过标注、被改过名的。
       sid 唯一 ⟹ 至多一个。
    3. 都没有(这一场什么都还没落)⟹ 返回旧形态的路径,由调用方决定建不建。
    """
    sid = validate_session_id(session_id)
    base = root()
    legacy = base / sid
    if legacy.is_dir():
        return legacy
    suffix = f"{LABEL_DIR_SEP}{sid}"
    try:
        for p in sorted(base.iterdir()):
            if p.is_dir() and p.name.endswith(suffix):
                return p
    except (FileNotFoundError, NotADirectoryError):
        pass          # 录制根都还不存在 ⟹ 走第 3 条
    return legacy


def rename_session_dir_to_label(session_id: str, label: str) -> Path:
    """把这一场的目录改成 `<标签>__<sid>`。已经叫那个名就原样返回。

    调用时机在 `session_meta.upsert_label` 里,即**开录之前**(标注存完才开录)
    ⟹ 改名那一刻没有并发的写入方抢路径。
    ⚠️ 「录完改标签」那次改名**可能**与在录的帧撞上:face/gesture 每个请求都重新
    解析一次目录,所以撞上的是"刚解析完旧路径、目录就没了"那一瞬间 ——
    那几帧会被留存层记成降级(`degraded_reasons`),不会静默。
    """
    sid = validate_session_id(session_id)
    target = root() / label_dir_name(label, sid)
    current = resolve_recording_dir(sid)
    if current == target:
        return current
    if target.exists():
        raise ValueError(f"目录名已被占用:{target.name}")
    current.mkdir(parents=True, exist_ok=True)
    os.replace(current, target)      # 同一个盘 ⟹ 原子
    return target


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


def _prepare(session_id: str, writer: str,
             probe_parts: tuple[str, ...] | None = None) -> bool:
    """本次能不能写。首件素材预检不过 → **抛**;之后只留痕、返回 False。

    `probe_parts`:探针要落在**哪个目录**下,相对 `media/`。缺省 `None` = 拿
    `writer` 当子目录名(face / gesture / audio 的老行为,逐字节不变)。
    `camera` 传**空元组** —— 因为 `camera.webm` 直接落在 `media/` **下面**、
    没有 `media/camera/` 这一层(spec §4 的架构图);不传空元组的话,预检会
    **建出一个永远空的 `media/camera/` 目录**,而 §7.7 的人工核对清单里有一句
    "素材**文件数与模态数一致**" —— 多一个空目录正是在那种核对里制造困惑的东西。

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
        d = media_dir(session_id, *(writer,) if probe_parts is None else probe_parts)
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
               "declared_ts": None, "source_endpoint": "media_retention", "reason": why,
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
        "source_endpoint": source,
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


CAMERA_FILENAME = "camera.webm"


class CameraAlreadyRetained(RuntimeError):
    """本场已经有 `media/camera.webm` 了 —— 再传会把**已有的那份覆盖掉**。

    为什么是**拒绝**而不是"覆盖 + 账本留痕":原始素材不可再生。本项目花整个 M2.6
    留原始媒体,理由就是"分析代码必然有缺陷、算错了要能重算" —— 而被覆盖掉的那一份
    没有第二次机会。账本留两行 sha 只能证明"曾经有过一份",**救不回字节**。

    ★ 它成立的前提是调用方守着「**一次录制 = 一场会话**」。合法路径下这个异常
    永远不该被触发;一旦触发,说明上游把多次录制塞进了同一场 —— 那正是要报出来的事。

    2026-09-28 实盘:使用者在**一个页面里连录 9 个学生**,9 次上传挤进同一场,
    前 8 份原生录像被原地覆盖,丢 2.1 GB。当时的实现正是"覆盖 + 账本留痕"。
    """


def retain_uploaded_video(session_id: str, data: bytes,
                          source: str = "") -> dict | None:
    """把前端 `MediaRecorder` 录的**原生音视频**原样存成 `media/camera.webm`。

    它存在的理由(spec §1.2 的 R2):服务端那一腿留的是**管线解码过的帧**
    (AssessmentPage 下 5 fps,而微表情是 40–200 ms 的 onset–apex–offset 结构
    ⟹ 在 5 fps 下是伪测量),前端这一腿留的才是**帧率没被钉死的原生流**。
    两者不是冗余,是两件事。

    ★ **一个会话只许有一份 `camera.webm`;第二份会被拒绝**(`CameraAlreadyRetained`),
    不是覆盖。前提是调用方守着「**一次录制 = 一场会话**」(前端每次「开始录制」铸新号)。
    这个前提原来写的是"第二次上传覆盖它,账本留痕即可" —— **2026-09-28 实盘推翻了它**:
    使用者的用法是**一个页面连录多个学生**,于是 9 次录制挤进同一场,
    8 份原生录像被原地覆盖,丢了 2.1 GB —— 而账本那两行 sha 证明不了任何字节还在。

    `_prepare` 的探针传空元组:文件落在 `media/` 本身,没有 `media/camera/` 这一层。
    """
    if not enabled():
        return None
    sid = validate_session_id(session_id)
    with _session_lock(sid):
        # ⚠️ 这个守卫必须在下面那个 `try` 的**外面**。放进去的话 `except Exception`
        #    会把它当"写失败"吞成 degraded —— 于是"拒绝覆盖"变成静默降级,
        #    正是它要消灭的那个形态。
        target = resolve_recording_dir(sid) / MEDIA_SUBDIR / CAMERA_FILENAME
        if target.exists():
            raise CameraAlreadyRetained(
                f"本场已经有 {MEDIA_SUBDIR}/{CAMERA_FILENAME} 了"
                f"({target.stat().st_size} 字节)—— 拒绝覆盖。"
                f"一份原生录像不可再生;要再录请开新的一场(前端每次「开始录制」会铸新号)。")
        if not _prepare(sid, "camera", probe_parts=()):
            return None
        try:
            (media_dir(sid) / CAMERA_FILENAME).write_bytes(data)
            rel = f"{MEDIA_SUBDIR}/{CAMERA_FILENAME}"
            rec = _record(sid, "video", "camera", 1, rel, data, None, source)
            _append_jsonl(sid, "camera", rec)
            return rec
        except Exception as exc:
            _mark_degraded(sid, "camera",
                           f"video: {type(exc).__name__}: {exc}")
            return None
