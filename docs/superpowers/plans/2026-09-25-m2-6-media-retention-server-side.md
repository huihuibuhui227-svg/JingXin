# M2.6 服务端媒体留存 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 face / gesture / voice 三个服务在"收下"画面和音频的同时,把**收到的原始字节**原样存一份到仓库外的共享盘,这样跑完一场真会话后素材还在,事后能重新算。

**Architecture:** 新增一个仓库根的共用模块 `media_retention.py`(与 `logging_config.py` / `session_clock.py` 同层)。三个服务的端点在**已经解析出 session_id、已经拿到字节**之后,加一行调用把字节交给它。模块自己持路径、守卫、会话锁、序号;不跨服务 import(face/gesture 不 import voice 包)。留存是**旁路新增的一次写**,不是替换原来的解码来源 —— 关掉留存时行为与今天逐字节相同。

**Tech Stack:** Python 3(标准库 `hashlib` / `json` / `pathlib` / `threading`)、FastAPI(端点已存在,只加调用)、pytest。

**Spec:** `docs/superpowers/specs/2026-09-25-m2-6-raw-media-retention-design.md`(本计划实现它的 §1.1 服务端部分;前端那两条腿见该 spec §5.4/§5.6,不在本计划内)

## Global Constraints

- **解释器固定** `~/miniconda3/envs/jingxin/bin/python`(不可用 `~/huihuibuhui/bin/python`,缺依赖)。
- **落盘位置只许 `~/shared/jingxin_recordings/`**(→ `/mnt/d`,Windows 盘)。**不写仓库、不写 WSL ext4.vhdx** —— 后者只增不减(下一步 §4.7)。
- **原样字节,不转码不缩放不裁剪**(录制需求 §5)。
- **留存默认开启**,只有 `JINGXIN_RETAIN_MEDIA` 取 `0/false/no/off` 才关。
- **目录根可覆盖**:`JINGXIN_RECORDINGS_DIR`(与 `transcript_store` 同一开关)。
- **会话 id 只允许** `[A-Za-z0-9_-]{1,128}`;它是目录名,不接受路径分隔符与 `..`。
- **失败要说话**:首件素材预检不过 → **抛**(请求 500);中途某件写失败 → **不阻断分析**,留痕。**任何情况都不得静默**。
- **不碰前端**;不碰 `code_data_supplement/`(别人的论文复现包,git 未跟踪)。
- 提交时**只 `git add` 本任务改的文件** —— 工作树里有大量未跟踪文件(含 `code_data_supplement/`),`git add -A` 会把它们全带上。

## Review Focus

以下五类输入/情形,spec 没有逐条写死,但**使用者一定会碰上**。每条都在对应任务的测试里钉住:

1. **`JINGXIN_RECORDINGS_DIR` 指向不可写的路径**(磁盘满、权限错、目录被删)—— 期望:第一件素材就让请求**当场失败**,而不是安静地什么都不存。写错文件却在报告里表现为"数据对不上"是本项目反复踩过的形态。
2. **客户端不带 `session_id`(落 `NONE` 桶)** —— 期望:照存不误(`NONE` 是既有设计里的一个正常会话),但使用者要知道 `NONE/` 目录会**混入不同场次的素材**。
3. **同一会话多帧并发到达** —— 期望:文件序号不重、不交错,`retention.jsonl` 的行数与盘上文件数一致。
4. **一段音频既有原始容器又有转换后的 WAV**(`/asr` 走 ffmpeg 那条路)—— 期望:两者**共用同一个序号**,文件名可对应,不是两个互不相干的编号。
5. **服务被中断在"写了文件但还没写账"之间**(进程被杀)—— 期望:账本是**追加**写的,已写下的行不会因此损坏;重抽脚本以**盘上文件**为准而不是以账本为准。

---

### Task 1: `media_retention.py` —— 落点、守卫、开关、预检

**Files:**
- Create: `media_retention.py`(仓库根)
- Test: `tests/test_media_retention.py`

**Interfaces:**
- Consumes: 无(本任务不依赖别的任务)
- Produces:
  - `DEFAULT_ROOT: Path`、`RETENTION_FILENAME = "retention.jsonl"`、`MEDIA_SUBDIR = "media"`、`SESSION_ID_PAT: re.Pattern`
  - `root() -> Path`、`enabled() -> bool`
  - `validate_session_id(session_id: str) -> str`(非法抛 `ValueError`)
  - `validate_modality(modality: str) -> str`(只允许 `"face"` / `"gesture"`)
  - `recording_dir(session_id: str) -> Path`(不存在则建)
  - `retain_frame(session_id, modality, data, declared_ts=None, source="") -> dict | None`
  - `retain_audio(session_id, kind, data, source="") -> dict | None`
  - `degraded_reasons(session_id: str) -> list[str]`

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_media_retention.py`:

```python
# tests/test_media_retention.py
"""M2.6 服务端留存:路径/守卫/开关/预检这一层。

这一层是"素材会不会被静默丢掉"的唯一防线,所以每条测试都要能说出
**撤掉哪个生产改动会让它变红**。
"""
import hashlib
import json
import threading
from pathlib import Path

import pytest

import media_retention


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    """每个测试都落在自己的临时目录里 —— 绝不碰真的 ~/shared。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


def test_default_root_is_the_shared_volume():
    """默认落点必须是 ~/shared/jingxin_recordings(→ /mnt/d)。

    红法:把 DEFAULT_ROOT 改成仓库内的路径 —— 那会让原始媒体进 git,
    也会撑 WSL 的 ext4.vhdx(下一步 §4.7)。
    """
    assert media_retention.DEFAULT_ROOT == Path.home() / "shared" / "jingxin_recordings"


def test_enabled_by_default_and_can_be_switched_off(monkeypatch):
    """默认开 —— "忘了开"正是丢素材的那条路径(spec §6)。"""
    assert media_retention.enabled() is True
    for off in ("0", "false", "FALSE", "no", "off", " off "):
        monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", off)
        assert media_retention.enabled() is False, off


def test_path_traversal_is_rejected():
    """会话 id 是客户端可控的,而它会被拼进路径 —— 必须拦在拼路径之前。

    红法:去掉 validate_session_id 里的正则。`../../jingxin/x` 就能在录制根
    **之外**建目录并写入原始媒体。
    """
    for bad in ("../x", "a/b", "..", "", "  ", "a" * 129, "a.b"):
        with pytest.raises(ValueError):
            media_retention.validate_session_id(bad)
    assert media_retention.validate_session_id("20260925_124346_6e4a") == "20260925_124346_6e4a"
    assert media_retention.validate_session_id("NONE") == "NONE"


def test_recording_dir_honors_the_env_override(_isolated_root):
    d = media_retention.recording_dir("s1")
    assert d == _isolated_root / "s1"
    assert d.is_dir()


def test_preflight_is_loud_when_the_root_is_not_writable(tmp_path, monkeypatch):
    """首件素材预检不过 → **抛**,不是 warning。

    红法:把 _prepare 里的 raise 改成 return False —— 于是磁盘坏了也一路静默,
    跑完一场才发现一张图都没存(而这一场是人重跑不回来的)。
    """
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)                       # 只读目录
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(blocked / "sub"))
    media_retention._reset_for_tests()
    if media_retention._can_write_here(blocked):   # root 用户跑测试时 chmod 拦不住
        pytest.skip("本环境以 root 运行,chmod 拦不住写")
    with pytest.raises(RuntimeError, match="预检"):
        media_retention.retain_frame("s1", "face", b"\xff\xd8jpeg", declared_ts=0)


def test_preflight_failure_is_only_loud_once(tmp_path, monkeypatch):
    """首件大声报过之后,后续件不再抛 —— 中断也换不回已丢的字节,还白耗人的时间。

    但要**留痕**(见 Task 4)。红法:把 _LOUD_DONE 判断去掉 → 每一帧都 500,
    整场用不了。
    """
    blocked = tmp_path / "blocked2"
    blocked.mkdir()
    blocked.chmod(0o500)
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(blocked / "sub"))
    media_retention._reset_for_tests()
    if media_retention._can_write_here(blocked):
        pytest.skip("本环境以 root 运行,chmod 拦不住写")
    with pytest.raises(RuntimeError):
        media_retention.retain_frame("s1", "face", b"x", declared_ts=0)
    assert media_retention.retain_frame("s1", "face", b"x", declared_ts=1) is None
    assert media_retention.degraded_reasons("s1")
```

- [ ] **Step 2: 跑测试,确认红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention.py -v`
Expected: 全部 FAIL／ERROR,`ModuleNotFoundError: No module named 'media_retention'`

- [ ] **Step 3: 写最小实现**

新建 `media_retention.py`:

```python
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


def recording_dir(session_id: str) -> Path:
    d = root() / validate_session_id(session_id)
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

    spec §6 的两级:intent 是"半路才发现等于已经丢了一半素材,而这一场是人重跑不回来
    的",所以第一件必须当场响;但中断也换不回已丢的字节、还白耗人的时间,所以只响一次。
    """
    if session_id in _PREFLIGHTED:
        return True
    try:
        d = recording_dir(session_id)
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


def validate_modality(modality: str) -> str:
    """模态名也是目录名的一部分 —— 与 session_id 同一类守卫。

    放在本步(而不是用它的 Task 2)是因为它是**守卫**,和 `validate_session_id` 是
    同一层的东西;Task 1 的预检路径也要先过它。
    """
    if modality not in ("face", "gesture"):
        raise ValueError(f"未知模态: {modality!r}(只允许 face / gesture)")
    return modality


def retain_frame(session_id: str, modality: str, data: bytes,
                 declared_ts: int | None = None, source: str = "") -> dict | None:
    """把一帧的**原始字节**存下来。返回写进账本的那一行;没存则 None。

    存的是服务端收到的**同一份 bytes**,不重新编码、不缩放(录制需求 §5)。
    """
    # 守卫与预检在本步就位(两条预检测试压着它们);**落盘部分由 Task 2 补完**。
    if not enabled():
        return None
    sid = validate_session_id(session_id)
    validate_modality(modality)
    with _session_lock(sid):
        if not _prepare(sid):
            return None
    raise NotImplementedError("Task 2 实现落盘部分")


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
```

- [ ] **Step 4: 跑测试,确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention.py -v`
Expected: 6 passed

- [ ] **Step 5: 跑全量套件,确认没伤到别的**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest -q`
Expected: 全部通过(基线 254 passed 起)

- [ ] **Step 6: 提交**

```bash
cd ~/jingxin
git add media_retention.py tests/test_media_retention.py
git commit -m "feat(m2.6): 留存模块的落点/守卫/开关/预检 —— 首件素材不过就大声失败"
```

---

### Task 2: 存帧 `retain_frame`

**Files:**
- Modify: `media_retention.py`(替换 `retain_frame` 的 `NotImplementedError`)
- Test: `tests/test_media_retention.py`(追加)

**Interfaces:**
- Consumes: Task 1 的 `_prepare` / `_session_lock` / `_bump_seq` / `recording_dir` / `enabled` / `MEDIA_SUBDIR`
- Produces: `retain_frame(session_id, modality, data, declared_ts=None, source="") -> dict | None`,写 `media/<modality>/000001.jpg` 与 `retention.jsonl` 一行

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_media_retention.py`:

```python
# ── Task 2:存帧 ────────────────────────────────────────────────────────────

def _jsonl(root: Path, sid: str = "s1") -> list[dict]:
    p = root / sid / media_retention.RETENTION_FILENAME
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_frame_bytes_land_verbatim(_isolated_root):
    """盘上的字节必须与收到的**逐字节相同**,账本里的 sha256 要能证明这一点。

    红法:把 `path.write_bytes(data)` 换成一进一出的 `cv2.imencode/imdecode`
    —— 看起来"还存了一张图",但重抽出来的值不再等于当场算的值,而这是留存存在的
    全部理由(spec §7.1)。
    """
    payload = b"\xff\xd8\xff\xe0not-a-real-jpeg-but-bytes-are-bytes"
    rec = media_retention.retain_frame("s1", "face", payload, declared_ts=1500,
                                       source="/analyze")
    assert rec is not None
    f = _isolated_root / "s1" / "media" / "face" / "000001.jpg"
    assert f.read_bytes() == payload
    assert rec["sha256"] == hashlib.sha256(payload).hexdigest()
    assert rec["bytes"] == len(payload)
    assert rec["kind"] == "frame" and rec["modality"] == "face" and rec["seq"] == 1
    assert rec["declared_ts"] == 1500 and rec["source"] == "/analyze"
    assert rec["session_id"] == "s1"
    assert rec["file"] == "media/face/000001.jpg"
    assert isinstance(rec["received_at_wall"], float)


def test_sequence_is_zero_padded_so_name_order_is_time_order(_isolated_root):
    """`000001` 而不是 `1` —— 字典序即时间序,重抽脚本才能直接按文件名排。

    红法:去掉 `:06d` 里的补零。第 10 帧会排到第 2 帧前面。
    """
    for i in range(12):
        media_retention.retain_frame("s1", "face", b"x", declared_ts=i)
    names = sorted(p.name for p in (_isolated_root / "s1" / "media" / "face").iterdir())
    assert names[0] == "000001.jpg" and names[1] == "000002.jpg"
    assert names[9] == "000010.jpg"
    assert [json.loads(l)["seq"] for l in
            (_isolated_root / "s1" / media_retention.RETENTION_FILENAME)
            .read_text(encoding="utf-8").splitlines()] == list(range(1, 13))


def test_two_modalities_count_separately(_isolated_root):
    """face 与 gesture 各自从 1 开始 —— 它们是两条独立的字节流。"""
    a = media_retention.retain_frame("s1", "face", b"f")
    b = media_retention.retain_frame("s1", "gesture", b"g")
    assert a["seq"] == 1 and b["seq"] == 1
    assert (_isolated_root / "s1" / "media" / "gesture" / "000001.jpg").exists()


def test_retention_off_writes_nothing(_isolated_root, monkeypatch):
    """关掉时**一个文件都不写**,且不报错(它是旁路,关了就该与今天一样)。"""
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    assert media_retention.retain_frame("s1", "face", b"x") is None
    assert not (_isolated_root / "s1" / "media").exists()
    assert media_retention.degraded_reasons("s1") == []


def test_none_bucket_is_retained_like_any_other_id(_isolated_root):
    """`NONE` 是既有设计里的一个正常会话(transcript_store 也给它建目录)。

    ⚠️ 代价要写在这里:不同场次、不同人的素材会**混进同一个 `NONE/` 目录**
    (与下一步 §3 第 11 条的既有问题同源)。真要区分只能靠 `received_at_wall`。
    """
    rec = media_retention.retain_frame("NONE", "face", b"x")
    assert rec is not None and rec["session_id"] == "NONE"


def test_concurrent_frames_do_not_collide(_isolated_root):
    """同一会话并发 40 帧 → 40 个文件、40 行账、序号无重复。

    红法:去掉 _session_lock 的 with —— "取序号→写文件→写账"三步会交错,
    出现同名文件互相覆盖(盘上文件数 < 账本行数)。
    """
    def burst():
        for _ in range(10):
            media_retention.retain_frame("s1", "face", b"x")
    ts = [threading.Thread(target=burst) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    files = list((_isolated_root / "s1" / "media" / "face").iterdir())
    assert len(files) == 40
    seqs = [json.loads(l)["seq"] for l in
            (_isolated_root / "s1" / media_retention.RETENTION_FILENAME)
            .read_text(encoding="utf-8").splitlines()]
    assert sorted(seqs) == list(range(1, 41))
```

- [ ] **Step 2: 跑测试,确认红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention.py -v -k frame or sequence or modalities or off_writes or none_bucket or concurrent`
Expected: 全部 FAIL,`NotImplementedError`

- [ ] **Step 3: 写实现**

把 `media_retention.py` 里的 `retain_frame` 换成:

```python
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
        if not _prepare(sid):
            return None
        seq = _bump_seq(sid, modality)
        rel = f"{MEDIA_SUBDIR}/{modality}/{seq:06d}.jpg"
        try:
            (recording_dir(sid) / rel).write_bytes(data)
            rec = _record(sid, "frame", modality, seq, rel, data, declared_ts, source)
            _append_jsonl(sid, rec)
            return rec
        except Exception as exc:
            _mark_degraded(sid, f"frame {modality}#{seq}: {type(exc).__name__}: {exc}")
            return None


def validate_modality(modality: str) -> str:
    """模态名也是目录名的一部分 —— 与 session_id 同一类守卫。"""
    if modality not in ("face", "gesture"):
        raise ValueError(f"未知模态: {modality!r}(只允许 face / gesture)")
    return modality
```

> `validate_modality` 在 Task 1 里已经就位,本步直接用它。

- [ ] **Step 4: 跑测试,确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention.py -v`
Expected: 12 passed

- [ ] **Step 5: 提交**

```bash
cd ~/jingxin
git add media_retention.py tests/test_media_retention.py
git commit -m "feat(m2.6): 存帧 —— 原样字节 + sha256 记账 + 序号零填充"
```

---

### Task 3: 存音频 `retain_audio`

**Files:**
- Modify: `media_retention.py`(替换 `retain_audio` 的 `NotImplementedError`)
- Test: `tests/test_media_retention.py`(追加)

**Interfaces:**
- Consumes: Task 1/2 的 `_prepare` / `_session_lock` / `_bump_seq` / `_current_seq` / `_append_jsonl` / `_record`
- Produces: `retain_audio(session_id, kind, data, source="") -> dict | None`;`sniff_audio_ext(data) -> str`

- [ ] **Step 1: 写失败的测试**

```python
# ── Task 3:存音频 ──────────────────────────────────────────────────────────

def test_raw_and_converted_share_one_sequence_number(_isolated_root):
    """一段回答的原始容器与转换后的 WAV **共用同一个序号**。

    为什么:它们说的是同一段话,文件名要能对上(`0001.webm` ↔ `0001_converted.wav`)。
    红法:让 converted 也走 _bump_seq —— 变成 0001.webm 与 0002_converted.wav,
    两件互不相干的编号,重抽时对不上是哪段回答。
    """
    webm = b"\x1a\x45\xdf\xa3fake-webm"
    wav = b"RIFF" + b"\x00" * 60
    r1 = media_retention.retain_audio("s1", "raw", webm, source="/asr")
    r2 = media_retention.retain_audio("s1", "converted", wav, source="/asr")
    assert r1["seq"] == 1 and r2["seq"] == 1
    assert r1["file"] == "media/audio/0001.webm"
    assert r2["file"] == "media/audio/0001_converted.wav"
    assert (_isolated_root / "s1" / r1["file"]).read_bytes() == webm
    assert (_isolated_root / "s1" / r2["file"]).read_bytes() == wav


def test_second_answer_gets_sequence_two(_isolated_root):
    media_retention.retain_audio("s1", "raw", b"RIFF" + b"\x00" * 60)
    r = media_retention.retain_audio("s1", "raw", b"\x1a\x45\xdf\xa3x")
    assert r["seq"] == 2 and r["file"] == "media/audio/0002.webm"


def test_extension_is_sniffed_from_content_not_assumed(_isolated_root):
    """扩展名由**内容**决定 —— 调用方不必知道自己在送什么容器。

    红法:写死 `.webm`。`/interview/answer_audio` 送的是 WAV,会被存成
    `0001.webm`,重抽时按扩展名选的解码器全错。
    """
    assert media_retention.sniff_audio_ext(b"RIFF....WAVEfmt ") == "wav"
    assert media_retention.sniff_audio_ext(b"\x1a\x45\xdf\xa3\x01\x00") == "webm"
    assert media_retention.sniff_audio_ext(b"\x00\x01\x02") == "bin"
    r = media_retention.retain_audio("s1", "raw", b"RIFF" + b"\x00" * 60)
    assert r["file"].endswith(".wav")


def test_audio_off_writes_nothing(_isolated_root, monkeypatch):
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    assert media_retention.retain_audio("s1", "raw", b"RIFF" + b"\x00" * 60) is None
```

- [ ] **Step 2: 跑测试,确认红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention.py -v -k audio or sniff or answer`
Expected: FAIL,`NotImplementedError` / `AttributeError: sniff_audio_ext`

- [ ] **Step 3: 写实现**

```python
def sniff_audio_ext(data: bytes) -> str:
    """按内容判容器。只认得出两种就够 —— 认不出就 `.bin`,不假装。"""
    if data[:4] == b"RIFF":
        return "wav"
    if data[:4] == b"\x1a\x45\xdf\xa3":       # EBML(webm/mkv)
        return "webm"
    return "bin"


def retain_audio(session_id: str, kind: str, data: bytes, source: str = "") -> dict | None:
    """把一段音频的**原始字节**存下来(kind: "raw" | "converted")。

    `converted` 是 `/asr` 经 ffmpeg 转出的 16k 单声道 WAV —— **那才是特征提取器真正
    读的 PCM**,所以它必须与原始上传一起留(spec §4 的"管线所见 + 原始上传都有据")。
    """
    if not enabled():
        return None
    if kind not in ("raw", "converted"):
        raise ValueError(f"未知音频种类: {kind!r}(只允许 raw / converted)")
    sid = validate_session_id(session_id)
    with _session_lock(sid):
        if not _prepare(sid):
            return None
        if kind == "converted":
            # 与同一段回答的 raw 共用序号;若调用方先存 converted(不该发生),退回 1。
            seq = _current_seq(sid, "audio") or 1
            rel = f"{MEDIA_SUBDIR}/audio/{seq:04d}_converted.wav"
        else:
            seq = _bump_seq(sid, "audio")
            rel = f"{MEDIA_SUBDIR}/audio/{seq:04d}.{sniff_audio_ext(data)}"
        try:
            (recording_dir(sid) / rel).write_bytes(data)
            rec = _record(sid, kind, None, seq, rel, data, None, source)
            _append_jsonl(sid, rec)
            return rec
        except Exception as exc:
            _mark_degraded(sid, f"audio {kind}#{seq}: {type(exc).__name__}: {exc}")
            return None
```

- [ ] **Step 4: 跑测试,确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention.py -v`
Expected: 16 passed

- [ ] **Step 5: 提交**

```bash
cd ~/jingxin
git add media_retention.py tests/test_media_retention.py
git commit -m "feat(m2.6): 存音频 —— raw 与 converted 共用序号,扩展名按内容嗅探"
```

---

### Task 4: 中途失败留痕(`degraded`)+ 源码级契约

**Files:**
- Modify: `media_retention.py`(无需改动,若 Task 1–3 写法正确;本任务只补测试与一处收口)
- Test: `tests/test_media_retention.py`(追加)

**Interfaces:**
- Consumes: `degraded_reasons` / `_mark_degraded`
- Produces: 无新接口(本任务是"证明前三个任务真的会留痕")

- [ ] **Step 1: 写失败的测试**

```python
# ── Task 4:中途失败留痕 + 守卫副本的契约 ────────────────────────────────────

def test_midway_write_failure_is_recorded_and_does_not_raise(_isolated_root, monkeypatch):
    """写盘中途失败 → **不抛**(不阻断分析)、但必须留痕。

    红法:把 except 里的 _mark_degraded 去掉、只 return None —— 于是这一场静默地
    少了一批素材,而报告里什么都看不出来。
    """
    media_retention.retain_frame("s1", "face", b"ok1")     # 先过预检
    calls = {"n": 0}
    real = Path.write_bytes

    def flaky(self, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky)
    assert media_retention.retain_frame("s1", "face", b"ok2") is None   # 不抛
    assert any("No space left" in r for r in media_retention.degraded_reasons("s1"))
    monkeypatch.undo()
    # 后续件照常能写(留痕之后不瘫痪)
    assert media_retention.retain_frame("s1", "face", b"ok3") is not None


def test_degraded_reasons_is_empty_when_all_is_well(_isolated_root):
    media_retention.retain_frame("s1", "face", b"x")
    assert media_retention.degraded_reasons("s1") == []
    assert media_retention.degraded_reasons("never-seen") == []


def test_guard_regex_matches_transcript_store():
    """本模块的守卫与 `transcript_store` 必须是**同一个口径**。

    红法:只改一份(B 本模块放宽一位长度,或允许点号)—— 同一个客户端请求在
    "原句落哪"与"像素落哪"两处得到不同判定。
    与三份 `_resolve_session_id` 的源码级契约测试同一手法。
    """
    from voice_interaction.asr import transcript_store
    assert media_retention.SESSION_ID_PAT.pattern == transcript_store.SESSION_ID_PAT.pattern
```

- [ ] **Step 2: 跑测试**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention.py -v -k midway or empty or regex`
Expected: 3 passed

> 若 `test_midway_write_failure_…` 红:说明 Task 2/3 的 `except` 没写全 —— 补上再跑。

- [ ] **Step 3: 跑全量 + 提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest -q`
Expected: 全部通过

```bash
cd ~/jingxin
git add tests/test_media_retention.py
git commit -m "test(m2.6): 中途写失败要留痕不阻断;守卫与 transcript_store 同口径"
```

---

### Task 5: face 服务接入

**Files:**
- Modify: `face_expression/api/app.py`(端点 `analyze_frame`,`:282-287` 那一带;文件顶部 import 区)
- Test: `tests/test_media_retention_face.py`(新建)

**Interfaces:**
- Consumes: Task 1–4 的 `media_retention.retain_frame`
- Produces: 无新接口(端点行为变化:每帧多一次落盘)

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_media_retention_face.py`。它复用 `tests/test_analyze_session_fallback.py` 已经验证过的替身手法(`importlib.import_module` + `_FakeUpload` + `_FakeRequest` + 假的 pipeline):

```python
# tests/test_media_retention_face.py
"""M2.6:face 端点真的把收到的字节交给留存了(接线正确)。

为什么单独一个文件:这是一个**接线**测试 —— 模块本身在 test_media_retention.py 里
已经测透,这里只回答"端点在正确的时机、用正确的参数调了它吗"。
"""
import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

import pytest


def _install_env_shims():
    if "python_multipart" not in sys.modules:
        stub = types.ModuleType("python_multipart")
        stub.__version__ = "0.0.20"
        sys.modules["python_multipart"] = stub


_install_env_shims()

import media_retention                                              # noqa: E402
face_app = importlib.import_module("face_expression.api.app")       # noqa: E402


class _FakeUpload:
    def __init__(self, data: bytes):
        self._data, self.filename, self.content_type = data, "frame.jpg", "image/jpeg"

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    async def form(self):
        raise RuntimeError('Form data requires "python-multipart" to be installed.')


class _FakePipeline:
    def __init__(self, session_id=None):
        self.session_id = session_id

    def process_frame(self, _image_rgb, timestamp_ms):
        return object(), None, {"timestamp": timestamp_ms / 1000.0, "focus_score": 0.5}

    def measured_fps(self):
        return 1.0

    def close(self):
        pass


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    monkeypatch.setattr(face_app, "get_or_create_pipeline", lambda sid: _FakePipeline(sid))
    monkeypatch.setattr(face_app, "cv2", types.SimpleNamespace(
        imread=lambda p: object(), cvtColor=lambda i, c: i, COLOR_BGR2RGB=1))
    monkeypatch.setattr(face_app, "session_loggers", {})
    return tmp_path


def _post(payload: bytes, sid: str):
    return asyncio.run(face_app.analyze_frame(
        request=_FakeRequest(), file=_FakeUpload(payload), session_id=sid, fps=30))


def test_endpoint_hands_the_bytes_to_retention(wired):
    """端点收到的字节,必须原样出现在盘上,且账本里 hashed 一致。

    红法:删掉端点里那一行 retain_frame —— 本测试立刻红(盘上没有文件),
    而"特征照样算得出来"这个假象会让问题一路滑到重抽那天才暴露。
    """
    payload = b"\xff\xd8\xff\xe0" + b"A" * 100
    _post(payload, "20260925_120000_aaaa")
    f = wired / "20260925_120000_aaaa" / "media" / "face" / "000001.jpg"
    assert f.read_bytes() == payload


def test_declared_ts_matches_the_session_clock(wired):
    """账本里的 `declared_ts` 必须是**服务端给这一帧的那个时间戳**(M2.5 的会话相对时钟)。

    为什么重要:重抽时必须喂当时那个值,不能用墙钟 —— 否则重抽出来的数不等于当场
    算的数,而"逐格相等"正是留存的验收判据(spec §7.1)。
    红法:把 declared_ts 换成 time.time()。
    """
    _post(b"\xff\xd8jpeg", "20260925_120000_bbbb")
    line = json.loads((wired / "20260925_120000_bbbb" / media_retention.RETENTION_FILENAME)
                      .read_text(encoding="utf-8").splitlines()[0])
    # 首帧 → 会话相对时钟约等于 0(不写死 == 0:机器慢时可能已经过了 1 ms)
    assert 0 <= line["declared_ts"] < 1000
    assert line["source"] == "/analyze"
    assert line["declared_ts"] != int(line["received_at_wall"])   # 不是墙钟


def test_retention_off_leaves_no_files_and_still_analyzes(wired, monkeypatch):
    """关掉留存 → 盘上什么都没有,但**请求照常成功**(它是旁路)。

    红法:把端点里的调用写在 if enabled() 之外、或让留存失败冒泡成 500 ——
    关掉留存就整场用不了。
    """
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    r = _post(b"\xff\xd8jpeg", "20260925_120000_cccc")
    assert not (wired / "20260925_120000_cccc" / "media").exists()
    assert r is not None
```

- [ ] **Step 2: 跑测试,确认红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention_face.py -v`
Expected: FAIL —— `FileNotFoundError`(盘上没有文件)

- [ ] **Step 3: 改端点**

在 `face_expression/api/app.py` 顶部 import 区(`from session_clock import SessionClock` 那一行下面)加:

```python
import media_retention
```

在 `analyze_frame` 里,把 `:282-287` 这一段:

```python
            # 获取或创建 VideoPipeline
            pipeline = get_or_create_pipeline(session_id)
            timestamp_ms = _clock_for(session_id).stamp_ms()

            # 处理帧
            result_obj, mesh_results, features_dict = pipeline.process_frame(
                image_rgb, timestamp_ms)
```

改成(只**插入**两行,不动原有语句的顺序):

```python
            # 获取或创建 VideoPipeline
            pipeline = get_or_create_pipeline(session_id)
            timestamp_ms = _clock_for(session_id).stamp_ms()
            # M2.6:先把这一帧的原始字节存一份,再算 —— 算的过程中抛异常也不丢素材。
            # 留存关掉时(默认开)这里什么都不做,行为与 M2.6 之前逐字节相同。
            media_retention.retain_frame(session_id, "face", contents,
                                         declared_ts=timestamp_ms, source="/analyze")

            # 处理帧
            result_obj, mesh_results, features_dict = pipeline.process_frame(
                image_rgb, timestamp_ms)
```

> 注意:`contents` 在 `with tempfile.NamedTemporaryFile(...)` 块里赋值,出了块仍在作用域内(Python 的 `with` 不引入新作用域)。**不要**改成再 `await file.read()` 一次 —— 请求体只能消费一次。

- [ ] **Step 4: 跑测试,确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention_face.py -v`
Expected: 3 passed

- [ ] **Step 5: 跑全量套件**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest -q`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
cd ~/jingxin
git add face_expression/api/app.py tests/test_media_retention_face.py
git commit -m "feat(m2.6): face 端点把收到的帧原样交给留存"
```

---

### Task 6: gesture 服务接入

**Files:**
- Modify: `gesture_analysis/api/app.py`(端点 `analyze_image`,`:297` 那一带;文件顶部 import 区)
- Test: `tests/test_media_retention_gesture.py`(新建)

**Interfaces:**
- Consumes: `media_retention.retain_frame`
- Produces: 无新接口

- [ ] **Step 1: 写失败的测试**

新建 `tests/test_media_retention_gesture.py`,与 Task 5 同一手法(gesture 侧已经有 `tests/test_gesture_detector_wiring.py` 用的替身套路):

```python
# tests/test_media_retention_gesture.py
"""M2.6:gesture 端点真的把收到的字节交给留存了。"""
import asyncio
import importlib
import json
import sys
import types

import pytest


def _install_env_shims():
    if "python_multipart" not in sys.modules:
        stub = types.ModuleType("python_multipart")
        stub.__version__ = "0.0.20"
        sys.modules["python_multipart"] = stub


_install_env_shims()

import media_retention                                              # noqa: E402
gesture_app = importlib.import_module("gesture_analysis.api.app")   # noqa: E402


class _FakeUpload:
    def __init__(self, data: bytes):
        self._data, self.filename, self.content_type = data, "f.jpg", "image/jpeg"

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    async def form(self):
        raise RuntimeError("no multipart")


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    monkeypatch.setattr(gesture_app, "cv2", types.SimpleNamespace(
        imdecode=lambda buf, flag: object(), cvtColor=lambda i, c: i, IMREAD_COLOR=1,
        COLOR_BGR2RGB=1))
    monkeypatch.setattr(gesture_app, "np", types.SimpleNamespace(
        frombuffer=lambda b, t: b, uint8="u8"))
    monkeypatch.setattr(gesture_app, "get_or_create_analyzers", lambda sid: {})
    monkeypatch.setattr(gesture_app, "get_or_create_detectors", lambda sid: {
        "hands": types.SimpleNamespace(detect=lambda img, ts: []),
        "pose": types.SimpleNamespace(detect=lambda img, ts: []),
    })
    return tmp_path


def test_gesture_endpoint_hands_the_bytes_to_retention(wired):
    """红法:删掉端点里那一行 retain_frame —— 手势素材从此静默不留。"""
    payload = b"\xff\xd8\xff\xe0" + b"G" * 77
    asyncio.run(gesture_app.analyze_image(
        request=_FakeRequest(), file=_FakeUpload(payload), session_id="20260925_120000_dddd"))
    f = wired / "20260925_120000_dddd" / "media" / "gesture" / "000001.jpg"
    assert f.read_bytes() == payload


def test_gesture_declared_ts_comes_from_its_own_clock(wired):
    """gesture 有自己的会话时钟(M2.5 Task 6 加的),不是复用 face 的。"""
    asyncio.run(gesture_app.analyze_image(
        request=_FakeRequest(), file=_FakeUpload(b"\xff\xd8jpeg"),
        session_id="20260925_120000_eeee"))
    line = json.loads((wired / "20260925_120000_eeee" / media_retention.RETENTION_FILENAME)
                      .read_text(encoding="utf-8").splitlines()[0])
    assert 0 <= line["declared_ts"] < 1000          # 首帧 ≈ 0(不写死 == 0,机器慢时可能过了 1 ms)
    assert line["source"] == "/analyze"
```

- [ ] **Step 2: 跑测试,确认红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention_gesture.py -v`
Expected: FAIL —— `FileNotFoundError`

- [ ] **Step 3: 改端点**

`gesture_analysis/api/app.py` 顶部 import 区加 `import media_retention`。

把 `:295-299` 这一段:

```python
        # M2.5:时间戳由服务端实测 —— gesture 以前连 fps 参数都没有,直接用默认 30,
        # 而客户端实际 1 帧/秒(spec §3.5 / §3.1)。
        timestamp_ms = _clock_for(session_id).stamp_ms()

        hand_groups = dets['hands'].detect(image_rgb, timestamp_ms)
```

改成:

```python
        # M2.5:时间戳由服务端实测 —— gesture 以前连 fps 参数都没有,直接用默认 30,
        # 而客户端实际 1 帧/秒(spec §3.5 / §3.1)。
        timestamp_ms = _clock_for(session_id).stamp_ms()
        # M2.6:先存原始字节再算(与 face 同一处、同一理由)。
        media_retention.retain_frame(session_id, "gesture", contents,
                                     declared_ts=timestamp_ms, source="/analyze")

        hand_groups = dets['hands'].detect(image_rgb, timestamp_ms)
```

- [ ] **Step 4: 跑测试,确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention_gesture.py -v`
Expected: 2 passed

- [ ] **Step 5: 跑全量 + 提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest -q`

```bash
cd ~/jingxin
git add gesture_analysis/api/app.py tests/test_media_retention_gesture.py
git commit -m "feat(m2.6): gesture 端点把收到的帧原样交给留存"
```

---

### Task 7: voice 服务接入(三个端点)

**Files:**
- Modify: `voice_interaction/api/app.py`(`/asr` `:200` 与 `:265-268`;`/interview/answer_audio` `:359`;`/research/answer_audio` 同上;顶部 import 区)
- Test: `tests/test_media_retention_voice.py`(新建)

**Interfaces:**
- Consumes: `media_retention.retain_audio`
- Produces: 无新接口

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_media_retention_voice.py
"""M2.6:voice 的两个回答端点把 WAV 原样交给留存。"""
import asyncio
import importlib
import io
import json
import struct
import sys
import types
import wave

import pytest

if "python_multipart" not in sys.modules:
    stub = types.ModuleType("python_multipart")
    stub.__version__ = "0.0.20"
    sys.modules["python_multipart"] = stub

import media_retention                                              # noqa: E402
voice_app = importlib.import_module("voice_interaction.api.app")    # noqa: E402


def _wav_bytes(n_bytes: int = 3200) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<%dh" % (n_bytes // 2), *([0] * (n_bytes // 2))))
    return buf.getvalue()


class _FakeUpload:
    def __init__(self, data: bytes):
        self._data, self.filename, self.content_type = data, "a.wav", "audio/wav"

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    async def form(self):
        raise RuntimeError("no multipart")


class _FakeUtt:
    text = "我 觉得 这个 问题 很 有意思"


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    monkeypatch.setattr(voice_app, "_transcribe_async",
                        lambda audio_data: _async_return(_FakeUtt()))
    monkeypatch.setattr(voice_app.transcript_store, "append_utterance",
                        lambda sid, utt, **kw: None)
    monkeypatch.setattr(voice_app, "log_recognition", lambda utt: None)
    monkeypatch.setattr(voice_app.voice_logger, "log_prosody", lambda *a, **kw: None)
    monkeypatch.setattr(voice_app.interview_assessment, "add_answer", lambda t: None)
    monkeypatch.setattr(voice_app.interview_assessment, "save_log", lambda: None)
    monkeypatch.setattr(voice_app.interview_assessment, "qa_pairs", [])
    return tmp_path


async def _async_return(v):
    return v


def test_interview_answer_audio_is_retained(wired):
    """红法:删掉端点里的 retain_audio —— 音频从此不留,而特征照样算得出来。"""
    payload = _wav_bytes()
    asyncio.run(voice_app.submit_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(payload),
        session_id="20260925_120000_ffff"))
    f = wired / "20260925_120000_ffff" / "media" / "audio" / "0001.wav"
    assert f.read_bytes() == payload


def test_two_answers_get_sequence_one_and_two(wired):
    for _ in range(2):
        asyncio.run(voice_app.submit_answer_audio(
            request=_FakeRequest(), audio=_FakeUpload(_wav_bytes()),
            session_id="20260925_120000_gggg"))
    d = wired / "20260925_120000_gggg" / "media" / "audio"
    assert sorted(p.name for p in d.iterdir()) == ["0001.wav", "0002.wav"]
```

- [ ] **Step 2: 跑测试,确认红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention_voice.py -v`
Expected: FAIL —— `FileNotFoundError`

- [ ] **Step 3: 改端点**

`voice_interaction/api/app.py` 顶部 import 区加 `import media_retention`。

**① `/interview/answer_audio`**(`:359` 拿到 `contents` 之后、校验通过之前就存,这样格式不合法的上传也留得下证据):

```python
        # 复用 /asr 逻辑
        contents = await audio.read()
        # M2.6:先存原始字节 —— 即使下面判格式不合法,这份上传也留了据。
        media_retention.retain_audio(sid, "raw", contents, source="/interview/answer_audio")
        if not contents.startswith(b'RIFF'):
            raise HTTPException(status_code=400, detail="仅支持 WAV 格式音频")
```

**② `/research/answer_audio`**:同样一行,`source="/research/answer_audio"`。

**③ `/asr`**:两处。

在 `:200-201`(`contents = await audio.read()` 之后):

```python
        contents = await audio.read()
        logger.info(f"读取音频数据: {len(contents)} bytes")
        media_retention.retain_audio(sid, "raw", contents, source="/asr")
```

在 `:265-268`(ffmpeg 转换成功、读到 PCM 之后),把转换产物也留一份 —— 那才是提取器真正读的:

```python
            logger.info("转换成功")

            # M2.6:留一份转换后的 16k 单声道 WAV(提取器真正读的就是它);
            # 与同一段回答的 raw 共用序号(spec §4「管线所见 + 原始上传都有据」)。
            with open(output_path, "rb") as fh:
                media_retention.retain_audio(sid, "converted", fh.read(), source="/asr")

            # 读取转换后的 WAV 文件
            with wave.open(output_path, 'rb') as wf:
```

> ⚠️ `retain_audio("converted", …)` 必须在那个 `finally` 删临时文件**之前**调用(`:281-285` 会把 `output_path` 删掉)。上面的位置就在它前面。

- [ ] **Step 4: 跑测试,确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention_voice.py -v`
Expected: 2 passed

- [ ] **Step 5: 跑全量 + 提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest -q`

```bash
cd ~/jingxin
git add voice_interaction/api/app.py tests/test_media_retention_voice.py
git commit -m "feat(m2.6): voice 三个端点留下原始音频与转换后的 WAV"
```

---

### Task 8: 重抽回放器 + 端到端验收

**Files:**
- Create: `experiments/replay_retained.py`
- Test: `tests/test_replay_retained.py`(新建,只钉纯函数那部分)

**Interfaces:**
- Consumes: `media_retention.RETENTION_FILENAME` 的账本格式、`VideoPipeline.process_frame(frame, timestamp_ms)`
- Produces: `load_frames(session_dir) -> list[tuple[Path, int]]`(按文件名排序的 (帧文件, declared_ts) 列表)

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_replay_retained.py
"""M2.6 验收工具:把留存的帧按**当时的时间戳**排好序交给重抽。"""
import json
from pathlib import Path

from experiments.replay_retained import load_frames
import media_retention


def _mk(root: Path, sid: str, entries):
    d = root / sid
    (d / "media" / "face").mkdir(parents=True)
    lines = []
    for seq, (name, ts) in enumerate(entries, start=1):
        (d / "media" / "face" / name).write_bytes(b"x")
        lines.append(json.dumps({"kind": "frame", "modality": "face", "seq": seq,
                                 "file": f"media/face/{name}", "declared_ts": ts,
                                 "bytes": 1, "sha256": "", "received_at_wall": 0.0,
                                 "source": "/analyze", "session_id": sid}))
    (d / media_retention.RETENTION_FILENAME).write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
    return d


def test_frames_come_back_in_time_order_with_their_own_timestamps(tmp_path):
    """回放要按**账本里的 declared_ts**,不是"文件的修改时间"、也不是重新计时。

    红法:改成用 index×200ms 重新造时间戳 —— 重抽出来的值不再等于当场算的值,
    而"逐格相等"正是留存的验收判据(spec §7.1)。
    """
    d = _mk(tmp_path, "s1", [("000001.jpg", 0), ("000002.jpg", 1200),
                             ("000010.jpg", 9000)])
    got = load_frames(d)
    assert [p.name for p, _ in got] == ["000001.jpg", "000002.jpg", "000010.jpg"]
    assert [ts for _, ts in got] == [0, 1200, 9000]


def test_frames_are_read_from_disk_not_from_the_ledger(tmp_path):
    """以**盘上文件**为准:账本里有一行但文件不在(进程被杀在两步之间)→ 跳过它。

    红法:改成遍历账本 —— 于是遇到坏行就崩在"读一个不存在的文件"上,
    而那正是"服务被中断"必然产生的形态。
    """
    d = _mk(tmp_path, "s1", [("000001.jpg", 0), ("000002.jpg", 100)])
    (d / "media" / "face" / "000002.jpg").unlink()
    got = load_frames(d)
    assert [p.name for p, _ in got] == ["000001.jpg"]
```

- [ ] **Step 2: 跑测试,确认红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_replay_retained.py -v`
Expected: FAIL —— `ModuleNotFoundError: experiments.replay_retained`

- [ ] **Step 3: 写脚本**

新建 `experiments/replay_retained.py`:

```python
"""把留存下来的帧按**当时的时间戳**重放一遍 —— M2.6 的验收工具。

为什么需要一个专门的脚本(spec §7.1):留存的意义是"事后能重算"。要证明这一点,
就得拿盘上的帧重新跑一遍提取器,看结果是不是与**当场活跑**写下的 CSV 逐格相等。
相等 ⟹ 存的确实是我抽过的那些东西;不等 ⟹ 存错了(重编码?顺序错?时间戳造错了?)。

用法(WSL 侧):
    PY=~/miniconda3/envs/jingxin/bin/python
    $PY experiments/replay_retained.py --session-id 20260925_120000_aaaa \
        --out ~/shared/jingxin_recordings/20260925_120000_aaaa/replay_face.csv
然后把 replay_face.csv 与 data/logs/face_au_log_<sid>.csv 逐格比。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from media_retention import RETENTION_FILENAME, recording_dir   # noqa: E402


def load_frames(session_dir: Path | str, modality: str = "face") -> list[tuple[Path, int]]:
    """读出本会话该模态的帧,按序号排好,带上**账本里记的当时时间戳**。

    以**盘上的文件**为准,不以账本为准:进程被杀在"写了文件、还没写账"之间时,
    账本会缺行;反过来若账本有行而文件不在,那一行直接跳过而不是崩。
    """
    session_dir = Path(session_dir)
    book = session_dir / RETENTION_FILENAME
    out: list[tuple[Path, int, int]] = []
    if book.exists():
        for line in book.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                      # 半行(被杀在半途)→ 跳过,不崩
            if rec.get("kind") != "frame" or rec.get("modality") != modality:
                continue
            p = session_dir / rec["file"]
            if not p.exists():
                continue
            out.append((p, int(rec["seq"]), int(rec.get("declared_ts") or 0)))
    out.sort(key=lambda t: t[1])
    return [(p, ts) for p, _seq, ts in out]


def replay_face(session_dir: Path, out_csv: Path) -> int:
    """按留存帧重跑 VideoPipeline,把每帧的序列化结果写成 CSV。返回帧数。"""
    import cv2
    from face_expression.pipeline.video_pipeline import VideoPipeline

    frames = load_frames(session_dir, "face")
    pipeline = VideoPipeline(session_id=session_dir.name)
    rows = []
    for path, ts in frames:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[跳过] 读不出:{path}", file=sys.stderr)
            continue
        _obj, _mesh, features = pipeline.process_frame(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ts)
        if features:
            rows.append(features)
    pipeline.close()
    if rows:
        keys = sorted({k for r in rows for k in r})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="重放留存帧(M2.6 验收)")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--root", default=None, help="默认取 JINGXIN_RECORDINGS_DIR")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    d = Path(args.root) / args.session_id if args.root else recording_dir(args.session_id)
    n = replay_face(d, Path(args.out))
    print(f"[完成] 重放 {n} 帧 -> {args.out}")
    print("下一步:与 data/logs/face_au_log_<sid>.csv 逐格比对(spec §7.1)")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试,确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_replay_retained.py -v`
Expected: 2 passed

- [ ] **Step 5: 跑全量套件**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest -q`
Expected: 全部通过

- [ ] **Step 6: 提交**

```bash
cd ~/jingxin
git add experiments/replay_retained.py tests/test_replay_retained.py
git commit -m "feat(m2.6): 重抽回放器 —— 按账本里的当时时间戳重放留存帧"
```

- [ ] **Step 7: 端到端验收(需要一场真会话,由使用者跑)**

待使用者跑完一场真会话(`~/shared/start_all.sh` 起服务 → Windows 浏览器开 `http://localhost:5173` → 答 ≥5 段不同的回答)。然后逐条打勾:

- [ ] `ls ~/shared/jingxin_recordings/<sid>/media/face/ | wc -l` **等于** `retention.face.jsonl` 里 face 帧行数
- [ ] `media/gesture/` 与 `media/audio/` 也有内容(文件数与模态数一致)
- [ ] 各 `retention.<writer>.jsonl` 里 **没有** `"kind": "degraded"` 的行
      (⚠️ 原计划写的是检查 `retention_degraded` 字样 —— 而实现从不写这个字样,
      那是一条**永不失败的空检查**,给的是假保证。已改成本实现真正会写的形态。)
- [ ] 每个 `retention.<writer>.jsonl` 的**每一行都能被 `json.loads` 解析**
      (碎片行是"两个进程共写一个文件"的症状;账本已按写入者分文件,这里做回归确认)
- [ ] 重抽:`$PY experiments/replay_retained.py --session-id <sid> --out /tmp/replay_face.csv`,
      与 `data/logs/face_au_log_<sid>.csv` **逐格相等**;且工具**退出码为 0**
      (它现在会在有缺口时报 2 —— "重放了 105/187 帧"不算通过)
- [ ] 仓库里没有原始媒体:`git status --porcelain | grep -i -E '\.(jpg|png|webm|wav)$'` → 空

- [ ] **合并门不回归**(spec §7.8;按 M2.5 账本的口径:它只证明"没有误伤",不是本设计的证据):

```bash
cd ~/jingxin/experiments/duration_audit
PY=~/miniconda3/envs/jingxin/bin/python
$PY reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_probe
$PY reaggregate_normalized.py --verify-legacy /tmp/legacy_probe   # 期望 0/2835510
```

---

## 完成之后

**这一计划之外的两块,都不是遗漏,是明确切出去的**:

1. **M2.6 的前端腿**(要动 `~/JingXin-frontend` 三个文件):原生视频录制 + `POST /session/{sid}/media` 端点。
2. **M2.6 的元数据层**(spec §5.5 / §5.6,裁定 R7):`meta.json` 模板与校验、题目时刻上报端点 `POST /session/{sid}/question`。**R7 已裁定它属于 M2.6**,所以这份计划没有把它做完 —— 它同样卡在前端上报那一步。

两块都**先不做** —— 使用者已定:服务端腿先落地、先跑一场自证;其余随后补,补完再录正式的 3–5 场(那批才带原生音视频、才是 M3 的验收素材)。

**给使用者的话(完成时要说清)**:跑一场之后,画面和声音就**自动**存在
`~/shared/jingxin_recordings/<场次 id>/media/` 下,不需要他再手工录。这就是他问的
"数据存储问题"落地了 —— **但只落地了媒体那一半**;`meta.json`(他手填的面试官评分、设备参数)
与每道题的起止时刻还没接上,那两块等前端腿。
