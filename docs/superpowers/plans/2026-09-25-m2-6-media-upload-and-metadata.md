# M2.6 前端腿·服务端半(N2-A):原生视频上传端点 + 阶段 A 元数据层

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让「前端的原生视频」和「这一场的元数据」在服务端有落点 —— 新增 `POST /session/{sid}/media` 与 `POST /session/{sid}/question` 两个端点,加上 `meta.json` 的模板与硬校验、会话收尾对账。

**Architecture:** 两个新端点都挂在**已有的语音服务**(:8001)—— 铸号、`session.json`、`~/shared` 写入只有它做过,spec §5.3 明确"不新起服务"。媒体字节仍走既有的 `media_retention.py`(仓库根扁平件);题目时刻与元数据是**元数据层**,不是媒体,另起一个同层扁平件 `session_meta.py`。两个端点都**只写 `~/shared`**、都过 `validate_session_id` 守卫。

**Tech Stack:** Python 3.11 / FastAPI / pytest;`~/miniconda3/envs/jingxin/bin/python`(不可用 `~/huihui/bin/python`)。前端不在本计划内。

**Spec:** `docs/superpowers/specs/2026-09-25-m2-6-raw-media-retention-design.md`(§5.3 / §5.5 / §5.6 / §6 / §7.5–7.8 / §8.5)
**上游需求(元数据字段的依据):** `docs/superpowers/specs/2026-09-22-jingxin-recording-requirements.md` §3.2 / §3.4 / §4
**本计划只覆盖 N2 的服务端半。** 前端半(`useCamera` 开音轨 + `MediaRecorder` + `api.ts` 上传 + `AssessmentPage` 上报题目时刻)是**另一份计划**,且它依赖 spec §8.2 的 R5 实测 —— 那**必须由使用者在 Windows 端浏览器做**(WSL 端浏览器摄像头不可用)。本计划不含任何前端改动。

## Global Constraints

- **落盘位置**:只写 `~/shared/jingxin_recordings/{session_id}/`(即 `/mnt/d`,9p)。**不写仓库、不写 WSL `ext4.vhdx`**。媒体文件不得进 git。
- **`session_id` 一律过守卫**:非法 id(含 `/`、`..`)→ **HTTP 400**,不拿去当目录名。
- **留存默认开**:只有 `JINGXIN_RETAIN_MEDIA=0` 才关。
- **测试隔离**:每条测试用 `JINGXIN_RECORDINGS_DIR` 指到 `tmp_path`,并 `monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)`(既有的 `_isolated_root` fixture 口径)。**一条没设环境变量的测试曾往 `~/shared` 真写过 `s1/retention.face.jsonl`**(`tests/conftest.py:31` 记着这个事故)。
- **不改任何 L0 列的定义**;不动 `report_frontend/`;不采自评量表。
- **记账字段名**照 spec §5.2 的既有形状:`kind / modality / seq / file / bytes / sha256 / received_at_wall / declared_ts / source_endpoint / session_id`。
- **`questions.jsonl` 与 `meta.json` 都落在会话根**(`~/shared/jingxin_recordings/{sid}/`),**不在 `media/` 下面**。
- 解释器固定 `~/miniconda3/envs/jingxin/bin/python`;全量套件基线 **317 passed**。

## Review Focus

本计划涉及、但 spec 没写死的输入;每条都在下面某个任务的测试里钉住:

1. **`ask_end < ask_start`(含相等)** —— 一个零长或倒挂的提问窗口会让 `response_latency` 变成无意义的数。必须**拒绝并说明**,不是存下去。
2. **同一 `(sid, qid)` 重复上报** —— spec §5.6 说"以后来的为准"、§7.6.2 说"落盘的只有后一个"。两条合读 = **upsert**:文件里该题**只有一行**。若做成纯追加,`questions.jsonl` 会出现同一题两行,下游得自己判重。
3. **`camera.webm` 重复上传** —— 架构图里它只有一个文件名。第二次上传**覆盖**该文件,但**账本保留两行**(sha 各不同)⟹ 被覆盖这件事有据可查,不是静默丢失。
4. **留存被关掉时两个端点的行为** —— 它们唯一的工作就是留存。**不许静默 200 让人以为存了**:要回 200 且**明说没存**(`stored: false` + 原因)。
5. **`meta.json` 的"缺项"不只是"键不存在"** —— 空串 / `null` / 空列表 / **`false`** 同样是缺。只判键在不在,会让一份全空模板"通过校验";不把 `false` 当缺,则**一份没做知情同意的场次会过校验**(`consent.archived` 的模板默认值就是 `false`)。

---

### Task 1: `media_retention.retain_uploaded_video` —— 原生音视频落成 `media/camera.webm`

**Files:**
- Modify: `media_retention.py`(`LEDGER_WRITERS` :34、`_prepare` :176、新增 `retain_uploaded_video`)
- Test: `tests/test_media_retention_video.py`(新建)

**Interfaces:**
- Consumes: 本模块既有的 `enabled()` / `validate_session_id()` / `_session_lock()` / `media_dir()` / `_record()` / `_append_jsonl()` / `_mark_degraded()`。
- Produces:
  - `CAMERA_FILENAME = "camera.webm"`(模块常量)
  - `LEDGER_WRITERS` 变成 `("face", "gesture", "audio", "camera")`
  - `_prepare(session_id: str, writer: str, probe_parts: tuple[str, ...] | None = None) -> bool`
  - `retain_uploaded_video(session_id: str, data: bytes, source: str = "") -> dict | None`

**为什么要动 `_prepare` 的签名(hard part):** `_prepare` 现在探的是 `media_dir(sid, writer)`,即 `media/<writer>/`。而 `camera.webm` 按 spec §4 的架构图直接落在 `media/` **下面**(不是 `media/camera/`)。不改签名的话,预检会**建出一个永远空的 `media/camera/` 目录** —— 而 §7.7 的人工核对清单里有一句"素材**文件数与模态数一致**",多一个空目录正是在那种核对里制造困惑的东西。`probe_parts` 默认 `None` 时行为与今天**逐字节相同**(既有 20 条 `test_media_retention.py` 用例不受影响)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_media_retention_video.py
"""前端 `MediaRecorder` 录的原生音视频必须原样落成 `media/camera.webm`。

它和帧/音频的**两处不同**(所以单开一个文件):
  ① 文件落在 `media/` **下面**,不是 `media/camera/` —— 不许建出空目录;
  ② 账本写入者是 `camera`(既有三个是 face/gesture/audio)。
"""
import importlib
from pathlib import Path

import pytest

media_retention = importlib.import_module("media_retention")

WEBM = b"\x1a\x45\xdf\xa3" + b"native-camera-payload" * 40      # EBML 头


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


def _write(pcm=WEBM, sid="20260925_203826_2449"):
    return media_retention.retain_uploaded_video(sid, pcm, source="/session/x/media")


def test_video_bytes_land_verbatim(_isolated):
    """存的是**同一份 bytes**:盘上逐字节等于收到的。"""
    rec = _write()
    p = _isolated / "20260925_203826_2449" / "media" / "camera.webm"
    assert p.read_bytes() == WEBM
    assert rec["bytes"] == len(WEBM)
    assert rec["kind"] == "video"
    assert rec["file"] == "media/camera.webm"
    assert rec["sha256"] == media_retention.hashlib.sha256(WEBM).hexdigest()


def test_no_empty_camera_directory_is_created(_isolated):
    """`media/camera/` **不许**出现 —— 它只会在人工核对时制造困惑。

    红法:把 `retain_uploaded_video` 里的 `_prepare(sid, "camera", probe_parts=())`
    改回 `_prepare(sid, "camera")`。
    """
    _write()
    assert not (_isolated / "20260925_203826_2449" / "media" / "camera").exists()


def test_video_ledger_is_its_own_writer(_isolated):
    """账本写给 `retention.camera.jsonl`,与另外三个写入者分开。"""
    _write()
    d = _isolated / "20260925_203826_2449"
    names = sorted(p.name for p in media_retention.ledger_files("20260925_203826_2449"))
    assert names == ["retention.camera.jsonl"], names
    assert "camera" in media_retention.LEDGER_WRITERS


def test_second_upload_overwrites_the_file_but_the_ledger_keeps_both(_isolated):
    """一个会话只有一个 `camera.webm` 这个名字。第二次上传覆盖文件,
    **但账本留两行、sha 各不同** ⟹ "被覆盖"这件事有据可查,不是静默丢失。"""
    first = _write(WEBM)
    second = _write(WEBM + b"-second-half")
    p = _isolated / "20260925_203826_2449" / "media" / "camera.webm"
    assert p.read_bytes() == WEBM + b"-second-half"          # 文件是后一个
    assert first["sha256"] != second["sha256"]
    lines = (media_retention.ledger_path("20260925_203826_2449", "camera")
             ).read_text(encoding="utf-8").splitlines()
    assert len([l for l in lines if l.strip()]) == 2          # 账本是两条


def test_write_failure_is_recorded_and_does_not_raise(_isolated):
    """中途写失败:不抛、记 degraded(与帧/音频同一口径)。

    ⚠️ 两处讲究(抄 `tests/test_media_retention.py:232` 的先例):
      ① **先成功写一件把预检过掉**。预检 `_write_probe` 自己也调 `Path.write_bytes`
         —— 直接让 `write_bytes` 全抛的话,失败落在**探针**上,而首件预检失败是
         **要大声抛 `RuntimeError`** 的(`_loud_or_silent`)。那样这个测试就在验
         另一件事,而且会以 RuntimeError 而非断言失败的形式红掉。
      ② 用 `pytest.MonkeyPatch.context()` 而**不是** `monkeypatch.undo()` ——
         后者会把 autouse fixture 设的 `JINGXIN_RECORDINGS_DIR` 一起撤掉,
         于是后面那句**真的写进 `~/shared`**。这个事故发生过(见那个文件的注释)。
    """
    assert _write() is not None                      # 先过预检
    calls = {"n": 0}
    real = Path.write_bytes

    def flaky(self, data):
        calls["n"] += 1
        if calls["n"] == 1:                          # 预检已过 ⟹ 这就是本次的写
            raise OSError(28, "No space left on device")
        return real(self, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_bytes", flaky)
        assert _write() is None                      # 不抛
    assert any("No space left" in r
               for r in media_retention.degraded_reasons("20260925_203826_2449"))


def test_retention_off_writes_nothing(_isolated, monkeypatch):
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    assert _write() is None
    assert not (_isolated / "20260925_203826_2449").exists()


def test_path_traversal_is_rejected(_isolated):
    with pytest.raises(ValueError):
        media_retention.retain_uploaded_video("../escape", WEBM)
```

- [ ] **Step 2: 跑测试确认它红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention_video.py -q`
Expected: FAIL —— `AttributeError: module 'media_retention' has no attribute 'retain_uploaded_video'`

- [ ] **Step 3: 最小实现**

三处改动。先扩写入者集合(`media_retention.py:34`):

```python
LEDGER_WRITERS = ("face", "gesture", "audio", "camera")
```

再给 `_prepare` 加探针目录参数(`media_retention.py:176`,只改签名与取 `d` 那一行):

```python
def _prepare(session_id: str, writer: str,
             probe_parts: tuple[str, ...] | None = None) -> bool:
    """本次能不能写。首件素材预检不过 → **抛**;之后只留痕、返回 False。

    `probe_parts`:探针要落在哪个目录下,相对 `media/`。缺省 `None` = 用 `writer`
    当子目录名(face/gesture/audio 的老行为)。`camera` 传**空元组** —— 因为
    `camera.webm` 直接落在 `media/` 下面,没有 `media/camera/` 这一层
    (spec §4 的架构图),传空元组才不会建出一个永远空的目录。
    其余说明见下(不变)。
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
    # ...以下逐字节不变
```

最后在 `retain_audio` 之后加:

```python
CAMERA_FILENAME = "camera.webm"


def retain_uploaded_video(session_id: str, data: bytes,
                          source: str = "") -> dict | None:
    """把前端 `MediaRecorder` 录的**原生音视频**原样存成 `media/camera.webm`。

    它存在的理由(spec §1.2 的 R2):服务端那一腿留的是**管线解码过的帧**
    (AssessmentPage 下 5 fps,微表情族在 5 fps 下是伪测量),前端这一腿留的才是
    **帧率没被钉死的原生流**。两者不是冗余,是两件事。

    **一个会话只有一个文件名** —— 第二次上传覆盖它。账本仍然一次一记,
    所以"哪一份被覆盖过、当时是什么"有据可查(测试钉住了这一点)。
    """
    if not enabled():
        return None
    sid = validate_session_id(session_id)
    with _session_lock(sid):
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
```

- [ ] **Step 4: 跑测试确认绿,并跑既有留存测试确认没回归**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_media_retention_video.py tests/test_media_retention.py tests/test_media_retention_concurrency.py -q`
Expected: PASS,全绿(`test_media_retention_concurrency.py:61` 钉着 `ledger_files` 只返回**已存在**的三个账本 —— 新增写入者**不会**凭空多出文件,所以它仍然绿)

- [ ] **Step 5: 反向复现(项目标准动作)**

把 `_prepare(sid, "camera", probe_parts=())` 改回 `_prepare(sid, "camera")` →
`test_no_empty_camera_directory_is_created` **必须红**;跑完改回来,并清 `__pycache__`。
(先 `find . -name __pycache__ -type d -prune -exec rm -rf {} +`,否则会跑变异字节码。)

- [ ] **Step 6: 提交**

```bash
git add media_retention.py tests/test_media_retention_video.py
git commit -m "feat(m2.6): 原生音视频落成 media/camera.webm —— 账本写入者加 camera,预检不再建空目录"
```

---

### Task 2: `POST /session/{session_id}/media` 端点

**Files:**
- Modify: `voice_interaction/api/app.py`(在既有 `/session/{session_id}/summary` :552 附近加)
- Test: `tests/test_session_media_endpoint.py`(新建)

**Interfaces:**
- Consumes: `media_retention.retain_uploaded_video()`(Task 1)、既有的 `media_retention.validate_session_id()`。
- Produces: `POST /session/{session_id}/media`(multipart,字段名 `file`)→
  `{"status": "success", "stored": true, "session_id": ..., "bytes": N, "sha256": "..."}`
  或留存关闭时 `{"status": "success", "stored": false, "reason": "..."}`。

**为什么字段名用 `file` 而不是 `video`:** 三个既有服务收帧/收音频的 multipart 字段名就是 `file`(`face_expression/api/app.py`、`gesture_analysis/api/app.py` 的 `file: UploadFile = File(...)`)。统一它,前端的 `api.ts` 就能用同一个 `FormData` 形状。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_session_media_endpoint.py
"""`POST /session/{sid}/media`:前端原生视频的落点(spec §5.3)。

它**不分析任何东西** —— 只落盘 + 记账。所以这里的断言全是"字节与台账"。
"""
import asyncio
import importlib

import pytest

import media_retention
voice_app = importlib.import_module("voice_interaction.api.app")

WEBM = b"\x1a\x45\xdf\xa3" + b"native-camera-payload" * 40
SID = "20260925_203826_2449"


class _FakeUpload:
    def __init__(self, data, name="camera.webm"):
        self._data, self.filename, self.content_type = data, name, "video/webm"

    async def read(self):
        return self._data


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


def test_uploaded_video_is_retained(_isolated):
    """200 + 文件逐字节在盘上 + 台账一行。"""
    r = asyncio.run(voice_app.submit_session_media(session_id=SID, file=_FakeUpload(WEBM)))
    assert r["status"] == "success" and r["stored"] is True
    p = _isolated / SID / "media" / "camera.webm"
    assert p.read_bytes() == WEBM
    assert r["sha256"] == media_retention.hashlib.sha256(WEBM).hexdigest()


def test_retention_off_says_so_instead_of_pretending(_isolated, monkeypatch):
    """留存关了 → 200 但**明说没存**。不许静默成功让人以为存下了。"""
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    r = asyncio.run(voice_app.submit_session_media(session_id=SID, file=_FakeUpload(WEBM)))
    assert r["status"] == "success"
    assert r["stored"] is False
    assert r["reason"], "说了没存,却没给原因"
    assert not (_isolated / SID).exists()


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", ""])
def test_illegal_session_id_is_400_and_writes_nothing(_isolated, bad):
    """路径穿越守卫:非法 id → 400,盘上不得出现越界目录。"""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(voice_app.submit_session_media(session_id=bad, file=_FakeUpload(WEBM)))
    assert ei.value.status_code == 400
    assert not (_isolated.parent / "escape").exists()
```

- [ ] **Step 2: 跑测试确认它红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_media_endpoint.py -q`
Expected: FAIL —— `AttributeError: module 'voice_interaction.api.app' has no attribute 'submit_session_media'`

- [ ] **Step 3: 最小实现**

在 `voice_interaction/api/app.py` 的 `/session/{session_id}/summary` **上方**加:

```python
@app.post("/session/{session_id}/media")
async def submit_session_media(session_id: str, file: UploadFile = File(...)):
    """收前端 `MediaRecorder` 录的**原生音视频**,原样落 `media/camera.webm`(spec §5.3)。

    为什么挂在语音服务:铸号、`session.json`、`~/shared` 的写入都归它,spec §5.3
    明确"不新起服务"。

    它**不做任何分析** —— 与另外两个 CV 服务收帧的端点不同,这里没有第二步。
    `session_id` 走**路径参数**而不是 query/表单:这一条路由的身份就是那个会话,
    让它在 URL 里可见比藏在表单里好排查(另外两个 CV 服务收 query/form 是历史包袱)。
    """
    try:
        sid = media_retention.validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"非法 session_id: {exc}")
    data = await file.read()
    rec = media_retention.retain_uploaded_video(sid, data, source="/session/media")
    if rec is None:
        # 两种"没存":留存被显式关掉 / 中途写失败。两者都**不许装成功**。
        why = ("留存已关闭(JINGXIN_RETAIN_MEDIA=0)" if not media_retention.enabled()
               else "落盘失败:" + "；".join(media_retention.degraded_reasons(sid)))
        return {"status": "success", "stored": False, "reason": why,
                "session_id": sid, "bytes": len(data)}
    return {"status": "success", "stored": True, "session_id": sid,
            "bytes": rec["bytes"], "sha256": rec["sha256"], "file": rec["file"]}
```

- [ ] **Step 4: 跑测试确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_media_endpoint.py -q`
Expected: PASS(6 passed)

- [ ] **Step 5: 反向复现**

把 `raise HTTPException(status_code=400, ...)` 那两行去掉(交给下面的通用 `except Exception` → 500)→
`test_illegal_session_id_is_400_and_writes_nothing` 必须红(400 变 500);改回来。

⚠️ 注意:本端点**没有**外层 `except Exception → 500` 包装(与另外两个 `answer_audio` 不同)——
`validate_session_id` 的 `ValueError` 已在上面显式转成 400,而 `retain_uploaded_video`
自己吞掉写失败并返回 `None`。所以这里不需要那层包装,加了反而会把 400 吞成 500。

- [ ] **Step 6: 提交**

```bash
git add voice_interaction/api/app.py tests/test_session_media_endpoint.py
git commit -m "feat(m2.6): POST /session/{sid}/media —— 前端原生视频的落点"
```

---

### Task 3: `session_meta.py` —— 题目时刻台账(upsert)

**Files:**
- Create: `session_meta.py`(仓库根扁平件,与 `media_retention.py` / `session_clock.py` 同层)
- Test: `tests/test_session_meta_questions.py`(新建)

**Interfaces:**
- Consumes: `media_retention.root()`、`media_retention.validate_session_id()`
- Produces:
  - `QUESTIONS_FILENAME = "questions.jsonl"`、`META_FILENAME = "meta.json"`
  - `questions_path(session_id: str) -> Path`(**不建目录**)
  - `upsert_question(session_id: str, *, qid: str, index: int, ask_start: float, ask_end: float, source: str = "") -> dict`
  - `read_questions(session_id: str) -> list[dict]`

**两处必须解释的设计决定(hard part):**

**① `qid` 是什么。** 实测:题库是**纯字符串列表**(`voice_interaction/pipeline/assessment_pipeline.py:101`、`:315`),**题目没有 id**。spec §5.6 的请求体同时要 `qid` 与 `index`,而今天只有文本能当身份。所以:
- `qid` = **题目文本原文**(唯一、稳定、跨会话可归组);
- `index` = 该题在本次会话里的 0 基序号(它标的是**顺序**,不是身份)。
- 存文本**不违反** M1 的"原句不进仓库"边界:那条管的是**候选人说的话**;题目是这个系统的固定输入,而且 `questions.jsonl` 落在 `~/shared`(**仓库外**)。

**② upsert 而不是纯追加。** spec §5.6 说"逐行追加……以后来的为准",而 §7.6.2 的验收写"**落盘的只有后一个**"。两条合读 → **upsert**:同一 `qid` 在文件里**只有一行**。理由:下游(M3 的 `response_latency`)要的就是一题一个时间窗;留重复行等于把判重推给下游。读-改-写在**本会话的锁**内完成,理由与 `transcript_store.append_utterance` 逐字相同(并发重试会拿旧快照盖掉新写入)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_session_meta_questions.py
"""题目时刻台账(spec §5.6)。它是 `response_latency` 的唯一来源。"""
import importlib

import pytest

session_meta = importlib.import_module("session_meta")

SID = "20260925_203826_2449"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    session_meta._reset_for_tests()
    return tmp_path


def test_reported_window_lands_on_disk(_isolated):
    rec = session_meta.upsert_question(SID, qid="请简单介绍一下你自己", index=0,
                                       ask_start=1758824000.0, ask_end=1758824006.5,
                                       source="/session/question")
    p = _isolated / SID / "questions.jsonl"
    assert p.exists()
    rows = session_meta.read_questions(SID)
    assert len(rows) == 1
    assert rows[0]["qid"] == "请简单介绍一下你自己"
    assert rows[0]["index"] == 0
    assert rows[0]["ask_end"] == 1758824006.5
    assert rec["session_id"] == SID


def test_same_qid_reported_twice_keeps_only_the_later(_isolated):
    """spec §7.6.2:同一 (sid, qid) 报两次 → **落盘的只有后一个**。

    红法:把 upsert 改成纯追加(`rows.append(rec)`)→ 本测试红在 len(rows)。
    """
    session_meta.upsert_question(SID, qid="Q", index=0, ask_start=1.0, ask_end=2.0)
    session_meta.upsert_question(SID, qid="Q", index=0, ask_start=1.0, ask_end=9.0)
    rows = session_meta.read_questions(SID)
    assert len(rows) == 1, f"同一题留了 {len(rows)} 行"
    assert rows[0]["ask_end"] == 9.0


def test_two_different_questions_both_stay(_isolated):
    session_meta.upsert_question(SID, qid="Q1", index=0, ask_start=1.0, ask_end=2.0)
    session_meta.upsert_question(SID, qid="Q2", index=1, ask_start=3.0, ask_end=4.0)
    assert [r["qid"] for r in session_meta.read_questions(SID)] == ["Q1", "Q2"]


@pytest.mark.parametrize("kw,why", [
    (dict(qid="  ", index=0, ask_start=1.0, ask_end=2.0), "空 qid"),
    (dict(qid="Q", index=-1, ask_start=1.0, ask_end=2.0), "负 index"),
    (dict(qid="Q", index=0, ask_start=0.0, ask_end=2.0), "ask_start 非正"),
    (dict(qid="Q", index=0, ask_start=1.0, ask_end=0.0), "ask_end 非正"),
    (dict(qid="Q", index=0, ask_start=5.0, ask_end=2.0), "ask_end 早于 ask_start"),
    (dict(qid="Q", index=0, ask_start=5.0, ask_end=5.0), "零长窗口"),
    (dict(qid="Q", index=0, ask_start="1.0", ask_end=2.0), "字符串当时间"),
])
def test_bad_windows_are_rejected_loudly(_isolated, kw, why):
    """倒挂 / 零长 / 非数字的窗口必须**拒绝**。

    零长的提问窗口会让 `response_latency = 首次开口 − ask_end` 变成无意义的数,
    而且它看起来像个正常值 —— 比显然的错更危险。
    """
    with pytest.raises(ValueError):
        session_meta.upsert_question(SID, **kw)
    assert not (_isolated / SID / "questions.jsonl").exists(), f"{why}:拒绝得不够干净"


def test_reading_a_session_with_no_questions_is_empty_not_an_error(_isolated):
    """读一场没报过题的会话:返回空列表,而且**不建目录**(读侧不该有副作用)。"""
    assert session_meta.read_questions("20260101_000000_aaaa") == []
    assert not (_isolated / "20260101_000000_aaaa").exists()


def test_path_traversal_is_rejected(_isolated):
    with pytest.raises(ValueError):
        session_meta.upsert_question("../escape", qid="Q", index=0,
                                     ask_start=1.0, ask_end=2.0)
```

- [ ] **Step 2: 跑测试确认它红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_meta_questions.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'session_meta'`

- [ ] **Step 3: 最小实现**

```python
# session_meta.py
"""阶段 A 元数据层:题目时刻台账 + `meta.json` 的模板与校验(spec §5.5 / §5.6)。

与 `media_retention.py` **分开**的理由:那个管的是**媒体字节**,这里管的是
**元数据**(题目时刻、人填的协变量)。两者落点也不同 —— 媒体在 `media/` 下面,
这里两样都在**会话根**。

它为什么必须在录制之前就位(spec §3.9 / §5.5):`response_latency` 的分子是
「首次开口墙钟 − `ask_end`」,而**录完就再也补不回来** —— 没有 `questions.jsonl`
的场次永久没有这个量。同理,该场的光照/增益/面试官评分都是"当时那一刻的状态",
不是稳定属性。
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from media_retention import root as _recordings_root, validate_session_id

QUESTIONS_FILENAME = "questions.jsonl"
META_FILENAME = "meta.json"

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _session_lock(session_id: str) -> threading.Lock:
    """逐会话锁。语义与 `transcript_store._session_lock` 相同:读-改-写必须串行,
    否则并发重试会拿旧快照把新写入整个盖掉,而且丢得无声无息。
    """
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(session_id, threading.Lock())


def _reset_for_tests() -> None:
    with _LOCKS_GUARD:
        _LOCKS.clear()


def _session_dir(session_id: str, *, create: bool) -> Path:
    """会话根目录。`create=False` 时**不建** —— 读侧不该有副作用。

    为什么不用 `media_retention.recording_dir`:那个函数**总是建目录**,
    于是"读一场根本不存在/没报过题的会话"会在盘上留下一个空会话目录。
    """
    sid = validate_session_id(session_id)
    d = _recordings_root() / sid
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def questions_path(session_id: str) -> Path:
    return _session_dir(session_id, create=False) / QUESTIONS_FILENAME


def read_questions(session_id: str) -> list[dict]:
    """本会话已上报的题目窗口。文件不存在 → `[]`(不是错误)。"""
    p = questions_path(session_id)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _validate_window(qid: Any, index: Any, ask_start: Any, ask_end: Any) -> None:
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError(f"qid 必须是非空字符串(题目文本原文),收到 {qid!r}")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"index 必须是非负整数,收到 {index!r}")
    for name, v in (("ask_start", ask_start), ("ask_end", ask_end)):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name} 必须是墙钟秒数(数字),收到 {v!r}")
        if not v > 0:
            raise ValueError(f"{name} 必须是正数,收到 {v!r}")
    if ask_end <= ask_start:
        raise ValueError(
            f"ask_end({ask_end}) 必须晚于 ask_start({ask_start})—— "
            f"零长或倒挂的提问窗口会让 response_latency 变成无意义的数")


def upsert_question(session_id: str, *, qid: str, index: int,
                    ask_start: float, ask_end: float,
                    source: str = "") -> dict:
    """记一道题的提问窗口。同一 `qid` **覆盖**前一次(spec §5.6 的"以后来的为准"、
    §7.6.2 的"落盘的只有后一个")。
    """
    _validate_window(qid, index, ask_start, ask_end)
    sid = validate_session_id(session_id)
    rec = {"qid": qid, "index": index,
           "ask_start": float(ask_start), "ask_end": float(ask_end),
           "source": source, "reported_at_wall": time.time(),
           "session_id": sid}
    with _session_lock(sid):
        p = _session_dir(sid, create=True) / QUESTIONS_FILENAME
        rows = [r for r in read_questions(sid) if r.get("qid") != qid]
        rows.append(rec)
        rows.sort(key=lambda r: (r["index"], r["qid"]))
        # 写临时文件再 replace:进程被杀时不会留下半行(单文件、整份重写,
        # 所以不需要 media_retention 那条 "追加必须单次 os.write" 的讲究)
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")
        os.replace(tmp, p)
    return rec
```

- [ ] **Step 4: 跑测试确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_meta_questions.py -q`
Expected: PASS(12 passed:5 个普通用例 + `test_bad_windows_are_rejected_loudly` 的 7 个参数)

- [ ] **Step 5: 反向复现**

把 `rows = [r for r in read_questions(sid) if r.get("qid") != qid]` 改成 `rows = read_questions(sid)`(退化成纯追加)→
`test_same_qid_reported_twice_keeps_only_the_later` 必须红;改回来并清 `__pycache__`。

- [ ] **Step 6: 提交**

```bash
git add session_meta.py tests/test_session_meta_questions.py
git commit -m "feat(m2.6): session_meta 题目时刻台账 —— qid 用题目文本,同题重复上报以后来的为准"
```

---

### Task 4: `POST /session/{session_id}/question` 端点

**Files:**
- Modify: `voice_interaction/api/app.py`
- Test: `tests/test_session_question_endpoint.py`(新建)

**Interfaces:**
- Consumes: `session_meta.upsert_question()`(Task 3)
- Produces: `POST /session/{session_id}/question`,请求体 `QuestionWindow`
  (`qid: str`, `index: int`, `ask_start: float`, `ask_end: float`)→
  `{"status": "success", "session_id": ..., "qid": ..., "index": ...}`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_session_question_endpoint.py
"""`POST /session/{sid}/question`:前端推题时刻的落点(spec §5.6)。"""
import importlib

import pytest

session_meta = importlib.import_module("session_meta")
voice_app = importlib.import_module("voice_interaction.api.app")

SID = "20260925_203826_2449"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    session_meta._reset_for_tests()
    return tmp_path


def _post(**kw):
    body = {"qid": "请简单介绍一下你自己", "index": 0,
            "ask_start": 1758824000.0, "ask_end": 1758824006.5}
    body.update(kw)
    return voice_app.submit_session_question(
        session_id=SID, body=voice_app.QuestionWindow(**body))


def test_reported_window_is_written(_isolated):
    r = _post()
    assert r["status"] == "success" and r["qid"] == "请简单介绍一下你自己"
    rows = session_meta.read_questions(SID)
    assert len(rows) == 1 and rows[0]["ask_end"] == 1758824006.5


def test_illegal_session_id_is_400():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        voice_app.submit_session_question(
            session_id="../escape",
            body=voice_app.QuestionWindow(qid="Q", index=0,
                                          ask_start=1.0, ask_end=2.0))
    assert ei.value.status_code == 400


def test_inverted_window_is_400_not_500(_isolated):
    """倒挂窗口是**请求本身**不合法 → 400。

    红法:去掉端点里那个 `except ValueError` → 它会掉进通用的 500,
    于是客户端拿到的是"服务器出错",而真相是它自己发错了 —— 排查方向全偏。
    """
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        _post(ask_start=5.0, ask_end=2.0)
    assert ei.value.status_code == 400
    assert not (_isolated / SID / "questions.jsonl").exists()
```

- [ ] **Step 2: 跑测试确认它红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_question_endpoint.py -q`
Expected: FAIL —— `AttributeError: ... has no attribute 'submit_session_question'`

- [ ] **Step 3: 最小实现**

在 `voice_interaction/api/app.py` 加请求体模型(放在既有的 `AnswerRequest` 附近)与端点:

```python
class QuestionWindow(BaseModel):
    """前端上报的一道题的提问窗口(spec §5.6)。

    时刻是**墙钟秒**(`time.time()` 量纲),**不是** M2.5 那个会话内相对时钟。
    两个基不要混:混了以后 `response_latency` 会算出一个看着正常、其实没意义的数。
    """
    qid: str          # 题目文本原文(题库没有 id,见 session_meta 模块开头)
    index: int        # 本场内的 0 基序号
    ask_start: float  # 推题那一刻
    ask_end: float    # **题问完那一刻**(不是回答提交时刻 —— 见下)
```

```python
@app.post("/session/{session_id}/question")
async def submit_session_question(session_id: str, body: QuestionWindow):
    """记一道题的提问窗口。`response_latency` 只此一途(spec §3.9 / §5.5)。

    ⚠️ `ask_end` 必须是**题问完**的时刻,不是"回答提交"的时刻:报告层的
    `response_latency = 首次开口墙钟 − ask_end`。若拿提交时刻当 `ask_end`,
    这个差值会恒等于 0 左右 —— 一个**看着正常、其实什么都没量**的数。
    """
    try:
        rec = session_meta.upsert_question(
            session_id, qid=body.qid, index=body.index,
            ask_start=body.ask_start, ask_end=body.ask_end,
            source="/session/question")
    except ValueError as exc:
        # 非法 id 与非法窗口都是**请求本身**的问题 → 400(不是 500)
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success", "session_id": rec["session_id"],
            "qid": rec["qid"], "index": rec["index"]}
```

并在文件顶部的 import 区加 `import session_meta`(与 `import media_retention` 同处)。

- [ ] **Step 4: 跑测试确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_question_endpoint.py -q`
Expected: PASS

- [ ] **Step 5: 反向复现**

把端点里的 `except ValueError` 去掉 → `test_inverted_window_is_400_not_500` 与
`test_illegal_session_id_is_400` 必须红(变 500);改回来并清 `__pycache__`。

- [ ] **Step 6: 提交**

```bash
git add voice_interaction/api/app.py tests/test_session_question_endpoint.py
git commit -m "feat(m2.6): POST /session/{sid}/question —— 提问窗口上报,response_latency 的唯一来源"
```

---

### Task 5: `meta.json` —— 模板 + 必填校验(只给模板与校验,不做界面)

**Files:**
- Create: `session_meta_template.json`(仓库根;模板本身要**版本化**,所以进 git)
- Modify: `session_meta.py`(加 `META_REQUIRED` / `missing_meta_fields` / `write_template`)
- Test: `tests/test_session_meta_template.py`(新建)

**Interfaces:**
- Consumes: Task 3 的 `_session_dir()` / `META_FILENAME`
- Produces:
  - `META_REQUIRED: tuple[str, ...]`(点分路径)
  - `missing_meta_fields(session_id: str) -> list[str]`
  - `write_template(session_id: str) -> Path`(把带 id 与时间戳的骨架写进会话根的 `meta.json`)

**为什么是"模板 + 校验"而不是界面**(spec §5.5 的原话):录制需求 §3.4 本来就把它写成一份
"录完**当场**逐条打勾、不要事后补"的清单;为一次性流程做界面是本末倒置。`write_template`
之所以有用,是让使用者**在原地填空**而不是从别处抄一份 —— 少一步就少一次漏填。

**必填项取自** `docs/superpowers/specs/2026-09-22-jingxin-recording-requirements.md` §3.2 那张表。
⚠️ **不含候选人自评量表** —— 那是 M5 标定用的真值,受伦理审查强制前置(spec §1.2)。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_session_meta_template.py
"""`meta.json`:模板能生成、缺项能报出来、且**不阻断分析**(spec §5.5 / §6)。

覆盖面来自录制需求 §3.2 —— 那一整张表就是"必填"的定义。
"""
import importlib
import json

import pytest

session_meta = importlib.import_module("session_meta")

SID = "20260925_203826_2449"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    session_meta._reset_for_tests()
    return tmp_path


def test_template_is_written_with_the_session_id_filled_in(_isolated):
    p = session_meta.write_template(SID)
    assert p == _isolated / SID / "meta.json"
    got = json.loads(p.read_text(encoding="utf-8"))
    assert got["session_id"] == SID
    assert got["recorded_at"], "生成时该把录制时刻填上"


def test_fresh_template_reports_every_required_field_as_missing(_isolated):
    """刚生成的空模板:所有必填项都该被点出来(不是"校验通过")。"""
    session_meta.write_template(SID)
    missing = session_meta.missing_meta_fields(SID)
    assert set(missing) == set(session_meta.META_REQUIRED), missing


def test_blank_and_null_count_as_missing_not_as_filled(_isolated):
    """空串 / None / `False` **同样是缺** —— 只判键在不在会让全空模板假绿。"""
    session_meta.write_template(SID)
    p = _isolated / SID / "meta.json"
    filled = json.loads(p.read_text(encoding="utf-8"))
    for path in session_meta.META_REQUIRED:
        _set(filled, path, "x")
    _set(filled, "consent.archived", True)     # 布尔项填成真才算填了
    p.write_text(json.dumps(filled, ensure_ascii=False), encoding="utf-8")
    assert session_meta.missing_meta_fields(SID) == []

    for bad in ("", None):
        _set(filled, "candidate.sex", bad)
        p.write_text(json.dumps(filled, ensure_ascii=False), encoding="utf-8")
        assert "candidate.sex" in session_meta.missing_meta_fields(SID), repr(bad)

    # `consent.archived: false` = "知情同意还没归档" ⟹ **必须算缺**
    # (模板刚生成时它就是 false;不把 False 当缺的话,没做知情同意的场次会过校验)
    _set(filled, "candidate.sex", "M")
    _set(filled, "consent.archived", False)
    p.write_text(json.dumps(filled, ensure_ascii=False), encoding="utf-8")
    assert "consent.archived" in session_meta.missing_meta_fields(SID)


def test_missing_meta_file_reports_all_required(_isolated):
    """连 meta.json 都没有 → 全部必填项都缺(不是抛异常)。"""
    assert set(session_meta.missing_meta_fields(SID)) == set(session_meta.META_REQUIRED)


def test_required_list_covers_the_recording_requirements_table():
    """把录制需求 §3.2 那张表的**每一项**都钉住,防止有人删项。"""
    for path in ["candidate.sex", "candidate.age", "candidate.native_language",
                 "candidate.dialect_region", "capture.device", "capture.resolution",
                 "capture.camera_distance_cm", "capture.lighting", "capture.mic_gain_db",
                 "interviewer_ratings.logical_thinking",
                 "interviewer_ratings.communication",
                 "interviewer_ratings.confidence", "consent.archived"]:
        assert path in session_meta.META_REQUIRED, f"漏了必填项 {path}"
```

- [ ] **Step 2: 跑测试确认它红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_meta_template.py -q`
Expected: FAIL —— `AttributeError: module 'session_meta' has no attribute 'write_template'`

- [ ] **Step 3: 最小实现**

先建模板文件 `session_meta_template.json`(仓库根):

```json
{
  "session_id": "",
  "recorded_at": "",
  "consent": { "archived": false, "path": "" },
  "candidate": { "sex": "", "age": null, "native_language": "", "dialect_region": "" },
  "capture": {
    "device": "", "resolution": "", "camera_distance_cm": null,
    "lighting": "", "mic_gain_db": null, "audio_sample_rate": null
  },
  "questions": [],
  "interviewer_ratings": {
    "logical_thinking": null, "communication": null, "confidence": null
  },
  "notes": ""
}
```

再往 `session_meta.py` 追加:

```python
TEMPLATE_PATH = Path(__file__).resolve().parent / "session_meta_template.json"

# 必填项(点分路径)。**依据是录制需求 §3.2 那张表**(每一项为何必须,见那张表的"为什么"列):
#   人的生理属性 —— pitch_mean 的 ICC=0.723,是解剖常量,不加协变量可能让分数变成性别探测器;
#   设备/取景     —— energy 的 ICC=0.654,是增益/距离代理;
#   面试官评分    —— 审查 Q3(f) 称其为"最该补的真值,成本最低、最贴用途";
#   题目/难度     —— 审查称"最明显的遗漏";
#   知情同意      —— 审查 §7.4 第 1 条,法定必留。
# ⚠️ 不含候选人自评量表 —— 受科技伦理审查强制前置(spec §1.2),不在阶段 A。
META_REQUIRED: tuple[str, ...] = (
    "candidate.sex", "candidate.age",
    "candidate.native_language", "candidate.dialect_region",
    "capture.device", "capture.resolution", "capture.camera_distance_cm",
    "capture.lighting", "capture.mic_gain_db",
    "interviewer_ratings.logical_thinking",
    "interviewer_ratings.communication",
    "interviewer_ratings.confidence",
    "consent.archived",
)


def meta_path(session_id: str) -> Path:
    return _session_dir(session_id, create=False) / META_FILENAME


def _get_path(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _is_blank(v: Any) -> bool:
    """`None` / 空串 / 空列表 / 空字典 / **`False`** 都算**没填**。

    最后那一条是给 `consent.archived` 的:它是布尔必填项,而模板里的默认值就是
    `false`,意思是"知情同意**还没归档**" —— 那正是我们要它非 `false` 的原因。
    不把 `False` 当缺的话,一份**根本没做知情同意**的场次会通过校验。
    只判"键在不在"更糟:一份全空模板会整份通过(而那正是模板刚生成时的样子)。
    """
    if v is None or v is False:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def read_meta(session_id: str) -> dict | None:
    p = meta_path(session_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def missing_meta_fields(session_id: str) -> list[str]:
    """本场 `meta.json` 还缺哪些必填项(空 = 齐了)。

    **meta.json 不存在 = 全部都缺** —— 不是异常。收尾对账要能报出这件事,
    而不是自己先崩掉。
    """
    meta = read_meta(session_id)
    if meta is None:
        return list(META_REQUIRED)
    return [p for p in META_REQUIRED if _is_blank(_get_path(meta, p))]


def write_template(session_id: str) -> Path:
    """把模板铺进会话根,`session_id` 与 `recorded_at` 先填好。

    让使用者**在原地填空**,而不是从别处抄一份 —— 少一步就少一次漏填。
    **已存在则抛**:不覆盖已填过的内容(那份是当场记的,补不回来)。
    """
    from datetime import datetime
    sid = validate_session_id(session_id)
    p = _session_dir(sid, create=True) / META_FILENAME
    if p.exists():
        raise FileExistsError(f"{p} 已存在 —— 不覆盖(要重来请先自己移走)")
    payload = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    payload["session_id"] = sid
    payload["recorded_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
```

测试文件里还需要一个 `_set` 小助手(写进 `tests/test_session_meta_template.py` 顶部):

```python
def _set(obj: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = obj.setdefault(part, {})
    obj[parts[-1]] = value
```

- [ ] **Step 4: 跑测试确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_meta_template.py -q`
Expected: PASS

- [ ] **Step 5: 反向复现**

把 `_is_blank` 里 `isinstance(v, str): return not v.strip()` 删掉(退化成只判 None)→
`test_blank_and_null_count_as_missing_not_as_filled` 必须红;改回来并清 `__pycache__`。

- [ ] **Step 6: 提交**

```bash
git add session_meta.py session_meta_template.json tests/test_session_meta_template.py
git commit -m "feat(m2.6): meta.json 模板 + 必填校验 —— 缺项要响,不当场填就永久没有"
```

---

### Task 6: 会话收尾对账(CLI)

**Files:**
- Modify: `session_meta.py`(加 `check_session()` 与 `__main__`)
- Test: `tests/test_session_closeout.py`(新建)

**Interfaces:**
- Consumes: Task 3/5 的全部;`media_retention.degraded_reasons()` / `ledger_files()`
- Produces:
  - `check_session(session_id: str, expected_questions: int | None = None) -> dict`
    (`{"media": {modality: 件数}, "video": bool, "degraded": [...], "missing_meta": [...],
      "reported_questions": [...], "missing_questions": [...]}`)
  - `python -m session_meta --check-session <sid> [--expected-questions N]`
  - `python -m session_meta --write-template <sid>`

**为什么不硬塞进 `refresh_manifest`**(spec §8.5 的原文要求"先确认它的现有形状"):
实测 `refresh_manifest(session_id, log_dir, root=None)` 只做一件事 ——
按 `payload["logs"][mod]["expected_file"]` 在 `log_dir` 下判存在。它管的是**仓库内
`data/logs` 的三份 CSV**,与"`~/shared` 里的媒体齐不齐"是**两套不同的期望**。
硬塞进去要么把它改成四不像,要么让它的参数语义漂移。**另写。**

**漏报对账的规则**(spec §6 最后一行):按 `questions.jsonl` 的**条数**与
`--expected-questions` 对账;**缺哪题报哪题**需要题目清单,而服务端只有题库 ——
所以这里报的是"报了几题、期望几题、`index` 缺了哪些号"。`response_latency`
缺一道就是缺一道,**不许静默少一题**。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_session_closeout.py
"""会话收尾对账(spec §6 的两条出口 + §7.6.3/4)。

它是"当场逐条打勾"那份清单的机器版 —— 靠流程不靠代码(spec §8.6),
但**至少要能响**。
"""
import importlib
import json
from pathlib import Path

import pytest

import media_retention
session_meta = importlib.import_module("session_meta")

SID = "20260925_203826_2449"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    session_meta._reset_for_tests()
    return tmp_path


def _fill_meta(sid=SID):
    session_meta.write_template(sid)
    p = session_meta.meta_path(sid)
    import json
    m = json.loads(p.read_text(encoding="utf-8"))
    m["candidate"].update(sex="M", age=30, native_language="zh", dialect_region="吴语")
    m["capture"].update(device="Logitech C920", resolution="1280x720",
                        camera_distance_cm=60, lighting="室内顶灯", mic_gain_db=12)
    m["interviewer_ratings"].update(logical_thinking=4, communication=3, confidence=5)
    m["consent"]["archived"] = True
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")


def test_everything_present_reports_nothing_missing(_isolated):
    _fill_meta()
    media_retention.retain_frame(SID, "face", b"\xff\xd8\xff\xe0jpeg", source="/analyze")
    media_retention.retain_uploaded_video(SID, b"\x1a\x45\xdf\xa3webm")
    session_meta.upsert_question(SID, qid="Q0", index=0, ask_start=1.0, ask_end=2.0)
    session_meta.upsert_question(SID, qid="Q1", index=1, ask_start=3.0, ask_end=4.0)

    got = session_meta.check_session(SID, expected_questions=2)
    assert got["media"]["face"] == 1
    assert got["video"] is True
    assert got["degraded"] == []
    assert got["missing_meta"] == []
    assert got["missing_questions"] == []


def test_missing_interviewer_rating_is_reported_but_does_not_raise(_isolated):
    """spec §7.6.3:删掉 interviewer_ratings 的**某一项** → 收尾报出来。
    注意是**单项**:整块删是"没填",删一项是"填漏了",两种都要报得上。"""
    _fill_meta()
    import json
    p = session_meta.meta_path(SID)
    m = json.loads(p.read_text(encoding="utf-8"))
    del m["interviewer_ratings"]["communication"]
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")

    got = session_meta.check_session(SID)          # 不抛
    assert "interviewer_ratings.communication" in got["missing_meta"]


def test_unreported_question_is_named_by_index(_isolated):
    """spec §7.6.4:两题只报一题 → 点名缺的那题(按 index)。"""
    _fill_meta()
    session_meta.upsert_question(SID, qid="Q0", index=0, ask_start=1.0, ask_end=2.0)
    got = session_meta.check_session(SID, expected_questions=2)
    assert got["missing_questions"] == [1], got["missing_questions"]


def test_degraded_reasons_surface_in_the_report(_isolated):
    """留存中途失败 → 收尾必须看得到(spec §6:绝不允许静默)。

    注入方式同 Task 1:先成功写一件过预检,再用 `MonkeyPatch.context()`
    (**不是** `monkeypatch.undo()` —— 那会把 `JINGXIN_RECORDINGS_DIR` 一起撤掉,
    于是真的写进 `~/shared`)。
    """
    _fill_meta()
    media_retention.retain_frame(SID, "face", b"\xff\xd8\xff\xe0ok", source="/analyze")
    calls = {"n": 0}
    real = Path.write_bytes

    def flaky(self, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        return real(self, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_bytes", flaky)
        media_retention.retain_frame(SID, "face", b"\xff\xd8\xff\xe0boo", source="/analyze")

    got = session_meta.check_session(SID)
    assert any("No space left" in d for d in got["degraded"]), got["degraded"]


def test_response_latency_is_computable_from_the_stored_ask_end(_isolated):
    """spec §7.6.1:拿 `questions.jsonl` + 一份带逐字时间戳的 transcript,
    `response_latency` **算得出**,且**用的是 `ask_end` 自己的值**(改成错值 → 结果跟着变)。

    ⚠️ 这里**只做算术演示,不落地成生产函数** —— spec §5.6 明确写"本条不在 M2.6 实现
    (M2.6 只负责把 `ask_end` 留下来)"。它存在的理由是**证明留下的那个数接得上**:
    `response_latency = 首次开口墙钟 − ask_end`,而 ASR 的逐字时间戳是**段内相对值**
    (从 0 起),所以"首次开口墙钟"必须由**段起始墙钟 + 段内偏移**合成。
    不把这个示范钉住,M3 真去算时才会发现两个基不是一回事。
    """
    seg_wall, first_char_offset = 1758824011.5, 0.75      # 段起始墙钟 / 段内偏移
    first_speech_wall = seg_wall + first_char_offset

    session_meta.upsert_question(SID, qid="Q0", index=0,
                                 ask_start=1758824000.0, ask_end=1758824006.0)
    ask_end = session_meta.read_questions(SID)[0]["ask_end"]
    assert round(first_speech_wall - ask_end, 3) == 6.25

    session_meta.upsert_question(SID, qid="Q0", index=0,        # 换成错值
                                 ask_start=1758824000.0, ask_end=1758824009.5)
    ask_end2 = session_meta.read_questions(SID)[0]["ask_end"]
    assert round(first_speech_wall - ask_end2, 3) == 2.75, "结果没跟着 ask_end 变 ⟹ 它没读那个字段"


def test_missing_meta_file_reports_all_required_and_does_not_raise(_isolated):
    got = session_meta.check_session(SID)
    assert set(got["missing_meta"]) == set(session_meta.META_REQUIRED)
    assert got["media"] == {}
    assert got["video"] is False
```

- [ ] **Step 2: 跑测试确认它红**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_closeout.py -q`
Expected: FAIL —— `AttributeError: module 'session_meta' has no attribute 'check_session'`

- [ ] **Step 3: 最小实现**

往 `session_meta.py` 追加:

```python
def _media_counts(session_id: str) -> dict[str, int]:
    """按**账本行数**数各模态的素材件数(不是数盘上文件)。

    为什么以账本为准:账本记的是"服务端**收到了**什么",而盘上文件会被覆盖
    (`camera.webm`)或被人手动动过。spec §7.7 的第一条判据正是
    "帧数 == 服务端实际收到的帧数(以 retention.jsonl 的行数为准)"。
    """
    from media_retention import ledger_files
    counts: dict[str, int] = {}
    for book in ledger_files(session_id):
        for line in book.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") in ("frame", "raw", "converted", "video"):
                key = rec.get("modality") or rec.get("kind")
                counts[key] = counts.get(key, 0) + 1
    return counts


def check_session(session_id: str,
                  expected_questions: int | None = None) -> dict:
    """会话收尾对账。**任何缺项都只报不抛** —— 分析结果不因元数据缺失而作废
    (spec §6:元数据缺项不阻断分析,但必须让人当场知道)。
    """
    from media_retention import degraded_reasons
    sid = validate_session_id(session_id)
    rows = read_questions(sid)
    reported = sorted(r["index"] for r in rows)
    missing_q: list[int] = []
    if expected_questions is not None:
        missing_q = [i for i in range(expected_questions) if i not in set(reported)]
    counts = _media_counts(sid)
    return {
        "session_id": sid,
        "media": counts,
        "video": counts.get("camera", 0) > 0,
        "degraded": degraded_reasons(sid),
        "missing_meta": missing_meta_fields(sid),
        "reported_questions": reported,
        "missing_questions": missing_q,
    }


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="阶段 A 元数据层:模板与收尾对账")
    ap.add_argument("--write-template", metavar="SESSION_ID",
                    help="把 meta.json 模板铺进该会话目录(已存在则不动)")
    ap.add_argument("--check-session", metavar="SESSION_ID",
                    help="对账:媒体件数 / 降级 / meta 缺项 / 题目漏报")
    ap.add_argument("--expected-questions", type=int, default=None,
                    help="本场实问题数,用来点名漏报的题(缺省 = 不对账题目)")
    args = ap.parse_args()

    if args.write_template:
        p = write_template(args.write_template)
        print(f"✅ 模板已铺到 {p} —— 请**当场**逐条填,不要事后补")
        sys.exit(0)

    if args.check_session:
        r = check_session(args.check_session, args.expected_questions)
        print(f"会话 {r['session_id']}")
        print(f"  素材件数(按账本):{r['media'] or '(一件都没有)'}")
        print(f"  前端原生视频:{'有' if r['video'] else '**没有**'}")
        print(f"  留存降级:{r['degraded'] or '无'}")
        print(f"  meta.json 缺项:{r['missing_meta'] or '无'}")
        print(f"  已报题号:{r['reported_questions'] or '(一题都没报)'}")
        if args.expected_questions is not None:
            print(f"  漏报题号:{r['missing_questions'] or '无'}")
        ok = not (r["degraded"] or r["missing_meta"] or r["missing_questions"])
        print("✅ 齐了" if ok else "⚠️ 上面点出来的项没齐 —— 这一场缺的东西补不回来")
        sys.exit(0 if ok else 1)

    ap.print_help()
    sys.exit(2)
```

- [ ] **Step 4: 跑测试确认绿**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_session_closeout.py -q`
Expected: PASS

- [ ] **Step 5: 反向复现**

把 `_is_blank` 的 str 分支删掉 → `test_missing_interviewer_rating_is_reported_but_does_not_raise`
仍绿(键不存在时 `_get_path` 返回 None),但把 `missing_meta_fields` 改成只返回 `[]` → 该测试必须红。
改回来并清 `__pycache__`。

- [ ] **Step 6: 手动跑一次 CLI 看输出**

```bash
JINGXIN_RECORDINGS_DIR=/tmp/closeout_probe \
  ~/miniconda3/envs/jingxin/bin/python -m session_meta --check-session 20260101_000000_aaaa
```
Expected: 打印全空的一行行 + `⚠️`,退出码 1(**不是** traceback)

- [ ] **Step 7: 提交**

```bash
git add session_meta.py tests/test_session_closeout.py
git commit -m "feat(m2.6): 会话收尾对账 CLI —— 媒体件数/降级/meta 缺项/题目漏报,只报不抛"
```

---

### Task 7: 端到端(脚本化)+ 不回归

**Files:**
- Create: `~/shared/m26_media_acceptance.sh`(与既有的 `t7_acceptance.sh` / `m21_acceptance.sh` 同处)
- Test: 本任务**不写 pytest**;它跑的是真 HTTP

**Interfaces:**
- Consumes: Task 1–6 全部
- Produces: 一份可自己跑的验收脚本,输出"逐条打勾"的结果

**这个脚本与 spec §7.7 的关系:** §7.7 那份清单是给**真会话**(N3)用的,而且要求
`camera.webm` 能被解出音轨(那要等前端腿)。本脚本是它的**服务端冒烟版**:
用 curl 造一场假会话,把两个新端点和收尾对账串起来跑通,证明"接线是对的"。
**它不替代 §7.7。**

- [ ] **Step 1: 写脚本**

```bash
#!/usr/bin/env bash
# M2.6 前端腿·服务端半的冒烟验收:两个新端点 + 收尾对账。
# 用法: bash ~/shared/m26_media_acceptance.sh
set -uo pipefail
PY=~/miniconda3/envs/jingxin/bin/python
WORK=$(mktemp -d); PORT=8093
export JINGXIN_RECORDINGS_DIR="$WORK/rec" JINGXIN_DATA_DIR="$WORK/data"
mkdir -p "$JINGXIN_RECORDINGS_DIR" "$JINGXIN_DATA_DIR"
cd ~/jingxin
$PY -m uvicorn voice_interaction.api.app:app --host 127.0.0.1 --port $PORT \
  > "$WORK/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; echo "(服务已停;临时目录 $WORK 留着给排查)"' EXIT
for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 0.5; done

SID=$(curl -s -X POST "http://127.0.0.1:$PORT/interview/start" | $PY -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
echo "会话 = $SID"

echo "① 原生视频上传"
head -c 40000 /dev/urandom > "$WORK/camera.webm"
curl -s -X POST "http://127.0.0.1:$PORT/session/$SID/media" \
     -F "file=@$WORK/camera.webm" | tee "$WORK/media.json"; echo
grep -q '"stored": *true' "$WORK/media.json" || { echo "❌ 没存下"; exit 1; }
cmp -s "$WORK/camera.webm" "$JINGXIN_RECORDINGS_DIR/$SID/media/camera.webm" \
  && echo "   ✓ 盘上字节与上传逐字节相同" || { echo "❌ 字节不一致"; exit 1; }
[ -d "$JINGXIN_RECORDINGS_DIR/$SID/media/camera" ] && { echo "❌ 建出了空的 media/camera/"; exit 1; }

echo "② 题目时刻上报(两题)"
for i in 0 1; do
  curl -s -X POST "http://127.0.0.1:$PORT/session/$SID/question" -H 'Content-Type: application/json' \
    -d "{\"qid\":\"题目${i}\",\"index\":$i,\"ask_start\":$((1758824000+i*10)).0,\"ask_end\":$((1758824006+i*10)).0}" >/dev/null
done
echo "   questions.jsonl 行数 = $(wc -l < "$JINGXIN_RECORDINGS_DIR/$SID/questions.jsonl")"

echo "③ 同题重复上报 → 只留后一个"
curl -s -X POST "http://127.0.0.1:$PORT/session/$SID/question" -H 'Content-Type: application/json' \
  -d '{"qid":"题目0","index":0,"ask_start":1758824000.0,"ask_end":1758824999.0}' >/dev/null
$PY - "$JINGXIN_RECORDINGS_DIR/$SID/questions.jsonl" <<'PYEOF'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1],encoding="utf-8") if l.strip()]
q0=[r for r in rows if r["qid"]=="题目0"]
assert len(q0)==1, f"题目0 留了 {len(q0)} 行"
assert q0[0]["ask_end"]==1758824999.0, q0[0]
print("   ✓ 仍是 2 行,且题目0 取的是后报的值")
PYEOF

echo "④ 倒挂窗口 → 400(不是 500)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/session/$SID/question" \
  -H 'Content-Type: application/json' -d '{"qid":"坏","index":2,"ask_start":9.0,"ask_end":1.0}')
[ "$code" = "400" ] && echo "   ✓ 400" || { echo "❌ 得到 $code"; exit 1; }

echo "⑤ 路径穿越 → 400"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/session/..%2Fescape/media" -F "file=@$WORK/camera.webm")
echo "   HTTP $code(期望 400 或 404;两者都不许落盘)"

echo "⑥ 收尾对账(meta 还没填,应当点出来)"
$PY -m session_meta --write-template "$SID" >/dev/null
$PY -m session_meta --check-session "$SID" --expected-questions 3
echo "   (退出码 $? —— 1 = 有缺项,正是期望)"

echo "全部冒烟判据通过。真会话的 §7.7 清单在录制时另跑。"
```

- [ ] **Step 2: 跑它**

Run: `bash ~/shared/m26_media_acceptance.sh`
Expected: 六段全过;第 ⑥ 段打印出 `meta.json 缺项`(空模板的所有必填项)与
`漏报题号: [2]`(只报了两题、期望三题)→ 这正是"让缺项响起来"的证据

- [ ] **Step 3: 全量套件 + 合并门**

```bash
cd ~/jingxin && ~/miniconda3/envs/jingxin/bin/python -m pytest -q
cd ~/jingxin/experiments/duration_audit
PY=~/miniconda3/envs/jingxin/bin/python
$PY reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_probe
$PY reaggregate_normalized.py --verify-legacy /tmp/legacy_probe     # 期望 0/2835510
```
Expected: 全套件 **317 + 新增** 全绿;合并门 **0 / 2835510**(它只证明"没有误伤",
不是本设计的证据 —— spec §3.1)

- [ ] **Step 4: 记录位置(本任务不产生 git 提交)**

⚠️ **`~/shared` 不在 jingxin 仓库里** —— 它是 `D:\Shared`,与仓库是两个地方。
所以这个脚本**不进 git**,按既有 `t7_acceptance.sh` / `m21_acceptance.sh` 的先例留在
`~/shared`。**不要**为它造一个空提交。

要进 git 的是**它的位置与用法**:在 `docs/下一步.md` §5 的"验收脚本(可自己跑)"那一行
追加 `~/shared/m26_media_acceptance.sh`(M2.6 前端腿·服务端半)。这一步并进 Task 6 之后的
文档提交(见"收尾")。

```bash
cd ~/jingxin
git add docs/下一步.md
git commit -m "docs(m2.6): 验收脚本清单补 m26_media_acceptance.sh(它按先例留在 ~/shared,不进仓库)"
```
⚠️ **若 `docs/下一步.md` 此刻还装着别的未提交改动,先按 §4.9 精确暂存,不要 `git add -A`。**

---

## 收尾:本计划做完之后

- **N2 的前端半**需要另写计划,且它的第一步是 spec §8.2 的 **R5 实测**(由使用者在
  Windows 端浏览器做):`useCamera` 改成 `audio: true` 后两路 `getUserMedia` 会不会打架、
  `camera.webm` 的音轨正不正常。**打架就退回"另存 `audio.webm`"**(两路不同步,
  只能靠 `received_at_wall` 粗对齐)。
- **N3(录 3–5 场)**在本计划 + 前端半都完成之后。录制时逐条跑 spec §7.7 那份清单。
- 更新 `docs/下一步.md` 的 §0.1 第 2 件与 §5 索引。
