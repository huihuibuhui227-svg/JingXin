# tests/test_m26_review_hardening.py
"""N2-A 终局复核(2026-09-25)提出的五项 Important 的回归钉子。

分开放的理由:这一个文件整体对应"复核发现了什么",不是某个模块的内部细节。
每条都先红后绿 —— 复核者给出的复现步骤就是这里的红法。

命中的是同一个失败家族:**看着正常、其实无意义的数** /
**说存下了、其实没存** / **该响的时候崩了**。
"""
import asyncio
import importlib
import json
import time
from pathlib import Path

import pytest

import media_retention
session_meta = importlib.import_module("session_meta")
voice_app = importlib.import_module("voice_interaction.api.app")

SID = "20260925_203826_2449"
NOW = time.time()          # 墙钟"现在" —— 合理窗口都相对它构造


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    session_meta._reset_for_tests()
    return tmp_path


# ─────────────────────────── I1:毫秒 / 会话内相对时钟 不许被静默收下
@pytest.mark.parametrize("kw,why", [
    (dict(ask_start=NOW * 1000, ask_end=NOW * 1000 + 6000), "毫秒(JS Date.now() 的量纲)"),
    (dict(ask_start=0.5, ask_end=12.5), "会话内相对时钟(M2.5 那个基)"),
    (dict(ask_start=NOW - 30 * 86400, ask_end=NOW - 30 * 86400 + 5), "30 天前"),
])
def test_implausible_wall_clock_windows_are_rejected(_isolated, kw, why):
    """spec §5.6 要的是**墙钟秒**,而"两个基不要混"这句只有在这里才拦得住。

    毫秒与相对时钟都不会报错:它们算出来的 `response_latency` 是
    `≈ −1.76e12 s` 这种**看着像个数、其实毫无意义**的值,而这一场**录完就补不回来**
    —— 采集端是唯一的拦截点。红法:去掉 `_validate_window` 里的量程闸。
    """
    body = dict(qid="Q", index=0, ask_start=NOW, ask_end=NOW + 5)
    body.update(kw)
    with pytest.raises(ValueError):
        session_meta.upsert_question(SID, **body)
    assert not (_isolated / SID / "questions.jsonl").exists(), f"{why}:拒绝得不够干净"


def test_a_normal_wall_clock_window_still_passes(_isolated):
    """量程闸不能把正常的窗口也拦掉(它离"现在"只有几秒)。"""
    session_meta.upsert_question(SID, qid="Q", index=0,
                                 ask_start=NOW - 30, ask_end=NOW - 24)
    assert len(session_meta.read_questions(SID)) == 1


# ─────────── I2:账本被写碎/被人手改过时,收尾对账**只报不抛**(它的立身之本)
def test_torn_ledger_line_is_reported_not_raised(_isolated):
    """`retention.face.jsonl` 里一行被截断 → 对账必须**照样出报告**。

    为什么这不是假想(复核者引的仓库内证据):`media_retention.py:331-334` 记着
    真实会话那 390 行里**碎了 77 行**。而 `degraded_reasons` 逐行 try/except,
    新写的 `_media_counts` 却没有 —— 于是 CLI 直接 traceback,
    与它 docstring 上"任何缺项都只报不抛"相反。

    红法:去掉 `_media_counts` 里的逐行 try/except。
    """
    root = _isolated / SID                      # 账本落在**会话根**,不是 media/ 下面
    (root / "media" / "face").mkdir(parents=True)
    good = {"kind": "frame", "modality": "face", "seq": 1, "file": "media/face/000001.jpg",
            "bytes": 10, "sha256": "x", "received_at_wall": 1.0, "declared_ts": 0,
            "source_endpoint": "/analyze", "session_id": SID}
    (root / "retention.face.jsonl").write_text(
        json.dumps(good) + "\n" + '{"kind":"frame","modality":"fa' + "\n",
        encoding="utf-8")

    got = session_meta.check_session(SID)          # 不抛
    assert got["media"]["face"] == 1, "好的那一行没数进去"
    assert got["unreadable_lines"] == 1, "碎行没被点出来"


# ─────────────────── I3:"存了个 0 字节"不许被当成"原生视频有了"
def test_empty_upload_is_rejected_instead_of_reported_as_stored(_isolated):
    """空请求体 → 400。0 字节的 `camera.webm` 是一份**看着像有、其实没有**的录像。

    红法:去掉端点里的空体闸 → 它会回 `stored:true, bytes:0`。
    """
    from fastapi import HTTPException

    class _Empty:
        async def read(self, *a):
            return b""              # 空体:第一次读就 EOF
    with pytest.raises(HTTPException) as ei:
        asyncio.run(voice_app.submit_session_media(session_id=SID, file=_Empty()))
    assert ei.value.status_code == 400
    assert not (_isolated / SID / "media" / "camera.webm").exists()


def test_degraded_rows_are_never_counted_as_material(_isolated):
    """降级行**不是素材**,哪怕它带着字节数。

    ⚠️ 为什么用一行 `bytes>0` 的降级行来钉:当前 `_mark_degraded` 写的降级行
    `bytes` 恰好是 0,会被"0 字节不算材料"那条(I3 的修)**顺手**滤掉 ——
    于是 `kind` 过滤就**没有任何测试绑住**了(复核实测:把它换成 `if True:`
    全套 356 条没有一条变红)。这里直接造一行带字节数的 degraded,
    把 `kind` 这一条单独钉住,不靠那个"恰好"。
    """
    root = _isolated / SID
    root.mkdir(parents=True)
    degraded = {"kind": "degraded", "modality": "face", "seq": None, "file": None,
                "bytes": 4096, "sha256": None, "received_at_wall": 1.0, "declared_ts": None,
                "source_endpoint": "media_retention", "reason": "落点不可写", "session_id": SID}
    (root / "retention.face.jsonl").write_text(json.dumps(degraded) + "\n", encoding="utf-8")

    got = session_meta.check_session(SID)
    assert got["media"] == {}, f"降级行被当成素材数进去了:{got['media']}"
    assert got["degraded"], "降级也没报出来"


def test_zero_byte_camera_row_is_not_counted_as_video(_isolated):
    """就算账本里已经有一行 0 字节的 camera(手改过 / 旧版本留的),
    收尾也不许说"原生视频:有"。"""
    d = _isolated / SID
    d.mkdir(parents=True)
    row = {"kind": "video", "modality": "camera", "seq": 1, "file": "media/camera.webm",
           "bytes": 0, "sha256": "e3b0c44298fc1c149afbf4c8996fb924", "received_at_wall": 1.0,
           "declared_ts": None, "source_endpoint": "/session/media", "session_id": SID}
    (d / "retention.camera.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (d / "media").mkdir(exist_ok=True)
    (d / "media" / "camera.webm").write_bytes(b"")

    got = session_meta.check_session(SID)
    assert got["video"] is False, "0 字节被当成有视频了"
    assert got["video_bytes"] == 0


# ─────────────────────── I4:一题都没报时,收尾不许报"齐了"
def test_verdict_is_never_green_with_no_reported_questions():
    """`response_latency` 是这个里程碑存在的理由,一题都没报时它必然拿不到。

    没给 `--expected-questions` 就绿灯,等于**替 operator 把这件事放过去了**。
    红法:把 `closeout_verdict` 里的 `no_questions` 那一项去掉。
    """
    clean = {"session_id": SID, "media": {"face": 1}, "video": True, "video_bytes": 10,
             "degraded": [], "missing_meta": [], "reported_questions": [],
             "missing_questions": [], "unreadable_lines": 0}
    ok, reasons = session_meta.closeout_verdict(clean, expected_questions=None)
    assert ok is False, "一题都没报却给了绿灯"
    assert any("题" in r for r in reasons), reasons

    # 报了题(或显式给了期望题数)才可能绿
    ok2, _ = session_meta.closeout_verdict({**clean, "reported_questions": [0]}, None)
    assert ok2 is True


def test_verdict_reports_each_gap_separately():
    """每一条缺口各自出现在 reasons 里 —— operator 要能一眼看出缺什么。"""
    bad = {"session_id": SID, "media": {}, "video": False, "video_bytes": 0,
           "degraded": ["写失败"], "missing_meta": ["candidate.sex"],
           "reported_questions": [0], "missing_questions": [1], "unreadable_lines": 2}
    ok, reasons = session_meta.closeout_verdict(bad, expected_questions=3)
    assert ok is False
    assert len(reasons) == 5, reasons          # 降级/meta/漏题/碎行/无视频


# ─────────────────────────── I7:单次上传要有上限
def test_upload_cap_cannot_be_hit_by_a_single_session():
    """上限**撑得住一整场**会话 —— 否则它护的是盘,毁的是这一场的素材。

    速率来自 2026-09-25 的 **R5 实测**(Windows Edge 153,`vp8,opus` @1280x720):
    10 秒录出 1722 KB ⟹ ≈ 172 KB/s。先写的 512 MB 只够 ≈50 分钟,
    而撞上限的后果是**这一场什么都没存**(不是少存一段)。
    红法:把 `MAX_UPLOAD_BYTES` 改回 512 MB。
    """
    measured_kbps = 1722 / 10          # KB/s,实测
    hours = voice_app.MAX_UPLOAD_BYTES / 1024 / measured_kbps / 3600
    assert hours > 1.0, f"上限只够 {hours*60:.0f} 分钟 —— 一场长会话就会撞上"


def test_oversized_upload_is_413_and_writes_nothing(_isolated, monkeypatch):
    """一次请求不许把盘写满:写满会让这一场**之后所有**留存变成 degraded,
    而这正是本里程碑存在的理由(把素材留下来)。

    上限用 monkeypatch 调小,免得测试真去造几百 MB。红法:去掉大小闸。
    """
    from fastapi import HTTPException
    monkeypatch.setattr(voice_app, "MAX_UPLOAD_BYTES", 1024)

    class _Big:
        async def read(self, n=-1):
            return b"\x1a\x45\xdf\xa3" + b"x" * 4096
    with pytest.raises(HTTPException) as ei:
        asyncio.run(voice_app.submit_session_media(session_id=SID, file=_Big()))
    assert ei.value.status_code == 413
    assert not (_isolated / SID / "media" / "camera.webm").exists()
