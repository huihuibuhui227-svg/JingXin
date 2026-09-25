# tests/test_research_session_minting.py
"""第 19 条(复审 I4):科研评估**自成一场**,不许沿用上一场面试的号。

失效现场(2026-09-24 复审,按代码路径判定;2026-09-25 本机**当场复现**了其中一条):
    A 新页面直接做科研 → 前端没有号 → `withSession` 不加参数 → 落 `NONE`;
    B 先面试、再科研 → 前端还留着**上一场面试的铸号** → 科研回答
      (`/research/answer_audio` → `transcript_store.append_utterance(sid, utt)`)
      被写进**面试会话**的目录 `~/shared/jingxin_recordings/<面试号>/transcript.json`。

    实测(2026-09-25 12:35):`POST /research/start` 的返回体是
    `{'status': 'started', 'question': '请描述一个你深入研究过的技术…'}` —— **没有 session_id**。

修法(使用者 2026-09-25 裁定,方案 A):`/research/start` 与 `/interview/start` 对称铸号。
"""

import importlib
import re

import pytest
from fastapi.testclient import TestClient

# ⚠️ 两个名字不要搞混(2026-09-25 实测):
#   * `from voice_interaction.api import app …` → 拿到的是 **FastAPI 实例**
#     (包 `__init__.py` 里 `from .app import app` 把子模块这个属性**遮蔽**了);
#   * `importlib.import_module("voice_interaction.api.app")` → 拿到**模块**,
#     端点里的 `tts_engine` 等全局在它上面。
from voice_interaction.api import app as voice_api

_voice_app_module = importlib.import_module("voice_interaction.api.app")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """把仓库外的录制根目录挪到 tmp,并把 TTS 换掉(本任务不测发声)。

    `JINGXIN_RECORDINGS_DIR` 是 `transcript_store._root()` 认的环境变量。
    """
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "rec"))
    monkeypatch.setattr(_voice_app_module.tts_engine, "speak", lambda *a, **k: None)
    # logger 的文件名带 session_id,不隔离的话会在仓库 data/logs 里留下测试产物
    import voice_interaction.utils.logger as voice_logger_module
    monkeypatch.setattr(voice_logger_module, "LOGS_DIR", str(tmp_path / "logs"))
    return TestClient(voice_api)


def test_research_start_mints_a_session_id(client):
    """★ 科研入口必须铸号(与 /interview/start 对称)。

    红法:退回 `return {"status": "started", "question": first_question}`
    → `session_id` 缺失 → 红。
    """
    body = client.post("/research/start").json()

    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{4}", body.get("session_id", "")), (
        f"科研评估没有铸号 —— 返回体是 {body}")


def test_research_start_writes_the_session_manifest(client, tmp_path):
    """铸了号就要建会话清单 —— 否则 `append_utterance` 会在另一处另建一个残缺目录。

    红法:只 `return` 铸号、不 `ensure_manifest` → 清单不存在 → 红。
    """
    sid = client.post("/research/start").json()["session_id"]

    assert (tmp_path / "rec" / sid / "session.json").exists(), "铸了号却没建会话清单"


def test_research_and_interview_get_different_session_ids(client):
    """★ 同一页面连做两场 → 两个号必须不同。

    相同就是"科研写进面试目录"那条老路(复审 I4 失效场景 B)。
    """
    interview_sid = client.post("/interview/start").json()["session_id"]
    research_sid = client.post("/research/start").json()["session_id"]

    assert interview_sid != research_sid, (
        f"科研与面试拿到了同一个号 {research_sid} —— 科研回答会被写进面试会话目录")
