# tests/test_question_total_is_reported.py
"""题库有几题,必须由服务端告诉前端 —— 前端那个 `/10` 是**写死的**。

实测(2026-09-26 场 `20260926_124439_0f5b`,使用者当场口述):
「我这边显示的是问题七杠十,但那边进度的话显示是六杠十」——两处显示同一个数,
一处 `currentIndex+1`、一处没加一,所以**永远差 1**;而分母 `10` 与题库(8 题)
无关,进度条因此永远到不了 100%。

差 1 是前端自己的 bug(改一处);分母则要靠服务端给 —— 前端手里没有任何
"共几题"的来源(`setQuestions` 是死代码,从未被调用)。两个入口都要给:
面试与科研是**对称的两个端点**,只给一个会让另一边继续编分母(第 19 条的教训)。
"""
import importlib
import re

import pytest
from fastapi.testclient import TestClient

from voice_interaction.api import app as voice_api

_voice_app_module = importlib.import_module("voice_interaction.api.app")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "rec"))
    monkeypatch.setattr(_voice_app_module.tts_engine, "speak", lambda *a, **k: None)
    import voice_interaction.utils.logger as voice_logger_module
    monkeypatch.setattr(voice_logger_module, "LOGS_DIR", str(tmp_path / "logs"))
    return TestClient(voice_api)


def test_interview_start_reports_the_question_total(client):
    """红法:返回体里没有 `total_questions` 这个键 —— 前端只能继续用写死的 10。"""
    body = client.post("/interview/start").json()
    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{4}", body.get("session_id", "")), (
        f"前提不成立(没铸号):{body}")
    assert body.get("total_questions") == len(_voice_app_module.interview_assessment.questions)
    assert body["total_questions"] > 0, "分母不该是 0"


def test_research_start_reports_the_question_total(client):
    """科研那条路同样要给 —— 两个入口对称(第 19 条)。"""
    body = client.post("/research/start").json()
    assert body.get("total_questions") == len(_voice_app_module.research_assessment.questions)
    assert body["total_questions"] > 0, "分母不该是 0"
