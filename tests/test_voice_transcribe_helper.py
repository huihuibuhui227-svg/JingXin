"""M1 Task 4:端点转写缝(`_transcribe`)与 vosk 拆除的守卫。

为什么本文件 import 的是 `voice_interaction.asr.transcribe`、而不是
`voice_interaction.api.app`:后者一 import 就构造 TTS 引擎与两条评估管线(慢,而且
会因为与本次改动无关的原因失败)—— 见 Ruling M1-14。缝被抽到一个只持有引擎的轻
模块里,测试只碰这个模块和 `asr_config.json`。

端点本身的正确性不在本文件:HTTP 层沿用哪些键、错误码是 400 还是 500,由 Task 7 的
端到端验收(真会话)压。这里不 import app,也就不可能"顺手断言到端点"却以为压住了它。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from voice_interaction.asr import transcribe as transcribe_mod
from voice_interaction.asr.funasr_engine import AsrUtterance

PCM = b"\x00\x01" * 160          # 320 字节。本层不解析音频,内容无所谓


class _FakeEngine:
    """按预设返回一个 AsrUtterance,或按预设抛错;顺带记下收到的 PCM。"""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.seen: list[bytes] = []

    def transcribe_pcm(self, pcm: bytes) -> AsrUtterance:
        self.seen.append(pcm)
        if self.error is not None:
            raise self.error
        return self.result


def test_transcribe_helper_delegates_to_engine_verbatim(monkeypatch):
    """缝只做转发:PCM 原样进引擎,引擎的 AsrUtterance **原样**出。

    断言身份(`is`)与"带空白的原文不被裁剪"两件事,是为了钉死"什么都不做":
    只有 `return asr_engine.transcribe_pcm(pcm)` 这一句能同时满足两者 —— 任何在这里
    重建对象、或顺手 `.strip()` 的实现都会红(strip 属于调用方:四处调用点各自
    `utt.text.strip()` 后再判空)。
    """
    utt = AsrUtterance(text=" 然后我们说 ", n_chars=5, n_segments=1, vad_split=False,
                       segments=[])
    engine = _FakeEngine(result=utt)
    monkeypatch.setattr(transcribe_mod, "asr_engine", engine)

    out = transcribe_mod.transcribe(PCM)

    assert engine.seen == [PCM]
    assert out is utt


def test_transcribe_helper_does_not_swallow_engine_errors(monkeypatch):
    """引擎抛错就抛错:缝里吞掉异常会把它变成"识别为空"。

    这个区别在端点上就是"服务不可用(500)"与"没听清(400)"的区别 —— 吞掉之后
    用户会以为是自己没说清,而真正的原因(ws 连不上)在报告里消失。
    """
    engine = _FakeEngine(error=RuntimeError("ws closed"))
    monkeypatch.setattr(transcribe_mod, "asr_engine", engine)

    with pytest.raises(RuntimeError, match="ws closed"):
        transcribe_mod.transcribe(PCM)


def test_transcribe_helper_never_turns_empty_text_into_a_placeholder(monkeypatch):
    """识别为空就返回空文本,不补 "0" 之类的占位。

    空与 "0" 在上层是两件事:`/interview/answer_audio` 靠空串判 400(未识别到有效
    语音),而 "0" 会被当成一句真回答送进评估,再被算进连接词密度的分母。
    """
    engine = _FakeEngine(result=AsrUtterance(text="", n_chars=0, n_segments=0,
                                             vad_split=False, segments=[]))
    monkeypatch.setattr(transcribe_mod, "asr_engine", engine)

    assert transcribe_mod.transcribe(PCM).text == ""


# ---------------------------------------------------------------------------
# vosk 拆除的守卫
# ---------------------------------------------------------------------------

VOSK_PAT = re.compile(r"vosk|KaldiRecognizer", re.IGNORECASE)

# 与验收时用的那条命令同一范围:`voice_interaction/` + `report_frontend/`。
PKG_ROOTS = ("voice_interaction", "report_frontend")

# 本任务必须改掉的两个识别点所在文件。把它们钉进"扫描集必须覆盖"的断言里:
# 否则 glob 一旦写歪(路径改了、rglob 写成了 glob),本测试会因为"一个文件都没扫到"
# 而静默变绿 —— 这正是本项目栽过的那类假绿。
REQUIRED_FILES = {"voice_interaction/api/app.py",
                  "voice_interaction/pipeline/speech_recognition_pipeline.py"}


def test_no_vosk_reference_left_in_voice_package():
    """voice 包与报告前端里都不许再出现 vosk / KaldiRecognizer。

    vosk-model-cn-0.22(2.0 GB)已随本任务删除,`pip install vosk` 也不再是依赖:
    任何残留引用都是会在运行时炸掉的死代码。"删除模型"这个动作不可逆,所以这条
    文本级守卫是它唯一的回归闸。
    """
    root = Path(__file__).resolve().parent.parent
    files = [f for pkg in PKG_ROOTS for f in sorted((root / pkg).rglob("*.py"))]
    rel = {str(f.relative_to(root)) for f in files}

    missing = REQUIRED_FILES - rel
    assert not missing, f"扫描集没覆盖识别点所在文件,本测试会假绿:{sorted(missing)}"

    hits = {}
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        nums = [i for i, line in enumerate(lines, 1) if VOSK_PAT.search(line)]
        if nums:
            hits[str(f.relative_to(root))] = nums
    assert hits == {}, f"仍有 vosk 引用:{hits}"
