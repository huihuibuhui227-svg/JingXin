"""端点侧的转写缝:一层的转发,好让端点与「引擎」解耦。

为什么单独一个模块(而不是写在 `api/app.py` 里,见 Ruling M1-14):
`api/app.py` 一 import 就构造 TTS 引擎与两条评估管线(还可能因为没有模型、
没有声卡之类与转写毫无关系的原因失败)。缝留在这里,测试就只 import 这个模块 +
`asr_config.json` —— 不碰 app、不碰 TTS、不碰评估管线。

`asr_engine` 是模块级可替换属性:测试与验收(真会话以外的场合)把它换成假引擎,
不用连网、也不用起 FunASR 服务。

本层**不做任何加工**:不 strip、不补占位文本、也不吞异常。三件事都各有其主:
  - strip / 判空 → 调用方(端点要按空串回 400「未识别到有效语音」)
  - 异常      → 调用方(端点把它转成 5xx,而不是伪装成"没听清")
  - 占位文本  → 没有这样的需求:空与 "0" 在上层是两件不同的事
"""

from __future__ import annotations

from .funasr_engine import AsrUtterance, FunASREngine

asr_engine = FunASREngine()


def transcribe(pcm: bytes) -> AsrUtterance:
    """把 16 kHz / 16 bit / 单声道 PCM 交给 ASR 引擎。异常上抛,不在这里吞。

    ⚠️ 同步函数,内部走的是 `funasr_client.recognize_pcm` → `asyncio.run`:
    **不能在已有事件循环里直接调用**(会 RuntimeError)。FastAPI 的端点都是 async,
    所以四处调用点一律写成 `await asyncio.to_thread(transcribe, audio_data)`
    (见 Ruling M1-6)。脚本与管道这类没有事件循环的地方照常直接调用。
    """
    return asr_engine.transcribe_pcm(pcm)
