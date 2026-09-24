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

import logging

from .funasr_engine import AsrUtterance, FunASREngine

logger = logging.getLogger(__name__)

asr_engine = FunASREngine()


def transcribe(pcm: bytes) -> AsrUtterance:
    """把 16 kHz / 16 bit / 单声道 PCM 交给 ASR 引擎。异常上抛,不在这里吞。

    ⚠️ 同步函数,内部走的是 `funasr_client.recognize_pcm` → `asyncio.run`:
    **不能在已有事件循环里直接调用**(会 RuntimeError)。FastAPI 的端点都是 async,
    所以四处调用点一律写成 `await asyncio.to_thread(transcribe, audio_data)`
    (见 Ruling M1-6)。脚本与管道这类没有事件循环的地方照常直接调用。
    """
    return asr_engine.transcribe_pcm(pcm)


def log_recognition(utt: AsrUtterance) -> None:
    """把一次识别的**结果摘要**写进服务日志 —— 只写长度与段数,**绝不写原句**。

    为什么单独一个函数、而不是让端点自己拼那句日志:`logging_config.setup_logging()`
    在**仓库内**装了 RotatingFileHandler(`data/logs/jingxin.log`)。原句一旦经过
    `logger.*`,就被写回了仓库 —— 而 spec D2 的全部意义就是让转写原句**只**落在
    仓库外的 `~/shared/jingxin_recordings/{session_id}/transcript.json`,验收判据之
    一更是 `grep -rn "<原句里的短语>" ~/jingxin → 0`。写进服务日志就等于把 M1 刚搬
    出去的内容又搬回来(见最终审查 I2)。

    日志仍要能回答"识别到什么程度了":长度 0 = 没识别到,长度异常短 = 音频太短。
    所以写长度,不是什么都不写。
    """
    text = (utt.text or "").strip()
    # 用 %-style 惰性格式化(不是 f-string):内容与参数分离,这句日志里
    # **没有**任何位置能装下原句 —— 见 tests/test_asr_log_privacy.py 的扫描守卫。
    logger.info("识别结果: %d 字, %d 段, vad_split=%s",
                len(text), utt.n_segments, utt.vad_split)
