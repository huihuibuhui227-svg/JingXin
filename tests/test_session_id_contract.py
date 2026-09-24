"""`NONE_SESSION` 的值在四个文件里各写一份,靠这条文本级测试守住同值。

为什么不是 import:face/gesture 从 voice 包导入方向是反的(还会拉起 TTS/vosk),
而 voice 内部从 `..asr.session` 取又会触发整个包的导入 —— 见 Ruling M1-2。
所以四个文件各持一份字面量,本测试读源码做交叉校验。
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    ROOT / "voice_interaction/asr/session.py",   # 定义处(不属本任务,只作基准)
    ROOT / "voice_interaction/utils/logger.py",
    ROOT / "face_expression/utils/logger.py",
    ROOT / "gesture_analysis/utils/logger.py",
]


def test_none_session_literal_is_identical_everywhere():
    """四处的字面量必须完全一致,否则报告侧的排除规则只对一部分日志生效。

    这条同时守住"各写各的":把某一处的 `NONE_SESSION = "NONE"` 换成从别处
    import,该文件就不再匹配,断言 `m` 直接失败。
    """
    seen = set()
    for src in SOURCES:
        text = src.read_text(encoding="utf-8")
        m = re.search(r'NONE_SESSION\s*=\s*"([^"]+)"', text)
        assert m, f"{src} 缺少 NONE_SESSION 常量"
        seen.add(m.group(1))
    assert seen == {"NONE"}, seen
