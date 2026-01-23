# voice_interaction/analyzers/tts_engine.py
"""
安全 TTS 引擎（避免 Windows 崩溃）
"""

import pyttsx3
import logging

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 180)
            self.engine.setProperty('volume', 0.9)
        except Exception as e:
            logger.error(f"TTS 初始化失败: {e}")
            self.engine = None

    def is_available(self) -> bool:
        """检查 TTS 引擎是否可用"""
        return self.engine is not None

    def speak(self, text: str) -> bool:
        if not self.is_available() or not text.strip():
            return False
        try:
            self.engine.say(text)
            self.engine.runAndWait()  # 关键！
            return True
        except Exception as e:
            print(f"TTS 错误: {e}")
            return False

    def stop(self):
        """安全停止"""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass