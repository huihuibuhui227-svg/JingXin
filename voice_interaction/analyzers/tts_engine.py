"""
语音合成模块

提供基于 pyttsx3 的文本转语音功能，包含语音选择、错误恢复和资源管理。
"""

import pyttsx3
from typing import Optional, Dict, Any
from ..config import TTS_CONFIG


class TTSEngine:
    """文本转语音引擎"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化TTS引擎

        参数:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or TTS_CONFIG.copy()
        self.engine: Optional[pyttsx3.Engine] = None
        self._is_initialized = False
        self._init_engine()

    def _init_engine(self) -> None:
        """初始化语音引擎（带错误恢复）"""
        try:
            if self.engine:
                self.engine.stop()

            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.config['rate'])
            self.engine.setProperty('volume', self.config['volume'])

            # 选择中文语音
            voices = self.engine.getProperty('voices')
            selected_voice = None

            if self.config['voice_preference'] == 'chinese':
                for voice in voices:
                    voice_info = (voice.name + voice.id).lower()
                    if any(keyword in voice_info for keyword in ['zh', 'chinese', 'huihui', 'tingting']):
                        selected_voice = voice
                        break

            if not selected_voice and voices:
                selected_voice = voices[0]

            if selected_voice:
                self.engine.setProperty('voice', selected_voice.id)
                print(f"🔊 已选择语音: {selected_voice.name}")

            self._is_initialized = True

        except Exception as e:
            print(f"⚠️ TTS 引擎初始化失败: {e}")
            self._is_initialized = False

    def speak(self, text: str) -> bool:
        """
        朗读文本

        参数:
            text: 要朗读的文本

        返回:
            是否成功朗读
        """
        if not text.strip():
            return True  # 空文本视为成功

        print(f"📢 {text}")

        if not self._is_initialized:
            print("⚠️ TTS 引擎未初始化，跳过朗读")
            return False

        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"❌ TTS 错误: {e}")
            # 尝试重新初始化
            self._init_engine()
            return False

    def stop(self) -> None:
        """停止语音播放"""
        if self.engine and self._is_initialized:
            try:
                self.engine.stop()
            except Exception as e:
                print(f"⚠️ 停止 TTS 时出错: {e}")

    def is_available(self) -> bool:
        """检查 TTS 引擎是否可用"""
        return self._is_initialized

    def __del__(self) -> None:
        """析构函数，清理资源"""
        self.stop()