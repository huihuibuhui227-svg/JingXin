"""
文本转语音管道

提供从文本到音频输出的完整处理流程
"""

import logging
import threading
import queue
import time
from typing import Optional
from ..models.voice_models import AudioData

logger = logging.getLogger(__name__)


class TTSPipeline:
    """文本转语音管道（线程安全）"""

    def __init__(self, rate: int = 180, volume: float = 0.9):
        """
        初始化TTS管道

        参数:
            rate: 语速
            volume: 音量 (0.0-1.0)
        """
        self.rate = rate
        self.volume = volume
        self.engine = None
        self._speech_queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()

        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)

            # 启动后台工作线程
            self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self._worker_thread.start()

            logger.info("✅ TTS引擎初始化成功（线程安全模式）")
            print("✅ TTS引擎已启动，后台线程运行中")
        except Exception as e:
            logger.error(f"❌ TTS初始化失败: {e}")
            print(f"❌ TTS初始化失败: {e}")
            self.engine = None

    def _speech_worker(self):
        """后台语音播放工作线程"""
        print("🎤 TTS后台工作线程已启动")

        while not self._stop_event.is_set():
            try:
                # 使用阻塞方式获取队列项
                text = self._speech_queue.get(timeout=1.0)

                if text is None:  # 停止信号
                    print("🛑 收到TTS停止信号")
                    break

                if self.engine and text:
                    try:
                        print(f"🔊 正在播放: {text[:50]}...")

                        # 直接播放（不先stop）
                        self.engine.say(text)
                        self.engine.runAndWait()

                        print("✅ 播放完成")

                        # 关键：播放完成后等待一小段时间，让引擎完全释放
                        time.sleep(0.5)

                    except Exception as e:
                        error_msg = str(e)
                        print(f"❌ TTS播放错误: {error_msg}")

                        # 无论什么错误，都重新初始化引擎
                        try:
                            import pyttsx3
                            print("🔄 重新初始化TTS引擎...")
                            self.engine = pyttsx3.init()
                            self.engine.setProperty('rate', self.rate)
                            self.engine.setProperty('volume', self.volume)
                            print("✅ TTS引擎已重新初始化")
                        except Exception as reinit_error:
                            print(f"❌ 重新初始化失败: {reinit_error}")
                            self.engine = None

                self._speech_queue.task_done()

            except queue.Empty:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                logger.error(f"❌ 语音工作线程错误: {e}")
                print(f"❌ 语音工作线程错误: {e}")

        print("🛑 TTS后台工作线程已停止")

    def is_available(self) -> bool:
        """检查TTS引擎是否可用"""
        return self.engine is not None

    def text_to_audio(self, text: str) -> Optional[AudioData]:
        """
        将文本转换为音频数据
        """
        if not self.is_available() or not text.strip():
            return None

        logger.warning("text_to_audio功能尚未完全实现")
        return None

    def speak(self, text: str) -> bool:
        """
        朗读文本（非阻塞，线程安全）

        参数:
            text: 要朗读的文本

        返回:
            是否成功加入队列
        """
        if not self.is_available() or not text.strip():
            return False

        try:
            # 不再清空队列，让任务自然执行
            self._speech_queue.put(text, timeout=2)
            print(f"📥 语音已加入队列: {text[:30]}...")
            return True
        except queue.Full:
            logger.warning("⚠️ 语音队列已满，跳过本次播放")
            print("⚠️ 语音队列已满")
            return False

    def speak_multiple(self, texts: list) -> bool:
        """
        朗读多个文本
        """
        if not self.is_available() or not texts:
            return False

        success = True
        for text in texts:
            if not self.speak(text):
                success = False

        return success

    def set_rate(self, rate: int) -> bool:
        """设置语速"""
        if not self.is_available():
            return False

        try:
            self.rate = rate
            self.engine.setProperty('rate', rate)
            return True
        except Exception as e:
            logger.error(f"设置语速失败: {e}")
            return False

    def set_volume(self, volume: float) -> bool:
        """设置音量"""
        if not self.is_available():
            return False

        try:
            volume = max(0.0, min(1.0, volume))
            self.volume = volume
            self.engine.setProperty('volume', volume)
            return True
        except Exception as e:
            logger.error(f"设置音量失败: {e}")
            return False

    def stop(self) -> None:
        """停止当前播放并关闭引擎"""
        print("🛑 正在停止TTS引擎...")
        self._stop_event.set()
        self._speech_queue.put(None)  # 发送停止信号

        if self._worker_thread:
            self._worker_thread.join(timeout=3)

        if self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                logger.error(f"停止TTS失败: {e}")

    def clear_queue(self) -> None:
        """清空语音队列"""
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get_nowait()
                self._speech_queue.task_done()
            except queue.Empty:
                break
