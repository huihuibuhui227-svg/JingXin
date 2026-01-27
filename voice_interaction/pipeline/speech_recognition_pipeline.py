"""
语音识别管道

提供从音频输入到文本输出的完整处理流程
"""

import os
import json
import queue
import logging
import numpy as np
from typing import Tuple, Optional
from vosk import Model, KaldiRecognizer
import sounddevice as sd
from ..models.voice_models import AudioData, SpeechRecognitionResult

logger = logging.getLogger(__name__)

# 自动定位模型路径
ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
MODEL_PATH = os.path.join(ROOT_DIR, "vosk-model-cn-0.22")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Vosk 模型未找到: {os.path.abspath(MODEL_PATH)}")


class SpeechRecognitionPipeline:
    """语音识别管道"""

    def __init__(self, sample_rate: int = 16000):
        """
        初始化语音识别管道

        参数:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.model = Model(MODEL_PATH)
        logger.info("Vosk 模型加载完成")

    def recognize_from_file(self, audio_file: str) -> SpeechRecognitionResult:
        """
        从音频文件识别语音

        参数:
            audio_file: 音频文件路径

        返回:
            语音识别结果
        """
        # TODO: 实现从文件识别的功能
        raise NotImplementedError("从文件识别功能尚未实现")

    def recognize_from_audio(
        self, 
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> SpeechRecognitionResult:
        """
        从音频数据识别语音

        参数:
            audio_data: 音频数据
            sample_rate: 音频采样率（如果为None则使用初始化时的采样率）

        返回:
            语音识别结果
        """
        sr = sample_rate or self.sample_rate

        # 创建音频数据对象
        audio_obj = AudioData(audio_data, sr)

        # 如果采样率不匹配，需要重采样
        if sr != self.sample_rate:
            # TODO: 实现重采样
            logger.warning(f"采样率不匹配: {sr} vs {self.sample_rate}，需要重采样")

        # 创建识别器
        recognizer = KaldiRecognizer(self.model, self.sample_rate)

        # 处理音频
        recognized_text = ""
        if isinstance(audio_data[0], np.int16):
            audio_bytes = audio_data.tobytes()
        else:
            audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()

        if recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(recognizer.Result())
            recognized_text = result.get("text", "").strip()
        else:
            result = json.loads(recognizer.PartialResult())
            partial = result.get("partial", "")
            if partial:
                recognized_text = partial

        # 创建识别结果
        recognition_result = SpeechRecognitionResult(
            text=recognized_text,
            confidence=1.0,  # Vosk不提供置信度，使用默认值
            is_final=True,
            audio_data=audio_obj
        )

        return recognition_result

    def listen_for_speech(
        self, 
        timeout: int = 30,
        pause_threshold: float = 1.2
    ) -> Tuple[SpeechRecognitionResult, np.ndarray]:
        """
        实时监听语音

        参数:
            timeout: 超时时间（秒）
            pause_threshold: 停顿阈值（秒）

        返回:
            (语音识别结果, 音频数据)
        """
        q = queue.Queue()
        audio_chunks = []
        recognized_text = ""

        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"录音状态: {status}")
            if not isinstance(indata, bytes):
                indata = bytes(indata)
            q.put(indata)
            audio_chunks.append(indata)

        print("🎤 请回答（说完后稍作停顿即可）...")

        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=8000,
                dtype='int16',
                channels=1,
                callback=callback
            ):
                recognizer = KaldiRecognizer(self.model, self.sample_rate)
                silence_count = 0
                total_time = 0.0

                while True:
                    try:
                        data = q.get(timeout=1.0)
                    except queue.Empty:
                        silence_count += 10
                        total_time += 1.0
                        if total_time > timeout:
                            break
                        continue

                    if not isinstance(data, bytes):
                        data = bytes(data)

                    total_time += 0.1

                    # 累积识别结果
                    if recognizer.AcceptWaveform(data):
                        res = json.loads(recognizer.Result())
                        text_chunk = res.get("text", "").strip()
                        if text_chunk:
                            recognized_text += text_chunk + " "
                            silence_count = 0
                            print(f"📝 你说的是: '{text_chunk}'")
                    else:
                        partial = json.loads(recognizer.PartialResult()).get("partial", "")
                        if not partial:
                            silence_count += 1
                        else:
                            silence_count = 0

                    if silence_count > 20 or total_time > timeout:
                        break

                # 合并音频
                if audio_chunks:
                    full_bytes = b''.join(audio_chunks)
                    audio_int16 = np.frombuffer(full_bytes, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                else:
                    audio_float32 = np.array([])

                if not recognized_text:
                    print("📝 未识别到有效语音")
                else:
                    print(f"✅ 最终识别结果: '{recognized_text.strip()}'")

                # 创建识别结果
                recognition_result = SpeechRecognitionResult(
                    text=recognized_text.strip(),
                    confidence=1.0,
                    is_final=True
                )

                return recognition_result, audio_float32

        except Exception as e:
            logger.error(f"录音或识别异常: {e}")
            print(f"❌ 语音交互失败: {e}")
            return SpeechRecognitionResult(text=""), np.array([])
