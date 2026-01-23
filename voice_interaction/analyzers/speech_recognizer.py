import os
import json
import queue
import logging
import numpy as np
from typing import Tuple
from vosk import Model, KaldiRecognizer
import sounddevice as sd

logger = logging.getLogger(__name__)

# 自动定位模型路径
ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
MODEL_PATH = os.path.join(ROOT_DIR, "vosk-model-small-cn-0.22")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Vosk 模型未找到: {os.path.abspath(MODEL_PATH)}")


class SpeechRecognizer:
    def __init__(self):
        self.sample_rate = 16000
        logger.info("正在加载 Vosk 模型...")
        self.model = Model(MODEL_PATH)
        logger.info("✅ Vosk 模型加载完成")

    def listen_for_speech(self, timeout: int = 30) -> Tuple[str, np.ndarray]:
        q = queue.Queue()
        audio_chunks = []
        recognized_text = ""  # 存储最终识别文本

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

                    # 关键：累积所有识别结果
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

                # 不要再调用 FinalResult()！
                recognized_text = recognized_text.strip()

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
                    print(f"✅ 最终识别结果: '{recognized_text}'")

                return recognized_text, audio_float32

        except Exception as e:
            logger.error(f"录音或识别异常: {e}")
            print(f"❌ 语音交互失败: {e}")
            return "", np.array([])