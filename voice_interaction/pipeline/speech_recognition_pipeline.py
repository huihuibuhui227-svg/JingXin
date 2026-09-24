"""
语音识别管道

提供从音频输入到文本输出的完整处理流程

识别后端是 FunASR(`asr/funasr_engine.py`,经 websocket 调远端服务):不再有本地声学
模型,旧后端与它那 2.0 GB 的中文模型已随本次改动一并删除。这条链路是**整段**识别:
一次调用给一段 PCM,没有逐块的 partial 结果 —— 所以 `listen_for_speech` 的循环只
负责"录到一段完整的话",识别在循环之后一次做完。
"""

import logging
import queue
import numpy as np
from typing import Tuple, Optional
import sounddevice as sd
from ..asr.funasr_engine import FunASREngine
from ..models.voice_models import AudioData, SpeechRecognitionResult

logger = logging.getLogger(__name__)


class SpeechRecognitionPipeline:
    """语音识别管道"""

    def __init__(self, sample_rate: int = 16000):
        """
        初始化语音识别管道

        参数:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        # 引擎从 asr_config.json 取 host/port/timeout(构造是纯读配置,不连网)
        self.engine = FunASREngine()
        logger.info("FunASR 引擎就绪")

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

        异常:上抛(与 `asr/transcribe.py` 的缝一致)。识别失败不是"没听清",不该被
        这里改写成空文本 —— 那样脚本会把服务故障当成用户没说话。
        """
        sr = sample_rate or self.sample_rate

        # 创建音频数据对象
        audio_obj = AudioData(audio_data, sr)

        # 如果采样率不匹配，需要重采样
        if sr != self.sample_rate:
            # TODO: 实现重采样
            logger.warning(f"采样率不匹配: {sr} vs {self.sample_rate}，需要重采样")

        # 转成 16 bit 单声道 PCM 字节交给引擎
        if isinstance(audio_data[0], np.int16):
            audio_bytes = audio_data.tobytes()
        else:
            audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()

        recognized_text = self.engine.transcribe_pcm(audio_bytes).text.strip()

        # 创建识别结果
        # confidence 恒为 1.0 是**占位**:本部署的 raw 里没有置信度字段
        # (asr_config 的 provenance 记着 asr_confidence_source="unavailable")。
        # 旧后端同样不提供置信度,故沿用同一个占位值,免得下游拿到 None/0 变行为。
        recognition_result = SpeechRecognitionResult(
            text=recognized_text,
            confidence=1.0,
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
            timeout: 最长录音时间（秒）
            pause_threshold: 停顿阈值（秒）：说话之后连续静音这么久就认为说完了

        返回:
            (语音识别结果, 音频数据)

        旧后端靠逐块 partial 判"还在不在说";FunASR 这条链路没有 partial,于是
        判据换成"还有没有音频块进来",`pause_threshold` 因此第一次真正生效
        (它此前是签名里从未被用到的参数)。没开口之前不会因为静音而提前结束 ——
        要等 `audio_chunks` 非空,否则一进来就会立刻超时返回空结果。
        """
        q = queue.Queue()
        audio_chunks = []

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
                silence_sec = 0.0
                total_time = 0.0

                while True:
                    if total_time > timeout:
                        break
                    try:
                        data = q.get(timeout=0.1)
                    except queue.Empty:
                        silence_sec += 0.1
                        total_time += 0.1
                        # 必须已经收到过音频:否则开头 1.2 秒的安静就被当成"说完了"
                        if audio_chunks and silence_sec >= pause_threshold:
                            print("⏸  停顿超过阈值，结束录音")
                            break
                        continue

                    total_time += 0.1
                    if data:
                        silence_sec = 0.0          # 还在说

                # 合并音频
                full_bytes = b''.join(audio_chunks) if audio_chunks else b''
                if full_bytes:
                    audio_int16 = np.frombuffer(full_bytes, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                else:
                    audio_float32 = np.array([])

                # 整段识别(FunASR 没有逐块 partial,识别放在循环之后一次做完)
                recognized_text = ""
                if full_bytes:
                    recognized_text = self.engine.transcribe_pcm(full_bytes).text.strip()

                if not recognized_text:
                    print("📝 未识别到有效语音")
                else:
                    print(f"✅ 最终识别结果: '{recognized_text}'")

                # 创建识别结果
                recognition_result = SpeechRecognitionResult(
                    text=recognized_text,
                    confidence=1.0,
                    is_final=True
                )

                return recognition_result, audio_float32

        except Exception as e:
            # 交互式录音路径:崩掉整个脚本比返回空结果更糟,所以这里兜住
            logger.error(f"录音或识别异常: {e}")
            print(f"❌ 语音交互失败: {e}")
            return SpeechRecognitionResult(text=""), np.array([])
