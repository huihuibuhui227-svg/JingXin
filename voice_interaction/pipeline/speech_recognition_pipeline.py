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
import time
import numpy as np
from typing import Sequence, Tuple, Optional
import sounddevice as sd
from ..asr.funasr_engine import FunASREngine, load_config
from ..models.voice_models import AudioData, SpeechRecognitionResult

logger = logging.getLogger(__name__)

# 采集块大小(帧):16 kHz 下一块 = 0.5 s,停录判定的粒度就是它(见 should_stop)。
# 这是**设备参数**而不是可调阈值,故留在代码里做具名常量;真正的阈值(静音能量下限)
# 一律来自 asr_config.json,停顿秒数则由调用方传参。
BLOCKSIZE = 8000


def chunk_energy(chunk: bytes) -> float:
    """一块 16 bit 单声道 PCM 的能量 = 归一化 RMS(0..1)。空块 → 0.0。

    判静音必须看**幅值**,不能看"有没有收到数据":静音也是数据 ——
    `sd.RawInputStream` 每 0.5 s 必送一块,而全零的 `bytes` 在 Python 里是真值。
    """
    if not chunk:
        return 0.0
    samples = np.frombuffer(chunk, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))


def should_stop(chunk_energies: Sequence[float], pause_threshold_s: float,
                chunk_seconds: float, energy_floor: float) -> bool:
    """末尾连续静音已达 `pause_threshold_s` → 该停录了。

    - **静音** = 能量不高于 `energy_floor`(安静房间的低幅底噪也算静音)
    - 只数**末尾**那一段连续静音:说话中的短暂停顿(不足阈值)不算说完,否则一句话会
      被从中间截断
    - 还没开口(所有块都没有超过下限)时恒为 False:留给 `timeout` 兜底,否则一进循环
      就停,用户来不及说第一句
    - 粒度是**一块**(`chunk_seconds` = 0.5 s):阈值低于一块时实际等同于一块
      (第一块静音就停)

    `energy_floor` 由调用方传入(pipeline 在 `__init__` 里从 asr_config.json 取),
    本函数因此是纯函数:不读配置、不碰设备、可离线测。
    """
    if not any(energy > energy_floor for energy in chunk_energies):
        return False
    quiet_chunks = 0
    for energy in reversed(chunk_energies):
        if energy > energy_floor:
            break
        quiet_chunks += 1
    return quiet_chunks * chunk_seconds >= pause_threshold_s


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
        # 静音能量下限同样只来自配置 —— 不在代码里写裸常量(缺键就在这里响亮地炸)
        self.silence_energy_floor = float(load_config()["silence_energy_floor"]["value"])
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

        停录判据是**能量**(`chunk_energy` → `should_stop`),不是"队列里还有没有数据":
        `sd.RawInputStream` 每 0.5 s 必送一块,**静音也是音频数据**(全零的 `bytes`
        也是真值),所以按"有没有数据"判静音会让计时被每一块重置、`pause_threshold`
        永远到不了、每次录音都录满 `timeout`(这条死规则在本任务的第一版里真实存在过,
        审查用假设备复现后才修掉)。

        粒度是一块(0.5 s):`pause_threshold` 低于 0.5 s 时等同于 0.5 s(第一块静音就停)
        —— 句中停顿比这短就会被截断,调用方要按自己的场景给够。
        还没开口时不会因为静音而提前停(`should_stop` 的前置判断),交给 `timeout` 兜底。
        """
        q = queue.Queue()
        audio_chunks = []
        chunk_seconds = BLOCKSIZE / self.sample_rate

        def callback(indata, frames, time, status):
            # 注意:形参 `time` 只在本回调作用域内遮蔽时间模块(回调里不用时间函数)
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
                blocksize=BLOCKSIZE,
                dtype='int16',
                channels=1,
                callback=callback
            ):
                chunk_energies: list[float] = []
                started = time.monotonic()

                while time.monotonic() - started <= timeout:
                    try:
                        data = q.get(timeout=0.1)
                    except queue.Empty:
                        continue          # 没块到:计时由 timeout 管,静音由能量管

                    chunk_energies.append(chunk_energy(data))
                    if should_stop(chunk_energies, pause_threshold, chunk_seconds,
                                   self.silence_energy_floor):
                        print("⏸  连续静音超过阈值，结束录音")
                        break

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
