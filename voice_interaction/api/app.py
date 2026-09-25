# api/app.py
"""
FastAPI应用

提供语音交互的Web API接口
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import os
import logging
import wave
import io

import numpy as np

from logging_config import setup_logging
import media_retention
import session_meta

setup_logging()
logger = logging.getLogger(__name__)

# 导入项目模块（使用绝对导入）
from voice_interaction.pipeline.tts_pipeline import TTSPipeline as TTSEngine
from voice_interaction.pipeline.assessment_pipeline import InterviewAssessmentPipeline, ResearchAssessmentPipeline
from voice_interaction.core.feature_extraction.prosody_extractor import ProsodyFeatureExtractor
from voice_interaction.config import API_CONFIG, FFMPEG_PATH
from voice_interaction.utils.logger import VoiceLogger
from voice_interaction.asr import session as session_mod, transcript_store
from voice_interaction.asr.connective_density import connective_density
from voice_interaction.asr.funasr_engine import load_config
from voice_interaction.asr.transcript_store import (normalize_session_id,
                                                    validate_session_id)
# 转写缝:引擎只由 asr/transcribe.py 持有,这里按名字取那一层转发(不直接摸引擎)。
# 换引擎(测试/验收)只需动那一个模块,不必碰本文件。
from voice_interaction.asr.transcribe import log_recognition
from voice_interaction.asr.transcribe import transcribe as _transcribe

app = FastAPI(
    title="Voice Interaction API",
    description="语音交互API，提供语音识别、语音合成和面试评估功能"
)

# 添加 CORS 中间件
cors_origins = os.getenv('CORS_ORIGINS', 'http://127.0.0.1:5000,http://localhost:5000,http://localhost:5173,http://127.0.0.1:5173').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... existing code ...

# ================== 初始化组件 ==================

tts_engine = TTSEngine()
interview_assessment = InterviewAssessmentPipeline()
research_assessment = ResearchAssessmentPipeline()
voice_logger = VoiceLogger(log_type='interview')
# 科研那条日志器同样要在**铸号时换成带号的那一个**(见 /research/start)。文件名是**构造函数**
# 算的 —— 事后改 `.session_id` 只换首列、不换文件名(M1 记过这个坑),所以不能沿用现造的。
research_logger = VoiceLogger(log_type='research')

# 音频契约:16 kHz / 16 bit / 单声道(四处 wave 校验与 ffmpeg 转换都用它)
SAMPLE_RATE = 16000

# 单次上传上限(终局复核 I7)。为什么需要它:`/session/{sid}/media` 收的是
# **整场**的原生视频,而盘就是 `D:\`(127 GB)。一个不受限的请求就能把它填满,
# 而**填满之后这一场剩下的每一次留存都会变成 degraded** —— 正好毁掉本里程碑
# 存在的理由(把素材留下来)。512 MB 对一场 30–60 分钟、1 Mbps 量级的 webm 很宽裕。
# 读的时候**分块**,不是先 read() 再判大小:后者在拒绝之前已经把整个文件读进内存了。
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

# 语调特征提取器。无状态、可跨请求共享(与上面三个单例同理)。
prosody_extractor = ProsodyFeatureExtractor(SAMPLE_RATE)

# 提取器产出名 → 日志列名。**这张表是必须的,不是装饰**:
# 提取器吐的是 `pitch_std` / `energy_std`,而 `VoiceLogger.fieldnames` 里的列叫
# `pitch_variation` / `energy_variation`,两边**不同名**。改名只发生在
# `ProsodyAnalyzer.analyze_pitch/analyze_energy`(那里写的就是 `"pitch_variation": pitch_std`),
# 而活路径直接吃提取器的返回值 —— 少了这张表,那两列会**静默留 0**,
# 其余每一列却都对,是最难发现的一类(守它的测试:`tests/test_prosody_wiring.py`)。
_EXTRACTOR_TO_LOG_COLUMNS = {
    "pitch_std": "pitch_variation",
    "energy_std": "energy_variation",
}


def _asr_meta() -> dict:
    """会话清单(session.json)里的 asr provenance 块(spec §6.4)。

    与 `transcript_store` 里那份同形、同源(都从 asr_config.json 现取),所以两处
    不会给出不同的值,代码里也不再存第二份 host/port。
    `asr_confidence` 恒为 null 且来源恒为 "unavailable":本部署的 raw 里没有置信度
    字段(实测,见 spec §3),必须显式落盘而不是省略(spec §9.1)。
    """
    cfg = load_config()
    return {
        "engine": "funasr",
        "endpoint": f"ws://{cfg['funasr_host']}:{cfg['funasr_port']}",
        "models": dict(cfg.get("models") or {}),
        "asr_confidence": None,
        "asr_confidence_source": "unavailable",
    }


async def _transcribe_async(pcm: bytes):
    """同步转写缝的 loop-safe 入口 —— 四个识别点一律走这里(Ruling M1-6)。

    为什么不能直接 `_transcribe(pcm)`:那一层同步调 `funasr_client.recognize_pcm`,
    而它内部是 `asyncio.run(...)`;FastAPI 的端点都跑在事件循环里,直接调会
    `RuntimeError: asyncio.run() cannot be called from a running event loop`。
    丢到工作线程后,`asyncio.run` 在那个线程里自建事件循环,既安全又不阻塞本循环
    (识别约 1.75 s,期间服务还能接别的请求)。
    """
    return await asyncio.to_thread(_transcribe, pcm)


def prosody_features_from_pcm(audio_data: bytes) -> dict:
    """把端点手里那段 PCM 变成 `VoiceLogger.log_prosody` 认的特征字典。

    **只吃 16 kHz / 16 bit / 单声道的小端 PCM** —— 调用点上游的 `wave` 校验已经
    保证了这一点(不满足会先 400),所以这里不重复做格式嗅探:嗅探只会让
    "格式不对"变成一处静默的近似,而上面那道闸是硬的。

    已知缺陷**照实登记、不在这里修**(接线 ≠ 定义;M3 逐列重构会重写提取器内部):

    - `speech_ratio` 用**自指阈值**(`centroid > mean(centroid) * 0.1`):阈值由本段
      自己的均值定,所以它天然贴着 1.0 —— 实测三段真回答为 0.99 / 1.00 / 1.00,
      MIT 那批 87.9% 恰为 1.0。spec §4.3 要删这一列。
      报告层已按 G4 封停(`evidence_gate.py`:「自指阈值,87.9% 恰为 1.0」),
      所以它出不了分;但它**已经真的在产出了**,别当成成果。
    - `pause_*` 的时长按"整窗安静"的帧数估:`librosa.feature.rms` 窗长 2048
      (0.128 s @16k),所以每段停顿的**两侧各被吃掉最多一个窗**,0.2 s 的静音
      实测只算得出 0.095 s(过不了提取器自己 `> 0.1 s` 的闸)。
      尾部静默另有登记缺陷(报告层 G4:「尾部静默被丢弃」)。
    - 提取器在"没人声"时给的是 **0.0 而不是"未测出"**,与"真的量到 0"不可分。
      这一层不做区分(`log_prosody` 的键缺省也是 0),真掩码是 M3 的事 ——
      见 `evidence_gate.py` 里 `is_valid` 那条「恒 1 且下游从未使用 → M3 改真掩码」。
      端点没有改 `is_valid`:本路径上 ASR 有文本才走到这里,音频必然非空,
      提取器也必然产出非空字典,派生出来的 `is_valid` 仍是**恒 1** ——
      那是"看着像修了"的假改动,不如把账记在 M3 名下。
    """
    audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    raw = prosody_extractor.extract_all_features(audio)
    return {_EXTRACTOR_TO_LOG_COLUMNS.get(k, k): v for k, v in raw.items()}


async def _prosody_async(pcm: bytes) -> dict:
    """`prosody_features_from_pcm` 的 loop-safe 入口 —— 与 `_transcribe_async` 同一条理由。

    `pyin` 是**同步重活**:实测 3 s 音频 ≈ 0.3 s、29.5 s ≈ 3.7 s,与 ASR 的 1.75 s
    同量级。端点都是 `async def`,在协程里直接算会卡住整个事件循环 ——
    期间服务连 `/health` 都不回(使用者那场会话每答一题都会撞上)。
    """
    return await asyncio.to_thread(prosody_features_from_pcm, pcm)


async def _resolve_session_id(request: Request, session_id: str | None) -> str:
    """取本次请求的会话 id:query 参数 `session_id` 或 multipart **表单字段**同名键。

    为什么两个位置都要认:请求体是 `multipart/form-data`(音频就是其中一个字段),前端
    很容易把 id 当**表单字段**提交。只认 query 的话,那种请求会拿 200、回答被静默归到
    `NONE` —— M1"三模块对上号"的目标无声失效,而且没有任何报错。

    `request.form()` 这里**不会**二次消费请求体:路由声明了 `File(...)`,FastAPI 在进端点
    之前已经把表单解析好并缓存在同一个 Request 上,这里只是读缓存(非表单请求会得到空
    FormData,故 except 仅是兜底)。

    校验也在这一步:先 `validate_session_id`(store 的公开守卫,id 会当目录名用),
    非法的 id 直接 400 —— 不必先花一次识别的时间再让它 500。
    """
    # 只认「参数不存在」为没给;给了空串也算**给了**,交给守卫判非法(见 normalize_session_id)
    if session_id is None:
        try:
            form = await request.form()
        except Exception:            # 不是表单请求 / 体已损坏 → 当作没给
            form = None
        # 用 `in` 而不是 `if value`:表单里给了空串与 query 同口径(给了 → 判非法)
        if form is not None and "session_id" in form:
            session_id = form.get("session_id")
    try:
        return validate_session_id(normalize_session_id(session_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class TextRequest(BaseModel):
    """文本请求模型"""
    text: str


class AnswerRequest(BaseModel):
    """回答请求模型"""
    answer: str


class QuestionWindow(BaseModel):
    """前端上报的一道题的提问窗口(spec §5.6)。

    时刻是**墙钟秒**(`time.time()` 量纲),**不是** M2.5 那个会话内相对时钟。
    两个基不要混:混了以后 `response_latency` 会算出一个看着正常、其实没意义的数。
    """
    qid: str          # 题目文本原文(题库没有 id,见 session_meta 模块开头)
    index: int        # 本场内的 0 基序号
    ask_start: float  # 推题那一刻
    ask_end: float    # **题问完那一刻**(不是回答提交时刻 —— 见下面端点)


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "Voice Interaction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - 健康检查",
            "/tts": "POST - 文本转语音",
            "/asr": "POST - 语音识别（上传WAV文件）",
            "/interview/start": "POST - 开始面试",
            "/interview/question": "GET - 获取下一个问题",
            "/interview/answer": "POST - 提交文本回答",
            "/interview/answer_audio": "POST - 提交语音回答（自动识别）",
            "/interview/evaluation": "GET - 获取综合评估",
            "/research/start": "POST - 开始科研评估",
            "/research/question": "GET - 获取下一个问题",
            "/research/answer": "POST - 提交文本回答",
            "/research/answer_audio": "POST - 提交语音回答（自动识别）",
            "/research/evaluation": "GET - 获取科研潜质评估"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


@app.post("/tts")
async def text_to_speech(request: TextRequest):
    """
    文本转语音
    """
    try:
        tts_engine.speak(request.text)
        return {"status": "success", "message": "语音播放已加入队列"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")


def _to_wav16k_bytes(contents: bytes) -> bytes:
    """把任意容器(webm/opus/mp3…)转成 **16kHz / 单声道 / 16bit** 的 WAV 字节。

    为什么要有它(2026-09-25 使用者第一场真会话的 0/20 就出在这里):
      浏览器 `MediaRecorder` 录出来的是 **webm/opus**(`useAudioRecorder.ts:19`),
      而 `/interview/answer_audio` 原先**只认 `RIFF`** —— 于是"用语音回答"这条
      **唯一的语音特征来源**,在真实使用里必然被 400 拒掉。
      `/asr` 早就收 webm 并在内部用 ffmpeg 转;这里把同一件事提成共用助手。

    临时文件在 `finally` 里删干净:它是**使用者上传的原始音频**,不该留在 /tmp。
    ffmpeg 失败**抛**(不是安静返回垃圾)—— 否则后面 `wave.open` 会报出一个与真实
    原因无关的 `not a WAVE file`,把排查方向带偏。
    """
    import tempfile
    import subprocess

    in_path = out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as f:
            f.write(contents)
            in_path = f.name
        out_path = in_path + "_16k.wav"
        cmd = [FFMPEG_PATH, "-i", in_path, "-ar", str(SAMPLE_RATE), "-ac", "1",
               "-sample_fmt", "s16", "-y", out_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg转换失败: {result.stderr.decode('utf-8', errors='ignore')}")
        with open(out_path, "rb") as fh:
            return fh.read()
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


@app.post("/asr")
async def speech_to_text(request: Request, audio: UploadFile = File(...),
                         session_id: str = None):
    """
    语音识别（ASR）：接收音频文件，返回识别文本
    支持：WAV、WebM、MP3 等格式（自动转换为 16kHz WAV）

    `session_id` 可省:query 参数(`?session_id=…`)或 multipart 表单字段都能给
    (见 `_resolve_session_id`)。给了就把这次识别累积进**仓库外**的该会话
    transcript.json(缺省落 NONE,报告侧整体排除)。纯 ASR 不写语音特征行 ——
    那是回答的语义。id 必须匹配 `[A-Za-z0-9_-]{1,128}`,否则 400。
    """
    import tempfile
    import subprocess

    sid = await _resolve_session_id(request, session_id)

    try:
        logger.info(f"收到ASR请求: {audio.filename}")

        contents = await audio.read()
        logger.info(f"读取音频数据: {len(contents)} bytes")
        # M2.6:原始上传原样留一份(扩展名由内容嗅探,webm/wav 都认)。
        # 把这一行的 seq 留住 —— 下面转换后的 WAV 要显式用它配对(审查 Important 4)。
        _raw_retained = media_retention.retain_audio(sid, "raw", contents, source="/asr")

        # 检查是否为标准 WAV 格式且符合要求
        if contents.startswith(b'RIFF') and len(contents) > 44:
            logger.info("检测到 WAV 格式")
            audio_stream = io.BytesIO(contents)
            with wave.open(audio_stream, 'rb') as wf:
                logger.info(f"采样率: {wf.getframerate()}Hz, 声道: {wf.getnchannels()}, 位深: {wf.getsampwidth() * 8}bit")

                if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == SAMPLE_RATE:
                    logger.info("格式符合要求，直接识别")
                    audio_data = wf.readframes(wf.getnframes())

                    logger.info("开始 ASR 识别...")
                    utt = await _transcribe_async(audio_data)
                    text = utt.text.strip()
                    if text:                       # 空结果不落盘:不留一场没有段的会话文件
                        transcript_store.append_utterance(sid, utt)

                    log_recognition(utt)
                    return {"text": text, "session_id": sid}
                else:
                    logger.info("格式不匹配，需要转换")
        else:
            logger.info("检测到非WAV格式")

        # 需要转换格式
        logger.info("开始格式转换...")

        input_path = None
        output_path = None
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as input_file:
                input_file.write(contents)
                input_path = input_file.name

            output_path = input_path.replace('.webm', '_converted.wav')

            logger.info(f"使用ffmpeg转换: {input_path} -> {output_path}")

            # 使用 subprocess 调用 ffmpeg
            cmd = [
                FFMPEG_PATH,
                '-i', input_path,
                '-ar', str(SAMPLE_RATE),
                '-ac', '1',
                '-sample_fmt', 's16',
                '-y',  # 覆盖输出文件
                output_path
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )

            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                logger.error(f"ffmpeg转换失败: {error_msg}")
                raise Exception(f"ffmpeg转换失败: {error_msg}")

            logger.info("转换成功")

            # M2.6:留一份转换后的 16k 单声道 WAV(提取器真正读的就是它);
            # 与同一段回答的 raw 共用序号(spec §4「管线所见 + 原始上传都有据」)。
            # ⚠️ 必须在下面那个 finally 删临时文件**之前**读。
            with open(output_path, "rb") as _retained_fh:
                media_retention.retain_audio(
                    sid, "converted", _retained_fh.read(), source="/asr",
                    seq=(_raw_retained or {}).get("seq"))

            # 读取转换后的 WAV 文件
            with wave.open(output_path, 'rb') as wf:
                logger.info(f"最终格式: {wf.getframerate()}Hz, {wf.getnchannels()}声道, {wf.getsampwidth() * 8}bit")
                audio_data = wf.readframes(wf.getnframes())

            # 使用 ASR 识别
            logger.info("开始 ASR 识别...")
            utt = await _transcribe_async(audio_data)
            text = utt.text.strip()
            if text:
                transcript_store.append_utterance(sid, utt)

            log_recognition(utt)
            return {"text": text, "session_id": sid}
        finally:
            if input_path and os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ASR失败")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")


# ========== 面试评估接口 ==========

@app.post("/interview/start")
async def start_interview():
    global voice_logger
    try:
        interview_assessment.reset()
        first_question = interview_assessment.get_next_question()
        if not first_question:
            raise HTTPException(status_code=500, detail="无法获取问题")

        # 会话id在这里诞生,再由前端显式下传给三个模块(spec D7);NONE 之外无来源。
        sid = session_mod.new_session_id()
        transcript_store.ensure_manifest(sid, _asr_meta())
        # 换个带 id 的 logger:文件名是**构造函数**算的(interview_emotion_log_<sid>.csv),
        # 事后改 .session_id 只换列、不换文件名 —— 而 session.json 的 expected_file
        # 记的就是带 id 的名字,不换文件名 refresh_manifest 会把 voice 日志标成 missing。
        voice_logger = VoiceLogger(log_type='interview', session_id=sid)

        tts_engine.speak(first_question)
        return {"status": "started", "session_id": sid, "question": first_question}
    except HTTPException:
        raise          # 别再包一层:否则 500 会变成"启动面试失败: 500: 无法获取问题"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动面试失败: {str(e)}")


@app.get("/interview/question")
async def get_next_question():
    try:
        question = interview_assessment.get_next_question()
        if question:
            tts_engine.speak(question)
            return {"question": question}
        else:
            return {"message": "面试已结束，请获取评估结果"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取问题失败: {str(e)}")


@app.post("/interview/answer")
async def submit_answer(request: AnswerRequest):
    """提交文本回答"""
    try:
        interview_assessment.add_answer(request.answer)
        try:
            interview_assessment.save_log()
        except Exception:
            pass
        return {"status": "success", "message": "回答已记录"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交回答失败: {str(e)}")


@app.post("/interview/answer_audio")
async def submit_answer_audio(request: Request, audio: UploadFile = File(...),
                              session_id: str = None):
    """提交语音回答，自动识别后记录

    `session_id` 可省:query 参数或 multipart 表单字段都能给(见 `_resolve_session_id`);
    非法 id(含路径分隔符/`..`)直接 400,不会拿去当目录名。
    """
    sid = await _resolve_session_id(request, session_id)
    try:
        # 复用 /asr 逻辑
        contents = await audio.read()
        # M2.6:先存原始字节 —— 即使下面判格式不合法,这份上传也留了据。
        _raw = media_retention.retain_audio(sid, "raw", contents,
                                            source="/interview/answer_audio")
        # 非 WAV(浏览器录的是 webm/opus)→ 转,不再硬 400。
        # 这条曾经让"用语音回答"在真实使用里彻底不通(使用者第一场会话 0/20 的根因之一)。
        if not contents.startswith(b'RIFF'):
            wav_bytes = _to_wav16k_bytes(contents)
            media_retention.retain_audio(sid, "converted", wav_bytes,
                                         source="/interview/answer_audio",
                                         seq=(_raw or {}).get("seq"))
        else:
            wav_bytes = contents

        audio_stream = io.BytesIO(wav_bytes)
        with wave.open(audio_stream, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
                raise HTTPException(status_code=400, detail="音频格式要求：16kHz, 16bit, 单声道")
            audio_data = wf.readframes(wf.getnframes())

        utt = await _transcribe_async(audio_data)
        text = utt.text.strip()

        if not text:
            raise HTTPException(status_code=400, detail="未识别到有效语音")

        # 原句只进仓库外那一个 transcript.json;本仓库只留数字(spec D2/D8)
        transcript_store.append_utterance(sid, utt)
        # 连接词密度:纯文本层,分母是字数(刻意不含时长,spec D4/D8);过短/空 → None(不写 0)
        density = connective_density(text)
        # question_index 与 assessment.save_log 的 enumerate 同为 0 基:回答前 qa_pairs 的长度
        # 就是这题的下标。
        question_index = len(interview_assessment.qa_pairs)
        voice_logger.session_id = sid          # 首列随会话(文件名在 /interview/start 里定)
        # 语调特征:拿**这段 PCM 的真值**,不再传空字典(§0.1 第 1 件)。
        # 放在文本闸**之后** —— 识别不出文字的请求上面已经 400 了,不必白算一次 pyin。
        prosody = await _prosody_async(audio_data)
        voice_logger.log_prosody(
            prosody, question_index=question_index, emotion="", feedback="",
            # 整句算一次:一个样本参与,n_rows=1;单个值的标准差按定义为 0.0
            # (与"没算出来"的 connective_density=None 是两回事,别混)
            connective_density=density, connective_density_std=0.0, n_rows=1)

        interview_assessment.add_answer(text)
        try:
            interview_assessment.save_log()
        except Exception:
            pass
        return {"status": "success", "session_id": sid, "recognized_text": text,
                "connective_density": density}
    except HTTPException:
        raise          # 400 要真的回 400,不能被下面这层包成 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音回答处理失败: {str(e)}")


@app.get("/interview/evaluation")
async def get_interview_evaluation():
    """获取面试评估结果并保存日志"""
    try:
        result = interview_assessment.get_comprehensive_evaluation()

        # 保存日志到文件
        log_path = interview_assessment.save_log()

        # 记录结构化日志
        voice_logger.log_assessment(
            total_questions=len(interview_assessment.questions),
            answered_questions=len(interview_assessment.qa_pairs),
            ai_model="qwen-plus",
            max_tokens=300,
            evaluation_result=result
        )

        return {
            "evaluation": result,
            "log_path": log_path,
            "csv_log": voice_logger.get_csv_path(),
            "json_log": voice_logger.get_json_path()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取评估结果失败: {str(e)}")


@app.post("/session/{session_id}/question")
async def submit_session_question(session_id: str, body: QuestionWindow):
    """记一道题的提问窗口。`response_latency` 只此一途(spec §3.9 / §5.5)。

    ⚠️ `ask_end` 必须是**题问完**的时刻,不是"回答提交"的时刻:报告层的
    `response_latency = 首次开口墙钟 − ask_end`。若拿提交时刻当 `ask_end`,
    这个差值会恒等于 0 左右 —— 一个**看着正常、其实什么都没量**的数。

    非法 `session_id` 与非法窗口都是**请求本身**的问题 → 400(不是 500):
    客户端得知道是它自己发错了,而不是服务器坏了。
    """
    try:
        rec = session_meta.upsert_question(
            session_id, qid=body.qid, index=body.index,
            ask_start=body.ask_start, ask_end=body.ask_end,
            source="/session/question")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "success", "session_id": rec["session_id"],
            "qid": rec["qid"], "index": rec["index"]}


@app.post("/session/{session_id}/media")
async def submit_session_media(session_id: str, file: UploadFile = File(...)):
    """收前端 `MediaRecorder` 录的**原生音视频**,原样落 `media/camera.webm`(spec §5.3)。

    为什么挂在语音服务:铸号、`session.json`、`~/shared` 的写入都归它,spec §5.3
    明确"不新起服务"。

    它**不做任何分析** —— 与另外两个 CV 服务收帧的端点不同,这里没有第二步。
    `session_id` 走**路径参数**而不是 query/表单:这一条路由的身份就是那个会话,
    让它在 URL 里可见比藏在表单里好排查(另外两个 CV 服务收 query/form 是历史包袱)。

    这里**刻意没有**外层 `except Exception → 500`(另外两个 `answer_audio` 有):
    `validate_session_id` 的 `ValueError` 已在下面显式转成 400,而
    `retain_uploaded_video` 自己吞掉中途写失败并返回 `None`。加那层包装只会
    把 400 吞成 500 —— 而 400/500 的区别正是"客户端发错了"与"服务器坏了"的区别。
    """
    try:
        sid = media_retention.validate_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"非法 session_id: {exc}")

    # 分块读 + 上限(见 MAX_UPLOAD_BYTES):超限**立刻**拒,不留半个文件。
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"上传超过上限 {MAX_UPLOAD_BYTES} 字节 —— "
                       f"一次请求不许把落盘目录撑满(那会让这一场之后的留存全部降级)")
        chunks.append(chunk)
    data = b"".join(chunks)

    # 空体**拒**,不是"存一个 0 字节的文件然后回 stored:true"(终局复核 I3):
    # 0 字节的 camera.webm 是一份"看着像有、其实没有"的录像,而收尾对账会照着
    # 账本说"原生视频:有"。空体是**请求本身**的毛病 → 400。
    if not data:
        raise HTTPException(status_code=400, detail="上传是空的 —— 没有可留存的录像")

    rec = media_retention.retain_uploaded_video(sid, data, source="/session/media")
    if rec is None:
        # 两种"没存":留存被显式关掉 / 中途写失败。两者都**不许装成功** ——
        # 这个端点唯一的工作就是留存,静默 200 会让人以为素材存下了。
        why = ("留存已关闭(JINGXIN_RETAIN_MEDIA=0)" if not media_retention.enabled()
               else "落盘失败:" + "；".join(media_retention.degraded_reasons(sid)))
        return {"status": "success", "stored": False, "reason": why,
                "session_id": sid, "bytes": len(data)}
    return {"status": "success", "stored": True, "session_id": sid,
            "bytes": rec["bytes"], "sha256": rec["sha256"], "file": rec["file"]}


@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str, type: str = "interview"):
    """获取语音评估会话的实时摘要（当前为全局单例，session_id 预留做向前兼容）"""
    pipeline = interview_assessment if type == "interview" else research_assessment

    try:
        evaluation = pipeline.get_comprehensive_evaluation()
    except Exception:
        evaluation = ""

    return {
        "status": "success",
        "session_id": session_id,
        "data": {
            "type": type,
            "total_questions": len(pipeline.questions),
            "answered": len(pipeline.qa_pairs),
            "qa_pairs": [
                {
                    "question": qa.question,
                    "answer": qa.answer,
                    "has_valid_answer": qa.has_valid_answer,
                }
                for qa in pipeline.qa_pairs
            ],
            "evaluation": evaluation,
            "valid_answers": pipeline.get_valid_answers(),
        },
    }


# ========== 科研评估接口 ==========

@app.post("/research/start")
async def start_research_assessment():
    global research_logger
    try:
        research_assessment.reset()
        first_question = research_assessment.get_next_question()
        if not first_question:
            raise HTTPException(status_code=500, detail="无法获取问题")

        # 科研评估**自成一场**(M2.1 / 第 19 条):与 /interview/start 对称铸号。
        # 不铸的话科研回答只能带客户端自己的 id —— 新页面直接做科研会落 NONE,
        # 而"先面试、再科研"会顺延上一场的号,把科研回答的原句写进**面试会话**的录制目录。
        sid = session_mod.new_session_id()
        transcript_store.ensure_manifest(sid, _asr_meta())
        # 科研那条日志器也**带号建**:不换的话科研会话的特征行恒落
        # `research_emotion_log_NONE_<时间戳>.csv`,读侧于是**永远**把 `语音（科研）`
        # 判成 missing —— 一场科研会话再完整也描述不出来。
        research_logger = VoiceLogger(log_type='research', session_id=sid)

        tts_engine.speak(first_question)
        return {"status": "started", "session_id": sid, "question": first_question}
    except HTTPException:
        raise          # 别再包一层:否则 500 会变成"启动科研评估失败: 500: 无法获取问题"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动科研评估失败: {str(e)}")


@app.get("/research/question")
async def get_research_question():
    try:
        question = research_assessment.get_next_question()
        if question:
            tts_engine.speak(question)
            return {"question": question}
        else:
            return {"message": "评估已结束，请获取评估结果"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取问题失败: {str(e)}")


@app.post("/research/answer")
async def submit_research_answer(request: AnswerRequest):
    try:
        research_assessment.add_answer(request.answer)
        try:
            research_assessment.save_log()
        except Exception:
            pass
        return {"status": "success", "message": "回答已记录"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交回答失败: {str(e)}")


@app.post("/research/answer_audio")
async def submit_research_answer_audio(request: Request, audio: UploadFile = File(...),
                                       session_id: str = None):
    """提交语音回答，自动识别后记录

    `session_id` 可省:query 参数或 multipart 表单字段都能给(见 `_resolve_session_id`);非法 id 400。
    """
    sid = await _resolve_session_id(request, session_id)
    try:
        contents = await audio.read()
        # M2.6:先存原始字节 —— 即使下面判格式不合法,这份上传也留了据。
        _raw = media_retention.retain_audio(sid, "raw", contents,
                                            source="/research/answer_audio")
        # 与面试侧同一条:非 WAV(浏览器录的是 webm/opus)→ 转,不再硬 400。
        if not contents.startswith(b'RIFF'):
            wav_bytes = _to_wav16k_bytes(contents)
            media_retention.retain_audio(sid, "converted", wav_bytes,
                                         source="/research/answer_audio",
                                         seq=(_raw or {}).get("seq"))
        else:
            wav_bytes = contents

        audio_stream = io.BytesIO(wav_bytes)
        with wave.open(audio_stream, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
                raise HTTPException(status_code=400, detail="音频格式要求：16kHz, 16bit, 单声道")
            audio_data = wf.readframes(wf.getnframes())

        utt = await _transcribe_async(audio_data)
        text = utt.text.strip()

        if not text:
            raise HTTPException(status_code=400, detail="未识别到有效语音")

        # 与面试侧同一条不变量:原句只进仓库外的 transcript,本仓库只留数字。
        # (连接词密度的特征行只写语音侧那一条日志 —— LOG_PREFIXES 里没有科研侧)
        transcript_store.append_utterance(sid, utt)

        # 与面试侧**同口径**地写下这条 L0 特征行(此前科研侧完全没有产出方):
        # 连接词密度是纯文本层,分母是字数(刻意不含时长,spec D4/D8);过短/空 → None(不写 0)。
        density = connective_density(text)
        question_index = len(research_assessment.qa_pairs)
        research_logger.session_id = sid     # 首列随会话(文件名在 /research/start 里定)
        # 与面试侧同一条:真值替掉空字典(§0.1 第 1 件,两处都接)。
        prosody = await _prosody_async(audio_data)
        research_logger.log_prosody(
            prosody, question_index=question_index, emotion="", feedback="",
            connective_density=density, connective_density_std=0.0, n_rows=1)

        research_assessment.add_answer(text)
        try:
            research_assessment.save_log()
        except Exception:
            pass
        return {"status": "success", "session_id": sid, "recognized_text": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音回答处理失败: {str(e)}")


@app.get("/research/evaluation")
async def get_research_evaluation():
    """获取科研评估结果并保存日志"""
    try:
        result = research_assessment.get_comprehensive_evaluation()

        # 保存日志到文件
        log_path = research_assessment.save_log()

        # 用**带会话号的那一个**(模块级,`/research/start` 里建的)—— 不再现造一个不带号的:
        # 现造会把评估行写进 `research_emotion_log_NONE_<时间戳>.csv`,与特征行分了家。
        research_logger.log_assessment(
            total_questions=len(research_assessment.questions),
            answered_questions=len(research_assessment.qa_pairs),
            ai_model="qwen-plus",
            max_tokens=300,
            evaluation_result=result
        )

        return {
            "evaluation": result,
            "log_path": log_path,
            "csv_log": research_logger.get_csv_path(),
            "json_log": research_logger.get_json_path()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取评估结果失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_CONFIG['host'], port=API_CONFIG['port'])
