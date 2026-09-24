# api/app.py
"""
FastAPI应用

提供语音交互的Web API接口
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import os
import logging
import wave
import io

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# 导入项目模块（使用绝对导入）
from voice_interaction.pipeline.tts_pipeline import TTSPipeline as TTSEngine
from voice_interaction.pipeline.assessment_pipeline import InterviewAssessmentPipeline, ResearchAssessmentPipeline
from voice_interaction.config import API_CONFIG, FFMPEG_PATH
from voice_interaction.utils.logger import VoiceLogger
from voice_interaction.asr import session as session_mod, transcript_store
from voice_interaction.asr.connective_density import connective_density
from voice_interaction.asr.funasr_engine import load_config
# 转写缝:引擎只由 asr/transcribe.py 持有,这里按名字取那一层转发(不直接摸引擎)。
# 换引擎(测试/验收)只需动那一个模块,不必碰本文件。
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

# 音频契约:16 kHz / 16 bit / 单声道(四处 wave 校验与 ffmpeg 转换都用它)
SAMPLE_RATE = 16000


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


class TextRequest(BaseModel):
    """文本请求模型"""
    text: str


class AnswerRequest(BaseModel):
    """回答请求模型"""
    answer: str


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


@app.post("/asr")
async def speech_to_text(audio: UploadFile = File(...), session_id: str = None):
    """
    语音识别（ASR）：接收音频文件，返回识别文本
    支持：WAV、WebM、MP3 等格式（自动转换为 16kHz WAV）

    `session_id` 可省:给了就把这次识别累积进**仓库外**的该会话 transcript.json
    (缺省落 NONE,报告侧整体排除)。纯 ASR 不写语音特征行 —— 那是回答的语义。
    """
    import tempfile
    import subprocess

    sid = session_id or session_mod.NONE_SESSION

    try:
        logger.info(f"收到ASR请求: {audio.filename}")

        contents = await audio.read()
        logger.info(f"读取音频数据: {len(contents)} bytes")

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

                    logger.info(f"识别结果: '{text}'")
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

            logger.info(f"识别结果: '{text}'")
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
async def submit_answer_audio(audio: UploadFile = File(...), session_id: str = None):
    """提交语音回答，自动识别后记录"""
    sid = session_id or session_mod.NONE_SESSION
    try:
        # 复用 /asr 逻辑
        contents = await audio.read()
        if not contents.startswith(b'RIFF'):
            raise HTTPException(status_code=400, detail="仅支持 WAV 格式音频")

        audio_stream = io.BytesIO(contents)
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
        voice_logger.log_prosody(
            {}, question_index=question_index, emotion="", feedback="",
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
    try:
        research_assessment.reset()
        first_question = research_assessment.get_next_question()
        if first_question:
            tts_engine.speak(first_question)
            return {"status": "started", "question": first_question}
        else:
            raise HTTPException(status_code=500, detail="无法获取问题")
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
async def submit_research_answer_audio(audio: UploadFile = File(...), session_id: str = None):
    """提交语音回答，自动识别后记录"""
    sid = session_id or session_mod.NONE_SESSION
    try:
        contents = await audio.read()
        if not contents.startswith(b'RIFF'):
            raise HTTPException(status_code=400, detail="仅支持 WAV 格式音频")

        audio_stream = io.BytesIO(contents)
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

        # 创建科研评估的日志记录器
        research_logger = VoiceLogger(log_type='research')
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
