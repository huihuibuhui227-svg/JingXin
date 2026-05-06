# api/app.py
"""
FastAPI应用

提供语音交互的Web API接口
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import threading
import os
import json
from vosk import Model, KaldiRecognizer
import wave
import io

# 导入项目模块（使用绝对导入）
from voice_interaction.pipeline.tts_pipeline import TTSPipeline as TTSEngine
from voice_interaction.pipeline.assessment_pipeline import InterviewAssessmentPipeline, ResearchAssessmentPipeline
from voice_interaction.config import API_CONFIG
from voice_interaction.utils.logger import VoiceLogger
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import threading
import os
import json
from vosk import Model, KaldiRecognizer
import wave
import io

# 导入项目模块（使用绝对导入）
from voice_interaction.pipeline.tts_pipeline import TTSPipeline as TTSEngine
from voice_interaction.pipeline.assessment_pipeline import InterviewAssessmentPipeline, ResearchAssessmentPipeline
from voice_interaction.config import API_CONFIG
from voice_interaction.utils.logger import VoiceLogger

app = FastAPI(
    title="Voice Interaction API",
    description="语音交互API，提供语音识别、语音合成和面试评估功能"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# --- Vosk 语音识别模型 ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'vosk-model-cn-0.22')
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ Vosk 模型未找到: {os.path.abspath(MODEL_PATH)}")

vosk_model = Model(MODEL_PATH)
SAMPLE_RATE = 16000


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
async def speech_to_text(audio: UploadFile = File(...)):
    """
    语音识别（ASR）：接收音频文件，返回识别文本
    支持：WAV、WebM、MP3 等格式（自动转换为 16kHz WAV）
    """
    import traceback
    import tempfile
    import subprocess
    import os

    try:
        print(f"📥 收到ASR请求: {audio.filename}")

        contents = await audio.read()
        print(f"✅ 读取音频数据: {len(contents)} bytes")

        # 检查是否为标准 WAV 格式且符合要求
        if contents.startswith(b'RIFF') and len(contents) > 44:
            print("📄 检测到 WAV 格式")
            audio_stream = io.BytesIO(contents)
            with wave.open(audio_stream, 'rb') as wf:
                print(f"   采样率: {wf.getframerate()}Hz, 声道: {wf.getnchannels()}, 位深: {wf.getsampwidth() * 8}bit")

                if wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == SAMPLE_RATE:
                    print("✅ 格式符合要求，直接识别")
                    audio_data = wf.readframes(wf.getnframes())

                    # 使用 Vosk 识别
                    print("🎤 开始Vosk识别...")
                    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
                    rec.AcceptWaveform(audio_data)
                    result = json.loads(rec.FinalResult())
                    text = result.get("text", "").strip()

                    print(f"✅ 识别结果: '{text}'")
                    return {"text": text}
                else:
                    print(f"⚠️ 格式不匹配，需要转换")
        else:
            print(f"📄 检测到非WAV格式")

        # 需要转换格式
        print("🔄 开始格式转换...")

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as input_file:
            input_file.write(contents)
            input_path = input_file.name

        output_path = input_path.replace('.webm', '_converted.wav')

        # ffmpeg 路径
        ffmpeg_path = r"C:\fffPMPG\bin\ffmpeg.exe"

        if not os.path.exists(ffmpeg_path):
            raise FileNotFoundError(f"ffmpeg.exe 未找到: {ffmpeg_path}")

        print(f"🔧 使用ffmpeg转换: {input_path} -> {output_path}")

        # 使用 subprocess 调用 ffmpeg
        cmd = [
            ffmpeg_path,
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
            print(f"❌ ffmpeg转换失败: {error_msg}")
            raise Exception(f"ffmpeg转换失败: {error_msg}")

        print("✅ 转换成功")

        # 读取转换后的 WAV 文件
        with wave.open(output_path, 'rb') as wf:
            print(f"   最终格式: {wf.getframerate()}Hz, {wf.getnchannels()}声道, {wf.getsampwidth() * 8}bit")
            audio_data = wf.readframes(wf.getnframes())

        # 清理临时文件
        os.unlink(input_path)
        os.unlink(output_path)

        # 使用 Vosk 识别
        print("🎤 开始Vosk识别...")
        rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        rec.AcceptWaveform(audio_data)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()

        print(f"✅ 识别结果: '{text}'")
        return {"text": text}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ASR失败: {str(e)}")
        print(f"📋 错误堆栈:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")


# ========== 面试评估接口 ==========

@app.post("/interview/start")
async def start_interview():
    try:
        interview_assessment.reset()
        first_question = interview_assessment.get_next_question()
        if first_question:
            tts_engine.speak(first_question)
            return {"status": "started", "question": first_question}
        else:
            raise HTTPException(status_code=500, detail="无法获取问题")
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
        return {"status": "success", "message": "回答已记录"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交回答失败: {str(e)}")


@app.post("/interview/answer_audio")
async def submit_answer_audio(audio: UploadFile = File(...)):
    """提交语音回答，自动识别后记录"""
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

        rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        rec.AcceptWaveform(audio_data)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()

        if not text:
            raise HTTPException(status_code=400, detail="未识别到有效语音")

        interview_assessment.add_answer(text)
        return {"status": "success", "recognized_text": text}
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
        return {"status": "success", "message": "回答已记录"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交回答失败: {str(e)}")


@app.post("/research/answer_audio")
async def submit_research_answer_audio(audio: UploadFile = File(...)):
    """提交语音回答，自动识别后记录"""
    try:
        contents = await audio.read()
        if not contents.startswith(b'RIFF'):
            raise HTTPException(status_code=400, detail="仅支持 WAV 格式音频")

        audio_stream = io.BytesIO(contents)
        with wave.open(audio_stream, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
                raise HTTPException(status_code=400, detail="音频格式要求：16kHz, 16bit, 单声道")
            audio_data = wf.readframes(wf.getnframes())

        rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        rec.AcceptWaveform(audio_data)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()

        if not text:
            raise HTTPException(status_code=400, detail="未识别到有效语音")

        research_assessment.add_answer(text)
        return {"status": "success", "recognized_text": text}
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
