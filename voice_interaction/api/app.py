
# app.py
"""
FastAPI应用

提供语音交互的Web API接口
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import threading

from ..analyzers import SpeechRecognizer, TTSEngine
from ..assessment import InterviewAssessment, ResearchAssessment
from ..config import API_CONFIG

app = FastAPI(
    title="Voice Interaction API",
    description="语音交互API，提供语音识别、语音合成和面试评估功能"
)

# 初始化组件
speech_recognizer = SpeechRecognizer()
tts_engine = TTSEngine()
interview_assessment = InterviewAssessment()
research_assessment = ResearchAssessment()


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
            "/interview/start": "POST - 开始面试",
            "/interview/question": "GET - 获取下一个问题",
            "/interview/answer": "POST - 提交回答",
            "/interview/evaluation": "GET - 获取综合评估",
            "/research/start": "POST - 开始科研评估",
            "/research/question": "GET - 获取下一个问题",
            "/research/answer": "POST - 提交回答",
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

    参数:
        request: 包含要朗读文本的请求

    返回:
        操作结果
    """
    try:
        # 异步播放语音
        threading.Thread(target=tts_engine.speak, args=(request.text,), daemon=True).start()
        return {"status": "success", "message": "语音播放已开始"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")


# ========== 面试评估接口 ==========

@app.post("/interview/start")
async def start_interview():
    """
    开始面试

    返回:
        第一个问题
    """
    try:
        interview_assessment.reset()
        first_question = interview_assessment.get_next_question()
        if first_question:
            # 异步播放语音
            threading.Thread(target=tts_engine.speak, args=(first_question,), daemon=True).start()
            return {"status": "started", "question": first_question}
        else:
            raise HTTPException(status_code=500, detail="无法获取问题")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动面试失败: {str(e)}")


@app.get("/interview/question")
async def get_next_question():
    """
    获取下一个问题

    返回:
        问题字符串
    """
    try:
        question = interview_assessment.get_next_question()
        if question:
            # 异步播放语音
            threading.Thread(target=tts_engine.speak, args=(question,), daemon=True).start()
            return {"question": question}
        else:
            return {"message": "面试已结束，请获取评估结果"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取问题失败: {str(e)}")


@app.post("/interview/answer")
async def submit_answer(request: AnswerRequest):
    """
    提交回答

    参数:
        request: 包含回答的请求

    返回:
        操作结果
    """
    try:
        interview_assessment.add_answer(request.answer)
        return {"status": "success", "message": "回答已记录"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交回答失败: {str(e)}")


@app.get("/interview/evaluation")
async def get_evaluation():
    """
    获取综合评估

    返回:
        评估结果
    """
    try:
        evaluation = interview_assessment.get_comprehensive_evaluation()
        return {"evaluation": evaluation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取评估失败: {str(e)}")


# ========== 科研评估接口 ==========

@app.post("/research/start")
async def start_research_assessment():
    """
    开始科研评估

    返回:
        第一个问题
    """
    try:
        research_assessment.reset()
        first_question = research_assessment.get_next_question()
        if first_question:
            # 异步播放语音
            threading.Thread(target=tts_engine.speak, args=(first_question,), daemon=True).start()
            return {"status": "started", "question": first_question}
        else:
            raise HTTPException(status_code=500, detail="无法获取问题")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动科研评估失败: {str(e)}")


@app.get("/research/question")
async def get_research_question():
    """
    获取下一个科研评估问题

    返回:
        问题字符串
    """
    try:
        question = research_assessment.get_next_question()
        if question:
            # 异步播放语音
            threading.Thread(target=tts_engine.speak, args=(question,), daemon=True).start()
            return {"question": question}
        else:
            return {"message": "评估已结束，请获取评估结果"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取问题失败: {str(e)}")


@app.post("/research/answer")
async def submit_research_answer(request: AnswerRequest):
    """
    提交科研评估回答

    参数:
        request: 包含回答的请求

    返回:
        操作结果
    """
    try:
        research_assessment.add_answer(request.answer)
        return {"status": "success", "message": "回答已记录"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交回答失败: {str(e)}")


@app.get("/research/evaluation")
async def get_research_evaluation():
    """
    获取科研潜质评估

    返回:
        评估结果
    """
    try:
        evaluation = research_assessment.evaluate_research_potential()
        return {"evaluation": evaluation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取评估失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_CONFIG['host'], port=API_CONFIG['port'])
