# app.py
"""
JingXin集成API应用

提供统一的RESTful API接口，协调面部表情、语音交互和手势分析三个子模块
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import threading
from typing import Optional

from ..integrator import JingXinIntegrator

app = FastAPI(
    title="JingXin API",
    description="多模态面试评估系统集成API"
)

# 初始化集成器
integrator = JingXinIntegrator()


class TextRequest(BaseModel):
    """文本请求模型"""
    text: str


class AnswerRequest(BaseModel):
    """回答请求模型"""
    answer: str


class ImageRequest(BaseModel):
    """图片请求模型"""
    image: str  # Base64编码的图片


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "JingXin API - 多模态面试评估系统",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - 健康检查",
            "/interview/start": "POST - 开始面试",
            "/interview/question": "GET - 获取下一个问题",
            "/interview/answer": "POST - 提交回答",
            "/interview/evaluation": "GET - 获取综合评估",
            "/analyze/frame": "POST - 分析视频帧",
            "/tts": "POST - 文本转语音"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    status = {
        "status": "healthy",
        "modules": {
            "face_expression": integrator.face_enabled,
            "voice_interaction": integrator.voice_enabled,
            "gesture_analysis": integrator.gesture_enabled
        }
    }
    return status


# ========== 面试接口 ==========

@app.post("/interview/start")
async def start_interview():
    """
    开始面试

    返回:
        面试开始状态
    """
    try:
        integrator.start_interview_session()
        return {"status": "started", "message": "面试会话已开始"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动面试失败: {str(e)}")


@app.get("/interview/question")
async def get_question():
    """
    获取下一个问题

    返回:
        问题文本
    """
    try:
        question = integrator.ask_question()
        if question:
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
        answer = integrator.get_answer()
        return {"status": "success", "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交回答失败: {str(e)}")


@app.get("/interview/evaluation")
async def get_evaluation():
    """
    获取综合评估

    返回:
        综合评估结果
    """
    try:
        evaluation = integrator.get_comprehensive_evaluation()
        return evaluation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取评估失败: {str(e)}")


# ========== 分析接口 ==========

@app.post("/analyze/frame")
async def analyze_frame(request: ImageRequest):
    """
    分析视频帧

    参数:
        request: 包含Base64编码图片的请求

    返回:
        分析结果
    """
    try:
        # 解码图片
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="无法解码图片")

        # 分析帧
        results = integrator.analyze_frame(frame)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


# ========== 语音接口 ==========

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
        threading.Thread(
            target=integrator.tts_engine.speak,
            args=(request.text,),
            daemon=True
        ).start()
        return {"status": "success", "message": "语音播放已开始"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
