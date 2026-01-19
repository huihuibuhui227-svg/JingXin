
# app.py
"""
FastAPI应用

提供手势和肩部情绪分析的Web API接口
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from typing import Optional

from ..analyzers import HandAnalyzer, ShoulderAnalyzer
from ..inference import EmotionInferencer
from ..config import API_CONFIG, MEDIAPIPE_CONFIG
import mediapipe as mp

app = FastAPI(
    title="Gesture Analysis API",
    description="手势和肩部情绪分析API"
)

# 初始化MediaPipe模型
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(**MEDIAPIPE_CONFIG['hands'])
pose = mp_pose.Pose(**MEDIAPIPE_CONFIG['pose'])

# 初始化分析器
left_hand_analyzer = HandAnalyzer(hand_id=0)
right_hand_analyzer = HandAnalyzer(hand_id=1)
shoulder_analyzer = ShoulderAnalyzer()
emotion_inferencer = EmotionInferencer()


class ImageRequest(BaseModel):
    """图片请求模型"""
    image: str  # Base64编码的图片


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "Gesture Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - 健康检查",
            "/analyze": "POST - 分析图片中的手势和肩部",
            "/reset": "POST - 重置分析器状态"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_image(request: ImageRequest):
    """
    分析图片中的手势和肩部

    参数:
        request: 包含Base64编码图片的请求

    返回:
        分析结果
    """
    try:
        # 解码图片
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解码图片")

        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理手部
        hand_results_raw = hands.process(image_rgb)
        detected_hands = 0
        hand_scores = []

        if hand_results_raw.multi_hand_landmarks:
            for hand_id, lm_obj in enumerate(hand_results_raw.multi_hand_landmarks):
                if hand_id >= 2: break
                analyzer = left_hand_analyzer if hand_id == 0 else right_hand_analyzer
                analyzer.update(lm_obj.landmark)
                hand_scores.append(analyzer.get_results()['resilience_score'])
                detected_hands += 1

        # 处理肩部
        shoulder_results_raw = pose.process(image_rgb)
        shoulder_score = 50.0

        if shoulder_results_raw.pose_landmarks:
            shoulder_analyzer.update(shoulder_results_raw.pose_landmarks.landmark)
            shoulder_score = shoulder_analyzer.get_results()['shoulder_score']

        # 计算手部平均分
        if detected_hands == 1:
            hand_score = hand_scores[0]
        elif detected_hands == 2:
            hand_score = sum(hand_scores) / len(hand_scores)
        else:
            hand_score = 50.0

        # 推断情绪
        hand_results = {"resilience_score": hand_score}
        shoulder_results = {"shoulder_score": shoulder_score}
        emotion_result = emotion_inferencer.infer_emotion(hand_results, shoulder_results)

        # 返回结果
        return {
            "status": "success",
            "detected_hands": detected_hands,
            "hand_score": hand_score,
            "shoulder_score": shoulder_score,
            "overall_score": emotion_result["overall_score"],
            "emotion_state": emotion_result["emotion_state"],
            "emoji": emotion_result["emoji"],
            "feedback": emotion_result["feedback"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.post("/reset")
async def reset_analyzers():
    """
    重置分析器状态

    返回:
        操作结果
    """
    try:
        left_hand_analyzer.reset()
        right_hand_analyzer.reset()
        shoulder_analyzer.reset()
        return {"status": "success", "message": "分析器已重置"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_CONFIG['host'], port=API_CONFIG['port'])
