from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import uuid
import time
import os
import logging
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer
from gesture_analysis.core.analysis.shoulder_analyzer import ShoulderAnalyzer
from gesture_analysis.core.analysis.arm_analyzer import ArmAnalyzer
from gesture_analysis.core.analysis.emotion_inferencer import EmotionInferencer
from gesture_analysis.utils.logger import GestureLogger
from gesture_analysis.config import API_CONFIG, MEDIAPIPE_CONFIG, LOGS_DIR
import mediapipe as mp

app = FastAPI(
    title="Gesture Analysis API",
    description="手势、肩部和手臂情绪分析API"
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

# 初始化MediaPipe模型
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(**MEDIAPIPE_CONFIG['hands'])
pose = mp_pose.Pose(**MEDIAPIPE_CONFIG['pose'])

# 会话管理
session_analyzers = {}
session_loggers = {}  # 每个会话的 GestureLogger 实例
SESSION_TIMEOUT = 300


def get_or_create_analyzers(session_id: str):
    """获取或创建会话的分析器实例"""
    current_time = time.time()

    # 清理过期会话
    expired_sessions = [
        sid for sid, (analyzers, last_used) in session_analyzers.items()
        if current_time - last_used > SESSION_TIMEOUT
    ]
    for sid in expired_sessions:
        del session_analyzers[sid]
        session_loggers.pop(sid, None)

    # 获取或创建新会话
    if session_id not in session_analyzers:
        analyzers = {
            'left_hand': HandAnalyzer(hand_id=0),
            'right_hand': HandAnalyzer(hand_id=1),
            'shoulder': ShoulderAnalyzer(),
            'left_arm': ArmAnalyzer(arm_id='left'),
            'right_arm': ArmAnalyzer(arm_id='right'),
            'emotion': EmotionInferencer()
        }
        session_analyzers[session_id] = (analyzers, current_time)
        # 同一会话的所有帧写入同一个文件（用会话创建时间戳作为文件名）
        from datetime import datetime
        session_ts = datetime.fromtimestamp(current_time).strftime('%Y%m%d_%H%M%S')
        log_path = str(LOGS_DIR / f'gesture_emotion_log_{session_ts}.csv')
        session_loggers[session_id] = (GestureLogger(log_file_path=log_path), log_path, current_time)
        logger.info(f"创建手势分析会话: {session_id}, 日志: {log_path}")
    else:
        analyzers, _ = session_analyzers[session_id]
        session_analyzers[session_id] = (analyzers, current_time)
        if session_id in session_loggers:
            entry = session_loggers[session_id]
            session_loggers[session_id] = (entry[0], entry[1], current_time)

    return session_analyzers[session_id][0]


class ImageRequest(BaseModel):
    """图片请求模型（保留以兼容旧代码）"""
    image: str  # Base64编码的图片


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "Gesture Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "/health": "GET - 健康检查",
            "/analyze": "POST - 分析图片中的手势和肩部（支持FormData）",
            "/reset": "POST - 重置分析器状态"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "active_sessions": len(session_analyzers)
    }


@app.post("/analyze")
async def analyze_image(
        file: UploadFile = File(...),
        session_id: str = None
):
    """
    分析图片中的手势和肩部

    参数:
        file: 上传的图片文件（FormData格式）
        session_id: 会话ID（可选，不提供则自动生成）

    返回:
        分析结果
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        logger.info(f"收到手势分析请求: session={session_id}, file={file.filename}")

        # 读取图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="无法解码图片")

        logger.info(f"图片加载成功: {image.shape}")

        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取或创建分析器
        analyzers = get_or_create_analyzers(session_id)

        # 处理手部
        hand_results_raw = hands.process(image_rgb)
        detected_hands = 0
        hand_scores = []

        if hand_results_raw.multi_hand_landmarks:
            for hand_id, lm_obj in enumerate(hand_results_raw.multi_hand_landmarks):
                if hand_id >= 2: break
                analyzer_key = 'left_hand' if hand_id == 0 else 'right_hand'
                analyzers[analyzer_key].update(lm_obj.landmark)
                hand_scores.append(analyzers[analyzer_key].get_results()['resilience_score'])
                detected_hands += 1

        # 处理肩部
        shoulder_results_raw = pose.process(image_rgb)
        shoulder_score = 50.0

        if shoulder_results_raw.pose_landmarks:
            analyzers['shoulder'].update(shoulder_results_raw.pose_landmarks.landmark)
            shoulder_score = analyzers['shoulder'].get_results()['shoulder_score']

        # 处理手臂
        left_arm_score = 50.0
        right_arm_score = 50.0
        if shoulder_results_raw.pose_landmarks:
            analyzers['left_arm'].update(shoulder_results_raw.pose_landmarks.landmark)
            analyzers['right_arm'].update(shoulder_results_raw.pose_landmarks.landmark)
            left_arm_result = analyzers['left_arm'].get_results()
            right_arm_result = analyzers['right_arm'].get_results()
            left_arm_score = left_arm_result.get('arm_score', 50.0) if left_arm_result.get('is_valid') else 50.0
            right_arm_score = right_arm_result.get('arm_score', 50.0) if right_arm_result.get('is_valid') else 50.0

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
        left_arm_results = {"arm_score": left_arm_score}
        right_arm_results = {"arm_score": right_arm_score}
        emotion_result = analyzers['emotion'].infer_emotion(
            hand_results,
            shoulder_results,
            left_arm_results,
            right_arm_results
        )

        logger.info(f"分析完成: {emotion_result['emotion_state']} (评分: {emotion_result['overall_score']:.1f})")

        # 获取详细的分析器结果
        left_hand_results = analyzers['left_hand'].get_results()
        right_hand_results = analyzers['right_hand'].get_results()
        shoulder_results = analyzers['shoulder'].get_results()
        left_arm_results = analyzers['left_arm'].get_results()
        right_arm_results = analyzers['right_arm'].get_results()

        # 将帧数据写入 CSV 日志（供 report_frontend 批量读取）
        if session_id in session_loggers:
            try:
                gesture_logger = session_loggers[session_id][0]
                gesture_logger.log(
                    left_hand_result=left_hand_results,
                    right_hand_result=right_hand_results,
                    shoulder_result=shoulder_results,
                    left_arm_result=left_arm_results,
                    right_arm_result=right_arm_results,
                    emotion_result=emotion_result
                )
            except Exception as log_err:
                logger.warning("CSV日志写入失败: %s", log_err)

        # 返回完整结果
        return {
            "status": "success",
            "session_id": session_id,
            "result": {
                # 检测状态
                "detected_hands": detected_hands,

                # 手部详细分析
                "hand": {
                    "left": {
                        "resilience_score": left_hand_results.get('resilience_score', 50.0),
                        "jitter": left_hand_results.get('jitter', 0.0),
                        "fist_status": left_hand_results.get('fist_status', False),
                        "spread": left_hand_results.get('spread', 0.0),
                        "is_valid": left_hand_results.get('is_valid', False)
                    },
                    "right": {
                        "resilience_score": right_hand_results.get('resilience_score', 50.0),
                        "jitter": right_hand_results.get('jitter', 0.0),
                        "fist_status": right_hand_results.get('fist_status', False),
                        "spread": right_hand_results.get('spread', 0.0),
                        "is_valid": right_hand_results.get('is_valid', False)
                    },
                    "average_score": hand_score
                },

                # 肩部详细分析
                "shoulder": {
                    "shoulder_score": shoulder_results.get('shoulder_score', 50.0),
                    "left_jitter": shoulder_results.get('left_jitter', 0.0),
                    "right_jitter": shoulder_results.get('right_jitter', 0.0),
                    "shrug_level": shoulder_results.get('shrug_level', 0.0),
                    "is_calibrated": shoulder_results.get('is_calibrated', False),
                    "is_valid": shoulder_results.get('is_valid', False)
                },

                # 手臂详细分析
                "arm": {
                    "left": {
                        "arm_score": left_arm_results.get('arm_score', 50.0),
                        "wrist_jitter": left_arm_results.get('wrist_jitter', 0.0),
                        "elbow_jitter": left_arm_results.get('elbow_jitter', 0.0),
                        "arm_angle": left_arm_results.get('arm_angle', 0.0),
                        "is_valid": left_arm_results.get('is_valid', False)
                    },
                    "right": {
                        "arm_score": right_arm_results.get('arm_score', 50.0),
                        "wrist_jitter": right_arm_results.get('wrist_jitter', 0.0),
                        "elbow_jitter": right_arm_results.get('elbow_jitter', 0.0),
                        "arm_angle": right_arm_results.get('arm_angle', 0.0),
                        "is_valid": right_arm_results.get('is_valid', False)
                    }
                },

                # 情绪推断结果
                "emotion": {
                    "overall_score": emotion_result["overall_score"],
                    "emotion_state": emotion_result["emotion_state"],
                    "emoji": emotion_result["emoji"],
                    "feedback": emotion_result["feedback"],
                    "used_features": emotion_result["used_features"]
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("手势分析失败")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """获取手势分析会话的实时摘要"""
    if session_id not in session_analyzers:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

    analyzers, _ = session_analyzers[session_id]

    left_hand = analyzers['left_hand'].get_results()
    right_hand = analyzers['right_hand'].get_results()
    shoulder = analyzers['shoulder'].get_results()
    left_arm = analyzers['left_arm'].get_results()
    right_arm = analyzers['right_arm'].get_results()

    # 计算手部平均分
    hand_scores = []
    if left_hand.get('is_valid'):
        hand_scores.append(left_hand.get('resilience_score', 50))
    if right_hand.get('is_valid'):
        hand_scores.append(right_hand.get('resilience_score', 50))
    avg_hand = sum(hand_scores) / len(hand_scores) if hand_scores else 50.0

    # 情绪推断
    left_arm_score = left_arm.get('arm_score', 50) if left_arm.get('is_valid') else 50.0
    right_arm_score = right_arm.get('arm_score', 50) if right_arm.get('is_valid') else 50.0
    emotion = analyzers['emotion'].infer_emotion(
        {"resilience_score": avg_hand},
        {"shoulder_score": shoulder.get('shoulder_score', 50)},
        {"arm_score": left_arm_score},
        {"arm_score": right_arm_score},
    )

    return {
        "status": "success",
        "session_id": session_id,
        "data": {
            "hand": {
                "left": left_hand,
                "right": right_hand,
                "average_score": round(avg_hand, 1),
            },
            "shoulder": shoulder,
            "arm": {
                "left": left_arm,
                "right": right_arm,
            },
            "emotion": {
                "overall_score": emotion["overall_score"],
                "emotion_state": emotion["emotion_state"],
                "feedback": emotion["feedback"],
            },
        },
    }


@app.post("/reset")
async def reset_analyzers(session_id: str = None):
    """
    重置分析器状态

    参数:
        session_id: 会话ID（可选，不提供则重置所有）

    返回:
        操作结果
    """
    try:
        if session_id:
            if session_id in session_analyzers:
                del session_analyzers[session_id]
                session_loggers.pop(session_id, None)
                return {"status": "success", "message": f"会话 {session_id} 已重置"}
            else:
                return {"status": "not_found", "message": f"会话 {session_id} 不存在"}
        else:
            session_analyzers.clear()
            session_loggers.clear()
            return {"status": "success", "message": "所有会话已重置"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_CONFIG['host'], port=API_CONFIG['port'])

