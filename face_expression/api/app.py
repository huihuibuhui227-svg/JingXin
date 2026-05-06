from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import cv2
import numpy as np
import uuid
import time

try:
    from face_expression.pipeline.video_pipeline import VideoPipeline
    import mediapipe as mp
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已正确安装 face_expression 模块")
    raise

app = FastAPI(
    title="Face Expression Analysis API",
    description="面部表情分析API，支持视频流实时分析"
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

# 会话管理：存储每个用户的 VideoPipeline 实例
session_pipelines = {}
SESSION_TIMEOUT = 300  # 会话超时时间（秒）


def get_or_create_pipeline(session_id: str, fps: int = 30) -> VideoPipeline:
    """获取或创建 VideoPipeline 实例"""
    current_time = time.time()

    # 清理过期会话
    expired_sessions = [
        sid for sid, (pipeline, last_used) in session_pipelines.items()
        if current_time - last_used > SESSION_TIMEOUT
    ]
    for sid in expired_sessions:
        del session_pipelines[sid]
        print(f"🗑️ 清理过期会话: {sid}")

    # 获取或创建新会话
    if session_id not in session_pipelines:
        try:
            pipeline = VideoPipeline(fps=fps, session_id=session_id)
            session_pipelines[session_id] = (pipeline, current_time)
            print(f"✅ 创建新会话: {session_id}")
        except Exception as e:
            print(f"❌ 创建会话失败: {e}")
            raise HTTPException(status_code=500, detail=f"会话初始化失败: {str(e)}")
    else:
        # 更新最后使用时间
        pipeline, _ = session_pipelines[session_id]
        session_pipelines[session_id] = (pipeline, current_time)

    return session_pipelines[session_id][0]


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "Face Expression Analysis API (Video Stream)",
        "version": "2.0.0",
        "endpoints": {
            "/analyze": "POST - 上传视频帧进行实时分析（支持会话）",
            "/health": "GET - 健康检查",
            "/session/{session_id}/reset": "POST - 重置会话"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "active_sessions": len(session_pipelines)
    }


@app.post("/analyze")
async def analyze_frame(
        file: UploadFile = File(...),
        session_id: str = None,
        fps: int = 30
):
    """
    上传视频帧进行实时分析

    参数:
        file: 视频帧图片
        session_id: 会话ID（可选，不提供则自动生成）
        fps: 帧率（默认30）

    返回:
        完整的分析结果，包含AU特征、情绪、紧张度、时间序列统计等
    """
    import traceback

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上传的文件必须是图片格式")

    # 生成或使用提供的 session_id
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        print(f"📥 收到帧上传请求: session={session_id}, file={file.filename}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        image = cv2.imread(temp_path)
        if image is None:
            os.unlink(temp_path)
            raise HTTPException(status_code=400, detail="无法读取图片")

        print(f"✅ 图片加载成功: {image.shape}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取或创建 VideoPipeline
        pipeline = get_or_create_pipeline(session_id, fps)

        # 处理帧
        result_obj, mesh_results, features_dict = pipeline.process_frame(image_rgb)

        os.unlink(temp_path)

        if result_obj is None:
            print("❌ 未检测到人脸")
            return JSONResponse(content={
                "status": "no_face",
                "session_id": session_id,
                "message": "未检测到人脸"
            })

        print(f"✅ 分析完成: {features_dict.get('dominant_emotion', 'unknown')} "
              f"(置信度: {features_dict.get('confidence', 0):.2f})")

        # 构建完整响应
        return JSONResponse(content={
            "status": "success",
            "session_id": session_id,
            "result": {
                # 基础信息
                "timestamp": features_dict.get("timestamp", 0),
                "focus_score": features_dict.get("focus_score", 0),

                # AU 特征
                "au_features": {
                    "au1_inner_brow_raise": features_dict.get("au1_inner_brow_raise", 0),
                    "au2_outer_brow_raise": features_dict.get("au2_outer_brow_raise", 0),
                    "au4_frown": features_dict.get("au4_frown", 0),
                    "au6_cheek_raise": features_dict.get("au6_cheek_raise", 0),
                    "au7_eye_squeeze": features_dict.get("au7_eye_squeeze", 0),
                    "au9_nose_wrinkle": features_dict.get("au9_nose_wrinkle", 0),
                    "au10_upper_lip_raise": features_dict.get("au10_upper_lip_raise", 0),
                    "au12_smile": features_dict.get("au12_smile", 0),
                    "au14_dimpler": features_dict.get("au14_dimpler", 0),
                    "au15_mouth_down": features_dict.get("au15_mouth_down", 0),
                    "au20_lip_stretcher": features_dict.get("au20_lip_stretcher", 0),
                    "au23_lip_compression": features_dict.get("au23_lip_compression", 0),
                    "au25_mouth_open": features_dict.get("au25_mouth_open", 0),
                    "au26_jaw_drop": features_dict.get("au26_jaw_drop", 0),
                    "avg_ear": features_dict.get("avg_ear", 0),
                    "head_yaw": features_dict.get("head_yaw", 0),
                    "head_pitch": features_dict.get("head_pitch", 0),
                    "symmetry_score": features_dict.get("symmetry_score", 1.0),
                    "blink_rate_per_min": features_dict.get("blink_rate_per_min", 0),
                    "eye_closed_sec": features_dict.get("eye_closed_sec", 0),
                    "is_blink": features_dict.get("is_blink", False),
                    "left_iris_x": features_dict.get("left_iris_x", 0),
                    "left_iris_y": features_dict.get("left_iris_y", 0),
                    "right_iris_x": features_dict.get("right_iris_x", 0),
                    "right_iris_y": features_dict.get("right_iris_y", 0),
                    "gaze_direction_x": features_dict.get("gaze_direction_x", 0),
                    "gaze_direction_y": features_dict.get("gaze_direction_y", 0),
                    "gaze_deviation": features_dict.get("gaze_deviation", 0)
                },

                # 情绪分析
                "emotion": {
                    "primary_emotion": features_dict.get("dominant_emotion", "neutral"),
                    "confidence": features_dict.get("confidence", 0),
                    "scores": {
                        k: v for k, v in features_dict.items()
                        if k.startswith("emotion_")
                    },
                    "composite_emotions": [],
                    "psychological_summary": ""
                },

                # 紧张度分析
                "tension": {
                    "tension_score": features_dict.get("tension_score", 0),
                    "tension_level": features_dict.get("tension_level", "low"),
                    "sources": {
                        "brow_furrow": features_dict.get("tension_sources_brow_furrow", 0),
                        "lip_compression": features_dict.get("tension_sources_lip_compression", 0),
                        "eye_closure": features_dict.get("tension_sources_eye_closure", 0),
                        "expression_instability": features_dict.get("tension_sources_expression_instability", 0),
                        "emotional_influence": features_dict.get("tension_sources_emotional_influence", 0)
                    }
                },

                # 微表情
                "micro_expressions": {
                    "au_name": features_dict.get("micro_exp_au_name"),
                    "intensity": features_dict.get("micro_exp_intensity"),
                    "duration_frames": features_dict.get("micro_exp_duration_frames"),
                    "onset_frame": features_dict.get("micro_exp_onset_frame")
                },

                # 时间序列统计（关键！）
                "temporal_stats": {
                    k: v for k, v in features_dict.items()
                    if any(k.endswith(suffix) for suffix in ["_trend", "_volatility", "_change_rate"])
                }
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        print(f"📋 错误堆栈:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    """重置指定会话"""
    if session_id in session_pipelines:
        del session_pipelines[session_id]
        return {"status": "success", "message": f"会话 {session_id} 已重置"}
    else:
        return {"status": "not_found", "message": f"会话 {session_id} 不存在"}


if __name__ == "__main__":
    import uvicorn


    # 启动前测试MediaPipe兼容性
    print("=" * 60)
    print("🔧 正在检查MediaPipe兼容性...")
    try:
        import mediapipe as mp

        print(f"✅ MediaPipe版本: {mp.__version__}")

        # 测试兼容性导入
        try:
            test_module = mp.solutions.face_mesh
            print("✅ 使用旧版API (mp.solutions)")
        except AttributeError:
            from mediapipe import solutions

            test_module = solutions.face_mesh
            print("✅ 使用新版API (mediapipe.solutions)")

        # 测试FaceMesh初始化
        test_mesh = test_module.FaceMesh(
            static_image_mode=False,  # 视频模式
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ FaceMesh初始化成功")
        test_mesh.close()
    except Exception as e:
        print(f"❌ MediaPipe检查失败: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 60)
    print("🚀 启动Face Expression API服务（视频流模式）...")
    print("📍 地址: http://0.0.0.0:8000")
    print("💡 特性: 会话管理、时间序列分析、微表情检测")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)

