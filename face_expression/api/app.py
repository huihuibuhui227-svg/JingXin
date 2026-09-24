from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import re
import cv2
import numpy as np
import time
import logging
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

try:
    from face_expression.pipeline.video_pipeline import VideoPipeline
    from face_expression.utils.logger import DataLogger, NONE_SESSION
    from face_expression.config import LOGS_DIR
    # close()(经 VideoPipeline 转发到探测器的 native 句柄)实测恒 5.0s,所以回收/重置
    # 路径一律走这个后台 helper(I1)。本模块自身 import 期不碰 mediapipe ——
    # `face_expression.pipeline.detector` 的 mediapipe import 在工厂函数体内。
    from face_expression.pipeline.detector import close_detached
except ImportError as e:
    logger.error(f"导入失败: {e}")
    logger.error("请确保已正确安装 face_expression 模块")
    raise

app = FastAPI(
    title="Face Expression Analysis API",
    description="面部表情分析API，支持视频流实时分析"
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

# 会话管理：存储每个用户的 VideoPipeline 实例
session_pipelines = {}
session_loggers = {}  # 每个会话的 DataLogger 实例
SESSION_TIMEOUT = 300  # 会话超时时间（秒）

# 会话 id 直接进日志文件名（`face_au_log_{session_id}.csv`），而它是客户端可控的
# （query 参数），所以只收 `[A-Za-z0-9_-]{1,128}` —— 路径分隔符与 `..` 一概不收。
# `NONE`（无 id 时的占位）按此模式本来就合法，不必为它开口子。
# 与 voice_interaction/asr/transcript_store.py 的守卫同模式，但**各持一份**：
# face 从 voice 包导入方向是反的，还会把 TTS/ASR 整条 import 链拉起来（见 Ruling M1-2）。
SESSION_ID_PAT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def validate_session_id(session_id: str) -> str:
    """校验客户端给的 session_id：非法一律 400。

    为什么必须在这一层拦：face 不写 transcript，`transcript_store.recording_dir()` 那道
    中央守卫覆盖不到它；而这个 id 会被拼进日志文件名 —— `../../x` 能让
    `face_au_log_../../x.csv` 落到日志目录**之外**。

    为什么不"顺手改成 NONE"：坏输入被吞掉之后客户端拿到的是 200 和一份看着正常的响应，
    问题只在报告里以"数据对不上"的形式浮出来。本项目一贯要求坏输入响亮失败。
    """
    if isinstance(session_id, str) and SESSION_ID_PAT.fullmatch(session_id):
        return session_id
    raise HTTPException(
        status_code=400,
        detail=f"非法 session_id: {session_id!r} —— 只允许字母/数字/下划线/连字符，1–128 位"
               f"（它会进日志文件名，不接受路径分隔符与 '..'）")


def normalize_session_id(raw: str | None) -> str:
    """把「客户端给的原始 id」规范化:**只有"参数没给"算没给**。

    与 `validate_session_id`(什么算非法)是**分工**,不是两层各判一次:
      * `None`(参数不存在 / 表单里没这个键)→ `NONE`,这是无会话客户端的正常路径;
      * 给了但内容是空的(`""` / `"  "`)→ **原样交出,由守卫判非法 → 400**。

    为什么空串不"顺手归 NONE":空串与"没给"在客户端那里是两件事 —— 后者是没接会话,
    前者是**参数拼错了**(例如 `?session_id=${sid}` 而 sid 为空)。静默归进 `NONE` 之后
    客户端拿到的是 200 和一份看着正常的响应,问题只在报告里以"数据对不上"的形式浮出来。
    三份副本(voice/face/gesture)由 tests/test_session_id_normalization.py 压着逐字相同。
    """
    return NONE_SESSION if raw is None else raw


async def _resolve_session_id(request: Request, session_id: str = None) -> str:
    """取本次请求的会话 id：query 参数 `session_id` 或 multipart **表单字段**同名键。

    为什么两个位置都要认：请求体是 `multipart/form-data`（帧图就是其中一个字段），前端很
    容易把 id 当**表单字段**提交。只认 query 的话，那种请求会拿 200 却静默归进 `NONE` ——
    M1「三模块按 session 对上号」的目标无声失效，而且所有这类客户端还会**共享同一个
    `session_pipelines["NONE"]`**（时序统计互相污染），报告侧看不出任何异常。

    `request.form()` 这里**不会**二次消费请求体：路由声明了 `File(...)`，FastAPI 在进端点
    之前已经把表单解析好并缓存在同一个 Request 上，这里只是读缓存；非表单请求会拿到空
    FormData，那个 except 只是兜底（本环境连 python-multipart 都没装，`form()` 会直接抛，
    没有兜底的话每个不带 query id 的请求都会炸成 500）。

    校验放在两个来源**合并之后**：表单字段来的 id 与 query 来的一样要过守卫。
    """
    # 只认「参数不存在」为没给;给了空串也算**给了**,交给守卫判非法(见 normalize_session_id)
    if session_id is None:
        try:
            form = await request.form()
        except Exception:            # 不是表单请求 / 体已损坏 / multipart 解析器不在 → 当作没给
            form = None
        # 用 `in` 而不是 `if value`:表单里给了空串与 query 同口径(给了 → 判非法)
        if form is not None and "session_id" in form:
            session_id = form.get("session_id")
    return validate_session_id(normalize_session_id(session_id))


def get_or_create_pipeline(session_id: str, fps: int = 30) -> VideoPipeline:
    """获取或创建 VideoPipeline 实例"""
    current_time = time.time()

    # 清理过期会话
    expired_sessions = [
        sid for sid, (pipeline, last_used) in session_pipelines.items()
        if current_time - last_used > SESSION_TIMEOUT
    ]
    for sid in expired_sessions:
        pipeline, _ = session_pipelines.pop(sid)
        close_detached(pipeline)  # 必须显式关:tasks 探测器持 native 句柄(spec §6.3);5.0s → 后台(I1)
        session_loggers.pop(sid, None)
        logger.info(f"清理过期会话: {sid}")

    # 获取或创建新会话
    if session_id not in session_pipelines:
        try:
            pipeline = VideoPipeline(fps=fps, session_id=session_id)
            session_pipelines[session_id] = (pipeline, current_time)
            # 同一会话的所有帧写入同一个文件，文件名带会话 id（报告侧按它归堆、NONE 单独一桶）
            log_path = os.path.join(LOGS_DIR, f'face_au_log_{session_id}.csv')
            face_logger = DataLogger(log_type='video', session_id=session_id)
            # 构造后再覆盖 log_file：Task 3 的 log() 会在新路径缺表头时补写，
            # 所以这样覆盖出来的仍是合法 CSV
            face_logger.log_file = log_path
            session_loggers[session_id] = (face_logger, current_time)
            logger.info(f"创建新会话: {session_id}, 日志: {log_path}")
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            raise HTTPException(status_code=500, detail=f"会话初始化失败: {str(e)}")
    else:
        # 更新最后使用时间
        pipeline, _ = session_pipelines[session_id]
        session_pipelines[session_id] = (pipeline, current_time)
        if session_id in session_loggers:
            session_loggers[session_id] = (session_loggers[session_id][0], current_time)

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
        request: Request,
        file: UploadFile = File(...),
        session_id: str = None,
        fps: int = 30
):
    """
    上传视频帧进行实时分析

    参数:
        file: 视频帧图片
        session_id: 会话ID（可选，**query 参数或 multipart 表单字段都能给**；
                    不提供则记入 NONE 桶；形状非法回 400）
        fps: 帧率（默认30）

    返回:
        完整的分析结果，包含AU特征、情绪、紧张度、时间序列统计等
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上传的文件必须是图片格式")

    # 无 id → NONE（不再每请求 mint 一个 uuid：那会让每一帧都变成新会话、写出新日志文件）
    session_id = await _resolve_session_id(request, session_id)

    try:
        logger.info(f"收到帧上传请求: session={session_id}, file={file.filename}")

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                contents = await file.read()
                temp_file.write(contents)
                temp_path = temp_file.name

            image = cv2.imread(temp_path)
            if image is None:
                raise HTTPException(status_code=400, detail="无法读取图片")

            logger.info(f"图片加载成功: {image.shape}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 获取或创建 VideoPipeline
            pipeline = get_or_create_pipeline(session_id, fps)

            # 处理帧
            result_obj, mesh_results, features_dict = pipeline.process_frame(image_rgb)

            # 将帧数据写入 CSV 日志（供 report_frontend 批量读取）
            if session_id in session_loggers:
                try:
                    face_logger, _ = session_loggers[session_id]
                    face_logger.log(features_dict)
                except Exception as log_err:
                    logger.warning("CSV日志写入失败: %s", log_err)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

        if result_obj is None:
            logger.warning("未检测到人脸")
            return JSONResponse(content={
                "status": "no_face",
                "session_id": session_id,
                "message": "未检测到人脸"
            })

        logger.info(f"分析完成: {features_dict.get('dominant_emotion', 'unknown')} "
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
        logger.exception("帧分析失败")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """获取会话的实时聚合统计数据"""
    if session_id not in session_pipelines:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    pipeline, _ = session_pipelines[session_id]
    return JSONResponse(content={
        "status": "success",
        "data": pipeline.get_summary(),
    })


@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    """重置指定会话。

    本端点的语义是**删掉整个会话**(不是"清空状态但保留会话"):下一次请求会走
    `get_or_create_pipeline` 重建一个全新的 `VideoPipeline`,连同全新的探测器 ——
    帧计数自然从 0 开始。所以这里**只 close() 释放 native 句柄,不调 reset()**:
    reset() 是留给"保留会话"那种语义的(`VideoPipeline.reset()` 本身有契约测钉着)。
    """
    if session_id in session_pipelines:
        pipeline, _ = session_pipelines.pop(session_id)
        close_detached(pipeline)  # 删之前先放掉探测器的 native 句柄(spec §6.3);5.0s → 后台(I1)
        session_loggers.pop(session_id, None)
        return {"status": "success", "message": f"会话 {session_id} 已重置"}
    else:
        return {"status": "not_found", "message": f"会话 {session_id} 不存在"}


if __name__ == "__main__":
    import uvicorn

    # 启动前自检:模型文件在不在、能不能真加载起来(spec §8)。
    # **不吞异常** —— 故障必须拦在启动这一步。旧的 `mp.solutions` 探针恰恰是反例:
    # 它 `except Exception` 吞掉一切,于是服务"启动看着正常、每帧静默 500"。
    logger.info("=" * 60)
    logger.info("正在自检人脸模型...")
    from face_expression.pipeline.detector import verify_models
    verify_models()
    logger.info("人脸模型自检通过")

    logger.info("=" * 60)
    logger.info("启动Face Expression API服务（视频流模式）...")
    logger.info("地址: http://0.0.0.0:8000")
    logger.info("特性: 会话管理、时间序列分析、微表情检测")
    logger.info("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)

