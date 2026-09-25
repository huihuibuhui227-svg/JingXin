from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import re
import time
import os
import logging
from logging_config import setup_logging
from session_clock import SessionClock
import media_retention

setup_logging()
logger = logging.getLogger(__name__)

from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer
from gesture_analysis.core.analysis.shoulder_analyzer import ShoulderAnalyzer
from gesture_analysis.core.analysis.arm_analyzer import ArmAnalyzer
from gesture_analysis.core.analysis.emotion_inferencer import EmotionInferencer
from gesture_analysis.utils.logger import GestureLogger, NONE_SESSION
from gesture_analysis.config import API_CONFIG, MEDIAPIPE_CONFIG, LOGS_DIR
# close() 实测恒 5.0s,所以回收/重置路径一律走这个后台 helper(I1)。**只 import 这一个
# 名字,不在这里 import HandDetector/PoseDetector** —— 那两个名字必须留在
# `get_or_create_detectors` 的函数体内按调用时解析,否则
# `tests/test_analyze_session_fallback.py` 对 `detectors.HandDetector` 的 monkeypatch
# 会失效(模块级 import 会把补丁之前的值绑死)。本模块自身 import 期不碰 mediapipe。
from gesture_analysis.core.detectors import close_detached

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

# 会话管理
session_analyzers = {}
session_loggers = {}  # 每个会话的 GestureLogger 实例
session_clocks = {}   # 每个会话一个相对时钟(M2.5 spec §4)
SESSION_TIMEOUT = 300


def _clock_for(session_id: str) -> SessionClock:
    """会话时钟。与 detectors / analyzers **同生命周期**,但分开存 —— 回收时都要动。"""
    if session_id not in session_clocks:
        session_clocks[session_id] = SessionClock()
    return session_clocks[session_id]


def _reset_session(session_id: str) -> bool:
    """删掉整个会话:分析器 + 日志器 + 探测器 + **时钟**。返回它是否真的存在过。

    与 face 侧同语义、同理由(见 face 的 `_reset_session`):时钟是 pop 而不是 reset,
    否则它就成了唯一跨会话存活的状态。
    """
    existed = session_id in session_analyzers
    if existed:
        del session_analyzers[session_id]
        session_loggers.pop(session_id, None)
    dets = detectors.pop(session_id, None)
    if dets is not None:
        for d in dets.values():
            close_detached(d)  # 5.0s/个 → 后台,否则 /reset 冻住整个服务(I1)
    clock = session_clocks.pop(session_id, None)
    if clock is not None:
        logger.info("会话 %s 收尾:实测 fps=%.3f(spec §5.5)", session_id, clock.measured_fps())
    return existed

# 按会话的探测器表（与 session_analyzers 同生命周期、同 TTL 回收 —— spec §6.3）。
# 迁移前 `hands`/`pose` 是模块级单例，所有客户端共用；tasks 的 VIDEO 模式把跟踪状态
# 挂在探测器实例上，单例会让并发会话互相污染跟踪（spec §3.5）。
detectors: dict = {}

# 会话 id 直接进日志文件名（`gesture_emotion_log_{session_id}.csv`），而它是客户端可控的
# （query 参数），所以只收 `[A-Za-z0-9_-]{1,128}` —— 路径分隔符与 `..` 一概不收。
# `NONE`（无 id 时的占位）按此模式本来就合法，不必为它开口子。
# 与 voice_interaction/asr/transcript_store.py 的守卫同模式，但**各持一份**：
# gesture 从 voice 包导入方向是反的，还会把 TTS/ASR 整条 import 链拉起来（见 Ruling M1-2）。
SESSION_ID_PAT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def validate_session_id(session_id: str) -> str:
    """校验客户端给的 session_id：非法一律 400。

    为什么必须在这一层拦：gesture 不写 transcript，`transcript_store.recording_dir()` 那道
    中央守卫覆盖不到它；而这个 id 会被拼进日志文件名 —— `../../x` 能让
    `gesture_emotion_log_../../x.csv` 落到日志目录**之外**。

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

    为什么两个位置都要认：请求体是 `multipart/form-data`（图片就是其中一个字段），前端很
    容易把 id 当**表单字段**提交。只认 query 的话，那种请求会拿 200 却静默归进 `NONE` ——
    M1「三模块按 session 对上号」的目标无声失效，而且所有这类客户端还会**共享同一个
    `session_analyzers["NONE"]`**（分析器状态互相污染），报告侧看不出任何异常。

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
        dets = detectors.pop(sid, None)
        if dets is not None:
            for d in dets.values():
                close_detached(d)  # 必须显式关:持 native 句柄(spec §6.3)。5.0s/个 → 后台(I1)
        clock = session_clocks.pop(sid, None)
        if clock is not None:
            logger.info("会话 %s 收尾:实测 fps=%.3f(spec §5.5)", sid, clock.measured_fps())

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
        # 同一会话的所有帧写入同一个文件，文件名带会话 id（报告侧按它归堆、NONE 单独一桶）
        log_path = str(LOGS_DIR / f'gesture_emotion_log_{session_id}.csv')
        # session_id 必须一起传：logger 用它写 CSV 首列（Task 3 的契约），只传路径的话
        # 首列会写成 NONE，而文件名写着真实 id —— 两边对不上，报告侧按首列归堆就漏了这些行
        session_loggers[session_id] = (GestureLogger(log_file_path=log_path, session_id=session_id),
                                       log_path, current_time)
        logger.info(f"创建手势分析会话: {session_id}, 日志: {log_path}")
    else:
        analyzers, _ = session_analyzers[session_id]
        session_analyzers[session_id] = (analyzers, current_time)
        if session_id in session_loggers:
            entry = session_loggers[session_id]
            session_loggers[session_id] = (entry[0], entry[1], current_time)

    return session_analyzers[session_id][0]


def get_or_create_detectors(session_id: str):
    """取或建本会话的探测器。VIDEO 模式的跟踪状态挂在实例上,所以必须按会话(spec D3)。"""
    current_time = time.time()

    expired = [sid for sid, (_, last) in session_analyzers.items()
               if current_time - last > SESSION_TIMEOUT]
    for sid in expired:
        dets = detectors.pop(sid, None)
        if dets is not None:
            for d in dets.values():
                close_detached(d)  # 同上:回收路径不许在事件循环里等 5.0s(I1)

    if session_id not in detectors:
        from gesture_analysis.core.detectors import HandDetector, PoseDetector
        from gesture_analysis.config import HAND_MODEL, POSE_MODEL
        detectors[session_id] = {
            'hands': HandDetector(HAND_MODEL, **{
                k: v for k, v in MEDIAPIPE_CONFIG['hands'].items()}),
            'pose': PoseDetector(POSE_MODEL, **{
                k: v for k, v in MEDIAPIPE_CONFIG['pose'].items() if k != 'tier'}),
        }
        logger.info(f"创建手势探测器会话: {session_id}")
    return detectors[session_id]


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
        request: Request,
        file: UploadFile = File(...),
        session_id: str = None
):
    """
    分析图片中的手势和肩部

    参数:
        file: 上传的图片文件（FormData格式）
        session_id: 会话ID（可选，**query 参数或 multipart 表单字段都能给**；
                    不提供则记入 NONE 桶；形状非法回 400）

    返回:
        分析结果
    """
    # 无 id → NONE（不再每请求 mint 一个 uuid：那会让每一帧都变成新会话、写出新日志文件）
    session_id = await _resolve_session_id(request, session_id)

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

        dets = get_or_create_detectors(session_id)

        # M2.5:时间戳由服务端实测 —— gesture 以前连 fps 参数都没有,直接用默认 30,
        # 而客户端实际 1 帧/秒(spec §3.5 / §3.1)。
        timestamp_ms = _clock_for(session_id).stamp_ms()
        # M2.6:先存原始字节再算(与 face 同一处、同一理由)。
        media_retention.retain_frame(session_id, "gesture", contents,
                                     declared_ts=timestamp_ms, source="/analyze")

        hand_groups = dets['hands'].detect(image_rgb, timestamp_ms)
        detected_hands = 0
        hand_scores = []

        for hand_id, landmarks in enumerate(hand_groups):
            if hand_id >= 2: break
            analyzer_key = 'left_hand' if hand_id == 0 else 'right_hand'
            analyzers[analyzer_key].update(landmarks)
            hand_scores.append(analyzers[analyzer_key].get_results()['resilience_score'])
            detected_hands += 1

        pose_landmarks = dets['pose'].detect(image_rgb, timestamp_ms)
        shoulder_score = 50.0

        if pose_landmarks:
            analyzers['shoulder'].update(pose_landmarks)
            shoulder_score = analyzers['shoulder'].get_results()['shoulder_score']

        left_arm_score = 50.0
        right_arm_score = 50.0
        if pose_landmarks:
            analyzers['left_arm'].update(pose_landmarks)
            analyzers['right_arm'].update(pose_landmarks)
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
            if _reset_session(session_id):
                return {"status": "success", "message": f"会话 {session_id} 已重置"}
            return {"status": "not_found", "message": f"会话 {session_id} 不存在"}
        else:
            for dets in detectors.values():
                for d in dets.values():
                    close_detached(d)  # 无 id 的 /reset 会关掉所有会话:实测同步时 20.04s(I1)
            detectors.clear()
            session_analyzers.clear()
            session_loggers.clear()
            session_clocks.clear()   # M2.5:无 id 的 /reset 连时钟一起清
            return {"status": "success", "message": "所有会话已重置"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    from gesture_analysis.core.detectors import verify_models
    verify_models()

    uvicorn.run(app, host=API_CONFIG['host'], port=API_CONFIG['port'])

