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
from gesture_analysis.core.analysis.upper_body_analyzer import UpperBodyAnalyzer
from gesture_analysis.core.feature_extraction.angles import (finger_angles, pose_angles,
                                                             shoulder_width)
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
            # 2026-09-26:上半身分析器此前**只活在 examples/ 里**,活服务从没建过它
            # ⟹ head_*/torso_* 六列结构性恒 0(实测该场 275 帧全是 0)。
            'upper_body': UpperBodyAnalyzer(),
            'emotion': EmotionInferencer(),
            # 同一套分析器再喂一份 **world(米制 3D)** 坐标:公式不变、输入空间变。
            # 那批量的"取景代理/未除尺度"来自画面坐标,这份与取景无关,且能与上面
            # 直接对照(M3 决定用哪一份时有据可依)。
            # ★ `metric=True` 是**必须显式给**的开关,它决定"除不除肩宽":米制已是解剖
            #   尺度,`*_jitter_world` 那 6 行明写「**不**除肩宽」(l0_columns.json)。
            #   漏了它 ⟹ 米制那批去找一个**从来没喂过**的肩宽 ⟹ 整批交空
            #   (不会崩、不会报错,只有真产出方看得见 —— 本批的 L0 测试就是这么抓到的)。
            'world': {
                'shoulder': ShoulderAnalyzer(metric=True),
                'left_arm': ArmAnalyzer(arm_id='left', metric=True),
                'right_arm': ArmAnalyzer(arm_id='right', metric=True),
            },
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


def process_frame(image_rgb, timestamp_ms: int, analyzers: dict, dets: dict) -> dict:
    """一帧 → 这一帧的全部产出（分析器的内部状态就地推进）。

    ★ 为什么从端点里抽出来（2026-09-27，B3）：`experiments/replay_retained.py` 的**手势腿**
    必须走**同一条接线**。重放腿若在脚本里重写一遍「喂谁、喂什么、拿哪个键」，它证的就是
    它自己抄的那份接线，而不是活路径那一份 —— 本项目已 5 次栽在「验证跑错了对象」上。
    抽出来之后，端点与重放腿调的是**同一个函数**（差别只剩 FastAPI 的取参与落盘）。

    返回两样东西，端点两样都要：
      · 各 analyzer 的结果字典 —— 响应体用它；
      · `log_kwargs` —— 直接 `**` 进 `GestureLogger.log()` 的那一组入参。重放腿用它写出
        一份**与活路径同列**的 CSV（列名映射也只此一份）。

    ⚠️ 姿态（以及它的 `shoulder_width`）**先算**：肩宽是画面坐标那批 jitter 的**分母**，
    手部分析器在下面就要用到它。把它排在手部之后，这一帧的 jitter 会拿**别处**的尺度
    做分母 —— 静默、且只在逐帧对照时才看得出来。
    """
    # 姿态那一路要 world(米制 3D)那一份:PoseLandmarker 本来就输出，
    # 本仓此前只读归一化那份(`grep world_landmarks` 零命中)。
    pose_landmarks, pose_world = dets['pose'].detect_with_world(image_rgb, timestamp_ms)
    # 双肩的**归一化图像距离** —— 协变量(报告层拿它当尺度基准消取景影响)，
    # 也是画面坐标那 10 个 jitter 的分母。单位**不是米**:本项目没有任何相机内参来源,
    # 给不出米制(见 angles.shoulder_width 与 l0_columns.json 那一行)。
    # 缺任一肩 / 本帧没有姿态 ⟹ `None` ⟹ 那批 jitter 这一帧交**空**:没有分母就没有
    # 这个率,不拿未归一化的分子顶上(= 静默换单位)。
    shoulder_scale = shoulder_width(pose_landmarks)

    hand_groups, handedness = dets['hands'].detect_with_handedness(image_rgb, timestamp_ms)
    detected_hands = 0
    hand_scores = []
    # 左右手**按模型给的 handedness 定,不按检出顺序**。
    # 2026-09-26 实测:在此之前是 `'left_hand' if hand_id == 0 else 'right_hand'`
    # —— hand_id 是"第几个被检出",换个姿势左右就互换,而日志里那两列看着像
    # 左右手。模型本来就算得出,只是没人读。
    #
    # ⚠️ 模型按**镜像(自拍)输入**判左右(官方文档原话:handedness 是
    #    "determined assuming the input image is mirrored"),而本项目的帧是画布
    #    原样绘制的**非镜像**图 ⟹ 标签要**翻过来**。原始标签与置信度都进日志
    #    (见 utils/logger.py 的四列),所以这个翻法是可审计的。
    handedness_info = {}
    used_slots = set()
    hand_lm_by_slot = {}          # 本帧每一槽收到的 hand landmarks

    for hand_id, landmarks in enumerate(hand_groups):
        if hand_id >= 2: break
        entry = handedness[hand_id] if hand_id < len(handedness) else None
        analyzer_key = None
        if entry is not None:
            label, conf = entry
            side = 'right' if label == 'Left' else 'left'      # ← 翻转,理由见上
            if f'{side}_hand' not in used_slots:
                analyzer_key = f'{side}_hand'
                handedness_info[analyzer_key] = (label, conf)
        if analyzer_key is None:
            # 模型没给 handedness(或它指的那一侧已被占):填还空着的槽,
            # 并**如实标记这一槽没有依据** —— 不假装知道它是左手还是右手。
            analyzer_key = next((c for c in ('left_hand', 'right_hand')
                                 if c not in used_slots), None)
            if analyzer_key is None:
                break
            handedness_info[analyzer_key] = None
        used_slots.add(analyzer_key)
        hand_lm_by_slot[analyzer_key] = landmarks     # 手指角度要用(下面)
        analyzers[analyzer_key].update(landmarks, timestamp_ms=timestamp_ms,
                                       shoulder_width=shoulder_scale)
        hand_scores.append(analyzers[analyzer_key].get_results()['resilience_score'])
        detected_hands += 1

    # 这两个 50.0 只喂**情绪推断**的输入(下面 hand_results/shoulder_results/…),
    # **不进日志**:日志那几列按"本帧有没有数据"写空(见 _fresh 与 _safe_get)。
    shoulder_score = 50.0
    left_arm_score = 50.0
    right_arm_score = 50.0
    angles_data = {}
    world_results = {}

    if pose_landmarks:
        analyzers['shoulder'].update(pose_landmarks, timestamp_ms=timestamp_ms,
                                     shoulder_width=shoulder_scale)
        analyzers['left_arm'].update(pose_landmarks, timestamp_ms=timestamp_ms,
                                     shoulder_width=shoulder_scale)
        analyzers['right_arm'].update(pose_landmarks, timestamp_ms=timestamp_ms,
                                      shoulder_width=shoulder_scale)
        analyzers['upper_body'].update(pose_landmarks, timestamp_ms=timestamp_ms,
                                       shoulder_width=shoulder_scale)   # ← 此前活服务从没喂过它
        shoulder_score = analyzers['shoulder'].get_results()['shoulder_score']
        left_arm_result = analyzers['left_arm'].get_results()
        right_arm_result = analyzers['right_arm'].get_results()
        left_arm_score = left_arm_result.get('arm_score', 50.0) if left_arm_result.get('is_valid') else 50.0
        right_arm_score = right_arm_result.get('arm_score', 50.0) if right_arm_result.get('is_valid') else 50.0
        # 姿态角度(肘/肩/头倾/头俯仰/肩线/躯干) —— 定义照 examples 移植,见 angles.py
        angles_data.update(pose_angles(pose_landmarks))
        # 双肩的归一化图像距离(上面那个 `shoulder_scale` 的**同一个值** ——
        # 两处各算一遍同一个 dist 就是两份口径,迟早漂)
        angles_data["shoulder_width"] = shoulder_scale

    if pose_world:
        # 同一套公式喂米制坐标。**两个分析器集各自独立**,不然状态会串。
        w = analyzers['world']
        # 米制那批**不除肩宽**(米制已是解剖尺度),所以不传 shoulder_width ——
        # 传了也不会用,但那个入参在四个 analyzer 里同名,传一个"看着有意义"的值
        # 只会让下一个读代码的人以为米制那批也除了肩宽。
        w['shoulder'].update(pose_world, timestamp_ms=timestamp_ms)
        w['left_arm'].update(pose_world, timestamp_ms=timestamp_ms)
        w['right_arm'].update(pose_world, timestamp_ms=timestamp_ms)
        wl, wr = w['left_arm'].get_results(), w['right_arm'].get_results()
        ws = w['shoulder'].get_results()
        # 名字与 logger._WORLD_COLUMNS **逐字一致**(两处不同名 = 静默的零值)
        world_results = {
            'left_wrist_jitter_world': wl.get('wrist_jitter'),
            'left_elbow_jitter_world': wl.get('elbow_jitter'),
            'right_wrist_jitter_world': wr.get('wrist_jitter'),
            'right_elbow_jitter_world': wr.get('elbow_jitter'),
            'left_arm_angle_world': wl.get('arm_angle'),
            'right_arm_angle_world': wr.get('arm_angle'),
            'left_shoulder_jitter_world': ws.get('left_jitter'),
            'right_shoulder_jitter_world': ws.get('right_jitter'),
        }

    # 手指角度:每指取三个关节(定义与 examples 一致)
    for _slot, _lms in hand_lm_by_slot.items():
        _fa = finger_angles(_lms)
        if _fa:
            angles_data[f"{_slot.replace('_hand', '')}_finger_angles"] = _fa

    # 计算手部平均分
    if detected_hands == 1:
        hand_score = hand_scores[0]
    elif detected_hands == 2:
        hand_score = sum(hand_scores) / len(hand_scores)
    else:
        hand_score = 50.0

    # 推断情绪
    emotion_result = analyzers['emotion'].infer_emotion(
        {"resilience_score": hand_score},
        {"shoulder_score": shoulder_score},
        {"arm_score": left_arm_score},
        {"arm_score": right_arm_score}
    )

    # 获取详细的分析器结果。
    # ⚠️ **本帧没喂过的槽交 None** —— 交对象等于把**上一帧的旧值**写进这一行
    #    (分析器保留上次结果),而"上一帧的度量"顶替"这一帧的度量"是不留痕迹的错;
    #    交 0/50 则是把"没测到"写成"测到一个值"。两种都不要,空就是空。
    def _fresh(key: str, fed: bool):
        return analyzers[key].get_results() if fed else None

    return {
        "detected_hands": detected_hands,
        "hand_score": hand_score,
        "emotion_result": emotion_result,
        "left_hand_results": _fresh('left_hand', 'left_hand' in used_slots),
        "right_hand_results": _fresh('right_hand', 'right_hand' in used_slots),
        "shoulder_results": _fresh('shoulder', bool(pose_landmarks)),
        "left_arm_results": _fresh('left_arm', bool(pose_landmarks)),
        "right_arm_results": _fresh('right_arm', bool(pose_landmarks)),
        "upper_body_results": _fresh('upper_body', bool(pose_landmarks)),
        # ★ 落盘那一组入参 —— 端点与重放腿**各传一份同一个字典**,列名映射只此一份。
        #   `hand_present` 直接来自上面那个帧循环的 `used_slots`,不从 `*_hand_results`
        #   反推:反推出来的只是它的代理,代理一旦被改坏,`hand_visible_*` 会**静默**
        #   开始把「不知道是哪只手」写成「这只手可见」。
        #   `handedness_info` 只回答"知不知道是哪只手",两者是两个事实
        #   (logger 侧按**合取**写:`1` 只在两者都成立时)。
        "log_kwargs": {
            "left_hand_result": _fresh('left_hand', 'left_hand' in used_slots),
            "right_hand_result": _fresh('right_hand', 'right_hand' in used_slots),
            "shoulder_result": _fresh('shoulder', bool(pose_landmarks)),
            "left_arm_result": _fresh('left_arm', bool(pose_landmarks)),
            "right_arm_result": _fresh('right_arm', bool(pose_landmarks)),
            "upper_body_result": _fresh('upper_body', bool(pose_landmarks)),
            "emotion_result": emotion_result,
            "angles_data": angles_data,
            "handedness_info": handedness_info,
            "world_results": world_results,
            "hand_present": {slot: slot in used_slots
                             for slot in ("left_hand", "right_hand")},
        },
    }


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

        # 一帧的全部产出:与 `experiments/replay_retained.py` 的手势腿**共用同一份**
        # —— 端点这边剩下的只有取图、取会话、落盘、拼响应(见 process_frame 的 docstring:
        # 重放腿不许自抄一遍接线,否则它证的是它自己抄的那份)。
        frame = process_frame(image_rgb, timestamp_ms, analyzers, dets)
        detected_hands = frame["detected_hands"]
        hand_score = frame["hand_score"]
        emotion_result = frame["emotion_result"]
        left_hand_results = frame["left_hand_results"]
        right_hand_results = frame["right_hand_results"]
        shoulder_results = frame["shoulder_results"]
        left_arm_results = frame["left_arm_results"]
        right_arm_results = frame["right_arm_results"]

        logger.info(f"分析完成: {emotion_result['emotion_state']} (评分: {emotion_result['overall_score']:.1f})")

        # 将帧数据写入 CSV 日志（供 report_frontend 批量读取）。
        # 入参整份来自 `frame["log_kwargs"]` —— 与重放腿传的是**同一个字典**。
        if session_id in session_loggers:
            try:
                gesture_logger = session_loggers[session_id][0]
                gesture_logger.log(**frame["log_kwargs"])
            except Exception as log_err:
                logger.warning("CSV日志写入失败: %s", log_err)

        # 响应里取值:**本帧没有数据(None)就交 null**,不补默认值。
        # 补 default(50.0/0.0)等于把"没测到"在响应层就变成一个看着合法的数,
        # 下游再也分不出真假;而面板是按 `is_valid` 判有没有数据的,null 不会被动用。
        def _g(res, key, default=None):
            return None if res is None else res.get(key, default)
        def _v(res, key, default=False):
            # 标志位例外:没数据就是 False(不是 null)—— 下游 `if x["is_valid"]` 要能直接判
            return default if res is None else res.get(key, default)

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
                        "resilience_score": _g(left_hand_results, 'resilience_score', 50.0),
                        "jitter": _g(left_hand_results, 'jitter', 0.0),
                        "fist_status": _g(left_hand_results, 'fist_status', False),
                        "spread": _g(left_hand_results, 'spread', 0.0),
                        "is_valid": _v(left_hand_results, 'is_valid', False)
                    },
                    "right": {
                        "resilience_score": _g(right_hand_results, 'resilience_score', 50.0),
                        "jitter": _g(right_hand_results, 'jitter', 0.0),
                        "fist_status": _g(right_hand_results, 'fist_status', False),
                        "spread": _g(right_hand_results, 'spread', 0.0),
                        "is_valid": _v(right_hand_results, 'is_valid', False)
                    },
                    "average_score": hand_score
                },

                # 肩部详细分析
                "shoulder": {
                    "shoulder_score": _g(shoulder_results, 'shoulder_score', 50.0),
                    "left_jitter": _g(shoulder_results, 'left_jitter', 0.0),
                    "right_jitter": _g(shoulder_results, 'right_jitter', 0.0),
                    "shrug_level": _g(shoulder_results, 'shrug_level', 0.0),
                    "is_calibrated": _g(shoulder_results, 'is_calibrated', False),
                    "is_valid": _v(shoulder_results, 'is_valid', False)
                },

                # 手臂详细分析
                "arm": {
                    "left": {
                        "arm_score": _g(left_arm_results, 'arm_score', 50.0),
                        "wrist_jitter": _g(left_arm_results, 'wrist_jitter', 0.0),
                        "elbow_jitter": _g(left_arm_results, 'elbow_jitter', 0.0),
                        "arm_angle": _g(left_arm_results, 'arm_angle', 0.0),
                        "is_valid": _v(left_arm_results, 'is_valid', False)
                    },
                    "right": {
                        "arm_score": _g(right_arm_results, 'arm_score', 50.0),
                        "wrist_jitter": _g(right_arm_results, 'wrist_jitter', 0.0),
                        "elbow_jitter": _g(right_arm_results, 'elbow_jitter', 0.0),
                        "arm_angle": _g(right_arm_results, 'arm_angle', 0.0),
                        "is_valid": _v(right_arm_results, 'is_valid', False)
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

