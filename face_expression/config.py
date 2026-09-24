"""
配置模块

管理 face_expression 模块的配置参数
"""

import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# 数据路径配置
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'input')      # 存放输入图片/视频
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')    # 存放输出结果
LOGS_DIR = os.path.join(DATA_DIR, 'logs')        # 存放日志文件

# 输出子目录配置
FACE_EXPRESSION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'face_expression')  # 面部表情分析输出目录

# 确保必要的目录存在
for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, LOGS_DIR, FACE_EXPRESSION_OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ====== MediaPipe 模型文件 ======
# 2026-09-24(M1.5):原来的 `MEDIAPIPE_CONFIG` 是**死代码** —— `pipeline/video_pipeline.py`
# 硬编码了全部 5 个 kwarg、从不 import 它(整仓唯一的消费者是 examples 和一个已被注释掉的
# 覆盖行)。迁移到 tasks API 时**删掉而不是照搬**(spec §1 / D4)。
#
# 路径在这里定义、由 `pipeline/detector.py` 读取,**不硬编码在抽取代码里**(spec §6.1):
# M3 重排特征时不该动到探测器路径。
#
# ⚠️ 模型文件不进 git(`.gitignore` 已排除 `models/mediapipe/`),要按 spec §6.1 单独下载。
# ⚠️ tasks 的 `FaceLandmarker` 默认就输出含虹膜的 478 点拓扑,与旧版
#    `refine_landmarks=True` 等价,所以 `au_calculator` 里 468–476 那些下标一个都不用改。
MEDIAPIPE_MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'mediapipe')
FACE_MODEL = os.path.join(MEDIAPIPE_MODELS_DIR, 'face_landmarker.task')

# ====== 眨眼检测配置 ======
EYE_CONFIG = {
    'EAR_THRESHOLD': 0.21,                        # 眼睛纵横比阈值（用于眨眼检测）
    'BLINK_BUFFER_SIZE': 5,                       # 眨眼缓冲区大小（秒）
    'YAW_BUFFER_SIZE': 30,                        # 头部偏转缓冲区大小（帧数）
    'MIN_BLINK_INTERVAL': 0.3                     # 最小眨眼间隔（秒）
}

# ====== 情绪识别配置 ======
EMOTION_CONFIG = {
    'FATIGUE_THRESHOLD': 1.0,                     # 疲劳检测阈值（闭眼持续时间）
    'FOCUS_HIGH_THRESHOLD': 0.03,                 # 高专注度头部偏转阈值
    'FOCUS_LOW_THRESHOLD': 0.08,                  # 低专注度头部偏转阈值
    'BLINK_RATE_THRESHOLD': 30,                   # 正常眨眼频率阈值（次/分钟）
    'SMILE_RATIO_THRESHOLD': 0.35                 # 微笑比例阈值
}

# ====== API 配置 ======
API_CONFIG = {
    'host': '0.0.0.0',                            # 监听地址
    'port': 8000,                                 # 监听端口
    'debug': False                                # 是否开启调试模式
}

# ====== 日志配置 ======
LOG_CONFIG = {
    'video_log_file': 'face_au_log_{timestamp}.csv',    # 视频分析日志文件名
    'static_log_file': 'static_face_log_{timestamp}.csv',  # 静态图片分析日志文件名
    'encoding': 'utf-8'                                 # 日志文件编码
}

# ====== 可选：添加环境变量覆盖支持（可选增强） ======
# 如果你需要从环境变量覆盖配置，可以取消下面注释
# import os
# FACE_MODEL = os.getenv('MP_FACE_MODEL', FACE_MODEL)