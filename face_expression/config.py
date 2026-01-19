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

# 确保必要的目录存在
for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ====== MediaPipe 配置 ======
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,                   # 视频模式下设为 False
    'max_num_faces': 1,                           # 最多检测 1 张人脸
    'refine_landmarks': True,                     # 使用精细关键点
    'min_detection_confidence': 0.8,              # 检测置信度阈值
    'min_tracking_confidence': 0.8                # 跟踪置信度阈值
}

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
    'video_log_file': 'face_au_log.csv',          # 视频分析日志文件名
    'static_log_file': 'static_face_log.csv',     # 静态图片分析日志文件名
    'encoding': 'utf-8'                           # 日志文件编码
}

# ====== 可选：添加环境变量覆盖支持（可选增强） ======
# 如果你需要从环境变量覆盖配置，可以取消下面注释
# import os
# MEDIAPIPE_CONFIG['min_detection_confidence'] = float(os.getenv('MP_MIN_DETECTION_CONFIDENCE', 0.8))