"""
配置模块

管理 gesture_analysis 模块的配置参数
支持通过环境变量覆盖关键路径，便于测试与部署隔离。
"""

import os
import sys
from typing import Dict, Any, Literal
from pathlib import Path

# ======================
# 路径配置（支持环境变量覆盖）
# ======================

# 基础路径：模块所在目录
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent

# 允许通过环境变量指定数据目录（例如：export JINGXIN_DATA_DIR=/custom/path）
CUSTOM_DATA_DIR = os.getenv("JINGXIN_DATA_DIR")
if CUSTOM_DATA_DIR:
    DATA_DIR = Path(CUSTOM_DATA_DIR).resolve()
else:
    DATA_DIR = PROJECT_ROOT / "data"

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = DATA_DIR / "logs"

# 确保必要目录存在（使用 pathlib 更安全）
for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ======================
# 配置类型定义（提升可读性与类型安全）
# ======================

MediaPipeHandsConfig = Dict[str, Any]
MediaPipePoseConfig = Dict[str, Any]
HandAnalysisConfig = Dict[str, float]
ShoulderAnalysisConfig = Dict[str, float]
EmotionScoreRanges = Dict[Literal[
    "very_relaxed", "relaxed", "neutral", "slightly_nervous", "nervous"
], int]

# ======================
# MediaPipe 配置
# ======================

MEDIAPIPE_CONFIG: Dict[str, Dict[str, Any]] = {
    'hands': {
        'static_image_mode': False,
        'max_num_hands': 2,
        'min_detection_confidence': 0.7,
        'min_tracking_confidence': 0.5
    },
    'pose': {
        'static_image_mode': False,
        'model_complexity': 1,
        'min_detection_confidence': 0.6,
        'min_tracking_confidence': 0.6
    }
}

# ======================
# 手部分析配置
# ======================

HAND_CONFIG: HandAnalysisConfig = {
    'history_length': 30,  # 历史数据长度（帧）
    'fist_threshold': 0.08,  # 握拳判定阈值（指尖到手掌心平均距离）
    'jitter_multiplier': 1000.0,  # 抖动惩罚系数（放大抖动影响）
    'spread_bonus_multiplier': 80.0,  # 手指张开奖励系数
    'spread_threshold': 0.25,  # 张开度阈值（手指展开程度）
    'fist_penalty': 20.0  # 握拳时的情绪分惩罚
}

# ======================
# 肩部分析配置
# ======================

SHOULDER_CONFIG: ShoulderAnalysisConfig = {
    'history_length': 30,
    'baseline_frames_needed': 30,  # 初始校准所需帧数
    'jitter_multiplier': 2000.0,  # 肩部抖动更敏感，系数更高
    'shrug_penalty': 30.0,  # 耸肩惩罚分
    'max_shrug_diff': 0.1,  # 左右肩高度差超过此值视为耸肩
    'baseline_smoothing': 0.9  # 基准线平滑因子（指数移动平均）
}

# ======================
# 情绪评估配置
# ======================

EMOTION_CONFIG: Dict[str, Any] = {
    'hand_weight': 0.6,  # 手部特征权重
    'shoulder_weight': 0.4,  # 肩部特征权重
    'score_ranges': {
        'very_relaxed': 80,
        'relaxed': 65,
        'neutral': 50,
        'slightly_nervous': 35,
        'nervous': 20
    }
}


# ======================
# 可视化配置
# ======================

def _find_chinese_font() -> str:
    """自动查找系统中可用的中文字体，避免 simhei.ttf 缺失导致崩溃"""
    candidates = [
        'simhei.ttf',
        'msyh.ttc',  # 微软雅黑
        'PingFang.ttc',  # macOS
        'STHeiti Light.ttc'
    ]

    # Windows 字体目录
    if sys.platform == "win32":
        win_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for font in candidates:
            if (win_fonts / font).exists():
                return str(win_fonts / font)

    # 尝试当前目录或系统默认
    for font in candidates:
        if (Path(font)).exists():
            return font

    # 最终 fallback：返回名称，由 PIL 处理（可能警告但不崩溃）
    return "simhei.ttf"


VISUALIZATION_CONFIG: Dict[str, Any] = {
    'font_path': _find_chinese_font(),
    'font_size': 24,
    'colors': {
        'text': (255, 255, 255),
        'success': (0, 255, 0),
        'warning': (255, 165, 0),
        'danger': (0, 0, 255),
        'info': (255, 255, 0)
    }
}

# ======================
# API 配置
# ======================

API_CONFIG: Dict[str, Any] = {
    'host': '0.0.0.0',
    'port': 8002,  # 避免与 face_expression 冲突
    'debug': False
}

# ======================
# 日志配置
# ======================

LOG_CONFIG: Dict[str, str] = {
    'gesture_log_file': 'gesture_emotion_log_{timestamp}.csv',
    'encoding': 'utf-8'
}