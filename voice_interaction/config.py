"""
配置模块

管理 voice_interaction 模块的配置参数。
支持通过环境变量或 .env 文件加载敏感信息（如 API 密钥）。
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# ======================
# 路径配置（支持环境变量覆盖）
# ======================

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

# 输出子目录配置
VOICE_OUTPUT_DIR = OUTPUT_DIR / "voice_interaction"  # 语音交互输出目录

# 确保必要目录存在
for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, LOGS_DIR, VOICE_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ======================
# 敏感 API 密钥（强制从环境变量加载）
# ======================

# 优先从环境变量读取，若缺失则留空（后续模块会报错提示）
DASHSCOPE_API_KEY: str = os.getenv('DASHSCOPE_API_KEY', '').strip()
BAIDU_API_KEY: str = os.getenv('BAIDU_API_KEY', '').strip()
BAIDU_SECRET_KEY: str = os.getenv('BAIDU_SECRET_KEY', '').strip()


# 安全检查：启动时可调用此函数提示缺失密钥
def check_api_keys() -> None:
    """检查关键 API 密钥是否配置，若缺失则打印警告"""
    missing = []
    if not DASHSCOPE_API_KEY:
        missing.append("DASHSCOPE_API_KEY")
    if not BAIDU_API_KEY or not BAIDU_SECRET_KEY:
        missing.append("BAIDU_API_KEY/BAIDU_SECRET_KEY")

    if missing:
        print("⚠️  警告：以下 API 密钥未配置，相关功能将不可用：")
        for key in missing:
            print(f"    - {key}")
        print("💡 建议：设置环境变量或创建 .env 文件")


# ======================
# 配置类型定义
# ======================

SpeechConfig = Dict[str, Any]
TTSConfig = Dict[str, Any]
AssessmentConfig = Dict[str, Any]

# ======================
# 语音识别配置
# ======================

SPEECH_CONFIG: SpeechConfig = {
    'sample_rate': 16000,
    'channels': 1,
    'chunk_duration': 0.5,  # 音频块时长（秒）
    'pause_threshold': 1.2,  # 停顿阈值（秒）
    'non_speaking_duration': 1.2,  # 非语音持续时长（秒）
    'timeout': 120,  # 最大监听超时（秒）
    'max_audio_length': 58  # 百度 API 限制：≤60秒
}

# ======================
# 语音合成配置
# ======================

TTS_CONFIG: TTSConfig = {
    'rate': 160,  # 语速 (50-400)
    'volume': 1.0,  # 音量 (0.0-1.0)
    'voice_preference': 'chinese'  # 语音偏好
}

# ======================
# 评估配置
# ======================

ASSESSMENT_CONFIG: AssessmentConfig = {
    'use_ai_feedback': True,  # 是否启用 DashScope AI 反馈
    'ai_model': 'qwen-plus',  # 使用的模型
    'max_tokens': 300,  # 最大生成 token 数
    'save_logs': True  # 是否保存评估日志
}

# ======================
# FFmpeg 配置
# ======================

FFMPEG_PATH: str = os.getenv('FFMPEG_PATH', 'ffmpeg')

# ======================
# API 配置
# ======================

API_CONFIG: Dict[str, Any] = {
    'host': '0.0.0.0',
    'port': 8001,  # 避免与 gesture_analysis (8002) 冲突
    'debug': False
}

# ======================
# 日志配置
# ======================

LOG_CONFIG: Dict[str, str] = {
    'encoding': 'utf-8',
    'interview_log_file': 'interview_log_{timestamp}.txt',
    'research_log_file': 'research_assessment_{timestamp}.txt'
}