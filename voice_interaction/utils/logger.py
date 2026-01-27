"""
日志工具模块

提供结构化日志记录功能，用于记录面试/科研评估的元数据和结果。
与 assessment 模块的 save_log() 互补：前者用于机器分析，后者用于人类阅读。
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from ..config import LOGS_DIR, LOG_CONFIG


class VoiceLogger:
    """语音交互结构化日志记录器"""

    def __init__(self, log_type: str = 'interview', log_dir: Optional[str] = None):
        """
        初始化日志记录器

        参数:
            log_type: 日志类型 ('interview' 或 'research')
            log_dir: 日志目录路径，若为 None 则使用 config.LOGS_DIR
        """
        if log_type not in ('interview', 'research'):
            raise ValueError("log_type 必须为 'interview' 或 'research'")

        self.log_type = log_type
        self.log_dir = Path(log_dir) if log_dir else Path(LOGS_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if log_type == 'interview':
            csv_file = f'interview_emotion_log_{timestamp}.csv'
            json_file = f'interview_emotion_log_{timestamp}.json'
        else:
            csv_file = f'research_emotion_log_{timestamp}.csv'
            json_file = f'research_emotion_log_{timestamp}.json'

        self.csv_file = self.log_dir / csv_file
        self.json_file = self.log_dir / json_file

        # 定义 CSV 字段 - 详细语音特征
        self.fieldnames = [
            "unix_timestamp",  # Unix 时间戳
            "timestamp",  # ISO 8601 时间戳
            "pitch_mean",  # 平均音调
            "pitch_variation",  # 音调变化
            "pitch_trend",  # 语调趋势（Hz）
            "pitch_direction",  # 语调方向（上扬/下降/平稳）
            "energy_mean",  # 平均能量
            "energy_variation",  # 能量变化
            "speech_ratio",  # 语音比例
            "duration_sec",  # 持续时间
            "pause_duration_mean",  # 平均停顿时间
            "pause_duration_max",  # 最长停顿时间
            "pause_frequency",  # 停顿频率（每分钟停顿次数）
            "emotion",  # 情绪状态
            "feedback",  # 反馈信息
            "question_index",  # 问题索引
            "is_valid"  # 是否有效
        ]

        # 写入 CSV 文件头
        self._write_csv_header()

    def _write_csv_header(self) -> None:
        """写入 CSV 文件头（幂等操作）"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_prosody(
            self,
            prosody_data: Dict[str, Any],
            question_index: int,
            emotion: str,
            feedback: str,
            is_valid: bool = True
    ) -> bool:
        """
        记录语音特征到结构化日志

        参数:
            prosody_data: 语音特征字典，包含 pitch_mean, pitch_variation, energy_mean, energy_variation, speech_ratio, duration_sec
            question_index: 问题索引
            emotion: 情绪状态
            feedback: 反馈信息
            is_valid: 是否有效

        返回:
            是否成功写入
        """
        try:
            now = datetime.now()
            data = {
                "unix_timestamp": now.timestamp(),
                "timestamp": now.isoformat(),
                "pitch_mean": prosody_data.get("pitch_mean", 0),
                "pitch_variation": prosody_data.get("pitch_variation", 0),
                "pitch_trend": prosody_data.get("pitch_trend", 0),
                "pitch_direction": prosody_data.get("pitch_direction", "无法判断"),
                "energy_mean": prosody_data.get("energy_mean", 0),
                "energy_variation": prosody_data.get("energy_variation", 0),
                "speech_ratio": prosody_data.get("speech_ratio", 0),
                "duration_sec": prosody_data.get("duration_sec", 0),
                "pause_duration_mean": prosody_data.get("pause_duration_mean", 0),
                "pause_duration_max": prosody_data.get("pause_duration_max", 0),
                "pause_frequency": prosody_data.get("pause_frequency", 0),
                "emotion": emotion,
                "feedback": feedback,
                "question_index": question_index,
                "is_valid": is_valid
            }

            # 写入 CSV
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(data)

            return True

        except Exception as e:
            print(f"❌ 语音特征日志记录失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def log_assessment(
            self,
            total_questions: int,
            answered_questions: int,
            ai_model: str,
            max_tokens: int,
            evaluation_result: Dict[str, Any]
    ) -> bool:
        """
        记录评估结果到结构化日志（已废弃，请使用 log_prosody）

        参数:
            total_questions: 总问题数
            answered_questions: 已回答问题数
            ai_model: 使用的 AI 模型
            max_tokens: 最大 token 数
            evaluation_result: 评估结果字典（来自 InterviewAssessment.get_comprehensive_evaluation()）

        返回:
            是否成功写入
        """
        print("⚠️  log_assessment 方法已废弃，请使用 log_prosody 方法")
        return False

    def get_csv_path(self) -> str:
        """获取 CSV 日志文件路径"""
        return str(self.csv_file)

    def get_json_path(self) -> str:
        """获取 JSON 日志文件路径"""
        return str(self.json_file)