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
            csv_file = LOG_CONFIG['interview_log_file'].replace('.txt', '.csv').format(timestamp=timestamp)
            json_file = LOG_CONFIG['interview_log_file'].replace('.txt', '.json').format(timestamp=timestamp)
        else:
            csv_file = LOG_CONFIG['research_log_file'].replace('.txt', '.csv').format(timestamp=timestamp)
            json_file = LOG_CONFIG['research_log_file'].replace('.txt', '.json').format(timestamp=timestamp)

        self.csv_file = self.log_dir / csv_file
        self.json_file = self.log_dir / json_file

        # 定义 CSV 字段
        self.fieldnames = [
            "timestamp",  # ISO 8601 时间戳
            "unix_timestamp",  # Unix 时间戳
            "log_type",  # 日志类型
            "total_questions",  # 总问题数
            "answered_questions",  # 已回答问题数
            "ai_model",  # 使用的 AI 模型
            "max_tokens",  # 最大 token 数
            "evaluation_text",  # AI 评估文本
            "is_valid",  # 评估是否成功
            "error_message"  # 错误信息（如果有）
        ]

        # 写入 CSV 文件头
        self._write_csv_header()

    def _write_csv_header(self) -> None:
        """写入 CSV 文件头（幂等操作）"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_assessment(
            self,
            total_questions: int,
            answered_questions: int,
            ai_model: str,
            max_tokens: int,
            evaluation_result: Dict[str, Any]
    ) -> bool:
        """
        记录评估结果到结构化日志

        参数:
            total_questions: 总问题数
            answered_questions: 已回答问题数
            ai_model: 使用的 AI 模型
            max_tokens: 最大 token 数
            evaluation_result: 评估结果字典（来自 InterviewAssessment.get_comprehensive_evaluation()）

        返回:
            是否成功写入
        """
        try:
            now = datetime.now()
            data = {
                "timestamp": now.isoformat(),
                "unix_timestamp": now.timestamp(),
                "log_type": self.log_type,
                "total_questions": total_questions,
                "answered_questions": answered_questions,
                "ai_model": ai_model,
                "max_tokens": max_tokens,
                "evaluation_text": evaluation_result.get("text", ""),
                "is_valid": evaluation_result.get("is_valid", False),
                "error_message": evaluation_result.get("error", "")
            }

            # 写入 CSV
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(data)

            # 写入 JSON（完整数据快照）
            json_data = {
                "metadata": data,
                "full_assessment": evaluation_result
            }
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"❌ 结构化日志记录失败: {e}")
            return False

    def get_csv_path(self) -> str:
        """获取 CSV 日志文件路径"""
        return str(self.csv_file)

    def get_json_path(self) -> str:
        """获取 JSON 日志文件路径"""
        return str(self.json_file)