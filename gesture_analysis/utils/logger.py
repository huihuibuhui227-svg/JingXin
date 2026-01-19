"""
日志工具模块

提供手势和肩部分析结果的日志记录功能
支持结构化 CSV 日志，便于后续分析与审计。
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from ..config import LOGS_DIR, LOG_CONFIG


class GestureLogger:
    """手势分析日志记录器"""

    def __init__(self, log_dir: Optional[str] = None, log_file_name: Optional[str] = None):
        """
        初始化日志记录器

        参数:
            log_dir: 日志目录路径，若为 None 则使用 config.LOGS_DIR
            log_file_name: 日志文件名模板，若为 None 则使用 config 中的模板
        """
        self.log_dir = Path(log_dir) if log_dir else Path(LOGS_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_template = log_file_name or LOG_CONFIG['gesture_log_file']
        self.log_file = self.log_dir / file_template.format(timestamp=timestamp)

        # 定义字段名（保持与数据流一致）
        self.fieldnames = [
            "timestamp",                # Unix 时间戳（秒）
            "timestamp_iso",            # ISO 8601 格式时间（便于阅读）
            "left_hand_score",
            "right_hand_score",
            "shoulder_score",
            "overall_score",
            "emotion_state",
            "feedback",
            "used_features",            # 新增：标明使用了哪些特征
            "is_valid"                  # 新增：整体结果是否有效
        ]

        # 写入文件头（仅一次）
        self._write_header()

    def _write_header(self) -> None:
        """写入 CSV 文件头（幂等操作）"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding=LOG_CONFIG['encoding']) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(
        self,
        left_hand_result: Optional[Dict[str, Any]],
        right_hand_result: Optional[Dict[str, Any]],
        shoulder_result: Optional[Dict[str, Any]],
        emotion_result: Optional[Dict[str, Any]]
    ) -> bool:
        """
        记录分析结果到日志文件

        参数:
            left_hand_result: 左手分析结果
            right_hand_result: 右手分析结果
            shoulder_result: 肩部分析结果
            emotion_result: 情绪评估结果

        返回:
            是否成功写入
        """
        try:
            now = datetime.now()
            data = {
                "timestamp": now.timestamp(),
                "timestamp_iso": now.isoformat(),
                "left_hand_score": self._safe_get(left_hand_result, 'resilience_score', 0.0),
                "right_hand_score": self._safe_get(right_hand_result, 'resilience_score', 0.0),
                "shoulder_score": self._safe_get(shoulder_result, 'shoulder_score', 0.0),
                "overall_score": self._safe_get(emotion_result, 'overall_score', 0.0),
                "emotion_state": self._safe_get(emotion_result, 'emotion_state', ''),
                "feedback": self._safe_get(emotion_result, 'feedback', ''),
                "used_features": self._safe_get(emotion_result, 'used_features', 'none'),
                "is_valid": bool(self._safe_get(emotion_result, 'is_valid', False))
            }

            with open(self.log_file, 'a', newline='', encoding=LOG_CONFIG['encoding']) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(data)
            return True

        except Exception as e:
            print(f"❌ 日志记录失败: {str(e)}")
            return False

    def _safe_get(self, obj: Optional[Dict], key: str, default):
        """安全获取字典值，避免 KeyError 或 AttributeError"""
        if not isinstance(obj, dict):
            return default
        return obj.get(key, default)

    def get_log_path(self) -> str:
        """获取日志文件路径"""
        return str(self.log_file)