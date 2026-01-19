"""
日志工具模块

提供数据日志记录功能
"""

import csv
import os
from datetime import datetime
from typing import Dict, Any, Optional
from ..config import LOGS_DIR, LOG_CONFIG


class DataLogger:
    """数据日志记录器"""

    def __init__(self, log_type: str = 'video'):
        """
        初始化日志记录器

        参数:
            log_type: 日志类型 ('video' 或 'static')
        """
        self.log_type = log_type
        self.log_file = os.path.join(LOGS_DIR,
                                     LOG_CONFIG['video_log_file'] if log_type == 'video'
                                     else LOG_CONFIG['static_log_file'])

        # 确保日志目录存在
        os.makedirs(LOGS_DIR, exist_ok=True)

        # 定义字段名
        if log_type == 'video':
            self.fieldnames = [
                "timestamp", "focus_score", "blink_rate_per_min", "au4_frown",
                "au12_eyebrow_raise", "au12_smile", "au9_nose_wrinkle",
                "au15_mouth_down", "au25_mouth_open", "eye_closed_sec", "emotion"
            ]
        else:
            self.fieldnames = [
                "timestamp", "image_path", "focus_score", "blink_status",
                "au4_frown", "au12_eyebrow_raise", "au12_smile", "au9_nose_wrinkle",
                "au15_mouth_down", "au25_mouth_open", "eye_closed_sec", "emotion"
            ]

        # 如果文件不存在，创建并写入表头
        if not os.path.exists(self.log_file):
            self._write_header()

    def _write_header(self) -> None:
        """写入CSV文件头"""
        try:
            with open(self.log_file, 'w', newline='', encoding=LOG_CONFIG['encoding']) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        except Exception as e:
            print(f"❌ 创建日志文件失败: {str(e)}")

    def log(self, data: Dict[str, Any]) -> bool:
        """
        记录数据到日志文件

        参数:
            data: 要记录的数据字典

        返回:
            bool: 记录是否成功
        """
        try:
            # 添加时间戳
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().timestamp()

            # 写入数据
            with open(self.log_file, 'a', newline='', encoding=LOG_CONFIG['encoding']) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(data)
            return True
        except Exception as e:
            print(f"❌ 日志记录失败: {str(e)}")
            return False

    def get_log_path(self) -> str:
        """获取日志文件路径"""
        return self.log_file