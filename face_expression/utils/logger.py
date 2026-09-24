"""
日志工具模块

提供数据日志记录功能
"""

import csv
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from ..config import LOGS_DIR, LOG_CONFIG

logger = logging.getLogger(__name__)

NONE_SESSION = "NONE"          # 无 id 时的显式占位,与另两个 logger 及 asr/session.py 同值(测试守住)


class DataLogger:
    """数据日志记录器"""

    def __init__(self, log_type: str = 'video', session_id: Optional[str] = None):
        """
        初始化日志记录器

        参数:
            log_type: 日志类型 ('video' 或 'static')
            session_id: 可选会话ID，用于生成固定文件名（同一会话写入同一文件）
        """
        self.log_type = log_type
        self.session_id = session_id or NONE_SESSION

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_template = (LOG_CONFIG['video_log_file'] if log_type == 'video'
                         else LOG_CONFIG['static_log_file'])
        sid_part = self.session_id if session_id else f"{NONE_SESSION}_{timestamp}"
        self.log_file = os.path.join(LOGS_DIR, file_template.format(timestamp=sid_part))

        os.makedirs(LOGS_DIR, exist_ok=True)

        if log_type == 'video':
            self.fieldnames = [
                "session_id",
                "timestamp", "focus_score", "blink_rate_per_min",
                "au1_inner_brow_raise", "au2_outer_brow_raise", "au4_frown",
                "au6_cheek_raise", "au7_eye_squeeze", "au9_nose_wrinkle",
                "au10_upper_lip_raise", "au12_smile", "au14_dimpler",
                "au15_mouth_down", "au20_lip_stretcher", "au23_lip_compression",
                "au25_mouth_open", "au26_jaw_drop", "avg_ear",
                "head_yaw", "head_pitch", "symmetry_score",
                "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
                "gaze_direction_x", "gaze_direction_y", "gaze_deviation",
                "eye_closed_sec", "is_blink",
                "dominant_emotion", "confidence",
                "tension_score", "tension_level",
                "micro_exp_au_name", "micro_exp_intensity",
                "micro_exp_duration_frames", "micro_exp_onset_frame"
            ]
        else:
            self.fieldnames = [
                "session_id",
                "timestamp", "image_path", "focus_score", "blink_status",
                "au4_frown", "au12_smile", "au9_nose_wrinkle",
                "au15_mouth_down", "au25_mouth_open", "eye_closed_sec", "emotion"
            ]

        if not os.path.exists(self.log_file):
            self._write_header()

    def _write_header(self) -> None:
        """写入CSV文件头"""
        try:
            with open(self.log_file, 'w', newline='', encoding=LOG_CONFIG['encoding']) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        except Exception as e:
            logger.error("创建日志文件失败: %s", e)

    def log(self, data: Dict[str, Any]) -> bool:
        """
        记录数据到日志文件

        参数:
            data: 要记录的数据字典

        返回:
            bool: 记录是否成功
        """
        try:
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now().timestamp()

            # session_id 由 logger 自己决定（构造时传入），不接受行数据覆盖
            row = {}
            for field in self.fieldnames:
                if field == 'session_id':
                    row[field] = self.session_id
                elif field in data:
                    row[field] = data[field]

            # 调用方可能换过 log_file（face_expression/api/app.py 就是），
            # 新路径上还没有表头，补一次，否则 CSV 的第一行数据会被当成表头。
            if not os.path.exists(self.log_file):
                self._write_header()

            with open(self.log_file, 'a', newline='', encoding=LOG_CONFIG['encoding']) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)
            return True
        except Exception as e:
            logger.error("日志记录失败: %s", e)
            return False

    def get_log_path(self) -> str:
        """获取日志文件路径"""
        return self.log_file
