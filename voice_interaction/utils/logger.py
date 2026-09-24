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

NONE_SESSION = "NONE"          # 无 id 时的显式占位,与另两个 logger 及 asr/session.py 同值(测试守住)


class VoiceLogger:
    """语音交互结构化日志记录器"""

    def __init__(self, log_type: str = 'interview', log_dir: Optional[str] = None,
                 session_id: Optional[str] = None):
        """
        初始化日志记录器

        参数:
            log_type: 日志类型 ('interview' 或 'research')
            log_dir: 日志目录路径，若为 None 则使用 config.LOGS_DIR
            session_id: 会话ID，用作文件名的一部分（缺省时退回 NONE + 时间戳）
        """
        if log_type not in ('interview', 'research'):
            raise ValueError("log_type 必须为 'interview' 或 'research'")

        self.log_type = log_type
        self.session_id = session_id or NONE_SESSION
        self.log_dir = Path(log_dir) if log_dir else Path(LOGS_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 文件名带会话ID：同一会话的日志归一堆，跨会话错配一眼可见
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sid_part = self.session_id if session_id else f"{NONE_SESSION}_{timestamp}"
        prefix = 'interview_emotion_log' if log_type == 'interview' else 'research_emotion_log'

        self.csv_file = self.log_dir / f'{prefix}_{sid_part}.csv'
        self.json_file = self.log_dir / f'{prefix}_{sid_part}.json'

        # 定义 CSV 字段 - 详细语音特征
        self.fieldnames = [
            "session_id",  # 会话ID（首列：报告侧按它归堆/排除）
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
            "is_valid",  # 是否有效
            "connective_density",  # 连接词密度（每百字，过短不出值 → 空）
            "connective_density_std",  # 该回答分句密度的标准差
            "n_rows"  # 参与密度计算的句数
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
            is_valid: bool = True,
            connective_density: Optional[float] = None,
            connective_density_std: Optional[float] = None,
            n_rows: Optional[int] = None
    ) -> bool:
        """
        记录语音特征到结构化日志

        参数:
            prosody_data: 语音特征字典，包含 pitch_mean, pitch_variation, energy_mean, energy_variation, speech_ratio, duration_sec
            question_index: 问题索引
            emotion: 情绪状态
            feedback: 反馈信息
            is_valid: 是否有效
            connective_density: 连接词密度（每百字），未计算时保持 None（**不写 0**）
            connective_density_std: 分句密度的标准差
            n_rows: 参与密度计算的句数

        返回:
            True(写入成功)。**写不进去则抛出**,不返回 False —— 见下。

        为什么失败必须抛出(用户裁定,最终审查 D2):唯一调用点
        (`voice_interaction/api/app.py` 的 `/interview/answer_audio`)**丢弃返回值**,
        所以 `return False` 等于"什么都没发生":客户端拿到 200 和一份看着正常的响应,
        而报告侧那一行**凭空消失** —— 不出错、不留痕,只是有效样本量悄悄少一个。
        这正是 M1 要杀的诚实轴,而且发生在 M1 自己造的那个槽上(连接词密度)。
        抛出之后,端点的 `except Exception → HTTPException(500)` 会把它变成 500。
        """
        try:
            now = datetime.now()
            data = {
                "session_id": self.session_id,
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
                "is_valid": is_valid,
                "connective_density": connective_density,
                "connective_density_std": connective_density_std,
                "n_rows": n_rows
            }

            # 写入 CSV
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(data)

            return True

        except Exception as e:
            # 打印留着(排障要看得见),但**不再把异常吃掉**:`raise` 原样上抛,
            # 由端点映射成 500。返回值不再承担"成功/失败"的语义。
            print(f"❌ 语音特征日志记录失败: {e}")
            import traceback
            traceback.print_exc()
            raise

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