# main/storage.py
"""
JingXin 数据存储模块（增强版）
- 线程安全：每个线程独立数据库连接
- 支持三种模态：面部、手势、语音
- 自动重连机制
- 详细错误日志
"""

import json
import pymssql
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class FileLogger:
    """文件日志记录器（JSONL 格式）- 线程安全"""

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # 线程锁

    def log(self, session_id: str, modality: str, source_module: str,
            data: Dict[str, Any]) -> bool:
        """写入日志到文件"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "modality": modality,
            "source_module": source_module,
            "data": data
        }

        try:
            with self._lock:  # 确保线程安全
                log_file = self.log_dir / f"{session_id}.jsonl"
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")
            return True
        except Exception as e:
            print(f"[FileLogger Error] 写入失败：{e}")
            return False

    def close(self):
        """清理资源"""
        pass


class SqlServerLogger:
    """
    SQL Server 日志记录器（线程安全增强版）
    - 每个线程独立数据库连接
    - 自动重连机制
    - 支持面部、手势、语音三种模态
    """

    def __init__(self, config: Optional[Dict] = None):
        self._local = threading.local()  # 线程局部存储
        self._global_lock = threading.Lock()  # 全局锁（用于初始化）
        self._conn_count = 0

        # 数据库配置
        self._config = config or {
            'host': 'localhost',
            'server': 'LAPTOP-1H16P3V0',
            'port': '1433',
            'user': 'jingxin',
            'password': '123456',
            'database': 'jingxin',
            'charset': 'UTF-8'
        }

        # 字段映射表（日志字段名 -> 数据库字段名）
        self._face_field_map = self._build_face_field_map()

        # 初始化连接
        self._get_connection()

    def _build_face_field_map(self) -> Dict[str, str]:
        """构建面部日志字段映射表"""
        return {
            # AU 趋势字段映射
            'au1_inner_brow_raise_trend': 'au1_trend',
            'au1_inner_brow_raise_volatility': 'au1_volatility',
            'au1_inner_brow_raise_change_rate': 'au1_change_rate',
            'au2_outer_brow_raise_trend': 'au2_trend',
            'au2_outer_brow_raise_volatility': 'au2_volatility',
            'au2_outer_brow_raise_change_rate': 'au2_change_rate',
            'au4_frown_trend': 'au4_trend',
            'au4_frown_volatility': 'au4_volatility',
            'au4_frown_change_rate': 'au4_change_rate',
            'au6_cheek_raise_trend': 'au6_trend',
            'au6_cheek_raise_volatility': 'au6_volatility',
            'au6_cheek_raise_change_rate': 'au6_change_rate',
            'au7_eye_squeeze_trend': 'au7_trend',
            'au7_eye_squeeze_volatility': 'au7_volatility',
            'au7_eye_squeeze_change_rate': 'au7_change_rate',
            'au9_nose_wrinkle_trend': 'au9_trend',
            'au9_nose_wrinkle_volatility': 'au9_volatility',
            'au9_nose_wrinkle_change_rate': 'au9_change_rate',
            'au10_upper_lip_raise_trend': 'au10_trend',
            'au10_upper_lip_raise_volatility': 'au10_volatility',
            'au10_upper_lip_raise_change_rate': 'au10_change_rate',
            'au12_smile_trend': 'au12_trend',
            'au12_smile_volatility': 'au12_volatility',
            'au12_smile_change_rate': 'au12_change_rate',
            'au14_dimpler_trend': 'au14_trend',
            'au14_dimpler_volatility': 'au14_volatility',
            'au14_dimpler_change_rate': 'au14_change_rate',
            'au15_mouth_down_trend': 'au15_trend',
            'au15_mouth_down_volatility': 'au15_volatility',
            'au15_mouth_down_change_rate': 'au15_change_rate',
            'au20_lip_stretcher_trend': 'au20_trend',
            'au20_lip_stretcher_volatility': 'au20_volatility',
            'au20_lip_stretcher_change_rate': 'au20_change_rate',
            'au23_lip_compression_trend': 'au23_trend',
            'au23_lip_compression_volatility': 'au23_volatility',
            'au23_lip_compression_change_rate': 'au23_change_rate',
            'au25_mouth_open_trend': 'au25_trend',
            'au25_mouth_open_volatility': 'au25_volatility',
            'au25_mouth_open_change_rate': 'au25_change_rate',
            'au26_jaw_drop_trend': 'au26_trend',
            'au26_jaw_drop_volatility': 'au26_volatility',
            'au26_jaw_drop_change_rate': 'au26_change_rate',

            # 其他趋势字段
            'avg_ear_trend': 'avg_ear_trend',
            'avg_ear_volatility': 'avg_ear_volatility',
            'avg_ear_change_rate': 'avg_ear_change_rate',
            'head_yaw_trend': 'head_yaw_trend',
            'head_yaw_volatility': 'head_yaw_volatility',
            'head_yaw_change_rate': 'head_yaw_change_rate',
            'head_pitch_trend': 'head_pitch_trend',
            'head_pitch_volatility': 'head_pitch_volatility',
            'head_pitch_change_rate': 'head_pitch_change_rate',
            'symmetry_score_trend': 'symmetry_score_trend',
            'symmetry_score_volatility': 'symmetry_score_volatility',
            'symmetry_score_change_rate': 'symmetry_score_change_rate',
            'blink_rate_per_min_trend': 'blink_rate_trend',
            'blink_rate_per_min_volatility': 'blink_rate_volatility',
            'blink_rate_per_min_change_rate': 'blink_rate_change_rate',
            'eye_closed_sec_trend': 'eye_closed_trend',
            'eye_closed_sec_volatility': 'eye_closed_volatility',
            'eye_closed_sec_change_rate': 'eye_closed_change_rate',
            'gaze_direction_x_trend': 'gaze_x_trend',
            'gaze_direction_x_volatility': 'gaze_x_volatility',
            'gaze_direction_x_change_rate': 'gaze_x_change_rate',
            'gaze_direction_y_trend': 'gaze_y_trend',
            'gaze_direction_y_volatility': 'gaze_y_volatility',
            'gaze_direction_y_change_rate': 'gaze_y_change_rate',
            'gaze_deviation_trend': 'gaze_dev_trend',
            'gaze_deviation_volatility': 'gaze_dev_volatility',
            'gaze_deviation_change_rate': 'gaze_dev_change_rate',
        }

    def _get_connection(self) -> Optional[pymssql.Connection]:
        """获取当前线程的独立数据库连接（带自动重连）"""
        # 检查是否已有连接
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            # 健康检查
            if self._check_connection_health(self._local.conn):
                return self._local.conn
            else:
                # 连接已断开，关闭并重建
                try:
                    self._local.conn.close()
                except:
                    pass
                self._local.conn = None

        # 创建新连接
        try:
            with self._global_lock:
                conn = pymssql.connect(**self._config)
                self._local.conn = conn
                self._conn_count += 1
                print(f"✅ [DB] 新连接建立 (线程：{threading.current_thread().name}, 总连接数：{self._conn_count})")
                return conn
        except Exception as e:
            print(f"❌ [DB] 连接失败：{e}")
            self._local.conn = None
            return None

    def _check_connection_health(self, conn: pymssql.Connection) -> bool:
        """检查连接是否健康"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return True
        except:
            return False

    def close_all(self):
        """关闭当前线程的连接"""
        if hasattr(self._local, 'conn') and self._local.conn:
            try:
                self._local.conn.close()
                self._conn_count -= 1
                print(f"🔒 [DB] 连接关闭 (线程：{threading.current_thread().name})")
            except:
                pass
            self._local.conn = None

    def log_session(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """插入会话元数据"""
        conn = self._get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            sql = """
            INSERT INTO sessions (session_id, name, gender, birth_date, start_time, session_type)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            raw_gender = metadata.get("gender", "")
            clean_gender = "男" if str(raw_gender).strip() == "男" else "女"

            cursor.execute(sql, (
                session_id,
                metadata.get("name", "anonymous"),
                clean_gender,
                metadata.get("birth_date", "2000-01-01"),
                metadata.get("start_time", datetime.now().isoformat()),
                metadata.get("session_type", "integrated_interview")
            ))
            conn.commit()
            print(f"✅ [DB] sessions 插入成功：{session_id}")
            return True
        except Exception as e:
            error_str = str(e)
            # 忽略重复插入错误
            if "Duplicate" not in error_str and "PRIMARY KEY" not in error_str and "23000" not in error_str:
                print(f"[SQL Session Error] {e}")
            return False

    def log_face(self, session_id: str, timestamp: float, data: Dict[str, Any]) -> bool:
        """插入面部日志（带字段映射）"""
        return self._insert_data("face_logs", session_id, timestamp, data,
                                 self._build_face_values)

    def log_gesture(self, session_id: str, timestamp: float, data: Dict[str, Any]) -> bool:
        """插入手势日志"""
        return self._insert_data("gesture_logs", session_id, timestamp, data,
                                 self._build_gesture_values)

    def log_voice(self, session_id: str, timestamp: float, data: Dict[str, Any]) -> bool:
        """插入语音日志（新增）"""
        return self._insert_data("voice_logs", session_id, timestamp, data,
                                 self._build_voice_values)

    def _insert_data(self, table: str, session_id: str, ts: float,
                     data: Dict, builder_func) -> bool:
        """通用插入方法（带重试机制）"""
        conn = self._get_connection()
        if not conn:
            return False

        max_retries = 2
        for attempt in range(max_retries):
            try:
                cursor = conn.cursor()

                # 动态获取表列名
                cursor.execute(f"SELECT TOP 0 * FROM {table}")
                columns = [col[0] for col in cursor.description if col[0] != 'id']

                # 构建值列表
                values = builder_func(columns, session_id, ts, data)

                # 验证字段数匹配
                if len(values) != len(columns):
                    print(f"[SQL Warning] 字段数不匹配：{len(values)} vs {len(columns)}")
                    print(f"  表：{table}")
                    return False

                # 执行插入
                placeholders = ", ".join(["%s"] * len(values))
                sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                cursor.execute(sql, values)
                conn.commit()
                return True

            except Exception as e:
                error_str = str(e)
                if attempt < max_retries - 1:
                    # 重试前重连
                    print(f"[SQL Retry] 第{attempt + 1}次重试...")
                    self._local.conn = None
                    conn = self._get_connection()
                    if not conn:
                        break
                else:
                    # 忽略重复插入错误
                    if "Duplicate" not in error_str and "PRIMARY KEY" not in error_str and "23000" not in error_str:
                        print(f"[SQL Insert Error in {table}] {e}")
                        import traceback
                        traceback.print_exc()
                    try:
                        conn.rollback()
                    except:
                        pass
                    return False

        return False

    def _build_face_values(self, columns: List[str], session_id: str,
                           ts: float, data: Dict) -> List[Any]:
        """构建面部日志值（带字段映射）"""
        vals = []
        for col in columns:
            if col == "session_id":
                vals.append(session_id)
            elif col == "timestamp":
                vals.append(datetime.fromtimestamp(ts))
            else:
                # 1. 先尝试直接匹配
                value = data.get(col, None)

                # 2. 如果直接匹配失败，尝试映射表
                if value is None:
                    for log_field, db_field in self._face_field_map.items():
                        if db_field == col and log_field in data:
                            value = data.get(log_field, None)
                            break

                vals.append(value)
        return vals

    def _build_gesture_values(self, columns: List[str], session_id: str,
                              ts: float, data: Dict) -> List[Any]:
        """构建手势日志值"""
        arm = data.get("arm", {})
        left = arm.get("left", {}).get("features", {})
        right = arm.get("right", {}).get("features", {})
        shoulder = data.get("shoulder", {})
        head_torso = data.get("upper_body", {}).get("features", {})
        emotion = data.get("emotion", {})

        vals = []
        for col in columns:
            if col == "session_id":
                vals.append(session_id)
            elif col == "timestamp":
                vals.append(datetime.fromtimestamp(ts))
            # 左手
            elif col == "left_wrist_x":
                vals.append(left.get("wrist_position", [None, None])[0])
            elif col == "left_wrist_y":
                vals.append(left.get("wrist_position", [None, None])[1])
            elif col == "left_elbow_x":
                vals.append(left.get("elbow_position", [None, None])[0])
            elif col == "left_elbow_y":
                vals.append(left.get("elbow_position", [None, None])[1])
            elif col == "left_shoulder_x":
                vals.append(left.get("shoulder_position", [None, None])[0])
            elif col == "left_shoulder_y":
                vals.append(left.get("shoulder_position", [None, None])[1])
            elif col == "left_wrist_jitter":
                vals.append(left.get("wrist_jitter", None))
            elif col == "left_elbow_jitter":
                vals.append(left.get("elbow_jitter", None))
            elif col == "left_arm_angle":
                vals.append(left.get("arm_angle", None))
            elif col == "left_is_valid":
                vals.append(int(bool(left.get("is_valid", True))))
            elif col == "left_arm_score":
                vals.append(left.get("arm_score", None))
            elif col == "left_arm_stability":
                vals.append(left.get("arm_stability", None))
            # 右手
            elif col == "right_wrist_x":
                vals.append(right.get("wrist_position", [None, None])[0])
            elif col == "right_wrist_y":
                vals.append(right.get("wrist_position", [None, None])[1])
            elif col == "right_elbow_x":
                vals.append(right.get("elbow_position", [None, None])[0])
            elif col == "right_elbow_y":
                vals.append(right.get("elbow_position", [None, None])[1])
            elif col == "right_shoulder_x":
                vals.append(right.get("shoulder_position", [None, None])[0])
            elif col == "right_shoulder_y":
                vals.append(right.get("shoulder_position", [None, None])[1])
            elif col == "right_wrist_jitter":
                vals.append(right.get("wrist_jitter", None))
            elif col == "right_elbow_jitter":
                vals.append(right.get("elbow_jitter", None))
            elif col == "right_arm_angle":
                vals.append(right.get("arm_angle", None))
            elif col == "right_is_valid":
                vals.append(int(bool(right.get("is_valid", True))))
            elif col == "right_arm_score":
                vals.append(right.get("arm_score", None))
            elif col == "right_arm_stability":
                vals.append(right.get("arm_stability", None))
            # 肩部
            elif col == "shoulder_left_x":
                vals.append(shoulder.get("left_x", None))
            elif col == "shoulder_right_x":
                vals.append(shoulder.get("right_x", None))
            elif col == "shoulder_left_y":
                vals.append(shoulder.get("left_y", None))
            elif col == "shoulder_right_y":
                vals.append(shoulder.get("right_y", None))
            elif col == "shoulder_left_jitter":
                vals.append(shoulder.get("left_jitter", None))
            elif col == "shoulder_right_jitter":
                vals.append(shoulder.get("right_jitter", None))
            elif col == "shoulder_height_diff":
                vals.append(shoulder.get("height_diff", None))
            elif col == "shoulder_shrug_level":
                vals.append(shoulder.get("shrug_level", None))
            elif col == "shoulder_is_valid":
                vals.append(int(bool(shoulder.get("is_valid", True))))
            elif col == "shoulder_is_calibrated":
                vals.append(int(bool(shoulder.get("is_calibrated", False))))
            elif col == "shoulder_score":
                vals.append(shoulder.get("score", None))
            # 头部和躯干
            elif col == "head_x":
                vals.append(head_torso.get("head_x", None))
            elif col == "head_y":
                vals.append(head_torso.get("head_y", None))
            elif col == "torso_x":
                vals.append(head_torso.get("torso_x", None))
            elif col == "torso_y":
                vals.append(head_torso.get("torso_y", None))
            elif col == "head_jitter":
                vals.append(head_torso.get("head_jitter", None))
            elif col == "torso_jitter":
                vals.append(head_torso.get("torso_jitter", None))
            elif col == "head_tilt":
                vals.append(head_torso.get("head_tilt", None))
            elif col == "torso_stability":
                vals.append(head_torso.get("torso_stability", None))
            elif col == "upper_body_is_valid":
                vals.append(int(bool(head_torso.get("is_valid", True))))
            elif col == "head_score":
                vals.append(head_torso.get("head_score", None))
            elif col == "torso_score":
                vals.append(head_torso.get("torso_score", None))
            # 情绪
            elif col == "overall_score":
                vals.append(emotion.get("overall_score", None))
            elif col == "emotion_state":
                vals.append(emotion.get("emotion_state", ""))
            elif col == "emoji":
                vals.append(emotion.get("emoji", ""))
            elif col == "feedback":
                vals.append(emotion.get("feedback", ""))
            elif col == "color_r":
                vals.append(emotion.get("color", [0, 0, 0])[0])
            elif col == "color_g":
                vals.append(emotion.get("color", [0, 0, 0])[1])
            elif col == "color_b":
                vals.append(emotion.get("color", [0, 0, 0])[2])
            elif col == "used_features":
                vals.append(",".join(emotion.get("used_features", [])) if emotion.get("used_features") else None)
            elif col == "is_valid":
                vals.append(int(bool(emotion.get("is_valid", True))))
            else:
                vals.append(None)
        return vals

    def _build_voice_values(self, columns: List[str], session_id: str,
                            ts: float, data: Dict) -> List[Any]:
        """
        构建语音日志值
        对应 voice_logs 表结构：
        unix_timestamp, timestamp, pitch_mean, pitch_variation, pitch_trend,
        pitch_direction, energy_mean, energy_variation, speech_ratio,
        duration_sec, pause_duration_mean, pause_duration_max, pause_frequency,
        emotion, feedback, question_index, is_valid
        """
        vals = []
        for col in columns:
            if col == "session_id":
                vals.append(session_id)
            elif col == "unix_timestamp":
                vals.append(ts)
            elif col == "timestamp":
                vals.append(datetime.fromtimestamp(ts))
            elif col == "pitch_mean":
                vals.append(data.get("pitch_mean", None))
            elif col == "pitch_variation":
                vals.append(data.get("pitch_variation", None))
            elif col == "pitch_trend":
                vals.append(data.get("pitch_trend", None))
            elif col == "pitch_direction":
                vals.append(data.get("pitch_direction", ""))
            elif col == "energy_mean":
                vals.append(data.get("energy_mean", None))
            elif col == "energy_variation":
                vals.append(data.get("energy_variation", None))
            elif col == "speech_ratio":
                vals.append(data.get("speech_ratio", None))
            elif col == "duration_sec":
                vals.append(data.get("duration_sec", None))
            elif col == "pause_duration_mean":
                vals.append(data.get("pause_duration_mean", None))
            elif col == "pause_duration_max":
                vals.append(data.get("pause_duration_max", None))
            elif col == "pause_frequency":
                vals.append(data.get("pause_frequency", None))
            elif col == "emotion":
                vals.append(data.get("emotion", ""))
            elif col == "feedback":
                vals.append(data.get("feedback", ""))
            elif col == "question_index":
                vals.append(data.get("question_index", None))
            elif col == "is_valid":
                vals.append(int(bool(data.get("is_valid", True))))
            else:
                vals.append(None)
        return vals