# main/integrator.py
"""
JingXin 多模态集成器（完整版）
- 协调面部、手势、语音三个子模块
- 统一日志记录（文件 + SQL Server）
- 线程安全、异常隔离
- 会话状态管理
- 模块健康检查
"""

import uuid
import time
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple
from enum import Enum
from datetime import datetime

from .storage import FileLogger, SqlServerLogger

# =========================================================================
# 子模块导入（带容错）
# =========================================================================
try:
    from face_expression.pipeline.video_pipeline import VideoPipeline

    FACE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 面部模块加载失败：{e}")
    FACE_AVAILABLE = False
    VideoPipeline = None

try:
    from gesture_analysis.pipeline.gesture_pipeline import GesturePipeline

    GESTURE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 手势模块加载失败：{e}")
    GESTURE_AVAILABLE = False
    GesturePipeline = None

try:
    from voice_interaction.pipeline.assessment_pipeline import InterviewAssessmentPipeline
    from voice_interaction.pipeline.speech_recognition_pipeline import SpeechRecognitionPipeline
    from voice_interaction.pipeline.voice_pipeline import VoiceProcessingPipeline

    VOICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 语音模块加载失败：{e}")
    VOICE_AVAILABLE = False
    InterviewAssessmentPipeline = None
    SpeechRecognitionPipeline = None
    VoiceProcessingPipeline = None


class SessionState(Enum):
    """会话状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ENDED = "ended"


class ModuleHealth(Enum):
    """模块健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class JingXinIntegrator:
    """多模态集成器"""

    def __init__(self, user_id=None, session_id=None):
        self.session_id = session_id or f"{user_id or 'anonymous'}_{uuid.uuid4().hex[:8]}"
        self.state = SessionState.IDLE
        self.start_time = None
        self.end_time = None

        # 日志系统
        self.file_logger = FileLogger()
        self.sql_logger = SqlServerLogger()
        self._session_logged = False

        # 1. 面部模块
        self.face_pipe = None
        if FACE_AVAILABLE and VideoPipeline:
            try:
                self.face_pipe = VideoPipeline(session_id=self.session_id)
                print("✅ [Face] VideoPipeline 初始化成功")
            except Exception as e:
                print(f"❌ [Face] 初始化失败：{e}")
                self.face_pipe = None

        # 2. 手势模块
        self.gesture_pipe = None
        if GESTURE_AVAILABLE and GesturePipeline:
            try:
                self.gesture_pipe = GesturePipeline()
                print("✅ [Gesture] GesturePipeline 初始化成功")
            except Exception as e:
                print(f"❌ [Gesture] 初始化失败：{e}")
                self.gesture_pipe = None

        # 3. 语音模块
        self.voice_assess = None
        self.voice_process = None
        self.speech_recog = None

        if VOICE_AVAILABLE:
            try:
                self.voice_assess = InterviewAssessmentPipeline()
                self.speech_recog = SpeechRecognitionPipeline()
                self.voice_process = VoiceProcessingPipeline()
                print("✅ [Voice] 所有 Pipeline 初始化成功")
            except Exception as e:
                print(f"❌ [Voice] 初始化失败：{e}")
                self.voice_assess = None
                self.speech_recog = None
                self.voice_process = None
        else:
            print("⚠️ [Voice] 模块不可用")

    def start_session(self, metadata: Optional[Dict] = None) -> bool:
        """启动会话"""
        try:
            self.state = SessionState.RUNNING
            self.start_time = time.time()
            self.end_time = None

            # 确保会话元数据已记录
            if metadata:
                self._ensure_session_logged(metadata)

            print(f"🚀 会话已启动: {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ 启动会话失败: {e}")
            return False

    def pause_session(self) -> bool:
        """暂停会话"""
        try:
            if self.state == SessionState.RUNNING:
                self.state = SessionState.PAUSED
                print(f"⏸️ 会话已暂停: {self.session_id}")
                return True
            return False
        except Exception as e:
            print(f"❌ 暂停会话失败: {e}")
            return False

    def resume_session(self) -> bool:
        """恢复会话"""
        try:
            if self.state == SessionState.PAUSED:
                self.state = SessionState.RUNNING
                print(f"▶️ 会话已恢复: {self.session_id}")
                return True
            return False
        except Exception as e:
            print(f"❌ 恢复会话失败: {e}")
            return False

    def end_session(self) -> bool:
        """结束会话"""
        try:
            self.state = SessionState.ENDED
            self.end_time = time.time()
            print(f"🏁 会话已结束: {self.session_id}")
            return True
        except Exception as e:
            print(f"❌ 结束会话失败: {e}")
            return False

    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        duration = None
        if self.start_time:
            end = self.end_time or time.time()
            duration = end - self.start_time

        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": duration,
            "modules": self.check_module_health()
        }

    def check_module_health(self) -> Dict[str, Dict[str, Any]]:
        """检查所有模块的健康状态"""
        health_status = {}

        # 检查面部模块
        if self.face_pipe is not None:
            try:
                health_status["face"] = {
                    "status": ModuleHealth.HEALTHY.value,
                    "available": True,
                    "message": "正常"
                }
            except Exception as e:
                health_status["face"] = {
                    "status": ModuleHealth.DEGRADED.value,
                    "available": True,
                    "message": f"异常: {str(e)}"
                }
        else:
            health_status["face"] = {
                "status": ModuleHealth.UNAVAILABLE.value,
                "available": False,
                "message": "未初始化"
            }

        # 检查手势模块
        if self.gesture_pipe is not None:
            try:
                health_status["gesture"] = {
                    "status": ModuleHealth.HEALTHY.value,
                    "available": True,
                    "message": "正常"
                }
            except Exception as e:
                health_status["gesture"] = {
                    "status": ModuleHealth.DEGRADED.value,
                    "available": True,
                    "message": f"异常: {str(e)}"
                }
        else:
            health_status["gesture"] = {
                "status": ModuleHealth.UNAVAILABLE.value,
                "available": False,
                "message": "未初始化"
            }

        # 检查语音模块
        voice_available = all([
            self.voice_assess is not None,
            self.voice_process is not None,
            self.speech_recog is not None
        ])

        if voice_available:
            try:
                health_status["voice"] = {
                    "status": ModuleHealth.HEALTHY.value,
                    "available": True,
                    "message": "正常"
                }
            except Exception as e:
                health_status["voice"] = {
                    "status": ModuleHealth.DEGRADED.value,
                    "available": True,
                    "message": f"异常: {str(e)}"
                }
        else:
            health_status["voice"] = {
                "status": ModuleHealth.UNAVAILABLE.value,
                "available": False,
                "message": "未初始化"
            }

        return health_status

    def is_session_active(self) -> bool:
        """检查会话是否处于活动状态"""
        return self.state == SessionState.RUNNING

    def _ensure_session_logged(self, metadata: Optional[Dict] = None) -> bool:
        """确保会话元数据写入数据库"""
        if not self._session_logged and self.sql_logger.conn:
            try:
                meta = metadata or {
                    "name": "unknown", "gender": "unknown", "birth_date": "2000-01-01",
                    "start_time": time.time(), "session_type": "integrated"
                }
                success = self.sql_logger.log_session(self.session_id, meta)
                if success:
                    self._session_logged = True
                return success
            except Exception as e:
                print(f"[SQL Warning] 插入 sessions 失败：{e}")
                return False
        return True

    # =========================================================================
    # 视觉接口
    # =========================================================================
    def process_video_frame(self, frame_rgb: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """处理视频帧（面部）"""
        if not self.face_pipe:
            print("[Face] 面部分析模块未初始化")
            return False

        try:
            # 检查输入帧的有效性
            if frame_rgb is None or len(frame_rgb) == 0:
                print("[Face] 输入帧为空")
                return False

            if not isinstance(frame_rgb, np.ndarray):
                print(f"[Face] 输入帧类型错误: {type(frame_rgb)}")
                return False

            # 记录会话元数据
            if metadata:
                self._ensure_session_logged(metadata)

            # 处理视频帧
            result = self.face_pipe.process_frame(frame_rgb)

            if result and isinstance(result, tuple) and len(result) >= 3:
                face_dict = result[2]
                if face_dict:
                    # 添加会话状态信息
                    face_dict['session_state'] = self.state.value if hasattr(self, 'state') else "unknown"

                    # 记录到文件日志
                    self.file_logger.log(self.session_id, "face", "face_expression", face_dict)

                    # 记录到数据库
                    if self.sql_logger.conn:
                        timestamp = face_dict.get("timestamp", time.time())
                        self.sql_logger.log_face(self.session_id, timestamp, face_dict)

                    return True
                else:
                    print("[Face] 未检测到面部")
                    return False
            else:
                print("[Face] 处理结果格式错误")
                return False
        except Exception as e:
            print(f"[Face Error] 处理视频帧失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_gesture_landmarks(self, hand_lms, pose_lms, metadata: Optional[Dict] = None) -> bool:
        """处理手势关键点"""
        if not self.gesture_pipe:
            print("[Gesture] 手势分析模块未初始化")
            return False

        try:
            # 记录会话元数据
            if metadata:
                self._ensure_session_logged(metadata)

            # 处理手势关键点
            gesture_dict = self.gesture_pipe.process(hand_lms, pose_lms)

            if gesture_dict:
                # 添加会话状态信息
                gesture_dict['session_state'] = self.state.value if hasattr(self, 'state') else "unknown"

                # 记录到文件日志
                self.file_logger.log(self.session_id, "gesture", "gesture_analysis", gesture_dict)

                # 记录到数据库
                if self.sql_logger.conn:
                    timestamp = gesture_dict.get("timestamp", time.time())
                    self.sql_logger.log_gesture(self.session_id, timestamp, gesture_dict)

                return True
            else:
                print("[Gesture] 未检测到手势")
                return False
        except Exception as e:
            print(f"[Gesture Error] 处理手势关键点失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # 语音接口
    # =========================================================================
    def _normalize_audio(self, audio_input: Union[np.ndarray, bytes, list]) -> Optional[np.ndarray]:
        """音频格式标准化"""
        try:
            if isinstance(audio_input, np.ndarray):
                data = audio_input
            elif isinstance(audio_input, bytes):
                data = np.frombuffer(audio_input, dtype=np.int16)
            elif isinstance(audio_input, list):
                data = np.array(audio_input)
            else:
                print(f"[Audio Norm Error] 不支持的音频类型: {type(audio_input)}")
                return None

            # 检查音频数据是否为空
            if len(data) == 0:
                print("[Audio Norm Error] 音频数据为空")
                return None

            # 数据类型转换和归一化
            if data.dtype == np.int16:
                return data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                return data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.float32 or data.dtype == np.float64:
                # 已经是浮点数，检查范围
                max_val = np.max(np.abs(data))
                if max_val > 1.0:
                    print(f"[Audio Norm Warning] 音频超出范围，最大值: {max_val}，将进行归一化")
                    return data / max_val
                return data.astype(np.float32)
            else:
                print(f"[Audio Norm Warning] 未知的数据类型: {data.dtype}")
                return data.astype(np.float32)
        except Exception as e:
            print(f"[Audio Norm Error] {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_voice_input(self, audio_or_text: Union[str, np.ndarray, bytes]) -> bool:
        """语音输入处理（ASR + 文本评估）"""
        if not self.voice_assess or not self.speech_recog:
            print("[Voice] 语音评估或识别模块未初始化")
            return False

        try:
            # 处理文本输入
            if isinstance(audio_or_text, str):
                text = audio_or_text.strip()
                if text:
                    self.voice_assess.add_answer(text)
                    self.file_logger.log(
                        self.session_id, "voice", "voice_interaction",
                        {
                            "type": "text",
                            "content": text,
                            "timestamp": time.time(),
                            "session_state": self.state.value if hasattr(self, 'state') else "unknown"
                        }
                    )
                    print(f"📝 [Text] {text[:30]}...")
                    return True
                return False

            # 处理音频输入
            clean_audio = self._normalize_audio(audio_or_text)
            if clean_audio is None:
                print("[Voice] 音频标准化失败")
                return False

            if len(clean_audio) < 1000:
                print(f"[Voice] 音频数据过短: {len(clean_audio)}")
                return False

            # 语音识别
            result = self.speech_recog.recognize_from_audio(clean_audio)
            text = ""
            confidence = 1.0

            if hasattr(result, 'text'):
                text = result.text
                confidence = getattr(result, 'confidence', 1.0)
            elif isinstance(result, dict):
                text = result.get('text', '')
                confidence = result.get('confidence', 1.0)

            text = str(text).strip()
            if text:
                self.voice_assess.add_answer(text)
                self.file_logger.log(
                    self.session_id, "voice", "voice_interaction",
                    {
                        "type": "asr",
                        "text": text,
                        "confidence": confidence,
                        "timestamp": time.time(),
                        "session_state": self.state.value if hasattr(self, 'state') else "unknown"
                    }
                )
                print(f"🎤 [ASR] {text[:30]}... (置信度: {confidence:.2f})")
                return True
            else:
                print("[Voice] 未识别到有效文本")
                return False
        except Exception as e:
            print(f"[Voice Error] 处理语音输入失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_voice_audio(self, audio_data: Union[np.ndarray, bytes]) -> bool:
        """语音韵律分析"""
        if not self.voice_process:
            print("[Voice] 语音处理模块未初始化")
            return False

        try:
            clean_audio = self._normalize_audio(audio_data)
            if clean_audio is None:
                print("[Voice] 音频标准化失败")
                return False

            if len(clean_audio) < 1000:
                print(f"[Voice] 音频数据过短: {len(clean_audio)}")
                return False

            result = self.voice_process.process_audio(clean_audio, extract_features=True, analyze=True)

            log_entry = {
                "type": "prosody",
                "timestamp": time.time(),
                "session_state": self.state.value if hasattr(self, 'state') else "unknown"
            }

            # 提取特征
            if hasattr(result, 'features') and result.features:
                feat = result.features
                log_entry['features'] = {
                    'pitch_mean': getattr(feat, 'pitch_mean', 0),
                    'pitch_std': getattr(feat, 'pitch_std', 0),
                    'energy_mean': getattr(feat, 'energy_mean', 0),
                    'energy_std': getattr(feat, 'energy_std', 0),
                    'speech_ratio': getattr(feat, 'speech_ratio', 0),
                    'duration_sec': getattr(feat, 'duration_sec', 0)
                }

            # 提取分析结果
            if hasattr(result, 'analysis') and result.analysis:
                anal = result.analysis
                log_entry['analysis'] = {
                    'feedback': getattr(anal, 'feedback', ''),
                    'is_valid': getattr(anal, 'is_valid', False),
                    'overall_score': getattr(anal, 'overall_score', 0)
                }

            # 记录日志
            self.file_logger.log(self.session_id, "voice", "voice_interaction", log_entry)

            # 如果有SQL连接，也记录到数据库
            if self.sql_logger.conn:
                try:
                    # 将特征转换为数据库格式
                    db_data = {}
                    if 'features' in log_entry:
                        db_data.update(log_entry['features'])
                    if 'analysis' in log_entry:
                        db_data['emotion'] = log_entry['analysis'].get('feedback', '')
                        db_data['feedback'] = log_entry['analysis'].get('feedback', '')
                        db_data['is_valid'] = log_entry['analysis'].get('is_valid', True)

                    self.sql_logger.log_voice(
                        self.session_id,
                        log_entry['timestamp'],
                        db_data
                    )
                except Exception as e:
                    print(f"[SQL Voice Log Error] {e}")

            # 打印反馈
            if hasattr(result, 'analysis') and hasattr(result.analysis, 'feedback'):
                print(f"📊 [Prosody] {result.analysis.feedback}")

            return True
        except Exception as e:
            print(f"[Prosody Error] 语音韵律分析失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # 报告与清理
    # =========================================================================
    def get_assessment_report(self) -> Optional[str]:
        """获取综合评估报告"""
        if not self.voice_assess:
            return "语音模块未启用"
        try:
            return self.voice_assess.get_comprehensive_evaluation()
        except Exception as e:
            return f"生成报告失败：{e}"

    def save_assessment_log(self) -> Optional[str]:
        """保存评估日志"""
        if not self.voice_assess:
            return None
        try:
            path = self.voice_assess.save_log()
            print(f"💾 评估日志已保存：{path}")
            return path
        except Exception as e:
            print(f"[Save Error] {e}")
            return None

    def cleanup(self):
        """清理资源"""
        if self.sql_logger:
            self.sql_logger.close_all()
        if self.file_logger:
            self.file_logger.close()
        print("🧹 资源已清理")