
"""
数据模型定义

定义手势分析中使用的数据模型和类型
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HandFeatures:
    """手部特征数据模型"""
    jitter: float
    fist_status: bool
    spread: float
    finger_positions: Dict[str, Tuple[float, float]]
    is_valid: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HandFeatures':
        """从字典创建HandFeatures实例"""
        return cls(
            jitter=float(data.get('jitter', 0.0)),
            fist_status=bool(data.get('fist_status', False)),
            spread=float(data.get('spread', 0.0)),
            finger_positions=data.get('finger_positions', {}),
            is_valid=bool(data.get('is_valid', False))
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'jitter': self.jitter,
            'fist_status': self.fist_status,
            'spread': self.spread,
            'finger_positions': self.finger_positions,
            'is_valid': self.is_valid
        }


@dataclass
class ArmFeatures:
    """手臂特征数据模型"""
    wrist_position: Tuple[float, float]
    elbow_position: Tuple[float, float]
    shoulder_position: Tuple[float, float]
    wrist_jitter: float
    elbow_jitter: float
    arm_angle: float
    is_valid: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArmFeatures':
        """从字典创建ArmFeatures实例"""
        return cls(
            wrist_position=tuple(data.get('wrist_position', (0.0, 0.0))),
            elbow_position=tuple(data.get('elbow_position', (0.0, 0.0))),
            shoulder_position=tuple(data.get('shoulder_position', (0.0, 0.0))),
            wrist_jitter=float(data.get('wrist_jitter', 0.0)),
            elbow_jitter=float(data.get('elbow_jitter', 0.0)),
            arm_angle=float(data.get('arm_angle', 0.0)),
            is_valid=bool(data.get('is_valid', False))
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'wrist_position': self.wrist_position,
            'elbow_position': self.elbow_position,
            'shoulder_position': self.shoulder_position,
            'wrist_jitter': self.wrist_jitter,
            'elbow_jitter': self.elbow_jitter,
            'arm_angle': self.arm_angle,
            'is_valid': self.is_valid
        }


@dataclass
class ShoulderFeatures:
    """肩部特征数据模型"""
    left_position: Tuple[float, float]
    right_position: Tuple[float, float]
    left_jitter: float
    right_jitter: float
    height_diff: float
    shrug_level: float
    is_valid: bool
    is_calibrated: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShoulderFeatures':
        """从字典创建ShoulderFeatures实例"""
        return cls(
            left_position=tuple(data.get('left_position', (0.0, 0.0))),
            right_position=tuple(data.get('right_position', (0.0, 0.0))),
            left_jitter=float(data.get('left_jitter', 0.0)),
            right_jitter=float(data.get('right_jitter', 0.0)),
            height_diff=float(data.get('height_diff', 0.0)),
            shrug_level=float(data.get('shrug_level', 0.0)),
            is_valid=bool(data.get('is_valid', False)),
            is_calibrated=bool(data.get('is_calibrated', False))
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'left_position': self.left_position,
            'right_position': self.right_position,
            'left_jitter': self.left_jitter,
            'right_jitter': self.right_jitter,
            'height_diff': self.height_diff,
            'shrug_level': self.shrug_level,
            'is_valid': self.is_valid,
            'is_calibrated': self.is_calibrated
        }


@dataclass
class UpperBodyFeatures:
    """上肢特征数据模型"""
    head_position: Tuple[float, float]
    torso_position: Tuple[float, float]
    head_jitter: float
    torso_jitter: float
    head_tilt: float
    torso_stability: float
    is_valid: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UpperBodyFeatures':
        """从字典创建UpperBodyFeatures实例"""
        return cls(
            head_position=tuple(data.get('head_position', (0.0, 0.0))),
            torso_position=tuple(data.get('torso_position', (0.0, 0.0))),
            head_jitter=float(data.get('head_jitter', 0.0)),
            torso_jitter=float(data.get('torso_jitter', 0.0)),
            head_tilt=float(data.get('head_tilt', 0.0)),
            torso_stability=float(data.get('torso_stability', 0.0)),
            is_valid=bool(data.get('is_valid', False))
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'head_position': self.head_position,
            'torso_position': self.torso_position,
            'head_jitter': self.head_jitter,
            'torso_jitter': self.torso_jitter,
            'head_tilt': self.head_tilt,
            'torso_stability': self.torso_stability,
            'is_valid': self.is_valid
        }


@dataclass
class EmotionResult:
    """情绪分析结果数据模型"""
    overall_score: float
    emotion_state: str
    emoji: str
    feedback: str
    color: Tuple[int, int, int]
    is_valid: bool
    used_features: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionResult':
        """从字典创建EmotionResult实例"""
        return cls(
            overall_score=float(data.get('overall_score', 50.0)),
            emotion_state=str(data.get('emotion_state', '中性')),
            emoji=str(data.get('emoji', '🟡')),
            feedback=str(data.get('feedback', '状态平稳，保持自然呼吸')),
            color=tuple(data.get('color', (255, 255, 0))),
            is_valid=bool(data.get('is_valid', False)),
            used_features=str(data.get('used_features', 'none'))
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'overall_score': self.overall_score,
            'emotion_state': self.emotion_state,
            'emoji': self.emoji,
            'feedback': self.feedback,
            'color': self.color,
            'is_valid': self.is_valid,
            'used_features': self.used_features
        }
