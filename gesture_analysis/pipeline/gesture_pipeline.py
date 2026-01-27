
"""
手势分析管道

提供完整的手势分析流程，包括特征提取、分析和情绪推断
"""

from typing import Dict, Any, Optional
from ..core.feature_extraction import (
    HandFeatureExtractor,
    ArmFeatureExtractor,
    ShoulderFeatureExtractor,
    UpperBodyFeatureExtractor
)
from ..core.analysis import (
    HandAnalyzer,
    ArmAnalyzer,
    ShoulderAnalyzer,
    UpperBodyAnalyzer,
    EmotionAnalyzer
)
from ..config import HAND_CONFIG, ARM_CONFIG, SHOULDER_CONFIG


class GesturePipeline:
    """手势分析管道"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化手势分析管道

        参数:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or {}

        # 初始化特征提取器
        self.left_hand_extractor = HandFeatureExtractor(
            hand_id=0,
            history_length=self.config.get('hand_history_length', HAND_CONFIG['history_length'])
        )
        self.right_hand_extractor = HandFeatureExtractor(
            hand_id=1,
            history_length=self.config.get('hand_history_length', HAND_CONFIG['history_length'])
        )
        self.left_arm_extractor = ArmFeatureExtractor(
            arm_id='left',
            history_length=self.config.get('arm_history_length', ARM_CONFIG['history_length'])
        )
        self.right_arm_extractor = ArmFeatureExtractor(
            arm_id='right',
            history_length=self.config.get('arm_history_length', ARM_CONFIG['history_length'])
        )
        self.shoulder_extractor = ShoulderFeatureExtractor(
            history_length=self.config.get('shoulder_history_length', SHOULDER_CONFIG['history_length'])
        )
        self.upper_body_extractor = UpperBodyFeatureExtractor(
            history_length=self.config.get('upper_body_history_length', ARM_CONFIG['history_length'])
        )

        # 初始化分析器
        self.left_hand_analyzer = HandAnalyzer(
            hand_id=0,
            config=self.config.get('hand_config')
        )
        self.right_hand_analyzer = HandAnalyzer(
            hand_id=1,
            config=self.config.get('hand_config')
        )
        self.left_arm_analyzer = ArmAnalyzer(
            arm_id='left',
            config=self.config.get('arm_config')
        )
        self.right_arm_analyzer = ArmAnalyzer(
            arm_id='right',
            config=self.config.get('arm_config')
        )
        self.shoulder_analyzer = ShoulderAnalyzer(
            config=self.config.get('shoulder_config')
        )
        self.upper_body_analyzer = UpperBodyAnalyzer(
            config=self.config.get('upper_body_config')
        )
        self.emotion_analyzer = EmotionAnalyzer(
            config=self.config.get('emotion_config')
        )

    def process(self, hand_landmarks, pose_landmarks) -> Dict[str, Any]:
        """
        处理输入的landmarks，返回完整的分析结果

        参数:
            hand_landmarks: MediaPipe手部landmarks
            pose_landmarks: MediaPipe姿态landmarks

        返回:
            包含所有分析结果的字典
        """
        results = {
            "hand": {
                "left": {"features": {}, "analysis": {}},
                "right": {"features": {}, "analysis": {}}
            },
            "arm": {
                "left": {"features": {}, "analysis": {}},
                "right": {"features": {}, "analysis": {}}
            },
            "shoulder": {"features": {}, "analysis": {}},
            "upper_body": {"features": {}, "analysis": {}},
            "emotion": {}
        }

        # 处理手部
        if hand_landmarks and hand_landmarks.multi_hand_landmarks and hand_landmarks.multi_handedness:
            for idx, (landmarks, handedness) in enumerate(
                    zip(hand_landmarks.multi_hand_landmarks, hand_landmarks.multi_handedness)
            ):
                if idx >= 2:
                    break
                label = handedness.classification[0].label
                if label == "Left":
                    features = self.left_hand_extractor.extract(landmarks.landmark)
                    analysis = self.left_hand_analyzer.analyze(features)
                    results["hand"]["left"]["features"] = features
                    results["hand"]["left"]["analysis"] = analysis
                else:
                    features = self.right_hand_extractor.extract(landmarks.landmark)
                    analysis = self.right_hand_analyzer.analyze(features)
                    results["hand"]["right"]["features"] = features
                    results["hand"]["right"]["analysis"] = analysis

        # 处理姿态
        if pose_landmarks and pose_landmarks.pose_landmarks:
            pose = pose_landmarks.pose_landmarks.landmark

            # 手臂
            left_arm_features = self.left_arm_extractor.extract(pose)
            left_arm_analysis = self.left_arm_analyzer.analyze(left_arm_features)
            results["arm"]["left"]["features"] = left_arm_features
            results["arm"]["left"]["analysis"] = left_arm_analysis

            right_arm_features = self.right_arm_extractor.extract(pose)
            right_arm_analysis = self.right_arm_analyzer.analyze(right_arm_features)
            results["arm"]["right"]["features"] = right_arm_features
            results["arm"]["right"]["analysis"] = right_arm_analysis

            # 肩部
            shoulder_features = self.shoulder_extractor.extract(pose)
            shoulder_analysis = self.shoulder_analyzer.analyze(shoulder_features)
            results["shoulder"]["features"] = shoulder_features
            results["shoulder"]["analysis"] = shoulder_analysis

            # 上肢
            upper_body_features = self.upper_body_extractor.extract(pose)
            upper_body_analysis = self.upper_body_analyzer.analyze(upper_body_features)
            results["upper_body"]["features"] = upper_body_features
            results["upper_body"]["analysis"] = upper_body_analysis

        # 情绪分析
        emotion_result = self.emotion_analyzer.analyze(
            hand_results=results["hand"]["left"]["analysis"] if results["hand"]["left"]["analysis"].get("is_valid") else None,
            shoulder_results=results["shoulder"]["analysis"],
            left_arm_results=results["arm"]["left"]["analysis"],
            right_arm_results=results["arm"]["right"]["analysis"]
        )
        results["emotion"] = emotion_result

        return results

    def reset(self) -> None:
        """重置所有提取器和分析器的状态"""
        self.left_hand_extractor.reset()
        self.right_hand_extractor.reset()
        self.left_arm_extractor.reset()
        self.right_arm_extractor.reset()
        self.shoulder_extractor.reset()
        self.upper_body_extractor.reset()

        self.left_hand_analyzer.reset()
        self.right_hand_analyzer.reset()
        self.left_arm_analyzer.reset()
        self.right_arm_analyzer.reset()
        self.shoulder_analyzer.reset()
        self.upper_body_analyzer.reset()

    def get_raw_features(self) -> Dict[str, Any]:
        """
        获取所有提取器的原始特征

        返回:
            包含所有原始特征的字典
        """
        return {
            "hand": {
                "left": self.left_hand_extractor.get_features(),
                "right": self.right_hand_extractor.get_features()
            },
            "arm": {
                "left": self.left_arm_extractor.get_features(),
                "right": self.right_arm_extractor.get_features()
            },
            "shoulder": self.shoulder_extractor.get_features(),
            "upper_body": self.upper_body_extractor.get_features()
        }

    def get_analysis_results(self) -> Dict[str, Any]:
        """
        获取所有分析器的分析结果

        返回:
            包含所有分析结果的字典
        """
        return {
            "hand": {
                "left": self.left_hand_analyzer.get_results(),
                "right": self.right_hand_analyzer.get_results()
            },
            "arm": {
                "left": self.left_arm_analyzer.get_results(),
                "right": self.right_arm_analyzer.get_results()
            },
            "shoulder": self.shoulder_analyzer.get_results(),
            "upper_body": self.upper_body_analyzer.get_results()
        }
