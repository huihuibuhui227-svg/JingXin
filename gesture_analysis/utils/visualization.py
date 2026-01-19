"""
可视化工具模块

提供手势和肩部分析结果的可视化功能
支持中文显示、关键点绘制和情绪状态反馈。
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple, Any, Dict
from ..config import VISUALIZATION_CONFIG


class Visualizer:
    """可视化工具类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化可视化工具

        参数:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or VISUALIZATION_CONFIG.copy()
        self.font: Optional[ImageFont.FreeTypeFont] = None
        self._init_font()

    def _init_font(self) -> None:
        """初始化中文字体，失败时降级处理"""
        try:
            self.font = ImageFont.truetype(
                self.config['font_path'],
                self.config['font_size']
            )
        except (OSError, IOError):
            # 字体加载失败，保留为 None，后续使用 OpenCV 默认字体
            self.font = None

    def put_chinese_text(
        self,
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_size: Optional[int] = None,
        color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        在图片上绘制中文文本（支持 BGR 图像）

        参数:
            img: OpenCV BGR 图像 (H, W, 3)
            text: 要绘制的文本
            position: 文本左上角位置 (x, y)
            font_size: 字体大小（仅当使用 PIL 时生效）
            color: BGR 颜色元组

        返回:
            绘制文本后的图像
        """
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            print("⚠️ 输入图像格式无效，跳过文本绘制")
            return img

        # 如果有可用中文字体，使用 PIL
        if self.font is not None and self._is_valid_text(text):
            try:
                # 转换 BGR → RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                draw = ImageDraw.Draw(pil_img)

                # PIL 使用 RGB，color 是 BGR，需反转
                rgb_color = tuple(reversed(color))
                draw.text(position, text, font=self.font, fill=rgb_color)

                # 转回 BGR
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                return img_bgr
            except Exception as e:
                print(f"⚠️ PIL 绘图失败，降级使用 OpenCV: {e}")

        # 降级：使用 OpenCV 默认字体（不支持中文，但不会崩溃）
        cv2.putText(
            img,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA
        )
        return img

    def _is_valid_text(self, text: str) -> bool:
        """检查文本是否非空且可绘制"""
        return isinstance(text, str) and len(text.strip()) > 0

    def draw_hand_landmarks(
        self,
        img: np.ndarray,
        landmarks: Any,
        connections: Optional[Any] = None
    ) -> np.ndarray:
        """
        绘制手部关键点（调用 MediaPipe）
        """
        try:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands

            connections = connections or mp_hands.HAND_CONNECTIONS
            mp_drawing.draw_landmarks(
                img, landmarks, connections,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        except Exception as e:
            print(f"⚠️ 手部关键点绘制失败: {e}")
        return img

    def draw_pose_landmarks(
        self,
        img: np.ndarray,
        landmarks: Any,
        connections: Optional[Any] = None
    ) -> np.ndarray:
        """
        绘制姿态关键点（调用 MediaPipe）
        """
        try:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose

            connections = connections or mp_pose.POSE_CONNECTIONS
            mp_drawing.draw_landmarks(
                img, landmarks, connections,
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
            )
        except Exception as e:
            print(f"⚠️ 姿态关键点绘制失败: {e}")
        return img

    def draw_emotion_result(
        self,
        img: np.ndarray,
        emotion_result: Optional[Dict[str, Any]],
        position: Tuple[int, int] = (10, 200)
    ) -> np.ndarray:
        """
        绘制情绪评估结果

        参数:
            img: OpenCV 图像
            emotion_result: 情绪评估结果字典
            position: 起始绘制位置 (x, y)

        返回:
            绘制后的图像
        """
        if not isinstance(emotion_result, dict):
            return self.put_chinese_text(
                img, "情绪分析不可用", position, color=(0, 0, 255)
            )

        try:
            # 第一行：综合评分
            score = emotion_result.get('overall_score', 0.0)
            state = emotion_result.get('emotion_state', '未知')
            text1 = f"整体情绪: {score:.1f}/100 ({state})"
            img = self.put_chinese_text(
                img, text1, position,
                color=self.config['colors']['info']
            )

            # 第二行：反馈
            emoji = emotion_result.get('emoji', '')
            feedback = emotion_result.get('feedback', '无反馈')
            text2 = f"{emoji} {feedback}"
            color = emotion_result.get('color', self.config['colors']['text'])
            img = self.put_chinese_text(
                img, text2,
                (position[0], position[1] + 35),
                color=color
            )

            # 第三行：有效性提示（调试用）
            if not emotion_result.get('is_valid', True):
                img = self.put_chinese_text(
                    img, "⚠️ 数据可能不可靠",
                    (position[0], position[1] + 70),
                    color=(0, 165, 255)
                )

        except Exception as e:
            print(f"⚠️ 情绪结果绘制异常: {e}")
            img = self.put_chinese_text(
                img, "情绪结果显示错误", position, color=(0, 0, 255)
            )

        return img