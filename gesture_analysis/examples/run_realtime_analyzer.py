"""
实时手势与肩部情绪评估示例脚本

演示如何使用 gesture_analysis 模块进行实时视频流分析。
完全复用 analyzers / inference / utils 模块，避免重复代码。
"""

import cv2
import time
import signal
import sys
from datetime import datetime

# 标准导入（假设从项目根目录运行：python -m gesture_analysis.examples.run_realtime_analyzer）
try:
    from gesture_analysis.analyzers import HandAnalyzer, ShoulderAnalyzer
    from gesture_analysis.inference import EmotionInferencer
    from gesture_analysis.utils import Visualizer, GestureLogger
    from gesture_analysis.config import MEDIAPIPE_CONFIG
except ImportError as e:
    print("❌ 导入失败！请确保从项目根目录运行：")
    print("   python -m gesture_analysis.examples.run_realtime_analyzer")
    print(f"   错误详情: {e}")
    sys.exit(1)

import mediapipe as mp


class RealtimeAnalyzer:
    """实时分析器封装类，便于资源管理和异常处理"""

    def __init__(self):
        self.hands = None
        self.pose = None
        self.cap = None
        self.running = False

        # 初始化分析器
        self.left_hand_analyzer = HandAnalyzer(hand_id=0)
        self.right_hand_analyzer = HandAnalyzer(hand_id=1)
        self.shoulder_analyzer = ShoulderAnalyzer()
        self.emotion_inferencer = EmotionInferencer()

        # 初始化工具
        self.visualizer = Visualizer()
        self.logger = GestureLogger()

        # 初始化 MediaPipe
        self._init_mediapipe()

    def _init_mediapipe(self):
        """初始化 MediaPipe 模型"""
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose

        self.hands = mp_hands.Hands(**MEDIAPIPE_CONFIG['hands'])
        self.pose = mp_pose.Pose(**MEDIAPIPE_CONFIG['pose'])

    def start(self):
        """启动实时分析"""
        print("=" * 60)
        print("实时手势与肩部情绪评估系统")
        print("=" * 60)
        print("ℹ️  启动后请保持自然坐姿1~2秒，系统将自动校准肩部基准")
        print("ℹ️  校准完成后，耸肩将被正确检测")
        print("🛑 按 'q' 键或 Ctrl+C 退出程序\n")

        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头！")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 注册信号处理器（支持 Ctrl+C）
        signal.signal(signal.SIGINT, self._signal_handler)

        self.running = True
        self._main_loop()

    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n⚠️  收到中断信号，正在退出...")
        self.running = False

    def _main_loop(self):
        """主分析循环"""
        frame_count = 0
        last_report_time = time.time()

        while self.running and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                continue

            # 镜像翻转 + 转 RGB
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # === 1. 手部分析 ===
            hand_results_raw = self.hands.process(image_rgb)
            left_result = {"resilience_score": 50.0, "is_valid": False}
            right_result = {"resilience_score": 50.0, "is_valid": False}

            if hand_results_raw.multi_hand_landmarks and hand_results_raw.multi_handedness:
                for idx, (landmarks, handedness) in enumerate(
                    zip(hand_results_raw.multi_hand_landmarks, hand_results_raw.multi_handedness)
                ):
                    if idx >= 2:  # 最多处理两只手
                        break
                    label = handedness.classification[0].label  # 'Left' or 'Right'
                    analyzer = self.left_hand_analyzer if label == "Left" else self.right_hand_analyzer
                    analyzer.update(landmarks.landmark)
                    if label == "Left":
                        left_result = analyzer.get_results()
                    else:
                        right_result = analyzer.get_results()

                    # 绘制关键点
                    image = self.visualizer.draw_hand_landmarks(image, landmarks)

            # === 2. 肩部分析 ===
            shoulder_result = {"shoulder_score": 50.0, "is_valid": False, "is_calibrated": False}
            pose_results_raw = self.pose.process(image_rgb)
            if pose_results_raw.pose_landmarks:
                self.shoulder_analyzer.update(pose_results_raw.pose_landmarks.landmark)
                shoulder_result = self.shoulder_analyzer.get_results()
                image = self.visualizer.draw_pose_landmarks(image, pose_results_raw.pose_landmarks)

            # === 3. 情绪推断 ===
            emotion_result = self.emotion_inferencer.infer_emotion(left_result, shoulder_result)

            # === 4. 可视化 ===
            image = self.visualizer.draw_emotion_result(image, emotion_result, position=(10, 200))

            # 显示左右手评分（仅当有效时）
            if left_result["is_valid"]:
                image = self.visualizer.put_chinese_text(
                    image, f"左手: {left_result['resilience_score']:.1f}", (10, 30), color=(0, 255, 0)
                )
            if right_result["is_valid"]:
                image = self.visualizer.put_chinese_text(
                    image, f"右手: {right_result['resilience_score']:.1f}", (400, 30), color=(0, 255, 0)
                )

            # 显示肩部状态
            shoulder_text = f"肩部: {shoulder_result['shoulder_score']:.1f}"
            if not shoulder_result["is_calibrated"]:
                shoulder_text += " (校准中...)"
            image = self.visualizer.put_chinese_text(
                image, shoulder_text, (10, 60), color=(0, 255, 255)
            )

            cv2.imshow('Gesture & Shoulder Emotion Analyzer', image)

            # === 5. 日志记录 ===
            self.logger.log(left_result, right_result, shoulder_result, emotion_result)

            # === 6. 终端报告（每2秒）===
            current_time = time.time()
            if current_time - last_report_time > 2:
                self._print_report(left_result, right_result, shoulder_result, emotion_result)
                last_report_time = current_time

            frame_count += 1

            # 退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _print_report(self, left, right, shoulder, emotion):
        """打印终端报告"""
        print("\n" + "="*60)
        print(f"📊 情绪评估报告（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）")
        print("="*60)
        print(f"左手: {left['resilience_score']:.1f} | 右手: {right['resilience_score']:.1f}")
        print(f"肩部: {shoulder['shoulder_score']:.1f} {'(校准中)' if not shoulder['is_calibrated'] else ''}")
        print(f"综合情绪: {emotion['overall_score']:.1f}/100 → {emotion['emoji']} {emotion['emotion_state']}")
        print(f"建议: {emotion['feedback']}")
        print(f"数据来源: {emotion['used_features']} | 有效: {emotion['is_valid']}")
        print("="*60)

    def stop(self):
        """停止并清理资源"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        if self.pose:
            self.pose.close()
        cv2.destroyAllWindows()
        print("\n✅ 程序已安全退出。")


def main():
    analyzer = None
    try:
        analyzer = RealtimeAnalyzer()
        analyzer.start()
    except Exception as e:
        print(f"❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if analyzer:
            analyzer.stop()


if __name__ == "__main__":
    main()