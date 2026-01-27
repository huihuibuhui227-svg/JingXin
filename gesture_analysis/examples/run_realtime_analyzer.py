"""
实时手势与姿态角度分析示例脚本

演示如何使用 gesture_analysis 模块进行实时视频流分析。
完全复用 analyzers / inference / utils 模块，避免重复代码。
"""

import cv2
import time
import signal
import sys
from datetime import datetime

try:
    from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer
    from gesture_analysis.core.analysis.shoulder_analyzer import ShoulderAnalyzer
    from gesture_analysis.core.analysis.arm_analyzer import ArmAnalyzer
    from gesture_analysis.core.analysis.upper_body_analyzer import UpperBodyAnalyzer
    from gesture_analysis.core.analysis.emotion_inferencer import EmotionInferencer
    from gesture_analysis.utils import Visualizer, GestureLogger
    from gesture_analysis.config import MEDIAPIPE_CONFIG
except ImportError as e:
    print("❌ 导入失败！请确保从项目根目录运行：")
    print("   python -m gesture_analysis.examples.run_realtime_analyzer")
    print(f"   错误详情: {e}")
    sys.exit(1)

import mediapipe as mp


class RealtimeAnalyzer:
    # 类变量，用于防止重复启动
    _instance = None
    _is_running = False

    def __init__(self, max_runtime_minutes=10):
        # 防止重复启动
        if RealtimeAnalyzer._is_running:
            raise RuntimeError("⚠️  程序已在运行中，请先关闭当前实例！")

        self.hands = None
        self.pose = None
        self.cap = None
        self.running = False

        # 设置最大运行时间（秒）
        self.max_runtime = max_runtime_minutes * 60
        self.start_time = None

        self.left_hand_analyzer = HandAnalyzer(hand_id=0)
        self.right_hand_analyzer = HandAnalyzer(hand_id=1)
        self.shoulder_analyzer = ShoulderAnalyzer()
        self.left_arm_analyzer = ArmAnalyzer(arm_id='left')
        self.right_arm_analyzer = ArmAnalyzer(arm_id='right')
        self.upper_body_analyzer = UpperBodyAnalyzer()
        self.emotion_inferencer = EmotionInferencer()

        self.visualizer = Visualizer()
        self.logger = GestureLogger()

        self._init_mediapipe()

    def _init_mediapipe(self):
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
        self.hands = mp_hands.Hands(**MEDIAPIPE_CONFIG['hands'])
        self.pose = mp_pose.Pose(**MEDIAPIPE_CONFIG['pose'])

    def _calculate_angle(self, a, b, c):
        """计算三个点形成的角度（度）"""
        import math
        ba = [a['x'] - b['x'], a['y'] - b['y']]
        bc = [c['x'] - b['x'], c['y'] - b['y']]
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        norm_ba = math.hypot(*ba)
        norm_bc = math.hypot(*bc)
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        cos_angle = dot / (norm_ba * norm_bc)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    def _calculate_finger_angles(self, landmarks):
        """计算手指关节角度"""
        finger_joints = {
            'thumb': [2, 3, 4],
            'index': [5, 6, 8],
            'middle': [9, 10, 12],
            'ring': [13, 14, 16],
            'pinky': [17, 18, 20]
        }

        angles = {}
        for finger, joints in finger_joints.items():
            a = {"x": landmarks[joints[0]].x, "y": landmarks[joints[0]].y}
            b = {"x": landmarks[joints[1]].x, "y": landmarks[joints[1]].y}
            c = {"x": landmarks[joints[2]].x, "y": landmarks[joints[2]].y}
            angles[finger] = self._calculate_angle(a, b, c)

        return angles

    def start(self):
        # 检查是否已在运行
        if RealtimeAnalyzer._is_running:
            print("⚠️  程序已在运行中，请先关闭当前实例！")
            return

        # 设置运行标记
        RealtimeAnalyzer._is_running = True
        self.start_time = time.time()
        
        print("=" * 60)
        print("实时手势与姿态角度分析系统")
        print("=" * 60)
        print(f"⏱️  最大运行时间: {self.max_runtime // 60} 分钟")
        print("ℹ️  启动后请保持自然坐姿1~2秒，系统将自动校准肩部基准")
        print("🛑 按 'q' 键或 Ctrl+C 退出程序\n")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            RealtimeAnalyzer._is_running = False
            raise RuntimeError("无法打开摄像头！")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        signal.signal(signal.SIGINT, self._signal_handler)
        self.running = True
        try:
            self._main_loop()
        finally:
            # 确保退出时清除运行标记
            RealtimeAnalyzer._is_running = False

    def _signal_handler(self, signum, frame):
        print("\n⚠️  收到中断信号，正在退出...")
        self.running = False

    def _main_loop(self):
        last_report_time = time.time()
        mp_pose = mp.solutions.pose

        while self.running and self.cap.isOpened():
            # 检查运行时间是否超过限制
            if self.start_time and (time.time() - self.start_time) > self.max_runtime:
                print(f"\n⏱️  已达到最大运行时间 {self.max_runtime // 60} 分钟，程序自动退出。")
                self.running = False
                break
            success, raw_frame = self.cap.read()
            if not success:
                continue

            # 显示帧：非镜像（他人视角）
            display_frame = cv2.flip(raw_frame, 1)
            h, w = display_frame.shape[:2]

            # === 1. 手部分析 ===
            flipped_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            hand_results_raw = self.hands.process(flipped_rgb)
            left_result = {"resilience_score": 50.0, "is_valid": False}
            right_result = {"resilience_score": 50.0, "is_valid": False}
            left_finger_angles = None
            right_finger_angles = None

            if hand_results_raw.multi_hand_landmarks and hand_results_raw.multi_handedness:
                for idx, (landmarks, handedness) in enumerate(
                        zip(hand_results_raw.multi_hand_landmarks, hand_results_raw.multi_handedness)
                ):
                    if idx >= 2:
                        break
                    label = handedness.classification[0].label
                    analyzer = self.left_hand_analyzer if label == "Left" else self.right_hand_analyzer
                    analyzer.update(landmarks.landmark)
                    if label == "Left":
                        left_result = analyzer.get_results()
                        left_finger_angles = self._calculate_finger_angles(landmarks.landmark)
                    else:
                        right_result = analyzer.get_results()
                        right_finger_angles = self._calculate_finger_angles(landmarks.landmark)

                    # 绘制手部
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing.draw_landmarks(display_frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # === 2. 姿态分析 ===
            shoulder_result = {"shoulder_score": 50.0, "is_valid": False, "is_calibrated": False}
            left_arm_result = {"arm_score": 50.0, "is_valid": False}
            right_arm_result = {"arm_score": 50.0, "is_valid": False}
            upper_body_result = {"head_score": 50.0, "torso_score": 50.0, "is_valid": False}
            raw_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            pose_results_raw = self.pose.process(raw_rgb)

            # 初始化角度变量
            shoulder_angle = None
            head_tilt_angle = None
            torso_angle = None
            left_elbow_angle = None
            right_elbow_angle = None
            left_shoulder_angle = None
            right_shoulder_angle = None

            if pose_results_raw.pose_landmarks:
                self.shoulder_analyzer.update(pose_results_raw.pose_landmarks.landmark)
                self.left_arm_analyzer.update(pose_results_raw.pose_landmarks.landmark)
                self.right_arm_analyzer.update(pose_results_raw.pose_landmarks.landmark)
                self.upper_body_analyzer.update(pose_results_raw.pose_landmarks.landmark)
                shoulder_result = self.shoulder_analyzer.get_results()
                left_arm_result = self.left_arm_analyzer.get_results()
                right_arm_result = self.right_arm_analyzer.get_results()
                upper_body_result = self.upper_body_analyzer.get_results()

                # 绘制姿态骨架
                mp_drawing = mp.solutions.drawing_utils
                connections = mp_pose.POSE_CONNECTIONS
                for connection in connections:
                    start_idx, end_idx = connection
                    start = pose_results_raw.pose_landmarks.landmark[start_idx]
                    end = pose_results_raw.pose_landmarks.landmark[end_idx]
                    if start.visibility > 0.5 and end.visibility > 0.5:
                        start_px = int((1 - start.x) * w)
                        start_py = int(start.y * h)
                        end_px = int((1 - end.x) * w)
                        end_py = int(end.y * h)
                        cv2.line(display_frame, (start_px, start_py), (end_px, end_py), (255, 255, 255), 1)

                # 提取关键点
                landmarks = pose_results_raw.pose_landmarks.landmark

                def get_lm_dict(name):
                    lm = landmarks[getattr(mp_pose.PoseLandmark, name)]
                    return {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}

                # 提取所有关键点
                left_shoulder = get_lm_dict("LEFT_SHOULDER")
                right_shoulder = get_lm_dict("RIGHT_SHOULDER")
                left_elbow = get_lm_dict("LEFT_ELBOW")
                right_elbow = get_lm_dict("RIGHT_ELBOW")
                left_wrist = get_lm_dict("LEFT_WRIST")
                right_wrist = get_lm_dict("RIGHT_WRIST")
                left_hip = get_lm_dict("LEFT_HIP")
                right_hip = get_lm_dict("RIGHT_HIP")
                left_ear = get_lm_dict("LEFT_EAR")
                right_ear = get_lm_dict("RIGHT_EAR")
                nose = get_lm_dict("NOSE")

                # 计算手臂角度（肘关节角度）
                left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist) if \
                    all(pt["visibility"] > 0.6 for pt in [left_shoulder, left_elbow, left_wrist]) else None
                right_elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist) if \
                    all(pt["visibility"] > 0.6 for pt in [right_shoulder, right_elbow, right_wrist]) else None

                # 计算肩部角度（肩关节角度）
                left_shoulder_angle = self._calculate_angle(left_hip, left_shoulder, left_elbow) if \
                    all(pt["visibility"] > 0.6 for pt in [left_hip, left_shoulder, left_elbow]) else None
                right_shoulder_angle = self._calculate_angle(right_hip, right_shoulder, right_elbow) if \
                    all(pt["visibility"] > 0.6 for pt in [right_hip, right_shoulder, right_elbow]) else None

                # 计算肩部倾斜角度（左右肩连线与水平线的夹角）
                if all(pt["visibility"] > 0.6 for pt in [left_shoulder, right_shoulder]):
                    shoulder_angle = self._calculate_angle(
                        {"x": left_shoulder["x"] - 0.1, "y": left_shoulder["y"]},
                        left_shoulder,
                        right_shoulder
                    )

                # 计算头部倾斜角度（左右耳连线与水平线的夹角）
                if all(pt["visibility"] > 0.6 for pt in [left_ear, right_ear]):
                    head_tilt_angle = self._calculate_angle(
                        {"x": left_ear["x"] - 0.1, "y": left_ear["y"]},
                        left_ear,
                        right_ear
                    )

                # 计算头部俯仰角度（鼻子到耳中点的连线与垂直线的夹角）
                if all(pt["visibility"] > 0.6 for pt in [nose, left_ear, right_ear]):
                    ear_center = {
                        "x": (left_ear["x"] + right_ear["x"]) / 2,
                        "y": (left_ear["y"] + right_ear["y"]) / 2
                    }
                    head_pitch_angle = abs(90 - self._calculate_angle(
                        {"x": nose["x"], "y": nose["y"] - 0.1},
                        nose,
                        ear_center
                    ))
                else:
                    head_pitch_angle = None

                # 计算躯干倾斜角度（肩部中点到臀部中点连线与垂直线的夹角）
                if all(pt["visibility"] > 0.6 for pt in [left_shoulder, right_shoulder, left_hip, right_hip]):
                    shoulder_center = {
                        "x": (left_shoulder["x"] + right_shoulder["x"]) / 2,
                        "y": (left_shoulder["y"] + right_shoulder["y"]) / 2
                    }
                    hip_center = {
                        "x": (left_hip["x"] + right_hip["x"]) / 2,
                        "y": (left_hip["y"] + right_hip["y"]) / 2
                    }
                    torso_angle = abs(90 - self._calculate_angle(
                        {"x": shoulder_center["x"], "y": shoulder_center["y"] - 0.1},
                        shoulder_center,
                        hip_center
                    ))

                # 绘制关键点
                joint_info = [
                    (left_shoulder, "左肩", (0, 255, 255)),
                    (right_shoulder, "右肩", (0, 255, 255)),
                    (left_elbow, "左肘", (255, 0, 255)),
                    (left_wrist, "左腕", (0, 165, 255)),
                    (right_elbow, "右肘", (255, 0, 255)),
                    (right_wrist, "右腕", (0, 165, 255)),
                    (left_hip, "左髋", (255, 255, 0)),
                    (right_hip, "右髋", (255, 255, 0)),
                    (nose, "鼻子", (255, 100, 100)),
                ]
                for pt, label_text, color in joint_info:
                    if pt["visibility"] > 0.6:
                        px = int((1 - pt["x"]) * w)
                        py = int(pt["y"] * h)
                        cv2.circle(display_frame, (px, py), 8, color, -1)
                        cv2.circle(display_frame, (px, py), 10, (255, 255, 255), 2)

            # === 3. 情绪推断 ===
            emotion_result = self.emotion_inferencer.infer_emotion(
                left_result, shoulder_result, left_arm_result, right_arm_result
            )

            # === 4. 可视化显示（分栏布局，避免重叠）===
            # 左侧栏：手部角度
            y_left = 30
            if left_finger_angles is not None:
                display_frame = self.visualizer.put_chinese_text(
                    display_frame, "左手角度:", (10, y_left), color=(0, 255, 0)
                )
                y_left += 25
                finger_names = {'thumb': '拇指', 'index': '食指', 'middle': '中指', 'ring': '无名指', 'pinky': '小指'}
                for finger, angle in left_finger_angles.items():
                    display_frame = self.visualizer.put_chinese_text(
                        display_frame, f"  {finger_names[finger]}: {angle:.1f}°", (10, y_left), color=(0, 200, 0)
                    )
                    y_left += 25

            y_left += 10
            if right_finger_angles is not None:
                display_frame = self.visualizer.put_chinese_text(
                    display_frame, "右手角度:", (10, y_left), color=(0, 255, 0)
                )
                y_left += 25
                for finger, angle in right_finger_angles.items():
                    display_frame = self.visualizer.put_chinese_text(
                        display_frame, f"  {finger_names[finger]}: {angle:.1f}°", (10, y_left), color=(0, 200, 0)
                    )
                    y_left += 25

            # 中左栏：手臂角度
            y_mid_left = 30
            display_frame = self.visualizer.put_chinese_text(
                display_frame, "手臂角度:", (200, y_mid_left), color=(255, 200, 0)
            )
            y_mid_left += 25
            if left_elbow_angle is not None:
                display_frame = self.visualizer.put_chinese_text(
                    display_frame, f"  左肘: {left_elbow_angle:.1f}°", (200, y_mid_left), color=(255, 200, 0)
                )
                y_mid_left += 25
            if right_elbow_angle is not None:
                display_frame = self.visualizer.put_chinese_text(
                    display_frame, f"  右肘: {right_elbow_angle:.1f}°", (200, y_mid_left), color=(255, 200, 0)
                )
                y_mid_left += 25
            if left_shoulder_angle is not None:
                display_frame = self.visualizer.put_chinese_text(
                    display_frame, f"  左肩: {left_shoulder_angle:.1f}°", (200, y_mid_left), color=(255, 200, 0)
                )
                y_mid_left += 25
            if right_shoulder_angle is not None:
                display_frame = self.visualizer.put_chinese_text(
                    display_frame, f"  右肩: {right_shoulder_angle:.1f}°", (200, y_mid_left), color=(255, 200, 0)
                )
                y_mid_left += 25

            # 中右栏：头部和肩部角度
            y_mid_right = 30
            display_frame = self.visualizer.put_chinese_text(
                display_frame, "头部角度:", (350, y_mid_right), color=(255, 100, 100)
            )
            y_mid_right += 25
            if head_tilt_angle is not None:
                display_frame = self.visualizer.put_chinese_text(
                    display_frame, f"  倾斜: {head_tilt_angle:.1f}°", (350, y_mid_right), color=(255, 100, 100)
                )
                y_mid_right += 25
            if head_pitch_angle is not None:
                display_frame = self.visualizer.put_chinese_text(
                    display_frame, f"  俯仰: {head_pitch_angle:.1f}°", (350, y_mid_right), color=(255, 100, 100)
                )
                y_mid_right += 25

            y_mid_right += 10
            display_frame = self.visualizer.put_chinese_text(
                display_frame, "肩部角度:", (350, y_mid_right), color=(0, 255, 255)
            )
            y_mid_right += 25
            if shoulder_angle is not None:
                display_frame = self.visualizer.put_chinese_text(
                    display_frame, f"  倾斜: {shoulder_angle:.1f}°", (350, y_mid_right), color=(0, 255, 255)
                )
                y_mid_right += 25

            # 右侧栏：躯干角度
            y_right = 30
            display_frame = self.visualizer.put_chinese_text(
                display_frame, "躯干角度:", (500, y_right), color=(255, 150, 150)
            )
            y_right += 25
            if torso_angle is not None:
                display_frame = self.visualizer.put_chinese_text(
                    display_frame, f"  倾斜: {torso_angle:.1f}°", (500, y_right), color=(255, 150, 150)
                )
                y_right += 25

            # 底部：情绪结果
            display_frame = self.visualizer.draw_emotion_result(display_frame, emotion_result, position=(10, 400))

            cv2.imshow('Gesture & Pose Angle Analyzer', display_frame)

            self.logger.log(
                left_result, right_result, shoulder_result,
                left_arm_result, right_arm_result, upper_body_result, emotion_result
            )

            if time.time() - last_report_time > 2:
                self._print_report(left_result, right_result, shoulder_result, emotion_result)
                last_report_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _print_report(self, left, right, shoulder, emotion):
        print("\n" + "="*60)
        print(f"📊 角度分析报告（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）")
        print("="*60)
        print(f"左手: {left['resilience_score']:.1f} | 右手: {right['resilience_score']:.1f}")
        print(f"肩部: {shoulder['shoulder_score']:.1f} {'(校准中)' if not shoulder['is_calibrated'] else ''}")
        print(f"综合情绪: {emotion['overall_score']:.1f}/100 → {emotion['emoji']} {emotion['emotion_state']}")
        print(f"建议: {emotion['feedback']}")
        print(f"数据来源: {emotion['used_features']} | 有效: {emotion['is_valid']}")
        print("="*60)

    def stop(self):
        self.running = False
        RealtimeAnalyzer._is_running = False  # 清除运行标记
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
