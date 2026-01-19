"""
JingXin 多模态集成器

统一调用 face_expression, gesture_analysis, voice_interaction 三个子模块，
实现面部表情、手势姿态、语音内容的联合分析。
"""

import cv2
import time
import numpy as np
from typing import Dict, Any, Optional
import threading
import queue

# 正确的模块导入路径（基于你的完整目录结构）
try:
    # 面部分析 - 使用 face_au_analyzer.py
    from face_expression.analyzers.face_au_analyzer import FaceAUAnalyzer

    # 手势分析 - 使用 hand_analyzer.py 和 shoulder_analyzer.py
    from gesture_analysis.analyzers.hand_analyzer import HandAnalyzer
    from gesture_analysis.analyzers.shoulder_analyzer import ShoulderAnalyzer
    from gesture_analysis.utils.visualization import Visualizer as GestureVisualizer

    # 语音交互 - 使用 speech_recognizer.py 和 tts_engine.py
    from voice_interaction.analyzers.speech_recognizer import SpeechRecognizer
    from voice_interaction.analyzers.tts_engine import TTSEngine
    from voice_interaction.assessment.interview_assessment import InterviewAssessment

except ImportError as e:
    print("❌ 模块导入失败！请确保从项目根目录运行：")
    print("   python -m main.examples.run_integrated_interview")
    print(f"   错误详情: {e}")
    exit(1)


class JingXinIntegrator:
    """多模态集成器"""

    def __init__(self):
        # 视频分析组件
        self.cap = None
        self.mp_hands = None
        self.mp_pose = None
        self.face_analyzer = None
        self.hand_analyzer_left = None
        self.hand_analyzer_right = None
        self.shoulder_analyzer = None
        self.gesture_visualizer = None

        # 语音组件
        self.speech_recognizer = None
        self.tts_engine = None
        self.interview_assessment = None

        # 分析结果缓存
        self.current_results = {
            "face": {"is_valid": False},
            "hand": {"is_valid": False},
            "shoulder": {"is_valid": False, "is_calibrated": False},
            "emotion": {"overall_score": 50.0, "emotion_state": "中性", "emoji": "🟡", "feedback": "系统初始化中", "color": (255, 255, 0)},
            "voice": {"text": "", "is_valid": False}
        }

        # 状态标志
        self.running = False
        self.interview_started = False
        self.current_question_index = 0

        # 初始化所有组件
        self._init_components()

    def _init_components(self):
        """初始化所有分析器和工具"""
        try:
            # 初始化 MediaPipe 模型（只初始化一次）
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )

            # 初始化分析器
            self.face_analyzer = FaceAUAnalyzer()
            self.hand_analyzer_left = HandAnalyzer(hand_id=0)
            self.hand_analyzer_right = HandAnalyzer(hand_id=1)
            self.shoulder_analyzer = ShoulderAnalyzer()

            # 初始化可视化工具
            self.gesture_visualizer = GestureVisualizer()

            # 初始化语音组件
            self.speech_recognizer = SpeechRecognizer()
            self.tts_engine = TTSEngine()
            self.interview_assessment = InterviewAssessment()

            print("✅ 所有组件初始化成功")

        except Exception as e:
            raise RuntimeError(f"组件初始化失败: {e}")

    def start_interview_session(self):
        """启动面试会话"""
        print("=" * 60)
        print("JingXin 多模态面试评估系统")
        print("=" * 60)
        print("✅ 系统已启动，正在初始化摄像头...")

        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头！")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 启动开场白
        self._start_introduction()

        # 启动主循环
        self.running = True
        self._main_loop()

    def _start_introduction(self):
        """播放开场白（同时预适应视频流）"""
        print("" + "=" * 60)
        print("🎥 视频流预适应阶段")
        print("=" * 60)
        print("系统正在预热摄像头和分析器...")
        print("请保持自然坐姿，让系统校准肩部基准线")
        print("=" * 60 + "")

        # 预适应阶段：持续显示视频流，让用户适应
        warmup_frames = 0
        max_warmup_frames = 150  # 约5秒（30fps）

        while warmup_frames < max_warmup_frames:
            success, image = self.cap.read()
            if not success:
                continue

            # 镜像翻转
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 预分析（用于校准）
            self.analyze_frame(image_rgb, image)

            # 显示提示信息
            cv2.putText(image, "系统预热中...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"剩余: {(max_warmup_frames - warmup_frames) // 30}秒", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('JingXin Integrated Analyzer', image)

            warmup_frames += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("✅ 视频流预适应完成")

        # 播放开场白
        if self.tts_engine and self.tts_engine.is_available():
            self.tts_engine.speak("你好，欢迎参加JingXin多模态面试评估。")
            self.tts_engine.speak("我会同时分析您的面部表情、手势姿态和语音内容。")
            self.tts_engine.speak("请保持自然坐姿，系统将在5秒后开始提问。")
            time.sleep(5)
        self.interview_started = True

    def _main_loop(self):
        """主分析循环"""
        frame_count = 0
        last_report_time = time.time()

        while self.running and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                continue

            # 镜像翻转
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 分析当前帧
            results = self.analyze_frame(image_rgb, image)

            # 处理语音交互（如果面试已开始且还有问题）
            if self.interview_started and self.current_question_index < len(self.interview_assessment.questions):
                self._handle_voice_interaction()

            # 可视化结果
            image = self._visualize_all(image, results)

            # 显示图像
            cv2.imshow('JingXin Integrated Analyzer', image)

            # 终端报告（每2秒）
            current_time = time.time()
            if current_time - last_report_time > 2:
                self._print_report(results)
                last_report_time = current_time

            frame_count += 1

            # 退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def analyze_frame(self, image_rgb, image_bgr=None):
        """
        分析视频帧（同时进行面部和肢体分析）

        参数:
            image_rgb: RGB 格式的图像
            image_bgr: BGR 格式的图像（用于可视化，可选）

        返回:
            包含所有分析结果的字典
        """
        results = {
            "face": {"is_valid": False},
            "hand": {"is_valid": False},
            "shoulder": {"is_valid": False, "is_calibrated": False},
            "emotion": {"overall_score": 50.0, "emotion_state": "中性", "emoji": "🟡", "feedback": "分析中", "color": (255, 255, 0)}
        }

        try:
            # 面部分析
            if self.face_analyzer:
                features, mesh_results, emotion = self.face_analyzer.process_frame(image_rgb)
                face_result = {
                    "is_valid": features is not None,
                    "features": features if features else {},
                    "emotion": emotion,
                    "mesh_results": mesh_results
                }
                results["face"] = face_result
                self.current_results["face"] = face_result

            # 手势和肩部分析
            hand_result, shoulder_result = self._analyze_gesture(image_rgb, image_bgr)
            results["hand"] = hand_result
            results["shoulder"] = shoulder_result
            self.current_results["hand"] = hand_result
            self.current_results["shoulder"] = shoulder_result

            # 融合情绪评估
            emotion_result = self._fuse_emotion(face_result, hand_result, shoulder_result)
            results["emotion"] = emotion_result
            self.current_results["emotion"] = emotion_result

        except Exception as e:
            print(f"⚠️ 帧分析失败: {e}")

        return results

    def _analyze_gesture(self, image_rgb, image_bgr=None):
        """分析手势和肩部"""
        hand_result = {"resilience_score": 50.0, "is_valid": False}
        shoulder_result = {"shoulder_score": 50.0, "is_valid": False, "is_calibrated": False}

        try:
            # 手部检测
            hands_results = self.mp_hands.process(image_rgb)
            if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
                for idx, (landmarks, handedness) in enumerate(
                    zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness)
                ):
                    if idx >= 2:
                        break
                    label = handedness.classification[0].label
                    analyzer = self.hand_analyzer_left if label == "Left" else self.hand_analyzer_right
                    analyzer.update(landmarks.landmark)
                    if label == "Left":
                        hand_result = analyzer.get_results()
                    else:
                        hand_result = analyzer.get_results()

                    # 绘制手部关键点（如果提供了BGR图像）
                    if image_bgr is not None:
                        image_bgr = self.gesture_visualizer.draw_hand_landmarks(image_bgr, landmarks)

            # 肩部检测
            pose_results = self.mp_pose.process(image_rgb)
            if pose_results.pose_landmarks:
                self.shoulder_analyzer.update(pose_results.pose_landmarks.landmark)
                shoulder_result = self.shoulder_analyzer.get_results()
                # 绘制肩部关键点
                if image_bgr is not None:
                    image_bgr = self.gesture_visualizer.draw_pose_landmarks(image_bgr, pose_results.pose_landmarks)

        except Exception as e:
            print(f"⚠️ 手势分析失败: {e}")

        return hand_result, shoulder_result

    def _handle_voice_interaction(self):
        """处理语音交互"""
        try:
            # 获取下一个问题
            question = self.interview_assessment.get_next_question()
            if question is None:
                return

            print(f"\n问题 {self.current_question_index + 1}: {question}")
            if self.tts_engine and self.tts_engine.is_available():
                self.tts_engine.speak(question)
                self.tts_engine.speak("请开始回答。")

            # 获取回答
            answer = ""
            if self.speech_recognizer:
                answer = self.speech_recognizer.listen_for_speech()

            if not answer:
                answer = "[无有效回答]"

            # 记录回答
            self.interview_assessment.add_answer(answer)
            self.current_results["voice"] = {"text": answer, "is_valid": True}
            self.current_question_index += 1

        except Exception as e:
            print(f"⚠️ 语音交互失败: {e}")

    def _fuse_emotion(self, face_result, hand_result, shoulder_result):
        """融合多模态情绪评估"""
        # 获取各模块评分
        face_score = 50.0
        if face_result.get("is_valid", False):
            features = face_result.get("features", {})
            if features:
                # 使用专注度作为面部评分
                face_score = features.get("focus_score", 50.0) * 100

        hand_score = hand_result.get("resilience_score", 50.0) if hand_result.get("is_valid", False) else 50.0
        shoulder_score = shoulder_result.get("shoulder_score", 50.0) if shoulder_result.get("is_valid", False) else 50.0

        # 加权融合（你可以调整权重）
        overall_score = (
            face_score * 0.3 +
            hand_score * 0.4 +
            shoulder_score * 0.3
        )

        # 映射到情绪状态
        if overall_score >= 80:
            emotion_state = "非常放松"
            emoji = "🟢"
            color = (0, 255, 0)
        elif overall_score >= 65:
            emotion_state = "放松"
            emoji = "🟢"
            color = (0, 255, 0)
        elif overall_score >= 50:
            emotion_state = "中性"
            emoji = "🟡"
            color = (255, 255, 0)
        elif overall_score >= 35:
            emotion_state = "轻微紧张"
            emoji = "🟠"
            color = (255, 165, 0)
        else:
            emotion_state = "紧张"
            emoji = "🔴"
            color = (0, 0, 255)

        feedback = f"综合情绪评分: {overall_score:.1f}/100 → {emoji} {emotion_state}"

        return {
            "overall_score": float(np.clip(overall_score, 0, 100)),
            "emotion_state": emotion_state,
            "emoji": emoji,
            "feedback": feedback,
            "color": color
        }

    def _visualize_all(self, image, results):
        """可视化所有分析结果"""
        try:
            # 面部可视化
            if results["face"].get("is_valid", False):
                mesh_results = results["face"].get("mesh_results")
                if mesh_results and mesh_results.multi_face_landmarks:
                    try:
                        import mediapipe as mp
                        mp_drawing = mp.solutions.drawing_utils
                        mp_face_mesh = mp.solutions.face_mesh

                        # 绘制面部网格
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=mesh_results.multi_face_landmarks[0],
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(
                                color=(80, 110, 10), thickness=1, circle_radius=1
                            )
                        )
                    except Exception as e:
                        print(f"⚠️ 面部网格绘制失败: {e}")

                # 显示面部表情信息
                emotion = results["face"].get("emotion", "未知")
                features = results["face"].get("features", {})
                if features:
                    # 使用OpenCV绘制所有AU特征（与单独运行的面部模块一致）
                    lines = [
                        f"Emotion: {emotion}",
                        f"Focus: {features.get('focus_score', 0.5):.2f}",
                        f"Blink: {features.get('blink_rate_per_min', 0.0):.1f}/min",
                        f"AU4_Frown: {features.get('au4_frown', 0.0):.3f}",
                        f"AU12_Raise: {features.get('au12_eyebrow_raise', 0.0):.3f}",
                        f"AU12_Smile: {features.get('au12_smile', 0.0):.3f}",
                        f"AU9_Wrinkle: {features.get('au9_nose_wrinkle', 0.0):.3f}",
                        f"AU15_Down: {features.get('au15_mouth_down', 0.0):.3f}",
                        f"AU25_Open: {features.get('au25_mouth_open', 0.0):.3f}",
                        f"EyeClosed: {features.get('eye_closed_sec', 0.0):.1f}s"
                    ]
                    for i, line in enumerate(lines):
                        cv2.putText(image, line, (10, 30 + i * 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 手势/肩部可视化
            if self.gesture_visualizer:
                # 显示手势分数（右侧）
                if results["hand"].get("is_valid", False):
                    image = self.gesture_visualizer.put_chinese_text(
                        image, f"左手: {results['hand']['resilience_score']:.1f}", (400, 30), color=(0, 255, 0)
                    )
                    image = self.gesture_visualizer.put_chinese_text(
                        image, f"右手: {results['hand']['resilience_score']:.1f}", (400, 60), color=(0, 255, 0)
                    )

                # 显示肩部分数（右侧）
                if results["shoulder"].get("is_valid", False):
                    shoulder_text = f"肩部: {results['shoulder']['shoulder_score']:.1f}"
                    if not results["shoulder"].get("is_calibrated", False):
                        shoulder_text += " (校准中...)"
                    image = self.gesture_visualizer.put_chinese_text(
                        image, shoulder_text, (400, 90), color=(0, 255, 255)
                    )

                # 综合情绪（底部）
                image = self.gesture_visualizer.draw_emotion_result(image, results["emotion"], position=(10, 430))

        except Exception as e:
            print(f"⚠️ 可视化失败: {e}")

        return image

    def get_answer(self):
        """获取当前回答"""
        return self.current_results["voice"].get("text", "")

    def get_comprehensive_evaluation(self):
        """获取综合评估"""
        if self.interview_assessment:
            return self.interview_assessment.get_comprehensive_evaluation()
        return {"text": "评估不可用", "is_valid": False}

    def _print_report(self, results):
        """打印终端报告"""
        print("\n" + "="*60)
        print(f"📊 多模态分析报告 ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        print("="*60)
        print(f"面部表情: {'有效' if results['face'].get('is_valid', False) else '无效'}")
        print(f"手势分析: {'有效' if results['hand'].get('is_valid', False) else '无效'}")
        print(f"肩部分析: {'有效' if results['shoulder'].get('is_valid', False) else '无效'}")
        print(f"综合情绪: {results['emotion']['overall_score']:.1f}/100 → {results['emotion']['emoji']} {results['emotion']['emotion_state']}")
        print("="*60)

    def stop(self):
        """停止并清理资源"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.mp_hands:
            self.mp_hands.close()
        if self.mp_pose:
            self.mp_pose.close()
        if self.tts_engine:
            self.tts_engine.stop()
        cv2.destroyAllWindows()
        print("\n✅ 多模态系统已安全退出。")


def main():
    """主函数：运行多模态集成系统"""
    integrator = None
    try:
        integrator = JingXinIntegrator()
        integrator.start_interview_session()
    except Exception as e:
        print(f"❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if integrator:
            integrator.stop()


if __name__ == "__main__":
    main()