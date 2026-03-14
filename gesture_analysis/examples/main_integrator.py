"""
镜心项目 - 多模态视频流整合分析器 (样式字典修复版)
修复：
1. 解决 iris_style.color 报错问题（样式是字典，需用 ['color'] 访问）。
2. 确保面部网格和虹膜在镜像画面中正确绘制。
"""

import cv2
import time
import threading
import queue
import sys
import signal
import math
from pathlib import Path

# === 1. 路径配置 ===
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# === 2. 导入模块 ===
try:
    from face_expression.pipeline.video_pipeline import VideoPipeline as FaceAUAnalyzer
    print("✓ 面部模块加载成功")
except ImportError as e:
    print(f"❌ 面部模块导入失败：{e}")
    sys.exit(1)

try:
    from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer
    from gesture_analysis.core.analysis.shoulder_analyzer import ShoulderAnalyzer
    from gesture_analysis.core.analysis.arm_analyzer import ArmAnalyzer
    from gesture_analysis.core.analysis.upper_body_analyzer import UpperBodyAnalyzer
    from gesture_analysis.core.analysis.emotion_inferencer import EmotionInferencer
    from gesture_analysis.utils import Visualizer, GestureLogger
    from gesture_analysis.config import MEDIAPIPE_CONFIG
    import mediapipe as mp
    print("✓ 肢体模块加载成功")
except ImportError as e:
    print(f"❌ 肢体模块导入失败：{e}")
    sys.exit(1)

# === 3. 全局配置 ===
SESSION_ID = time.strftime("%Y%m%d_%H%M%S")
MAX_RUNTIME_MINUTES = 10
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

face_queue = queue.Queue(maxsize=2)
gesture_queue = queue.Queue(maxsize=2)

data_lock = threading.Lock()
latest_face_data = {"features": None, "mp_results": None}
latest_gesture_data = {
    "hand": {}, "pose_angles": {}, "finger_angles": {},
    "emotion": {}, "landmarks": {"hands": None, "pose": None},
    "raw_results": {}
}

stop_event = threading.Event()
start_time = time.time()

# === 4. 肢体分析核心类 ===
class FullGestureAnalyzer:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(**MEDIAPIPE_CONFIG['hands'])
        self.pose = mp.solutions.pose.Pose(**MEDIAPIPE_CONFIG['pose'])

        self.left_hand_analyzer = HandAnalyzer(hand_id=0)
        self.right_hand_analyzer = HandAnalyzer(hand_id=1)
        self.shoulder_analyzer = ShoulderAnalyzer()
        self.left_arm_analyzer = ArmAnalyzer(arm_id='left')
        self.right_arm_analyzer = ArmAnalyzer(arm_id='right')
        self.upper_body_analyzer = UpperBodyAnalyzer()
        self.emotion_inferencer = EmotionInferencer()
        self.visualizer = Visualizer()
        self.logger = GestureLogger()
        self.is_calibrated = False

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        display_frame = cv2.flip(frame_bgr, 1)
        flipped_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        raw_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- 1. 手部 ---
        hand_results = self.hands.process(flipped_rgb)
        left_result = {"resilience_score": 50.0, "is_valid": False}
        right_result = {"resilience_score": 50.0, "is_valid": False}
        left_finger_angles = None
        right_finger_angles = None

        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for idx, (landmarks, handedness) in enumerate(zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)):
                if idx >= 2: break
                label = handedness.classification[0].label
                analyzer = self.left_hand_analyzer if label == "Left" else self.right_hand_analyzer
                analyzer.update(landmarks.landmark)
                res = analyzer.get_results()
                angles = self._calc_finger_angles(landmarks.landmark)

                if label == "Left":
                    left_result, left_finger_angles = res, angles
                else:
                    right_result, right_finger_angles = res, angles

        # --- 2. 姿态 ---
        pose_results = self.pose.process(raw_rgb)
        shoulder_result = {"shoulder_score": 50.0, "is_valid": False, "is_calibrated": self.is_calibrated}
        left_arm_result = {"arm_score": 50.0, "is_valid": False}
        right_arm_result = {"arm_score": 50.0, "is_valid": False}
        upper_body_result = {"head_score": 50.0, "torso_score": 50.0, "is_valid": False}

        angles_data = {k: None for k in ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
                                         "shoulder_tilt", "head_tilt", "head_pitch", "torso_tilt"]}

        if pose_results.pose_landmarks:
            lms = pose_results.pose_landmarks.landmark
            self.shoulder_analyzer.update(lms)
            self.left_arm_analyzer.update(lms)
            self.right_arm_analyzer.update(lms)
            self.upper_body_analyzer.update(lms)

            shoulder_result = self.shoulder_analyzer.get_results()
            left_arm_result = self.left_arm_analyzer.get_results()
            right_arm_result = self.right_arm_analyzer.get_results()
            upper_body_result = self.upper_body_analyzer.get_results()

            def get_lm(name):
                lm = lms[getattr(mp.solutions.pose.PoseLandmark, name)]
                return {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}

            try:
                ls, rs = get_lm("LEFT_SHOULDER"), get_lm("RIGHT_SHOULDER")
                le, re = get_lm("LEFT_ELBOW"), get_lm("RIGHT_ELBOW")
                lw, rw = get_lm("LEFT_WRIST"), get_lm("RIGHT_WRIST")
                lh, rh = get_lm("LEFT_HIP"), get_lm("RIGHT_HIP")
                lear, rear = get_lm("LEFT_EAR"), get_lm("RIGHT_EAR")
                nose = get_lm("NOSE")

                if all(p["visibility"]>0.6 for p in [ls, le, lw]): angles_data["left_elbow"] = self._calc_angle(ls, le, lw)
                if all(p["visibility"]>0.6 for p in [rs, re, rw]): angles_data["right_elbow"] = self._calc_angle(rs, re, rw)
                if all(p["visibility"]>0.6 for p in [lh, ls, le]): angles_data["left_shoulder"] = self._calc_angle(lh, ls, le)
                if all(p["visibility"]>0.6 for p in [rh, rs, re]): angles_data["right_shoulder"] = self._calc_angle(rh, rs, re)
                if all(p["visibility"]>0.6 for p in [ls, rs]): angles_data["shoulder_tilt"] = self._calc_angle({"x": ls["x"]-0.1, "y": ls["y"]}, ls, rs)
                if all(p["visibility"]>0.6 for p in [lear, rear]): angles_data["head_tilt"] = self._calc_angle({"x": lear["x"]-0.1, "y": lear["y"]}, lear, rear)

                if all(p["visibility"]>0.6 for p in [nose, lear, rear]):
                    ear_c = {"x": (lear["x"]+rear["x"])/2, "y": (lear["y"]+rear["y"])/2}
                    angles_data["head_pitch"] = abs(90 - self._calc_angle({"x": nose["x"], "y": nose["y"]-0.1}, nose, ear_c))

                if all(p["visibility"]>0.6 for p in [ls, rs, lh, rh]):
                    sc = {"x": (ls["x"]+rs["x"])/2, "y": (ls["y"]+rs["y"])/2}
                    hc = {"x": (lh["x"]+rh["x"])/2, "y": (lh["y"]+rh["y"])/2}
                    angles_data["torso_tilt"] = abs(90 - self._calc_angle({"x": sc["x"], "y": sc["y"]-0.1}, sc, hc))

                if not self.is_calibrated and shoulder_result.get('is_valid'):
                    self.is_calibrated = True
                    shoulder_result['is_calibrated'] = True
            except: pass

        emotion_result = self.emotion_inferencer.infer_emotion(left_result, shoulder_result, left_arm_result, right_arm_result)

        return {
            "hand": {"left": left_result, "right": right_result, "left_angles": left_finger_angles, "right_angles": right_finger_angles},
            "pose_angles": angles_data,
            "emotion": emotion_result,
            "landmarks": {"hands": hand_results.multi_hand_landmarks, "pose": pose_results.pose_landmarks},
            "raw_results": {
                "shoulder": shoulder_result, "left_arm": left_arm_result,
                "right_arm": right_arm_result, "upper_body": upper_body_result
            }
        }

    def _calc_angle(self, a, b, c):
        ba = [a['x'] - b['x'], a['y'] - b['y']]
        bc = [c['x'] - b['x'], c['y'] - b['y']]
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        norm_ba = math.hypot(*ba)
        norm_bc = math.hypot(*bc)
        if norm_ba == 0 or norm_bc == 0: return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / (norm_ba * norm_bc)))))

    def _calc_finger_angles(self, lms):
        joints = {'thumb': [2,3,4], 'index': [5,6,8], 'middle': [9,10,12], 'ring': [13,14,16], 'pinky': [17,18,20]}
        angles = {}
        for f, j in joints.items():
            a = {"x": lms[j[0]].x, "y": lms[j[0]].y}
            b = {"x": lms[j[1]].x, "y": lms[j[1]].y}
            c = {"x": lms[j[2]].x, "y": lms[j[2]].y}
            angles[f] = self._calc_angle(a,b,c)
        return angles

    def close(self):
        self.hands.close()
        self.pose.close()

# === 5. 线程函数 ===

def face_worker():
    analyzer = FaceAUAnalyzer(fps=30, session_id=SESSION_ID)
    while not stop_event.is_set():
        try:
            frame = face_queue.get(timeout=0.5)
            if frame is None: break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_obj, mp_results, features = analyzer.process_frame(rgb_frame)

            with data_lock:
                latest_face_data['features'] = features
                latest_face_data['mp_results'] = mp_results
            face_queue.task_done()
        except queue.Empty: continue
        except Exception as e:
            pass
    print("[Face] Thread exited.")

def gesture_worker():
    analyzer = FullGestureAnalyzer()
    while not stop_event.is_set():
        try:
            frame = gesture_queue.get(timeout=0.5)
            if frame is None: break

            data = analyzer.process(frame)

            with data_lock:
                latest_gesture_data.update({
                    "hand": data["hand"],
                    "pose_angles": data["pose_angles"],
                    "emotion": data["emotion"],
                    "landmarks": data["landmarks"],
                    "raw_results": data["raw_results"]
                })
            gesture_queue.task_done()
        except queue.Empty: continue
        except Exception as e:
            pass
    analyzer.close()
    print("[Gesture] Thread exited.")

# === 6. 统一绘图函数 (最终修复：字典访问样式) ===

def draw_everything(frame, face_feat, face_mp, gest_data):
    h, w = frame.shape[:2]
    # 镜像画面作为底图
    display_frame = cv2.flip(frame, 1)

    # ==========================================
    # 1. 绘制面部网格 (关键修复：字典访问样式)
    # ==========================================
    if face_mp and face_mp.multi_face_landmarks:
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        # 获取样式字典
        tesselation_style = mp_drawing_styles.get_default_face_mesh_tesselation_style()
        iris_style = mp_drawing_styles.get_default_face_mesh_iris_connections_style()

        # 安全提取颜色和粗细 (兼容 dict 或 object)
        def get_color(style):
            if isinstance(style, dict): return style.get('color', (255, 255, 255))
            return getattr(style, 'color', (255, 255, 255))

        def get_thickness(style):
            if isinstance(style, dict): return style.get('thickness', 1)
            return getattr(style, 'thickness', 1)

        tess_color = get_color(tesselation_style)
        tess_thick = get_thickness(tesselation_style)
        iris_color = get_color(iris_style)
        iris_thick = get_thickness(iris_style)

        for fl in face_mp.multi_face_landmarks:
            landmarks = fl.landmark

            # --- 绘制网格 (Tesselation) ---
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx = connection[0]
                end_idx = connection[1]

                pt1 = landmarks[start_idx]
                pt2 = landmarks[end_idx]

                # 【核心修复】翻转 X 坐标
                x1, y1 = int((1.0 - pt1.x) * w), int(pt1.y * h)
                x2, y2 = int((1.0 - pt2.x) * w), int(pt2.y * h)

                if pt1.z > -0.5 and pt2.z > -0.5:
                    cv2.line(display_frame, (x1, y1), (x2, y2), tess_color, tess_thick)

            # --- 绘制虹膜 (Iris) ---
            for connection in mp_face_mesh.FACEMESH_IRISES:
                start_idx = connection[0]
                end_idx = connection[1]

                pt1 = landmarks[start_idx]
                pt2 = landmarks[end_idx]

                x1, y1 = int((1.0 - pt1.x) * w), int(pt1.y * h)
                x2, y2 = int((1.0 - pt2.x) * w), int(pt2.y * h)

                cv2.line(display_frame, (x1, y1), (x2, y2), iris_color, iris_thick)

    # 2. 绘制肢体骨架
    if gest_data.get("landmarks", {}).get("pose"):
        mp_pose = mp.solutions.pose
        pose_lms = gest_data["landmarks"]["pose"].landmark

        connections = mp_pose.POSE_CONNECTIONS
        for conn in connections:
            s_idx, e_idx = conn
            start = pose_lms[s_idx]
            end = pose_lms[e_idx]
            if start.visibility > 0.5 and end.visibility > 0.5:
                sx, sy = int((1 - start.x) * w), int(start.y * h)
                ex, ey = int((1 - end.x) * w), int(end.y * h)
                cv2.line(display_frame, (sx, sy), (ex, ey), (255, 255, 255), 2)

        joint_map = {
            "LEFT_SHOULDER": (0, 255, 255), "RIGHT_SHOULDER": (0, 255, 255),
            "LEFT_ELBOW": (255, 0, 255), "RIGHT_ELBOW": (255, 0, 255),
            "LEFT_WRIST": (0, 165, 255), "RIGHT_WRIST": (0, 165, 255),
            "LEFT_HIP": (255, 255, 0), "RIGHT_HIP": (255, 255, 0),
            "NOSE": (255, 100, 100)
        }
        for name, color in joint_map.items():
            lm = pose_lms[getattr(mp_pose.PoseLandmark, name)]
            if lm.visibility > 0.6:
                px, py = int((1 - lm.x) * w), int(lm.y * h)
                cv2.circle(display_frame, (px, py), 8, color, -1)
                cv2.circle(display_frame, (px, py), 10, (255, 255, 255), 2)

    if gest_data.get("landmarks", {}).get("hands"):
        mp_drawing = mp.solutions.drawing_utils
        for hl in gest_data["landmarks"]["hands"]:
            mp_drawing.draw_landmarks(display_frame, hl, mp.solutions.hands.HAND_CONNECTIONS)

    visualizer = Visualizer()
    hand_info = gest_data.get("hand", {})
    angles = gest_data.get("pose_angles", {})
    emotion = gest_data.get("emotion", {})

    # ==========================================
    # 3. 绘制面部 AU 数据 (左侧固定区域 x=10)
    # ==========================================
    if face_feat:
        tension_info = face_feat.get('psychological_signals', {})
        tension_level = tension_info.get('tension_level', 'unknown')
        dominant_emotion = face_feat.get('dominant_emotion', 'unknown')
        confidence = face_feat.get('confidence', 0.0)
        emotion_text = f"{dominant_emotion} ({confidence:.2f})"

        lines = [
            f"Emotion: {emotion_text}",
            f"Tension: {tension_level.upper()}",
            f"AU1: {face_feat.get('au1_inner_brow_raise', 0):.2f}",
            f"AU2: {face_feat.get('au2_outer_brow_raise', 0):.2f}",
            f"AU4: {face_feat.get('au4_frown', 0):.2f}",
            f"AU6: {face_feat.get('au6_cheek_raise', 0):.2f}",
            f"AU7: {face_feat.get('au7_eye_squeeze', 0):.2f}",
            f"AU9: {face_feat.get('au9_nose_wrinkle', 0):.2f}",
            f"AU10: {face_feat.get('au10_upper_lip_raise', 0):.2f}",
            f"AU12: {face_feat.get('au12_smile', 0):.2f}",
            f"AU14: {face_feat.get('au14_dimpler', 0):.2f}",
            f"AU15: {face_feat.get('au15_mouth_down', 0):.2f}",
            f"AU20: {face_feat.get('au20_lip_stretcher', 0):.2f}",
            f"AU23: {face_feat.get('au23_lip_compression', 0):.2f}",
            f"AU25: {face_feat.get('au25_mouth_open', 0):.2f}",
            f"AU26: {face_feat.get('au26_jaw_drop', 0):.2f}"
        ]
        for i, line in enumerate(lines):
            cv2.putText(display_frame, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ==========================================
    # 4. 绘制肢体角度数据 (整体右移，避免重叠)
    # ==========================================

    # A. 手部角度 -> 移至 x=280
    y_left = 30
    finger_names = {'thumb': '拇指', 'index': '食指', 'middle': '中指', 'ring': '无名指', 'pinky': '小指'}
    x_hand = 280

    if hand_info.get("left_angles"):
        display_frame = visualizer.put_chinese_text(display_frame, "左手角度:", (x_hand, y_left), color=(0, 255, 0))
        y_left += 25
        for f, ang in hand_info["left_angles"].items():
            display_frame = visualizer.put_chinese_text(display_frame, f"  {finger_names[f]}: {ang:.1f}°", (x_hand, y_left), color=(0, 200, 0))
            y_left += 25

    y_left += 10
    if hand_info.get("right_angles"):
        display_frame = visualizer.put_chinese_text(display_frame, "右手角度:", (x_hand, y_left), color=(0, 255, 0))
        y_left += 25
        for f, ang in hand_info["right_angles"].items():
            display_frame = visualizer.put_chinese_text(display_frame, f"  {finger_names[f]}: {ang:.1f}°", (x_hand, y_left), color=(0, 200, 0))
            y_left += 25

    # B. 手臂角度 -> 移至 x=480
    y_mid_l = 30
    x_arm = 480
    display_frame = visualizer.put_chinese_text(display_frame, "手臂角度:", (x_arm, y_mid_l), color=(255, 200, 0))
    y_mid_l += 25
    arm_items = [("左肘", angles.get("left_elbow")), ("右肘", angles.get("right_elbow")),
                 ("左肩", angles.get("left_shoulder")), ("右肩", angles.get("right_shoulder"))]
    for name, val in arm_items:
        if val is not None:
            display_frame = visualizer.put_chinese_text(display_frame, f"  {name}: {val:.1f}°", (x_arm, y_mid_l), color=(255, 200, 0))
            y_mid_l += 25

    # C. 头/肩角度 -> 移至 x=680
    y_mid_r = 30
    x_head = 680
    display_frame = visualizer.put_chinese_text(display_frame, "头部角度:", (x_head, y_mid_r), color=(255, 100, 100))
    y_mid_r += 25
    if angles.get("head_tilt") is not None:
        display_frame = visualizer.put_chinese_text(display_frame, f"  倾斜：{angles['head_tilt']:.1f}°", (x_head, y_mid_r), color=(255, 100, 100))
        y_mid_r += 25
    if angles.get("head_pitch") is not None:
        display_frame = visualizer.put_chinese_text(display_frame, f"  俯仰：{angles['head_pitch']:.1f}°", (x_head, y_mid_r), color=(255, 100, 100))
        y_mid_r += 25

    y_mid_r += 10
    display_frame = visualizer.put_chinese_text(display_frame, "肩部角度:", (x_head, y_mid_r), color=(0, 255, 255))
    y_mid_r += 25
    if angles.get("shoulder_tilt") is not None:
        display_frame = visualizer.put_chinese_text(display_frame, f"  倾斜：{angles['shoulder_tilt']:.1f}°", (x_head, y_mid_r), color=(0, 255, 255))
        y_mid_r += 25

    # D. 躯干角度 -> 移至 x=900
    y_right = 30
    x_torso = 900
    display_frame = visualizer.put_chinese_text(display_frame, "躯干角度:", (x_torso, y_right), color=(255, 150, 150))
    y_right += 25
    if angles.get("torso_tilt") is not None:
        display_frame = visualizer.put_chinese_text(display_frame, f"  倾斜：{angles['torso_tilt']:.1f}°", (x_torso, y_right), color=(255, 150, 150))
        y_right += 25

    # E. 底部情绪结果
    if emotion:
        display_frame = visualizer.draw_emotion_result(display_frame, emotion, position=(10, h - 80))

    # F. 底部状态栏
    cv2.rectangle(display_frame, (0, h-25), (w, h), (0, 0, 0), -1)
    cv2.putText(display_frame, f"Session: {SESSION_ID} | Time: {int(time.time()-start_time)}s | Press 'q' to quit",
                (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return display_frame

# === 7. 主程序 ===

def main():
    global start_time
    start_time = time.time()

    print("="*60)
    print("🪞 镜心项目 - 样式字典修复版")
    print("="*60)
    print(f"Session: {SESSION_ID}")
    print("修复：AttributeError: 'dict' object has no attribute 'color'")
    print("="*60)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    t1 = threading.Thread(target=face_worker, daemon=True)
    t2 = threading.Thread(target=gesture_worker, daemon=True)
    t1.start()
    t2.start()

    try:
        while not stop_event.is_set():
            if time.time() - start_time > MAX_RUNTIME_MINUTES * 60:
                break

            ret, frame = cap.read()
            if not ret: break

            if face_queue.full(): face_queue.get_nowait()
            if gesture_queue.full(): gesture_queue.get_nowait()
            face_queue.put(frame.copy())
            gesture_queue.put(frame.copy())

            with data_lock:
                f_feat = latest_face_data['features']
                f_mp = latest_face_data['mp_results']
                g_data = latest_gesture_data.copy()

            out_frame = draw_everything(frame, f_feat, f_mp, g_data)
            cv2.imshow("JingXin Full Integration", out_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        face_queue.put(None)
        gesture_queue.put(None)
        t1.join()
        t2.join()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 退出完成")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: stop_event.set())
    main()