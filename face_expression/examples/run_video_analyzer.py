import cv2
import time
import csv
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 使用绝对导入
try:
    from face_expression.analyzers.face_au_analyzer import FaceAUAnalyzer
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已安装必要的依赖: pip install opencv-python-headless mediapipe")
    exit(1)

# 尝试导入 MediaPipe（可选）
mp_drawing = None
mp_drawing_styles = None
mp_face_mesh = None
try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    print("✓ MediaPipe 已加载")
except ImportError as e:
    print(f"⚠️ MediaPipe 导入失败: {e}")
    print("将运行在无 MediaPipe 模式下（仅显示基础信息）")


def main():
    """主函数：实时视频流AU分析"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头实际分辨率: {actual_w} x {actual_h}")

    if actual_w < 640 or actual_h < 480:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("⚠️ 分辨率过低，已回退到 640x480")

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    try:
        analyzer = FaceAUAnalyzer(fps=fps)
    except Exception as e:
        print(f"❌ 分析器初始化失败: {e}")
        return

    # 创建日志文件
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / 'data' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "face_au_log.csv"

    fieldnames = [
        "timestamp", "focus_score", "blink_rate_per_min", "au4_frown", "au12_eyebrow_raise",
        "au12_smile", "au9_nose_wrinkle", "au15_mouth_down", "au25_mouth_open", "eye_closed_sec", "emotion"
    ]

    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print("按 'q' 退出。数据将记录 10 分钟...")
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features, results, emotion = analyzer.process_frame(frame_rgb)

        annotated_frame = frame.copy()
        if results and results.multi_face_landmarks and mp_drawing is not None:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        # 在视频窗口上显示 AU 值（英文+数字）
        if features:
            lines = [
                f"Focus: {features['focus_score']}",
                f"Blink: {features['blink_rate_per_min']}/min",
                f"AU4_Frown: {features['au4_frown']}",
                f"AU12_Raise: {features['au12_eyebrow_raise']}",
                f"AU12_Smile: {features['au12_smile']}",
                f"AU9_Wrinkle: {features['au9_nose_wrinkle']}",
                f"AU15_Down: {features['au15_mouth_down']}",
                f"AU25_Open: {features['au25_mouth_open']}",
                f"EyeClosed: {features['eye_closed_sec']}s"
            ]
            for i, line in enumerate(lines):
                cv2.putText(annotated_frame, line, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Facial AU Analyzer (Press 'q' to quit)", annotated_frame)

        # 在控制台实时显示情绪
        if emotion != "无人脸":
            print(f"[{time.strftime('%H:%M:%S')}] 当前情绪: {emotion}")

        # 写入 CSV
        if features:
            features_copy = features.copy()  # 避免修改原始字典
            features_copy["emotion"] = emotion
            with open(log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(features_copy)

        # 运行 10 分钟后自动退出
        if time.time() - start_time > 600:  # 10 分钟 = 600 秒
            print("✅ 10 分钟数据采集完成，正在保存...")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"📊 数据已保存至 {log_path}")


if __name__ == "__main__":
    main()