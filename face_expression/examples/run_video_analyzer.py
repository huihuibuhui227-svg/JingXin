import cv2
import time
import csv
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from face_expression.pipeline.video_pipeline import VideoPipeline as FaceAUAnalyzer
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头实际分辨率: {actual_w} x {actual_h}")

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    session_id = time.strftime("%Y%m%d_%H%M%S")
    try:
        analyzer = FaceAUAnalyzer(fps=fps, session_id=session_id)
    except Exception as e:
        print(f"❌ 分析器初始化失败: {e}")
        return

    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / 'data' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"face_au_log_{session_id}.csv"

    # === 完整字段列表（182+ 列，严格匹配 face_logs 表结构）===
    BASE_AU_FIELDS = [
        "au1_inner_brow_raise", "au2_outer_brow_raise", "au4_frown",
        "au6_cheek_raise", "au7_eye_squeeze", "au9_nose_wrinkle",
        "au10_upper_lip_raise", "au12_smile", "au14_dimpler",
        "au15_mouth_down", "au20_lip_stretcher", "au23_lip_compression",
        "au25_mouth_open", "au26_jaw_drop", "avg_ear"
    ]

    HEAD_GAZE_FIELDS = [
        "head_yaw", "head_pitch", "symmetry_score",
        "blink_rate_per_min", "eye_closed_sec", "is_blink",
        "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
        "gaze_direction_x", "gaze_direction_y", "gaze_deviation"
    ]

    TS_PREFIXES = BASE_AU_FIELDS + HEAD_GAZE_FIELDS
    TS_FIELDS = []
    for prefix in TS_PREFIXES:
        for suffix in ["_trend", "_volatility", "_change_rate"]:
            TS_FIELDS.append(prefix + suffix)

    KEY_FIELDS = [
        "session_id", "timestamp", "focus_score",
        *BASE_AU_FIELDS,
        *HEAD_GAZE_FIELDS,
        "tension_score", "tension_level",
        "tension_sources_brow_furrow",
        "tension_sources_lip_compression",
        "tension_sources_eye_closure",
        "tension_sources_expression_instability",
        "tension_sources_emotional_influence",
        "micro_exp_au_name", "micro_exp_intensity",
        "micro_exp_duration_frames", "micro_exp_onset_frame",
        "dominant_emotion", "confidence", "emotion_state", "is_valid"
    ] + TS_FIELDS

    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDS)
        writer.writeheader()

    print("按 'q' 退出。数据将记录最多 10 分钟...")
    start_time = time.time()

    # 尝试导入 MediaPipe
    mp_drawing = mp_drawing_styles = mp_face_mesh = None
    try:
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        print("✓ MediaPipe 已加载")
    except ImportError:
        print("⚠️ MediaPipe 未安装")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_obj, results, features = analyzer.process_frame(frame_rgb)

        annotated_frame = frame.copy()
        if results and results.multi_face_landmarks and mp_drawing is not None:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        # === 实时显示全部 AU 特征（完整 16 行）===
        if features:
            tension_info = features.get('psychological_signals', {})
            tension_level = tension_info.get('tension_level', 'unknown')
            dominant_emotion = features.get('dominant_emotion', 'unknown')
            confidence = features.get('confidence', 0.0)
            emotion_text = f"{dominant_emotion} ({confidence:.2f})"

            lines = [
                f"Emotion: {emotion_text}",
                f"Tension: {tension_level.upper()}",
                f"AU1: {features.get('au1_inner_brow_raise', 0):.2f}",
                f"AU2: {features.get('au2_outer_brow_raise', 0):.2f}",
                f"AU4: {features.get('au4_frown', 0):.2f}",
                f"AU6: {features.get('au6_cheek_raise', 0):.2f}",
                f"AU7: {features.get('au7_eye_squeeze', 0):.2f}",
                f"AU9: {features.get('au9_nose_wrinkle', 0):.2f}",
                f"AU10: {features.get('au10_upper_lip_raise', 0):.2f}",
                f"AU12: {features.get('au12_smile', 0):.2f}",
                f"AU14: {features.get('au14_dimpler', 0):.2f}",
                f"AU15: {features.get('au15_mouth_down', 0):.2f}",
                f"AU20: {features.get('au20_lip_stretcher', 0):.2f}",
                f"AU23: {features.get('au23_lip_compression', 0):.2f}",
                f"AU25: {features.get('au25_mouth_open', 0):.2f}",
                f"AU26: {features.get('au26_jaw_drop', 0):.2f}"
            ]
            for i, line in enumerate(lines):
                cv2.putText(annotated_frame, line, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Facial AU & Emotion Analyzer", annotated_frame)

        # === 日志输出 ===
        if features:
            current_time_str = time.strftime('%H:%M:%S')
            tension_level = features.get('tension_level', 'low')
            dominant_emotion = features.get('dominant_emotion', 'unknown')
            confidence = features.get('confidence', 0.0)

            print(f"[{current_time_str}] 情绪: {dominant_emotion} "
                  f"(置信度: {confidence:.2f}), "
                  f"紧张度: {tension_level.upper()}, "
                  f"Tension Score: {features.get('tension_score', 0):.3f}")

            # 构建完整日志行（使用 features 字典，它来自 result.to_dict()）
            row = {}
            for key in KEY_FIELDS:
                row[key] = features.get(key, "")

            # 安全写入 CSV
            try:
                with open(log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=KEY_FIELDS)
                    writer.writerow(row)
            except Exception as e:
                print(f"⚠️ 写入日志失败: {e}")

        # 退出条件
        if time.time() - start_time > 600 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"📊 数据已保存至: {log_path}")


if __name__ == "__main__":
    main()