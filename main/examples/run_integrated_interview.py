# main/examples/run_integrated_interview.py
"""
JingXin 集成面试终端（交互式调试版）
- 启动时收集被测者信息（姓名、性别、出生日期）
- 提供三种运行模式：视频 / 语音 / 多模态
- 显示原始摄像头画面（无分析叠加）
- 所有分析通过 JingXinIntegrator 统一调度与日志记录
"""

import argparse
import cv2
import numpy as np
import time
import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# === 导入集成器 ===
try:
    from main.integrator import JingXinIntegrator
except ImportError as e:
    print(f"❌ 无法导入 JingXinIntegrator: {e}")
    sys.exit(1)

# === 初始化 MediaPipe ===
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    )
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False, min_detection_confidence=0.5
    )
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )
except ImportError:
    print("❌ MediaPipe 未安装，请运行: pip install mediapipe")
    sys.exit(1)


def collect_participant_info():
    """交互式收集被测者信息"""
    print("\n" + "="*50)
    print("📋 请填写被测者基本信息")
    print("="*50)

    while True:
        name = input("姓名: ").strip()
        if name: break
        print("⚠️ 姓名不能为空")

    while True:
        gender = input("性别 (男/女): ").strip()
        if gender in ["男", "女"]: break
        print("⚠️ 请输入 '男' 或 '女'")

    while True:
        birth_date = input("出生日期 (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(birth_date, "%Y-%m-%d")
            break
        except ValueError:
            print("⚠️ 日期格式错误，使用 2000-01-01")
            birth_date = "2000-01-01"
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{name}_{timestamp}"
    metadata = {
        "name": name,
        "gender": gender,
        "birth_date": birth_date,
        "start_time": datetime.now().isoformat(),
        "session_type": "integrated_interview"
    }

    log_dir = Path(project_root) / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_file = log_dir / f"{session_id}_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 会话ID: {session_id}")
    return session_id, metadata


def run_video_only(integrator, cap, metadata):
    print("\n🎥 模式: 仅视频分析（按 'q' 返回主菜单）")
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_landmarks = mp_face_mesh.process(frame_rgb)
            hand_landmarks = mp_hands.process(frame_rgb)
            pose_landmarks = mp_pose.process(frame_rgb)

            if frame_count == 0:
                integrator.process_video_frame(frame_rgb, metadata=metadata)
                integrator.process_gesture_landmarks(hand_landmarks, pose_landmarks, metadata=metadata)
            else:
                integrator.process_video_frame(frame_rgb)
                integrator.process_gesture_landmarks(hand_landmarks, pose_landmarks)

            cv2.imshow("JingXin - Video Only", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            frame_count += 1
    finally:
        cv2.destroyAllWindows()
    print(f"\n✅ 视频分析完成，共处理 {frame_count} 帧")


def run_multimodal(integrator, cap, metadata):
    print("\n🎬 模式: 多模态同步分析（按 'q' 返回主菜单）")
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_landmarks = mp_face_mesh.process(frame_rgb)
            hand_landmarks = mp_hands.process(frame_rgb)
            pose_landmarks = mp_pose.process(frame_rgb)

            if frame_count == 0:
                integrator.process_video_frame(frame_rgb, metadata=metadata)
                integrator.process_gesture_landmarks(hand_landmarks, pose_landmarks, metadata=metadata)
            else:
                integrator.process_video_frame(frame_rgb)
                integrator.process_gesture_landmarks(hand_landmarks, pose_landmarks)

            cv2.imshow("JingXin - Multimodal", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            frame_count += 1
    finally:
        cv2.destroyAllWindows()
    print(f"\n✅ 多模态分析完成，共处理 {frame_count} 帧")


def main():
    session_id, metadata = collect_participant_info()
    integrator = JingXinIntegrator(session_id=session_id)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    try:
        while True:
            print("\n" + "="*50)
            print("[1] 仅视频流分析")
            print("[2] 多模态同步分析")
            print("[0] 退出")
            print("="*50)

            choice = input("请选择: ").strip()
            if choice == "1":
                run_video_only(integrator, cap, metadata)
            elif choice == "2":
                run_multimodal(integrator, cap, metadata)
            elif choice == "0":
                break
    finally:
        cap.release()
        mp_hands.close()
        mp_pose.close()
        mp_face_mesh.close()


if __name__ == "__main__":
    main()