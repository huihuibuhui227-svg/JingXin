# main/examples/collector.py
"""
JingXin 数据采集端（Terminal 1）
- 用户交互、硬件采集、画面预览
- 通过 ZeroMQ 发送数据
"""
import cv2
import numpy as np
import time
import sys
import zmq
import threading
import pyaudio
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

DATA_ADDR = "tcp://127.0.0.1:5555"
CONTROL_ADDR = "tcp://127.0.0.1:5556"
AUDIO_RATE = 16000
CHUNK_SIZE = 1024


class DataCollector:
    def __init__(self):
        self.context = zmq.Context()
        self.data_socket = self.context.socket(zmq.PUSH)
        self.data_socket.connect(DATA_ADDR)
        self.control_socket = self.context.socket(zmq.REQ)
        self.control_socket.connect(CONTROL_ADDR)

        self.is_running = False
        self.audio_stream = None
        self.cap = None

    def collect_info(self):
        print("\n" + "=" * 50)
        print("📋 被测者信息")
        print("=" * 50)
        name = input("姓名：").strip() or "Anonymous"
        gender = input("性别 (男/女): ").strip() or "男"
        birth_date = input("出生日期 (YYYY-MM-DD): ").strip() or "2000-01-01"
        return {"name": name, "gender": gender, "birth_date": birth_date,
                "start_time": datetime.now().isoformat()}

    def show_menu(self):
        print("\n" + "=" * 50)
        print("[1] 仅视频  [2] 仅语音  [3] 多模态  [4] 完整面试  [0] 退出")
        print("=" * 50)
        return input("选择：").strip()

    def audio_thread(self):
        p = pyaudio.PyAudio()
        try:
            self.audio_stream = p.open(format=pyaudio.paInt16, channels=1,
                                       rate=AUDIO_RATE, input=True,
                                       frames_per_buffer=CHUNK_SIZE)
            print("🎤 麦克风已启动")
            while self.is_running:
                try:
                    data = self.audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    self.data_socket.send_pyobj({'type': 'audio', 'data': data, 'ts': time.time()})
                except:
                    break
        except Exception as e:
            print(f"❌ 音频错误：{e}")
        finally:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            p.terminate()

    def run_session(self, mode, metadata):
        self.is_running = True
        print("📡 启动分析端...")
        self.control_socket.send_pyobj({"cmd": "START", "mode": mode, "metadata": metadata})
        resp = self.control_socket.recv_pyobj()
        if resp.get("status") != "OK":
            print(f"❌ 启动失败：{resp.get('msg')}")
            return

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        t_audio = threading.Thread(target=self.audio_thread, daemon=True)
        t_audio.start()

        print("🎥 视频已启动 (按 'q' 结束)")

        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret: break
                _, img_bytes = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                self.data_socket.send_pyobj({'type': 'video', 'data': img_bytes.tobytes(), 'ts': time.time()})

                cv2.putText(frame, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("JingXin Collector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.is_running = False
            if self.cap: self.cap.release()
            cv2.destroyAllWindows()

            print("\n📡 获取报告...")
            self.control_socket.send_pyobj({"cmd": "STOP"})
            report = self.control_socket.recv_pyobj()

            if report:
                print("\n" + "=" * 50)
                print("📊 评估报告")
                print("=" * 50)
                print(report.get("text", "无报告"))
                if report.get("saved_path"):
                    print(f"💾 保存至：{report['saved_path']}")

    def run(self):
        try:
            self.control_socket.send_pyobj({"cmd": "PING"})
            self.control_socket.setsockopt(zmq.RCVTIMEO, 2000)
            resp = self.control_socket.recv_pyobj()
            print("✅ 已连接到分析端")
            self.control_socket.setsockopt(zmq.RCVTIMEO, -1)
        except:
            print("❌ 无法连接分析端，请先运行 analyzer.py")
            return

        metadata = self.collect_info()

        while True:
            choice = self.show_menu()
            if choice == "0":
                self.control_socket.send_pyobj({"cmd": "EXIT"})
                break
            elif choice in ["1", "2", "3", "4"]:
                mode_map = {"1": "video", "2": "voice", "3": "multimodal", "4": "interview"}
                self.run_session(mode_map[choice], metadata)
            else:
                print("⚠️ 无效选择")


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()