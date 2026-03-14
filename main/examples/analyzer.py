# main/examples/analyzer.py
"""
JingXin 数据分析端（Terminal 2）
- 核心 AI 分析、数据库写入
- 线程安全处理
"""
import sys
import time
import threading
import zmq
import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main.integrator import JingXinIntegrator

DATA_ADDR = "tcp://0.0.0.0:5555"
CONTROL_ADDR = "tcp://0.0.0.0:5556"


class AnalyzerService:
    def __init__(self):
        self.integrator = None
        self.is_processing = False
        self.mode = "multimodal"
        self.metadata = {}

        self.context = zmq.Context()
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.bind(DATA_ADDR)
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind(CONTROL_ADDR)

        self.workers = []

    def start_workers(self):
        t_video = threading.Thread(target=self.video_worker, daemon=True)
        t_audio = threading.Thread(target=self.audio_worker, daemon=True)
        self.workers.extend([t_video, t_audio])
        t_video.start()
        t_audio.start()
        print("✅ 工作线程已启动")

    def video_worker(self):
        print("🧵 [Video] 就绪")
        while True:
            try:
                if not self.is_processing:
                    time.sleep(0.1)
                    continue
                msg = self.data_socket.recv_pyobj(flags=zmq.NOBLOCK)
                if msg['type'] == 'video':
                    img_bytes = np.frombuffer(msg['data'], dtype=np.uint8)
                    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.integrator.process_video_frame(frame_rgb)
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"[Video Error] {e}")

    def audio_worker(self):
        print("🧵 [Audio] 就绪")
        audio_buffer = []
        last_process_time = 0

        while True:
            try:
                if not self.is_processing:
                    time.sleep(0.1)
                    continue
                msg = self.data_socket.recv_pyobj(flags=zmq.NOBLOCK)
                if msg['type'] == 'audio':
                    audio_buffer.append(msg['data'])
                    if time.time() - last_process_time > 2.0 and len(audio_buffer) > 0:
                        full_bytes = b''.join(audio_buffer)
                        audio_np = np.frombuffer(full_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        audio_buffer = []
                        if len(audio_np) > 1000 and self.mode in ["voice", "multimodal", "interview"]:
                            self.integrator.process_voice_input(audio_np)
                            self.integrator.analyze_voice_audio(audio_np)
                        last_process_time = time.time()
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                print(f"[Audio Error] {e}")

    def handle_control(self):
        print("📡 [Control] 监听指令...")
        while True:
            try:
                req = self.control_socket.recv_pyobj()
                cmd = req.get("cmd")
                resp = {"status": "OK"}

                if cmd == "PING":
                    resp = {"status": "OK", "msg": "PONG"}
                elif cmd == "START":
                    self.mode = req.get("mode", "multimodal")
                    self.metadata = req.get("metadata", {})
                    session_id = f"{self.metadata.get('name')}_{int(time.time())}"
                    self.integrator = JingXinIntegrator(session_id=session_id)
                    self.is_processing = True
                    print(f"🚀 会话开始：{session_id} | 模式：{self.mode}")
                elif cmd == "STOP":
                    self.is_processing = False
                    time.sleep(0.5)
                    report_text = self.integrator.get_assessment_report()
                    save_path = self.integrator.save_assessment_log()
                    resp = {"status": "OK", "text": report_text, "saved_path": save_path}
                    print("🛑 会话结束")
                elif cmd == "EXIT":
                    print("👋 分析端退出")
                    if self.integrator:
                        self.integrator.cleanup()
                    sys.exit(0)

                self.control_socket.send_pyobj(resp)
            except Exception as e:
                print(f"[Control Error] {e}")
                self.control_socket.send_pyobj({"status": "ERROR", "msg": str(e)})

    def run(self):
        self.start_workers()
        self.handle_control()


if __name__ == "__main__":
    print("🚀 JingXin Analyzer Starting...")
    service = AnalyzerService()
    service.run()