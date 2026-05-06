import os
import sys
import subprocess
import threading
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime

# ... existing code ...

app = Flask(__name__, static_folder='report_frontend/static', template_folder='templates')

CORS(app, resources={r"/*": {"origins": "*"}})

# ... existing code ...


# 配置输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'output')


# --- 辅助函数：安全地运行 Python 脚本 ---
# ... existing code ...

def run_script(module_name):
    """在后台线程运行脚本，避免阻塞网页"""
    try:
        cmd = [sys.executable, '-m', f'report_frontend.{module_name}']
        print(f"🚀 正在启动任务：{cmd}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            return {"status": "success", "message": "任务完成！", "logs": result.stdout}
        else:
            return {"status": "error", "message": "任务失败", "logs": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ... existing code ...

# --- 路由：主页 ---
@app.route('/')
def index():
    return render_template('dashboard.html')


# --- API: 获取 output 目录下的文件列表 (用于展示热力图等) ---
@app.route('/api/files/<folder_name>')
def get_files(folder_name):
    """列出 data/output/<folder_name> 下的图片文件"""
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    if not os.path.exists(folder_path):
        return jsonify([])

    files = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.html')):
            files.append({
                "name": f,
                "url": f"/output/{folder_name}/{f}"
            })
    # 按时间排序，最新的在前
    files.sort(key=lambda x: x['name'], reverse=True)
    return jsonify(files)


# --- API: 提供静态文件访问 (图片和报告) ---
@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# --- API: 触发各个模块运行 ---
@app.route('/api/run/<module>', methods=['POST'])
def trigger_module(module):
    """
    接收前端请求，运行对应的模块
    module 参数可以是：
    - feature_engine (面部+眼动)
    - gesture_analysis (肢体 - 需你自己封装一个运行脚本)
    - voice_assessment (语音 - 需你自己封装)
    - report_generator (生成报告)
    """

    # 映射前端传来的名字到实际的 python 模块名
    module_map = {
        "face": "feature_engine",  # 运行面部特征提取
        "gesture": "visualizer",  # 假设你有一个脚本专门画肢体图，或者复用 visualizer
        "voice": "run_interview_assessment_voice",  # 假设这是你的语音脚本
        "report": "report_generator"  # 运行报告生成
    }

    target_module = module_map.get(module)

    if not target_module:
        return jsonify({"status": "error", "message": "未知的模块"})

    # 在新线程中运行，防止网页卡死
    thread = threading.Thread(target=run_script, args=(target_module,))
    thread.start()

    return jsonify({"status": "started", "message": f"任务 [{module}] 已启动，请在后台查看日志或稍后刷新页面。"})


if __name__ == '__main__':
    print("=" * 50)
    print("🌐 JingXin 总控平台启动中...")
    print("📂 数据目录:", OUTPUT_DIR)
    print("🔗 访问地址：http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)