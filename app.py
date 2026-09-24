import os
import sys
import subprocess
import threading
import uuid
import time
import logging
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='report_frontend/static', template_folder='templates')

# CORS 配置：通过环境变量 CORS_ORIGINS 设置允许的来源，默认仅允许本地
cors_origins = os.getenv('CORS_ORIGINS', 'http://127.0.0.1:5000,http://localhost:5000,http://localhost:5173,http://127.0.0.1:5173').split(',')
CORS(app, resources={r"/*": {"origins": cors_origins}})

# 配置输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'output')

# 允许访问的子目录白名单
ALLOWED_FOLDERS = {'face_expression', 'gesture_analysis', 'voice_interaction'}

# 任务状态追踪
task_status = {}


def run_script(task_id, module_name, extra_args=None):
    """在后台线程运行脚本，避免阻塞网页。

    `extra_args`:附加到命令行的参数(M2 起用来透传 `--session-id`)。
    """
    task_status[task_id]["status"] = "running"
    try:
        cmd = [sys.executable, '-m', f'report_frontend.{module_name}']
        if extra_args:
            cmd.extend(extra_args)
        logger.info(f"正在启动任务 {task_id}：{cmd}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            task_status[task_id].update({
                "status": "success",
                "message": "任务完成！",
                "logs": result.stdout,
                "finished_at": time.time()
            })
        else:
            task_status[task_id].update({
                "status": "error",
                "message": "任务失败",
                "logs": result.stderr,
                "finished_at": time.time()
            })
    except Exception as e:
        task_status[task_id].update({
            "status": "error",
            "message": str(e),
            "finished_at": time.time()
        })


# --- 路由：主页 ---
@app.route('/')
def index():
    return render_template('dashboard.html')


# --- API: 获取 output 目录下的文件列表 ---
@app.route('/api/files/<folder_name>')
def get_files(folder_name):
    """列出 data/output/<folder_name> 下的文件"""
    if folder_name not in ALLOWED_FOLDERS:
        return jsonify({"status": "error", "message": "不允许访问该目录"}), 403

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
    files.sort(key=lambda x: x['name'], reverse=True)
    return jsonify(files)


# --- API: 提供静态文件访问 ---
@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# --- API: 触发各个模块运行 ---
@app.route('/api/run/<module>', methods=['POST'])
def trigger_module(module):
    # 实时报告走独立流程（需要 session_id）
    if module == "report_live":
        return _trigger_live_report()

    module_map = {
        "face": "feature_engine",
        "gesture": "visualizer",
        "voice": "run_interview_assessment_voice",
        "report": "report_generator"
    }

    target_module = module_map.get(module)

    if not target_module:
        return jsonify({"status": "error", "message": "未知的模块"})

    task_id = str(uuid.uuid4())[:8]
    task_status[task_id] = {
        "module": module,
        "status": "pending",
        "started_at": time.time(),
        "finished_at": None,
        "message": "",
        "logs": ""
    }

    # M2:报告模块接受 `--session-id`,由调用方(query 参数)决定描述哪一场
    extra = None
    sid = request.args.get("session_id")
    if module == "report" and sid:
        extra = ["--session-id", sid]

    thread = threading.Thread(target=run_script, args=(task_id, target_module, extra))
    thread.start()

    return jsonify({
        "status": "started",
        "task_id": task_id,
        "message": f"任务 [{module}] 已启动，task_id: {task_id}"
    })


def _trigger_live_report():
    """触发实时报告生成"""
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id", "").strip()

    if not session_id:
        return jsonify({"status": "error", "message": "缺少 session_id 参数"}), 400

    task_id = str(uuid.uuid4())[:8]
    task_status[task_id] = {
        "module": "report_live",
        "status": "pending",
        "session_id": session_id,
        "started_at": time.time(),
        "finished_at": None,
        "message": "",
        "logs": ""
    }

    thread = threading.Thread(target=_run_live_report, args=(task_id, session_id))
    thread.start()

    return jsonify({
        "status": "started",
        "task_id": task_id,
        "session_id": session_id,
        "message": f"实时报告任务已启动，session: {session_id}, task_id: {task_id}"
    })


def _run_live_report(task_id, session_id):
    """在后台线程中运行实时报告生成"""
    task_status[task_id]["status"] = "running"
    try:
        from report_frontend.report_generator import ReportGenerator
        gen = ReportGenerator()
        path = gen.generate_report_live(session_id)
        task_status[task_id].update({
            "status": "success",
            "message": "实时报告生成完成！",
            "logs": f"报告路径: {path}" if path else "生成失败",
            "finished_at": time.time()
        })
    except Exception as e:
        logger.exception("实时报告生成失败")
        task_status[task_id].update({
            "status": "error",
            "message": str(e),
            "finished_at": time.time()
        })


# --- API: 查询任务状态 ---
@app.route('/api/task/<task_id>')
def get_task_status(task_id):
    """查询后台任务的执行状态"""
    status = task_status.get(task_id)
    if not status:
        return jsonify({"status": "error", "message": "任务不存在"}), 404
    return jsonify({"task_id": task_id, **status})


# --- API: 获取结构化评估报告 JSON ---
@app.route('/api/report/structured')
def get_structured_report():
    """
    运行 report_generator 流水线，返回结构化评估 JSON。
    可选参数: type=interview|research (过滤日志类型)
    """
    try:
        from report_frontend.data_loader import LogDataLoader
        from report_frontend.feature_engine import PsychologicalFeatureEngine
        from report_frontend.research_mapper import ResearchCapabilityMapper

        # M2:报告要描述**哪一场**由调用方指定;不给就取最新一场(加载器会在报告头写明)。
        session_id = request.args.get("session_id") or None
        loader = LogDataLoader()
        data = loader.get_fused_latest_data(session_id)

        if not data:
            return jsonify({"status": "error", "message": "未找到评估日志数据"})

        engine = PsychologicalFeatureEngine(data)
        features = engine.extract_all_features()

        mapper = ResearchCapabilityMapper()
        result = mapper.map_features_to_scores(features)

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        logger.exception("获取结构化评估报告失败")
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("JingXin 总控平台启动中...")
    logger.info(f"数据目录: {OUTPUT_DIR}")
    logger.info("访问地址：http://127.0.0.1:5000")
    logger.info("=" * 50)
    app.run(debug=True, port=5000)
