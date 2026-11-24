import os
import re
import sys
import hashlib
import json as _json
import subprocess
import threading
import urllib.error
import urllib.request
import uuid
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, Response, render_template, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from logging_config import setup_logging
from report_frontend import brand
import session_purge

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

# 素材浏览页(`/api/recordings*`)。注册在这里而不是另起一个服务:它要发的
# 是**磁盘上的文件**(录像 / 日志 / 帧),而 Flask 的 `send_file(conditional=True)`
# 原生支持 Range 请求 —— 视频拖拽与断点续传都是白送的。另起一个服务就得自己写。
# 报告落点也一并交给它,免得两处各算一遍。
import recordings_browser

# 数据导出 / 图表(`/api/export/*`)。原始 CSV 与转写此前只有"在盘上"，没有出口；
# 复核与换算法都需要它们(2026-10-26 使用者要求)。matplotlib 出 PNG，不依赖 CDN。
import exports

app.config['OUTPUT_DIR'] = OUTPUT_DIR
# 日志落点也交给它:`purge`(真删)要连同 `data/logs/` 里那三份 CSV 一起删,
# 少删一处就会留下"有日志没素材"的半截状态。同样只在这里定义一次。
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'logs')
app.config['LOGS_DIR'] = LOGS_DIR
app.register_blueprint(exports.bp)
app.register_blueprint(recordings_browser.bp)

# 报告**不写在子目录里** —— `report_generator` 把 `*_Assessment_Report_*.html`
# 直接落在 OUTPUT_DIR 根下。而前端「报告列表」页请求的是文件夹名,于是它去找一个
# 根本不存在的子目录 ⟹ 既不在 ALLOWED_FOLDERS 里(回 403)、目录也不存在
# ⟹ **列表永远是空的**(2026-09-26 实测:使用者反复报"报告列表承接不成功")。
# 修法:给根目录一个显式别名,连同正确的静态 URL 前缀一起映射。
# (没有把报告挪进子目录,是因为读它的地方不止这一处:面板自己的打开流程、
#  `--session-id` 直接跑、以及 §5 的验收判据都按现在这个位置写。)
REPORT_FOLDER = "reports"          # 前端用这个别名请求根下的报告

#: 「什么算一份报告」—— **只在这里定义一次**。
#:
#: 根目录下混着三类产物,而它们的名字长得不像但都在 `*.html` 里:
#:   · `Research_Assessment_Report_*.html`  14 份 —— 报告本体
#:   · `radar_chart_*.html`                 15 份 —— 单份 4.8 MB 的 plotly 图
#:   · `evidence_*.html`                    55 份 —— 旧引擎的逐维证据图(产出方已摘除)
#:
#: 这个判据此前被三处消费,而三处各写各的:`Home.tsx` 与 `ReportsList.tsx` 在前端
#: 各自 `filter(/Assessment_Report.*\.html$/)`,而 `templates/dashboard.html`
#: **一处都没写**。而那张卡片取 `slice(0, 6)` 且按名字**倒序** —— 小写 `r`(114)
#: 排在大写 `R`(82)前面,所以**就算请求发出去**,那 6 个 iframe 也全是雷达图。
#: (2026-10-26 实测:它其实连请求都没发,原因是另一个缺陷 —— `'report': ''`
#:  被 `if (!folder) return` 当成 falsy 拦掉了,见 `dashboard.html` 的 `folderMap`。
#:  两个缺陷叠在一起,所以此前没人发现接口这一层也错。)
#:
#: 判据收在后端,`/api/files/reports` 只回报告。前端那两处过滤留着是幂等的,
#: 但事实只有这一份。
REPORT_FILENAME_RE = brand.REPORT_FILENAME_RE

# 含受试者后缀的报告文件名（如 Research_Assessment_Report_20261122_131902-姓名.html）。
# 未认证演示模式对外隐藏后缀真实姓名，并通过不可逆短 token 提供脱敏版 HTML。
_NAMED_REPORT_RE = re.compile(
    r"^((?:HengCai_Evidence_Report|Research_Assessment_Report)_(?:Live_)?\d{8}_\d{6})-([^/\\]+)\.html$",
    re.IGNORECASE,
)
_SUBJECT_DIV_RE = re.compile(
    r'<div\s+class="subject(?:\s+[^"]*)?"[^>]*>.*?</div>',
    re.S,
)
_REDACTED_SUBJECT_HTML = (
    '<div class="subject subject-missing">受试者身份：'
    '<strong>已脱敏（演示模式）</strong>'
    '（仅展示多模态行为证据与缺口；管理员登录后可见完整身份信息）</div>'
)
_REDACTED_SUBJECT_OBJ = {
    "status": "redacted",
    "note": "演示模式已脱敏隐藏受试者身份（管理员登录后可见完整信息）",
}

_REPORT_SID_CACHE: dict[tuple[str, int, int], str | None] = {}
_REPORT_SID_LOCK = threading.Lock()
_REDACTED_TOKENS: dict[str, str] = {}
_REDACTED_TOKENS_LOCK = threading.Lock()
_TASK_MAX_ENTRIES = 200

# 任务状态追踪
task_status = {}


def _prune_task_status() -> None:
    if len(task_status) <= _TASK_MAX_ENTRIES:
        return
    finished = sorted(
        (
            (v.get("finished_at") or v.get("started_at") or 0.0, k)
            for k, v in task_status.items()
            if v.get("status") in ("success", "error")
        ),
        key=lambda x: x[0],
    )
    drop = len(task_status) - _TASK_MAX_ENTRIES
    for _, k in finished[:drop]:
        task_status.pop(k, None)


def _report_token(filename: str) -> str:
    tok = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
    with _REDACTED_TOKENS_LOCK:
        _REDACTED_TOKENS[tok] = filename
    return tok


def _resolve_redacted_token(token: str) -> str | None:
    if not token or not re.fullmatch(r"[0-9a-f]{16}", token):
        return None
    with _REDACTED_TOKENS_LOCK:
        hit = _REDACTED_TOKENS.get(token)
    if hit and os.path.isfile(os.path.join(OUTPUT_DIR, hit)):
        return hit
    if os.path.isdir(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if REPORT_FILENAME_RE.search(f) and hashlib.sha256(f.encode("utf-8")).hexdigest()[:16] == token:
                with _REDACTED_TOKENS_LOCK:
                    _REDACTED_TOKENS[token] = f
                return f
    return None


def _extract_report_session_id(full_path: str) -> str | None:
    try:
        st = os.stat(full_path)
    except OSError:
        return None
    key = (full_path, st.st_mtime_ns, st.st_size)
    with _REPORT_SID_LOCK:
        if key in _REPORT_SID_CACHE:
            return _REPORT_SID_CACHE[key]
    sid: str | None = None
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(8192)
        m = session_purge.SID_IN_TEXT.search(head)
        if m:
            sid = m.group(0)
    except OSError:
        sid = None
    with _REPORT_SID_LOCK:
        if len(_REPORT_SID_CACHE) > 2000:
            _REPORT_SID_CACHE.clear()
        _REPORT_SID_CACHE[key] = sid
    return sid


def _public_report_ref(filename: str, *, is_admin: bool) -> tuple[str, str, bool]:
    """返回 `(对外文件名, URL, 是否已脱敏)`。"""
    m = _NAMED_REPORT_RE.match(filename)
    if (not is_admin) and m and not re.fullmatch(r"[0-9a-fA-F_-]+", m.group(2)):
        safe_name = f"{m.group(1)}-已脱敏.html"
        return safe_name, f"/output/redacted/{_report_token(filename)}", True
    return filename, f"/output/{filename}", False


def _redact_report_html(html_text: str) -> str:
    return _SUBJECT_DIV_RE.sub(_REDACTED_SUBJECT_HTML, html_text)


def _serve_report_html(filename: str, *, redact: bool):
    full = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(full):
        return jsonify({"status": "error", "message": "报告文件不存在"}), 404
    if not redact:
        return send_from_directory(OUTPUT_DIR, filename)
    with open(full, "r", encoding="utf-8", errors="replace") as fh:
        body = _redact_report_html(fh.read())
    resp = Response(body, mimetype="text/html; charset=utf-8")
    resp.headers["X-JX-Redacted"] = "1"
    return resp


def _parse_report_path_from_stdout(stdout: str) -> str | None:
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("REPORT_PATH="):
            cand = line[len("REPORT_PATH="):].strip()
            if cand:
                return cand
    return None


def run_script(task_id, module_name, extra_args=None, is_admin: bool = False):
    """在后台线程运行脚本，避免阻塞网页。

    `extra_args`:附加到命令行的参数(M2 起用来透传 `--session-id`)。
    """
    task_status[task_id]["status"] = "running"
    try:
        cmd = [sys.executable, '-m', f'report_frontend.{module_name}']
        if extra_args:
            cmd.extend(extra_args)
        logger.info(f"正在启动任务 {task_id}：{cmd}")

        env = dict(os.environ)
        env["JX_NO_BROWSER"] = "1"
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
            env=env,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            report_path = _parse_report_path_from_stdout(result.stdout)
            update_data = {
                "status": "success",
                "message": "任务完成！",
                "logs": result.stdout,
                "finished_at": time.time()
            }
            if report_path:
                fname = os.path.basename(report_path)
                disp_name, url, redacted = _public_report_ref(fname, is_admin=is_admin)
                update_data.update({
                    "report_name": disp_name,
                    "report_url": url,
                    "redacted": redacted,
                })
            task_status[task_id].update(update_data)
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
    finally:
        _prune_task_status()


# --- 路由：主页 ---
@app.route('/')
def index():
    return render_template('dashboard.html')


# --- API: 获取 output 目录下的文件列表 ---
@app.route('/api/files/<folder_name>')
def get_files(folder_name):
    """列出 data/output/<folder_name> 下的文件"""
    if folder_name == REPORT_FOLDER:
        folder_path, url_prefix = OUTPUT_DIR, "/output"
    elif folder_name in ALLOWED_FOLDERS:
        folder_path, url_prefix = os.path.join(OUTPUT_DIR, folder_name), f"/output/{folder_name}"
    else:
        return jsonify({"status": "error", "message": "不允许访问该目录"}), 403

    if not os.path.exists(folder_path):
        return jsonify([])

    is_admin = recordings_browser._is_admin()
    files = []
    for f in os.listdir(folder_path):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.html')):
            continue
        # 报告别名只回**报告**。根目录下还躺着 radar_chart_*.html(每份 4.8 MB)
        # 与 evidence_*.html(旧引擎证据图),它们也叫 .html —— 原样返回的话,
        # 面板那张卡片取前 6 个 iframe 全是雷达图(见 `REPORT_FILENAME_RE` 上方说明)。
        if folder_name == REPORT_FOLDER and not REPORT_FILENAME_RE.search(f):
            continue
        full_path = os.path.join(folder_path, f)
        try:
            mtime = os.path.getmtime(full_path)
        except OSError:
            mtime = 0.0
        if folder_name == REPORT_FOLDER:
            disp_name, url, redacted = _public_report_ref(f, is_admin=is_admin)
            files.append({
                "name": disp_name,
                "display_name": disp_name,
                "url": url,
                "session_id": _extract_report_session_id(full_path),
                "mtime": mtime,
                "redacted": redacted,
            })
        else:
            files.append({
                "name": f,
                "url": f"{url_prefix}/{f}",
                "mtime": mtime,
            })
    if folder_name == REPORT_FOLDER:
        files.sort(key=lambda x: (x.get('mtime', 0.0), x['name']), reverse=True)
    else:
        files.sort(key=lambda x: x['name'], reverse=True)
    return jsonify(files)


# --- API: 提供静态文件访问（未认证演示模式对报告 HTML 自动脱敏受试者身份） ---
@app.route('/output/redacted/<token>')
def serve_redacted_report(token):
    real_name = _resolve_redacted_token(token)
    if not real_name:
        return jsonify({"status": "error", "message": "报告文件不存在或令牌无效"}), 404
    is_admin = recordings_browser._is_admin(allow_query=True)
    return _serve_report_html(real_name, redact=not is_admin)


@app.route('/output/<path:filename>')
def serve_output(filename):
    is_admin = recordings_browser._is_admin(allow_query=True)
    # 根目录 HTML 报告：未登录管理员一律剥除身份横幅
    if os.path.dirname(filename) == "" and REPORT_FILENAME_RE.search(filename):
        return _serve_report_html(filename, redact=not is_admin)
    return send_from_directory(OUTPUT_DIR, filename)


# --- API: 触发各个模块运行 ---
@app.route('/api/run/<module>', methods=['POST'])
def trigger_module(module):
    # 实时报告走独立流程（需要 session_id）
    if module == "report_live":
        return _trigger_live_report()
    if module == "voice_series":
        return _trigger_voice_series()

    module_map = {
        "report": "report_generator"
    }

    target_module = module_map.get(module)

    if not target_module:
        return jsonify({
            "status": "error",
            "message": f"不支持运行模块 {module!r}（仅支持 report / report_live / voice_series）"
        }), 400

    is_admin = recordings_browser._is_admin()
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

    thread = threading.Thread(target=run_script, args=(task_id, target_module, extra, is_admin))
    thread.start()

    return jsonify({
        "status": "started",
        "task_id": task_id,
        "message": f"任务 [{module}] 已启动，task_id: {task_id}"
    })


def _trigger_voice_series():
    """管理员按会话触发语音序列重建（data/l0_series/<sid>.npz）。"""
    if recordings_browser._admin_password() is None:
        return recordings_browser._password_missing()
    if not recordings_browser._is_admin():
        return jsonify({"status": "error", "message": "需要管理员密码"}), 401
    body = request.get_json(silent=True) or {}
    sid = (body.get("session_id") or request.args.get("session_id") or "").strip()
    if not sid:
        return jsonify({"status": "error", "message": "缺少 session_id 参数"}), 400
    force = bool(body.get("force", False))
    task_id = str(uuid.uuid4())[:8]
    task_status[task_id] = {
        "module": "voice_series",
        "status": "pending",
        "session_id": sid,
        "started_at": time.time(),
        "finished_at": None,
        "message": "",
        "logs": "",
    }
    thread = threading.Thread(target=_run_voice_series, args=(task_id, sid, force))
    thread.start()
    return jsonify({
        "status": "started",
        "task_id": task_id,
        "session_id": sid,
        "message": f"语音序列重建任务已启动，session: {sid}, task_id: {task_id}",
    })


def _run_voice_series(task_id: str, session_id: str, force: bool = False):
    task_status[task_id]["status"] = "running"
    try:
        from report_frontend.voice_series import extract_session
        got = extract_session(session_id, force=force)
        task_status[task_id].update({
            "status": "success" if got else "error",
            "message": f"语音序列重建完成（共 {len(got)} 段）" if got else "未找到可提取的原始音频",
            "logs": f"extracted_segments={len(got)}",
            "finished_at": time.time(),
        })
    except Exception as e:
        logger.exception("语音序列重建失败")
        task_status[task_id].update({
            "status": "error",
            "message": str(e),
            "finished_at": time.time(),
        })
    finally:
        _prune_task_status()


def _trigger_live_report():
    """触发实时报告生成"""
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id", "").strip()

    if not session_id:
        return jsonify({"status": "error", "message": "缺少 session_id 参数"}), 400

    is_admin = recordings_browser._is_admin()
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

    thread = threading.Thread(target=_run_live_report, args=(task_id, session_id, is_admin))
    thread.start()

    return jsonify({
        "status": "started",
        "task_id": task_id,
        "session_id": session_id,
        "message": f"实时报告任务已启动，session: {session_id}, task_id: {task_id}"
    })


def _run_live_report(task_id, session_id, is_admin: bool = False):
    """在后台线程中运行实时报告生成"""
    task_status[task_id]["status"] = "running"
    try:
        os.environ.setdefault("JX_NO_BROWSER", "1")
        from report_frontend.report_generator import ReportGenerator
        gen = ReportGenerator()
        path = gen.generate_report_live(session_id)
        # 「没产出」不许报成功(第 17 条同族)—— `generate_report_live` 失败时正是 `return ""`,
        # 而这条路的旧实现无条件报「实时报告生成完成！」,把静默失败原样搬到了隔壁这条路上。
        update_data = {
            "status": "success" if path else "error",
            "message": "实时报告生成完成！" if path else "实时报告生成失败",
            "logs": f"报告路径: {path}" if path else "生成失败",
            "finished_at": time.time()
        }
        if path:
            fname = os.path.basename(path)
            disp_name, url, redacted = _public_report_ref(fname, is_admin=is_admin)
            update_data.update({
                "report_name": disp_name,
                "report_url": url,
                "redacted": redacted,
            })
        task_status[task_id].update(update_data)
    except Exception as e:
        logger.exception("实时报告生成失败")
        task_status[task_id].update({
            "status": "error",
            "message": str(e),
            "finished_at": time.time()
        })
    finally:
        _prune_task_status()


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
    可选参数: session_id(要描述哪一场;缺省取最新一场)

    M2.1(第 18 条):①**本场没有任何日志时也照常回答**(spec §6 行 1/2 要求"报告仍生成"),
    不再回 `{"status":"error","message":"未找到评估日志数据"}` —— 那种回答会让前端显示
    「暂无报告数据」,而盘上明明有别的会话;②回传 `session_id` 与 `sources`,让读的人
    **从 JSON 里就能确认这是哪一场、哪个模态没进来**(此前"写明是哪一场"只落在 HTML 报告里)。
    """
    try:
        from report_frontend.data_loader import LogDataLoader

        # M2:报告要描述**哪一场**由调用方指定;不给就取最新一场(加载器会在报告头写明)。
        session_id = request.args.get("session_id") or None
        loader = LogDataLoader()
        data = loader.get_fused_latest_data(session_id)

        # ── M4.5:默认走 **L1 指标层**(2026-10-18 切换)────────────────────────
        #
        # 旧链(`feature_engine` → `research_mapper` 的 14 个模糊匹配槽)留着是因为
        # 切换要能和旧结果并排看一轮;`?engine=legacy` 就能取到旧载荷。
        #
        # 为什么换:`research_mapper` 的槽是从扁平特征键里按**列名子串**模糊匹配的,
        # 不保证量的是同一个东西(历史上「皱眉」就匹配到过从 micro_exp 字符串解析出的垃圾)。
        # L1 每条指标有公式、有 `n_valid`、有置信度、有 gap,且逐条能对到规格。
        #
        # 顺带换掉一句**假陈述**:旧载荷每个维度都写着
        # `algorithm: "加权线性组合 + 认知负荷推断"`,而这一层既没有加权组合
        # (权重只用于删槽后的记账),也没有认知负荷推断(本仓不做推断)。
        which = (request.args.get("engine") or "l1").strip().lower()

        if which == "legacy":
            from report_frontend.feature_engine import PsychologicalFeatureEngine
            from report_frontend.research_mapper import ResearchCapabilityMapper

            # 空 data 也把链走完:实测 features={} → coverage 0/20、五维全 None、图表照常生成
            features = PsychologicalFeatureEngine(data).extract_all_features()
            result = ResearchCapabilityMapper().map_features_to_scores(features)
        elif which == "l1":
            from report_frontend.l1_report import build_l1_report
            from report_frontend import scene_profiles

            raw_sp = (request.args.get("scene_profile") or "").strip() or None
            try:
                norm_sp = scene_profiles.normalize_scene_profile_id(raw_sp) if raw_sp else None
            except ValueError as exc:
                return jsonify({"status": "error", "message": str(exc)}), 400
            # `logs` 直接给已经融合好的那份,不让 L1 层自己再去 `data/` 里找一遍
            # (读取口径只有 `LogDataLoader` 一处)。
            result = build_l1_report(loader.target_session, logs=data, scene_profile=norm_sp)
        else:
            return jsonify({"status": "error",
                            "message": f"engine 只能是 l1 或 legacy,收到 {which!r}"}), 400

        # ── 报告页的「这一场是谁的、什么时候测的」────────────────────────────
        # 与 HTML 报告头**同一口径**(2026-10-26 使用者:报告分辨不出人)。
        # 读不到标注**不许**让报告失败 —— 报告页至少还能看数;缺什么就印"未标注"。
        subject: dict = {}
        try:
            from session_meta import read_label
            rec = read_label(loader.target_session) or {}
            subject = {k: rec[k] for k in ("name", "student_id", "department", "serial")
                       if rec.get(k)}
        except Exception:
            logger.exception("读本场标注失败(报告照常返回)")

        is_admin = recordings_browser._is_admin()
        if subject and not is_admin:
            subject = dict(_REDACTED_SUBJECT_OBJ)

        # 「测试时间」= 铸号时刻(会话 id 自带),不是本请求的时刻。
        # 「报告生成时间」= 这一次生成的时刻。两个都回,前端同屏印。
        try:
            from report_frontend.report_generator import session_started_at
            _started = session_started_at(loader.target_session)
        except Exception:
            _started = None

        return jsonify({
            "status": "success",
            "session_id": loader.target_session,
            "sources": loader.selected_sessions,
            "engine": which,
            "result": result,
            "subject": subject,
            "is_admin": is_admin,
            "tested_at": _started.strftime("%Y-%m-%d %H:%M:%S") if _started else None,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as e:
        logger.exception("获取结构化评估报告失败")
        return jsonify({"status": "error", "message": str(e)}), 500


# --- API: 聚合系统健康检查（绕过系统代理，1.2s 并行探活） ---
def _probe_http_health(name: str, url: str, timeout: float = 1.2) -> tuple[str, dict]:
    t0 = time.perf_counter()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            dt_ms = round((time.perf_counter() - t0) * 1000, 1)
            payload = _json.loads(raw) if raw.strip().startswith("{") else {}
            status = str(payload.get("status") or ("healthy" if resp.status == 200 else "degraded"))
            info = {
                "ok": resp.status == 200 and status == "healthy",
                "status": status,
                "latency_ms": dt_ms,
                "url": url,
            }
            for extra_k in (
                "asr_endpoint",
                "asr_reachable",
                "backend",
                "supported_backends",
                "pitch_backend",
                "vad_backend",
                "live_pitch_backend",
                "live_vad_backend",
                "whisper_conf_ready",
                "landmarker_3d_matrix",
                "blendshapes_count",
                "live_framing",
            ):
                if extra_k in payload:
                    info[extra_k] = payload[extra_k]
            return name, info
    except Exception as exc:
        dt_ms = round((time.perf_counter() - t0) * 1000, 1)
        return name, {
            "ok": False,
            "status": "offline",
            "latency_ms": dt_ms,
            "url": url,
            "error": type(exc).__name__,
        }


@app.route('/api/system/health', methods=['GET'])
def system_health():
    """聚合面板、面部、语音(+ASR)、手势四路服务的实时就绪状态。"""
    targets = {
        "face": os.getenv("JX_FACE_HEALTH_URL", "http://127.0.0.1:8000/health"),
        "voice": os.getenv("JX_VOICE_HEALTH_URL", "http://127.0.0.1:8001/health"),
        "gesture": os.getenv("JX_GESTURE_HEALTH_URL", "http://127.0.0.1:8002/health"),
    }
    services: dict[str, dict] = {
        "panel": {"ok": True, "status": "healthy", "latency_ms": 0.0, "url": "http://127.0.0.1:5000"},
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_probe_http_health, k, u) for k, u in targets.items()]
        for fut in futs:
            k, info = fut.result()
            services[k] = info

    all_ok = all(v.get("ok") for v in services.values())
    any_ok = any(v.get("ok") for k, v in services.items() if k != "panel")
    overall = "healthy" if all_ok else ("degraded" if any_ok else "offline")
    return jsonify({
        "status": overall,
        "brand": brand.BRAND_NAME,
        "subtitle": brand.BRAND_SUBTITLE,
        "software_name": brand.COPYRIGHT_SOFTWARE_NAME,
        "admin_configured": recordings_browser._admin_password() is not None,
        "is_admin": recordings_browser._is_admin(),
        "services": services,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


# ── 题库:查看 / 上传替换 / 恢复默认 ──────────────────────────────────────
# 两个模块(面试 / 科研)。题库落盘见 `question_bank.py`;语音服务在新开一场时
# 读取,**上传即生效、不用重启服务**(2026-10-26 使用者要求)。
@app.route('/api/question-bank/<module>', methods=['GET'])
def get_question_bank(module):
    import question_bank
    try:
        return jsonify(question_bank.load(module))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/api/question-bank/<module>/banks', methods=['POST'])
@recordings_browser._admin_required()
def create_question_bank(module):
    """新建一套题库（不覆盖已有的任何一套）。"""
    body = request.get_json(silent=True) or {}
    import question_bank
    try:
        return jsonify(question_bank.create_bank(module, body.get("name"),
                                                 body.get("questions")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/api/question-bank/<module>/banks/<bank_id>', methods=['PUT'])
@recordings_browser._admin_required()
def update_question_bank(module, bank_id):
    body = request.get_json(silent=True) or {}
    import question_bank
    try:
        return jsonify(question_bank.update_bank(module, bank_id,
                                                 name=body.get("name"),
                                                 items=body.get("questions")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/api/question-bank/<module>/banks/<bank_id>', methods=['DELETE'])
@recordings_browser._admin_required()
def delete_question_bank(module, bank_id):
    import question_bank
    try:
        return jsonify(question_bank.delete_bank(module, bank_id))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


#: 两套随仓库走的示例题库（8 题 + 参考答案）。用户从页面下载后走「导入题库文件」，
#: 不直接写进 data/question_bank.json —— 导入仍由使用者自己确认。
SAMPLE_BANKS = {
    "interview": "面试题库_8题_含参考答案.json",
    "research": "科研题库_8题_含参考答案.json",
}


@app.route('/api/question-bank/sample/<module>', methods=['GET'])
def download_sample_question_bank(module):
    name = SAMPLE_BANKS.get(module)
    if not name:
        return jsonify({"error": f"未知模块 {module!r}"}), 404
    path = os.path.join(os.path.dirname(__file__), 'docs', '题库', name)
    if not os.path.isfile(path):
        return jsonify({"error": "示例题库文件不在盘上"}), 404
    return send_file(path, as_attachment=True, download_name=name,
                     mimetype='application/json')


@app.route('/api/question-bank/<module>/active', methods=['POST'])
@recordings_browser._admin_required()
def activate_question_bank(module):
    """把某套设为"下一场评估用这一套"；`id=null` 表示回到内置默认。"""
    body = request.get_json(silent=True) or {}
    import question_bank
    try:
        return jsonify(question_bank.activate(module, body.get("id")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


# ── 本地按需辅助接口（严格隔离：零指标计算、零打分、红线审查 + 毫秒级回退） ──
@app.route('/api/report/llm-assist', methods=['POST'])
def report_llm_assist():
    """按需生成脱敏报告的可读化摘要或基于证据缺口的追问提纲。"""
    body = request.get_json(silent=True) or {}
    session_id = (body.get("session_id") or request.args.get("session_id") or "").strip() or None
    mode = (body.get("mode") or "summary").strip()
    use_llm = bool(body.get("use_llm", True))
    try:
        from report_frontend.data_loader import LogDataLoader
        from report_frontend.l1_report import build_l1_report
        from report_frontend.local_llm_assistant import assist_report

        loader = LogDataLoader()
        data = loader.get_fused_latest_data(session_id)
        payload = build_l1_report(
            loader.target_session,
            logs=data,
            scene_profile=(body.get("scene_profile") or "").strip() or None,
        )
        res = assist_report(payload, mode=mode, use_llm=use_llm)
        return jsonify({
            "status": "success",
            "session_id": loader.target_session,
            **res,
        })
    except Exception as exc:
        logger.exception("报告本地辅助解读失败")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route('/api/question-bank/llm-search', methods=['POST'])
def question_bank_llm_search():
    """毫秒级本地 TF-IDF 题库检索（可选本地小模型扩展查询词）。"""
    body = request.get_json(silent=True) or {}
    module = (body.get("module") or "interview").strip()
    query = str(body.get("query") or "")
    top_k = int(body.get("top_k") or 5)
    use_llm = bool(body.get("use_llm", False))
    try:
        from report_frontend.local_llm_assistant import search_question_bank
        res = search_question_bank(module, query, top_k=top_k, use_llm=use_llm)
        return jsonify({"status": "success", **res})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        logger.exception("题库语义检索失败")
        return jsonify({"status": "error", "message": str(exc)}), 500


def _load_json_if_exists(path: str) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = _json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.exception("读取基准 JSON 失败: %s", path)
        return None


def _summarize_model_macros(models_dict: dict | None) -> list[dict]:
    if not isinstance(models_dict, dict):
        return []
    out: list[dict] = []
    for model_name, payload in models_dict.items():
        if not isinstance(payload, dict):
            continue
        macro = payload.get("macro") if isinstance(payload.get("macro"), dict) else payload
        out.append({
            "model": model_name,
            "pearson_r": macro.get("pearson_r"),
            "spearman_rho": macro.get("spearman_rho"),
            "rmse": macro.get("rmse"),
            "mae": macro.get("mae"),
            "acc_1_minus_mae": macro.get("acc_1_minus_mae"),
        })
    return out


@app.route('/api/research/benchmarks', methods=['GET'])
def get_research_benchmarks():
    """只读聚合离线科研方法学与层内校验基准（零 PII、零真实受试者明细、绝不参与评分）。"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    reinforce_dir = os.path.join(project_root, "07_三方向与评审筹备", "系统补强")

    mc_raw = _load_json_if_exists(
        os.getenv("JX_BENCH_MULTICORPUS_PATH")
        or os.path.join(base_dir, "research_lab", "results", "multicorpus_benchmark_20261124.json")
    )
    bs_raw = _load_json_if_exists(
        os.getenv("JX_BENCH_BLENDSHAPE_PATH")
        or os.path.join(reinforce_dir, "blendshape_calibration_benchmark_20261124.json")
    )
    ref_raw = _load_json_if_exists(
        os.getenv("JX_BENCH_REFEREE_PATH")
        or os.path.join(base_dir, "report_frontend", "referee_ml_metrics.json")
    )
    gpu_raw = _load_json_if_exists(
        os.getenv("JX_BENCH_GPU_VS_CPU_PATH")
        or os.path.join(reinforce_dir, "gpu_vs_cpu_referee_benchmark_20261124.json")
    )

    multicorpus_section: dict = {"available": False}
    if mc_raw:
        ext = mc_raw.get("extraction_summary") or {}
        mit = mc_raw.get("experiment_1_mit") or {}
        rv = mc_raw.get("experiment_2_recruitview") or {}
        rv_ta = rv.get("track_a_full_metadata_text") or {}
        rv_tb = rv.get("track_b_multimodal") or {}
        ch = mc_raw.get("experiment_3_chalearn") or {}
        ch_best_name = ch.get("best_model") or "AudioVisual-Ridge"
        ch_best_targets = ((ch.get("models") or {}).get(ch_best_name) or {}).get("per_target") or {}
        rt = mc_raw.get("experiment_4_regime_transfer") or {}
        multicorpus_section = {
            "available": True,
            "generated_at": mc_raw.get("generated_at"),
            "extraction": {
                "total_clips_succeeded": ext.get("total_clips_succeeded"),
                "clips_per_sec": ext.get("clips_per_sec"),
                "workers": ext.get("workers"),
            },
            "mit": {
                "n_interviews": mit.get("n_interviews"),
                "n_participants": mit.get("n_participants"),
                "cv_scheme": mit.get("cv_scheme"),
                "best_model": mit.get("best_model"),
                "models": _summarize_model_macros(mit.get("models")),
                "gpu_mlp_hardware": mit.get("gpu_mlp_hardware"),
            },
            "recruitview": {
                "cv_scheme": rv.get("cv_scheme"),
                "track_a_n_clips": rv_ta.get("n_clips"),
                "track_a_n_users": rv_ta.get("n_users"),
                "track_a_models": _summarize_model_macros(rv_ta.get("models")),
                "track_b_n_clips": rv_tb.get("n_clips"),
                "track_b_n_users": rv_tb.get("n_users"),
                "track_b_best_model": rv_tb.get("best_model"),
                "track_b_models": _summarize_model_macros(rv_tb.get("models")),
            },
            "chalearn": {
                "split_counts": ch.get("split_counts"),
                "n_total_features": ch.get("n_total_features"),
                "best_model": ch_best_name,
                "models": _summarize_model_macros(ch.get("models")),
                "best_model_dimensions": [
                    {
                        "key": k,
                        "pearson_r": v.get("pearson_r"),
                        "spearman_rho": v.get("spearman_rho"),
                        "acc_1_minus_mae": v.get("acc_1_minus_mae"),
                    }
                    for k, v in ch_best_targets.items()
                    if isinstance(v, dict)
                ],
            },
            "regime_transfer": {
                "shared_features": rt.get("shared_features") or [],
                "chinese_corpus": rt.get("chinese_corpus") or {},
                "english_corpus": rt.get("english_corpus") or {},
                "regimes": rt.get("regimes") or {},
            },
        }

    blendshape_section: dict = {"available": False}
    if bs_raw and isinstance(bs_raw.get("corpus_summary"), dict):
        blendshape_section = {
            "available": True,
            "generated_at": bs_raw.get("generated_at"),
            "corpus_summary": bs_raw["corpus_summary"],
        }

    referee_section: dict = {"available": False}
    if ref_raw:
        referee_section = {
            "available": True,
            "trained_at": ref_raw.get("trained_at"),
            "scoring_participation": False,
            "non_scoring_guarantee": ref_raw.get("non_scoring_guarantee"),
            "dataset": ref_raw.get("dataset") or {},
            "stage_segmentation": ref_raw.get("stage_segmentation") or {},
            "evidence_gate": ref_raw.get("evidence_gate") or {},
            "cpu_runtime": ref_raw.get("cpu_runtime") or {},
        }

    gpu_section: dict = {"available": False}
    if gpu_raw:
        gpu_section = {
            "available": True,
            "generated_at": gpu_raw.get("generated_at"),
            "hardware": gpu_raw.get("hardware") or {},
            "cpu_referee": gpu_raw.get("cpu_referee") or {},
            "gpu_neural_sequence_referee": gpu_raw.get("gpu_neural_sequence_referee") or {},
            "comparison": gpu_raw.get("comparison") or {},
        }

    payload = {
        "status": "success",
        "research_only": True,
        "scoring_participation": False,
        "boundary_note": (
            "本页面展示公开学术数据集与工程消融方法学验证结果，旨在论证跨通道信号稳定性、姿态解耦与"
            "不确定性核验有效性；相关权重与统计数值均不参与任何真实访谈报告的计分、排序或个体评价。"
        ),
        "multicorpus": multicorpus_section,
        "blendshape_calibration": blendshape_section,
        "referee_ml": referee_section,
        "gpu_vs_cpu": gpu_section,
    }
    return Response(_json.dumps(payload, ensure_ascii=False, allow_nan=False), mimetype="application/json")


# ── 多场景声明式配置接口（一核三基地共用底层） ──
@app.route('/api/scene-profiles', methods=['GET'])
def api_scene_profiles():
    """列出全部声明式场景档案或查询单一档案。"""
    from report_frontend import scene_profiles

    sid = (request.args.get("id") or "").strip() or None
    if sid:
        try:
            return jsonify({"status": "success", "profile": scene_profiles.get_scene_profile(sid)})
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
    return jsonify({
        "status": "success",
        "default_profile": scene_profiles.DEFAULT_SCENE_PROFILE_ID,
        "profiles": scene_profiles.list_scene_profiles(),
    })


# ── 标准 JSON-RPC 2.0 Career MCP 协议与求职证据对照端点（基地二） ──
@app.route('/api/mcp/career', methods=['POST'])
def api_mcp_career_rpc():
    """标准 JSON-RPC 2.0 Career MCP 协议桥接端点（严禁计算匹配分或筛人结论）。"""
    from mcp_career.server import handle_jsonrpc

    req = request.get_json(silent=True)
    if req is None:
        return jsonify({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON"}}), 400
    resp = handle_jsonrpc(req)
    if resp is None:
        return ("", 204)
    return jsonify(resp)


@app.route('/api/career/positions', methods=['GET'])
def api_career_positions():
    from mcp_career.server import tool_search_career_positions

    return jsonify({
        "status": "success",
        **tool_search_career_positions(
            keyword=request.args.get("keyword", ""),
            track=request.args.get("track", ""),
            level=request.args.get("level", ""),
        ),
    })


@app.route('/api/career/decompose-jd', methods=['POST'])
def api_career_decompose_jd():
    from mcp_career.server import tool_decompose_jd_competencies

    body = request.get_json(silent=True) or {}
    try:
        res = tool_decompose_jd_competencies(jd_text=str(body.get("jd_text") or ""))
        return jsonify({"status": "success", **res})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/api/career/generate-task', methods=['POST'])
def api_career_generate_task():
    from mcp_career.server import tool_generate_ai_critique_task

    body = request.get_json(silent=True) or {}
    try:
        res = tool_generate_ai_critique_task(
            position_id=str(body.get("position_id") or ""),
            jd_text=str(body.get("jd_text") or ""),
        )
        return jsonify({"status": "success", **res})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/api/career/align-evidence', methods=['POST'])
def api_career_align_evidence():
    from mcp_career.server import tool_align_session_evidence_to_jd

    body = request.get_json(silent=True) or {}
    try:
        res = tool_align_session_evidence_to_jd(
            session_id=str(body.get("session_id") or ""),
            position_id=str(body.get("position_id") or ""),
            jd_text=str(body.get("jd_text") or ""),
        )
        return jsonify({"status": "success", **res})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


# ── 长周期个体纵向随访与多场科研对比端点（基地三 + 基地一） ──
@app.route('/api/longitudinal/subjects', methods=['GET', 'POST'])
def api_longitudinal_subjects():
    from report_frontend import longitudinal_tracker as lt

    if request.method == 'GET':
        return jsonify({
            "status": "success",
            "compliance_note": lt.VERIFICATION_COMPLIANCE_NOTE,
            "metrics": list(lt.TRACKED_METRICS),
            "subjects": lt.list_subjects(),
        })
    body = request.get_json(silent=True) or {}
    try:
        rec = lt.upsert_subject(
            subject_code=str(body.get("subject_code") or ""),
            scene_profile=str(body.get("scene_profile") or "longitudinal_followup"),
            alias_note=str(body.get("alias_note") or ""),
            consent_mode=str(body.get("consent_mode") or "anonymous_kinematics_only"),
        )
        return jsonify({"status": "success", "subject": rec})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/api/longitudinal/subjects/<subject_code>/visits', methods=['POST'])
def api_longitudinal_attach_visit(subject_code):
    from report_frontend import longitudinal_tracker as lt

    body = request.get_json(silent=True) or {}
    try:
        rec = lt.attach_visit(
            subject_code=subject_code,
            session_id=str(body.get("session_id") or ""),
            visit_label=str(body.get("visit_label") or ""),
            privacy_mode=str(body.get("privacy_mode") or "anonymous_kinematics_only"),
        )
        return jsonify({"status": "success", "subject": rec})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/api/longitudinal/subjects/<subject_code>/trajectory', methods=['GET'])
def api_longitudinal_trajectory(subject_code):
    from report_frontend import longitudinal_tracker as lt

    try:
        data = lt.get_subject_trajectory(subject_code)
        return jsonify({"status": "success", **data})
    except KeyError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/api/reports/compare', methods=['GET'])
def api_reports_compare():
    from report_frontend import longitudinal_tracker as lt

    raw = (request.args.get("session_ids") or "").strip()
    sids = [s.strip() for s in raw.split(",") if s.strip()]
    if not sids:
        return jsonify({"status": "error", "message": "请提供至少两个 comma 分隔的 session_ids"}), 400
    scene_profile = (request.args.get("scene_profile") or "research_interview").strip()
    try:
        res = lt.compare_sessions(sids, scene_profile=scene_profile)
        return jsonify({"status": "success", **res})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


# ── 导师/管理员非打分人工复核注记（基地一保形升级闭环） ──
@app.route('/api/report/review-notes/<session_id>', methods=['GET'])
def api_get_reviewer_note(session_id):
    from report_frontend import reviewer_notes

    try:
        return jsonify({"status": "success", "note": reviewer_notes.read_reviewer_note(session_id)})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route('/api/report/review-notes/<session_id>', methods=['POST'])
@recordings_browser._admin_required()
def api_save_reviewer_note(session_id):
    from report_frontend import reviewer_notes

    body = request.get_json(silent=True) or {}
    try:
        saved = reviewer_notes.write_reviewer_note(
            session_id,
            verification_status=str(body.get("verification_status") or "verified_consistent"),
            focus_tags=list(body.get("focus_tags") or []),
            note_text=str(body.get("note_text") or ""),
        )
        return jsonify({"status": "success", "note": saved})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info(f"{brand.BRAND_FULL_TITLE} 总控服务启动中...")
    logger.info(f"数据目录: {OUTPUT_DIR}")
    logger.info("访问地址：http://127.0.0.1:5000")
    logger.info("=" * 50)
    # 绑地址可配:默认只绑本机;**要让别的电脑访问就把 PANEL_HOST 设成 0.0.0.0**
    # (tools/start_all.sh 在 JX_BIND=0.0.0.0 时会自动设,并同步把面板的地址
    #  加进另外三个服务的 CORS 白名单 —— 少一样,远端浏览器就连不上后端)。
    #
    # ⚠️ debug **默认关**:debug=True 会启用 Werkzeug reloader,任何被监视的 .py
    #    一改就重启进程,而任务表 `task_status` 是内存字典 ⟹ 正在轮询的报告任务
    #    会凭空消失(2026-10-26 实盘:面板今天自动重启 42 次,前端一直"正在生成…")。
    #    另外交互式调试器(PIN)也不该常开在能被 tailnet 访问的端口上。
    #    真要调试时显式打开:JX_PANEL_DEBUG=1 python app.py
    debug = os.getenv('JX_PANEL_DEBUG', '0') == '1'
    app.run(debug=debug, host=os.getenv('PANEL_HOST', '127.0.0.1'), port=5000)
