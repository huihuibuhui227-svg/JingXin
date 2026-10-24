# session_purge.py
"""把一场素材从盘上**彻底**抹掉。

**两个调用方共用这一份**:面板的管理员端点(`recordings_browser`)与语音服务的
`POST /session/{sid}/discard`(录制页在「不留存 / 不同意」那条路上调它)。
写两遍的代价本仓太熟了 —— 一处改了另一处没改,而失效形态是"删了一半":
场次目录没了、日志 CSV 还在,报告层于是看到一场**有日志没素材**的会话,
看起来只是"这一场没采到",不是"这一场被作废了"。

⚠️ **删的是四处**(2026-10-25 扩的,原先只有前两处):
   · 场次目录(帧 / 音频 / 留存账本 / 标注)
   · `data/logs/` 里那三份(或更少)按 sid 落盘的 CSV
   · `data/output/` 里**正文点名了这个 sid** 的报告,以及只被它引用的图表
   · `data/l0_series/<sid>.npz`(f0/rms 逐帧序列)

   后两处是审计实测补上的:原先只删前两处,于是用户点「不留存」、界面报成功、
   场次目录没了 —— 而一份**正文里写着这个 sid** 的报告还躺在 `data/output/`
   里(实测 42 份 / 17 个 sid)。docstring 说的是「当它没发生过」,那就得算派生品。
   钉子:`tests/test_session_purge_leaves_no_trace.py`。

⚠️ **仍然不碰**的两类,各有理由:
   · `data/logs/jingxin.log*` —— 服务日志,按行交织了所有场次(实测 252 个 sid
     在同几份文件里),删一行就得重写整份文件;要单独裁定。
   · `NONE` 桶 —— 见 `purge_session` 里的显式拒绝。

⚠️ **这里没有回收站,是刻意的**。与 `/trash`(移进回收站)的分工:
   · `trash` —— 素材页上翻**旧**素材时点的「删除」。误点的代价必须可救。
   · `purge` —— 刚录完那一场的「不留存」/「不同意录制」。语义是"当它没发生过";
     往回收站里留一份等于**还是留存了**,与那个语义自相矛盾。
   两条路由**两个明确的按钮**分开,不靠人记得点哪个。

⚠️ 安全边界:本模块假定调用方已经做过鉴权。面板那条有管理员密码;
   语音服务那条**没有**(与它现有的 `/session/{sid}/label`、`/question` 一致 ——
   那些端点本来就能改身份数据)。服务只绑 loopback、经 tailscale serve 发布,
   边界是 tailnet。这一条由使用者确认过。
"""

import os
import re
import shutil
from pathlib import Path

import media_retention as mr

# 铸出来的 sid 形态。**比 `mr.validate_session_id` 严** —— 那个是给写侧用的
# (要放行 `NONE` 之类),这里只认真正的场次,`NONE` 另有明确处置(见下)。
SID_PAT = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{4}$")

# ⚠️ **另一个**形态。`SID_PAT` 带 `^…$` 锚点,是给"这个字符串整体是不是一个 sid"
#    (`fullmatch`)用的;拿它去 `findall` 扫一份 HTML 会**一个都找不到** —— 锚点要求
#    匹配落在整份文档的首尾。2026-09-28 实测踩到:报告索引因此恒为空,而界面上
#    看到的是"所有场次都没有报告",不是报错。(grep 那次没锚点,所以找得到 ——
#    同一个 sid,两个模式,一个命中一个不命中。)
SID_IN_TEXT = re.compile(r"\d{8}_\d{6}_[0-9a-f]{4}")
NONE_BUCKET = "NONE"

# 按 sid 落盘的日志前缀。**只有这四个** —— `data/logs/interview/` 下面那批
# `assessment_note_<时间戳>.csv` 不带 sid,认不出属于哪一场,所以不碰。
LOG_PREFIXES = ("face_au_log", "gesture_emotion_log",
                "interview_emotion_log", "research_emotion_log")


def default_logs_dir() -> Path:
    """日志落点。默认与本模块同级(仓库根)的 `data/logs` —— 两个服务都在同一个仓
    里跑,所以这就是它们真正写的地方(`face/gesture/voice` 三个 config 各自算出来
    的也是它)。调用方可以显式覆盖(面板就是把它自己那份配置传进来的)。"""
    return Path(__file__).parent / "data" / "logs"


def default_output_dir() -> Path:
    """报告与图表的落点(`report_frontend.report_generator` 的默认 `output_dir`)。"""
    return Path(__file__).parent / "data" / "output"


def default_series_dir() -> Path:
    """`report_frontend.voice_series` 写 `<sid>.npz`(f0/rms 逐帧序列)的地方。"""
    return Path(__file__).parent / "data" / "l0_series"


def log_csvs(session_id: str, *, logs_dir: str | os.PathLike | None = None) -> list[Path]:
    """这一场的日志 CSV。名字是拼的,但 sid **先过了正则**。"""
    if not SID_PAT.fullmatch(session_id):
        return []
    d = Path(logs_dir) if logs_dir else default_logs_dir()
    out = []
    for prefix in LOG_PREFIXES:
        p = d / f"{prefix}_{session_id}.csv"
        if p.is_file():
            out.append(p)
    return out


def purge_session(session_id: str, *, logs_dir: str | os.PathLike | None = None,
                  output_dir: str | os.PathLike | None = None,
                  series_dir: str | os.PathLike | None = None) -> dict:
    """真删一场。返回 `{"session_dir": …, "logs": [...], "reports": [...],
    "charts": [...], "series": [...]}`。

    `session_id` 不合法 → `ValueError`;目录、日志、报告、序列**全都不存在**
    → `FileNotFoundError`。
    """
    if session_id == NONE_BUCKET:
        # NONE 桶里是**所有没带 session_id 的请求**的素材,不属于任何一场。
        # 「本场作废」只会对着刚铸的号说,不可能是它;允许删它等于给了一个
        # 一次抹掉整个无主素材堆的按钮。
        raise ValueError("NONE 桶不是一场,不允许 purge(要清理请走回收站)")
    if not SID_PAT.fullmatch(session_id):
        raise ValueError(f"不是合法的 session_id:{session_id!r}")

    # ⚠️ 目录解析**复用** `media_retention`(它认得 `<标签>__<sid>` 与旧的
    #    `root()/<sid>`),不自己再拼一遍 —— 本仓在「四个地方各自算过一遍目录」
    #    上栽过,失效形态是素材静默写不进去。
    d = mr.resolve_recording_dir(session_id)
    root = mr.root().resolve()
    resolved = d.resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise ValueError(f"解析出的目录在录制根之外:{d}")

    out_dir = Path(output_dir) if output_dir else default_output_dir()
    ser_dir = Path(series_dir) if series_dir else default_series_dir()

    # ★ **派生品也是这一场的数据**。docstring 的语义是「当它没发生过」,而实测
    #   purge 原先只碰两处,盘上还留着 42 份**正文带 sid** 的报告(17 个不同 sid)、
    #   它们引用的图表、以及 `<sid>.npz` 逐帧序列。用户点「不留存」、界面报成功、
    #   场次目录没了 —— 但报告还在。钉子:`tests/test_session_purge_leaves_no_trace.py`。
    reports = _reports_naming(session_id, out_dir)
    charts = _charts_only_of(reports, out_dir)

    removed = {"session_dir": None, "logs": [], "reports": [], "charts": [], "series": []}
    if d.is_dir():
        shutil.rmtree(d)
        removed["session_dir"] = d.name
    for p in log_csvs(session_id, logs_dir=logs_dir):
        p.unlink()                      # 删不掉就让异常冒上去 —— 不许"看着成功"
        removed["logs"].append(p.name)
    for p in reports:
        p.unlink()
        removed["reports"].append(p.name)
    for p in charts:
        p.unlink()
        removed["charts"].append(p.name)
    npz = ser_dir / f"{session_id}.npz"
    if npz.is_file():
        npz.unlink()
        removed["series"].append(npz.name)

    if not any(removed[k] for k in removed):
        raise FileNotFoundError(f"没有这一场:{session_id}")
    # ★ **删完要留印记**。不留的话,后续任何一次写入都会把这个 sid 的目录重新建出来
    #   (2026-09-28 实测:删完之后又来了一次 `/question`,于是录制根下多了一个只含
    #   `questions.jsonl` 的裸目录,而素材列表照常把它列成一场)。
    mr.mark_purged(session_id)
    removed["tombstone"] = True
    return removed


# 报告里引用图表的那条链是 `<iframe src="radar_chart_<epoch>.html">`。
# ⚠️ 只认这三种前缀。实测 596 份非报告 HTML **没有一份正文含 sid** —— 所以图表
#    只能顺着报告找,不能反过来"扫到哪个删哪个"。
CHART_REF = re.compile(r"(?:radar_chart|evidence_\w+|gaze_trajectory)_\d+\.html")
REPORT_GLOB = "Research_Assessment_Report_*.html"


def _reports_naming(session_id: str, out_dir: Path) -> list[Path]:
    """`out_dir` 里**正文点名了这个 sid** 的报告。

    ⚠️ 判据是**正文里真的出现这个 sid**(`SID_IN_TEXT`,无锚点),不是文件名、
    也不是"最近一份"。实测 48 份报告里有 6 份正文**不含任何 sid**(旧格式 /
    生成时没拿到场次)—— 认不出归属的东西**不许猜**,猜错就是删别人的报告。
    """
    if not out_dir.is_dir():
        return []
    hits = []
    for p in sorted(out_dir.glob(REPORT_GLOB)):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue                    # 读不了就不动它 —— 宁可不删,不许删错
        if session_id in SID_IN_TEXT.findall(txt):
            hits.append(p)
    return hits


def _charts_only_of(reports: list[Path], out_dir: Path) -> list[Path]:
    """这些报告引用的图表,**减去**还被别的(幸存的)报告引用的那些。

    实测当前 288 份被引用的图表**没有一份被两个 sid 共用**,所以"减去"这一步
    现在恒不生效 —— 但少它就会在报告层哪天改成共用图表时删掉别人报告里的图,
    那种坏法在界面上只表现为破图,很难归因。
    """
    if not reports:
        return []
    wanted: set[str] = set()
    for p in reports:
        wanted |= set(CHART_REF.findall(p.read_text(encoding="utf-8", errors="ignore")))
    if not wanted:
        return []

    doomed = {p.name for p in reports}
    still_referenced: set[str] = set()
    for p in out_dir.glob(REPORT_GLOB):
        if p.name in doomed:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        still_referenced |= set(CHART_REF.findall(txt)) & wanted

    out = []
    for name in sorted(wanted - still_referenced):
        p = out_dir / name
        if p.is_file():
            out.append(p)
    return out
