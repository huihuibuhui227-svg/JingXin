# session_purge.py
"""把一场素材从盘上**彻底**抹掉。

**两个调用方共用这一份**:面板的管理员端点(`recordings_browser`)与语音服务的
`POST /session/{sid}/discard`(录制页在「不留存 / 不同意」那条路上调它)。
写两遍的代价本仓太熟了 —— 一处改了另一处没改,而失效形态是"删了一半":
场次目录没了、日志 CSV 还在,报告层于是看到一场**有日志没素材**的会话,
看起来只是"这一场没采到",不是"这一场被作废了"。

⚠️ **删的是两处**:
   · 场次目录(帧 / 音频 / 留存账本 / 标注)
   · `data/logs/` 里那三份(或更少)按 sid 落盘的 CSV

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


def purge_session(session_id: str, *, logs_dir: str | os.PathLike | None = None) -> dict:
    """真删一场。返回 `{"session_dir": …, "logs": [...]}`。

    `session_id` 不合法 → `ValueError`;目录与日志都不存在 → `FileNotFoundError`。
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

    removed = {"session_dir": None, "logs": []}
    if d.is_dir():
        shutil.rmtree(d)
        removed["session_dir"] = d.name
    for p in log_csvs(session_id, logs_dir=logs_dir):
        p.unlink()                      # 删不掉就让异常冒上去 —— 不许"看着成功"
        removed["logs"].append(p.name)

    if removed["session_dir"] is None and not removed["logs"]:
        raise FileNotFoundError(f"没有这一场:{session_id}")
    # ★ **删完要留印记**。不留的话,后续任何一次写入都会把这个 sid 的目录重新建出来
    #   (2026-09-28 实测:删完之后又来了一次 `/question`,于是录制根下多了一个只含
    #   `questions.jsonl` 的裸目录,而素材列表照常把它列成一场)。
    mr.mark_purged(session_id)
    removed["tombstone"] = True
    return removed
