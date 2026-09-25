# session_meta.py
"""阶段 A 元数据层:题目时刻台账 + `meta.json` 的模板与校验(spec §5.5 / §5.6)。

与 `media_retention.py` **分开**的理由:那个管的是**媒体字节**,这里管的是
**元数据**(题目时刻、人填的协变量)。两者落点也不同 —— 媒体在 `media/` 下面,
这里两样都在**会话根**。

它为什么必须在录制之前就位(spec §3.9 / §5.5):`response_latency` 的分子是
「首次开口墙钟 − `ask_end`」,而**录完就再也补不回来** —— 没有 `questions.jsonl`
的场次永久没有这个量。同理,该场的光照/增益/面试官评分都是"当时那一刻的状态",
不是稳定属性。

## `qid` 是什么(本题库没有题目 id)

实测:题库是**纯字符串列表**(`voice_interaction/pipeline/assessment_pipeline.py:101`、`:315`),
**题目没有 id**。而 spec §5.6 的请求体同时要 `qid` 与 `index`。所以:

- `qid` = **题目文本原文** —— 唯一、稳定、跨会话可归组;
- `index` = 该题在**本次会话**里的 0 基序号 —— 它标的是**顺序**,不是身份。

存文本**不违反** M1 的"原句不进仓库"边界:那条管的是**候选人说的话**;题目是这个
系统的固定输入,而且 `questions.jsonl` 落在 `~/shared`(**仓库外**)。
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from media_retention import root as _recordings_root, validate_session_id

QUESTIONS_FILENAME = "questions.jsonl"
META_FILENAME = "meta.json"

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _session_lock(session_id: str) -> threading.Lock:
    """逐会话锁。语义与 `transcript_store._session_lock` 相同:读-改-写必须串行,
    否则并发重试会拿旧快照把新写入整个盖掉,而且丢得无声无息。
    """
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(session_id, threading.Lock())


def _reset_for_tests() -> None:
    with _LOCKS_GUARD:
        _LOCKS.clear()


def _session_dir(session_id: str, *, create: bool) -> Path:
    """会话根目录。`create=False` 时**不建** —— 读侧不该有副作用。

    为什么不用 `media_retention.recording_dir`:那个函数**总是建目录**,
    于是"读一场根本不存在/没报过题的会话"会在盘上留下一个空会话目录。
    """
    sid = validate_session_id(session_id)
    d = _recordings_root() / sid
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def questions_path(session_id: str) -> Path:
    return _session_dir(session_id, create=False) / QUESTIONS_FILENAME


def _unreadable_line_count(path: Path) -> int:
    """这个 JSONL 里有多少行**解析不了**(被写碎 / 被人手改坏)。

    为什么要数而不是直接抛(终局复核 I2):`media_retention.py` 记着真实会话那
    390 行里**碎了 77 行**(三个进程共写一个账本的后果)。收尾对账的立身之本就是
    "只报不抛" —— 它崩了,operator 连"这场缺什么"都看不到,比缺更糟。
    """
    if not path.exists():
        return 0
    bad = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
        except Exception:
            bad += 1
    return bad


def read_questions(session_id: str) -> list[dict]:
    """本会话已上报的题目窗口。文件不存在 → `[]`(不是错误)。

    **解析不了的行跳过,不抛** —— 与 `media_retention.degraded_reasons` 同一条口径。
    (碎行数由 `check_session` 通过 `_unreadable_line_count` 报出来,不在这里吞掉。)
    `upsert_question` 随后整份重写该文件,所以碎行会被顺手清掉。
    """
    p = questions_path(session_id)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# 墙钟量程闸的容差(终局复核 I1)。1 天:报告是**当场**推的,离"现在"远过这个数
# 就一定不是墙钟秒。放宽到一天是为了容下"录完过一会儿才补推"这种正常情形。
_WALL_CLOCK_TOLERANCE_SEC = 86400.0


def _validate_window(qid: Any, index: Any, ask_start: Any, ask_end: Any) -> None:
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError(f"qid 必须是非空字符串(题目文本原文),收到 {qid!r}")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"index 必须是非负整数,收到 {index!r}")
    now = time.time()
    for name, v in (("ask_start", ask_start), ("ask_end", ask_end)):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name} 必须是墙钟秒数(数字),收到 {v!r}")
        if not v > 0:
            raise ValueError(f"{name} 必须是正数,收到 {v!r}")
        # 量程闸:光验"是个正数"是不够的 —— 毫秒与会话内相对时钟**都是正数**。
        # 而 spec §5.6 那句"两个基不要混"只有在这里拦得住(采集端是唯一的拦截点,
        # 这一场录完就补不回来)。两种错法定量后果一样:response_latency 会算出
        # ≈ −1.76e12 s 这种**看着像个数、其实毫无意义**的值,不会报任何错。
        if abs(v - now) > _WALL_CLOCK_TOLERANCE_SEC:
            off_days = abs(v - now) / 86400.0
            raise ValueError(
                f"{name}={v} 离此刻 {off_days:.1f} 天,不像是**墙钟秒** —— "
                f"要么是毫秒(JS `Date.now()` 的量纲,大 1000 倍),"
                f"要么是会话内相对时钟(M2.5 那个基)。spec §5.6 要的是墙钟秒;"
                f"两个基混了以后 response_latency 会变成一个看着正常、其实无意义的数")
    if ask_end <= ask_start:
        raise ValueError(
            f"ask_end({ask_end}) 必须晚于 ask_start({ask_start})—— "
            f"零长或倒挂的提问窗口会让 response_latency 变成无意义的数")


def upsert_question(session_id: str, *, qid: str, index: int,
                    ask_start: float, ask_end: float,
                    source: str = "") -> dict:
    """记一道题的提问窗口。同一 `qid` **覆盖**前一次(spec §5.6 的"以后来的为准"、
    §7.6.2 的"落盘的只有后一个")。

    为什么是 upsert 而不是纯追加:下游(M3 的 `response_latency`)要的就是
    **一题一个时间窗**;留重复行等于把判重推给下游。
    读-改-写在**本会话的锁**内完成。
    """
    _validate_window(qid, index, ask_start, ask_end)
    sid = validate_session_id(session_id)
    rec = {"qid": qid, "index": index,
           "ask_start": float(ask_start), "ask_end": float(ask_end),
           "source": source, "reported_at_wall": time.time(),
           "session_id": sid}
    with _session_lock(sid):
        p = _session_dir(sid, create=True) / QUESTIONS_FILENAME
        rows = [r for r in read_questions(sid) if r.get("qid") != qid]
        rows.append(rec)
        rows.sort(key=lambda r: (r["index"], r["qid"]))
        # 写临时文件再 replace:进程被杀时不会留下半行(单文件、整份重写,
        # 所以不需要 media_retention 那条「追加必须单次 os.write」的讲究)
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")
        os.replace(tmp, p)
    return rec


# ---------------------------------------------------------------- 阶段 A 元数据

TEMPLATE_PATH = Path(__file__).resolve().parent / "session_meta_template.json"

# 必填项(点分路径)。**依据是录制需求 §3.2 那张表**(每项为何必须,见那张表的"为什么"列):
#
#   人的生理属性 —— `pitch_mean` 的 **ICC = 0.723**,它主要是解剖常量,
#                   不加协变量**可能让分数部分成为性别探测器**;
#   设备 / 取景   —— `energy` 的 **ICC = 0.654**,是设备增益/距离代理;
#   面试官评分    —— 审查 Q3(f) 称其为"你现在最该补的真值,成本最低、最贴用途";
#   题目 / 难度   —— 审查称"最明显的遗漏"(数据集里 74 个不同题目,协变量里一个都没有);
#   知情同意      —— 审查 §7.4 第 1 条,法定必留(类型/用途/留存期/能否拒绝/申诉渠道)。
#
# ⚠️ **不含候选人自评量表** —— 那是 M5 标定用的真值,受《科技伦理审查办法(试行)》
#    强制前置(spec §1.2),不在阶段 A。
#
# `questions` 只校验**非空**。逐条 `difficulty` 的空白不在这里判 —— 本场实问题数
# 服务端不知道(题目可以中途结束),逐题覆盖由 `--check-session` 的题号对账负责。
# 这一条列进来是为了拦住"整个题库块忘了填"这个真实失败形态(§3.2 明列它必填)。
META_REQUIRED: tuple[str, ...] = (
    "candidate.sex", "candidate.age",
    "candidate.native_language", "candidate.dialect_region",
    "capture.device", "capture.resolution", "capture.camera_distance_cm",
    "capture.lighting", "capture.mic_gain_db",
    "interviewer_ratings.logical_thinking",
    "interviewer_ratings.communication",
    "interviewer_ratings.confidence",
    "consent.archived",
    "questions",
)


def meta_path(session_id: str) -> Path:
    return _session_dir(session_id, create=False) / META_FILENAME


def _get_path(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _is_blank(v: Any) -> bool:
    """`None` / 空串 / 空列表 / 空字典 / **`False`** 都算**没填**。

    最后那一条是给 `consent.archived` 的:它是布尔必填项,而模板里的默认值就是
    `false`,意思是"知情同意**还没归档**" —— 那正是我们要它非 `false` 的原因。
    不把 `False` 当缺的话,一份**根本没做知情同意**的场次会通过校验。
    只判"键在不在"更糟:一份全空模板会整份通过(而那正是模板刚生成时的样子)。
    """
    if v is None or v is False:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def read_meta(session_id: str) -> dict | None:
    p = meta_path(session_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def missing_meta_fields(session_id: str) -> list[str]:
    """本场 `meta.json` 还缺哪些必填项(空 = 齐了)。

    **meta.json 不存在 = 全部都缺** —— 不是异常。收尾对账要能报出这件事,
    而不是自己先崩掉。
    """
    meta = read_meta(session_id)
    if meta is None:
        return list(META_REQUIRED)
    return [p for p in META_REQUIRED if _is_blank(_get_path(meta, p))]


def write_template(session_id: str) -> Path:
    """把模板铺进会话根,`session_id` 与 `recorded_at` 先填好。

    让使用者**在原地填空**,而不是从别处抄一份 —— 少一步就少一次漏填。
    **已存在则抛**:不覆盖已填过的内容(那份是当场记的,补不回来)。
    """
    from datetime import datetime
    sid = validate_session_id(session_id)
    p = _session_dir(sid, create=True) / META_FILENAME
    if p.exists():
        raise FileExistsError(f"{p} 已存在 —— 不覆盖(要重来请先自己移走)")
    payload = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    payload["session_id"] = sid
    payload["recorded_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 会话收尾对账


def _media_counts(session_id: str) -> tuple[dict[str, int], int, int]:
    """按**账本行数**数各模态的素材件数(不是数盘上文件)。

    为什么以账本为准:账本记的是"服务端**收到了**什么",而盘上文件会被覆盖
    (`camera.webm`)或被人手动动过。spec §7.7 的第一条判据正是
    "帧数 == 服务端实际收到的帧数(以 retention.jsonl 的行数为准)"。

    返回 `(各模态件数, 原生视频的**最大字节数**, 解析不了的行数)`。

    - **只算 `bytes > 0` 的行**(终局复核 I3):0 字节的 `camera.webm` 是"看着像有、
      其实没有"的录像。件数照数(它确实是个文件),但"有没有视频"看字节。
    - 逐行 try/except(终局复核 I2):碎行只跳过并计数,不抛 —— 见
      `_unreadable_line_count` 的说明。
    """
    from media_retention import ledger_files
    counts: dict[str, int] = {}
    video_bytes = 0
    unreadable = 0
    for book in ledger_files(session_id):
        for line in book.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                unreadable += 1
                continue
            if rec.get("kind") in ("frame", "raw", "converted", "video"):
                if not rec.get("bytes"):
                    continue          # 0 字节不记材料(它的 kind 是真的,内容不是)
                key = rec.get("modality") or rec.get("kind")
                counts[key] = counts.get(key, 0) + 1
                if rec.get("kind") == "video":
                    video_bytes = max(video_bytes, int(rec["bytes"]))
    return counts, video_bytes, unreadable


def check_session(session_id: str,
                  expected_questions: int | None = None) -> dict:
    """会话收尾对账。**任何缺项都只报不抛** —— 分析结果不因元数据缺失而作废
    (spec §6:元数据缺项不阻断分析,但必须让人当场知道)。

    为什么不硬塞进 `transcript_store.refresh_manifest`(spec §8.5 要求先确认它的形状):
    实测那个函数只按 `payload["logs"][mod]["expected_file"]` 在 `log_dir` 下判存在
    —— 它管的是**仓库内 `data/logs` 的三份 CSV**,与"`~/shared` 里的媒体齐不齐、
    `meta.json` 填没填"是**两套不同的期望**。硬塞进去要么把它改成四不像,
    要么让它的参数语义漂移。**另写。**
    """
    from media_retention import degraded_reasons
    sid = validate_session_id(session_id)
    rows = read_questions(sid)
    reported = sorted(r["index"] for r in rows if isinstance(r.get("index"), int))
    missing_q: list[int] = []
    if expected_questions is not None:
        missing_q = [i for i in range(expected_questions) if i not in set(reported)]
    counts, video_bytes, unreadable = _media_counts(sid)
    unreadable += _unreadable_line_count(questions_path(sid))
    return {
        "session_id": sid,
        "media": counts,
        "video": video_bytes > 0,
        "video_bytes": video_bytes,
        "degraded": degraded_reasons(sid),
        "missing_meta": missing_meta_fields(sid),
        "reported_questions": reported,
        "missing_questions": missing_q,
        "unreadable_lines": unreadable,
    }


def closeout_verdict(result: dict,
                     expected_questions: int | None = None) -> tuple[bool, list[str]]:
    """把对账结果折成一个**判断 + 逐条理由**(终局复核 I4)。

    为什么单独一个函数:原先那个判断是内联在 `__main__` 里的,于是
    "一题都没报"这种情形**没法被测试钉住** —— 而它恰恰是最危险的一种绿灯:
    `response_latency` 是这个里程碑存在的理由,一题都没报时它必然拿不到,
    收尾却会印 `✅ 齐了`。抽出来之后,CLI 与测试走同一条判断。
    """
    reasons: list[str] = []
    if result.get("degraded"):
        reasons.append(f"留存降级 {len(result['degraded'])} 条")
    if result.get("missing_meta"):
        reasons.append(f"meta.json 缺 {len(result['missing_meta'])} 项")
    if result.get("missing_questions"):
        reasons.append(f"漏报题号 {result['missing_questions']}")
    if result.get("unreadable_lines"):
        reasons.append(f"账本有 {result['unreadable_lines']} 行解析不了(被写碎或手改过)")
    if not result.get("video"):
        reasons.append("没有可用的原生视频(缺失,或存下来是 0 字节)")
    # 一题都没报时**不许绿**:没给 --expected-questions 就绿灯等于替 operator
    # 把"response_latency 拿不到"这件事放过去了。
    if not result.get("reported_questions") and expected_questions is None:
        reasons.append("一题都没报(questions.jsonl 空)—— response_latency 无来源;"
                       "确实没有题目时才可忽略,否则补推或加 --expected-questions")
    return (not reasons), reasons


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="阶段 A 元数据层:模板与收尾对账")
    ap.add_argument("--write-template", metavar="SESSION_ID",
                    help="把 meta.json 模板铺进该会话目录(已存在则不动)")
    ap.add_argument("--check-session", metavar="SESSION_ID",
                    help="对账:媒体件数 / 降级 / meta 缺项 / 题目漏报")
    ap.add_argument("--expected-questions", type=int, default=None,
                    help="本场实问题数,用来点名漏报的题(缺省 = 不对账题目)")
    args = ap.parse_args()

    if args.write_template:
        try:
            p = write_template(args.write_template)
        except FileExistsError:
            # 3–5 场录制里一定会撞到(复核 Minor 1:help 写的是"已存在则不动",
            # 而实现是抛)。人当场看到的是"已存在,未改动"比 traceback 有用。
            print(f"ℹ️ {args.write_template} 已有 meta.json,未改动 —— "
                  f"要重来请先自己移走那份(别覆盖:里面是当场记的)")
            sys.exit(0)
        print(f"✅ 模板已铺到 {p} —— 请**当场**逐条填,不要事后补")
        sys.exit(0)

    if args.check_session:
        r = check_session(args.check_session, args.expected_questions)
        print(f"会话 {r['session_id']}")
        print(f"  素材件数(按账本):{r['media'] or '(一件都没有)'}")
        print(f"  前端原生视频:"
              f"{'有,' + str(r['video_bytes']) + ' 字节' if r['video'] else '**没有**'}")
        print(f"  留存降级:{r['degraded'] or '无'}")
        print(f"  meta.json 缺项:{r['missing_meta'] or '无'}")
        print(f"  已报题号:{r['reported_questions'] or '(一题都没报)'}")
        if args.expected_questions is not None:
            print(f"  漏报题号:{r['missing_questions'] or '无'}")
        if r["unreadable_lines"]:
            print(f"  ⚠️ 账本有 {r['unreadable_lines']} 行解析不了(被写碎或手改过)")
        ok, reasons = closeout_verdict(r, args.expected_questions)
        print("✅ 齐了" if ok else "⚠️ 没齐 —— 这一场缺的东西补不回来:")
        for why in reasons:
            print(f"    · {why}")
        sys.exit(0 if ok else 1)

    ap.print_help()
    sys.exit(2)
