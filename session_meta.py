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


def read_questions(session_id: str) -> list[dict]:
    """本会话已上报的题目窗口。文件不存在 → `[]`(不是错误)。"""
    p = questions_path(session_id)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _validate_window(qid: Any, index: Any, ask_start: Any, ask_end: Any) -> None:
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError(f"qid 必须是非空字符串(题目文本原文),收到 {qid!r}")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"index 必须是非负整数,收到 {index!r}")
    for name, v in (("ask_start", ask_start), ("ask_end", ask_end)):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name} 必须是墙钟秒数(数字),收到 {v!r}")
        if not v > 0:
            raise ValueError(f"{name} 必须是正数,收到 {v!r}")
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


def _media_counts(session_id: str) -> dict[str, int]:
    """按**账本行数**数各模态的素材件数(不是数盘上文件)。

    为什么以账本为准:账本记的是"服务端**收到了**什么",而盘上文件会被覆盖
    (`camera.webm`)或被人手动动过。spec §7.7 的第一条判据正是
    "帧数 == 服务端实际收到的帧数(以 retention.jsonl 的行数为准)"。
    """
    from media_retention import ledger_files
    counts: dict[str, int] = {}
    for book in ledger_files(session_id):
        for line in book.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") in ("frame", "raw", "converted", "video"):
                key = rec.get("modality") or rec.get("kind")
                counts[key] = counts.get(key, 0) + 1
    return counts


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
    reported = sorted(r["index"] for r in rows)
    missing_q: list[int] = []
    if expected_questions is not None:
        missing_q = [i for i in range(expected_questions) if i not in set(reported)]
    counts = _media_counts(sid)
    return {
        "session_id": sid,
        "media": counts,
        "video": counts.get("camera", 0) > 0,
        "degraded": degraded_reasons(sid),
        "missing_meta": missing_meta_fields(sid),
        "reported_questions": reported,
        "missing_questions": missing_q,
    }


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
        p = write_template(args.write_template)
        print(f"✅ 模板已铺到 {p} —— 请**当场**逐条填,不要事后补")
        sys.exit(0)

    if args.check_session:
        r = check_session(args.check_session, args.expected_questions)
        print(f"会话 {r['session_id']}")
        print(f"  素材件数(按账本):{r['media'] or '(一件都没有)'}")
        print(f"  前端原生视频:{'有' if r['video'] else '**没有**'}")
        print(f"  留存降级:{r['degraded'] or '无'}")
        print(f"  meta.json 缺项:{r['missing_meta'] or '无'}")
        print(f"  已报题号:{r['reported_questions'] or '(一题都没报)'}")
        if args.expected_questions is not None:
            print(f"  漏报题号:{r['missing_questions'] or '无'}")
        ok = not (r["degraded"] or r["missing_meta"] or r["missing_questions"])
        print("✅ 齐了" if ok else "⚠️ 上面点出来的项没齐 —— 这一场缺的东西补不回来")
        sys.exit(0 if ok else 1)

    ap.print_help()
    sys.exit(2)
