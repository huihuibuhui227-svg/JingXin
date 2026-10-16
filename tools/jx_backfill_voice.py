"""拿**留存音频段**把缺失的语音日志离线补出来(甲-4 的下游,2026-10-03)。

## 为什么要补

`data/logs/` 里语音日志的分布是断的 —— 实测(2026-10-03):

| 场次 | 留存回答段 | `interview_emotion_log_<sid>.csv` |
|---|---|---|
| `20260926_153202_2b11` / `153854_1592` / `155559_caf0`(**三场正式素材**) | 各 8 段 | **文件根本不存在** |
| `20260925_230809_0947` | 4 段 | 不存在 |
| `20260927_195323_35e1` / `195528_afb2` / `212143_95a8` | 2 / 6 / 8 段 | 2 / 6 / 8 行(活跑写的) |

三场**正式**素材的语音日志不在这台机器上(它们在开发机的 `~/shared` 时期产生,
随"以服务器上的为准"那次统一留在了那边;`~/shared` 下**一个 `interview_emotion_log_*`
都没有**,实测 `find` 过)。**音频段在**,所以那些行是**可重算的** —— 这正是本工具。

## 它做什么

对每一场:用 `experiments/replay_retained.py` 的 `replay_voice`(它走的是**活路径那个
真产出方** `voice_interaction.api.app.prosody_features_from_pcm`)把留存回答段重跑一遍,
产出**符合 `VoiceLogger` 32 列契约**的一张表,写进 `data/logs/interview_emotion_log_<sid>.csv`。

⭐ 之所以能这么做,是因为甲-4 先把重放腿的产物做成了**活路径那张表**:
实测场 `..._95a8` 逐列比,**表头 32 列逐字一致、24/32 列逐字相同**(含 `reaction_time`)、
未登记差异 0。没有那一步,补出来的文件喂给报告只会让语音系的槽**成片掉光**。

## 它**不**做什么(边界,写清楚了才好用)

* **绝不覆盖已有行**:活的日志有 ≥1 行数据即**跳过**。本工具补的是"当时没落下来"的那些行,
  不是"重算一遍看看一样不一样"(那是 `replay_retained.py` 的验收用途)。
* **不碰没有留存音频段的场次**。录制语料(26 场"老师面试学生")的音频只在
  `media/camera.webm` 里,`media/audio/` 是空的 ⟹ 本工具一场都补不了。
  那是另一件事(要从视频抽音轨 + 定"一行是什么"),而且**音轨里师生都在**,
  见下面「待裁定」。
* **不跑 ASR**。所以 `n_chars` / `chars_per_sec` / `connective_density` /
  `connective_density_std` / `n_rows` 这 5 列**写出来是空的** ——
  它们要文本,而文本来自 ASR。留空不编(登记在
  `voice_interaction/utils/replay_contract.py` 的 `NOT_WRITTEN_BY_REPLAY`)。
* **不写 `is_valid`**。活路径恒写 `True` 是调用方的占位判断,重放不复制一个它没做的判断。

## 溯源(为什么必须留痕)

补出来的 CSV 与活跑写的 CSV **同形同名**,看文件本身分不出来。所以:

* 顶掉任何既有文件之前**先备份**(`.csv.voice_backfill.bak`,再跑 `.1.bak`…… 首个备份永不覆盖,
  与 `jx_backfill_face.py` 同一套);
* 每次处理都记进 `data/logs/.voice_backfill_manifest.json`:哪一场、从哪份账本、
  用了哪些段、写了几行、哪些列留空、以及**两条推断规则**(到达时刻取 `raw` 记录、
  题号按 `ask_end` 推)—— 将来有人问"这几行是活跑的还是补的",清单能回答。

## 用法

    PY=~/anaconda3/envs/jingxin/bin/python
    $PY tools/jx_backfill_voice.py --dry-run      # 先看会补哪几场、各补几行
    $PY tools/jx_backfill_voice.py                # 补(幂等:已有行的场次跳过)
    $PY tools/jx_backfill_voice.py --session 20260926_153202_2b11

⚠️ 录制根由 `JINGXIN_RECORDINGS_DIR` 决定(与 `media_retention.root()` **同一个**开关)——
本工具把它与 `--sessions-dir` **绑成一处**,免得"扫的目录"与"读 `questions.jsonl`
的目录"不是同一个(那会让 `reaction_time` 整组静默留空)。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# `replay_voice` 住在 `experiments/`(它被 git 跟踪),本工具**复用它**而不是再写一遍
# "读账本 → 喂真产出方 → 拼契约行" —— 那正是重放腿存在的意义。
sys.path.insert(0, str(REPO / "experiments"))

LOG_DIR = REPO / "data" / "logs"
MANIFEST = LOG_DIR / ".voice_backfill_manifest.json"

#: 备份后缀。与 `jx_backfill_face.py` 的 `.predrop.bak` 同一套「首个备份永不覆盖」,
#: 只是换个前缀 —— 两种回填的备份混在一起时还能一眼看出是谁留下的。
BACKUP_SUFFIX = ".voice_backfill"

#: 本工具依赖重放腿的**形状契约**;形状变了就必须重新验一遍"补出来的表与活跑同形"。
#: 记进清单,好让将来的人知道这些行是**哪一版规则**产出来的。
RULES_VERSION = "replay-contract/2026-10-03(到达时刻取 raw 记录;题号=最后一个 ask_end ≤ 到达)"


def voice_log_path(sid: str, log_dir: Path | None = None) -> Path:
    """日志路径。`log_dir` 可注入 —— 测试不许往真的 `data/logs/` 里写东西。"""
    return (log_dir or LOG_DIR) / f"interview_emotion_log_{sid}.csv"


def log_row_count(p: Path) -> int | None:
    """CSV 的数据行数(表头不算)。文件不存在 → `None`。"""
    if not p.is_file():
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def backup_path(out_csv: Path) -> Path:
    """给一份要顶掉的 CSV 选备份名 —— **首个备份永不覆盖**(与 `jx_backfill_face.py` 同)。"""
    bak = out_csv.with_suffix(f".csv{BACKUP_SUFFIX}.bak")
    if not bak.exists():
        return bak
    i = 1
    while True:
        alt = out_csv.with_suffix(f".csv{BACKUP_SUFFIX}.{i}.bak")
        if not alt.exists():
            return alt
        i += 1


def _ledger_records(session_dir: Path, kind: str = "converted") -> list[dict]:
    """读留存账本里某一 kind 的记录。**不要求文件还在** —— 这是"账本说发生过什么"。

    ⚠️ 为什么本工具要自己读一遍账本,而不是只用 `replay_retained.load_audio_segments`:
    那个函数**以盘上的文件为准**(它的定义就是"有没有素材"),而本工具还得分得清
    「本场**没答过题**」与「答过、但**盘上没了**」—— 这两件事的处置完全不同
    (前者无事可做,后者是素材缺失要报出来)。用同一个函数当唯一判据的话,
    第二种会被说成第一种,而那是一句**假话**。
    """
    import replay_retained

    led = session_dir / f"{replay_retained.LEDGER_PREFIX}.audio{replay_retained.LEDGER_SUFFIX}"
    if not led.is_file():
        return []
    out = []
    for line in led.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                     # 碎行跳过(与 load_audio_segments 同口径)
        if rec.get("kind") == kind:
            out.append(rec)
    return out


def answer_segments(session_dir: Path) -> list[dict]:
    """本场**提交的回答**段(`/interview/answer_audio`)的账本记录,按 sha256 去重。

    判据取自**账本**(不要求文件还在),理由见 `_ledger_records`。
    去重是按 sha256:**同一段回答常被留两份**(预览 `/asr` 一次 + 提交一次),
    字节逐字相同时哈希相同 ⟹ 不去重行数会翻倍。
    ⚠️ 只看 `converted`:那是端点在用的那份字节(`raw` 是上传的原容器,
    两者字节不同 ⟹ 哈希天然不同,不能互相 join,见「丙、19」)。
    """
    seen: set[str] = set()
    out: list[dict] = []
    for rec in _ledger_records(session_dir):
        if rec.get("source_endpoint") != "/interview/answer_audio":
            continue
        key = str(rec.get("sha256") or rec.get("file"))
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def candidate_sessions(root: Path, only: str | None = None) -> list[Path]:
    """候选场次 = 有 `media/audio/` 且账本里有**提交的回答**段。

    不候选的两种,理由不同,**都要能说出来**(调用方会打出来):
    · 只有 `/asr` 预览段的场次(实测 `..._2449` / `253a` / `90e7`):本场没答过题;
    · 根本没有 `media/audio/` 的场次(录制语料那 26 场):音频只在 `camera.webm` 里。
    """
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        sid = d.name.split("__")[-1]
        if only and sid != only:
            continue
        if not (d / "media" / "audio").is_dir():
            continue
        if answer_segments(d):
            out.append(d)
    return out


def backfill_one(session_dir: Path, dry_run: bool = False,
                 log_dir: Path | None = None) -> dict:
    """补一场。返回结果字典(`status` = `skip` / `dry-run` / `backfilled` / `failed`)。

    ⚠️ **顺序是有意的**:先看"已经有几行" ⟹ 有活行就**立刻**返回,连账本都不读。
    所以"绝不覆盖活行"这条不依赖任何下游判断(测试也不需要造音频就能验它)。
    """
    import replay_retained
    import session_meta

    sid = session_dir.name.split("__")[-1]
    out = voice_log_path(sid, log_dir)

    have = log_row_count(out)
    if have:                       # 有活行 ⟹ 一行都不动(本工具不覆盖)
        return {"session": sid, "status": "skip",
                "why": f"已有 {have} 行数据,不覆盖(要复核请用 replay_retained.py 比)"}

    segs = answer_segments(session_dir)
    if not segs:
        return {"session": sid, "status": "skip",
                "why": "账本里没有 `/interview/answer_audio` 段(只有预览 ⟹ 本场没答过题)"}
    on_disk, _rep = replay_retained.load_audio_segments(session_dir)
    if not on_disk:
        # 账本说答过、盘上一份都没有 —— 这是**素材缺失**,不是"没答过题"。
        return {"session": sid, "status": "skip",
                "why": f"账本记着 {len(segs)} 段提交的回答,但盘上一份都没有(素材缺失)"}

    questions = session_meta.read_questions(sid)
    with tempfile.TemporaryDirectory(prefix="voice_backfill_") as td:
        tmp = Path(td) / "rows.csv"
        n = replay_retained.replay_voice(              # 走真产出方,与验收那条腿**同一段代码**
            session_dir, tmp, answers_only=True, dedupe=True,
            questions=questions, session_id=sid)
        rows = log_row_count(tmp) if tmp.is_file() else 0
        if not rows:
            return {"session": sid, "status": "skip",
                    "why": f"重放出来 0 行({len(segs)} 段候选,盘上 {len(on_disk)} 段)"
                           f"—— 段格式没过闸,见上面的 [跳过] 行"}

        result = {
            "session": sid,
            "status": "dry-run" if dry_run else "backfilled",
            "rows": rows,
            "segments": len(segs),
            "questions": len(questions),
            "log_before": have,
            "rules": RULES_VERSION,
            "source": "media/audio/<n>_converted.wav(留存账本 converted 段)",
        }
        if dry_run:
            return result

        # 先备份再写:顺序不能反 —— 反了就是"覆盖了才发现该备份"。
        if out.is_file():
            bak = backup_path(out)
            shutil.move(str(out), str(bak))
            result["backup"] = str(bak.relative_to(REPO))
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(out))
    return result


def main() -> int:
    import media_retention

    ap = argparse.ArgumentParser(description="拿留存音频段补出缺失的语音日志(甲-4 下游)")
    ap.add_argument("--sessions-dir", default=None,
                    help="录制根。默认 = `media_retention.root()`(即 JINGXIN_RECORDINGS_DIR);"
                         "显式给了就把它设进环境变量,保证读 questions.jsonl 用的是同一个根")
    ap.add_argument("--session", default=None, help="只处理这一场(按 sid)")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 场(试跑用)")
    ap.add_argument("--dry-run", action="store_true", help="只报告会补哪几场,不写文件")
    args = ap.parse_args()

    if args.sessions_dir:
        os.environ["JINGXIN_RECORDINGS_DIR"] = args.sessions_dir
    root = media_retention.root()
    if not root.is_dir():
        print(f"✗ 录制根不存在:{root}", file=sys.stderr)
        return 2

    todo = candidate_sessions(root, only=args.session)
    if args.limit:
        todo = todo[: args.limit]
    print(f"录制根 {root}")
    print(f"候选 {len(todo)} 场(有留存回答段){'  [dry-run]' if args.dry_run else ''}\n")
    if not todo:
        print("没有候选场次。")
        return 0

    results = []
    for i, d in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {d.name[:56]}", flush=True)
        t0 = time.time()
        try:
            r = backfill_one(d, dry_run=args.dry_run)
        except Exception as e:                    # 失败要说出来,不吞
            r = {"session": d.name.split("__")[-1], "status": "failed",
                 "why": f"{type(e).__name__}: {e}"}
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        tag = {"skip": "·", "dry-run": "?", "backfilled": "✓", "failed": "✗"}[r["status"]]
        detail = (f"{r.get('rows')} 行(段 {r.get('segments')},题台账 {r.get('questions')})"
                  if r.get("rows") else (r.get("why") or ""))
        print(f"    {tag} {r['status']}: {detail}  ({r['seconds']}s)")

    if not args.dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        old = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
        for r in results:
            if r["status"] in ("backfilled", "failed"):
                old[r["session"]] = r
        MANIFEST.write_text(json.dumps(old, ensure_ascii=False, indent=2, sort_keys=True),
                            encoding="utf-8")
        print(f"\n清单已写入 {MANIFEST}")

    n_ok = sum(1 for r in results if r["status"] in ("backfilled", "dry-run"))
    n_skip = sum(1 for r in results if r["status"] == "skip")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'─' * 60}\n补出 {n_ok}   跳过 {n_skip}   失败 {n_fail}")
    if n_fail:
        print("失败清单(需要处理,不要当成成功):")
        for r in results:
            if r["status"] == "failed":
                print(f"  ✗ {r['session']}  {r.get('why')}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
