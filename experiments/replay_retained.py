"""把留存下来的帧/音频按**当时的时间戳**重放一遍 —— M2.6 的验收工具。

为什么需要一个专门的脚本(spec §7.1):留存的意义是"事后能重算"。要证明这一点,
就得拿盘上的帧重新跑一遍提取器,看结果是不是与**当场活跑**写下的 CSV 逐格相等。
相等 ⟹ 存的确实是我抽过的那些东西;不等 ⟹ 存错了(重编码?顺序错?时间戳造错了?)。

用法(WSL 侧):
    PY=~/miniconda3/envs/jingxin/bin/python
    $PY experiments/replay_retained.py --session-id 20260925_120000_aaaa \
        --out ~/shared/jingxin_recordings/20260925_120000_aaaa/replay_face.csv
然后把 replay_face.csv 与 data/logs/face_au_log_<sid>.csv 逐格比。

三路重放腿(2026-09-27 起):`--modality face|gesture|voice`。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from media_retention import LEDGER_PREFIX, LEDGER_SUFFIX, recording_dir   # noqa: E402


def _ledger_books(session_dir: Path) -> list[Path]:
    """本目录下的全部账本文件(每个写入者一个;也认旧的单文件形态)。"""
    return sorted(set(session_dir.glob(f"{LEDGER_PREFIX}.*{LEDGER_SUFFIX}"))
                  | set(session_dir.glob(f"{LEDGER_PREFIX}{LEDGER_SUFFIX}")))


def load_frames(session_dir: Path | str, modality: str = "face"
                ) -> tuple[list[tuple[Path, int]], dict]:
    """读出本会话该模态的帧,按序号排好,带上**账本里记的当时时间戳**。

    返回 `(帧列表, 缺口报告)`。**缺口报告不是装饰**,它是这个工具最容易骗人的地方:

      以**盘上的文件**为准,账本只用来取 `declared_ts`。进程被杀在"写了文件、还没写账"
      之间时,账本会缺行 —— 那些帧**盘上还在,但没有当时的时间戳**。§7.1 要求重抽必须
      喂当时那个值、**不许另编一个**(编了就不是"逐格相等"了),所以它们的正确处置是
      **被报出来**,而不是静默跳过。

    实测代价(2026-09-25 使用者第一场真会话):账本被跨进程追加写碎了 25%,187 个 face
    帧只有 105 帧还有可用时间戳 —— 而当时的写法只打印"重放 105 帧",看的人根本不知道
    少了 82 帧、更不知道原因在账本。

    返回的 report 含:`frames_on_disk` / `frames_with_ts` / `missing_ts` / `torn_lines`。
    """
    session_dir = Path(session_dir)

    # ① 以盘为准:帧文件本身才是"有没有素材"的判据
    on_disk = sorted((session_dir / "media" / modality).glob("*.jpg"))
    by_seq: dict[int, Path] = {}
    for p in on_disk:
        stem = p.stem
        if stem.isdigit():
            by_seq[int(stem)] = p

    # ② 账本只用来取 declared_ts(以及后续扩展字段)
    ts_by_file: dict[str, int] = {}
    torn = 0
    for book in _ledger_books(session_dir):
        for line in book.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                torn += 1                     # 碎行/半行 → 计数,不崩
                continue
            if rec.get("kind") != "frame" or rec.get("modality") != modality:
                continue
            if rec.get("declared_ts") is None:
                continue
            rel = str(rec.get("file") or "")
            if rel:
                ts_by_file[rel] = int(rec["declared_ts"])

    frames: list[tuple[Path, int]] = []
    missing: list[Path] = []
    for seq in sorted(by_seq):
        p = by_seq[seq]
        rel = f"media/{modality}/{p.name}"
        if rel in ts_by_file:
            frames.append((p, ts_by_file[rel]))
        else:
            missing.append(p)

    report = {
        "frames_on_disk": len(by_seq),
        "frames_with_ts": len(frames),
        "missing_ts": [p.name for p in missing],
        "torn_lines": torn,
    }
    return frames, report


def replay_face(session_dir: Path, out_csv: Path) -> int:
    """按留存帧重跑 VideoPipeline,把每帧的序列化结果写成 CSV。返回帧数。"""
    import cv2
    from face_expression.pipeline.video_pipeline import VideoPipeline

    frames, report = load_frames(session_dir, "face")
    print(f"[素材] 盘上 {report['frames_on_disk']} 帧,其中 {report['frames_with_ts']} 帧有时间戳;"
          f"账本碎行 {report['torn_lines']} 行")
    if report["missing_ts"]:
        # 不静默:这些帧的图还在,但当时的时间戳丢了 —— §7.1 不允许另编一个
        print(f"⚠️  {len(report['missing_ts'])} 帧**没有可用时间戳**,本次不重放;"
              f"前几个:{report['missing_ts'][:5]}", file=sys.stderr)
    pipeline = VideoPipeline(session_id=session_dir.name)
    rows = []
    for path, ts in frames:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[跳过] 读不出:{path}", file=sys.stderr)
            continue
        _obj, _mesh, features = pipeline.process_frame(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ts)
        if features:
            rows.append(features)
    pipeline.close()
    if rows:
        keys = sorted({k for r in rows for k in r})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def replay_gesture(session_dir: Path, out_csv: Path) -> int:
    """按留存帧重跑手势服务的那一条路,把每帧的**日志行**写成 CSV。返回帧数。

    ★ 为什么调 `gesture_app.process_frame` 而不是在这里自己拼一条(2026-09-27,B3):
    「重放」的全部意义是**走真产出方**。若本脚本自己写一遍"先喂谁、再喂谁、拿哪个键",
    它证的就是它自己抄的那份接线 —— 端点里漏传一个 `shoulder_width`、或 `world_results`
    的键名写错一个字母,这条腿照样绿,而活路径是空的(本项目已 5 次栽在"验证跑错了对象")。
    所以这里调的是**端点调的那个函数**,落盘也走**真的 `GestureLogger`**(列名映射只此一份)。

    ⚠️ 时间戳一律用账本里的 `declared_ts`(纪律 3):mediapipe 的 VIDEO 模式把时间戳当
    模型输入,喂墙钟会得出"推理不可复现"的**假**结论。
    """
    import cv2
    import importlib

    # ⚠️ 必须 `importlib.import_module`,不能 `import gesture_analysis.api.app as ...`:
    # 那个包 `__init__.py` 做了 `from .app import app` ⟹ 属性链上取到的是 **FastAPI 实例**,
    # 而这里要的是**模块**(端点在模块的 globals 里找 `get_or_create_analyzers`)。
    # `importlib.import_module` 走 `sys.modules`,交的是模块本体(既有测试同一手法)。
    gesture_app = importlib.import_module("gesture_analysis.api.app")
    glog = importlib.import_module("gesture_analysis.utils.logger")

    frames, report = load_frames(session_dir, "gesture")
    print(f"[素材] 盘上 {report['frames_on_disk']} 帧,其中 {report['frames_with_ts']} 帧有时间戳;"
          f"账本碎行 {report['torn_lines']} 行")
    if report["missing_ts"]:
        # 不静默:这些帧的图还在,但当时的时间戳丢了 —— §7.1 不允许另编一个
        print(f"⚠️  {len(report['missing_ts'])} 帧**没有可用时间戳**,本次不重放;"
              f"前几个:{report['missing_ts'][:5]}", file=sys.stderr)

    sid = session_dir.name
    # `get_or_create_analyzers` 会顺手建一个 **GestureLogger**(写 data/logs/)——
    # 重放自己往 `--out` 写一份,不需要它那一份。把落盘目录换到一个丢弃目录,
    # 免得重放一遍就在 data/logs/ 里多出一个空 CSV(与本会话的真日志同名,更难分辨)。
    with tempfile.TemporaryDirectory(prefix="replay_gesture_") as junk:
        old_logs_dir = gesture_app.LOGS_DIR
        gesture_app.LOGS_DIR = Path(junk)
        try:
            analyzers = gesture_app.get_or_create_analyzers(sid)
            dets = gesture_app.get_or_create_detectors(sid)
        finally:
            gesture_app.LOGS_DIR = old_logs_dir
        gesture_app.session_loggers.clear()

        if out_csv.exists():
            out_csv.unlink()          # GestureLogger 只在文件不存在时写表头 ⟹ 不删会追加
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        logger = glog.GestureLogger(log_file_path=str(out_csv), session_id=sid)

        n = 0
        for path, ts in frames:
            img = cv2.imread(str(path))
            if img is None:
                print(f"[跳过] 读不出:{path}", file=sys.stderr)
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = gesture_app.process_frame(rgb, ts, analyzers, dets)
            logger.log(**frame["log_kwargs"])
            n += 1

        for d in dets.values():
            d.close()
    return n


def load_audio_segments(session_dir: Path | str, kind: str = "converted"
                        ) -> tuple[list[tuple[Path, dict]], dict]:
    """读出本会话留存的音频段(账本为准)。返回 `(段列表, 缺口报告)`。

    与 `load_frames` **同一判据**:以盘上的文件为准,账本只用来取元数据(这里取的是
    `source_endpoint` / `seq` / `sha256`,好让调用方认出「同一次回答被留了两份」)。

    为什么只取 `kind="converted"`:`raw` 是浏览器给的原始容器(webm/opus),而活路径
    `voice_interaction/api/app.py` 的 `prosody_features_from_pcm` 只吃
    **16 kHz/16 bit/单声道 PCM**(端点的 `wave` 闸保证过这一点)。
    `converted` 就是过了那道闸、被 `wave` 读出来的那段字节的来源 ⟹ 重放必须用它,
    用 `raw` 得先自己转码一次,那就在重放里插了一层当时没有的处理。

    ★ **时间戳(纪律 3)在音频这一路是空的,而且是如实空的**:
    账本里音频行的 `declared_ts` **恒为 `null`**(`retain_audio` 传的就是 `None` ——
    那是逐帧模态的字段),而活路径用的那个函数 `prosody_features_from_pcm(pcm)`
    **只吃字节、不吃时间戳**。所以本腿没有可以喂的时间戳,**也不许另编一个**:
    音频的时间基准就在这份 16 kHz 字节的采样率里,是当时那批字节自带的。
    报告里把 `declared_ts_present` 数出来(预期 0),免得后来人以为这里有东西被漏掉。
    """
    session_dir = Path(session_dir)
    ledger = session_dir / f"{LEDGER_PREFIX}.audio{LEDGER_SUFFIX}"

    records: list[dict] = []
    torn = 0
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                torn += 1                       # 碎行/半行 → 计数,不崩
                continue
            if rec.get("kind") != kind:
                continue
            records.append(rec)

    segments: list[tuple[Path, dict]] = []
    missing: list[str] = []
    for rec in records:
        rel = str(rec.get("file") or "")
        p = session_dir / rel
        if rel and p.exists():
            segments.append((p, rec))
        else:
            missing.append(rel or "<无 file 字段>")

    report = {
        "ledger_records": len(records),
        "files_on_disk": len(segments),
        "missing_files": missing,
        "torn_lines": torn,
        "declared_ts_present": sum(1 for r in records if r.get("declared_ts") is not None),
    }
    return segments, report


def replay_voice(session_dir: Path, out_csv: Path, answers_only: bool = False,
                 dedupe: bool = True, questions: list[dict] | None = None,
                 session_id: str | None = None) -> int:
    """按留存音频重跑**语音服务的那一条路**,写出 **`VoiceLogger` 契约形状**的一张表。

    ★ 与手势腿同一条理由(见 `replay_gesture` 的说明):「重放」的全部意义是**走真产出方**。
    所以这里调的是端点调的那个函数 `voice_interaction.api.app.prosody_features_from_pcm`
    —— 端点里 `await _prosody_async(audio_data)`(它内部 `asyncio.to_thread` 包的就是
    这个函数)拿到的正是同一份字节。若本脚本自己写一遍"读 wav → 调提取器 → 改名",
    它证的就是它自己抄的那份接线:端点里漏接一个键、改名表漏一行,这条腿照样绿。

    ## 表格形状(甲-4,2026-10-03 改)

    以前这里写的是**自己拼的列**(`file`/`seq`/`source_endpoint`/`seconds` + 提取器的键),
    而报告层读的是 `VoiceLogger` 的 **32 列契约** ⟹ 那份 CSV 喂给报告时语音系的槽
    **成片掉光**,也与同一场活跑的日志逐列比不了。现在:

    * 表头与行的形状都取自 `VoiceLogger`(`build_row` + `fieldnames`)—— **一处实现**;
    * 产不出的列**逐条登记**在 `voice_interaction/utils/replay_contract.py` 的
      `NOT_WRITTEN_BY_REPLAY` 里并写明为什么,**留空不编**(要文本的那 5 列来自 ASR,
      而本腿只负责声学;`is_valid` 是活路径调用方的占位判断,重放不复制它没做的判断);
    * `reaction_time` 那一组**产出了** —— 它要的两样东西重放都有:`ask_end` 来自
      `questions.jsonl`,到达时刻来自留存账本。见下面两条推断规则。

    ## 两条**推断**规则(重放与活路径唯一的语义差,不许静默)

    1. **到达时刻**用**同 `seq` 的 `raw` 记录**的 `received_at_wall`
       (`replay_contract.arrival_wall_of`)。同一次上传落 `raw`/`converted` **两条**记录,
       而 `_record()` 每条各采一次 `time.time()` ⟹ 差 150~283 ms;`reaction_time` 用的是
       raw 那条(实测与活日志吻合 **±5 ms**)。找不到 raw 记录就返回 `None` ⟹
       `reaction_time` **整组不写**,不退回那个已知偏晚的时刻。
    2. **题号**按「最后一个 `ask_end ≤ 到达时刻`」推断
       (`replay_contract.infer_question_index`)。活路径知道题号(它正在处理那一题),
       重放只有账本。实测(场 `20260927_212143_95a8`,8 段回答)与活日志 **8/8 全对**。

    这两条都由 `tests/test_voice_replay_contract.py` 钉住。
    """
    import importlib
    import wave

    from voice_interaction.utils.logger import VoiceLogger
    from voice_interaction.utils.replay_contract import (
        NOT_WRITTEN_BY_REPLAY, arrival_wall_of, infer_question_index)

    voice_app = importlib.import_module("voice_interaction.api.app")

    segments, report = load_audio_segments(session_dir)
    # 「到达时刻」只在 **raw** 记录上(见 docstring 第 1 条)—— 单读一份 raw 账本备查。
    raw_segments, _raw_report = load_audio_segments(session_dir, kind="raw")
    raw_records = [rec for _p, rec in raw_segments]
    questions = list(questions or [])
    sid = session_id or session_dir.name.split("__")[-1]
    print(f"[素材] 账本 {report['ledger_records']} 段(converted),盘上找得到 {report['files_on_disk']} 段;"
          f"账本碎行 {report['torn_lines']} 行;"
          f"带 declared_ts 的音频行 {report['declared_ts_present']} 段(音频这一路应为 0);"
          f"raw 账本 {len(raw_records)} 行(取到达时刻用);题目台账 {len(questions)} 题")
    if report["missing_files"]:
        # 不静默:这些段账本有、盘上没有 —— 它们不会出现在 CSV 里,而看的人会以为
        # 那一段本来就没有。
        print(f"⚠️  {len(report['missing_files'])} 段**账本有、盘上没有**,本次不重放;"
              f"前几个:{report['missing_files'][:5]}", file=sys.stderr)

    if answers_only:
        _before = len(segments)
        segments = [(p, r) for p, r in segments
                    if r.get("source_endpoint") == "/interview/answer_audio"]
        # 筛选结果**必须报出来**:0 段时若不说话,读的人会以为"这一场没有音频",
        # 而真相是"这一场没有**提交过回答**"(实测场 `..._2449`:8 段全是 `/asr` 预览)。
        print(f"[筛选] --answers-only:保留 {len(segments)}/{_before} 段"
              f"(`/interview/answer_audio`)")
        if not segments and _before:
            print(f"⚠️  {_before} 段音频都不是提交的回答(是 `/asr` 预览)—— 本场没答过题,"
                  f"不是「没有音频」;不加 `--answers-only` 可重放那批预览段", file=sys.stderr)

    if dedupe:
        seen: set[str] = set()
        kept = []
        for p, r in segments:
            # 账本里的 sha256 就是当时那份字节的哈希(media_retention._record 算的);
            # 缺了它才退回文件名 —— 不在这里另算一次哈希。
            key = str(r.get("sha256") or p.name)
            if key in seen:
                continue
            seen.add(key)
            kept.append((p, r))
        dropped = len(segments) - len(kept)
        if dropped:
            print(f"[去重] 按 sha256 去掉 {dropped} 段逐字节相同的重复留存"
                  f"(同一段回答常被留两份:预览 `/asr` 一次 + `/interview/answer_audio` 一次)")
        segments = kept

    rows: list[dict] = []
    no_reaction: list[int] = []
    # ⚠️ 用一个**只借形状**的 logger:表头写进临时目录(与 `replay_gesture` 同一条理由),
    #    免得重放一遍就在 data/logs/ 里多出一个空 CSV。
    with tempfile.TemporaryDirectory(prefix="replay_voice_") as junk:
        vlog = VoiceLogger(log_type="interview", log_dir=junk, session_id=sid)
        for path, rec in segments:
            with wave.open(str(path), "rb") as wf:
                ch, sw, fr, n = (wf.getnchannels(), wf.getsampwidth(),
                                 wf.getframerate(), wf.getnframes())
                # 与端点同一道闸(voice_interaction/api/app.py 的 `wave` 校验):
                # 不满足就**报出来**,不静默跳过 —— 跳过的段在 CSV 里就是"没有这一行",
                # 而看的人会以为那一段本来就没有。
                if (ch, sw, fr) != (1, 2, 16000):
                    print(f"[跳过] {path.name}:格式 {ch}ch/{sw*8}bit/{fr}Hz,"
                          f"端点会 400 —— 不是{1}ch/16bit/16000Hz", file=sys.stderr)
                    continue
                pcm = wf.readframes(n)
            feats = voice_app.prosody_features_from_pcm(pcm)

            arrived = arrival_wall_of(rec, raw_records)
            qidx = infer_question_index(arrived, questions)
            window = next((w for w in questions if w.get("index") == qidx), None)
            if arrived is None or window is None:
                no_reaction.append(rec.get("seq"))
            else:
                # 走**活路径同一个函数**,不在这里重写公式(那会让"重放"证的是自己抄的那份)。
                feats.update(voice_app.reaction_time_features(feats, arrived, window))

            rows.append(vlog.build_row(
                feats, question_index=qidx, emotion="", feedback="",
                # ⚠️ `is_valid` 传 None(留空),不是 True:活路径那个 True 是调用方的占位判断,
                #    重放不复制一个它没有做出的判断。见 `NOT_WRITTEN_BY_REPLAY`。
                is_valid=None))

        if rows:
            keys = list(rows[0])
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)

    print(f"[契约] 表头 = `VoiceLogger` 的 {len(vlog.fieldnames)} 列(形状取自生产侧,非本脚本拼的);"
          f"产不出的 {len(NOT_WRITTEN_BY_REPLAY)} 列**留空**:{sorted(NOT_WRITTEN_BY_REPLAY)}")
    if no_reaction:
        print(f"⚠️  {len(no_reaction)} 段的 `reaction_time` 一组没写(缺 raw 到达时刻或题台账);"
              f"段序号:{no_reaction[:5]}", file=sys.stderr)
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="重放留存帧(M2.6 验收)")
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--root", default=None, help="默认取 JINGXIN_RECORDINGS_DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modality", default="face", choices=("face", "gesture", "voice"),
                    help="重放哪一路(M3 B3 起手势也有腿、M3.4 起语音也有腿)")
    ap.add_argument("--answers-only", action="store_true",
                    help="语音腿专用:只要 `/interview/answer_audio` 那批(去掉 `/asr` 预览)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="语音腿专用:不按 sha256 去掉逐字节相同的重复留存")
    args = ap.parse_args(argv)

    d = Path(args.root) / args.session_id if args.root else recording_dir(args.session_id)
    if args.modality == "gesture":
        n = replay_gesture(d, Path(args.out))
        print(f"[完成] 重放手势 {n} 帧 -> {args.out}")
        print("下一步:与 data/logs/gesture_emotion_log_<sid>.csv 逐格比对(spec §7.1)")
        _frames, report = load_frames(d, args.modality)
        missing, torn = len(report["missing_ts"]), report["torn_lines"]
    elif args.modality == "voice":
        import session_meta
        n = replay_voice(d, Path(args.out), answers_only=args.answers_only,
                         dedupe=not args.no_dedupe,
                         # 题台账与 sid 从**生产侧同一个读取器**取(`session_meta.read_questions`),
                         # 不在这里自己解析 jsonl —— 那是重放腿最该避免的"自己抄一份接线"。
                         questions=session_meta.read_questions(args.session_id),
                         session_id=args.session_id)
        print(f"[完成] 重放语音 {n} 段 -> {args.out}")
        print("下一步:与 data/logs/interview_emotion_log_<sid>.csv 的对应行逐格比对。"
              "列形状两边相同;差异只该出现在"
              "① 登记为「重放写不出」的那 6 列(留空 vs 活路径的值)"
              "② `unix_timestamp`/`timestamp`(活路径写落盘时刻,重放只有留存记录的到达时刻)")
        _segs, report = load_audio_segments(d)
        missing, torn = len(report["missing_files"]), report["torn_lines"]
    else:
        n = replay_face(d, Path(args.out))
        print(f"[完成] 重放 {n} 帧 -> {args.out}")
        print("下一步:与 data/logs/face_au_log_<sid>.csv 逐格比对(spec §7.1)")
        _frames, report = load_frames(d, args.modality)
        missing, torn = len(report["missing_ts"]), report["torn_lines"]

    # 退出码:只有**盘上每一份素材都重放了**才算通过 —— 有缺口就是 2,
    # 免得调用方把"重放了 105/187"当成成功。
    if missing or torn:
        print(f"[不完整] 缺口 {missing} 份 / 碎行 {torn} 行", file=sys.stderr)
        return 2
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
