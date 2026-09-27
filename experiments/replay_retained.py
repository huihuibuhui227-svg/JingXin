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
                 dedupe: bool = True) -> int:
    """按留存音频重跑**语音服务的那一条路**,把每段的**特征字典**写成 CSV。返回段数。

    ★ 与手势腿同一条理由(见 `replay_gesture` 的说明):「重放」的全部意义是**走真产出方**。
    所以这里调的是端点调的那个函数 `voice_interaction.api.app.prosody_features_from_pcm`
    —— 端点里 `await _prosody_async(audio_data)`(它内部 `asyncio.to_thread` 包的就是
    这个函数)拿到的正是同一份字节。若本脚本自己写一遍"读 wav → 调提取器 → 改名",
    它证的就是它自己抄的那份接线:端点里漏接一个键、改名表漏一行,这条腿照样绿。
    ⚠️ 走的是 `prosody_features_from_pcm`,因此**不经过** `reaction_time_features`
    (那要用墙钟与题目台账,重放里没有,也不许编)—— 本腿只对"提取器产出"负责。
    """
    import importlib
    import wave

    voice_app = importlib.import_module("voice_interaction.api.app")

    segments, report = load_audio_segments(session_dir)
    print(f"[素材] 账本 {report['ledger_records']} 段(converted),盘上找得到 {report['files_on_disk']} 段;"
          f"账本碎行 {report['torn_lines']} 行;"
          f"带 declared_ts 的音频行 {report['declared_ts_present']} 段(音频这一路应为 0)")
    if report["missing_files"]:
        print(f"⚠️  {len(report['missing_files'])} 段**账本有、盘上没有**,本次不重放;"
              f"前几个:{report['missing_files'][:5]}", file=sys.stderr)

    if answers_only:
        segments = [(p, r) for p, r in segments
                    if r.get("source_endpoint") == "/interview/answer_audio"]

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
        row = {"file": path.name, "seq": rec.get("seq"),
               "source_endpoint": rec.get("source_endpoint"),
               "seconds": round(len(pcm) / 2 / 16000, 2)}
        row.update(feats)
        rows.append(row)

    if rows:
        keys = list(rows[0])
        keys += sorted({k for r in rows for k in r} - set(keys))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
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
        n = replay_voice(d, Path(args.out), answers_only=args.answers_only,
                         dedupe=not args.no_dedupe)
        print(f"[完成] 重放语音 {n} 段 -> {args.out}")
        print("下一步:与 data/logs/interview_emotion_log_<sid>.csv 的对应行逐格比对"
              "(注意:活路径那几行还多出 `reaction_time` 一组,它们要墙钟与题目台账,重放里没有)")
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
