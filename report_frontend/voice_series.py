# report_frontend/voice_series.py
"""**M4.9 语音序列重建** —— 从原始音频重新算逐帧序列,不再只留聚合量。

## 为什么要有这一层(诊断,不是猜测)

`prosody_extractor` 里 `librosa.pyin` 本来就产出**逐帧**的
`f0, voiced_flag, voiced_prob`,以及逐帧的 `rms`。但它只把
`mean / std / p10 / p90 / mean(voiced_prob)` 写进语音日志,**序列算完就扔**。

后果:L1 的 V 族 12 条里有 8 条(真 VAD 帧占比、停顿分位数、音高中位数、
首末半音差、音高/能量每秒变化)需要序列,而聚合量**在数学上无法反推**这些量。

**问题不是"特征提取写得不好",是"算完没存"。** 所以修法不是重写提取器,
而是**把已经算出来的东西落盘**,并允许对历史素材重算一遍。

## 素材在哪(实测,2026-10-18)

```
/home/zgy/JingXin/recordings/<会话>/media/camera.webm      37 场(VP8 1280x720 + Opus 48 kHz)
/home/zgy/JingXin/recordings/<会话>/media/audio/NNNN.webm   逐段原始音频(带 /asr 与 /interview/answer_audio)
/home/zgy/JingXin/recordings/<会话>/media/audio/NNNN_converted.wav  逐段 16 kHz 单声道
```

`retention.audio.jsonl` 的 `source_endpoint` 把两路分开:
`/interview/answer_audio` 是**回答**,`/asr` 是**提问**。实测 `20260927_212143_95a8`
的 seq 是 `asr, asr, answer, asr, answer, …` 交替,8 段回答落在 seq 3,5,…,17。

⟹ 回答音频可以**逐题**取到,不必从整场录音里切,也就不需要跨模态对齐。

## 成本(实测)

```
解码 30 s webm → wav     0.43 s
librosa.load(16 kHz)     2.13 s
librosa.pyin(30 s)       1.81 s   ⟹ 帧移 0.032 s、31.3 帧/秒
整场 10 分钟 ≈ 36 s ⟹ 37 场 ≈ 22 分钟(可后台跑)
```

## 两条不许

1. **不许用聚合量反推序列。** `pitch_std` 反推不出中位数,`p90/p10` 也**不是**"首末差"。
   要序列就重新算,算不出就说算不出。
2. **不许把缓存当成原始素材。** `npz` 是派生品,可随时删、可随时从音频重建;
   它是**加速缓存**,不是事实来源 —— 所以 `load` 失败必须回退到"没有序列",而不是猜。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import media_retention

#: pyin 的搜索范围(Hz)。与 `prosody_extractor` 同一组参数 —— 两处必须一致,
#: 否则同一个人的音高在两条链上不一样。
FMIN = 60.0
FMAX = 500.0

#: 重采样率。与 `prosody_extractor` 一致。
SAMPLE_RATE = 16000

#: 判"停顿"的最短静音时长(秒)。
#:
#: 这是**定义**,不是标定:语音学里 200–250 ms 是"停顿"的常用下限,更短的间隙是
#: 音节间的过渡,不是停顿。取 0.2 s 并**显式登记在这里**,而不是散落在代码各处。
#: 改这个值会改变 `pause_*` 的全部读数,所以它必须是一个能被审计的常量。
MIN_PAUSE_SEC = 0.2


@dataclass
class VoiceSeries:
    """一段音频的逐帧序列。`index` 为题号;`None` 表示整场。"""

    session_id: str
    index: int | None
    frame_sec: float
    f0: np.ndarray            # Hz;不浊音处为 nan
    voiced_prob: np.ndarray
    rms: np.ndarray
    source: str = ""

    @property
    def voiced_flag(self) -> np.ndarray:
        """浊音帧。**用 `f0` 是否有限判定**,不用 `voiced_prob > 阈值` ——
        `voiced_prob` 当门控实测无效(见 `prosody_extractor` 里那段实测记录:
        门控去掉的噪声帧占比与全体相同,还会把最干净的段删空)。"""
        return np.isfinite(self.f0)

    @property
    def duration_sec(self) -> float:
        return len(self.f0) * self.frame_sec

    @property
    def voiced_sec(self) -> float:
        return float(self.voiced_flag.sum()) * self.frame_sec


# ── 目录与缓存 ────────────────────────────────────────────────────────────────

def series_dir() -> Path:
    """缓存目录。可用 `JINGXIN_SERIES_DIR` 覆盖(测试用)。"""
    env = os.environ.get("JINGXIN_SERIES_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data" / "l0_series"


def _cache_path(session_id: str) -> Path:
    return series_dir() / f"{session_id}.npz"


def save_series(series: list[VoiceSeries]) -> Path:
    """写缓存。用 `npz`(不需要 pandas/parquet 依赖),键名带题号。"""
    if not series:
        raise ValueError("空序列不落盘")
    sid = series[0].session_id
    d = series_dir()
    d.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    meta: list[str] = []
    for s in series:
        key = "whole" if s.index is None else str(s.index)
        payload[f"{key}__f0"] = s.f0
        payload[f"{key}__vp"] = s.voiced_prob
        payload[f"{key}__rms"] = s.rms
        meta.append(f"{key}|{s.frame_sec}|{s.source}")
    payload["__meta__"] = np.array(meta)
    p = _cache_path(sid)
    np.savez_compressed(p, **payload)
    return p


def load_series(session_id: str) -> list[VoiceSeries] | None:
    """读缓存。**读不出就返回 `None`**(缓存是派生品,损坏不该让上层炸掉)。"""
    p = _cache_path(session_id)
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            meta = [str(x) for x in z["__meta__"]]
            out: list[VoiceSeries] = []
            for m in meta:
                key, frame_sec, source = m.split("|", 2)
                out.append(VoiceSeries(
                    session_id=session_id,
                    index=None if key == "whole" else int(key),
                    frame_sec=float(frame_sec),
                    f0=z[f"{key}__f0"], voiced_prob=z[f"{key}__vp"],
                    rms=z[f"{key}__rms"], source=source))
            return out or None
    except Exception:
        return None


# ── 从素材里取音频 ────────────────────────────────────────────────────────────

def whole_session_audio(session_id: str) -> Path | None:
    """整场录音 `media/camera.webm`(VP8 视频 + Opus 48 kHz 单声道)。

    ## 为什么需要它(2026-10-18 实测)

    **正式场的语音日志全部是 0 行** —— 24 场实名正式场,
    `interview_emotion_log_<sid>.csv` 一个都没有数据行;
    同时**没有** `questions.jsonl`、没有 `transcript.json`、没有逐段回答音频。
    只有 `camera.webm`(24/24 场都有,Opus 48 kHz)。

    ⟹ 正式语料的 V 族**只能从整场录音重算**。这不是"退而求其次":
    整场音频里每个浊音帧、每段停顿都是原始测量,比任何聚合量都更接近事实。

    ## 代价(必须说清)

    整场音频**没有题目边界** ⟹ 只能出**会话级**指标,
    `response_latency_sec`(需要"提问结束"时刻)与 `speech_rate`(需要 `n_chars`)
    **产不出来**;其余 10 条可产。
    """
    root = media_retention.resolve_recording_dir(session_id)
    cam = root / "media" / "camera.webm"
    return cam if cam.exists() else None


def audio_source(session_id: str) -> tuple[str, list[tuple[int | None, Path]]]:
    """本场能用什么音频。返回 `("answer", [(题号, 路径)…])` 或 `("whole", [(None, 路径)])`。

    优先级:**逐题音频 > 整场录音**。前者能出会话级**和**逐题两套指标,
    后者只能出会话级 —— 能细就不粗。
    """
    per_answer = answer_audio_files(session_id)
    if per_answer:
        return "answer", list(per_answer)
    whole = whole_session_audio(session_id)
    if whole is not None:
        return "whole", [(None, whole)]
    return "none", []

def answer_audio_files(session_id: str) -> list[tuple[int, Path]]:
    """逐题的回答音频,按账本 `seq` 顺序配题号。

    判据:`retention.audio.jsonl` 里 `source_endpoint == "/interview/answer_audio"`
    的 raw 行,按 `seq` 排;第 k 条配题号 k。

    ⚠️ 与 `event_windows.build_windows` 的口径**不同也不冲突**:那里按时间边界切,
    这里按账本顺序配。两者都成立时应当一致(实测 8/8 一致);不一致时以
    **时间边界**为准(那是题目的真实发生顺序),本函数只负责"文件是哪个"。
    """
    root = media_retention.resolve_recording_dir(session_id)
    ledger = root / f"{media_retention.LEDGER_PREFIX}.audio{media_retention.LEDGER_SUFFIX}"
    if not ledger.exists():
        return []
    import json
    seqs: list[int] = []
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("kind") == "raw" and rec.get("source_endpoint") == "/interview/answer_audio":
            seqs.append(int(rec["seq"]))
    out: list[tuple[int, Path]] = []
    for k, seq in enumerate(sorted(seqs)):
        wav = root / "media" / "audio" / f"{seq:04d}_converted.wav"
        webm = root / "media" / "audio" / f"{seq:04d}.webm"
        path = wav if wav.exists() else (webm if webm.exists() else None)
        if path is not None:
            out.append((k, path))
    return out


# ── 提取 ──────────────────────────────────────────────────────────────────────

def extract_file(path: str | Path, session_id: str = "", index: int | None = None,
                 sr: int = SAMPLE_RATE) -> VoiceSeries:
    """一段音频 → 逐帧序列。**这是唯一真正做重活的地方。**

    `librosa.pyin` 的第三个返回值是逐帧"是否浊音"的概率;这里**只存不用**
    —— 门控实验证明它不能用来筛帧(见 `prosody_extractor` 的实测记录)。
    """
    import librosa  # 延迟导入:没装音频库时不影响其它模块

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if len(y) == 0:
        # 空音频 = 一帧都没有。交**空数组**而不是补零 —— 补零会算出"完全静音"的读数。
        return VoiceSeries(session_id=session_id, index=index, frame_sec=0.0,
                           f0=np.array([]), voiced_prob=np.array([]),
                           rms=np.array([]), source=str(path))
    f0, _voiced_flag, voiced_prob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr)
    frame_sec = float(len(y)) / sr / len(f0)
    # RMS 用与 pyin 相同的帧数/帧移,保证两列可以逐帧对齐(不对齐就无法做"每秒变化量")
    hop = max(1, int(round(frame_sec * sr)))
    rms = librosa.feature.rms(y=y, frame_length=max(hop * 2, 512), hop_length=hop)[0]
    rms = _fit_length(rms, len(f0))
    return VoiceSeries(session_id=session_id, index=index, frame_sec=frame_sec,
                       f0=np.asarray(f0, dtype=float),
                       voiced_prob=np.asarray(voiced_prob, dtype=float),
                       rms=np.asarray(rms, dtype=float), source=str(path))


def _fit_length(a: np.ndarray, n: int) -> np.ndarray:
    """把 rms 的长度对齐到 f0 的帧数(截断或补 nan)。

    补 **nan** 而不是补 0:尾部多出来的帧没有测量值,补 0 会变成"那里完全静音"。
    """
    if len(a) == n:
        return a
    if len(a) > n:
        return a[:n]
    out = np.full(n, np.nan)
    out[:len(a)] = a
    return out


def ensure_series(session_id: str | None) -> list[VoiceSeries] | None:
    """**报告读侧要的那一份序列** —— 有缓存就读,没缓存就现提。

    ## 为什么这条要在报告路径上

    2026-10-26 实测:使用者新录一场,报告正文里「沟通表达流畅度」整维
    **证据不足(0/3)」,而报告头说语音已读入。原因只是
    `data/l0_series/<sid>.npz` 没生成 —— V 族 12 条里有 8 条只吃这份序列。

    手动跑一次 `extract_session`(34 秒)之后,同一场:

        覆盖 6/14 → 8/14 · 语音 4/12 → 12/12 · 沟通表达流畅度 0/3 → 2/3

    报告里那句提示原本写着「跑一次提取就有,不用重录」—— 它指向一个
    读者并不知道怎么跑的工具。**系统自己能做的事,不该推给使用者。**

    ## 与「报告是读侧、不许有副作用」不冲突

    `tools/jx_backfill_series.py` 那句约束的本意是"报告不许偷偷改写素材"。
    `data/l0_series/*.npz` 是本模块自己声明的**派生缓存**:「可随时删、
    可随时从音频重建」。生成它不改变任何事实来源 —— 输入相同则输出相同。

    * 允许:从**已在盘上的音频**生成派生缓存;
    * 禁止:改日志、改 CSV、改录像。

    ## 三条边界

    * 缓存已在 ⟹ **直接返回**,不重算(单场 34 秒,面板点十次就是六分钟);
    * 没有音频 ⟹ **不试**(`audio_source` 回 `none`;那是真的模态缺失,
      提取一万次也不会出东西,而且会白等);
    * 提取抛错 ⟹ **返回 `None` 并让报告照出**(ffmpeg 缺失、音频损坏都可能)。
      缺 8 条序列指标是**可报的缺口**,没有报告不可报。
    """
    if not session_id:
        return None
    cached = load_series(session_id)
    if cached is not None:
        return cached
    try:
        if audio_source(session_id)[0] == "none":
            return None
    except Exception:  # noqa: BLE001 - 读盘出错按"没有"处理,别让报告炸掉
        return None
    try:
        got = extract_session(session_id)
    except Exception as e:  # noqa: BLE001 - 提取失败是可报的缺口,不是致命错误
        print(f"   ⚠️  语音序列提取失败({session_id}):{e} —— 8 条序列指标按缺口报")
        return None
    return got or None


def extract_session(session_id: str, force: bool = False) -> list[VoiceSeries]:
    """提取并缓存。有逐题音频就逐题,否则退回整场录音。

    已有缓存且 `force=False` 时直接读缓存。
    """
    if not force:
        cached = load_series(session_id)
        if cached is not None:
            return cached
    _kind, items = audio_source(session_id)
    out: list[VoiceSeries] = []
    for index, path in items:
        out.append(extract_file(path, session_id=session_id, index=index))
    if out:
        save_series(out)
    return out
