"""转写**在盘上**,只是不在那个目录里 —— 报告不该报「没有转写」。

## 为什么

`report_frontend/transcript_chars.py` 只读 `data/transcripts/<sid>.json`
(`tools/jx_transcribe_recordings.py` 的批量产物,`.gitignore` 里的缓存)。
而**活的那条路**把转写写在**录制目录**里:`voice_interaction/asr/transcript_store.py`
每次 `/interview/answer_audio` 都把识别结果累积进
`<录制目录>/transcript.json`,形状是

    {"session_id":…, "recorded_at":…, "asr":{…}, "segments":[…],
     "merged": {"text": …, "n_chars": …, "n_segments": …, "vad_split": …}}

2026-10-25 实测(WSL 本机,`~/shared/jingxin_recordings` 下 21 场):
**每一场都有 `transcript.json`**,`merged.n_chars` 从 47 到 673;
而 `data/transcripts/` 是**空的**(0 个文件)。

⟹ 后果不是报错,是**报告静默地说错话**:T 族 3 条 + `speech_rate` 全部
报「上游只存了聚合量 / 没有转写」,报告头还写着
「本场没有 ASR 转写 —— 可跑 tools/jx_transcribe_recordings.py 生成」。
让人去跑一个**本来就不需要跑**的工具,而真转写就在他脚边的录制目录里。

这与本仓记过的形态同族(「量错了对象」):代码没坏,是**看错了地方**。

## 这个文件守什么

1. 缓存(`data/transcripts/`)里没有时,**回退到录制目录的 `transcript.json`**。
2. 两种形状都认:活路径的 `merged.*`,以及批量工具的扁平 `text` / `n_chars`
   (它们被 `--sessions-dir` 指到录制根时也写在那儿,不该因此读不出)。
3. 都没有 ⟹ `None`,**不是 0**(0 字是"一场没人说话",与"没转写过"是两回事)。
4. 缓存优先 —— 它是**显式**转写产物;回退只补空缺,不覆盖已有结果。
"""
from __future__ import annotations

import json

import pytest

from report_frontend import transcript_chars

_SID = "20260924_220729_07d7"


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """三处路径全指到 tmp:转写缓存、录制根。"""
    cache = tmp_path / "transcripts"
    cache.mkdir()
    rec = tmp_path / "recordings"
    rec.mkdir()
    monkeypatch.setenv("JINGXIN_TRANSCRIPTS_DIR", str(cache))
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(rec))
    return cache, rec


def _live_transcript(rec, sid=_SID, text="我叫王明。", n_chars=6, **merged_extra):
    """活路径的转写(形状照抄 `transcript_store._merge` + `append_utterance`)。"""
    d = rec / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.json").write_text(json.dumps({
        "session_id": sid,
        "recorded_at": "2026-09-24T22:07:37+08:00",
        "asr": {"engine": "funasr", "models": {"asr_offline": "x"}},
        "merged": {"text": text, "n_chars": n_chars, "n_segments": 1,
                   "vad_split": False, **merged_extra},
        "segments": [{"index": 0, "text": text, "n_chars": n_chars}],
    }, ensure_ascii=False), encoding="utf-8")
    return d


def test_chars_come_from_the_recording_transcript_when_the_cache_is_empty(dirs):
    """★ 红法:去掉回退分支 ⟹ 这里红(本机实测就是这个状态)。"""
    _cache, rec = dirs
    _live_transcript(rec, n_chars=673)
    assert transcript_chars.n_chars(_SID) == 673


def test_text_comes_from_the_recording_transcript_when_the_cache_is_empty(dirs):
    """★ 红法:同上。T 族三条全靠这句正文,读不到就全部 `not_stored`。"""
    _cache, rec = dirs
    _live_transcript(rec, text="呃,我叫王明,本科读的是计算机。")
    assert transcript_chars.text(_SID) == "呃,我叫王明,本科读的是计算机。"


def test_the_flat_shape_is_also_readable(dirs):
    """★ 红法:只认 `merged.*` ⟹ 这里红。

    `tools/jx_transcribe_recordings.py` 写的是**扁平**的 `text` / `n_chars`。
    它默认落在 `data/transcripts/`,但 `--sessions-dir` 指到录制根时产物就
    跟着落在录制目录里 —— 同一份东西两种形状,不该只认一种。
    """
    _cache, rec = dirs
    d = rec / _SID
    d.mkdir()
    (d / "transcript.json").write_text(json.dumps({
        "session_id": _SID, "text": "扁平形状的正文", "n_chars": 7,
        "audio_seconds": 12.5,
    }, ensure_ascii=False), encoding="utf-8")
    assert transcript_chars.n_chars(_SID) == 7
    assert transcript_chars.text(_SID) == "扁平形状的正文"


def test_the_cache_wins_when_both_exist(dirs):
    """★ 红法:把回退提到缓存之前 ⟹ 这里红。

    缓存是**显式**跑过转写工具的产物;回退只补空缺。反过来会让"我特意转的那份"
    被录制目录里更早的那份悄悄盖掉 —— 又是一次静默的「量错了对象」。
    """
    cache, rec = dirs
    _live_transcript(rec, text="录制目录里的旧版", n_chars=100)
    (cache / f"{_SID}.json").write_text(json.dumps({
        "session_id": _SID, "text": "缓存里的新版", "n_chars": 200,
    }, ensure_ascii=False), encoding="utf-8")
    assert transcript_chars.n_chars(_SID) == 200
    assert transcript_chars.text(_SID) == "缓存里的新版"


def test_no_transcript_anywhere_is_still_none_not_zero(dirs):
    """★ 红法:回退分支里 `return 0` ⟹ 这里红。

    两处都没有 = 真的没转写过。回 0 会让下游印出「0.00 字/秒」,
    一个看着像测量、其实是"这场没人说话"的数。
    """
    assert transcript_chars.n_chars(_SID) is None
    assert transcript_chars.text(_SID) is None


def test_broken_recording_transcript_is_none_not_an_exception(dirs):
    """★ 红法:去掉 try ⟹ 这里红。录制目录里的文件可能是正在写的中途产物。"""
    _cache, rec = dirs
    d = rec / _SID
    d.mkdir()
    (d / "transcript.json").write_text("{写到一半", encoding="utf-8")
    assert transcript_chars.n_chars(_SID) is None
    assert transcript_chars.text(_SID) is None


def test_empty_merged_text_is_none_not_empty_string(dirs):
    """★ 红法:把 `text()` 的 `if not t.strip()` 去掉 ⟹ 这里红。

    空串会被 T 族算成"0 字",与"没转写过"混为一谈 —— 前者是"一场没人说话",
    后者是"还没转写",补救动作完全不同。
    """
    _cache, rec = dirs
    _live_transcript(rec, text="   ", n_chars=0)
    assert transcript_chars.text(_SID) is None
    assert transcript_chars.n_chars(_SID) is None


def test_labeled_recording_dir_is_found(dirs):
    """★ 红法:按 `root()/<sid>` 自己拼路径 ⟹ 这里红。

    存过标注的场次目录叫 `<标签>__<sid>`(见 `media_retention.resolve_recording_dir`)。
    自己拼路径会找一个不存在的目录,然后静默报「没有转写」。
    """
    _cache, rec = dirs
    # 标签用**一眼看得出是编的**字样:真实素材的目录名是 `<姓名>-<学号>-<专业>__<sid>`,
    # 而本仓是公开仓库,测试里不该出现任何形似真实素材的目录名(见 PII 约束)。
    named = rec / f"示例标签__{_SID}"
    named.mkdir(parents=True)
    (named / "transcript.json").write_text(json.dumps({
        "session_id": _SID, "merged": {"text": "带标签的目录", "n_chars": 6},
    }, ensure_ascii=False), encoding="utf-8")
    assert transcript_chars.n_chars(_SID) == 6
    assert transcript_chars.text(_SID) == "带标签的目录"
