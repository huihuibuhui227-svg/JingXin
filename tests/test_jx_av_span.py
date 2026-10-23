"""`tools/jx_av_span.py` —— 流式 webm 量不出时长这件事,必须有测试钉住。

为什么值得单独测:这个工具存在的原因是「**`ffprobe` 返回 N/A 时 exit code 仍是 0**」。
如果哪天有人把它改回 `-show_entries format=duration`,代码不会报错、不会崩,
只会**打印一排 None** —— 而那看起来像"这些场次没有文件",不像"量错了字段"。
本仓的纪律是:静默失效的东西必须有东西拦住。
"""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("jx_av_span", ROOT / "tools/jx_av_span.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


av = _load()


def test_pts_span_returns_none_when_the_file_has_no_track(tmp_path):
    """没有音轨的文件 ⟹ None,不是 0.0,也不是抛异常。

    0.0 会被下游当成"这一段是零秒",与"量不到"是两件事。
    """
    f = tmp_path / "empty.webm"
    f.write_bytes(b"not a real container")
    assert av.pts_span(f, "a:0") is None


def test_pts_span_returns_none_for_a_missing_file(tmp_path):
    assert av.pts_span(tmp_path / "nope.webm", "a:0") is None


def test_ledger_span_returns_none_without_declared_ts(tmp_path):
    """`declared_ts` 为 null 时不能算成 0 —— 那会把"没有时刻"变成"零跨度"。"""
    f = tmp_path / "retention.camera.jsonl"
    f.write_text(json.dumps({"declared_ts": None}) + "\n", encoding="utf-8")
    assert av.ledger_span(f) is None


def test_ledger_span_reads_milliseconds(tmp_path):
    """`declared_ts` 是**毫秒**。读成秒会让跨度差 1000 倍。"""
    f = tmp_path / "retention.face.jsonl"
    f.write_text(
        json.dumps({"declared_ts": 1_000_000}) + "\n"
        + json.dumps({"declared_ts": 1_009_500}) + "\n",
        encoding="utf-8")
    assert av.ledger_span(f) == pytest.approx(9.5)


def test_ledger_span_tolerates_a_truncated_last_line(tmp_path):
    """账本是追加写的 —— 进程被杀时最后一行可能是半截 JSON。不许因此整场读不出。"""
    f = tmp_path / "retention.gesture.jsonl"
    f.write_text(
        json.dumps({"declared_ts": 0}) + "\n"
        + json.dumps({"declared_ts": 2_000}) + "\n"
        + '{"declared_ts": 4',   # 截断
        encoding="utf-8")
    assert av.ledger_span(f) == pytest.approx(2.0)


def test_sessions_skips_underscore_and_none_dirs(tmp_path):
    """`_回收站` / `_已作废` / `NONE` 不是素材 —— 与 `corpus.audit_corpus` 同一套排除。"""
    for name in ("_回收站", "_已作废", "NONE", "20260101_000000_aaaa"):
        (tmp_path / name).mkdir()
    got = [p.name for p in av.sessions(tmp_path)]
    assert got == ["20260101_000000_aaaa"]


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="没有 ffprobe")
def test_streaming_webm_has_no_format_duration_but_pts_work(tmp_path):
    """★ 本工具存在的理由,用真文件验一次。

    Chrome MediaRecorder 产的流式 webm:**`format.duration` 是 N/A**(且 exit=0),
    但 **packet 的 pts 量得出来**。如果这条测试红了,说明 ffmpeg 换了行为 ——
    那时才需要重新考虑本工具的写法。
    """
    f = tmp_path / "t.webm"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libvpx", "-c:a", "libopus", "-y", str(f)],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not f.exists():
        pytest.skip(f"ffmpeg 造不出样本:{r.stderr[:200]}")

    span = av.pts_span(f, "v:0")
    assert span is not None, "pts 量不到 —— 工具的前提不成立了"
    assert span[1] - span[0] > 1.0, f"2 秒的样本量出 {span[1]-span[0]:.2f} 秒"
