"""A3 存量回填工具 `patch/backfill_transcript_asr.py` 的测试。

**不依赖真实数据**:所有样本都在 `tmp_path` 里现造,跑测试不会碰
`data/transcripts/`,也不会连 ASR 服务(`asr_block` 只读 `asr_config.json`)。

三类样本(任务要求的最小覆盖):
  * **缺块** —— 扁平存量形状(`segments` 在顶层)与活路径形状(`merged`)各一份;
  * **已有块** —— 旧 5 键块,必须**跳过**(幂等),字节都不许动;
  * **坏文件** —— 非法 JSON 与"顶层不是 dict",必须**跳过 + 计数**,退出码 3。

断言:dry-run 不写、apply 只加一个键、二次运行幂等、坏文件跳过且计数正确。
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from voice_interaction.asr import transcript_store

# ── 把补丁包里的工具按路径加载(它不在包里,`import` 不到) ─────────────────────
# 顺序:显式环境变量 → 仓库 `tools/`(若哪天搬进来) → 补丁包原位。
_TOOL_CANDIDATES = (
    os.environ.get("JX_BACKFILL_TOOL"),
    str(Path(__file__).resolve().parent.parent / "tools" / "backfill_transcript_asr.py"),
    "/home/zgy/huihuibui/衡才智析/07_三方向与评审筹备/系统补强/patch/backfill_transcript_asr.py",
)


def _load_tool():
    for cand in _TOOL_CANDIDATES:
        if cand and Path(cand).is_file():
            spec = importlib.util.spec_from_file_location("_backfill_asr_tool", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("找不到 backfill_transcript_asr.py(可用 JX_BACKFILL_TOOL 指定)")


tool = _load_tool()

FLAT_SEG_1 = {"index": 0, "text": "欢迎", "n_chars": 2,
              "timestamps_ms": [[210, 450]], "punc_array": [1],
              "ts_origin": "segment_relative"}
FLAT_SEG_2 = {"index": 1, "text": "你好", "n_chars": 2,
              "timestamps_ms": [[480, 700]], "punc_array": [1],
              "ts_origin": "segment_relative"}

#: 存量批量工具产出的扁平形状(**没有** `asr` 块)
FLAT_DOC = {
    "session_dir": "20260928_120000-3-张三-1-金融学院__20260928_120000_abcd",
    "session_id": "20260928_120000_abcd",
    "name": "张三",
    "student_id": "1",
    "consent": None,
    "audio_seconds": 1.4,
    "text": "欢迎你好",
    "n_chars": 4,
    "n_segments": 2,
    "vad_split": True,
    "segments": [FLAT_SEG_1, FLAT_SEG_2],
    "extract_seconds": 0.2,
    "asr_seconds": 0.9,
    "source": "media/camera.webm (opus 音轨)",
    "notes": ["会话级转写,非逐题"],
}

#: 活路径(`transcript.json`)形状:`merged` 在顶层,计数只在 `merged` 里
MERGED_DOC = {
    "session_id": "20260928_130000_beef",
    "recorded_at": "2026-09-28T13:00:00+08:00",
    "merged": {"text": "欢迎", "n_chars": 2, "n_segments": 1, "vad_split": False},
    "segments": [dict(FLAT_SEG_1)],
}

#: 已有 `asr` 块(补丁前写下的 5 键形状)—— 必须原样保留
EXISTING_DOC = {
    "session_id": "20260928_140000_cafe",
    "segments": [dict(FLAT_SEG_1)],
    "n_chars": 2,
    "n_segments": 1,
    "asr": {"engine": "funasr", "endpoint": "ws://1.2.3.4:5",
            "models": {"vad": "old-model"},
            "asr_confidence": None, "asr_confidence_source": "unavailable"},
}


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _write(p: Path, obj) -> Path:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


@pytest.fixture()
def corpus(tmp_path):
    """缺块 ×2、已有块 ×1、坏文件 ×2 —— 全部现造。"""
    d = tmp_path / "transcripts"
    d.mkdir()
    files = {
        "flat.json": _write(d / "flat.json", FLAT_DOC),
        "merged.json": _write(d / "merged.json", MERGED_DOC),
        "existing.json": _write(d / "existing.json", EXISTING_DOC),
        "broken.json": (d / "broken.json"),                      # 非法 JSON
        "notdict.json": _write(d / "notdict.json", [1, 2, 3]),   # 顶层不是 dict
    }
    files["broken.json"].write_text('{"session_id": "x", "segments": [', encoding="utf-8")
    return d, files


# ── 1. dry-run 一个字节都不写 ────────────────────────────────────────────────
def test_dry_run_writes_nothing(corpus, capsys):
    d, files = corpus
    before = {n: _md5(p) for n, p in files.items()}

    assert tool.main(["--dir", str(d)]) == 3          # 有坏文件 ⟹ 退出码 3
    out = capsys.readouterr().out

    assert {n: _md5(p) for n, p in files.items()} == before, "dry-run 竟然改了文件"
    assert sorted(x.name for x in d.iterdir()) == sorted(files), "dry-run 留下了临时文件"
    assert "2 份待补" in out and "1 份已有 asr 块" in out and "2 份读不出" in out
    assert "SUMMARY scanned=5 todo=2 skipped=1 broken=2 written=0" in out


# ── 2. apply:只加一个键,其余原封不动 ───────────────────────────────────────
def test_apply_adds_asr_and_keeps_every_other_field(corpus):
    d, files = corpus
    assert tool.main(["--dir", str(d), "--apply"]) == 3

    for name, original in (("flat.json", FLAT_DOC), ("merged.json", MERGED_DOC)):
        doc = json.loads(files[name].read_text(encoding="utf-8"))
        assert "asr" in doc and isinstance(doc["asr"], dict)

        # 非 asr 字段:值 + 相对顺序都不许变
        assert [k for k in doc if k != "asr"] == list(original), f"{name} 键序被改了"
        assert {k: v for k, v in doc.items() if k != "asr"} == original
        # `asr` 落在 `segments` 之后(= 新产出文件的位置)
        keys = list(doc)
        assert keys[keys.index("segments") + 1] == "asr"


# ── 3. 与写入器同一套定义(不是第二个拼装函数) ──────────────────────────────
def test_asr_block_comes_from_the_live_writer(corpus):
    d, files = corpus
    tool.main(["--dir", str(d), "--apply"])

    doc = json.loads(files["flat.json"].read_text(encoding="utf-8"))
    asr = doc["asr"]
    expected = transcript_store.asr_block(None, len(doc["segments"]), doc["n_chars"])

    # body 逐键同值、同序;只多两个审计键(而且排在最末)
    assert list(asr)[:len(expected)] == list(expected)
    assert {k: asr[k] for k in expected} == expected
    assert list(asr)[len(expected):] == ["backfilled", "backfilled_at"]
    assert asr["backfilled"] is True
    assert isinstance(asr["backfilled_at"], str) and "T" in asr["backfilled_at"]

    # 旧契约的 5 键也必须在(下游按名取)
    for k in ("engine", "endpoint", "models", "asr_confidence", "asr_confidence_source"):
        assert k in asr


# ── 4. 未知信息写 null + unavailable,不猜 ───────────────────────────────────
def test_unknown_confidence_is_null_not_guessed(corpus):
    d, files = corpus
    tool.main(["--dir", str(d), "--apply"])
    asr = json.loads(files["flat.json"].read_text(encoding="utf-8"))["asr"]

    assert asr["confidence"] is None and asr["asr_confidence"] is None
    assert asr["confidence_source"] == asr["asr_confidence_source"] == "unavailable"
    # 端点是配置文件的事实,不是伪造的
    cfg = json.loads((Path(transcript_store.__file__).parent / "asr_config.json")
                     .read_text(encoding="utf-8"))
    assert asr["endpoint"] == f"ws://{cfg['funasr_host']}:{cfg['funasr_port']}"
    assert asr["models"] == cfg["models"]


# ── 5. 已有块跳过:字节都不动 ────────────────────────────────────────────────
def test_existing_block_is_left_byte_identical(corpus):
    d, files = corpus
    before = _md5(files["existing.json"])
    tool.main(["--dir", str(d), "--apply"])
    assert _md5(files["existing.json"]) == before


# ── 6. 坏文件跳过 + 报错计数正确 ────────────────────────────────────────────
def test_broken_files_are_skipped_and_counted(corpus, capsys):
    d, files = corpus
    broken_before = {n: _md5(files[n]) for n in ("broken.json", "notdict.json")}

    code = tool.main(["--dir", str(d), "--apply"])
    out = capsys.readouterr().out

    assert code == 3, "有坏文件时退出码必须是 3"
    assert "SUMMARY scanned=5 todo=2 skipped=1 broken=2 written=2" in out
    assert "✗ broken.json" in out and "✗ notdict.json" in out
    assert {n: _md5(files[n]) for n in broken_before} == broken_before, "坏文件被动了"
    # 坏文件不能挡住好文件
    assert "asr" in json.loads(files["flat.json"].read_text(encoding="utf-8"))


# ── 7. 幂等:第二次 --apply 0 变更、md5 不变 ─────────────────────────────────
def test_second_apply_is_idempotent(tmp_path, capsys):
    d = tmp_path / "transcripts"
    d.mkdir()
    _write(d / "flat.json", FLAT_DOC)
    _write(d / "merged.json", MERGED_DOC)

    assert tool.main(["--dir", str(d), "--apply"]) == 0
    first = {p.name: _md5(p) for p in d.glob("*.json")}
    stamp = json.loads((d / "flat.json").read_text(encoding="utf-8"))["asr"]["backfilled_at"]
    capsys.readouterr()

    assert tool.main(["--dir", str(d), "--apply"]) == 0
    out = capsys.readouterr().out

    assert {p.name: _md5(p) for p in d.glob("*.json")} == first
    assert "SUMMARY scanned=2 todo=0 skipped=2 broken=0 written=0" in out
    # backfilled_at 不该被第二轮重刷(时间戳变了就说明它又写了一遍)
    assert json.loads((d / "flat.json").read_text(encoding="utf-8"))["asr"]["backfilled_at"] == stamp


# ── 8. 编码/格式/原子写 ─────────────────────────────────────────────────────
def test_output_is_utf8_indent2_without_trailing_newline(corpus):
    d, files = corpus
    tool.main(["--dir", str(d), "--apply"])
    raw = files["flat.json"].read_bytes()

    doc = json.loads(raw.decode("utf-8"))
    assert raw == json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
    assert not raw.endswith(b"\n"), "存量文件没有尾换行,回填也不该加"
    assert "张三".encode("utf-8") in raw, "中文被转义了(ensure_ascii 应为 False)"
    assert not list(d.glob("*.tmp")), "原子写留下了 .tmp"


# ── 9. 备份:指定目录 + 不覆盖更早的原件 ─────────────────────────────────────
def test_backup_dir_keeps_relative_path_and_never_overwrites(tmp_path):
    d = tmp_path / "transcripts"
    d.mkdir()
    p = _write(d / "flat.json", FLAT_DOC)
    orig = p.read_bytes()
    bak = tmp_path / "bak"

    assert tool.main(["--dir", str(d), "--apply", "--backup-dir", str(bak)]) == 0
    assert (bak / "flat.json").read_bytes() == orig, "备份不是回填前的原件"
    # 默认不往数据目录里塞 *.pre-asr-bak
    assert not list(d.glob("*.pre-asr-bak"))

    # 再跑一次(0 变更)也得把备份留着,且不改它
    first_bak = _md5(bak / "flat.json")
    tool.main(["--dir", str(d), "--apply", "--backup-dir", str(bak)])
    assert _md5(bak / "flat.json") == first_bak


def test_default_backup_lands_next_to_the_file(tmp_path):
    d = tmp_path / "transcripts"
    d.mkdir()
    p = _write(d / "flat.json", FLAT_DOC)
    orig = p.read_bytes()

    tool.main(["--dir", str(d), "--apply"])
    bak = d / "flat.json.pre-asr-bak"
    assert bak.is_file() and bak.read_bytes() == orig
    # .pre-asr-bak 不能被下一轮当成输入
    assert tool.main(["--dir", str(d), "--apply"]) == 0


# ── 10. 空目录 / 不存在的目录 ───────────────────────────────────────────────
def test_missing_dir_is_an_error(tmp_path):
    assert tool.main(["--dir", str(tmp_path / "nope")]) == 2


def test_empty_dir_is_not_an_error(tmp_path, capsys):
    assert tool.main(["--dir", str(tmp_path)]) == 0
    assert "没有 .json" in capsys.readouterr().out
