# tests/test_backfill_voice.py
"""语音日志离线回填(甲-4 下游)的**不许伤害**那几条。

## 为什么单独一组

这个工具的产物与活跑写的日志**同形同名** —— 看文件本身分不出来。所以它的安全性
不靠"看起来对",靠这几条硬规矩:

1. **绝不覆盖活行**:有 ≥1 行数据就跳过(且这个判断在**读账本之前**,不依赖下游);
2. **先备份再写**,备份名**首个永不覆盖**(与 `jx_backfill_face.py` 同一套);
3. **没有提交的回答段就不动**(只有 `/asr` 预览的场次没答过题,不是"没有音频");
4. 录制根只有**一处**来源(`media_retention.root()`)⟹ 扫的目录与读
   `questions.jsonl` 的目录不可能不是同一个(否则 `reaction_time` 会整组**静默**留空)。

⚠️ 这里**不**测"补出来的值对不对" —— 那是 `tests/test_voice_replay_contract.py`
(形状/划分/两条推断)+ `docs/下一步.md`「甲、4」那段端到端验收(24/32 列逐字相同)的事。
本文件只管"它会不会弄坏已有东西"。

红法(实测):把 `backfill_one` 里那段 `if have: return skip` 删掉 ⟹
`test_existing_rows_are_never_overwritten` 红;把 `backup_path` 的循环去掉、
恒返回第一个后缀 ⟹ `test_backup_ladder_never_clobbers` 红。
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

backfill = importlib.import_module("jx_backfill_voice")


def _session_dir(tmp_path: Path, sid: str, endpoints: list[str]) -> Path:
    """造一个**只够 walk 账本**的场次目录(不需要真音频:跳过的路径不读字节)。"""
    d = tmp_path / "recordings" / f"20260101_000000-1-甲-000-系__{sid}"
    (d / "media" / "audio").mkdir(parents=True)
    lines = []
    for i, ep in enumerate(endpoints, 1):
        for kind in ("raw", "converted"):
            lines.append(json.dumps({
                "kind": kind, "modality": None, "seq": i,
                "file": f"media/audio/{i:04d}{'' if kind == 'raw' else '_converted'}"
                        f"{'.webm' if kind == 'raw' else '.wav'}",
                "bytes": 10, "sha256": f"{kind}-{i}", "received_at_wall": 100.0 + i,
                "declared_ts": None, "source_endpoint": ep, "session_id": sid,
            }, ensure_ascii=False))
    (d / "retention.audio.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


# ── 备份阶梯 ───────────────────────────────────────────────────────────
def test_backup_ladder_never_clobbers(tmp_path):
    """★ 首个备份永不覆盖:连顶三次,三份都要在。

    红法(实测):把 `backup_path` 的 `while` 循环删掉、恒返回 `.bak` ⟹ 本条红。
    为什么值得单独守:`--pending` 那种例行命令会对同一场跑很多次,而
    **第一次**留下的那份备份往往才是唯一的物证(当年 face 那边就是它被顶掉了)。
    """
    out = tmp_path / "interview_emotion_log_x.csv"
    out.write_text("header\n", encoding="utf-8")
    names = []
    for i in range(3):
        bak = backfill.backup_path(out)
        assert not bak.exists(), f"第 {i + 1} 次选到了一个已存在的备份名"
        bak.write_text(f"第{i + 1}代\n", encoding="utf-8")
        names.append(bak.name)
    assert names == ["interview_emotion_log_x.csv.voice_backfill.bak",
                     "interview_emotion_log_x.csv.voice_backfill.1.bak",
                     "interview_emotion_log_x.csv.voice_backfill.2.bak"]
    for n, expect in zip(names, ("第1代", "第2代", "第3代")):
        assert (tmp_path / n).read_text(encoding="utf-8").strip() == expect


# ── 行数判据 ───────────────────────────────────────────────────────────
def test_row_count_distinguishes_missing_header_only_and_rows(tmp_path):
    """`None`(文件不在)/ `0`(只有表头)/ `N` —— 三态必须分得开。

    这是"要不要补"的唯一判据:把 `0` 与"文件不在"混成一个值,就会把活跑的
    **表头空文件**当成"已经有日志",于是那一场永远补不上(而报告里它显示为
    「未采集到对应数据」,没人知道差一步)。
    """
    p = tmp_path / "a.csv"
    assert backfill.log_row_count(p) is None
    p.write_text("h1,h2\n", encoding="utf-8")
    assert backfill.log_row_count(p) == 0
    p.write_text("h1,h2\n1,2\n3,4\n", encoding="utf-8")
    assert backfill.log_row_count(p) == 2


# ── 绝不覆盖活行 ───────────────────────────────────────────────────────
def test_existing_rows_are_never_overwritten(tmp_path):
    """★ 有活行 ⟹ 立刻跳过,**连账本都不读**(传一个不存在的场次目录也必须跳过)。

    传不存在的目录是刻意的:它证明这条判断在**读账本之前**,所以"不覆盖活行"
    不依赖任何下游环节(下游一旦报错,也不会变成"于是写了")。
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    sid = "20260101_000000_aaaa"
    out = backfill.voice_log_path(sid, log_dir)
    out.write_text("h\n活行\n", encoding="utf-8")

    r = backfill.backfill_one(tmp_path / sid, log_dir=log_dir)   # ← 目录**不存在**
    assert r["status"] == "skip" and "已有 1 行" in r["why"], r
    assert out.read_text(encoding="utf-8") == "h\n活行\n", "活行被动了"
    assert not list(log_dir.glob("*.bak")), "跳过的场次不该留下备份(什么都没顶掉)"


def test_header_only_log_is_a_candidate_for_backfill(tmp_path):
    """只有表头的日志**要**补(它是"当时没落下来",不是"已经有数据")。

    这条与上一条是一对:上一条守"别覆盖",这条守"别漏补"。实测里
    `data/logs/` 有 85 份语音日志是**只有表头**的 —— 把它们当"已完成",
    那 85 场就永远补不上。
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    sid = "20260101_000000_bbbb"
    out = backfill.voice_log_path(sid, log_dir)
    out.write_text("h1,h2\n", encoding="utf-8")
    assert backfill.log_row_count(out) == 0


# ── 只认提交的回答段 ───────────────────────────────────────────────────
def test_session_without_submitted_answers_is_skipped_with_the_real_reason(tmp_path):
    """只有 `/asr` 预览段的场次 ⟹ 跳过,而且理由要说**对**。

    实测三场(`..._2449` / `..._253a` / `..._90e7`)的留存段全是 `/asr` 预览。
    把这种场次说成"没有音频"是**假话**,而且会让人去查采集端 ——
    查了也没用。真实原因是"本场没答过题"。
    """
    d = _session_dir(tmp_path, "20260101_000000_cccc", ["/asr", "/asr"])
    assert backfill.answer_segments(d) == []
    r = backfill.backfill_one(d, log_dir=tmp_path / "logs")
    assert r["status"] == "skip"
    assert "没答过题" in r["why"], r
    assert "没有音频" not in r["why"].replace("不是「没有音频」", ""), (
        f"理由里把「没答过题」说成了「没有音频」:{r['why']}")


def test_answer_segments_are_deduped_by_sha(tmp_path):
    """同一段回答被留两份(预览 + 提交、字节相同)时要**去重**,否则行数会翻倍。

    ⚠️ 去重按 `sha256` 且**只在 `converted` 里** —— `raw`(webm)与 `converted`(wav)
    字节不同、哈希不同,按哈希 join 会一条都配不上(实测过,见「丙、19」)。
    """
    d = _session_dir(tmp_path, "20260101_000000_dddd",
                     ["/interview/answer_audio", "/interview/answer_audio"])
    # 把第 2 段的 sha 改得与第 1 段相同(模拟"同一段被留了两份")
    p = d / "retention.audio.jsonl"
    lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    for rec in lines:
        if rec["kind"] == "converted" and rec["seq"] == 2:
            rec["sha256"] = "converted-1"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n",
                 encoding="utf-8")
    assert len(backfill.answer_segments(d)) == 1, "逐字节相同的重复留存没被去掉"


# ── 一个根 ─────────────────────────────────────────────────────────────
def test_recordings_root_has_a_single_source(tmp_path, monkeypatch):
    """★ 扫描用的根与读 `questions.jsonl` 用的根必须是**同一个**开关。

    两者一旦分家,`infer_question_index` 会查不到题台账 ⟹ `reaction_time` 整组
    **静默留空** —— 而报告那边只会显示「未采集到对应数据」,没人知道是路径不对。
    所以本工具不自己解析根:它读 `media_retention.root()`(生产侧同一个函数)。
    """
    import media_retention

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    assert media_retention.root() == tmp_path
    # 工具的默认值就是它 —— 没有第二处解析
    src = (REPO / "tools" / "jx_backfill_voice.py").read_text(encoding="utf-8")
    assert "media_retention.root()" in src
    assert "os.getenv(\"JINGXIN_RECORDINGS_DIR\"" not in src, (
        "工具自己解析录制根了 —— 那就成了第二处来源,与 session_meta 会漂")
