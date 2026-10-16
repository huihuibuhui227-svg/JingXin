"""`tools/jx_backfill_face.py` 里两个**出过错**的判据的钉子。

都是 2026-10-03 复查 `--uniform`（全量统一重跑）时揪出来的：

- `backup_path`：原先直接 `shutil.move(out_csv, out_csv + ".predrop.bak")`，
  **无条件覆盖**同名备份。而 `--uniform` 会对已经补过一次的场次再跑一遍，
  于是那份"回填前的原始 CSV"（检出 0.1% 的、也就是"0.1% → 93%"这个结论**唯一的
  物证**）会被回填后的结果顶掉。这条钉住"首个备份永不覆盖"。

- `detection_rate`：判据是"除 id/时间戳外还有非空值"。现有 215 份 CSV 里
  `no_face` 一次都没出现过（无脸的表示法是整行留空），所以这条防御目前是 no-op；
  但要是哪天有人开始写 `no_face` 哨兵值，无脸行就会被算成检出 ⟹ 检出率**静默虚高**。
  这条钉住"no_face 不算检出"，与 `framing.is_detected` 同一判据。

历史教训（已修，留在这里防回退）：我第一版 `detection_rate` 查的是 CSV 的 `emotion`
列，而那列**根本不存在**（logger 的 `fieldnames` 里是 `dominant_emotion`）⟹
每一场都算成 0%。所以下面专门有一条钉列名。
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    """按路径加载 `tools/jx_backfill_face.py`（`tools/` 不是包）。"""
    path = ROOT / "tools" / "jx_backfill_face.py"
    spec = importlib.util.spec_from_file_location("jx_backfill_face_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bf = _load_module()


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ──────────────────────────────────────────────────────────────
# backup_path：★ 首个备份永不覆盖
# ──────────────────────────────────────────────────────────────

def test_first_backup_uses_the_plain_suffix(tmp_path):
    """没有任何备份时，就用 `.csv.predrop.bak`。

    这条红了 ⟹ 有人改了后缀命名（会让 `--uniform` 找不到第一份原始备份）。
    """
    out = tmp_path / "face_au_log_20260101_000000_abcd.csv"
    assert bf.backup_path(out).name == "face_au_log_20260101_000000_abcd.csv.predrop.bak"


def test_existing_backup_is_never_clobbered(tmp_path):
    """★ 核心：`.predrop.bak` 已存在时，必须让位到 `.predrop.1.bak`。

    这条红了 ⟹ 又回到"覆盖同名备份"的写法，那份回填前的原始 CSV 会被顶掉
    （也就是这条测试存在的唯一理由）。
    """
    out = tmp_path / "x.csv"
    (tmp_path / "x.csv.predrop.bak").write_text("最初那份，不许动", encoding="utf-8")
    assert bf.backup_path(out).name == "x.csv.predrop.1.bak"


def test_backup_ladder_keeps_climbing(tmp_path):
    """连跑第三次要落到 `.predrop.2.bak` —— 谁都不丢。

    这条红了 ⟹ 序号阶梯断了（第二份备份会盖掉第一份）。
    """
    out = tmp_path / "x.csv"
    (tmp_path / "x.csv.predrop.bak").write_text("第 0 份", encoding="utf-8")
    (tmp_path / "x.csv.predrop.1.bak").write_text("第 1 份", encoding="utf-8")
    assert bf.backup_path(out).name == "x.csv.predrop.2.bak"


def test_backup_path_does_not_touch_the_filesystem(tmp_path):
    """选名必须是**纯**查询：不许顺手创建任何文件。

    这条红了 ⟹ `backup_path` 里混进了写操作（`--dry-run` 会变得不干净）。
    """
    out = tmp_path / "x.csv"
    before = sorted(p.name for p in tmp_path.iterdir())
    bf.backup_path(out)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# ──────────────────────────────────────────────────────────────
# detection_rate：★ 判据要跟着数据实际长什么样走
# ──────────────────────────────────────────────────────────────

FIELDS = ["session_id", "timestamp", "timestamp_iso", "dominant_emotion", "face_scale", "bs_brow_down"]


def test_missing_file_is_none_not_zero(tmp_path):
    """文件不存在返回 `None`（区别于"存在但一行都没有"的 0.0）。

    这条红了 ⟹ 调用方分不清"这场没日志"和"这场日志全空"，
    于是会把没日志的场次当成 0% 去补。
    """
    assert bf.detection_rate(tmp_path / "nope.csv") is None


def test_all_empty_rows_are_zero(tmp_path):
    """只有 id/时间戳的整行留空 = 没检出 ⟹ 0%。

    这条红了 ⟹ `_CSV_SKIP` 漏了某列，虚高的检出率会让该补的场次被跳过。
    """
    p = _write_csv(tmp_path / "a.csv", [
        {"session_id": "s", "timestamp": "1", "timestamp_iso": "2026-01-01T00:00:00",
         "dominant_emotion": "", "face_scale": "", "bs_brow_down": ""},
        {"session_id": "s", "timestamp": "2", "timestamp_iso": "2026-01-01T00:00:01",
         "dominant_emotion": "", "face_scale": "", "bs_brow_down": ""},
    ], FIELDS)
    assert bf.detection_rate(p) == 0.0


def test_a_row_with_a_real_feature_counts_as_detected(tmp_path):
    """有真特征值的行 = 检出。

    这条红了 ⟹ 判据反了（会把可用的场次报成 0%）。
    """
    p = _write_csv(tmp_path / "a.csv", [
        {"session_id": "s", "timestamp": "1", "timestamp_iso": "T",
         "dominant_emotion": "", "face_scale": "", "bs_brow_down": ""},
        {"session_id": "s", "timestamp": "2", "timestamp_iso": "T",
         "dominant_emotion": "anger", "face_scale": "0.42", "bs_brow_down": "0.11"},
    ], FIELDS)
    assert bf.detection_rate(p) == 0.5


def test_dominant_emotion_is_the_column_that_exists(tmp_path):
    """★ 列名是 `dominant_emotion`，**不是** `emotion`。

    这是我自己踩过的坑：第一版查 `r.get('emotion')`，而那列不存在 ⟹
    每一场（含本该 73% 的）都被算成 0%。
    这条红了 ⟹ 列名又写错了，所有场次会一起变成 0%。
    """
    p = _write_csv(tmp_path / "a.csv", [
        {"session_id": "s", "timestamp": "1", "timestamp_iso": "T",
         "dominant_emotion": "anger", "face_scale": "0.5", "bs_brow_down": "0.2"},
    ], FIELDS)
    assert bf.detection_rate(p) == 1.0


def test_no_face_sentinel_row_is_not_a_detection(tmp_path):
    """★ 防御：只写了 `dominant_emotion='no_face'` 的行不算检出。

    当前素材上这是 no-op（215 份 CSV 里 `no_face` 一次都没出现过，"无脸"就是整行留空）。
    这条红了 ⟹ 有人把哨兵值写进了 CSV，而无脸行被算成检出 ⟹ 检出率静默虚高，
    该补的场次会被判成"已经好了"。
    判据与 `face_expression.pipeline.framing.is_detected` 对齐。
    """
    p = _write_csv(tmp_path / "a.csv", [
        {"session_id": "s", "timestamp": "1", "timestamp_iso": "T",
         "dominant_emotion": "no_face", "face_scale": "", "bs_brow_down": ""},
        {"session_id": "s", "timestamp": "2", "timestamp_iso": "T",
         "dominant_emotion": "anger", "face_scale": "0.5", "bs_brow_down": "0.2"},
    ], FIELDS)
    assert bf.detection_rate(p) == 0.5


@pytest.mark.parametrize("val", ["", "None"])
def test_none_string_counts_as_empty(tmp_path, val):
    """`'None'` 这个字符串也算空 —— 它在 CSV 里是 Python `None` 落盘的产物。

    这条红了 ⟹ 会把一堆 `None` 当成检出（判据从"有数值"退化成"有字"）。
    """
    p = _write_csv(tmp_path / "a.csv", [
        {"session_id": "s", "timestamp": "1", "timestamp_iso": "T",
         "dominant_emotion": val, "face_scale": val, "bs_brow_down": val},
    ], FIELDS)
    assert bf.detection_rate(p) == 0.0


# ──────────────────────────────────────────────────────────────
# ledger_sig：例行回填（--pending）的判据
# ──────────────────────────────────────────────────────────────

def test_ledger_sig_is_none_without_a_ledger(tmp_path):
    """没有台账就没有指纹 —— 调用方据此把这场排除在外（没留存帧就无从重放）。

    这条红了 ⟹ 没台账的场次会被判成"变过"，白白进重放流程然后失败。
    """
    assert bf.ledger_sig(tmp_path) is None


def test_ledger_sig_changes_when_the_ledger_grows(tmp_path):
    """★ 核心：台账长了，指纹必须变。

    这条红了 ⟹ 新录的帧不会被发现，`--pending` 会**漏掉刚录完的场次**
    （例行回填静默失效，而它现在是从"补救"变成"必需"的一环）。
    """
    led = tmp_path / "retention.face.jsonl"
    led.write_text('{"kind":"frame"}\n', encoding="utf-8")
    before = bf.ledger_sig(tmp_path)
    assert isinstance(before, list) and len(before) == 2
    led.write_text('{"kind":"frame"}\n{"kind":"frame"}\n', encoding="utf-8")
    assert bf.ledger_sig(tmp_path) != before, "台账长了，指纹却没变"


def test_ledger_sig_is_stable_when_nothing_changes(tmp_path):
    """★ 反过来：没变就必须**一模一样**。

    这条红了 ⟹ 指纹里混进了会漂的东西（时间戳、随机数、内存地址），
    于是每次 `--pending` 都把全语料重跑一遍 —— 那 `--pending` 就白加了。
    """
    (tmp_path / "retention.face.jsonl").write_text('{"kind":"frame"}\n', encoding="utf-8")
    assert bf.ledger_sig(tmp_path) == bf.ledger_sig(tmp_path)


# ──────────────────────────────────────────────────────────────
# _segments：★ 时间戳回退 = 活跑 /reset 过 ⟹ 必须换新管线
# ──────────────────────────────────────────────────────────────

def _frames(ts_list):
    """只关心时间戳，路径给 None（`_segments` 不看路径）。"""
    return [(None, t) for t in ts_list]


def test_segments_monotonic_is_one_segment():
    """全程递增 ⟹ 只有一段（常规场次不该被切开）。

    这条红了 ⟹ 单调的场次也被切碎，每段重建管线，白花时间且滚动特征被反复重置。
    """
    segs = bf._segments(_frames([0, 100, 200, 300]))
    assert len(segs) == 1
    assert segs[0] == [0, 1, 2, 3]


def test_segments_splits_at_a_timestamp_regression():
    """★ 核心：时间戳回退处必须切开。

    实测 `..._1320` 是**两段录制拼起来的**（帧 491 处时钟从 611829 归零到 0）。
    活跑在 `/reset` 时把 `SessionClock` 和 `VideoPipeline` 一起 pop 了，所以能过；
    重放若不换管线，mediapipe 抛 "timestamp_ms 回退"，而守卫的"上一次"会永远停在
    回退前那个值 ⟹ **回退点之后所有帧全被跳过**（实测丢了 612/1491 行）。

    这条红了 ⟹ 又回到"一条管线喂到底"，那场会静默丢 40% 的行。
    """
    segs = bf._segments(_frames([0, 100, 200, 0, 100, 200]))
    assert len(segs) == 2, f"应当在归零处切开，实得 {segs}"
    assert segs[0] == [0, 1, 2]
    assert segs[1] == [3, 4, 5]


def test_segments_treats_equal_timestamps_as_a_cut():
    """相等也算回退（mediapipe 要求**严格**递增，不是不递减）。

    这条红了 ⟹ 撞值的两帧会被喂成非法序列，第二帧起报错。
    """
    segs = bf._segments(_frames([0, 100, 100, 200]))
    assert len(segs) == 2
    assert segs[0] == [0, 1]
    assert segs[1] == [2, 3]


def test_segments_handles_multiple_resets():
    """多次重置 ⟹ 多段。"""
    segs = bf._segments(_frames([0, 10, 0, 10, 0, 10]))
    assert len(segs) == 3


def test_segments_never_loses_a_frame():
    """★ 所有帧都必须落在某一段里 —— 段划分不许吞帧。

    这条红了 ⟹ 丢行（本轮修的正是这个形态）。
    """
    ts = [0, 100, 200, 0, 100, 0, 50, 900]
    segs = bf._segments(_frames(ts))
    flat = [i for seg in segs for i in seg]
    assert flat == list(range(len(ts))), f"帧序号不完整: {flat}"
