# tests/test_none_bucket_rows.py
"""A4:`LogDataLoader._none_bucket_rows()` 的语义钉子。

## 这个数是什么

NONE 桶(没带 `session_id` 的日志)不进任何聚合,但**存在多少行**要告诉读者 ——
报告头里那句「另有 N 行日志没带会话编号」就是它(`sources_disclosure`)。

## A4 改了什么

旧写法:每份 NONE 文件 `pd.read_csv` 整读一遍,只为拿 `len(df)`;
新写法:`csv.reader` 数行,结构不规整的文件交回 pandas。

## 本文件守什么

不是"新实现能跑",而是**"新实现与 pandas 整读逐字节同答"**:

* 每条用例都比对 `_count_none_rows(p)` 与 `_pandas_rows(p)`(后者是把旧写法逐字
  搬过来的参考实现),所以断言的是**等价**而不是"某个我猜的数";
* 覆盖:正常 / 表头 / 空文件 / 表头只有一份但有行 / 字段数不符(多、少)/
  文件不存在 / 指向目录 / 权限不可读 / 引号里有逗号与换行 / BOM / GBK /
  只有空白的行 / 幂等(连问三次同一个数);
* 另有一条"真实数据存在才跑"的用例:拿盘上真实 NONE 桶比一次
  —— 没有大数据的环境不受影响(自动 skip)。
"""

import random

import pandas as pd
import pytest

from report_frontend.data_loader import LogDataLoader

_ENC_CHAIN = ("utf-8", "gbk", "gb2312", "utf-8-sig")


def _pandas_rows(path) -> int:
    """**旧写法的参考实现**(逐字照搬 A4 之前的循环体 + `_read_csv_safe` 的编码链)。

        try:
            df = loader._read_csv_safe(p)
            total += 0 if df is None else len(df)
        except Exception:
            pass
    """
    try:
        for enc in _ENC_CHAIN:
            try:
                return len(pd.read_csv(path, encoding=enc))
            except (UnicodeDecodeError, ValueError):
                continue
        return 0
    except Exception:
        return 0


def _loader(log_dir) -> LogDataLoader:
    return LogDataLoader(str(log_dir))


def _write(log_dir, name: str, data: bytes):
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / name
    p.write_bytes(data)
    return p


def _both(p, log_dir) -> tuple[int, int]:
    """返回 (新实现, pandas 参考实现) —— 每个用例都必须相等。"""
    return _loader(log_dir)._count_none_rows(p), _pandas_rows(p)


# ── 1. 正常 ────────────────────────────────────────────────────────────

def test_counts_header_plus_rows_like_pandas(tmp_path):
    """正常:3 行数据 → 3(表头不算)。"""
    p = _write(tmp_path, "face_au_log_NONE.csv",
               b"session_id,timestamp,focus_score\nNONE,0.0,0.1\nNONE,1.0,0.2\nNONE,2.0,0.3\n")
    new, old = _both(p, tmp_path)
    assert new == old == 3, f"数到 {new} 行(参考实现 {old} 行),应当是 3"


def test_counts_across_multiple_files_and_subdirs(tmp_path):
    """聚合口径:递归到子目录、文件名不带尾段/带尾段两种形态都要数。"""
    _write(tmp_path, "face_au_log_NONE.csv", b"a,b\n1,2\n3,4\n")
    _write(tmp_path / "sub", "interview_emotion_log_NONE_20260924_221800.csv",
           b"a,b\n1,2\n")
    _write(tmp_path, "gesture_emotion_log_20260924_221800_abcd.csv", b"a,b\n1,2\n9,9\n")
    got = _loader(tmp_path)._none_bucket_rows()
    want = sum(_pandas_rows(p) for p in tmp_path.rglob("*_log_NONE*.csv"))
    assert got == want == 3, (
        f"数到 {got} 行(参考 {want}):两份 NONE 是 2+1,非 NONE 那份不该进来")


def test_glob_is_a_substring_match_and_that_is_unchanged(tmp_path):
    """glob 是**子串**匹配:`*_log_NONE*` 连 `_log_NONEX` 也会命中。

    这是 A4 **之前**就有的行为(通配对 A4 是常量),在这里钉住:
    优化只许改"怎么数",不许改"数哪些文件"。
    """
    _write(tmp_path, "face_au_log_NONE.csv", b"a,b\n1,2\n")
    _write(tmp_path, "face_au_log_20260924_221800_abcd.csv", b"a,b\n1,2\n3,4\n5,6\n")
    _write(tmp_path, "face_au_log_NONEX.csv", b"a,b\n1,2\n3,4\n")
    got = _loader(tmp_path)._none_bucket_rows()
    want = sum(_pandas_rows(p) for p in tmp_path.rglob("*_log_NONE*.csv"))
    assert got == want, f"通配口径变了:新 {got} vs 旧 {want}"
    assert got == 3, f"数到 {got} 行:1 行(face_au_log_NONE)+ 2 行(_log_NONEX 被子串命中)"


# ── 2. 表头 / 空文件 ───────────────────────────────────────────────────

@pytest.mark.parametrize("data", [
    b"",                                     # 真空文件
    b"\n\n\n",                               # 只有换取行
    b"a,b\n",                                # 只有表头
    b"a,b",                                  # 只有表头且没结尾换行
    b"   \n",                                # 只有空白行
    b"a,b\r\n",                              # CRLF 只有表头
])
def test_empty_and_header_only_count_zero(tmp_path, data):
    """空文件 / 只有表头 / 只有空白行 → 0 行(与 pandas 的 EmptyDataError 同答)。"""
    p = _write(tmp_path, "face_au_log_NONE.csv", data)
    new, old = _both(p, tmp_path)
    assert new == old == 0, f"{data!r} 数到 {new} 行(参考实现 {old}),应当是 0"


def test_blank_lines_are_skipped_like_pandas(tmp_path):
    """文件中间的空行不算数据行(pandas 的 `skip_blank_lines` 默认开)。"""
    p = _write(tmp_path, "face_au_log_NONE.csv", b"a,b\n1,2\n\n3,4\n\n")
    new, old = _both(p, tmp_path)
    assert new == old == 2


def test_header_column_lists_need_not_match_across_files(tmp_path):
    """各份 NONE 文件的表头**列可以不同**(face/gesture/voice 三套各是各的),
    数行只看行数,与列名无关。"""
    _write(tmp_path, "face_au_log_NONE.csv", b"session_id,au1,au2\nNONE,0.1,0.2\n")
    _write(tmp_path, "interview_emotion_log_NONE_20260924_221800.csv",
           b"session_id,pitch_mean\nNONE,120.5\nNONE,118.0\n")
    assert _loader(tmp_path)._none_bucket_rows() == 3


def test_short_row_counts_and_long_row_matches_pandas(tmp_path):
    """字段数不符时不猜,必须与 pandas 同答。

    * **比表头少**的行:pandas 补 NaN,但仍算一行;
    * **比表头多**的行:pandas 在"不止一行数据"时报 ParserError,``_read_csv_safe``
      跑完四条编码链 → None → **整份算 0 行**(历史行为,不许改)。
    """
    short = _write(tmp_path / "short", "face_au_log_NONE.csv", b"a,b,c\n1,2\n3,4,5\n")
    new, old = _both(short, tmp_path / "short")
    assert new == old == 2, f"短行那份:{new} vs pandas {old}"

    long_ = _write(tmp_path / "long", "face_au_log_NONE.csv", b"a,b\n1,2\n3,4,5\n")
    new, old = _both(long_, tmp_path / "long")
    assert new == old, f"多字段那份:{new} vs pandas {old}"
    assert new == 0, "pandas 对多出来的字段报 ParserError ⟹ 这份应当是 0 行"


# ── 3. 文件不存在 / 目录 / 读不出 ───────────────────────────────────────

def test_missing_file_counts_zero(tmp_path):
    """文件不存在:旧实现是 FileNotFoundError 被外层吞掉 → 0;新实现同答。"""
    missing = tmp_path / "face_au_log_NONE.csv"
    assert _loader(tmp_path)._count_none_rows(missing) == 0


def test_directory_named_like_a_none_file_counts_zero(tmp_path):
    """rglob 会把**目录**也匹配上(`*_log_NONE*.csv`)。
    旧实现 pandas 抛 IsADirectoryError → 被吞 → 0;新实现不许抛、也不许数成别的。"""
    (tmp_path / "face_au_log_NONE.csv").mkdir()
    _write(tmp_path, "face_au_log_NONE_20260924_221800.csv", b"a,b\n1,2\n")
    assert _loader(tmp_path)._none_bucket_rows() == 1


@pytest.mark.skipif(not hasattr(__import__("os"), "geteuid"), reason="仅 POSIX")
def test_unreadable_file_counts_zero(tmp_path):
    """权限不足:两侧都是 0,且**不抛**。"""
    import os

    if os.geteuid() == 0:                     # root 读得动,这条在那台机器上没意义
        pytest.skip("以 root 运行,权限位不起作用")
    p = _write(tmp_path, "face_au_log_NONE.csv", b"a,b\n1,2\n")
    p.chmod(0o000)
    try:
        assert _loader(tmp_path)._count_none_rows(p) == 0
    finally:
        p.chmod(0o644)


# ── 4. 引号 / 逗号 / 换行 / 编码 ────────────────────────────────────────

def test_quoted_comma_and_newline_are_one_row(tmp_path):
    """引号里的逗号与**换行**属于同一个字段,不能切行 —— 这正是
    `sum(1 for _ in f)` 数不对的原因。"""
    data = (b'session_id,feedback\n'
            b'NONE,"a, b, c"\n'                       # 引号里有逗号
            b'NONE,"first line\nsecond line"\n'       # 引号里有换行
            b'NONE,"he said ""hi"""\n')               # 双写转义
    p = _write(tmp_path, "interview_emotion_log_NONE.csv", data)
    new, old = _both(p, tmp_path)
    assert new == old == 3, f"数到 {new} 行(参考实现 {old}),应当是 3"


def test_unclosed_quote_matches_pandas(tmp_path):
    """引号没闭合:pandas 报 ParserError("EOF inside string")→ 四条编码链跑完 → 0 行。
    新实现也必须交出 0(而不是把后面几行吃成一条)。"""
    p = _write(tmp_path, "interview_emotion_log_NONE.csv", b'a,b\n"oops,2\n3,4\n')
    new, old = _both(p, tmp_path)
    assert new == old == 0


def test_bom_is_neither_counted_nor_breaks_the_header(tmp_path):
    """UTF-8 BOM(`utf-8-sig` 那份编码就是为它留的):行数不变。"""
    p = _write(tmp_path, "face_au_log_NONE.csv",
               b"\xef\xbb\xbfsession_id,timestamp\nNONE,1\nNONE,2\n")
    new, old = _both(p, tmp_path)
    assert new == old == 2


def test_gbk_file_counts_like_pandas(tmp_path):
    """GBK 中文(编码链第 2 位):行数不变。"""
    text = "session_id,feedback\nNONE,中文反馈\nNONE,第二条\n"
    p = _write(tmp_path, "interview_emotion_log_NONE.csv", text.encode("gbk"))
    new, old = _both(p, tmp_path)
    assert new == old == 2


@pytest.mark.parametrize("data", [
    b"\xff\xfe\x00\x00garbage",               # 四条编码全解不开
    b"\x80\x81\x82\x83\x84",
])
def test_undecodable_file_counts_zero(tmp_path, data):
    """四条编码全解不开 → 与 pandas 一样算 0 行(不是抛,也不是猜)。"""
    p = _write(tmp_path, "face_au_log_NONE.csv", data)
    new, old = _both(p, tmp_path)
    assert new == old == 0


def test_crlf_and_cr_line_endings_match_pandas(tmp_path):
    """本仓 NONE 桶**全是 CRLF**;裸 CR 行尾也必须同答(pandas 与 csv 在这里切行规则
    不同,所以这条同时守"裸 CR 要交回 pandas"那条路)。"""
    for name, data in {
        "crlf": b"a,b\r\n1,2\r\n3,4\r\n",
        "cr": b"a,b\r1,2\r3,4\r",
        "mixed": b"a,b\r\n1,2\n3,4\r\n",
    }.items():
        sub = tmp_path / name
        p = _write(sub, "face_au_log_NONE.csv", data)
        new, old = _both(p, sub)
        assert new == old, f"{name}: 新 {new} vs pandas {old}"


# ── 5. 幂等 ────────────────────────────────────────────────────────────

def test_idempotent_and_stateless(tmp_path):
    """连问三次同一个数;并且"先数一遍别的目录"不影响结果(没有跨调用状态)。"""
    _write(tmp_path / "a", "face_au_log_NONE.csv", b"a,b\n1,2\n3,4\n")
    _write(tmp_path / "b", "face_au_log_NONE.csv", b"a,b\n1,2\n")
    la, lb = _loader(tmp_path / "a"), _loader(tmp_path / "b")
    assert [la._none_bucket_rows() for _ in range(3)] == [2, 2, 2]
    assert lb._none_bucket_rows() == 1
    assert la._none_bucket_rows() == 2, "数过别的目录之后同一个 loader 的结果不许变"


# ── 6. 与 pandas 的随机对照(小、快、固定种子)──────────────────────────

def test_random_small_files_match_pandas(tmp_path):
    """固定种子的随机小文件:正常/多字段/少字段/空行/引号/BOM/GBK 混着来,
    每份都必须与 pandas 整读同答。"""
    rng = random.Random(20261121)
    pieces = ["1", "x", "", " ", "a,b", '"q"', '"a\nb"', '"unclosed', "中文"]
    n = 0
    for i in range(60):
        ncol = rng.randint(1, 4)
        rows = [",".join(f"c{k}" for k in range(ncol))]
        for _ in range(rng.randint(0, 5)):
            rows.append(",".join(rng.choice(pieces) for _ in range(ncol)))
        text = ("\r\n" if rng.random() < 0.5 else "\n").join(rows)
        if rng.random() < 0.8:
            text += "\r\n" if rng.random() < 0.5 else "\n"
        enc = "gbk" if rng.random() < 0.15 else "utf-8"
        try:
            data = text.encode(enc)
        except UnicodeEncodeError:
            data = text.encode("utf-8")
        if rng.random() < 0.1:
            data = b"\xef\xbb\xbf" + data
        sub = tmp_path / f"f{i}"
        p = _write(sub, "face_au_log_NONE.csv", data)
        new, old = _both(p, sub)
        assert new == old, f"第 {i} 份({data!r}):新 {new} vs pandas {old}"
        n += 1
    assert n == 60


# ── 7. 真实数据(没有就 skip)──────────────────────────────────────────

def test_real_none_bucket_matches_pandas_when_present():
    """盘上真有 NONE 桶时:逐份与 pandas 比一次,并确认聚合值没变。

    仓库 `data/logs` 在干净检出里带 82 份 NONE 文件 / 832 行;换台机器没有大数据时
    这条自动 skip —— 它守的是真素材上的等价,**不该**变成"必须有素材才能跑"的门槛。
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    log_dir = repo / "data" / "logs"
    files = sorted(log_dir.rglob("*_log_NONE*.csv")) if log_dir.is_dir() else []
    if not files:
        pytest.skip("盘上没有 NONE 桶素材(干净环境),跳过真实数据对照")

    loader = LogDataLoader(str(log_dir))
    total = 0
    for p in files:
        new = loader._count_none_rows(p)
        old = _pandas_rows(p)
        assert new == old, f"{p.name}:新 {new} 行 vs pandas {old} 行"
        total += new
    assert loader._none_bucket_rows() == total, "聚合值与逐份之和必须相等"
    # 这一份数会进报告头;把"数出来是多少"固定下来,免得改坏了没人发现
    assert total > 0, "真实素材数出 0 行 —— 通配或数行逻辑回归了"
