# tests/test_prose_references.py
"""散文里的**引用**必须指得着 —— 文档条目标号 / 行号 / 测试名。

## 为什么需要这个文件

本仓对「散文里的引用会静默失效」有成套纪律,但**只管到了 L0 表**:

* `tests/test_l0_reference_paths.py` —— 裸引用要**恰好命中一个**跟踪文件、行号要在界内、
  豁免表不许闲置。它的 `_refs()` / `_naked_refs()` 都是从 `l0.load()` 取的 ⟹ **只扫表**。
* `tests/test_the_table_carries_no_line_number_references` —— 同样只扫表。

于是**文档正文与测试代码里的引用没有任何守卫**。2026-10-03 复核时实测踩到:
我给「丙」加的条目编号是 11,而「丁」里**已经有一个 11** —— 两处引用都"指得着",
但指的是两个不同的东西,靠肉眼看不出来。

## 管什么(三条,判据都取**能精确判定**的)

0. 扫描面:根 `README.md` + 顶层 `docs/*.md` + 六个代码目录的 `*.py`;
1. `「甲、1」`/`「丙、17」` 这类**条目标号**引用 ⟹ `docs/下一步.md` 里必须真的有那个编号;
2. `<路径>.py:<行号>` ⟹ 目标文件存在时,行号不许超界(文件变短、引用指到尾部之外 ⟹ 红);
3. 被引用的**测试名** ⟹ `tests/` 里要有同名定义(或它是某个已定义名字的前缀,见下)。

## 不管什么(如实说,免得读的人以为它管得比实际宽)

* **不扫 `docs/superpowers/`** —— 那是历史时点记录,仓库规定不许改(`docs/下一步.md`「二、6」:
  "账本里的历史记录不改 —— 改了就成篡改")。那里的行号引用**本来就该过期**。
* **不判"路径写不写得对"** —— 散文里大量 `xxx.py` 是举例,不是引用。
  只判**已经能解析到唯一跟踪文件**的那些的行号是否越界。
* **不扫模块级文档串** —— 那里合法地写**格式示例**(如 `test_l0_reference_paths.py` 的
  文档串里有 `app.py:605 / :867` 这种示例)。判据只扫文档串以外的正文。
* **测试名允许"前缀引用"** —— 如正文里写 `test_segments_*` 指一整族;
  也允许出现在**记录改名的那一行**(含「原」/「改名」/「已移」/「曾」/「旧名」),
  因为本仓规矩是「更正必须能**引述**旧说法」。

⚠️ 测试名这条判据**刻意做得窄**:2026-10-03 设计它时连撞三次误报
(通配前缀、临时**文件名**、记录改名的引述),三次都是靠上面这组豁免消掉的。
再放宽就会变成"白名单垃圾桶"(见 `test_l0_reference_paths` 的同族警告)——
若将来它开始频繁假红,**收窄判据或删掉它**,不要加豁免表。
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: 不扫的目录前缀(历史时点记录;见模块文档串)。
_SKIP_PREFIXES = ("docs/superpowers", "__pycache__")

#: **本文件自己**要豁免:它是"讲引用格式"的地方,文档串里必须能写**反例**
#: (如 `「丙、99」`、`framing.py:99999`、`test_this_never_existed_at_all`)来演示红法。
#: 不豁免的话,这些反例会被自己当成坏引用抓走 —— 那是"黑名单被自己引用的更正句误伤"
#: 的同族形态(`test_l0_column_table.py` 里记过这个坑)。
_SELF = Path(__file__).name

#: 记录"改名/搬迁"的行:那里出现旧测试名是**合法的引述**。
_RENAME_MARKERS = ("原 ", "原`", "改名", "已移", "曾", "旧名")


def _scan_files() -> list[Path]:
    """要扫的文件:根 `README.md` + 顶层 `docs/*.md` + 六个代码目录下的 `*.py`(不含本文件)。

    ⚠️ 根 `README.md` **必须在内**:它是最常被读的文档,而它 2026-10-03 之前
    一直不在任何引用守卫的射程里(本函数第一版也漏了它)。
    """
    readme = REPO / "README.md"
    out = ([readme] if readme.is_file() else []) + list((REPO / "docs").glob("*.md"))
    for r in ("tests", "report_frontend", "face_expression", "tools",
              "gesture_analysis", "voice_interaction"):
        out += [p for p in (REPO / r).rglob("*.py") if "__pycache__" not in str(p)]
    return [p for p in out
            if not str(p.relative_to(REPO)).startswith(_SKIP_PREFIXES)
            and p.name != _SELF]


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _body_without_module_docstring(p: Path) -> str:
    """挖掉模块级文档串 —— 那里合法地写格式示例。"""
    text = _read(p)
    if p.suffix != ".py":
        return text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    if not tree.body or not isinstance(tree.body[0], ast.Expr):
        return text
    first = tree.body[0]
    if not isinstance(getattr(first, "value", None), ast.Constant) or \
            not isinstance(first.value.value, str):
        return text
    lo, hi = first.lineno, (first.end_lineno or first.lineno)
    return "\n".join(l for i, l in enumerate(text.splitlines(), 1) if not (lo <= i <= hi))


def _tracked_files() -> list[str]:
    """git 跟踪的文件(相对路径)。

    ⚠️ 按 git 而不是 rglob:`test_l0_reference_paths` 记着,仓库根下有一个**未被跟踪的
    镜像目录**,rglob 会让 `logger.py` 这类名字**全都变成不唯一**。
    """
    try:
        out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout
    except Exception:
        return []
    return [l for l in out.splitlines() if l.strip()]


def _doc_item_numbers() -> set:
    txt = _read(REPO / "docs" / "下一步.md")
    return {int(m.group(1)) for m in re.finditer(r"^(\d+)\.\s", txt, re.M)}


def _defined_test_names() -> tuple:
    defined, stems = set(), set()
    for f in (REPO / "tests").glob("test_*.py"):
        stems.add(f.stem)
        defined |= set(re.findall(r"^\s*def (test_\w+)", _read(f), re.M))
    return defined, stems


def test_doc_item_references_point_at_real_items():
    """`「丙、17」` 这类条目标号引用,`docs/下一步.md` 里必须真的有那个编号。

    ★ 这条是在 2026-10-03 的**真实事故**之后加的:我往「丙」加的条目用了编号 11,
    而「丁」里**已经有一个 11** —— 两处引用都"指得着",但指的是两个不同的条目,
    肉眼看不出来。本测试命中**编号不存在**的情况;编号撞车另有下面那条钉着。

    红法(实测):把任一引用改成不存在的编号(如 `「丙、99」`)⟹ 红。
    """
    items = _doc_item_numbers()
    assert items, "读不到 `docs/下一步.md` 的条目标号 —— 判据本身失效了"

    bad = []
    for p in _scan_files():
        for m in re.finditer(r"「([甲乙丙丁戊])、(\d+)」", _read(p)):
            if int(m.group(2)) not in items:
                bad.append(f"{p.relative_to(REPO)}: 「{m.group(1)}、{m.group(2)}」")
    assert not bad, (
        "这些条目标号引用指向**不存在**的条目:\n  " + "\n  ".join(bad)
        + "\n—— 要么改引用,要么在 `docs/下一步.md` 里补上那一条。")


def test_doc_items_are_not_duplicated_within_one_section():
    """同一节里的条目编号不许重复 —— 重复会让 `「丙、11」` 这种引用**有歧义**。

    各节**各自从 1 开始**是本文件的既有约定(`docs/下一步.md` 的「二/三/四」都从 1 起),
    所以只查"同一节内不重复"。2026-10-03 我正是在「丙」里写出了与「丁」重复的 11。

    红法(实测):在同一节里加第二个 `11. ` ⟹ 红。

    ⚠️ 节键必须用**标题所在的行号**,不能用标题文字:文档里有两个都以
    「2026-09-27 的…」开头的 `###` 标题(§三、§四),按文字(哪怕截断)当键会把它们
    **并成一节**,于是把两节各自从 1 开始的合法编号误报成重复 —— 本测试第一版就是这么假红的。
    """
    lines = _read(REPO / "docs" / "下一步.md").splitlines()
    seen = {}          # (标题行号, 号) -> 条目所在行号
    dup = []
    sec = 0            # 最近一个标题的行号(0 = 文首)
    for i, l in enumerate(lines, 1):
        if l.startswith("### ") or l.startswith("## "):
            sec = i
        m = re.match(r"^(\d+)\.\s", l)
        if m:
            key = (sec, int(m.group(1)))
            if key in seen:
                dup.append(f"第 {sec} 行标题下:「{lines[sec - 1][:24]}…」的 "
                           f"{key[1]} 号在第 {seen[key]} 行与第 {i} 行各一次")
            else:
                seen[key] = i
    assert not dup, (
        "同一节里条目编号重复(引用会变得有歧义):\n  " + "\n  ".join(dup))


def test_line_references_in_prose_are_in_bounds():
    """正文里的 `<路径>.py:<行号>` 引用,行号不许超出那个文件的实际行数。

    为什么值得管:`test_l0_reference_paths` 已经为 **L0 表**立了同样的判据,并记着一次
    真实事故 —— 表里原有 78 处 `evidence_gate.py:NN`,B0 只往那个文件加了几行,
    78 处**集体静默失效**。文档与测试里的行号引用至今没有守卫。

    只管**能解析到唯一跟踪文件**的引用;解析不到的当举例放过(散文里大量 `xxx.py` 是举例)。

    红法(实测):把正文里任一已知文件的行号改成超出其行数的数(如 `framing.py:99999`)⟹ 红。
    """
    tracked = _tracked_files()
    assert tracked, "取不到 git 跟踪文件列表 —— 判据本身失效了"
    by_base = {}
    for f in tracked:
        by_base.setdefault(Path(f).name, []).append(f)
    line_count = {f: len(_read(REPO / f).splitlines()) for f in tracked}

    bad = []
    for p in _scan_files():
        for m in re.finditer(r"([A-Za-z_][\w/]*\.py):(\d+)", _body_without_module_docstring(p)):
            name, num = Path(m.group(1)).name, int(m.group(2))
            cands = by_base.get(name, [])
            if len(cands) != 1:
                continue                      # 举例 / 名字不唯一 —— 另有判据管表里的那种
            n = line_count.get(cands[0], 0)
            if num > n:
                bad.append(f"{p.relative_to(REPO)} -> {cands[0]}:{num}(该文件只有 {n} 行)")
    assert not bad, (
        "这些行号引用**超出目标文件的行数**(文件变短了?):\n  " + "\n  ".join(bad)
        + "\n—— 行号会随任何一次编辑静默失效,请改成**符号名**引用"
          "(`evidence_gate.QUARANTINE['x']` 那种),或把行号更新到正确位置。")


def test_cited_test_names_exist():
    """正文里点名的**测试名**必须真的存在 —— 改名之后旧名会变成一句空话。

    判据刻意窄(见模块文档串):允许**前缀引用**(`test_segments_*` 指一整族)、
    允许**文件名**(`test_report_layer.py`)、允许出现在**记录改名的那一行**
    (本仓规矩:更正必须能引述旧说法)。

    红法(实测):在任一非历史文件的正文里写一个不存在的测试名
    (如 `test_this_never_existed_at_all`)⟹ 红。
    """
    defined, stems = _defined_test_names()
    assert defined, "一条测试名都没读到 —— 判据本身失效了"

    bad = []
    for p in _scan_files():
        for line in _body_without_module_docstring(p).splitlines():
            if any(mk in line for mk in _RENAME_MARKERS):
                continue
            for tok in re.findall(r"\b(test_[a-z0-9_]{8,})\b", line):
                if f"{tok}.py" in line:
                    continue
                if tok in defined or tok in stems:
                    continue
                if any(d.startswith(tok) for d in defined):
                    continue
                bad.append(f"{p.relative_to(REPO)}: {tok}")
    assert not bad, (
        "这些被点名的测试**不存在**(改名了?):\n  " + "\n  ".join(bad)
        + "\n—— 要么改引用,要么那一行是在记录改名(那就带上「原」/「改名」/「已移」这类标记)。")
