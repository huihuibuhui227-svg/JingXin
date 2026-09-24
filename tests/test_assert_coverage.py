# tests/test_assert_coverage.py
"""断言覆盖率检查:找出「声明了却从未执行」的 assert。

背景:报告层那一轮被同一类缺陷烧了**九次** —— 测试在改动前后都通过,因为它断言的不是
被守护的那个维度、fixture 在到达被守护路径前就 short-circuit、循环体一次都没跑。其中两次
是靠一份临时脚本(记录每个测试真正执行了哪些行,再与 ast 里的 assert 行对差)才抓到的,
那份脚本一直没进仓库。本文件把它固化成常驻测试。

做法:用 sys.settrace 记录每个 test item 在 **call 阶段**真正执行过的、位于被扫描测试模块
内的源码行(ast 解析出的 assert 行号与之同一坐标系),两者对差,列出「从未跑到」的 assert。

⚠️ 盲区 —— 不要过度信任本检查:

1. **它只能抓「assert 从未执行」,抓不到「缺断言」。** 第九次失败模式是:测试跑了、断言了、
   也通过了,但断言的性质比 docstring 声称的弱(比如 docstring 说"必须带置信度",代码却只
   断言了"不为 None")。此时每一行 assert 都执行了,本检查给不出任何信号。这类缺陷只能靠
   人逐条读 docstring 与断言的对应关系。
2. **断言在执行的行上断言了错的对象**(维度写错)同样抓不到 —— 行执行了,语义错了。
3. **恒真断言**(如 `assert x == x`)抓不到。
4. **扫描范围只有 `tests/test_*.py`。** 放在别处的测试模块(别的目录、别的命名)完全不在
   覆盖范围内 —— 不报错,也不计数。往 `tests/` 之外加测试,本检查是静默的。

因此本检查的价值是**下界**:它保证每条 assert 至少被跑到过一次,不保证每条 assert 有意义。
内部还有三道防「假绿」的闸:nodeid 对不上 ast 表(unmapped)、被收集却没进 call 阶段
(never_called)、源码里是 test* 但没被收集(never_collected)—— 三者任一非空都会失败,
以免报告退化成一句「0 个问题」。
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SELF_PATH = Path(__file__).resolve()

# 地板:收集路径坏掉时必须响亮失败,而不是报「0 个问题」。
MIN_TESTS = 40
MIN_MODULES = 3


class CoverageReport(NamedTuple):
    """一次断言覆盖率检查的结果。"""

    n_tests: int  # call 阶段被追踪到的 test item 数
    n_modules: int  # 其中真正跑过测试、且 assert 表对得上的模块数
    n_assert_lines: int  # 解析到的 assert 语句总数
    unexecuted: dict[str, list[int]]  # nodeid → 从未执行过的 assert 行号
    unmapped: list[str]  # nodeid → ast 表里找不到对应测试函数的(静默盲区)
    never_called: list[str]  # nodeid → 被收集但从没进 call 阶段的(跳过/前置失败)
    never_collected: list[str]  # "模块::限定名" → 源码里是 test* 但 pytest 没收集它
    inner_exit_code: int  # 内层 pytest 运行的返回码


# ---------------------------------------------------------------- ast 侧


def _assert_lines_of(node: ast.AST) -> list[int]:
    return sorted({n.lineno for n in ast.walk(node) if isinstance(n, ast.Assert)})


def assert_map_from_source(source: str) -> dict[str, list[int]]:
    """源码里 测试函数限定名 → 函数体内 assert 语句的行号。

    限定名与 pytest nodeid 的中间段一致:类方法为 ``Class::method``,模块级函数为 ``func``。
    嵌套函数里的 assert 一并计入(它们是该测试的一部分,没跑到就是没跑到)。
    """
    found: dict[str, list[int]] = {}

    def walk(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test"):
                    found[prefix + node.name] = _assert_lines_of(node)
            elif isinstance(node, ast.ClassDef):
                walk(node.body, prefix + node.name + "::")

    walk(ast.parse(source).body)
    return found


def qualname_from_nodeid(nodeid: str) -> str | None:
    """nodeid → 测试限定名(与 ``assert_map_from_source`` 的键同一坐标系)。

    只取 nodeid 的 ``::`` 段 —— 路径段不用,因为它相对 rootdir,直接 abspath 会解析到
    错误位置(实测:CWD 恰好是仓库根时真套件能对上,tmp_path 里的模块则静默对不上,
    报告退化成「0 个问题」)。模块绝对路径改由 ``item.path`` 取。
    """
    parts = nodeid.split("::")
    if len(parts) < 2:
        return None
    func = parts[-1].split("[")[0]
    cls = parts[-2] if len(parts) >= 3 else None
    return f"{cls}::{func}" if cls else func


# ---------------------------------------------------------------- 追踪侧


class _TracePlugin:
    """记录每个 test item 在 call 阶段执行过的、被扫描模块内的源码行。"""

    def __init__(self, tracked: set[Path]) -> None:
        self._tracked = {os.path.abspath(str(p)) for p in tracked}
        self._current: str | None = None
        self.executed: dict[str, set[tuple[str, int]]] = {}
        self.modules: dict[str, str] = {}
        self.collected: list[str] = []

    def _trace(self, frame, event, arg):
        if event == "call":
            # 不在 call 阶段、或该帧不在被扫描模块内 → 不给这个帧开行事件。
            # 全局 trace 函数仍会对它调用的新帧触发,所以测试里调用的辅助函数照样被记。
            if self._current is None or os.path.abspath(frame.f_code.co_filename) not in self._tracked:
                return None
            return self._trace
        if event == "line" and self._current is not None:
            self.executed[self._current].add(
                (os.path.abspath(frame.f_code.co_filename), frame.f_lineno)
            )
        return self._trace

    def pytest_collection_modifyitems(self, items):
        # 模块路径在这里就记下(不只在 call 阶段):被跳过的测试不进 call 阶段,
        # 但它在 never_collected 里不该被误报成「没收集到」。
        for item in items:
            self.collected.append(item.nodeid)
            self.modules[item.nodeid] = os.path.abspath(str(item.path))

    # hookwrapper=True 而非裸生成器:pytest 9 下裸生成器的 hookimpl 不会被当成包装器,
    # hook 静默不触发(追踪不到任何行,却报「0 个问题」——正是本文件要抓的那种假绿)。
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        self._current = item.nodeid
        self.executed.setdefault(item.nodeid, set())
        previous = sys.settrace(self._trace)
        try:
            yield
        finally:
            sys.settrace(previous)
            self._current = None


# ---------------------------------------------------------------- 检查器


def check_assert_coverage(paths) -> CoverageReport:
    """跑 ``paths`` 里的测试模块(在**本进程内**跑,不起子进程),返回覆盖率报告。

    与 ``pytest tests/`` 的区别只有一条:多了 call 阶段的行追踪。测试本身的行为不变。
    """
    modules = [Path(p).resolve() for p in paths]
    by_module: dict[str, dict[str, list[int]]] = {}
    n_assert_lines = 0
    for module in modules:
        parsed = assert_map_from_source(module.read_text(encoding="utf-8"))
        by_module[str(module)] = parsed
        n_assert_lines += sum(len(lines) for lines in parsed.values())

    plugin = _TracePlugin(set(modules))
    args = [str(m) for m in modules] + ["-q", "--no-header", "-p", "no:cacheprovider"]
    exit_code = pytest.main(args, plugins=[plugin])

    unexecuted: dict[str, list[int]] = {}
    unmapped: list[str] = []
    modules_with_tests: set[str] = set()
    for nodeid, executed in plugin.executed.items():
        module = plugin.modules.get(nodeid)
        qualname = qualname_from_nodeid(nodeid)
        expected = by_module.get(module, {}).get(qualname) if module and qualname else None
        if expected is None:
            # 对不上 = 这个测试的 assert 根本没进对比表,是一处静默的盲区,必须报出来。
            unmapped.append(nodeid)
            continue
        modules_with_tests.add(module)
        ran = {lineno for filename, lineno in executed if filename == module}
        missing = sorted(set(expected) - ran)
        if missing:
            unexecuted[nodeid] = missing

    # 被收集但从没进 call 阶段(跳过 / 前置失败):它的 assert 一行都没被核实。
    never_called = sorted(set(plugin.collected) - set(plugin.executed))

    # 源码里写着 test* 却压根没被 pytest 收集(改错名、放在非 Test* 的类里……)。
    collected_per_module: dict[str, set[str]] = {}
    for nodeid in plugin.collected:
        qualname = qualname_from_nodeid(nodeid)
        module = plugin.modules.get(nodeid)
        if qualname and module:
            collected_per_module.setdefault(module, set()).add(qualname)
    never_collected = sorted(
        f"{Path(module).name}::{qualname}"
        for module, declared in by_module.items()
        for qualname in set(declared) - collected_per_module.get(module, set())
    )

    return CoverageReport(
        n_tests=len(plugin.executed),
        n_modules=len(modules_with_tests),
        n_assert_lines=n_assert_lines,
        unexecuted=unexecuted,
        unmapped=sorted(unmapped),
        never_called=never_called,
        never_collected=never_collected,
        inner_exit_code=exit_code,
    )


def suite_paths() -> list[Path]:
    """``tests/`` 下除本文件外的全部测试模块(不递归进本文件,避免自指死循环)。"""
    return sorted(p for p in TESTS_DIR.rglob("test_*.py") if p.resolve() != SELF_PATH)


# ---------------------------------------------------------------- 常驻检查


def test_no_assert_is_dead():
    """报告层那九次事故的兜底:每条 assert 都必须至少被执行过一次。"""
    report = check_assert_coverage(suite_paths())

    # 内层套件不绿时,「哪些行执行过」不再可信(失败的测试会停在半路),先修内层。
    assert report.inner_exit_code == 0, (
        f"内层套件没跑绿(返回码 {report.inner_exit_code}),此行覆盖率结论不可信;先修内层"
    )

    # 地板先于结论:收集路径坏掉时要报「收集不到」,不是报「0 个问题」。
    assert report.n_tests >= MIN_TESTS, (
        f"只追踪到 {report.n_tests} 个测试(地板 {MIN_TESTS}),收集路径坏了"
    )
    assert report.n_modules >= MIN_MODULES, (
        f"只有 {report.n_modules} 个测试模块真的跑过(地板 {MIN_MODULES})"
    )
    assert report.n_assert_lines > 0, "一条 assert 都没解析到,ast 侧坏了"
    assert report.unmapped == [], (
        "以下测试对不上 ast 里的测试函数,它们的 assert 根本没进对比表(静默盲区):\n"
        + "\n".join(f"  {nodeid}" for nodeid in report.unmapped)
    )
    assert report.never_called == [], (
        "以下测试被收集但从未进入 call 阶段(跳过 / 前置失败),其 assert 一行都没被核实:\n"
        + "\n".join(f"  {nodeid}" for nodeid in report.never_called)
        + "\n(跳过的测试,其断言不在覆盖保证之内,且本检查不会替它背书)"
    )
    assert report.never_collected == [], (
        "以下函数源码里看着是测试,但 pytest 没收集到它(改名 / 不在 Test* 类里 / 有参数):\n"
        + "\n".join(f"  {name}" for name in report.never_collected)
    )

    assert report.unexecuted == {}, (
        "以下测试里有从未执行过的 assert(声明了却没跑到,等于没测):\n"
        + "\n".join(
            f"  {nodeid}  行 {lines}" for nodeid, lines in sorted(report.unexecuted.items())
        )
    )


# ---------------------------------------------------------------- 自证能红

_VACUOUS = '''\
def test_all_asserts_run():
    value = 2
    assert value == 2
    assert isinstance(value, int)


def test_assert_after_early_return():
    value = 1
    assert value == 1
    if value == 1:
        return
    assert value == 99, "early-return 之后永远到不了"


def test_assert_in_never_entered_loop():
    total = 0
    for item in []:
        assert item is not None, "空列表里的肢体"
    assert total == 0


class TestSyntheticClass:
    """类方法的限定名是 "Class::method",与模块级函数不是同一条映射路径。"""

    def test_method_all_asserts_run(self):
        assert 1 + 1 == 2

    def test_method_with_dead_assert(self):
        flag = False
        assert flag is False
        if flag:
            assert flag is True, "flag 为假时到不了"
'''


def _line_of(source: str, snippet: str) -> int:
    """源码里唯一一处含 ``snippet`` 的 assert 语句的行号。"""
    hits = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assert) and snippet in ast.unparse(node)
    ]
    assert len(hits) == 1, f"{snippet!r} 命中 {len(hits)} 处,预期 1 处"
    return hits[0]


def test_checker_flags_dead_asserts_and_clears_sound_ones(tmp_path):
    """检查器必须能红。不能红的检查器,正是它自己要抓的那类缺陷。

    合成模块共 5 个测试:2 个全部跑到(模块级 1 + 类方法 1),3 个带死 assert(提前 return 1 +
    空循环 1 + 类方法 1)。检查器必须只点名那 3 个带死 assert 的,行号精确指向死掉的那一条,
    且不得牵连跑到的断言。
    """
    module = tmp_path / "test_synthetic_assert_coverage.py"
    module.write_text(_VACUOUS, encoding="utf-8")

    report = check_assert_coverage([module])

    assert report.n_tests == 5, f"合成分支应被收集到 5 个测试,实际 {report.n_tests}"
    assert report.n_modules == 1
    assert report.unmapped == [], f"合成分支没对上 ast 表:{report.unmapped}"
    assert report.inner_exit_code == 0, "合成模块本身应当是绿的 —— 死断言不等于失败断言"

    flagged = {nodeid.split("::")[-1]: lines for nodeid, lines in report.unexecuted.items()}
    assert set(flagged) == {
        "test_assert_after_early_return",
        "test_assert_in_never_entered_loop",
        "test_method_with_dead_assert",
    }, f"点名错误:{sorted(flagged)}"
    assert flagged["test_assert_after_early_return"] == [_line_of(_VACUOUS, "99")]
    assert flagged["test_assert_in_never_entered_loop"] == [
        _line_of(_VACUOUS, "item is not None")
    ]
    assert flagged["test_method_with_dead_assert"] == [_line_of(_VACUOUS, "flag is True")]
    # 全跑到的两个测试(模块级 + 类方法)都不得被点名 —— 否则检查器是「见谁都喊」,
    # 而不是「抓到真死断言」。
    assert not any("all_asserts_run" in nodeid for nodeid in report.unexecuted)
