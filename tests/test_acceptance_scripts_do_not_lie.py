"""验收脚本不许**假绿**:任何一步失败,结尾就不许说"通过"。

## 为什么

2026-10-25 在服务器上跑 `tools/m26_media_acceptance.sh`,输出是这样的:

    tools/m26_media_acceptance.sh: line 48: .../questions.jsonl: No such file or directory
    Traceback (most recent call last):
    FileNotFoundError: .../questions.jsonl
    ...
    全部冒烟判据通过。真会话的 §7.7 清单在录制时另跑。

**中间抛了 Python 异常、第 ②③ 步一条断言都没执行,结尾照样说"全部通过"。**
原因是那两步没有 `|| exit`,而结尾那句是**无条件** `echo` 的。

这与本仓反复记的形态同族,但更坏一层:前面那些「静默失败」至少是**没说话**,
这一条是**说了反话** —— 照它做的人会以为验收过了。

## 这个文件守什么

1. 每条验收脚本的**最后一句结论**必须由退出码决定(`exit $FAIL` / `|| exit`),
   不许是无条件 `echo ...通过`。
2. 脚本里不许出现**写死的墙钟时间戳**:`session_meta._validate_window` 有一条
   量程闸(离此刻超过 1 天 ⟹ 400),而 m26 里写死的 `1758824000` 是 2025-09 的数
   ⟹ 2026-10-25 跑,第 ② 步**必然** 400。写死的时间戳是**会过期的代码**,
   而过期的那天脚本不会变红,只会静默少跑几步(见上面那条)。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

#: 真·端到端验收脚本(会用 curl 打服务、会判红)。`start_all.sh` / `jx_*.sh` 不是验收。
ACCEPTANCE = ("m2_acceptance.sh", "m21_acceptance.sh", "m26_media_acceptance.sh",
              "t7_acceptance.sh")

#: 十位数以上、形似 Unix 秒的整数字面量。**只查会执行的行**(注释里的示例不算)。
_EPOCH = re.compile(r"(?<![\w.])1[0-9]{9}(?:\.0)?(?![\w.])")


def _code_lines(text: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.split("#", 1)[0]
        if stripped.strip():
            out.append((i, stripped))
    return out


def _text(name: str) -> str:
    p = TOOLS / name
    assert p.is_file(), f"找不到 {p}"
    return p.read_text(encoding="utf-8")


def test_each_acceptance_script_ends_with_a_verdict_from_its_exit_code():
    """★ 红法:把结尾那句无条件 `echo ...通过` 加回去 ⟹ 这里红。

    判据落在**最后一条非空、非注释行**上:它必须是 `exit $VAR` 这种由变量决定的退出,
    而不是一条 `echo`。
    """
    bad = []
    for name in ACCEPTANCE:
        lines = _code_lines(_text(name))
        assert lines, f"{name} 是空的?"
        last_no, last = lines[-1]
        if not re.fullmatch(r"exit\s+\$\{?\w+\}?", last.strip()):
            bad.append(f"{name}:{last_no} 结尾不是 `exit $VAR` —— {last.strip()}")
    assert not bad, ("以下验收脚本的结论与退出码脱钩(失败也会说通过):\n  "
                     + "\n  ".join(bad))


def test_no_acceptance_script_hardcodes_a_wall_clock_timestamp():
    """★ 红法:写 `"ask_start":1758824000.0` ⟹ 这里红。

    那是 2025-09 的秒数,而 `session_meta` 的量程闸是「离此刻超过 1 天就 400」
    ⟹ 写死的那天起,这一步就静默不执行了(而旧版脚本结尾照样说"通过")。
    要用时间就当场算:`$(date +%s)` 或 Python 的 `time.time()`。
    """
    bad = []
    for name in ACCEPTANCE:
        for lineno, line in _code_lines(_text(name)):
            for m in _EPOCH.finditer(line):
                bad.append(f"{name}:{lineno} 写死了时间戳 {m.group(0)} —— {line.strip()}")
    assert not bad, ("以下时间戳会随日期过期(量程闸是「离此刻 1 天内」):\n  "
                     + "\n  ".join(bad))


def test_every_bad_call_feeds_the_exit_code():
    """★ 红法:把 `bad()` 里的 `FAIL=1` 删掉 ⟹ 这里红。

    `exit $FAIL` 只在**有东西真的置位**时才有意义。2026-10-25 发现 `m2_acceptance.sh`
    是这条的另一半:`bad()` 只打一行红字,结尾是 `exit $RC` —— 而 `$RC` 只装了
    **第 ⑤ 步**的结果 ⟹ 第 ①②③④ 步挂多少条,退出码都是 0。
    (与 m26/t7 同一天、同一形态;m26 最糟:结尾直接说"全部冒烟判据通过"。)

    判据:凡是调用了 `bad` 的脚本,`bad()` 的定义里必须**置位**某个变量
    (形如 `X=1`),且最后一行 `exit` 引用的正是那个变量。
    """
    problems = []
    for name in ACCEPTANCE:
        text = _text(name)
        if not re.search(r"(?m)^\s*bad\(\)", text):
            continue  # 不用 bad 的脚本(如 m21 直接 echo)由上面那条管
        m = re.search(r"(?ms)^bad\(\)\s*\{(.*?)^\}", text) or \
            re.search(r"(?m)^bad\(\)\s*\{(.*)\}\s*$", text)
        assert m, f"{name} 有 `bad()` 调用但找不到它的定义"
        body = m.group(1)
        setter = re.search(r"\b([A-Za-z_]\w*)=1\b", body)
        if not setter:
            problems.append(f"{name}: `bad()` 只打印,不置位 ⟹ 失败不影响退出码")
            continue
        var = setter.group(1)
        last = _code_lines(text)[-1][1]
        if var not in last:
            problems.append(f"{name}: `bad()` 置位 {var},但结尾 `{last.strip()}` 没引用它")
    assert not problems, "以下脚本的失败判定传不到退出码:\n  " + "\n  ".join(problems)
