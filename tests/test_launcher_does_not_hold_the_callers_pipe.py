"""起服务的脚本不许攥着调用方的 stdout —— 否则 `... | grep` 永远等不到 EOF。

## 为什么

`tools/m2_acceptance.sh` 第 24 行是:

    bash tools/start_all.sh 2>&1 | grep -E "✓|✗|已在跑" | sed 's/^/  /'

2026-10-25 实测:**这一行永不返回**。`timeout 600` 到了,日志只停在「── 起服务 ──」。
`docs/开发者文档.md` §8 第 12 条把它登记成"已知卡死点",建议"先单独把 `start_all.sh`
跑完"—— 于是这条验收得靠人**手工分两步**,而照着文档做的人会以为它自己卡死了。

根因在 `start_all.sh` 的 `start()`:

    ( cd "$cwd" && setsid "$@" > "/tmp/jx_$port.log" 2>&1 < /dev/null & )

重定向只加在 `setsid` 上,而 `cd ... && setsid ...` 这条**后台列表**还多套了一层子 shell。
那层子 shell **继承了调用方的 stdout**(那条管道),而且要等 `setsid` 起完才退出。
外层 `( )` 立刻返回、脚本退出、它变成孤儿 —— 它仍然握着管道的**写端**,于是 `grep`
永远等不到 EOF。

隔离实验(`&&` 是唯一变量,三轮稳定):

| 启动语句 | 管道是否返回 |
|---|---|
| `( setsid sleep 55 >/dev/null 2>&1 </dev/null & )` | ✅ 立刻 |
| `( cd /tmp && setsid sleep 55 >/dev/null 2>&1 </dev/null & )` | ❌ 不返回 |
| `( cd /tmp; setsid sleep 55 >/dev/null 2>&1 </dev/null & )` | ✅ 立刻 |

差别就在 `&&` 多出来的那层子 shell。修法:重定向**移到整组外面**,让任何后代都拿不到
调用方的 stdout:

    ( cd "$cwd" && setsid "$@" ) > "/tmp/jx_$port.log" 2>&1 < /dev/null &

## 这个文件守什么

把 `start_all.sh` 里**真的那个** `start()` 函数抽出来,起一个假服务(不起任何真服务),
然后照 m2 的接法用管道接住 —— 断言它**很快**返回。抽不到函数就红(而不是静默跳过)。
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START_ALL = ROOT / "tools" / "start_all.sh"

#: 假服务的进程名标记 —— 收尾时只按这个杀,绝不用宽匹配(会把跑测试的 shell 一起杀)。
_MARK = "JX_PIPE_PROBE"


def _start_fn() -> str:
    """从 `start_all.sh` 里抽出 `start()` 函数的原文。

    抽不到就**红**:这条测试的全部价值在于它跑的是**生产里那一行**,
    抽不到还"通过"就等于什么都没验。
    """
    text = START_ALL.read_text(encoding="utf-8")
    m = re.search(r"^start\(\)\s*\{.*?^\}", text, re.S | re.M)
    assert m, f"没在 {START_ALL} 里找到 `start()` 函数 —— 它被改名/改形了?"
    body = m.group(0)
    assert "setsid" in body, "`start()` 里没有 setsid —— 抽到的不是那个函数"
    return body


def test_starting_a_service_does_not_hold_the_callers_pipe(tmp_path):
    """★ 红法:把重定向挪回 `setsid` 那一侧 ⟹ 这里超时红。

    复刻 m2 的接法:`bash <脚本> 2>&1 | cat`。脚本只做一件事 —— 起一个假服务。
    真服务一个都不起(命令是 `bash -c 'exec -a <标记> sleep 7'`)。
    """
    probe = tmp_path / "probe.sh"
    probe.write_text(
        "set -u\n"
        + _start_fn() + "\n"
        "UP=() ; SKIP=() ; BAD=()\n"
        f'start 59998 "假服务" /tmp bash -c \'exec -a {_MARK} sleep 7\'\n'
        'echo "started"\n',
        encoding="utf-8")

    t0 = time.time()
    try:
        p = subprocess.run(["bash", "-c", f"bash {probe} 2>&1 | cat"],
                           capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", _MARK], capture_output=True)
        raise AssertionError(
            "起完假服务后,管道 20s 还没关 —— 有孤儿进程攥着调用方的 stdout。\n"
            "`start()` 里的重定向要移到**整组外面**:\n"
            '    ( cd "$cwd" && setsid "$@" ) > "/tmp/jx_$port.log" 2>&1 < /dev/null &')
    finally:
        subprocess.run(["pkill", "-9", "-f", _MARK], capture_output=True)

    elapsed = time.time() - t0
    assert p.returncode == 0, f"探针脚本退出码 {p.returncode}:{p.stderr.decode()[:400]}"
    assert b"started" in p.stdout, f"没看到 started:{p.stdout.decode()[:400]}"
    # 假服务活 7s;脚本本身应当**立刻**结束,而不是陪着它活完。
    assert elapsed < 5, f"管道 {elapsed:.1f}s 才关 —— 脚本在等那个孤儿进程"
