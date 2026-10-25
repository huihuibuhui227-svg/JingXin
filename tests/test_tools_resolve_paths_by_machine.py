"""`tools/*.sh` 不许写死解释器与仓库路径 —— 一律走 `jx_env.sh` 解析。

## 为什么

本仓有**两台**机器在用同一批脚本,布局不同(§10.6 路径对照):

| | 笔记本(WSL) | 服务器 |
|---|---|---|
| 后端仓 | `~/jingxin` | `~/JingXin/jingxin` |
| 解释器 | `~/miniconda3/envs/jingxin/bin/python` | `~/anaconda3/envs/jingxin/bin/python` |

`jx_env.sh` 就是为了消掉这个差异而存在的(它自己按机器探两处),但**脚本们
不一定用了它**。2026-10-25 实测:`tools/m26_media_acceptance.sh` 仍然写死
`PY=~/miniconda3/envs/jingxin/bin/python` + `cd ~/jingxin` ⟹ 在服务器上**第一行就废**:

    tools/m26_media_acceptance.sh: line 14: cd: /home/zgy/jingxin: No such file or directory
    tools/m26_media_acceptance.sh: line 22: /home/zgy/miniconda3/.../python: No such file or directory
    会话 =
    ① 原生视频上传
    ❌ 没存下

而它是 `docs/开发者文档.md` §10.2 第 8 步点名要跑的那条「无外部样本依赖」的验收。
⟹ 照文档做的人在服务器上**验收跑不起来**,而报错看起来像"上传坏了"。

## 这个文件守什么

只守**会真的把脚本跑废**的那两种写死:

- 解释器路径里带 `miniconda3/envs/jingxin` / `anaconda3/envs/jingxin` 字面量;
- `cd ~/jingxin`(仓库根写死成笔记本布局)。

**不**守文档字符串/注释里的示例路径 —— 那些是给人照着敲的,`jx_env.sh` 自己的
头注释里就列着两种布局。判据只看**会执行的代码行**(去掉 `#` 之后的文本)。
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

#: 免检的脚本。
#: · `jx_env.sh` —— 它就是解析器本身,两种布局都写在这里是它的职责;
#: · `jx_night_run.sh` —— 只调 `claude`,不碰解释器与仓库(实测 grep 无命中);
#: · `tunnel_to_server.sh` —— 只做 ssh 隧道,同样不碰。
EXEMPT = {"jx_env.sh", "jx_night_run.sh", "tunnel_to_server.sh"}

_HARDCODED_PY = re.compile(r"(?:mini|ana)conda3/envs/jingxin/bin/python")
_HARDCODED_REPO = re.compile(r"cd\s+~?/?jingxin\b(?!/)")


def _scripts() -> list[Path]:
    return sorted(p for p in TOOLS.glob("*.sh") if p.name not in EXEMPT)


def _code_lines(text: str) -> list[tuple[int, str]]:
    """去掉整行注释与行尾注释后的可执行文本。

    ⚠️ 只切 `#` 之后是够用的:本仓这些脚本里没有把 `#` 当数据用的行
    (`"${x#prefix}"` 那种参数展开确实存在,但它出现在 `jx_env.sh` 里,已免检)。
    """
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.split("#", 1)[0]
        if stripped.strip():
            out.append((i, stripped))
    return out


def test_every_shell_script_resolves_the_interpreter_through_jx_env():
    """★ 红法:新写一个 `PY=~/miniconda3/...` 的脚本 ⟹ 这里红。

    判据是**两件事都要成立**:脚本里没有写死的解释器,且它确实 source 了
    `jx_env.sh`(否则即使没写死,它也没有可用的 `$JX_PY`)。
    """
    bad: list[str] = []
    for p in _scripts():
        text = p.read_text(encoding="utf-8")
        code = _code_lines(text)
        for lineno, line in code:
            if _HARDCODED_PY.search(line):
                bad.append(f"{p.name}:{lineno} 写死了解释器 —— {line.strip()}")
        if not re.search(r"source\s+.*jx_env\.sh", text):
            bad.append(f"{p.name} 没有 source jx_env.sh(拿不到 $JX_PY)")
    assert not bad, "以下脚本换一台机器就跑不起来:\n  " + "\n  ".join(bad)


def test_every_shell_script_resolves_the_repo_root_through_jx_env():
    """★ 红法:写 `cd ~/jingxin` ⟹ 这里红(服务器上那是 `/home/zgy/jingxin`,不存在)。

    `$REPO` 由 `jx_env.sh` 从脚本自身位置推出来 —— 脚本在哪台机器、哪个目录,
    它就算哪个,不需要任何人再写一次。
    """
    bad: list[str] = []
    for p in _scripts():
        for lineno, line in _code_lines(p.read_text(encoding="utf-8")):
            if _HARDCODED_REPO.search(line):
                bad.append(f"{p.name}:{lineno} —— {line.strip()}")
    assert not bad, ("以下脚本把仓库根写死成笔记本布局"
                     "(服务器上是 ~/JingXin/jingxin):\n  " + "\n  ".join(bad))
