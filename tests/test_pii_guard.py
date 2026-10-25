"""公开仓库的 PII 守卫:真实学生姓名 + 学号**不许进仓库**。

## 为什么需要这道闸

本仓的 GitHub 远端是**公开仓库**。素材目录名的形状是

    <姓名>-<学号>-<专业>__<session_id>
    某某甲-<9位像真学号>-示例学院__20260928_212130_5b10

`tools/jx_inventory.py` 生成素材表时会把目录名整个写进文档,测试里的
fixture 也常直接抄一场真实目录名当例子 —— 一次不留神就是 9 个 commit
(2026-10-03 ~ 2026-10-24,实测)带着真实姓名+学号推上去了。

2026-10-25 的决定:**已经推上去的历史不改写**(改写要强推、所有克隆失效,
且 GitHub 仍留 dangling commit),**从现在起守住**。所以这道闸必须
在**提交前**拦住,而不是事后扫。

## 判据

不是"文件里出现中文" —— 那会淹在正常内容里。判据是**素材目录名的形状**
(见 `jx_pii_guard._NAME_RE`):中文姓名 2-4 字 + 9 位学号 + 专业 + `__` + session_id。

其中学号是 **9 位数字**。假值一律用 9 个 0 打头(`000000000` / `000000001`),
**真实学号不会全 0** ⟹ 判据自带豁免,不需要维护白名单。

## 这个文件守什么

1. 扫描器认得真实形状、放过假值;
2. 扫描器**真的会**在仓库已跟踪文件上跑(不是只测它自己);
3. 扫描器认得 `git diff --cached` 的补丁输出(提交前钩子用的是这条路径);
4. 仓库当前是干净的 —— 这条是**回归地板**:将来谁再抄一场真实目录名进来,这里红。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import jx_pii_guard as guard  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

#: 一个**像真学号**的样例 —— 运行时拼出来,不写死成字面量。
#:
#: 为什么不写字面量:守卫扫的是**已跟踪文件的当前内容**,写死的样例会让
#: `test_the_tracked_tree_is_clean_right_now` 自己红(2026-10-25 实测踩到)。
#: 而样例又必须像真学号(不规则 9 位),否则测不到"扫描器认得出真实形状"。
#: 拼接是唯一能同时满足两边的写法。
_REAL_LOOKING = "某某" + "甲-" + "8" + "142" + "57203" + "-示例学院__20260928_212130_5b10"


def test_the_scanner_catches_a_real_looking_recording_name():
    """★ 红法:把 `_NAME_RE` 的 `\\d{9}` 改成 `\\d{3}`(或删掉这条正则)⟹ 这里红。"""
    text = f"见 /data/{_REAL_LOOKING}/media/camera.webm"
    hits = guard.scan_text(text)
    assert hits, "扫描器没认出真实形状的素材目录名"
    assert hits[0][1] == _REAL_LOOKING


def test_the_scanner_lets_fake_names_through():
    """★ 红法:把豁免规则去掉(学号全 0 也算违规)⟹ 这里红。

    仓库里现有的假值都是 9 个 0 打头:`示例甲-000000001-示例学院__…`、
    `候选丙-000000000-投资学__…`、`张三-987654321-金融8班__…`。
    没有这条,守卫会把它们也判红 —— 一个到处误报的闸会被绕过,等于没有。
    """
    for fake in ("示例甲-000000001-示例学院__20260929_180047_6fee",
                 "候选丙-000000000-投资学__20260929_200333_2e84",
                 "张三-987654321-金融8班__20260929_180047_6fee",
                 "姬某-000000000-投资学__20260929_195403_040b"):
        assert guard.scan_text(f"目录 {fake} 存在") == [], f"假值被判红了:{fake}"


def test_the_scanner_reads_a_git_patch_not_just_plain_text():
    """★ 红法:让 `scan_patch()` 直接扫整段(不去 `+` 前缀)⟹ 这里红。

    提交前钩子拿到的不是文件内容,是 `git diff --cached` 的补丁。
    补丁里 `+++ b/path` 这种行会让"整段扫"误报,而真正的新增行带 `+` 前缀。
    """
    patch = (
        "diff --git a/x.md b/x.md\n"
        "--- a/x.md\n"
        "+++ b/x.md\n"
        "@@ -1 +1 @@\n"
        "-旧行\n"
        f"+{_REAL_LOOKING}\n"
    )
    hits = guard.scan_patch(patch)
    assert hits, "补丁里的新增行没被扫到"
    assert any("某某甲" in h[1] for h in hits)


def test_the_scanner_does_not_flag_removed_lines():
    """★ 红法:把 `scan_patch` 里的 `startswith("+")` 去掉 ⟹ 这里红。

    **删掉**一行真实姓名不该被拦 —— 那正是清理动作。拦住它等于让人没法修。
    """
    patch = (
        "diff --git a/x.md b/x.md\n"
        "--- a/x.md\n"
        "+++ b/x.md\n"
        "@@ -1 +1 @@\n"
        f"-{_REAL_LOOKING}\n"
        "+（已脱敏）\n"
    )
    assert guard.scan_patch(patch) == [], "删除行被误判成违规"


def test_the_tracked_tree_is_clean_right_now():
    """★ 回归地板:谁再抄一场真实目录名进已跟踪文件,这里红。

    这条**故意扫真实仓库**,不是扫样例字符串 —— 上面四条测的是"扫描器对不对",
    这条测的是"仓库现在干不干净"。两者缺一不可。
    """
    hits = guard.scan_tracked_files(REPO)
    assert not hits, (
        "已跟踪文件里有真实姓名+学号:\n  "
        + "\n  ".join(f"{f}: {m}" for f, m in hits))


def test_the_guard_is_wired_as_a_pre_commit_hook():
    """★ 红法:删掉 `tools/install_pii_hook.sh` 的安装逻辑 ⟹ 这里红。

    扫描器写得再对,不接进提交流程就等于没有 —— 本仓反复栽的形态
    (「验证跑错了对象」的同族:闸装了但没接到路上)。
    """
    installer = REPO / "tools" / "install_pii_hook.sh"
    assert installer.exists(), "没有安装脚本 —— 钩子装不上"
    body = installer.read_text(encoding="utf-8")
    assert "pre-commit" in body, "安装脚本没有写 pre-commit 钩子"
    assert "jx_pii_guard" in body, "钩子没调扫描器"
