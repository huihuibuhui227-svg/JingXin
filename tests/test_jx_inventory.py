# tests/test_jx_inventory.py
"""**结构图生成器**(`tools/jx_inventory.py`)的钉子。

## 为什么它需要测试

它是本仓唯一"自己数自己"的产出 —— `docs/资源总览.md` 里的每个数都从这里出。
而这个文件此前**一条测试都没有**(2026-10-24 审计发现),于是三类错都能静默发生:

1. **判据拿不到**:`python tools/jx_inventory.py` 的 `sys.path[0]` 是 `tools/`,
   不是仓库根 ⟹ `from report_frontend.corpus import FORMAL_PATTERN` **必然**
   ModuleNotFoundError,而 `except` 分支静默退回本地正则副本。两份正则当时恰好
   一样,所以没人看出来 —— 「判据只从一处取」这个设计**从来没生效过**。
2. **回收目录混进素材表**:`_回收站` / `_已作废` 被当成"试跑"列进根目录表
   (实测 43 行 vs 41 个有效目录)。
3. **问题数报 0 而实际有 8**:只读 `meta.json` 的 `questions` 块,而 12 个目录
   的台账在 `questions.jsonl` 里(正式场 24 场两处都没有)。

## 不依赖真素材

全部在 `tmp_path` 里铺目录;不读 `~/JingXin/recordings`。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# ⚠️ **假名 + 假学号。** 判据只看形状(`-序号-姓名-学号-学院`),不看真值 ⟹
# 没有理由把真实候选人的姓名学号写进测试。本仓是**公开仓库**,
# 而 `~/JingXin/recordings/` 下是真实受访者(2026-10-24 发现)。
FORMAL_NAME = "20260929_180047-2-示例甲-000000001-示例学院__20260929_180047_6fee"


@pytest.fixture(scope="module")
def inv():
    """按**脚本方式**加载 —— `sys.path[0]` 仍是仓库根(conftest 已塞),不影响断言。"""
    spec = importlib.util.spec_from_file_location("jx_inventory", ROOT / "tools/jx_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 1 判据必须来自 `corpus`,不许静默降级 ─────────────────────────────────

def test_formal_pattern_is_imported_not_fallback_copied(tmp_path):
    """★ **在子进程里**按脚本方式跑,证明 `sys.path` 那段真的有用。

    ## 为什么必须开子进程

    第一版写成"在 pytest 进程里调 `_is_formal` 再看 `_FORMAL_FALLBACK`"——
    **那是恒绿的**:`tests/conftest.py` 早就把仓库根塞进了 `sys.path`,
    所以 pytest 进程里 `from report_frontend.corpus import …` 永远成功,
    删掉 `tools/jx_inventory.py` 顶部那段 `sys.path.insert` 也照样绿。
    而真实调用方式是 `python tools/jx_inventory.py`,那时 `sys.path[0]` 是
    `tools/` —— 两种环境的差别正是这条缺陷的全部。

    ⟹ 这里造一个只含 `jx_inventory.py` 符号链接的 `tools/`,把子进程的
    `cwd` 设成 `tmp_path`(于是 `-c` 的 `sys.path[0]` = `''` = tmp_path,
    找不到 `report_frontend`),再清掉 `PYTHONPATH`。这与真实调用等价。

    红法:删掉 `tools/jx_inventory.py` 顶部那段 `sys.path.insert` ⟹ 本条红。
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "jx_inventory.py").symlink_to(ROOT / "tools" / "jx_inventory.py")
    code = (
        "import sys, json, pathlib\n"
        f"sys.path.insert(0, {str(tools)!r})\n"
        "import jx_inventory as m\n"
        f"m._is_formal(pathlib.Path({FORMAL_NAME!r}))\n"
        "print('FALLBACK=' + json.dumps(m._FORMAL_FALLBACK, ensure_ascii=False))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path),
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"子进程失败:\n{r.stdout}\n{r.stderr}"
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("FALLBACK=")), "")
    assert line == "FALLBACK=[]", (
        f"`report_frontend.corpus` 导入失败,已退回本地正则副本:{line}\n"
        "⟹ 「判据只从一处取」没有生效,两份正则会各自漂。")


def test_fallback_is_recorded_when_import_really_fails(inv, monkeypatch):
    """降级**必须留痕** —— 静默降级正是「两处各持一份判据」的入口。

    做法:让 `report_frontend.corpus` 的导入真的抛错。
    """
    import builtins
    real_import = builtins.__import__

    def _boom(name, *a, **kw):
        if name == "report_frontend.corpus":
            raise ModuleNotFoundError("simulated")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _boom)
    monkeypatch.delitem(sys.modules, "report_frontend.corpus", raising=False)
    inv._FORMAL_FALLBACK.clear()
    try:
        assert inv._is_formal(Path(FORMAL_NAME)) is True   # 本地副本仍判对
        assert inv._FORMAL_FALLBACK, "降级了却没记录 ⟹ 产出一字不提,读者无从知道"
    finally:
        inv._FORMAL_FALLBACK.clear()


# ── 2 回收目录不是素材 ────────────────────────────────────────────────────

def test_recycle_and_void_dirs_are_not_in_the_material_table(inv, tmp_path):
    """★ `_回收站` / `_已作废` / `NONE` 不许出现在素材表里。

    红法:把 `sessions_table` 的过滤条件改回只排 `NONE` ⟹ 本条红
    (实测会多出 2 行「试跑 / 目录名不是 …」)。
    规则与 `corpus.audit_corpus` **同一套**(`startswith("_")`),不是另立一份。
    """
    (tmp_path / FORMAL_NAME).mkdir()
    for junk in ("_回收站", "_已作废", "NONE", "_试跑"):
        (tmp_path / junk).mkdir()
    rows = inv.sessions_table(tmp_path)
    body = "\n".join(rows)
    for junk in ("_回收站", "_已作废", "NONE", "_试跑"):
        assert junk not in body, f"{junk} 不是素材,不该进表:\n{body}"
    assert "候选A" in body, f"正式场应当进表(脱敏后叫 候选A):\n{body}"


# ── 3.5 目录名脱敏 ────────────────────────────────────────────────────────

def test_real_names_and_student_ids_are_redacted_in_the_table(inv, tmp_path):
    """★ 素材表里的**真实姓名与学号不许出现** —— 本仓是公开仓库。

    2026-10-24 发现:重刷后的 `docs/资源总览.md` 把 24 场正式素材的
    `-序号-姓名-学号-学院` 目录名整列了出来,即真实受访大学生的姓名 + 学号。
    `git` 一推上去就撤不回来。

    这条钉的是**生成器**而不是文档本身:文档是生成物,手改下次就回来。

    红法:把 `sessions_table` 里的 `redact(d.name)` 改回 `d.name` ⟹ 本条红。
    """
    real = "20260929_180047-2-张三-261210823-金融8班__20260929_180047_6fee"
    (tmp_path / real).mkdir()
    body = "\n".join(inv.sessions_table(tmp_path))
    assert "张三" not in body, f"真实姓名进了表:\n{body}"
    assert "261210823" not in body, f"真实学号进了表:\n{body}"
    # 代号要能替代它,且学号**位数保留**(位数本身是判据的一部分)
    assert "候选A" in body, body
    assert "000000000" in body, f"学号应换成同长度占位符:\n{body}"


def test_redaction_keeps_the_shape_readable(inv):
    """脱敏后仍要能看出「这是正式场」—— 只换姓名学号,不动序号与学院。"""
    real = "20260929_180047-2-张三-261210823-金融8班__20260929_180047_6fee"
    inv._assign_codes([Path(real)])
    out = inv.redact(real)
    assert "20260929_180047-2-" in out
    assert "金融8班" in out
    assert "000000000" in out
    assert out.count("候选") == 1


# ── 3 问题数口径 ──────────────────────────────────────────────────────────

def test_question_count_prefers_the_ledger_the_downstream_reads(inv, tmp_path):
    """★ 问题数取 `questions.jsonl` —— 即 `session_meta.read_questions()` 读的那份。

    红法:把 `_question_count` 改回只读 `meta.json` ⟹ 本条红(会返回 0)。
    这条错的后果不是"数字不好看":读者会拿「0 题」去解释「为什么没有逐题窗口」,
    而真相是 8 题。
    """
    d = tmp_path / "s"
    d.mkdir()
    (d / "questions.jsonl").write_text(
        "".join(json.dumps({"index": i}) + "\n" for i in range(8)), encoding="utf-8")
    n, src = inv._question_count(d)
    assert (n, src) == (8, "questions.jsonl")


def test_question_count_falls_back_to_meta_only_when_the_ledger_is_absent(inv, tmp_path):
    """没有 `questions.jsonl` 时才退 `meta.json`;两处都没有 ⟹ `(0, "无")`。"""
    d = tmp_path / "s"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"questions": [{"q": i} for i in range(3)]}),
                                 encoding="utf-8")
    assert inv._question_count(d) == (3, "meta.json")
    (d / "meta.json").unlink()
    assert inv._question_count(d) == (0, "无")


def test_question_count_agrees_with_what_m4_would_see(inv, tmp_path):
    """★ 与 `session_meta.read_questions()` 同源 —— 不是"看起来一样",是同一份文件。

    钉的是**口径同源**这件事:表里报的题数必须等于 M4 切窗口时会读到的条数,
    否则两个数各自"对"却说的不是一件事。
    """
    d = tmp_path / "s"
    d.mkdir()
    n_expected = 5
    (d / "questions.jsonl").write_text(
        "".join(json.dumps({"qid": f"q{i}", "index": i,
                            "ask_start": 1.0, "ask_end": 2.0}) + "\n"
                for i in range(n_expected)), encoding="utf-8")
    n, _ = inv._question_count(d)
    assert n == n_expected


# ── 4 录制根的两个开关 ────────────────────────────────────────────────────

def _load_with_env(monkeypatch, **env):
    """在**给定的环境变量**下重新加载一次模块(模块级常量在 import 时就定死了)。"""
    for k in ("JX_RECORDINGS", "JINGXIN_RECORDINGS_DIR"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location(
        "jx_inventory_reload", ROOT / "tools/jx_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_recordings_root_accepts_the_tools_env_name(monkeypatch, tmp_path):
    """`JX_RECORDINGS`(`tools/jx_env.sh` 导出的那个)优先。

    红法:把 `RECORDINGS` 的 `_pick` 改回只认 `JX_RECORDINGS` 却不回写
    `JINGXIN_RECORDINGS_DIR` ⟹ 下面第二条断言红。
    """
    import os
    mod = _load_with_env(monkeypatch, JX_RECORDINGS=str(tmp_path))
    assert mod.RECORDINGS == tmp_path
    # ★ 回写 —— 否则本脚本与 `corpus.audit_corpus()`(读生产那套开关)
    # 可以看的不是同一块盘。2026-10-24 实测:一个 41 个目录、另一个 5 个。
    assert os.environ["JINGXIN_RECORDINGS_DIR"] == str(tmp_path)


def test_recordings_root_accepts_the_production_env_name(monkeypatch, tmp_path):
    """`JINGXIN_RECORDINGS_DIR`(`media_retention.root()` 那个)单独设也要认。

    红法:同上 —— 只认 `JX_RECORDINGS` ⟹ 本条红(会退回 `~/JingXin/recordings`)。
    """
    mod = _load_with_env(monkeypatch, JINGXIN_RECORDINGS_DIR=str(tmp_path))
    assert mod.RECORDINGS == tmp_path


def test_recordings_root_falls_back_when_neither_is_set(monkeypatch):
    """两个都没设时退回候选列表里**第一个存在的**,不许直接抛。"""
    mod = _load_with_env(monkeypatch)
    assert isinstance(mod.RECORDINGS, Path)
