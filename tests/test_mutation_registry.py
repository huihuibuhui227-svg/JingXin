# tests/test_mutation_registry.py
"""**可执行的「红法」** —— 把"改 X ⟹ 测试变红"从散文变成被检查过的事实。

## 为什么需要这个文件

本仓的纪律里写着「反向复现是标准动作」,但那条纪律此前只靠**人写一句话**落实:
测试的 docstring 里写「红法:改 X ⟹ 红」。**那句话没人验过** —— 而 2026-10-03
一天之内我就写错了两次(把某个权重挪走会不会红、解封会不会让死权重变小),
都是靠事后手工跑变异才发现的。

散文里的复现声明有个结构性毛病:**它错了不会响**。本文件把它换成数据 + 一个真的去跑的测试:

1. `MUTATIONS` 登记 `(测试, 文件, 旧片段, 新片段, 说明)`;
2. `test_mutation_registry_is_well_formed`(**每次跑都执行**)—— 钉登记表自身的完整性与新度:
   测试要存在、文件要在、`旧片段` 在文件里**只出现一次**(否则改动位置有歧义);
3. `test_declared_mutations_really_turn_their_test_red`(**手动开**)——
   在**临时副本**里:先确认原样是绿的,再施加变异、确认变红,最后还原。
   **绝不碰工作区**;每次施加前清 `__pycache__`(仓库纪律:同一秒内"写入→跑→还原"
   会跑到变异字节码)。

## 怎么跑

    JX_RUN_MUTATIONS=1 ~/anaconda3/envs/jingxin/bin/python -m pytest tests/test_mutation_registry.py -q

或 `python tools/jx_selfcheck.py --mutations`。

## 边界(如实说)

* 只支持**片段替换**型变异。像"新建一个文件来触发护栏"那种
  (`tests/test_prose_references.py` 的几条)不在此列 —— 它们各自的 docstring 里
   已注明是手工实测过的。
* 子进程**不继承** `JX_RUN_MUTATIONS`(否则副本里的本文件会再跑一遍变异,递归下去)。
* 副本会包含 `models/` 但排除 `data/`、`.git/`:所以只有**纯单元测试**能登记进来。
  基线圈就是用来发现这一点的 —— 若某个 nodeid 在副本里本来就红,报告会说"基线不绿",
  而不是把它当成"变异生效"。
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parents[1]

#: 子进程开关:设了它,这个文件里的"真跑变异"那条才会执行。
_RUN_ENV = "JX_RUN_MUTATIONS"


class Mutation(NamedTuple):
    nodeid: str      # `tests/test_x.py::test_y`
    path: str        # 相对仓库根
    old: str         # 必须在文件里**恰好出现一次**
    new: str
    note: str        # 这条钉的是哪个生产改动


#: ── 登记表 ────────────────────────────────────────────────────────────
#: 挑选原则:**每一条都对应一个我真的写进过 docstring 的「红法」声明**,
#: 而且改动落在**生产文件或登记数据**上(不是改测试自己)。
MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "tests/test_backfill_face.py::test_existing_backup_is_never_clobbered",
        "tools/jx_backfill_face.py",
        "    if not bak.exists():\n        return bak",
        "    if True:\n        return bak",
        "备份选名退回「无条件用同一个后缀」⟹ 又会覆盖掉最初那份原始 CSV"),
    Mutation(
        "tests/test_face_framing.py::test_spread_is_not_just_the_first_n",
        "face_expression/pipeline/framing.py",
        "    if total <= n:\n        return list(range(total))",
        "    return list(range(min(n, total)))",
        "`spread_indices` 退回「取前 n 帧」⟹ 开场没坐定的场次又被误判"),
    Mutation(
        "tests/test_report_layer.py::test_eye_contact_consumer_uses_the_eye_relative_origin",
        "report_frontend/feature_engine.py",
        "                contact = (1.0 - s / _IRIS_MAX_HORIZONTAL_EXCURSION).clip(lower=0.0, upper=1.0)",
        "                contact = (1.0 - (s - 0.5).abs() * 2).clip(lower=0.0, upper=1.0)",
        "eye_contact 的原点退回画面中心 0.5 ⟹ 正视镜头的人得 0 分(本轮修掉的那个 bug)"),
    Mutation(
        "tests/test_report_layer.py::test_tension_score_is_a_hand_weighted_composite_golden_value",
        "face_expression/core/analysis/tension_engine.py",
        "0.25 * min(au4 * 1.3, 1.0) +",
        "0.30 * min(au4 * 1.3, 1.0) +",
        "`tension_score` 的硬编码权重被调 ⟹ 金标准抓"),
    Mutation(
        "tests/test_dimension_live_weight.py::test_registered_dead_weight_matches_reality",
        "report_frontend/evidence_gate.py",
        '    "focus_score": Quarantine(',
        '    "connective_density": Quarantine("探针", "M3"),\n    "focus_score": Quarantine(',
        "新封停一个槽 ⟹ 那个维度的死权重变大,登记表抓"),
    Mutation(
        "tests/test_dimension_live_weight.py::test_every_dimension_declares_full_slot_weight",
        "report_frontend/research_mapper.py",
        '("gaze_direction_x_std", 0.2, False,',
        '("gaze_direction_x_std", 0.3, False,',
        "维度内槽权重和不再等于 1 ⟹ 死权重那个数就不可比了"),
    Mutation(
        "tests/test_report_layer.py::test_no_slot_points_at_a_column_marked_for_deletion",
        "l0_columns.json",
        '"planned": "**M3.2 删列**。⚠️ 删之前必须先处理',
        '"planned": "M3.2 再议。⚠️ 删之前必须先处理',
        "把 `gaze_deviation` 从待删名单里移走 ⟹ 而槽还指着它,守卫抓"),
)


def _defined_test_names(rel: str) -> set:
    p = REPO / rel
    if not p.is_file():
        return set()
    return set(re.findall(r"^\s*def (test_\w+)", p.read_text(encoding="utf-8"), re.M))


def test_mutation_registry_is_well_formed():
    """★ 登记表自身的完整性 —— **每次跑都执行**(不依赖 `JX_RUN_MUTATIONS`)。

    钉四件事:
    1. `old != new`(否则是空变异,跑出来一定"不红");
    2. 目标文件存在;
    3. `old` 在文件里**恰好出现一次** —— 出现 0 次是登记过期(生产代码改了),
       出现多次则**改动位置有歧义**(替哪一处?);
    4. `nodeid` 指向的测试真的存在。

    红法(实测):把某条 `old` 改成一个文件里不存在/出现两次的片段 ⟹ 红;
    把 `nodeid` 改成一个不存在的测试名 ⟹ 红。
    """
    assert MUTATIONS, "登记表空了 —— 那这个文件就什么都没钉"
    bad = []
    for m in MUTATIONS:
        if m.old == m.new:
            bad.append(f"{m.nodeid}: old 与 new 相同(空变异)")
        p = REPO / m.path
        if not p.is_file():
            bad.append(f"{m.nodeid}: 目标文件不在了: {m.path}")
            continue
        n = p.read_text(encoding="utf-8").count(m.old)
        if n != 1:
            bad.append(f"{m.nodeid}: `old` 在 {m.path} 里出现 {n} 次(必须恰好 1 次)")

        rel, _, fn = m.nodeid.partition("::")
        if fn not in _defined_test_names(rel):
            bad.append(f"{m.nodeid}: 这个测试名在 {rel} 里找不到定义")

    assert not bad, (
        "变异登记表本身有问题:\n  " + "\n  ".join(bad)
        + "\n—— `old` 出现 0 次通常是**生产代码改过、登记过期**;"
          "出现多次则要写得更长以消歧。")


def _copy_repo(dst: Path) -> None:
    """把仓库复制到 `dst`(排除大的/不该带的),供子进程在副本里跑。"""
    shutil.copytree(
        REPO, dst, symlinks=True,
        ignore=shutil.ignore_patterns("data", ".git", "__pycache__", "node_modules",
                                      ".pytest_cache", "*.pyc", "*.pyo"))


def _clear_pycache(root: Path) -> None:
    """清掉副本里的字节码缓存 —— 否则可能跑到**变异前**编译出来的 .pyc。

    (本仓纪律:`test_report_layer.py` 的「变异验证必须清 `__pycache__`」。)
    """
    for d in root.rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)


def _child_env(tmp: Path) -> dict:
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp / "numba")
    env["JOBLIB_TEMP_FOLDER"] = str(tmp / "joblib")
    env["MPLCONFIGDIR"] = str(tmp / "mpl")
    # ★ 必须摘掉开关:否则副本里的本文件会再跑一遍变异,无限递归。
    env.pop(_RUN_ENV, None)
    return env


def _run_one(nodeid: str, cwd: Path, tmp: Path) -> tuple:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", nodeid, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=cwd, env=_child_env(tmp), capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


@pytest.mark.skipif(not os.environ.get(_RUN_ENV),
                    reason=f"真跑变异很慢(每条一个 pytest 子进程);设 {_RUN_ENV}=1 才执行")
def test_declared_mutations_really_turn_their_test_red():
    """★ 每条登记都要**实测**:副本里原样是绿的 ⟹ 施加变异后必须变红。

    三阶段,顺序不能省:
    1. **基线**:原样跑一遍,必须绿。副本里本来就红的(例如它需要 `data/`)
       ⟹ 报"基线不绿",**不算**变异生效 —— 否则这条测试会因为环境问题"假通过";
    2. **变异 + 清 `__pycache__`**:再跑,必须红;
    3. 还原目标文件。
    """
    tmp = Path(tempfile.mkdtemp(prefix="jx_mut_"))
    copy = tmp / "repo"
    results, problems = [], []
    try:
        _copy_repo(copy)
        for m in MUTATIONS:
            target = copy / m.path
            original = target.read_text(encoding="utf-8")

            _clear_pycache(copy)
            base_code, base_out = _run_one(m.nodeid, copy, tmp)
            if base_code != 0:
                problems.append(f"{m.nodeid}: **基线不绿**(副本里原样就红)—— "
                                f"多半是它需要 data/,不该登记进本表\n{base_out[-600:]}")
                results.append((m.nodeid, "基线不绿", "-"))
                continue

            target.write_text(original.replace(m.old, m.new, 1), encoding="utf-8")
            _clear_pycache(copy)
            mut_code, mut_out = _run_one(m.nodeid, copy, tmp)
            target.write_text(original, encoding="utf-8")

            if mut_code == 0:
                problems.append(f"{m.nodeid}: **变异之后仍然绿** ⟹ docstring 里的「红法」是假的。"
                                f"变异:{m.note}\n{mut_out[-600:]}")
                results.append((m.nodeid, "仍绿 ✗", m.note))
            else:
                results.append((m.nodeid, "变红 ✓", m.note))

        report = "\n".join(f"  {nodeid:74s} {verdict:8s} {note}"
                           for nodeid, verdict, note in results)
        assert not problems, (
            f"变异登记表与实际不符({len(problems)}/{len(MUTATIONS)} 条):\n"
            + "\n\n".join(problems) + "\n\n逐条结果:\n" + report)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
