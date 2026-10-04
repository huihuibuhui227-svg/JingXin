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
        '("gaze_direction_x_std", 0.2857142857142857, False,',
        '("gaze_direction_x_std", 0.3, False,',
        "维度内槽权重和不再等于 1 ⟹ 死权重那个数就不可比了"),
    Mutation(
        "tests/test_report_layer.py::test_no_slot_points_at_a_column_marked_for_deletion",
        "report_frontend/research_mapper.py",
        '                    ("symmetry_score", 0.25, True, "面部对称性", "support"),',
        '                    ("gaze_deviation", 0.2, False, "视线偏差", "core"),\n'
        '                    ("symmetry_score", 0.25, True, "面部对称性", "support"),',
        "把 `gaze_deviation` 槽**加回**到它指向的待删列上 ⟹ 守卫抓"
        "(⚠️ 原登记是「把列移出待删名单」,那条 2026-10-03 删槽后**已失效** —— "
        "槽都不是槽了,移不移名单守卫都不管;换成本条)"),
    Mutation(
        "tests/test_report_layer.py::test_the_doomed_column_rule_catches_a_renamed_keyword",
        "tests/test_report_layer.py",
        "    return sorted(c for c in doomed if keyword == c or keyword in c or c in keyword)",
        "    return sorted(c for c in doomed if keyword == c)",
        "⚠️ 本条变异改的是**守卫自己**(不是生产文件)—— 本表唯一的例外类:"
        "「待删列」判据退回**等值**比对 ⟹ `blink_rate` 命中不到 `blink_rate_per_min`,"
        "那个洞会第二次静默出现。没有这条正向对照,上一行那条守卫在**名单为空时照样绿**。"),
    Mutation(
        "tests/test_dimension_live_weight.py::test_registered_dead_weight_matches_reality",
        "report_frontend/evidence_gate.py",
        '    "pause_duration": Quarantine(',
        '    "pauseXduration": Quarantine(',
        "解封一个**只被**封停、不指向待删列的槽(`pause_duration`,w=1/7)⟹ "
        "该维死权重 5/7 → 4/7,登记表抓。"
        "⚠️ 2026-10-04 改过 new:原来写的是 `pause_duration探针`(**加后缀**),"
        "而封停查找改成**双向子串**之后 `pause_duration ⊂ pause_duration探针` 仍然命中 ⟹ "
        "那条变异**不再红**(机制自己报的「变异之后仍然绿」)。改成真正不相关的名字才等价于解封"),
    Mutation(
        "tests/test_dimension_live_weight.py::test_registered_dead_weight_matches_reality",
        "report_frontend/research_mapper.py",
        '                    ("n_chars", 0.4, True, "回答详尽度", "support"),',
        '                    ("au7_freq", 0.3, False, "眼部挤压 (费力)", "core"),\n'
        '                    ("n_chars", 0.4, True, "回答详尽度", "support"),',
        "把 2026-10-03 删掉的 `au7_freq` **加回来** ⟹ 死权重 0.0 → 0.3,登记表抓。"
        "⭐ 这是那个「名字推不出源列」的洞的**唯一兜底**:`au7_freq` → `micro_exp_au_name` "
        "在名字上没有任何公共子串,守卫看不见它 —— 能挡住它复活的只有这张死权重登记表"),
    Mutation(
        "tests/test_dimension_live_weight.py::test_registered_dead_weight_matches_reality",
        "report_frontend/evidence_thresholds.json",
        '    "pitch": 5,',
        '    "pitch": 10,',
        "把 `pitch` 族的门槛退回**帧级默认 10** ⟹ 每段回答一个样本的 `pitch_variation` "
        "结构性不可达(n_valid 8 < 10),该维死权重 5/7 → **7/7**。"
        "⭐ 这是死权重表里「门槛单位错配」那条判据**唯一的正向对照** —— "
        "2026-10-03 之前没有它,本表把 communication_fluency 登记成 5/7 而真值是 7/7"),
    Mutation(
        "tests/test_evidence_gate.py::test_answer_level_families_are_not_gated_by_the_frame_level_default",
        "report_frontend/evidence_thresholds.json",
        '    "pitch": 5,',
        '    "pitch": 10,',
        "同一改(门槛单位错配)由**另一条独立判据**也抓住 —— 两条守卫各自都能红,"
        "不是一条在替另一条背书"),
    Mutation(
        "tests/test_voice_replay_contract.py::test_the_row_shape_is_the_contract",
        "voice_interaction/utils/logger.py",
        '            "voiced_prob_mean": prosody_data.get("voiced_prob_mean"),',
        '            "voiced_prob": prosody_data.get("voiced_prob_mean"),',
        "把 `build_row` 里一个列的键**改名**(列名与键名不一致是重放腿最典型的漂法)⟹ "
        "形状守卫抓。⚠️ 真实后果是**静默**的:`DictWriter(fieldnames=...)` 遇到对不上的键"
        "会把那一列写成空,不报错 —— 所以这条必须钉在**形状**上,不能只钉集合"),
    Mutation(
        "tests/test_voice_replay_contract.py::test_contract_is_partitioned_into_written_and_registered_not_written",
        "voice_interaction/utils/replay_contract.py",
        '    "is_valid": "活路径恒写 True',
        '    "is_valid_X": "活路径恒写 True',
        "把 `is_valid` 从「重放写不出」的登记里划掉 ⟹ 契约 32 列不再被不重不漏地划分,"
        "守卫抓(新版 `VoiceLogger` 加一列而没人归类时,红的也是这一条)"),
    Mutation(
        "tests/test_voice_replay_contract.py::test_arrival_wall_uses_the_raw_record_not_the_converted_one",
        "voice_interaction/utils/replay_contract.py",
        '        if rec.get("kind") == "raw" and rec.get("seq") == seq:',
        '        if rec.get("kind") == "converted" and rec.get("seq") == seq:',
        "把到达时刻从 **raw** 记录改成 **converted** 记录 ⟹ `reaction_time` 会系统性偏晚"
        "150~283 ms(实测),而那个偏差在一个 1~65 秒量级的量上**看不出来** —— "
        "⭐ 这是 甲-4 那次实测定位出来的坑,只有这条变异守着它"),
    Mutation(
        "tests/test_backfill_voice.py::test_existing_rows_are_never_overwritten",
        "tools/jx_backfill_voice.py",
        "    if have:                       # 有活行 ⟹ 一行都不动(本工具不覆盖)",
        "    if False:                      # 有活行 ⟹ 一行都不动(本工具不覆盖)",
        "把语音回填的「有活行就跳过」去掉 ⟹ 它会开始**覆盖活跑写下的行**。"
        "补出来的与活跑的**同形同名**,看文件分不出来 —— 所以这条必须红"),
    Mutation(
        "tests/test_backfill_voice.py::test_backup_ladder_never_clobbers",
        "tools/jx_backfill_voice.py",
        "    if not bak.exists():\n        return bak",
        "    if True:\n        return bak",
        "备份选名退回「无条件用同一个后缀」⟹ 跑第二遍就把最初那份原始日志顶掉"
        "(与 face 回填那条同形;当年那份 `.predrop.bak` 正是这么丢的)"),
    Mutation(
        "tests/test_dimension_live_weight.py::test_registered_dead_weight_matches_reality",
        "tests/test_dimension_live_weight.py",
        "    hits = [q for q in QUARANTINE if q == kw or q in kw or kw in q]",
        "    hits = [q for q in QUARANTINE if q == kw]",
        "⚠️ 同「守护卫自己」那一类例外:封停查找退回**等值** ⟹ `energy`(封停键 "
        "`energy_mean` 更长)抓不到 ⟹ `confidence_level` 死权重 0.2 → 0.0。"
        "⭐ 这是「封停键比槽关键词长」那个洞的**唯一**对照 —— "
        "它与 `test_report_layer.py` 那条「待删列」守卫是同型洞的镜像(那边是关键词更短)"),
    Mutation(
        "tests/test_l1_sample_size.py::test_the_sparse_gaze_column_carries_its_own_sample_size",
        "report_frontend/feature_engine.py",
        '                        features[f"face_{col}_sample_size"] = int(len(series))',
        '                        pass  # 探针:去掉样本量',
        "把「3. Gaze 方向统计」的 `_sample_size` 拿掉 ⟹ 稀疏 gaze 列没有自己的样本量,"
        "G3 与**报告显示的 `n_valid`** 退到模态行数。实测后果:场 `6d57` 印出 `n_valid=200`"
        "而真值 12 ⟹ `logical_thinking` 拿到 **99.05「卓越」**"),
    Mutation(
        "tests/test_l1_sample_size.py::test_long_series_also_carries_a_sample_size",
        "report_frontend/feature_engine.py",
        '            stats[f"{base_name}_sample_size"] = len(series)',
        '            pass  # 探针:长序列不发样本量',
        "把 `_extract_numeric_stats` 的样本量挪回「只有短序列」那一支(即修前的样子)⟹ "
        "长序列指标查不到自己的样本量 ⟹ 全语料 33 场的 `n_valid` 被模态行数高估(可达 4 倍)"),
    Mutation(
        "tests/test_l1_sample_size.py::test_indicator_sample_size_finds_it_through_a_statistic_suffix",
        "report_frontend/research_mapper.py",
        '    for suffix in ("_mean", "_std", "_min", "_max", "_sum", "_trend"):',
        "    for suffix in ():",
        "查法退回「只认 `<基名>_mean` 的兄弟键」⟹ 槽消费的是 `_std` 时(如 "
        "`gaze_direction_x_std`「视线稳定性」)查不到同列已登记的样本量"),
    Mutation(
        "tests/test_l1_sample_size.py::test_a_dispersion_key_uses_its_own_value_for_g2",
        "report_frontend/research_mapper.py",
        "                chk = gate(found_key, found_val, n_valid=n_valid, std=g2_std)",
        "                chk = gate(found_key, found_val, n_valid=n_valid, std=std)",
        "G2 的输入退回「只有伴随 `_std` 才判」⟹ 键自身是离散度时(`gaze_direction_x_std`)"
        "G2 **静默跳过**。实测:全语料 278 个过门槽里 **29 个**都落在这一条上,"
        "而报告里看不出这 29 次 G2 没执行"),
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
