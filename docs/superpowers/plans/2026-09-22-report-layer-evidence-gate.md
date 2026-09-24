# 报告层真实化与证据门(①)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让报告层停止输出无证据的分数与情绪/心理结论 —— 加入证据门、封停已审计为坏的列、把置信度改成三值(可到"无")、删掉杜撰常模与硬编码叙事。

**Architecture:** 把它做成一个**独立可测的纯函数模块** `evidence_gate.py`(证据门规则 + 封停名单 + 临时阈值,全部数据驱动),再由 `research_mapper.py` 接入。叙事层与措辞的改动(提交 B)不引入新模块,只重写既有字符串。**不碰采集层。**

**Tech Stack:** Python 3.11 · pandas · numpy · pytest(需安装,见前置)

**Spec:** `docs/superpowers/specs/2026-09-22-jingxin-report-layer-evidence-gate-design.md`

## Global Constraints

- **解释器固定为 `~/miniconda3/envs/jingxin/bin/python`。** 不可用 `~/huihui/bin/python` —— `report_frontend/__init__.py` 会连带导入 `visualizer`,而后者依赖 plotly,只有 jingxin 环境有。
- **范围仅 `report_frontend/` 与 `templates/`。** 一行都不改 `face_expression/`、`gesture_analysis/`、`voice_interaction/`、`app.py`。
- **不安装任何包。** pytest 由使用者自己执行下方命令安装。
- **提交只加精确路径(`git add <file>`)。** 仓库当前有 **369 个未跟踪文件**,`git add -A` / `git add .` 会污染提交。
- **先建分支再动代码。** 当前在 `main`。分支名:`fix/report-layer-evidence-gate`。
- **临时阈值必须写在 `report_frontend/evidence_thresholds.json` 里**,带 `_provisional: true` 与 `_version`;**禁止**写成代码里的裸常量(spec §5.1)。
- **禁止词表**(spec §5.6):`焦虑` `紧张` `压力` `抗压` `情绪稳定` `说谎` `诚信` `录用` `人格` `心理画像` `常模`。这些词不得出现在 `report_frontend/` 与 `templates/` 的任何输出字符串中(代码注释与测试断言除外)。

---

## 前置:安装 pytest(由使用者执行)

```bash
~/miniconda3/envs/jingxin/bin/python -m pip install pytest
```

验证:

```bash
~/miniconda3/envs/jingxin/bin/python -m pytest --version
```

预期:输出 `pytest 8.x` 或更高。

**若使用者不愿引入 pytest**,备选是把所有测试改写为 stdlib `unittest`,用 `python -m unittest discover tests` 运行 —— 本计划不采用该路径,但需使用者明示。

---

## 文件结构

| 文件 | 职责 | 动作 |
|---|---|---|
| `report_frontend/evidence_gate.py` | 证据门四关 + 封停名单 + 置信度三值。纯函数,不依赖 pandas | **新建** |
| `report_frontend/evidence_thresholds.json` | 临时阈值登记(带 `_provisional` / `_version`) | **新建** |
| `report_frontend/research_mapper.py` | 接入证据门;删假简历兜底、BASELINE_FILL、三条硬编码代理、杜撰 baselines;修权重和 | 改 |
| `report_frontend/visualizer.py` | 删 `[60]*5` 与 '常模基准' 图例;删死代码;眼动图按 §5.5 处置 | 改 |
| `report_frontend/report_generator.py` | 删硬编码叙事层,改为证据状态输出 | 改 |
| `report_frontend/feature_engine.py` | 删 `fluency_score`/`fluency_proxy` 死分支;`question_index` 不再被跳过 | 改 |
| `tests/conftest.py` | 把项目根加入 `sys.path` | **新建** |
| `tests/test_evidence_gate.py` | 证据门四关 + 封停名单单测 | **新建** |
| `tests/test_report_layer.py` | 零输入回归 / 兜底不可达 / 置信度可达性 / 禁止词扫描 | **新建** |

---

# 提交 A:机制(证据门 + 封停 + 置信度 + 权重)

## Task 1: 测试脚手架

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: 无
- Produces: 可用的 pytest 环境;`tests/` 下所有测试可 `from report_frontend.X import Y`

- [ ] **Step 1: 建分支**

```bash
cd /home/huihuibuhui/jingxin
git checkout -b fix/report-layer-evidence-gate
```

- [ ] **Step 2: 写 conftest.py(把项目根加入 sys.path)**

```python
# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 3: 写冒烟测试**

```python
# tests/test_smoke.py
def test_package_imports():
    """report_frontend 包可导入(会连带导入 visualizer → 需要 plotly)。"""
    import report_frontend

    assert report_frontend.ResearchCapabilityMapper is not None
    assert report_frontend.ReportGenerator is not None
```

- [ ] **Step 4: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_smoke.py -v`
Expected: PASS。若报 `ModuleNotFoundError: plotly`,说明用错了解释器。

- [ ] **Step 5: 提交**

```bash
git add tests/conftest.py tests/test_smoke.py
git commit -m "test: 为报告层引入 pytest 脚手架"
```

---

## Task 2: 证据门模块

**Files:**
- Create: `report_frontend/evidence_thresholds.json`
- Create: `report_frontend/evidence_gate.py`
- Create: `tests/test_evidence_gate.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `check_g1_has_value(v) -> Check`
  - `check_g2_not_constant(values) -> Check` —— 由序列判定
  - `check_g2_from_std(std) -> Check` —— 由伴随的 `_std` 判定(报告层只有标量,用这个)
  - `check_g3_enough_samples(n_valid, min_required) -> Check`
  - `check_g4_not_proxy(key) -> Check`
  - `gate(key, value, n_valid, values=None, std=None) -> Check` —— 依次跑 G1→G4,返回第一个失败;全过返回 `Check(ok=True)`
  - `confidence_from(n_ok: int, n_slots: int) -> Confidence`
  - `is_quarantined(key: str) -> Optional[Quarantine]`
  - `user_message(check: Check) -> str` —— **报告层唯一允许使用的缺口文案来源**;禁止直接渲染 `Check.reason`
  - `Check` 为 frozen dataclass,字段顺序为 `ok: bool`, `gate_name: str`, `reason: str`(下游一律用属性访问,勿用位置参数构造)
  - `Quarantine` 为 frozen dataclass,字段 `reason: str`, `unblock: str`, `permanent: bool`
  - `Confidence = Literal["高","中","低","无"]`
  - `load_thresholds() -> dict` —— 读 json,启动时校验 `_provisional` 与 `_version` 存在

- [ ] **Step 1: 写阈值文件**

```json
{
  "_version": "0.1.0-provisional",
  "_provisional": true,
  "_note": "临时阈值。依据=物理下限,非标定值。M5 标定到位后必须替换,替换即新模型版本(spec §5.1)。",
  "_units": "n_valid 的单位随指标而定:frames 或 segments",
  "_default_n_valid": 10,
  "thresholds": {
    "blink": 20,
    "gaze": 30,
    "au": 30,
    "head_pose": 30,
    "pause": 2,
    "pitch": 10,
    "energy": 10,
    "speech": 2
  }
}
```

- [ ] **Step 2: 写失败的测试**

```python
# tests/test_evidence_gate.py
import pytest

from report_frontend.evidence_gate import (
    Check,
    confidence_from,
    gate,
    is_quarantined,
    load_thresholds,
)


class TestG1HasValue:
    def test_none_fails(self):
        assert gate("pitch_median", None, 100).ok is False

    def test_nan_fails(self):
        assert gate("pitch_median", float("nan"), 100).ok is False

    def test_number_passes(self):
        assert gate("pitch_median", 120.0, 100).ok is True


class TestG2NotConstant:
    def test_single_value_is_constant(self):
        c = gate("pitch_median", 1.0, 100, values=[1.0] * 50)
        assert c.ok is False
        assert c.gate_name == "G2"

    def test_all_zero_is_constant(self):
        assert gate("pitch_median", 0.0, 100, values=[0.0] * 50).ok is False

    def test_varying_passes(self):
        assert gate("pitch_median", 1.0, 100, values=[1.0, 2.0, 3.0]).ok is True


class TestG2FromStd:
    """报告层只有标量,用伴随的 _std 判定常量性。"""

    def test_zero_std_is_constant(self):
        c = gate("pitch_median", 100.0, 500, std=0.0)
        assert c.ok is False
        assert c.gate_name == "G2"

    def test_nonzero_std_passes(self):
        assert gate("pitch_median", 100.0, 500, std=0.3).ok is True

    def test_missing_std_does_not_block(self):
        assert gate("pitch_median", 100.0, 500, std=None).ok is True


class TestG3EnoughSamples:
    # ⚠️ 不能用 blink_* 系列:阈值 20 只由 "blink" 产出,而所有含 blink 的键都在封停
    # 名单里,会被 G4 拦下 —— 该边界用任何未被封停的键都测不到。改用 pitch(阈值 10,
    # pitch_median 未被封停)。

    def test_below_threshold_fails(self):
        c = gate("pitch_median", 120.0, 9, values=[1.0, 2.0])
        assert c.ok is False
        assert c.gate_name == "G3"

    def test_at_threshold_passes(self):
        assert gate("pitch_median", 120.0, 10, values=[1.0, 2.0]).ok is True


class TestG4NotProxy:
    def test_quarantined_column_fails(self):
        c = gate("face_focus_score_mean", 0.3, 500, values=[0.3, 0.4])
        assert c.ok is False
        assert c.gate_name == "G4"

    def test_permanent_quarantine_has_no_unblock(self):
        q = is_quarantined("overall_score")
        assert q is not None and q.permanent is True

    def test_clean_column_passes(self):
        # 不能用 interview_pause_duration_mean —— "pause_duration" 是它的子串,会被 G4 拦。
        assert gate("interview_duration_mean", 0.8, 500,
                    values=[0.7, 0.9]).ok is True


class TestConfidence:
    def test_zero_is_wu(self):
        assert confidence_from(0, 4) == "无"

    def test_minority_is_di(self):
        assert confidence_from(1, 4) == "低"

    def test_majority_is_zhong(self):
        assert confidence_from(3, 4) == "中"

    def test_high_is_unreachable_this_round(self):
        """① 阶段无外部效标验证,'高' 不可达(spec §5.1)。"""
        for n_ok in range(0, 10):
            assert confidence_from(n_ok, 10) != "高"


class TestUserMessage:
    def test_user_message_hides_maintainer_text(self):
        """报告层只能用 user_message;封停理由等内部文案不得外泄(spec §5.6)。"""
        from report_frontend.evidence_gate import user_message

        c = gate("face_focus_score_mean", 0.3, 500, values=[0.3, 0.4])
        assert c.ok is False
        msg = user_message(c)
        assert msg, "缺口文案不得为空"
        for leak in ("封停", "jitter", "M3", "精确重构"):
            assert leak not in msg, f"内部文案外泄:{leak}"

    def test_user_message_per_gate(self):
        from report_frontend.evidence_gate import user_message

        assert user_message(gate("pitch_median", None, 100)) == "未采集到对应数据"
        assert user_message(gate("pitch_median", 1.0, 100, std=0.0)) == "本次会话内无变化"
        assert user_message(gate("pitch_median", 1.0, 3, values=[1.0, 2.0])) == "有效样本不足"
        assert user_message(
            gate("face_focus_score_mean", 0.3, 500, values=[0.3, 0.4])
        ) == "该指标本轮停用"


class TestThresholds:
    def test_thresholds_are_marked_provisional(self):
        t = load_thresholds()
        assert t["_provisional"] is True
        assert t["_version"]

    def test_default_threshold_lives_in_json(self):
        """未登记指标的默认阈值也必须是登记过的临时值,不得是代码里的裸常量。"""
        from report_frontend.evidence_gate import _threshold_for

        data = load_thresholds()
        assert "_default_n_valid" in data
        assert _threshold_for("some_unregistered_metric") == data["_default_n_valid"]

    def test_longest_match_wins(self):
        """阈值表里 "au" 早于 "pause" 且是其子串(p-au-se)。

        若按"注册表第一个匹配"实现,pause_* 会静默拿到 au 的阈值 30 而不是 2。
        必须与 is_quarantined 一样按最长匹配。
        """
        from report_frontend.evidence_gate import _threshold_for

        assert _threshold_for("pause_duration_mean") == 2
        assert _threshold_for("au12_smile_mean") == 30


class TestGateOrder:
    def test_first_failure_wins(self):
        """G1 先于 G3。"""
        c = gate("blink_rate", None, 0)
        assert c.gate_name == "G1"
```

- [ ] **Step 3: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_evidence_gate.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'report_frontend.evidence_gate'`

- [ ] **Step 4: 实现 evidence_gate.py**

```python
# report_frontend/evidence_gate.py
"""证据门:任何指标要进打分,必须四关全过。

设计依据:docs/superpowers/specs/2026-09-22-jingxin-report-layer-evidence-gate-design.md §5.1/§5.2

四关:
  G1 有值      非 NaN
  G2 非常量    会话内方差 > 0
  G3 样本够    n_valid >= 登记阈值
  G4 非代理    不在封停名单

本模块不依赖 pandas —— 纯函数,可独立测试。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

Confidence = Literal["高", "中", "低", "无"]


@dataclass(frozen=True)
class Check:
    ok: bool
    gate_name: str = ""
    reason: str = ""


@dataclass(frozen=True)
class Quarantine:
    reason: str
    unblock: str
    permanent: bool = False


# 封停名单(spec §5.2)。key 为**子串**,按最长匹配生效。
# 解封条件写在 unblock 里,便于 M3 修好后逐条解封。
QUARANTINE: dict[str, Quarantine] = {
    # ---- 等 M3 ----
    "focus_score": Quarantine("3 个硬编码取值,99.7% 恒 0.3", "M3"),
    "symmetry_score": Quarantine("未除人脸尺度,近常量,取景代理", "M3"),
    "tension_score": Quarantine("伪合成,5 项里 2 项恒定", "M3"),
    "tension_level": Quarantine("字符串分箱", "M3"),
    "tension_sources_": Quarantine("au4/au23 副本或恒定量", "M3"),
    "gaze_direction_y": Quarantine("恒负,解剖常量", "M3"),
    "gaze_deviation": Quarantine("与 gaze_direction_y 组内 r=-0.994", "M3"),
    "hand_score": Quarantine("可由同 block jitter 精确重构", "M3"),
    "shoulder_score": Quarantine("可由同 block jitter 精确重构", "M3"),
    "arm_score": Quarantine("可由同 block jitter 精确重构", "M3"),
    "head_score": Quarantine("可由同 block jitter 精确重构", "M3"),
    "torso_score": Quarantine("可由同 block jitter 精确重构", "M3"),
    "fist_status": Quarantine("fist_threshold=0.08 -> 有效帧 100% 判握拳", "M3"),
    # ↓ 这 7 条来自 spec §5.3 的**槽位级**处置,§5.2 的列级表没覆盖它们。
    #   仅靠 §5.2 会让这 7 个指标绕过 G4 直接进打分(实测其中 3 个真的出了分)。
    #   已由 Task 3 的修复轮补入(Task 2 完成时遗漏)。
    "gaze_stability": Quarantine("由被封停的 gaze_deviation 派生", "M3 修 gaze_direction_x"),
    "au4_freq": Quarantine("从 micro_exp 字符串解析出的垃圾匹配", "M3 修 au4 landmark"),
    "au7_freq": Quarantine("au7 + avg_ear 恒等 1.0,与 au4 互补", "M3"),
    "jitter": Quarantine("经 x5 归一化恒饱和", "M3 归一化(除肩宽除时间)"),
    "speech_ratio": Quarantine("自指阈值,87.9% 恰为 1.0", "M3 真 VAD"),
    "eye_contact": Quarantine("iris 图像坐标到画面中心距离,是取景代理", "M3 改眼内相对坐标"),
    "fluency_score": Quarantine("生产分支为死代码,实测走 BASELINE_FILL", "M3 真 VAD"),
    # 永久封停:unblock 是给人看的显示文本,逻辑判断一律用 permanent 字段。
    # 不要用 unblock == "永不" 反推 —— 字符串是显示层,不是逻辑层。
    "upper_body_head_tilt": Quarantine("参考系错位 180 度,分支命中率 0%", "永不", permanent=True),
    "shoulder_is_calibrated": Quarantine("纯时长变量,控时长后相关 0.014", "永不", permanent=True),
    "overall_score": Quarantine("属性不存在,getattr 走默认值", "永不", permanent=True),
    "emotion_state": Quarantine("属性不存在,恒 neutral", "永不", permanent=True),
    "emotion_": Quarantine("7 个分量结构性恒 0,单形归一化互竞", "永不", permanent=True),
    "dominant_emotion": Quarantine("argmax,非置信度", "永不", permanent=True),
    "micro_exp_onset_frame": Quarantine("环形缓冲下标,恒在 5~13", "永不", permanent=True),
    "fluency_proxy": Quarantine("与 fluency_score 同自由度,数了两遍", "永不", permanent=True),
    # ---- 等对应里程碑 ----
    "blink_rate": Quarantine("恒 0(值写进深拷贝)", "M2 修序列化"),
    "eye_closed_sec": Quarantine("恒 0;+=1/30 在 10fps 下低估 3 倍", "M2 修序列化"),
    "pitch_variation": Quarantine("实际取的是 pitch_mean,名字与计算不符", "改名后"),
    "pause_duration": Quarantine("尾部静默被丢弃", "M3 修停顿检测"),
    "energy_mean": Quarantine("设备增益/距离代理,跨会话不可比", "M3 会话内归一"),
    "energy_level": Quarantine("同上", "M3 会话内归一"),
    "is_valid": Quarantine("恒 1 且下游从未使用", "M3 改真掩码"),
}

_THRESHOLD_FILE = Path(__file__).resolve().parent / "evidence_thresholds.json"


def load_thresholds() -> dict:
    with _THRESHOLD_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("_provisional") is not True or not data.get("_version"):
        raise ValueError(
            "evidence_thresholds.json 必须带 _provisional: true 与 _version —— "
            "临时阈值必须可被识别为临时(spec §5.1)"
        )
    if "_default_n_valid" not in data:
        raise ValueError(
            "evidence_thresholds.json 必须带 _default_n_valid —— "
            "未登记指标的默认阈值也属于临时值,不得写成代码里的裸常量(spec §5.1)"
        )
    return data


def _threshold_for(key: str) -> int:
    """按**最长**子串匹配阈值,与 is_quarantined 用同一套规则。

    不能用"注册表里第一个匹配" —— 阈值表里 "au" 早于 "pause",而 "au" 是
    "pause" 的子串(p-au-se),会让所有 pause_* 键静默拿到 au 的阈值 30
    而不是自己登记的 2。同模块内两套匹配规则是缺陷。
    """
    data = load_thresholds()
    thresholds = data["thresholds"]
    k = key.lower()
    matched = [name for name in thresholds if name in k]
    if not matched:
        # 默认值也来自 JSON —— 不得写成裸常量(spec §5.1)
        return int(data["_default_n_valid"])
    return thresholds[max(matched, key=len)]


def is_quarantined(key: str) -> Optional[Quarantine]:
    """按最长子串匹配封停名单。"""
    k = key.lower()
    matched = [s for s in QUARANTINE if s in k]
    if not matched:
        return None
    return QUARANTINE[max(matched, key=len)]


def check_g1_has_value(value) -> Check:
    if value is None:
        return Check(False, "G1", "无值")
    try:
        if math.isnan(float(value)):
            return Check(False, "G1", "NaN")
    except (TypeError, ValueError):
        return Check(False, "G1", f"非数值:{type(value).__name__}")
    return Check(True)


def check_g2_not_constant(values: Optional[Sequence[float]]) -> Check:
    if not values:
        return Check(True)  # 无序列信息时不在本关拦截
    finite = [v for v in values if v is not None and not _isnan(v)]
    if len(finite) < 2:
        return Check(False, "G2", "有效取值不足 2 个,无法排除常量")
    if max(finite) == min(finite):
        return Check(False, "G2", f"会话内恒为 {finite[0]}")
    return Check(True)


def check_g2_from_std(std) -> Check:
    """用伴随的 _std 判定常量性。

    报告层拿到的只有标量(feature_engine 已把序列聚合掉了),
    但 feature_engine 对每个数值列同时产出 _std,足以判定"会话内是否恒定"。
    std 缺失时不在本关拦截。
    """
    if std is None:
        return Check(True)
    try:
        if float(std) == 0.0:
            return Check(False, "G2", "会话内标准差为 0(常量)")
    except (TypeError, ValueError):
        return Check(True)
    return Check(True)


def check_g3_enough_samples(n_valid: int, min_required: int) -> Check:
    if n_valid < min_required:
        return Check(False, "G3", f"有效样本 {n_valid} < 阈值 {min_required}")
    return Check(True)


def check_g4_not_proxy(key: str) -> Check:
    q = is_quarantined(key)
    if q:
        return Check(False, "G4", f"已封停:{q.reason}(解封:{q.unblock})")
    return Check(True)


def gate(key: str, value, n_valid: int,
         values: Optional[Sequence[float]] = None,
         std=None) -> Check:
    """依次跑 G1 -> G4,返回第一个失败;全过返回 Check(ok=True)。

    values 与 std 二选一:有序列用 values,只有标量用 std。
    """
    g2 = check_g2_not_constant(values) if values is not None else check_g2_from_std(std)
    for check in (
        check_g1_has_value(value),
        g2,
        check_g3_enough_samples(n_valid, _threshold_for(key)),
        check_g4_not_proxy(key),
    ):
        if not check.ok:
            return check
    return Check(True)


_USER_MESSAGES: dict[str, str] = {
    "G1": "未采集到对应数据",
    "G2": "本次会话内无变化",
    "G3": "有效样本不足",
    "G4": "该指标本轮停用",
}


def user_message(check: Check) -> str:
    """面向用户的证据缺口说明。

    ⚠️ 报告层只准用这个,**不得直接渲染 `check.reason`** —— reason 面向维护者,
    含封停理由等内部信息(如"可由同 block jitter 精确重构")。spec §5.6 对
    quarantine 文本的豁免只覆盖"留在代码里不进输出";一旦渲染进报告就不再是
    内部文本,而报告必须中性、可辩护。
    """
    return _USER_MESSAGES.get(check.gate_name, "证据不足")


def confidence_from(n_ok: int, n_slots: int) -> Confidence:
    """置信度三值化。'高' 在本轮不可达(spec §5.1)。"""
    if n_ok <= 0:
        return "无"
    if n_slots > 0 and n_ok * 2 < n_slots:
        return "低"
    return "中"


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True
```

- [ ] **Step 5: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_evidence_gate.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add report_frontend/evidence_gate.py report_frontend/evidence_thresholds.json tests/test_evidence_gate.py
git commit -m "feat: 新增证据门模块(四关 + 封停名单 + 置信度三值)"
```

---

## Task 3: research_mapper 接入证据门

**Files:**
- Modify: `report_frontend/research_mapper.py`
- Create: `tests/test_report_layer.py`

**Interfaces:**
- Consumes: Task 2 的 `gate`, `confidence_from`, `Check`
- Produces:
  - `map_features_to_scores(features)` 返回的每个 dimension 增加 `evidence_gaps: list[str]`;无有效证据时 `score is None`、`confidence == "无"`
  - 顶层返回增加 `total_score`(可能为 `None`)、`evidence_gaps: list[str]`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_report_layer.py
from report_frontend.research_mapper import ResearchCapabilityMapper


def test_zero_input_produces_no_scores():
    """spec §6.1:零输入必须不再产出分数。基线是 47.08。"""
    r = ResearchCapabilityMapper().map_features_to_scores({})

    assert r["total_score"] is None
    for key, dim in r["dimensions"].items():
        assert dim["score"] is None, f"{key} 零输入却仍出分"
        assert dim["confidence"] == "无", f"{key} 置信度应为无"
    assert r["evidence_gaps"], "必须给出 evidence_gaps"


def test_no_fake_resume_fallback():
    """spec §5.3:假简历兜底必须删除。"""
    r = ResearchCapabilityMapper().map_features_to_scores({})
    for dim in r["dimensions"].values():
        for ev in dim["evidence_chain"]:
            assert ev["feature"] != "logic_keyword_density", "假简历兜底仍在"
            assert "BASELINE" not in ev["feature"], "BASELINE_FILL 仍可达"
            assert "代理" not in ev["feature"], "硬编码代理仍在"


def test_quarantined_columns_are_rejected():
    """spec §5.2:封停列即使有值也不得进入证据链。"""
    feats = {"face": {"face_focus_score_mean": 0.3,
                      "face_symmetry_score_mean": 0.98}}
    r = ResearchCapabilityMapper().map_features_to_scores(feats)
    for dim in r["dimensions"].values():
        for ev in dim["evidence_chain"]:
            assert "focus_score" not in ev["feature"]
            assert "symmetry_score" not in ev["feature"]


def test_confidence_can_be_none_and_low():
    """spec §6.5:置信度必须能取到'无'与'低'。"""
    seen = set()
    seen.add(ResearchCapabilityMapper().map_features_to_scores({})
             ["dimensions"]["logical_thinking"]["confidence"])
    feats = {"voice_research": {"interview_pause_duration_mean": 0.8}}
    r = ResearchCapabilityMapper().map_features_to_scores(feats)
    for dim in r["dimensions"].values():
        seen.add(dim["confidence"])
    assert "无" in seen


def test_dimension_weights_sum_to_one():
    """spec §5.3:comm 维度权重和曾为 1.4。"""
    m = ResearchCapabilityMapper()
    for dim_key, rule in m.mapping_rules.items():
        total = sum(w for _, w, _, _, _ in rule["indicators"])
        assert abs(total - 1.0) < 1e-9, f"{dim_key} 权重和为 {total}"
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -v`
Expected: FAIL(零输入仍返回 47.08、假简历兜底仍在、comm 权重和 1.4)

- [ ] **Step 3: 删除假简历兜底数据与 baselines(两处必须同时删)**

删除 `research_mapper.py:19-38` 的 `self.demo_text_data`,以及 `:40-47` 的 `self.baselines`。`__init__` 仅保留 `self.mapping_rules` 与 `self.dimension_weights`。

⚠️ **删 `self.baselines` 的同时必须删掉它在 `map_features_to_scores` 里的唯一使用点**,否则运行即 `AttributeError`。删除这段(原 `:234-243`):

```python
bl = self.baselines.get(keyword, {})
percentile = None
if bl:
    std_dev = bl["std"] if bl["std"] > 0 else 0.001
    z = (matched_val - bl["mean"]) / std_dev
    try:
        percentile = int(0.5 * (1 + math.erf(z / math.sqrt(2))) * 100)
    except:
        percentile = 50
    inference_data[keyword] = {"val": matched_val, "percentile": percentile}
```

并把 `evidence_item` 里的 `"percentile": percentile` 一并删除。此后 `inference_data` 只保留 `{"val": ...}`(供调试),不再含百分位。

⚠️ **连带必须处理的两处,否则真实会话会崩**(Task 3 实现时发现,均按最小方式处置):

1. `_generate_deep_inference` 有 3 处读 `[...]["percentile"]`,其中 `data['gaze_stability']['percentile']` 在它的 `try` 块**之外**,另两处在布尔表达式里 —— 都逃过 `except KeyError`。删掉百分位后必崩。
   **处置:三个槽位直接留空**(`gaze_info = jitter_info = au_info = ""`),**不要用 `.get("percentile")`** —— `.get` 会落到"视线稳定性正常 / 肢体略有紧张 / 面临一定认知挑战"这类**无证据断言**分支(此前因百分位总能算出而不可达)。在一个以"删除伪造"为目标的改动里重新打开无证据断言是反向的。代价是模板句会出现"结合其，显示出…"的双逗号;该函数已被 Task 7 整段删除,属过渡态。

2. `score=None` 带来两处未防护比较:`self._get_level(final_total)` 的 `None >= 90`,以及 `_generate_summary_narrative` 里的 `if v['score'] > 0`。
   **处置:**前者改为 `self._get_level(final_total) if final_total is not None else "证据不足"`(选"证据不足"是为了让 `report_generator.py:149` 的 `result['total_level'].split()[0]` 继续可用);后者改为 `is not None`,与 Step 8 的 `valid` 定义保持一致。

- [ ] **Step 4: 修正 `communication_fluency` 的权重和**

```python
"communication_fluency": {
    ...
    "indicators": [
        ("fluency_score", 0.4, True, "语音流畅度", "core"),
        ("fluency_proxy", 0.4, True, "流畅度代理指标", "core"),
        ("speech_ratio", 0.3, True, "有效说话占比", "core"),
        ("pitch_variation", 0.2, True, "语调变化", "support"),
        ("pause_duration", 0.1, False, "平均停顿时长", "support"),
    ]
},
```

改为(总权重 1.0;`fluency_proxy` 是永久封停项,直接删除):

```python
"communication_fluency": {
    ...
    "indicators": [
        ("fluency_score", 0.4, True, "语音流畅度", "core"),
        ("speech_ratio", 0.3, True, "有效说话占比", "core"),
        ("pitch_variation", 0.2, True, "语调变化", "support"),
        ("pause_duration", 0.1, False, "平均停顿时长", "support"),
    ]
},
```

- [ ] **Step 5: 前置 —— feature_engine 产出每个模态的有效行数**

G3 需要一个真实的 `n_valid`。报告层拿到的全是聚合标量,**没有 `n_valid_frames`**(那是 M4 的 L1 才有的)。当前唯一真实可用的量是**每个模态源 DataFrame 的行数** —— 它决定了一个均值是否有意义。

在 `feature_engine.py` 的 `extract_all_features` 里,每个模态结束处加一行,产出 `<模态>__n_rows`:

```python
# extract_all_features(),每个模态提取完之后
# 键名写成 _n_rows(单下划线):mapper 的扁平化是 f"{modality}_{k}",
# 加成 face__n_rows(两个下划线),正好等于 Step 6 的查询键。
# 写成 __n_rows 会得到 face___n_rows,查询永远取到默认值 0 —— 且因为
# 零输入本来就该全无证据,5 条测试仍会全绿,缺陷会被完美掩盖。
# 守卫用 `in self.features` 而非 `in self.data`:前者表示该模态提取成功。
# 用 self.data 的话,面部数据缺失时会 KeyError,落进下面的宽 except 返回 {} ——
# 连手势与语音特征一起丢掉。行数对每个模态都是纯附加项,不该有这种放大效应。
if 'face' in self.features:
    self.features['face']['_n_rows'] = float(len(self.data['face']))
if 'gesture' in self.features:
    self.features['gesture']['_n_rows'] = float(len(self.data['gesture']))
for key in ('voice_interview', 'voice_research'):
    if key in self.features:
        self.features[key]['_n_rows'] = float(len(self.data[key]))
```

⚠️ `_n_rows` 以单下划线开头,`_normalize_keys` 会保留它;它**不参与指标匹配**(所有 `mapping_rules` 关键词都不含 `n_rows`),只被 G3 读取。

- [ ] **Step 6: 删除步骤 2/3/4 的兜底逻辑,改为证据门**

把 `map_features_to_scores` 里第 2 步(假简历 fallback)、第 3 步(智能降级三条硬编码代理)、第 4 步(兜底 BASELINE_FILL)整段删除,替换为:

```python
from .evidence_gate import confidence_from, gate, user_message

# ...(在方法内,替换原第 2~4 步)

# 顶层缺口累加器(方法开始处初始化一次,供返回值的 evidence_gaps 用)
all_evidence_gaps = []

# ...进维度循环后,每个维度内:
matched = []          # [(keyword, weight, is_positive, human_name, raw_value, matched_key)]
dim_gaps = []         # 本维度的缺口,二者都要声明在维度循环内

for keyword, weight, is_positive, human_name, importance in indicators:
    found_key, found_val = self._fuzzy_match(keyword, all_features)
    if found_key is None:
        dim_gaps.append(f"{human_name}: 未采集到对应数据")
        continue

    # 伴随的 _std 用于 G2(常量判定);模态行数用于 G3(样本量)
    std_key = (found_key[:-len("_mean")] + "_std") if found_key.endswith("_mean") else None
    std = all_features.get(std_key) if std_key else None
    # 模态名按**实际存在的前缀**比对,不要用 split("_",1)[0] ——
    # 对 voice_research_research_pitch_variation_mean 它会得到 'voice',
    # 于是所有语音指标的 n_valid 恒为 0,被 G3 无差别拦下。
    n_rows_by_modality = {k[:-len("_n_rows")]: v
                          for k, v in all_features.items() if k.endswith("_n_rows")}
    modality = next((m for m in n_rows_by_modality
                     if found_key.startswith(m + "_")), None)
    n_valid = int(n_rows_by_modality.get(modality, 0))

    chk = gate(found_key, found_val, n_valid=n_valid, std=std)
    if not chk.ok:
        # ⚠️ 只准用 user_message。chk.reason 含维护者文案(封停理由等),
        # 而 dim_gaps 会被 Task 6 渲染进报告的"证据缺口"一节 —— 直接用会外泄内部信息。
        dim_gaps.append(f"{human_name}: {user_message(chk)}")
        continue
    matched.append((keyword, weight, is_positive, human_name, found_val, found_key))

# 维度循环末尾,把本维缺口并入顶层:
all_evidence_gaps.extend(dim_gaps)
```

⚠️ **作用域是这里的关键**:`matched` / `dim_gaps` 必须在**维度循环内**声明(原代码的 `evidence_chain = []` 就在这个位置),`all_evidence_gaps` 在**方法开头**声明一次。写错作用域会让所有维度共用一个缺口列表。

其中 `_fuzzy_match(keyword, all_features) -> (key, value) | (None, None)` 抽出原第 1/2 步的匹配逻辑(只保留精确匹配与 `_mean/_std/_sum` 宽松匹配,**不含降级与兜底**)。

- [ ] **Step 7: 分数只在有证据时产出**

```python
if not matched:
    dimension_results[dim_key] = {
        "display_name": rule_config["name"],
        "score": None,
        "level": "证据不足",
        "narrative": "本次未采集到足以评估该行为线索的有效样本。",
        "evidence_chain": [],
        "evidence_gaps": dim_gaps,
        "confidence": "无",
        "matched_indicators": f"0/{len(indicators)}",
        "stats": {},
    }
    continue
```

有证据时:

```python
confidence = confidence_from(len(matched), len(indicators))
score = round(max(0.0, min(100.0, weighted_sum / total_weight * 100)), 2)
```

- [ ] **Step 8: 汇总总分(可能为 None)**

```python
valid = [d for d in dimension_results.values() if d["score"] is not None]
if not valid:
    final_total = None
else:
    num = sum(d["score"] * self.dimension_weights[k]
              for k, d in dimension_results.items() if d["score"] is not None)
    den = sum(self.dimension_weights[k]
              for k, d in dimension_results.items() if d["score"] is not None)
    final_total = round(num / den, 2)
```

- [ ] **Step 9: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py tests/test_evidence_gate.py -v`
Expected: 全部 PASS

- [ ] **Step 10: 真实会话回归**

```bash
~/miniconda3/envs/jingxin/bin/python -c "
import sys; sys.path.insert(0, '.')
from report_frontend import LogDataLoader, PsychologicalFeatureEngine, ResearchCapabilityMapper
d = LogDataLoader().get_fused_latest_data()
f = PsychologicalFeatureEngine(d).extract_all_features()
r = ResearchCapabilityMapper().map_features_to_scores(f)
print('总分:', r['total_score'])
for k, v in r['dimensions'].items():
    print(f\"  {k}: score={v['score']} conf={v['confidence']}\")
print('gaps:', r['evidence_gaps'][:5])
"
```

Expected:**不再出现 `conf=高`**(spec §6.2 基线是 4/5 维为"高")。

- [ ] **Step 11: 提交**

```bash
# ⚠️ 必须含 feature_engine.py —— Step 5 改了它(产出 _n_rows)。
# 漏掉它会把"消费者"提交而"生产者"不提交,提交后 n_valid 恒为 0、G3 全盘拦下。
git add report_frontend/research_mapper.py report_frontend/feature_engine.py tests/test_report_layer.py
git commit -m "feat: research_mapper 接入证据门,删除兜底与硬编码代理"
```

---

## Task 3b: `report_generator` 的过渡防护与诚实短接

**背景**:Task 3 让 `score` 可为 `None`,而 `report_generator._generate_deep_text_analysis` 按 `x[1]['score']` 排序 → `TypeError`,被 `generate_report` 的宽 `except` 吞成空报告。加最小防护后暴露出更坏的问题:零证据时会走通到该函数第 4 节,打印「候选人在**证据不足**维度表现最为突出,显示出良好的**科研天赋**」—— **零证据下的才能断言,正是本次改动要消灭的伪造。**

**关键事实**:Task 3 + 槽位级封停后,**真实会话上 5 个维度全部 0/4**,即"无任何维度出分"是当前唯一可达情形。因此一个"全无证据即短接"的守卫覆盖今天所有可能发生的会话。

任务 6 会整段重写该函数;本任务是过渡,代码注释必须写明这一点。

- [ ] **Step 1: 全无证据短接**

在 `_generate_deep_text_analysis` 顶部(生成 `html_parts` 之前)插入:

```python
    # 过渡短接:Task 3 起 score 可能为 None。经 Task 3 + 槽位级封停后,
    # 真实会话上 5 个维度全部 0/4 —— "无任何维度出分"是当前唯一可达情形。
    # 此情形下直接给诚实的空报告,不进入下面那些基于 .get(..., 0) 默认值的段落,
    # 否则会打印"候选人在证据不足维度表现最为突出,显示出良好的科研天赋"这类
    # 零证据下的才能断言 —— 正是本次改动要消灭的伪造。Task 6 会整段重写本函数。
    scored = [(k, v) for k, v in result['dimensions'].items() if v['score'] is not None]
    if not scored:
        gaps = "".join(f"<li>{g}</li>" for g in result.get("evidence_gaps", []))
        return (
            "<h3>行为观测摘要</h3>"
            "<p><strong>本次会话未采集到足以支撑评估的有效证据。</strong></p>"
            f"<ul>{gaps}</ul>"
        )
```

原第 4 节里的 `scored = [...]` 由上面这处承接(不要重复声明)。

- [ ] **Step 2: 分数卡片不渲染 None**

`_build_html_report` 里 `result['total_score']` 直接插值会印出"综合科研潜力评分为 **None** 分"。改为:

```python
        _score_display = result['total_score'] if result['total_score'] is not None else "—"
```

并在分数卡片处用它替代 `{result['total_score']}`。(`result['total_level']` 已由 Task 3 在 `None` 时置为「证据不足」,无需处理。)

- [ ] **Step 3: 测试**

```python
def test_zero_evidence_report_makes_no_claims():
    """零证据时不得出现任何才能/心理素质断言(spec §5.4)。"""
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores({})
    html = ReportGenerator(output_dir="/tmp")._generate_deep_text_analysis({}, result)
    for claim in ("科研天赋", "心理素质", "最为突出", "表现最为"):
        assert claim not in html, f"零证据下仍出现断言:{claim}"
    assert "证据不足" in html or "未采集到" in html


def test_all_slot_level_quarantine_keys_are_pinned():
    """7 个槽位级封停键必须逐个被钉住,不能只测真实键。"""
    from report_frontend.evidence_gate import is_quarantined

    for entry in ("gaze_stability", "au4_freq", "au7_freq", "jitter",
                  "speech_ratio", "eye_contact", "fluency_score"):
        assert is_quarantined(entry) is not None, f"{entry} 未被封停"
```

- [ ] **Step 4: 运行并提交**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest -q`
Expected: 全部 PASS

```bash
git add report_frontend/report_generator.py tests/test_report_layer.py
git -c user.name='huihuibuhui227' -c user.email='huihuibuhui227@gmail.com' \
  commit -m "fix: 零证据时短接报告叙事层,分数卡片不渲染 None"
```

---

## Task 4: 删死分支与恢复 question_index

**Files:**
- Modify: `report_frontend/feature_engine.py`
- Modify: `tests/test_report_layer.py`

**Interfaces:**
- Consumes: 无
- Produces: `feature_engine` 不再产出 `fluency_score`/`fluency_proxy`(死分支);`question_index` 作为协变量保留

- [ ] **Step 1: 写失败的测试**

```python
def test_question_index_is_not_dropped():
    """spec §8-3:question_index 曾被 'index' 跳过规则静默丢弃。"""
    import pandas as pd

    from report_frontend.feature_engine import PsychologicalFeatureEngine

    df = pd.DataFrame({"question_index": [0, 1, 2], "pitch_mean": [100.0, 120.0, 110.0]})
    feats = PsychologicalFeatureEngine({"voice_research": df}).extract_all_features()
    keys = " ".join(feats.get("voice_research", {}).keys())
    assert "question_index" in keys, "question_index 仍被丢弃"


def test_dead_fluency_branch_removed():
    """spec §5.3:该分支要求列名同时含 speech_ratio 与 mean,故为死代码。

    ⚠️ 必须喂入 `speech_ratio_mean` 这一**(唯一能触发该分支的)列名形状**。
    只喂真实日志的 `speech_ratio` 会让本测试**改动前后都通过** —— 零约束力。
    Task 4 实现时实测:改动前 `'speech_ratio' -> []` 而
    `'speech_ratio_mean' -> ['research_fluency_score_mean', 'research_fluency_proxy']`。
    """
    import pandas as pd

    from report_frontend.feature_engine import PsychologicalFeatureEngine

    # 真实日志的形状(永不触发分支)+ 唯一能触发分支的形状,两者都要在
    df = pd.DataFrame({"speech_ratio": [0.9, 0.8, 0.95],
                       "speech_ratio_mean": [0.9, 0.8, 0.95]})
    feats = PsychologicalFeatureEngine({"voice_research": df}).extract_all_features()
    keys = " ".join(feats.get("voice_research", {}).keys())
    assert "fluency_score" not in keys
    assert "fluency_proxy" not in keys
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k "question_index or dead_fluency" -v`
Expected: FAIL

- [ ] **Step 3: 修跳过规则(`feature_engine.py:117`)**

```python
# 改前
if any(x in col.lower() for x in ['id', 'index', 'unnamed', 'timestamp']):
    continue

# 改后:放行 question_index 作为协变量
_SKIP_COLS = ('id', 'unnamed', 'timestamp')

if any(x in col.lower() for x in _SKIP_COLS):
    continue
if 'index' in col.lower() and 'question' not in col.lower():
    continue
```

- [ ] **Step 4: 删死分支(`feature_engine.py:289-293`)**

删除 `if 'speech_ratio' in col_lower and 'mean' in col_lower:` 整个分支。

- [ ] **Step 5: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add report_frontend/feature_engine.py tests/test_report_layer.py
git commit -m "fix: 保留 question_index 协变量,删除 fluency 死分支"
```

---

# 提交 B:内容(叙事层 + 常模 + 措辞)

## Task 5: 删杜撰常模与百分位

**Files:**
- Modify: `report_frontend/visualizer.py`
- Modify: `report_frontend/research_mapper.py`
- Modify: `tests/test_report_layer.py`

**Interfaces:**
- Consumes: 无
- Produces: 雷达图无"常模基准"trace;证据链无 `percentile` 字段

- [ ] **Step 1: 写失败的测试**

```python
def test_no_fabricated_percentile():
    """spec §5.5:百分位只能来自真实常模。

    ⚠️ 必须喂**能过证据门**的输入。否则 evidence_chain 全为空、嵌套循环零断言,
    本测试对**任何**实现都通过 —— 包括把百分位加回来的实现。
    Task 5 复审实测:原输入 `interview_pause_duration_mean` 让 5 个维度全部
    score=None / chain=[],是彻底的空断言("回归守卫"的标签夸大了它)。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0}}
    r = ResearchCapabilityMapper().map_features_to_scores(feats)

    total = sum(len(d["evidence_chain"]) for d in r["dimensions"].values())
    assert total > 0, "输入未过证据门,本测试退化为空断言"
    for dim in r["dimensions"].values():
        for ev in dim["evidence_chain"]:
            assert "percentile" not in ev or ev["percentile"] is None


def test_radar_has_no_norm_baseline():
    """spec §5.5:[60]*5 与 '常模基准' 图例必须删除。"""
    from report_frontend.visualizer import ReportVisualizer

    result = ResearchCapabilityMapper().map_features_to_scores({})
    fig = ReportVisualizer(output_dir="/tmp")._build_radar_figure(result)
    names = [t.name for t in fig.data]
    assert "常模基准" not in names
    # 正向断言:否则"一张轨迹都没有的图"也会通过,候选分轨迹的存续无人钉住
    assert names == ["候选人得分"]
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k "percentile or radar" -v`
Expected: FAIL

- [ ] **Step 3: 确认 research_mapper 已无常模残留**

百分位计算已在 **Task 3 Step 3** 随 `self.baselines` 一并删除(那里删 `self.baselines` 就必须同步删使用点,否则 `AttributeError`)。这里只做确认:

```bash
grep -n "baselines\|percentile" report_frontend/research_mapper.py
```

Expected:0 命中(若 Task 3 已正确完成)。若有残留,就地删除。

- [ ] **Step 4: 删 visualizer 的常模线**

删除 `create_capability_radar` 里的 `baselines = [60, 60, 60, 60, 60]`、`baselines += [baselines[0]]`、以及 `fig.add_trace(go.Scatterpolar(r=baselines, ..., name='常模基准', ...))`。

同时把 `create_capability_radar` 中"画图 + `_save_fig`"拆出纯函数,便于测试:

```python
def _build_radar_figure(self, result: Dict[str, Any]):
    """构造雷达图,不落盘 —— 便于测试(spec §6)。"""
    ...  # 原 create_capability_radar 的 fig 构造部分,去掉 baselines

def create_capability_radar(self, result: Dict[str, Any]) -> str:
    fig = self._build_radar_figure(result)
    return self._save_fig(fig, "capability_radar")
```

- [ ] **Step 5: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add report_frontend/visualizer.py report_frontend/research_mapper.py tests/test_report_layer.py
git commit -m "fix: 删除杜撰常模与基于它的百分位"
```

---

## Task 6: 重写硬编码叙事层

**Files:**
- Modify: `report_frontend/report_generator.py:123-250`
- Modify: `tests/test_report_layer.py`

**Interfaces:**
- Consumes: Task 3 的 `score=None` / `confidence="无"` / `evidence_gaps`
- Produces: `_generate_deep_text_analysis` 只输出真实数值 + 证据状态,不含任何解读

- [ ] **Step 0: 修复 Task 3 遗留的两条空断言(同一测试文件)**

**背景**:控制器对 `tests/test_report_layer.py` 做了一次**系统扫描**(`sys.settrace` 记录每个测试实际执行的行,与 AST 里的 `assert` 行求差),结果:**12 个测试里恰好 2 个的断言行从未执行**。这两条正好守着 Task 3 的核心主张 —— 意味着"假简历兜底回归""封停列重新进链"都不会被任何测试拦住。

扫描结果(其余 10 个测试的断言全部执行):

| 测试 | 断言行 | 从未执行 |
|---|---|---|
| `test_no_fake_resume_fallback` | 3 | **3** |
| `test_quarantined_columns_are_rejected` | 2 | **2** |

两处根因相同:fixture 让证据链为空 → 遍历 `evidence_chain` 的循环**零次执行**。

把 `tests/test_report_layer.py` 里这两个函数整体替换为:

```python
def test_no_fake_resume_fallback():
    """spec §5.3:假简历兜底与 BASELINE_FILL 必须不可达。

    ⚠️ 原版用零输入 fixture,证据链为空 → 循环零断言,对任何实现都通过。
    改为喂一个只让 1 个槽过门的输入,断言 matched 恰为 1/4 且缺口为 3。
    若 BASELINE_FILL 回归,缺失的 3 个槽会被填空 → matched 变 4/4 → 本测试变红。
    控制器已实测该 fixture 输出:score=100.0 / conf=低 / matched=1/4 / 缺口 3。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]

    assert dim["matched_indicators"] == "1/4", "缺失槽被填充(BASELINE_FILL 回归?)"
    assert len(dim["evidence_gaps"]) == 3
    for ev in dim["evidence_chain"]:
        # 裸键名是假简历兜底的签名;带模态前缀才是真测量
        assert ev["feature"] != "logic_keyword_density", "裸键名 = 假简历兜底的签名"
        assert "BASELINE" not in ev["feature"]
        assert "代理" not in ev["feature"]


def test_quarantined_columns_are_rejected():
    """spec §5.2:封停列即使有值也不得进入证据链。

    ⚠️ 原版只喂封停列 → 全部被拒 → 链为空 → 循环零断言。
    改为同时喂一个干净列让链非空,再断言封停列不在其中。
    若 G4 被移除,focus_score 会进链 → 本测试变红。
    控制器已实测:链为 [voice_research_logic_keyword_density],focus_score 不在其中。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0},
             "face": {"face_focus_score_mean": 0.3,
                      "face_focus_score_std": 0.05,
                      "_n_rows": 100.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]

    assert dim["evidence_chain"], "证据链为空 —— 本测试退化为空断言"
    for ev in dim["evidence_chain"]:
        assert "focus_score" not in ev["feature"]
        assert "symmetry_score" not in ev["feature"]
```

**验收要求**:改完后**重跑一遍上述扫描**(同一段 `sys.settrace` 脚本),断言"存在未执行断言的测试数 = 0"。这是本轮唯一能证明修复到位的办法 —— 光看代码看不出循环跑没跑。

- [ ] **Step 1: 写失败的测试**

```python
import re

BANNED = ["焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
          "录用", "人格", "心理画像", "常模"]


def test_deep_analysis_has_no_banned_words():
    """spec §5.4 + §5.6:叙事层不得含情绪/心理/诚信构念。

    ⚠️ 只覆盖**叙事层自己写的句子**。报告里出现的禁止词有另一个来源:
    `research_mapper` 提供的两个标签 —— `display_name`「抗压与情绪稳定性」
    (含 抗压、情绪稳定)与 `human_name`「面部紧张度」(含 紧张)。
    实测确认全 mapper 只有这两处命中。

    它们属 **Task 7** 的改名范围(Task 7 改完 `display_name` 与 `human_name`
    后,**须恢复全量扫描** —— 见计划 Task 7 Step 3b)。
    本测试用剔除这两个标签的方式,把叙事层自己的输出隔离出来测。
    剔除是精确字符串替换,所以叙事层**自己在别处**写出的禁止词仍会被抓到。
    """
    from report_frontend.report_generator import ReportGenerator

    feats = {"face": {"face_tension_score_mean": 0.5},
             "gesture": {"gesture_left_hand_jitter_mean": 0.02}}
    result = ResearchCapabilityMapper().map_features_to_scores(feats)
    html = ReportGenerator()._generate_deep_text_analysis(feats, result)

    # 剔除 mapper 提供的标签(Task 7 修复后本段移除,恢复全量)
    for label in ("抗压与情绪稳定性", "面部紧张度"):
        html = html.replace(label, "")

    for word in BANNED:
        assert word not in html, f"叙事层出现禁止词:{word}"


def test_deep_analysis_handles_none_scores():
    """零证据时不得崩溃(score 为 None,不能参与排序)。"""
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores({})
    html = ReportGenerator()._generate_deep_text_analysis({}, result)
    assert "证据不足" in html


def test_no_hardcoded_gaze_claim():
    """spec §5.4:那句'未出现异常的回避行为'是纯硬编码。"""
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores({})
    html = ReportGenerator()._generate_deep_text_analysis({}, result)
    assert "未出现异常的回避行为" not in html
    assert "如外科医生般" not in html
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k "banned or none_scores or hardcoded_gaze" -v`
Expected: FAIL

- [ ] **Step 3: 重写 `_generate_deep_text_analysis`**

⚠️ **这四处必须一并清掉**(Task 3 复审发现,均在本函数内;Task 3b 的短接让它们在今天不可达,但重写时必须真正删除,不能只是被跳过):

| 位置 | 问题 |
|---|---|
| 原 `:165,176,194,215` | 四处 `['percentile']` 读的是已删除的字段,默认值 `50` → 每场会话都印「优于 **50%** 的受试者」「处于 **50%** 的水平」「超越了 **50%** 的人群」—— **四条恒定在 50 的杜撰常模**,比改动前更假(Task 3 之前至少还是算出来的) |
| 原 `:137` | 查 `face_micro_exp_au_name_au4_freq`,而实际产出的键是 `face_micro_exp_micro_exp_au_name_au4_freq` → 永远取默认 0 → `:178-181` 那句「微表情监测**未检测到显著的焦虑特征**」**无条件打印**(spec §5.4 点名必删) |
| 原 `:129,174` | `face_feats.get(..., 0)` 默认值驱动「展现了极佳的**情绪控制力**和**心理稳定性**」 |
| 原 `:157` | 「综合科研潜力评分为 X 分」+「毫秒级量化分析」+「常模参照模型」 |

⚠️ **必须保留 Task 3b 建立的两条性质**,否则重写会打破现已通过的测试:

1. **全无证据时不得进入任何基于默认值的段落。** 现有测试 `test_zero_evidence_report_makes_no_claims` 钉住这条(零证据下不得出现「科研天赋」「心理素质」「最为突出」)。重写后的函数在 `scored` 为空时仍须直接返回诚实的证据缺口摘要。
2. **分数不得渲染为 `None`。** `_build_html_report` 里的 `_score_display` 兜底须保留。

这两条是 Task 3b 的产物,`grep -n "_score_display\|scored = "` 可定位现状。

整段(原 `:123-250`)替换为只陈述测量事实的版本。每个维度输出:

```python
def _render_dimension_block(self, dim_key: str, dim: Dict[str, Any]) -> str:
    """渲染单个维度的证据状态。不解读,不推断,不加形容词。"""
    if dim["score"] is None:
        gaps = "".join(f"<li>{g}</li>" for g in dim.get("evidence_gaps", []))
        return f"""
        <h3>{dim['display_name']}</h3>
        <p><strong>证据不足</strong> —— 本次未采集到足以评估该行为线索的有效样本。</p>
        <ul>{gaps}</ul>
        """

    rows = "".join(
        f"<tr><td>{e['human_name']}</td><td>{e['raw_value']}</td>"
        f"<td>{e['normalized_score']}</td><td>{e['weight']}</td></tr>"
        for e in dim["evidence_chain"]
    )
    return f"""
    <h3>{dim['display_name']}</h3>
    <p>依据 {dim['matched_indicators']} 个指标；置信度：<strong>{dim['confidence']}</strong>。</p>
    <table><thead><tr><th>指标</th><th>原始值</th><th>归一值</th><th>权重</th></tr></thead>
    <tbody>{rows}</tbody></table>
    """
```

顶层函数:

```python
def _generate_deep_text_analysis(self, features, result) -> str:
    parts = []
    total = result["total_score"]
    if total is None:
        parts.append("<p>本次会话未采集到足以支撑评估的有效证据。</p>")
    else:
        parts.append(
            f"<p>综合行为观测摘要：{result['total_level']}"
            f"(置信度上限：{max((d['confidence'] for d in result['dimensions'].values()), key=_CONF_ORDER)})</p>"
        )
    for dim_key, dim in result["dimensions"].items():
        parts.append(self._render_dimension_block(dim_key, dim))
    if result.get("evidence_gaps"):
        parts.append("<h3>证据缺口</h3><ul>"
                     + "".join(f"<li>{g}</li>" for g in result["evidence_gaps"])
                     + "</ul>")
    return "".join(parts)
```

并在模块顶部:

```python
_CONF_ORDER = {"无": 0, "低": 1, "中": 2, "高": 3}
```

- [ ] **Step 4: 同步修 `_build_html_report`**

`report_generator.py:310` 的 `{result['total_score']}` 与 `:312` 的 `{result['total_level']}` 在零证据时须显示"证据不足",不能显示 `None`:

```python
_score_display = result["total_score"] if result["total_score"] is not None else "—"
```

- [ ] **Step 5: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 6: 端到端验证**

```bash
~/miniconda3/envs/jingxin/bin/python -c "
import sys; sys.path.insert(0, '.')
from report_frontend import ReportGenerator
p = ReportGenerator(output_dir='/tmp/jxreport').generate_report()
print('报告:', p)
" 2>&1 | tail -5
```

然后人工打开生成的 HTML,确认:
- 出现"证据不足"
- 无"情绪状态与抗压能力深度剖析"整节
- 无"未检测到焦虑特征"
- 无"如外科医生般"

- [ ] **Step 7: 提交**

```bash
git add report_frontend/report_generator.py tests/test_report_layer.py
git commit -m "refactor: 重写报告叙事层,删除硬编码解读"
```

---

- [ ] **Step 8: 审查修复轮(2 Important + 2 顺手项)**

审查判定 **Needs fixes**。两条 Important **都是同一类**:新测试用了**零证据 fixture**,而禁止词与硬编码句住在**出分路径**上 —— 于是两条测试在改前改后都通过,守不住它们声称要守的东西。(这是本计划同类失效的**第 7、8 次**。)

**统一修法:把这三条测试的 fixture 换成"混合 fixture",让它走通出分路径。** 控制器已实测该 fixture 会使 `logical_thinking` 出分(即进入被删的那些段落),而当前实现下硬编码句与禁止词命中均为**空** —— 所以修完后测试有约束力:

```python
_MIXED = {"voice_research": {"logic_keyword_density": 0.05,
                             "logic_keyword_density_std": 0.01,
                             "_n_rows": 100.0}}
```

**(a) `test_no_hardcoded_gaze_claim`** —— 把 `ResearchCapabilityMapper().map_features_to_scores({})` 改为 `map_features_to_scores(_MIXED)`,`_generate_deep_text_analysis({}, result)` 保持。原版因零证据短接,永远到不了那三句。

**(b) `test_deep_analysis_has_no_banned_words`** —— 同样换成 `_MIXED`。否则它只扫"无证据"那几句,而禁止词原本住在出分路径。

**(c) `test_deep_analysis_handles_none_scores`** —— 换成 `_MIXED`,使其 docstring 声称的"`score` 为 `None` 不能参与排序"真的被走到(零证据输入根本到不了 `max(..., key=_CONF_ORDER.get)`)。

**(d) `test_quarantined_columns_are_rejected` 的 `symmetry_score` 断言结构性不可证伪** —— `symmetry_score` 是 `stress_resilience` 的槽,而该测试只查 `logical_thinking`;且新 fixture 根本没喂对称性列。`grep -rn symmetry tests/` 显示这一行是它唯一出现处,即它**哪儿都没被钉住**。

⚠️ 注意:补 `stress_resilience` 的链断言**无效** —— 该维四个槽(`tension_score`/`jitter`/`gaze_deviation`/`symmetry_score`)**全部封停**,链必为空,断言仍然恒真。**正确修法是断言"进不了链、且以缺口形式出现"**:

```python
def test_quarantined_columns_are_rejected():
    """spec §5.2:封停列即使有值也不得进入证据链。

    ⚠️ 原版只喂封停列 → 全部被拒 → 链为空 → 循环零断言。
    改为同时喂一个干净列让链非空,再断言封停列不在其中。
    若 G4 被移除,focus_score 会进链 → 本测试变红。

    symmetry_score 属 stress_resilience 维度,而该维四个槽全部封停、链必为空,
    故不能靠"链里没有它"来钉(那恒真)。改为断它**出现在证据缺口里** ——
    若删掉它的封停条目且喂入该列,它会进链、缺口里便不再有它 → 本测试变红。
    控制器实测:stress_resilience 链=0、缺口=4、缺口含「面部对称性」。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0},
             "face": {"face_focus_score_mean": 0.3,
                      "face_focus_score_std": 0.05,
                      "face_symmetry_score_mean": 0.98,
                      "face_symmetry_score_std": 0.02,
                      "_n_rows": 100.0}}
    dims = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]

    lt = dims["logical_thinking"]
    assert lt["evidence_chain"], "证据链为空 —— 本测试退化为空断言"
    for ev in lt["evidence_chain"]:
        assert "focus_score" not in ev["feature"]

    sr = dims["stress_resilience"]
    assert sr["evidence_chain"] == [], "封停列进了链"
    assert any("对称" in g for g in sr["evidence_gaps"]), "对称性槽未被处理"
```

**(e) 缺口重复渲染**(Minor,顺手) —— `_render_dimension_block` 的逐维 `<ul>` 与顶层「证据缺口」段打印的是同一批字符串(`all_evidence_gaps` 就是各维缺口的并集),实测 19 条缺口被渲染两遍。**出分路径下删掉顶层那一段**(保留逐维列表,信息更全);**短接路径仍需要它**(那时不渲染任何维度)。

**(f) 删 `_get_percentile_badge`**(Minor) —— 本任务删掉了它唯一的调用方,grep 显示只剩定义。spec §5.5 要求不留任何百分位机器。**顺手删掉。**

**验收**(修复后必须全部做到):

1. `pytest -q` 全绿
2. 对 (a)(b)(c) 三条,证明"若把对应改动回退,测试会失败" —— 本计划已栽过八次空断言
3. 重跑空断言扫描(§Step 0 的脚本),断言"存在未执行断言的测试数 = 0"
4. 生成一份报告,确认缺口不再重复出现

---

## Task 7: 措辞替换与眼动图处置

**Files:**
- Modify: `report_frontend/research_mapper.py`(`:377` 硬编码注入、维度 display_name / description)
- Modify: `report_frontend/visualizer.py`(眼动图 + 死代码)
- Modify: `tests/test_report_layer.py`

**Interfaces:**
- Consumes: 无
- Produces: 全仓库 `report_frontend/` + `templates/` 禁止词 0 命中

- [ ] **Step 1: 写禁止词全仓扫描测试**

```python
import ast
from pathlib import Path

SCOPE = [Path("report_frontend"), Path("templates")]

# 禁止词表(spec §5.6)。此处在 Task 7 内重新定义,不依赖 Task 6 —— 读者可能乱序阅读。
BANNED = ["焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
          "录用", "人格", "心理画像", "常模"]


def test_no_banned_words_in_output_strings():
    """spec §5.6:禁止词不得出现在任何字符串字面量里(含 docstring)。

    注释不算 AST Constant,所以说明性注释不受影响。
    """
    offenders = []
    for root in SCOPE:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for word in BANNED:
                        if word in node.value:
                            offenders.append(f"{path}:{node.lineno} {word}")
    assert not offenders, "字符串字面量含禁止词:\n" + "\n".join(offenders)
```

- [ ] **Step 2: 运行,确认失败**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/test_report_layer.py -k banned_words_in_output -v`
Expected: FAIL,列出 `research_mapper.py` 的 `"抗压与情绪稳定性"`、`"自信度"`、`pitch_info="语调丰富"` 等

- [ ] **Step 3: 改维度名与描述**

`research_mapper.py` 的 `mapping_rules`:

| 原 | 改为 |
|---|---|
| `"name": "抗压与情绪稳定性"` | `"name": "情境行为稳定性"` |
| `"description": "评估高压下的情绪控制力、生理指标平稳度及焦虑水平。"` | `"description": "观测会话中的可测行为量。"` |
| `"name": "自信度"` | `"name": "行为表现活跃度"` |
| `"description": "评估自我效能感、肢体开放度及眼神交流质量。"` | `"description": "观测肢体与注视相关的可测量。"` |
| `"name": "逻辑思维与专注度"` | `"name": "话语结构特征"` |
| `"name": "认知负荷效率"` | `"name": "言语流畅特征"`(审查 Q5:"效率"= 产出/投入,系统没有"产出") |
| **`"human_name": "面部紧张度"`** | **`"human_name": "眉间收缩与唇部压缩"`** |
| `inference_template` 里的"抗压能力""科研自信心" | 删除模板机制(见 Step 4) |

⚠️ **`human_name` 这一行是 Task 6 上报的缺口** —— 原改名表只覆盖 `display_name`,而实测 `human_name` 里也有一个含禁止词的:「面部紧张度」(含 **紧张**),它由 `_render_dimension_block` 渲染进报告的指标表与证据缺口清单。**不改它,Task 6 那条禁止词测试即使到 Task 7 之后仍然会红。**

新名「眉间收缩与唇部压缩」**是控制器给的默认**(该槽实际覆盖 au4 眉间收缩与 au23 唇部压缩);spec §5.6 建议的"面部紧张相关动作单元活动率"仍含"紧张",不可用。**使用者可否决。**

改完后请**再全量扫一次** `research_mapper.py` 的 `name` / `description` / `human_name`,确认 0 命中(不能只改表里列的这两个)。

**命名已由使用者于 2026-09-22 拍板,采用上表默认值。** spec §7.1 的待定项关闭。

⚠️ 唯一约束:新名字里**不得含禁止词**。特别地,**不能用"压力情境下的行为稳定性"** —— "压力"在 §5.6 的禁止词表里,用了会被本任务 Step 1 的扫描测试打回。

- [ ] **Step 3a: 恢复禁止词全量扫描(Task 6 收窄的那条)**

Task 6 的 `test_deep_analysis_has_no_banned_words` 曾收窄为"剔除 `抗压与情绪稳定性` 与 `面部紧张度` 两个 mapper 标签后扫描",原因是那两个字符串属 Task 7 范围。

**Step 3 改名完成后,把那段剔除逻辑删掉,恢复全量:**

```python
    html = ReportGenerator()._generate_deep_text_analysis(feats, result)

    # Task 7 已修完 display_name 与 human_name,恢复全量扫描(不再剔除标签)
    for word in BANNED:
        assert word not in html, f"叙事层出现禁止词:{word}"
```

**验收**:恢复全量后该测试仍须 PASS。若红,说明 mapper 还有未改净的标签 —— 回到 Step 3 的"全量扫一次"。

- [ ] **Step 3b: 清理报告头部与标题字符串**

`report_generator.py:301-302` 的 `"🔬 JingXin 科研能力评估报告"` / `"基于多模态心理特征的深度分析与判推"`,以及 `:155` 的 `"常模参照模型"`:

| 位置 | 原 | 改为 |
|---|---|---|
| `:301` | `JingXin 科研能力评估报告` | `JingXin 面试行为观测报告` |
| `:302` | `基于多模态心理特征的深度分析与判推` | `基于多模态行为量的结构化观测` |
| `:155` | `并通过常模参照模型进行了深度判推` | 删 |
| `:155` | `进行了毫秒级量化分析` | 删(采样率 ≈10fps,不是毫秒级) |

同时把 `report_generator.py` 模块顶部注释与 `visualizer.py` 的 `create_capability_radar` 标题 `"📊 科研能力五维模型评估"` 改为 `"📊 五维行为观测"`。

- [ ] **Step 4: 删推理模板与硬编码注入**

删除 `_generate_deep_inference`(`:360-382`)整段,连同 `mapping_rules` 里的 `inference_template` 字段。`narrative` 改由 Task 6 的证据状态渲染产生,不再有模板填空。

这同时删掉 `:377` 的 `pitch_info="语调丰富", eye_info="眼神交流充分", hand_info="手势自然"`。

- [ ] **Step 5: 眼动图按 spec §5.5 处置**

删除 `create_gaze_plot_from_df` 里的两张图与 `add_shape(rect, x0=-1, y0=-1, x1=1, y1=1)`;函数体改为:

```python
def create_gaze_plot_from_df(self, df_face) -> Optional[str]:
    """M3 之前不生成眼动图。

    gaze_direction_y 是解剖常量、iris_x/y 是图像归一化坐标(编码人脸位置),
    两者都不能支撑"注视热力图"这个标题。见 spec §5.5。
    """
    return None
```

同时删除死代码 `visualizer.py:141-164`(函数体为 `return None` 的模拟眼动版本)。

- [ ] **Step 6: 运行,确认通过**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 7: 全仓人工复核**

```bash
grep -rn "焦虑\|紧张\|压力\|抗压\|情绪稳定\|说谎\|诚信\|录用\|人格\|心理画像\|常模" report_frontend/ templates/ --include="*.py" --include="*.html"
```

Expected:仅剩注释与 `evidence_gate.py` 的封停理由说明(那是给维护者看的,不进入输出)。

- [ ] **Step 8: 提交**

```bash
git add report_frontend/research_mapper.py report_frontend/visualizer.py tests/test_report_layer.py
git commit -m "refactor: 中性化维度命名与措辞,停用眼动图与死代码"
```

---

---

## Task 8: 修订设计文档的三处前提(spec §1)

**Files:**
- Modify: `docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md`

**Interfaces:**
- Consumes: 无
- Produces: 无(纯文档)

这三处不涉及代码,但与 ① 同期完成,否则设计文档会与增补矛盾。

- [ ] **Step 1: 修订 §6.3 阶段 1**

把 `| 1 | 三个数据集标定(**先上线**) | 低 |` 这一行替换为:数据集标定**不上线**,并写入 spec §1.1 的两条可选路径(a:有自有标注前 L2 不上线;b:数据集线严格限定为离线研究结论)。

同时在 §6.3 表格上方加一句:

> ⚠️ RecruitView 许可(CC BY-NC 4.0)明文禁止将本数据集**及在其上训练的模型**用于真实招聘、雇佣筛选或**心理画像**。故"数据集标定 → 先上线"这一组合被移除。详见增补 §1.1。

- [ ] **Step 2: 修订 §4.4 的四个理由**

删除第 1 条("声学特征可跨语言迁移,语言特征不能")中的该前提,替换为:

> 1. **跨语言可迁移性不对称(仅适用于非 f0 类特征)** —— 语言特征不能跨语言迁移。注意:f0 类特征是**例外**,普通话的词汇声调压缩了情感语音中的音高变异,故 f0 类特征同样不可跨语言直接搬运(增补 §1.2)。

保留其余三条(失败模式不同 / 隐私等级不同 / 时间线不同)。

- [ ] **Step 3: §6.2 补伦理审查**

在 §6.2 的"具体量表版本与授权需确认后使用"之后加:

> **⚠️ 强制前置项:** 按《科技伦理审查办法(试行)》第 2 条第(一)项,以人为研究参与者(含"利用个人信息数据")的科技活动**必须通过科技伦理审查**。采集候选人自评量表用于标定落入该条,2023-12-01 已施行。此项与量表授权并列,不得省略。

- [ ] **Step 4: 提交**

```bash
git add docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md
git commit -m "docs: 按心理测量学审查修订设计文档三处前提"
```

---

## 收尾:回归与验收

- [ ] **Step 1: 全量测试**

Run: `~/miniconda3/envs/jingxin/bin/python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 2: 采集层回归防线必须仍然通过**

Run: `~/miniconda3/envs/jingxin/bin/python experiments/duration_audit/reaggregate_normalized.py --stats legacy --verify-legacy`
Expected:0/2835510 格差异。**① 只改报告层,这条防线若被打破说明改错了范围。**

- [ ] **Step 3: 真实会话端到端**

按 Task 6 Step 6 跑一遍,人工确认报告内容。

- [ ] **Step 4: 与 spec 逐条对照**

打开 spec §6 的 7 条验证方式,逐条确认有对应测试或人工步骤。
