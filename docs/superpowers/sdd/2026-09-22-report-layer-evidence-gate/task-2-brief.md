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

