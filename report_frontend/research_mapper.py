# report_frontend/research_mapper.py

import pandas as pd
import numpy as np
import math
from typing import Dict, Any, List, Optional
import warnings
import json

from .evidence_gate import confidence_from, gate, normalize_value, user_message

warnings.filterwarnings('ignore')

# 置信度从低到高的顺序。取值域由 evidence_gate.Confidence 定义(高在本轮不可达)。
# 取"置信度上限"时用 max(..., key=CONF_ORDER.get);上限由本模块算进 coverage,
# 报告层直接渲染该字段,不再自己排一次序 —— 两处各存一份顺序表会漂移。
CONF_ORDER = {"无": 0, "低": 1, "中": 2, "高": 3}

# 聚合呈现的说明句(spec §5.6:综合分改为「区间 + 置信度 + 依据」)。
# 本系统没有标定样本,不同指标的量纲无法折算到同一尺度,故不给综合分,也不给评级。
# 覆盖事实与区间由 coverage(payload)与报告头部的覆盖卡承载,这里不重复计数。
NO_COMPOSITE_STATEMENT = ("未产出综合评分，也不给评级：本系统尚无标定样本，"
                          "不同指标的量纲无法折算为同一尺度。")


def _as_float(value) -> Optional[float]:
    """能当有限浮点数用就返回它,否则 None(非数值 / NaN / inf 一律 None)。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _observed_interval(mean, std) -> Optional[List[float]]:
    """本场会话内的实测区间 = 该指标会话内均值 ± 会话内标准差。

    ⚠️ 这是**本场会话自己的测量**:不是人群中位置、不是置信区间、不是任何人群参照
    —— spec §5.5 禁止一切暗示本系统拥有人群基线的区间。标准差为 0(常量)时没有
    区间可言(那种槽本就过不了 G2),返回 None,由报告层写明"未采集到变异信息"。
    下界截到 0:这些指标都是非负量纲。
    """
    spread = _as_float(std)
    center = _as_float(mean)
    if spread is None or center is None or spread <= 0:
        return None
    return [max(0.0, round(center - spread, 4)), round(center + spread, 4)]


def _indicator_sample_size(found_key: str, all_features: Dict[str, Any]) -> Optional[int]:
    """该指标**自己**的有效样本量 —— `feature_engine` 的 `<基名>_sample_size`。

    取不到(键不存在 / 不是整数)时返回 None,由调用方退回模态行数。

    为什么不能用模态行数代替它(最终审查 I1):语音日志里「回答太短」的那一行,密度列是
    空格(`connective_density=None` 不写 0,spec §8),特征引擎按 `dropna()` 丢掉它 ——
    而模态行数把它算在内。用模态行数会让报告在只由 3 行构成的均值上印「有效样本量 20」,
    更糟的是**让一行都不带的日志过掉 G3 门槛** —— 门槛正是 M1 存在的理由(诚实轴),
    单位错了它就只是一枚橡皮图章。

    `found_key` 通常是 `<基名>_mean`;没有 `_mean` 后缀的键(如
    `face_eye_contact_ratio`)直接查 `<键>_sample_size`。
    """
    base = found_key[:-len("_mean")] if found_key.endswith("_mean") else found_key
    raw = all_features.get(f"{base}_sample_size")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class ResearchCapabilityMapper:
    """
    行为指标映射引擎 (最终修复版 - 包含 stats 字段)
    """

    def __init__(self):
        # 说明:曾有的 self.demo_text_data(内置演示文本)与 self.baselines(无出处的
        # 常模均值/标准差)已删除 —— 前者让连接词密度在无文本列时凭空
        # 得到常量,后者既伪造证据又产出杜撰百分位(spec §5.3 / §5.5)。
        self.mapping_rules = {
            "logical_thinking": {
                "name": "话语结构特征",
                "description": "观测文本结构与面部动作单元相关的可测量。",
                "algorithm": "加权线性组合 + 认知负荷推断",
                "indicators": [
                    ("connective_density", 0.4, True, "连接词密度", "core"),
                    ("focus_score", 0.3, True, "面部专注度", "core"),
                    ("gaze_stability", 0.2, True, "视线稳定性", "core"),
                    ("au4_freq", 0.1, False, "困惑微表情 (皱眉)", "support"),
                ]
            },
            "stress_resilience": {
                "name": "情境行为稳定性",
                "description": "观测会话中的可测行为量。",
                "algorithm": "多模态生理信号融合 (面部 + 肢体 + 眼动)",
                "indicators": [
                    ("tension_score", 0.3, False, "眉间收缩与唇部压缩", "core"),
                    ("jitter", 0.3, False, "肢体抖动", "core"),
                    ("gaze_deviation", 0.2, False, "视线偏差", "core"),
                    ("symmetry_score", 0.2, True, "面部对称性", "support"),
                ]
            },
            "communication_fluency": {
                "name": "沟通表达流畅度",
                "description": "观测语音韵律与停顿相关的可测量。",
                "algorithm": "韵律特征与停顿分析",
                "indicators": [
                    ("fluency_score", 0.4, True, "语音流畅度", "core"),
                    ("speech_ratio", 0.3, True, "有效说话占比", "core"),
                    ("pitch_variation", 0.2, True, "语调变化", "support"),
                    ("pause_duration", 0.1, False, "平均停顿时长", "support"),
                ]
            },
            "confidence_level": {
                "name": "行为表现活跃度",
                "description": "观测肢体与注视相关的可测量。",
                "algorithm": "眼动 - 肢体多模态耦合模型",
                "indicators": [
                    ("hand_score", 0.3, True, "手势自信分", "core"),
                    ("shoulder_score", 0.2, True, "肩部放松度", "support"),
                    ("eye_contact", 0.3, True, "眼神接触比例", "core"),
                    ("energy", 0.2, True, "语音能量", "support"),
                ]
            },
            "cognitive_efficiency": {
                "name": "言语流畅特征",
                "description": "观测面部动作单元频率与回答长度的可测量。",
                "algorithm": "微表情频率分析与响应延迟回归",
                "indicators": [
                    ("au7_freq", 0.3, False, "眼部挤压 (费力)", "core"),
                    ("blink_rate", 0.2, True, "眨眼频率", "support"),
                    ("text_avg_length", 0.2, True, "回答详尽度", "support"),
                    ("reaction_time", 0.3, False, "反应延迟", "core"),
                ]
            }
        }

        self.dimension_weights = {
            "logical_thinking": 0.25,
            "stress_resilience": 0.25,
            "communication_fluency": 0.20,
            "confidence_level": 0.15,
            "cognitive_efficiency": 0.15
        }

    def map_features_to_scores(self, features: Dict[str, Any]) -> Dict[str, Any]:
        print("\n⚖️ 正在执行行为指标映射...")
        print("-" * 70)

        # 1. 扁平化
        all_features = {}
        for modality, feats in features.items():
            for k, v in feats.items():
                full_key = f"{modality}_{k}" if not k.startswith(modality) else k
                all_features[full_key] = v

        # 2. 各模态的有效行数 —— 证据门 G3(样本量)的唯一真实来源。
        # feature_engine 产出 <模态>__n_rows(L1 的 n_valid_frames 尚未落地,
        # 见 spec §5.1 G3 与演进表 §4)。取法必须用实际的模态前缀比对,
        # 不能用 split("_", 1)[0] —— 那会把 "voice_research_*" 切成 "voice",
        # 于是语音系指标永远拿不到行数、被 G3 无差别拦下。
        n_rows_by_modality = {k[:-len("__n_rows")]: v
                              for k, v in all_features.items() if k.endswith("__n_rows")}

        # 检查生理指标
        critical_physio_keys = ['tension_score', 'jitter', 'au7_freq', 'gaze_stability']
        found_physio = [k for k in critical_physio_keys if any(k in fk for fk in all_features.keys())]
        if found_physio:
            print(f"      ✅ 检测到真实生理特征：{', '.join(found_physio)}")

        dimension_results = {}
        # 顶层证据缺口:所有维度的缺口汇总(方法级,只在此声明一次)
        all_evidence_gaps = []

        for dim_key, rule_config in self.mapping_rules.items():
            indicators = rule_config["indicators"]
            weighted_sum = 0.0
            total_weight = 0.0
            evidence_chain = []
            positive_factors = []
            negative_factors = []
            inference_data = {}
            # 这两个必须声明在维度循环内 —— 声明在方法级会让 5 个维度共用一个列表,
            # 证据缺口串味(Ruling 2)。
            matched = []      # [(keyword, weight, is_positive, human_name, raw_value, matched_key)]
            dim_gaps = []     # 本维度的证据缺口

            # 3. 逐个指标过证据门:四关全过才允许进加权(spec §5.1)
            for keyword, weight, is_positive, human_name, importance in indicators:
                found_key, found_val = self._fuzzy_match(keyword, all_features)
                if found_key is None:
                    dim_gaps.append(f"{human_name}: 未采集到对应数据")
                    continue

                # 伴随的 _std 用于 G2(常量判定);模态行数用于 G3(样本量)。
                # 取法两种都试:真实数据的列形如 `<基名>_mean` + `<基名>_std`,而有些
                # 键(如 face_eye_contact_ratio)没有 _mean 后缀 —— 只查前者会让
                # 这些键**静默**拿不到标准差:G2 退回 fail-open,区间也失去变异信息。
                std = None
                for std_key in (
                    (found_key[:-len("_mean")] + "_std") if found_key.endswith("_mean") else None,
                    found_key + "_std",
                ):
                    if std_key and std_key in all_features:
                        std = all_features[std_key]
                        break
                modality = next((m for m in n_rows_by_modality if found_key.startswith(m + "_")), None)
                # 样本量优先取**该指标自己**的 `<基名>_sample_size`(特征引擎按 dropna
                # 后真正带值的行数产出),取不到才退回模态行数 —— 理由见
                # `_indicator_sample_size`(过短回答的空格行不能被算进样本量)。
                n_valid = _indicator_sample_size(found_key, all_features)
                if n_valid is None:
                    n_valid = int(n_rows_by_modality.get(modality, 0))

                chk = gate(found_key, found_val, n_valid=n_valid, std=std)
                if not chk.ok:
                    # ⚠️ 只准用 user_message。chk.reason 含维护者文案(封停理由等),
                    # 而 dim_gaps 会被渲染进报告的"证据缺口"一节 —— 直接用会外泄内部信息。
                    dim_gaps.append(f"{human_name}: {user_message(chk)}")
                    continue
                # std 与 n_valid 一并带下去:报告要按 spec §5.4 的
                # 「值 + 有效样本量 + 置信度 + evidence_gaps」渲染,样本量是其中一项;
                # 会话内标准差用于给出**本场实测区间**(spec §5.6 的"区间"只能来自
                # 本场会话自己的测量)。
                matched.append((keyword, weight, is_positive, human_name, found_val,
                                found_key, std, n_valid))

            # 本维缺口并入顶层
            all_evidence_gaps.extend(dim_gaps)

            # 4. 计算 —— 只有过门的指标参与加权
            for keyword, weight, is_positive, human_name, matched_val, matched_key, std, n_valid in matched:
                normalized_val = self._dynamic_normalize(matched_val, keyword)

                if is_positive:
                    final_val = normalized_val
                else:
                    final_val = 1.0 - normalized_val

                final_val = max(0.0, min(1.0, final_val))
                contribution = final_val * weight

                # 调试输出
                if keyword in ['tension_score', 'jitter', 'gaze_stability']:
                    print(
                        f"         🧮 {human_name}: 原始={matched_val:.2f}, 因子={final_val:.2f}, 贡献=+{contribution:.3f}")

                weighted_sum += contribution
                total_weight += weight

                status = "✅ 强支撑" if final_val > 0.7 else "⚠️ 弱支撑" if final_val < 0.4 else "➖ 中性"

                # 只留原始值供调试;百分位已随杜撰基线一并删除(spec §5.5:
                # 没有真实常模就不输出百分位)
                inference_data[keyword] = {"val": matched_val}

                evidence_item = {
                    "feature": matched_key,
                    "human_name": human_name,
                    "raw_value": round(matched_val, 4),
                    "normalized_score": round(final_val, 2),
                    "contribution": round(contribution, 3),
                    "status": status,
                    "direction": "正向" if is_positive else "反向",
                    "weight": weight,
                    # spec §5.4 的「值 + 有效样本量 + 置信度 + evidence_gaps」需要样本量;
                    # 区间只由本场会话自己的测量构成(spec §5.5/§5.6)。
                    "n_valid": n_valid,
                    "std": _as_float(std),
                    "observed_interval": _observed_interval(matched_val, std),
                }
                evidence_chain.append(evidence_item)

                if final_val > 0.65:
                    positive_factors.append(human_name)
                elif final_val < 0.35:
                    negative_factors.append(human_name)

            # 5. 无任何指标过门 → 不出分,只出缺口(spec §5.1 硬规则)
            if not matched:
                dimension_results[dim_key] = {
                    "display_name": rule_config["name"],
                    "score": None,
                    "level": "证据不足",
                    "evidence_chain": [],
                    "evidence_gaps": dim_gaps,
                    "confidence": "无",
                    "matched_indicators": f"0/{len(indicators)}",
                    "stats": {},
                }
                continue

            confidence = confidence_from(len(matched), len(indicators))
            score = round(max(0.0, min(100.0, weighted_sum / total_weight * 100)), 2)
            level = self._get_level(score)

            # 说明:此处曾产出 narrative / simple_narrative(模板填空式评语),
            # 已随 `_generate_deep_inference` 一并删除(spec §5.4:不解读、不推断)。
            # 报告层改为直接渲染证据状态,不再有可填空的句子。
            dimension_results[dim_key] = {
                "display_name": rule_config["name"],
                "description": rule_config["description"],
                "algorithm": rule_config["algorithm"],
                "score": score,
                "level": level,
                "evidence_chain": evidence_chain,
                "positive_factors": positive_factors,
                "negative_factors": negative_factors,
                "evidence_gaps": dim_gaps,
                "confidence": confidence,
                "matched_indicators": f"{len(matched)}/{len(indicators)}",
                "stats": inference_data
            }
            # 控制台输出同样不得印未标定标尺上的点分与档位 —— 报告层一个字都不渲染
            # 复合分/评级(spec §5.4 :157-158),控制台是同一个主张的另一个出口。
            print(f"   ✅ [{rule_config['name']}] 过门指标 {len(matched)}/{len(indicators)}，"
                  f"置信度：{confidence}")

        # 总分 —— 只在有维度过门时产出;一个都没有时 total_score 为 None(spec §6.1)
        valid = [d for d in dimension_results.values() if d["score"] is not None]
        if not valid:
            final_total = None
        else:
            num = sum(d["score"] * self.dimension_weights[k]
                      for k, d in dimension_results.items() if d["score"] is not None)
            den = sum(self.dimension_weights[k]
                      for k, d in dimension_results.items() if d["score"] is not None)
            final_total = round(num / den, 2)

        coverage = self._coverage(dimension_results)

        return {
            "total_score": final_total,
            "total_level": self._get_level(final_total) if final_total is not None else "证据不足",
            "dimensions": dimension_results,
            # ⚠️ total_score / total_level 只为 API 稳定保留:它们是**未标定标尺**上的
            # 复合点分与五档评语,报告层一个字都不渲染(spec §5.4 :157-158 / §5.6)。
            # 聚合呈现改用 coverage(覆盖事实 + 逐指标实测区间 + 置信度上限)。
            "coverage": coverage,
            "summary_narrative": NO_COMPOSITE_STATEMENT,
            "evidence_gaps": all_evidence_gaps,
            "model_metadata": {"version": "JingXin-Mapper-v11-EvidenceGate",
                               "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
        }

    def _coverage(self, dimension_results: Dict[str, Any]) -> Dict[str, Any]:
        """本次观测的覆盖事实 —— 聚合呈现的**唯一**来源(spec §5.4 / §5.6)。

        为什么不是分数:spec §5.4 :157-158 要求「综合…评分为 X 分,评级为 Y」改为
        区间 + 置信度;§5.6 要求「科研潜力评分 X 分」改为区间 + 置信度 + 依据。
        而系统没有标定样本(§5.5 禁止一切人群参照区间),所以能给的只有本场会话
        自己的测量:每个过门指标的实测区间。覆盖率低的会话(今天 1/20 槽)在旧渲染下
        会印出「100.0 / 卓越」或「0.0 / 待提升」—— 那是把未标定标尺上的点分当对人的评定。

        槽数取自各维自己渲染的那两个数字(matched_indicators),与报告正文同源。
        """
        passed_slots = []
        n_slots = 0
        for dim_key, dim in dimension_results.items():
            got, _, total = dim["matched_indicators"].partition("/")
            n_slots += int(total)
            for ev in dim["evidence_chain"]:
                passed_slots.append({
                    "dimension": dim_key,
                    "display_name": dim["display_name"],
                    "indicator": ev["human_name"],
                    "feature": ev["feature"],
                    "raw_value": ev["raw_value"],
                    "observed_interval": ev["observed_interval"],
                    "n_valid": ev["n_valid"],
                    "confidence": dim["confidence"],
                })
        return {
            "n_slots": n_slots,
            "n_passed": len(passed_slots),
            "confidence_cap": max((d["confidence"] for d in dimension_results.values()),
                                  key=CONF_ORDER.get),
            "passed_slots": passed_slots,
        }

    def _dynamic_normalize(self, val: float, keyword: str) -> float:
        """按登记表把原始值折到 0–1(spec §5.1)。

        实现全部搬进 `evidence_gate.normalize_value` —— 满量程、分箱边界一律读
        `evidence_thresholds.json`;本方法保留只为既有调用点稳定,体内不得再出现裸常量。
        """
        return normalize_value(val, keyword)

    def _get_level(self, score: float) -> str:
        if score >= 90:
            return "卓越"
        elif score >= 80:
            return "优秀"
        elif score >= 70:
            return "良好"
        elif score >= 60:
            return "合格"
        else:
            return "待提升"

    def _fuzzy_match(self, keyword: str, all_features: Dict[str, Any]):
        """按关键词在扁平化特征里找第一个数值指标。

        只保留原来第 1 步(精确子串)与第 2 步(去掉 _mean/_std/_sum 的宽松匹配),
        不含第 3 步的硬编码代理与第 4 步的 BASELINE_FILL 兜底 —— 那两类伪造已在
        ① 中删除(spec §5.3),现在由证据门决定谁能进场。

        返回 (真实键名, 值);找不到返回 (None, None)。键名不带 "(宽松)" 尾注,
        因为下游要用它去查伴随的 _std(spec §5.1 G2)与模态行数(G3)。
        """
        for f_key, f_val in all_features.items():
            if isinstance(f_val, (int, float)) and keyword.lower() in f_key.lower():
                return f_key, f_val

        for f_key, f_val in all_features.items():
            clean_key = f_key.replace('_mean', '').replace('_std', '').replace('_sum', '')
            if isinstance(f_val, (int, float)) and keyword.lower() in clean_key:
                return f_key, f_val

        return None, None

if __name__ == "__main__":
    from data_loader import LogDataLoader
    from feature_engine import PsychologicalFeatureEngine

    print("=== 测试 Research Mapper (修复版) ===")
    loader = LogDataLoader()
    data = loader.get_fused_latest_data()

    if data:
        engine = PsychologicalFeatureEngine(data)
        features = engine.extract_all_features()
        if features:
            mapper = ResearchCapabilityMapper()
            result = mapper.map_features_to_scores(features)
            print(f"\n📝 总结：{result['summary_narrative']}")
            print("\n✅ 完成！")
    else:
        print("❌ 数据加载失败。")