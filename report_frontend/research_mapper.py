# report_frontend/research_mapper.py

import pandas as pd
import numpy as np
import math
from typing import Dict, Any, List, Optional
import warnings
import json

from .evidence_gate import confidence_from, gate, user_message

warnings.filterwarnings('ignore')


class ResearchCapabilityMapper:
    """
    科研能力映射引擎 (最终修复版 - 包含 stats 字段)
    """

    def __init__(self):
        # 说明:曾有的 self.demo_text_data(内置演示文本)与 self.baselines(无出处的
        # 常模均值/标准差)已删除 —— 前者让 logic_keyword_density 在无文本列时凭空
        # 得到常量,后者既伪造证据又产出杜撰百分位(spec §5.3 / §5.5)。
        self.mapping_rules = {
            "logical_thinking": {
                "name": "逻辑思维与专注度",
                "description": "评估思维严密性、语言逻辑结构及视觉注意力集中程度。",
                "algorithm": "加权线性组合 + 认知负荷推断",
                "inference_template": "候选人在逻辑构建上表现{level}，结合其{gaze_info}，显示出{conclusion}的科研思维潜质。",
                "indicators": [
                    ("logic_keyword_density", 0.4, True, "逻辑关键词密度", "core"),
                    ("focus_score", 0.3, True, "面部专注度", "core"),
                    ("gaze_stability", 0.2, True, "视线稳定性", "core"),
                    ("au4_freq", 0.1, False, "困惑微表情 (皱眉)", "support"),
                ]
            },
            "stress_resilience": {
                "name": "抗压与情绪稳定性",
                "description": "评估高压下的情绪控制力、生理指标平稳度及焦虑水平。",
                "algorithm": "多模态生理信号融合 (面部 + 肢体 + 眼动)",
                "inference_template": "在压力情境下，候选人表现出{level}的生理稳定性，{jitter_info}，预示其{conclusion}的科研抗压能力。",
                "indicators": [
                    ("tension_score", 0.3, False, "面部紧张度", "core"),
                    ("jitter", 0.3, False, "肢体抖动", "core"),
                    ("gaze_deviation", 0.2, False, "视线偏差", "core"),
                    ("symmetry_score", 0.2, True, "面部对称性", "support"),
                ]
            },
            "communication_fluency": {
                "name": "沟通表达流畅度",
                "description": "评估语言组织能力、语调丰富度及表达连贯性。",
                "algorithm": "韵律特征与停顿分析",
                "inference_template": "语言表达流畅度{level}，{pitch_info}，反映出其{conclusion}的学术沟通能力。",
                "indicators": [
                    ("fluency_score", 0.4, True, "语音流畅度", "core"),
                    ("speech_ratio", 0.3, True, "有效说话占比", "core"),
                    ("pitch_variation", 0.2, True, "语调变化", "support"),
                    ("pause_duration", 0.1, False, "平均停顿时长", "support"),
                ]
            },
            "confidence_level": {
                "name": "自信度",
                "description": "评估自我效能感、肢体开放度及眼神交流质量。",
                "algorithm": "眼动 - 肢体多模态耦合模型",
                "inference_template": "自信水平{level}，眼神接触{eye_info}，手势{hand_info}，表明其{conclusion}的科研自信心。",
                "indicators": [
                    ("hand_score", 0.3, True, "手势自信分", "core"),
                    ("shoulder_score", 0.2, True, "肩部放松度", "support"),
                    ("eye_contact", 0.3, True, "眼神接触比例", "core"),
                    ("energy", 0.2, True, "语音能量", "support"),
                ]
            },
            "cognitive_efficiency": {
                "name": "认知负荷效率",
                "description": "评估处理复杂信息时的脑力消耗效率。",
                "algorithm": "微表情频率分析与响应延迟回归",
                "inference_template": "认知处理效率{level}，{au_info}，暗示其{conclusion}的复杂问题解决能力。",
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
        print("\n⚖️ 正在执行深度映射与心理科研能力判推...")
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

                # 伴随的 _std 用于 G2(常量判定);模态行数用于 G3(样本量)
                std_key = (found_key[:-len("_mean")] + "_std") if found_key.endswith("_mean") else None
                std = all_features.get(std_key) if std_key else None
                modality = next((m for m in n_rows_by_modality if found_key.startswith(m + "_")), None)
                n_valid = int(n_rows_by_modality.get(modality, 0))

                chk = gate(found_key, found_val, n_valid=n_valid, std=std)
                if not chk.ok:
                    # ⚠️ 只准用 user_message。chk.reason 含维护者文案(封停理由等),
                    # 而 dim_gaps 会被渲染进报告的"证据缺口"一节 —— 直接用会外泄内部信息。
                    dim_gaps.append(f"{human_name}: {user_message(chk)}")
                    continue
                matched.append((keyword, weight, is_positive, human_name, found_val, found_key))

            # 本维缺口并入顶层
            all_evidence_gaps.extend(dim_gaps)

            # 4. 计算 —— 只有过门的指标参与加权
            for keyword, weight, is_positive, human_name, matched_val, matched_key in matched:
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
                    "narrative": "本次未采集到足以评估该行为线索的有效样本。",
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

            narrative = self._generate_deep_inference(rule_config, score, level, inference_data, positive_factors,
                                                      negative_factors)

            dimension_results[dim_key] = {
                "display_name": rule_config["name"],
                "description": rule_config["description"],
                "algorithm": rule_config["algorithm"],
                "score": score,
                "level": level,
                "narrative": narrative,
                "simple_narrative": self._generate_simple_narrative(rule_config["name"], score, level,
                                                                    positive_factors, negative_factors),
                "evidence_chain": evidence_chain,
                "positive_factors": positive_factors,
                "negative_factors": negative_factors,
                "evidence_gaps": dim_gaps,
                "confidence": confidence,
                "matched_indicators": f"{len(matched)}/{len(indicators)}",
                "stats": inference_data
            }
            print(f"   ✅ [{rule_config['name']}] 得分：{score} ({level})")

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

        return {
            "total_score": final_total,
            "total_level": self._get_level(final_total) if final_total is not None else "证据不足",
            "dimensions": dimension_results,
            "summary_narrative": self._generate_summary_narrative(dimension_results, final_total),
            "evidence_gaps": all_evidence_gaps,
            "model_metadata": {"version": "JingXin-Mapper-v10.1-FixedStats",
                               "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
        }

    def _dynamic_normalize(self, val: float, keyword: str) -> float:
        val = float(val)
        if any(k in keyword for k in ['freq', 'ratio', 'stability', 'contact']):
            return min(1.0, max(0.0, val))
        if 'score' in keyword:
            return min(1.0, max(0.0, val / 100.0 if val > 1 else val))
        if 'density' in keyword:
            return min(1.0, max(0.0, val * 50))
        if any(k in keyword for k in ['jitter', 'deviation']):
            return min(1.0, max(0.0, val * 5))
        if 'pause' in keyword:
            if 0.5 <= val <= 3.0:
                return 0.0
            elif val < 0.5:
                return 0.5
            else:
                return min(1.0, (val - 3.0) / 5.0)
        if 'length' in keyword:
            return min(1.0, max(0.0, val / 30.0))
        if 'energy' in keyword:
            return min(1.0, max(0.0, val * 250.0))
        if 'pitch' in keyword and 'variation' in keyword:
            return min(1.0, max(0.0, val / 80.0))
        return min(1.0, max(0.0, val / 10.0)) if val >= 0 else 0.0

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

    def _generate_simple_narrative(self, dim_name, score, level, positives, negatives):
        text = f"在**{dim_name}**方面，评估结果为**{level}**（{score}分）。"
        if positives: text += f" 优势：{', '.join(positives)}。"
        if negatives: text += f" 建议：{', '.join(negatives)}。"
        return text

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

    def _generate_deep_inference(self, rule_config, score, level, data, positives, negatives):
        template = rule_config.get("inference_template", "")
        level_desc = "出色" if score >= 80 else "良好" if score >= 70 else "一般"
        conclusion = "极具潜力胜任高强度科研工作" if score >= 80 else "具备优秀科研素养" if score >= 70 else "具备基本素养，需加强训练"

        # 这三句原本由杜撰常模算出的百分位驱动。常模百分位已删(spec §5.5),
        # 因此不再产出"超过常人 X%""正常/偏低"这类无常模支撑的断言,直接留空。
        gaze_info = ""
        jitter_info = ""
        au_info = ""

        try:
            narrative = template.format(level=level_desc, gaze_info=gaze_info, jitter_info=jitter_info,
                                        pitch_info="语调丰富", eye_info="眼神交流充分", hand_info="手势自然",
                                        au_info=au_info, conclusion=conclusion)
        except KeyError:
            narrative = self._generate_simple_narrative(rule_config["name"], score, level, positives, negatives)

        return narrative

    def _generate_summary_narrative(self, dimensions, total_score):
        # score 现在可能是 None(证据不足),不能用 > 0 比较
        valid_dims = {k: v for k, v in dimensions.items() if v['score'] is not None}
        if not valid_dims: return "数据不足。"
        top_dim = max(valid_dims.items(), key=lambda x: x[1]['score'])
        bottom_dim = min(valid_dims.items(), key=lambda x: x[1]['score'])
        summary = f"综合科研潜力评分：**{total_score}** ({self._get_level(total_score)})。"
        summary += f" 核心优势在于**{top_dim[1]['display_name']}**；建议关注**{bottom_dim[1]['display_name']}**的提升。"
        return summary


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