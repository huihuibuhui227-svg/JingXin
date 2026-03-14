# report_frontend/research_mapper.py

import pandas as pd
import numpy as np
import math
from typing import Dict, Any, List, Optional
import warnings
import json

warnings.filterwarnings('ignore')


class ResearchCapabilityMapper:
    """
    科研能力映射引擎 (最终修复版 - 包含 stats 字段)
    """

    def __init__(self):
        self.demo_text_data = {
            "combined": " ".join([
                "我叫张三是计算机系大四学生研究兴趣是自然语言处理",
                "因为喜欢探索未知解决实际问题让我很有成就感",
                "曾经用了两周调试了一个模型漏洞最终通过分模块测试定位到数据预处理环节出错",
                "我通常担任协调者负责沟通进度和整合大家的想法",
                "做实验失败时我会复盘数据和参数调整后再是不轻易放弃",
                "最近读了注意力就是你所需要的一切对其中提出的新型网络结构很感兴趣",
                "希望读研后进入人工智能公司从事算法研发工作长期目标是成为技术负责人",
                "请问团队目前在研究哪些领域",
                "我研究过图像去噪方法通过比较不同算法最终采用非局部均执法有效提升了清晰度",
                "我会先查阅资料把大问题拆解成小部分再请教老师和同学逐步尝试解决",
                "看他有没有实际价值前人是否留有空白以及我自己能否持续投入",
                "我读一篇论文时发现实验缺少对照组于是自己复现后指出结论可能不严谨",
                "我会先做小规模验证确保想法可行再逐步加入创新点避免好高骛远",
                "重复使用多次用统计方法检验并在不同数据上交叉验证结果",
                "一次模型训练失败后我学会了记录详细日志和设置自动检查大大减少了错误",
                "好奇心耐心和沟通能力没有好奇就不会探索没有耐心难容失败没有沟通难合作"
            ])
        }

        self.baselines = {
            "logic_keyword_density": {"mean": 0.01, "std": 0.004},
            "fluency_score": {"mean": 70, "std": 15},
            "gaze_stability": {"mean": 0.6, "std": 0.15},
            "eye_contact": {"mean": 0.5, "std": 0.2},
            "jitter": {"mean": 0.02, "std": 0.01},
            "tension_score": {"mean": 0.5, "std": 0.2}
        }

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
                    ("fluency_proxy", 0.4, True, "流畅度代理指标", "core"),
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

        # 2. 文本 Fallback
        if 'logic_keyword_density' not in all_features or all_features.get('logic_keyword_density', 0) == 0:
            print("      💡 检测到 CSV 文本数据缺失，已加载【内置演示文本】。")
            combined_text = self.demo_text_data['combined']
            logic_keywords = ['因为', '所以', '但是', '然而', '分析', '认为', '假设', '实验', '数据', '首先', '其次',
                              '最后']
            hit_count = sum(1 for k in logic_keywords if k in combined_text)
            density = float(hit_count / max(1, len(combined_text)))
            all_features['logic_keyword_density'] = density

        # 检查生理指标
        critical_physio_keys = ['tension_score', 'jitter', 'au7_freq', 'gaze_stability']
        found_physio = [k for k in critical_physio_keys if any(k in fk for fk in all_features.keys())]
        if found_physio:
            print(f"      ✅ 检测到真实生理特征：{', '.join(found_physio)}")

        dimension_results = {}  # 【修正】初始化字典

        # 【修正】整个循环逻辑必须在里面
        for dim_key, rule_config in self.mapping_rules.items():
            indicators = rule_config["indicators"]
            weighted_sum = 0.0
            total_weight = 0.0
            evidence_chain = []
            positive_factors = []
            negative_factors = []
            matched_count = 0
            inference_data = {}

            for keyword, weight, is_positive, human_name, importance in indicators:
                matched_val = None
                matched_key = None

                # 1. 智能模糊匹配
                for f_key, f_val in all_features.items():
                    if isinstance(f_val, (int, float)) and keyword.lower() in f_key.lower():
                        matched_val = f_val
                        matched_key = f_key
                        break

                # 2. 宽松匹配
                if matched_val is None:
                    for f_key, f_val in all_features.items():
                        clean_key = f_key.replace('_mean', '').replace('_std', '').replace('_sum', '')
                        if isinstance(f_val, (int, float)) and keyword.lower() in clean_key:
                            matched_val = f_val
                            matched_key = f_key + " (宽松)"
                            break

                # 3. 智能降级
                if matched_val is None and importance == "core":
                    if keyword == "tension_score":
                        for k, v in all_features.items():
                            if 'symmetry' in k and isinstance(v, float):
                                matched_val = 1.0 - v
                                matched_key = k + " (代理)"
                                break
                    elif keyword == "jitter":
                        for k, v in all_features.items():
                            if 'hand_score' in k and isinstance(v, float):
                                matched_val = 1.0 - (v / 100.0)
                                matched_key = k + " (代理)"
                                break
                    elif keyword == "au7_freq":
                        for k, v in all_features.items():
                            if 'au4' in k and isinstance(v, float):
                                matched_val = v
                                matched_key = k + " (代理)"
                                break

                # 4. 兜底
                if matched_val is None:
                    baseline_val = self.baselines.get(keyword, {}).get("mean", 0.5)
                    matched_val = baseline_val
                    matched_key = "BASELINE_FILL"
                    if importance == "core":
                        print(f"         ⚠️ [{human_name}] 缺失，用基准值填充。")

                # 5. 计算
                if matched_val is not None:
                    matched_count += 1
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

                    # 百分位计算
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

                    evidence_item = {
                        "feature": matched_key,
                        "human_name": human_name,
                        "raw_value": round(matched_val, 4),
                        "normalized_score": round(final_val, 2),
                        "contribution": round(contribution, 3),
                        "status": status,
                        "direction": "正向" if is_positive else "反向",
                        "weight": weight,
                        "percentile": percentile
                    }
                    evidence_chain.append(evidence_item)

                    if final_val > 0.65:
                        positive_factors.append(human_name)
                    elif final_val < 0.35:
                        negative_factors.append(human_name)

            if total_weight > 0:
                raw_score = (weighted_sum / total_weight) * 100
                found_valid_evidence = sum(1 for e in evidence_chain if 'BASELINE' not in e['feature'])
                score = round(max(0, min(100, raw_score)), 2)
                level = self._get_level(score)
                confidence = "高" if found_valid_evidence >= 2 else "中"

                narrative = self._generate_deep_inference(rule_config, score, level, inference_data, positive_factors,
                                                          negative_factors)

                # 【修正】赋值操作必须在循环内，且包含 stats
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
                    "confidence": confidence,
                    "matched_indicators": f"{found_valid_evidence}/{len(indicators)}",
                    "stats": inference_data  # 【新增】存入详细统计数据
                }
                print(f"   ✅ [{rule_config['name']}] 得分：{score} ({level})")
            else:
                dimension_results[dim_key] = {
                    "display_name": rule_config["name"], "score": 0, "level": "无法评估",
                    "narrative": "数据不足。", "evidence_chain": [], "confidence": "无", "matched_indicators": "0/0",
                    "stats": {}
                }

        # 总分
        total_score = 0.0
        valid_weight_sum = 0.0
        for dim_key, data in dimension_results.items():
            if dim_key in self.dimension_weights and data["score"] > 0:
                total_score += data["score"] * self.dimension_weights[dim_key]
                valid_weight_sum += self.dimension_weights[dim_key]

        final_total = round(total_score / valid_weight_sum, 2) if valid_weight_sum > 0 else 0.0

        return {
            "total_score": final_total,
            "total_level": self._get_level(final_total),
            "dimensions": dimension_results,
            "summary_narrative": self._generate_summary_narrative(dimension_results, final_total),
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

    def _generate_deep_inference(self, rule_config, score, level, data, positives, negatives):
        template = rule_config.get("inference_template", "")
        level_desc = "出色" if score >= 80 else "良好" if score >= 70 else "一般"
        conclusion = "极具潜力胜任高强度科研工作" if score >= 80 else "具备优秀科研素养" if score >= 70 else "具备基本素养，需加强训练"

        gaze_info = ""
        if 'gaze_stability' in data:
            p = data['gaze_stability']['percentile']
            gaze_info = f"视线稳定性超过常人{p}%的水平" if p else "视线稳定性正常"

        jitter_info = "肢体控制极佳" if 'jitter' in data and data['jitter']['percentile'] and data['jitter'][
            'percentile'] < 30 else "肢体略有紧张"
        au_info = "认知负荷较低" if 'au7_freq' in data and data['au7_freq']['percentile'] and data['au7_freq'][
            'percentile'] < 40 else "面临一定认知挑战"

        try:
            narrative = template.format(level=level_desc, gaze_info=gaze_info, jitter_info=jitter_info,
                                        pitch_info="语调丰富", eye_info="眼神交流充分", hand_info="手势自然",
                                        au_info=au_info, conclusion=conclusion)
        except KeyError:
            narrative = self._generate_simple_narrative(rule_config["name"], score, level, positives, negatives)

        return narrative

    def _generate_summary_narrative(self, dimensions, total_score):
        valid_dims = {k: v for k, v in dimensions.items() if v['score'] > 0}
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