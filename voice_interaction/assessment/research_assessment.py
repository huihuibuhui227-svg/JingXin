# voice_interaction/assessment/research_assessment.py

"""
科研评估模块（离线版）

提供科研潜质评估功能，支持多维度评分和结构化输出。
不依赖任何外部 API，完全本地运行。
"""

import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional


class ResearchAssessment:
    """科研潜质评估器（离线版）"""

    def __init__(self):
        self.questions = [
            "请描述一个你深入研究过的技术、科学或学术问题，你是如何解决的？",
            "当你遇到无法立即解决的难题时，通常会采取哪些步骤？",
            "你如何判断一个研究课题是否值得深入探索？",
            "请分享一次你通过批判性思维发现他人研究中漏洞的经历。",
            "在科研中，你如何平衡创新与可行性？",
            "你通常如何验证你的假设或实验结果的可靠性？",
            "描述一次你从失败实验中学到重要经验的经历。",
            "你认为优秀的科研工作者最重要的三个特质是什么？为什么？"
        ]
        self.qa_pairs: List[Dict[str, Any]] = []

    def get_next_question(self) -> Optional[str]:
        if len(self.qa_pairs) < len(self.questions):
            return self.questions[len(self.qa_pairs)]
        else:
            return None

    def add_answer(self, answer: str, prosody: Dict[str, Any] = None):
        index = len(self.qa_pairs)
        question = self.questions[index] if index < len(self.questions) else "Unknown"
        self.qa_pairs.append({
            "question": question,
            "answer": answer,
            "prosody": prosody or {}
        })

    def evaluate_research_potential(self) -> Dict[str, Any]:
        """
        评估科研潜质（完全本地，无API）
        返回: {"text": "评估报告", "is_valid": True}
        """
        if not self.qa_pairs:
            return {"text": "未收到任何回答。", "is_valid": True}

        core_competency = self._analyze_core_competency()
        prosody_feedback = self._analyze_prosody()

        full_report = (
            "【科研潜质评估】\n"
            f"{core_competency}\n\n"
            "【语音表达表现】\n"
            f"{prosody_feedback}"
        )
        return {"text": full_report, "is_valid": True}

    def _analyze_core_competency(self) -> str:
        answers = [pair["answer"] for pair in self.qa_pairs if pair["answer"] != "[无有效回答]"]
        if not answers:
            return "未检测到有效回答内容，无法评估科研潜质。"

        full_text = " ".join(answers).lower()

        # 关键词定义
        research_keywords = ["实验", "数据", "论文", "方法", "分析", "模型", "验证", "创新", "研究", "课题", "文献", "算法", "nlp", "自然语言", "假设", "变量", "控制", "显著性", "收敛", "过拟合", "正则化", "早停", "平衡", "可行性", "可靠性", "复现", "严谨", "批判性思维"]
        problem_solving = ["解决", "克服", "应对", "处理", "优化", "改进", "调试", "失败", "挫折", "困难", "挑战", "复盘", "调整", "迭代", "学习", "经验"]
        teamwork = ["合作", "团队", "沟通", "协调", "帮助", "讨论", "协作", "配合", "集体", "整合", "进度", "导师", "合作者", "反馈", "交流"]
        motivation = ["兴趣", "热爱", "目标", "规划", "长期", "坚持", "动力", "热情", "志向", "成就感", "研发", "负责人", "产出", "发表", "影响力", "成长", "探索"]

        feedback = []

        # 科研意识
        research_score = sum(1 for w in research_keywords if w in full_text)
        if research_score >= 3:
            feedback.append("✅ 科研意识强：能提及具体研究方法、技术细节或实验设计，展现出扎实的学术基础。")
        elif research_score >= 1:
            feedback.append("🟡 具备基本科研认知，但可补充更多技术细节或量化结果。")
        else:
            feedback.append("⚠️ 回答中较少体现科研相关经验，建议加强研究背景描述。")

        # 问题解决能力
        ps_score = sum(1 for w in problem_solving if w in full_text)
        if ps_score >= 2:
            feedback.append("✅ 问题解决能力强：能描述应对失败或调试的过程，体现韧性与工程能力。")
        elif ps_score >= 1:
            feedback.append("🟡 有解决问题的意识，建议补充具体策略和结果。")
        else:
            feedback.append("⚠️ 未充分展示解决复杂问题的经验，可举例说明。")

        # 批判性思维（科研特色）
        critical_thinking = sum(1 for w in ["批判", "质疑", "反思", "漏洞", "验证", "严谨"] if w in full_text)
        if critical_thinking >= 1:
            feedback.append("✨ 具备批判性思维：能反思研究过程或指出潜在问题，展现科研潜力。")

        # 内在动机
        mot_score = sum(1 for w in motivation if w in full_text)
        if mot_score >= 2:
            feedback.append("✅ 动机明确：展现出清晰的职业规划与科研热情。")
        elif mot_score >= 1:
            feedback.append("🟡 有一定目标感，长期规划可更具体。")
        else:
            feedback.append("⚠️ 动机表述较模糊，建议明确发展方向。")

        return "\n".join(feedback)

    def _analyze_prosody(self) -> str:
        all_prosody = [pair["prosody"] for pair in self.qa_pairs if pair.get("prosody")]
        if not all_prosody:
            return "未获取到语音特征数据，无法进行语调分析。"

        pitch_vars = []
        speech_ratios = []

        for p in all_prosody:
            pv = p.get("pitch_variation")
            if isinstance(pv, (int, float)) and not np.isnan(pv):
                pitch_vars.append(pv)
            sr = p.get("speech_ratio")
            if isinstance(sr, (int, float)) and not np.isnan(sr):
                speech_ratios.append(sr)

        parts = []

        if pitch_vars:
            avg_pitch = np.mean(pitch_vars)
            if avg_pitch < 20:
                parts.append("语调较为平缓，可能显得不够自信或缺乏热情。")
            else:
                parts.append("语调富有变化，表达生动，展现出良好的沟通意愿。")

        if speech_ratios:
            avg_speech = np.mean(speech_ratios)
            if avg_speech > 0.6:
                parts.append("表达流畅，停顿合理，逻辑清晰。")
            else:
                parts.append("存在较多停顿或犹豫，建议加强表达的连贯性。")

        if not parts:
            return "未能从语音中提取有效表达特征。"

        return "".join(parts)

    def save_log(self) -> str:
        """保存评估日志（本地）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs/research"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"research_{timestamp}.txt")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== AI 科研能力评估记录 ===\n\n")
            for i, pair in enumerate(self.qa_pairs):
                f.write(f"Q{i + 1}: {pair['question']}\n")
                f.write(f"A{i + 1}: {pair['answer']}\n")
                if pair["prosody"]:
                    f.write(f"Prosody: {pair['prosody']}\n")
                f.write("\n")

            eval_result = self.evaluate_research_potential()
            f.write("=== 评估报告 ===\n")
            f.write(eval_result["text"])

        return os.path.abspath(log_path)