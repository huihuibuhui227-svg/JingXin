# voice_interaction/assessment/interview_assessment.py

import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any


class InterviewAssessment:
    def __init__(self):
        self.questions = [
            "请简单介绍一下你自己，包括教育背景和研究兴趣。",
            "你为什么想从事科研工作？",
            "描述一次你解决复杂问题的经历。",
            "你在团队合作中通常扮演什么角色？",
            "你如何应对科研中的失败或挫折？",
            "你最近读过哪些与你研究方向相关的论文？",
            "你未来五年的职业规划是什么？",
            "你有什么问题想问我们吗？"
        ]
        self.qa_pairs: List[Dict[str, Any]] = []

    def get_next_question(self) -> str:
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

    def get_comprehensive_evaluation(self) -> Dict[str, str]:
        if not self.qa_pairs:
            return {"text": "未收到任何回答。"}

        core_competency = self._analyze_core_competency()
        prosody_feedback = self._analyze_prosody()

        full_report = (
            "【核心胜任力与品质评估】\n"
            f"{core_competency}\n\n"
            "【语音表达表现】\n"
            f"{prosody_feedback}"
        )
        return {"text": full_report}

    def _analyze_core_competency(self) -> str:
        answers = [pair["answer"] for pair in self.qa_pairs if pair["answer"] != "[无有效回答]"]
        if not answers:
            return "未检测到有效回答内容，无法评估胜任力。"

        full_text = " ".join(answers).lower()

        # 关键词定义
        research_keywords = ["实验", "数据", "论文", "方法", "分析", "模型", "验证", "创新", "研究", "课题", "文献", "算法", "nlp", "自然语言"]
        problem_solving = ["解决", "克服", "应对", "处理", "优化", "改进", "调试", "失败", "挫折", "困难", "挑战", "复盘", "调整"]
        teamwork = ["合作", "团队", "沟通", "协调", "帮助", "讨论", "协作", "配合", "集体", "整合", "进度"]
        motivation = ["兴趣", "热爱", "目标", "规划", "长期", "坚持", "动力", "热情", "志向", "成就感", "研发", "负责人"]

        feedback = []

        # 科研意识
        research_score = sum(1 for w in research_keywords if w in full_text)
        if research_score >= 2:
            feedback.append("✅ 科研意识强：能提及具体研究方向或技术细节，展现出学术基础。")
        elif research_score == 1:
            feedback.append("🟡 具备基本科研认知，但可补充更多技术细节。")
        else:
            feedback.append("⚠️ 回答中较少体现科研相关经验，建议加强研究背景描述。")

        # 问题解决能力
        ps_score = sum(1 for w in problem_solving if w in full_text)
        if ps_score >= 2:
            feedback.append("✅ 问题解决能力强：能描述应对失败或调试的过程，体现韧性。")
        elif ps_score >= 1:
            feedback.append("🟡 有解决问题的意识，建议补充具体策略和结果。")
        else:
            feedback.append("⚠️ 未充分展示解决复杂问题的经验，可举例说明。")

        # 团队合作
        team_score = sum(1 for w in teamwork if w in full_text)
        if team_score >= 1:
            feedback.append("✅ 团队协作意识良好：强调沟通与协调，符合科研合作需求。")
        else:
            feedback.append("⚠️ 较少提及团队角色，建议突出协作经验。")

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs/interview"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"interview_{timestamp}.txt")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== AI 语音面试记录 ===\n\n")
            for i, pair in enumerate(self.qa_pairs):
                f.write(f"Q{i + 1}: {pair['question']}\n")
                f.write(f"A{i + 1}: {pair['answer']}\n")
                if pair["prosody"]:
                    f.write(f"Prosody: {pair['prosody']}\n")
                f.write("\n")

            eval_result = self.get_comprehensive_evaluation()
            f.write("=== 评估报告 ===\n")
            f.write(eval_result["text"])

        return os.path.abspath(log_path)