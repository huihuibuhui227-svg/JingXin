"""
评估管道

提供面试和科研评估的完整流程
"""

from typing import List, Dict, Any, Optional
from ..models.voice_models import (
    QuestionAnswerPair,
    AssessmentResult,
    InterviewSession
)


class AssessmentPipeline:
    """评估管道基类"""

    def __init__(self):
        """初始化评估管道"""
        self.qa_pairs: List[QuestionAnswerPair] = []

    def add_qa_pair(
        self,
        question: str,
        answer: str,
        prosody_features: Optional[Any] = None,
        prosody_analysis: Optional[Any] = None
    ) -> None:
        """
        添加问答对

        参数:
            question: 问题
            answer: 回答
            prosody_features: 语音特征
            prosody_analysis: 语音分析结果
        """
        qa_pair = QuestionAnswerPair(
            question=question,
            answer=answer,
            prosody_features=prosody_features,
            prosody_analysis=prosody_analysis
        )
        self.qa_pairs.append(qa_pair)

    def get_valid_answers(self) -> List[str]:
        """获取有效回答列表"""
        return [
            qa.answer
            for qa in self.qa_pairs
            if qa.has_valid_answer
        ]

    def evaluate(self) -> AssessmentResult:
        """
        执行评估

        返回:
            评估结果
        """
        raise NotImplementedError("子类必须实现evaluate方法")

    def reset(self) -> None:
        """重置评估状态"""
        self.qa_pairs = []


class InterviewAssessmentPipeline(AssessmentPipeline):
    """面试评估管道"""

    def __init__(self):
        """初始化面试评估管道"""
        super().__init__()
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

    def add_answer(self, answer: str) -> None:
        """
        添加回答（自动关联当前问题）

        参数:
            answer: 回答文本
        """
        if len(self.qa_pairs) < len(self.questions):
            question = self.questions[len(self.qa_pairs)]
            self.add_qa_pair(question, answer)

    def get_comprehensive_evaluation(self) -> str:
        """
        获取综合评估结果

        返回:
            评估文本
        """
        result = self.evaluate()
        return result.text

    def save_log(self) -> str:
        """
        保存评估日志

        返回:
            日志文件路径
        """
        import os
        from datetime import datetime
        import time

        # 创建输出目录
        output_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            ),
            "data", "logs", "interview"
        )
        os.makedirs(output_dir, exist_ok=True)

        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"interview_emotion_log_{timestamp}.csv")

        # 写入日志（CSV格式）
        with open(log_file, 'w', encoding='utf-8') as f:
            # 写入CSV表头
            f.write("unix_timestamp,timestamp,pitch_mean,pitch_variation,pitch_trend,pitch_direction,energy_mean,energy_variation,speech_ratio,duration_sec,pause_duration_mean,pause_duration_max,pause_frequency,emotion,feedback,question_index,is_valid\n")

            # 写入每条记录
            for i, qa in enumerate(self.qa_pairs):
                if qa.prosody_features and qa.prosody_analysis:
                    features = qa.prosody_features
                    analysis = qa.prosody_analysis

                    # 生成时间戳
                    timestamp = time.time()
                    datetime_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

                    # 获取情绪标签（从feedback中提取）
                    emotion = "积极"  # 默认值
                    if "兴奋" in analysis.feedback:
                        emotion = "兴奋"
                    elif "积极" in analysis.feedback:
                        emotion = "积极"

                    # 写入CSV行
                    f.write(f"{timestamp},{datetime_str},{features.pitch_mean},{features.pitch_std},")
                    f.write(f"{features.pitch_trend},{features.pitch_direction},")
                    f.write(f"{features.energy_mean},{features.energy_std},{features.speech_ratio},")
                    f.write(f"{features.duration_sec},{features.pause_duration_mean},{features.pause_duration_max},")
                    f.write(f"{features.pause_frequency},{emotion},{analysis.feedback},{i},{analysis.is_valid}\n")

        return log_file

    def get_next_question(self) -> Optional[str]:
        """
        获取下一个问题

        返回:
            下一个问题文本，如果没有更多问题则返回None
        """
        if len(self.qa_pairs) < len(self.questions):
            return self.questions[len(self.qa_pairs)]
        return None

    def evaluate(self) -> AssessmentResult:
        """
        执行面试评估

        返回:
            评估结果
        """
        if not self.qa_pairs:
            return AssessmentResult(
                text="未收到任何回答。",
                is_valid=True
            )

        core_competency = self._analyze_core_competency()
        prosody_feedback = self._analyze_prosody()

        full_report = (
            "【核心胜任力与品质评估】\n"
            f"{core_competency}\n\n"
            "【语音表达表现】\n"
            f"{prosody_feedback}"
        )

        return AssessmentResult(
            text=full_report,
            is_valid=True,
            metadata={
                "question_count": len(self.qa_pairs),
                "valid_answer_count": len(self.get_valid_answers())
            }
        )

    def _analyze_core_competency(self) -> str:
        """分析核心胜任力"""
        answers = self.get_valid_answers()
        if not answers:
            return "未检测到有效回答内容，无法评估胜任力。"

        full_text = " ".join(answers).lower()

        # 能力维度关键词
        research_keywords = ["实验", "数据", "论文", "方法", "分析", "模型", "验证", "创新", "研究", "课题", "文献", "算法"]
        problem_solving = ["解决", "克服", "应对", "处理", "优化", "改进", "调试", "失败", "挫折", "困难", "挑战"]
        teamwork = ["合作", "团队", "沟通", "协调", "帮助", "讨论", "协作", "配合", "集体"]
        motivation = ["兴趣", "热爱", "目标", "规划", "长期", "坚持", "动力", "热情", "志向"]
        critical_thinking = ["思考", "逻辑", "推理", "质疑", "反思", "深度", "本质", "原因"]

        feedback = []

        # 科研意识
        research_score = sum(1 for w in research_keywords if w in full_text)
        if research_score >= 3:
            feedback.append("✅ 科研意识强：能具体提及研究方法、论文或技术细节，展现出扎实的学术基础。")
        elif research_score >= 1:
            feedback.append("🟡 科研意识一般：有科研相关表述，但缺乏具体案例或深度。")
        else:
            feedback.append("🔴 科研意识薄弱：回答中较少体现科研经验或学术思维。")

        # 问题解决能力
        ps_score = sum(1 for w in problem_solving if w in full_text)
        if ps_score >= 2:
            feedback.append("✅ 问题解决能力强：能清晰描述面对挑战的应对策略，体现抗压与应变能力。")
        elif ps_score >= 1:
            feedback.append("🟡 具备基本问题解决意识，但解决方案可更具体、结构化。")
        else:
            feedback.append("🔴 未充分展示解决复杂问题的经验，建议加强实例描述。")

        # 团队合作
        team_score = sum(1 for w in teamwork if w in full_text)
        if team_score >= 1:
            feedback.append("✅ 团队协作意识良好：强调合作价值，符合科研工作对沟通能力的要求。")
        else:
            feedback.append("⚠️ 较少提及团队合作，建议在科研场景中突出协作经验。")

        # 内在动机
        mot_score = sum(1 for w in motivation if w in full_text)
        if mot_score >= 2:
            feedback.append("✅ 动机明确：展现出清晰的职业规划与科研热情，稳定性高。")
        elif mot_score >= 1:
            feedback.append("🟡 有一定目标感，但长期发展路径可更具体。")
        else:
            feedback.append("⚠️ 动机表述模糊，建议明确科研兴趣与个人驱动力。")

        # 批判性思维
        ct_score = sum(1 for w in critical_thinking if w in full_text)
        if ct_score >= 1:
            feedback.append("✨ 具备批判性思维：能进行反思或深入分析，展现科研潜力。")

        return "\n".join(feedback)

    def _analyze_prosody(self) -> str:
        """分析语音表达表现"""
        all_prosody = [
            qa.prosody_analysis
            for qa in self.qa_pairs
            if qa.prosody_analysis and qa.prosody_analysis.is_valid
        ]

        if not all_prosody:
            return "未获取到语音特征数据，无法进行语调分析。"

        feedback_lines = []

        for i, prosody in enumerate(all_prosody, start=1):
            line = f"【回答 {i}】"
            parts = []

            # 语调分析
            if prosody.pitch_variation > 40:
                parts.append("语调起伏大，富有表现力")
            elif prosody.pitch_variation < 20:
                parts.append("语调平缓，可能显得不够自信")
            else:
                parts.append("语调自然，有适度变化")

            # 流畅度分析
            if prosody.speech_ratio > 0.6:
                parts.append("表达流畅")
            elif prosody.speech_ratio > 0.3:
                parts.append("表达较连贯")
            else:
                parts.append("停顿较多，略显犹豫")

            # 音量分析
            if prosody.energy_mean > 0.8:
                parts.append("声音洪亮")
            elif prosody.energy_mean < 0.5:
                parts.append("声音偏轻")
            else:
                parts.append("音量适中")

            line += "：" + "；".join(parts)
            feedback_lines.append(line)

        # 综合建议
        avg_pitch = sum(p.pitch_variation for p in all_prosody) / len(all_prosody)
        avg_speech = sum(p.speech_ratio for p in all_prosody) / len(all_prosody)

        suggestions = []
        if avg_pitch < 20:
            suggestions.append("尝试在关键观点处提高音调，增强感染力")
        if avg_speech < 0.4:
            suggestions.append("适当减少停顿，提升表达流畅度")
        if not suggestions:
            suggestions.append("整体语音表达良好，继续保持！")

        overall = "\n\n【语音表达建议】" + "；".join(suggestions)
        return "\n".join(feedback_lines) + overall


class ResearchAssessmentPipeline(AssessmentPipeline):
    """科研评估管道"""

    def __init__(self):
        """初始化科研评估管道"""
        super().__init__()
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

    def add_answer(self, answer: str) -> None:
        """
        添加回答（自动关联当前问题）

        参数:
            answer: 回答文本
        """
        if len(self.qa_pairs) < len(self.questions):
            question = self.questions[len(self.qa_pairs)]
            self.add_qa_pair(question, answer)

    def get_comprehensive_evaluation(self) -> str:
        """
        获取综合评估结果

        返回:
            评估文本
        """
        result = self.evaluate()
        return result.text

    def save_log(self) -> str:
        """
        保存评估日志

        返回:
            日志文件路径
        """
        import os
        from datetime import datetime
        import time

        # 创建输出目录
        output_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            ),
            "data", "logs", "research"
        )
        os.makedirs(output_dir, exist_ok=True)

        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"research_emotion_log_{timestamp}.csv")

        # 写入日志（CSV格式）
        with open(log_file, 'w', encoding='utf-8') as f:
            # 写入CSV表头
            f.write("unix_timestamp,timestamp,pitch_mean,pitch_variation,pitch_trend,pitch_direction,energy_mean,energy_variation,speech_ratio,duration_sec,pause_duration_mean,pause_duration_max,pause_frequency,emotion,feedback,question_index,is_valid\n")

            # 写入每条记录
            for i, qa in enumerate(self.qa_pairs):
                if qa.prosody_features and qa.prosody_analysis:
                    features = qa.prosody_features
                    analysis = qa.prosody_analysis

                    # 生成时间戳
                    timestamp = time.time()
                    datetime_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

                    # 获取情绪标签（从feedback中提取）
                    emotion = "积极"  # 默认值
                    if "兴奋" in analysis.feedback:
                        emotion = "兴奋"
                    elif "积极" in analysis.feedback:
                        emotion = "积极"

                    # 写入CSV行
                    f.write(f"{timestamp},{datetime_str},{features.pitch_mean},{features.pitch_std},")
                    f.write(f"{features.pitch_trend},{features.pitch_direction},")
                    f.write(f"{features.energy_mean},{features.energy_std},{features.speech_ratio},")
                    f.write(f"{features.duration_sec},{features.pause_duration_mean},{features.pause_duration_max},")
                    f.write(f"{features.pause_frequency},{emotion},{analysis.feedback},{i},{analysis.is_valid}\n")

        return log_file

    def get_next_question(self) -> Optional[str]:
        """
        获取下一个问题

        返回:
            下一个问题文本，如果没有更多问题则返回None
        """
        if len(self.qa_pairs) < len(self.questions):
            return self.questions[len(self.qa_pairs)]
        return None

    def evaluate(self) -> AssessmentResult:
        """
        执行科研评估

        返回:
            评估结果
        """
        if not self.qa_pairs:
            return AssessmentResult(
                text="未收到任何回答。",
                is_valid=True
            )

        research_capability = self._analyze_research_capability()
        prosody_feedback = self._analyze_prosody()

        full_report = (
            "【科研能力评估】\n"
            f"{research_capability}\n\n"
            "【语音表达表现】\n"
            f"{prosody_feedback}"
        )

        return AssessmentResult(
            text=full_report,
            is_valid=True,
            metadata={
                "question_count": len(self.qa_pairs),
                "valid_answer_count": len(self.get_valid_answers())
            }
        )

    def _analyze_research_capability(self) -> str:
        """分析科研能力"""
        answers = self.get_valid_answers()
        if not answers:
            return "未检测到有效回答内容，无法评估科研能力。"

        full_text = " ".join(answers).lower()

        # 能力维度关键词
        methodology = ["方法", "实验", "数据", "分析", "验证", "模型", "算法", "测试"]
        critical_thinking = ["质疑", "反思", "逻辑", "推理", "深度", "本质", "原因", "漏洞"]
        innovation = ["创新", "新颖", "独特", "突破", "改进", "优化", "原创"]
        feasibility = ["可行", "现实", "实用", "实现", "落地", "应用"]
        persistence = ["坚持", "反复", "多次", "尝试", "失败", "挫折", "困难"]

        feedback = []

        # 方法论能力
        method_score = sum(1 for w in methodology if w in full_text)
        if method_score >= 3:
            feedback.append("✅ 方法论能力强：能清晰描述研究方法和实验设计，展现系统性思维。")
        elif method_score >= 1:
            feedback.append("🟡 方法论能力一般：有方法意识，但描述不够具体。")
        else:
            feedback.append("🔴 方法论能力薄弱：缺乏对研究方法的系统描述。")

        # 批判性思维
        ct_score = sum(1 for w in critical_thinking if w in full_text)
        if ct_score >= 2:
            feedback.append("✅ 批判性思维强：能深入分析问题本质，发现潜在问题。")
        elif ct_score >= 1:
            feedback.append("🟡 具备一定批判性思维，但深度有待提升。")
        else:
            feedback.append("⚠️ 批判性思维不足：建议加强逻辑推理和反思能力。")

        # 创新能力
        innovation_score = sum(1 for w in innovation if w in full_text)
        if innovation_score >= 2:
            feedback.append("✅ 创新意识强：能提出新颖观点或解决方案。")
        elif innovation_score >= 1:
            feedback.append("🟡 有一定创新意识，但可更加大胆。")
        else:
            feedback.append("⚠️ 创新意识一般：建议多思考突破性方案。")

        # 可行性评估
        feas_score = sum(1 for w in feasibility if w in full_text)
        if feas_score >= 1:
            feedback.append("✅ 可行性意识好：能平衡创新与现实约束。")
        else:
            feedback.append("⚠️ 可行性评估不足：建议关注实际应用场景。")

        # 坚持与韧性
        persist_score = sum(1 for w in persistence if w in full_text)
        if persist_score >= 2:
            feedback.append("✅ 坚韧性强：面对挫折能持续尝试，展现科研韧性。")
        elif persist_score >= 1:
            feedback.append("🟡 有一定韧性，但可进一步加强。")
        else:
            feedback.append("⚠️ 韧性表现不足：建议多分享克服困难的经历。")

        return "\n".join(feedback)

    def _analyze_prosody(self) -> str:
        """分析语音表达表现"""
        all_prosody = [
            qa.prosody_analysis
            for qa in self.qa_pairs
            if qa.prosody_analysis and qa.prosody_analysis.is_valid
        ]

        if not all_prosody:
            return "未获取到语音特征数据，无法进行语调分析。"

        feedback_lines = []

        for i, prosody in enumerate(all_prosody, start=1):
            line = f"【回答 {i}】"
            parts = []

            # 语调分析
            if prosody.pitch_variation > 40:
                parts.append("语调起伏大，富有表现力")
            elif prosody.pitch_variation < 20:
                parts.append("语调平缓，可能显得不够自信")
            else:
                parts.append("语调自然，有适度变化")

            # 流畅度分析
            if prosody.speech_ratio > 0.6:
                parts.append("表达流畅")
            elif prosody.speech_ratio > 0.3:
                parts.append("表达较连贯")
            else:
                parts.append("停顿较多，略显犹豫")

            # 音量分析
            if prosody.energy_mean > 0.8:
                parts.append("声音洪亮")
            elif prosody.energy_mean < 0.5:
                parts.append("声音偏轻")
            else:
                parts.append("音量适中")

            line += "：" + "；".join(parts)
            feedback_lines.append(line)

        # 综合建议
        avg_pitch = sum(p.pitch_variation for p in all_prosody) / len(all_prosody)
        avg_speech = sum(p.speech_ratio for p in all_prosody) / len(all_prosody)

        suggestions = []
        if avg_pitch < 20:
            suggestions.append("尝试在关键观点处提高音调，增强感染力")
        if avg_speech < 0.4:
            suggestions.append("适当减少停顿，提升表达流畅度")
        if not suggestions:
            suggestions.append("整体语音表达良好，继续保持！")

        overall = "\n\n【语音表达建议】" + "；".join(suggestions)
        return "\n".join(feedback_lines) + overall
