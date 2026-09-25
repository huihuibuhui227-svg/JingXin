"""
评估管道

提供面试和科研评估的完整流程
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..models.voice_models import (
    QuestionAnswerPair,
    AssessmentResult,
    InterviewSession
)

# 评估侧「人类可读」日志的根目录(仓库内 data/logs 之下,面试/科研各一个子目录)。
# 拎成模块级常量:两个 save_log 各算一遍路径只会漂移,而测试要把它指到临时目录
# (与 T3 对 logger 的 LOGS_DIR 同一手法)。
ASSESSMENT_LOG_ROOT = Path(__file__).resolve().parents[2] / "data" / "logs"

# ⚠️ 产物文件名**不得**落在 `<模态>_<描述>_log_<时间戳>.csv` 这个形态里。
# 那个形态正是报告侧 LogDataLoader 的模态命名空间(`report_frontend/data_loader.py`
# 的 `file_pattern`):它递归扫 `data/logs`,在每个模态里取**文件名时间戳最新**的那份。
# 而本文件名里的时间戳取自**回答**时刻,必然晚于会话开始时铸进 M1 会话日志文件名的
# 那个 —— 于是它每次都被选中;又因为经 API 它永远只有表头(prosody 列只有 examples/
# 那两个脚本会填),装载器随即把它当空表丢掉 → `voice_interview` 一个模态都到不了
# 特征引擎 →「连接词密度」渲染成「未采集到对应数据」。
# 现在叫 `assessment_note_<时间戳>.csv`:整串里没有 `_log_`,**结构上**不可能被
# 那个正则命中(而不是靠"名字里恰好没有某个词")。
# 它的消费者是 `voice_interaction/utils/visualize.py`,那里按同一个词根匹配。
NOTE_STEM = "assessment_note"


def _assessment_note_path(kind: str) -> Path:
    """本次调用的产物路径(时间戳命名:一次调用一份,不覆盖既有文件)。"""
    out_dir = ASSESSMENT_LOG_ROOT / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{NOTE_STEM}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


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
        保存评估日志(人类可读版)

        返回:
            日志文件路径
        """
        import time

        log_file = _assessment_note_path("interview")

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

        return str(log_file)

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
        """语音表达的**观测**文本 —— 只报"量到了几段",不下判语。

        这里原先按 `pitch_variation > 40` / `speech_ratio > 0.6` / `energy_mean` 0.5–0.8
        三条**无出处的硬编码阈值**下判语(「语调起伏大,富有表现力」「表达流畅」
        「音量适中」「整体语音表达良好,继续保持!」)。已删(§3 第 28 条)。三个数的依据都站不住:

        - `speech_ratio` 是**自指阈值** —— 实测真数据里 8 段有 7 段恰为 1.0,
          所以那条判语**恒为**「表达流畅」,不是量出来的(spec §4.3 要删这一列);
        - `energy_mean` 真值约 0.02–0.06,而阈值是 0.5–0.8 ⟹ 恒判「声音偏轻」;
        - `pitch_variation` 实测可达 221 Hz,而那个数是 pyin 的 f0 轨迹不干净造成的
          (见 N1 账本 §6),阈值 40 一撞就出「富有表现力」。

        **它本来"炸不了"**:`qa_pairs` 唯一的写入者是 `add_qa_pair`,而活路径
        (`add_answer`)不传 prosody ⟹ 恒走下面那句提前返回。但 `add_qa_pair`
        **接受** prosody 参数 ⟹ 这是颗**随时能引爆**的雷:实测喂一个
        `pitch_variation=45 / speech_ratio=0.8 / energy_mean=0.5` 的假对象,
        它立刻吐「语调起伏大,富有表现力;表达流畅;音量适中」。所以删掉的是**雷**,
        不只是死代码。真要做韵律判语,得等 M3 把列定义与阈值依据一起重做。
        """
        all_prosody = [
            qa.prosody_analysis
            for qa in self.qa_pairs
            if qa.prosody_analysis and qa.prosody_analysis.is_valid
        ]

        if not all_prosody:
            return "未获取到语音特征数据，无法进行语调分析。"

        # 有数据也只说"量到了几段"这一件事实;判语留给 M3(那时才有依据)。
        return f"获取到 {len(all_prosody)} 段语音特征数据；本轮不给语调判语(阈值依据未立，见 M3)。"


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
        保存评估日志(人类可读版)

        返回:
            日志文件路径
        """
        import time

        log_file = _assessment_note_path("research")

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

        return str(log_file)

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
        """语音表达的**观测**文本 —— 只报"量到了几段",不下判语。

        这里原先按 `pitch_variation > 40` / `speech_ratio > 0.6` / `energy_mean` 0.5–0.8
        三条**无出处的硬编码阈值**下判语(「语调起伏大,富有表现力」「表达流畅」
        「音量适中」「整体语音表达良好,继续保持!」)。已删(§3 第 28 条)。三个数的依据都站不住:

        - `speech_ratio` 是**自指阈值** —— 实测真数据里 8 段有 7 段恰为 1.0,
          所以那条判语**恒为**「表达流畅」,不是量出来的(spec §4.3 要删这一列);
        - `energy_mean` 真值约 0.02–0.06,而阈值是 0.5–0.8 ⟹ 恒判「声音偏轻」;
        - `pitch_variation` 实测可达 221 Hz,而那个数是 pyin 的 f0 轨迹不干净造成的
          (见 N1 账本 §6),阈值 40 一撞就出「富有表现力」。

        **它本来"炸不了"**:`qa_pairs` 唯一的写入者是 `add_qa_pair`,而活路径
        (`add_answer`)不传 prosody ⟹ 恒走下面那句提前返回。但 `add_qa_pair`
        **接受** prosody 参数 ⟹ 这是颗**随时能引爆**的雷:实测喂一个
        `pitch_variation=45 / speech_ratio=0.8 / energy_mean=0.5` 的假对象,
        它立刻吐「语调起伏大,富有表现力;表达流畅;音量适中」。所以删掉的是**雷**,
        不只是死代码。真要做韵律判语,得等 M3 把列定义与阈值依据一起重做。
        """
        all_prosody = [
            qa.prosody_analysis
            for qa in self.qa_pairs
            if qa.prosody_analysis and qa.prosody_analysis.is_valid
        ]

        if not all_prosody:
            return "未获取到语音特征数据，无法进行语调分析。"

        # 有数据也只说"量到了几段"这一件事实;判语留给 M3(那时才有依据)。
        return f"获取到 {len(all_prosody)} 段语音特征数据；本轮不给语调判语(阈值依据未立，见 M3)。"
