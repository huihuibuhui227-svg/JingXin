"""
面试评估模块

提供面试流程管理和综合评估功能，支持 AI 智能评估与结构化输出。
"""

from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from ..config import ASSESSMENT_CONFIG, LOG_CONFIG, LOGS_DIR, DASHSCOPE_API_KEY

InterviewAssessmentResult = Dict[str, Any]


class InterviewAssessment:
    """面试评估器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化面试评估器

        参数:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or ASSESSMENT_CONFIG.copy()
        self.questions = self._get_default_questions()
        self.qa_pairs: List[Tuple[str, str]] = []
        self.current_index = 0
        self._cached_evaluation: Optional[InterviewAssessmentResult] = None

    def _get_default_questions(self) -> List[str]:
        """获取默认面试问题"""
        return [
            "请用1-2分钟简单介绍一下你自己，包括你的教育背景和研究兴趣。",
            "你曾经遇到过最困难的学术或技术问题是什么？你是如何解决的？",
            "当你在研究中遇到失败或实验反复不成功时，通常会怎么应对？",
            "请描述一次你主动学习新知识或新技能的经历。是什么驱动你去学的？",
            "在团队合作中，如果你和队友意见严重分歧，你会怎么处理？",
            "你认为自己最大的优点和不足分别是什么？这些特质如何影响你的科研工作？",
            "如果有一个完全自由的研究课题，不受经费和时间限制，你最想探索什么？为什么？",
            "你如何看待科研中的'重复性工作'？会觉得枯燥吗？",
            "你平时通过哪些方式保持对前沿科技或学术动态的关注？",
            "最后，你有什么问题想问我们吗？"
        ]

    def reset(self) -> None:
        """重置面试状态"""
        self.qa_pairs = []
        self.current_index = 0
        self._cached_evaluation = None

    def get_next_question(self) -> Optional[str]:
        """
        获取下一个问题

        返回:
            问题字符串，如果没有更多问题则返回 None
        """
        if self.current_index < len(self.questions):
            question = self.questions[self.current_index]
            self.current_index += 1
            return question
        return None

    def add_answer(self, answer: str) -> None:
        """
        添加回答

        参数:
            answer: 回答文本
        """
        if self.current_index > 0 and answer is not None:
            question = self.questions[self.current_index - 1]
            self.qa_pairs.append((question, str(answer).strip()))
            # 清除缓存的评估结果
            self._cached_evaluation = None

    def get_comprehensive_evaluation(self) -> InterviewAssessmentResult:
        """
        获取综合评估（带缓存和降级）

        返回:
            结构化评估结果字典，包含：
            - text: 评估文本
            - is_valid: 是否成功调用 AI
            - error: 错误信息（如果有）
        """
        if self._cached_evaluation is not None:
            return self._cached_evaluation

        result: InterviewAssessmentResult = {
            "text": "",
            "is_valid": False,
            "error": ""
        }

        # 检查是否启用 AI
        if not self.config['use_ai_feedback']:
            result["text"] = "（AI综合评估已关闭）"
            result["is_valid"] = True
            self._cached_evaluation = result
            return result

        # 检查是否有数据
        if not self.qa_pairs:
            result["text"] = "（暂无回答数据）"
            result["is_valid"] = True
            self._cached_evaluation = result
            return result

        # 检查 API 密钥
        if not DASHSCOPE_API_KEY.strip():
            result["text"] = "（缺少 DashScope API 密钥，无法生成 AI 评估）"
            result["error"] = "DASHSCOPE_API_KEY 未配置"
            self._cached_evaluation = result
            return result

        try:
            qa_text = "\n".join([f"问题：{q}\n回答：{a}" for q, a in self.qa_pairs])

            prompt = f"""你是一位资深人力资源专家和科研导师，请根据以下候选人在模拟面试中的全部回答，对其做出全面、客观的综合评估。

重点关注：
1. 人格特质（如诚信、责任感、抗压能力、合作精神、主动性等）
2. 科研或技术岗位所需的核心素质（如批判性思维、解决问题能力、创新意识、逻辑表达、学术严谨性等）
3. 是否具备胜任目标岗位（如研究员、工程师、科研助理等）的潜力
4. 是否存在明显风险或短板

请用一段150字以内的专业评语总结，并给出是否推荐录用的倾向性意见。

候选人全部问答如下：
{qa_text}

综合评估："""

            from dashscope import Generation
            import dashscope
            dashscope.api_key = DASHSCOPE_API_KEY

            response = Generation.call(
                model=self.config['ai_model'],
                prompt=prompt,
                max_tokens=self.config['max_tokens']
            )

            if hasattr(response, 'output') and hasattr(response.output, 'text'):
                result["text"] = response.output.text.strip()
                result["is_valid"] = True
            else:
                result["text"] = f"（AI评估返回格式异常）"
                result["error"] = "Unexpected API response format"

        except Exception as e:
            error_msg = str(e)
            result["text"] = f"（AI综合评估失败: {error_msg[:100]}...）"
            result["error"] = error_msg

        self._cached_evaluation = result
        return result

    def save_log(self) -> str:
        """
        保存面试日志

        返回:
            日志文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = LOG_CONFIG['interview_log_file'].format(timestamp=timestamp)
        log_path = LOGS_DIR / log_file

        evaluation_result = self.get_comprehensive_evaluation()
        evaluation_text = evaluation_result["text"]

        with open(log_path, "w", encoding='utf-8') as f:
            f.write("=== AI 语音模拟面试记录 ===\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for i, (question, answer) in enumerate(self.qa_pairs, 1):
                f.write(f"问题 {i}: {question}\n")
                f.write(f"回答: {answer}\n\n")

            f.write("=== AI 综合评估报告 ===\n")
            f.write(evaluation_text + "\n")

        return str(log_path)