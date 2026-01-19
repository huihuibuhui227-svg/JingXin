"""
科研评估模块

提供科研潜质评估功能，支持多维度评分和结构化输出。
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from ..config import ASSESSMENT_CONFIG, LOG_CONFIG, LOGS_DIR, DASHSCOPE_API_KEY

ResearchAssessmentResult = Dict[str, Any]


class ResearchAssessment:
    """科研潜质评估器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化科研评估器

        参数:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or ASSESSMENT_CONFIG.copy()
        self.questions = self._get_default_questions()
        self.answers: List[str] = []
        self.current_index = 0
        self._cached_evaluation: Optional[ResearchAssessmentResult] = None

    def _get_default_questions(self) -> List[str]:
        """获取默认科研评估问题"""
        return [
            "请描述一个你深入研究过的技术、科学或学术问题，你是如何解决的？",
            "当你遇到无法立即解决的难题时，通常会采取哪些步骤？",
            "你如何判断一个研究课题是否值得深入探索？",
            "请分享一次你通过批判性思维发现他人研究中漏洞的经历。",
            "在科研中，你如何平衡创新与可行性？",
            "你通常如何验证你的假设或实验结果的可靠性？",
            "描述一次你从失败实验中学到重要经验的经历。",
            "你认为优秀的科研工作者最重要的三个特质是什么？为什么？"
        ]

    def reset(self) -> None:
        """重置评估状态"""
        self.answers = []
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
            self.answers.append(str(answer).strip())
            # 清除缓存的评估结果
            self._cached_evaluation = None

    def evaluate_research_potential(self) -> ResearchAssessmentResult:
        """
        评估科研潜质（带缓存和降级）

        返回:
            结构化评估结果字典，包含：
            - text: 评估文本
            - is_valid: 是否成功调用 AI
            - error: 错误信息（如果有）
        """
        if self._cached_evaluation is not None:
            return self._cached_evaluation

        result: ResearchAssessmentResult = {
            "text": "",
            "is_valid": False,
            "error": ""
        }

        if not self.config['use_ai_feedback']:
            result["text"] = "（AI评估已关闭）"
            result["is_valid"] = True
            self._cached_evaluation = result
            return result

        if not self.answers:
            result["text"] = "（暂无回答数据）"
            result["is_valid"] = True
            self._cached_evaluation = result
            return result

        if not DASHSCOPE_API_KEY.strip():
            result["text"] = "（缺少 DashScope API 密钥，无法生成 AI 评估）"
            result["error"] = "DASHSCOPE_API_KEY 未配置"
            self._cached_evaluation = result
            return result

        try:
            answers_text = "\n".join([f"Q{i + 1}: {ans}" for i, ans in enumerate(self.answers)])

            prompt = f"""你是一位资深科研导师，请基于以下8个问题的回答，对候选人的科研潜质进行综合评估。

评估维度（每项1-5分）：
1. 问题解决能力
2. 批判性思维
3. 创新意识
4. 科研韧性
5. 学术表达

回答内容：
{answers_text}

请输出：
- 综合评分（1-5星）
- 各维度得分
- 一段100字以内的总结建议

格式：
综合评分：★★★★☆（4.2/5）
问题解决能力：4 | 批判性思维：3 | 创新意识：5 | 科研韧性：4 | 学术表达：3
总结：该候选人展现出较强的创新意识和问题解决能力，但在批判性思维和学术表达上需加强，建议多参与学术讨论。"""

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
            result["text"] = f"AI评估失败: {error_msg[:100]}..."
            result["error"] = error_msg

        self._cached_evaluation = result
        return result

    def save_log(self) -> str:
        """
        保存评估日志

        返回:
            日志文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = LOG_CONFIG['research_log_file'].format(timestamp=timestamp)
        log_path = LOGS_DIR / log_file

        evaluation_result = self.evaluate_research_potential()
        evaluation_text = evaluation_result["text"]

        with open(log_path, "w", encoding='utf-8') as f:
            f.write("=== 科研潜质语音评估报告 ===\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for i, (question, answer) in enumerate(zip(self.questions, self.answers), 1):
                f.write(f"问题 {i}: {question}\n")
                f.write(f"回答: {answer}\n\n")

            f.write("=== AI 综合评估 ===\n")
            f.write(evaluation_text + "\n")

        return str(log_path)