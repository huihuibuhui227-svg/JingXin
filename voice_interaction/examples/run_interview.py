"""
面试示例脚本

演示如何使用 voice_interaction 模块进行语音面试。
完全复用 analyzers / assessment / utils 模块，避免重复代码。
"""

import signal
import sys
from datetime import datetime

# 标准导入（从项目根目录运行：python -m voice_interaction.examples.run_interview）
try:
    from voice_interaction.analyzers import SpeechRecognizer, TTSEngine
    from voice_interaction.assessment import InterviewAssessment
    from voice_interaction.utils import VoiceLogger
    from voice_interaction.config import ASSESSMENT_CONFIG
except ImportError as e:
    print("❌ 导入失败！请确保从项目根目录运行：")
    print("   python -m voice_interaction.examples.run_interview")
    print(f"   错误详情: {e}")
    sys.exit(1)


class InterviewRunner:
    """面试运行器封装类，便于资源管理和异常处理"""

    def __init__(self):
        self.assessment = None
        self.recognizer = None
        self.tts = None
        self.logger = None
        self.running = False

        # 初始化组件
        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        self.assessment = InterviewAssessment()
        self.recognizer = SpeechRecognizer()
        self.tts = TTSEngine()
        self.logger = VoiceLogger(log_type='interview')

    def start(self):
        """启动面试流程"""
        print("=" * 60)
        print("AI 语音模拟面试系统")
        print("=" * 60)

        # 注册信号处理器（支持 Ctrl+C）
        signal.signal(signal.SIGINT, self._signal_handler)

        self.running = True

        # 开场白
        if not self._safe_speak("你好，欢迎参加AI语音模拟面试。"):
            return
        if not self._safe_speak("我会逐个朗读问题，请自由回答，说完稍作停顿即可。"):
            return

        # 逐个提问
        while self.running:
            question = self.assessment.get_next_question()
            if question is None:
                break

            print(f"\n问题: {question}")
            if not self._safe_speak(question):
                break
            if not self._safe_speak("请开始回答。"):
                break

            # 获取回答
            answer = self.recognizer.listen_for_speech()
            if not answer:
                answer = "[无有效回答]"

            # 记录回答
            self.assessment.add_answer(answer)

        if not self.running:
            print("\n⚠️  面试被中断")
            return

        # 生成评估
        if not self._safe_speak("正在分析您的整体表现，请稍候..."):
            return
        print("\n正在生成综合评估...")

        evaluation_result = self.assessment.get_comprehensive_evaluation()
        evaluation_text = evaluation_result["text"]

        # 播报结果
        print(f"\n综合评估报告:\n{evaluation_text}")
        if not self._safe_speak("以下是您的综合评估报告："):
            return
        if not self._safe_speak(evaluation_text):
            return

        if not self._safe_speak("感谢您的参与！本次模拟面试已结束。"):
            return

        # 保存日志
        human_log_path = self.assessment.save_log()
        self.logger.log_assessment(
            total_questions=len(self.assessment.questions),
            answered_questions=len(self.assessment.qa_pairs),
            ai_model=ASSESSMENT_CONFIG['ai_model'],
            max_tokens=ASSESSMENT_CONFIG['max_tokens'],
            evaluation_result=evaluation_result
        )
        struct_log_path = self.logger.get_csv_path()

        print(f"\n📝 人类可读报告已保存至: {human_log_path}")
        print(f"📊 结构化日志已保存至: {struct_log_path}")

    def _safe_speak(self, text: str) -> bool:
        """安全播报文本，处理 TTS 异常"""
        if not self.running:
            return False
        try:
            success = self.tts.speak(text)
            return success
        except Exception as e:
            print(f"⚠️  TTS 播报失败: {e}")
            return False

    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n⚠️  收到中断信号，正在退出...")
        self.running = False

    def stop(self):
        """停止并清理资源"""
        self.running = False
        if self.tts:
            self.tts.stop()
        print("\n✅ 面试系统已安全退出。")


def main():
    runner = None
    try:
        runner = InterviewRunner()
        runner.start()
    except Exception as e:
        print(f"❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if runner:
            runner.stop()


if __name__ == "__main__":
    main()