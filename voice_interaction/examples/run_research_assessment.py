"""
科研评估示例脚本

演示如何使用 voice_interaction 模块进行科研潜质评估。
"""

import signal
import sys
from datetime import datetime

# 标准导入
try:
    from voice_interaction.analyzers import SpeechRecognizer, TTSEngine
    from voice_interaction.assessment import ResearchAssessment
    from voice_interaction.utils import VoiceLogger
    from voice_interaction.config import ASSESSMENT_CONFIG
except ImportError as e:
    print("❌ 导入失败！请确保从项目根目录运行：")
    print("   python -m voice_interaction.examples.run_research_assessment")
    print(f"   错误详情: {e}")
    sys.exit(1)


class ResearchAssessmentRunner:
    """科研评估运行器"""

    def __init__(self):
        self.assessment = None
        self.recognizer = None
        self.tts = None
        self.logger = None
        self.running = False
        self._init_components()

    def _init_components(self):
        self.assessment = ResearchAssessment()
        self.recognizer = SpeechRecognizer()
        self.tts = TTSEngine()
        self.logger = VoiceLogger(log_type='research')

    def start(self):
        print("=" * 60)
        print("科研潜质语音评估系统")
        print("=" * 60)

        signal.signal(signal.SIGINT, self._signal_handler)
        self.running = True

        # 开场白
        messages = [
            "你好！我是科研潜质评估助手。",
            "我们将进行8个问题的深度访谈。",
            "每个问题后，请自由回答，说完稍作停顿即可。",
            "准备好了吗？我们开始吧。"
        ]
        for msg in messages:
            if not self._safe_speak(msg):
                return

        # 逐个提问
        while self.running:
            question = self.assessment.get_next_question()
            if question is None:
                break

            print(f"\n问题: {question}")
            if not self._safe_speak(question):
                break

            # 获取回答
            answer = self.recognizer.listen_for_speech()
            if not answer:
                answer = "[无有效回答]"

            self.assessment.add_answer(answer)

        if not self.running:
            print("\n⚠️  评估被中断")
            return

        # 生成评估
        if not self._safe_speak("正在生成科研潜质评估报告..."):
            return
        print("\n正在生成科研潜质评估报告...")

        evaluation_result = self.assessment.evaluate_research_potential()
        evaluation_text = evaluation_result["text"]

        # 播报结果
        print(f"\n科研潜质评估报告:\n{evaluation_text}")
        if not self._safe_speak("以下是你的科研潜质评估报告："):
            return

        # 分段播报总结
        lines = evaluation_text.split('\n')
        for line in lines[:3]:
            if "总结：" in line:
                if not self._safe_speak(line):
                    break
                break
            if not self._safe_speak(line):
                break

        if not self._safe_speak("感谢参与！完整报告已保存。"):
            return

        # 保存日志
        human_log_path = self.assessment.save_log()
        self.logger.log_assessment(
            total_questions=len(self.assessment.questions),
            answered_questions=len(self.assessment.answers),
            ai_model=ASSESSMENT_CONFIG['ai_model'],
            max_tokens=ASSESSMENT_CONFIG['max_tokens'],
            evaluation_result=evaluation_result
        )
        struct_log_path = self.logger.get_csv_path()

        print(f"\n📝 人类可读报告已保存至: {human_log_path}")
        print(f"📊 结构化日志已保存至: {struct_log_path}")

    def _safe_speak(self, text: str) -> bool:
        if not self.running:
            return False
        try:
            return self.tts.speak(text)
        except Exception as e:
            print(f"⚠️  TTS 播报失败: {e}")
            return False

    def _signal_handler(self, signum, frame):
        print("\n⚠️  收到中断信号，正在退出...")
        self.running = False

    def stop(self):
        self.running = False
        if self.tts:
            self.tts.stop()
        print("\n✅ 科研评估系统已安全退出。")


def main():
    runner = None
    try:
        runner = ResearchAssessmentRunner()
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