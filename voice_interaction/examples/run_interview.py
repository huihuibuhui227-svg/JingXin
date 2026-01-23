# voice_interaction/examples/run_interview.py

import signal
import sys
import os
import numpy as np

try:
    from voice_interaction.analyzers.speech_recognizer import SpeechRecognizer
    from voice_interaction.analyzers.prosody_analyzer import analyze_prosody  # 注意：这里可能有拼写错误！
    from voice_interaction.assessment.interview_assessment import InterviewAssessment

    from voice_interaction.analyzers.tts_engine import TTSEngine
except ImportError as e:
    print("❌ 导入失败！请从项目根目录运行：")
    print("   python -m voice_interaction.examples.run_interview")
    print(f"   错误详情: {e}")
    sys.exit(1)


class InterviewRunner:
    def __init__(self):
        self.assessment = InterviewAssessment()
        self.recognizer = SpeechRecognizer()
        self.tts = TTSEngine()
        self.running = True

    def _safe_speak(self, text: str):
        """安全播报文本"""
        if not text.strip():
            return
        try:
            print(f"📢 正在播报: '{text}'")
            self.tts.speak(text)
        except Exception as e:
            print(f"⚠️ TTS 异常: {e}")

    def start(self):
        print("=" * 60)
        print("🎤 AI 语音面试系统（含语调分析）")
        print("=" * 60)

        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))

        # 开场白
        self._safe_speak("欢迎参加AI语音面试。")

        while self.running:
            question = self.assessment.get_next_question()
            if not question:
                break

            print(f"\n📌 问题: {question}")
            self._safe_speak(question)

            # 录音 + 识别
            text, audio = self.recognizer.listen_for_speech(timeout=30)

            # 处理空回答
            if not text:
                text = "[无有效回答]"

            # 分析语调特征
            prosody = {}
            if len(audio) > 0:
                try:
                    prosody = analyze_prosody(audio)
                    print(f"📊 语调特征已提取: {list(prosody.keys())}")
                except Exception as e:
                    print(f"⚠️ 语调分析失败: {e}")
            else:
                print("⚠️ 音频为空，跳过语调分析")

            # 关键：传递 prosody 到评估系统
            self.assessment.add_answer(text, prosody)

        # 生成最终报告
        result = self.assessment.get_comprehensive_evaluation()
        print(f"\n📄 最终评估报告:\n{result['text']}")
        self._safe_speak("面试结束，感谢参与！")


def main():
    runner = InterviewRunner()
    runner.start()


if __name__ == "__main__":
    main()