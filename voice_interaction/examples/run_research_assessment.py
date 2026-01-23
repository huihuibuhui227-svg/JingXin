# voice_interaction/examples/run_research_assessment.py

import signal
import sys
import os
import numpy as np

try:
    from voice_interaction.analyzers.speech_recognizer import SpeechRecognizer
    from voice_interaction.analyzers.prosody_analyzer import analyze_prosody
    from voice_interaction.assessment.research_assessment import ResearchAssessment
    from voice_interaction.analyzers.tts_engine import TTSEngine
except ImportError as e:
    print("❌ 导入失败！请从项目根目录运行：")
    print("   python -m voice_interaction.examples.run_research_assessment")
    print(f"   错误详情: {e}")
    sys.exit(1)


class ResearchRunner:
    def __init__(self):
        self.assessment = ResearchAssessment()
        self.recognizer = SpeechRecognizer()
        self.tts = TTSEngine()
        self.running = True

    def _safe_speak(self, text: str):
        if not text.strip():
            return
        try:
            print(f"📢 正在播报: '{text}'")
            self.tts.speak(text)
        except Exception as e:
            print(f"⚠️ TTS 异常: {e}")

    def start(self):
        print("=" * 60)
        print("🔬 科研潜质语音评估系统（离线版）")
        print("=" * 60)

        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))

        self._safe_speak("欢迎参加AI科研能力评估。")

        while self.running:
            question = self.assessment.get_next_question()
            if not question:
                break

            print(f"\n📌 问题: {question}")
            self._safe_speak(question)

            # 录音 + 识别
            text, audio = self.recognizer.listen_for_speech(timeout=30)

            if not text:
                text = "[无有效回答]"

            # 分析语调
            prosody = {}
            if len(audio) > 0:
                try:
                    prosody = analyze_prosody(audio)
                    print(f"📊 语调特征已提取: {prosody}")
                except Exception as e:
                    print(f"⚠️ 语调分析失败: {e}")
            else:
                print("⚠️ 音频为空，跳过语调分析")

            self.assessment.add_answer(text, prosody)

        # 生成最终报告（✅ 关键：调用 evaluate_research_potential）
        result = self.assessment.evaluate_research_potential()
        print(f"\n" + "=" * 60)
        print("📄 最终评估报告:")
        print("=" * 60)
        print(result["text"])
        print("=" * 60)
        self._safe_speak("评估结束，感谢参与！")


def main():
    runner = ResearchRunner()
    runner.start()


if __name__ == "__main__":
    main()