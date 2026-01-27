"""
运行面试评估系统（使用语音识别）

使用方法：
    python -m voice_interaction.examples.run_interview_assessment_voice
"""

from voice_interaction.pipeline.assessment_pipeline import InterviewAssessmentPipeline
from voice_interaction.pipeline.tts_pipeline import TTSPipeline
from voice_interaction.pipeline.speech_recognition_pipeline import SpeechRecognitionPipeline
from voice_interaction.pipeline.voice_pipeline import VoiceProcessingPipeline
import numpy as np


def main():
    """主函数：运行面试评估"""
    print("=" * 60)
    print("JingXin 面试评估系统（语音识别版）")
    print("=" * 60)
    print("ℹ️  系统会自动识别您的语音回答")
    print("🛑 输入 'quit' 退出程序\n")

    # 初始化组件
    interview = InterviewAssessmentPipeline()
    tts = TTSPipeline()
    recognizer = SpeechRecognitionPipeline()
    voice_processor = VoiceProcessingPipeline()

    # 开始面试
    while True:
        question = interview.get_next_question()
        if not question:
            print("\n" + "=" * 60)
            print("面试已结束！正在生成评估报告...")
            print("=" * 60)
            break

        # 显示问题
        print(f"\n问题: {question}")

        # 语音播报问题
        if tts.is_available():
            tts.speak(question)

        # 获取用户回答（使用语音识别）
        print("\n🎤 请回答（说完后稍作停顿即可）...")
        result, audio_data = recognizer.listen_for_speech(
            timeout=30,
            pause_threshold=1.2
        )
        answer = result.text

        if not answer:
            print("\n❌ 未识别到有效语音，请重新回答")
            continue

        # 显示识别结果
        print(f"✓ 识别结果: {answer}")

        # 分析语音特征
        # 确保音频数据格式正确
        if len(audio_data) > 0:
            # 将音频数据转换回 int16 格式
            audio_int16 = (audio_data * 32768.0).astype(np.int16)
            # 再转换回 float32，但使用正确的归一化
            audio_normalized = audio_int16.astype(np.float32) / 32768.0
            voice_result = voice_processor.process_audio(audio_normalized)
            features = voice_result.get("features")
            analysis = voice_result.get("analysis")
        else:
            features = None
            analysis = None

        # 记录回答（包含语音特征和分析）
        interview.add_qa_pair(
            question=question,
            answer=answer,
            prosody_features=features,
            prosody_analysis=analysis
        )

    # 获取评估结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    result = interview.get_comprehensive_evaluation()
    print(result)

    # 保存日志
    log_path = interview.save_log()
    print(f"\n✓ 评估报告已保存至: {log_path}")


if __name__ == "__main__":
    main()
