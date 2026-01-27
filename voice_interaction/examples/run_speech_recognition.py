"""
运行语音识别示例

使用方法：
    python -m voice_interaction.examples.run_speech_recognition
"""

from voice_interaction.pipeline.speech_recognition_pipeline import SpeechRecognitionPipeline


def main():
    """主函数：运行语音识别"""
    print("=" * 60)
    print("JingXin 语音识别系统")
    print("=" * 60)
    print("ℹ️  按回车键开始录音")
    print("🛑 输入 'quit' 退出程序\n")

    # 初始化语音识别管道
    recognizer = SpeechRecognitionPipeline()

    while True:
        input("\n按回车键开始录音...")

        # 监听语音
        result, audio_data = recognizer.listen_for_speech(
            timeout=30,
            pause_threshold=1.2
        )

        if result.text:
            print(f"\n✓ 识别结果: {result.text}")
        else:
            print("\n❌ 未识别到有效语音")


if __name__ == "__main__":
    main()
