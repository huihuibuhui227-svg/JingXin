"""
语音交互日志可视化示例脚本

演示如何使用 voice_interaction 模块的可视化工具来分析面试和科研评估日志
"""

# 标准导入（从项目根目录运行）
try:
    from voice_interaction.utils.visualize import plot_interview_logs, plot_research_logs
except ImportError as e:
    print("❌ 导入失败！请确保从项目根目录运行：")
    print("   python -m voice_interaction.examples.visualize_""logs")
    print(f"   错误详情: {e}")
    exit(1)


def main():
    """主函数：可视化语音交互日志"""
    print("=" * 60)
    print("语音交互日志可视化")
    print("=" * 60)
    print("请选择要可视化的日志类型：")
    print("1. 面试评估日志")
    print("2. 科研评估日志")

    choice = input("请输入选项 (1 或 2): ").strip()

    if choice == '1':
        print("正在可视化面试评估日志...")
        success = plot_interview_logs()
    elif choice == '2':
        print("正在可视化科研评估日志...")
        success = plot_research_logs()
    else:
        print("❌ 无效的选项")
        success = False

    if success:
        print("=" * 60)
        print("✅ 图表已成功显示")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ 图表显示失败")
        print("=" * 60)


if __name__ == "__main__":
    main()