"""
语音交互数据可视化工具

提供面试和科研评估日志的可视化功能
"""

import os
from typing import Optional


def plot_interview_logs(csv_path: str = None) -> bool:
    """
    从CSV文件生成面试评估日志可视化图表

    Args:
        csv_path (str): CSV文件路径，如果为None则查找最新的面试日志文件

    Returns:
        bool: 可视化是否成功
    """
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # 如果未指定路径，查找最新的面试日志文件
    if csv_path is None:
        log_dir = Path(__file__).parent.parent.parent / 'data' / 'logs'
        log_files = glob.glob(str(log_dir / 'interview_log_*.csv'))
        if not log_files:
            print(f"❌ 在 {log_dir} 目录下未找到面试日志文件")
            return False
        csv_path = max(log_files, key=os.path.getmtime)
        print(f"正在读取日志文件: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"❌ 文件 {csv_path} 不存在")
        return False

    try:
        df = pd.read_csv(csv_path)
        if df.empty or len(df) == 1:  # 只有表头的情况
            print("❌ 日志文件为空或只有表头，无法生成图表")
            return False

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 设置中文字体支持
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        available_font = None
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, '测试', transform=ax.transAxes)
                plt.close(fig)
                available_font = font
                break
            except:
                continue

        if available_font is None:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            use_chinese = False
            title_prefix = "Interview Assessment Analysis"
        else:
            use_chinese = True
            title_prefix = "面试评估分析"

        # 设置字体大小
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title_prefix, fontsize=16)

        # 1. 回答问题数量 vs 时间
        axes[0, 0].plot(df['timestamp'], df['answered_questions'], 
                       marker='o', linestyle='-', color='blue')
        axes[0, 0].set_title('Answered Questions Over Time' if not use_chinese else '已回答问题数量随时间变化')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True)

        # 2. 评估有效性分布（饼图）
        valid_counts = df['is_valid'].value_counts()
        labels = ['Valid' if not use_chinese else '有效', 'Invalid' if not use_chinese else '无效']
        axes[0, 1].pie(valid_counts.values, labels=[labels[i] for i in range(len(labels))], 
                       autopct='%1.1f%%')
        axes[0, 1].set_title('Evaluation Validity' if not use_chinese else '评估有效性分布')

        # 3. 总问题数 vs 已回答问题数（柱状图）
        x = range(len(df))
        width = 0.35
        axes[1, 0].bar([i - width/2 for i in x], df['total_questions'], width, 
                      label='Total' if not use_chinese else '总问题数', color='blue')
        axes[1, 0].bar([i + width/2 for i in x], df['answered_questions'], width, 
                      label='Answered' if not use_chinese else '已回答', color='green')
        axes[1, 0].set_title('Questions Overview' if not use_chinese else '问题概览')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(x)
        axes[1, 0].legend()
        axes[1, 0].grid(True, axis='y')

        # 4. 评估文本长度分布
        df['text_length'] = df['evaluation_text'].str.len()
        axes[1, 1].hist(df['text_length'], bins=20, color='purple', edgecolor='black')
        axes[1, 1].set_title('Evaluation Text Length Distribution' if not use_chinese else '评估文本长度分布')
        axes[1, 1].set_xlabel('Text Length')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, axis='y')

        plt.tight_layout()
        plt.show()
        return True

    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {str(e)}")
        return False


def plot_research_logs(csv_path: str = None) -> bool:
    """
    从CSV文件生成科研评估日志可视化图表

    Args:
        csv_path (str): CSV文件路径，如果为None则查找最新的科研评估日志文件

    Returns:
        bool: 可视化是否成功
    """
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # 如果未指定路径，查找最新的科研评估日志文件
    if csv_path is None:
        log_dir = Path(__file__).parent.parent.parent / 'data' / 'logs'
        log_files = glob.glob(str(log_dir / 'research_assessment_*.csv'))
        if not log_files:
            print(f"❌ 在 {log_dir} 目录下未找到科研评估日志文件")
            return False
        csv_path = max(log_files, key=os.path.getmtime)
        print(f"正在读取日志文件: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"❌ 文件 {csv_path} 不存在")
        return False

    try:
        df = pd.read_csv(csv_path)
        if df.empty or len(df) == 1:  # 只有表头的情况
            print("❌ 日志文件为空或只有表头，无法生成图表")
            return False

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 设置中文字体支持
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        available_font = None
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, '测试', transform=ax.transAxes)
                plt.close(fig)
                available_font = font
                break
            except:
                continue

        if available_font is None:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            use_chinese = False
            title_prefix = "Research Assessment Analysis"
        else:
            use_chinese = True
            title_prefix = "科研评估分析"

        # 设置字体大小
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title_prefix, fontsize=16)

        # 1. 回答问题数量 vs 时间
        axes[0, 0].plot(df['timestamp'], df['answered_questions'], 
                       marker='o', linestyle='-', color='blue')
        axes[0, 0].set_title('Answered Questions Over Time' if not use_chinese else '已回答问题数量随时间变化')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True)

        # 2. 评估有效性分布（饼图）
        valid_counts = df['is_valid'].value_counts()
        labels = ['Valid' if not use_chinese else '有效', 'Invalid' if not use_chinese else '无效']
        axes[0, 1].pie(valid_counts.values, labels=[labels[i] for i in range(len(labels))], 
                       autopct='%1.1f%%')
        axes[0, 1].set_title('Evaluation Validity' if not use_chinese else '评估有效性分布')

        # 3. 总问题数 vs 已回答问题数（柱状图）
        x = range(len(df))
        width = 0.35
        axes[1, 0].bar([i - width/2 for i in x], df['total_questions'], width, 
                      label='Total' if not use_chinese else '总问题数', color='blue')
        axes[1, 0].bar([i + width/2 for i in x], df['answered_questions'], width, 
                      label='Answered' if not use_chinese else '已回答', color='green')
        axes[1, 0].set_title('Questions Overview' if not use_chinese else '问题概览')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(x)
        axes[1, 0].legend()
        axes[1, 0].grid(True, axis='y')

        # 4. 评估文本长度分布
        df['text_length'] = df['evaluation_text'].str.len()
        axes[1, 1].hist(df['text_length'], bins=20, color='purple', edgecolor='black')
        axes[1, 1].set_title('Evaluation Text Length Distribution' if not use_chinese else '评估文本长度分布')
        axes[1, 1].set_xlabel('Text Length')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, axis='y')

        plt.tight_layout()
        plt.show()
        return True

    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {str(e)}")
        return False


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 获取日志目录
    log_dir = Path(__file__).parent.parent.parent / 'data' / 'logs'

    print("=" * 60)
    print("语音交互日志可视化")
    print("=" * 60)
    print("\n请选择要可视化的日志类型：")
    print("1. 面试评估日志")
    print("2. 科研评估日志")

    choice = input("\n请输入选项 (1 或 2): ").strip()

    if choice == '1':
        success = plot_interview_logs()
    elif choice == '2':
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
