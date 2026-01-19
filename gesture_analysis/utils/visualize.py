"""
手势与肩部分析数据可视化工具

提供手势和肩部特征时序图展示功能
"""

import os
from typing import Optional


def plot_gesture_features_from_csv(csv_path: str = "gesture_emotion_log.csv") -> bool:
    """
    从CSV文件生成手势和肩部特征时序图

    Args:
        csv_path (str): CSV文件路径

    Returns:
        bool: 可视化是否成功
    """
    if not os.path.exists(csv_path):
        print(f"❌ 文件 {csv_path} 不存在，请先运行 run_realtime_analyzer.py")
        return False

    # ====== 安全导入 pandas 和 matplotlib ======
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"❌ 可视化依赖未安装: {str(e)}")
        print("请运行: pip install pandas matplotlib")
        return False

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("❌ CSV文件为空，无法生成图表")
            return False

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # ====== 安全设置中文支持 ======
        # 尝试多种中文字体，避免因字体缺失导致崩溃
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

        # 测试哪种字体可用
        available_font = None
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                # 创建临时图形测试字体
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, '测试', transform=ax.transAxes)
                plt.close(fig)
                available_font = font
                break
            except:
                continue

        if available_font is None:
            # 如果没有中文字体可用，使用默认字体但禁用中文
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            use_chinese = False
            title_prefix = "Gesture and Shoulder Emotion Analysis"
        else:
            use_chinese = True
            title_prefix = "手势与肩部情绪分析"

        # 设置字体大小
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title_prefix, fontsize=16)

        # 1. 左右手评分 vs 时间
        axes[0, 0].plot(df['timestamp'], df['left_hand_score'], color='blue',
                        label='Left Hand' if not use_chinese else '左手评分')
        axes[0, 0].plot(df['timestamp'], df['right_hand_score'], color='green',
                        label='Right Hand' if not use_chinese else '右手评分')
        axes[0, 0].set_title('Hand Scores' if not use_chinese else '手部抗压能力评分')
        axes[0, 0].set_ylabel('Score (0-100)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. 肩部评分 vs 时间
        axes[0, 1].plot(df['timestamp'], df['shoulder_score'], color='orange',
                        label='Shoulder' if not use_chinese else '肩部评分')
        axes[0, 1].set_title('Shoulder Score' if not use_chinese else '肩部紧张度评分')
        axes[0, 1].set_ylabel('Score (0-100)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. 综合情绪评分 vs 时间
        axes[1, 0].plot(df['timestamp'], df['overall_score'], color='purple',
                        label='Overall' if not use_chinese else '综合情绪')
        axes[1, 0].set_title('Overall Emotion Score' if not use_chinese else '综合情绪评分')
        axes[1, 0].set_ylabel('Score (0-100)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. 情绪状态分布（饼图）
        emotion_counts = df['emotion_state'].value_counts()
        axes[1, 1].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Emotion Distribution' if not use_chinese else '情绪状态分布')

        plt.tight_layout()
        plt.show()
        return True

    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {str(e)}")
        return False


if __name__ == "__main__":
    success = plot_gesture_features_from_csv()
    if success:
        print("✅ 图表已成功显示")
    else:
        print("❌ 图表显示失败")
