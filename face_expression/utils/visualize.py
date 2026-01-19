"""
数据可视化工具

提供AU特征时序图展示功能
"""

import os
from typing import Optional


def plot_features_from_csv(csv_path: str = "face_au_log.csv") -> bool:
    """
    从CSV文件生成AU特征时序图

    Args:
        csv_path (str): CSV文件路径

    Returns:
        bool: 可视化是否成功
    """
    if not os.path.exists(csv_path):
        print(f"❌ 文件 {csv_path} 不存在，请先运行 face_au_analyzer.py")
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
            title_prefix = "Facial Action Units (AU) Time Series Analysis"
        else:
            use_chinese = True
            title_prefix = "面部动作单元（AU）时序分析"

        # 设置字体大小
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(title_prefix, fontsize=16)

        # 1. 专注度 vs 时间
        axes[0, 0].plot(df['timestamp'], df['focus_score'], color='blue',
                        label='Focus Score' if not use_chinese else '专注度')
        axes[0, 0].set_title('Focus Score' if not use_chinese else '专注度变化')
        axes[0, 0].set_ylabel('0.0 ~ 1.0')
        axes[0, 0].grid(True)

        # 2. 眨眼频率 vs 时间
        axes[0, 1].plot(df['timestamp'], df['blink_rate_per_min'], color='red',
                        label='Blink Rate' if not use_chinese else '眨眼频率')
        axes[0, 1].set_title('Blink Rate (per min)' if not use_chinese else '眨眼频率 (次/分钟)')
        axes[0, 1].set_ylabel('per min' if not use_chinese else '次/分钟')
        axes[0, 1].grid(True)

        # 3. AU4 皱眉 vs 时间
        axes[1, 0].plot(df['timestamp'], df['au4_frown'], color='green',
                        label='AU4 Frown' if not use_chinese else 'AU4 皱眉')
        axes[1, 0].set_title('AU4 Frown Intensity' if not use_chinese else 'AU4 皱眉强度')
        axes[1, 0].set_ylabel('0.0 ~ 1.0')
        axes[1, 0].grid(True)

        # 4. AU12 微笑 vs 时间
        axes[1, 1].plot(df['timestamp'], df['au12_smile'], color='orange',
                        label='AU12 Smile' if not use_chinese else 'AU12 微笑')
        axes[1, 1].set_title('AU12 Smile Intensity' if not use_chinese else 'AU12 微笑强度')
        axes[1, 1].set_ylabel('0.0 ~ 1.5')
        axes[1, 1].grid(True)

        # 5. AU15 嘴角下拉 vs 时间
        axes[2, 0].plot(df['timestamp'], df['au15_mouth_down'], color='purple',
                        label='AU15 Mouth Down' if not use_chinese else 'AU15 嘴角下拉')
        axes[2, 0].set_title('AU15 Mouth Down' if not use_chinese else 'AU15 嘴角下拉')
        axes[2, 0].set_ylabel('Higher = More Downward' if not use_chinese else '值越大越向下')
        axes[2, 0].grid(True)

        # 6. AU25 张嘴 vs 时间
        axes[2, 1].plot(df['timestamp'], df['au25_mouth_open'], color='brown',
                        label='AU25 Mouth Open' if not use_chinese else 'AU25 张嘴')
        axes[2, 1].set_title('AU25 Mouth Open Degree' if not use_chinese else 'AU25 张嘴程度')
        axes[2, 1].set_ylabel('Higher = More Open' if not use_chinese else '值越大嘴张得越开')
        axes[2, 1].grid(True)

        plt.tight_layout()
        plt.show()
        return True

    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {str(e)}")
        return False


if __name__ == "__main__":
    success = plot_features_from_csv()
    if success:
        print("✅ 图表已成功显示")
    else:
        print("❌ 图表显示失败")