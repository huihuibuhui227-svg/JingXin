"""
手势与肩部分析日志可视化脚本（整合版）

功能：
1. 自动查找 data/logs/ 下最新的 gesture_emotion_log_*.csv
2. 读取日志数据，生成9个关键特征的时序图
3. 自动适配中文字体（失败则使用英文）
4. 图表保存至 data/output/gesture_analysis/
5. 支持直接运行或作为模块调用

作者：镜心项目 - 肢体分析模块
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import glob

# ==============================
# 1. 路径配置（自动识别项目根目录）
# ==============================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # 假设结构: project_root/gesture_analysis/examples/
sys.path.insert(0, str(PROJECT_ROOT))

# 默认日志和输出目录
DEFAULT_LOG_DIR = PROJECT_ROOT / "data" / "logs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "gesture_analysis"


# ==============================
# 2. 工具函数
# ==============================

def find_latest_log_file(log_dir: Path = None) -> Path | None:
    """查找最新的手势日志文件"""
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None
    log_files = list(log_dir.glob("gesture_emotion_log_*.csv"))
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime)


def plot_and_save_gesture_features(
        csv_path: str | Path,
        output_dir: str | Path = None,
        show_plot: bool = False
) -> bool:
    """
    从CSV生成肢体情绪特征时序图并保存

    Args:
        csv_path: 日志CSV路径
        output_dir: 输出目录，默认为 data/output/gesture_analysis
        show_plot: 是否在保存后显示交互窗口（默认False）

    Returns:
        bool: 是否成功
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ 日志文件不存在: {csv_path}")
        return False

    # 确保输出目录
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 安全导入依赖
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"❌ 可视化依赖未安装: {e}")
        print("请运行: pip install pandas matplotlib")
        return False

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("❌ CSV文件为空，无法生成图表")
            return False

        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # ==============================
        # 字体处理：优先中文，失败则英文
        # ==============================
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        use_chinese = False

        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                fig_test, ax_test = plt.subplots()
                ax_test.text(0.5, 0.5, '测试')
                plt.close(fig_test)
                use_chinese = True
                break
            except:
                continue

        if not use_chinese:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

        # 设置字体大小
        plt.rcParams.update({
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })

        # ==============================
        # 创建3x3子图
        # ==============================
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        title_prefix = "手势与肩部情绪分析" if use_chinese else "Gesture and Shoulder Emotion Analysis"
        fig.suptitle(title_prefix, fontsize=16)

        def _(en, cn):
            return cn if use_chinese else en

        # 1. 左右手评分
        axes[0, 0].plot(df['timestamp'], df['left_hand_score'], color='blue', label=_('Left Hand', '左手评分'))
        axes[0, 0].plot(df['timestamp'], df['right_hand_score'], color='green', label=_('Right Hand', '右手评分'))
        axes[0, 0].set_title(_('Hand Scores', '手部抗压能力评分'))
        axes[0, 0].set_ylabel('Score (0-100)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. 肩部评分
        axes[0, 1].plot(df['timestamp'], df['shoulder_score'], color='orange', label=_('Shoulder', '肩部评分'))
        axes[0, 1].set_title(_('Shoulder Score', '肩部紧张度评分'))
        axes[0, 1].set_ylabel('Score (0-100)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. 综合情绪评分
        axes[0, 2].plot(df['timestamp'], df['overall_score'], color='purple', label=_('Overall', '综合情绪'))
        axes[0, 2].set_title(_('Overall Emotion Score', '综合情绪评分'))
        axes[0, 2].set_ylabel('Score (0-100)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # 4. 手肘抖动
        axes[1, 0].plot(df['timestamp'], df['left_elbow_jitter'], color='red', label=_('Left Elbow', '左肘抖动'))
        axes[1, 0].plot(df['timestamp'], df['right_elbow_jitter'], color='darkred', label=_('Right Elbow', '右肘抖动'))
        axes[1, 0].set_title(_('Elbow Jitter', '手肘抖动幅度'))
        axes[1, 0].set_ylabel('Jitter Level')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 5. 手臂角度
        axes[1, 1].plot(df['timestamp'], df['left_arm_angle'], color='cyan', label=_('Left Arm', '左手臂角度'))
        axes[1, 1].plot(df['timestamp'], df['right_arm_angle'], color='teal', label=_('Right Arm', '右手臂角度'))
        axes[1, 1].set_title(_('Arm Angle', '手臂角度变化'))
        axes[1, 1].set_ylabel('Angle (degrees)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 6. 手臂稳定性
        axes[1, 2].plot(df['timestamp'], df['left_arm_stability'], color='magenta', label=_('Left Arm', '左手臂稳定性'))
        axes[1, 2].plot(df['timestamp'], df['right_arm_stability'], color='purple',
                        label=_('Right Arm', '右手臂稳定性'))
        axes[1, 2].set_title(_('Arm Stability', '手臂稳定性'))
        axes[1, 2].set_ylabel('Stability (0-1)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        # 7. 头部抖动
        axes[2, 0].plot(df['timestamp'], df['head_jitter'], color='brown', label=_('Head', '头部抖动'))
        axes[2, 0].set_title(_('Head Jitter', '头部抖动幅度'))
        axes[2, 0].set_ylabel('Jitter Level')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # 8. 头部倾斜角度
        axes[2, 1].plot(df['timestamp'], df['head_tilt'], color='olive', label=_('Head Tilt', '头部倾斜角度'))
        axes[2, 1].set_title(_('Head Tilt Angle', '头部倾斜角度变化'))
        axes[2, 1].set_ylabel('Angle (degrees)')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # 9. 躯干抖动
        axes[2, 2].plot(df['timestamp'], df['torso_jitter'], color='gray', label=_('Torso', '躯干抖动'))
        axes[2, 2].set_title(_('Torso Jitter', '躯干抖动幅度'))
        axes[2, 2].set_ylabel('Jitter Level')
        axes[2, 2].legend()
        axes[2, 2].grid(True)

        plt.tight_layout()

        # 生成带时间戳的文件名
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"gesture_analysis_{timestamp_str}.png"

        # 保存高分辨率图像
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存到: {output_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return True

    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数：查找最新日志并可视化"""
    print("=" * 60)
    print("手势与肩部分析日志可视化（整合版）")
    print("=" * 60)

    log_path = find_latest_log_file(DEFAULT_LOG_DIR)

    if not log_path:
        print(f"❌ 在 {DEFAULT_LOG_DIR} 目录下未找到日志文件")
        print("请先运行 run_realtime_analyzer.py 生成日志数据")
        return

    print(f"正在读取日志文件: {log_path}")
    success = plot_and_save_gesture_features(log_path, DEFAULT_OUTPUT_DIR, show_plot=False)

    print("=" * 60)
    if success:
        print("✅ 图表已成功生成并保存")
    else:
        print("❌ 图表生成失败")
    print("=" * 60)


if __name__ == "__main__":
    main()