# visualize_logs.py
"""
手势与肩部分析日志可视化示例脚本

演示如何使用 gesture_analysis 模块的可视化工具来分析日志数据
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 使用绝对导入
from gesture_analysis.utils.visualize import plot_gesture_features_from_csv


def find_latest_log_file(log_dir):
    """查找最新的手势日志文件"""
    import glob
    log_files = glob.glob(os.path.join(log_dir, "gesture_emotion_log_*.csv"))
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime)


def main():
    """主函数：可视化手势和肩部日志数据"""
    log_dir = os.path.join(str(project_root), 'data', 'logs')

    # 查找最新的日志文件
    log_path = find_latest_log_file(log_dir)

    if not log_path:
        print(f"❌ 在 {log_dir} 目录下未找到手势日志文件")
        print("请先运行 run_realtime_analyzer.py 生成日志数据")
        return

    print("=" * 60)
    print("手势与肩部分析日志可视化")
    print("=" * 60)
    print(f"正在读取日志文件: {log_path}")

    success = plot_gesture_features_from_csv(log_path)

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
