import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 使用绝对导入
from face_expression.utils.visualize import plot_features_from_csv


def main():
    """主函数：可视化AU日志数据"""
    log_dir = os.path.join(str(project_root), 'data', 'logs')
    log_path = os.path.join(log_dir, "face_au_log.csv")

    if not os.path.exists(log_path):
        print(f"❌ 日志文件 {log_path} 不存在")
        print("请先运行 run_video_analyzer.py 生成日志数据")
        return

    plot_features_from_csv(log_path)


if __name__ == "__main__":
    main()