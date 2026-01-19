import os
import csv
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 使用绝对导入
try:
    from face_expression.analyzers.image_analyzer import StaticFaceAnalyzer
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已在项目根目录运行，或已正确安装 face_expression 模块")
    exit(1)


def main():
    """主函数：分析静态图片"""
    # 设置要分析的图片路径
    project_root = Path(__file__).parent.parent.parent
    image_path = project_root / 'data' / 'input' / 'test.jpg'

    # 确保输入目录存在
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果测试图片不存在，提示用户
    if not image_path.exists():
        print(f"⚠️ 测试图片不存在: {image_path}")
        print("请将测试图片放入 data/input/ 目录，或修改代码中的图片路径")
        return

    try:
        analyzer = StaticFaceAnalyzer()
        features = analyzer.analyze_image(str(image_path))
    except Exception as e:
        print(f"❌ 分析器初始化失败: {e}")
        return

    if features is None:
        print("分析失败，请检查图片路径和内容")
        return

    # 打印结果到控制台
    print("=" * 60)
    print(f"📊 图片分析结果: {image_path}")
    print("=" * 60)
    print(f"专注度: {features['focus_score']}")
    print(f"眨眼状态: {features['blink_status']}")
    print(f"AU4 皱眉: {features['au4_frown']}")
    print(f"AU12 眉毛上扬: {features['au12_eyebrow_raise']}")
    print(f"AU12 微笑: {features['au12_smile']}")
    print(f"AU9 皱鼻: {features['au9_nose_wrinkle']}")
    print(f"AU15 嘴角下拉: {features['au15_mouth_down']}")
    print(f"AU25 张嘴: {features['au25_mouth_open']}")
    print(f"当前情绪: {features['emotion']}")
    print("=" * 60)

    # 保存到CSV
    log_dir = project_root / 'data' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "static_face_log.csv"

    fieldnames = [
        "timestamp", "image_path", "focus_score", "blink_status", "au4_frown", "au12_eyebrow_raise",
        "au12_smile", "au9_nose_wrinkle", "au15_mouth_down", "au25_mouth_open", "eye_closed_sec", "emotion"
    ]

    # 如果是第一次运行，创建CSV文件并写入表头
    if not log_path.exists():
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # 写入本次分析结果
    features["image_path"] = str(image_path)
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(features)

    print(f"✅ 结果已保存至 {log_path}")


if __name__ == "__main__":
    main()