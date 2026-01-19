
# 面部微表情分析模块 (Face Expression Analyzer)

## 模块简介

本模块是JingXin项目的子模块，提供基于MediaPipe的面部动作单元(AU)分析和情绪识别功能，支持实时视频流和静态图片两种分析模式。

## 模块结构

```
face_expression/
├── __init__.py              # 模块初始化文件
├── README.md                # 模块说明文档
├── analyzers/               # 分析器模块
│   ├── __init__.py
│   ├── landmarks.py         # 面部关键点索引配置
│   ├── face_au_analyzer.py  # 实时视频分析器
│   └── image_analyzer.py    # 静态图片分析器
├── inference/               # 推理模块
│   ├── __init__.py
│   └── emotion_infer.py     # 情绪推断引擎
└── utils/                   # 工具模块
    ├── __init__.py
    └── visualize.py         # 数据可视化工具
```

## 功能特性

### 1. 实时视频流分析
- 实时检测面部关键点
- 计算眨眼频率
- 提取多个AU特征值
- 实时情绪识别
- 疲劳检测

### 2. 静态图片分析
- 单张图片AU特征提取
- 情绪状态识别
- 专注度评估

### 3. 数据可视化
- AU特征时序分析
- 多维度数据展示
- 支持中文显示

## 使用方法

### 1. 实时视频分析

```python
from face_expression import FaceAUAnalyzer
import cv2

# 初始化分析器
analyzer = FaceAUAnalyzer(fps=30)

# 打开摄像头
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 处理帧
    features, results, emotion = analyzer.process_frame(frame_rgb)

    # 显示结果
    print(f"当前情绪: {emotion}")
    print(f"专注度: {features['focus_score']}")

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 2. 静态图片分析

```python
from face_expression import StaticFaceAnalyzer

# 初始化分析器
analyzer = StaticFaceAnalyzer()

# 分析图片
features = analyzer.analyze_image("path/to/image.jpg")

# 打印结果
if features:
    print(f"专注度: {features['focus_score']}")
    print(f"当前情绪: {features['emotion']}")
    print(f"AU4 皱眉: {features['au4_frown']}")
    print(f"AU12 微笑: {features['au12_smile']}")
```

### 3. 数据可视化

```python
from face_expression.utils import plot_features_from_csv

# 可视化日志数据
plot_features_from_csv("path/to/log.csv")
```

## AU特征说明

| AU编号 | 特征名称 | 说明 |
|--------|---------|------|
| AU4 | 皱眉 | 眉心距离与脸宽的比值 |
| AU12 | 眉毛上扬/微笑 | 眉眼垂直距离/嘴宽与眼距比值 |
| AU9 | 皱鼻 | 鼻翼与鼻尖距离 |
| AU15 | 嘴角下拉 | 嘴角与嘴中心的垂直距离 |
| AU25 | 张嘴 | 上下唇垂直距离与脸高的比值 |

## 情绪识别

支持识别以下情绪状态：
- 😊 愉悦
- 🧠 专注
- ❓ 困惑
- 😮 惊讶
- 🤢 厌恶
- 😢 悲伤
- 💬 说话
- 😴 疲劳（仅视频模式）
- 😐 中性

## 依赖要求

- mediapipe>=0.10.0
- opencv-python>=4.9.0
- numpy>=1.26.0
- scipy>=1.10.0
- matplotlib>=3.0.0
- pandas>=1.0.0

## 注意事项

1. 确保摄像头权限已开启（视频分析）
2. 图片应包含清晰的正脸（静态分析）
3. 光线条件良好可提高识别准确率
4. 建议使用Python 3.8或更高版本

## 许可证

本模块仅供学习和研究使用。
