# Face Expression Analyzer Module

## 概述

面部表情分析模块，提供面部动作单元(AU)分析和情绪识别功能。该模块基于MediaPipe Face Mesh技术，能够实时检测和分析面部表情特征，用于评估面试者的情绪状态和专注度。

## 功能特性

### 核心功能
- **面部动作单元(AU)检测**: 检测皱眉、眉毛抬起、微笑、皱鼻、嘴角下撇、张嘴等关键AU
- **情绪识别**: 基于AU特征识别基本情绪状态
- **专注度评估**: 通过头部偏转和眨眼频率评估专注程度
- **疲劳检测**: 监测闭眼持续时间，识别疲劳状态

### 分析能力
- 实时面部特征提取
- 动作单元强度量化
- 情绪状态分类
- 眨眼频率统计
- 眼睛闭合时长监测

## 模块结构

```
face_expression/
├── __init__.py                 # 模块初始化
├── config.py                   # 配置文件
├── api/
│   ├── __init__.py
│   └── app.py                # FastAPI接口
├── core/
│   ├── __init__.py
│   ├── analysis/              # 分析器
│   └── feature_extraction/    # 特征提取
├── models/
│   ├── __init__.py
│   ├── features.py           # 特征模型
│   └── results.py           # 结果模型
├── pipeline/
│   ├── __init__.py
│   ├── image_pipeline.py      # 图片分析流程
│   └── video_pipeline.py      # 视频分析流程
├── utils/
│   ├── __init__.py
│   ├── logger.py            # 日志工具
│   └── visualize.py         # 可视化工具
└── examples/
    ├── __init__.py
    ├── run_image_analyzer.py
    └── run_video_analyzer.py
```

## 安装依赖

```bash
pip install mediapipe opencv-python numpy pandas matplotlib
```

## 配置说明

### 路径配置
- `DATA_DIR`: 数据根目录 (默认: `data`)
- `INPUT_DIR`: 输入文件目录 (默认: `data/input`)
- `OUTPUT_DIR`: 输出结果目录 (默认: `data/output`)
- `LOGS_DIR`: 日志文件目录 (默认: `data/logs`)
- `FACE_EXPRESSION_OUTPUT_DIR`: 面部表情输出目录 (默认: `data/output/face_expression`)

### MediaPipe配置
```python
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,          # 视频模式
    'max_num_faces': 1,                # 最多检测1张人脸
    'refine_landmarks': True,          # 使用精细关键点
    'min_detection_confidence': 0.8,     # 检测置信度阈值
    'min_tracking_confidence': 0.8       # 跟踪置信度阈值
}
```

### 眨眼检测配置
```python
EYE_CONFIG = {
    'EAR_THRESHOLD': 0.21,              # 眼睛纵横比阈值
    'BLINK_BUFFER_SIZE': 5,             # 眨眼缓冲区大小(秒)
    'YAW_BUFFER_SIZE': 30,              # 头部偏转缓冲区大小(帧数)
    'MIN_BLINK_INTERVAL': 0.3           # 最小眨眼间隔(秒)
}
```

### 情绪识别配置
```python
EMOTION_CONFIG = {
    'FATIGUE_THRESHOLD': 1.0,           # 疲劳检测阈值(秒)
    'FOCUS_HIGH_THRESHOLD': 0.03,       # 高专注度头部偏转阈值
    'FOCUS_LOW_THRESHOLD': 0.08,        # 低专注度头部偏转阈值
    'BLINK_RATE_THRESHOLD': 30,         # 正常眨眼频率阈值(次/分钟)
    'SMILE_RATIO_THRESHOLD': 0.35        # 微笑比例阈值
}
```

## 使用方法

### 图片分析

```python
from face_expression import ImagePipeline

# 初始化管道
pipeline = ImagePipeline()

# 分析图片
result = pipeline.process_image('path/to/image.jpg')

# 获取结果
print(f"情绪: {result.emotion}")
print(f"专注度: {result.features.focus_score}")
```

### 视频分析

```python
from face_expression import VideoPipeline

# 初始化管道
pipeline = VideoPipeline()

# 分析视频
results = pipeline.process_video('path/to/video.mp4')

# 查看结果
for frame_result in results:
    print(f"帧 {frame_result.frame_id}: {frame_result.emotion}")
```

### 实时分析

```python
from face_expression import VideoPipeline
import cv2

pipeline = VideoPipeline()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 分析帧
    result = pipeline.process_frame(frame)

    # 显示结果
    print(f"情绪: {result.emotion}, 专注度: {result.features.focus_score}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### 使用API服务

```bash
# 启动API服务
python -m face_expression.api.app
```

API端点:
- `POST /analyze/image`: 分析单张图片
- `POST /analyze/video`: 分析视频文件
- `GET /health`: 健康检查

## 日志说明

日志文件保存在 `data/logs/face_expression/` 目录下:

- `face_au_log.csv`: 视频分析日志
- `static_face_log.csv`: 静态图片分析日志

日志字段:
- `timestamp`: 时间戳
- `focus_score`: 专注度评分
- `blink_rate_per_min`: 眨眼频率(次/分钟)
- `au4_frown`: AU4 皱眉强度
- `au12_eyebrow_raise`: AU12 眉毛抬起强度
- `au12_smile`: AU12 微笑强度
- `au9_nose_wrinkle`: AU9 皱鼻强度
- `au15_mouth_down`: AU15 嘴角下撇强度
- `au25_mouth_open`: AU25 张嘴强度
- `eye_closed_sec`: 闭眼持续时间(秒)
- `emotion`: 情绪状态

## 输出说明

分析结果保存在 `data/output/face_expression/` 目录下:

- 可视化图片/视频
- 分析报告
- 统计图表

## 注意事项

1. **光照条件**: 确保光线充足，避免背光或强光直射
2. **面部角度**: 保持正对摄像头，偏转角度不宜过大
3. **距离**: 建议距离摄像头0.5-1.5米
4. **遮挡**: 避免面部被头发、口罩等遮挡

## 性能优化

- 使用GPU加速MediaPipe推理
- 降低视频分辨率以提升处理速度
- 调整检测置信度阈值平衡准确率和速度

## 故障排除

### 问题: 检测不到人脸
- 检查光照是否充足
- 确认面部未被遮挡
- 调整 `min_detection_confidence` 参数

### 问题: 情绪识别不准确
- 确保表情自然明显
- 调整 `EMOTION_CONFIG` 中的阈值参数
- 增加训练数据

### 问题: 性能较慢
- 降低视频分辨率
- 调整 `max_num_faces` 参数
- 启用GPU加速

## 版本历史

- v1.0.0: 初始版本
  - 实现基础AU检测
  - 添加情绪识别功能
  - 支持图片和视频分析

## 许可证

Copyright © JingXin Team
